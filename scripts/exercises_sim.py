"""The measuring exercises (hardware/leg_p0/software/exercises.py, docs/measuring.md) in simulation, and the robot's
logs of them next to the simulator's prediction.

    .venv/bin/python scripts/exercises_sim.py predict [--model M] [--flex hip_roll=5]     # what the simulator expects
    .venv/bin/python scripts/exercises_sim.py compare <ex_*.csv> [--flex hip_roll=5 ...]  # robot vs simulator

The simulator runs the same scripted joint targets (hardware/leg_p0/software/exercise_poses.py) through the evaluator's
servo model (no policy), 5 V, the robot-like servo settings of §13 (hip rolls 2.5x stiffer, fast ramps), deterministic
(no randomization). --flex adds the give the encoders can't see (EvalConfig.hidden_flex, N*m/rad), as in §14.

E1 tip-over angles come from the rigid model's geometry: the tilt at which the centre of mass is straight above the
edge the robot pivots on. Robot vs model: delta_x = h * (tan(model) - tan(robot)) is how far the real centre of mass
sits from the model's along that direction (h = its height).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hardware" / "leg_p0" / "software"))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import exercise_poses as ep  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, quat_rot_t  # noqa: E402

MODEL = "hw_ank3_b150_legs_pelvis_10dof"


# ------------------------------------------------------------------ simulator
def make_env(model, flex=(), volt=0.68, kp_hip_roll=2.5):
    cfg = EvalConfig(action_scale=1.0, ref_table="", ref_residual=False, prof_acc=(25.0, 60.0, 60.0), floor_tc=0.02,
                     reset_joint_noise=0.0, reset_tilt=0.0, imu_filter=True, gyro_range=17.45, obs_noise=0.0,
                     strength_scale=volt, speed_scale=volt, hidden_flex=tuple(flex), max_tilt=1.2)
    env = CpuBiped(model, cfg, seed=0)
    env.reset()
    env.servo["kp"] = env.servo["kp"] * np.array([kp_hip_roll if j.endswith("hip_roll") else 1.0 for j in env.joints])
    assert list(env.joints) == ep.JOINTS and np.allclose(env.q_nom, ep.Q_NOM)
    return env


def drive(env, traj, seconds, every=None):
    """Run the scripted targets; every(env, t, stage, rep, q_cmd) after each 20 ms step. Returns the fall time."""
    for k in range(int(seconds / env.cfg.ctrl_dt)):
        t = k * env.cfg.ctrl_dt
        q_cmd, stage, rep = traj(t)
        env.step(q_cmd - env.q_nom)
        if every:
            every(env, t, stage, rep, q_cmd)
        if env.fallen():
            return t
    return None


def true_tilt(env):
    g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
    return math.degrees(math.atan2(g[1], -g[2])), math.degrees(math.atan2(g[0], -g[2]))     # roll + = leaning left


def sim_shift(model, flex, side="L", rate=0.5, s_max=10.0):
    env = make_env(model, flex)
    tr = ep.ShiftRamp(side, rate, s_max)
    far = 1 if side == "L" else 0
    out = dict(s_unload=None, s_tip=None, curve=[])
    def ev(env, t, stage, rep, q):
        s = abs(math.degrees(tr.s(t)))
        low, _ = env.tip_low()
        roll, _ = true_tilt(env)
        if out["s_unload"] is None and low[far] > 0.001:
            out["s_unload"] = s
        if out["s_tip"] is None and abs(roll) > 10:
            out["s_tip"] = s
        if abs(s - round(s)) < 0.5 * rate * 0.02 and stage == "ramp":
            out["curve"].append((round(s), roll, 1000 * low[far]))
    drive(env, tr, tr.duration, ev)
    return out


def sim_lift(model, flex, side="R", shift=9.5, reps=((0.01, 0.5),), hold=2.5):
    """Each rep from a fresh start (on the robot a tip-over ends the session). Per rep: the lifted board's centre and
    lowest point above the floor during the hold (mm), roll on one foot (deg, + = leaning left), peak roll rate from
    the lowering to the unshift (deg/s), fell."""
    li = 0 if side == "L" else 1
    res = []
    for h, lt in reps:
        env = make_env(model, flex)
        tr = ep.LiftReps(side, shift, [(h, lt)], hold=hold)
        r = dict(cen=[], low=[], roll=[], peak_w=0.0)
        def ev(env, t, stage, rep, q, r=r):
            low, cen = env.tip_low()
            roll, _ = true_tilt(env)
            if stage == "hold":
                r["low"].append(1000 * low[li]); r["cen"].append(1000 * cen[li, 2] - 2.0); r["roll"].append(roll)
            if stage in ("lower", "land"):
                r["peak_w"] = max(r["peak_w"], math.degrees(abs(env.d.qvel[3])))
        r["fell"] = drive(env, tr, tr.duration, ev) is not None
        med = lambda v: float(np.median(v)) if v else float("nan")      # noqa: E731
        r.update(clear_mm=med(r["low"]), centre_mm=med(r["cen"]), roll_deg=med(r["roll"]))
        res.append(r)
    return res


def tip_angles(model, q=None, boards="LR"):
    """Rigid pose q (default the stance), standing on `boards`: tilt at which the centre of mass is above the pivot
    edge, each way (deg); COM height (mm)."""
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"{model}.xml"))
    d = mujoco.MjData(m)
    meta = json.loads((ROOT / "models" / f"{model}.json").read_text())
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in meta["joints"]]
    d.qpos[:] = 0; d.qpos[3] = 1
    for a_, v in zip(adr, ep.Q_NOM if q is None else q):
        d.qpos[a_] = v
    mujoco.mj_forward(m, d)
    corners = []
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "_tip" in n and n[0] in boards and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        corners.append(d.geom_xpos[g] + R @ (s * np.array([sx, sy, sz])))
    P = np.array(corners); c = d.subtree_com[1].copy()
    zmin = P[:, 2].min()
    out = {"com_h_mm": 1000 * (c[2] - zmin)}
    for name, axis, coord, sign in (("forward", 1, 0, 1), ("back", 1, 0, -1), ("left", 0, 1, 1), ("right", 0, 1, -1)):
        for th in np.radians(np.arange(0.0, 80.0, 0.05)):
            ang = th * (sign if axis == 1 else -sign)        # rotate so the robot leans toward `name`
            ca, sa = math.cos(ang), math.sin(ang)
            Rm = (np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]]) if axis == 1 else
                  np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]))
            Pr = P @ Rm.T; cr = Rm @ c
            low = Pr[Pr[:, 2] < Pr[:, 2].min() + 1e-4]
            piv = (low[:, coord] * sign).max() * sign
            if (cr[coord] - piv) * sign >= 0:
                out[name] = math.degrees(th)
                break
    return out


# ------------------------------------------------------------------ robot logs
FK = {}


def board_tilts(model, q, g):
    """Each board's tilt (deg, + = its left edge up) in the gravity-levelled frame, from encoders + IMU."""
    if model not in FK:
        m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"{model}.xml")); d = mujoco.MjData(m)
        meta = json.loads((ROOT / "models" / f"{model}.json").read_text())
        adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in meta["joints"]]
        gid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_tip") for s in "LR"]
        FK[model] = (m, d, adr, gid)
    m, d, adr, gid = FK[model]
    d.qpos[:] = 0; d.qpos[3] = 1
    for a_, v in zip(adr, q):
        d.qpos[a_] = v
    mujoco.mj_kinematics(m, d)
    gz = g / np.linalg.norm(g); x = np.array([1.0, 0, 0]); x -= x @ gz * gz; x /= np.linalg.norm(x); y = np.cross(-gz, x)
    R = np.stack([x, y, -gz])
    P = (R @ (d.geom_xpos[gid] - d.xpos[1]).T).T
    tilt = [math.degrees(math.asin(np.clip((R @ d.geom_xmat[g_].reshape(3, 3))[2, 1], -1, 1))) for g_ in gid]
    return np.array(tilt), P


def read_log(path):
    with open(path) as f:
        r = csv.reader(f); hd = next(r); rows = [x for x in r if len(x) == len(hd)]
    c = {h: i for i, h in enumerate(hd)}
    num = lambda n: np.array([float(x[c[n]]) for x in rows])                 # noqa: E731
    L = dict(t=num("t"), G=np.stack([num(f"g{a}") for a in "xyz"], 1), W=np.stack([num(f"w{a}") for a in "xyz"], 1),
             Q=np.stack([num(f"q_{n}") for n in ep.JOINTS], 1), CMD=np.stack([num(f"cmd_{n}") for n in ep.JOINTS], 1),
             stage=[x[c["stage"]] for x in rows], rep=np.array([int(x[c["rep"]]) for x in rows]),
             mark=[x[c["mark"]] for x in rows])
    L["roll"] = np.degrees(np.arctan2(L["G"][:, 1], -L["G"][:, 2]))
    L["pitch"] = np.degrees(np.arctan2(L["G"][:, 0], -L["G"][:, 2]))
    return L, json.loads(Path(path).with_suffix(".json").read_text())


def compare(a, flexes):
    L, meta = read_log(a.log)
    ex = meta["exercise"]
    print(f"{a.log}: {ex}, stopped by {meta.get('stop')}, marks {meta.get('marks')}")
    if ex == "hold":
        ta = tip_angles(a.model)
        print(f"E1 tip-over angles (deg), model {a.model}: centre of mass {ta['com_h_mm']:.0f} mm up")
        if a.rest:                                    # measured standing untouched on the floor
            rest = dict(roll=a.rest[0], pitch=a.rest[1])
        else:                                         # quiet (gyro < 2 deg/s), near-upright samples
            w = np.degrees(np.linalg.norm(L["W"], axis=1))
            ok = (w < 2) & (np.hypot(L["roll"], L["pitch"]) < 6)
            ok = ok if ok.any() else np.ones(len(w), bool)
            rest = dict(roll=float(np.median(L["roll"][ok])), pitch=float(np.median(L["pitch"][ok])))
        print(f"  standing (reference), the pelvis reads roll {rest['roll']:+.1f}, pitch {rest['pitch']:+.1f} deg "
              f"(model 0: the plate is level when the boards are flat). Tip angles below are measured from there.")
        for key, name, col in (("f", "forward", "pitch"), ("b", "back", "pitch"), ("l", "left", "roll"), ("r", "right", "roll")):
            idx = [i for i, m_ in enumerate(L["mark"]) if m_ == key]
            if not idx:
                print(f"  {name:8s} model {ta[name]:5.1f}   robot: no '{key}' marks"); continue
            rob = float(np.median(np.abs(L[col][idx] - rest[col])))
            dx = ta["com_h_mm"] * (math.tan(math.radians(ta[name])) - math.tan(math.radians(rob)))
            print(f"  {name:8s} model {ta[name]:5.1f}   robot {rob:5.1f} (n={len(idx)})  -> real centre of mass "
                  f"{abs(dx):.0f} mm {'closer to' if dx > 0 else 'further from'} that edge than the model's")
        pi = [i for i, m_ in enumerate(L["mark"]) if m_ == "p"]
        if pi:
            win = sorted({j for i in pi for j in range(max(0, i - 25), min(len(L["t"]), i + 25))})
            bt = np.array([board_tilts(a.model, L["Q"][j], L["G"][j])[0] for j in win])
            rest = np.array([board_tilts(a.model, L["Q"][j], L["G"][j])[0] for j in range(0, min(50, len(L["t"])))])
            print(f"E2 give: with both boards flat, the encoders + IMU put the boards at {np.round(rest.mean(0), 1)} deg at rest "
                  f"and up to {np.round(np.abs(bt - rest.mean(0)).max(0), 1)} deg away from that while pushed "
                  f"(pelvis roll seen by the IMU {np.ptp(L['roll'][win]):.1f} deg peak to peak). Rigid legs: 0.")
    elif ex.startswith("shift"):
        side = meta["side"]
        s_log = np.degrees(L["CMD"][:, 6] - ep.Q_NOM[6]) * (1 if side == "L" else -1)
        print(f"E6 sideways shift onto the {side} board, {meta['rate_deg_s']} deg/s:")
        for key, what in (("u", "far board unloaded"), ("t", "tipping")):
            idx = [i for i, m_ in enumerate(L["mark"]) if m_ == key]
            if idx:
                i = idx[0]
                print(f"  robot {what:20s} at shift {s_log[i]:4.1f} deg (pelvis roll {L['roll'][i]:+.1f})")
        tilt_stop = str(meta.get("stop", "")).startswith("tilt")
        if tilt_stop:
            i = next(i for i, s_ in enumerate(L["stage"]) if s_ == "return")
            print(f"  robot tilt cut-out             at shift {s_log[i]:4.1f} deg")
        print("  robot along the ramp: shift deg -> pelvis roll (IMU) / board tilts L, R (encoders + IMU; flat = 0):")
        for sd in range(1, int(meta["max_shift_deg"]) + 1):
            i = next((i for i, (s_, st) in enumerate(zip(s_log, L["stage"])) if st == "ramp" and s_ >= sd), None)
            if i is None:
                break
            bt, _ = board_tilts(a.model, L["Q"][i], L["G"][i])
            print(f"     {sd:3d}: {L['roll'][i]:+5.1f} / {bt[0]:+5.1f}, {bt[1]:+5.1f}")
        for fl in flexes:
            r = sim_shift(a.model, fl, side, meta["rate_deg_s"], meta["max_shift_deg"])
            print(f"  SIM {fmt(fl):18s} far board unloads at {fmt_s(r['s_unload'])}, tips at {fmt_s(r['s_tip'])}; "
                  f"roll at 2/4/6/8 deg: {[round(x[1], 1) for x in r['curve'] if x[0] in (2, 4, 6, 8)]}")
    elif ex.startswith("lift"):
        reps = [tuple(x) for x in meta["reps"]]
        sims = {fmt(fl): sim_lift(a.model, fl, meta["side"], meta["shift_deg"], reps, meta["hold_s"]) for fl in flexes}
        print(f"E4/E5 one-foot lifts of the {meta['side']} board (rigid-leg lift, lowering time):")
        for k, (h, lt) in enumerate(reps):
            ix = np.where(L["rep"] == k)[0]
            if not len(ix):
                print(f"  rep {k}: not reached"); continue
            mk = "".join(L["mark"][i] for i in ix)
            hold = [i for i in ix if L["stage"][i] == "hold"]
            land = [i for i in ix if L["stage"][i] in ("lower", "land")]
            roll = np.median(L["roll"][hold]) if hold else float("nan")
            pw = np.degrees(np.abs(L["W"][land, 0])).max() if land else float("nan")
            line = f"  rep {k}: {1000 * h:4.0f} mm, {lt:4.2f} s | robot marks '{mk}' roll {roll:+5.1f} peak roll rate {pw:4.0f} deg/s"
            for name, res in sims.items():
                r = res[k]
                line += f" | SIM {name}: centre/lowest {r['centre_mm']:4.1f}/{r['clear_mm']:4.1f} mm roll {r['roll_deg']:+5.1f} peak {r['peak_w']:4.0f}{' FELL' if r['fell'] else ''}"
            print(line)
        oi = [(m_, i) for i, m_ in enumerate(L["mark"]) if m_ in ("o", "i")]
        if oi:
            q1 = ep.lift_pose(ep.shift_pose(math.radians(meta["shift_deg"]) * (1 if meta["side"] == "R" else -1)),
                              meta["side"], ep.lift_u(reps[0][0]))
            st = "L" if meta["side"] == "R" else "R"
            t1 = tip_angles(a.model, q1, st)
            out_dir, in_dir = ("left", "right") if st == "L" else ("right", "left")
            for key, d_ in (("o", out_dir), ("i", in_dir)):
                v = [abs(L["roll"][i]) for m_, i in oi if m_ == key]
                if v:
                    dx = t1["com_h_mm"] * (math.tan(math.radians(t1[d_])) - math.tan(math.radians(float(np.median(v)))))
                    print(f"E3 one-foot tip-over {'outward' if key == 'o' else 'inward'}: model {t1[d_]:.1f}, robot "
                          f"{np.median(v):.1f} deg (n={len(v)}) -> real centre of mass {abs(dx):.0f} mm "
                          f"{'closer to' if dx > 0 else 'further from'} that edge")
        if any(s_ == "return" for s_ in L["stage"]) and str(meta.get("stop", "")).startswith("tilt"):
            i = next(i for i, s_ in enumerate(L["stage"]) if s_ == "return")
            print(f"  robot tilt cut-out in rep {L['rep'][i - 1]}")


def fmt(fl):
    return "rigid" if not fl else " ".join(f"{j}={k:g}" for j, k in fl)


def fmt_s(s):
    return "never" if s is None else f"{s:4.1f} deg"


def predict(a, flexes):
    ta = tip_angles(a.model)
    print(f"{a.model}: centre of mass {ta['com_h_mm']:.0f} mm above the soles in the stance pose")
    print(f"E1 tip-over angles (rigid): forward {ta['forward']:.1f}, back {ta['back']:.1f}, left {ta['left']:.1f}, "
          f"right {ta['right']:.1f} deg")
    for fl in flexes:
        r = sim_shift(a.model, fl, "L", 0.5, a.max_shift)
        print(f"E6 shift [{fmt(fl)}]: far board unloads at {fmt_s(r['s_unload'])}, tips at {fmt_s(r['s_tip'])}; "
              f"pelvis roll at 2/4/6/8 deg: {[round(x[1], 1) for x in r['curve'] if x[0] in (2, 4, 6, 8)]}")
    q1 = ep.lift_pose(ep.shift_pose(math.radians(9.5)), "R", ep.lift_u(0.010))
    t1 = tip_angles(a.model, q1, "L")
    print(f"E3 one-foot tip-over angles (rigid, on the left board, 9.5 deg shift, right board 10 mm up): outward (left) "
          f"{t1['left']:.1f}, inward (right) {t1['right']:.1f}, forward {t1['forward']:.1f}, back {t1['back']:.1f} deg")
    reps = ep.parse_reps([2, 4, 6, 8, 12], [1.0])
    for fl in flexes:
        res = sim_lift(a.model, fl, "R", 9.5, reps)
        print(f"E4 lifts [{fmt(fl)}] (rigid-leg mm -> board centre / lowest point above the floor mm, roll on one foot): "
              + ", ".join(f"{1000 * h:.0f} -> {r['centre_mm']:.1f}/{r['clear_mm']:.1f} ({r['roll_deg']:+.1f})"
                          f"{' FELL' if r['fell'] else ''}" for (h, _), r in zip(reps, res)))
    reps = ep.parse_reps([12], [2.0, 1.0, 0.5, 0.25, 0.1])
    for fl in flexes:
        res = sim_lift(a.model, fl, "R", 9.5, reps)
        print(f"E5 landings [{fmt(fl)}] 12 mm, lowering s -> peak roll rate deg/s: "
              + ", ".join(f"{lt} -> {r['peak_w']:.0f}{' FELL' if r['fell'] else ''}" for (_, lt), r in zip(reps, res)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["predict", "compare"])
    ap.add_argument("log", nargs="?")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--flex", nargs="*", default=["", "hip_roll=5"],
                    help="hidden-give variants to simulate, each 'joint=k[,joint=k]' ('' = rigid)")
    ap.add_argument("--max-shift", type=float, default=9.5)
    ap.add_argument("--rest", type=float, nargs=2, metavar=("ROLL", "PITCH"),
                    help="E1: IMU roll, pitch (deg) standing untouched on the floor (default: the log's quiet upright samples)")
    a = ap.parse_args()
    flexes = [tuple((kv.split("=")[0], float(kv.split("=")[1])) for kv in f.split(",") if kv) for f in a.flex]
    if a.cmd == "predict":
        predict(a, flexes)
    else:
        compare(a, flexes)


if __name__ == "__main__":
    main()
