"""Ankle-roll design study (2026-11-09): what an ankle roll joint per leg changes, with deterministic controllers
on the CPU simulator (models from scripts/make_ankle_model.py).

    .venv/bin/python scripts/ankle_study.py balance [--model hw_pad_board150w_legs_ankle_10dof]
    .venv/bin/python scripts/ankle_study.py walk --steps 6 --fwd 0.03

balance: shift the weight over the left board with the hip rolls while the ankle rolls keep both boards flat (the
legs lean like a parallelogram, the pelvis stays level), lift the right board, hold it up, put it down.
walk: quasi-static walking: shift over the stance board, lift, swing forward (stance leg sweeps back, pelvis
advances), put down, shift over it, next leg; every phase slow enough to be near-static.
"""
import argparse, json, math, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run, quat_rot_t  # noqa: E402
from slender_biped.rock_walk import leg_ik, leg_fk  # noqa: E402

BASE = {"hw_pad_board150w_legs_ankle_10dof": "pad_board150w_legs", "hw_pad_board150w_ankle_10dof": "pad_board150w"}
DT = 0.02
# joint indices (10-dof): L hip yaw 0, roll 1, pitch 2, knee 3, ankle roll 4; R 5..9
LR, LP, LK, LA, RR, RP, RK, RA = 1, 2, 3, 4, 6, 7, 8, 9


def ease(x):
    x = min(1.0, max(0.0, x)); return 0.5 - 0.5 * math.cos(math.pi * x)


def make_env(model, tc, seed, dr, imu=True):
    cfg, _ = load_run(BASE[model])
    e = dict(cfg["env"], model=model)
    over = dict(floor_tc=tc, action_scale=1.0, reset_joint_noise=0.0, reset_tilt=0.0, dr=dr, push_vel=0.0,
                ref_table="", ref_residual=False)
    if imu:
        over.update(imu_filter=True, imu_gate=0.15, gyro_range=17.45)
    return CpuBiped(model, EvalConfig.from_train(e, **over), seed=seed)


def shift_pose(phi, stance_left, ksign):
    """Hip and ankle rolls (rad offsets) leaning both legs by phi toward the stance side, boards kept flat."""
    dq = np.zeros(10)
    s = 1.0 if stance_left else -1.0
    # left leg: hip roll -phi*s, ankle roll +ksign*phi*s; right leg mirrored (mirrored axes: same signs swapped)
    dq[LR] = -phi * s; dq[LA] = ksign * phi * s
    dq[RR] = +phi * s; dq[RA] = -ksign * phi * s
    return dq


FB = dict(ka=0.0, kad=0.0, kc=0.0, env=None, side=0, gate=0.0, dphi=0.0)


def com_correction():
    """Shift correction from the centre of mass relative to the stance board (simulator truth here; on the robot it
    comes from the joint encoders and the IMU tilt): lean less when the centre of mass is outboard of the board
    centre, more when inboard. Integrates kc rad/s per metre of error, clipped to +-0.08 rad."""
    env, side = FB["env"], FB["side"]
    if env is None or side == 0 or FB["kc"] == 0.0:
        return 0.0
    m, d = env.m, env.d
    com = (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    _, c = env.tip_low()
    b = c[0] if side > 0 else c[1]
    e = side * (com[1] - b[1])                       # > 0: centre of mass outboard of the stance board centre
    FB["dphi"] = float(np.clip(FB["dphi"] - FB["gate"] * FB["kc"] * e * DT, -0.08, 0.08))
    return FB["dphi"]


def feedback(obs):
    """Ankle strategy: both ankle rolls turn the legs (same physical direction: opposite numeric signs, the right
    axis is mirrored) by ka x IMU roll + kad x roll rate."""
    roll = math.asin(max(-1.0, min(1.0, float(obs[1]))))
    u = FB["ka"] * roll + FB["kad"] * float(obs[3])
    dq = np.zeros(10)
    dq[LA] += u; dq[RA] -= u
    return dq


def run(ctl, env, seconds, rec=False):
    obs = env.reset((0.0, 0.0, 0.0))
    x0 = env.d.qpos[0]
    out = dict(fell=None, lift=np.zeros(2), roll=[], trace=[])
    for k in range(int(seconds / DT)):
        t = k * DT
        obs, _ = env.step(np.clip(ctl(t, obs) + feedback(obs), -1, 1))
        low, c = env.tip_low()
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        roll = math.degrees(math.asin(np.clip(g[1], -1, 1)))
        out["lift"] = np.maximum(out["lift"], low)
        out["roll"].append(roll)
        if rec:
            R = env.d.geom_xmat[env.tip_gid].reshape(2, 3, 3)
            com = (env.m.body_mass[:, None] * env.d.xipos).sum(0) / env.m.body_mass.sum()
            out["trace"].append((t + DT, roll, *(1000 * low), *np.degrees(np.arcsin(R[:, 2, 1])), 1000 * (com[1] - c[0, 1]),
                                 1000 * (com[1] - c[1, 1])))
        if env.fallen():
            out["fell"] = t + DT
            break
    out["fwd"] = float(env.d.qpos[0] - x0)
    return out


def balance(a):
    def ctl(t, obs):
        phi = a.phi * ease((t - 0.5) / 2.0) * (1 - ease((t - 6.8) / 1.5))   # unshift once the foot is down
        up = ease((t - 3.0) / 0.8) * (1 - ease((t - 6.0) / 0.8))
        # put the foot down with `land` rad of knee still bent (the pelvis droops on the lifted side, so a fully
        # extended leg would land early and push the robot over the stance board), released during the unshift
        land = a.land * ease((t - 6.0) / 0.8) * (1 - ease((t - 6.8) / 1.5))
        dq = shift_pose(phi, True, a.ksign)
        dq[RK] += a.lift * up + land; dq[RP] += a.ratio * (a.lift * up + land)
        return dq
    return ctl


def walk(a):
    """Quasi-static: per step T: shift over the stance board (t_sh), lift + swing (t_sw), put down, unshift is the
    next step's shift to the other side. Fore-aft: swing foot -d/2 -> +d/2, stance foot +d/2 -> -d/2 (planar IK)."""
    hp0, kn0 = math.radians(16), math.radians(32)
    x0, z0, _ = leg_fk(hp0, kn0)

    def foot(dx):
        hp, kn = leg_ik(x0 + dx, z0, hp0, kn0)
        return hp - hp0, kn - kn0
    t_sh, t_up, t_sw, t_dn = a.t_shift, 0.5, a.t_swing, 0.5
    T = t_sh + t_up + t_sw + t_dn

    def ctl(t, obs):
        dq = np.zeros(10)
        t0 = 1.0
        if t < t0:
            return dq
        k = int((t - t0) // T)
        tau = (t - t0) - k * T
        if k >= a.steps:                              # done: shift back to the middle and stand
            prev_left = (k - 1) % 2 == 0
            phi = (a.phi + FB["dphi"]) * (1 - ease(tau / t_sh))
            FB["side"] = 0
            dq += shift_pose(phi, prev_left, a.ksign)
            dl = a.fwd if a.steps > 1 else a.fwd / 2
            for leg, x in ((0, -dl / 2 if prev_left else dl / 2), (1, dl / 2 if prev_left else -dl / 2)):
                hp, kn = foot(x * (1 - ease(tau / t_sh)))
                dq[2 + 5 * leg] += hp; dq[3 + 5 * leg] += kn
            return dq
        stance_left = k % 2 == 0                      # first step: right foot swings, left stance
        sw = 1 if stance_left else 0
        st = 1 - sw
        d = a.fwd if k > 0 else a.fwd / 2             # first step from side by side: half
        d_prev = 0.0 if k == 0 else (a.fwd / 2 if k == 1 else a.fwd)
        x_sw0 = -d_prev / 2                           # each step starts where the previous one left the feet
        x_st0 = d_prev / 2
        # shift: from the previous stance side (or the middle) to this stance side
        prev = None if k == 0 else (not stance_left)
        a_in = ease(tau / t_sh)
        if FB["side"] != (1 if stance_left else -1):
            FB["dphi"] = 0.0
        FB["side"] = 1 if stance_left else -1
        FB["gate"] = a_in                                    # correct only once the weight is coming across
        corr = com_correction()
        if prev is None:
            dq += shift_pose(a.phi * a_in + corr, stance_left, a.ksign)
        else:
            dq += shift_pose(a.phi * (1 - a_in), prev, a.ksign) + shift_pose(a.phi * a_in + corr, stance_left, a.ksign)
        up = ease((tau - t_sh) / t_up) * (1 - ease((tau - t_sh - t_up - t_sw) / t_dn))
        prog = ease((tau - t_sh - t_up) / t_sw)
        x_sw = x_sw0 + (d / 2 - x_sw0) * prog
        x_st = x_st0 + (-d / 2 - x_st0) * prog
        for leg, x in ((sw, x_sw), (st, x_st)):
            hp, kn = foot(x)
            dq[2 + 5 * leg] += hp; dq[3 + 5 * leg] += kn
        dq[3 + 5 * sw] += a.lift * up; dq[2 + 5 * sw] += a.ratio * a.lift * up
        # single support: keep the pelvis level with the stance hip roll (the stance board is flat, so the IMU roll
        # is the pelvis roll); the swing foot then lands at full extension instead of pushing the robot over
        if a.kh and up > 0:
            roll = math.asin(max(-1.0, min(1.0, float(obs[1]))))
            dq[1 + 5 * st] += a.kh * roll * (1 if st == 0 else -1) * up
        # touchdown with `land` rad of knee still bent on the swing leg, released over the next shift
        land_sw = a.land * ease((tau - t_sh - t_up - t_sw) / t_dn)
        land_st = a.land * (1 - ease(tau / 0.4)) if k > 0 else 0.0     # straighten before the weight comes on
        dq[3 + 5 * sw] += land_sw; dq[2 + 5 * sw] += a.ratio * land_sw
        dq[3 + 5 * st] += land_st; dq[2 + 5 * st] += a.ratio * land_st
        return dq
    return ctl, 1.0 + (a.steps + 1) * T + 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["balance", "walk"])
    ap.add_argument("--model", default="hw_pad_board150w_legs_ankle_10dof", choices=list(BASE))
    ap.add_argument("--phi", type=float, default=0.16, help="leg lean (rad) of the parallelogram shift")
    ap.add_argument("--ksign", type=float, default=1.0, help="ankle roll sign relative to the hip roll")
    ap.add_argument("--lift", type=float, default=0.4)
    ap.add_argument("--land", type=float, default=0.0, help="knee bend (rad) left in at touchdown, released after")
    ap.add_argument("--ratio", type=float, default=1.0, help="hip pitch per knee in the lift (1 = board stays level)")
    ap.add_argument("--fwd", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--t-shift", type=float, default=1.5)
    ap.add_argument("--t-swing", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--ka", type=float, default=0.0, help="ankle feedback, rad per rad of IMU roll")
    ap.add_argument("--kad", type=float, default=0.0, help="ankle feedback, rad per rad/s of roll rate")
    ap.add_argument("--kh", type=float, default=0.0, help="stance hip roll per rad of IMU roll in single support")
    ap.add_argument("--kc", type=float, default=0.0, help="centre-of-mass correction of the shift, rad/s per metre")
    a = ap.parse_args()
    FB.update(ka=a.ka, kad=a.kad, kc=a.kc)
    for tc in (0.004, 0.1, 0.3, 0.8):
        res = []
        for s in range(a.seeds):
            if a.mode == "balance":
                ctl, secs = balance(a), 10.0
            else:
                ctl, secs = walk(a)
            env = make_env(a.model, tc, s, s > 0)
            FB.update(env=env, side=0, dphi=0.0)
            r = run(ctl, env, secs, rec=a.trace and s == 0)
            res.append(r)
            if a.trace and s == 0 and tc == 0.004:
                for row in r["trace"][::10]:
                    print("   t %.1f roll %+5.1f low L/R %5.1f/%5.1f board roll L/R %+5.1f/%+5.1f CoM-y vs L/R board %+6.1f/%+6.1f" % row)
        up = sum(r["fell"] is None for r in res)
        print(f"floor {tc}: up {up}/{len(res)}  max lift L/R {np.median([1000 * r['lift'][0] for r in res]):5.1f}/"
              f"{np.median([1000 * r['lift'][1] for r in res]):5.1f} mm  |roll| max {np.median([np.max(np.abs(r['roll'])) for r in res]):4.1f} deg"
              f"  fwd {100 * np.median([r['fwd'] for r in res]):+5.1f} cm  falls at {[r['fell'] for r in res if r['fell']]}", flush=True)


if __name__ == "__main__":
    main()
