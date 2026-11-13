"""Fit the simulator to the robot's walks (2026-11-12, walking_clean.md §17): run policies in the robot-like
evaluator (robot4_eval's settings, hip give k = 4) with extra EvalConfig overrides and report what the robot logs
show, measured the same way (encoders + IMU, robot_walk_stats): time to the fall, fall direction, walking pitch,
encoder step reach per foot.

    .venv/bin/python scripts/gap_fit.py s10_robust c3m_T12 --set 'hidden_off=[["knee",-0.045]]' drag_back=2 --n 8
    .venv/bin/python scripts/gap_fit.py --logs hardware/leg_p0/software/stand_s10_robust_20261112_10*.csv

Robot 2026-11-12 (vx 0.015, 4 trials each): mean time to fall s7 18 s, s8 15, s10 24, c3m 14; 11/16 falls backward;
walking pitch +1 to +3 deg back, creeping to +4..+6 before the fall; reach L > R (s10 12 / 6 mm).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

MODEL = "hw_ank3_b150_legs_pelvis_10dof"
_FK = {}


def _fk_setup(model):
    import mujoco
    if model not in _FK:
        m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"{model}.xml"))
        _FK[model] = (m, mujoco.MjData(m), [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_tip") for s in "LR"],
                      json.loads((ROOT / "models" / f"{model}.json").read_text())["joints"])
    return _FK[model]


def _quat_from_g(g):
    import mujoco
    z = -g / np.linalg.norm(g)
    x = np.array([1.0, 0, 0]) - z * z[0]; x /= np.linalg.norm(x)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.vstack([x, np.cross(z, x), z]).flatten()); return q


def walk_stats(t, g, q, model=MODEL):
    """Encoder + IMU walk statistics, as from a robot log. g: gravity in the pelvis frame; q: encoder joint angles."""
    import mujoco
    m, d, tid, _ = _fk_setup(model)
    P = np.zeros((len(t), 2, 3))
    for i in range(len(t)):
        d.qpos[:] = 0; d.qpos[2] = 0.3; d.qpos[3:7] = _quat_from_g(g[i]); d.qpos[7:] = q[i]
        mujoco.mj_kinematics(m, d); P[i] = d.geom_xpos[tid]
    P *= 1000
    dz = P[:, 0, 2] - P[:, 1, 2]; dx = P[:, 0, 0] - P[:, 1, 0]
    sw, i, n = [], 0, len(t)
    while i < n:
        if abs(dz[i]) > 5:
            s = np.sign(dz[i]); j = i
            while j < n and np.sign(dz[j]) == s and abs(dz[j]) > 1.5: j += 1
            i0 = i
            while i0 > 0 and abs(dz[i0]) > 1.5 and np.sign(dz[i0]) == s: i0 -= 1
            if j < n and np.max(np.abs(dz[i0:j])) > 8:
                sw.append(("L" if s > 0 else "R", (dx[j] - dx[i0]) * s / 2))
            i = j
        else:
            i += 1
    roll = np.degrees(np.arctan2(g[:, 1], -g[:, 2])); pitch = np.degrees(np.arctan2(-g[:, 0], -g[:, 2]))
    walk = t > 2.4
    k = len(t) - 1
    while k > 0 and max(abs(roll[k]), abs(pitch[k])) > 8: k -= 1
    early = walk & (t < 7.2)
    late = (t > t[k] - 4.8) & (t < t[k]) & walk
    med = lambda a, s: float(np.median(a[s])) if s.any() else float("nan")   # noqa: E731
    return dict(T=float(t[-1]), steps=len(sw), reachL=float(np.median([r for f, r in sw if f == "L"] or [np.nan])),
                reachR=float(np.median([r for f, r in sw if f == "R"] or [np.nan])), pitch_early=med(pitch, early),
                pitch_late=med(pitch, late), roll_end=float(roll[-1]), pitch_end=float(pitch[-1]))


def fall_dir(r, fell):
    if not fell:
        return "-"
    rr, pp = r["roll_end"], r["pitch_end"]
    if abs(pp) >= abs(rr) * 0.8:
        return "back" if pp > 0 else "fwd"
    return "left" if rr > 0 else "right"


def log_stats(path):
    rows = list(csv.reader(open(path))); h = rows[0]; A = np.array(rows[1:], float)
    c = lambda ks: A[:, [h.index(k) for k in ks]]   # noqa: E731
    _, _, _, joints = _fk_setup(MODEL)
    r = walk_stats(A[:, 0], c(["gx", "gy", "gz"]), c([f"q_{n}" for n in joints]))
    return r, math.hypot(r["roll_end"], r["pitch_end"]) > 15


def episode(a):
    run, seed, vx, seconds, model, extra = a
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, quat_rot_t, yaw_of
    rr, _, pp = run.partition(":")
    cfg, rd = load_run(ROOT / "runs" / rr)
    p = rd / pp if pp else (rd / "params.pkl" if (rd / "params.pkl").exists() else sorted(rd.glob("params_*M.pkl"))[-1])
    pol = NumpyPolicy(p)
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0,
                dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
                dr_kp=(0.75, 1.25), kp_hip_roll=2.5, hidden_off=(), drag_back=0.0, hidden_flex=(("hip_roll", 4.0),))
    over.update(extra)
    env = CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=55000 + seed)
    obs = env.reset((vx, 0, 0)); y0 = yaw_of(env.d.qpos[3:7]); x0 = env.d.qpos[0:2].copy()
    T, G, Q = [], [], []
    for i in range(int(seconds / 0.02)):
        env.cmd[2] = float(np.clip((y0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi, -.3, .3))
        obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        T.append(i * 0.02); G.append(quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0])))
        Q.append(env.d.qpos[7:] + env.flex + env.q_off)
        if env.fallen():
            break
    r = walk_stats(np.array(T), np.array(G), np.array(Q), model)
    fwd = np.array([math.cos(y0), math.sin(y0)])
    r["dist"] = float((env.d.qpos[0:2] - x0) @ fwd)
    return run, r, env.fallen()


def report(name, R):
    falls = [r for r, f in R if f]
    dirs = {}
    for r, f in R:
        dd = fall_dir(r, f); dirs[dd] = dirs.get(dd, 0) + 1
    m = lambda k: np.nanmean([r[k] for r, _ in R])   # noqa: E731
    dist = f" fwd {100 * m('dist'):5.1f} cm" if "dist" in R[0][0] else ""
    print(f"{name:34s} falls {len(falls)}/{len(R)} T {m('T'):5.1f} s  {' '.join(f'{k}:{v}' for k, v in sorted(dirs.items()) if k != '-'):18s}"
          f" pitch early {m('pitch_early'):+5.1f} late {m('pitch_late'):+5.1f}  reach L {m('reachL'):5.1f} R {m('reachR'):5.1f} mm{dist}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--vx", type=float, default=0.015)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    if a.logs:
        groups = {}
        for f in a.logs:
            groups.setdefault(Path(f).stem[6:-16], []).append(log_stats(f))
        for k, R in groups.items():
            report(f"robot {k}", R)
    if a.runs:
        extra = {k: ast.literal_eval(v) for k, v in (x.split("=", 1) for x in a.set)}
        jobs = [(r, s, a.vx, a.seconds, a.model, extra) for r in a.runs for s in range(a.n)]
        with Pool(min(12, len(jobs))) as p:
            res = p.map(episode, jobs)
        for r in a.runs:
            report(f"sim {r} {' '.join(a.set)}"[:34], [(x, f) for rr, x, f in res if rr == r])


if __name__ == "__main__":
    main()
