"""Robot-round-4 screening (2026-11-12): the robot-like evaluator of walking_clean.md §13-14 with the encoder-invisible
hip give calibrated on the E1 tilt test (measuring.md), on the model with the pelvis shell.

    .venv/bin/python scripts/robot4_eval.py <run> [<run> ...] [--params P] [--vx 0.015 0.03] [--flex 4 5] [--n 6]

Per run, flex and speed (n episodes x 30 s, 5 V, 10 ms more command delay at random): falls, forward distance per
episode, mean step length (forward travel / touchdowns), time with |pelvis roll| > 4 deg (where the real feet start to
slip inward), peak |roll|, and the board-centre gap (min, end; it starts at 81 mm). Steadier = fewer falls, less time
past 4 deg, gap kept.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, quat_rot_t, yaw_of  # noqa: E402

MODEL = "hw_ank3_b150_legs_pelvis_10dof"


def episode(args):
    run, params, flex, vx, seed, seconds, model, extra = args
    import mujoco
    cfg, rd = load_run(ROOT / "runs" / run)
    pp = Path(params) if params else rd / "params.pkl"
    if not pp.exists():
        pp = sorted(rd.glob("params_*M.pkl"))[-1]
    pol = NumpyPolicy(pp)
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0,
                dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
                dr_kp=(0.75, 1.25), kp_hip_roll=2.5, hidden_flex=((("hip_roll", float(flex)),) if flex else ()))
    over.update(extra)
    env = CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=91000 + seed)
    obs = env.reset((vx, 0, 0)); y0 = yaw_of(env.d.qpos[3:7])
    tid = [mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_tip") for s in "LR"]
    x0 = env.d.qpos[0:2].copy(); fwd = np.array([math.cos(y0), math.sin(y0)])
    contact_prev = np.array([True, True]); td = 0; over4 = 0; peak = 0.0; gaps = []
    n = int(seconds / 0.02); fell = None
    for k in range(n):
        env.cmd[2] = float(np.clip((y0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi, -.3, .3))
        obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        roll = abs(math.degrees(math.atan2(g[1], -g[2])))
        over4 += roll > 4.0; peak = max(peak, roll)
        low, _ = env.tip_low(); c = low < 0.0015
        td += int(np.sum(c & ~contact_prev)); contact_prev = c
        p = env.d.geom_xpos[tid]; yv = quat_rot_t(env.d.qpos[3:7], p[0] - p[1])[1]
        gaps.append(1000 * abs(yv))
        if env.fallen():
            fell = k * 0.02; break
    dist = float((env.d.qpos[0:2] - x0) @ fwd)
    return dict(fell=fell, dist=dist, step=dist / max(td, 1), td=td, over4=over4 / (k + 1), peak=peak,
                gap_min=min(gaps), gap_end=float(np.median(gaps[-50:])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--params")
    ap.add_argument("--vx", type=float, nargs="+", default=[0.015, 0.03])
    ap.add_argument("--flex", type=float, nargs="+", default=[4.0, 5.0])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--set", nargs="*", default=[], help="EvalConfig overrides k=python-literal")
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()
    import ast
    extra = {k: ast.literal_eval(v) for k, v in (x.split("=", 1) for x in a.set)}
    jobs, keys = [], []
    for r in a.runs:
        rr, _, pp = r.partition(":")
        for fl in a.flex:
            for vx in a.vx:
                keys.append((r, fl, vx))
                jobs += [(rr, (str(ROOT / "runs" / rr / pp) if pp else a.params), fl, vx, s, a.seconds, a.model, extra) for s in range(a.n)]
    with Pool(min(12, len(jobs))) as p:
        res = p.map(episode, jobs)
    for i, (r, fl, vx) in enumerate(keys):
        R = res[i * a.n:(i + 1) * a.n]
        falls = [x["fell"] for x in R if x["fell"] is not None]
        m = lambda k: float(np.mean([x[k] for x in R]))          # noqa: E731
        print(f"{r:28s} k={fl:<4g} vx={vx:<6g} falls {len(falls)}/{a.n} {'(' + ','.join(f'{f:.0f}' for f in falls) + ' s)' if falls else '':16s}"
              f" fwd {100 * m('dist'):5.1f} cm  step {1000 * m('step'):4.1f} mm  |roll|>4: {100 * m('over4'):4.1f}%  peak {m('peak'):4.1f}"
              f"  gap min {m('gap_min'):3.0f} end {m('gap_end'):3.0f} mm", flush=True)


if __name__ == "__main__":
    main()
