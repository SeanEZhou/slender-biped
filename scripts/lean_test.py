"""How much initial lean can each stand controller recover from? (fitted 5 V servos, quiet, 1 s blend-in)

    .venv-hw/bin/python scripts/lean_test.py hw_stand_fit5v_p005 hw_stand_fasthip5v hw_stand_fit5v_robust balancer

On the robot the IMU read +3 to +7 deg of forward lean at the moment the policy engaged (2026-11-06 logs). Each run
starts in the stance pose with the whole robot tilted by a fixed pitch (+ = forward), pads on the floor, and counts
the fraction still standing after 20 s.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "hardware/leg_p0/software")]
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

BASE = dict(prof_acc=(7.0, 35.0), kp=8.0, latency_sub=5, strength_scale=0.68, speed_scale=0.68,
            reset_joint_noise=0.02, reset_tilt=0.0, dr=False, push_vel=0.0)
LEANS = [-4, -2, 0, 2, 4, 6]


def episode(args):
    name, lean, seed = args
    import policy_stand as ps
    ctrl = ps.Balancer() if name == "balancer" else ps.Policy(name)
    cfg = EvalConfig.from_train(ctrl.env, **BASE)
    env = CpuBiped("hw_p0_8dof", cfg, seed=seed)
    env.reset()
    m, d = env.m, env.d
    th = math.radians(lean)                    # pitch about +y: + leans forward
    d.qpos[3:7] = [math.cos(th / 2), 0.0, math.sin(th / 2), 0.0]
    d.qvel[:] = 0
    mujoco.mj_kinematics(m, d)
    low, _ = env.tip_low()
    d.qpos[2] += -low.min() + 0.0005
    mujoco.mj_forward(m, d)
    obs = env.obs()
    rl = name != "balancer"
    for k in range(int(20.0 / cfg.ctrl_dt)):
        a = np.clip(ctrl(obs), -1, 1) if rl else ctrl(obs)
        obs, _ = env.step(min(1.0, k * cfg.ctrl_dt) * a if rl else a)
        if rl:
            obs[22:30] = a
        if env.fallen():
            return 0.0
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrls", nargs="+")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    print(f"standing at 20 s by initial lean (deg, + forward), {a.n} runs each:")
    print(f"  {'controller':22s}" + "".join(f"{x:>7d}" for x in LEANS))
    with mp.Pool(max(1, mp.cpu_count() - 3)) as pool:
        for c in a.ctrls:
            row = []
            for lean in LEANS:
                r = pool.map(episode, [(c, lean, 4000 + s) for s in range(a.n)])
                row.append(100 * np.mean(r))
            print(f"  {c:22s}" + "".join(f"{v:6.0f}%" for v in row), flush=True)


if __name__ == "__main__":
    main()
