"""Passive screen of foot-pad designs: the robot holds its stance pose (no balance control) and is set down tilted on
a floor; the largest tilt it still rights itself from tells how much the pads alone give.

    .venv-hw/bin/python scripts/pad_passive.py [designs ...]
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

TILTS = [1, 2, 3, 4, 6, 8, 10]
FLOORS = {"hard": 0.004, "rug1": 0.1, "rug2": 0.2, "rug3": 0.3}
DIRS = 8


def stands(args):
    model, floor, tilt, k = args
    cfg = EvalConfig(prof_acc=(7.0, 35.0, 35.0), strength_scale=0.68, speed_scale=0.68, latency_sub=5,
                     reset_joint_noise=0.0, reset_tilt=0.0, floor_tc=FLOORS[floor], floor_damp=1.0, friction=0.8)
    env = CpuBiped(model, cfg, seed=k)
    env.reset()
    m, d = env.m, env.d
    a = 2 * math.pi * k / DIRS
    th = math.radians(tilt)
    d.qpos[3:7] = [math.cos(th / 2), math.sin(th / 2) * math.cos(a), math.sin(th / 2) * math.sin(a), 0]
    mujoco.mj_kinematics(m, d)
    low, _ = env.tip_low()
    d.qpos[2] += -low.min() + 0.0005
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    for _ in range(int(4.0 / cfg.ctrl_dt)):
        env.step(np.zeros(8))
        if env.fallen():
            return 0.0
    return 1.0


def main():
    names = sys.argv[1:] or ["orig", "board60", "board90", "board120", "board90w", "disc50", "disc70", "dome50",
                             "dome70", "claw45", "claw60", "talon45", "talon60"]
    jobs = [(f"hw_pad_{n}_8dof", f, t, k) for n in names for f in FLOORS for t in TILTS for k in range(DIRS)]
    with mp.Pool() as p:
        r = p.map(stands, jobs)
    res = {}
    for j, v in zip(jobs, r):
        res.setdefault(j[:3], []).append(v)
    print(f"passive (stance held, no balance control): fraction of {DIRS} tilt directions that stay up 4 s")
    print(f"{'design':10s} {'floor':5s} " + " ".join(f"{t:>4d}°" for t in TILTS))
    for n in names:
        for f in FLOORS:
            print(f"{n:10s} {f:5s} " + " ".join(f"{100 * np.mean(res[(f'hw_pad_{n}_8dof', f, t)]):4.0f}%" for t in TILTS))


if __name__ == "__main__":
    main()
