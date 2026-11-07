"""Recovery tests for stand policies on the fitted servo model (simulated).

    .venv-hw/bin/python scripts/recover_eval.py hw_recover_slowhip hw_recover_p191 [--n 12] [--volt 0.68]

Each policy runs on the model and servo profile it was trained for (its config: model, prof_acc), with the
fitted 5 V servos (--volt 0.68; 1.0 for 7.4 V), a hardware-like start and a 1 s blend-in. Three tests, each the
fraction still standing after 10 s:
  lean  - placed tilted forward/back by a fixed pitch (+ = forward)
  kick  - placed upright, then the whole robot given a fore-aft speed (m/s, + = forward) at t = 1.5 s
  shove - random 2-D shoves (std m/s per axis) every 2-4 s for 20 s
Also counts recovery steps (touchdowns per run) in the kick test.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "hardware/leg_p0/software")]
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

LEANS = [-6, -4, -2, 2, 4, 6]
KICKS = [-0.2, -0.15, -0.1, 0.1, 0.15, 0.2]
SHOVES = [0.05, 0.1, 0.15]


def make(name, volt, push, seed):
    import policy_stand as ps
    pol = ps.Policy(name)
    env_cfg = pol.env
    cfg = EvalConfig.from_train(env_cfg, prof_acc=tuple(env_cfg["prof_acc"]), kp=8.0, latency_sub=5,
                                strength_scale=volt, speed_scale=volt, reset_joint_noise=0.02, reset_tilt=0.02,
                                dr=False, push_vel=push)
    env = CpuBiped(env_cfg["model"], cfg, seed=seed)
    return pol, cfg, env


def run(pol, cfg, env, obs, seconds, on_step=None):
    contact_prev = np.ones(2, bool)
    touchdowns = 0
    for k in range(int(seconds / cfg.ctrl_dt)):
        a = np.clip(pol(obs), -1, 1)
        obs, _ = env.step(min(1.0, k * cfg.ctrl_dt) * a)
        obs[22:30] = a
        if on_step:
            on_step(k)
        low, _ = env.tip_low()
        contact = low < 0.002
        touchdowns += int(np.sum(contact & ~contact_prev))
        contact_prev = contact
        if env.fallen():
            return 0.0, touchdowns
    return 1.0, touchdowns


def episode(args):
    name, test, x, volt, seed = args
    pol, cfg, env = make(name, volt, x if test == "shove" else 0.0, seed)
    obs = env.reset()
    m, d = env.m, env.d
    if test == "lean":
        th = math.radians(x)
        d.qpos[3:7] = [math.cos(th / 2), 0.0, math.sin(th / 2), 0.0]
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        low, _ = env.tip_low()
        d.qpos[2] += -low.min() + 0.0005
        mujoco.mj_forward(m, d)
        obs = env.obs()
        return run(pol, cfg, env, obs, 10.0)
    if test == "kick":
        def kick(k):
            if k == int(1.5 / cfg.ctrl_dt):
                d.qvel[0] += x
        return run(pol, cfg, env, obs, 10.0, kick)
    return run(pol, cfg, env, obs, 20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--volt", type=float, default=0.68)
    ap.add_argument("--out", default=str(ROOT / "results/recover_eval.json"))
    a = ap.parse_args()
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    with mp.Pool(max(1, mp.cpu_count() - 3)) as pool:
        for r in a.runs:
            key = f"{r}@{a.volt}"
            res[key] = {}
            for test, xs in (("lean", LEANS), ("kick", KICKS), ("shove", SHOVES)):
                row, steps = [], []
                for x in xs:
                    rr = np.array(pool.map(episode, [(r, test, x, a.volt, 5000 + s) for s in range(a.n)]))
                    row.append(100 * rr[:, 0].mean())
                    steps.append(rr[:, 1].mean())
                res[key][test] = dict(x=xs, standing=row, touchdowns=steps)
                print(f"{r:22s} V{a.volt:.2f} {test:5s} " + "  ".join(f"{x:+g}:{v:3.0f}%" for x, v in zip(xs, row))
                      + ("   steps/run " + " ".join(f"{s:.1f}" for s in steps) if test == "kick" else ""), flush=True)
            out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
