"""Which unmodelled effect makes the simulator reproduce the hardware failure of hw_stand_push2?

    .venv-hw/bin/python scripts/gap_probe.py [--n 16]

On the robot (2026-11-06, 5 V, 1 s blend-in, started from the commanded stance) hw_stand_push2 fell in ~1.1 s every
time. On the fitted servo model with the same start and blend-in it survives 20 s in ~40 % of randomized runs, so the
model is still optimistic. Each variant below adds one candidate effect to the nominal fitted 5 V model and reports how
fast the policy falls. Sensor delay is applied here by feeding the policy the observation from k control steps ago.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "hardware/leg_p0/software")]
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

BASE = dict(prof_acc=(7.0, 35.0), kp=8.0, latency_sub=5, strength_scale=0.68, speed_scale=0.68,
            reset_joint_noise=0.02, reset_tilt=0.02, dr=False, push_vel=0.0)   # quiet: no shoves
VARIANTS = {
    "fitted model (nominal)": ({}, 0),
    "+ sensor delay 20 ms": ({}, 1),
    "+ sensor delay 40 ms": ({}, 2),
    "+ command dead time 40 ms": (dict(delay_steps=2), 0),
    "+ command dead time 60 ms": (dict(delay_steps=3), 0),
    "+ hip profile x0.6": (dict(prof_acc=(4.2, 35.0)), 0),
    "+ servo kp 4": (dict(kp=4.0), 0),
    "+ backlash 1 deg": (dict(deadzone=0.0175), 0),
    "+ pad friction 0.4": (dict(friction=0.4), 0),
    "+ sensor 20 ms + dead time 40 ms": (dict(delay_steps=2), 1),
}


def episode(args):
    run, name, seed, secs = args
    import policy_stand as ps
    pol = ps.Policy(run)
    over, obs_delay = VARIANTS[name]
    cfg = EvalConfig.from_train(pol.env, **dict(BASE, **over))
    env = CpuBiped("hw_p0_8dof", cfg, seed=seed)
    obs = env.reset()
    hist = [obs.copy()] * (obs_delay + 1)
    for k in range(int(secs / cfg.ctrl_dt)):
        seen = hist[0].copy()
        a = np.clip(pol(seen), -1, 1)
        obs, _ = env.step(min(1.0, k * cfg.ctrl_dt) * a)        # 1 s blend-in, like the robot runner
        obs[22:30] = a
        hist = hist[1:] + [obs.copy()]
        if env.fallen():
            return k * cfg.ctrl_dt
    return secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hw_stand_push2")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--secs", type=float, default=10.0)
    a = ap.parse_args()
    print(f"{a.run}: robot (5 V) fell at ~1.1 s every time")
    with mp.Pool(max(1, mp.cpu_count() - 3)) as pool:
        for name in VARIANTS:
            t = np.array(pool.map(episode, [(a.run, name, 3000 + s, a.secs) for s in range(a.n)]))
            print(f"  {name:36s} standing at {a.secs:.0f} s {100 * np.mean(t >= a.secs):3.0f} %   median fall/end "
                  f"{np.median(t):4.1f} s", flush=True)


if __name__ == "__main__":
    main()
