"""Does a stand policy take steps? Sim (fitted 5 V servos) vs the robot's logs.

    .venv-hw/bin/python scripts/step_probe.py hw_stand_fasthip5v hw_stand_fit5v_p005 [--logs 'hardware/leg_p0/software/stand_hw_stand_fasthip5v_*.csv']

Scissor = left/right hip-pitch commands more than 4 deg apart (the policy swinging one leg against the other).
In sim also counts the time a pad is >2 mm off the floor.
"""
from __future__ import annotations
import argparse, glob, multiprocessing as mp, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "hardware/leg_p0/software")]
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

BASE = dict(prof_acc=(7.0, 35.0), kp=8.0, latency_sub=5, strength_scale=0.68, speed_scale=0.68,
            reset_joint_noise=0.02, reset_tilt=0.02, dr=False)
SCISSOR = np.radians(4.0)


def episode(args):
    name, push, seed = args
    import policy_stand as ps
    pol = ps.Policy(name)
    cfg = EvalConfig.from_train(pol.env, **BASE, push_vel=push)
    env = CpuBiped("hw_p0_8dof", cfg, seed=seed)
    obs = env.reset()
    n = sc = air = 0
    for k in range(int(20.0 / cfg.ctrl_dt)):
        a = np.clip(pol(obs), -1, 1)
        obs, _ = env.step(min(1.0, k * cfg.ctrl_dt) * a)
        obs[22:30] = a
        q = env.prev_cmd
        n += 1
        sc += abs(q[2] - q[6]) > SCISSOR
        low, _ = env.tip_low()
        air += low.max() > 0.002
        if env.fallen():
            return n * cfg.ctrl_dt, sc / n, air / n, 0
    return n * cfg.ctrl_dt, sc / n, air / n, 1


def hw(pattern):
    J = ["cmd_L_hip_pitch", "cmd_R_hip_pitch"]
    tot = scs = 0
    for f in sorted(glob.glob(pattern)):
        d = np.genfromtxt(f, delimiter=",", names=True)
        s = np.abs(d[J[0]] - d[J[1]]) > SCISSOR
        tot += len(s); scs += s.sum()
    return scs / max(tot, 1), tot * 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--logs")
    a = ap.parse_args()
    with mp.Pool(max(1, mp.cpu_count() - 3)) as pool:
        for r in a.runs:
            for push in (0.0, 0.05):
                res = np.array(pool.map(episode, [(r, push, 7000 + s) for s in range(a.n)]))
                print(f"sim {r:24s} push {push:.2f}: standing {100*res[:,3].mean():3.0f}%  scissor {100*res[:,1].mean():4.1f}% "
                      f"of time  pad in air {100*res[:,2].mean():4.1f}%", flush=True)
    if a.logs:
        f, secs = hw(a.logs)
        print(f"robot logs ({secs:.0f} s): scissor {100*f:.1f}% of time")


if __name__ == "__main__":
    main()
