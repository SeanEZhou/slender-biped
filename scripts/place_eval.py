"""Hand-placement test for stand policies (simulated): set down tilted and already tipping, on hard and soft floors.

    .venv-hw/bin/python scripts/place_eval.py hw_recover_p191 hw_recover_soft [--n 16] [--volt 0.68]

Each start: tilt of a fixed size in a random direction, tipping at 0.3 rad/s std per axis (like letting go by hand),
policy at full authority from the first step (`policy_stand.py run --ramp 0`) or blended in over --ramp s.
Floors: hard (solref 0.004), medium mat (0.03) and soft mat (0.08), friction 0.8. Score = fraction standing at 8 s.
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
sys.path[:0] = [str(ROOT), str(ROOT / "hardware/leg_p0/software")]
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

TILTS = [3, 5, 7, 9]
FLOORS = {"hard": 0.004, "mat": 0.03, "soft": 0.08}
RUGS = {"rug1": 0.1, "rug2": 0.2, "rug3": 0.3}   # original 31 x 18 mm pad sinks ~3 / ~9 / ~8+ mm (scaled by pad area)


def episode(args):
    name, tilt_deg, floor, volt, ramp, seed, imu, floors, zero = args
    import policy_stand as ps
    pol = ps.Policy(name)
    e = pol.env
    cfg = EvalConfig.from_train(e, prof_acc=tuple(e["prof_acc"]), kp=8.0, latency_sub=5, strength_scale=volt,
                                speed_scale=volt, reset_joint_noise=0.02, reset_tilt=0.0, dr=False, push_vel=0.0,
                                friction=0.8, floor_tc=floors[floor],
                                **(dict(imu_filter=True, imu_gate=0.15, gyro_range=17.45) if imu else {}))
    env = CpuBiped(e["model"], cfg, seed=seed)
    env.reset()
    rng = np.random.default_rng(1000 + seed)
    m, d = env.m, env.d
    a = rng.uniform(-math.pi, math.pi)
    th = math.radians(tilt_deg)
    ax = np.array([math.cos(a), math.sin(a), 0.0])
    d.qpos[3:7] = [math.cos(th / 2), *(math.sin(th / 2) * ax)]
    mujoco.mj_kinematics(m, d)
    low, _ = env.tip_low()
    d.qpos[2] += -low.min() + 0.0005
    w = np.array([*(0.3 * rng.normal(size=2)), 0.0])
    d.qvel[:] = 0
    d.qvel[0:3] = np.cross(w, [0, 0, d.qpos[2]])
    mujoco.mj_forward(m, d)
    d.qvel[3:6] = d.xmat[1].reshape(3, 3).T @ w
    mujoco.mj_forward(m, d)
    env.sync_imu()
    obs = env.obs()
    for k in range(int(8.0 / cfg.ctrl_dt)):
        act = np.zeros(8) if zero else np.clip(pol(obs), -1, 1)
        s = min(1.0, k * cfg.ctrl_dt / ramp) if ramp > 0 else 1.0
        obs, _ = env.step(s * act)
        obs[22:30] = act
        if env.fallen():
            return 0.0
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--volt", type=float, default=0.68)
    ap.add_argument("--ramp", type=float, default=0.0)
    ap.add_argument("--rugs", action="store_true", help="rug floors instead of hard / mat / soft")
    ap.add_argument("--imu", action="store_true", help="policy sees the firmware IMU filter (new firmware settings)")
    ap.add_argument("--zero", action="store_true", help="no policy: servos just hold the stance pose (the pad alone)")
    a = ap.parse_args()
    floors = RUGS if a.rugs else FLOORS
    jobs = [(r, t, f, a.volt, a.ramp, s, a.imu, floors, a.zero) for r in a.runs for f in floors for t in TILTS for s in range(a.n)]
    with mp.Pool() as pool:
        res = pool.map(episode, jobs)
    out = {}
    for j, v in zip(jobs, res):
        out.setdefault(j[:3], []).append(v)
    print(f"standing at 8 s, {a.n} starts each, volt {a.volt}, ramp {a.ramp} s, imu filter {a.imu}{', NO POLICY (stance held)' if a.zero else ''}")
    print(f"{'run':24s} {'floor':5s} " + " ".join(f"{t:>4d}deg" for t in TILTS))
    for r in a.runs:
        for f in floors:
            print(f"{r:24s} {f:5s} " + " ".join(f"{100 * np.mean(out[(r, t, f)]):6.0f}%" for t in TILTS))


if __name__ == "__main__":
    main()
