"""Trace one walk of a policy: every `every` s, phase, roll / pitch, lowest point of each foot, pelvis x / y.

    .venv/bin/python scripts/policy_trace.py <run> [--params p.pkl] [--vx 0.008] [--seconds 12] [--floor-tc 0.004] [--seed 0]
"""
import argparse, math, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, quat_rot_t  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--params"); ap.add_argument("--vx", type=float, default=0.008)
ap.add_argument("--seconds", type=float, default=12); ap.add_argument("--floor-tc", type=float, default=0.004)
ap.add_argument("--seed", type=int, default=70000); ap.add_argument("--every", type=float, default=0.2); ap.add_argument("--mu", type=float)
a = ap.parse_args()
cfg, rd = load_run(a.run)
pol = NumpyPolicy(rd / (a.params or sorted(rd.glob("params_*M.pkl"))[-1].name)) if a.params != "zero" else (lambda o: np.zeros(10))
env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], floor_tc=a.floor_tc, push_vel=0.0, imu_filter=True,
                                                          imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0, **({"friction": a.mu} if a.mu else {})), seed=a.seed)
obs = env.reset((a.vx, 0.0, 0.0)); dt = env.cfg.ctrl_dt; x0 = env.d.qpos[0]
for k in range(int(a.seconds / dt)):
    obs, _ = env.step(pol(obs))
    g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
    low, c = env.tip_low()
    if k % max(1, int(a.every / dt)) == 0 or env.fallen():
        u = (env.phase / (2 * math.pi)) % 1 if hasattr(env, "phase") else float("nan")
        print(f"t {k * dt:5.2f} u {u:.3f} roll {math.degrees(math.asin(g[1])):+5.1f} pitch {math.degrees(math.asin(g[0])):+5.1f} "
              f"low L {1000 * low[0]:5.1f} R {1000 * low[1]:5.1f} pelvis x {1000 * (env.d.qpos[0] - x0):+6.1f} y {1000 * env.d.qpos[1]:+5.1f}"
              + ("  FALL" if env.fallen() else ""))
    if env.fallen():
        break
