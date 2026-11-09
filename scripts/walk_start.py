"""First steps of a walking policy as started on the robot (2026-11-08 hardware trials: ~70 % of falls on step 1).

    .venv/bin/python scripts/walk_start.py b150_walkstride_L --vx 0.16 --ramp 1.0 --n 20

Starts in the stance pose, upright and still (as after the runner's Enter), on a rug, with the robot's IMU filter,
servo/mass randomization per seed, and optionally the runner's --ramp blend (reference and action scaled 0 -> 1 over
`ramp` s; training never had it). Reports falls within `seconds` and the tilt direction when 10 deg is first passed.
"""
import argparse, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import math  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, quat_rot_t, yaw_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--params", default=None)
ap.add_argument("--vx", type=float, default=0.08)
ap.add_argument("--ramp", type=float, default=0.0)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--floor-tc", type=float, default=0.2)
ap.add_argument("--no-imu", action="store_true")
ap.add_argument("--quiet", action="store_true")
ap.add_argument("--tilt", default="0,0", help="pitch,roll deg: the robot really leans this way at engage (+ = forward, left)")
ap.add_argument("--set", nargs="*", default=[], help="EvalConfig overrides k=json, e.g. strength_scale=0.7")
ap.add_argument("--imu-bias", default="0,0", help="pitch,roll deg the IMU reads while the robot is upright")
a = ap.parse_args()
cfg, rd = load_run(a.run)
params = a.params or sorted(rd.glob("params_*M.pkl"))[-1].name
pol = NumpyPolicy(rd / params)
over = dict(floor_tc=a.floor_tc, reset_joint_noise=0.0, reset_tilt=0.0)
import json  # noqa: E402
for kv in a.set:
    k, v = kv.split("=", 1)
    v = json.loads(v)
    over[k] = tuple(v) if isinstance(v, list) else v
if not a.no_imu:
    over.update(imu_filter=True, imu_gate=0.15, gyro_range=17.45)
tp, tr = (math.radians(float(x)) for x in a.tilt.split(","))
bp, br = (math.radians(float(x)) for x in a.imu_bias.split(","))
Rb = (np.array([[1, 0, 0], [0, math.cos(br), -math.sin(br)], [0, math.sin(br), math.cos(br)]])
      @ np.array([[math.cos(-bp), 0, math.sin(-bp)], [0, 1, 0], [-math.sin(-bp), 0, math.cos(-bp)]]))
falls, first, dirs = [], [], []
for s in range(a.n):
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=30000 + s)
    obs = env.reset((a.vx, 0.0, 0.0))
    if tp or tr:                               # lean about the feet: rotate the base, drop it back onto the floor
        m, d = env.m, env.d
        qy = d.qpos[3:7].copy()
        qt = np.zeros(4); mujoco.mju_euler2Quat(qt, np.array([-tr, tp, 0.0]), "xyz")   # +roll arg = lean left (gravity +y in the body)
        qn = np.zeros(4); mujoco.mju_mulQuat(qn, qy, qt)
        d.qpos[3:7] = qn
        mujoco.mj_kinematics(m, d)
        low, _ = env.tip_low()
        d.qpos[2] += -low.min() + 0.0005
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        env.sync_imu()
        obs = env.obs()
    if bp or br:
        obs[0:3] = Rb @ obs[0:3]
    yaw0 = yaw_of(env.d.qpos[3:7])
    fell, t10, rp10 = None, None, None
    for k in range(int(a.seconds / env.cfg.ctrl_dt)):
        t = k * env.cfg.ctrl_dt
        env.ramp = min(1.0, t / a.ramp) if a.ramp > 0 else 1.0
        err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
        env.cmd[2] = float(np.clip(err, -0.3, 0.3)); obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        if bp or br:
            obs[0:3] = Rb @ obs[0:3]
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        p, r = np.degrees(np.arcsin(np.clip(g[0], -1, 1))), np.degrees(np.arcsin(np.clip(g[1], -1, 1)))
        if t10 is None and np.hypot(p, r) > 10:
            t10, rp10 = t, (p, r)
        if env.fallen():
            fell = t
            break
    falls.append(fell)
    dirs.append(("back" if p < -abs(r) else "fwd" if p > abs(r) else "left" if r > 0 else "right") if fell is not None else "")
    first.append((t10, rp10))
    if not a.quiet:
        print(f"seed {s}: {'fell at %.2f s' % fell if fell is not None else 'up'}"
              + (f", 10 deg at {t10:.2f} s (pitch {rp10[0]:+.0f}, roll {rp10[1]:+.0f})" if t10 is not None else ""))
nf = sum(f is not None for f in falls)
early = sum(f is not None and f < 2.0 for f in falls)
rolls = [rp[1] for t, rp in first if t is not None and t < 2.0]
print(f"{a.run} {params} vx {a.vx} ramp {a.ramp} tilt {a.tilt} imu-bias {a.imu_bias} {' '.join(a.set)}: falls {nf}/{a.n} in {a.seconds:.0f} s, "
      f"{early} within 2 s; early 10-deg roll median {np.median(rolls) if rolls else float('nan'):+.0f} deg; "
      f"fall directions {dict((k, dirs.count(k)) for k in set(dirs) if k)}")
