"""Count real steps before the first fall (2026-11-09 goal: 6 steps without falling).

    .venv/bin/python scripts/count_steps.py <run> [--params params_0010M.pkl] [--vx 0.012] [--n 10] [--seconds 20]

A step: a board leaves the floor (lowest point > 3 mm), stays up >= 0.1 s, and lands >= 1 cm further forward (along
the start heading). Starts from standing, the robot's IMU filter in the loop, servo/mass randomization per seed.
Prints, per floor: walks reaching 6 steps, steps before a fall or the end (median, min-max), falls.
"""
import argparse, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of  # noqa: E402


def episode(env, pol, vx, seconds, heading_gain=1.0):
    obs = env.reset((vx, 0.0, 0.0))
    yaw0 = yaw_of(env.d.qpos[3:7]); fwd = np.array([np.cos(yaw0), np.sin(yaw0)])
    up = np.zeros(2); peak = np.zeros(2); x_off = np.zeros(2); steps = 0; fell = False
    for k in range(int(seconds / env.cfg.ctrl_dt)):
        err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
        env.cmd[2] = float(np.clip(heading_gain * err, -0.3, 0.3)); obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        low, c = env.tip_low()
        for i in range(2):
            if low[i] > 0.003 and up[i] == 0:
                x_off[i] = c[i, :2] @ fwd
            if low[i] > 0.0015:
                up[i] += env.cfg.ctrl_dt; peak[i] = max(peak[i], low[i])
            else:
                if up[i] >= 0.1 and peak[i] > 0.003 and c[i, :2] @ fwd - x_off[i] >= 0.01:
                    steps += 1
                up[i] = 0.0; peak[i] = 0.0
        if env.fallen():
            fell = True
            break
    return steps, fell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--params")
    ap.add_argument("--vx", type=float, default=0.012)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--floors", default="0.004,0.1,0.3,0.8")
    ap.add_argument("--roll-bias", type=float, default=0.0, help="deg the IMU roll reads off (the robot: ~+2)")
    a = ap.parse_args()
    cfg, rd = load_run(a.run)
    params = a.params or sorted(rd.glob("params_*M.pkl"))[-1].name
    pol = NumpyPolicy(rd / params)
    print(f"{a.run} {params} vx {a.vx}" + (f" roll bias {a.roll_bias:+.1f} deg" if a.roll_bias else ""))
    for tc in (float(x) for x in a.floors.split(",")):
        over = dict(floor_tc=tc, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45,
                    imu_tilt_bias=(np.radians(a.roll_bias), 0.0), dr_imu_tilt=0.0)
        res = [episode(CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=60000 + s), pol,
                       a.vx, a.seconds) for s in range(a.n)]
        st = np.array([r[0] for r in res]); fl = sum(r[1] for r in res)
        six = sum(1 for r in res if r[0] >= 6)
        print(f"  floor {tc:5.3f}: 6+ steps {six}/{a.n}  steps median {np.median(st):.0f} ({st.min()}-{st.max()})  falls {fl}/{a.n}", flush=True)


if __name__ == "__main__":
    main()
