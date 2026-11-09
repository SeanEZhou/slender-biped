"""One-foot balance test (2026-11-09): how long a balance policy keeps the right board off the floor.

    .venv/bin/python scripts/balance_eval.py <run> [--params ...] [--n 8] [--seconds 20]

Per floor: falls, the longest continuous time with the right board's lowest point > 3 mm while the left stays down,
and the total such time over the walk. IMU filter on, randomized servos and masses.
"""
import argparse, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--params")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--floors", default="0.004,0.1,0.3,0.8")
    a = ap.parse_args()
    cfg, rd = load_run(a.run)
    params = a.params or sorted(rd.glob("params_*M.pkl"))[-1].name
    pol = NumpyPolicy(rd / params)
    vx = cfg["env"]["cmd_vx"][1]
    print(f"{a.run} {params}")
    for tc in (float(x) for x in a.floors.split(",")):
        over = dict(floor_tc=tc, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45)
        rows = []
        for s in range(a.n):
            env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=70000 + s)
            obs = env.reset((vx, 0.0, 0.0)); run_t = best = total = 0.0; fell = False
            for k in range(int(a.seconds / env.cfg.ctrl_dt)):
                obs, _ = env.step(pol(obs))
                low, _ = env.tip_low()
                if low[1] > 0.003 and low[0] < 0.0015:
                    run_t += env.cfg.ctrl_dt; total += env.cfg.ctrl_dt; best = max(best, run_t)
                else:
                    run_t = 0.0
                if env.fallen():
                    fell = True
                    break
            rows.append((fell, best, total))
        b = np.array([r[1] for r in rows]); t = np.array([r[2] for r in rows])
        print(f"  floor {tc:5.3f}: falls {sum(r[0] for r in rows)}/{a.n}  longest on one foot {np.median(b):.2f} s (max {b.max():.2f})"
              f"  total {np.median(t):.1f} s", flush=True)


if __name__ == "__main__":
    main()
