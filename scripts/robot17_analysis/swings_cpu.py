"""Round 17: each foot's air phases in the CPU sim (fitted robot, act_lpf 0.7, vx 0.015, 6 x 20 s): full swings
(> 0.2 s in the air) vs taps (short re-lifts, e.g. a bounce after landing). Per full swing: peak clearance (mm).
    swings_cpu.py <run> ...
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np


def ep(a):
    run, seed = a
    import eval6
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run
    cfg, rd = load_run(ROOT / "runs" / run); pol = NumpyPolicy(rd / "params.pkl")
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0,
                dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
                dr_kp=(0.75, 1.25), kp_hip_roll=2.5, drag_back=0.0)
    over.update(eval6.SIMS["fitted"]); over.update(act_lpf=0.7)
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=93000 + seed)
    obs = env.reset((0.015, 0, 0)); LOW = []
    for i in range(1000):
        obs, _ = env.step(pol(obs)); LOW.append(env.tip_low()[0].copy())
        if env.fallen():
            break
    LOW = np.array(LOW); c = LOW < 0.0015; sw, taps = [], 0
    for s in range(2):
        t0 = None
        for t in range(100, len(LOW)):
            if c[t - 1, s] and not c[t, s]:
                t0 = t
            elif not c[t - 1, s] and c[t, s] and t0 is not None:
                if t - t0 >= 10:
                    sw.append(LOW[t0:t, s].max() * 1000)
                else:
                    taps += 1
                t0 = None
    return sw, taps, len(LOW) == 1000


if __name__ == "__main__":
    for run in sys.argv[1:]:
        with Pool(3) as p:
            R = p.map(ep, [(run, s) for s in range(6)])
        P = np.concatenate([r[0] for r in R]); taps = sum(r[1] for r in R); up = sum(r[2] for r in R)
        print(f"{run:12s} up {up}/6  full swings {len(P):3d}  peak clearance p10 {np.percentile(P, 10):5.1f} p50 {np.median(P):5.1f} mm  "
              f"under 10 mm {np.mean(P < 10) * 100:3.0f}%  | taps (air < 0.2 s) {taps}")
