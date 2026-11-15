"""Round 10: how high each policy lifts its feet in simulation (fitted sim + pelvis mass 1.5 cm forward, act_lpf 0.6,
vx 0.02): per swing, the lowest board point's peak height (mm); median and 10th percentile over 6 x 20 s, plus the
share of swings that peak under 10 mm (toe-catch risk on rug).   clear.py <run> ...
"""
import math, os, sys
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
    over.update(eval6.SIMS["fitted"]); over.update(com_dx=0.015, act_lpf=0.6)
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=83000 + seed)
    obs = env.reset((0.02, 0, 0)); peaks = [0.0, 0.0]; up = [False, False]; P = []
    for i in range(1000):
        obs, _ = env.step(pol(obs))
        low = env.tip_low()[0]
        for s in range(2):
            if low[s] > 0.002:
                up[s] = True; peaks[s] = max(peaks[s], low[s])
            elif up[s]:
                P.append(peaks[s] * 1000); up[s] = False; peaks[s] = 0.0
        if env.fallen():
            break
    return P

if __name__ == "__main__":
    for run in sys.argv[1:]:
        with Pool(3) as p:
            P = np.concatenate([np.array(x) for x in p.map(ep, [(run, s) for s in range(6)])])
        P = P[P > 3]
        print(f"{run:12s} swings {len(P):4d}  peak clearance median {np.median(P):5.1f} mm  p10 {np.percentile(P, 10):5.1f}  "
              f"under 10 mm {np.mean(P < 10) * 100:4.0f}%")
