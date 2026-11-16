"""Round 17: what the body does around each touchdown (CPU sim, fitted robot + act_lpf 0.7, vx 0.015, 6 x 20 s).
Per touchdown: body roll rate toward the landing foot over the 40 ms before contact, and how much the roll / pitch /
yaw rate changes from before to after (deg/s); body roll angle toward the landing foot at contact (deg).
    td_cpu.py <run[:params]> ...
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np


def ep(a):
    spec, seed = a
    import eval6
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, roll_of
    run, _, pp = spec.partition(":")
    cfg, rd = load_run(ROOT / "runs" / run); pol = NumpyPolicy(rd / (pp or "params.pkl"))
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0,
                dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
                dr_kp=(0.75, 1.25), kp_hip_roll=2.5, drag_back=0.0)
    over.update(eval6.SIMS["fitted"]); over.update(act_lpf=0.7)
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=91000 + seed)
    obs = env.reset((0.015, 0, 0)); W = []; LOW = []; RL = []
    for i in range(1000):
        obs, _ = env.step(pol(obs))
        q = env.d.qpos[3:7]
        w = np.degrees(env.d.qvel[3:6].copy()); W.append(w); LOW.append(env.tip_low()[0].copy())
        # roll (deg, + = right side down) from the body z axis
        R = np.zeros(9); import mujoco; mujoco.mju_quat2Mat(R, q); R = R.reshape(3, 3)
        RL.append(np.degrees(np.arcsin(-R[2, 1])))
        if env.fallen():
            break
    W, LOW, RL = map(np.array, (W, LOW, RL)); c = LOW < 0.0015; out = []
    n = len(W) - 30 if len(W) == 1000 else len(W) - 60
    for t in range(100, n):
        for s in range(2):                      # s 0 = left board (y > 0): landing on it means rolling toward +x? sign below
            if c[t, s] and not c[t - 1, s] and not c[t - 3:t, s].any():
                sg = 1.0 if s == 1 else -1.0    # toward the landing foot: sign chosen so the median is comparable L vs R
                pre, post = W[t - 2:t].mean(0), W[t + 1:t + 3].mean(0)
                out.append([sg * pre[0], *(post - pre), sg * RL[t]])
    return out


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        with Pool(3) as p:
            X = np.array([r for x in p.map(ep, [(spec, s) for s in range(6)]) for r in x])
        print(f"{spec:24s} n {len(X):3d} | roll rate toward landing foot before p50 {np.median(X[:, 0]):+5.1f} p90 {np.percentile(X[:, 0], 90):+5.1f} | "
              f"|change| roll {np.median(np.abs(X[:, 1])):5.1f} pitch {np.median(np.abs(X[:, 2])):5.1f} yaw {np.median(np.abs(X[:, 3])):5.1f} | "
              f"roll at contact {np.median(X[:, 4]):+5.1f} deg")
