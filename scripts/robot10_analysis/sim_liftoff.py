"""Round 11 prep: the robot logs' lift-off measure (roll and roll rate toward the stance side at each lift-off) in the
fitted simulator (pelvis mass 1.5 cm forward, act_lpf 0.6, vx 0.02, 6 x 20 s), plus how often the settle gate held
and for how long. Does the simulator reproduce k5's late, still-rolling lift-offs?   sim_liftoff.py <run> ...
"""
import math, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
LIFT_H = float(os.environ.get("LIFT_H", "0.006"))   # the robot logs' lift-off: the foot 6 mm up (FK)

def ep(a):
    run, seed, extra = a
    import eval6
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run
    cfg, rd = load_run(ROOT / "runs" / run); pol = NumpyPolicy(rd / "params.pkl")
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0,
                dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
                dr_kp=(0.75, 1.25), kp_hip_roll=2.5, drag_back=0.0)
    over.update(eval6.SIMS["fitted"]); over.update(com_dx=0.015, act_lpf=0.6); over.update(extra)
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=83000 + seed)
    obs = env.reset((0.02, 0, 0)); dt = env.cfg.ctrl_dt
    prev = None; out = []; holds = []; R = []; fell = 0
    for i in range(1000):
        obs, _ = env.step(pol(obs))
        q = env.d.qpos[3:7]; g = np.array([2 * (q[1] * q[3] - q[0] * q[2]), 2 * (q[2] * q[3] + q[0] * q[1]), q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2])
        R.append(math.degrees(math.atan2(g[1], g[2])))
        if env.gate_held == 0 and holds and holds[-1] < 0: holds[-1] = -holds[-1]
        if env.gate_held > 0:
            if holds and holds[-1] < 0: holds[-1] = -env.gate_held
            else: holds.append(-1)
        low = env.tip_low()[0]; c = low < LIFT_H
        if prev is not None and i > 50:
            for s in range(2):
                if prev[s] and not c[s] and c[1 - s]:
                    # s lifts; stance is 1-s. stance L (s=1 lifts R) -> roll toward the left; sign fixed below by data
                    sd = 1.0 if s == 0 else -1.0
                    k = max(0, len(R) - 1 - int(0.1 / dt))
                    out.append((sd * R[-1], sd * (R[-1] - R[k]) / (0.1)))
        prev = c
        if env.fallen():
            fell = 1; break
    return out, [abs(h) * dt for h in holds], fell

if __name__ == "__main__":
    import json
    extra = json.loads(os.environ.get("OVER", "{}"))
    if "hidden_flex" in extra: extra["hidden_flex"] = tuple(tuple(e) for e in extra["hidden_flex"])
    for run in sys.argv[1:]:
        with Pool(3) as p:
            res = p.map(ep, [(run, s, extra) for s in range(6)])
        X = np.array(sum([r[0] for r in res], [])); H = sum([r[1] for r in res], [])
        # orient: the median lift-off roll is toward the stance side (positive), as on the robot
        if np.median(X[:, 0]) < 0: X = -X
        print(f"{run:12s} lift-offs {len(X):4d}  roll {np.median(X[:,0]):+5.1f} (p90 {np.percentile(X[:,0],90):+5.1f})  rate {np.median(X[:,1]):+6.1f} deg/s "
              f"(p75 {np.percentile(X[:,1],75):+5.1f})  gate holds {len(H)} median {np.median(H) if H else 0:.2f}s max {max(H) if H else 0:.2f}s  falls {sum(r[2] for r in res)}/6")
