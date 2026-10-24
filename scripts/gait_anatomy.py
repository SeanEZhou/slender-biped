"""Where does foot travel happen: in the air (stepping) or on the ground (sliding/shuffling)?

    .venv/bin/python scripts/gait_anatomy.py <run> --params params_XXXXM.pkl --test walk40 --seeds 3
Prints per-foot horizontal travel split into airborne vs in-contact, lift-offs, mean air time and
mean airborne step length (tip travel per airborne phase).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of  # noqa: E402
from evaluate import TESTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--params", default="params.pkl")
ap.add_argument("--test", default="walk40")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seconds", type=float, default=20.0)
a = ap.parse_args()
cfg, rd = load_run(a.run)
t = TESTS[a.test]
rows = []
for s in range(a.seeds):
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"]), seed=10000 + s)
    pol = NumpyPolicy(rd / a.params)
    obs = env.reset(t["sched"][0][1])
    yaw0 = yaw_of(env.d.qpos[3:7])
    low, c = env.tip_low()
    con_prev, c_prev = low < 0.0015, c.copy()
    air_tr, gnd_tr, lifts, air_t, cur_air, air_steps = np.zeros(2), np.zeros(2), np.zeros(2), [], np.zeros(2), []
    lift_pos = c.copy()
    for k in range(int(a.seconds / env.cfg.ctrl_dt)):
        if t.get("heading_hold"):
            err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
            env.cmd[2] = float(np.clip(err, -0.3, 0.3)); obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        low, c = env.tip_low()
        con = low < 0.0015
        dxy = np.linalg.norm(c[:, :2] - c_prev[:, :2], axis=1)
        both = con & con_prev
        gnd_tr += np.where(both, dxy, 0.0)
        air_tr += np.where(~both, dxy, 0.0)
        lo = con_prev & ~con
        lifts += lo
        lift_pos[lo] = c[lo]
        cur_air = np.where(con, cur_air, cur_air + env.cfg.ctrl_dt)
        for i in np.where(con & ~con_prev)[0]:
            air_t.append(cur_air[i]); air_steps.append(float(np.linalg.norm(c[i, :2] - lift_pos[i, :2])))
        cur_air = np.where(con, 0.0, cur_air)
        con_prev, c_prev = con, c.copy()
        if env.fallen():
            break
    rows.append((air_tr.sum(), gnd_tr.sum(), lifts.sum(), np.mean(air_t) if air_t else 0, np.median(air_steps) if air_steps else 0,
                 np.linalg.norm(env.d.qpos[:2])))
r = np.array(rows)
print(f"{a.run} {a.params} {a.test} ({a.seconds:.0f} s, {a.seeds} seeds, medians): "
      f"airborne travel {np.median(r[:,0]):.2f} m | ground (sliding) travel {np.median(r[:,1]):.2f} m | "
      f"lift-offs {np.median(r[:,2]):.0f} | mean air time {np.median(r[:,3])*1000:.0f} ms | "
      f"airborne step {np.median(r[:,4])*100:.1f} cm | base displacement {np.median(r[:,5]):.2f} m")
