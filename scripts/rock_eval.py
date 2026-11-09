"""Simulate the deterministic rocking walker (slender_biped/rock_walk.py) on the rug floors.

    .venv/bin/python scripts/rock_eval.py [--params results/rock_walk.json] [--model ...] [--n 6]
"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run, quat_rot_t, yaw_of  # noqa: E402
from slender_biped.rock_walk import RockWalker  # noqa: E402

BASE_RUN = {"hw_pad_board150w_legs_8dof": "pad_board150w_legs", "hw_pad_board150w_8dof": "pad_board150w"}


def episode(params, model, seed, floor_tc, friction=None, seconds=20.0, imu=True, dr=True, record=False):
    cfg, _ = load_run(BASE_RUN[model])
    over = dict(floor_tc=floor_tc, action_scale=1.0, reset_joint_noise=0.0, reset_tilt=0.0, dr=dr, push_vel=0.0)
    if friction is not None:
        over["friction"] = friction
    if imu:
        over.update(imu_filter=True, imu_gate=0.15, gyro_range=17.45)
    env = CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=seed)
    obs = env.reset((0.0, 0.0, 0.0))
    ctl = RockWalker(params, ctrl_dt=env.cfg.ctrl_dt)
    x0 = env.d.qpos[0:2].copy(); yaw0 = yaw_of(env.d.qpos[3:7])
    con_prev = np.ones(2, bool); air = np.zeros(2); peak = np.zeros(2); steps = []; rolls = []; tr = []
    n = int(seconds / env.cfg.ctrl_dt); k = 0; fell = False
    for k in range(n):
        obs, _ = env.step(ctl(obs))
        low, c = env.tip_low()
        con = low < 0.0015
        for i in np.where(con & ~con_prev)[0]:
            if air[i] >= 0.06 - 1e-9:
                steps.append((air[i], peak[i]))
        air = np.where(con, 0.0, air + env.cfg.ctrl_dt); peak = np.where(con, 0.0, np.maximum(peak, low))
        con_prev = con
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        rolls.append(np.degrees(np.arcsin(np.clip(g[1], -1, 1))))
        if record:
            tr.append([k * env.cfg.ctrl_dt, np.degrees(np.arcsin(np.clip(g[0], -1, 1))), rolls[-1], *(1000 * low)])
        if env.fallen():
            fell = True
            break
    dur = (k + 1) * env.cfg.ctrl_dt
    fwd = float((env.d.qpos[0:2] - x0) @ np.array([np.cos(yaw0), np.sin(yaw0)]))
    st = np.array(steps) if steps else np.zeros((0, 2))
    out = dict(survived=dur, fell=fell, fwd=fwd, steps=len(st), air_ms=1000 * float(np.median(st[:, 0])) if len(st) else 0.0,
               clear_mm=1000 * float(np.median(st[:, 1])) if len(st) else 0.0, roll_pp=float(np.ptp(rolls[len(rolls) // 4:])) if rolls else 0.0,
               yaw_deg=float(np.degrees((yaw_of(env.d.qpos[3:7]) - yaw0 + np.pi) % (2 * np.pi) - np.pi)))
    if record:
        out["trace"] = tr
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params")
    ap.add_argument("--model", default="hw_pad_board150w_legs_8dof")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    params = json.loads(Path(a.params).read_text())["params"] if a.params else {}
    for kv in a.set:
        k, v = kv.split("="); params[k] = float(v)
    for tc in (0.004, 0.1, 0.3, 0.8, 1.2):
        t0 = time.time()
        rs = [episode(params, a.model, 40000 + s, tc, seconds=a.seconds) for s in range(a.n)]
        print(f"floor {tc:5.3f}: up {sum(not r['fell'] for r in rs)}/{a.n}  fwd {np.median([r['fwd'] for r in rs]):+.2f} m  "
              f"steps {np.median([r['steps'] for r in rs]):.0f}  air {np.median([r['air_ms'] for r in rs]):.0f} ms  "
              f"clear {np.median([r['clear_mm'] for r in rs]):.0f} mm  roll p-p {np.median([r['roll_pp'] for r in rs]):.1f} deg  "
              f"yaw {np.median([r['yaw_deg'] for r in rs]):+.0f} deg   ({time.time() - t0:.0f} s)", flush=True)
