"""Does it walk straight? Sideways drift and crab angle at a fixed speed command (round 5).

    .venv/bin/python scripts/drift_eval.py <run> --params params_XXXXM.pkl [--vx 0.15] [--seeds 20] [--vy 0.0]

Heading hold on (outer loop pins the pelvis yaw, as an operator or the IMU would), 20 s per seed:
  forward / lateral travel along / across the start heading, lateral offset when the robot first passes 2 m forward
  (the "stays on the line" number), crab angle (travel direction minus facing, 1 s windows) and survival.
With --vy, the sideways command is held at that value (does it respond to sideways commands at all?).
With --line-hold K, an outer loop steers back to the start line from the robot's own stance-leg odometry
(CpuBiped.odo_v, integrated): vy_cmd = clip(-K * y_est, +-0.1). No external sensing; the estimate keeps its bias.
With --true-offset (diagnostic), a policy with the path-offset input (obs_line) is given the true offset instead of
its estimate: what the controller could do with perfect position sensing.
--set overrides EvalConfig fields (sensor errors, odo_src, line_frame, ...); --tag keeps such variants apart.
Medians and IQR over seeds; writes results/drift/<run>__<params>__vx<vx>_vy<vy>.json.
"""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--params", default="params.pkl")
ap.add_argument("--vx", type=float, default=0.15)
ap.add_argument("--vy", type=float, default=0.0)
ap.add_argument("--seeds", type=int, default=20)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--line-hold", type=float, default=0.0)
ap.add_argument("--true-offset", action="store_true")
ap.add_argument("--tag", default="", help="suffix for the output file (diagnostic variants)")
ap.add_argument("--set", nargs="*", default=[], help="EvalConfig overrides k=json, e.g. odo_bias=0 'odo_src=\"flow\"'")
a = ap.parse_args()
cfg, rd = load_run(a.run)
pol = NumpyPolicy(rd / a.params)
ec = dataclasses.replace(EvalConfig.from_train(cfg["env"]), **{k: json.loads(v) for k, v in (kv.split("=", 1) for kv in a.set)})
eps = []
for s in range(a.seeds):
    env = CpuBiped(cfg["env"]["model"], ec, seed=30000 + s)
    obs = env.reset((a.vx, a.vy, 0.0))
    yaw0 = yaw_of(env.d.qpos[3:7])
    p0 = env.d.qpos[0:2].copy()
    fwd, left = np.array([np.cos(yaw0), np.sin(yaw0)]), np.array([-np.sin(yaw0), np.cos(yaw0)])
    w = int(1.0 / env.cfg.ctrl_dt)
    hist, crab, lat2, fell, y_est = [], [], None, False, 0.0
    for k in range(int(a.seconds / env.cfg.ctrl_dt)):
        err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
        env.cmd[2] = float(np.clip(err, -0.3, 0.3))
        if a.line_hold:
            y_est += env.odo_v[1] * env.cfg.ctrl_dt
            env.cmd[1] = float(np.clip(a.vy - a.line_hold * y_est, -0.1, 0.1))
        obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        if env.fallen():
            fell = True
            break
        d = env.d.qpos[0:2] - p0
        if a.true_offset:
            env.y_est = float(d @ left)
            obs = env.obs()
        hist.append((d[0], d[1], yaw_of(env.d.qpos[3:7])))
        if lat2 is None and d @ fwd >= 2.0:
            lat2 = float(d @ left)
        if k >= w and k % w == 0:
            x0, y0, _ = hist[-w]
            x1, y1, yw = hist[-1]
            crab.append(np.degrees((np.arctan2(y1 - y0, x1 - x0) - yw + np.pi) % (2 * np.pi) - np.pi))
    d = env.d.qpos[0:2] - p0
    eps.append(dict(fell=fell, forward_m=float(d @ fwd), lateral_m=float(d @ left), lateral_at_2m=lat2,
                    crab_deg=float(np.median(crab)) if crab else None))


def q(xs):
    xs = [x for x in xs if x is not None]
    return [round(float(np.percentile(xs, p)), 3) for p in (25, 50, 75)] if xs else None


lat = [e["lateral_m"] for e in eps]
l2 = [e["lateral_at_2m"] for e in eps]
out = dict(run=a.run, params=a.params, vx=a.vx, vy=a.vy, line_hold=a.line_hold, overrides=a.set, seeds=a.seeds,
           fell=sum(e["fell"] for e in eps) / len(eps),
           reach_2m=sum(x is not None for x in l2) / len(eps),
           on_line_2m=sum(x is not None and abs(x) <= 0.2 for x in l2) / len(eps),   # passes 2 m within 0.2 m of the line
           abs_lateral_at_2m_med=float(np.median([abs(x) for x in l2 if x is not None])) if any(x is not None for x in l2) else None,
           forward_q=q([e["forward_m"] for e in eps]), lateral_q=q(lat), abs_lateral_med=float(np.median(np.abs(lat))),
           lateral_mean=float(np.mean(lat)), abs_crab_med=float(np.median([abs(e["crab_deg"]) for e in eps if e["crab_deg"] is not None])),
           episodes=eps)
tag = (f"_lh{a.line_hold}" if a.line_hold else "") + (f"__{a.tag}" if a.tag else "")
dst = ROOT / "results" / "drift" / f"{Path(str(a.run)).name}__{Path(a.params).stem}__vx{a.vx}_vy{a.vy}{tag}.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
print(f"{Path(str(a.run)).name} {a.params} vx {a.vx} vy {a.vy} lh {a.line_hold}: fell {out['fell']:.0%} reach2m {out['reach_2m']:.0%} "
      f"on-line@2m {out['on_line_2m']:.0%} |lat|@2m {out['abs_lateral_at_2m_med']} | fwd {out['forward_q']} "
      f"lat {out['lateral_q']} mean {out['lateral_mean']:+.2f} | |crab| {out['abs_crab_med']:.0f} deg")
