"""Heading from an integrated gyro (no magnetometer, e.g. MPU6050) instead of the true yaw (round 5, E64).

    .venv/bin/python scripts/gyro_heading_eval.py <run> --params params_XXXXM.pkl --mode gyro --bias 0.05

Every other round-5 number gives the heading hold, and the path-frame offset estimate, the true pelvis yaw. Here the
yaw comes from a gyro: yaw_est = yaw_true + bias*t + scale*(yaw_true - yaw0), with the bias sign alternating across
seeds. The yaw error steers the heading hold and rotates the world-frame odometry / flow velocity that the path-offset
estimate integrates, as the body-to-world rotation would on hardware. Modes: true (perfect yaw), gyro, none (no heading
hold, yaw-rate command 0). 0.15 m/s command; reports the on-the-line rate at 2 m and the offset at the end of the walk.
Appends one line to results/drift/gyro_heading.txt.
"""
import argparse
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
ap.add_argument("--mode", choices=("true", "gyro", "none"), default="gyro")
ap.add_argument("--bias", type=float, default=0.05, help="gyro bias, deg/s")
ap.add_argument("--scale", type=float, default=0.03, help="gyro scale error (fraction)")
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--seeds", type=int, default=30)
a = ap.parse_args()


class GyroBiped(CpuBiped):
    yaw_err = 0.0

    def _update_estimators(self, phase_next, cmd_vx):
        super()._update_estimators(phase_next, cmd_vx)
        c, s = np.cos(self.yaw_err), np.sin(self.yaw_err)
        v = self.odo_vw
        self.odo_vw = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


cfg, rd = load_run(a.run)
pol = NumpyPolicy(rd / a.params)
ec = EvalConfig.from_train(cfg["env"])
res = []
for sd in range(a.seeds):
    env = GyroBiped(cfg["env"]["model"], ec, seed=30000 + sd)
    obs = env.reset((0.15, 0.0, 0.0))
    yaw0 = yaw_of(env.d.qpos[3:7])
    p0 = env.d.qpos[0:2].copy()
    fwd, left = np.array([np.cos(yaw0), np.sin(yaw0)]), np.array([-np.sin(yaw0), np.cos(yaw0)])
    b = (1 if sd % 2 == 0 else -1) * np.radians(a.bias)
    dt = env.cfg.ctrl_dt
    lat2, fell = None, False
    for k in range(int(a.seconds / dt)):
        dy = (yaw_of(env.d.qpos[3:7]) - yaw0 + np.pi) % (2 * np.pi) - np.pi
        env.yaw_err = b * k * dt + a.scale * dy if a.mode == "gyro" else 0.0
        if a.mode == "none":
            env.cmd[2] = 0.0
        else:
            env.cmd[2] = float(np.clip((-(dy + env.yaw_err) + np.pi) % (2 * np.pi) - np.pi, -0.3, 0.3))
        obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        if env.fallen():
            fell = True
            break
        d = env.d.qpos[0:2] - p0
        if lat2 is None and d @ fwd >= 2.0:
            lat2 = float(d @ left)
    d = env.d.qpos[0:2] - p0
    res.append(dict(fell=fell, fwd=float(d @ fwd), lat=abs(float(d @ left)), on=lat2 is not None and abs(lat2) <= 0.2,
                    head=abs(np.degrees((yaw_of(env.d.qpos[3:7]) - yaw0 + np.pi) % (2 * np.pi) - np.pi))))
lat = [r["lat"] for r in res]
line = (f"{a.run} {a.params} mode {a.mode} bias {a.bias if a.mode == 'gyro' else 0} deg/s "
        f"scale {a.scale if a.mode == 'gyro' else 0} {a.seconds:.0f}s x{a.seeds}: "
        f"fell {np.mean([r['fell'] for r in res]):.0%} on-line@2m {np.mean([r['on'] for r in res]):.0%} "
        f"|lat|@end median {np.median(lat):.2f} m p90 {np.percentile(lat, 90):.2f} | "
        f"fwd {np.median([r['fwd'] for r in res]):.1f} m | |heading|@end {np.median([r['head'] for r in res]):.1f} deg")
print(line, flush=True)
out = ROOT / "results" / "drift" / "gyro_heading.txt"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "a") as f:
    f.write(line + "\n")
