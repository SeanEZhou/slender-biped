"""Does it walk or skitter? Gait-shape metrics at a fixed speed command (round 4).

    .venv/bin/python scripts/gait_quality.py <run> --params params_XXXXM.pkl [--vx 0.15] [--seeds 5]

Per touchdown that follows >= 60 ms of air time (a step; shorter blips are chatter):
  cadence (steps/s), air time, step length (tip travel between lift-off and touchdown), swing clearance
  (highest tip point during the swing), step width (lateral gap between consecutive left/right touchdowns). Also: share of tip travel spent sliding on the ground, double-support
  fraction, pelvis lateral sway (peak-to-peak of the sideways velocity integral) and pelvis roll/pitch RMS,
  forward speed along the start heading (vx) and in the body frame (body_vx), heading drift (deg/s, net over
  the episode, with the heading-hold loop on) and survival. Medians over seeds. Writes results/gait/<run>__<vx>.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, quat_rot_t, yaw_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--params", default="params.pkl")
ap.add_argument("--vx", type=float, default=0.15)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--seconds", type=float, default=20.0)
a = ap.parse_args()
cfg, rd = load_run(a.run)
pol = NumpyPolicy(rd / a.params)
rows = []
for s in range(a.seeds):
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"]), seed=20000 + s)
    obs = env.reset((a.vx, 0.0, 0.0))
    yaw0 = yaw_of(env.d.qpos[3:7])
    low, c = env.tip_low()
    con_prev = low < 0.0015
    lift_pos, air, peak = c.copy(), np.zeros(2), np.zeros(2)
    steps, gnd, tot, both, lat, rp, vbx, x0 = [], 0.0, 0.0, 0, [], [], [], env.d.qpos[0:2].copy()
    tds, widths = [], []
    c_prev = c.copy()
    n = int(a.seconds / env.cfg.ctrl_dt)
    k = 0
    for k in range(n):
        err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
        env.cmd[2] = float(np.clip(err, -0.3, 0.3)); obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        low, c = env.tip_low()
        con = low < 0.0015
        dxy = np.linalg.norm(c[:, :2] - c_prev[:, :2], axis=1)
        tot += dxy.sum()
        gnd += dxy[con & con_prev].sum()
        both += int(con.all())
        td = con & ~con_prev
        fwd = np.array([np.cos(yaw0), np.sin(yaw0)])
        for i in np.where(td)[0]:
            if air[i] >= 0.06 - 1e-9:                         # air time accumulated up to this touchdown
                steps.append((air[i], float((c[i, :2] - lift_pos[i, :2]) @ fwd), peak[i]))
                if tds and tds[-1][0] != i:                    # step width: lateral gap to the other foot's touchdown
                    widths.append(abs(float((c[i, :2] - tds[-1][1]) @ np.array([-fwd[1], fwd[0]]))))
                tds.append((i, c[i, :2].copy()))
        lo = con_prev & ~con
        lift_pos[lo] = c[lo]
        air = np.where(con, 0.0, air + env.cfg.ctrl_dt)
        peak = np.where(con, 0.0, np.maximum(peak, low))
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        rp.append(g[:2])
        vb = quat_rot_t(env.d.qpos[3:7], env.d.qvel[0:3])
        lat.append(vb[1])
        vbx.append(vb[0])
        con_prev, c_prev = con, c.copy()
        if env.fallen():
            break
    dur = (k + 1) * env.cfg.ctrl_dt
    st = np.array(steps) if steps else np.zeros((0, 3))
    late = st[len(st) // 5:] if len(st) > 5 else st          # skip the first second-ish of start-up steps
    y = np.cumsum(lat) * env.cfg.ctrl_dt
    y = y - np.convolve(y, np.ones(50) / 50, mode="same")    # remove drift: sway about a 1 s moving mean
    rows.append(dict(
        survived_s=dur, body_vx=float(np.mean(vbx[50:])) if len(vbx) > 60 else 0.0,
        yaw_drift_dps=float(np.degrees((yaw_of(env.d.qpos[3:7]) - yaw0 + np.pi) % (2 * np.pi) - np.pi)) / dur, vx=float((env.d.qpos[0:2] - x0) @ np.array([np.cos(yaw0), np.sin(yaw0)])) / dur,
        cadence=len(st) / dur, air_ms=1000 * float(np.median(late[:, 0])) if len(late) else 0.0,
        step_cm=100 * float(np.median(late[:, 1])) if len(late) else 0.0,
        clear_mm=1000 * float(np.median(late[:, 2])) if len(late) else 0.0,
        width_mm=1000 * float(np.median(widths[len(widths) // 5:])) if len(widths) > 5 else 0.0,
        slide_frac=gnd / max(tot, 1e-9), double_support=both / (k + 1),
        sway_mm=1000 * float(np.ptp(y[50:-50])) if len(y) > 120 else 0.0,
        tilt_rms_deg=float(np.degrees(np.sqrt(np.mean(np.sum(np.array(rp) ** 2, 1))))),
    ))
med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
out = ROOT / "results" / "gait"
out.mkdir(parents=True, exist_ok=True)
(out / f"{a.run}__{a.vx:.2f}.json").write_text(json.dumps(dict(params=a.params, median=med, rows=rows), indent=1))
print(f"{a.run} {a.params} vx_cmd {a.vx}: " + " | ".join(f"{k} {v:.3g}" for k, v in med.items()))
