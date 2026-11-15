"""Round 13 (2026-11-15): how early can the IMU tell a k5 fall from a normal k5 step? Robot logs of k5_high2.

Per log: roll / pitch relative to the moment the clock starts, their rates low-passed at --lpf Hz, the prediction
angle + tau * rate. A fall = the log ends tilted past 18 deg. Reports, per threshold: share of falls caught, the lead
time (trigger to 18 deg) and false triggers per minute of normal walking (more than 2 s before a fall's end).

    .venv-hw/bin/python scripts/robot13_analysis/detector.py [glob] [--tau 0.15] [--lpf 3]
"""
import argparse, csv, glob, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pat", nargs="?", default="hardware/leg_p0/software/stand_k5_high2_2026111[45]_*.csv")
ap.add_argument("--tau", type=float, default=0.15)
ap.add_argument("--lpf", type=float, default=3.0)
a = ap.parse_args()
logs = [p for p in sorted(glob.glob(a.pat)) if not p.endswith("_frames.csv")]
TH = [(r, p) for r in (6, 8, 10, 12, 14) for p in (8, 10, 12, 15)]
res = {th: dict(caught=0, lead=[], false=0) for th in TH}
nf, walk_s, env_r, env_p = 0, 0.0, [], []
for path in logs:
    r = list(csv.reader(open(path)))
    h, A = r[0], np.array(r[1:], float)
    if len(A) < 100 or "phase" not in h:
        continue
    t = A[:, 0]; g = A[:, [h.index(c) for c in ("gx", "gy", "gz")]]; w = A[:, [h.index(c) for c in ("wx", "wy", "wz")]]
    roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2])); pitch = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))   # rate = body-y gyro (mjx_env._pitch_of)
    ph = A[:, h.index("phase")]
    i0 = int(np.argmax(ph > 0)) if np.any(ph > 0) else 0
    if i0 == 0:
        continue
    al = 1 - math.exp(-2 * math.pi * a.lpf * 0.02)
    wf = np.zeros((len(t), 2)); x = w[i0, :2].copy()
    for i in range(len(t)):
        x += al * (w[i, :2] - x); wf[i] = x
    rr = roll - roll[i0]; pr = pitch - pitch[i0]
    pred_r = rr + a.tau * np.degrees(wf[:, 0]); pred_p = pr + a.tau * np.degrees(wf[:, 1])
    tilt = np.hypot(rr, pr)
    fall = np.any(tilt[-10:] > 18)
    end = int(np.argmax(tilt > 18)) if fall else len(t)
    nf += fall
    normal = np.arange(len(t)) >= i0
    normal &= (t < t[end - 1] - 2.0) if fall else True
    walk_s += normal.sum() * 0.02
    env_r += list(np.abs(pred_r[normal])); env_p += list(np.abs(pred_p[normal]))
    for th in TH:
        trig = (np.abs(pred_r) > th[0]) | (np.abs(pred_p) > th[1])
        trig[:i0] = False
        # false triggers: rising edges in normal walking
        e = trig & ~np.r_[False, trig[:-1]] & normal
        res[th]["false"] += int(e.sum())
        if fall:
            j = np.where(trig[:end + 1] & (t[:end + 1] >= t[end] - 2.0))[0]
            if len(j):
                res[th]["caught"] += 1; res[th]["lead"].append(t[end] - t[j[0]])
print(f"{len(logs)} logs, {nf} falls, {walk_s / 60:.1f} min of normal walking")
print("normal |pred roll| p50/p95/p99/max: " + " ".join(f"{np.percentile(env_r, q):.1f}" for q in (50, 95, 99, 100)))
print("normal |pred pitch| p50/p95/p99/max: " + " ".join(f"{np.percentile(env_p, q):.1f}" for q in (50, 95, 99, 100)))
print("roll th / pitch th | falls caught | median lead s (p10) | false /min")
for th in TH:
    d = res[th]; L = d["lead"]
    print(f"{th[0]:4d} {th[1]:4d} | {d['caught']:3d}/{nf} | {np.median(L) if L else 0:.2f} ({np.percentile(L, 10) if L else 0:.2f}) | {d['false'] / (walk_s / 60):.2f}")
