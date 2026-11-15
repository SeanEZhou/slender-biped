"""Robot logs, per trial and per batch: time up, steps, fall direction, roll at each landing (L / R), step width and
foot reach at landing (encoder FK, pelvis frame), body shake (gyro roll/pitch above 3 Hz, RMS deg/s).
Usage (repo root): .venv/bin/python scripts/robot10_analysis/batch.py <label>=<glob> ...
"""
import sys, csv, glob, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import gap_fit, mujoco
m, d, tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle_roll","R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle_roll"]
def fk(q, g=None):
    d.qpos[:] = 0; d.qpos[2] = .3; d.qpos[3:7] = [1, 0, 0, 0] if g is None else gap_fit._quat_from_g(g); d.qpos[7:] = q
    mujoco.mj_kinematics(m, d); return d.geom_xpos[tid[0]].copy(), d.geom_xpos[tid[1]].copy()
def hp(x, dt, fc=3.0):
    a = np.exp(-2*np.pi*fc*dt); y = np.zeros_like(x); lp = x[0].copy()
    for i in range(len(x)): lp = a*lp + (1-a)*x[i]; y[i] = x[i]-lp
    return y
for arg in sys.argv[1:]:
    lab, pat = arg.split("=", 1)
    files = [f for f in sorted(glob.glob(pat)) if not f.endswith("_frames.csv")]
    T = []; NS = []; FD = []; RL = {"L": [], "R": []}; WD = {"L": [], "R": []}; RX = {"L": [], "R": []}; SH = []
    for f in files:
        r = list(csv.reader(open(f))); h = r[0]; A = np.array(r[1:], float); t = A[:, 0]
        if len(t) < 50: continue
        dt = np.median(np.diff(t))
        g = A[:, [h.index(k) for k in ("gx","gy","gz")]]; q = A[:, [h.index(f"q_{n}") for n in J]]; w = A[:, [h.index(k) for k in ("wx","wy","wz")]]
        roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2])); roll -= roll[:25].mean()
        fp = -np.degrees(np.arctan2(-g[:, 0], -g[:, 2]))
        bad = np.where((np.abs(roll) > 20) | (np.abs(fp - 4) > 22))[0]; end = bad[0] if len(bad) else len(t)
        fd = "-" if not len(bad) else (("L" if roll[end] < 0 else "R") if abs(roll[end]) > abs(fp[end] - 4) else ("fwd" if fp[end] > 4 else "back"))
        dz = np.zeros(end); geo = []
        for i in range(end):
            a_, b_ = fk(q[i], g[i]); dz[i] = 1000 * (a_[2] - b_[2])
        up = False; n = 0
        for i in range(end):
            if not up and abs(dz[i]) > 8: up = True; side = "L" if dz[i] > 0 else "R"
            elif up and abs(dz[i]) < 3:
                up = False; n += 1; j = min(i + 5, end - 1)
                a0, b0 = fk(q[j]); RL[side].append(roll[i]); WD[side].append(1000 * (a0[1] - b0[1]))
                RX[side].append(1000 * ((a0[0] - b0[0]) if side == "L" else (b0[0] - a0[0])))
        s = slice(int(2.5 / dt), end)
        if end > int(3.5 / dt):
            wh = hp(np.degrees(w[s, :2]), dt); SH.append(np.sqrt((wh ** 2).sum(1).mean()))
        T.append(t[end - 1]); NS.append(n); FD.append(fd)
    T = np.array(T); NS = np.array(NS)
    print(f"== {lab}: {len(T)} trials, up median {np.median(T):.1f} s (max {T.max():.1f}), steps median {np.median(NS):.0f} "
          f"(max {NS.max()}), {np.sum(NS)/np.sum(np.maximum(T-2,0.1)):.2f} steps/s; falls {sum(x != '-' for x in FD)}: "
          + " ".join(f"{k}:{FD.count(k)}" for k in ("L", "R", "fwd", "back") if FD.count(k))
          + f"; shake {np.median(SH):.1f} deg/s")
    for sd in "LR":
        if RL[sd]:
            print(f"   {sd} landings n {len(RL[sd]):3d}: roll median {np.median(RL[sd]):+.1f} (p10 {np.percentile(RL[sd],10):+.1f} p90 {np.percentile(RL[sd],90):+.1f})"
                  f"  width {np.median(WD[sd]):.0f} mm (p90 {np.percentile(WD[sd],90):.0f})  reach {np.median(RX[sd]):+.0f} mm (p90 {np.percentile(RX[sd],90):+.0f})")
