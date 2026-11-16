"""Round 14 robot trials of the recovery-step policies (2026-11-15 15:30-15:55): per step (a key or the capture point),
the command (dx, dy outward, side), the largest lift of each foot during the step (encoder FK with the IMU attitude,
relative to the other foot; the give makes this a few mm low), the tilt at each phase (shift / first swing /
transfer / closing swing / settle) and the outcome (fell during the step / after / up), with the fall direction.

    .venv/bin/python scripts/robot14_analysis/steps.py hardware/leg_p0/software/stand_r14*_20261115_15*.csv
"""
import csv, glob, sys
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import gap_fit, mujoco
m, d, tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_roll", "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_roll"]

def fk(q, g):
    d.qpos[:] = 0; d.qpos[2] = .3; d.qpos[3:7] = gap_fit._quat_from_g(g); d.qpos[7:] = q
    mujoco.mj_kinematics(m, d); return d.geom_xpos[tid[0]].copy(), d.geom_xpos[tid[1]].copy()

PH = [("shift", 0, .08), ("swing1", .08, .27), ("transfer", .27, .45), ("swing2", .45, .64), ("settle", .64, 1.01)]
for p in [p for a in sys.argv[1:] for p in sorted(glob.glob(a)) if not p.endswith("_frames.csv")]:
    r = list(csv.reader(open(p))); h = r[0]
    if len(r) < 60 or "rec_on" not in h:
        continue
    A = np.array(r[1:], float); t = A[:, 0]
    g = A[:, [h.index(c) for c in ("gx", "gy", "gz")]]; q = A[:, [h.index("q_" + n) for n in J]]
    roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2])); pitch = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))
    on = A[:, h.index("rec_on")] > 0.5; u = A[:, h.index("rec_u")]
    cmd = A[:, [h.index(c) for c in ("rec_dx", "rec_dy", "rec_side")]]
    i0 = int(np.searchsorted(t, 2.3)); r0, p0 = np.median(roll[i0:i0 + 20]), np.median(pitch[i0:i0 + 20])
    rr, pr = roll - r0, pitch - p0; tilt = np.hypot(rr, pr)
    fell = tilt[-5:].max() > 18
    k_end = int(np.argmax(tilt > 18)) if fell else len(t)
    fd = "-" if not fell else (("right" if rr[k_end] > 0 else "left") if abs(rr[k_end]) > abs(pr[k_end]) else ("fwd" if pr[k_end] > 0 else "back"))
    print(f"\n{p.split('/')[-1]}: {t[-1]:.1f} s, {'FELL ' + fd + f' at {t[k_end]:.1f} s' if fell else 'up'}")
    starts = np.where(on & ~np.r_[False, on[:-1]])[0]
    for s in starts:
        e = s + int(np.argmax(~on[s:])) if np.any(~on[s:]) else len(t)
        c = cmd[s]
        dirn = ("fwd" if c[0] > 0.005 else "back" if c[0] < -0.005 else "") + (" out" if c[1] > 0.005 else "") or "in place"
        z = np.array([fk(q[i], g[i]) for i in range(s, min(e, k_end))])
        if len(z) == 0:
            continue
        dz = 1000 * (z[:, 0, 2] - z[:, 1, 2])            # left above right (mm)
        A_ = "L" if c[2] > 0 else "R"
        liftA = dz.max() if A_ == "L" else -dz.min(); liftB = -dz.min() if A_ == "L" else dz.max()
        seg = []
        for nm, a, b in PH:
            sel = (u[s:e] >= a) & (u[s:e] < b) & (np.arange(s, e) < k_end)
            if sel.any():
                ix = np.arange(s, e)[sel]
                seg.append(f"{nm} {rr[ix].max() if abs(rr[ix].max()) > abs(rr[ix].min()) else rr[ix].min():+4.0f}/{pr[ix].max() if abs(pr[ix].max()) > abs(pr[ix].min()) else pr[ix].min():+4.0f}")
        out = "FELL in step" if fell and s <= k_end < e else ("fell after" if fell and e <= k_end < e + 150 else "ok")
        trig = "key" if np.any(np.abs(np.abs(c[:2]) - 0.04) < 1e-3) or np.all(np.abs(c[:2]) < 1e-4) else "cp"
        print(f"  {t[s]:5.1f} s {trig:3s} {A_} foot {dirn:9s} ({c[0] * 100:+.1f}, {c[1] * 100:.1f} cm) | lift A {liftA:4.0f} mm B {liftB:4.0f} mm"
              f" | roll/pitch peak per phase: {', '.join(seg)} | {out}")
