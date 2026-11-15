"""Round 13 robot stand-push trials (2026-11-15 11:19-11:26): per push, the tilt's time course and what the policy
did about it. A push = the tilt (from the engage attitude) leaving 2.5 deg; it ends in a fall if the log stops past 18 deg
within 3 s. Reports: direction, peak rate, time from 2.5 deg to 6 / 10 / 18 deg, and the commanded joint motion over
that window (largest change of any joint's command from its value at onset, and per joint group), and the knee
difference (a foot lifting).

    .venv-hw/bin/python scripts/robot13_analysis/stand_push.py hardware/leg_p0/software/stand_*_20261115_112*.csv
"""
import csv, glob, sys
import numpy as np

def main():
    paths = [p for a in sys.argv[1:] for p in sorted(glob.glob(a)) if not p.endswith("_frames.csv")]
    for p in paths:
        r = list(csv.reader(open(p)))
        if len(r) < 60:
            continue
        h, A = r[0], np.array(r[1:], float)
        t = A[:, 0]; g = A[:, [h.index(c) for c in ("gx", "gy", "gz")]]; w = np.degrees(A[:, [h.index(c) for c in ("wx", "wy")]])
        roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2])); pitch = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))
        ph = A[:, h.index("phase")]
        i0 = int(np.argmax(ph > 0)) if np.any(ph > 0) else 0
        i0 = max(i0, int(np.searchsorted(t, t[0] + 0.5)))
        b = slice(i0, i0 + 25)
        rr, pr = roll - np.median(roll[b]), pitch - np.median(pitch[b])
        tilt = np.hypot(rr, pr)
        names = [c[4:] for c in h if c.startswith("cmd_")]
        cmd = np.degrees(A[:, [h.index("cmd_" + n) for n in names]])
        q = np.degrees(A[:, [h.index("q_" + n) for n in names]])
        kd = q[:, names.index("L_knee")] - q[:, names.index("R_knee")]
        fell = tilt[-5:].max() > 18
        print(f"\n{p.split('/')[-1]}: {t[-1] - t[i0]:.1f} s engaged, {'FELL' if fell else 'up'} (start roll {np.median(roll[b]):+.1f} pitch {np.median(pitch[b]):+.1f})")
        i = i0 + 25
        while i < len(t):
            if tilt[i] < 2.5:
                i += 1; continue
            s = i
            e = s
            while e < len(t) - 1 and tilt[e] > 1.5 and t[e] - t[s] < 3.0:
                e += 1
            pk = s + int(np.argmax(tilt[s:e + 1]))
            d = "roll " + ("L" if rr[pk] > 0 else "R") if abs(rr[pk]) > abs(pr[pk]) else "pitch " + ("fwd" if pr[pk] > 0 else "back")
            def tt(th):
                j = np.where(tilt[s:e + 1] > th)[0]
                return f"{t[s + j[0]] - t[s]:.2f}" if len(j) else "  - "
            dc = cmd[s:pk + 1] - cmd[s]
            grp = {g_: np.max(np.abs(dc[:, [k for k, n in enumerate(names) if n.endswith(g_)]])) for g_ in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll")}
            trk = np.max(np.abs((q - cmd)[s:pk + 1]), 0)
            print(f"  t {t[s] - t[i0]:5.1f}  {d:9s} peak {tilt[pk]:5.1f} deg at +{t[pk] - t[s]:.2f} s, rate max {np.max(np.abs(w[s:pk + 1])):4.0f} deg/s"
                  f" | to 6/10/18 deg: {tt(6)} {tt(10)} {tt(18)} s | cmd change max " + " ".join(f"{k.split('_')[-1][:5]} {v:4.1f}" for k, v in grp.items())
                  + f" | knee diff max {np.max(np.abs(kd[s:pk + 1])):4.1f} | worst tracking err {trk.max():4.1f} ({names[int(np.argmax(trk))]})")
            i = e + 1

if __name__ == "__main__":
    main()
