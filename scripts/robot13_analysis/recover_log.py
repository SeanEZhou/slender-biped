"""Round 13 (2026-11-15): robot logs of shove / recovery trials (policy_stand run, with or without --recover).

Per log: time up, how it ended (fell = tilt past 18 deg at the end), supervisor stops and what followed each (resumed
walking / fell while recovering / still recovering at the end), the time from a stop to the resume, the largest tilt
in recovery, and shoves found in the IMU (a jump in the low-passed body rate while no step is due is hard to tell from
a stumble, so the shove times are only candidates). Steps: knee-difference swings (> 12 deg) as in the gate guard.

    .venv-hw/bin/python scripts/robot13_analysis/recover_log.py hardware/leg_p0/software/stand_r13*_2026111[56]_*.csv
"""
import csv
import glob
import math
import sys

import numpy as np


def load(path):
    r = list(csv.reader(open(path)))
    h, A = r[0], np.array(r[1:], float)
    return h, A


def main():
    paths = [p for a in sys.argv[1:] for p in sorted(glob.glob(a)) if not p.endswith("_frames.csv")]
    print("log | s up | end | stops: resumed / fell / open | stop->resume s | max tilt in recovery | knee swings")
    for p in paths:
        h, A = load(p)
        if len(A) < 20:
            continue
        t = A[:, 0]
        g = A[:, [h.index(c) for c in ("gx", "gy", "gz")]]
        roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2])); pitch = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))
        ph = A[:, h.index("phase")] if "phase" in h else np.zeros(len(t))
        i0 = int(np.argmax(ph > 0)) if np.any(ph > 0) else 0
        tilt = np.hypot(roll - roll[i0], pitch - pitch[i0])
        fell = bool(np.any(tilt[-10:] > 18))
        names = [c[2:] for c in h if c.startswith("q_")]
        kl, kr = h.index("q_L_knee"), h.index("q_R_knee")
        kd = np.degrees(A[:, kl] - A[:, kr])
        sw = np.abs(kd) > 12
        swings = int(np.sum(sw & ~np.r_[False, sw[:-1]]))
        sup = A[:, h.index("sup_on")] > 0.5 if "sup_on" in h else np.zeros(len(t), bool)
        starts = np.where(sup & ~np.r_[False, sup[:-1]])[0]
        res, fl, op, dur, mt = 0, 0, 0, [], []
        for s in starts:
            e = s + int(np.argmax(~sup[s:])) if np.any(~sup[s:]) else None
            if e is None:
                if fell:
                    fl += 1
                else:
                    op += 1
                mt.append(tilt[s:].max())
            else:
                res += 1; dur.append(t[e] - t[s]); mt.append(tilt[s:e].max())
        print(f"{p.split('/')[-1]} | {t[-1] - t[i0]:5.1f} | {'FELL' if fell else 'up  '} | {len(starts):2d}: {res} / {fl} / {op} | "
              f"{np.median(dur) if dur else float('nan'):.1f} | {max(mt) if mt else float('nan'):.1f} | {swings}")


if __name__ == "__main__":
    main()
