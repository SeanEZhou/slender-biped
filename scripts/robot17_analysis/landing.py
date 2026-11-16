"""Round 17: how hard each touchdown is on the robot (from the policy_stand log, no video needed).

Per scheduled touchdown (gait clock crossing u 0.396 / 0.896): the landing leg's knee and hip-pitch speed (measured,
deg/s) over the last 60 ms before it (the foot's descent), and the body's jolt after it: the gyro's tick-to-tick change
(roll and pitch, deg/s per 20 ms), RMS and peak over the 0.2 s after touchdown. Steps before a fall are left out of
the medians (the last 2), so a fall's own crash doesn't count.   landing.py <stand_*.csv> ...
"""
import sys
import numpy as np
import csv

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]


def td_phases(run):
    """Touchdown cycle fractions of the run's reference table -> side (k5's c_T12HH: R 0.3958, L 0.8958; the first step is the right foot)."""
    try:
        tab = json.loads((ROOT / "runs" / run / "config.json").read_text())["env"]["ref_table"]
        st = np.load(ROOT / "models" / "ref" / f"{tab}.npz")["stance"][-1]          # (nphase, 2)
    except (OSError, KeyError):
        return {0.3958: "R", 0.8958: "L"}
    n = len(st); out = {}
    for i in range(n):
        for leg, side in ((0, "L"), (1, "R")):
            if st[i - 1, leg] < 0.5 <= st[i, leg]:
                out[round(i / n, 4)] = side
    return out


def _f(x):
    try:
        return float(x)
    except ValueError:
        return {"True": 1.0, "False": 0.0}.get(x, np.nan)


def steps(path, TD):
    with open(path) as f:
        r = csv.reader(f); hdr = next(r); A = np.array([[_f(x) for x in row] for row in r if len(row) == len(hdr)])
    if A.ndim != 2 or len(A) < 50:
        return []
    d = {h: A[:, i] for i, h in enumerate(hdr)}
    u = (d["phase"] / (2 * np.pi)) % 1.0
    w = np.degrees(np.c_[d["wx"], d["wy"]])
    dw = np.r_[np.zeros((1, 2)), np.diff(w, axis=0)]
    jolt = np.linalg.norm(dw, axis=1)
    out = []
    for i in range(1, len(d["t"])):
        for utd, side in TD.items():
            if (u[i - 1] < utd <= u[i]) and u[i] - u[i - 1] < 0.5:
                if i + 10 >= len(d["t"]) or i < 3:
                    continue
                kn = np.degrees(d[f"qd_{side}_knee"][i - 3:i + 1])
                hp = np.degrees(d[f"qd_{side}_hip_pitch"][i - 3:i + 1])
                win = jolt[i:i + 10]
                out.append(dict(t=d["t"][i], side=side, knee_v=np.abs(kn).max(), hip_v=np.abs(hp).max(),
                                jolt_rms=np.sqrt(np.mean(win ** 2)), jolt_pk=win.max(),
                                w_pk=np.abs(w[i:i + 10]).max()))
    return out


if __name__ == "__main__":
    G = {}
    for p in sys.argv[1:]:
        name = p.split("stand_")[-1].rsplit("_2026", 1)[0]
        s = steps(p, td_phases(name))
        if len(s) < 3:
            continue
        if len(s) > 4:
            s = s[:-2]
        name = p.split("stand_")[-1].rsplit("_2026", 1)[0]
        G.setdefault(name, []).extend(s)
    print(f"{'run':16s} {'n':>4s} | knee speed before td p50/p90 | hip | jolt rms p50/p90 | jolt peak p50/p90 | gyro peak p50 (deg/s)")
    for run, L in G.items():
        g = type("G", (), {k: np.array([x[k] for x in L]) for k in L[0] if k != "side"})
        q = lambda a, x: np.quantile(a, x)
        print(f"{run:16s} {len(L):4d} | {np.median(g.knee_v):6.0f} {q(g.knee_v, .9):6.0f} | {np.median(g.hip_v):5.0f} | "
              f"{np.median(g.jolt_rms):6.1f} {q(g.jolt_rms, .9):6.1f} | {np.median(g.jolt_pk):6.1f} {q(g.jolt_pk, .9):6.1f} | "
              f"{np.median(g.w_pk):5.0f}")
