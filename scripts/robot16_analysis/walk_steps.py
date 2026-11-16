"""Round 16 (2026-11-15): k5 walking trials, step by step. For each step (gait clock touchdown at u 0.396 / 0.896):
the side, roll and pitch at the touchdown (from the standing attitude), roll / pitch rate there, the largest roll over
the following stance (to the next lift-off), the gate hold before the next lift-off, and the landing foot's board
roll (encoder FK, ankle roll included). Per trial: how it ended and the last steps before the end.
Usage (repo root): .venv/bin/python scripts/robot16_analysis/walk_steps.py <csv> ...
"""
import sys, csv, math, numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
TD = (0.3958, 0.8958); LO = (0.2083, 0.7083)     # c_T12HH; the first touchdown / lift-off (TD[0], LO[0]) is the RIGHT foot
# 2026-11-16 correction: until then this script called TD[0] a left touchdown and read the left board there (the stance
# board, not the landing one). The reference tables' stance columns are [left, right]; the right foot swings first.
import mujoco, gap_fit
_m, _d, _tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_roll", "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_roll"]


def boards(q, g):
    """World roll / pitch (deg) of the L / R boards from the encoders and the IMU attitude (yaw ignored)."""
    _d.qpos[:] = 0; _d.qpos[2] = 0.3; _d.qpos[3:7] = gap_fit._quat_from_g(g); _d.qpos[7:] = q
    mujoco.mj_kinematics(_m, _d)
    out = []
    for gid in _tid:
        R = _d.geom_xmat[gid].reshape(3, 3)
        out.append((math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(-np.clip(R[2, 0], -1, 1)))))
    return out

def load(f):
    r = list(csv.reader(open(f))); h = r[0]; A = np.array(r[1:], float)
    col = lambda k: A[:, h.index(k)]  # noqa: E731
    return h, A, col

def table_events(f):
    """TD / LO of the run's reference table (right first), from the log name stand_<run>_<date>_<time>.csv."""
    import json, pathlib
    run = f.split("/")[-1][6:-20]
    try:
        tab = json.loads((pathlib.Path("runs") / run / "config.json").read_text())["env"]["ref_table"]
        st = np.load(f"models/ref/{tab}.npz")["stance"][-1]
    except (OSError, KeyError):
        return TD, LO
    n = len(st)
    td = tuple(next(i / n for i in range(n) if st[i - 1, leg] < 0.5 <= st[i, leg]) for leg in (1, 0))
    lo = tuple(next(i / n for i in range(n) if st[i - 1, leg] >= 0.5 > st[i, leg]) for leg in (1, 0))
    return td, lo


def analyse(f, verbose=True):
    TD, LO = table_events(f)
    h, A, col = load(f)
    t = col("t"); g = A[:, [h.index(k) for k in ("gx", "gy", "gz")]]; w = A[:, [h.index(k) for k in ("wx", "wy", "wz")]]
    roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2])); pitch = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))
    eng = np.where(np.abs(np.diff(col("phase"))) > 1e-6)[0]
    if not len(eng):
        return None
    k0 = eng[0]
    roll -= roll[max(0, k0 - 25):k0 + 1].mean(); pitch -= pitch[max(0, k0 - 25):k0 + 1].mean()
    u = col("phase") / (2 * math.pi); held = col("gate_held")
    q = A[:, [h.index(f"q_{n}") for n in J]]
    b0 = np.array([boards(q[k], g[k]) for k in range(max(0, k0 - 25), k0 + 1)]).mean(0)   # standing: the zero
    steps = []
    for i in range(k0, len(t) - 1):
        for j, td in enumerate(TD):
            if (u[i] - td) % 1.0 > (u[i + 1] - td) % 1.0 and u[i] != u[i + 1]:
                side = "R" if j == 0 else "L"   # u 0.208-0.396 the right board swings: it lands at 0.396
                # stance after this touchdown until the next lift-off
                nxt = [k for k in range(i + 1, len(t) - 1) if any((u[k] - lo) % 1.0 > (u[k + 1] - lo) % 1.0 and u[k] != u[k + 1] for lo in LO)]
                e = nxt[0] if nxt else len(t) - 1
                seg = slice(i + 1, e + 1)
                bd = np.array(boards(q[i + 1], g[i + 1])) - b0
                fi = 0 if side == "L" else 1
                steps.append(dict(t=t[i + 1], side=side, broll=bd[fi, 0], bpitch=bd[fi, 1], roll=roll[i + 1], pitch=pitch[i + 1], wr=math.degrees(w[i + 1, 0]),
                                  wp=math.degrees(w[i + 1, 1]), rmax=roll[seg][np.argmax(np.abs(roll[seg]))] if e > i else 0.0,
                                  hold=(held[seg].max() if e > i else 0) * 0.02, dur=t[e] - t[i + 1]))
    tilt = np.hypot(roll, pitch)
    bad = np.where(tilt > 20)[0]
    if len(bad):
        b = bad[0]; how = ("fell L" if roll[b] < 0 else "fell R") if abs(roll[b]) > abs(pitch[b]) else ("fell fwd" if pitch[b] > 0 else "fell back")
        tend = t[b]
    else:
        how, tend = "stopped up", t[-1]
    steps = [s for s in steps if s["t"] <= tend]
    if verbose:
        print(f"\n{f.split('/')[-1]}: {how} at {tend - t[k0]:.1f} s after engage, {len(steps)} touchdowns")
        for s in steps[-6:] if how != "stopped up" else steps[:0]:
            print(f"   {s['t'] - t[k0]:5.1f}s {s['side']} board roll {s['broll']:+5.1f} pitch {s['bpitch']:+5.1f} | body roll {s['roll']:+5.1f} rate {s['wr']:+5.0f} | pitch {s['pitch']:+5.1f} rate {s['wp']:+5.0f}"
                  f" | stance roll peak {s['rmax']:+5.1f} hold {s['hold']:.2f}s stance {s['dur']:.2f}s")
    return dict(how=how, T=tend - t[k0], steps=steps)

if __name__ == "__main__":
    R = [r for r in (analyse(f) for f in sorted(x for x in sys.argv[1:] if not x.endswith("_frames.csv"))) if r]
    allst = [s for r in R for s in r["steps"]]
    print("\nall trials:", len(R), "| ends:", {k: sum(r["how"] == k for r in R) for k in set(r["how"] for r in R)})
    print("time up (s):", " ".join(f"{r['T']:.0f}" for r in R), "| touchdowns:", " ".join(str(len(r["steps"])) for r in R))
    for sd in "LR":
        S = [s for s in allst if s["side"] == sd]
        a = lambda k: np.array([s[k] for s in S])  # noqa: E731
        print(f"{sd} landings n {len(S)}: roll {np.median(a('roll')):+.1f} (p10 {np.percentile(a('roll'),10):+.1f} p90 {np.percentile(a('roll'),90):+.1f}),"
              f" roll rate at landing |median| {np.median(np.abs(a('wr'))):.0f} p90 {np.percentile(np.abs(a('wr')),90):.0f} deg/s,"
              f" stance roll peak |p50| {np.median(np.abs(a('rmax'))):.1f} p90 {np.percentile(np.abs(a('rmax')),90):.1f},"
              f" gate hold p50 {np.median(a('hold')):.2f} p90 {np.percentile(a('hold'),90):.2f} s, pitch {np.median(a('pitch')):+.1f}")
        print(f"   landing board (world, from standing): roll p10 {np.percentile(a('broll'),10):+.1f} p50 {np.median(a('broll')):+.1f} p90 {np.percentile(a('broll'),90):+.1f};"
              f" pitch p10 {np.percentile(a('bpitch'),10):+.1f} p50 {np.median(a('bpitch')):+.1f} p90 {np.percentile(a('bpitch'),90):+.1f};"
              f" |tilt| > 5 deg in {np.mean(np.hypot(a('broll'), a('bpitch')) > 5):.0%}")
    # does the swing grow before falls? stance roll peak, by steps before the end, fallen vs stopped trials
    for lab, sel in (("fallen", [r for r in R if r["how"].startswith("fell")]), ("stopped up", [r for r in R if r["how"] == "stopped up"])):
        if not sel: continue
        rows = []
        for back in range(8, 0, -1):
            v = [abs(r["steps"][-back]["rmax"]) for r in sel if len(r["steps"]) >= back]
            rows.append(f"{np.mean(v):4.1f}")
        first = np.mean([np.mean([abs(s["rmax"]) for s in r["steps"][2:6]]) for r in sel if len(r["steps"]) > 6])
        print(f"{lab}: |stance roll peak| deg, steps 3-6 avg {first:.1f}; last 8 steps before the end: " + " ".join(rows))
