"""Round 18 (2026-11-16): how each trial ended, and how high each swing foot actually got on the robot.

Per trial: time walking, how it ended (tilt > 20 deg = a fall), touchdowns, falls per touchdown.
Per swing (the run's reference table gives lift-off / touchdown u): the swinging board's lowest corner above the standing
board's lowest corner (encoder FK + IMU attitude), measured and commanded (FK of the cmd_ joints, same attitude):
peak over the swing, and the share of the swing's middle half spent below 5 / 10 mm (skimming the floor).
Also the reference table's own clearance (flat body, ref joints only) for comparison.
Usage (repo root): .venv/bin/python scripts/robot18_analysis/swing_clear.py <stand_*.csv> ...
"""
import sys, csv, math, json, numpy as np
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import mujoco, gap_fit

ROOT = Path(__file__).resolve().parents[2]
_m, _d, _tid, _ = gap_fit._fk_setup(gap_fit.MODEL)
J = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_roll", "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_roll"]
_C = np.array([[sx, sy, -1] for sx in (-1, 1) for sy in (-1, 1)], float)


def low(q, g):
    """Lowest corner z (mm) of the L / R boards, body attitude from gravity g (pelvis frame)."""
    _d.qpos[:] = 0; _d.qpos[2] = 0.3; _d.qpos[3:7] = gap_fit._quat_from_g(g) if g is not None else [1, 0, 0, 0]; _d.qpos[7:] = q
    mujoco.mj_kinematics(_m, _d)
    out = []
    for gid in _tid:
        R = _d.geom_xmat[gid].reshape(3, 3); c = _d.geom_xpos[gid]
        out.append(1000 * min((c + R @ (_C[k] * _m.geom_size[gid]))[2] for k in range(4)))
    return np.array(out)


def table(run):
    cfg = json.loads((ROOT / "runs" / run / "config.json").read_text())["env"]
    z = np.load(ROOT / "models" / "ref" / f"{cfg['ref_table']}.npz")
    st = z["stance"][-1]; n = len(st); ev = []
    for i in range(n):
        for leg, side in ((0, "L"), (1, "R")):
            if st[i - 1, leg] >= 0.5 > st[i, leg]:
                ev.append((i / n, "lo", side))
            if st[i - 1, leg] < 0.5 <= st[i, leg]:
                ev.append((i / n, "td", side))
    return cfg["ref_table"], z, ev


def ref_clear(z, ev):
    """The table's own swing clearance (flat body), peak per side, mm."""
    q = z["q"][-1] if z["q"].ndim == 3 else z["q"]
    n = len(q); L = np.array([low(q[i], None) for i in range(n)])
    return {s: float(np.max(np.abs(L[:, 0] - L[:, 1]))) for s in "LR"}


def analyse(f):
    r = list(csv.reader(open(f))); h = r[0]; r = [x for x in r[1:] if len(x) == len(h)]
    if len(r) < 50:
        return None
    A = np.array([[float(v) if v not in ("True", "False") else float(v == "True") for v in x] for x in r])
    col = lambda k: A[:, h.index(k)]  # noqa: E731
    run = f.split("/")[-1][6:-20]
    tab, z, ev = table(run)
    t = col("t"); g = A[:, [h.index(k) for k in ("gx", "gy", "gz")]]
    q = A[:, [h.index(f"q_{n}") for n in J]]; qc = A[:, [h.index(f"cmd_{n}") for n in J]]
    u = col("phase") / (2 * math.pi)
    eng = np.where(np.abs(np.diff(u)) > 1e-6)[0]
    if not len(eng):
        return None
    k0 = eng[0]
    roll = np.degrees(np.arctan2(-g[:, 1], -g[:, 2])); pitch = np.degrees(np.arctan2(g[:, 0], -g[:, 2]))
    roll -= roll[max(0, k0 - 25):k0 + 1].mean(); pitch -= pitch[max(0, k0 - 25):k0 + 1].mean()
    bad = np.where(np.hypot(roll, pitch) > 20)[0]; bad = bad[bad > k0]
    if len(bad):
        b = bad[0]; how = ("fell L" if roll[b] < 0 else "fell R") if abs(roll[b]) > abs(pitch[b]) else ("fell fwd" if pitch[b] > 0 else "fell back")
    else:
        b, how = len(t) - 1, "stopped up"
    # event crossings
    cross = []
    for i in range(k0, b):
        for (ue, kind, side) in ev:
            if u[i] != u[i + 1] and (u[i] - ue) % 1.0 > (u[i + 1] - ue) % 1.0:
                cross.append((i + 1, kind, side))
    Lm = np.array([low(q[i], g[i]) for i in range(k0, b + 1)]); Lc = np.array([low(qc[i], g[i]) for i in range(k0, b + 1)])
    sw = []
    for a, (i, kind, side) in enumerate(cross):
        if kind != "lo":
            continue
        nx = [c for c in cross[a + 1:] if c[1] == "td" and c[2] == side]
        if not nx:
            continue
        j = nx[0][0]; fi = 0 if side == "L" else 1
        cm = Lm[i - k0:j - k0, fi] - Lm[i - k0:j - k0, 1 - fi]; cc = Lc[i - k0:j - k0, fi] - Lc[i - k0:j - k0, 1 - fi]
        if len(cm) < 4:
            continue
        mid = cm[len(cm) // 4: 3 * len(cm) // 4]
        sw.append(dict(side=side, peak=cm.max(), cpeak=cc.max(), lt5=np.mean(mid < 5), lt10=np.mean(mid < 10), dur=t[j] - t[i]))
    tds = sum(1 for c in cross if c[1] == "td")
    return dict(f=f.split("/")[-1], run=run, tab=tab, how=how, T=t[b] - t[k0], tds=tds, sw=sw, ref=ref_clear(z, ev))


if __name__ == "__main__":
    R = [x for x in (analyse(f) for f in sorted(a for a in sys.argv[1:] if not a.endswith("_frames.csv"))) if x]
    for x in R:
        print(f"{x['f']}: {x['how']} after {x['T']:.1f} s, {x['tds']} touchdowns")
    print(f"\n{'run':14s} trials falls touchdowns falls/td | swing peak clearance mm: measured p10/p50/p90, commanded p50, table |"
          " mid-swing < 5 / < 10 mm | swing s")
    for run in dict.fromkeys(x["run"] for x in R):
        X = [x for x in R if x["run"] == run]; S = [s for x in X for s in x["sw"]]
        a = lambda k: np.array([s[k] for s in S])  # noqa: E731
        nf = sum(x["how"] != "stopped up" for x in X); ntd = sum(x["tds"] for x in X)
        print(f"{run:14s} {len(X):3d} {nf:5d} {ntd:7d} {nf / max(ntd, 1):8.1%} | {np.percentile(a('peak'), 10):5.1f} {np.median(a('peak')):5.1f}"
              f" {np.percentile(a('peak'), 90):5.1f}   {np.median(a('cpeak')):5.1f}   {X[0]['ref']['L']:5.1f} | {np.mean(a('lt5')):5.0%} {np.mean(a('lt10')):5.0%}"
              f" | {np.median(a('dur')):.2f}")
