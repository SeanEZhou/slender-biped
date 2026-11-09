"""Foot-pad variants of the robot model: same body and servos, only the pads at the shin tips change.

    .venv-hw/bin/python scripts/make_pads.py            # writes models/hw_pad_<name>_8dof{,_train}.{xml,json}
    .venv-hw/bin/python scripts/make_pads.py --list

Each pad is built from MuJoCo primitives in a "pad frame": origin at the centre of the sole, x forward, y to the
robot's left, z up, sole plane z = 0 (the same height as the original 31 x 18 mm pad, so the standing pose, the
policy inputs and the stand height are unchanged). The pad frame is the original tip's frame, which is pitched so
the sole is level in the stance pose.

Geom naming: the first primitive is `<S>_tip` (S = L/R), the rest `<S>_tip_<k>`. The env treats every `<S>_tip*`
geom as foot: tip height = lowest point over the foot's geoms, every one gets a floor contact pair (and the floor
randomization), and left pads collide with right pads and the other shin.

Mass: printed PLA at ~0.75 g/cm^3 effective (walls + 25 % infill) from the primitives' volume, plus 2 g of screws,
added to the shin's inertial. `pad_area` (m^2, sole footprint) goes in the model JSON: the env scales the rug
softness by it (a bigger sole sinks less into the same rug).
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
BASE = "hw_p0p191_8dof"
RHO = 750.0          # kg/m^3, printed PLA effective
SCREWS = 0.0         # the original pad's screws are already in the CAD shin
ORIG_PAD = 0.031209 * 0.018 * 0.005 * RHO   # the original pad, already in the CAD shin: only the difference is added
# Torso and head on the pelvis (not in the leg CAD model): (mass kg, COM in the pelvis frame m, box dims for inertia m).
# hardware/torso_p0/torso_report.json (213 g at (-2.9, 0, 111.9) mm, printed) and hardware/head_p0/head_report.json
# (H1, 160 g solid at (-17.4, 0.3, 320.8) mm; printed with infill it is lighter: 140 g nominal, the env's per-body
# mass randomization covers the spread). The torso/head CAD frame is the leg CAD frame = the sim pelvis frame.
PAYLOAD = [(0.213, (-0.0029, 0.0, 0.1119), (0.090, 0.110, 0.240)),
           (0.140, (-0.0174, 0.0003, 0.3208), (0.080, 0.075, 0.090))]


# ------------------------------------------------------------------ primitives (pad frame, metres)
def box(cx, cy, L, W, t, z0=0.0, pitch=0.0):
    return dict(type="box", pos=[cx, cy, z0 + t / 2], size=[L / 2, W / 2, t / 2], pitch=pitch)


def disc(cx, cy, R, t):
    return dict(type="cylinder", pos=[cx, cy, t / 2], size=[R, t / 2], pitch=0.0)


def dome(cx, cy, R, h):
    return dict(type="ellipsoid", pos=[cx, cy, h], size=[R, R, h], pitch=0.0)


def claw(angle_deg, length, r, hub=0.0):
    """Capsule lying on the sole plane from the hub out to `length` along heading angle (0 = forward)."""
    a = math.radians(angle_deg)
    u = np.array([math.cos(a), math.sin(a), 0.0])
    p1, p2 = u * hub, u * length
    return dict(type="capsule", fromto=[*(p1 + [0, 0, r]), *(p2 + [0, 0, r])], size=[r], pitch=0.0)


def bar(L, W, t=0.004):
    """Collision stand-in for a long pad: capsule along x, as wide as the pad."""
    r = W / 2
    return dict(type="capsule", fromto=[-(L / 2 - r), 0, t / 2, L / 2 - r, 0, t / 2], size=[r], pitch=0.0)


def ball(R, t=0.004):
    return dict(type="sphere", pos=[0, 0, t / 2], size=[R], pitch=0.0)


def snowboard(L, W, t=0.004, kick=0.012, kick_deg=25.0):
    """Flat board of length L (incl. the kicks) with the last `kick` m at each end bent up by kick_deg."""
    Lf = L - 2 * kick
    k = math.radians(kick_deg)
    parts = [box(0, 0, Lf, W, t)]
    for s in (1, -1):                               # kick sections, hinged at the flat's bottom end edge
        cx = s * (Lf / 2 + (kick / 2) * math.cos(k)) - s * (t / 2) * math.sin(k)
        cz = (kick / 2) * math.sin(k) + (t / 2) * math.cos(k)
        parts.append(dict(type="box", pos=[cx, 0, cz], size=[kick / 2, W / 2, t / 2], pitch=-s * k))
    return parts


def _vol(p):
    s = p["size"]
    if p["type"] == "box":
        return 8 * s[0] * s[1] * s[2]
    if p["type"] == "cylinder":
        return math.pi * s[0] ** 2 * 2 * s[1]
    if p["type"] == "ellipsoid":
        return 4 / 3 * math.pi * s[0] * s[1] * s[2] / 2          # printed as a half-dome
    if p["type"] == "capsule":
        f = np.array(p["fromto"])
        return math.pi * s[0] ** 2 * np.linalg.norm(f[3:] - f[:3]) + 4 / 3 * math.pi * s[0] ** 3
    raise ValueError(p["type"])


# name: (primitives, sole footprint area m^2, description[, proxies]): proxies = cheap stand-ins used only for the
# left-pad vs right-pad (and pad vs other shin) collisions (box-box costs MJX a lot); default: the capsule primitives
def designs():
    d = {}
    d["orig"] = ([box(0, 0, 0.031209, 0.018, 0.005)], 0.031209 * 0.018, "original 31 x 18 mm pad",
                 [bar(0.031209, 0.018)])
    for L in (60, 90, 120):
        for W in (36,):
            Lm, Wm = L / 1000, W / 1000
            d[f"board{L}"] = (snowboard(Lm, Wm), (Lm - 0.024) * Wm,
                              f"snowboard {L} x {W} mm, 12 mm kicks at both ends bent up 25 deg", [bar(Lm, Wm)])
    d["board90w"] = (snowboard(0.090, 0.050), (0.090 - 0.024) * 0.050, "snowboard 90 x 50 mm", [bar(0.090, 0.050)])
    d["board120w"] = (snowboard(0.120, 0.050), (0.120 - 0.024) * 0.050, "snowboard 120 x 50 mm", [bar(0.120, 0.050)])
    d["board150"] = (snowboard(0.150, 0.040, kick=0.015), (0.150 - 0.030) * 0.040,
                     "snowboard 150 x 40 mm, 15 mm kicks", [bar(0.150, 0.040)])
    d["board150w"] = (snowboard(0.150, 0.050, kick=0.015), (0.150 - 0.030) * 0.050,
                      "snowboard 150 x 50 mm, 15 mm kicks", [bar(0.150, 0.050)])
    for D in (50, 70):
        R = D / 2000
        d[f"disc{D}"] = ([disc(0, 0, R, 0.004)], math.pi * R * R, f"flat round pad, {D} mm across", [ball(R)])
        d[f"dome{D}"] = ([dome(0, 0, R, 0.006)], math.pi * (R * 0.5) ** 2,
                         f"domed round pad, {D} mm across, 6 mm tall (contact patch ~half the radius)", [ball(R)])
    for Lc in (45, 60):
        r = 0.004
        parts = [disc(0, 0, 0.009, 0.004)] + [claw(a, Lc / 1000, r, hub=0.006) for a in (0, 120, -120)]
        d[f"claw{Lc}"] = (parts, math.pi * 0.009 ** 2 + 3 * 2 * r * (Lc / 1000 - 0.006),
                          f"three {Lc} mm claws (8 mm thick) at 0 / +-120 deg from an 18 mm hub")
        parts = [disc(0, 0, 0.009, 0.004)] + [claw(a, Lc / 1000, r, hub=0.006) for a in (30, -30, 180)]
        d[f"talon{Lc}"] = (parts, math.pi * 0.009 ** 2 + 3 * 2 * r * (Lc / 1000 - 0.006),
                           f"bird foot: two {Lc} mm toes forward at +-30 deg and one back")
    return d


# ------------------------------------------------------------------ maths
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def f6(v):
    return " ".join(f"{x:.6g}" for x in v)


def combine(J0, m0, p0, pts):
    """Inertia (about the new COM), mass and COM of a body plus point/box masses [(m, pos, J_own)]."""
    allp = [(m0, p0, J0)] + pts
    M = sum(m for m, _, _ in allp)
    com = sum(m * np.asarray(p) for m, p, _ in allp) / M
    J = np.zeros((3, 3))
    for m, p, Jo in allp:
        d = np.asarray(p) - com
        J += Jo + m * (d @ d * np.eye(3) - np.outer(d, d))
    return J, M, com


def add_payload(xml):
    bm = re.search(r'(<body name="pelvis"[^\n]*\n[^\n]*\n\s*)<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"/>',
                   xml)
    p0 = np.array([float(x) for x in bm.group(2).split()])
    m0 = float(bm.group(3))
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = [float(x) for x in bm.group(4).split()]
    J0 = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    pts = [(m, np.array(c), m / 12 * np.diag([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2]))
           for m, c, d in PAYLOAD]
    J, M, com = combine(J0, m0, p0, pts)
    fi = [J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]]
    return xml.replace(bm.group(0), f'{bm.group(1)}<inertial pos="{f6(com)}" mass="{M:.6g}" fullinertia="{f6(fi)}"/>')


def build(xml: str, meta: dict, name: str, parts, area, payload=True, proxies=None):
    out = xml
    tip_line = {}
    for S in "LR":
        m = re.search(rf'^(\s*)<geom name="{S}_tip" ([^\n]*)/>\n', out, re.M)
        indent, attrs = m.group(1), m.group(2)
        pos = np.array([float(x) for x in re.search(r'pos="([^"]+)"', attrs).group(1).split()])
        quat = np.array([float(x) for x in re.search(r'quat="([^"]+)"', attrs).group(1).split()])
        half_h = float(re.search(r'size="([^"]+)"', attrs).group(1).split()[2])
        R0 = quat_R(quat)
        sole = pos + R0 @ np.array([0, 0, -half_h])                      # sole centre, shin frame
        fr = re.search(r'friction="([^"]+)"', attrs).group(1)
        lines, masses = [], []
        prox = proxies if proxies is not None else [p for p in parts if p["type"] == "capsule"]
        allp = [(f"{S}_tip" if k == 0 else f"{S}_tip_{k}", p, True) for k, p in enumerate(parts)]
        allp += [(f"{S}_padcol_{k}", p, False) for k, p in enumerate(prox)]
        for gname, p, is_foot in allp:
            if p["type"] == "capsule":
                f = np.array(p["fromto"])
                a, b = sole + R0 @ f[:3], sole + R0 @ f[3:]
                geo = f'type="capsule" size="{f6(p["size"])}" fromto="{f6([*a, *b])}"'
                c_pad = (f[:3] + f[3:]) / 2
            else:
                qp = np.array([math.cos(p["pitch"] / 2), 0, math.sin(p["pitch"] / 2), 0])
                q = quat_mul(quat, qp)
                c_pad = np.array(p["pos"])
                geo = f'type="{p["type"]}" size="{f6(p["size"])}" pos="{f6(sole + R0 @ c_pad)}" quat="{f6(q)}"'
            cls = "foot" if is_foot else "col"
            lines.append(f'{indent}<geom name="{gname}" class="{cls}" {geo} mass="0" friction="{fr}"/>\n')
            if is_foot:
                masses.append((RHO * _vol(p), sole + R0 @ c_pad))
        masses.append((-ORIG_PAD, sole + R0 @ np.array([0, 0, 0.0025])))
        tip_line[S] = (m.group(0), "".join(lines), masses)
    for S in "LR":
        old, new, masses = tip_line[S]
        out = out.replace(old, new)
        # fold the pad mass into the shin inertial (point masses + screws at the sole centre; own inertia ignored)
        bm = re.search(rf'(<body name="{S}_shin"[^\n]*\n[^\n]*\n\s*)<inertial pos="([^"]+)" mass="([^"]+)" '
                       rf'fullinertia="([^"]+)"/>', out)
        p0 = np.array([float(x) for x in bm.group(2).split()])
        m0 = float(bm.group(3))
        I = np.array([float(x) for x in bm.group(4).split()])
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz = I
        J = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        pts = [(m0, p0)] + masses + [(SCREWS, masses[0][1])]
        M = sum(mm for mm, _ in pts)
        com = sum(mm * pp for mm, pp in pts) / M
        Jc = J.copy()
        for mm, pp in pts:
            dvec = pp - com
            Jc += mm * (np.dot(dvec, dvec) * np.eye(3) - np.outer(dvec, dvec))
        fi = [Jc[0, 0], Jc[1, 1], Jc[2, 2], Jc[0, 1], Jc[0, 2], Jc[1, 2]]
        out = out.replace(bm.group(0), f'{bm.group(1)}<inertial pos="{f6(com)}" mass="{M:.6g}" fullinertia="{f6(fi)}"/>')
    if payload:
        out = add_payload(out)
    # contact pairs: replace every pair that mentions a tip, then add the full pad set
    pair_lines = re.findall(r'^\s*<pair [^\n]*\n', out, re.M)
    floor_tip = next(p for p in pair_lines if 'geom1="floor" geom2="L_tip"' in p)
    leg_leg = next(p for p in pair_lines if 'geom1="L_thigh_col_l" geom2="R_thigh_col_l"' in p)
    indent = re.match(r"^(\s*)", floor_tip).group(1)
    for p in pair_lines:
        if "_tip" in p:
            out = out.replace(p, "", 1)
    # the lower shin capsule ends ~8 mm above the sole: on a deep rug the pad sinks that far and the fur brushes the
    # shin tip, which is harmless (and the training model has no shin-floor contact): no floor pair for it
    for p in pair_lines:
        if 'geom1="floor"' in p and "_shin_col_l" in p:
            out = out.replace(p, "", 1)
    n = len(parts)
    gl = ["L_tip"] + [f"L_tip_{k}" for k in range(1, n)]
    gr = ["R_tip"] + [f"R_tip_{k}" for k in range(1, n)]
    ft_attrs = floor_tip.strip()[len('<pair geom1="floor" geom2="L_tip" '):]
    ll_attrs = leg_leg.strip()[len('<pair geom1="L_thigh_col_l" geom2="R_thigh_col_l" '):]
    new = [f'{indent}<pair geom1="floor" geom2="{g}" {ft_attrs}\n' for g in gl + gr]
    npx = len(proxies if proxies is not None else [p for p in parts if p["type"] == "capsule"])
    pl = [f"L_padcol_{k}" for k in range(npx)]
    pr = [f"R_padcol_{k}" for k in range(npx)]
    new += [f'{indent}<pair geom1="{a}" geom2="{b}" {ll_attrs}\n' for a in pl for b in pr]
    new += [f'{indent}<pair geom1="{a}" geom2="R_shin_col_l" {ll_attrs}\n' for a in pl]
    new += [f'{indent}<pair geom1="L_shin_col_l" geom2="{b}" {ll_attrs}\n' for b in pr]
    first_pair = re.search(r'^\s*<pair ', out, re.M)
    out = out[:first_pair.start()] + "".join(new) + out[first_pair.start():]
    meta = json.loads(json.dumps(meta))
    meta["name"] = f"hw_pad_{name}_8dof"
    meta["cfg"]["variant"] = f"hw_pad_{name}"
    meta["cfg"]["foot"] = f"pad_{name}"
    meta["pad_area"] = area
    meta["payload"] = "torso + H1 head (140 g)" if payload else "none"
    meta["pad_mass"] = sum(mm for mm, _ in tip_line["L"][2]) + SCREWS
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--legs-only", action="store_true",
                    help="no torso/head on the pelvis (the robot as tested in docs/sim_to_real.md): hw_pad_<d>_legs_8dof")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    ds = designs()
    if a.list:
        for k, (parts, area, desc, *_) in ds.items():
            print(f"{k:10s} {area * 1e4:6.1f} cm^2  {desc}")
        return
    meta = json.loads((MODELS / f"{BASE}.json").read_text())
    for k in a.names or ds:
        parts, area, desc, *px = ds[k]
        for suffix in ("", "_train"):
            xml = (MODELS / f"{BASE}{suffix}.xml").read_text()
            tag = f"{k}_legs" if a.legs_only else k
            out, m = build(xml, meta, tag, parts, area, payload=not a.legs_only, proxies=px[0] if px else None)
            (MODELS / f"hw_pad_{tag}_8dof{suffix}.xml").write_text(out)
        m["pad_desc"] = desc
        (MODELS / f"hw_pad_{tag}_8dof.json").write_text(json.dumps(m, indent=1))
        print(f"hw_pad_{tag}_8dof: {len(parts)} geoms per foot, area {area * 1e4:.1f} cm^2, "
              f"pad mass {m['pad_mass'] * 1000:.1f} g")


if __name__ == "__main__":
    main()
