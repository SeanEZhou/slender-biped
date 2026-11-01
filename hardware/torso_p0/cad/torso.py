"""Printable T1 torso: four stacking tubes on the P08 plate, plus two removable hip panels below it.

    .venv-cad/bin/python hardware/torso_p0/cad/export_legs.py <legs_dir>        # once
    .venv-cad/bin/python hardware/torso_p0/cad/torso.py <legs_dir> [--no-check]

Frame: the leg CAD (+X forward, +Y left, +Z up, mm). Shape: T1_shape.json. Arm openings and seams: constants below.
Outputs: exports/print_ready, renders, torso_report.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad"))
sys.path.insert(0, str(ROOT.parent / "leg_p0" / "cad"))
import shape as sh  # noqa: E402

Z0 = sh.PLATE_TOP
WALL = 2.4                 # params.yaml fit.min_wall
SLIDE = 0.15               # per side, sleeve inside the tube above
SLEEVE_T, SLEEVE_H = 1.6, 8.0
SEAMS = (Z0 + 73, Z0 + 120, Z0 + 165)     # waist, under the chest, through the arm-opening centre
ARM_R, ARM_X, ARM_Z = 24.0, -5.0, Z0 + 165
SCREWS = {SEAMS[0]: (180, 75, -75), SEAMS[1]: (180, 75, -75), SEAMS[2]: (40, -40, 140, -140)}   # deg, 0 = front
SLOPED_SEAMS = (SEAMS[2],)                 # steep upper chest: the sleeve follows the wall instead of staying straight
PILOT_DEPTH = 6.5
M3_CLEAR, M3_PILOT = 3.4, 2.9              # pilot: M3 self-tapping into print, set by the thread test
NECK_SCREWS, NECK_PAD = (180, 75, -75), 4.0   # deg, 0 = front; pad depth from the outside, keeps the neck bore open
NUT_AF, NUT_H = 5.7, 2.6                   # params.yaml fit.m3_nut_*
PLATE = dict(x0=-42.0, x1=34.0, y=60.0, z0=25.7, z1=Z0)
PLATE_HOLES = (-30.0, 0.0, 22.0), (-7.0, 7.0)   # biped.part_pelvis_biped
EAR_T, FLOOR_T = 4.0, 3.0
SPLIT_X = -4.0
PANEL_GAP = 0.2            # ear under the plate, tongue over the rear floor
DENSITY = 1.24e-3          # g/mm3, PLA/PETG; thin walls print effectively solid
BED = 256.0
ENGINE = "manifold"


def diff(a, *b):
    return trimesh.boolean.difference([a, *b], engine=ENGINE)


def inter(*a):
    return trimesh.boolean.intersection(list(a), engine=ENGINE)


def union(*a):
    return trimesh.boolean.union(list(a), engine=ENGINE)


def box(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    b.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    return b


def rod(p, d, r, length):
    """Cylinder from point p along unit vector d."""
    c = trimesh.creation.cylinder(radius=r, height=length, sections=48)
    t = trimesh.geometry.align_vectors((0, 0, 1), d)
    t[:3, 3] = np.array(p) + np.array(d) * length / 2
    c.apply_transform(t)
    return c


_cache = {}


def ring(z, inset):
    if inset not in _cache:
        tb = [(zp, a - inset, f - inset, b - inset, c, n) for zp, a, f, b, c, n in sh.BODY]
        _cache[inset] = sh.surface(tb, nt=720, dz=0.5)
    v = _cache[inset]
    return v[int(np.argmin(np.abs(v[:, 0, 2] - z)))]


PHI = np.linspace(-np.pi, np.pi, 360, endpoint=False)


def radial(z, inset, cx):
    """Section radius vs angle PHI about (cx, 0)."""
    p = ring(z, inset)
    a = np.arctan2(p[:, 1], p[:, 0] - cx)
    r = np.hypot(p[:, 1], p[:, 0] - cx)
    o = np.argsort(a)
    return np.interp(PHI, a[o], r[o], period=2 * np.pi)


def loft(rings, cx):
    """Closed solid from [(z, r(PHI))] rings."""
    n = len(PHI)
    v = np.vstack([np.stack([cx + r * np.cos(PHI), r * np.sin(PHI), np.full(n, z)], 1) for z, r in rings])
    f = []
    for i in range(len(rings) - 1):
        a = i * n + np.arange(n)
        b = i * n + (np.arange(n) + 1) % n
        f += [np.stack([a, b, b + n], 1), np.stack([a, b + n, a + n], 1)]
    lo, hi = len(v), len(v) + 1
    v = np.vstack([v, (cx, 0, rings[0][0]), (cx, 0, rings[-1][0])])
    f += [np.stack([np.full(n, lo), (np.arange(n) + 1) % n, np.arange(n)], 1)]
    top = (len(rings) - 1) * n
    f += [np.stack([np.full(n, hi), top + np.arange(n), top + (np.arange(n) + 1) % n], 1)]
    m = trimesh.Trimesh(v, np.concatenate(f), process=True)
    if m.volume < 0:
        m.invert()
    return m


def boss(c, rs, ph, poly):
    """Prism 10 mm wide from section poly (depth inward from radius rs, height above c) at angle ph about c."""
    import manifold3d as m3
    er = np.array([np.cos(ph), np.sin(ph), 0.0])
    et = np.array([-np.sin(ph), np.cos(ph), 0.0])
    mf = m3.CrossSection([poly], m3.FillRule.Positive).extrude(10.0).to_mesh()
    d, z, w = np.asarray(mf.vert_properties)[:, :3].T
    v = c + np.outer(rs - d, er) + np.outer(w - 5.0, et) + np.outer(z, (0, 0, 1))
    b = trimesh.Trimesh(v, np.asarray(mf.tri_verts), process=True)
    if b.volume < 0:
        b.invert()
    return b


def neck_parts(rim, outer):
    """Screw pads for a head fitted over the top SLEEVE_H of the neck. Returns (pads, pilots)."""
    cx = float(np.mean(ring(rim, 0.0)[:, 0]))
    r = radial(rim - SLEEVE_H / 2, 0.0, cx)
    c = np.array([cx, 0.0, rim])
    poly = [(0, -SLEEVE_H - NECK_PAD), (NECK_PAD, -SLEEVE_H), (NECK_PAD, 0), (0, 0)]      # 45 deg underside: prints unsupported
    pads, pilots = [], []
    for deg in NECK_SCREWS:
        ph = np.radians(deg)
        rs = float(np.interp(np.arctan2(np.sin(ph), np.cos(ph)), PHI, r, period=2 * np.pi))
        er = np.array([np.cos(ph), np.sin(ph), 0.0])
        pads.append(inter(boss(c, rs, ph, poly), outer))
        pilots.append(rod(c + er * (rs + 1.0) - np.array((0, 0, SLEEVE_H / 2)), -er, M3_PILOT / 2, NECK_PAD + 3.0))
    return pads, pilots


def surface_normal(z, ph, cx):
    """Outward unit normal of the outer shell at height z, angle ph about (cx, 0)."""
    i = int(np.argmin(np.abs(PHI - np.arctan2(np.sin(ph), np.cos(ph)))))
    er = np.array([np.cos(ph), np.sin(ph), 0.0])
    et = np.array([-np.sin(ph), np.cos(ph), 0.0])
    r0 = radial(z, 0.0, cx)
    dr_dz = (radial(z + 1.0, 0.0, cx)[i] - radial(z - 1.0, 0.0, cx)[i]) / 2.0
    dr_dph = (r0[(i + 3) % len(PHI)] - r0[i - 3]) / (2.0 * (PHI[1] - PHI[0]) * 3)
    n = np.cross(dr_dph * er + r0[i] * et, dr_dz * er + np.array((0, 0, 1.0)))
    return n / np.linalg.norm(n)


def seam_parts(zs):
    """Sleeve on the lower tube and the screw features. Returns (add_to_lower, cut_lower, cut_upper)."""
    cx = float(np.mean(ring(zs, 0.0)[:, 0]))
    zz = np.linspace(zs, zs + SLEEVE_H, 9)
    radii = [radial(z, WALL + SLIDE, cx) for z in zz]
    sloped = zs in SLOPED_SEAMS
    if sloped:
        r_z = np.minimum.accumulate(radii, axis=0)      # follows a wall narrowing upward, still drops straight on
        rings = list(zip(zz, r_z))
    else:
        r_z = np.min(radii, axis=0)[None]               # one straight sleeve inside the narrowest section
        rings = [(zs, r_z[0]), (zs + SLEEVE_H, r_z[0])]
    r_up = r_z[0]
    r_in = r_up - SLEEVE_T
    step = next(h for h in np.arange(4.0, 30.0, 1.0) if (radial(zs - h, WALL, cx) - r_in).max() <= h)     # 45 deg max
    r_low = radial(zs - step, WALL, cx)
    outer = loft([(zs - step, r_low + WALL / 2)] + rings, cx)
    bore = loft([(zs - step - 0.01, r_low)] + [(z, r - SLEEVE_T) for z, r in (rings if sloped else rings[:1])] + [(zs + SLEEVE_H + 1.0, rings[-1][1] - SLEEVE_T)], cx)
    sleeve = diff(outer, bore)
    bosses, cut_l, cut_u = [], [], []
    for deg in SCREWS[zs]:
        ph = np.radians(deg)
        rs = float(np.interp(np.arctan2(np.sin(ph), np.cos(ph)), PHI, r_up, period=2 * np.pi))
        er = np.array([np.cos(ph), np.sin(ph), 0.0])
        c = np.array([cx, 0.0, zs])
        if sloped:                                      # screw square to the wall, so the head seats flat
            zm = zs + SLEEVE_H / 2
            rm = float(np.interp(np.arctan2(np.sin(ph), np.cos(ph)), PHI, r_z[len(zz) // 2], period=2 * np.pi))
            p = c + er * rm + np.array((0, 0, SLEEVE_H / 2))
            n = surface_normal(zm, ph, cx)
            depth = rs - rm + (n @ er) * PILOT_DEPTH + 1.6
            low = SLEEVE_H / 2 - n[2] * PILOT_DEPTH - 1.6
            poly = [(0, low - depth), (depth, low), (depth, SLEEVE_H), (0, SLEEVE_H)]     # 45 deg underside: prints unsupported
        else:
            p = c + er * rs + np.array((0, 0, SLEEVE_H / 2))
            n = er
            poly = [(0, -7.5), (6.5, -1.0), (6.5, SLEEVE_H), (0, SLEEVE_H)]               # 45 deg underside: prints unsupported
        bosses.append(boss(c, rs, ph, poly))
        cut_l.append(rod(p + n * 1.0, -n, M3_PILOT / 2, PILOT_DEPTH + 1.0))
        cut_u.append(rod(p - n * 0.5, n, M3_CLEAR / 2, 8.0))
    keep = union(outer, box(-200, 200, -200, 200, zs - 12, zs))
    return union(sleeve, *[inter(b, keep) for b in bosses]), cut_l, cut_u


def hex_prism(x, y, z0, h):
    p = trimesh.creation.cylinder(radius=NUT_AF / 2 / np.cos(np.pi / 6), height=h, sections=6)
    p.apply_translation((x, y, z0 + h / 2))
    return p


def build():
    outer_full = sh.solid(sh.BODY)
    top = outer_full.bounds[1][2]
    base_shape = sh.cut_legs(outer_full)                       # T1 as approved: no arm openings
    outer = inter(base_shape, box(-200, 200, -200, 200, -100, top - 0.5))
    bottom = outer.bounds[0][2]
    cavity = inter(sh.solid(sh.BODY, inset=WALL), box(-200, 200, -200, 200, bottom + WALL, top + 5))
    skin = diff(outer, cavity)

    edges = (Z0,) + SEAMS + (top,)
    add, cuts_low, cuts_up = {}, {}, {}
    for zs in SEAMS:
        add[zs], cuts_low[zs], cuts_up[zs] = seam_parts(zs)
    arms = [rod((ARM_X, s * 20.0, ARM_Z), (0, s, 0), ARM_R, 100.0) for s in (1, -1)]

    parts = {}
    names = ("T03_pelvis_ring", "T04_midriff", "T05_chest", "T06_shoulders")
    for i, name in enumerate(names):
        z0, z1 = edges[i], edges[i + 1]
        p = inter(skin, box(-200, 200, -200, 200, z0, z1))
        if i < 3:
            p = union(p, add[z1])
            p = diff(p, *cuts_low[z1])
        if i > 0:
            p = diff(p, *cuts_up[z0])
        if i == 3:                                              # neck: same 8 mm overlap and M3x6, for a head fitted over it
            pads, pilots = neck_parts(z1, outer)
            p = diff(union(p, *pads), *pilots)
        if i == 0:                                              # floor on the plate, bolted through its M3 holes
            fl = inter(box(-60, 60, -16, 16, Z0, Z0 + FLOOR_T), outer)
            holes = [rod((x, y, Z0 - 1), (0, 0, 1), M3_CLEAR / 2, FLOOR_T + 2) for x in PLATE_HOLES[0] for y in PLATE_HOLES[1]]
            p = union(p, diff(fl, *holes))
        p = diff(p, *arms)
        parts[name] = p

    plate = box(PLATE["x0"] - 0.3, PLATE["x1"] + 0.3, -PLATE["y"] - 0.3, PLATE["y"] + 0.3, PLATE["z0"], PLATE["z1"] + 1)
    low = diff(inter(skin, box(-200, 200, -200, 200, -100, Z0)), plate)
    front = inter(low, box(SPLIT_X, 200, -200, 200, -100, 100))
    rear = inter(low, box(-200, SPLIT_X, -200, 200, -100, 100))
    z_e0 = PLATE["z0"] - EAR_T
    ear_f = inter(box(14, 60, -14, 14, z_e0, PLATE["z0"] - PANEL_GAP), outer)
    ear_r = inter(box(-70, -22, -14, 14, z_e0, PLATE["z0"] - PANEL_GAP), outer)
    for x in PLATE_HOLES[0]:
        tgt = 1 if x > 10 else (-1 if x < -10 else 0)
        if tgt:
            for y in PLATE_HOLES[1]:
                h = [rod((x, y, z_e0 - 1), (0, 0, 1), M3_CLEAR / 2, EAR_T + 2), hex_prism(x, y, z_e0 - 0.01, NUT_H)]
                if tgt > 0:
                    ear_f = diff(ear_f, *h)
                else:
                    ear_r = diff(ear_r, *h)
    # crotch lap: a tongue on the front panel over the rear panel's floor, one screw from below
    f0 = bottom + WALL
    tongue = union(box(-4.01, 2, -6, 6, f0 - 0.01, f0 + 5), box(-16, -4, -6, 6, f0 + PANEL_GAP, f0 + 5))
    tongue = diff(inter(tongue, outer), rod((-10, 0, f0), (0, 0, 1), M3_PILOT / 2, 4.5))
    parts["T01_hip_front"] = union(front, ear_f, tongue)
    parts["T02_hip_rear"] = diff(union(rear, ear_r), rod((-10, 0, bottom - 1), (0, 0, 1), M3_CLEAR / 2, WALL + 2))
    return parts


FLIP = {"T01_hip_front", "T02_hip_rear"}     # hip panels print top face down, narrowing upward


def bed_pose(name, m):
    t = m.copy()
    if name in FLIP:
        t.apply_transform(trimesh.transformations.rotation_matrix(np.pi, (1, 0, 0)))
    c = t.bounds.mean(axis=0)
    t.apply_translation((BED / 2 - c[0], BED / 2 - c[1], -t.bounds[0][2]))
    return t


def overhang(t):
    """Share of surface steeper than 50 deg from vertical and facing down, excluding the bed face (mm2, deg)."""
    n, a, zc = t.face_normals, t.area_faces, t.triangles_center[:, 2]
    down = (n[:, 2] < -np.sin(np.radians(50))) & (zc > 0.3)
    return round(float(a[down].sum())), round(float(np.degrees(np.arcsin(np.clip(-n[zc > 0.3, 2].min(), 0, 1)))))


def check(parts, legs_dir):
    import biped
    from build123d import export_stl
    rep = {"pieces": {}}
    for name, m in parts.items():
        t = bed_pose(name, m)
        area, worst = overhang(t)
        rep["pieces"][name] = dict(watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)),
                                   bbox_on_bed_mm=[round(float(v), 1) for v in t.extents],
                                   fits_bed=bool(t.extents.max() <= BED), mass_g=round(m.volume * DENSITY),
                                   overhang_over_50deg_mm2=area)
    rep["total_mass_g"] = sum(p["mass_g"] for p in rep["pieces"].values())
    com = sum(m.volume * m.center_mass for m in parts.values()) / sum(m.volume for m in parts.values())
    rep["com_mm"] = [round(float(v), 1) for v in com]
    rep["between_pieces_mm3"] = {f"{a}+{b}": round(float(inter(parts[a], parts[b]).volume), 1)
                                 for a, b in itertools.combinations(parts, 2)
                                 if (parts[a].bounds[0] < parts[b].bounds[1]).all() and (parts[b].bounds[0] < parts[a].bounds[1]).all()}
    fixed = {}
    for f in sorted(Path(legs_dir).glob("*.stl")):
        if "pelvis_biped" in f.name or "L_servo_hip_yaw" in f.name:
            part = trimesh.load(f)
            fixed[f.stem] = {n: round(float(inter(m, part).volume), 1) for n, m in parts.items()}
    rep["fixed_parts_mm3"] = fixed
    items = [i for i in biped.leg_items() if i.kind in ("part", "servo", "horn")]
    poses = [dict(zip(("yaw", "roll", "pitch", "knee"), p)) for p in itertools.product((-15, 0, 25), (-10, 0, 25), (-25, 0, 80), (0,))]
    poses += [dict(yaw=0, roll=0, pitch=16, knee=32), dict(yaw=0, roll=25, pitch=80, knee=105)]
    hits = []
    with tempfile.TemporaryDirectory() as tmp:
        for q in poses:
            left, _ = biped.posed_pair(items, biped.q(**q), biped.q())
            for it in left:
                if it.link.endswith("pelvis") or it.shape.bounding_box().max.Z < -70:
                    continue
                p = f"{tmp}/x.stl"
                export_stl(it.shape, p, tolerance=0.2, angular_tolerance=0.3)
                pm = trimesh.load(p)
                for n, m in parts.items():
                    if pm.bounds[1][2] > m.bounds[0][2]:
                        v = float(inter(m, pm).volume)
                        if v > 1.0:
                            hits.append(dict(pose=q, part=it.name, piece=n, mm3=round(v)))
    rep["moving_parts"] = dict(poses=len(poses), hits=hits)
    rep["panel_slide_in_hits"] = slide_check(parts, legs_dir)
    return rep


def slide_check(parts, legs_dir):
    """Hip panels slide in along X with the legs on: test every 4 mm along the path, stance pose, both legs."""
    fixed = [(f.stem, trimesh.load(f)) for f in sorted(Path(legs_dir).glob("stance_*.stl"))]
    hits = []
    for name, sign in (("T01_hip_front", 1), ("T02_hip_rear", -1)):
        lo = parts[name].bounds[0][2] - 1
        near = [(n, m) for n, m in fixed if m.bounds[1][2] > lo]
        for dx in np.arange(4.0, 84.0, 4.0):
            t = parts[name].copy()
            t.apply_translation((sign * dx, 0, 0))
            for n, m in near:
                if (t.bounds[0] < m.bounds[1]).all() and (m.bounds[0] < t.bounds[1]).all():
                    v = float(inter(t, m).volume)
                    if v > 1.0:
                        hits.append(dict(piece=name, dx=float(dx), part=n, mm3=round(v)))
    return hits


def leg_colour(item):
    if item["kind"] == "servo":
        return "0.03 0.03 0.035 1"
    return "0.85 0.65 0.25 1" if item["kind"] == "horn" else "0.96 0.96 0.96 1"


def render(parts, legs_dir):
    import mujoco
    from PIL import Image
    man = json.loads((Path(legs_dir) / "manifest.json").read_text())
    shades = dict(T01_hip_front=.96, T02_hip_rear=.84, T03_pelvis_ring=.84, T04_midriff=.96, T05_chest=.84, T06_shoulders=.96)
    ex = dict(T01_hip_front=(70, 0, -25), T02_hip_rear=(-70, 0, -25), T03_pelvis_ring=(0, 0, 25), T04_midriff=(0, 0, 60),
              T05_chest=(0, 0, 95), T06_shoulders=(0, 0, 135))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        asset = [f'<mesh name="{i["name"]}" file="{Path(legs_dir) / i["file"]}" scale=".001 .001 .001" inertia="shell"/>' for i in man]
        for n, m in parts.items():
            m.export(f"{tmp}/{n}.stl")
            asset.append(f'<mesh name="{n}" file="{tmp}/{n}.stl" scale=".001 .001 .001" inertia="shell"/>')
        for tag, spread in (("exploded", 1),):
            rows = []
            for yaw in (0, -38, -90, -150):
                g = "".join(f'<geom type="mesh" mesh="{i["name"]}" rgba="{leg_colour(i)}"/>' for i in man)
                for n in parts:
                    s = shades[n]
                    o = [spread * v * 0.001 for v in ex[n]]
                    g += f'<geom type="mesh" mesh="{n}" pos="{o[0]} {o[1]} {o[2]}" rgba="{s} {s} {s} 1"/>'
                xml = f"""<mujoco><compiler angle="degree"/>
<visual><global offwidth="1600" offheight="1200"/><headlight ambient=".42 .42 .42" diffuse=".5 .5 .5" specular=".1 .1 .1"/></visual>
<asset><texture type="skybox" builtin="flat" rgb1=".5 .52 .55" rgb2=".5 .52 .55" width="8" height="8"/>{"".join(asset)}</asset>
<default><geom contype="0" conaffinity="0"/></default>
<worldbody><light pos="1 .6 2" dir="-.5 -.3 -1" directional="true" diffuse=".35 .35 .35"/>
<body pos="0 0 {-sh.FLOOR * 0.001}" euler="0 0 {yaw}">{g}</body></worldbody></mujoco>"""
                mdl = mujoco.MjModel.from_xml_string(xml)
                d = mujoco.MjData(mdl)
                mujoco.mj_forward(mdl, d)
                r = mujoco.Renderer(mdl, 900, 500)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = (0, 0, 0.40)
                cam.azimuth, cam.elevation, cam.distance = 180, 0, 2.0
                mdl.vis.global_.orthographic = 1
                mdl.vis.global_.fovy = 0.92
                r.update_scene(d, camera=cam)
                rows.append(r.render().copy())
                r.close()
            p = ROOT / "renders" / f"torso_{tag}.png"
            p.parent.mkdir(exist_ok=True)
            Image.fromarray(np.hstack(rows)).save(p)
            out[tag] = str(p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("legs")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--assembled", default="", help="also write robot-frame STLs to this folder")
    a = ap.parse_args()
    parts = build()
    (ROOT / "exports" / "print_ready").mkdir(parents=True, exist_ok=True)
    for name, m in parts.items():
        bed_pose(name, m).export(ROOT / "exports" / "print_ready" / f"{name}_print.stl")
    if a.assembled:                                             # robot-frame copies for renders; not kept in the repo
        Path(a.assembled).mkdir(parents=True, exist_ok=True)
        for name, m in parts.items():
            m.export(Path(a.assembled) / f"{name}.stl")
    print(render(parts, a.legs))
    if not a.no_check:
        rep = check(parts, a.legs)
        (ROOT / "torso_report.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps({k: v for k, v in rep.items() if k != "moving_parts"}, indent=0))
        print("moving", rep["moving_parts"]["poses"], "poses, hits", len(rep["moving_parts"]["hits"]), rep["moving_parts"]["hits"][:10])
