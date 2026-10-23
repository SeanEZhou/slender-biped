"""Measured morphology: silhouette dimensions, packaging, mass and inertia audits.

All numbers are measured from the compiled MuJoCo model in the `stand` keyframe,
not copied from the config, so they reflect what is actually simulated.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .render import stand_data


def geom_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def mesh_world_vertices(m, d, gid):
    mid = m.geom_dataid[gid]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[a:a + n]
    return v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]


def exterior_names(m):
    out = []
    for g in range(m.ngeom):
        if m.geom_group[g] == 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            out.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
    return out


def span(pts, axis):
    return float(pts[:, axis].max() - pts[:, axis].min())


def box_surface_points(m, d, gid, n=6):
    h = m.geom_size[gid]
    R = d.geom_xmat[gid].reshape(3, 3)
    pts = []
    g = np.linspace(-1, 1, n)
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        for sgn in (-1, 1):
            for u in g:
                for v in g:
                    p = np.zeros(3)
                    p[ax] = sgn
                    p[o[0]], p[o[1]] = u, v
                    pts.append(p * h)
    return np.array(pts) @ R.T + d.geom_xpos[gid]


def packaging_check(m, d, tol=-0.0012):
    """tol < 0 demands that much clearance inside the exterior (room for the printed wall)."""
    """For every servo envelope: max distance its surface protrudes outside the union of exterior hulls."""
    hulls = []
    for nm in exterior_names(m):
        gid = geom_id(m, nm)
        hulls.append((nm, ConvexHull(mesh_world_vertices(m, d, gid))))
    res = []
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not nm.startswith("env_"):
            continue
        pts = box_surface_points(m, d, g)
        # signed "outside" distance per hull = max over facet planes (exact outside a convex hull up to corners)
        dist = np.full(len(pts), np.inf)
        for _, h in hulls:
            sd = (pts @ h.equations[:, :3].T + h.equations[:, 3]).max(1)
            dist = np.minimum(dist, sd)
        worst = float(dist.max())
        res.append({"envelope": nm, "max_protrusion_mm": round(worst * 1000, 2),
                    "frac_outside": float((dist > tol).mean()), "ok": worst <= tol})
    return res


def contacts_at_stand(m, d):
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        out.append((n1, n2, float(c.dist)))
    return out


def rocker_arc(m, gid, max_pitch_deg=20.0):
    """Fore-aft arc length of the tip surface usable as the contact point over +/-max_pitch.

    For a sphere the contact point is fixed relative to the shin regardless of tip pitch
    (zero CoP authority); for the ellipsoid it rolls along the sagittal profile.
    """
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return 0.0, float(m.geom_size[gid][0])
    a, _, c = m.geom_size[gid]
    th = np.radians(np.linspace(-max_pitch_deg, max_pitch_deg, 201))
    # contact point on ellipse x=a cos t, z=c sin t whose normal is tilted by th: tan(th) = (c/a) * ... param
    t = np.arctan2(-np.cos(th) * c, np.sin(th) * a)  # support point for direction (sin th, -cos th)
    x = a * np.cos(t)
    z = c * np.sin(t)
    arc = float(np.sum(np.hypot(np.diff(x), np.diff(z))))
    return arc, float(a * a / c)


def measure(m, meta):
    d = stand_data(m)
    ext = {nm: mesh_world_vertices(m, d, geom_id(m, nm)) for nm in exterior_names(m)}
    allp = np.concatenate(list(ext.values()))
    tips = []
    for side in "LR":
        gid = geom_id(m, f"{side}_tip")
        tips.append(gid)
    # height: top of exterior above floor (floor at z=0)
    height_total = float(allp[:, 2].max())
    no_pod = np.concatenate([v for k, v in ext.items() if k != "pod_shell"])
    height_body = float(no_pod[:, 2].max())
    hip_names = [k for k in ext if k.startswith("pelvis") or "roll_cover" in k or "hip_cover" in k]
    hip = np.concatenate([ext[k] for k in hip_names])
    pelvis_w = span(hip, 1)
    shin_w = max(span(ext[f"{s}_shin_shell"], 1) for s in "LR")
    shin_d = max(span(ext[f"{s}_shin_shell"], 0) for s in "LR")
    thigh_w = max(span(ext[f"{s}_thigh_shell"], 1) for s in "LR")
    knee_w = max(span(ext[f"{s}_knee_cover"], 1) for s in "LR")
    cfg = meta["cfg"]
    tip = tips[0]
    if m.geom_type[tip] == mujoco.mjtGeom.mjGEOM_SPHERE:
        fp = {"type": "A sphere", "length_mm": 2e3 * m.geom_size[tip][0], "width_mm": 2e3 * m.geom_size[tip][0]}
    else:
        fp = {"type": "B ellipsoid rocker", "length_mm": 2e3 * m.geom_size[tip][0], "width_mm": 2e3 * m.geom_size[tip][1]}
    arc, rad = rocker_arc(m, tip)
    fp["rollable_arc_mm_pm20deg"] = round(arc * 1000, 1)
    fp["sagittal_radius_mm"] = round(rad * 1000, 1)
    fp["static_contact"] = "single point (rigid); real rubber patch a few mm - no support polygon"
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    com = d.subtree_com[pelvis_id].copy()
    stand_tips = [d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_contact")] for s in "LR"]
    shin_len = cfg["shin"]["length"]
    out = {
        "model": meta["name"],
        "total_mass_kg": float(sum(m.body_mass)),
        "height_body_m": height_body, "height_with_pod_m": height_total,
        "pelvis_origin_height_m": float(d.xpos[pelvis_id][2]),
        "hip_width_m": pelvis_w, "hip_width_ratio": pelvis_w / height_body,
        "shin_max_width_front_m": shin_w, "shin_max_depth_side_m": shin_d,
        "shin_width_ratio": shin_w / height_body,
        "thigh_max_width_m": thigh_w, "thigh_width_ratio": thigh_w / height_body,
        "knee_cover_width_m": knee_w,
        "thigh_length_m": cfg["thigh"]["length"], "shin_length_m": shin_len,
        "shin_to_thigh": shin_len / cfg["thigh"]["length"],
        "hip_spacing_m": cfg["pelvis"]["hip_spacing"],
        "stance_width_m": float(abs(stand_tips[0][1] - stand_tips[1][1])),
        "com_stand_m": com.tolist(),
        "com_height_m": float(com[2]),
        "com_x_minus_tip_x_m": float(com[0] - 0.5 * (stand_tips[0][0] + stand_tips[1][0])),
        "footprint": fp,
        "targets": {"hip_width_ratio": [0.24, 0.28], "shin_width_ratio": [0.09, 0.12]},
    }
    out["meets_hip_target"] = out["hip_width_ratio"] <= 0.28
    out["meets_shin_target"] = out["shin_width_ratio"] <= 0.12
    out["packaging"] = packaging_check(m, d)
    out["packaging_ok"] = all(p["ok"] for p in out["packaging"])
    out["stand_contacts"] = contacts_at_stand(m, d)
    return out, d


def mass_table(meta):
    by_cat = defaultdict(float)
    by_body = defaultdict(float)
    for body, item, kg in meta["mass_log"]:
        cat = item.split(":")[0] if ":" in item else "electronics/fasteners/misc"
        by_cat[cat] += kg
        side_free = body.replace("L_", "").replace("R_", "")
        by_body[side_free] += kg
    return dict(by_cat), dict(by_body)


def link_inertials(m):
    out = []
    for b in range(1, m.nbody):
        out.append({"body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), "mass_kg": float(m.body_mass[b]),
                    "com_local_m": m.body_ipos[b].round(5).tolist(), "principal_inertia_kgm2": m.body_inertia[b].tolist()})
    return out


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
