"""The T1 torso surface, defined by ../T1_shape.json: a smooth loft of superellipse sections.

Robot frame: +X forward, +Y left, +Z up, mm. Table heights (zp) are measured from the top of the P08 plate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

PLATE_TOP = 30.7          # P08 top face, z in the leg CAD frame
FLOOR = -367.3            # underside of the pads in stance
SHAPE = json.loads((Path(__file__).resolve().parents[1] / "T1_shape.json").read_text())
BODY = [tuple(r) for r in SHAPE["table"]]
CUT = tuple(tuple(p) for p in SHAPE["side_cut_bezier_yz"])    # side openings seen from the front, Bezier (y, z)


def surface(table, nt=240, dz=1.0):
    """Vertices (nz, nt, 3) of the loft. Section parameters are interpolated along z, then smoothed."""
    t = np.array(table, float)
    zp = np.arange(t[0, 0], t[-1, 0] + dz / 2, dz)
    c = np.stack([PchipInterpolator(t[:, 0], t[:, k])(zp) for k in range(1, 6)], axis=1)
    c = gaussian_filter1d(c, SHAPE["smoothing_sigma_mm"] / dz, axis=0, mode="nearest")
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    ct, st = np.cos(th)[None], np.sin(th)[None]
    a, bf, bb, cx, n = (c[:, k][:, None] for k in range(5))
    b = bb + (bf - bb) * (0.5 + 0.5 * np.tanh(2.5 * ct))
    x = cx + b * np.sign(ct) * np.abs(ct) ** (2 / n)
    y = a * np.sign(st) * np.abs(st) ** (2 / n)
    z = np.broadcast_to(zp[:, None], x.shape) + PLATE_TOP
    return np.stack([x, y, z], axis=-1)


def solid(table, inset=0.0):
    table = [(z, a - inset, f - inset, b - inset, c, n) for z, a, f, b, c, n in table]
    v = surface(table)
    nz, nt, _ = v.shape
    idx = np.arange(nz * nt).reshape(nz, nt)
    a, b = idx[:-1], np.roll(idx, -1, axis=1)[:-1]
    c, d = np.roll(idx, -1, axis=1)[1:], idx[1:]
    tris = [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    verts = v.reshape(-1, 3)
    lo, hi = len(verts), len(verts) + 1
    verts = np.vstack([verts, v[0].mean(axis=0), v[-1].mean(axis=0)])
    ring0, ring1 = idx[0], idx[-1]
    tris.append(np.stack([np.full(nt, lo), np.roll(ring0, -1), ring0], 1))
    tris.append(np.stack([np.full(nt, hi), ring1, np.roll(ring1, -1)], 1))
    m = trimesh.Trimesh(verts, np.concatenate(tris), process=True)
    if m.volume < 0:
        m.invert()
    return m


def cut_tool(sign):
    """Prism through the body along X, removing everything outboard of the CUT curve on one side."""
    import manifold3d as m3
    t = np.linspace(0, 1, 60)[:, None]
    p0, p1, p2 = (np.array(p) for p in CUT)
    curve = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t ** 2 * p2
    poly = np.vstack([curve, [[150.0, CUT[2][1]], [150.0, -150.0], [CUT[0][0], -150.0]]])
    mf = m3.CrossSection([poly[::-1]], m3.FillRule.Positive).extrude(300.0).to_mesh()
    uvw = np.asarray(mf.vert_properties)[:, :3]
    v = np.stack([uvw[:, 2] - 150.0, sign * uvw[:, 0], uvw[:, 1]], axis=1)   # (u, v, w) -> (y, z, x)
    tm = trimesh.Trimesh(v, np.asarray(mf.tri_verts), process=True)
    if tm.volume < 0:
        tm.invert()
    return tm


def cut_legs(m):
    for sign in (1, -1):
        m = trimesh.boolean.difference([m, cut_tool(sign)], engine="manifold")
    return m
