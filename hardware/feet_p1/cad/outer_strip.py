"""Outer strip: a clip-on widening of the A02 foot_ankle board (150 x 50 mm, ends kicked up 25 deg) on its OUTER edge,
for sideways stability (2026-11-12).

    .venv/bin/python hardware/feet_p1/cad/outer_strip.py      # writes hardware/feet_p1/exports/F_outer_strip{15,25}.{stl,3mf}

Why a strip and not a half circle (scripts/make_outpad_model.py, results/robot4/NOTES.md): a robot tipping sideways
pivots on the outermost edge of its standing board. A half-disc on the outer edge only adds a single outermost point,
the apex: tipping onto it the robot balances on a point and falls fore or aft. A full-length strip moves the whole
straight outer edge out. Simulator (with the hip give that matches the robot's E1 tilt test): the sideways tip angle
goes 8.5 deg -> 12.1 with 15 mm, 14.5 with 25 mm (a 30 mm half-disc: 10.0), and the policies that fell sideways
(h13, h9) stop falling with a 15 mm strip.

Section (seen from the toe, left foot; the same part fits the right foot turned 180 deg about vertical):

              lip (screws)            strip
        +-----------------+-----+                        z = 6.5
        |  board top, 7-12 |root |                     |
   =====+=======board======+     +---------------------+  z = 4.3
        |                  |                          |
   =====+==================+--------------------------+  z = 0.3 (0.3 mm above the sole: the board stands as
      sole (z = 0)       board outer edge (y = 0)    y = W         before; the strip touches once the
      board rolls outward ~1 deg)

The strip follows the board: 120 mm flat, the 15 mm end kicks bent up 25 deg. The lip lies on the board's top
surface, 7 mm wide past the servo socket (which stands 9.3 mm in from the edge) and 12 mm where the screws are
(|x| 26..58 mm; the rib is 17 mm in). Two M2 x 6 self-tapping screws through the lip into the board: drill the board
1.6 mm through the lip's holes (the screw tips stop 0.5 mm above the sole). Print it
standing on its bottom face (the z = 0.3 face on the bed), no supports needed (the kicks rise at 25 deg).
"""
from __future__ import annotations

import math
from pathlib import Path

import manifold3d as m3d
import numpy as np
import trimesh

T_BOARD = 4.0          # A02 sole thickness (hardware/ankle_p1/cad/ankle.py ANKLE["sole_t"])
LIFT = 0.3             # strip bottom above the sole
TOP = 6.5              # lip top (and the strip's root, where it joins the lip)
T_STRIP = 4.0          # strip thickness outboard of the root
ROOT_W = 5.0           # full-height root along the board edge (the lip's bending joint)
L_FLAT = 120.0         # board flat length (150 - 2 x 15)
KICK, KICK_DEG = 15.0, 25.0
LIP_W, EAR_W = 7.0, 12.0
EAR_X = (26.0, 58.0)
SCREW_X, SCREW_Y, SCREW_D = 42.0, -7.5, 2.3     # M2 clearance
CORNER_R = 6.0
SEG = 64
OUT = Path(__file__).resolve().parents[1] / "exports"


def box(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def strip(W):
    body = box(-L_FLAT / 2, L_FLAT / 2, 0.0, W, LIFT, LIFT + T_STRIP) + box(-L_FLAT / 2 + 2, L_FLAT / 2 - 2, 0.0, ROOT_W, LIFT, TOP)
    for s in (1, -1):              # toe / heel kick: the same 15 mm at 25 deg as the board, pivoting on the sole line
        k = box(0.0, KICK, 0.0, W, LIFT, LIFT + T_STRIP).rotate((0.0, -s * KICK_DEG, 0.0)) if s > 0 else \
            box(-KICK, 0.0, 0.0, W, LIFT, LIFT + T_STRIP).rotate((0.0, KICK_DEG, 0.0))
        body = body + k.translate((s * L_FLAT / 2, 0.0, 0.0))
    # round the outer corners of the kicked ends (plan view) so the strip doesn't catch the rug when it turns
    lip = box(-L_FLAT / 2 + 2, L_FLAT / 2 - 2, -LIP_W, 0.0, T_BOARD, TOP)
    for s in (1, -1):
        lip = lip + box(min(s * EAR_X[0], s * EAR_X[1]), max(s * EAR_X[0], s * EAR_X[1]), -EAR_W, 0.0, T_BOARD, TOP)
    part = body + lip
    for s in (1, -1):
        part = part - m3d.Manifold.cylinder(TOP + 2, SCREW_D / 2, SCREW_D / 2, SEG).translate((s * SCREW_X, SCREW_Y, -1))
    # outer-corner rounding: cut the square corners at the strip's far edge, ends of the kicks
    cr = m3d.CrossSection.square((CORNER_R, CORNER_R)) - m3d.CrossSection.circle(CORNER_R, SEG)
    xe = L_FLAT / 2 + KICK * math.cos(math.radians(KICK_DEG))
    for sx in (1, -1):
        for sy, y in ((1, W), (-1, -EAR_W)) if False else ((1, W),):
            c = cr.extrude(40).translate((0, 0, -10))
            c = c.mirror((1, 0, 0)) if sx < 0 else c
            part = part - c.translate((sx * (xe - CORNER_R), y - CORNER_R, 0))
    return part


def export(W):
    mesh = strip(W).to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    assert tm.is_watertight and tm.volume > 0
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"F_outer_strip{W:g}"
    tm.export(OUT / f"{name}.stl")
    try:
        tm.export(OUT / f"{name}.3mf")
    except Exception as e:
        print(f"3mf skipped: {e}")
    ext = tm.bounds[1] - tm.bounds[0]
    print(f"{name}: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm, {tm.volume / 1000:.2f} cm3 = {tm.volume * 1.24e-3:.1f} g PLA")


if __name__ == "__main__":
    for W in (15.0, 25.0):
        export(W)
