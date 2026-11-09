"""Printable big foot pads (2026-11-08): the original P07 pad's stem and wedge with a larger sole fused under it.

    .venv/bin/python hardware/feet_p1/cad/pads.py board120          # -> hardware/feet_p1/exports/F_board120.{stl,3mf}
    .venv/bin/python hardware/feet_p1/cad/pads.py --all

Same shapes as the simulated designs (scripts/make_pads.py): the new sole lies in the original pad's sole plane
(pitched 16 deg toe-up so it is level in the stance pose), centred on the original sole, so the robot stands at the
same height. Plug the stem into the shin tip (the P06 with two M3 holes) and fit 2 x M3x10, front and back, into
the stem's 2.9 mm pilot.

The pad is the same part for both legs (the original P07 is symmetric about its own centre line).

Printing: boards and talons print ON THEIR SIDE (the exports are already oriented): every layer is then the board's
side profile, so the upturned ends need no support and the layers run along the board, the strong way. Discs print
sole-down. PLA or PETG, 3 walls, 25 % infill. A TPU sole (or a strip of rubber/felt glued on) is optional; the sim
assumes the fur gives the grip (friction 0.4-1.2).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import manifold3d as m3d
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P07 = ROOT / "hardware/leg_p0/exports/stl/P07_pad.stl"
OUT = HERE.parent / "exports"
T = 4.0          # sole plate thickness, mm (as in the sim)


def to_manifold(tm):
    return m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                                 tri_verts=np.asarray(tm.faces, np.uint32)))


def to_trimesh(mf):
    mesh = mf.to_mesh()
    return trimesh.Trimesh(vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts, process=False)


def box(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def rot_y(mf, deg, about):
    """Rotate about a y-parallel axis through point `about` (x, z); + deg lifts +x ends (toe up)."""
    ax, az = about
    return mf.translate([-ax, 0, -az]).rotate([0, -deg, 0]).translate([ax, 0, az])


def board(L, W, kick=12.0, kick_deg=25.0):
    Lf = L - 2 * kick
    flat = box(-Lf / 2, Lf / 2, -W / 2, W / 2, 0, T)
    parts = [flat]
    for s in (1, -1):
        k = box(0, kick, -W / 2, W / 2, 0, T) if s > 0 else box(-kick, 0, -W / 2, W / 2, 0, T)
        k = rot_y(k, kick_deg if s > 0 else -kick_deg, (0, 0)).translate([s * Lf / 2, 0, 0])
        end = box(s * Lf / 2 - 0.01 if s > 0 else s * Lf / 2, s * Lf / 2 if s > 0 else s * Lf / 2 + 0.01,
                  -W / 2, W / 2, 0, T)
        parts.append(m3d.Manifold.batch_hull([end, k]))
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


def disc(D):
    return m3d.Manifold.cylinder(T, D / 2, D / 2, 96)


def talon(Lc, r=4.0, hub=9.0, angles=(30, -30, 180)):
    parts = [m3d.Manifold.cylinder(T, hub, hub, 64)]
    for a in angles:
        cap = m3d.Manifold.batch_hull([m3d.Manifold.sphere(r, 32).translate([6.0, 0, r]),
                                       m3d.Manifold.sphere(r, 32).translate([Lc, 0, r])])
        cap = cap ^ box(-1, Lc + r + 1, -r - 1, r + 1, 0, 2 * r)          # flat underside on the sole plane
        parts.append(cap.rotate([0, 0, a]))
    return m3d.Manifold.batch_boolean(parts, m3d.OpType.Add)


DESIGNS = {
    "board60": (lambda: board(60, 36), "side"), "board90": (lambda: board(90, 36), "side"),
    "board120": (lambda: board(120, 36), "side"), "board90w": (lambda: board(90, 50), "side"), "board120w": (lambda: board(120, 50), "side"),
    "board150": (lambda: board(150, 40, kick=15.0), "side"),
    "board150w": (lambda: board(150, 50, kick=15.0), "side"),
    "disc50": (lambda: disc(50), "flat"), "disc70": (lambda: disc(70), "flat"),
    "talon45": (lambda: talon(45), "flat"), "talon60": (lambda: talon(60), "flat"),
    "claw45": (lambda: talon(45, angles=(0, 120, -120)), "flat"),
    "claw60": (lambda: talon(60, angles=(0, 120, -120)), "flat"),
}


def sole_frame(tm):
    """Origin (sole centre) and rotation (columns x toe-forward along the sole, y, z up out of the sole)."""
    i = int(np.argmax(tm.facets_area))
    n = tm.facets_normal[i]                       # outward = down and a little forward
    z = -n / np.linalg.norm(n)
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    verts = tm.vertices[np.unique(tm.faces[tm.facets[i]].ravel())]
    c = verts.mean(0)
    return c, np.column_stack([x, y, z])


def build(name):
    pad = trimesh.load(P07)
    c, R = sole_frame(pad)
    make, orient = DESIGNS[name]
    sole = make()
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, c
    sole_tm = to_trimesh(sole).apply_transform(M)
    part = to_manifold(pad) + to_manifold(sole_tm)
    tm = to_trimesh(part)
    # print orientation
    Minv = np.eye(4)
    Minv[:3, :3], Minv[:3, 3] = R.T, -R.T @ c
    tm.apply_transform(Minv)                      # sole frame: sole on z = 0
    if orient == "side":
        tm.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))   # y -> z: lie on its side
    tm.apply_translation([0, 0, -tm.bounds[0][2]])
    tm.apply_translation([-tm.bounds.mean(0)[0], -tm.bounds.mean(0)[1], 0])
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for n in (DESIGNS if a.all else a.names):
        tm = build(n)
        tm.export(OUT / f"F_{n}.stl")
        tm.export(OUT / f"F_{n}.3mf")
        print(f"F_{n}: {tm.extents.round(1)} mm on the bed, {tm.volume / 1000:.1f} cm^3, "
              f"~{tm.volume / 1000 * 0.75:.0f} g printed, watertight {tm.is_watertight}")


if __name__ == "__main__":
    main()
