"""IMU adapter slab: bolts to a 14 mm hole pair on the P08 pelvis plate and carries the MPU-6050 board (GY-521).

    .venv/bin/python hardware/imu/cad/imu_slab.py      # writes hardware/imu/exports/imu_slab.{stl,3mf}

The P08 plate has M3 clearance holes in pairs 14 mm apart (y = +-7 at x = -30, 0, 22; leg_p0/cad/biped.py).
The GY-521 board's two mounting holes are 15 mm apart along its long edge (measured 2026-11-06), so it can't
bolt straight on. Layout of the slab, seen from above (slab x along both hole rows):

      +---------------------------+
      |   o (2.9)        o (2.9)  |   MPU row, 15 mm apart, y = ROW_DY: two bosses 3 mm tall; M3 self-taps into them
      |                           |   board rests on the bosses, solder-pin edge hangs past the far end
      |  O (3.4)          O (3.4) |   plate row, 14 mm apart, y = 0: M3 x 10 + nut under the plate
      +---------------------------+

The rows are ROW_DY apart so the plate screws' heads (5.5 mm) clear the board edge (holes 2.7 mm in from it).
The slab is 2.5 mm thick so the board's solder joints (~2 mm) clear the plate. It stops short of the pin row.

v2 (2026-11-11): the MPU screw holes stand on two bosses BOSS_H = 3 mm above the slab, so the board seats on the
bosses and its screws can be driven fully home without pressing the components on its underside into the slab
(in v1 they could only go in halfway, and the board flexed). The board then sits 3 mm higher, 5.5 mm above the plate.
Screw: M3 x 6, through the board into the boss (5.5 mm of plastic, pilot 2.9 mm, through).
"""
from __future__ import annotations

from pathlib import Path

import manifold3d as m3d
import numpy as np
import trimesh

T = 2.5               # slab thickness (board height above the plate)
PLATE_PITCH = 14.0    # P08 hole pair
PLATE_D = 3.4         # M3 clearance (leg_p0/params.yaml fit.m3_clear_d)
MPU_PITCH = 15.0      # GY-521 hole pair
MPU_D = 2.9           # M3 self-taps into PLA
ROW_DY = 7.5          # plate row -> MPU row
BOSS_H = 3.0          # MPU bosses above the slab top
BOSS_D = 5.5          # boss diameter: ~1.3 mm of wall around the pilot, within the GY-521's pad around its hole
HOLE_EDGE = 2.7       # GY-521: hole centre to the board's hole edge
BOARD_DEPTH = 16.0    # GY-521, hole edge to pin edge
PIN_KEEPOUT = 4.0     # leave the last 4 mm of the board (the solder-pin row) unsupported
X_HALF = 12.5         # slab 25 mm long (board is ~20.5)
Y0 = -4.0             # 1.25 mm of plastic beyond the plate-screw heads
Y1 = ROW_DY - HOLE_EDGE + BOARD_DEPTH - PIN_KEEPOUT    # = 16.8
CORNER_R = 2.0
SEG = 96

OUT = Path(__file__).resolve().parents[1] / "exports"


def rounded_rect(x0, x1, y0, y1, r):
    c = m3d.CrossSection.circle(r, SEG)
    pts = [(x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r)]
    return m3d.CrossSection.batch_hull([c.translate(p) for p in pts])


def slab():
    sec = rounded_rect(-X_HALF, X_HALF, Y0, Y1, CORNER_R)
    for x in (-PLATE_PITCH / 2, PLATE_PITCH / 2):
        sec -= m3d.CrossSection.circle(PLATE_D / 2, SEG).translate((x, 0.0))
    body = sec.extrude(T)
    for x in (-MPU_PITCH / 2, MPU_PITCH / 2):
        boss = m3d.CrossSection.circle(BOSS_D / 2, SEG).translate((x, ROW_DY)).extrude(BOSS_H).translate((0, 0, T))
        body = body + boss
    for x in (-MPU_PITCH / 2, MPU_PITCH / 2):
        pilot = m3d.CrossSection.circle(MPU_D / 2, SEG).translate((x, ROW_DY)).extrude(T + BOSS_H + 2).translate((0, 0, -1))
        body = body - pilot
    return body


def main():
    mesh = slab().to_mesh()
    tm = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    assert tm.is_watertight and tm.volume > 0
    OUT.mkdir(parents=True, exist_ok=True)
    tm.export(OUT / "imu_slab.stl")
    try:
        tm.export(OUT / "imu_slab.3mf")
    except Exception as e:                          # 3MF needs lxml/networkx; the STL is enough for Bambu Studio
        print(f"3mf skipped: {e}")
    ext = tm.bounds[1] - tm.bounds[0]
    print(f"imu_slab: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm, {tm.volume / 1000:.2f} cm3 -> {OUT}")


if __name__ == "__main__":
    main()
