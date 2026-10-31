"""One small plate to find the hole size an M3 screw threads into and grips.

    .venv-cad/bin/python hardware/torso_p0/cad/thread_test.py

Output: fit_tests/m3_thread_test_print.stl. Each hole is engraved with its diameter. Print flat, then set
M3_PILOT in torso.py to the size that drives firmly and holds.
"""
from __future__ import annotations

from pathlib import Path

import trimesh
from build123d import Align, Box, Cylinder, Plane, Pos, Text, export_stl, extrude

OUT = Path(__file__).resolve().parents[1] / "fit_tests"
DIAS = (2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9)
PITCH, MARGIN, WIDTH, THICK, ENGRAVE = 7.0, 4.5, 12.0, 6.0, 0.4


def plate():
    length = PITCH * (len(DIAS) - 1) + 2 * MARGIN
    body = Box(length, WIDTH, THICK, align=(Align.MIN, Align.MIN, Align.MIN))
    for k, d in enumerate(DIAS):
        x = MARGIN + PITCH * k
        body -= Pos(x, 4.0, THICK / 2) * Cylinder(d / 2, THICK + 1)
        label = Plane.XY.offset(THICK - ENGRAVE) * Pos(x, 9.2, 0) * Text(f"{d}", font_size=3.2)
        body -= extrude(label, ENGRAVE + 0.1)
    return body


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    path = OUT / "m3_thread_test_print.stl"
    export_stl(plate(), str(path), tolerance=0.02, angular_tolerance=0.15)
    m = trimesh.load(path)
    print(f"watertight {m.is_watertight}, extents {m.extents.round(1).tolist()} mm, {m.volume * 1.24e-3:.1f} g if solid")
