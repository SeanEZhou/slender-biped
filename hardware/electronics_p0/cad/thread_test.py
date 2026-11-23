"""M2.5 pilot test: a small thin plate with the three likeliest pilot sizes for the Pi and Waveshare screws.

    .venv-cad/bin/python hardware/electronics_p0/cad/thread_test.py

Output: fit_tests/m25_thread_test_print.stl. Print flat. Each hole has its diameter engraved under it.
Set M25_PILOT in mount.py to the size that drives firmly and holds.
"""
from __future__ import annotations

from pathlib import Path

import trimesh
from build123d import Align, Box, Cylinder, Plane, Pos, Text, export_stl, extrude

OUT = Path(__file__).resolve().parents[1] / "fit_tests"
ROWS = ((2.4, 2.5, 2.6),)                        # M2.5 (Pi and Waveshare)
PITCH, MARGIN, ROW, THICK, ENGRAVE = 7.0, 4.0, 9.5, 3.0, 0.4


def plate():
    length = PITCH * 2 + 2 * MARGIN
    width = ROW * len(ROWS) + 1.0
    body = Box(length, width, THICK, align=(Align.MIN, Align.MIN, Align.MIN))
    for r, dias in enumerate(ROWS):
        y = width - 3.5 - ROW * r
        for k, d in enumerate(dias):
            x = MARGIN + PITCH * k
            body -= Pos(x, y, THICK / 2) * Cylinder(d / 2, THICK + 1)
            label = Plane.XY.offset(THICK - ENGRAVE) * Pos(x, y - 4.2, 0) * Text(f"{d}", font_size=2.4)
            body -= extrude(label, ENGRAVE + 0.1)
    return body


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    path = OUT / "m25_thread_test_print.stl"
    export_stl(plate(), str(path), tolerance=0.02, angular_tolerance=0.15)
    m = trimesh.load(path)
    print(f"watertight {m.is_watertight}, extents {m.extents.round(1).tolist()} mm, {m.volume * 1.24e-3:.1f} g if solid")
