"""Printable replacement horns for the STS3215 (driven 25T spline horn + passive rear horn), with fit variants.

    .venv-cad/bin/python hardware/leg_p0/cad/horns.py

Baseline dimensions are the Feetech STS3215 spec sheet, section 10 Accessories (page 7), in params.yaml
servo.horn_print: 19.95 mm discs, 4-M3 on a 14 mm circle, 9 mm hubs. Driven horn 4.5 thick (2.5 plate + 2.0 hub),
25T spline for the 5.9 mm shaft, 3.6 mm deep, 3.2 mm centre-screw hole through the 0.9 mm floor. Passive horn
3.1 thick (2.1 plate + 1.0 hub), 6.05 mm bore that turns on the servo's rear shaft. The drawing gives no tooth
height or flank shape: the spline profile is an approximation (trapezoidal teeth, spline_tooth_h).

The 0.3 mm spline teeth are narrower than a 0.4 mm nozzle's line and are lost in slicing, whatever the layer
height. With a 0.4 nozzle use the round-bore variants (the shaft's teeth press into a plain bore); with a 0.2 nozzle
or resin use the spline variants. All parts print outer face down; the driven horn's blind bore opens upward.

    exports/stl/print_ready/horns/STS3215_driven_{tight,nominal,loose}.stl   spline, bore valleys 6.0 / 6.1 / 6.2
    exports/stl/print_ready/horns/STS3215_driven_round_5p7.stl, _5p8.stl    plain bores for a 0.4 mm nozzle
    exports/stl/print_ready/horns/STS3215_passive_6p05 ... _6p35.stl        rear-shaft bore variants
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build123d import Cylinder, Pos, Polyline, extrude, export_step, export_stl, make_face  # noqa: E402

from leg import S  # noqa: E402

HP = S["horn_print"]
DR, PA = HP["driven"], HP["passive"]
ROOT = Path(__file__).resolve().parents[1]
OUT_STL = ROOT / "exports" / "stl" / "print_ready" / "horns"
OUT_STEP = ROOT / "exports" / "step" / "horns"


def spline_bore(valley_d, depth):
    """Female spline cutter, axis Z, z=0...depth: valleys at valley_d/2, tooth tips spline_tooth_h further in."""
    n = HP["spline_teeth"]
    ro, ri, p = valley_d / 2, valley_d / 2 - HP["spline_tooth_h"], 2 * math.pi / n
    pts = [(r * math.cos((k + f) * p), r * math.sin((k + f) * p))
           for k in range(n) for f, r in ((-0.15, ro), (0.15, ro), (0.35, ri), (0.65, ri))]
    cut = extrude(make_face(Polyline(*pts, pts[0])), depth)
    lead = HP["lead_in"]                         # entry chamfer so the shaft starts straight
    return cut + Pos(0, 0, depth - lead / 2) * Cylinder(ro + lead, lead)


def plate(plate_t, total_t):
    """Disc (outer face on the bed at z=0) + hub on top, minus the 4 mounting holes."""
    body = Pos(0, 0, plate_t / 2) * Cylinder(HP["outer_d"] / 2, plate_t)
    body += Pos(0, 0, (plate_t + total_t) / 2) * Cylinder(HP["hub_d"] / 2, total_t - plate_t)
    r = HP["bolt_circle_d"] / 2
    for a in range(4):                           # on the diagonals: holes square to the case at a 45 deg spline phase
        x, y = r * math.cos(math.pi / 4 + a * math.pi / 2), r * math.sin(math.pi / 4 + a * math.pi / 2)
        body -= Pos(x, y, plate_t / 2) * Cylinder(HP["mount_hole_d"] / 2, plate_t + 0.2)
    return body


def driven(valley_d=None, round_d=None):
    t, depth = DR["total_t"], DR["spline_depth"]
    body = plate(DR["plate_t"], t)
    if round_d:
        body -= Pos(0, 0, t - depth / 2 + 0.005) * Cylinder(round_d / 2, depth + 0.01)
        body -= Pos(0, 0, t - HP["lead_in"] / 2) * Cylinder(round_d / 2 + HP["lead_in"], HP["lead_in"])
    else:
        body -= Pos(0, 0, t - depth) * spline_bore(valley_d, depth + 0.01)
    return body - Pos(0, 0, t / 2) * Cylinder(DR["center_d"] / 2, t + 0.2)


def passive(bore_d):
    t = PA["total_t"]
    return plate(PA["plate_t"], t) - Pos(0, 0, t / 2) * Cylinder(bore_d / 2, t + 0.2)


def variants():
    base = HP["spline_nominal_d"] + HP["spline_clear"]
    for name, off in HP["spline_fit_offsets"].items():
        yield f"STS3215_driven_{name}", driven(valley_d=base + off)
    for d in HP["round_bore_variants"]:
        yield f"STS3215_driven_round_{d:.1f}".replace(".", "p"), driven(round_d=d)
    for d in HP["passive_bore_variants"]:
        yield f"STS3215_passive_{d:.2f}".replace(".", "p"), passive(d)


def build():
    OUT_STL.mkdir(parents=True, exist_ok=True)
    OUT_STEP.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, s in variants():
        export_stl(s, str(OUT_STL / f"{name}.stl"), tolerance=0.005, angular_tolerance=0.05)
        export_step(s, str(OUT_STEP / f"{name}.step"))
        bb = s.bounding_box()
        out[name] = {"bbox_mm": [round(v, 2) for v in bb.size], "volume_cm3": round(s.volume / 1000, 3),
                     "valid": s.is_valid}
    return out


if __name__ == "__main__":
    for k, v in build().items():
        print(k, v)
