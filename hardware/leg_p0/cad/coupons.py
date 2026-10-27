"""Small calibration coupons (print these FIRST, ~1 h total). Exported to hardware/leg_p0/fit_tests/.

    A  case_pocket_gauge   three windows of the real cradle cross-section at clearance 0.25 / 0.35 / 0.45 mm
                           per side (1 / 2 / 3 notches). Slide each over the servo case from its far end.
    B  horn_arm_coupon     a main-horn arm and an idler-horn arm exactly as on the brackets (hole pattern,
                           M3 horn-screw access). Bolt to both horns; check alignment and wobble.
    C  passive_support     = the idler arm of coupon B (P0 uses the servo's own idler bearing as the passive
                           coaxial support, so there is no separate bearing/shaft to test).
    D  hole_and_nut_gauge  M2 clearance 2.2/2.4/2.6, M2 self-tap pilots 1.6/1.8/2.0, M3 clearance 3.2/3.4/3.6,
                           M3 nut traps 5.5/5.7/5.9 AF, and the 4.6 mm screw-head seat used on the cradles.
    E  pad_socket_coupon   the shin-tip socket + pad stem pair (stem should slide in with light friction).

After printing: measure (see verification.md) and edit params.yaml -> fit.* / servo.*, then rebuild.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build123d import Cylinder, Pos, Rot, export_stl, export_step  # noqa: E402

import leg  # noqa: E402
from leg import S, FIT, P, box, hex_prism, servo_case  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "fit_tests"


def notches(n, x0, y, z1):
    out = None
    for k in range(n):
        c = box(x0 + 4 * k, x0 + 4 * k + 2, y - 1, y + 1, z1 - 1, z1 + 1)
        out = c if out is None else out + c
    return out


def coupon_a():
    parts = None
    for k, c in enumerate((0.25, 0.35, 0.45)):
        # slice of the cradle in the servo frame: 8 mm long, at the case's far end (includes both raised areas)
        z_out = max(S["horn_bump"]["z"], S["idler_bump"]["z"]) + c + 2.4
        blk = box(-20.0, -12.0, -S["case_wid"] / 2 - c - 2.4, S["case_wid"] / 2 + c + 2.4, -z_out, z_out)
        blk -= servo_case(clear=c, extend_open=40.0)
        blk -= notches(k + 1, -19.0, 0, z_out)
        # lay it flat for printing (window axis vertical) and space them out
        g = Rot(0, 90, 0) * blk
        g = Pos(0, k * 36.0, 0) * g
        parts = g if parts is None else parts + g
    return parts


def arm(which):
    o = S["horn_hole_offset"]
    t, r = P["chain"]["arm_t"], P["chain"]["arm_disc_r"]
    a = Pos(0, 0, t / 2) * Cylinder(r, t) + box(0, 30, -r, r, 0, t)
    a -= Pos(0, 0, t / 2) * Cylinder(S["horn_center_screw_head_d"] / 2 + 0.5, t + 1)
    for dx in (-o, o):
        for dy in (-o, o):
            a -= Pos(dx, dy, t / 2) * Cylinder(FIT["horn_screw"]["clear_d"] / 2, t + 1)
    if which == "idler":
        a -= box(22, 26, -1, 1, t - 1, t + 1)            # one notch = idler arm
    return a


def coupon_b():
    return arm("horn") + Pos(0, 26, 0) * arm("idler")


def coupon_d():
    base = box(0, 92, 0, 40, 0, 6)
    xs = [8 + 10 * i for i in range(9)]
    row = [(1.1, 2.2), (1.2, 2.4), (1.3, 2.6), (0.8, 1.6), (0.9, 1.8), (1.0, 2.0), (1.6, 3.2), (1.7, 3.4), (1.8, 3.6)]
    for x, (r, _) in zip(xs, row):
        base -= Pos(x, 10, 3) * Cylinder(r, 8)
    for k, af in enumerate((5.5, 5.7, 5.9)):
        base -= Pos(12 + 14 * k, 28, 3) * Cylinder(1.7, 8)
        base -= Pos(12 + 14 * k, 28, 6 - FIT["m3_nut_h"] / 2 + 0.01) * hex_prism(af, FIT["m3_nut_h"])
    base -= Pos(62, 28, 6 - 0.5) * Cylinder(2.3, 2.0)        # cradle screw-head seat (4.6 mm)
    base -= Pos(62, 28, 3) * Cylinder(FIT["m2_clear_d"] / 2, 8)
    return base


def coupon_e():
    st = P["pad"]["stem"]
    sock = box(-st["d"] / 2 - 3, st["d"] / 2 + 3, -st["w"] / 2 - 3, st["w"] / 2 + 3, 0, st["h"] + 2)
    sock -= box(-st["d"] / 2 - 0.2, st["d"] / 2 + 0.2, -st["w"] / 2 - 0.2, st["w"] / 2 + 0.2, 2, st["h"] + 3)
    stem = Pos(30, 0, 0) * (box(-10, 10, -7, 7, 0, 3) + box(-st["d"] / 2, st["d"] / 2, -st["w"] / 2, st["w"] / 2, 3, 3 + st["h"]))
    return sock + stem


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in (("A_case_pocket_gauge", coupon_a), ("B_C_horn_and_idler_arms", coupon_b),
                     ("D_hole_and_nut_gauge", coupon_d), ("E_pad_socket_coupon", coupon_e)):
        s = fn()
        export_stl(s, str(OUT / f"{name}.stl"), tolerance=0.02)
        export_step(s, str(OUT / f"{name}.step"))
        bb = s.bounding_box()
        out[name] = {"bbox_mm": [round(v, 1) for v in bb.size], "volume_cm3": round(s.volume / 1000, 2)}
    return out


if __name__ == "__main__":
    print(build())
