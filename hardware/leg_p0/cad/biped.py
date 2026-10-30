"""Two-leg bench assembly: P08 biped pelvis (both hip-yaw cups in one piece) + left leg + mirrored right leg.

    .venv-cad/bin/python hardware/leg_p0/cad/biped.py            # export P08, render, inter-leg collision check

The right leg is the mirror of the left (exports/stl_right_mirror/). The servo is symmetric about its own
length/width plane, so every mirrored pocket still holds a real (unmirrored) servo, just turned 180 deg.
Outputs: exports/{stl,stl/print_ready,step,3mf}/P08_pelvis_biped.*, renders/biped_*.png,
biped_report.json.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build  # noqa: E402  (sets the MuJoCo GL env before mujoco is imported)
import leg  # noqa: E402
from build123d import Axis, Cylinder, Location, Mesher, Pos, export_step, export_stl  # noqa: E402

STEM = "P08_pelvis_biped"
JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee")
SPACING = leg.P["pelvis"]["biped_hip_spacing"]
DY = SPACING / 2 - leg.HY          # the left leg is modelled at hip_y: shift it out to half the biped spacing


def part_pelvis_biped():
    """One plate, both hip-yaw cups SPACING apart, no desk clamp.
    v1 bench piece: hold it by hand; an M3 grid on the centreline is for a later handle / torso."""
    pe = leg.P["pelvis"]
    pz0, pz1 = leg.PELVIS_Z0, leg.PELVIS_Z0 + pe["plate_t"]
    y_out = pe["y_outer"] + DY             # same margin outside each cup as the single-leg plate
    plate = leg.box(pe["x0"], pe["x1"], -y_out, y_out, pz0, pz1)
    plate = leg.try_fillet(plate, plate.edges().filter_by(Axis.Z), 6.0)
    add, cut = (s.moved(Location((0, DY, 0))) for s in leg.yaw_cup_mount(pz0, pz1))
    body = plate + add + leg.mirror_right(add)
    body -= cut
    body -= leg.mirror_right(cut)
    for x in (-30.0, 0.0, 22.0):                     # future mounts, between the two cup risers
        for y in (-7.0, 7.0):
            body -= Pos(x, y, (pz0 + pz1) / 2) * Cylinder(leg.FIT["m3_clear_d"] / 2, 20)
    return body


def leg_items():
    """Left-leg items without the single-leg fixture (P01 plate, P02 tongue)."""
    return [it for it in leg.build_items() if it.name not in ("pelvis_plate", "clamp_tongue")]


def posed_pair(items, q_left, q_right):
    """Both legs posed. Right-leg angles follow the same sign conventions mirrored (roll + = abduction,
    yaw + = toe out), so q_right = q_left is a symmetric pose."""
    out = Location((0, DY, 0))
    left = [leg.Item("L_" + it.name, it.shape.moved(out), "L_" + it.link, it.printable, it.color, it.kind)
            for it in leg.pose(items, q_left)]
    right = [leg.Item("R_" + it.name, leg.mirror_right(it.shape.moved(out)), "R_" + it.link, it.printable, it.color,
                      it.kind) for it in leg.pose(items, q_right)]
    return left, right


def inter_leg_hits(left, right, pelvis):
    """Overlaps between the two legs, and between the pelvis and anything that moves."""
    hits = []
    for a in left:
        for b in right:
            if a.kind == "keepout" and b.kind == "keepout":
                continue
            v = build._overlap(a.shape, b.shape)
            if v:
                hits.append((a.name, b.name, round(v, 1)))
    for it in left + right:
        if it.link.endswith("pelvis"):
            continue
        v = build._overlap(pelvis, it.shape)
        if v:
            hits.append(("pelvis_biped", it.name, round(v, 1)))
    return hits


def q(yaw=0.0, roll=0.0, pitch=0.0, knee=0.0):
    return dict(zip(JOINTS, (yaw, roll, pitch, knee)))


def check(items, pelvis):
    """Inter-leg clearance over the bench ranges (yaw -15..25, roll -10..25, pitch -25..80, knee 0..105).
    Legs can only meet when they move toward each other: adduction (roll < 0), toe-in, scissoring."""
    t0 = time.time()
    res = {"symmetric": [], "same_direction_sway": [], "scissor_and_turn": []}
    # 1. symmetric: both legs adduct by the same amount (worst case for the knees and pads)
    for roll in (0, -3, -5, -7, -10):
        for pk in ((0, 0), (16, 32), (45, 90), (80, 105)):
            left, right = posed_pair(items, q(roll=roll, pitch=pk[0], knee=pk[1]), q(roll=roll, pitch=pk[0], knee=pk[1]))
            res["symmetric"].append({"roll": roll, "pitch": pk[0], "knee": pk[1], "hits": inter_leg_hits(left, right, pelvis)})
    # 2. lateral sway: both legs tilt the same way in the world (one abducts, the other adducts)
    for sway in (5, 10):
        left, right = posed_pair(items, q(roll=sway, pitch=16, knee=32), q(roll=-sway, pitch=16, knee=32))
        res["same_direction_sway"].append({"left_roll": sway, "right_roll": -sway, "hits": inter_leg_hits(left, right, pelvis)})
    # 3. one leg forward, one back (walking), with toe-in / toe-out yaw
    for (pl, kl), (pr, kr) in itertools.product(((-25, 0), (16, 32), (80, 105), (35, 75)), repeat=2):
        if (pl, kl) >= (pr, kr):
            continue
        for yl, yr in ((0, 0), (-15, -15), (25, 25)):
            left, right = posed_pair(items, q(yaw=yl, pitch=pl, knee=kl), q(yaw=yr, pitch=pr, knee=kr))
            res["scissor_and_turn"].append({"left": [yl, 0, pl, kl], "right": [yr, 0, pr, kr],
                                            "hits": inter_leg_hits(left, right, pelvis)})
    res["seconds"] = round(time.time() - t0)
    return res


def min_gap_neutral(items):
    """Closest approach between the legs at neutral and at stance (mm)."""
    out = {}
    for tag, qq in (("neutral", q()), ("stance", q(pitch=16, knee=32))):
        left, right = posed_pair(items, qq, qq)
        inner_l = min(it.shape.bounding_box().min.Y for it in left if it.kind != "keepout")
        out[tag] = round(2 * inner_l, 1)          # mirror: the gap is twice the left leg's innermost Y
    return out


def export(pelvis):
    E = build.EXP
    export_stl(pelvis, str(E / "stl" / f"{STEM}.stl"), tolerance=0.02, angular_tolerance=0.15)
    export_step(pelvis, str(E / "step" / f"{STEM}.step"))
    m = Mesher()
    m.add_shape(pelvis, linear_deflection=0.02, angular_deflection=0.15)
    m.write(str(E / "3mf" / f"{STEM}.3mf"))
    import trimesh
    t = trimesh.load(E / "stl" / f"{STEM}.stl")
    t.apply_transform(trimesh.geometry.align_vectors((0, 0, 1), (0, 0, -1)))   # plate face down, cups up
    c = t.bounds.mean(axis=0)
    t.apply_translation((build.BED_CENTER[0] - c[0], build.BED_CENTER[1] - c[1], -t.bounds[0][2]))
    t.export(E / "stl" / "print_ready" / f"{STEM}_print.stl")
    return {"watertight": bool(t.is_watertight), "bbox_mm": [round(v, 1) for v in t.extents],
            "volume_cm3": round(t.volume / 1000, 1)}


def render(items, pelvis):
    import imageio.v2 as iio
    vis = [i for i in items if i.kind in ("part", "servo", "horn")]
    pv = leg.Item("pelvis_biped", pelvis, "pelvis", color=(0.22, 0.23, 0.26, 1))
    views = [("front", (180, 0), True, 0.5), ("three_quarter", (135, -15), False, 0.0)]
    for tag, (ql, qr) in {"stance": (q(pitch=16, knee=32), q(pitch=16, knee=32)),
                          "stride": (q(pitch=35, knee=75), q(pitch=-10, knee=15))}.items():
        left, right = posed_pair(vis, ql, qr)
        imgs = build.render_views([pv] + left + right, views, size=(900, 1000), center=(0.0, 0.0, -0.17))
        for k, im in imgs.items():
            iio.imwrite(build.REN / f"biped_{tag}_{k}.png", im)


if __name__ == "__main__":
    items = leg_items()
    pelvis = part_pelvis_biped()
    rep = {"pelvis": export(pelvis), "gap_between_legs_mm": min_gap_neutral(items)}
    render(items, pelvis)
    rep["clearance"] = check(items, pelvis)
    n_hit = sum(bool(r["hits"]) for k in ("symmetric", "same_direction_sway", "scissor_and_turn") for r in rep["clearance"][k])
    rep["poses_with_hits"] = n_hit
    (build.ROOT / "biped_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("pelvis", "gap_between_legs_mm", "poses_with_hits")}, indent=1))
    for k in ("symmetric", "same_direction_sway", "scissor_and_turn"):
        for r in rep["clearance"][k]:
            if r["hits"]:
                print(k, {kk: vv for kk, vv in r.items() if kk != "hits"}, r["hits"][:4])
