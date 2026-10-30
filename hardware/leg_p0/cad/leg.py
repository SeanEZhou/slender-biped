"""Parametric CAD for the P0 single leg (build123d / OpenCascade).

Joint chain (left leg; world frame X forward, Y left, Z up; neutral pose = straight leg):

    pelvis plate --(hip yaw, about +Z)--> yaw bracket --(hip roll, about +X)--> roll bracket
      --(hip pitch, about -Y: + = flexion)--> thigh --(knee, about +Y: + = flexion)--> shin + pad

Every joint uses the same pattern: the PARENT link holds the servo CASE in a cradle that grips only the
far end of the case (beyond `cheek_clear_r` from the output axis); the CHILD link is a U-bracket whose
two arms bolt to the main horn and to the idler horn. The idler horn turns on the servo's own rear
bearing, so each joint has a coaxial passive support and no extra bearing to misalign.

All geometry is built in world coordinates at the neutral pose; `pose()` moves links by joint angles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from build123d import (Axis, Box, Cylinder, Location, Plane, Pos, Solid, Wire, Face, Vector, fillet, Compound,
                       Rot, Part, mirror, Mode, RegularPolygon, extrude)

HERE = Path(__file__).resolve().parent
P = yaml.safe_load((HERE.parent / "params.yaml").read_text())
S, FIT, C = P["servo"], P["fit"], P["chain"]

# ----------------------------------------------------------------------------- derived chain levels
HALF_L = S["case_len"] / 2                      # 22.7
END_FAR = HALF_L + S["output_x"]                # axis -> far end of the case (35.2)
END_NEAR = HALF_L - S["output_x"]               # axis -> output end of the case (10.2)
HALF_W = S["case_wid"] / 2                      # 12.4
HY = C["hip_y"]
CL = FIT["pocket_clearance"]
ARM_T, ARM_R = C["arm_t"], C["arm_disc_r"]
WEB_R, WEB_T, GAP = C["web_r"], C["web_t"], C["stack_gap"]
HORN_Z, IDLER_Z = S["horn_face_z"], S["idler_face_z"]
HS, CS = FIT["horn_screw"], FIT["case_screw"]                  # M3x6 horn screws, M2x6 case screws
Z_CRADLE = max(S["horn_bump"]["z"], S["idler_bump"]["z"]) + CL + C["cheek_t"]   # cradle wall outer face
Z_CASE_HEAD = S["case_face_z"] + CL + CS["grip"]               # case-screw head seat: grip mm of wall under it

Z_YAW = 0.0                                                    # yaw servo centre
Z_ROLL = -(HORN_Z + ARM_T + C["roll_drop"] + END_FAR)          # roll axis
Z_PITCH = Z_ROLL - WEB_R - WEB_T - GAP - HALF_W                # pitch axis
Z_KNEE = Z_PITCH - C["thigh_len"]                              # knee axis
Z_PAD = Z_KNEE - C["shin_len"]                                 # pad heel underside, straight leg
_PAD = P["pad"]
PAD_TOE_RISE = _PAD["length"] * math.tan(math.radians(_PAD["stance_pitch"]))
Z_SHIN_TIP = Z_PAD + PAD_TOE_RISE + _PAD["thickness"]          # shin tip / pad top (flat mating face)

AXES = {  # joint: (point, direction); positive rotation follows the sim's sign conventions
    "hip_yaw": ((0.0, HY, Z_YAW), (0, 0, 1)),
    "hip_roll": ((0.0, HY, Z_ROLL), (1, 0, 0)),
    "hip_pitch": ((0.0, HY, Z_PITCH), (0, -1, 0)),
    "knee": ((0.0, HY, Z_KNEE), (0, 1, 0)),
}
# Servo frames (servo X = length, output at +X; servo Z = output axis, main horn at +Z), world placement:
SERVOS = {
    "hip_yaw": dict(x=(1, 0, 0), z=(0, 0, -1), axis=AXES["hip_yaw"][0], parent="pelvis", child="yaw"),
    "hip_roll": dict(x=(0, 0, -1), z=(1, 0, 0), axis=AXES["hip_roll"][0], parent="yaw", child="roll"),
    "hip_pitch": dict(x=(1, 0, 0), z=(0, 1, 0), axis=AXES["hip_pitch"][0], parent="roll", child="thigh"),
    "knee": dict(x=(0, 0, -1), z=(0, 1, 0), axis=AXES["knee"][0], parent="thigh", child="shin"),
}
CHAIN = ["pelvis", "yaw", "roll", "thigh", "shin"]
JOINT_OF_LINK = {"yaw": "hip_yaw", "roll": "hip_roll", "thigh": "hip_pitch", "shin": "knee"}
CHEEK_CLEAR = {"hip_yaw": C["cheek_clear_r"], "hip_roll": C["cheek_clear_r"], "hip_pitch": 25.0, "knee": C["cheek_clear_r"]}


def servo_plane(name: str) -> Plane:
    """Plane whose origin is the servo case centre, x/z along the servo frame."""
    s = SERVOS[name]
    x, z, a = Vector(*s["x"]), Vector(*s["z"]), Vector(*s["axis"])
    return Plane(origin=a - x * S["output_x"], x_dir=x, z_dir=z)


def in_servo(name: str, shape):
    return servo_plane(name).location * shape


# ----------------------------------------------------------------------------- primitives
def box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2) * Box(x1 - x0, y1 - y0, z1 - z0)


def hex_prism(af, h):
    """Hex nut trap (across-flats af) centred at the origin, axis Z."""
    return Pos(0, 0, -h / 2) * extrude(RegularPolygon(af / math.sqrt(3), 6), h)


def cyl_between(p0, p1, r):
    p0, p1 = Vector(*p0), Vector(*p1)
    d = p1 - p0
    return Plane(origin=p0, z_dir=d).location * Pos(0, 0, d.length / 2) * Cylinder(r, d.length)


def try_fillet(shape, edges, r):
    try:
        out = fillet(edges, r)
        return out if out.volume > 0 else shape
    except Exception:
        return shape


def superellipse_wire(w, d, n, z, cx=0.0, cy=0.0, npts=72):
    pts = []
    for i in range(npts):
        t = 2 * math.pi * i / npts
        c, s = math.cos(t), math.sin(t)
        x = cx + d / 2 * math.copysign(abs(c) ** (2 / n), c)
        y = cy + w / 2 * math.copysign(abs(s) ** (2 / n), s)
        pts.append((x, y, z))
    return Wire.make_polygon(pts, close=True)


def tube(sections, cx=0.0, cy=0.0, wall=None):
    """Smooth loft through superellipse sections [(z, {w,d,n}), ...]; hollow with closed ends if wall."""
    outer = Solid.make_loft([superellipse_wire(s["w"], s["d"], s["n"], z, cx, cy) for z, s in sections])
    if not wall:
        return outer
    zs = [z for z, _ in sections]
    top, bot = max(zs), min(zs)
    inner_secs = []
    for z, s in sections:
        zi = min(max(z, bot + wall + 1.0), top - wall - 1.0)
        inner_secs.append((zi, dict(w=s["w"] - 2 * wall, d=s["d"] - 2 * wall, n=s["n"])))
    inner = Solid.make_loft([superellipse_wire(s["w"], s["d"], s["n"], z, cx, cy) for z, s in inner_secs])
    return outer - inner


# ----------------------------------------------------------------------------- servo dummy (NOT printable)
def servo_case(clear=0.0, extend_open=0.0):
    """Case + raised areas in the servo frame. With clear>0 it is the pocket cutter (open end extended)."""
    c = clear
    body = box(-HALF_L - c, HALF_L + c + extend_open, -HALF_W - c, HALF_W + c, -S["case_face_z"] - c, S["case_face_z"] + c)
    hb, ib = S["horn_bump"], S["idler_bump"]
    m = c + (FIT["bump_margin"] if clear > 0 else 0.0)        # recesses only need to clear the bumps
    body += box(hb["x0"] - m, hb["x1"] + m, -hb["half_w"] - m, hb["half_w"] + m, 0, hb["z"] + c)
    body += box(ib["x0"] - m, ib["x1"] + m, -ib["half_w"] - m, ib["half_w"] + m, -ib["z"] - c, 0)
    if clear == 0:  # recesses around the output on both faces (horn hubs sit in them)
        for sgn in (1, -1):
            z0 = 14.4 if sgn > 0 else -15.9
            body -= Pos(S["output_x"], 0, z0 + 0.75) * Cylinder(11.0, 1.5)
    return body


def horns():
    """Main horn (+Z) and idler horn (-Z) in the servo frame: these turn with the CHILD link."""
    r = S["horn_d"] / 2
    ox = S["output_x"]
    main = Pos(ox, 0, (16.2 + HORN_Z) / 2) * Cylinder(r, HORN_Z - 16.2)
    main += Pos(ox, 0, (14.7 + 16.2) / 2) * Cylinder(4.5, 1.5)
    idl = Pos(ox, 0, -(15.6 + IDLER_Z) / 2) * Cylinder(r, IDLER_Z - 15.6)
    idl += Pos(ox, 0, -(14.4 + 15.6) / 2) * Cylinder(3.0, 1.2)
    return main, idl


def connector_keepout():
    k = S["connector_keepout"]
    return box(k["x0"], k["x1"], -k["half_w"], k["half_w"], -S["case_face_z"] - k["depth"], -S["case_face_z"])


def horn_screw_heads(face_z, sgn):
    """M3x6 horn-screw pan heads on the outside of an arm, servo frame."""
    out = None
    o, ox = S["horn_hole_offset"], S["output_x"]
    z0 = sgn * (face_z + ARM_T)
    for dx in (-o, o):
        for dy in (-o, o):
            h = Pos(ox + dx, dy, z0 + sgn * HS["head_h"] / 2) * Cylinder(HS["head_d"] / 2, HS["head_h"])
            out = h if out is None else out + h
    return out


# ----------------------------------------------------------------------------- cradle (parent side)
def cradle(name, walls_outer=None, cap=True):
    """Block around the far end of the servo case (beyond cheek_clear_r from its axis) minus the
    clearance envelope; M2x6 case screws at every case mounting hole inside the cradle, counterbored so only
    case_screw.grip mm of wall sits under the head (the bump recesses make the wall ~5 mm thick there).
    Returned in world coordinates. The pocket is open toward the output end."""
    t = C["cheek_t"]
    x_cut = S["output_x"] - CHEEK_CLEAR[name]
    z_out = Z_CRADLE
    x0 = -HALF_L - CL - (t if cap else 0.0)
    blk = box(x0, x_cut, -HALF_W - CL - t, HALF_W + CL + t, -z_out, z_out)
    blk = try_fillet(blk, blk.edges().filter_by(Axis.X), 2.0)
    blk -= servo_case(clear=CL, extend_open=40.0)
    for (hx, hy), sgn in [(h, 1) for h in S["mount_holes_top"]] + [(h, -1) for h in S["mount_holes_bot"]]:
        if hx < x_cut - 2.5:
            blk -= Pos(hx, hy, sgn * (z_out - 2)) * Cylinder(CS["clear_d"] / 2, 8)
            cb = z_out + 1.0 - Z_CASE_HEAD
            blk -= Pos(hx, hy, sgn * (Z_CASE_HEAD + cb / 2)) * Cylinder(CS["seat_d"] / 2, cb)   # head counterbore
    return in_servo(name, blk)


# ----------------------------------------------------------------------------- U-bracket arms (child side)
def arm_plate(name, which, length_to_web):
    """One U-bracket arm on the main ('horn') or idler horn: disc around the axis + strip toward the web
    (servo +X direction, past the output end). Servo frame -> world."""
    o, ox = S["horn_hole_offset"], S["output_x"]
    if which == "horn":
        z0, z1 = HORN_Z, HORN_Z + ARM_T
    else:
        z0, z1 = -IDLER_Z - ARM_T, -IDLER_Z
    zc, h = (z0 + z1) / 2, z1 - z0
    a = Pos(ox, 0, zc) * Cylinder(ARM_R, h)
    a += box(ox, ox + length_to_web, -ARM_R, ARM_R, z0, z1)
    a -= Pos(ox, 0, zc) * Cylinder(S["horn_center_screw_head_d"] / 2 + 0.5, h + 1)   # M3 horn-screw access
    for dx in (-o, o):
        for dy in (-o, o):
            a -= Pos(ox + dx, dy, zc) * Cylinder(HS["clear_d"] / 2, h + 1)
    return in_servo(name, a)


def u_bracket(name, web_extra=None):
    """Two arms + a web beyond the output end (web inner face at web_r from the axis)."""
    ox = S["output_x"]
    arms = arm_plate(name, "horn", WEB_R + WEB_T) + arm_plate(name, "idler", WEB_R + WEB_T)
    web = box(ox + WEB_R, ox + WEB_R + WEB_T, -ARM_R, ARM_R, -IDLER_Z - ARM_T, HORN_Z + ARM_T)
    return arms + in_servo(name, web)


# ----------------------------------------------------------------------------- links
PELVIS_Z0 = HORN_Z + ARM_T + 3.5        # pelvis plate underside: above the yaw top arm and its screw heads


def yaw_cup_mount(pz0, pz1):
    """Left hip-yaw cup hung under a plate spanning z pz0..pz1. Returns (solid to add, solid to cut)."""
    cup = cradle("hip_yaw")
    # riser from the cup's top (idler) wall up to the plate
    bb = cup.bounding_box()
    riser = box(bb.min.X, bb.max.X, bb.min.Y, bb.max.Y, bb.max.Z - 1.0, pz0 + 0.5)
    riser -= box(bb.min.X + 3, bb.max.X - 3, bb.min.Y + 3, bb.max.Y - 3, bb.max.Z - 2, pz0 - 1.5)  # hollow
    # cable slot above the yaw connector + screwdriver access to the cup's top screws
    kc = in_servo("hip_yaw", box(-6.5, 3.5, -11.0, 11.0, -40, -10)).bounding_box()
    cut = box(kc.min.X, kc.max.X, kc.min.Y, kc.max.Y, pz0 - 30, pz1 + 1)
    z_wall = max(S["horn_bump"]["z"], S["idler_bump"]["z"]) + CL + C["cheek_t"]     # top of the cup wall
    for hx, hy in S["mount_holes_bot"]:
        if hx < S["output_x"] - CHEEK_CLEAR["hip_yaw"] - 2.5:
            p = servo_plane("hip_yaw").location * Pos(hx, hy, 0)
            cut += box(p.position.X - 3.2, p.position.X + 3.2, p.position.Y - 3.2, p.position.Y + 3.2,
                       z_wall + 0.3, pz1 + 1)                           # screwdriver access, keeps the wall
    return cup + riser, cut


def part_pelvis_plate():
    """Single-leg desk fixture: plate + left yaw cup + clamp-tongue bolt pattern."""
    pz0 = PELVIS_Z0
    pz1 = pz0 + P["pelvis"]["plate_t"]
    pe = P["pelvis"]
    plate = box(pe["x0"], pe["x1"], pe["y_inner"], pe["y_outer"], pz0, pz1)
    plate = try_fillet(plate, plate.edges().filter_by(Axis.Z), 6.0)
    add, cut = yaw_cup_mount(pz0, pz1)
    body = plate + add - cut
    # clamp-tongue bolt pattern (M3) under the inner edge, and the same pattern at the back edge (later use)
    for x in (-25.0, 22.0):
        for y in (pe["y_inner"] + 8, pe["y_inner"] + 22):
            body -= Pos(x, y, (pz0 + pz1) / 2) * Cylinder(FIT["m3_clear_d"] / 2, 20)
    # mirrored mount pattern for the future right hip cup (M3), plus a matching cable slot
    for x in (-38.0, -24.0):
        for dy in (-17.0, 17.0):
            body -= Pos(x, -HY + dy, (pz0 + pz1) / 2) * Cylinder(FIT["m3_clear_d"] / 2, 20)
    return body


def part_tongue():
    pe, tg = P["pelvis"], P["tongue"]
    pz0 = HORN_Z + ARM_T + 3.5
    y1 = pe["y_inner"] + tg["overlap"]
    y0 = y1 - tg["length"]
    w = tg["width"]
    t = box(-w / 2, w / 2, y0, y1, pz0 - tg["t"], pz0)
    t = try_fillet(t, t.edges().filter_by(Axis.Z), 6.0)
    # stiffening ribs on top (outside the plate overlap)
    for x in (-w / 2 + 8, w / 2 - 8):
        t += box(x - 2, x + 2, y0 + 5, pe["y_inner"] - 1, pz0, pz0 + 8)
    for x in (-25.0, 22.0):
        for y in (pe["y_inner"] + 8, pe["y_inner"] + 22):
            t -= Pos(x, y, pz0 - tg["t"] / 2) * Cylinder(FIT["m3_clear_d"] / 2, 20)
            t -= Pos(x, y, pz0 - tg["t"] + FIT["m3_nut_h"] / 2 - 0.01) * hex_prism(FIT["m3_nut_af"], FIT["m3_nut_h"])
    for x in (-w / 2 + 16, w / 2 - 16):          # bolt-down / wood-screw holes
        for y in (y0 + 15, y0 + 75):
            t -= Pos(x, y, pz0 - tg["t"] / 2) * Cylinder(2.3, 20)
    # desk-edge alignment groove (the desk edge goes here: keeps the leg clear of the desk)
    t -= box(-w / 2 - 1, w / 2 + 1, pe["y_inner"] - 0.6, pe["y_inner"] + 0.6, pz0 - 0.6, pz0 + 0.1)
    return t


def part_yaw_bracket():
    """U-bracket on the yaw horns + the roll-servo saddle hanging under its bottom (horn-side) arm."""
    u = u_bracket("hip_yaw")
    saddle = cradle("hip_roll")
    # the saddle's front wall stops short of the web's outer face: extend it flush so the part prints web-down
    # with both on the bed (otherwise the saddle hovers 0.5 mm above the plate)
    front = saddle.faces().filter_by(Axis.X).sort_by(Axis.X)[-1]
    gap = u.bounding_box().max.X - front.center().X
    if gap > 0.01:
        saddle += extrude(front, amount=gap)
    bb = saddle.bounding_box()
    # bridge: round boss (r 20 about the yaw axis, so it clears the pelvis cup at any yaw angle) joining the
    # bottom arm to the saddle cap; everything else of this part beyond r 20 stays below the cup wall
    top = -HORN_Z
    bridge = Pos(0, HY, (bb.max.Z - 2.5 + top) / 2) * Cylinder(20.0, top - (bb.max.Z - 2.5))
    bridge -= Pos(0, HY, top - 2) * Cylinder(S["horn_center_screw_head_d"] / 2 + 0.5, 10)   # horn-screw access
    body = u + saddle + bridge
    # keep the yaw horn screw heads accessible from below: counterbores through the bridge only, stopping at
    # the arm's outer face so the M3 heads still bear on the arm
    o = S["horn_hole_offset"]
    cb0, cb1 = bb.max.Z - 3.5, top - ARM_T
    for dx in (-o, o):
        for dy in (-o, o):
            body -= Pos(dx, HY + dy, (cb0 + cb1) / 2) * Cylinder(HS["head_d"] / 2 + 0.4, cb1 - cb0)
            # the boss overlaps the bottom arm and was added after its holes were cut: re-cut them
            body -= Pos(dx, HY + dy, (cb1 + top) / 2) * Cylinder(HS["clear_d"] / 2, top - cb1 + 1.0)
    return body


def part_roll_bracket():
    """U-bracket on the roll horns; its web carries the pitch-servo cradle."""
    u = u_bracket("hip_roll")
    cr = cradle("hip_pitch")
    bb = cr.bounding_box()
    web_z1 = Z_ROLL - WEB_R
    web = box(min(bb.min.X, -IDLER_Z - ARM_T), HORN_Z + ARM_T, bb.min.Y, bb.max.Y, web_z1 - WEB_T, web_z1)
    web = try_fillet(web, web.edges().filter_by(Axis.Z), 3.0)
    return u + web + cr      # the cradle's top wall overlaps the web, so they fuse


def part_thigh():
    """Hip-pitch fork + slender hollow tube + knee-servo cradle (direct-drive knee)."""
    th = P["thigh"]
    # crotch: closes the fork below the pitch servo and is the root of the tube
    y_in, y_out = HY - IDLER_Z - ARM_T, HY + HORN_Z + ARM_T
    zc1 = Z_PITCH - C["pitch_web_r"]
    zc0 = zc1 - WEB_T - 4.0
    cr = C["thigh_crotch"]
    crotch = tube([(zc0, dict(w=y_out - y_in, d=cr["d"], n=3.2)), (zc1, dict(w=y_out - y_in, d=cr["d"] - 4, n=3.2))],
                  cx=cr["x_offset"], cy=(y_in + y_out) / 2)
    # fork arms: disc + strip pointing DOWN from the pitch axis (servo +X is forward, so build arms
    # explicitly in world instead of using u_bracket's forward web)
    arms = None
    o = S["horn_hole_offset"]
    for y0, y1 in ((HY + HORN_Z, HY + HORN_Z + ARM_T), (HY - IDLER_Z - ARM_T, HY - IDLER_Z)):
        a = Pos(0, (y0 + y1) / 2, Z_PITCH) * (Rot(90, 0, 0) * Cylinder(ARM_R, y1 - y0))
        back = ARM_R if y0 > HY else C["inner_arm_back"]     # inner arm cut back: clears the connectors
        a += box(-back, ARM_R, y0, y1, zc0 + 1, Z_PITCH)
        a -= Pos(0, (y0 + y1) / 2, Z_PITCH) * (Rot(90, 0, 0) * Cylinder(S["horn_center_screw_head_d"] / 2 + 0.5, 10))
        for dx in (-o, o):
            for dz in (-o, o):
                a -= Pos(dx, (y0 + y1) / 2, Z_PITCH + dz) * (Rot(90, 0, 0) * Cylinder(HS["clear_d"] / 2, 10))
        arms = a if arms is None else arms + a
    knee_cr = cradle("knee")
    kb = knee_cr.bounding_box()
    knee_cr = try_fillet(knee_cr, knee_cr.edges().filter_by(Axis.Z), 5.0)   # (no top-edge blend: corner slivers)
    z_top, z_bot = zc0 + 0.5, kb.max.Z - 3.0      # tube overlaps the cradle top by 3 mm (clean boolean seam)
    zm = z_top + (z_bot - z_top) * 0.5
    shell = tube([(z_bot, th["bot"]), (zm, th["mid"]), (z_top, th["top"])], cx=0.0, cy=HY + 0.5, wall=th["wall"])
    # cap between tube and knee cradle (cradle top wall is the cap) + cable exit at the back of the crotch
    body = arms + crotch + shell + knee_cr
    # zip-tie lugs on the inner (-Y) side of the tube for the knee-servo cable
    for z in (z_top - 20, zm, z_bot + 12):         # on the back face (hidden from the front silhouette)
        x = -th["top"]["d"] / 2 - 1.0
        lug = box(x - 4, x + 10, HY - 9, HY - 1, z - 2.5, z + 2.5) - box(x - 2.8, x - 0.8, HY - 6.5, HY - 3.5, z - 5, z + 5)
        body += lug
    return body


def part_shin():
    sh, pad = P["shin"], P["pad"]
    y_in, y_out = HY - IDLER_Z - ARM_T, HY + HORN_Z + ARM_T
    zc1 = Z_KNEE - WEB_R
    zc0 = zc1 - WEB_T - 4.0
    arms = None
    o = S["horn_hole_offset"]
    for y0, y1 in ((HY + HORN_Z, HY + HORN_Z + ARM_T), (HY - IDLER_Z - ARM_T, HY - IDLER_Z)):
        a = Pos(0, (y0 + y1) / 2, Z_KNEE) * (Rot(90, 0, 0) * Cylinder(ARM_R, y1 - y0))
        back = ARM_R if y0 > HY else C["inner_arm_back"]     # inner arm cut back: clears the connectors
        a += box(-back, ARM_R, y0, y1, zc0 + 1, Z_KNEE)
        a -= Pos(0, (y0 + y1) / 2, Z_KNEE) * (Rot(90, 0, 0) * Cylinder(S["horn_center_screw_head_d"] / 2 + 0.5, 10))
        for dx in (-o, o):
            for dz in (-o, o):
                a -= Pos(dx, (y0 + y1) / 2, Z_KNEE + dz) * (Rot(90, 0, 0) * Cylinder(HS["clear_d"] / 2, 10))
        arms = a if arms is None else arms + a
    crotch = tube([(zc0, dict(w=y_out - y_in, d=sh["top"]["d"], n=3.2)), (zc1, dict(w=y_out - y_in, d=sh["top"]["d"] - 4, n=3.2))],
                  cx=0.0, cy=(y_in + y_out) / 2)
    z_top = zc0 + 0.5
    z_bot = Z_SHIN_TIP
    zm = z_top + (z_bot - z_top) * sh["mid_frac"]
    shell = tube([(z_bot, sh["bot"]), (zm, sh["mid"]), (z_top, sh["top"])], cx=0.0, cy=HY + 0.5, wall=sh["wall"])
    body = arms + crotch + shell
    # pad socket in the tip + a rear M2 cross screw into the pad stem
    st = pad["stem"]
    body -= box(-st["d"] / 2 - 0.2, st["d"] / 2 + 0.2, HY + 0.5 - st["w"] / 2 - 0.2, HY + 0.5 + st["w"] / 2 + 0.2,
                z_bot - 1, z_bot + st["h"] + 0.3)
    body -= Pos(-sh["bot"]["d"] / 2, HY + 0.5, z_bot + st["h"] / 2) * (Rot(0, 90, 0) * Cylinder(FIT["m2_clear_d"] / 2, 12))
    return body


def part_pad():
    """Replaceable flat contact cap, 30 x 18 mm. Flat top mates the shin tip; the underside is pitched by
    stance_pitch (toe up) so it is level at the sim stance pose (hip 16, knee 32). Vertical stem."""
    pad = P["pad"]
    st = pad["stem"]
    L, W = pad["length"], pad["width"]
    cy = HY + 0.5
    p = box(-L / 2, L / 2, cy - W / 2, cy + W / 2, Z_PAD - 20, Z_SHIN_TIP)
    cutter = box(-80, 80, cy - 40, cy + 40, Z_PAD - 80, Z_PAD)
    p -= cutter.rotate(Axis((-L / 2, cy, Z_PAD), (0, -1, 0)), pad["stance_pitch"])
    p = try_fillet(p, p.edges().filter_by(Axis.Z), pad["corner_r"])
    p += box(-st["d"] / 2, st["d"] / 2, cy - st["w"] / 2, cy + st["w"] / 2, Z_SHIN_TIP - 0.5, Z_SHIN_TIP + st["h"])
    p -= Pos(0, cy, Z_SHIN_TIP + st["h"] / 2) * (Rot(0, 90, 0) * Cylinder(0.8, st["d"] + 2))   # M2 pilot
    return p


# ----------------------------------------------------------------------------- assembly
@dataclass
class Item:
    name: str
    shape: object
    link: str
    printable: bool = True
    color: tuple = (0.92, 0.90, 0.85, 1.0)
    kind: str = "part"       # part | servo | horn | keepout | hardware


def build_items():
    shell, dark, servo_c, horn_c, ko = (0.93, 0.91, 0.86, 1), (0.22, 0.23, 0.26, 1), (0.15, 0.15, 0.17, 1), (0.85, 0.65, 0.2, 1), (0.9, 0.2, 0.2, 0.35)
    items = [
        Item("pelvis_plate", part_pelvis_plate(), "pelvis", color=dark),
        Item("clamp_tongue", part_tongue(), "pelvis", color=dark),
        Item("yaw_bracket", part_yaw_bracket(), "yaw", color=dark),
        Item("roll_bracket", part_roll_bracket(), "roll", color=dark),
        Item("thigh", part_thigh(), "thigh", color=shell),
        Item("shin", part_shin(), "shin", color=shell),
        Item("pad", part_pad(), "shin", color=(0.08, 0.08, 0.09, 1)),
    ]
    for name, s in SERVOS.items():
        items.append(Item(f"servo_{name}", in_servo(name, servo_case()), s["parent"], printable=False, color=servo_c, kind="servo"))
        m, i = horns()
        items.append(Item(f"horn_{name}", in_servo(name, m), s["child"], printable=False, color=horn_c, kind="horn"))
        items.append(Item(f"idler_{name}", in_servo(name, i), s["child"], printable=False, color=horn_c, kind="horn"))
        items.append(Item(f"keepout_{name}", in_servo(name, connector_keepout()), s["parent"], printable=False, color=ko, kind="keepout"))
        # screw heads: 8 M3 horn screws (child link) and the M2 case screws in the cradle counterbores (parent link)
        heads = horn_screw_heads(HORN_Z, 1) + horn_screw_heads(IDLER_Z, -1)
        items.append(Item(f"hornscrews_{name}", in_servo(name, heads), s["child"], printable=False, color=(0.7, 0.7, 0.72, 1), kind="hardware"))
        x_cut = S["output_x"] - CHEEK_CLEAR[name]
        ch = None
        for (hx, hy), sgn in [(h, 1) for h in S["mount_holes_top"]] + [(h, -1) for h in S["mount_holes_bot"]]:
            if hx < x_cut - 2.5:
                hd = Pos(hx, hy, sgn * (Z_CASE_HEAD + CS["head_h"] / 2)) * Cylinder(CS["head_d"] / 2, CS["head_h"])
                ch = hd if ch is None else ch + hd
        items.append(Item(f"casescrews_{name}", in_servo(name, ch), s["parent"], printable=False, color=(0.7, 0.7, 0.72, 1), kind="hardware"))
    return items


def link_transform(shape, link, q):
    """Move a shape on `link` to joint angles q (deg): rotate about neutral axes, distal first."""
    idx = CHAIN.index(link)
    for l in reversed(CHAIN[1:idx + 1]):
        j = JOINT_OF_LINK[l]
        ang = q.get(j, 0.0)
        if ang:
            p, d = AXES[j]
            shape = shape.rotate(Axis(p, d), ang)
    return shape


def pose(items, q):
    return [Item(it.name, link_transform(it.shape, it.link, q), it.link, it.printable, it.color, it.kind) for it in items]


STANCE = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": 16.0, "knee": 32.0}


def mirror_right(shape):
    """Right-leg variant: mirror about the XZ plane (Y -> -Y)."""
    return mirror(shape, about=Plane.XZ)


def report_levels():
    return {"z_yaw_servo_centre": Z_YAW, "z_roll_axis": Z_ROLL, "z_pitch_axis": Z_PITCH, "z_knee_axis": Z_KNEE,
            "z_pad_bottom_straight": Z_PAD, "yaw_to_roll_mm": Z_YAW - Z_ROLL, "roll_to_pitch_mm": Z_ROLL - Z_PITCH,
            "hip_y": HY}
