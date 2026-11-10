"""Ankle roll for the P0 leg (2026-11-09, v2): the servo socket in the foot, the roll fork on a straight shin.

New parts only; the P06 shin and the P07 pad / F_board150w files are untouched.

    A01 shin_ankle   straight: the P06 knee fork at the top (identical: same knee servo, horns and screws), a straight
                     tube, and a U-fork at the bottom whose arms bolt to the ankle servo's main horn (front) and idler
                     horn (back), like the child side of every other joint
    A02 foot_ankle   the F_board150w sole (150 x 50 mm, last 15 mm each end bent up 25 deg) with the ankle servo's
                     socket on top: the servo stands upright, output end up, gripped by the lower part of its case;
                     a rib along the board stiffens it

Rest pose: the leg straight (all joints 0). The board is flat with the leg straight (no 16 deg wedge), centred under
the knee and hip, and the knee-axis-to-sole length is the leg_p0 150 mm, so the leg is as long as before.

Geometry (left leg, the leg_p0 world frame, mm): the ankle axis is horizontal, along X, on the shin centre line,
ANKLE["axis_h"] above the sole (38.55: the case's far end is 35.2 mm below the axis, + 0.35 clearance + a 3 mm
floor). Main horn forward, connectors on the back face (a window in the socket's back wall).

Right leg: the mirror image (Y -> -Y); the servo keeps its main horn forward, so a positive servo command rolls the
right foot the opposite way (mirrored, as for hip roll).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "leg_p0" / "cad"))

from build123d import Axis, Box, Cylinder, Plane, Pos, Rot, Solid, Vector, Polyline, make_face, extrude  # noqa: E402

import leg  # noqa: E402
from leg import (S, C, P, box, tube, superellipse_wire, try_fillet, servo_case, horns, horn_screw_heads,  # noqa: E402
                 connector_keepout, Item, HY, HALF_L, HALF_W, CL, ARM_T, ARM_R, WEB_R, WEB_T, HORN_Z, IDLER_Z, HS, CS,
                 Z_KNEE, Z_CRADLE, Z_CASE_HEAD, END_FAR, FIT)

# ----------------------------------------------------------------------------- parameters
ANKLE = dict(
    floor_t=3.0,          # socket floor under the servo case (the board is part of it)
    cheek_clear=14.0,     # the socket grips the case beyond this distance below the axis (shin fork discs: r 10.5)
    roll_range=20.0,      # deg, the sim's joint range
    sole_t=4.0,
    board=dict(L=150.0, W=50.0, kick=15.0, kick_deg=25.0),
    rib=dict(L=110.0, W=16.0, h=4.0),      # along the board, on top, through the socket (stiffens the sole)
    leg_len=C["shin_len"],                 # knee axis to sole, straight leg (leg_p0: 150)
    fork_pitch=16.0,      # deg: the ankle fork (and with it servo, socket and board) pitched toe-up on the straight shin,
                          # so the roll axis is level and the board flat in the bent stance (hip 16 / knee 32). 0 = v2
                          # (axis level with the leg straight; rest pose straight)
)
ANKLE["axis_h"] = END_FAR + CL + ANKLE["floor_t"]     # 38.55
CY = HY + 0.5                                         # shin / pad centre line (leg_p0)
Z_SOLE = Z_KNEE - ANKLE["leg_len"]
A = Vector(0.0, CY, Z_SOLE + ANKLE["axis_h"])         # a point on the ankle axis
_FP = math.radians(ANKLE["fork_pitch"])
AXIS = ((A.X, A.Y, A.Z), (math.cos(_FP), 0.0, math.sin(_FP)))
CHEEK = ANKLE["cheek_clear"]
REST = dict(leg.STANCE) if ANKLE["fork_pitch"] else {}   # the rest pose: the bent stance (v3) or straight (v2)


def tilt(shape):
    """Pitch a shape built with the roll axis level (servo upright, board flat) toe-up about the ankle axis by
    fork_pitch: that is its place on the straight shin with the leg straight (level again in the bent stance)."""
    return shape.rotate(Axis((A.X, A.Y, A.Z), (0, -1, 0)), ANKLE["fork_pitch"]) if ANKLE["fork_pitch"] else shape

# servo frame (leg.py convention): x along the case toward the output end (up), z = output axis (main horn forward)
SERVO_PLANE = Plane(origin=A - Vector(0, 0, 1) * S["output_x"], x_dir=(0, 0, 1), z_dir=(1, 0, 0))


def in_servo(shape):
    return SERVO_PLANE.location * shape


# ----------------------------------------------------------------------------- shin
def knee_fork():
    """The P06 shin's top, unchanged: arms on the knee horns + crotch (leg.part_shin)."""
    sh = P["shin"]
    y_in, y_out = HY - IDLER_Z - ARM_T, HY + HORN_Z + ARM_T
    zc1 = Z_KNEE - WEB_R
    zc0 = zc1 - WEB_T - 4.0
    arms = None
    o = S["horn_hole_offset"]
    for y0, y1 in ((HY + HORN_Z, HY + HORN_Z + ARM_T), (HY - IDLER_Z - ARM_T, HY - IDLER_Z)):
        a = Pos(0, (y0 + y1) / 2, Z_KNEE) * (Rot(90, 0, 0) * Cylinder(ARM_R, y1 - y0))
        back = ARM_R if y0 > HY else C["inner_arm_back"]
        a += box(-back, ARM_R, y0, y1, zc0 + 1, Z_KNEE)
        a -= Pos(0, (y0 + y1) / 2, Z_KNEE) * (Rot(90, 0, 0) * Cylinder(S["horn_center_screw_head_d"] / 2 + 0.5, 10))
        for dx in (-o, o):
            for dz in (-o, o):
                a -= Pos(dx, (y0 + y1) / 2, Z_KNEE + dz) * (Rot(90, 0, 0) * Cylinder(HS["clear_d"] / 2, 10))
        arms = a if arms is None else arms + a
    crotch = tube([(zc0, dict(w=y_out - y_in, d=sh["top"]["d"], n=3.2)), (zc1, dict(w=y_out - y_in, d=sh["top"]["d"] - 4, n=3.2))],
                  cx=0.0, cy=(y_in + y_out) / 2)
    return arms + crotch, zc0


def ankle_fork():
    """U-fork on the ankle horns (servo frame): arms on the main horn (front) and idler horn (back), web above the
    servo's output end (WEB_R from the axis; the case corner is at r 16.1). Returned in world coordinates, with the
    z of the web's top face."""
    o, ox = S["horn_hole_offset"], S["output_x"]
    u = None
    for which in ("horn", "idler"):
        z0, z1 = (HORN_Z, HORN_Z + ARM_T) if which == "horn" else (-IDLER_Z - ARM_T, -IDLER_Z)
        zc, h = (z0 + z1) / 2, z1 - z0
        a = Pos(ox, 0, zc) * Cylinder(ARM_R, h)
        a += box(ox, ox + WEB_R + WEB_T, -ARM_R, ARM_R, z0, z1)
        a -= Pos(ox, 0, zc) * Cylinder(S["horn_center_screw_head_d"] / 2 + 0.5, h + 1)
        for dx in (-o, o):
            for dy in (-o, o):
                a -= Pos(ox + dx, dy, zc) * Cylinder(HS["clear_d"] / 2, h + 1)
        u = a if u is None else u + a
    u += box(ox + WEB_R, ox + WEB_R + WEB_T, -ARM_R, ARM_R, -IDLER_Z - ARM_T, HORN_Z + ARM_T)
    return tilt(in_servo(u)), A.Z + WEB_R + WEB_T


def part_shin_ankle():
    fork, zc0 = knee_fork()
    afork, z_web = ankle_fork()
    sh = P["shin"]
    z_top = zc0 + 0.5
    d_web = HORN_Z + IDLER_Z + 2 * ARM_T                # fork's fore-aft extent
    xc_web = (HORN_Z - IDLER_Z) / 2                     # its centre (main horn face sits 1 mm further out)
    wc = tilt(Pos(xc_web, CY, z_web - 1.0) * Box(1, 1, 1)).center()      # the (pitched) web's top centre
    line_x = lambda z: wc.X * (z_top - z) / (z_top - (wc.Z - 1.5))   # noqa: E731  straight axis: knee -> fork web
    secs = [(wc.Z - 1.5, line_x(wc.Z - 1.5), dict(w=2 * ARM_R, d=d_web - 4.0, n=4.0)),
            (wc.Z + 8.0, line_x(wc.Z + 8.0), dict(w=19.0, d=30.0, n=2.8)),
            (z_web + 26.0, line_x(z_web + 26.0), dict(w=sh["mid"]["w"], d=sh["mid"]["d"], n=sh["mid"]["n"])),
            (z_top, 0.0, sh["top"])]
    shell = Solid.make_loft([superellipse_wire(s["w"], s["d"], s["n"], z, cx, CY) for z, cx, s in secs])
    body = fork + shell + afork
    # zip-tie lugs on the back of the tube for the knee -> ankle servo cable
    for z in (z_top - 14.0, z_web + 14.0):
        x = -11.5 + line_x(z)
        lug = box(x - 5, x + 1, CY - 4, CY + 4, z - 2.5, z + 2.5) - box(x - 3.8, x - 1.8, CY - 1.5, CY + 1.5, z - 5, z + 5)
        body += lug
    return body


# ----------------------------------------------------------------------------- foot
def board_solid():
    """F_board150w sole: underside on Z_SOLE, flat with the leg straight, centred under the shin."""
    b, T = ANKLE["board"], ANKLE["sole_t"]
    L, W, k, kd = b["L"], b["W"], b["kick"], b["kick_deg"]
    Lf = L - 2 * k
    sol = box(-Lf / 2, Lf / 2, CY - W / 2, CY + W / 2, Z_SOLE, Z_SOLE + T)
    for s in (1, -1):
        kick = box(0, k, -W / 2, W / 2, 0, T) if s > 0 else box(-k, 0, -W / 2, W / 2, 0, T)
        kick = kick.rotate(Axis((0, 0, 0), (0, -s, 0)), kd)      # toe / heel up
        sol += Pos(s * Lf / 2, CY, Z_SOLE) * kick
    return sol


def socket():
    """Block around the lower part of the ankle servo's case (servo frame), and the cutters: pocket (open upward,
    toward the output end), M2 case screws with counterbores on the front and back faces, connector window in the
    back wall."""
    t = C["cheek_t"]
    x_cut = S["output_x"] - CHEEK
    x0 = -HALF_L - CL - ANKLE["floor_t"]
    blk = box(x0, x_cut, -HALF_W - CL - t, HALF_W + CL + t, -Z_CRADLE, Z_CRADLE)
    blk = try_fillet(blk, blk.edges().filter_by(Axis.X), 2.5)
    cut = servo_case(clear=CL, extend_open=40.0)
    # the servo slides in far end first, so the recesses for the raised face areas run out to the open end (else a
    # strip of wall above the idler bump, between its recess and the connector window, blocks it)
    m = CL + FIT["bump_margin"]
    for b, sgn in ((S["horn_bump"], 1), (S["idler_bump"], -1)):
        z0, z1 = sorted((0.0, sgn * (b["z"] + CL)))
        cut += box(b["x0"] - m, x_cut + 1.0, -b["half_w"] - m, b["half_w"] + m, z0, z1)
    for (hx, hy), sgn in [(h, 1) for h in S["mount_holes_top"]] + [(h, -1) for h in S["mount_holes_bot"]]:
        if hx < x_cut - 2.5:
            cut += Pos(hx, hy, sgn * (Z_CRADLE - 2)) * Cylinder(CS["clear_d"] / 2, 8)
            cb = Z_CRADLE + 1.0 - Z_CASE_HEAD
            cut += Pos(hx, hy, sgn * (Z_CASE_HEAD + cb / 2)) * Cylinder(CS["seat_d"] / 2, cb)
    k = S["connector_keepout"]                     # plugs + cable bend, idler (back) face
    cut += box(k["x0"] - 1.0, x_cut + 1.0, -k["half_w"] - 0.5, k["half_w"] + 0.5, -Z_CRADLE - 2, -S["case_face_z"])
    return blk, cut


def part_foot_ankle():
    blk, cut = socket()
    r = ANKLE["rib"]
    rib = box(-r["L"] / 2, r["L"] / 2, CY - r["W"] / 2, CY + r["W"] / 2, Z_SOLE + ANKLE["sole_t"] - 0.5, Z_SOLE + ANKLE["sole_t"] + r["h"])
    rib = try_fillet(rib, rib.edges().filter_by(Axis.Z), 5.0)
    body = board_solid() + rib + in_servo(blk)
    return tilt(body - in_servo(cut))


# ----------------------------------------------------------------------------- assembly
def build_items():
    shell, foot_c, servo_c, horn_c = (0.93, 0.91, 0.86, 1), (0.08, 0.08, 0.09, 1), (0.15, 0.15, 0.17, 1), (0.85, 0.65, 0.2, 1)
    items = [Item("shin_ankle", part_shin_ankle(), "shin", color=shell),
             Item("foot_ankle", part_foot_ankle(), "foot", color=foot_c)]
    items.append(Item("servo_ankle_roll", tilt(in_servo(servo_case())), "foot", printable=False, color=servo_c, kind="servo"))
    m, i = horns()
    items.append(Item("horn_ankle_roll", tilt(in_servo(m)), "shin", printable=False, color=horn_c, kind="horn"))
    items.append(Item("idler_ankle_roll", tilt(in_servo(i)), "shin", printable=False, color=horn_c, kind="horn"))
    items.append(Item("keepout_ankle_roll", tilt(in_servo(connector_keepout())), "foot", printable=False, color=(0.9, 0.2, 0.2, 0.35), kind="keepout"))
    heads = horn_screw_heads(HORN_Z, 1) + horn_screw_heads(IDLER_Z, -1)
    items.append(Item("hornscrews_ankle_roll", tilt(in_servo(heads)), "shin", printable=False, color=(0.7, 0.7, 0.72, 1), kind="hardware"))
    x_cut = S["output_x"] - CHEEK
    ch = None
    for (hx, hy), sgn in [(h, 1) for h in S["mount_holes_top"]] + [(h, -1) for h in S["mount_holes_bot"]]:
        if hx < x_cut - 2.5:
            hd = Pos(hx, hy, sgn * (Z_CASE_HEAD + CS["head_h"] / 2)) * Cylinder(CS["head_d"] / 2, CS["head_h"])
            ch = hd if ch is None else ch + hd
    items.append(Item("casescrews_ankle_roll", tilt(in_servo(ch)), "foot", printable=False, color=(0.7, 0.7, 0.72, 1), kind="hardware"))
    return items


def leg_items():
    """The whole left leg with the ankle: leg_p0 parts down to the thigh, then the ankle shin and foot."""
    base = [it for it in leg.build_items() if it.name not in ("shin", "pad")]
    return base + build_items()


def transform(shape, link, q):
    """Pose a shape: ankle roll first (foot only), then the leg_p0 chain (the foot rides on the shin)."""
    if link == "foot":
        ang = q.get("ankle_roll", 0.0)
        if ang:
            shape = shape.rotate(Axis(*AXIS), ang)
        link = "shin"
    return leg.link_transform(shape, link, q)


def pose(items, q):
    return [Item(it.name, transform(it.shape, it.link, q), it.link, it.printable, it.color, it.kind) for it in items]
