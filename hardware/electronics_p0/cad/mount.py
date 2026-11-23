"""Electronics mount P0 (v2, fits the T1 torso): one printed spine on the P08 hip plate that carries the battery, the
Raspberry Pi 3 B+, the Waveshare Bus Servo Adapter (A) and the IMU (the old imu_slab is built into its base), stacked
along the torso's length so it all passes the torso's waist. A separate strap holds the battery.

    .venv-cad/bin/python hardware/electronics_p0/cad/torso_fit.py     # once: free space inside the torso tubes
    .venv-cad/bin/python hardware/electronics_p0/cad/mount.py

Outputs: exports/print_ready/*.stl, exports/assembled/*.stl (leg CAD frame, components as boxes), renders/mount_*.png,
mount_report.json.

Frame: the leg CAD frame (+X forward, +Y left, +Z up, mm). Heights below are zp = z - PLATE_TOP. The mount sits on
T03's 3 mm floor (zp 3); without the torso it sits on the plate, 3 mm lower.

    side view (from the left)             The torso's waist (T03/T04 seam) leaves about 55 mm front to back and
                                          72 mm side to side, rounded, so the wide parts stack in the middle:
            Pi  |=|  ribs: lever            - behind the spine, low: battery pouch (inner 86 x 35 x 18), held by a
           (USB |=|  connectors, buck,        strap screwed into thickened pouch walls; the leads leave the top
            up) |=|  cables (free)            next to the spine and pass through a slot to the front
                |_|                         - behind the spine, high: Pi on 3 mm standoffs, USB end up, powered
          battery| |  Waveshare               from the GPIO 5 V pins
           pouch | |                        - in front, low: IMU on the base, Waveshare above it on 4 mm standoffs
                 | |  IMU                     (servo sockets down, USB-C and DC terminals up)
          =======base=======                - in front, high: two ribs; the lever connectors and the buck converter
                                              ride free on their cables; zip-tie holes through the ribs
Fixing: 4 x M3 down through counterbores in the base, through T03's floor and the plate: at (0, +-7) into nuts under the
plate (T03's two middle floor screws, longer), at (22, +-7) into the front hip panel's ear nuts (nuts without the
torso). Pilots self-tap: M3 as the frame (params.yaml fit.m3_pilot_d, 2.9), M2.5 2.5 (Pi and Waveshare).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import manifold3d as m3d
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
P08 = ROOT.parent / "leg_p0" / "exports" / "stl" / "P08_pelvis_biped.stl"
CACHE = ROOT / "torso_cache"
TUBES = ("T03_pelvis_ring", "T04_midriff", "T05_chest", "T06_shoulders")
SEG = 48
PLATE_TOP = 30.7
DENSITY, FILL = 1.24e-3, 0.6              # g/mm3 PLA/PETG; rough solid fraction of the prints
MARGIN = 1.0                              # clearance asked of every part against the torso's free space

import yaml  # noqa: E402

FIT = yaml.safe_load((ROOT.parent / "leg_p0" / "params.yaml").read_text())["fit"]
M3_CLEAR, M3_PILOT = FIT["m3_clear_d"], FIT["m3_pilot_d"]     # the frame's M3 sizes (pilot 2.9, torso thread test)
M3_HEAD_D, M3_HEAD_H = 6.2, 3.0
M25_PILOT = 2.5                          # M2.5 (Pi and Waveshare): the size the Waveshare's screws were found to hold in

# --- base, on T03's floor
BASE_Z, BASE_T = 3.0, 6.0
FLOOR = BASE_Z + BASE_T                   # 9: the pouch floor
BASE_X0, BASE_X1, BASE_INSET = -17.5, 26.0, 1.5
BASE_HOLES = ((0.0, -7.0), (0.0, 7.0), (22.0, -7.0), (22.0, 7.0))   # P08 pairs: M3 down through counterbores into nuts

# --- spine: lower (behind it the battery), upper (behind it the Pi)
LS_X0, LS_X1, LS_HY, LS_TOP = 3.3, 6.3, 24.0, 100.0     # back face clears the x = 0 screw heads
US_X0, US_X1, US_HY, US_Z0, TOP = 4.4, 7.4, 28.5, 95.0, 180.0
RIB_Y0, RIB_Y1, RIB_X1, RIB_Z0, RIB_Z1 = 12.0, 14.5, 15.0, 85.0, 176.0
ZIP_Z, ZIP_X, ZIP_W, ZIP_H = (108.0, 130.0, 152.0, 170.0), 11.0, 2.2, 4.5     # cable-tie holes through the ribs
LEAD_SLOT = (11.0, 86.0, 100.0)           # |y|, zp: battery leads and Pi power through the spine

# --- battery: Ovonic 2S 2200 mAh, 85 x 34 x 17, leads out of the top end
BAT_L, BAT_W, BAT_T, ROOM = 85.0, 34.0, 17.0, 1.0
IN_L, IN_W, IN_D = BAT_L + ROOM, BAT_W + ROOM, BAT_T + ROOM
POUCH_WALL, TOP_DEPTH = 2.4, 10.0
RIM_X = LS_X0 - IN_D                      # -16.5: open back of the pouch
SIDE_Y0, SIDE_Y1 = IN_W / 2, IN_W / 2 + POUCH_WALL
POUCH_TOP = FLOOR + IN_L                  # 95: top wall underside

# --- battery strap: flat bar across the back of the pouch, two M3 into thickened pouch walls
STRAP_Z0, STRAP_H, STRAP_T, STRAP_HY = 41.0, 22.0, 3.0, 23.5
STRAP_BOSS_Y1, STRAP_PILOT_Y, STRAP_PILOT_DEPTH = 23.5, 20.5, 8.0
PAD_Y, PAD_T = 15.0, 0.8                  # 0.2 mm off a 17.0 mm pack

# --- Raspberry Pi 3 B+ (official drawing): 85 x 56 x 1.4, M2.5 holes 58 x 49 (3.5 from the SD end and the long edges);
# USB / Ethernet at the far end (+2.1 overhang), ~16 tall; USB pairs at 29 and 47 from the power edge, Ethernet 10.25
PI_L, PI_W, PI_T, PI_TALL, PI_OVER = 85.0, 56.0, 1.4, 16.0, 2.1
PI_HOLES = ((3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5))   # (from the SD end, from the power edge)
PI_STANDOFF, PI_BOSS_R, PI_FRONT_BOSS, PI_Z0 = 3.0, 3.0, 3.0, 103.0

# --- Waveshare Bus Servo Adapter (A): 42.5 x 32.5, M2.5 holes 38 x 28; along y, servo sockets down, power edge up
WS_L, WS_W, WS_T, WS_PX, WS_PY = 42.5, 32.5, 1.6, 38.0, 28.0
WS_STANDOFF, WS_BOSS_R, WS_BOT = 4.0, 3.0, 30.0
WS_TOP_Z = WS_BOT + WS_W

# --- IMU (GY-521) on the base in front of the spine (hardware/imu/cad/imu_slab.py bosses)
MPU_PITCH, MPU_BOSS_H, MPU_BOSS_D, MPU_PILOT_DEPTH = 15.0, 3.0, 5.5, 8.0
MPU_X, MPU_HOLE_EDGE, MPU_DEPTH, MPU_LEN = 10.0, 2.7, 16.0, 21.0

SCREWS_G = 6 * 0.45 + 8 * 0.2
FLOATING = ("lever connectors + buck converter (on their cables)", 20.0, (15.0, 0.0, 130.0))


# ---------------------------------------------------------------- helpers
def box(x0, x1, y0, y1, z0, z1):
    return m3d.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def sym_box(x0, x1, hy, z0, z1):
    return box(x0, x1, -hy, hy, z0, z1)


def cyl_z(x, y, z0, z1, r):
    return m3d.Manifold.cylinder(z1 - z0, r, r, SEG).translate((x, y, z0))


def cyl_x(y, z, x0, x1, r):
    return m3d.Manifold.cylinder(x1 - x0, r, r, SEG).rotate((0, 90, 0)).translate((x0, y, z))


def vol(man):
    return 0.0 if man.is_empty() else man.volume()


def to_tm(man, dz=PLATE_TOP):
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(float)
    v[:, 2] += dz
    return trimesh.Trimesh(v, np.asarray(mesh.tri_verts), process=True)


def load_free():
    p = CACHE / "free_sections.pkl"
    if not p.exists():
        sys.exit("no torso cache: run cad/torso_fit.py first")
    raw = pickle.loads(p.read_bytes())
    return {t: {zp: m3d.CrossSection(polys, m3d.FillRule.EvenOdd).simplify(0.01) for zp, polys in secs.items()}
            for t, secs in raw.items()}


# ---------------------------------------------------------------- positions
def pi_point(px, py):
    """Pi board coordinates -> (y, zp) behind the upper spine: USB end up, power edge at -y, GPIO edge at +y."""
    return -PI_W / 2 + py, PI_Z0 + px


def pi_holes():
    return [pi_point(px, py) for px, py in PI_HOLES]


def ws_point(wx, wy):
    """Waveshare coordinates (wx along the long edge from the barrel-jack end, wy from the power edge) -> (y, zp)."""
    return WS_L / 2 - wx, WS_TOP_Z - wy


def ws_holes():
    i = (WS_L - WS_PX) / 2, (WS_W - WS_PY) / 2
    return [ws_point(i[0] + a, i[1] + b) for a in (0, WS_PX) for b in (0, WS_PY)]


def mpu_holes():
    return [(MPU_X, -MPU_PITCH / 2), (MPU_X, MPU_PITCH / 2)]


def strap_holes():
    return [(-STRAP_PILOT_Y, STRAP_Z0 + STRAP_H / 2), (STRAP_PILOT_Y, STRAP_Z0 + STRAP_H / 2)]


def ws_box(wx0, wx1, wy0, wy1, x0, x1):
    (ya, za), (yb, zb) = ws_point(wx0, wy0), ws_point(wx1, wy1)
    return box(x0, x1, min(ya, yb), max(ya, yb), min(za, zb), max(za, zb))


# ---------------------------------------------------------------- parts
def mount(base_outline):
    body = base_outline.extrude(BASE_T).translate((0, 0, BASE_Z))
    body += sym_box(LS_X0, LS_X1, LS_HY, FLOOR - 0.5, LS_TOP)                       # lower spine
    body += sym_box(US_X0, US_X1, US_HY, US_Z0, TOP)                                 # upper spine
    body += sym_box(LS_X0, US_X1, LS_HY, US_Z0, LS_TOP)                              # step between them
    for s in (-1, 1):                                                                 # pouch side walls, strap bosses
        y0, y1 = sorted((s * SIDE_Y0, s * SIDE_Y1))
        body += box(RIM_X, LS_X0 + 0.5, y0, y1, FLOOR - 0.5, POUCH_TOP + POUCH_WALL)
        y0, y1 = sorted((s * SIDE_Y0, s * STRAP_BOSS_Y1))
        body += box(RIM_X, RIM_X + STRAP_PILOT_DEPTH + 2.0, y0, y1, STRAP_Z0, STRAP_Z0 + STRAP_H)
        y0, y1 = sorted((s * RIB_Y0, s * RIB_Y1))                                    # front ribs, 45 deg underside
        x0 = LS_X1 - 0.5
        rib = m3d.CrossSection([[(x0, RIB_Z0), (RIB_X1, RIB_Z0 + RIB_X1 - x0), (RIB_X1, RIB_Z1), (x0, RIB_Z1)]])
        body += rib.extrude(y1 - y0).rotate((90, 0, 0)).translate((0, y1, 0))
    body += sym_box(RIM_X, RIM_X + TOP_DEPTH, SIDE_Y1, POUCH_TOP, POUCH_TOP + POUCH_WALL)   # top wall (back 10 mm)
    for x, y in mpu_holes():
        body += cyl_z(x, y, FLOOR - 0.5, FLOOR + MPU_BOSS_H, MPU_BOSS_D / 2)
    for y, z in pi_holes():
        body += cyl_x(y, z, US_X0 - PI_STANDOFF, US_X0 + 0.5, PI_BOSS_R)
        body += cyl_x(y, z, US_X1 - 0.5, US_X1 + PI_FRONT_BOSS, PI_BOSS_R)
    for y, z in ws_holes():
        body += cyl_x(y, z, LS_X1 - 0.5, LS_X1 + WS_STANDOFF, WS_BOSS_R)

    # --- cuts
    for x, y in BASE_HOLES:
        body -= cyl_z(x, y, BASE_Z - 1, FLOOR + 1, M3_CLEAR / 2)
        body -= cyl_z(x, y, FLOOR - M3_HEAD_H, FLOOR + 1, M3_HEAD_D / 2)
    for x, y in mpu_holes():
        top = FLOOR + MPU_BOSS_H
        body -= cyl_z(x, y, top - MPU_PILOT_DEPTH, top + 1, M3_PILOT / 2)
    for y, z in strap_holes():
        body -= cyl_x(y, z, RIM_X - 1, RIM_X + STRAP_PILOT_DEPTH, M3_PILOT / 2)
    for y, z in pi_holes():                                    # through standoff, spine and front boss
        body -= cyl_x(y, z, US_X0 - PI_STANDOFF - 1, US_X1 + PI_FRONT_BOSS + 1, M25_PILOT / 2)
    for y, z in ws_holes():                                    # through standoff and spine; the M2.5x8 ends inside
        body -= cyl_x(y, z, LS_X0 - 1, LS_X1 + WS_STANDOFF + 1, M25_PILOT / 2)
    hy, z0, z1 = LEAD_SLOT
    body -= sym_box(LS_X0 - 1, US_X1 + 1, hy, z0, z1)
    for z in ZIP_Z:
        body -= box(ZIP_X - ZIP_W / 2, ZIP_X + ZIP_W / 2, -RIB_Y1 - 1, RIB_Y1 + 1, z - ZIP_H / 2, z + ZIP_H / 2)
    return body


def strap():
    body = sym_box(RIM_X - STRAP_T, RIM_X, STRAP_HY, STRAP_Z0, STRAP_Z0 + STRAP_H)
    body += sym_box(RIM_X - 0.01, RIM_X + PAD_T, PAD_Y, STRAP_Z0 + 2, STRAP_Z0 + STRAP_H - 2)
    for y, z in strap_holes():
        body -= cyl_x(y, z, RIM_X - STRAP_T - 1, RIM_X + 1, M3_CLEAR / 2)
    return body


def components():
    """Keep-out boxes: name -> (solid, grams). Plugs and leads get 0 g."""
    pb = US_X0 - PI_STANDOFF                       # Pi board's underside
    pt = pb - PI_T                                 # its component side
    stand = None
    for y, z in pi_holes():
        cy = cyl_x(y, z, pb - 0.1, US_X0 + 0.1, PI_BOSS_R)
        stand = cy if stand is None else stand + cy
    c = {
        "battery": (box(LS_X0 - BAT_T, LS_X0, -BAT_W / 2, BAT_W / 2, FLOOR + 0.5, FLOOR + 0.5 + BAT_L), 112),
        "battery_leads": (box(RIM_X + TOP_DEPTH + 0.5, LS_X0, -LEAD_SLOT[0], LEAD_SLOT[0], POUCH_TOP, LEAD_SLOT[2]), 0),
        "pi": (box(pt - PI_TALL, pb, -PI_W / 2, PI_W / 2, PI_Z0, PI_Z0 + PI_L + PI_OVER), 50),
        "pi_underside": (box(pb, pb + 1.5, -PI_W / 2, PI_W / 2, PI_Z0, PI_Z0 + PI_L) - stand, 0),
        "pi_sd_card": (box(pb, pb + 1.5, -6, 6, PI_Z0 - 2.5, PI_Z0 + 15), 0),
        "pi_usb_plugs": (box(pt - PI_TALL, pt, -7.0, 9.0, PI_Z0 + PI_L + PI_OVER, PI_Z0 + PI_L + PI_OVER + 10), 0),
    }
    wb = LS_X1 + WS_STANDOFF                       # Waveshare board's back
    wf = wb + WS_T                                 # its component side
    c.update({
        "ws_board": (ws_box(0, WS_L, 0, WS_W, wb, wf + 4.0), 10),
        "ws_barrel_jack": (ws_box(0, 9, -3, 14, wf, wf + 11.0), 0),
        "ws_terminal": (ws_box(13, 21, -3, 8, wf, wf + 10.0), 0),
        "ws_terminal_wires": (ws_box(13, 21, -18, -3, wf, wf + 10.0), 0),
        "ws_usb_c_plug": (ws_box(24, 35, -21, 0, wf - 1.0, wf + 8.0), 0),
        "ws_servo_sockets": (ws_box(4, 28, 26, WS_W, wf, wf + 6.0), 0),
        "ws_servo_plugs": (ws_box(4, 28, WS_W, WS_W + 11, wf - 0.5, wf + 7.0), 0),
        "ws_uart_header": (ws_box(33, 40, 28, 31, wf, wf + 8.5), 0),
    })
    x0 = MPU_X - MPU_HOLE_EDGE
    zb = FLOOR + MPU_BOSS_H
    c["imu"] = (box(x0, x0 + MPU_DEPTH, -MPU_LEN / 2, MPU_LEN / 2, zb, zb + 2.6), 3)
    c["imu_header"] = (box(x0 + MPU_DEPTH - 4, x0 + MPU_DEPTH, -MPU_LEN / 2, MPU_LEN / 2, zb + 2.6, zb + 21), 0)
    return c


GROUP = {"battery_leads": "battery", "pi_underside": "pi", "pi_sd_card": "pi", "pi_usb_plugs": "pi",
         "imu_header": "imu"}


def group(n):
    return "ws" if n.startswith("ws_") else GROUP.get(n, n)


# ---------------------------------------------------------------- torso fit
def torso_check(asm, free, top):
    """Assembly order: T03 on the plate first, the mount lowered into it, then T04, T05, T06 lowered over it.
    The mount's section at height h must pass every T03 section above h on the way down, and every section of a
    later tube that slides past it (that tube's sections from its bottom up to h). Returns per tube the worst area
    (mm2) of the mount outside that free space, at margin 0 and MARGIN."""
    res = {}
    hs = list(range(int(BASE_Z) + 1, int(np.ceil(top)) + 1))
    sec = {h: asm.slice(h - 0.5) for h in hs}
    for t in TUBES:
        f = free[t]
        lo, hi = min(f), max(f)
        worst = {0.0: (0.0, None), MARGIN: (0.0, None)}
        acc = None
        order = range(hi, lo - 1, -1) if t == TUBES[0] else range(lo, hi + 1)
        allowed = {}
        for s in order:                         # running intersection: T03 from its top down, the others bottom up
            if s in f:
                acc = f[s] if acc is None else acc ^ f[s]
            allowed[s] = acc
        for h in hs:
            if t == TUBES[0]:
                if h > hi:
                    continue
                a = allowed[max(h, lo)]
            else:
                if h < lo:
                    continue
                a = allowed[min(h, hi)]
            for m in worst:
                out = (sec[h] - (a.offset(-m, m3d.JoinType.Round) if m else a)).area()
                if out > worst[m][0]:
                    worst[m] = (round(out, 1), h)
        res[t] = {f"margin_{m:g}mm": {"outside_mm2": w[0], "at_zp": w[1]} for m, w in worst.items()}
    return res


def base_outline(free):
    """Inside every T03 section the base passes on its way down (the later tubes stop above it), inset, cut to the
    base's x range."""
    acc = None
    for s, cs in free[TUBES[0]].items():
        if s >= BASE_Z:
            acc = cs if acc is None else acc ^ cs
    region = acc.offset(-BASE_INSET, m3d.JoinType.Round)
    rect = m3d.CrossSection.square((BASE_X1 - BASE_X0, 200.0)).translate((BASE_X0, -100.0))
    return (region ^ rect).offset(-2.0, m3d.JoinType.Round).offset(2.0, m3d.JoinType.Round).simplify(0.05)


# ---------------------------------------------------------------- main
def main():
    free = load_free()
    a = mount(base_outline(free))
    b = strap()
    comps = components()
    rep = {"parts": {}, "checks": {}, "torso_fit": {}, "screws": {}}
    for name, man in (("E01_mount", a), ("E02_battery_strap", b)):
        tm = to_tm(man, 0.0)
        assert tm.is_watertight and tm.volume > 0, name
        ext = tm.bounds[1] - tm.bounds[0]
        rep["parts"][name] = {"watertight": bool(tm.is_watertight), "bodies": int(len(tm.split(only_watertight=False))),
                              "size_mm": [round(float(e), 1) for e in ext], "solid_g": round(tm.volume * DENSITY, 1),
                              "printed_g_est": round(tm.volume * DENSITY * FILL, 1)}
    ck = rep["checks"]
    ck["mount_x_strap_mm3"] = round(vol(a ^ b), 2)
    hits = {}
    names = list(comps)
    for n in names:
        for part, man in (("mount", a), ("strap", b)):
            v = vol(comps[n][0] ^ man)
            if v > 0.05:
                hits[f"{n} x {part}"] = round(v, 2)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            if group(n1) != group(n2):
                v = vol(comps[n1][0] ^ comps[n2][0])
                if v > 0.05:
                    hits[f"{n1} x {n2}"] = round(v, 2)
    ck["overlaps"] = hits
    pouch_in = sym_box(RIM_X, LS_X0, SIDE_Y0, FLOOR, POUCH_TOP)
    ck["battery_outside_pouch_mm3"] = round(vol(comps["battery"][0] - pouch_in), 2)
    ck["strap_pad_to_battery_mm"] = round((LS_X0 - BAT_T) - (RIM_X + PAD_T), 2)

    asm = a + b
    for man, g in comps.values():
        asm += man
    top = asm.bounding_box()[5]
    rep["torso_fit"] = torso_check(asm, free, top)
    rep["torso_fit"]["top_zp"] = round(top, 1)
    rep["torso_fit"]["note"] = ("mount on T03's floor (zp 3); worst area of any part outside the free space of each "
                                "tube as it slides past, with 0 and 1 mm clearance")

    items = [(n, to_tm(m).volume * DENSITY * FILL, to_tm(m).center_mass) for n, m in (("E01_mount", a), ("E02_battery_strap", b))]
    items += [(n, g, to_tm(m).center_mass) for n, (m, g) in comps.items() if g]
    items.append(("screws", SCREWS_G, np.array([0.0, 0.0, PLATE_TOP + 60.0])))
    items.append((FLOATING[0], FLOATING[1], np.array(FLOATING[2]) + (0, 0, PLATE_TOP)))
    m_tot = sum(g for _, g, _ in items)
    com = sum(g * c for _, g, c in items) / m_tot
    rep["mass"] = {"items_g": {n: round(g, 1) for n, g, _ in items}, "total_g": round(m_tot, 1),
                   "com_mm_leg_frame": [round(float(c), 1) for c in com],
                   "com_height_above_plate_top_mm": round(float(com[2] - PLATE_TOP), 1)}
    rep["screws"] = {
        "base": "4 x M3 down through the counterbores (2.8 mm of base under the head): with the torso (+3 mm floor, 5 mm "
                "plate) M3x14 into the nuts / ear nuts, M3x16 if an ear nut doesn't catch; without the torso M3x12 + nut",
        "battery strap": "2 x M3x10 (3 mm bar + 7 into the pouch-wall bosses)",
        "raspberry pi": "4 x M2.5x10 (1.4 board + 3 standoff + 3 spine + 3 front boss, pilot 2.5 through)",
        "waveshare": "4 x M2.5x8 (1.6 board + 4 standoff + 2.4 of the 3 mm spine; pilot 2.5 through)",
        "imu": "2 x M3x6 or M3x8 (pilot 2.9, 8 deep from the boss top)",
    }

    out = ROOT / "exports"
    (out / "print_ready").mkdir(parents=True, exist_ok=True)
    (out / "assembled").mkdir(parents=True, exist_ok=True)
    pa = to_tm(a, 0.0)
    pa.apply_translation((-pa.bounds[:, 0].mean(), -pa.bounds[:, 1].mean(), -pa.bounds[0, 2]))
    pa.export(out / "print_ready" / "E01_mount.stl")
    pb = to_tm(b, 0.0)                                         # flat: bar on the bed, pad up
    pb.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0)))
    pb.apply_translation((-pb.bounds[:, 0].mean(), -pb.bounds[:, 1].mean(), -pb.bounds[0, 2]))
    pb.export(out / "print_ready" / "E02_battery_strap.stl")
    to_tm(a).export(out / "assembled" / "E01_mount.stl")
    to_tm(b).export(out / "assembled" / "E02_battery_strap.stl")
    trimesh.util.concatenate([to_tm(m) for m, g in comps.values()]).export(out / "assembled" / "NOT_PRINTABLE_components.stl")
    (ROOT / "mount_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if "--no-render" not in sys.argv:
        render(to_tm(a), to_tm(b), {n: to_tm(m) for n, (m, g) in comps.items()})


def render(ta, tb, comps):
    """MuJoCo renders (a real depth buffer)."""
    import os
    import tempfile
    os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco
    from PIL import Image

    def colour(n):
        for k, c in (("E01", ".9 .89 .86 1"), ("E02", ".85 .55 .23 1"), ("P08", ".62 .60 .56 1"),
                     ("battery_leads", ".75 .15 .12 1"), ("battery", ".18 .45 .25 1"), ("pi_usb", ".55 .55 .6 1"),
                     ("pi", ".24 .6 .3 1"), ("ws_board", ".13 .17 .28 1"), ("ws_barrel", ".08 .08 .08 1"),
                     ("ws_terminal_wires", ".75 .15 .12 1"), ("ws_terminal", ".2 .6 .3 1"), ("ws_", ".9 .9 .85 1"),
                     ("imu_header", ".1 .1 .1 1"), ("imu", ".12 .31 .63 1"), ("T0", ".7 .74 .8 .28")):
            if n.startswith(k):
                return c
        return ".5 .5 .5 1"

    meshes = {"E01_mount": ta, "E02_battery_strap": tb,
              **{n: m for n, m in comps.items() if n not in ("pi_underside", "pi_sd_card")}}
    if P08.exists():
        meshes["P08"] = trimesh.load(P08)
    torso = {t: trimesh.load(CACHE / f"{t}.stl") for t in TUBES if (CACHE / f"{t}.stl").exists()}
    (ROOT / "renders").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        asset = []
        for n, tm in {**meshes, **torso}.items():
            tm.export(f"{tmp}/{n}.stl")
            asset.append(f'<mesh name="{n}" file="{tmp}/{n}.stl" scale=".001 .001 .001"/>')
        for tag, with_torso, az, el, dist in (("front", False, 215, -15, 0.36), ("back", False, 35, -15, 0.36),
                                              ("side", False, 270, -3, 0.36), ("in_torso", True, 220, -12, 0.36)):
            names = list(meshes) + (list(torso) if with_torso else [])
            geoms = "".join(f'<geom type="mesh" mesh="{n}" rgba="{colour(n)}"/>' for n in names)
            xml = f"""<mujoco><compiler angle="degree"/>
<visual><global offwidth="1400" offheight="1400"/>
<headlight ambient=".38 .38 .38" diffuse=".45 .45 .45" specular=".05 .05 .05"/></visual>
<asset><texture type="skybox" builtin="flat" rgb1="1 1 1" rgb2="1 1 1" width="8" height="8"/>{"".join(asset)}</asset>
<default><geom contype="0" conaffinity="0"/></default>
<worldbody><light pos=".3 -.2 .6" dir="-.4 .3 -1" directional="true" diffuse=".45 .45 .45" castshadow="false"/>
{geoms}</worldbody></mujoco>"""
            mdl = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(mdl)
            mujoco.mj_forward(mdl, d)
            r = mujoco.Renderer(mdl, 900, 900)
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = (0.0, 0.0, (PLATE_TOP + (110 if with_torso else 95)) * 1e-3)
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            r.update_scene(d, camera=cam)
            img = r.render()
            if with_torso:                                   # crop the white margin, keep 24 px
                ys, xs = np.nonzero((img < 245).any(axis=2))
                img = img[max(ys.min() - 24, 0):ys.max() + 25, max(xs.min() - 24, 0):xs.max() + 25]
            Image.fromarray(img).save(ROOT / "renders" / f"mount_{tag}.png")
            r.close()


if __name__ == "__main__":
    main()
