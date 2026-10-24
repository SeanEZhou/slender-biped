"""Parametric MJCF generator for the slender biped.

Three kinds of geoms are emitted, kept strictly separate:
  * exterior shells (group 0): convex-hull meshes; carry the printed-shell mass with
    shell inertia; never collide.
  * internal actuator envelopes (group 1): servo case boxes carrying the servo mass;
    never collide. Rendered translucent orange (red if they protrude through the
    exterior) so exterior silhouette and internal envelope can be compared.
  * collision geoms (group 3): simple primitives slightly inside the shells, zero
    mass. The foot tip geom is both the visible rubber cap and the only ground
    contact surface below the shin -- there is no hidden sole.
Point masses for electronics/fasteners/wiring live in group 4 (invisible).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from .config import JOINTS, SIDES, joint_servos, load_actuators, model_name

COLORS = {
    "shell": "0.92 0.90 0.85 1",
    "joint": "0.19 0.20 0.22 1",
    "rubber": "0.07 0.07 0.08 1",
    "accent": "0.62 0.10 0.12 1",
    "env": "1.0 0.55 0.10 0.55",
    "env_bad": "1.0 0.05 0.05 0.85",
    "col": "0.2 0.6 1.0 0.35",
    "inert": "0.5 0.5 0.5 0",
}


def _spow(a, e):
    return np.sign(a) * np.abs(a) ** e


def superellipsoid(ax, ay, az, n=2.0, taper_bottom=1.0, nu=18, nv=40):
    """Vertex cloud of a superellipsoid; optional linear y-taper (1 at top -> taper_bottom at bottom)."""
    pts = []
    for u in np.linspace(-np.pi / 2, np.pi / 2, nu):
        for v in np.linspace(-np.pi, np.pi, nv, endpoint=False):
            x = ax * _spow(np.cos(u), 2 / n) * _spow(np.cos(v), 2 / n)
            y = ay * _spow(np.cos(u), 2 / n) * _spow(np.sin(v), 2 / n)
            z = az * _spow(np.sin(u), 2 / n)
            s = taper_bottom + (1 - taper_bottom) * (z + az) / (2 * az)
            pts.append((x, y * s, z))
    return np.unique(np.round(np.array(pts), 7), axis=0)


def superellipse_ring(w, d, n, z, nv=40):
    v = np.linspace(-np.pi, np.pi, nv, endpoint=False)
    return np.stack([d / 2 * _spow(np.cos(v), 2 / n), w / 2 * _spow(np.sin(v), 2 / n), np.full(nv, z)], 1)


def tube(z_top, top, z_bot, bot, rings=8, cap_top=0.5, cap_bot=0.5, nv=40, mid=None):
    """Superellipse tube with domed ends (convex hull). top/bot/mid: dicts with w, d, n (mid also z).

    Without `mid` the taper is linear; with it the profile is piecewise linear through mid,
    which lets a shell stay full-width over a servo and taper below it.
    """
    secs = [(z_top, top)] + ([(mid["z"], mid)] if mid else []) + [(z_bot, bot)]
    pts = []
    for (za, a), (zb, b) in zip(secs[:-1], secs[1:]):
        for t in np.linspace(0, 1, rings):
            w = a["w"] + t * (b["w"] - a["w"])
            d = a["d"] + t * (b["d"] - a["d"])
            n = a.get("n", 2) + t * (b.get("n", 2) - a.get("n", 2))
            pts.append(superellipse_ring(w, d, n, za + t * (zb - za), nv))
    for sec, z0, sgn, cap in ((top, z_top, 1, cap_top), (bot, z_bot, -1, cap_bot)):
        h = cap * min(sec["w"], sec["d"]) / 2
        for phi in np.linspace(0, np.pi / 2, 6)[1:]:
            c = np.cos(phi)
            if c < 1e-3:
                pts.append(np.array([[0, 0, z0 + sgn * h]]))
            else:
                pts.append(superellipse_ring(sec["w"] * c, sec["d"] * c, sec.get("n", 2), z0 + sgn * h * np.sin(phi), nv))
    return np.unique(np.round(np.concatenate(pts), 7), axis=0)


def hull_area(pts):
    return ConvexHull(pts).area


def armature(spec, servo_model):
    """Reflected rotor inertia J_rotor * N^2 (estimate; see configs/actuators.yaml)."""
    n = spec.get("gear_ratio") or spec.get("gear_ratio_assumed")
    if spec.get("rotor_inertia") and n:
        return float(spec["rotor_inertia"]) * float(n) ** 2
    return servo_model["armature_fallback"]


def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


@dataclass
class Body:
    name: str
    pos: tuple
    joint: dict | None = None
    xml: list = field(default_factory=list)
    children: list = field(default_factory=list)


class Builder:
    def __init__(self, cfg: dict, act: dict | None = None):
        self.cfg = cfg
        self.act = act or load_actuators()
        self.servos = joint_servos(cfg, self.act)
        self.meshes: list[tuple[str, np.ndarray]] = []
        self.mass_log: list[tuple[str, str, float]] = []  # (body, item, kg)
        self.envelopes: list[dict] = []  # servo boxes for packaging check
        self.exteriors: list[str] = []  # geom names of exterior shells
        self.sm = self.act["servo_model"]

    # ---------------------------------------------------------------- helpers
    def mesh_geom(self, body: Body, name, pts, material, mass=None, wall=None, pos=(0, 0, 0)):
        mname = f"m_{name}"
        self.meshes.append((mname, pts))
        if mass is None:
            m = self.cfg["material"]
            mass = hull_area(pts) * wall * m["shell_density"] * m["rib_factor"]
        self.mass_log.append((body.name, f"shell:{name}", mass))
        self.exteriors.append(name)
        body.xml.append(
            f'<geom name="{name}" class="vis" type="mesh" mesh="{mname}" material="{material}" '
            f'pos="{fmt(pos)}" mass="{mass:.6g}"/>')

    def inert(self, body: Body, name, mass, pos):
        self.mass_log.append((body.name, name, mass))
        body.xml.append(f'<geom name="{body.name}_{name}" class="inert" type="sphere" size="0.003" '
                        f'pos="{fmt(pos)}" mass="{mass:.6g}"/>')

    def servo_box(self, body: Body, jname, spec, center, axes):
        """axes: mapping of 'L','W','H' -> body axis index (0=x,1=y,2=z)."""
        half = [0.0, 0.0, 0.0]
        for k, ax in axes.items():
            half[ax] = spec["dims"][k] / 2
        name = f"env_{jname}"
        self.mass_log.append((body.name, f"servo:{spec['model']}", spec["mass"]))
        self.envelopes.append({"name": name, "joint": jname, "model": spec["model"]})
        body.xml.append(f'<geom name="{name}" class="env" type="box" size="{fmt(half)}" '
                        f'pos="{fmt(center)}" mass="{spec["mass"]:.6g}"/>')

    def col(self, body: Body, name, gtype, size, pos=(0, 0, 0), fromto=None, extra=""):
        if fromto is not None:
            body.xml.append(f'<geom name="{name}" class="col" type="{gtype}" size="{fmt(size)}" fromto="{fmt(fromto)}" {extra}/>')
        else:
            body.xml.append(f'<geom name="{name}" class="col" type="{gtype}" size="{fmt(size)}" pos="{fmt(pos)}" {extra}/>')

    # ------------------------------------------------------------------ parts
    def foot_geom(self, body: Body, side, contact_z, mount_pitch):
        """Tip geom whose lowest point (at nominal stance orientation) is at contact_z."""
        c = self.cfg
        if c["foot"] == "A":
            fa = c["foot_A"]
            r = fa["radius"]
            pos = (0, 0, contact_z + r)
            body.xml.append(
                f'<geom name="{side}_tip" class="foot" type="sphere" size="{r}" pos="{fmt(pos)}" '
                f'mass="{fa["mass"]}" friction="{fmt(fa["friction"])}"/>')
            self.mass_log.append((body.name, "tip:rubber_sphere", fa["mass"]))
            tip_top = contact_z + 2 * r
        elif c["foot"][0] in "FP":  # flat rubber sole (F: control frame) or flat pad at the shin tip (P)
            ff = c[f"foot_{c['foot']}"]
            a, b, cc = ff["length"] / 2, ff["width"] / 2, ff["thickness"] / 2
            th = mount_pitch
            x0 = a - ff["toe_frac"] * 2 * a  # sole centre relative to the ankle (toe longer than heel)
            h = abs(a * math.sin(th)) + cc * math.cos(th)  # box support depth
            # sole centre = point below the ankle + x0 along the levelled sole + cc along its normal
            pos = (x0 * math.cos(th) - cc * math.sin(th), 0, contact_z + x0 * math.sin(th) + cc * math.cos(th))
            quat = (math.cos(th / 2), 0, -math.sin(th / 2), 0)
            body.xml.append(
                f'<geom name="{side}_tip" class="foot" type="box" size="{fmt([a, b, cc])}" '
                f'pos="{fmt(pos)}" quat="{fmt(quat)}" mass="{ff["mass"]}" friction="{fmt(ff["friction"])}"/>')
            self.mass_log.append((body.name, "tip:rubber_sole", ff["mass"]))
            tip_top = contact_z + 2 * h
        else:
            fb = c["foot_B"]
            a, b, cc = fb["half_length"], fb["half_width"], fb["half_height"]
            th = mount_pitch
            h = math.sqrt((a * math.sin(th)) ** 2 + (cc * math.cos(th)) ** 2)  # support depth
            pos = (fb.get("x_offset", 0.0), 0, contact_z + h)
            # rotate about -Y by th => toe up by th (matches ankle sign convention)
            quat = (math.cos(th / 2), 0, -math.sin(th / 2), 0)
            body.xml.append(
                f'<geom name="{side}_tip" class="foot" type="ellipsoid" size="{fmt([a, b, cc])}" '
                f'pos="{fmt(pos)}" quat="{fmt(quat)}" mass="{fb["mass"]}" friction="{fmt(fb["friction"])}"/>')
            self.mass_log.append((body.name, "tip:rubber_rocker", fb["mass"]))
            tip_top = contact_z + 2 * h
        body.xml.append(f'<site name="{side}_contact" pos="0 0 {contact_z:.6g}" size="0.002"/>')
        body.xml.append(f'<site name="{side}_touch" type="sphere" pos="{fmt(pos)}" size="0.017" rgba="0 1 0 0"/>')
        return tip_top

    def build_leg(self, side: str) -> Body:
        c, s = self.cfg, SIDES[side]
        P, H, T, S = c["pelvis"], c["hip"], c["thigh"], c["shin"]
        lim = {k: np.radians(v) for k, v in c["joint_limits_deg"].items()}
        sv = self.servos
        horn = H["horn_clearance"]

        def joint(jn, axis):
            spec = sv[jn]
            return {"name": f"{side}_{jn}", "axis": axis, "range": lim[jn], "spec": spec}

        # --- hip yaw link: carries roll servo (shaft along X, case hangs below yaw joint) ---
        yaw = Body(f"{side}_hip_yaw", (0, s * P["hip_spacing"] / 2, P["yaw_joint_z"]),
                   joint("hip_yaw", (0, 0, s)))
        rs = sv["hip_roll"]
        roll_z = -H["roll_below_yaw"]
        rc = (0, 0, roll_z + rs["dims"]["L"] / 2 - rs["shaft_offset"])
        self.servo_box(yaw, f"{side}_hip_roll", rs, rc, {"H": 0, "W": 1, "L": 2})
        self.inert(yaw, "bracket", H["bracket_mass"], (0, 0, -0.004))
        rcl = H.get("roll_cover_clearance", 0.0035)
        cover = superellipsoid(rs["dims"]["H"] / 2 + horn + 0.002, rs["dims"]["W"] / 2 + rcl,
                               rs["dims"]["L"] / 2 + rcl, n=6)
        self.mesh_geom(yaw, f"{side}_roll_cover", cover + np.array(rc), "joint", wall=0.0015)

        # --- hip roll link: carries pitch servo (shaft along Y, case extends backward) ---
        roll = Body(f"{side}_hip_roll", (0, 0, roll_z), joint("hip_roll", (s, 0, 0)))
        ps = sv["hip_pitch"]
        pitch_z = -H["pitch_below_roll"]
        pc = (-(ps["dims"]["L"] / 2 - ps["shaft_offset"]), 0, pitch_z)
        self.servo_box(roll, f"{side}_hip_pitch", ps, pc, {"L": 0, "H": 1, "W": 2})
        self.inert(roll, "cross_bracket", H["cross_mass"], (0, 0, pitch_z / 2))
        ch = H["cover_half"]
        self.mesh_geom(roll, f"{side}_hip_cover", superellipsoid(*ch, n=H["cover_exponent"]) + np.array(pc),
                       "shell", wall=0.0016)
        r_h = min(ch[1], ch[2]) * 0.85  # capsules, not boxes: fewer contacts and MJX-friendly
        self.col(roll, f"{side}_hip_col", "capsule", [r_h],
                 fromto=[pc[0] - ch[0] * 0.85 + r_h, 0, pc[2], pc[0] + ch[0] * 0.85 - r_h, 0, pc[2]])

        # --- thigh ---
        thigh = Body(f"{side}_thigh", (0, 0, pitch_z), joint("hip_pitch", (0, -1, 0)))
        L1 = T["length"]
        self.mesh_geom(thigh, f"{side}_thigh_shell", tube(T["top_z"], T["top"], -L1, T["bottom"], cap_bot=0.2, mid=T.get("mid")),
                       "shell", wall=T["shell_wall"])
        ks = sv["knee"]
        if T["knee_drive"] == "remote":
            kc = (0, 0, T["knee_servo_z"])
            self.inert(thigh, "belt_pulleys", T["belt_mass"], (0, 0, (T["knee_servo_z"] - L1) / 2))
        else:
            kc = (0, 0, -L1 + ks["dims"]["L"] / 2 - ks["shaft_offset"])
        self.servo_box(thigh, f"{side}_knee", ks, kc, {"W": 0, "H": 1, "L": 2})
        self.inert(thigh, "misc", T["misc_mass"], (0, 0, -L1 * 0.5))
        kh = c["knee_cover"]["half"]
        self.mesh_geom(thigh, f"{side}_knee_cover", superellipsoid(*kh, n=c["knee_cover"].get("n", 2.0)) + np.array([0, 0, -L1]),
                       "joint", wall=0.0015)
        rt = min(T["top"]["w"], T["top"]["d"]) / 2 * 0.85
        rb = min(T["bottom"]["w"], T["bottom"]["d"]) / 2 * 0.85
        self.col(thigh, f"{side}_thigh_col_u", "capsule", [rt], fromto=[0, 0, T["top_z"] - rt, 0, 0, -L1 * 0.5])
        self.col(thigh, f"{side}_thigh_col_l", "capsule", [rb], fromto=[0, 0, -L1 * 0.5, 0, 0, -L1 + kh[2]])
        self.col(thigh, f"{side}_knee_col", "sphere", [min(kh) * 0.92], pos=(0, 0, -L1))

        # --- shin ---
        shin = Body(f"{side}_shin", (0, 0, -L1), joint("knee", (0, 1, 0)))
        L2 = S["length"]
        pose = {k: np.radians(v) for k, v in c["nominal_pose_deg"].items()}
        stance_pitch = pose["knee"] - pose["hip_pitch"]  # toe-up correction to level the tip at stance
        self.inert(shin, "misc", S["misc_mass"], (0, 0, -L2 * 0.45))
        if c["ankle"]:
            z_ank = -(L2 - c["ankle_above_contact"])
            self.mesh_geom(shin, f"{side}_shin_shell", tube(S["top_z"], S["top"], z_ank, S["bottom"], cap_bot=0.3, mid=S.get("mid")),
                           "shell", wall=S["shell_wall"])
            asv = sv["ankle"]
            self.servo_box(shin, f"{side}_ankle", asv, (0, 0, S["ankle_servo_z"]), {"H": 0, "W": 1, "L": 2})
            self.inert(shin, "pushrod", S["pushrod_mass"], (0, 0, (S["ankle_servo_z"] + z_ank) / 2))
            if c.get("ankle_roll"):  # control frame: ankle pitch link, then ankle roll carrying the sole
                rsv = sv["ankle_roll"]
                self.servo_box(shin, f"{side}_ankle_roll", rsv, (0, 0, c["ankle_roll_servo_z"]), {"H": 0, "W": 1, "L": 2})
                link = Body(f"{side}_ankle_link", (0, 0, z_ank), joint("ankle", (0, -1, 0)))
                ah = 0.0075
                self.mesh_geom(link, f"{side}_ankle_cover", superellipsoid(ah, ah, ah, n=2.0), "joint", wall=0.0012)
                foot = Body(f"{side}_foot", (0, 0, 0), joint("ankle_roll", (s, 0, 0)))
                link.children.append(foot)
                shin.children.append(link)
            else:
                foot = Body(f"{side}_foot", (0, 0, z_ank), joint("ankle", (0, -1, 0)))
                ah = 0.0062
                self.mesh_geom(foot, f"{side}_ankle_cover", superellipsoid(ah, ah, ah, n=2.0), "joint", wall=0.0012)
                shin.children.append(foot)
            tip_top = self.foot_geom(foot, side, -c["ankle_above_contact"], 0.0)
            if c["foot"][0] in "FP":  # visual strut ankle -> sole (massless, no contact)
                foot.xml.append(f'<geom name="{side}_ankle_strut" class="vis" type="capsule" size="0.005" '
                                f'fromto="0 0 0 0 0 {tip_top:.6g}" material="joint" mass="0"/>')
            col_bot = z_ank + 0.004
        else:
            foot = None
            tip_top = self.foot_geom(shin, side, -L2, stance_pitch if c["foot"] != "A" else 0.0)
            ftop = -L2 + (tip_top + L2) * 0.55
            self.mesh_geom(shin, f"{side}_shin_shell", tube(S["top_z"], S["top"], ftop, S["bottom"], cap_bot=0.3, mid=S.get("mid")),
                           "shell", wall=S["shell_wall"])
            col_bot = tip_top + 0.006
        r1 = min(S["top"]["w"], S["top"]["d"]) / 2 * 0.85
        r2 = min(S["bottom"]["w"], S["bottom"]["d"]) / 2 * 0.85
        self.col(shin, f"{side}_shin_col_u", "capsule", [r1], fromto=[0, 0, -0.018, 0, 0, -L2 * 0.45])
        self.col(shin, f"{side}_shin_col_l", "capsule", [r2], fromto=[0, 0, -L2 * 0.45, 0, 0, col_bot])

        yaw.children.append(roll)
        roll.children.append(thigh)
        thigh.children.append(shin)
        return yaw

    def build_pelvis(self) -> Body:
        c = self.cfg
        P = c["pelvis"]
        pel = Body("pelvis", (0, 0, 0.5))
        hw, hd, hh = P["width"] / 2, P["depth"] / 2, P["height"] / 2
        self.mesh_geom(pel, "pelvis_shell", superellipsoid(hd, hw, hh, n=P.get("exponent", 3.2), taper_bottom=P["bottom_taper"]),
                       "shell", wall=P["shell_wall"])
        pel.xml.append(f'<geom name="pelvis_band" class="vis" type="box" size="{hd*0.995:.4f} {hw*0.55:.4f} 0.0015" '
                       f'pos="0 0 {hh*0.35:.4f}" material="accent" mass="0"/>')
        for nm, (m, x, y, z) in P["electronics"].items():
            self.inert(pel, nm, m, (x, y, z))
        ys = self.servos["hip_yaw"]
        for side, s in SIDES.items():
            zc = P["yaw_joint_z"] + 0.002 + ys["dims"]["H"] / 2
            off = ys["dims"]["L"] / 2 - ys["shaft_offset"]
            if P.get("yaw_servo_dir", "inward") == "inward":  # case points at the midline: shallow pelvis
                yc = (0, s * (P["hip_spacing"] / 2 - off), zc)
                self.servo_box(pel, f"{side}_hip_yaw", ys, yc, {"W": 0, "L": 1, "H": 2})
            else:
                yc = (off, s * P["hip_spacing"] / 2, zc)
                self.servo_box(pel, f"{side}_hip_yaw", ys, yc, {"L": 0, "W": 1, "H": 2})
        pod = P.get("pod", {})
        if pod.get("enabled"):
            sx, sy, sz = pod["size"]
            pz = hh + sz / 2 - 0.002
            self.mesh_geom(pel, "pod_shell", superellipsoid(sx / 2, sy / 2, sz / 2, n=3.0) + np.array([0, 0, pz]),
                           "joint", mass=pod["shell_mass"])
            self.inert(pel, "battery", pod["battery_mass"], (0, 0, pz))
            r_p = sz / 2 * 0.85
            self.col(pel, "pod_col", "capsule", [r_p], fromto=[-sx / 2 * 0.85 + r_p, 0, pz, sx / 2 * 0.85 - r_p, 0, pz])
        r_pel = min(hd, hh) * 0.88
        y_pel = hw * 0.88 * P["bottom_taper"] - r_pel
        self.col(pel, "pelvis_col", "capsule", [r_pel], fromto=[0, -y_pel, 0, 0, y_pel, 0])
        pel.xml.append('<site name="imu" pos="0 0 0.02" size="0.004" rgba="0 1 0 1"/>')
        for side in SIDES:
            pel.children.append(self.build_leg(side))
        return pel

    # -------------------------------------------------------------- emission
    def emit_body(self, b: Body, depth=2) -> str:
        ind = "  " * depth
        lines = [f'{ind}<body name="{b.name}" pos="{fmt(b.pos)}">']
        if b.name == "pelvis":
            lines.append(f'{ind}  <freejoint name="root"/>')
        if b.joint:
            j = b.joint
            lines.append(f'{ind}  <joint name="{j["name"]}" axis="{fmt(j["axis"])}" range="{fmt(j["range"])}" '
                         f'armature="{armature(j["spec"], self.sm):.6g}"/>')
        lines += [f"{ind}  {x}" for x in b.xml]
        for ch in b.children:
            lines.append(self.emit_body(ch, depth + 1))
        lines.append(f"{ind}</body>")
        return "\n".join(lines)

    def joint_list(self) -> list[str]:
        out = []
        for side in SIDES:
            for j in JOINTS:
                if j in self.servos:
                    out.append(f"{side}_{j}")
        return out

    def contact_pairs(self) -> str:
        """Explicit contact pairs (all geoms have contype = conaffinity = 0).

        floor <-> every collision geom (non-tip floor contact = fall); left <-> right leg pairs
        that can actually meet. Dropped: hip covers vs each other (joint limits keep them apart)
        and a tip vs its own leg. Keeps MJX (GPU) supported and fast, and CPU eval identical.
        """
        c = self.cfg
        fc = c[f"foot_{c['foot']}"]["friction"]  # [slide, torsional, rolling]
        tip_fr = f"{fc[0]} {fc[0]} {fc[1]} {fc[2]} {fc[2]}"
        body = ["hip_col", "thigh_col_u", "thigh_col_l", "knee_col", "shin_col_u", "shin_col_l"]
        lines = []
        lean = c.get("contact_set", "full") == "train"
        for s in "LR":
            condim = c[f"foot_{c['foot']}"].get("condim", c.get("tip_condim", 4))
            lines.append(f'    <pair geom1="floor" geom2="{s}_tip" condim="{condim}" friction="{tip_fr}" '
                         f'solref="0.004 1" solimp="0.9 0.97 0.001"/>')
            if lean:  # training set: falls are detected kinematically and end the episode
                continue
            for b in body:
                lines.append(f'    <pair geom1="floor" geom2="{s}_{b}" condim="3" friction="0.8 0.8 0.005 0.0001 0.0001"/>')
        for b in [] if lean else ["pelvis_col"] + (["pod_col"] if c["pelvis"].get("pod", {}).get("enabled") else []):
            lines.append(f'    <pair geom1="floor" geom2="{b}" condim="3" friction="0.8 0.8 0.005 0.0001 0.0001"/>')
        cross = [("thigh_col_l", "thigh_col_l"), ("knee_col", "knee_col"), ("shin_col_u", "shin_col_u"),
                 ("shin_col_l", "shin_col_l"), ("knee_col", "shin_col_u"), ("shin_col_u", "knee_col"),
                 ("shin_col_l", "tip"), ("tip", "shin_col_l"), ("tip", "tip"), ("shin_col_u", "shin_col_l"),
                 ("shin_col_l", "shin_col_u")]
        if lean:
            cross = [("thigh_col_l", "thigh_col_l"), ("knee_col", "knee_col"), ("shin_col_l", "shin_col_l"),
                     ("shin_col_l", "tip"), ("tip", "shin_col_l")]
        for a, b in cross:
            lines.append(f'    <pair geom1="L_{a}" geom2="R_{b}" condim="3" friction="0.4 0.4 0.005 0.0001 0.0001"/>')
        return "\n".join(lines)

    def build(self, keyframe_z: float | None = None, keyframe_q: list | None = None) -> str:
        self.meshes, self.mass_log, self.envelopes, self.exteriors = [], [], [], []
        root = self.build_pelvis()
        sm = self.sm
        assets = "\n".join(f'    <mesh name="{n}" vertex="{fmt(p.ravel())}" inertia="shell"/>' for n, p in self.meshes)
        acts, sens = [], []
        for jn in self.joint_list():
            spec = self.servos[jn.split("_", 1)[1]]
            ts = spec["stall_torque"]
            acts.append(f'    <motor name="{jn}" joint="{jn}" ctrlrange="{-ts:.4g} {ts:.4g}"/>')
            sens.append(f'    <jointpos name="{jn}_pos" joint="{jn}"/>')
            sens.append(f'    <jointvel name="{jn}_vel" joint="{jn}"/>')
            sens.append(f'    <actuatorfrc name="{jn}_frc" actuator="{jn}"/>')
        for side in SIDES:
            sens.append(f'    <touch name="{side}_touch" site="{side}_touch"/>')
        key = ""
        if keyframe_z is not None:
            q = [0, 0, keyframe_z, 1, 0, 0, 0] + list(keyframe_q)
            key = f'  <keyframe>\n    <key name="stand" qpos="{fmt(q)}"/>\n  </keyframe>'
        return f"""<mujoco model="slender_biped_{model_name(self.cfg)}">
  <!-- GENERATED by models/generate_model.py from configs/geometry.yaml + configs/actuators.yaml. Do not edit. -->
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="1"/>
  <size memory="20M"/>
  <visual>
    <global offwidth="1600" offheight="1600"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
  </visual>
  <default>
    <joint armature="{sm['armature_fallback']}" frictionloss="{sm['frictionloss']}" damping="0.0"/>
    <geom solref="0.004 1"/>
    <default class="vis"><geom contype="0" conaffinity="0" group="0"/></default>
    <default class="env"><geom contype="0" conaffinity="0" group="1" rgba="{COLORS['env']}"/></default>
    <default class="col"><geom contype="0" conaffinity="0" group="3" mass="0" rgba="{COLORS['col']}" condim="3" friction="0.6 0.005 0.0001"/></default>
    <default class="inert"><geom contype="0" conaffinity="0" group="4" rgba="{COLORS['inert']}"/></default>
    <default class="foot"><geom contype="0" conaffinity="0" group="0" material="rubber" condim="4" solref="0.004 1" solimp="0.9 0.97 0.001"/></default>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.86 0.88 0.91" width="256" height="1536"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.80 0.80 0.80" rgb2="0.70 0.70 0.70" width="512" height="512" mark="edge" markrgb="0.6 0.6 0.6"/>
    <material name="floor" texture="grid" texrepeat="20 20" reflectance="0.05"/>
    <material name="shell" rgba="{COLORS['shell']}" specular="0.5" shininess="0.6"/>
    <material name="joint" rgba="{COLORS['joint']}" specular="0.6" shininess="0.8"/>
    <material name="rubber" rgba="{COLORS['rubber']}" specular="0.1" shininess="0.1"/>
    <material name="accent" rgba="{COLORS['accent']}" specular="0.4" shininess="0.5"/>
{assets}
  </asset>
  <worldbody>
    <light name="key" pos="1 -1 2" dir="-0.5 0.5 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <light name="fill" pos="-1 1 1.5" dir="0.5 -0.5 -1" directional="true" castshadow="false" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" type="plane" size="5 5 0.05" material="floor" contype="0" conaffinity="0" condim="3" friction="1.0 0.005 0.0001"/>
    <camera name="front_ortho" pos="1.0 0 0.2" xyaxes="0 1 0 0 0 1" projection="orthographic" fovy="0.46"/>
    <camera name="side_ortho" pos="0 -1.0 0.2" xyaxes="1 0 0 0 0 1" projection="orthographic" fovy="0.46"/>
    <camera name="back_ortho" pos="-1.0 0 0.2" xyaxes="0 -1 0 0 0 1" projection="orthographic" fovy="0.46"/>
    <camera name="three_quarter" mode="targetbodycom" target="pelvis" pos="0.62 -0.55 0.30" fovy="42"/>
    <camera name="three_quarter_back" mode="targetbodycom" target="pelvis" pos="-0.55 -0.62 0.30" fovy="42"/>
    <camera name="track" mode="trackcom" pos="0.75 -0.75 0.36" xyaxes="0.707 0.707 0 -0.091 0.091 0.984" fovy="36"/>
{self.emit_body(root)}
  </worldbody>
  <contact>
{self.contact_pairs()}
  </contact>
  <actuator>
{chr(10).join(acts)}
  </actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
    <velocimeter name="imu_vel_privileged" site="imu"/>
{chr(10).join(sens)}
  </sensor>
{key}
</mujoco>
"""

    def nominal_q(self) -> list[float]:
        pose = {k: math.radians(v) for k, v in self.cfg["nominal_pose_deg"].items()}
        return [pose[jn.split("_", 1)[1]] for jn in self.joint_list()]
