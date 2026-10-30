"""CAD -> MuJoCo: the as-built two-leg robot (P08 pelvis + 2 x P0 leg + P30 pads) for training.

    .venv-cad/bin/python hardware/leg_p0/cad/sim_model.py [--actuator-set hw_p0] [--name hw_p0]

Unlike models/generate_model.py (parametric shells), every link here comes from the printable CAD:
  * mass, COM and inertia per link from the actual part solids, with a print model for the printed
    parts (perimeter walls solid + infill fraction inside), servo cases at their real poses, horns
    and screws;
  * joint axes and link lengths from the CAD kinematics, hips biped_hip_spacing apart (80 mm);
  * joint limits = the CAD collision-free range;
  * the contact tip is the real P07 pad sole (30 x 18 mm, pitched so it is level at stance);
  * collision capsules fitted to the CAD envelopes; visual meshes are the CAD parts (eval model only).
Writes models/<name>_8dof{,_train}.{xml,json} (same joint/geom names as the parametric models, so
the MJX env, CPU evaluation and every script work unchanged) and models/meshes/hw_p0_<spacing>mm/*.stl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import biped  # noqa: E402
import leg  # noqa: E402
import mujoco  # noqa: E402
import trimesh  # noqa: E402
import yaml  # noqa: E402
from build123d import export_stl  # noqa: E402

MODEL_DIR = ROOT / "models"
JOINTS = ["hip_yaw", "hip_roll", "hip_pitch", "knee"]
LINK_BODY = {"pelvis": "pelvis", "yaw": "hip_yaw", "roll": "hip_roll", "thigh": "thigh", "shin": "shin"}
# print model: (perimeter walls, infill) as recommended in hardware/leg_p0/README.md
PRINT = {"yaw_bracket": (4, 0.40), "roll_bracket": (4, 0.40), "thigh": (4, 0.15), "shin": (4, 0.15),
         "pad": (3, 0.30), "pelvis_biped": (4, 0.30)}
LINE_W = 0.45            # mm per perimeter
RHO = 1.24e-6            # kg/mm^3 (PLA/PETG)
SERVO_CASE_KG, HORN_KG, IDLER_KG = 0.050, 0.003, 0.002     # 55 g per servo with both horns
M3_KG, M2_KG = 0.00045, 0.00020                              # screw masses (steel pan head)
EXTRA = {  # (link, kg, point mm in left-leg world coords) -- wiring and the pelvis electronics
    "pelvis_electronics": ("pelvis", 0.020, (0.0, 0.0, 33.0)),   # IMU + bus adapter + connectors (power is tethered)
    "cable_yaw": ("yaw", 0.004, (-15.0, 30.0, -30.0)),           # one servo lead per link
    "cable_roll": ("roll", 0.004, (-20.0, 30.0, -90.0)),
    "cable_thigh": ("thigh", 0.006, (-15.0, 22.0, -160.0)),      # knee lead + extension on the zip-tie lugs
}
ROM_DEG = {"hip_yaw": (-20, 30), "hip_roll": (-14, 30), "hip_pitch": (-30, 85), "knee": (0, 110)}  # CAD collision-free
NOMINAL_DEG = {"hip_yaw": 0.0, "hip_roll": 0.0, "hip_pitch": 16.0, "knee": 32.0}                   # pad level


def to_mesh(shape, tol=0.1):
    with tempfile.NamedTemporaryFile(suffix=".stl") as f:
        export_stl(shape, f.name, tolerance=tol, angular_tolerance=0.3)
        return trimesh.load(f.name)


def printed_mass(mesh, walls, infill):
    shell = min(mesh.volume, mesh.area * walls * LINE_W)
    return RHO * (shell + infill * max(mesh.volume - shell, 0.0))


def mass_props(mesh, kg):
    """(kg, com mm, inertia about com kg*mm^2) of a solid scaled to `kg`."""
    m = mesh.copy()
    if not m.is_volume:
        m = m.convex_hull
    return kg, np.array(m.center_mass), np.array(m.moment_inertia) * kg / m.volume


def combine(parts):
    M = sum(p[0] for p in parts)
    c = sum(p[0] * p[1] for p in parts) / M
    I = np.zeros((3, 3))
    for m, com, Ic in parts:
        r = com - c
        I += Ic + m * (r @ r * np.eye(3) - np.outer(r, r))
    return M, c, I


def collect(shift_y):
    """Per-link (mass parts, visual meshes, mass log) for the LEFT leg + pelvis, left-leg world coords (mm)."""
    links = {k: dict(parts=[], meshes=[]) for k in LINK_BODY}
    log = []
    items = biped.leg_items()
    for it in items:
        if it.kind == "keepout":
            continue
        mesh = to_mesh(it.shape)
        mesh.apply_translation((0, shift_y, 0))
        if it.kind == "part":
            kg = printed_mass(mesh, *PRINT[it.name])
        elif it.kind == "servo":
            kg = SERVO_CASE_KG
        elif it.kind == "horn":
            kg = HORN_KG if it.name.startswith("horn_") else IDLER_KG
        else:   # screw heads: count the bodies (8 M3 horn screws, n M2 case screws)
            n = len(mesh.split(only_watertight=False))
            kg = n * (M3_KG if it.name.startswith("hornscrews") else M2_KG)
        links[it.link]["parts"].append(mass_props(mesh, kg))
        links[it.link]["meshes"].append(mesh)
        log.append((it.link, it.name, round(kg * 1000, 2)))
    pel = to_mesh(biped.part_pelvis_biped())          # already at its biped position (both cups)
    kg = printed_mass(pel, *PRINT["pelvis_biped"])
    # the pelvis plate is shared: give the left half-link half of it, the mirror adds the other half
    links["pelvis"]["parts"].append(mass_props(pel, kg / 2))
    links["pelvis"]["shared_mesh"] = pel
    log.append(("pelvis", "pelvis_biped (half)", round(kg / 2 * 1000, 2)))
    for name, (link, kg, p) in EXTRA.items():
        links[link]["parts"].append((kg, np.array(p) + (0, shift_y if link != "pelvis" else 0, 0), np.zeros((3, 3))))
        log.append((link, name, kg * 1000))
    return links, log


def fmt(v):
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def mirror_I(I):
    M = np.diag([1.0, -1.0, 1.0])
    return M @ I @ M


def build(name, actuator_set, spacing_mm=None):
    act = yaml.safe_load((ROOT / "configs" / "actuators.yaml").read_text())
    assign = act["assignments"][actuator_set]
    spec = {j: dict(act["catalog"][assign[j]], model=assign[j]) for j in JOINTS}
    sm = act["servo_model"]
    hy = (spacing_mm or biped.SPACING) / 2                 # yaw axis |y| in mm
    shift = hy - leg.HY
    links, log = collect(shift)
    zyaw = leg.AXES["hip_yaw"][0][2]
    origin = {"pelvis": np.array([0.0, 0.0, zyaw]), "yaw": np.array([0.0, hy, zyaw]),
              "roll": np.array([0.0, hy, leg.Z_ROLL]), "thigh": np.array([0.0, hy, leg.Z_PITCH]),
              "shin": np.array([0.0, hy, leg.Z_KNEE])}
    pad_tag = "" if (leg.P["pad"]["length"], leg.P["pad"]["width"]) == (30.0, 18.0) else f"_pad{leg.P['pad']['length']:.0f}"
    mesh_dir = MODEL_DIR / "meshes" / f"hw_p0_{2 * hy:.0f}mm{pad_tag}"   # shared by every servo variant of this geometry
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # ---- inertials (body frame = world-aligned at neutral, origin at the joint point), metres
    inert = {}
    for lk, d in links.items():
        m, c, I = combine(d["parts"])
        inert[lk] = (m, (c - origin[lk]) / 1000, I * 1e-6)
    # pelvis: left half + its mirror
    m, c, I = inert["pelvis"]
    cm = c * np.array([1, -1, 1])
    pel = combine([(m, c * 1000, I * 1e6), (m, cm * 1000, mirror_I(I) * 1e6)])
    inert["pelvis"] = (pel[0], pel[1] / 1000, pel[2] * 1e-6)

    # ---- visual meshes (eval model only)
    for lk, d in links.items():
        ms = d["meshes"] + ([d["shared_mesh"]] if "shared_mesh" in d else [])
        if not ms:
            continue
        mm = trimesh.util.concatenate(ms)
        mm.apply_translation(-origin[lk])
        mm.apply_scale(0.001)
        mm.export(mesh_dir / f"L_{lk}.stl")
        if lk != "pelvis":
            r = mm.copy()
            r.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
            r.invert()
            r.export(mesh_dir / f"R_{lk}.stl")

    # ---- collision primitives, left-leg body frames (mm), fitted to the CAD envelopes
    def rel(lk, *pts):
        return [((np.array(p, float) - origin[lk]) / 1000) for p in pts]
    cy = hy + 0.5                                           # tube centreline (leg.py: cy = HY + 0.5)
    th, sh = leg.P["thigh"], leg.P["shin"]
    r_tu = 0.85 * min(th["top"]["w"], th["top"]["d"]) / 2
    r_tl = 0.85 * min(th["bot"]["w"], th["bot"]["d"]) / 2
    r_su = 0.85 * min(sh["top"]["w"], sh["top"]["d"]) / 2
    r_sl = 0.85 * min(sh["bot"]["w"], sh["bot"]["d"]) / 2
    zk = leg.Z_KNEE
    knee_c = zk + leg.S["output_x"]                         # knee servo case centre (case runs up the thigh)
    stack = (hy - 23.6, hy + 24.6)                          # knee servo + horns + screw heads, lateral extent
    r_k = 14.0
    pitch_c = (-leg.S["output_x"], hy, leg.Z_PITCH)
    r_h = 10.5
    col = {
        "roll": [("hip_col", "capsule", r_h, rel("roll", (pitch_c[0] - 22.7 + r_h, hy, leg.Z_PITCH),
                                                   (pitch_c[0] + 22.7 - r_h, hy, leg.Z_PITCH)))],
        "thigh": [("thigh_col_u", "capsule", r_tu, rel("thigh", (0, cy, leg.Z_PITCH - 15 - r_tu), (0, cy, -160.0))),
                  ("thigh_col_l", "capsule", r_tl, rel("thigh", (0, cy, -160.0), (0, cy, knee_c + 22.7 - 5))),
                  ("knee_col", "capsule", r_k, rel("thigh", (0, stack[0] + r_k, knee_c), (0, stack[1] - r_k, knee_c)))],
        "shin": [("shin_col_u", "capsule", r_su, rel("shin", (0, cy, zk - 12.0), (0, cy, -300.0))),
                 ("shin_col_l", "capsule", r_sl, rel("shin", (0, cy, -300.0), (0, cy, leg.Z_SHIN_TIP + 6.0)))],
        "pelvis": [],
        "yaw": [],
    }
    # pad sole: box on the pitched underside of P07
    th_p = math.radians(leg.P["pad"]["stance_pitch"])
    n_down = np.array([math.sin(th_p), 0.0, -math.cos(th_p)])     # outward normal of the contact face
    u = np.array([math.cos(th_p), 0.0, math.sin(th_p)])           # along the sole (toe direction, +x)
    pad = to_mesh(next(it for it in biped.leg_items() if it.name == "pad").shape)
    pad.apply_translation((0, shift, 0))
    v = pad.vertices
    face = v[(v @ n_down) > (v @ n_down).max() - 0.05]             # vertices on the contact face
    half_t = leg.P["pad"]["thickness"] / 2
    half_l = (np.ptp(face @ u)) / 2
    half_w = (np.ptp(face[:, 1])) / 2
    c_face = u * ((face @ u).min() + half_l) + np.array([0, face[:, 1].mean(), 0]) + n_down * (face @ n_down).max()
    c_face = c_face - u * (c_face @ u) + u * ((face @ u).min() + half_l)   # keep the along-sole coordinate exact
    box_c = c_face - n_down * half_t
    tip = dict(size=np.array([half_l, half_w, half_t]) / 1000, pos=(box_c - origin["shin"]) / 1000,
               quat=np.array([math.cos(th_p / 2), 0.0, -math.sin(th_p / 2), 0.0]))
    pel_bb = links["pelvis"]["shared_mesh"].bounds
    r_p = 0.85 * min(np.ptp(pel_bb[:, 0]), np.ptp(pel_bb[:, 2])) / 2
    pc = pel_bb.mean(axis=0)
    col["pelvis"].append(("pelvis_col", "capsule", r_p, [((pc[0], -(pel_bb[1][1] - r_p), pc[2]) - origin["pelvis"]) / 1000,
                                                         ((pc[0], pel_bb[1][1] - r_p, pc[2]) - origin["pelvis"]) / 1000]))

    lim = {j: np.radians(ROM_DEG[j]) for j in JOINTS}

    def arm(j):
        s = spec[j]
        return float(s["rotor_inertia"]) * float(s["gear_ratio"]) ** 2

    def inertial_xml(lk, s):
        m, c, I = inert[lk]
        if s < 0:
            c, I = c * np.array([1, -1, 1]), mirror_I(I)
        return (f'<inertial pos="{fmt(c)}" mass="{m:.6g}" fullinertia="{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} '
                f'{I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"/>')

    def geoms(lk, side, s, visual):
        out = []
        if visual and (mesh_dir / f"{side}_{lk}.stl").exists():
            mat = "shell" if lk in ("thigh", "shin") else "joint"
            out.append(f'<geom name="{side}_{lk}_vis" class="vis" type="mesh" mesh="{side}_{lk}" material="{mat}"/>')
        for gname, gtype, r, (a, b) in col[lk]:
            a, b = a.copy(), b.copy()
            if s < 0:
                a[1], b[1] = -a[1], -b[1]
            out.append(f'<geom name="{side}_{gname}" class="col" type="{gtype}" size="{r/1000:.6g}" fromto="{fmt(np.r_[a, b])}"/>')
        if lk == "shin":
            p = tip["pos"].copy()
            if s < 0:
                p[1] = -p[1]
            fr = [1.0, 0.004, 0.0004]
            out.append(f'<geom name="{side}_tip" class="foot" type="box" size="{fmt(tip["size"])}" pos="{fmt(p)}" '
                       f'quat="{fmt(tip["quat"])}" mass="0" friction="{fmt(fr)}"/>')
            out.append(f'<site name="{side}_touch" type="sphere" pos="{fmt(p)}" size="0.02" rgba="0 1 0 0"/>')
        return out

    def leg_xml(side, visual):
        s = 1.0 if side == "L" else -1.0
        axes = {"hip_yaw": (0, 0, s), "hip_roll": (s, 0, 0), "hip_pitch": (0, -1, 0), "knee": (0, 1, 0)}
        chain = [("yaw", "hip_yaw"), ("roll", "hip_roll"), ("thigh", "hip_pitch"), ("shin", "knee")]
        xml, prev = [], origin["pelvis"]
        for depth, (lk, j) in enumerate(chain):
            p = (origin[lk] - prev) / 1000
            if s < 0:
                p[1] = -p[1]
            ind = "  " * (3 + depth)
            body = "thigh" if lk == "thigh" else ("shin" if lk == "shin" else LINK_BODY[lk])
            xml.append(f'{ind}<body name="{side}_{body}" pos="{fmt(p)}">')
            xml.append(f'{ind}  <joint name="{side}_{j}" axis="{fmt(axes[j])}" range="{fmt(lim[j])}" armature="{arm(j):.6g}"/>')
            xml.append(f"{ind}  {inertial_xml(lk, s)}")
            xml += [f"{ind}  {g}" for g in geoms(lk, side, s, visual)]
            prev = origin[lk]
        for depth in reversed(range(4)):
            xml.append("  " * (3 + depth) + "</body>")
        return "\n".join(xml)

    joint_names = [f"{sd}_{j}" for sd in "LR" for j in JOINTS]

    def contact_pairs(lean):
        tip_fr = "1 1 0.004 0.0004 0.0004"
        lines = []
        body = ["hip_col", "thigh_col_u", "thigh_col_l", "knee_col", "shin_col_u", "shin_col_l"]
        for sd in "LR":
            lines.append(f'    <pair geom1="floor" geom2="{sd}_tip" condim="3" friction="{tip_fr}" solref="0.004 1" solimp="0.9 0.97 0.001"/>')
            if not lean:
                for b in body:
                    lines.append(f'    <pair geom1="floor" geom2="{sd}_{b}" condim="3" friction="0.8 0.8 0.005 0.0001 0.0001"/>')
        if not lean:
            lines.append('    <pair geom1="floor" geom2="pelvis_col" condim="3" friction="0.8 0.8 0.005 0.0001 0.0001"/>')
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

    def model_xml(visual, lean, key=""):
        assets = ""
        if visual:
            assets = "\n".join(f'    <mesh name="{p.stem}" file="{p.relative_to(MODEL_DIR)}"/>'   # relative to the XML
                               for p in sorted(mesh_dir.glob("*.stl")))
        pel_geoms = [f'<geom name="pelvis_vis" class="vis" type="mesh" mesh="L_pelvis" material="joint"/>'] if visual else []
        for gname, gtype, r, (a, b) in col["pelvis"]:
            pel_geoms.append(f'<geom name="{gname}" class="col" type="{gtype}" size="{r/1000:.6g}" fromto="{fmt(np.r_[a, b])}"/>')
        acts, sens = [], []
        for jn in joint_names:
            ts = spec[jn.split("_", 1)[1]]["stall_torque"]
            acts.append(f'    <motor name="{jn}" joint="{jn}" ctrlrange="{-ts:.4g} {ts:.4g}"/>')
            sens.append(f'    <jointpos name="{jn}_pos" joint="{jn}"/>')
            sens.append(f'    <jointvel name="{jn}_vel" joint="{jn}"/>')
            sens.append(f'    <actuatorfrc name="{jn}_frc" actuator="{jn}"/>')
        return f"""<mujoco model="{name}_8dof">
  <!-- GENERATED by hardware/leg_p0/cad/sim_model.py from the P0 CAD + configs/actuators.yaml. Do not edit. -->
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
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
    <default class="vis"><geom contype="0" conaffinity="0" group="0" mass="0"/></default>
    <default class="col"><geom contype="0" conaffinity="0" group="3" mass="0" rgba="0.2 0.6 1.0 0.35" condim="3" friction="0.6 0.005 0.0001"/></default>
    <default class="foot"><geom contype="0" conaffinity="0" group="0" material="rubber" condim="3" solref="0.004 1" solimp="0.9 0.97 0.001"/></default>
  </default>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="0.86 0.88 0.91" width="256" height="1536"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.80 0.80 0.80" rgb2="0.70 0.70 0.70" width="512" height="512" mark="edge" markrgb="0.6 0.6 0.6"/>
    <material name="floor" texture="grid" texrepeat="20 20" reflectance="0.05"/>
    <material name="shell" rgba="0.92 0.90 0.85 1" specular="0.5" shininess="0.6"/>
    <material name="joint" rgba="0.19 0.20 0.22 1" specular="0.6" shininess="0.8"/>
    <material name="rubber" rgba="0.07 0.07 0.08 1" specular="0.1" shininess="0.1"/>
{assets}
  </asset>
  <worldbody>
    <light name="key" pos="1 -1 2" dir="-0.5 0.5 -1" directional="true" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <light name="fill" pos="-1 1 1.5" dir="0.5 -0.5 -1" directional="true" castshadow="false" diffuse="0.25 0.25 0.25"/>
    <geom name="floor" type="plane" size="5 5 0.05" material="floor" contype="0" conaffinity="0" condim="3" friction="1.0 0.005 0.0001"/>
    <camera name="three_quarter" mode="targetbodycom" target="pelvis" pos="0.62 -0.55 0.30" fovy="42"/>
    <camera name="track" mode="trackcom" pos="0.75 -0.75 0.36" xyaxes="0.707 0.707 0 -0.091 0.091 0.984" fovy="36"/>
    <body name="pelvis" pos="0 0 0.5">
      <freejoint name="root"/>
      {inertial_xml("pelvis", 1)}
      {chr(10).join("      " + g for g in pel_geoms)}
      <site name="imu" pos="0 0 0.03" size="0.004" rgba="0 1 0 1"/>
{leg_xml("L", visual)}
{leg_xml("R", visual)}
    </body>
  </worldbody>
  <contact>
{contact_pairs(lean)}
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

    q_nom = [math.radians(NOMINAL_DEG[jn.split("_", 1)[1]]) for jn in joint_names]
    m = mujoco.MjModel.from_xml_string(model_xml(False, True))
    d = mujoco.MjData(m)
    d.qpos[7:] = q_nom
    mujoco.mj_kinematics(m, d)
    lows = []
    for sd in "LR":
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{sd}_tip")
        R = d.geom_xmat[g].reshape(3, 3)
        lows.append(d.geom_xpos[g][2] - np.abs(m.geom_size[g] * R[2, :]).sum())
    z = 0.5 - min(lows)
    key = f'  <keyframe>\n    <key name="stand" qpos="{fmt([0, 0, z, 1, 0, 0, 0] + q_nom)}"/>\n  </keyframe>'
    out = {}
    for tag, visual, lean in (("", True, False), ("_train", False, True)):
        xml = model_xml(visual, lean, key)
        nm = f"{name}_8dof{tag}"
        (MODEL_DIR / f"{nm}.xml").write_text(xml)
        mm = mujoco.MjModel.from_xml_path(str(MODEL_DIR / f"{nm}.xml"))   # validate (mesh paths are relative to it)
        meta = {"name": nm, "joints": joint_names, "nominal_q": q_nom, "stand_base_z": z,
                "cfg": {"variant": name, "foot": f"P{leg.P['pad']['length']:.0f}", "ankle": False, "actuator_set": actuator_set,
                        "contact_set": "train" if lean else "full", "hip_spacing": 2 * hy / 1000,
                        "joint_limits_deg": ROM_DEG, "nominal_pose_deg": NOMINAL_DEG,
                        "foot_P30": {"friction": [1.0, 0.004, 0.0004]}},
                "servos": spec, "mass_log": log, "total_mass": float(sum(mm.body_mass)),
                "link_mass": {lk: inert[lk][0] for lk in inert}, "source": "hardware/leg_p0/cad/sim_model.py"}
        (MODEL_DIR / f"{nm}.json").write_text(json.dumps(meta, indent=1, default=float))
        out[nm] = meta
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--actuator-set", default="hw_p0")
    ap.add_argument("--name", default="hw_p0")
    ap.add_argument("--spacing", type=float, help="hip spacing mm (default: params.yaml pelvis.biped_hip_spacing)")
    ap.add_argument("--pad", type=float, nargs=2, metavar=("LEN", "WID"), help="pad size mm (default 30 x 18)")
    a = ap.parse_args()
    if a.pad:   # P07 is rebuilt from leg.P at call time; the socket/stem stay the same
        leg.P["pad"]["length"], leg.P["pad"]["width"] = a.pad
    res = build(a.name, a.actuator_set, a.spacing)
    for nm, meta in res.items():
        print(f"{nm}: total {meta['total_mass']*1000:.1f} g, stand_z {meta['stand_base_z']:.4f} m, links (g) "
              + ", ".join(f"{k} {v*1000:.1f}" for k, v in meta["link_mass"].items()))
