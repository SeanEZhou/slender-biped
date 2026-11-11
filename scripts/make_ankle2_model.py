"""Simulation models of the ankle-roll leg as designed (hardware/ankle_p1, v2, 2026-11-09), with a choice of pads.

    .venv/bin/python scripts/make_ankle2_model.py                 # every pad design
    .venv/bin/python scripts/make_ankle2_model.py b150 d70        # some
    .venv/bin/python scripts/make_ankle2_model.py --list

Writes models/hw_ank2_<pad>_legs_10dof{,_train}.{xml,json} (legs + pelvis, as tested on the robot); with --full
(torso and head on) models/hw_ank<2|3>_<pad>_10dof from models/hw_pad_board150w_ankle_10dof.

Built from models/hw_pad_board150w_legs_ankle_10dof (the design-study model: same hips, thighs, servos, contact
settings) with the v2 lower leg:

* shin (A01): straight, its mass and inertia from the CAD (printed shin + knee and ankle horns and screws);
  visual mesh models/meshes/ankle2/<S>_shin.stl (hardware/ankle_p1/cad/build_ankle.py simmesh)
* ankle roll joint 111.45 mm below the knee axis on the shin centre line, axis along x (mirrored on the right leg),
  range +-0.6 rad (the CAD is collision free to +-45 deg)
* foot: the ankle servo (55 g) standing in its socket, the socket + rib (14 g), the pad (printed, 0.75 g/cm^3), screws
  and cable (4 g); pad sole 38.55 mm below the ankle axis, flat with the leg straight, centred under the knee
* rest pose: all joints 0 (leg straight), knee range -1.2 .. 1.92 rad (the knee may bend "forward": the CAD is
  collision free to -90 deg)

Pad designs: geoms `<S>_tip`, `<S>_tip_<k>` (floor contacts, the env's "foot"), collision stand-ins `<S>_padcol_<k>`
(left vs right foot, foot vs other shin), plus the servo socket as one more stand-in.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
BASE = "hw_pad_board150w_legs_ankle_10dof"
_spec = importlib.util.spec_from_file_location("make_pads", ROOT / "scripts" / "make_pads.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

ANKLE_Z = -0.11145            # ankle axis below the knee (shin frame)
SOLE_Z = -0.150               # sole below the knee, leg straight
Y0 = 0.0005                   # shin centre line (shin frame)
RHO = 750.0
SHIN_MASS = 0.0497            # A01 32.5 g + knee horns/screws 8.6 g + ankle horns/screws 8.6 g
SHIN_COM = (0.0, Y0, -0.062)
SERVO = (0.055, (0.0, 0.0, -0.0125), (0.0248, 0.0318, 0.0454))     # kg, centre in the foot frame, box (x, y, z)
SOCKET = (0.014, (0.0, 0.0, -0.030))
EXTRA = (0.004, (0.0, 0.0, -0.020))
T = 0.004


def designs():
    d = {}
    d["b150"] = (mp.snowboard(0.150, 0.050, kick=0.015), 0.150 * 0.050, "snowboard 150 x 50 mm (as built)", [mp.bar(0.150, 0.050)])
    d["b100"] = (mp.snowboard(0.100, 0.050, kick=0.012), 0.100 * 0.050, "snowboard 100 x 50 mm", [mp.bar(0.100, 0.050)])
    d["r80"] = ([mp.box(0, 0, 0.080, 0.045, T)], 0.080 * 0.045, "flat rectangle 80 x 45 mm", [mp.bar(0.080, 0.045)])
    d["r60"] = ([mp.box(0, 0, 0.060, 0.040, T)], 0.060 * 0.040, "flat rectangle 60 x 40 mm", [mp.bar(0.060, 0.040)])
    for D in (55, 70):
        R = D / 2000
        d[f"d{D}"] = ([mp.disc(0, 0, R, T)], math.pi * R * R, f"flat round pad, {D} mm across", [mp.ball(R)])
    # foot-shaped, 96 mm long: heel disc (36 mm) behind, a 36 mm wide arch, a 46 mm forefoot disc ahead
    foot = [mp.box(0.0025, 0, 0.055, 0.036, T), mp.disc(-0.025, 0, 0.018, T), mp.disc(0.030, 0, 0.023, T)]
    d["foot"] = (foot, 0.055 * 0.036 + math.pi * (0.018 ** 2 + 0.023 ** 2) / 2, "foot-shaped 96 x 46 mm (heel 36, forefoot 46)",
                 [mp.bar(0.096, 0.040)])
    return d


def f6(v):
    return " ".join(f"{x:.6g}" for x in v)


def box_I(m, dims):
    a, b, c = dims
    return np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])


def foot_inertial(parts):
    """(mass, com, fullinertia) of the foot body: servo box, socket and extras as points, the pad from its parts."""
    pts = [(SERVO[0], np.array(SERVO[1]), box_I(SERVO[0], SERVO[2])),
           (SOCKET[0], np.array(SOCKET[1]), box_I(SOCKET[0], (0.042, 0.031, 0.020))),
           (EXTRA[0], np.array(EXTRA[1]), np.zeros((3, 3)))]
    sole = np.array([0, 0, SOLE_Z - ANKLE_Z])
    for p in parts:
        m = RHO * mp._vol(p)
        c = sole + (np.array(p["pos"]) if "pos" in p else (np.array(p["fromto"][:3]) + np.array(p["fromto"][3:])) / 2)
        pts.append((m, c, np.zeros((3, 3))))
    M = sum(m for m, _, _ in pts)
    com = sum(m * c for m, c, _ in pts) / M
    J = np.zeros((3, 3))
    for m, c, I in pts:
        d = c - com
        J += I + m * (d @ d * np.eye(3) - np.outer(d, d))
    # the pad's own inertia: a plate spanning its footprint
    return M, com, [J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]]


def geom_line(name, p, cls, fr, sgn):
    sole = np.array([0, 0, SOLE_Z - ANKLE_Z])
    if p["type"] == "capsule":
        f = np.array(p["fromto"])
        geo = f'type="capsule" size="{f6(p["size"])}" fromto="{f6([*(sole + f[:3]), *(sole + f[3:])])}"'
    else:
        q = [math.cos(p["pitch"] / 2), 0, math.sin(p["pitch"] / 2), 0]
        geo = f'type="{p["type"]}" size="{f6(p["size"])}" pos="{f6(sole + np.array(p["pos"]))}" quat="{f6(q)}"'
    return f'<geom name="{name}" class="{cls}" {geo} mass="0" friction="{fr}"/>'


FOOT_QUAT = None     # ank3: the foot frame pitched -16 deg (toe up with the leg straight, flat in the stance pose)


def convert(xml, parts, prox, train):
    out = xml
    fr = "1 0.004 0.0004"
    for S, sgn in (("L", 1), ("R", -1)):
        out = re.sub(rf'\s*<body name="{S}_ankle_servo"[^>]*>\s*<inertial[^>]*/>\s*</body>', "", out)
        # shin inertial
        m = re.search(rf'(<body name="{S}_shin"[^\n]*\n[^\n]*\n\s*)<inertial [^>]*/>', out)
        L = 0.12
        I = [SHIN_MASS * L * L / 12, SHIN_MASS * L * L / 12, SHIN_MASS * 0.02 ** 2 / 6, 0, 0, 0]
        com = np.array(SHIN_COM) * np.array([1, sgn, 1])
        out = out.replace(m.group(0), f'{m.group(1)}<inertial pos="{f6(com)}" mass="{SHIN_MASS}" fullinertia="{f6(I)}"/>')
        out = re.sub(rf'(<joint name="{S}_knee" [^>]*range=")[^"]*"', r'\g<1>-1.2 1.91986"', out)
        # foot body
        i0 = out.index(f'<body name="{S}_foot"')
        i1 = out.index("</body>", i0) + len("</body>")
        ind = "                "
        M, c, fi = foot_inertial(parts)
        fq = f' quat="{FOOT_QUAT}"' if FOOT_QUAT else ""
        lines = [f'<body name="{S}_foot" pos="0 {sgn * Y0:.4f} {ANKLE_Z}"{fq}>',
                 f'  <joint name="{S}_ankle_roll" axis="{sgn} 0 0" range="-0.6 0.6" armature="0.00297562"/>',
                 f'  <inertial pos="{f6(c * np.array([1, sgn, 1]))}" mass="{M:.6g}" fullinertia="{f6(fi)}"/>']
        if not train:
            lines += [f'  <geom name="{S}_foot_vis" class="vis" type="mesh" mesh="{S}_foot" material="joint"/>',
                      f'  <geom name="{S}_socket_vis" class="vis" type="box" size="0.0213 0.0158 0.0115" pos="0 0 {SOLE_Z - ANKLE_Z + T + 0.0115 - 0.001:.4f}" material="rubber"/>']
        for k, p in enumerate(parts):
            lines.append("  " + geom_line(f"{S}_tip" if k == 0 else f"{S}_tip_{k}", p, "foot", fr, sgn))
        sock = dict(type="capsule", fromto=[0, 0, 0.012, 0, 0, 0.040], size=[0.016], pitch=0.0)
        for k, p in enumerate(prox + [sock]):
            lines.append("  " + geom_line(f"{S}_padcol_{k}", p, "col", fr, sgn))
        lines.append(f'  <site name="{S}_touch" type="sphere" pos="0 0 {SOLE_Z - ANKLE_Z:.5f}" size="0.02" rgba="0 1 0 0"/>')
        lines.append("</body>")
        out = out[:i0] + ("\n" + ind).join(lines) + out[i1:]
    # meshes
    if not train:
        for S in "LR":
            out = out.replace(f'file="meshes/hw_p0_80mm/{S}_shin.stl"', f'file="meshes/ankle2/{S}_shin.stl"')
        out = out.replace('<mesh name="L_shin"', '<mesh name="L_foot" file="meshes/ankle2/L_foot.stl"/>\n    <mesh name="R_foot" file="meshes/ankle2/R_foot.stl"/>\n    <mesh name="L_shin"', 1)
    # contact pairs: drop every tip / padcol pair, add the new set
    pair_lines = re.findall(r'^\s*<pair [^\n]*\n', out, re.M)
    floor_tip = next(p for p in pair_lines if 'geom1="floor" geom2="L_tip"' in p)
    leg_leg = next(p for p in pair_lines if 'geom1="L_thigh_col_l" geom2="R_thigh_col_l"' in p)
    ind = re.match(r"^(\s*)", floor_tip).group(1)
    for p in pair_lines:
        if "_tip" in p or "padcol" in p:
            out = out.replace(p, "", 1)
    n, npx = len(parts), len(prox) + 1
    ft = floor_tip.strip()[len('<pair geom1="floor" geom2="L_tip" '):]
    ll = leg_leg.strip()[len('<pair geom1="L_thigh_col_l" geom2="R_thigh_col_l" '):]
    new = [f'{ind}<pair geom1="floor" geom2="{S}_tip{"" if k == 0 else f"_{k}"}" {ft}\n' for S in "LR" for k in range(n)]
    new += [f'{ind}<pair geom1="L_padcol_{a}" geom2="R_padcol_{b}" {ll}\n' for a in range(npx) for b in range(npx)]
    new += [f'{ind}<pair geom1="L_padcol_{a}" geom2="R_shin_col_l" {ll}\n' for a in range(npx)]
    new += [f'{ind}<pair geom1="L_shin_col_l" geom2="R_padcol_{b}" {ll}\n' for b in range(npx)]
    first = re.search(r'^\s*<pair ', out, re.M)
    out = out[:first.start()] + "".join(new) + out[first.start():]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--full", action="store_true", help="torso and head on (base hw_pad_board150w_ankle_10dof: their mass "
                                                       "lumped into the pelvis); writes hw_ank<2|3>_<pad>_10dof")
    ap.add_argument("--bent", action="store_true", help="ank3: rest pose hip 16 / knee 32 deg, the ankle fork angled 16 deg on "
                                                       "the straight shin so the roll axis is level and the board flat in it")
    a = ap.parse_args()
    D = designs()
    global FOOT_QUAT, BASE
    if a.full:
        BASE = "hw_pad_board150w_ankle_10dof"
    q_nom = [0.0] * 10
    if a.bent:
        FOOT_QUAT = "0.990268 0 -0.139173 0"
        q_nom = [0.0, 0.0, 0.2792526803190927, 0.5585053606381855, 0.0] * 2
    if a.list:
        for k, v in D.items():
            print(f"{k:6s} {v[2]}  area {v[1] * 1e4:.0f} cm^2")
        return
    import mujoco
    sys.path.insert(0, str(ROOT))
    from slender_biped import feet
    for name in a.names or list(D):
        parts, area, desc, prox = D[name]
        mname = f"hw_ank{3 if a.bent else 2}_{name}_{'' if a.full else 'legs_'}10dof"
        for suf, train in (("", False), ("_train", True)):
            xml = (MODELS / f"{BASE}{suf}.xml").read_text()
            (MODELS / f"{mname}{suf}.xml").write_text(convert(xml, parts, prox, train))
        m = mujoco.MjModel.from_xml_path(str(MODELS / f"{mname}.xml"))
        d = mujoco.MjData(m)
        d.qpos[:] = 0; d.qpos[2] = 1.0; d.qpos[3] = 1.0; d.qpos[7:] = q_nom
        mujoco.mj_kinematics(m, d)
        gid, typ, size = feet.foot_geoms(m)
        low = feet.lowest(np, d.geom_xpos[gid], d.geom_xmat[gid].reshape(2, -1, 3, 3), typ, size)
        z0 = 1.0 - float(low.min())
        key = "0 0 %.6f 1 0 0 0 " % z0 + " ".join(f"{v:.6g}" for v in q_nom)
        for suf in ("", "_train"):
            f = MODELS / f"{mname}{suf}.xml"
            f.write_text(re.sub(r'<key name="stand" qpos="[^"]*"/>', f'<key name="stand" qpos="{key}"/>', f.read_text()))
        meta = json.loads((MODELS / f"{BASE}.json").read_text())
        meta.update(name=mname, nominal_q=q_nom, stand_base_z=z0, pad_area=area, pad_desc=desc,
                    total_mass=float(sum(m.body_mass)),
                    note=f"ankle roll as designed (hardware/ankle_p1 v2, scripts/make_ankle2_model.py): foot holds the "
                         f"servo, straight shin, rest pose straight; pad {desc}")
        meta["cfg"] = dict(meta["cfg"], variant=mname.replace("_legs_10dof", "").replace("_10dof", ""), foot=name)
        if a.full:
            meta["note"] += "; torso and head on (their mass in the pelvis)"
        if a.bent:
            meta["note"] = meta["note"].replace("rest pose straight", "rest pose hip 16 / knee 32 deg, ankle fork angled 16 deg")
        (MODELS / f"{mname}.json").write_text(json.dumps(meta, indent=1))
        mujoco.MjModel.from_xml_path(str(MODELS / f"{mname}_train.xml"))
        fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "L_foot")
        print(f"{mname}: {desc}; stand z {z0:.4f}, mass {sum(m.body_mass):.3f} kg, foot {m.body_mass[fb] * 1000:.0f} g")


if __name__ == "__main__":
    main()
