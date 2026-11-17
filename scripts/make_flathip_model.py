"""Round 20 (2026-11-17): the flat-hip leg. The hip servos lie horizontally in a row along the hip plate instead of
stacking down the leg, so the thigh and shin can take the whole leg length.

Today (hw_ank3_b150_m2_10dof, hip plate to sole, straight leg): hip yaw servo 60 mm, hip roll servo 34 mm, thigh
138 mm, shin 111 mm, foot 37 mm. The yaw and roll servos take ~25 % of the length and the thigh + shin ~65 %.

Flat hip, front to back along the hip plate: pitch / roll / yaw (the kinematic chain stays yaw -> roll -> pitch, so
the joint order, the policy's inputs and outputs, the gait tables and the robot software are unchanged):
  * hip pitch servo: shaft sideways, body lying fore-aft (45 mm), its output (the pitch axis) at the front, under the
    hip plate (axis --zp below it);
  * hip roll servo: shaft fore-aft, body lying sideways behind the pitch servo; the roll axis is a fore-aft line at
    the pitch axis' height (it now passes through the hip instead of 34 mm above it);
  * hip yaw servo: shaft vertical, fixed to the pelvis behind the roll servo (axis --yaw-x behind the pitch axis); it
    sticks out behind the hip plate and up into a round butt plate (--butt kg).
  * thigh and shin: the freed length, same overall height (hip plate to sole, straight leg) and the same 55 : 45
    thigh : shin ratio unless --thigh / --shin say otherwise.
Masses: the servo bodies keep their (re-massed) masses at their new places; the thigh and shin shells grow with
length (--shell-frac of the thigh, --shin-shell-frac of the shin is length-proportional). The yaw servos (2 x
--yaw-m) move from under the plate to behind it. --rebalance moves the torso + head mass forward so the whole robot's
centre of mass stays where it is today (the torso position is free in the design); the json says by how much.

    .venv/bin/python scripts/make_flathip_model.py [--out hw_flathip_10dof] [--rebalance] [--thigh 0.178 --shin 0.144]
"""
import argparse, json, math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco

M = Path(__file__).resolve().parents[1] / "models"
SRC = "hw_ank3_b150_m2_10dof"


def f3(v):
    return " ".join(f"{x:.6g}" for x in v)


def box_I(m, a, b, c):
    """Principal inertia of a solid box with full edges a (x), b (y), c (z)."""
    return np.array([m * (b * b + c * c), m * (a * a + c * c), m * (a * a + b * b)]) / 12.0


def fullI(d):
    return f3([d[0], d[1], d[2], 0, 0, 0])


def shift_I(I, m, c):
    c = np.asarray(c, float)
    return I + m * (c @ c * np.eye(3) - np.outer(c, c))


def build(a, src_xml, train):
    tree = ET.ElementTree(ET.fromstring(src_xml))
    root = tree.getroot()
    root.set("model", a.out)
    wb = root.find("worldbody")
    pelvis = wb.find("body[@name='pelvis']")
    zp, yx = -a.zp, -a.yaw_x
    st, ss = a.thigh / 0.138, a.shin / 0.11145
    info = {}
    # ---- pelvis: yaw servos move from under the plate to behind it, plus the butt plate
    pin = pelvis.find("inertial")
    c0 = np.array(pin.get("pos").split(), float); m0 = float(pin.get("mass"))
    I0 = np.array(pin.get("fullinertia").split(), float)
    J = np.array([[I0[0], I0[3], I0[4]], [I0[3], I0[1], I0[5]], [I0[4], I0[5], I0[2]]])
    Jo = shift_I(J, m0, c0)                               # about the pelvis origin
    moves = []
    for s in (1, -1):
        old = np.array([0.0, 0.04 * s, -0.02]); new = np.array([yx - 0.008, 0.04 * s, 0.006])
        Ib = np.diag(box_I(a.yaw_m, 0.0452, 0.0247, 0.035))
        Jo = Jo - shift_I(Ib, a.yaw_m, old) + shift_I(Ib, a.yaw_m, new)
        moves.append((a.yaw_m, old, new))
    mb, cb = a.butt, np.array([yx + 0.005, 0.0, 0.004])  # round butt plate: a 2 mm shell, ~65 mm radius
    Ib = np.eye(3) * (2.0 / 3.0) * mb * 0.065 ** 2
    Jo = Jo + shift_I(Ib, mb, cb)
    m1 = m0 + mb
    c1 = (m0 * c0 + sum(m * (n - o) for m, o, n in moves) + mb * cb) / m1
    pelvis.find("inertial").set("pos", f3(c1)); pelvis.find("inertial").set("mass", f"{m1:.6g}")
    Jc = Jo - m1 * (c1 @ c1 * np.eye(3) - np.outer(c1, c1))
    pelvis.find("inertial").set("fullinertia", f3([Jc[0, 0], Jc[1, 1], Jc[2, 2], Jc[0, 1], Jc[0, 2], Jc[1, 2]]))
    info["pelvis_com_shift_mm"] = (1000 * (c1 - c0)).round(1).tolist()
    info["pelvis_mass_added"] = mb
    for g in list(pelvis.findall("geom")):
        if g.get("type") == "mesh":
            pelvis.remove(g)
    if not train:
        ET.SubElement(pelvis, "geom", {"name": "plate_vis", "class": "vis", "type": "box", "size": "0.035 0.062 0.004",
                                       "pos": "-0.02 0 0.002", "material": "shell"})
        ET.SubElement(pelvis, "geom", {"name": "butt_vis", "class": "vis", "type": "ellipsoid", "size": "0.06 0.07 0.04",
                                       "pos": f3([yx - 0.004, 0, 0.012]), "material": "shell", "rgba": "0.92 0.90 0.85 0.45"})
        ET.SubElement(pelvis, "geom", {"name": "torso_vis", "class": "vis", "type": "box", "size": "0.035 0.055 0.07",
                                       "pos": f3([c0[0], 0, 0.11]), "material": "shell"})
        for s, nm in ((1, "L"), (-1, "R")):
            ET.SubElement(pelvis, "geom", {"name": f"{nm}_yawservo_vis", "class": "vis", "type": "box",
                                           "size": "0.0226 0.0124 0.0175", "pos": f3([yx - 0.008, 0.04 * s, 0.006]),
                                           "material": "joint"})
    # ---- legs
    for side, s in (("L", 1), ("R", -1)):
        yaw = pelvis.find(f"body[@name='{side}_hip_yaw']")
        yaw.set("pos", f3([yx, 0.04 * s, zp]))
        # yaw body: the roll servo (shaft fore-aft, lying sideways), centred between the yaw axis and the pitch servo
        m_y = float(yaw.find("inertial").get("mass"))
        cr = np.array([-yx * 0.5 + 0.004, 0, 0])          # ~ -0.0385 world x for yaw_x 0.085
        yaw.find("inertial").set("pos", f3(cr)); yaw.find("inertial").set("fullinertia", fullI(box_I(m_y, 0.035, 0.0452, 0.0247)))
        for g in list(yaw.findall("geom")):
            yaw.remove(g)
        if not train:
            ET.SubElement(yaw, "geom", {"name": f"{side}_rollservo_vis", "class": "vis", "type": "box",
                                        "size": "0.0175 0.0226 0.0124", "pos": f3(cr), "material": "joint"})
        roll = yaw.find(f"body[@name='{side}_hip_roll']")
        roll.set("pos", f3([-yx, 0, 0]))
        m_r = float(roll.find("inertial").get("mass"))
        cp = np.array([-0.0145, 0, 0])                     # pitch servo body, its output at the front end
        roll.find("inertial").set("pos", f3(cp)); roll.find("inertial").set("fullinertia", fullI(box_I(m_r, 0.0452, 0.035, 0.0247)))
        for g in list(roll.findall("geom")):
            roll.remove(g)
        ET.SubElement(roll, "geom", {"name": f"{side}_hip_col", "class": "col", "type": "capsule", "size": "0.0125",
                                     "fromto": f3([yx - 0.02, 0, 0, 0.012, 0, 0])})
        if not train:
            ET.SubElement(roll, "geom", {"name": f"{side}_pitchservo_vis", "class": "vis", "type": "box",
                                         "size": "0.0226 0.0175 0.0124", "pos": f3(cp), "material": "joint"})
        th = roll.find(f"body[@name='{side}_thigh']")
        th.set("pos", "0 0 0")
        ti = th.find("inertial"); m_t = float(ti.get("mass")); kt = a.shell_frac * (st - 1) + 1
        ct = np.array(ti.get("pos").split(), float); It = np.array(ti.get("fullinertia").split(), float)
        ti.set("pos", f3([ct[0], ct[1], ct[2] * st])); ti.set("mass", f"{m_t * kt:.6g}")
        ti.set("fullinertia", f3([It[0] * kt * st * st, It[1] * kt * st * st, It[2] * kt, It[3] * kt, It[4] * kt * st, It[5] * kt * st]))
        for g in list(th.findall("geom")):
            if g.get("type") == "mesh":
                th.remove(g); continue
            if g.get("fromto"):
                v = np.array(g.get("fromto").split(), float); v[2] *= st; v[5] *= st; g.set("fromto", f3(v))
        if not train:
            ET.SubElement(th, "geom", {"name": f"{side}_thigh_vis", "class": "vis", "type": "capsule", "size": "0.014",
                                       "fromto": f3([0, 0, -0.012, 0, 0, -a.thigh + 0.012]), "material": "shell"})
            ET.SubElement(th, "geom", {"name": f"{side}_kneeservo_vis", "class": "vis", "type": "box",
                                       "size": "0.0124 0.0175 0.0226", "pos": f3([0, 0, -a.thigh + 0.03]), "material": "joint"})
        sh = th.find(f"body[@name='{side}_shin']")
        sh.set("pos", f3([0, 0, -a.thigh]))
        si = sh.find("inertial"); m_s = float(si.get("mass")); ks = a.shin_shell_frac * (ss - 1) + 1
        cs = np.array(si.get("pos").split(), float); Is = np.array(si.get("fullinertia").split(), float)
        si.set("pos", f3([cs[0], cs[1], cs[2] * ss])); si.set("mass", f"{m_s * ks:.6g}")
        si.set("fullinertia", f3([Is[0] * ks * ss * ss, Is[1] * ks * ss * ss, Is[2] * ks, Is[3] * ks, Is[4] * ks * ss, Is[5] * ks * ss]))
        for g in list(sh.findall("geom")):
            if g.get("type") == "mesh":
                sh.remove(g); continue
            if g.get("fromto"):
                v = np.array(g.get("fromto").split(), float); v[2] *= ss; v[5] *= ss; g.set("fromto", f3(v))
        if not train:
            ET.SubElement(sh, "geom", {"name": f"{side}_shin_vis", "class": "vis", "type": "capsule", "size": "0.011",
                                       "fromto": f3([0, 0, -0.008, 0, 0, -a.shin + 0.008]), "material": "shell"})
        ft = sh.find(f"body[@name='{side}_foot']")
        p = np.array(ft.get("pos").split(), float); p[2] = -a.shin; ft.set("pos", f3(p))
        for g in list(ft.findall("geom")):
            if g.get("type") == "mesh":
                ft.remove(g)
        info[f"{side}_thigh_mass"] = m_t * kt; info[f"{side}_shin_mass"] = m_s * ks
    # the hip servos' new collision capsules and the yaw body's removed geoms: drop contact pairs naming missing geoms
    names = {g.get("name") for g in root.iter("geom")}
    con = root.find("contact")
    for pr in list(con):
        if pr.get("geom1") not in names or pr.get("geom2") not in names:
            con.remove(pr)
    return tree, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw_flathip_10dof")
    ap.add_argument("--zp", type=float, default=0.022, help="pitch (and roll) axis below the hip plate, m")
    ap.add_argument("--yaw-x", type=float, default=0.085, help="yaw axis behind the pitch axis, m")
    ap.add_argument("--thigh", type=float, default=None)
    ap.add_argument("--shin", type=float, default=None)
    ap.add_argument("--yaw-m", type=float, default=0.045, help="kg per yaw servo + bracket moved behind the plate")
    ap.add_argument("--butt", type=float, default=0.025, help="kg, the round butt plate")
    ap.add_argument("--shell-frac", type=float, default=0.35)
    ap.add_argument("--shin-shell-frac", type=float, default=0.7)
    ap.add_argument("--rebalance", action="store_true")
    a = ap.parse_args()
    old_len = 0.0604 + 0.0342 + 0.138 + 0.11145          # hip plate -> ankle body today
    free = old_len - a.zp
    if a.thigh is None and a.shin is None:
        a.thigh = round(free * 0.138 / (0.138 + 0.11145), 4); a.shin = round(free - a.thigh, 4)
    elif a.shin is None:
        a.shin = round(free - a.thigh, 4)
    src_meta = json.loads((M / f"{SRC}.json").read_text())
    for suffix in (".xml", "_train.xml"):
        tree, info = build(a, (M / f"{SRC}{suffix}").read_text(), suffix != ".xml")
        ET.indent(tree, "  ")
        (M / f"{a.out}{suffix}").write_text(ET.tostring(tree.getroot(), encoding="unicode"))
    m = mujoco.MjModel.from_xml_path(str(M / f"{a.out}.xml")); d = mujoco.MjData(m)
    m0 = mujoco.MjModel.from_xml_path(str(M / f"{SRC}.xml")); d0 = mujoco.MjData(m0)
    q_nom = np.array(src_meta["nominal_q"])

    def stand(mm, dd):
        dd.qpos[:] = 0; dd.qpos[3] = 1; dd.qpos[7:] = q_nom; mujoco.mj_kinematics(mm, dd); mujoco.mj_comPos(mm, dd)
        low = min(dd.geom_xpos[mm.geom(f"{s}_tip").id][2] for s in "LR") - 0.002
        return -low, dd.subtree_com[0].copy(), float(mm.body_subtreemass[0])
    z0, com0, mt0 = stand(m0, d0)
    z1, com1, mt1 = stand(m, d)
    if a.rebalance:
        # move upper-body mass forward: shift the pelvis com so the robot's com x matches today's
        dx = (com0[0] - com1[0]) * mt1 / float(m.body_mass[m.body("pelvis").id])
        for suffix in (".xml", "_train.xml"):
            p = M / f"{a.out}{suffix}"; tree = ET.parse(p); pin = tree.getroot().find("worldbody/body[@name='pelvis']/inertial")
            c = np.array(pin.get("pos").split(), float); c[0] += dx; pin.set("pos", f3(c)); tree.write(p, encoding="unicode")
        upper = float(json.loads((M / f"{SRC}.json").read_text()).get("upper_kg", 0.3))
        info["rebalance_pelvis_com_dx_mm"] = round(1000 * dx, 1)
        info["rebalance_torso_head_forward_mm"] = round(1000 * dx * float(m.body_mass[m.body("pelvis").id]) / upper, 1)
        m = mujoco.MjModel.from_xml_path(str(M / f"{a.out}.xml")); d = mujoco.MjData(m)
        z1, com1, mt1 = stand(m, d)
    meta = dict(src_meta); meta["name"] = a.out; meta["total_mass"] = mt1; meta["stand_base_z"] = float(z1)
    meta["flathip"] = dict(thigh=a.thigh, shin=a.shin, zp=a.zp, yaw_x=a.yaw_x, yaw_m=a.yaw_m, butt=a.butt,
                           rebalance=a.rebalance, leg_ik=[a.thigh, round(a.shin + 0.03235, 5)], **info)
    meta["mass_note"] = (f"flat-hip leg from {SRC} (scripts/make_flathip_model.py): servos pitch / roll / yaw front to "
                         f"back under the hip plate, thigh {1000 * a.thigh:.0f} mm, shin {1000 * a.shin:.0f} mm")
    (M / f"{a.out}.json").write_text(json.dumps(meta, indent=1))
    print(f"{a.out}: thigh {1000 * a.thigh:.0f} mm, shin {1000 * a.shin:.0f} mm (today 138 / 111); total {mt1:.3f} kg "
          f"(today {mt0:.3f}); standing pelvis height {1000 * z1:.0f} mm (today {1000 * z0:.0f}) at the same joint angles;"
          f" robot com x {1000 * com1[0]:+.1f} mm (today {1000 * com0[0]:+.1f}), z {1000 * com1[2]:.0f} (today {1000 * com0[2]:.0f})")
    print(json.dumps(meta["flathip"]))


if __name__ == "__main__":
    main()
