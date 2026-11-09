"""Add an ankle roll joint to a pad model (2026-11-09 design study): <base>.xml/_train.xml/.json -> <base>_ankle.

    .venv/bin/python scripts/make_ankle_model.py hw_pad_board150w_legs_8dof [--axis-h 0.030] [--servo-mass 0.070]

Each shin gets a child body `<S>_foot` carrying the pad geoms (board, kicks, collision capsule, touch site), hinged
about the board's long axis `axis_h` m above the board centre: joint `<S>_ankle_roll` (axis mirrored on the right leg
like the hip roll, so the mirror image of a joint vector is the plain left/right swap), driven by the hip-roll servo
(STS3215 C001 1:345, 7.4 V; actuator set `<set>_ankleroll`). The servo plus bracket (`servo_mass`) is added to the
shin just above the joint, a 10 g bracket to the foot. Joint order: hip yaw, hip roll, hip pitch, knee, ankle roll.
The leg length is unchanged (a real design would shorten the shin to fit the servo).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.r_[0.0, v]), qconj(q))[1:]


def fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def attr(tag, name):
    m = re.search(rf'\b{name}="([^"]*)"', tag)
    return np.array([float(x) for x in m.group(1).split()]) if m else None


def setattr_(tag, name, value):
    if re.search(rf'\b{name}="[^"]*"', tag):
        return re.sub(rf'\b{name}="[^"]*"', f'{name}="{value}"', tag)
    return tag.replace("<geom ", f'<geom {name}="{value}" ', 1).replace("<site ", f'<site {name}="{value}" ', 1)


def convert(xml, axis_h, servo_mass):
    out = xml
    for S, sgn in (("L", 1), ("R", -1)):
        tip = re.search(rf'<geom name="{S}_tip" [^>]*/>', out).group(0)
        c, q_b = attr(tip, "pos"), attr(tip, "quat")
        n_up = qrot(q_b, np.array([0, 0, 1.0]))
        p_a = c + axis_h * n_up                                  # ankle axis point in the shin frame
        qi = qconj(q_b)
        lines = [l for l in out.split("\n")]
        foot_tags, keep = [], []
        for l in lines:
            s = l.strip()
            if re.match(rf'<(geom|site) name="{S}_(tip|tip_\d+|padcol_\d+|touch)"', s):
                t = s
                pos = attr(t, "pos")
                if pos is not None:
                    t = setattr_(t, "pos", fmt(qrot(qi, pos - p_a)))
                ft = attr(t, "fromto")
                if ft is not None:
                    t = setattr_(t, "fromto", fmt(np.r_[qrot(qi, ft[:3] - p_a), qrot(qi, ft[3:] - p_a)]))
                qq = attr(t, "quat")
                if qq is not None:
                    t = setattr_(t, "quat", fmt(qmul(qi, qq)))
                foot_tags.append(t)
            else:
                keep.append(l)
        out = "\n".join(keep)
        # servo mass on the shin, just above the joint (along the shin axis)
        servo_p = p_a + 0.02 * n_up
        foot = (f'<body name="{S}_ankle_servo" pos="{fmt(servo_p)}">\n'      # welded to the shin (no joint)
                f'                <inertial pos="0 0 0" mass="{servo_mass}" diaginertia="1.2e-5 1.2e-5 7e-6"/>\n'
                f'              </body>\n'
                f'              <body name="{S}_foot" pos="{fmt(p_a)}" quat="{fmt(q_b)}">\n'
                f'                <joint name="{S}_ankle_roll" axis="{sgn} 0 0" range="-0.349066 0.349066" armature="0.00297562"/>\n'
                f'                <inertial pos="0 0 {-axis_h:.4f}" mass="0.010" diaginertia="2e-6 4e-6 4e-6"/>\n'
                + "".join(f"                {t}\n" for t in foot_tags) +
                f'              </body>')
        # insert before the closing tag of <S>_shin: the first "</body>" after the shin's opening line
        i = out.index(f'<body name="{S}_shin"')
        j = out.index("</body>", i)
        indent = "              "
        out = out[:j] + foot + "\n" + indent[:-2] + out[j:]
    # actuators: insert ankle motors after each knee motor
    for S in ("L", "R"):
        km = re.search(rf'\s*<motor name="{S}_knee"[^>]*/>', out).group(0)
        out = out.replace(km, km + f'\n    <motor name="{S}_ankle_roll" joint="{S}_ankle_roll" ctrlrange="-1.91 1.91"/>', 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--axis-h", type=float, default=0.030)
    ap.add_argument("--servo-mass", type=float, default=0.070)
    a = ap.parse_args()
    name = a.base.replace("_8dof", "") + "_ankle_10dof"
    for suf in ("", "_train"):
        src = MODELS / f"{a.base}{suf}.xml"
        if src.exists():
            (MODELS / f"{name}{suf}.xml").write_text(convert(src.read_text(), a.axis_h, a.servo_mass))
    meta = json.loads((MODELS / f"{a.base}.json").read_text())
    J = meta["joints"]
    meta["joints"] = J[:4] + ["L_ankle_roll"] + J[4:] + ["R_ankle_roll"]
    nq = list(meta["nominal_q"])
    meta["nominal_q"] = nq[:4] + [0.0] + nq[4:] + [0.0]
    meta["cfg"] = dict(meta["cfg"], actuator_set=meta["cfg"]["actuator_set"] + "_ankleroll")
    meta["total_mass"] = meta["total_mass"] + 2 * (a.servo_mass + 0.010)
    meta["note"] = (f"{a.base} + ankle roll (scripts/make_ankle_model.py): hinge {1000 * a.axis_h:.0f} mm above the board "
                    f"centre along the board, STS3215 C001 1:345 7.4 V, {1000 * a.servo_mass:.0f} g servo+bracket on the shin")
    (MODELS / f"{name}.json").write_text(json.dumps(meta, indent=1))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODELS / f"{name}.xml"))
    print(name, "nq", m.nq, "nu", m.nu, "mass %.3f kg" % sum(m.body_mass), [m.joint(i).name for i in range(m.njnt)])


if __name__ == "__main__":
    sys.exit(main())
