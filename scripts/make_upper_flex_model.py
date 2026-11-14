"""Round 7 (2026-11-13): the upper body (torso + head) on spring hinges above the hip plate, so the full body rings as
the robot does (measuring.md E7 taps: roll 4.8-5.7 Hz, zeta ~0.15; E8 wiggle: pitch resonance ~4.3 Hz).

The pelvis keeps the legs model's pelvis (hip plate, hip servos); the difference to the full model's pelvis (mass,
centre of mass, inertia) becomes a body `upper` hung from the pelvis on three passive hinges (pitch, roll, yaw) at
height --z with stiffness for the given natural frequencies (upper body alone on a fixed pelvis, gravity included) and
damping for the given damping ratio. The body is the pelvis's last child, so its joints come after the 10 servo
joints in qpos / qvel. The IMU stays on the pelvis.

    .venv/bin/python scripts/make_upper_flex_model.py --base hw_ank3_b150_m2_10dof --legs hw_ank3_b150_legs_pelvis_m2_10dof \
        --name hw_ank3_b150_m2uf_10dof --f 4.3 5.2 4.5 --zeta 0.12 0.12 0.25 --z 0.04
"""
import argparse, json, re
from pathlib import Path
import numpy as np
MD = Path(__file__).resolve().parents[1] / "models"

def inertial(xml):
    m = re.search(r'<body name="pelvis"[^>]*>\s*<freejoint[^>]*/>\s*<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"/>', xml)
    c = np.array(m.group(1).split(), float); mass = float(m.group(2)); I6 = np.array(m.group(3).split(), float)
    I = np.array([[I6[0], I6[3], I6[4]], [I6[3], I6[1], I6[5]], [I6[4], I6[5], I6[2]]])
    return m, c, mass, I

def at_origin(c, m, I):
    return I + m * (c @ c * np.eye(3) - np.outer(c, c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="hw_ank3_b150_m2_10dof"); ap.add_argument("--legs", default="hw_ank3_b150_legs_pelvis_m2_10dof")
    ap.add_argument("--name", default="hw_ank3_b150_m2uf_10dof")
    ap.add_argument("--f", type=float, nargs=3, default=[4.3, 5.2, 4.5], help="Hz: pitch, roll, yaw")
    ap.add_argument("--zeta", type=float, nargs=3, default=[0.12, 0.12, 0.25])
    ap.add_argument("--z", type=float, default=0.04, help="m: hinge height above the pelvis origin")
    a = ap.parse_args()
    _, cl, ml, Il = inertial((MD / f"{a.legs}_train.xml").read_text())
    out = {}
    for suf in ("", "_train"):
        xml = (MD / f"{a.base}{suf}.xml").read_text()
        mt, cf, mf, If = inertial(xml)
        mu = mf - ml; cu = (mf * cf - ml * cl) / mu
        Iu_o = at_origin(cf, mf, If) - at_origin(cl, ml, Il)
        Iu = Iu_o - mu * (cu @ cu * np.eye(3) - np.outer(cu, cu))
        ev = np.linalg.eigvalsh(Iu)
        if ev.min() <= 0:   # differences of CAD inertias can come out slightly indefinite: lift to a valid tensor
            Iu = Iu + (abs(ev.min()) + 1e-6) * np.eye(3)
        h = np.array([0.0, 0.0, a.z]); r = cu - h                       # centre of mass relative to the hinge
        Ih = Iu + mu * (r @ r * np.eye(3) - np.outer(r, r))
        g = 9.81; ks, ds = [], []
        for ax, f, z in zip((1, 0, 2), a.f, a.zeta):
            I = Ih[ax, ax]; w = 2 * np.pi * f
            k = I * w ** 2 + (mu * g * r[2] if ax != 2 else 0.0)        # gravity tips the upper body: stiffer spring
            ks.append(k); ds.append(2 * z * np.sqrt(I * (k - (mu * g * r[2] if ax != 2 else 0.0))))
        xml = xml.replace(mt.group(0), mt.group(0).split("<inertial")[0]
                          + f'<inertial pos="{cl[0]:.6g} {cl[1]:.6g} {cl[2]:.6g}" mass="{ml:.6g}" fullinertia="'
                          + " ".join(f"{x:.6g}" for x in (Il[0, 0], Il[1, 1], Il[2, 2], Il[0, 1], Il[0, 2], Il[1, 2])) + '"/>')
        pc = cu - h
        body = (f'      <body name="upper" pos="0 0 {a.z:.4f}">\n'
                f'        <joint name="upper_pitch" axis="0 1 0" stiffness="{ks[0]:.5g}" damping="{ds[0]:.5g}" armature="0" frictionloss="0" range="-0.3 0.3"/>\n'
                f'        <joint name="upper_roll" axis="1 0 0" stiffness="{ks[1]:.5g}" damping="{ds[1]:.5g}" armature="0" frictionloss="0" range="-0.3 0.3"/>\n'
                f'        <joint name="upper_yaw" axis="0 0 1" stiffness="{ks[2]:.5g}" damping="{ds[2]:.5g}" armature="0" frictionloss="0" range="-0.3 0.3"/>\n'
                f'        <inertial pos="{pc[0]:.6g} {pc[1]:.6g} {pc[2]:.6g}" mass="{mu:.6g}" fullinertia="'
                + " ".join(f"{x:.6g}" for x in (Iu[0, 0], Iu[1, 1], Iu[2, 2], Iu[0, 1], Iu[0, 2], Iu[1, 2])) + '"/>\n'
                f'      </body>\n')
        i = xml.index("  </worldbody>"); j = xml.rindex("    </body>", 0, i)       # the pelvis's closing tag
        xml = xml[:j] + body + xml[j:]
        xml = re.sub(r'(<key name="stand" qpos="[^"]+)"', r'\1 0 0 0"', xml)
        (MD / f"{a.name}{suf}.xml").write_text(xml)
        out[suf] = dict(mass=mu, com=cu.tolist(), k=ks, d=ds, I_hinge=np.diag(Ih).tolist())
    meta = json.loads((MD / f"{a.base}.json").read_text())
    meta.update(name=a.name, upper_flex=dict(f=a.f, zeta=a.zeta, z=a.z, **out[""]),
                note=meta.get("note", "") + f"; upper body (torso + head) on spring hinges at z {a.z} (pitch/roll/yaw {a.f} Hz, zeta {a.zeta})")
    (MD / f"{a.name}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(out[""], indent=None))

if __name__ == "__main__":
    main()
