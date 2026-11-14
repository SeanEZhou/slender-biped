"""Re-mass the full-body model to the robot as weighed (2026-11-13, docs/measuring.md "Full body"): legs, pelvis and
hip plates 0.9 kg (model 1.205), torso and head 0.3 kg (model 0.243). Every leg body and the pelvis shell are scaled
by 0.9 / 1.205, the upper body (the full model's pelvis minus the legs model's) by 0.3 / 0.243, keeping positions.

    .venv/bin/python scripts/make_mass_model.py [--legs 0.9] [--upper 0.3] [--out hw_ank3_b150_m2_10dof]
"""
import argparse, json, re
from pathlib import Path
import numpy as np
M = Path(__file__).resolve().parents[1] / "models"

def full(I):  # fullinertia = ixx iyy izz ixy ixz iyz
    a = I; return np.array([[a[0], a[3], a[4]], [a[3], a[1], a[5]], [a[4], a[5], a[2]]])
def shift(Ic, m, c):  # com-frame inertia -> about origin
    c = np.asarray(c); return Ic + m * (c @ c * np.eye(3) - np.outer(c, c))
def pelvis_inertial(xml):
    s = re.search(r'<body name="pelvis".*?<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"', xml, re.S)
    return np.array(s.group(1).split(), float), float(s.group(2)), np.array(s.group(3).split(), float)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--legs", type=float, default=0.9); ap.add_argument("--upper", type=float, default=0.3)
    ap.add_argument("--src", default="hw_ank3_b150_10dof"); ap.add_argument("--legsrc", default="hw_ank3_b150_legs_pelvis_10dof")
    ap.add_argument("--out", default="hw_ank3_b150_m2_10dof"); a = ap.parse_args()
    legs_xml = (M / f"{a.legsrc}.xml").read_text()
    cs, ms, Is = pelvis_inertial(legs_xml)
    legs_total = json.loads((M / f"{a.legsrc}.json").read_text())["total_mass"]
    k_leg = a.legs / legs_total
    for suffix in (".xml", "_train.xml"):
        xml = (M / f"{a.src}{suffix}").read_text()
        cf, mf, If = pelvis_inertial(xml)
        Jf, Js = shift(full(If), mf, cf), shift(full(Is), ms, cs)
        mu, cu = mf - ms, (mf * cf - ms * cs) / (mf - ms); Ju = Jf - Js
        k_up = a.upper / mu if mu > 1e-6 else 0.0
        if mu <= 1e-6: cu = cs
        m_new = ms * k_leg + mu * k_up; c_new = (ms * k_leg * cs + mu * k_up * cu) / m_new
        J_new = Js * k_leg + Ju * k_up; Ic = J_new - m_new * (c_new @ c_new * np.eye(3) - np.outer(c_new, c_new))
        fi = " ".join(f"{v:.6g}" for v in (Ic[0, 0], Ic[1, 1], Ic[2, 2], Ic[0, 1], Ic[0, 2], Ic[1, 2]))
        first = [True]
        def rep(mo):
            pos, mass, fin = mo.group(1), float(mo.group(2)), np.array(mo.group(3).split(), float)
            if first[0]:
                first[0] = False
                return f'<inertial pos="{" ".join(f"{v:.6g}" for v in c_new)}" mass="{m_new:.6g}" fullinertia="{fi}"'
            return f'<inertial pos="{pos}" mass="{mass * k_leg:.6g}" fullinertia="{" ".join(f"{v * k_leg:.6g}" for v in fin)}"'
        out = re.sub(r'<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"', rep, xml)
        (M / f"{a.out}{suffix}").write_text(out)
    meta = json.loads((M / f"{a.src}.json").read_text()); meta["name"] = a.out
    import mujoco
    mm = mujoco.MjModel.from_xml_path(str(M / f"{a.out}.xml")); meta["total_mass"] = float(sum(mm.body_mass))
    meta["mass_note"] = f"re-massed from {a.src}: legs+pelvis {a.legs} kg, torso+head {a.upper} kg (weighed 2026-11-13)"
    (M / f"{a.out}.json").write_text(json.dumps(meta, indent=1))
    print(f"{a.out}: total {meta['total_mass']:.3f} kg; pelvis body {m_new:.3f} kg, com {1000 * c_new[2]:.0f} mm above the hips; legs x{k_leg:.3f}, upper body {mu:.3f} -> {a.upper} kg, its com {1000 * cu[2]:.0f} mm up")

if __name__ == "__main__":
    main()
