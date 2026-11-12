"""Legs-only model with the pelvis shell as built: T03 pelvis ring (bolted through the P08 plate's six holes) and the
T01 / T02 hip panels (front and rear), from hardware/torso_p0. The rest of the torso and the head stay off.

    .venv-cad/bin/python scripts/make_pelvis_model.py --parts      # mass properties from the torso CAD -> models/pelvis_shell.json
    .venv/bin/python scripts/make_pelvis_model.py                  # writes models/hw_ank3_b150_legs_pelvis_10dof{,_train}.{xml,json}

The parts' mass, centre of mass and inertia come from the CAD solids (hardware/torso_p0/cad/torso.py build(), the
same density as its report, 1.24 g/cm^3: thin walls print effectively solid) in the leg CAD frame, which is the sim
pelvis frame. Plus the screws and nuts that hold them (6 x M3x12 + nuts, 1 x M3x6: ~8 g at the plate). Lumped into
the pelvis body's inertial; geometry, joints and contacts are unchanged, so the stand pose and policies' inputs are too.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
BASE = "hw_ank3_b150_legs_10dof"
NAME = "hw_ank3_b150_legs_pelvis_10dof"
PARTS_JSON = MODELS / "pelvis_shell.json"
PARTS = ("T03_pelvis_ring", "T01_hip_front", "T02_hip_rear")
HARDWARE = (0.008, (-0.0027, 0.0, 0.028))          # kg, m: screws and nuts at the plate's hole pattern


def f6(v):
    return " ".join(f"{x:.6g}" for x in v)


def parts_from_cad():
    sys.path.insert(0, str(ROOT / "hardware" / "torso_p0" / "cad"))
    import torso
    built = torso.build()
    out = {}
    for n in PARTS:
        m = built[n]
        out[n] = dict(mass_kg=m.volume * torso.DENSITY / 1000, com_m=(np.asarray(m.center_mass) / 1000).tolist(),
                      inertia_kgm2=(np.asarray(m.moment_inertia) * torso.DENSITY * 1e-9).tolist())
    PARTS_JSON.write_text(json.dumps(out, indent=1))
    for n, v in out.items():
        print(f"{n}: {1000 * v['mass_kg']:.1f} g at {np.round(np.array(v['com_m']) * 1000, 1)} mm")


def combine(J0, m0, p0, pts):
    allp = [(m0, np.asarray(p0), J0)] + pts
    M = sum(m for m, _, _ in allp)
    com = sum(m * np.asarray(p) for m, p, _ in allp) / M
    J = np.zeros((3, 3))
    for m, p, Jo in allp:
        d = np.asarray(p) - com
        J += Jo + m * (d @ d * np.eye(3) - np.outer(d, d))
    return J, M, com


def add_shell(xml, shell):
    bm = re.search(r'(<body name="pelvis"[^\n]*\n(?:[^\n]*\n)*?\s*)<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"/>', xml)
    p0 = np.array([float(x) for x in bm.group(2).split()])
    m0 = float(bm.group(3))
    Ixx, Iyy, Izz, Ixy, Ixz, Iyz = [float(x) for x in bm.group(4).split()]
    J0 = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    pts = [(v["mass_kg"], np.array(v["com_m"]), np.array(v["inertia_kgm2"])) for v in shell.values()]
    pts.append((HARDWARE[0], np.array(HARDWARE[1]), np.zeros((3, 3))))
    J, M, com = combine(J0, m0, p0, pts)
    fi = [J[0, 0], J[1, 1], J[2, 2], J[0, 1], J[0, 2], J[1, 2]]
    return xml.replace(bm.group(0), f'{bm.group(1)}<inertial pos="{f6(com)}" mass="{M:.6g}" fullinertia="{f6(fi)}"/>'), (m0, p0, M, com)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", action="store_true", help="recompute the parts' mass properties from the torso CAD")
    a = ap.parse_args()
    if a.parts:
        parts_from_cad()
        return
    shell = json.loads(PARTS_JSON.read_text())
    for suf in ("", "_train"):
        xml, info = add_shell((MODELS / f"{BASE}{suf}.xml").read_text(), shell)
        (MODELS / f"{NAME}{suf}.xml").write_text(xml)
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MODELS / f"{NAME}.xml"))
    mujoco.MjModel.from_xml_path(str(MODELS / f"{NAME}_train.xml"))
    meta = json.loads((MODELS / f"{BASE}.json").read_text())
    added = sum(v["mass_kg"] for v in shell.values()) + HARDWARE[0]
    meta.update(name=NAME, total_mass=float(sum(m.body_mass)),
                note=meta.get("note", "") + f"; pelvis shell as built (T03 pelvis ring, T01/T02 hip panels, screws: "
                                            f"{1000 * added:.0f} g, scripts/make_pelvis_model.py)")
    meta["cfg"] = dict(meta["cfg"], variant=meta["cfg"].get("variant", "") + "_pelvis")
    (MODELS / f"{NAME}.json").write_text(json.dumps(meta, indent=1))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand"))
    mujoco.mj_forward(m, d)
    com = d.subtree_com[1]
    m0, p0, M, pc = info
    print(f"{NAME}: pelvis {1000 * m0:.0f} g -> {1000 * M:.0f} g, its COM z {1000 * p0[2]:.1f} -> {1000 * pc[2]:.1f} mm; "
          f"robot {sum(m.body_mass):.3f} kg, COM {1000 * com[2]:.0f} mm above the floor in the stand pose")


if __name__ == "__main__":
    main()
