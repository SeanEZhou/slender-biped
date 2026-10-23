"""Generate MJCF models from configs/geometry.yaml + configs/actuators.yaml.

    python models/generate_model.py                       # standard set (A/B x 8/10 DoF + variants)
    python models/generate_model.py --variant long_shin --foot A --no-ankle
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slender_biped.config import MODEL_DIR, load_geometry, model_name  # noqa: E402
from slender_biped.model_builder import Builder  # noqa: E402

STANDARD = [
    ("baseline", "A", False), ("baseline", "B", False),
    ("baseline", "A", True), ("baseline", "B", True),
    ("long_shin", "B", True), ("slim_dynamixel", "B", True), ("direct_knee", "B", True),
]


def lowest_foot_z(m, d):
    zs = []
    for side in "LR":
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_tip")
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE:
            zs.append(d.geom_xpos[gid][2] - m.geom_size[gid][0])
        else:  # ellipsoid support function along -z
            R = d.geom_xmat[gid].reshape(3, 3)
            e = m.geom_size[gid]
            zs.append(d.geom_xpos[gid][2] - np.linalg.norm(e * R[2, :]))
    return min(zs)


def generate(variant="baseline", foot="B", ankle=True, actuator_set=None, overrides=None, write=True):
    cfg = load_geometry(variant, foot=foot, ankle=ankle, actuator_set=actuator_set, overrides=overrides)
    b = Builder(cfg)
    q = b.nominal_q()
    m = mujoco.MjModel.from_xml_string(b.build())
    d = mujoco.MjData(m)
    d.qpos[7:] = q
    mujoco.mj_kinematics(m, d)
    z = 0.5 - lowest_foot_z(m, d)
    xml = b.build(keyframe_z=z, keyframe_q=q)
    m = mujoco.MjModel.from_xml_string(xml)  # validate
    name = model_name(cfg)
    meta = {
        "name": name, "cfg": cfg, "joints": b.joint_list(), "nominal_q": q, "stand_base_z": z,
        "servos": {k: v for k, v in b.servos.items()},
        "mass_log": b.mass_log, "envelopes": b.envelopes, "exteriors": b.exteriors,
        "total_mass": float(sum(m.body_mass)),
    }
    if write:
        if cfg.get("contact_set", "full") == "train":
            name += "_train"
            meta["name"] = name
        (MODEL_DIR / f"{name}.xml").write_text(xml)
        (MODEL_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))
    return xml, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--foot", choices=["A", "B"], default="B")
    ap.add_argument("--no-ankle", action="store_true")
    ap.add_argument("--actuator-set")
    a = ap.parse_args()
    jobs = [(a.variant, a.foot, not a.no_ankle)] if a.variant else STANDARD
    for v, f, ank in jobs:
        _, meta = generate(v, f, ank, a.actuator_set)
        print(f"{meta['name']:28s} mass={meta['total_mass']*1000:6.1f} g  stand_z={meta['stand_base_z']:.4f} m")
    # canonical name from the spec layout points at the selected primary design (ball tip, 8 DoF)
    src = MODEL_DIR / "baseline_A_8dof.xml"
    if src.exists():
        (MODEL_DIR / "slender_biped.xml").write_text(src.read_text())


if __name__ == "__main__":
    main()
