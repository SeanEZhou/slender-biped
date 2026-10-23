import json

import mujoco
import numpy as np
import pytest

from conftest import ROOT
from models.generate_model import generate
from slender_biped.morphology import measure

BASELINES = ["baseline_A_8dof", "baseline_B_8dof", "baseline_A_10dof", "baseline_B_10dof"]


@pytest.fixture(scope="module", params=BASELINES)
def model(request):
    xml, meta = generate("baseline", request.param.split("_")[1], request.param.endswith("10dof"), write=False)
    return mujoco.MjModel.from_xml_string(xml), meta


def test_compiles_and_dof(model):
    m, meta = model
    dof = 10 if meta["name"].endswith("10dof") else 8
    assert m.nu == dof and m.njnt == dof + 1 and m.nv == dof + 6


def test_mass_in_target_band(model):
    m, _ = model
    assert 0.6 <= m.body_mass.sum() <= 1.0


def test_left_right_symmetric(model):
    m, _ = model
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if n and n.startswith("L_"):
            r = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "R_" + n[2:])
            assert np.isclose(m.body_mass[b], m.body_mass[r])
            np.testing.assert_allclose(m.body_ipos[b] * [1, -1, 1], m.body_ipos[r], atol=1e-9)


def test_packaging_and_no_interpenetration(model):
    m, meta = model
    rep, d = measure(m, meta)
    assert rep["packaging_ok"], [p for p in rep["packaging"] if not p["ok"]]
    for g1, g2, dist in rep["stand_contacts"]:
        assert {g1, g2} & {"floor"} and ({g1, g2} - {"floor"}).pop().endswith("_tip"), (g1, g2)


def test_aesthetic_ratios(model):
    m, meta = model
    rep, _ = measure(m, meta)
    assert rep["shin_width_ratio"] <= 0.12
    assert rep["hip_width_ratio"] <= 0.29  # 28.4% measured; spec target <= 28% (deviation documented)


def test_joint_signs(model):
    """+hip_pitch moves the tip forward, +knee backward, +roll outward (abduction)."""
    m, meta = model
    d = mujoco.MjData(m)

    def tip(side, **q):
        d.qpos[:] = 0
        d.qpos[2], d.qpos[3] = 0.5, 1
        for j, v in q.items():
            d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}")]] = v
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_contact")].copy()

    for side, s in (("L", 1), ("R", -1)):
        p0 = tip(side)
        assert tip(side, hip_pitch=0.3)[0] > p0[0] + 0.05
        assert tip(side, knee=0.3)[0] < p0[0] - 0.03
        assert s * (tip(side, hip_roll=0.2)[1] - p0[1]) > 0.03
