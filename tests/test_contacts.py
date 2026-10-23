import mujoco
import numpy as np
import pytest

from models.generate_model import generate


@pytest.mark.parametrize("foot", ["A", "B"])
@pytest.mark.parametrize("ankle", [False, True])
def test_tip_is_lowest_collision_geom(foot, ankle):
    """No hidden sole: in the stand pose the tip geom is the lowest colliding geom of each leg."""
    xml, meta = generate("baseline", foot, ankle, write=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    low = {}
    colliding = set(m.pair_geom1) | set(m.pair_geom2)  # explicit-pair model: contype is 0 everywhere
    for g in range(m.ngeom):
        if g not in colliding or m.geom_bodyid[g] == 0:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        # conservative lower bound: centre minus bounding radius
        z = d.geom_xpos[g][2] - m.geom_rbound[g]
        low[name] = z
    for s in "LR":
        tip = low.pop(f"{s}_tip")
        others = [v for k, v in low.items() if k.startswith(s)]
        assert tip < min(others)


@pytest.mark.parametrize("foot,maxlen", [("A", 0.021), ("B", 0.041)])
def test_footprint_small(foot, maxlen):
    xml, _ = generate("baseline", foot, True, write=False)
    m = mujoco.MjModel.from_xml_string(xml)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "L_tip")
    assert 2 * m.geom_size[g][0] <= maxlen
    if foot == "B":
        assert 0.010 <= 2 * m.geom_size[g][1] <= 0.020


def test_stand_settles_on_tips_only():
    xml, _ = generate("baseline", "B", True, write=False)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(20):
        mujoco.mj_step(m, d)
    names = {tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                           mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)))) for c in d.contact[:d.ncon]}
    assert names == {("L_tip", "floor"), ("R_tip", "floor")}
