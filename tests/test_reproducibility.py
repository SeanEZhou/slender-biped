import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))


def test_sanity_sim_deterministic():
    from sanity_sim import run
    a = run("baseline_B_10dof", "random", seconds=0.6, seed=3, video=False)
    b = run("baseline_B_10dof", "random", seconds=0.6, seed=3, video=False)
    assert a == b
