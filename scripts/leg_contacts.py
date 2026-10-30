"""How often do the two legs touch each other? (CPU, full contact model)

    .venv/bin/python scripts/leg_contacts.py <run> --params params_XXXXM.pkl [--test walk40] [--n 10]
Prints the fraction of control steps with any left-right leg contact and the geom pairs involved.
"""
import argparse
import collections
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from evaluate import TESTS  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, run_episode  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--params", default="params.pkl")
ap.add_argument("--test", default="walk40")
ap.add_argument("--n", type=int, default=10)
a = ap.parse_args()
cfg, run_dir = load_run(a.run)
pol = NumpyPolicy(run_dir / a.params)
pairs, hit_steps, steps = collections.Counter(), 0, 0
orig = CpuBiped.step


def step(self, action):
    global hit_steps, steps
    out = orig(self, action)
    m, d = self.m, self.d
    hit = False
    for i in range(d.ncon):
        c = d.contact[i]
        n1, n2 = mujoco.mj_id2name(m, 5, c.geom1), mujoco.mj_id2name(m, 5, c.geom2)
        if n1 and n2 and {n1[:2], n2[:2]} == {"L_", "R_"}:
            pairs[tuple(sorted((n1, n2)))] += 1
            hit = True
    hit_steps += hit
    steps += 1
    return out


CpuBiped.step = step
t = TESTS[a.test]
for s in range(a.n):
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], dr=False), seed=10000 + s)
    run_episode(env, pol, t["seconds"], t["sched"], heading_hold=t.get("heading_hold", False))
print(f"{a.run} {a.params} {a.test}: leg-leg contact in {hit_steps / max(steps, 1):.1%} of control steps; "
      f"pairs {dict(pairs.most_common(4))}")
