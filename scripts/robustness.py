"""Robustness sweep (Phase 3): same held-out seeds for every policy and condition.

    .venv/bin/python scripts/robustness.py walk_A8_dr_s0 walk_B8_dr_s0 --test stand --n 50
Writes results/robustness_<test>.json (merged across calls) and results/plots/robustness_<test>.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluate import evaluate  # noqa: E402

CONDITIONS = {
    "nominal": ({}, False),
    "ideal_sensors": ({"obs_noise": 0.0}, False),
    "dr": ({}, True),
    "friction_0.5": ({"friction": 0.5}, False),
    "friction_0.35": ({"friction": 0.35}, False),
    "mass_x1.2": ({"mass_scale": 1.2}, False),
    "strength_x0.75": ({"strength_scale": 0.75}, False),
    "latency_12ms": ({"latency_sub": 6}, False),
    "noise_x2": ({"obs_noise": 2.0}, False),
    "slope_5deg": ({"slope_deg": 5.0}, False),
    "push_0.3": ({"push_vel": 0.3}, False),
    "push_0.5": ({"push_vel": 0.5}, False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--test", default="stand")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--conds", nargs="*")
    ap.add_argument("--params", default="params.pkl")
    a = ap.parse_args()
    path = ROOT / "results" / f"robustness_{a.test}.json"
    allres = json.loads(path.read_text()) if path.exists() else {}
    conds = a.conds or list(CONDITIONS)
    for run in a.runs:
        for c in conds:
            over, dr = CONDITIONS[c]
            # strength_scale is set on the eval env only; the policy is unchanged
            summ, _ = evaluate(run, a.test, a.n, dr, a.params, a.procs, over, tag=f"__rob_{c}")
            key_run = run if a.params == "params.pkl" else f"{run}@{a.params[7:12]}"
            allres.setdefault(key_run, {})[c] = summ
            key = "walk_1.0m" if a.test == "walk" else "alive_15s"
            print(f"{run:28s} {c:16s} {key}={summ.get(key, float('nan')):.2f} fall={summ['fall_rate']:.2f} "
                  f"steps/s={summ['steps_per_s_median']:.1f}", flush=True)
            path.write_text(json.dumps(allres, indent=1))
    # plot
    runs = list(allres)
    key = "walk_1.0m" if a.test == "walk" else "alive_15s"
    fig, ax = plt.subplots(figsize=(1.2 * len(CONDITIONS) + 2, 3.8), dpi=130)
    w = 0.8 / max(1, len(runs))
    for i, r in enumerate(runs):
        vals = [allres[r].get(c, {}).get(key, np.nan) for c in CONDITIONS]
        ax.bar(np.arange(len(CONDITIONS)) + i * w, vals, w, label=r)
    ax.set_xticks(np.arange(len(CONDITIONS)) + w * (len(runs) - 1) / 2)
    ax.set_xticklabels(list(CONDITIONS), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"success ({key})")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title(f"robustness, {a.test} test, n={a.n} held-out seeds per bar")
    fig.tight_layout()
    (ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "plots" / f"robustness_{a.test}.png")


if __name__ == "__main__":
    main()
