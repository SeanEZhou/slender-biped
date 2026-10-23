"""Plot training curves (eval reward and episode length) for one or more runs.

    .venv/bin/python scripts/plot_training.py stand_A8_dr_s0 stand_B8_dr_s0 --out results/plots/curves_stand.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run):
    rows = [json.loads(line) for line in (ROOT / "runs" / run / "progress.jsonl").read_text().splitlines() if line.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctrl-dt", type=float, default=0.02)
    a = ap.parse_args()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), dpi=130)
    for r in a.runs:
        rows = load(r)
        st = [x["step"] / 1e6 for x in rows]
        axs[0].plot(st, [x.get("eval/episode_reward", float("nan")) for x in rows], marker="o", ms=3, label=r)
        axs[1].plot(st, [x.get("eval/avg_episode_length", float("nan")) * a.ctrl_dt for x in rows], marker="o", ms=3, label=r)
    axs[0].set_title("eval episode reward")
    axs[1].set_title("eval episode length [s] (cap 20 s)")
    for ax in axs:
        ax.set_xlabel("env steps [M]")
        ax.grid(alpha=0.3)
    axs[1].legend(fontsize=7)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(a.out)


if __name__ == "__main__":
    sys.exit(main())
