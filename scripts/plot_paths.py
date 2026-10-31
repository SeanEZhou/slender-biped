"""Top-down walking paths, before and after the round-5 drift work (README figure).

    .venv/bin/python scripts/plot_paths.py   -> results/plots/walk_paths_r5.png

20 seeds per policy (the drift_eval seeds), 0.15 m/s command, heading hold on, 20 s. Each policy is evaluated with
its own sensor configuration (round-4 walker: no velocity input; legs: stance-leg odometry; flow: optical flow).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of  # noqa: E402

BLUE, INK, INK2, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURFACE,
})
POLICIES = [
    ("r4k_cat2_ft", "params_0030M.pkl", "Round 4 catwalk walker: no sense of its own drift"),
    ("r5q_legs_path", "params_0030M.pkl", "Round 5, leg odometry only (no extra sensor)"),
    ("r5p_flow", "params_0020M.pkl", "Round 5, with a downward optical-flow sensor"),
]


def paths(run, params, seeds=20, seconds=20.0):
    cfg, rd = load_run(run)
    pol = NumpyPolicy(rd / params)
    out = []
    for s in range(seeds):
        env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"]), seed=30000 + s)
        obs = env.reset((0.15, 0.0, 0.0))
        yaw0 = yaw_of(env.d.qpos[3:7])
        p0 = env.d.qpos[0:2].copy()
        R = np.array([[np.cos(yaw0), np.sin(yaw0)], [-np.sin(yaw0), np.cos(yaw0)]])
        xy = [(0.0, 0.0)]
        for _ in range(int(seconds / env.cfg.ctrl_dt)):
            err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
            env.cmd[2] = float(np.clip(err, -0.3, 0.3))
            obs[-5:-2] = env.cmd
            obs, _ = env.step(pol(obs))
            if env.fallen():
                break
            xy.append(tuple(R @ (env.d.qpos[0:2] - p0)))
        out.append(np.array(xy))
    return out


fig, axes = plt.subplots(len(POLICIES), 1, figsize=(8, 7.2), sharex=True, sharey=True)
for ax, (run, params, title) in zip(axes, POLICIES):
    ps = paths(run, params)
    ax.axhspan(-0.2, 0.2, color=GRID, lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (4, 4)), zorder=1)
    ax.axvline(2.0, color=INK2, lw=0.8, zorder=1)
    on = 0
    for p in ps:
        ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.2, alpha=0.7, zorder=2)
        i = np.argmax(p[:, 0] >= 2.0)
        on += bool(p[i, 0] >= 2.0 and abs(p[i, 1]) <= 0.2)
    ax.set_title(title, loc="left")
    ax.text(0.995, 0.96, f"{on} of {len(ps)} pass 2 m within 0.2 m of the line", transform=ax.transAxes,
            ha="right", va="top", color=INK2, zorder=3,
            bbox=dict(boxstyle="square,pad=0.25", facecolor=SURFACE, edgecolor="none"))
    ax.set_ylabel("sideways (m)")
    ax.grid(False)
axes[0].set_ylim(-1.6, 1.6)
axes[0].set_xlim(-0.1, 4.3)
axes[-1].set_xlabel("forward along the start heading (m)")
fig.text(0.01, 0.005, "20 seeds per walker, 0.15 m/s command, heading held, 20 s. Gray band: ±0.2 m of the line; "
         "vertical line: 2 m.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 1))
out = ROOT / "results" / "plots" / "walk_paths_r5.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(out)
