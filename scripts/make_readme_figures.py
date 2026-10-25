"""Figures for the README (all numbers read from results/eval/*.json and runs/*/progress.jsonl).

    .venv/bin/python scripts/make_readme_figures.py

Writes:
  assets/concept/*_small.jpg              README-sized copies of the concept art
  assets/renders/contact_candidates.png   tip/foot candidates: full body + tip close-up
  results/plots/policy_progression.png    ball-tip walking policies, CPU eval (100 seeds)
  results/plots/contact_comparison.png    standing stillness by contact type (control experiment)
  results/plots/training_curves.png       standing reward designs, control-experiment curves, walking regression
Palette: reference categorical slots 1-4 (validated light mode); values are direct-labelled.
"""
from __future__ import annotations

import json
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

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EVAL = ROOT / "results" / "eval"
PLOTS = ROOT / "results" / "plots"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURFACE,
})


def summ(name):
    d = json.loads((EVAL / f"{name}.json").read_text())
    s, eps = d["summary"], d["episodes"]
    if "liftoffs_after_settle" in eps[0]:
        s["nolift_frac"] = np.mean([(not e["fell"]) and e["liftoffs_after_settle"] == 0 for e in eps])
    return s


def progress(run):
    rows = [json.loads(x) for x in (ROOT / "runs" / run / "progress.jsonl").read_text().splitlines() if x.strip()]
    cfg = json.loads((ROOT / "runs" / run / "config.json").read_text())
    dt = cfg["env"]["ctrl_dt"]
    st = np.array([r["step"] / 1e6 for r in rows])
    ln = np.array([r.get("eval/avg_episode_length", np.nan) * dt for r in rows])
    rw = np.array([r.get("eval/episode_reward", np.nan) for r in rows])
    return st, ln, rw


def bar_labels(ax, bars, fmt, horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.text(v + ax.get_xlim()[1] * 0.015, b.get_y() + b.get_height() / 2, fmt(v), va="center", color=INK, fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, v + ax.get_ylim()[1] * 0.02, fmt(v), ha="center", color=INK, fontsize=9)


# ------------------------------------------------------------------ concept art
def concept():
    from PIL import Image
    for src, dst in (("Aigis_Battle.webp", "aigis_small.jpg"),):
        if not (ROOT / "assets" / "concept" / src).exists():  # originals are kept out of the repo
            continue
        im = Image.open(ROOT / "assets" / "concept" / src).convert("RGBA")
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))  # flatten transparency onto black (Aigis art is cut out)
        bg.alpha_composite(im)
        im = bg.convert("RGB")
        im.thumbnail((700, 700 * im.height // im.width if im.width > im.height else 900))
        im.save(ROOT / "assets" / "concept" / dst, quality=85)


# ------------------------------------------------------------------ contact candidates
def contact_candidates():
    import mujoco
    from slender_biped.render import Renderer, stand_data
    models = [("baseline_A_8dof", "A: 16 mm ball\n(selected design)"), ("baseline_B_8dof", "B: 32×13 mm rocker"),
              ("baseline_P20_8dof", "P20: 20×14 mm pad"), ("baseline_P30_8dof", "P30: 30×18 mm pad"),
              ("footed_F60_8dof", "control: 60 mm sole,\nno ankle"), ("footed_F60_12dof", "control: 60 mm sole,\n2-DoF ankle")]
    fig, axs = plt.subplots(2, len(models), figsize=(2.3 * len(models), 6.4), dpi=130,
                            gridspec_kw={"height_ratios": [1.45, 1]})
    for i, (name, label) in enumerate(models):
        m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"{name}.xml"))
        m.mat_reflectance[:] = 0
        d = stand_data(m)
        r = Renderer(m, 360, 520)
        r.set_groups(shells=True)
        axs[0, i].imshow(r.render(d, "three_quarter"))
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        tip = d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "L_tip")]
        cam.lookat[:] = tip + np.array([0, 0, 0.012])
        cam.azimuth, cam.elevation, cam.distance = 120, -12, 0.13
        r.close()
        r = Renderer(m, 360, 360)
        r.set_groups(shells=True)
        axs[1, i].imshow(r.render(d, cam))
        r.close()
        axs[0, i].set_title(label, fontsize=9.5, fontweight="bold" if i == 0 else "normal")
        for ax in axs[:, i]:
            ax.axis("off")
    axs[1, 0].text(-0.05, 0.5, "tip close-up", transform=axs[1, 0].transAxes, rotation=90, va="center", ha="right", color=INK2)
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "renders" / "contact_candidates.png")
    plt.close(fig)


# ------------------------------------------------------------------ policy progression (ball tip walking)
def policy_progression():
    rows = [("v5", "walk_A8_v5_dr_s0__walk40_r2", "walk_A8_v5_dr_s0__startstop_best"),
            ("v6b", "walk_A8_v6b_ft__walk40_r2", "walk_A8_v6b_ft__startstop_best"),
            ("stride", "stride_A8_ft__walk40", "stride_A8_ft__startstop"),
            ("cadence", "cadence_A8_ft__walk40", "cadence_A8_ft__startstop")]
    labels = [r[0] for r in rows]
    vx = [summ(r[1])["mean_vx_median"] for r in rows]
    ss = [100 * summ(r[2])["startstop_success"] for r in rows]
    rated = [max(summ(r[1])["rms_over_rated_median"]) for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), dpi=140)
    x = np.arange(len(rows))
    for ax, vals, title, fmt, top in (
            (axs[0], vx, "Walking speed (m/s) at a 0.15 m/s command", lambda v: f"{v:.3f}", 0.17),
            (axs[1], ss, "Walk → stop → walk success (%)", lambda v: f"{v:.0f}%", 110),
            (axs[2], rated, "Peak joint RMS torque / vendor rated", lambda v: f"{v:.2f}×", 1.1)):
        ax.set_ylim(0, top)
        bars = ax.bar(x, vals, width=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2)
        bar_labels(ax, bars, fmt)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axs[0].axhline(0.15, color=INK2, lw=1, ls="--")
    axs[0].text(3.45, 0.152, "command", color=INK2, fontsize=8.5, ha="right", va="bottom")
    axs[2].axhline(1.0, color=INK2, lw=1, ls="--")
    axs[2].text(3.45, 1.01, "rated", color=INK2, fontsize=8.5, ha="right", va="bottom")
    fig.suptitle("Ball-tip walking policies v5 → v6b → stride → cadence, each fine-tuned from the previous one "
                 "(CPU eval, 100 held-out seeds)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOTS / "policy_progression.png")
    plt.close(fig)


# ------------------------------------------------------------------ contact comparison (control experiment)
def contact_comparison():
    rows = [("Ball 16 mm", "q2_stand_A8__stand", BLUE),
            ("Ball 16 mm, ideal 100 Hz control", "q2_stand_A8_ideal100__stand", BLUE),
            ("Pad 20×14 mm", "q2_stand_P20__stand", ORANGE),
            ("Pad 20×14 mm, gentle starts", "q2_stand_P20_gentle__stand_stdstart", ORANGE),
            ("Pad 30×18 mm, seed 0", "q2_stand_P30__stand", ORANGE),
            ("Pad 30×18 mm, seed 1 (75 M steps)", "q2_stand_P30_s1_cont__stand", ORANGE),
            ("Sole 60 mm, no ankle", "ctl_stand_F60_8rigid__stand", AQUA),
            ("Sole 40 mm + ankle", "ctl_stand_F40_12__stand", AQUA),
            ("Sole 60 mm + ankle", "ctl_stand_F60_12__stand_0045M", AQUA),
            ("Sole 90 mm + ankle", "ctl_stand_F90_12__stand", AQUA)]
    rows = rows[::-1]
    y = np.arange(len(rows))
    s = [summ(r[1]) for r in rows]
    nolift = [100 * v["nolift_frac"] for v in s]
    lifts = [v["liftoffs_median"] for v in s]
    cols = [r[2] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), dpi=140, sharey=True)
    axs[0].set_xlim(0, 112)
    b = axs[0].barh(y, nolift, color=cols, edgecolor=SURFACE, linewidth=2, height=0.7)
    bar_labels(axs[0], b, lambda v: f"{v:.0f}%", horizontal=True)
    axs[0].set_title("Runs that never lift a tip (30 s stand, %)", fontsize=10.5)
    axs[1].set_xlim(0, 560)
    b = axs[1].barh(y, lifts, color=cols, edgecolor=SURFACE, linewidth=2, height=0.7)
    bar_labels(axs[1], b, lambda v: f"{v:.0f}", horizontal=True)
    axs[1].set_title("Tip lift-offs per 28 s (median)", fontsize=10.5)
    axs[0].set_yticks(y, [r[0] for r in rows], fontsize=9)
    for ax in axs:
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(color=BLUE, label="ball tip"), Patch(color=ORANGE, label="flat pad at the shin tip"),
                           Patch(color=AQUA, label="flat sole (control frame)")],
                  loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Can it stand still? Same legs, different ground contact (CPU eval, 100 held-out seeds)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PLOTS / "contact_comparison.png")
    plt.close(fig)


# ------------------------------------------------------------------ training curves
def training_curves():
    # Round-1 standing curves (v0/v1) are omitted: they were logged with stochastic in-training evals (E5).
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), dpi=140)
    for run, label, col in [("ctl_stand_A8", "ball", BLUE), ("q2_stand_P20", "pad 20 mm", YELLOW),
                            ("q2_stand_P30", "pad 30 mm", ORANGE), ("ctl_stand_F60_12", "sole 60 mm", AQUA)]:
        st, ln, _ = progress(run)
        axs[0].plot(st, ln, color=col, lw=2, marker="o", ms=4, label=label)
    axs[0].legend(frameon=False, fontsize=8.5, loc="center right")
    axs[0].set_title("Control experiment: quiet-stand training", fontsize=10.5)
    axs[0].set_ylabel("eval episode length (s, cap 20)")
    st, _, rw = progress("walk_A8_v5_dr_s0")
    axs[1].plot(st, rw, color=BLUE, lw=2, marker="o", ms=4)
    i = int(np.nanargmax(rw))
    axs[1].annotate("rule-selected checkpoint", (st[i], rw[i]), xytext=(st[i] + 25, rw[i] + 250), color=INK,
                    fontsize=8.5, arrowprops=dict(arrowstyle="->", color=INK2))
    axs[1].annotate("late regression", (st[-1], rw[-1]), xytext=(st[-1] - 70, rw[-1] - 1300), color=INK, fontsize=8.5,
                    arrowprops=dict(arrowstyle="->", color=INK2))
    axs[1].set_ylim(0, rw[i] * 1.18)
    axs[1].set_title("Ball tip, walking v5: peak, then regression", fontsize=10.5)
    axs[1].set_ylabel("in-training eval reward")
    for ax in axs:
        ax.set_xlabel("environment steps (M)")
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(PLOTS / "training_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    concept()
    policy_progression()
    contact_comparison()
    training_curves()
    contact_candidates()
    print("ok")
