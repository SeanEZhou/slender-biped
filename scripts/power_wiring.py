"""Current through each power link, and the voltage left at each servo, for different wiring layouts.

    .venv/bin/python scripts/power_wiring.py [--ohm-per-hop 0.1]

Servo currents use the power_audit.py model on its walk40 test (same seeds). A layout is a list of feeds, each one
a chain of servos: the link into a servo carries that servo and everything after it. Voltage drop is R x I summed
along the feed, with no feedback: real servos would slow down and draw less, so large drops only show it won't work.
Writes results/power/wiring.json and assets/power_wiring.png.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np  # noqa: E402

import power_audit as pa  # noqa: E402
from slender_biped import evaluation as ev  # noqa: E402

RUNS = (("r5p_flow", "params_0020M.pkl"), ("r5q_legs_path", "params_0030M.pkl"),
        ("r4k_cat2_ft", "params_0030M.pkl"), ("r4j_d65_c4_cont", "params_0020M.pkl"))
# joint order: L yaw, roll, pitch, knee, R yaw, roll, pitch, knee
LAYOUTS = {"single chain": [[0, 1, 2, 3, 4, 5, 6, 7]],
           "one branch per leg": [[0, 1, 2, 3], [4, 5, 6, 7]],
           "hips and knees fed separately": [[0, 1, 2], [3], [4, 5, 6], [7]]}
VOLTS, CONTACT_A, WIN = 7.4, 3.0, 10       # 10 x 2 ms = the 20 ms window a supply sees
PLOT_RUN, PLOT_T = "r5p_flow", (10.0, 14.0)
COLOURS = dict(zip(LAYOUTS, ("#eb6834", "#2a78d6", "#1baf7a")))


def avg20(x):
    return np.convolve(x, np.ones(WIN) / WIN, mode="valid")


def servo_currents(run, params):
    with mp.get_context("fork").Pool(min(20, os.cpu_count() or 4)) as pool:
        out = pool.map(pa.episode, [(run, params, "walk40", s) for s in range(20)])
    specs = out[0][2]
    ts, i_stall = np.array([s["stall_torque"] for s in specs]), np.array([s["stall_current"] for s in specs])
    eps = []
    for _, logs, _ in out:
        tau = np.concatenate([l["tau"] for l in logs])
        u = np.clip(np.concatenate([l["u_req"] for l in logs]), -1, 1)
        i_m = i_stall * tau / ts
        eps.append(np.where(u * i_m > 0, u * i_m, 0.0) + pa.I_IDLE)
    return eps


def links(ep, feed):
    return np.stack([ep[:, feed[k:]].sum(1) for k in range(len(feed))], 1)


def score(eps, layout, ohm):
    """Worst feed and worst servo of a layout."""
    first = [np.concatenate([links(e, f)[:, 0] for e in eps]) for f in layout]
    first20 = [np.concatenate([avg20(links(e, f)[:, 0]) for e in eps]) for f in layout]
    sag20 = [np.concatenate([avg20(ohm * links(e, f).sum(1)) for e in eps]) for f in layout]
    return dict(feed_rms_A=max(float(np.sqrt((a ** 2).mean())) for a in first),
                feed_peak_20ms_A=max(float(a.max()) for a in first20),
                feed_peak_2ms_A=max(float(a.max()) for a in first),
                feed_over_contact_pct=max(float(100 * (a > CONTACT_A).mean()) for a in first20),
                drop_p99_V=max(float(np.percentile(a, 99)) for a in sag20),
                drop_peak_V=max(float(a.max()) for a in sag20),
                drop_over_10pct_time_pct=max(float(100 * (a > 0.1 * VOLTS).mean()) for a in sag20))


def plot(ep, ohm, dt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i0, i1 = int(PLOT_T[0] / dt), int(PLOT_T[1] / dt)
    plt.rcParams.update({"font.size": 11, "text.color": "#0b0b0b", "axes.edgecolor": "#b5b4ad", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), dpi=150, facecolor="#fcfcfb")
    for name, layout in LAYOUTS.items():
        feed = max(layout, key=lambda f: links(ep, f)[:, 0].mean())
        cur = avg20(links(ep, feed)[:, 0])
        volts = VOLTS - np.max([avg20(ohm * links(ep, f).sum(1)) for f in layout], axis=0)
        t = np.arange(len(cur)) * dt
        ax[0].plot(t[i0:i1], cur[i0:i1], color=COLOURS[name], lw=1.6, label=name)
        ax[1].plot(t[i0:i1], volts[i0:i1], color=COLOURS[name], lw=1.6, label=name)
    ax[0].axhline(CONTACT_A, color="#52514e", lw=1, ls="--")
    ax[1].axhline(0.9 * VOLTS, color="#52514e", lw=1, ls="--")
    ax[0].set_title("Current through the busiest feed connector", loc="left", fontsize=12)
    ax[1].set_title("Estimated voltage at the worst servo", loc="left", fontsize=12)
    ax[0].set_ylabel("A, 20 ms average")
    ax[1].set_ylabel("V, 20 ms average")
    ax[0].set_ylim(0, 6.2)
    ax[1].set_ylim(3.6, VOLTS + 0.7)
    ax[0].text(PLOT_T[1], 5.75, "dashed: connector rating, 3 A", ha="right", color="#52514e", fontsize=10)
    ax[1].text(PLOT_T[1], 7.72, "dashed: 10 % below 7.4 V", ha="right", color="#52514e", fontsize=10)
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.set_xlabel("time walking (s)")
        a.set_xlim(*PLOT_T)
        a.grid(axis="y", color="#e6e5e0", lw=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    hand, lab = ax[0].get_legend_handles_labels()
    fig.legend(hand, lab, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohm-per-hop", type=float, default=0.1, help="estimate: one lead, out and back, with its contacts")
    a = ap.parse_args()

    # same logging patch as power_audit.py: keep the full substep logs
    step0, reset0 = ev.CpuBiped.step, ev.CpuBiped.reset

    def step(self, action):
        obs, log = step0(self, action)
        self._log.append(log)
        return obs, log

    def reset(self, cmd=(0.0, 0.0, 0.0)):
        self._log = []
        return reset0(self, cmd)
    ev.CpuBiped.step, ev.CpuBiped.reset = step, reset

    dt = ev.load_run(PLOT_RUN)[0]["env"]["sim_dt"]
    rep = dict(ohm_per_hop=a.ohm_per_hop, volts=VOLTS, runs={})
    for run, params in RUNS:
        eps = servo_currents(run, params)
        rep["runs"][run] = {name: score(eps, layout, a.ohm_per_hop) for name, layout in LAYOUTS.items()}
        if run == PLOT_RUN:
            plot(eps[0], a.ohm_per_hop, dt, ROOT / "assets" / "power_wiring.png")
    (ROOT / "results" / "power" / "wiring.json").write_text(json.dumps(rep, indent=1))
    for run, layouts in rep["runs"].items():
        print(run)
        for name, s in layouts.items():
            print(f"  {name:30s} " + "  ".join(f"{k} {v:.2f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
