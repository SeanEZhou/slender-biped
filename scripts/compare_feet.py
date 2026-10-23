"""A/B (or any) comparison table + survival histograms from existing eval files.

    .venv/bin/python scripts/compare_feet.py --pairs stand_A8_step_nodr_s0:stand_B8_step_nodr_s0 \
        --tests stand --out results/compare_stand_v1
Reads results/eval/<run>__<test>.json; writes <out>.md and <out>.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("alive_15s", "alive @15 s", "{:.0%}"), ("alive_30s", "alive @30 s", "{:.0%}"),
        ("survival_median_s", "median survival [s]", "{:.1f}"), ("steps_per_s_median", "steps / s", "{:.1f}"),
        ("stationary_frac", "stationary (0 steps)", "{:.0%}"), ("drift_median_m", "drift [m]", "{:.2f}"),
        ("walk_1.0m", "walk ≥1 m", "{:.0%}"), ("walk_2.0m", "walk ≥2 m", "{:.0%}"),
        ("mean_vx_median", "mean vx [m/s]", "{:.3f}"), ("startstop_success", "start/stop ok", "{:.0%}"),
        ("sat_frac_median", "duty saturation frac", "{:.3f}"), ("overspeed_events_total", "over-speed events", "{:d}"),
        ("max_tilt_median", "median peak tilt [deg]", "{:.1f}")]


def load(run, test):
    p = ROOT / "results" / "eval" / f"{run}__{test}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="runA:runB")
    ap.add_argument("--tests", nargs="+", default=["stand"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    md = []
    runs = []
    for p in a.pairs:
        runs += [r for r in p.split(":") if r not in runs]
    fig, axs = plt.subplots(1, len(a.tests), figsize=(5 * len(a.tests), 3.5), dpi=130, squeeze=False)
    for ti, test in enumerate(a.tests):
        data = {r: load(r, test) for r in runs}
        data = {r: d for r, d in data.items() if d}
        if not data:
            continue
        md.append(f"### {test}\n")
        md.append("| metric | " + " | ".join(data) + " |")
        md.append("|---|" + "---|" * len(data))
        for key, label, fmt in ROWS:
            vals = []
            for r, d in data.items():
                v = d["summary"].get(key)
                vals.append("—" if v is None else fmt.format(int(v) if fmt == "{:d}" else v))
            if all(v == "—" for v in vals):
                continue
            md.append(f"| {label} | " + " | ".join(vals) + " |")
        for r, d in data.items():
            s = d["summary"]
            md.append(f"\n`{r}` peak |τ|/stall per joint {s['peak_tau_frac_max']}, RMS |τ| [N·m] {s['rms_tau_median']}, RMS/rated {s.get('rms_over_rated_median')}, "
                      f"peak speed / no-load {s['speed_util_max']}")
        md.append("")
        ax = axs[0, ti]
        for r, d in data.items():
            surv = [e["survived_s"] for e in d["episodes"]]
            ax.hist(surv, bins=np.linspace(0, max(surv) + 1e-6, 16), alpha=0.5, label=r)
        ax.set_title(f"{test}: survival per episode (n={len(surv)})")
        ax.set_xlabel("seconds")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out + ".png")
    Path(a.out + ".md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
