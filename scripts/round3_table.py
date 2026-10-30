"""Markdown tables of the round-3 (hardware model) results from results/eval and results/power.

    .venv/bin/python scripts/round3_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV, PW = ROOT / "results" / "eval", ROOT / "results" / "power"


def summ(run, test, ckpt):
    p = EV / f"{run}__{test}_{ckpt}.json"
    return json.loads(p.read_text())["summary"] if p.exists() else None


def power(run, test):
    p = PW / f"{run}__{test}.json"
    return json.loads(p.read_text())["total"] if p.exists() else None


def ckpt(run):
    f = ROOT / "results" / "eval_hw" / f"{run}.txt"
    return f.read_text().split()[-1] if f.exists() and f.read_text().startswith("run") else None


def walk_rows(runs):
    print("| run | model | 2 m in 40 s | vx @ 0.15 | steps/s | start/stop | overspeed events | RMS/rated max | walk A mean / p99 / 20 ms peak |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        c = ckpt(r)
        if not c:
            continue
        c = (ROOT / "results" / "eval_hw" / f"{r}.txt").read_text().splitlines()[0].split()[-1]
        w, s = summ(r, "walk40", c), summ(r, "startstop", c)
        pw = power(r, "walk40")
        model = json.loads((ROOT / "runs" / r / "config.json").read_text())["env"]["model"]
        if not w:
            continue
        print(f"| {r} @{c[7:11].lstrip('0')}M | {model} | {w['walk_2.0m']:.0%} | {w['mean_vx_median']:.3f} | {w['steps_per_s_median']:.1f} | "
              f"{s['startstop_success']:.0%} | {w['overspeed_events_total']} | {max(w['rms_over_rated_median']):.2f} | "
              + (f"{pw['mean_A']:.2f} / {pw['p99_A']:.2f} / {pw['peak_20ms_A']:.1f}" if pw else "–") + " |")


def stand_rows(runs):
    print("| run | model | alive 30 s | no-lift stance | drift | heading | push: median survival (20 s) | stand A mean / 20 ms peak |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        f = ROOT / "results" / "eval_hw" / f"{r}.txt"
        if not f.exists() or not f.read_text().startswith("run"):
            continue
        c = f.read_text().splitlines()[0].split()[-1]
        st, pu = summ(r, "stand", c), summ(r, "push", c)
        pw = power(r, "stand")
        model = json.loads((ROOT / "runs" / r / "config.json").read_text())["env"]["model"]
        print(f"| {r} @{c[7:11].lstrip('0')}M | {model} | {st['alive_30s']:.0%} | {st['nolift_frac']:.0%} | {st['drift_median_m']:.2f} m | "
              f"{st.get('heading_change_abs_median_deg', 0):.0f}° | " + (f"{pu['survival_median_s']:.1f} s" if pu else "–") + " | "
              + (f"{pw['mean_A']:.2f} / {pw['peak_20ms_A']:.1f}" if pw else "–") + " |")


if __name__ == "__main__":
    runs = sorted(p.stem for p in (ROOT / "results" / "eval_hw").glob("*.txt"))
    walk_rows([r for r in runs if "walk" in r])
    print()
    stand_rows([r for r in runs if "stand" in r])
