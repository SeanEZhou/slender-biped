"""Acceptance-gate evaluation on CPU MuJoCo (full contact model), deterministic seeds 10000+i.

    .venv/bin/python scripts/evaluate.py <run_name> --test stand --n 100 [--dr] [--params params.pkl]
tests: stand (30 s, zero command) | walk (20 s @ 0.15 m/s) | startstop (fwd 6 s, stop 6 s, fwd 6 s)
       | push (stand 20 s with 0.3 m/s kicks every 2-4 s)
Writes results/eval/<run>__<test>[_dr].json and a survival plot. All episodes are kept.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import os
from pathlib import Path

# CPU-only: never touch the GPU from evaluation (training owns it). Params unpickling imports JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

TESTS = {
    "stand": dict(seconds=30.0, sched=[(0.0, (0.0, 0.0, 0.0))]),
    "walk": dict(seconds=20.0, sched=[(0.0, (0.15, 0.0, 0.0))], heading_hold=True),
    "walk40": dict(seconds=40.0, sched=[(0.0, (0.15, 0.0, 0.0))], heading_hold=True),  # 2 m gate without a 20 s cap
    "walk05": dict(seconds=20.0, sched=[(0.0, (0.05, 0.0, 0.0))], heading_hold=True),   # power vs speed (round 3)
    "walk10": dict(seconds=20.0, sched=[(0.0, (0.10, 0.0, 0.0))], heading_hold=True),
    "walk20": dict(seconds=20.0, sched=[(0.0, (0.20, 0.0, 0.0))], heading_hold=True),
    "walkpush": dict(seconds=40.0, sched=[(0.0, (0.15, 0.0, 0.0))], heading_hold=True, push=0.2),  # robustness (round 3)
    "walk_open": dict(seconds=20.0, sched=[(0.0, (0.15, 0.0, 0.0))]),  # no heading hold
    "startstop": dict(seconds=18.0, sched=[(0.0, (0.15, 0.0, 0.0)), (6.0, (0.0, 0.0, 0.0)), (12.0, (0.15, 0.0, 0.0))],
                      heading_hold=True),
    "turn": dict(seconds=10.0, sched=[(0.0, (0.0, 0.0, 0.3))]),   # ideal: +172 deg heading change
    "push": dict(seconds=20.0, sched=[(0.0, (0.0, 0.0, 0.0))], push=0.3),
}


def _worker(args):
    run, params, test, seed, dr, over = args
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, run_episode
    cfg, run_dir = load_run(run)
    env_cfg = cfg["env"]
    t = TESTS[test]
    over = dict(over)
    model = over.pop("model", env_cfg["model"])   # --set model='"<name>"': same policy on another model (sensitivity)
    ec = EvalConfig.from_train(env_cfg, **{"dr": dr, "push_vel": t.get("push", 0.0), **over})
    env = CpuBiped(model, ec, seed=10000 + seed)
    pol = NumpyPolicy(run_dir / params)
    res, _ = run_episode(env, pol, t["seconds"], t["sched"], record=(test == "startstop"),
                         heading_hold=t.get("heading_hold", False))
    res["seed"] = 10000 + seed
    if test == "startstop":
        tr = res.pop("trace")
        vx = np.array(tr["vx"])
        tt = np.arange(len(vx)) * ec.ctrl_dt
        def win(a, b):
            m = (tt >= a) & (tt < b)
            return float(vx[m].mean()) if m.any() else float("nan")
        res.update(v_go1=win(2, 6), v_stop=win(9, 12), v_go2=win(14, 18))
        res["startstop_ok"] = (not res["fell"]) and res["v_go1"] > 0.05 and abs(res["v_stop"]) < 0.04 and res["v_go2"] > 0.05
    if test.startswith("walk"):
        res["milestones"] = {str(mm): (res["forward_m"] >= mm) for mm in (0.25, 0.5, 1.0, 2.0)}
    return res


def summarize(eps, test, ts_stall=None):
    surv = np.array([e["survived_s"] for e in eps])
    out = {"n": len(eps), "survival_median_s": float(np.median(surv)), "survival_p10_s": float(np.percentile(surv, 10))}
    for s_ in (5, 15, 30):
        out[f"alive_{s_}s"] = float(np.mean(surv >= s_ - 1e-6))
    out["fall_rate"] = float(np.mean([e["fell"] for e in eps]))
    alive = [e for e in eps if not e["fell"]]
    pick = alive or eps
    out["touchdowns_median"] = float(np.median([e["touchdowns"] for e in pick]))
    out["steps_per_s_median"] = float(np.median([e["steps_per_s"] for e in pick]))
    out["bounces_median"] = float(np.median([e.get("bounces", 0) for e in pick]))
    out["stationary_frac"] = float(np.mean([e["touchdowns"] == 0 and not e["fell"] for e in eps]))
    if "liftoffs_after_settle" in eps[0]:
        # quiet stance: survived, no tip/sole lift-off after 2 s, < 2 cm total horizontal tip travel
        out["quiet_frac"] = float(np.mean([(not e["fell"]) and e["liftoffs_after_settle"] == 0 and e["tip_travel_m"] < 0.02
                                           for e in eps]))
        out["nolift_frac"] = float(np.mean([(not e["fell"]) and e["liftoffs_after_settle"] == 0 for e in eps]))
        out["liftoffs_median"] = float(np.median([e["liftoffs_after_settle"] for e in pick]))
        out["tip_travel_median_m"] = float(np.median([e["tip_travel_m"] for e in pick]))
        out["base_speed_rms_median"] = float(np.median([e["base_speed_rms"] for e in pick]))
        out["step_length_median_m"] = float(np.median([e["step_length_m"] for e in pick]))
    out["drift_median_m"] = float(np.median([e["drift_m"] for e in pick]))
    out["forward_median_m"] = float(np.median([e["forward_m"] for e in eps]))
    out["mean_vx_median"] = float(np.median([e["mean_vx"] for e in pick]))
    out["peak_tau_frac_max"] = np.max([e["peak_tau_frac"] for e in eps], 0).round(3).tolist()
    out["rms_tau_median"] = np.median([e["rms_tau"] for e in eps], 0).round(3).tolist()  # N*m
    if "rms_tau_frac" in eps[0]:
        out["rms_tau_frac_median"] = np.median([e["rms_tau_frac"] for e in eps], 0).round(3).tolist()
        out["rms_over_rated_median"] = np.median([e["rms_over_rated"] for e in eps], 0).round(3).tolist()
    out["speed_util_max"] = np.max([e["speed_util"] for e in eps], 0).round(3).tolist()
    out["sat_events_total"] = int(sum(e["sat_events"] for e in eps))
    out["sat_frac_median"] = float(np.median([e["sat_frac"] for e in eps]))
    out["overspeed_events_total"] = int(sum(e["overspeed_events"] for e in eps))
    out["max_tilt_median"] = float(np.median([e["max_tilt_deg"] for e in eps]))
    out["energy_proxy_median"] = float(np.median([e["energy_proxy"] for e in eps]))
    out["heading_change_abs_median_deg"] = float(np.median([abs(e.get("heading_change_deg", 0.0)) for e in eps]))
    if test.startswith("walk"):
        for mm in ("0.25", "0.5", "1.0", "2.0"):
            out[f"walk_{mm}m"] = float(np.mean([e["milestones"][mm] for e in eps]))
    if test == "startstop":
        out["startstop_success"] = float(np.mean([e["startstop_ok"] for e in eps]))
        for k in ("v_go1", "v_stop", "v_go2"):
            out[k + "_median"] = float(np.nanmedian([e[k] for e in eps]))
    return out


def evaluate(run, test="stand", n=100, dr=False, params="params.pkl", procs=8, over=None, tag=""):
    over = over or {}
    with mp.get_context("spawn").Pool(procs) as pool:
        eps = pool.map(_worker, [(run, params, test, i, dr, over) for i in range(n)])
    summ = summarize(eps, test)
    out_dir = ROOT / "results" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{run}__{test}{'_dr' if dr else ''}{tag}"
    (out_dir / f"{name}.json").write_text(json.dumps({"run": run, "test": test, "dr": dr, "params": params,
                                                     "overrides": over, "summary": summ, "episodes": eps}, indent=1))
    return summ, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--test", default="stand", choices=list(TESTS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--set", nargs="*", help="EvalConfig overrides k=json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    over = {k: json.loads(v) for k, v in (kv.split("=") for kv in (a.set or []))}
    summ, _ = evaluate(a.run, a.test, a.n, a.dr, a.params, a.procs, over, a.tag)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
