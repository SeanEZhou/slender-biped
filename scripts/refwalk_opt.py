"""Tune the deterministic 10-joint walker (slender_biped/rock_walk.py RefWalk: a reference table + IMU feedback on the
hip and ankle rolls and the hip pitches) in the CPU simulator, 2026-11-10.

    .venv/bin/python scripts/refwalk_opt.py [--table 0] [--vx 0.02] [--gens 25] [--pop 16] [--procs 10]

Cross-entropy search over k_r, k_w, ank, k_p, k_pd. Each candidate walks 15 s on 4 randomized robots (servo, mass,
friction as in training; the robot's IMU filter) x 2 floors (hard, medium rug). Score per walk: falls cost 10 (less
the time survived), otherwise +5, + 20 x forward metres, - 0.05 x RMS roll and pitch rate (deg/s): steady, not just
upright. Writes results/refwalk_<table>.json (loaded by policy_stand.py `run --run refwalk`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

KEYS = ("k_r", "k_w", "ank", "k_p", "k_pd")
SECONDS = 15.0
DT = 0.02
WOB = 0.05


def make_env(tc, seed):
    from slender_biped.evaluation import CpuBiped, EvalConfig, load_run
    cfg, _ = load_run("a3s_b150")
    over = dict(floor_tc=tc, action_scale=1.0, ref_table="", ref_residual=False, reset_joint_noise=0.0, reset_tilt=0.0,
                dr=True, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45)
    return CpuBiped("hw_ank3_b150_legs_10dof", EvalConfig.from_train(cfg["env"], **over), seed=seed)


def walk(args):
    params, tc, seed = args
    from slender_biped.rock_walk import RefWalk
    env = make_env(tc, seed)
    ctl = RefWalk(params)
    obs = env.reset((0.0, 0.0, 0.0))
    x0 = env.d.qpos[0]
    rates = []
    n = int(SECONDS / DT)
    for k in range(n):
        obs, _ = env.step(np.clip(ctl(obs), -1, 1))
        rates.append(np.degrees(env.d.qvel[3:5]))
        if env.fallen():
            return dict(fell=True, t=(k + 1) * DT, fwd=float(env.d.qpos[0] - x0), wob=99.0)
    r = np.array(rates[int(1.5 / DT):])
    return dict(fell=False, t=SECONDS, fwd=float(env.d.qpos[0] - x0), wob=float(np.sqrt(np.mean(r ** 2, 0)).sum()))


def score(res):
    s = 0.0
    for r in res:
        s += (-10.0 + 5.0 * r["t"] / SECONDS) if r["fell"] else (5.0 + 20.0 * r["fwd"] - WOB * r["wob"])
    return s / len(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, default=0)
    ap.add_argument("--vx", type=float, default=0.02)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--wob", type=float, default=0.05, help="score cost per deg/s of RMS roll + pitch rate")
    ap.add_argument("--tag", default="", help="suffix of the results file")
    a = ap.parse_args()
    global WOB
    WOB = a.wob
    from slender_biped.rock_walk import RefWalk
    tab = RefWalk.TABLES[a.table]
    base = dict(table=float(a.table), vx=a.vx)
    jobs = [(tc, s) for tc in (0.004, 0.3) for s in range(a.seeds)]
    mu = np.zeros(len(KEYS)); mu[KEYS.index("ank")] = 1.0
    sd = np.array([0.5, 0.1, 0.8, 0.5, 0.1])
    best = (-1e9, None, None)
    rng = np.random.default_rng(0)
    with Pool(a.procs) as pool:
        r0 = pool.map(walk, [(dict(base, **dict(zip(KEYS, mu))), tc, s) for tc, s in jobs])
        print(f"open loop + ank 1: score {score(r0):+.2f}  falls {sum(r['fell'] for r in r0)}/{len(r0)}  "
              f"fwd {np.mean([r['fwd'] for r in r0]) * 100:.1f} cm", flush=True)
        for g in range(a.gens):
            X = mu + sd * rng.standard_normal((a.pop, len(KEYS)))
            X[0] = mu
            cands = [dict(base, **dict(zip(KEYS, x))) for x in X]
            res = pool.map(walk, [(c, tc, s) for c in cands for tc, s in jobs])
            per = [res[i * len(jobs):(i + 1) * len(jobs)] for i in range(a.pop)]
            sc = np.array([score(p) for p in per])
            order = np.argsort(-sc)
            elite = X[order[: max(2, a.pop // 4)]]
            mu = elite.mean(0); sd = np.maximum(0.7 * sd + 0.3 * elite.std(0), 0.01)
            i = order[0]
            if sc[i] > best[0]:
                best = (float(sc[i]), cands[i], per[i])
            pb = per[i]
            print(f"gen {g:2d}: best {sc[i]:+.2f} (falls {sum(r['fell'] for r in pb)}/{len(pb)}, fwd "
                  f"{np.mean([r['fwd'] for r in pb]) * 100:.1f} cm, wobble {np.median([r['wob'] for r in pb]):.1f} deg/s) "
                  f"mean {sc.mean():+.2f}  mu {dict(zip(KEYS, np.round(mu, 3)))}", flush=True)
    s, p, pb = best
    out = ROOT / "results" / f"refwalk_{tab}{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(params=p, score=s, falls=sum(r["fell"] for r in pb), walks=len(pb),
                                   fwd_cm=round(float(np.mean([r["fwd"] for r in pb])) * 100, 1),
                                   wobble=round(float(np.median([r["wob"] for r in pb])), 1),
                                   note="scripts/refwalk_opt.py: 15 s walks, 4 randomized robots x hard floor + rug"), indent=1))
    print("best", s, p, "->", out)


if __name__ == "__main__":
    main()
