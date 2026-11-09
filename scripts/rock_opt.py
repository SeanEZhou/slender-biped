"""Tune the deterministic rocking walker (slender_biped/rock_walk.py) in the CPU simulator.

    .venv/bin/python scripts/rock_opt.py --gens 40 --out results/rock_walk.json [--model hw_pad_board150w_legs_8dof]

A small evolution strategy (rank-weighted mean, step size adapted on success) over the controller's parameters, scored
on every candidate in the same randomized conditions: five floors from hard to a rug the board sinks ~12 mm into, two
servo / mass randomizations each, the robot's IMU filter. Score per episode: staying up, real foot lift (swing
clearance up to 15 mm, only if it actually steps), forward travel and heading hold. Standing still scores 4 of a
possible ~13 + forward travel: most of the survival credit needs about 0.8 steps per second.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

BOUNDS = dict(T=(0.5, 1.6), a_len=(0.0, 0.3), a_roll=(-0.15, 0.15), psi=(-np.pi, np.pi), hip_ratio=(0.0, 1.0),
              lift=(0.0, 0.7), lift_c=(0.2, 0.8), lift_w=(0.2, 0.9), step=(0.0, 0.15), k_p=(-1.5, 1.5),
              k_pd=(-0.3, 0.3), roll_target=(0.02, 0.15), k_amp=(0.0, 20.0))
KEYS = list(BOUNDS)
LO = np.array([BOUNDS[k][0] for k in KEYS]); HI = np.array([BOUNDS[k][1] for k in KEYS])
FLOORS = (0.004, 0.1, 0.3, 0.8, 1.2)


def decode(z):
    return {k: float(v) for k, v in zip(KEYS, LO + (HI - LO) * np.clip(z, 0, 1))}


def score_one(args):
    import rock_eval
    params, model, seed, tc, seconds, w_fwd = args
    r = rock_eval.episode(params, model, seed, tc, seconds=seconds)
    # standing still is the easy local optimum (scores 4): most of the survival credit needs ~0.8 steps/s
    stepping = min(1.0, r["steps"] / (0.8 * seconds))
    s = r["survived"] / seconds * (4.0 + 6.0 * stepping) - (5.0 if r["fell"] else 0.0)
    s += 3.0 * min(r["clear_mm"], 15.0) / 15.0 * stepping
    s += w_fwd * max(r["fwd"], 0.0) * (not r["fell"]) - 0.03 * abs(r["yaw_deg"])
    return s, r


def evaluate(pool, cands, model, seconds, w_fwd, gen):
    jobs = [(decode(z), model, 50000 + 97 * gen + s, tc, seconds, w_fwd) for z in cands for tc in FLOORS for s in range(2)]
    out = pool.map(score_one, jobs)
    per = len(FLOORS) * 2
    return [(float(np.mean([o[0] for o in out[i * per:(i + 1) * per]])), [o[1] for o in out[i * per:(i + 1) * per]])
            for i in range(len(cands))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hw_pad_board150w_legs_8dof")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--w-fwd", type=float, default=30.0, help="score per metre of forward travel")
    ap.add_argument("--init", help="start from this result json")
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--out", default="results/rock_walk.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    from slender_biped.rock_walk import DEFAULT
    p0 = json.loads(Path(a.init).read_text())["params"] if a.init else DEFAULT
    mean = np.clip(np.array([(p0[k] - LO[i]) / (HI[i] - LO[i]) for i, k in enumerate(KEYS)]), 0, 1)
    sigma = a.sigma
    mu = a.pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
    best = (-1e9, None, None)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(min(12, os.cpu_count() or 4)) as pool:
        for gen in range(a.gens):
            t0 = time.time()
            eps = rng.normal(size=(a.pop, len(KEYS)))
            cands = np.clip(mean + sigma * eps, 0, 1)
            cands[0] = mean                                         # re-score the mean every generation
            res = evaluate(pool, cands, a.model, a.seconds, a.w_fwd, gen)
            sc = np.array([r[0] for r in res])
            order = np.argsort(-sc)
            new_mean = np.clip(np.sum(w[:, None] * cands[order[:mu]], 0), 0, 1)
            sigma *= 1.05 if sc[order[0]] > sc[0] else 0.93         # grow while the mean is beaten, shrink otherwise
            sigma = float(np.clip(sigma, 0.02, 0.4))
            mean = new_mean
            i = order[0]
            if sc[i] > best[0]:
                best = (float(sc[i]), decode(cands[i]), res[i][1])
                eps_r = best[2]
                out.write_text(json.dumps(dict(model=a.model, score=best[0], params=best[1], gen=gen,
                                               episodes=[{k: v for k, v in r.items() if k != "trace"} for r in eps_r]), indent=1))
            r = res[i][1]
            print(f"gen {gen:3d}  best {sc[i]:7.2f}  mean-cand {sc[0]:7.2f}  sigma {sigma:.3f}  | up {sum(not e['fell'] for e in r)}/{len(r)}"
                  f"  fwd {np.median([e['fwd'] for e in r]):+.2f} m  steps {np.median([e['steps'] for e in r]):.0f}"
                  f"  clear {np.median([e['clear_mm'] for e in r]):.0f} mm  ({time.time() - t0:.0f} s)", flush=True)
    print("best", json.dumps(best[1]))


if __name__ == "__main__":
    main()
