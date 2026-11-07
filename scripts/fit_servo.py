"""Fit the simulator's servo model to hardware step tests (hardware/leg_p0/software/policy_stand.py step-test).

    .venv-hw/bin/python scripts/fit_servo.py [--quick]

Each recorded step test is replayed in the CPU simulator (slender_biped/evaluation.py, the same servo code as the
MJX training env) at 2 ms resolution: the tested joint steps +A, 0, -A, 0 from the stance pose while the others hold,
either with the pelvis held fixed ("air") or standing on the pads ("stand"). A grid over the servo parameters
(command dead time, firmware motion-profile acceleration, position gain kp) is scored by the RMS error between the
simulated and measured joint trajectories, separately for hips (1:345) and knees (1:191). The supply is fixed by
--volt (strength and speed scale; 5 V on the 7.4 V servos ~ 0.68).

Writes results/servo_fit.json and results/servo_fit.png (measured vs fitted vs the original servo model).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slender_biped.evaluation import CpuBiped, EvalConfig  # noqa: E402

SW = ROOT / "hardware/leg_p0/software"
# step-test logs from 2026-11-06 (5 V bench supply, servo P gain 32). The acceleration register (254 vs 0) did not
# change the hips, so their ACC-0 runs count too; knees only from ACC 254 (their ACC-0 behaviour differs per servo).
DATA = [
    ("steptest_air_20261106_113322.csv", "air", 254),
    ("steptest_air_20261106_113757.csv", "air", 254),
    ("steptest_air_20261106_112945.csv", "air", 0),
    ("steptest_stand_20261106_113021.csv", "stand", 0),
]
JOINTS = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee"]
NOM = np.radians([0, 0, 16, 32] * 2)
DT = 0.002


def load_segments():
    """[(file, mode, accel_reg, joint, [(target_offset_deg, t[], q_deg[]), ...]), ...] in test order."""
    out = []
    for fn, mode, acc in DATA:
        p = SW / fn
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p)))
        by_joint = {}
        for r in rows:
            by_joint.setdefault(r["joint"], []).append(r)
        for jn, rs in by_joint.items():
            segs, cur = [], None
            for r in rs:
                tgt, t = float(r["target_offset_deg"]), float(r["t"])
                if cur is None or tgt != cur[0] or t < cur[1][-1]:
                    cur = (tgt, [], [])
                    segs.append(cur)
                cur[1].append(t)
                cur[2].append(float(r[f"q_{jn}_deg"]))
            out.append((fn, mode, acc, jn, [(a, np.array(b), np.array(c)) for a, b, c in segs]))
    return out


def simulate(mode, jn, segs, params, volt):
    """Simulated trajectory of joint jn (deg) at the measured sample times, for the measured step sequence."""
    delay_ms, acc, kp = params
    cfg = EvalConfig(ctrl_dt=DT, sim_dt=DT, kp=kp, latency_sub=0, delay_steps=int(round(delay_ms / (DT * 1000))),
                     prof_acc=(acc, acc) if acc > 0 else (0.0, 0.0), strength_scale=volt, speed_scale=volt,
                     obs_noise=0.0, reset_joint_noise=0.0, reset_tilt=0.0)
    env = CpuBiped("hw_p0_8dof", cfg, seed=0)
    env.reset()
    k = JOINTS.index(jn)
    d = env.d
    pose0 = d.qpos[0:7].copy()
    if mode == "air":                           # held up: lift the pelvis so the pads hang clear of the floor
        pose0[2] += 0.15
        d.qpos[0:7] = pose0

    def step(target_rad):
        env.step((target_rad - env.q_nom) / cfg.action_scale)
        if mode == "air":
            d.qpos[0:7] = pose0
            d.qvel[0:6] = 0.0

    tgt = NOM.copy()
    for _ in range(int(0.6 / DT)):              # settle at stance
        step(tgt)
    sims = []
    for off, t, _q in segs:
        tgt = NOM.copy()
        tgt[k] = NOM[k] + math.radians(off)
        n = int(math.ceil(t[-1] / DT)) + 1
        ts, qs = [], []
        for i in range(n):
            step(tgt)
            ts.append((i + 1) * DT)
            qs.append(math.degrees(d.qpos[7 + k]))
        sims.append(np.interp(t, ts, qs))
    return sims


def score(args):
    seg_sets, params, volt = args
    err, n = 0.0, 0
    for mode, jn, segs in seg_sets:
        try:
            sims = simulate(mode, jn, segs, params, volt)
        except Exception:
            return params, float("inf")
        for (off, t, q), s in zip(segs, sims):
            err += float(np.sum((s - q) ** 2))
            n += len(q)
    return params, math.sqrt(err / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volt", type=float, default=0.68, help="strength and speed scale (5 V / 7.4 V)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    data = load_segments()
    groups = {
        "hip": [(m, jn, segs) for fn, m, acc, jn, segs in data if "hip" in jn],
        "knee": [(m, jn, segs) for fn, m, acc, jn, segs in data if "knee" in jn and acc == 254],
    }
    grid = dict(delay_ms=[0, 10, 20, 30, 40, 50, 60, 80], acc=[0, 3, 5, 7, 9, 12, 20, 35, 60],
                kp=[2, 4, 8, 16])
    if a.quick:
        grid = dict(delay_ms=[20, 40], acc=[0, 7, 35], kp=[4, 8])
    combos = list(itertools.product(grid["delay_ms"], grid["acc"], grid["kp"]))
    result = {"volt_scale": a.volt, "grid": grid, "data": [d[0] for d in DATA]}
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for g, sets in groups.items():
            print(f"{g}: {len(sets)} joint tests, {len(combos)} parameter sets", flush=True)
            scores = pool.map(score, [(sets, c, a.volt) for c in combos])
            scores.sort(key=lambda x: x[1])
            base = score((sets, (0, 0, 8.0), 1.0))[1]           # the original idealized servo at 7.4 V
            base5 = score((sets, (0, 0, 8.0), a.volt))[1]        # ... at 5 V
            best, rms = scores[0]
            print(f"  best delay {best[0]} ms, profile acc {best[1]} rad/s^2 ({math.degrees(best[1]):.0f} deg/s^2), "
                  f"kp {best[2]}: RMS {rms:.2f} deg   (original model {base:.2f} deg, at 5 V {base5:.2f} deg)")
            print("  next best:", [(p, round(s, 2)) for p, s in scores[1:6]])
            result[g] = dict(delay_ms=best[0], acc=best[1], kp=best[2], rms_deg=rms, rms_original=base,
                             rms_original_5v=base5, top=[[list(p), s] for p, s in scores[:15]])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/servo_fit.json").write_text(json.dumps(result, indent=1))
    plot(data, result, a.volt)


def plot(data, result, volt):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    picks = [d for d in data if d[0] == DATA[0][0] and d[3] in ("L_hip_pitch", "R_hip_roll", "L_knee", "R_knee")]
    picks += [d for d in data if d[0] == DATA[3][0] and d[3] in ("L_hip_pitch", "R_hip_roll")]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.2 * len(picks)), squeeze=False)
    for ax, (fn, mode, acc, jn, segs) in zip(axes[:, 0], picks):
        g = "knee" if "knee" in jn else "hip"
        fit = (result[g]["delay_ms"], result[g]["acc"], result[g]["kp"])
        sim_fit = simulate(mode, jn, segs, fit, volt)
        sim_old = simulate(mode, jn, segs, (0, 0, 8.0), 1.0)
        t0 = 0.0
        for i, (off, t, q) in enumerate(segs):
            ax.plot(t0 + t, q, "k", lw=1.5, label="measured" if i == 0 else None)
            ax.plot(t0 + t, sim_fit[i], "C0", lw=1.2, label="fitted model" if i == 0 else None)
            ax.plot(t0 + t, sim_old[i], "C3--", lw=1.0, label="original model (training)" if i == 0 else None)
            t0 += t[-1]
        ax.set_title(f"{jn}, {mode}, {fn[9:24]}", fontsize=9)
        ax.set_ylabel("deg")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(ROOT / "results/servo_fit.png", dpi=110)
    print("plot: results/servo_fit.png")


if __name__ == "__main__":
    main()
