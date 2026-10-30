"""Electrical power and current drawn by the servos while a trained policy runs (CPU MuJoCo).

    .venv/bin/python scripts/power_audit.py <run> --params params_XXXXM.pkl --test walk40 --n 20 [--volts 7.4]

Model (same DC-motor curve as the servo model in slender_biped/actuator.py and the MJX env):
    tau = tau_stall * (u - qd/omega0)  ->  motor current  I_m = I_stall * tau / tau_stall
    supply current (PWM H-bridge, duty |u|):  I_s = u * I_m  while motoring (u * I_m > 0), else 0
      (braking/back-driven phases recirculate in the bridge; regeneration into the supply is ignored,
       which is conservative for wiring and supply sizing)
    + idle electronics per servo (I_IDLE).
Gearbox friction is not in the current model, so real currents are somewhat higher (see the doc).
Physics runs at 2 ms, so "peak" means a 2 ms average; the 20 ms window is what a supply/BEC sees.
Writes results/power/<run>__<test>.json and a current trace plot for the first episode.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

I_IDLE = 0.03   # A per servo: MCU + bus transceiver + encoder, torque enabled, no load (estimate)


def episode(args):
    run, params, test, seed = args
    from evaluate import TESTS
    from slender_biped.config import load_actuators
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, run_episode
    cfg, run_dir = load_run(run)
    ec = EvalConfig.from_train(cfg["env"], dr=False)     # nominal servos and masses
    env = CpuBiped(cfg["env"]["model"], ec, seed=10000 + seed)
    pol = NumpyPolicy(run_dir / params)
    t = TESTS[test]
    # run_episode keeps tau/qd/u_req at every physics step; re-run it with record so we can keep them all
    res, _ = run_episode(env, pol, t["seconds"], t["sched"], heading_hold=t.get("heading_hold", False), record=False)
    act = load_actuators()
    specs = [act["catalog"][act["assignments"][env.meta["cfg"]["actuator_set"]][j.split("_", 1)[1]]] for j in env.joints]
    return res, env._log, specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--test", default="walk40")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--volts", type=float, default=7.4)
    a = ap.parse_args()

    # monkey-patch CpuBiped.step to keep the full substep logs (run_episode keeps them only for summaries)
    from slender_biped import evaluation as ev
    orig = ev.CpuBiped.step

    def step(self, action):
        obs, log = orig(self, action)
        self._log.append(log)
        return obs, log
    ev.CpuBiped.step = step
    orig_reset = ev.CpuBiped.reset

    def reset(self, cmd=(0.0, 0.0, 0.0)):
        self._log = []
        return orig_reset(self, cmd)
    ev.CpuBiped.reset = reset

    with mp.get_context("fork").Pool(min(a.n, os.cpu_count() or 4)) as pool:
        out = pool.map(episode, [(a.run, a.params, a.test, s) for s in range(a.n)])

    specs = out[0][2]
    from slender_biped.evaluation import load_run
    cfg, _ = load_run(a.run)
    joints = json.loads((ROOT / "models" / f"{cfg['env']['model']}.json").read_text())["joints"]
    ts = np.array([s["stall_torque"] for s in specs])
    i_stall = np.array([s["stall_current"] for s in specs])
    w0 = np.array([s["noload_speed"] for s in specs])
    dt = cfg["env"]["sim_dt"]
    per_ep, all_I, all_P_mech, traces = [], [], [], []
    for res, logs, _ in out:
        if not logs:
            continue
        tau = np.concatenate([l["tau"] for l in logs])
        qd = np.concatenate([l["qd"] for l in logs])
        u = np.clip(np.concatenate([l["u_req"] for l in logs]), -1, 1)
        i_m = i_stall * tau / ts
        i_s = np.where(u * i_m > 0, u * i_m, 0.0) + I_IDLE
        all_I.append(i_s)
        all_P_mech.append(np.maximum(tau * qd, 0.0))
        traces.append(i_s)
        per_ep.append(dict(survived_s=res["survived_s"], fell=res["fell"], mean_vx=res["mean_vx"]))
    I = np.concatenate(all_I)                # (T, nj) supply current per servo, A
    tot = I.sum(1)
    win = int(round(0.02 / dt))
    tot20 = np.convolve(tot, np.ones(win) / win, mode="valid")
    Pm = np.concatenate(all_P_mech)
    V = a.volts
    rep = dict(
        run=a.run, params=a.params, test=a.test, n=len(per_ep), volts=V,
        alive_frac=float(np.mean([not e["fell"] for e in per_ep])), mean_vx=float(np.mean([e["mean_vx"] for e in per_ep])),
        per_servo={j: dict(mean_A=float(I[:, k].mean()), rms_A=float(np.sqrt((I[:, k] ** 2).mean())),
                           p99_A=float(np.percentile(I[:, k], 99)), peak_A=float(I[:, k].max()),
                           stall_A=float(i_stall[k]), mean_mech_W=float(Pm[:, k].mean()))
                   for k, j in enumerate(joints)},
        total=dict(mean_A=float(tot.mean()), rms_A=float(np.sqrt((tot ** 2).mean())), p99_A=float(np.percentile(tot, 99)),
                   peak_2ms_A=float(tot.max()), peak_20ms_A=float(tot20.max()), p99_20ms_A=float(np.percentile(tot20, 99)),
                   mean_W=float(tot.mean() * V), peak_20ms_W=float(tot20.max() * V),
                   mean_mech_W=float(Pm.sum(1).mean()), Wh_per_min=float(tot.mean() * V / 60.0),
                   all_stall_A=float(i_stall.sum())),
        model_note="I_supply = u*I_motor while motoring (no regen) + %.2f A idle/servo; gearbox friction not included" % I_IDLE,
    )
    outdir = ROOT / "results" / "power"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{a.run}__{a.test}"
    (outdir / f"{stem}.json").write_text(json.dumps(rep, indent=1))
    t = rep["total"]
    print(f"{stem}: n={rep['n']} alive {rep['alive_frac']:.2f} vx {rep['mean_vx']:.3f} | total @ {V} V: mean {t['mean_A']:.2f} A "
          f"({t['mean_W']:.1f} W), RMS {t['rms_A']:.2f} A, p99 {t['p99_A']:.2f} A, peak 20 ms {t['peak_20ms_A']:.2f} A, "
          f"peak 2 ms {t['peak_2ms_A']:.2f} A (all-stall {t['all_stall_A']:.1f} A); mech {t['mean_mech_W']:.2f} W")
    for j, s in rep["per_servo"].items():
        print(f"   {j:12s} mean {s['mean_A']:.2f}  rms {s['rms_A']:.2f}  p99 {s['p99_A']:.2f}  peak {s['peak_A']:.2f} A (stall {s['stall_A']:.1f})")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = traces[0]
        tt = np.arange(len(tr)) * dt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=110)
        ax[0].plot(tt, tr.sum(1), lw=0.6, color="0.3", label="total (2 ms)")
        k = np.ones(win) / win
        ax[0].plot(tt[win - 1:], np.convolve(tr.sum(1), k, mode="valid"), lw=1.2, color="C3", label="total (20 ms avg)")
        ax[0].set_ylabel("supply current (A)")
        ax[0].legend(loc="upper right")
        for kk, j in enumerate(joints[:4]):
            ax[1].plot(tt, np.convolve(tr[:, kk], k, mode="same"), lw=0.9, label=j)
        ax[1].set_ylabel("left-leg servo current (A, 20 ms avg)")
        ax[1].set_xlabel("time (s)")
        ax[1].legend(loc="upper right", ncol=4, fontsize=8)
        fig.suptitle(f"{a.run} {a.params} — {a.test} @ {V} V")
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}.png")
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
