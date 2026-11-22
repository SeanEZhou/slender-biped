"""Round 28: reproduce the robot's falls in the GPU 'sim robot'.

For every fallen trial of the given run(s) (results/robot28/falls.json): the checkpoint that ran (by its actions), then
from the log states at t_fall - dt (dt in --deltas) the fitted body velocity and --n randomized sim robots running the
same policy gate-free. A start reproduces the fall if more than half the walkers fall (tilt > 30 deg) within dt + 0.8 s
to the same side (roll sign; 'forward' = pitch). Controls: --controls log states per trial from steady walking at least
4 s before any fall; walkers should not fall from them.

Writes results/robot28/repro_<run>.json (per start: fall share, side shares, fitted velocity and fit errors).

    scripts/gpu_python.sh scripts/robot28/repro.py <run> [<run> ...] [--deltas 0.4 0.8 1.2 1.6 2.0 2.5] [--controls 4]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import robostate as rs
from push_eval import load_policy
from slender_biped.mjx_env import SlenderBipedEnv

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--deltas", type=float, nargs="+", default=[-0.6, -0.3, 0.0, 0.4, 0.8, 1.6])
ap.add_argument("--controls", type=int, default=4); ap.add_argument("--n", type=int, default=32); ap.add_argument("--T", type=float, default=3.4)
ap.add_argument("--policy", default=None, help="run[:params] to run instead of the logged one (evaluating a new policy)")
ap.add_argument("--files", nargs="*", default=None); ap.add_argument("--tag", default="")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
falls = json.load(open(rs.ROOT / "results/robot28/falls.json"))
D = rs.ROOT / "hardware/leg_p0/software"
rng = np.random.default_rng(0)

def setv(kv):
    k, v = kv.split("=", 1); x = json.loads(v)
    return k, (tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x)

SIMS = {}
for run in a.runs:
    trials = [r for r in falls if r["run"] == run and (a.files is None or r["file"] in a.files)]
    if not trials:
        continue
    _, _, d = load_policy(a.policy or run)
    if a.policy and "envs" in dir():
        pass                                     # one policy: the same envs and compiled sims for every run
    else:
        envs = {g_: SlenderBipedEnv(rs.robot_cfg(d, gate=g_, **dict(setv(kv) for kv in a.set))) for g_ in (False, True)}
    env = envs[False]
    env_log = env if a.policy is None else SlenderBipedEnv(rs.robot_cfg(load_policy(run)[2]))
    sims = SIMS if a.policy else {}
    out = []
    for r in trials:
        f = str(D / r["file"])
        if a.policy:
            pname = a.policy
        else:
            pk, perr = rs.which_params(run, f, env_log); pname = f"{run}:{pk}"
        L = rs.Log(f, env_log)
        gate = bool("gate_held" in L.h and L.A[:, L.h.index("gate_held")].max() > 0 and a.policy is None)
        if (pname, gate) not in sims:
            W, p, _ = load_policy(pname)
            sims[(pname, gate)] = (rs.Sim(envs[gate], W, a.n, a.T, dr=True), rs.Sim(envs[gate], W, len(rs.GRID), 0.2, dr=False))
        sim, fsim = sims[(pname, gate)]
        moving = np.where(np.abs(np.diff(L.ph)) > 1e-6)[0]
        if not len(moving):
            continue
        t_start = L.t[moving[0]] + 1.0
        starts = []
        if r["fell"]:
            starts += [("pre", dt, r["t_fall"] - dt) for dt in a.deltas if t_start < r["t_fall"] - dt < r["T"] - 0.25]
        t_ok_end = (r["t_fall"] - 4.0) if r["fell"] else L.t[-1] - 2.0
        if t_ok_end - t_start > 3.0 and a.controls:
            for tc in rng.uniform(t_start + 2.0, t_ok_end - a.T, a.controls) if t_ok_end - a.T > t_start + 2.0 else []:
                starts.append(("ctl", None, float(tc)))
        for kind, dt, t0 in starts:
            i = int(np.argmin(np.abs(L.t - t0)))
            V, e_fit, e_zero = rs.fit_velocity(fsim, L, i)
            R = sim.run(L.state(i), np.tile(V, (a.n, 1)))
            horizon = min(sim.nst, int(round(((r["T"] - L.t[i] if kind == "pre" else a.T) + 0.8) / env.cfg.ctrl_dt)))
            fell_k = (R["tilt"][:horizon] > 30)
            fell = fell_k.any(0)
            kf = np.argmax(fell_k, 0)
            rl = R["roll"][kf, np.arange(a.n)]; pt = R["pitch"][kf, np.arange(a.n)]
            side = np.where(~fell, "", np.where(np.abs(rl) >= np.abs(pt) * 0.8, np.where(rl < 0, "left", "right"),
                                                 np.where(pt > 0, "forward", "back")))
            sh = {s_: float(np.mean(side == s_)) for s_ in ("left", "right", "forward", "back")}
            rec = dict(file=r["file"], policy=pname, gate=gate, kind=kind, dt=dt, t0=float(L.t[i]), u=float(L.ph[i] / 2 / np.pi % 1),
                       roll=float(L.roll[i]), fall=float(fell.mean()), sides=sh, robot_side=r.get("side"),
                       t_fall_sim=float(np.median(kf[fell]) * env.cfg.ctrl_dt) if fell.any() else None,
                       V=V.tolist(), fit_err=e_fit, fit_err_v0=e_zero)
            rec["repro"] = bool(kind == "pre" and (sh.get(r.get("side"), 0.0) > 0.5 or
                                (r.get("side") == "forward" and fell.mean() > 0.5)))
            out.append(rec)
            print(f"{r['file'][6:-4]:28s} {kind} {'' if dt is None else f'{-dt:+.1f}s':6s} t {L.t[i]:5.1f} u {rec['u']:.2f} roll {L.roll[i]:+5.1f} "
                  f"| V {V[0]:+.2f},{V[1]:+.2f} fit {e_fit:.2f} (v0 {e_zero:.2f}) | fell {fell.mean()*100:3.0f}% "
                  + " ".join(f"{k_[0]}{v_*100:.0f}" for k_, v_ in sh.items() if v_) + (f"  robot {r.get('side')}" if kind == 'pre' else "")
                  + ("  REPRO" if rec["repro"] else ""), flush=True)
    tag = (a.tag or (a.policy.replace(":", "_") if a.policy else "logged"))
    json.dump(out, open(rs.ROOT / f"results/robot28/repro_{run}_{tag}.json", "w"), indent=1)
