"""Round 28: a reset bank of robot states (models/bank/<name>.npz) for mjx_env reset_bank.

From the Nov 19-20 logs of the 1.2 s-step policies (c_K2gr / c_K2ip / c_T12HH): 'bad' states every 0.1 s from 1.2 s
before each fall's onset to 0.3 s before the log ends (the stuck double supports and the terrible steps), and 'normal'
states every 0.7 s of steady walking. Each with its body velocity fitted by shooting with the logged policy (gate as it
ran). The trials in results/robot28/holdout.json (every third fall) are left out for testing. The bad states are repeated so they make about half the bank.

    scripts/gpu_python.sh scripts/robot28/build_bank.py [--name robot28] [--runs ...]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import robostate as rs
from push_eval import load_policy
from slender_biped.mjx_env import SlenderBipedEnv

ap = argparse.ArgumentParser(); ap.add_argument("--name", default="robot28")
ap.add_argument("--runs", nargs="*", default=["z7_anc2", "z0_y2c", "z6_e", "z7_rgq", "z7_var", "z5_shv10", "z5_shv2", "k5_high2"])
ap.add_argument("--bad_share", type=float, default=0.5)
ap.add_argument("--fitpol", default="z7_anc2:params_0012M.pkl", help="policy for the velocity fits (one compile); 'logged' = the one that ran")
a = ap.parse_args()
falls = json.load(open(rs.ROOT / "results/robot28/falls.json"))
D = rs.ROOT / "hardware/leg_p0/software"
keys = ("qe", "quat", "v", "w", "qd", "phase", "last", "last2", "tgt", "g", "vx")
S = {k: [] for k in keys}; meta = []; FS = {}
for run in a.runs:
    hold = set(json.load(open(rs.ROOT / "results/robot28/holdout.json")))
    trials = [r for r in falls if r["run"] == run and r["file"] not in hold]
    _, _, d = load_policy(run)
    envs = {g_: SlenderBipedEnv(rs.robot_cfg(d, gate=g_)) for g_ in (False, True)}
    fs = {}
    for r in trials:
        f = str(D / r["file"]); L = rs.Log(f, envs[False])
        moving = np.where(np.abs(np.diff(L.ph)) > 1e-6)[0]
        if len(moving) < 50:
            continue
        if a.fitpol == "logged":
            pk, _ = rs.which_params(run, f, envs[False]); pname = f"{run}:{pk}"
            gate = bool("gate_held" in L.h and L.A[:, L.h.index("gate_held")].max() > 0)
        else:
            pname, gate = a.fitpol, False
        if (pname, gate) not in FS:
            W, _, dp = load_policy(pname)
            FS[(pname, gate)] = rs.Sim(SlenderBipedEnv(rs.robot_cfg(dp, gate=gate)), W, len(rs.GRID), 0.2, dr=False)
        fsim = FS[(pname, gate)]
        t_w = L.t[moving[0]] + 1.0
        ts = []
        if r["fell"]:
            ts += [("bad", t) for t in np.arange(max(r["t_fall"] - 1.2, t_w), r["T"] - 0.3, 0.1)]
        t_end = (r["t_fall"] - 3.0) if r["fell"] else r["T"] - 1.0
        ts += [("normal", t) for t in np.arange(t_w + 1.0, t_end, 0.7)]
        for kind, t in ts:
            i = int(np.argmin(np.abs(L.t - t)))
            if i < 3 or i > len(L.t) - 12:
                continue
            V, e, e0 = rs.fit_velocity(fsim, L, i)
            st = L.state(i)
            S["qe"].append(st["q"]); S["quat"].append(st["quat"]); S["v"].append(V); S["w"].append(st["w"]); S["qd"].append(st["qd"])
            S["phase"].append(st["phase"]); S["last"].append(st["last"]); S["last2"].append(st["last2"]); S["tgt"].append(st["tgt"])
            S["g"].append(st["g"]); S["vx"].append(max(st["vx"], 0.0) if st["vx"] > 1e-3 else 0.0)
            meta.append((r["file"], kind, float(L.t[i]), e))
        print(f"{r['file']}: {sum(1 for m in meta if m[0] == r['file'])} states ({pname}, gate {gate})", flush=True)
A = {k: np.asarray(v, np.float32) for k, v in S.items()}
import pickle; pickle.dump((A, meta), open(rs.ROOT / "models" / "bank" / f"{a.name}_raw.pkl", "wb"))
kind = np.array([m[1] for m in meta])
bad, nor = np.where(kind == "bad")[0], np.where(kind == "normal")[0]
rep = max(1, int(round(a.bad_share / (1 - a.bad_share) * len(nor) / max(len(bad), 1))))
idx = np.concatenate([nor] + [bad] * rep)
out = rs.ROOT / "models" / "bank"; out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{a.name}.npz", **{k: v[idx] for k, v in A.items()}, kind=kind[idx], src=np.array([m[0] for m in meta])[idx],
         t=np.array([m[2] for m in meta])[idx], fit_err=np.array([m[3] for m in meta])[idx])
print(f"bank {a.name}: {len(bad)} bad x{rep} + {len(nor)} normal = {len(idx)} states; fit err p50 {np.median([m[3] for m in meta]):.2f} deg")
