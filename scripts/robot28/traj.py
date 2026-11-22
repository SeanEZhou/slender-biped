"""Round 28: from a log state (fitted velocity), the sim robots' roll / pitch over time against the robot's log."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import robostate as rs
from push_eval import load_policy
from slender_biped.mjx_env import SlenderBipedEnv
ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("policy"); ap.add_argument("--t", type=float, nargs="+")
ap.add_argument("--T", type=float, default=2.0); ap.add_argument("--n", type=int, default=32); ap.add_argument("--set", nargs="*", default=[])
ap.add_argument("--nodr", action="store_true")
a = ap.parse_args()
import json
kw = {}
for kv in a.set:
    k, v = kv.split("=", 1); x = json.loads(v); kw[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
W, p, d = load_policy(a.policy); env = SlenderBipedEnv(rs.robot_cfg(d, **kw))
sim = rs.Sim(env, W, a.n, a.T, dr=not a.nodr); fsim = rs.Sim(env, W, len(rs.GRID), 0.2, dr=False)
L = rs.Log(a.log, env); dt = env.cfg.ctrl_dt
for t0 in a.t:
    i = int(np.argmin(np.abs(L.t - t0))); V, e, e0 = rs.fit_velocity(fsim, L, i)
    R = sim.run(L.state(i), np.tile(V, (a.n, 1)))
    print(f"\n{Path(a.log).name} t {L.t[i]:.2f} u {L.ph[i]/2/np.pi%1:.2f} V {V[0]:+.2f},{V[1]:+.2f} fit {e:.2f} | fell {(R['tilt']>30).any(0).mean()*100:.0f}%")
    print("   +t     u_log u_sim | robot roll/pitch | sim roll p50 (p10..p90) / pitch p50")
    for k in range(0, sim.nst, 5):
        j = i + 1 + k
        rl = f"{L.ph[j]/2/np.pi%1:.2f}  | {L.roll[j]:+6.1f} {L.pitch[j]:+5.1f}" if j < len(L.t) else "  end |              "
        print(f"   +{(k+1)*dt:4.2f}  {rl}   {np.median(R['phase'][k])/2/np.pi%1:.2f} | {np.median(R['groll'][k]):+6.1f} ({np.percentile(R['groll'][k],10):+6.1f}..{np.percentile(R['groll'][k],90):+6.1f}) / {np.median(R['gpitch'][k]):+5.1f}")
