"""Round 8 (2026-11-13 night): what tips the round-7 policies forward. Per run x simulator condition: falls and their
direction, walking pitch (IMU, + = back), knee - hip (encoders, deg; robot standing: pitch = knee-hip - 12),
the foot proxy pitch - (knee - hip), forward distance.

    .venv/bin/python scripts/robot8_analysis/lean.py --runs h4_t05 r10_lcp_h4 --cond '{"base": {}, "k09": {"hidden_off": [["knee", -0.09]]}}'
"""
import argparse, json, math, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

def tup(v):
    return tuple(tup(x) for x in v) if isinstance(v, list) else v

def episode(a):
    run, sim, cname, cond, seed, vx, seconds, lpf = a
    import eval6
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, quat_rot_t, yaw_of
    rr, _, pp = run.partition(":")
    cfg, rd = load_run(ROOT / "runs" / rr)
    pol = NumpyPolicy(rd / (pp or "params.pkl"))
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0,
                dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
                dr_kp=(0.75, 1.25), kp_hip_roll=2.5, drag_back=0.0)
    over.update(eval6.SIMS[sim]); over.update({k: tup(v) for k, v in cond.items()})
    if lpf is not None: over["act_lpf"] = lpf
    env = CpuBiped(cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=82000 + seed)
    obs = env.reset((vx, 0, 0)); y0 = yaw_of(env.d.qpos[3:7]); x0 = env.d.qpos[0:2].copy()
    P, KH = [], []; up = np.zeros(2, bool); lifts = 0
    n = int(seconds / 0.02)
    for i in range(n):
        env.cmd[2] = float(np.clip((y0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi, -.3, .3))
        obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0])); P.append(math.degrees(math.atan2(-g[0], -g[2])))
        q = np.degrees(env.d.qpos[7:7 + env.nj] + env.flex + env.q_off)
        KH.append(q[[3, 8]].mean() - q[[2, 7]].mean())
        u = env.tip_low()[0] > 0.005; lifts += int(np.sum(u & ~up)); up = u
        if env.fallen():
            break
    P, KH = np.array(P), np.array(KH); T = np.arange(len(P)) * 0.02
    s = (T > 2.0) & (T < T[-1] - 1.0)
    if s.sum() < 30: s = T > 0
    fwd = np.array([math.cos(y0), math.sin(y0)])
    fell = env.fallen(); fdir = "-"
    if fell:
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        r, p = math.degrees(math.atan2(g[1], -g[2])), math.degrees(math.atan2(-g[0], -g[2]))
        fdir = ("F" if p < 0 else "B") if abs(p) > abs(r) else ("L" if r > 0 else "R")
    return run, cname, dict(T=T[-1], fell=fell, fdir=fdir, p05=P[(T > 0.4) & (T < 0.6)].mean() if len(P) > 30 else np.nan,
                            pw=P[s].mean(), kh=KH[s].mean(), foot=(P[s] - KH[s]).mean(), dist=float((env.d.qpos[0:2] - x0) @ fwd), lifts=lifts / max(T[-1], 1.0) * 30)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+"); ap.add_argument("--sim", default="r7"); ap.add_argument("--cond", default='{"base": {}}')
    ap.add_argument("--n", type=int, default=6); ap.add_argument("--vx", type=float, default=0.015)
    ap.add_argument("--seconds", type=float, default=20.0); ap.add_argument("--lpf", type=float, default=None)
    a = ap.parse_args()
    conds = json.loads(a.cond)
    jobs = [(r, a.sim, c, conds[c], s, a.vx, a.seconds, a.lpf) for r in a.runs for c in conds for s in range(a.n)]
    with Pool(min(int(os.environ.get("EVAL_PROCS", 3)), len(jobs))) as p:
        res = p.map(episode, jobs)
    for c in conds:
        for run in a.runs:
            R = [x for rr, cc, x in res if rr == run and cc == c]
            m = lambda k: float(np.nanmean([x[k] for x in R]))  # noqa: E731
            dirs = "".join(sorted(x["fdir"] for x in R if x["fell"]))
            print(f"{c:10s} {run:16s} falls {sum(x['fell'] for x in R)}/{len(R)} {dirs:6s} T {m('T'):4.1f} pitch@0.5 {m('p05'):+5.1f}"
                  f" walk {m('pw'):+5.1f} knee-hip {m('kh'):5.1f} foot {m('foot'):+6.1f} fwd {100*m('dist'):5.1f}cm lifts/30s {m('lifts'):4.1f}", flush=True)

if __name__ == "__main__":
    main()
