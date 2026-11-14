"""Round-6 screen (2026-11-13): full-body policies in the CPU evaluator on the re-massed model with the series give,
no cable pull. Per run: falls, time, fall direction, pitch at the start / walking, |roll| p95, roll past 4 deg,
reach per foot (encoder estimator), 4-8 Hz gyro RMS (the robot's shake), command share at 4-8 Hz.

    .venv/bin/python scripts/eval6.py <run[:params]> ... [--sim fitted|hard|old] [--n 12] [--vx 0.015]
"""
import argparse, ast, math, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from multiprocessing import Pool
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

SER = (("hip_roll", 10.0, 0.15, 2.5), ("ankle_roll", 10.0, 0.15, 2.5), ("knee", 15.0), ("hip_pitch", 15.0), ("hip_yaw", 10.0))
SIMS = {
    "fitted": dict(hidden_flex=SER, flex_series=True, flex_tc=0.005, hidden_off=(("knee", -0.06),)),
    "hard": dict(hidden_flex=tuple((e[0], e[1] * 0.6) + tuple(e[2:3]) + ((e[3] * 0.6,) if len(e) > 3 else ()) for e in SER),
                 flex_series=True, flex_tc=0.007, hidden_off=(("knee", -0.09),)),
    # round 7: the frame mode fitted to the E8 wiggle, 60 ms dead time, the measured standing lean
    "r7": dict(hidden_flex=SER, flex_series=True, flex_tc=0.005, hidden_off=(("knee", -0.025),), prof_acc=(7.0, 35.0, 35.0),
               ring=(("pitch", 4.2, 0.2, 1.0, 0.02), ("roll", 5.3, 0.25, 0.7, 0.02)), ring_delay=0.02, dr_delay_steps=(2, 3)),
    "r7hard": dict(hidden_flex=SER, flex_series=True, flex_tc=0.005, hidden_off=(("knee", -0.045),), prof_acc=(7.0, 35.0, 35.0),
                   ring=(("pitch", 4.2, 0.15, 1.4, 0.02), ("roll", 5.3, 0.18, 1.0, 0.02)), ring_delay=0.03, dr_delay_steps=(3, 3)),
    "old": dict(hidden_flex=(("hip_roll", 3.5),), flex_series=False, flex_tc=0.01, hidden_off=(("knee", -0.045),)),
}

def episode(a):
    run, model, sim, seed, vx, seconds, extra = a
    import gap_fit
    from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, quat_rot_t, yaw_of
    rr, _, pp = run.partition(":")
    cfg, rd = load_run(ROOT / "runs" / rr)
    p = rd / pp if pp else (rd / "params.pkl" if (rd / "params.pkl").exists() else sorted(rd.glob("params_*M.pkl"))[-1])
    pol = NumpyPolicy(p)
    over = dict(floor_tc=0.02, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45, dr_imu_tilt=0.0,
                dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
                dr_kp=(0.75, 1.25), kp_hip_roll=2.5, drag_back=0.0)
    over.update(SIMS[sim]); over.update(extra)
    env = CpuBiped(model or cfg["env"]["model"], EvalConfig.from_train(cfg["env"], **over), seed=81000 + seed)
    obs = env.reset((vx, 0, 0)); y0 = yaw_of(env.d.qpos[3:7]); x0 = env.d.qpos[0:2].copy()
    T, G, Q, W, C = [], [], [], [], []
    for i in range(int(seconds / 0.02)):
        env.cmd[2] = float(np.clip((y0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi, -.3, .3))
        obs[-5:-2] = env.cmd
        obs, _ = env.step(pol(obs))
        T.append(i * 0.02); G.append(quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0])))
        Q.append(env.d.qpos[7:7 + env.nj] + env.flex + env.q_off); W.append(np.degrees(env.d.qvel[3:6])); C.append(np.degrees(env.cmd_buf[0]))
        if env.fallen():
            break
    T, G, Q, W, C = map(np.array, (T, G, Q, W, C))
    r = gap_fit.walk_stats(T, G, Q, gap_fit.MODEL)
    roll = np.degrees(np.arctan2(G[:, 1], -G[:, 2])); pitch = np.degrees(np.arctan2(-G[:, 0], -G[:, 2]))
    s = (T > 2.4) & (T < T[-1] - 1.0)
    if s.sum() < 60: s = T > 0
    fr = np.fft.rfftfreq(int(s.sum()), 0.02)
    def bp(x):
        X = np.fft.rfft(x - x.mean(0), axis=0); X[(fr < 4) | (fr >= 8)] = 0; return np.fft.irfft(X, len(x), axis=0)
    shake = float(np.sqrt(np.mean(np.sum(bp(W[s][:, :2]) ** 2, 1)))) if s.sum() > 20 else np.nan
    Fc = np.abs(np.fft.rfft(C[s] - C[s].mean(0), axis=0)) ** 2
    cmd48 = float(Fc[(fr >= 4) & (fr < 8)].sum() / max(Fc[fr > 0.2].sum(), 1e-9) * 100)
    fwd = np.array([math.cos(y0), math.sin(y0)])
    r.update(fell=env.fallen(), dist=float((env.d.qpos[0:2] - x0) @ fwd), pitch0=float(pitch[:50].mean()),
             roll95=float(np.percentile(np.abs(roll[s]), 95)), over4=float(np.mean(np.abs(roll[s]) > 4)), shake=shake, cmd48=cmd48)
    return run, r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+"); ap.add_argument("--sim", default="fitted", choices=list(SIMS))
    ap.add_argument("--model", default="hw_ank3_b150_m2_10dof"); ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--vx", type=float, default=0.015); ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    import gap_fit
    extra = {k: ast.literal_eval(v) for k, v in (x.split("=", 1) for x in a.set)}
    jobs = [(r, a.model, a.sim, s, a.vx, a.seconds, extra) for r in a.runs for s in range(a.n)]
    with Pool(min(int(os.environ.get("EVAL_PROCS", 3)), len(jobs))) as p:
        res = p.map(episode, jobs)
    for run in a.runs:
        R = [x for rr, x in res if rr == run]
        m = lambda k: float(np.nanmean([x[k] for x in R]))  # noqa: E731
        dirs = {}
        for x in R:
            d = gap_fit.fall_dir(x, x["fell"]); dirs[d] = dirs.get(d, 0) + 1
        print(f"{run:24s} [{a.sim}] falls {sum(x['fell'] for x in R):2d}/{len(R)} T {m('T'):4.1f}s {' '.join(f'{k}:{v}' for k, v in sorted(dirs.items()) if k != '-'):14s}"
              f" pitch0 {m('pitch0'):+4.1f} walk {m('pitch_late'):+4.1f} |roll|p95 {m('roll95'):4.1f} >4deg {100*m('over4'):3.0f}%"
              f" reach L/R {m('reachL'):4.1f}/{m('reachR'):4.1f} fwd {100*m('dist'):4.1f}cm shake {m('shake'):4.1f} cmd4-8 {m('cmd48'):3.1f}%", flush=True)

if __name__ == "__main__":
    main()
