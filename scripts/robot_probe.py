"""Does a policy show the robot's failure modes in simulation? (2026-11-11)

    .venv/bin/python scripts/robot_probe.py <run> [--params P] [--vx 0.015] [--n 6] [--seconds 30] [--floor 0.02]
                                            [--volt 0.68,1.0] [--delay 0,1,2] [--tug 0,0.5,1.0]

On the robot (logs of 2026-11-10/11) c3f_T12: (1) shakes sideways at ~4.8 Hz: 69-73 % of the roll rate's energy in
4-6.5 Hz, the policy's hip / ankle roll commands oscillating with it in phase (coherence 0.87), worse at 7.4 V;
(2) marches nearly in place, the right foot stepping less than the left; c3m_T12 (3) is pulled over by its cables.

Per condition (supply factor: 1.0 = 7.4 V, 0.68 = 5 V; extra command dead time in 20 ms steps; a steady pull on
the pelvis in N, random horizontal direction, on for 2-5 s at a time half of the walk), the robot's IMU filter in
the loop, servo / mass randomization per seed:

* command jitter: std of the joint targets minus their 100 ms mean, deg, joint average (robot logs 2026-11-11:
  c3m / c3f 0.45-0.6, the act_lpf 0.7 policies 0.1-0.3); board-centre separation (body frame) median / min, mm
* pelvis beside the standing board (body frame; + = outward past its centre): median at lift-off (very negative =
  lifting before the weight is over the standing foot, robot falls mode A) and the 95th percentile over single
  support (large + = leaning out over the support leg, mode B)
* falls; roll-rate share in 4-6.5 Hz and its peak frequency; shake = RMS second difference of the roll/pitch rate
  (deg/s, the robot logs' "gyro zigzag"); action zigzag (RMS second difference of the actions)
* forward cm in the walk; mean step length of the left and right foot (mm, lift-off to touchdown along the heading)
"""
import argparse, math, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, yaw_of, quat_rot_t  # noqa: E402


def episode(env, pol, vx, seconds, tug, rng, filt=None, ramp=0.0, blend=0.0, hold_clock=False):
    obs = env.reset((vx, 0.0, 0.0))
    dt = env.cfg.ctrl_dt
    yaw0 = yaw_of(env.d.qpos[3:7]); fwd = np.array([np.cos(yaw0), np.sin(yaw0)])
    p0 = env.d.qpos[0:2].copy()
    w, acts = [], []
    up = np.zeros(2, bool); lift_x = np.zeros(2); steps = [[], []]
    fell = False
    t_next, on, F, F_goal = rng.uniform(1, 4), False, np.zeros(3), np.zeros(3)
    a_prev = None
    lpf = env.cfg.act_lpf; af = None; cmds = []; width = []; lift_off = []; lean = []
    tilt_early = 0.0
    for k in range(int(seconds / dt)):
        t = k * dt
        if blend > 0:                         # the robot runner's --ramp: policy and reference faded in over `blend` s
            env.ramp = min(1.0, (t + dt) / blend)
            if hold_clock and env.ramp < 1.0:
                env.phase = 0.0                # runner fix: the gait clock waits for the blend
                obs[-2:] = [0.0, 1.0]
        if tug > 0 and t >= t_next:
            on = not on
            ang = rng.uniform(0, 2 * np.pi)
            F_goal = np.array([np.cos(ang), np.sin(ang), 0.0]) * tug * on
            t_next = t + rng.uniform(2, 5)
            if not ramp:
                F = F_goal
        if tug > 0 and ramp:                      # a cable tightening: the pull builds over `ramp` s
            F = F + np.clip(F_goal - F, -tug / ramp * dt, tug / ramp * dt)
        env.d.xfrc_applied[1, :3] = F
        err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
        env.cmd[2] = float(np.clip(err, -0.3, 0.3)); obs[-5:-2] = env.cmd
        a = pol(obs)
        if filt is not None:
            a_prev = a if a_prev is None else filt * a_prev + (1 - filt) * a
            a = a_prev
        acts.append(a.copy())
        af = a if af is None or not lpf else lpf * af + (1 - lpf) * a     # what the servos get (the run's filter)
        cmds.append(af * env.cfg.action_scale)
        obs, _ = env.step(a)
        w.append(env.d.qvel[3:5].copy())
        if t < 5.0:
            tilt_early = max(tilt_early, float(np.degrees(np.arccos(np.clip(quat_rot_t(env.d.qpos[3:7], np.array([0, 0, 1.0]))[2], -1, 1)))))
        low, c = env.tip_low()
        width.append(abs(quat_rot_t(env.d.qpos[3:7], c[0] - c[1])[1]))
        # pelvis beside the standing board, body frame, + = outward past the board centre (mode B: tipping over the
        # support leg), - = still inside it (mode A at lift-off: the weight not yet over the standing foot)
        out = [quat_rot_t(env.d.qpos[3:7], env.d.qpos[0:3] - c[j])[1] * (1 if j == 0 else -1) for j in range(2)]
        if up.sum() == 1:
            lean.append(out[int(np.argmin(up))])
        for i in range(2):
            if not up[i] and low[i] > 0.003:
                up[i] = True; lift_x[i] = c[i, :2] @ fwd; lift_off.append(out[1 - i])
            elif up[i] and low[i] < 0.0015:
                up[i] = False; steps[i].append(c[i, :2] @ fwd - lift_x[i])
        if env.fallen():
            fell = True
            break
    env.d.xfrc_applied[:] = 0
    w = np.degrees(np.array(w)); acts = np.array(acts)
    x = w[:, 0] - w[:, 0].mean()
    F_ = np.abs(np.fft.rfft(x)) ** 2; fr = np.fft.rfftfreq(len(x), dt)
    share = F_[(fr > 4) & (fr < 6.5)].sum() / max(F_[fr > 0.2].sum(), 1e-9)
    pk = fr[fr > 2][np.argmax(F_[fr > 2])] if len(fr[fr > 2]) else 0
    shake = np.sqrt(np.mean(np.sum((w[2:] - 2 * w[1:-1] + w[:-2]) ** 2, 1))) if len(w) > 3 else 0
    azig = np.sqrt(np.mean(np.sum((acts[2:] - 2 * acts[1:-1] + acts[:-2]) ** 2, 1))) if len(acts) > 3 else 0
    fwd_d = (env.d.qpos[0:2] - p0) @ fwd
    C = np.array(cmds)
    hp = np.degrees(np.std(C[2:-2] - np.array([C[i:len(C) - 4 + i] for i in range(5)]).mean(0), 0)).mean() if len(C) > 10 else 0
    return dict(fell=fell, hp=hp, te=tilt_early, early=fell and len(w) * dt < 5.0, lo=np.median(lift_off) if lift_off else np.nan,
                lean=np.percentile(lean, 95) if lean else np.nan, wmed=np.median(width), wmin=np.min(width), share=share, pk=pk, shake=shake, azig=azig, fwd=fwd_d,
                sL=np.mean(steps[0]) if steps[0] else 0.0, sR=np.mean(steps[1]) if steps[1] else 0.0,
                nL=len(steps[0]), nR=len(steps[1]), T=len(w) * dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--params")
    ap.add_argument("--vx", type=float, default=0.015)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--floor", type=float, default=0.02)
    ap.add_argument("--volt", default="0.68,1.0")
    ap.add_argument("--delay", default="1")
    ap.add_argument("--tug", default="0")
    ap.add_argument("--model")
    ap.add_argument("--ramp", type=float, default=0.0, help="s for a pull to build up (0 = sudden)")
    ap.add_argument("--filt", type=float, default=None, help="extra action low-pass on top of the run's (alpha)")
    ap.add_argument("--blend", type=float, default=0.0, help="emulate the runner's --ramp (s)")
    ap.add_argument("--hold-clock", action="store_true", help="with --blend: gait clock held at 0 during the blend")
    ap.add_argument("--set", nargs="*", default=[], help="EvalConfig overrides k=python-literal, e.g. prof_acc=(25,60,60)")
    a = ap.parse_args()
    cfg, rd = load_run(a.run)
    params = a.params or sorted(rd.glob("params_*M.pkl"))[-1].name
    pol = NumpyPolicy(rd / params)
    model = a.model or cfg["env"]["model"]
    print(f"{a.run} {params} on {model}, vx {a.vx}, floor tc {a.floor}, {a.n} x {a.seconds:.0f} s {' '.join(a.set)}",
          flush=True)
    for v in (float(x) for x in a.volt.split(",")):
        for dl in (int(x) for x in a.delay.split(",")):
            for tug in (float(x) for x in a.tug.split(",")):
                over = dict(floor_tc=a.floor, push_vel=0.0, imu_filter=True, imu_gate=0.15, gyro_range=17.45,
                            dr_imu_tilt=0.0, dr_volt=(v, v), dr_delay_steps=(dl, dl))
                over.update({k: __import__("ast").literal_eval(x) for k, x in (kv.split("=", 1) for kv in a.set)})
                rng = np.random.default_rng(5)
                res = [episode(CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=81000 + s), pol, a.vx,
                               a.seconds, tug, rng, a.filt, a.ramp, a.blend, a.hold_clock) for s in range(a.n)]
                m = lambda k: np.median([r[k] for r in res])  # noqa: E731
                print(f"  volt {v:4.2f} delay {dl} tug {tug:3.1f} N{' ramp %.1f s' % a.ramp if tug and a.ramp else ''}: falls {sum(r['fell'] for r in res)}/{a.n} (first 5 s {sum(r['early'] for r in res)}, peak tilt there {m('te'):4.1f} deg) | "
                      f"roll 4-6.5 Hz {100 * m('share'):3.0f}% (peak {m('pk'):3.1f} Hz) shake {m('shake'):4.1f} deg/s "
                      f"act zig {m('azig'):.3f} cmd hp {m('hp'):.2f} deg | width {1000 * m('wmed'):3.0f} min {1000 * m('wmin'):3.0f} mm | pelvis vs standing board: at lift-off {1000 * m('lo'):+4.0f} lean p95 {1000 * m('lean'):+4.0f} mm | fwd {100 * m('fwd'):4.1f} cm | step L {1000 * m('sL'):4.1f} R "
                      f"{1000 * m('sR'):4.1f} mm ({m('nL'):.0f}/{m('nR'):.0f} steps)", flush=True)


if __name__ == "__main__":
    main()
