"""Non-learning stand balancer for the as-built robot, tuned in simulation on the fitted servo model.

    .venv-hw/bin/python scripts/pd_balance.py tune  [--volt 0.68] [--n 24]   # grid-search the gains
    .venv-hw/bin/python scripts/pd_balance.py eval  --gains '{...}'          # evaluate one gain set

The controller holds the stance pose (hip pitch 16, knee 32 deg) and feeds back the pelvis tilt from the IMU, the same
two signals the RL policy gets (gravity direction and gyro):
    pitch, pitch rate  -> both hip pitches together (and optionally both knees): move the pelvis over the pads
    roll,  roll rate   -> hip rolls in opposite directions: sway the pelvis sideways over the stance
    q_target = stance + [hip_pitch: kp_p*pitch + kd_p*wy, knee: kk_p*pitch + kk_d*wy,
                         L_roll: +(kp_r*roll + kd_r*wx), R_roll: -(kp_r*roll + kd_r*wx)]
Signs are searched, not assumed. Evaluated on slender_biped/evaluation.CpuBiped with the servo model fitted to the
hardware step tests (results/servo_fit.json): firmware motion profile, dead time, 5 V or 7.4 V strength and speed.
Writes results/pd_balance.json.
"""
from __future__ import annotations

import argparse
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

ACTION_SCALE = 0.35
# fitted servo model (results/servo_fit.json "chosen"), with the same randomization the RL runs train on
SERVO = dict(prof_acc=(7.0, 35.0), kp=8.0, latency_sub=5, dr_prof_acc=(0.6, 1.5), dr_delay_steps=(0, 1))


TRIM_MAX = 0.12                                # rad: the self-trim may move the stance by at most ~7 deg


def controller(obs, gains, state=None, dt=0.02):
    """obs = the policy's 35-dim observation (gravity 0:3, gyro 3:6, ...). Returns joint offsets (rad) from stance.
    Optional self-trim (ki_p, ki_r with a state dict): slowly integrates the lean into the hip angles so the stance
    settles with the weight centred over the pads, whatever the real centre of mass is."""
    g, w = obs[0:3], obs[3:6]
    pitch = math.atan2(g[0], -g[2])            # + leaning forward
    roll = math.atan2(g[1], -g[2])             # + leaning left
    ip = ir = 0.0
    if state is not None:
        state["ip"] = float(np.clip(state.get("ip", 0.0) + gains.get("ki_p", 0.0) * pitch * dt, -TRIM_MAX, TRIM_MAX))
        state["ir"] = float(np.clip(state.get("ir", 0.0) + gains.get("ki_r", 0.0) * roll * dt, -TRIM_MAX, TRIM_MAX))
        ip, ir = state["ip"], state["ir"]
    up = gains["kp_p"] * pitch + gains["kd_p"] * w[1] + ip
    uk = gains.get("kk_p", 0.0) * pitch + gains.get("kk_d", 0.0) * w[1]
    ur = gains["kp_r"] * roll + gains["kd_r"] * w[0] + ir
    d = np.zeros(8)                            # L yaw roll pitch knee, R yaw roll pitch knee
    d[[2, 6]] = up
    d[[3, 7]] = uk
    d[1], d[5] = ur, -ur
    return np.clip(d, -0.35, 0.35)


# start conditions: "train" = the RL evaluation's reset (+-4.6 deg per joint, 2.3 deg tilt, 1 sigma);
# "hw" = a commanded stance on the real robot (~1 deg)
STARTS = {"train": dict(reset_joint_noise=0.08, reset_tilt=0.04), "hw": dict(reset_joint_noise=0.02, reset_tilt=0.02)}


def episode(gains, seed, volt, push, secs=10.0, dr=True, start="hw"):
    cfg = EvalConfig(dr=dr, push_vel=push, push_interval_s=(2.0, 4.0), strength_scale=volt, speed_scale=volt,
                     **SERVO, **STARTS[start])
    env = CpuBiped("hw_p0_8dof", cfg, seed=seed)
    obs = env.reset()
    n = int(secs / cfg.ctrl_dt)
    tilt_sum = 0.0
    state = {}
    for k in range(n):
        obs, _ = env.step(controller(obs, gains, state, cfg.ctrl_dt) / ACTION_SCALE)
        g = obs[0:3]
        tilt_sum += math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        if env.fallen():
            return k * cfg.ctrl_dt, tilt_sum / (k + 1)
    return secs, tilt_sum / n


def evaluate(gains, n=24, volt=0.68, push=0.0, secs=10.0, start="hw"):
    res = [episode(gains, 1000 + s, volt, push, secs, start=start) for s in range(n)]
    t = np.array([r[0] for r in res])
    return dict(survive=float(np.mean(t >= secs - 1e-6)), mean_time=float(t.mean()),
                mean_tilt=float(np.mean([r[1] for r in res])))


def _job(args):
    gains, n, volt, push, start = args
    return gains, evaluate(gains, n, volt, push, start=start)


def tune(a):
    zero = dict(kp_p=0.0, kd_p=0.0, kp_r=0.0, kd_r=0.0)
    print(f"start '{a.start}'; baseline, plain stance hold (no feedback):",
          evaluate(zero, a.n, a.volt, start=a.start), flush=True)
    # stage 1: pitch loop (roll off), both signs
    g1 = [dict(kp_p=p, kd_p=d, kp_r=0.0, kd_r=0.0) for p, d in
          itertools.product([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3])]
    with mp.Pool(max(1, mp.cpu_count() - 2)) as pool:
        r1 = sorted(pool.map(_job, [(g, a.n, a.volt, a.tune_push, a.start) for g in g1]), key=lambda x: -x[1]["mean_time"])
        print("pitch loop, best 5:", [(g["kp_p"], g["kd_p"], round(r["mean_time"], 2)) for g, r in r1[:5]], flush=True)
        best = r1[0][0]
        # stage 2: roll loop on top of the best pitch loop
        g2 = [dict(best, kp_r=p, kd_r=d) for p, d in
              itertools.product([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15, 0.3])]
        r2 = sorted(pool.map(_job, [(g, a.n, a.volt, a.tune_push, a.start) for g in g2]), key=lambda x: -x[1]["mean_time"])
        print("roll loop, best 5:", [(g["kp_r"], g["kd_r"], round(r["mean_time"], 2)) for g, r in r2[:5]], flush=True)
        best = r2[0][0]
        # stage 3: knees join the pitch loop
        g3 = [dict(best, kk_p=p, kk_d=d) for p, d in
              itertools.product([-2.0, -1.0, 0.0, 1.0, 2.0], [-0.15, 0.0, 0.15])]
        r3 = sorted(pool.map(_job, [(g, a.n, a.volt, a.tune_push, a.start) for g in g3]), key=lambda x: -x[1]["mean_time"])
        print("knee terms, best 3:", [(g["kk_p"], g["kk_d"], round(r["mean_time"], 2)) for g, r in r3[:3]], flush=True)
        best = r3[0][0]
    final = {"gains": best, "volt": a.volt, "start": a.start,
             "quiet": evaluate(best, a.n, a.volt, 0.0, 20.0, a.start),
             "push_0.3": evaluate(best, a.n, a.volt, 0.3, 20.0, a.start),
             "baseline_quiet": evaluate(zero, a.n, a.volt, 0.0, 20.0, a.start),
             "baseline_push_0.3": evaluate(zero, a.n, a.volt, 0.3, 20.0, a.start)}
    print(json.dumps(final, indent=1))
    out = ROOT / "results/pd_balance.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    final["tune_push"] = a.tune_push
    prev[f"volt_{a.volt}_{a.start}" + (f"_push{a.tune_push}" if a.tune_push else "")] = final
    out.write_text(json.dumps(prev, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "eval"])
    ap.add_argument("--volt", type=float, default=0.68)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--gains", type=json.loads)
    ap.add_argument("--start", choices=list(STARTS), default="hw")
    ap.add_argument("--tune-push", type=float, default=0.0, help="score gain sets under random shoves of this size (m/s)")
    a = ap.parse_args()
    if a.cmd == "tune":
        tune(a)
    else:
        print(evaluate(a.gains, a.n, a.volt, a.push, 20.0, a.start))


if __name__ == "__main__":
    main()
