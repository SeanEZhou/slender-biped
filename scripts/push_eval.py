"""Round 13 (2026-11-15) push tests on the GPU simulator (MJX, the training env with its randomization).

    scripts/gpu_python.sh scripts/push_eval.py stand <run[:params]> [--n 24] [--sim train|loose]
    scripts/gpu_python.sh scripts/push_eval.py walk  <run[:params]> [--rec self|stance|<run>] [--no-sup] [--n 16]

stand: the robot stands (command 0); at t = 2 s one shove (force on the pelvis for 0.15 s, the robot gains dv) forward,
  back, left or right (heading frame), then 5 s more. Per direction and dv: share still up, share back "home" at the end
  (both pads down, every joint within 6 deg of the stance, tilt within 3 deg of the start, turning < 10 deg/s), and
  recovery steps (touchdowns after the shove).
walk: walking at 0.015 m/s; at t = 4-6 s one shove; 14 s in all. With the supervisor (default on, the run's sup_*
  settings or the defaults) the command drops to 0 on a predicted tip, the recovery policy (--rec: self = the walker at
  command 0, stance = no action: the stance pose, or another run) runs until quiet, then walking resumes. Per direction and
  dv: up, supervisor triggers, resumed (walked >= 2 cm in the last 4 s), forward distance after the shove.
Both run the policy deterministically with the run-time action filter (--act-lpf, default 0.7 as k5 runs on the robot).
"""
from __future__ import annotations

import argparse
import functools
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from brax.envs.wrappers import training as bw  # noqa: E402

from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn  # noqa: E402
from slender_biped.runcfg import env_config  # noqa: E402

DIRS = {"fwd": 0.0, "back": math.pi, "left": 0.5 * math.pi, "right": -0.5 * math.pi}
SER = (("hip_roll", 10.0, 0.15, 2.5), ("ankle_roll", 10.0, 0.15, 2.5), ("knee", 15.0), ("hip_pitch", 15.0), ("hip_yaw", 10.0))
LOOSE = (("hip_roll", 5.0, 0.15, 1.25),) + SER[1:]
# the CPU screen's fitted robot (scripts/eval6.py, base settings + SIMS["fitted"]): one rug, 5 V, the measured servo
# ramps and IMU filter, the fitted give; masses and friction still randomized. "loose": the hip roll's give doubled
# (round 11: reproduces k5's sideways falls). "train": the training distribution (harsher: soft rugs to 0.8 s, sag).
ROBOT = dict(dr_floor_tc=(0.02, 0.02), dr_floor_damp=(1.0, 1.0), imu_filter=True, imu_gate=0.15, gyro_range=17.45,
             dr_imu_tilt=0.0, dr_volt=(0.68, 0.68), dr_delay_steps=(0, 1), prof_acc=(25.0, 60.0, 60.0), noise_qd=0.25,
             dr_kp=(0.75, 1.25), kp_hip_roll=2.5, drag_back=0.0, hidden_flex=SER, flex_series=True, flex_tc=0.005,
             dr_flex=(1.0, 1.0), dr_flex_tc=(), hidden_off=(("knee", -0.06, -0.06, 0.0),), dr_friction=(0.6, 1.0))


def load_policy(spec):
    run, _, pp = spec.partition(":")
    d = ROOT / "runs" / run
    p = d / pp if pp else sorted(d.glob("params_*M.pkl"))[-1]
    with open(p, "rb") as f:
        norm, pol = pickle.load(f)[:2]
    mean, std = jp.asarray(norm.mean["state"]), jp.asarray(norm.std["state"])
    L = pol["params"]
    keys = sorted(L, key=lambda s: int(s.split("_")[-1]))
    Ws = [(jp.asarray(L[k]["kernel"]), jp.asarray(L[k]["bias"])) for k in keys]

    def f(obs):
        x = (obs - mean) / std
        for i, (W, b) in enumerate(Ws):
            x = x @ W + b
            if i < len(Ws) - 1:
                x = jax.nn.swish(x)
        return jp.tanh(x[..., : x.shape[-1] // 2])
    return f, p, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", choices=["stand", "walk", "steps"])
    ap.add_argument("run")
    ap.add_argument("--rec", default="self")
    ap.add_argument("--no-sup", action="store_true")
    ap.add_argument("--n", type=int, default=24, help="envs per direction and dv")
    ap.add_argument("--dv", type=float, nargs="*")
    ap.add_argument("--sim", choices=["robot", "loose", "train"], default="robot")
    ap.add_argument("--act-lpf", type=float, default=0.7)
    ap.add_argument("--dur", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="EnvConfig overrides k=json")
    ap.add_argument("--vx", type=float, default=0.015, help="walk test: forward command (m/s)")
    a = ap.parse_args()
    W, p, d = load_policy(a.run)
    steps = a.test == "steps"           # round 14: recovery-step policies stepping on command (random targets), no shoves
    stand = a.test in ("stand", "steps")
    dvs = a.dv or ([0.1, 0.15, 0.2, 0.25, 0.3, 0.4] if stand else [0.0, 0.1, 0.15, 0.2, 0.25])
    over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=a.act_lpf,
                cmd_resample_s=(1000.0, 1000.0), reset_hard_prob=0.0,
                reset_joint_noise=0.02, reset_tilt=0.02)          # a hardware-like start (training: 0.08 / 0.04)
    if stand:
        over.update(cmd_zero_prob=1.0, sup=False)
    import json as _j
    renv = _j.loads((d / "config.json").read_text())["env"]
    rec_walk = bool(renv.get("rec_walk", False))
    if renv.get("rec_table"):   # round 14: recovery-step policies
        over.update(rec_rand=1.5 if steps else 0.0, rec_rest_s=(0.6, 1.5), rec_cp=not steps, rec_rand_walk=0.0)
        if rec_walk and not stand:     # round 15: walk, capture-point step, resume (no supervisor)
            over.update(sup=False, cmd_vx=(a.vx, a.vx))
    if steps:
        dvs = [0.0]
    else:
        over.update(cmd_vx=(a.vx, a.vx), cmd_zero_prob=0.0, cmd_wz=(0.0, 0.0), sup=not a.no_sup)
    if a.sim in ("robot", "loose"):
        over.update(ROBOT)
    if a.sim == "loose":
        over.update(hidden_flex=LOOSE)
    import json
    for kv in a.set:
        k, v = kv.split("=", 1)
        x = json.loads(v)
        over[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
    cfg = env_config(d, **over)
    env = SlenderBipedEnv(cfg)
    if a.rec == "self":
        R = W
    elif a.rec == "stance":
        R = lambda obs: jp.zeros(obs.shape[:-1] + (env.nj,))  # noqa: E731
    else:
        R = load_policy(a.rec)[0]
    combos = [(dn, dv) for dv in dvs for dn in DIRS]
    N = len(combos) * a.n
    ang = jp.array(np.repeat([DIRS[c[0]] for c in combos], a.n))
    dvv = jp.array(np.repeat([c[1] for c in combos], a.n))
    T = 12.0 if steps else 7.0 if stand else 14.0
    nst = int(T / cfg.ctrl_dt)
    rng = np.random.default_rng(a.seed)
    t_push = (np.full(N, 2.0) if stand else rng.uniform(4.0, 6.0, N)) / cfg.ctrl_dt
    t_push = jp.array(t_push.astype(np.int32))
    n_dur = max(1, int(round(a.dur / cfg.ctrl_dt)))
    # randomized bodies and floors as in training, no auto-reset (a fallen env just keeps falling; `dead` marks it)
    wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(
        randomization_fn, rng=jax.random.split(jax.random.PRNGKey(a.seed + 1), N)))
    st = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(a.seed), N))
    q_home = st.pipeline_state.qpos[:, 7:7 + env.nj]

    def yaw_of(q):
        return jp.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def body(carry, k):
        st, dead, nsup, x_push, td, sup_t, q_pre = carry
        obs = st.obs["state"]
        on = st.info["sup_on"]
        act = jp.where(on[:, None], R(obs), W(obs))
        hit = k == t_push
        yaw = yaw_of(st.pipeline_state.qpos[:, 3:7])
        mass = jp.sum(env.sys.body_mass)
        fmag = mass * dvv / (n_dur * cfg.ctrl_dt)
        f = jp.stack([fmag * jp.cos(yaw + ang), fmag * jp.sin(yaw + ang), jp.zeros(N)], 1)
        info = dict(st.info)
        info["push_f"] = jp.where(hit[:, None], f, info["push_f"])
        info["push_left"] = jp.where(hit, n_dur + 1, info["push_left"])   # the env counts it down before applying
        x_push = jp.where(hit[:, None], st.pipeline_state.qpos[:, 0:2], x_push)
        q_pre = jp.where(hit[:, None], st.pipeline_state.qpos[:, 7:7 + env.nj], q_pre)   # home: the pose held before the shove
        st = st.replace(info=info)
        st2 = wenv.step(st, act)
        dead = dead | (st2.done > 0)
        after = k > t_push
        td = td + jp.where(after & ~dead, st2.metrics["touchdowns"], 0.0)
        nsup = nsup + jp.where(~dead, (st2.info["sup_on"] & ~on).astype(jp.int32), 0)
        sup_t = sup_t + jp.where(~dead & st2.info["sup_on"], 1, 0)
        return (st2, dead, nsup, x_push, td, sup_t, q_pre), (st2.pipeline_state.qpos[:, 0:2], st2.pipeline_state.qvel[:, 3:5],
                                                              st2.info["prev_cmd"])

    zeros = jp.zeros(N)
    carry = (st, jp.zeros(N, bool), jp.zeros(N, jp.int32), st.pipeline_state.qpos[:, 0:2], zeros, jp.zeros(N, jp.int32), q_home)
    carry, (xy, wxy, qc) = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(nst)))(carry)
    st, dead, nsup, x_push, td, sup_t, q_home = carry
    dead = np.asarray(dead)
    qpos = np.asarray(st.pipeline_state.qpos)
    xy, wxy = np.asarray(xy), np.asarray(wxy)
    up = ~dead
    # home: both pads down, joints near the stance, level, still
    low, _ = jax.vmap(env._tip_low)(st.pipeline_state)
    both = np.all(np.asarray(low) < 0.0025, 1)
    qerr = np.max(np.abs(qpos[:, 7:7 + env.nj] - np.asarray(q_home)), 1)
    from slender_biped.mjx_env import _quat_rot_t
    g = np.asarray(jax.vmap(lambda q: _quat_rot_t(q, jp.array([0.0, 0.0, -1.0])))(st.pipeline_state.qpos[:, 3:7]))
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
    still = np.max(np.abs(np.degrees(wxy[-25:])), axis=(0, 2)) < 10.0
    home = up & both & (np.degrees(qerr) < 6.0) & (tilt < 3.0 + 2.0) & still
    print(f"(of those up: both pads down {both[up].mean():.0%}, joints within 6 deg {(np.degrees(qerr) < 6)[up].mean():.0%} "
          f"(median worst {np.degrees(np.median(qerr[up])) if up.any() else 0:.1f} deg), tilt < 5 deg {(tilt < 5)[up].mean():.0%}, "
          f"still {still[up].mean():.0%})")
    yaw0 = 0.0
    tp = np.asarray(t_push)
    print(f"{a.test} test, {a.run} ({p.relative_to(ROOT)})" + ("" if stand else f", recovery {a.rec}, supervisor "
          f"{'off' if a.no_sup else 'on'}") + f", sim {a.sim}, act-lpf {a.act_lpf}, {a.n} per cell, shove {a.dur:.2f} s")
    if stand:
        print("dv m/s | " + " | ".join(f"{dn:>17s}" for dn in DIRS) + " |  all up / home")
        print("       | " + " | ".join(f"{'up home steps':>17s}" for _ in DIRS))
    else:
        print("dv m/s | " + " | ".join(f"{dn:>22s}" for dn in DIRS) + " |  all up / resumed")
        print("       | " + " | ".join(f"{'up trig resum dist':>22s}" for _ in DIRS))
    td = np.asarray(td); nsup = np.asarray(nsup)
    if rec_walk and not stand:          # round 15: "trig" = capture-point recovery steps started
        nsup = np.asarray(st.info["rec_n"])
    tot = {}
    for dv in dvs:
        cells = []
        sel_all = np.zeros(N, bool)
        for dn in DIRS:
            i = [j for j, c in enumerate(combos) if c == (dn, dv)][0]
            s = np.arange(i * a.n, (i + 1) * a.n)
            sel_all[s] = True
            if stand:
                cells.append(f"{up[s].mean():4.0%} {home[s].mean():4.0%} {td[s].mean():5.1f}")
            else:
                last = int(4.0 / cfg.ctrl_dt)
                # forward distance (heading at the start ~ the env's random yaw; use path length along the final heading)
                dist_after = np.array([np.linalg.norm(xy[-1, j] - np.asarray(x_push)[j]) for j in s])
                walked_last = np.array([np.linalg.norm(xy[-1, j] - xy[-last, j]) for j in s])
                res = up[s] & (walked_last > 0.02)
                cells.append(f"{up[s].mean():4.0%} {nsup[s].mean():4.1f} {res.mean():4.0%} {100 * np.mean(dist_after * up[s]):4.1f}cm")
        if stand:
            print(f"{dv:6.2f} | " + " | ".join(f"{c:>17s}" for c in cells) + f" | {up[sel_all].mean():4.0%} / {home[sel_all].mean():4.0%}")
        else:
            last = int(4.0 / cfg.ctrl_dt)
            walked_last = np.linalg.norm(xy[-1] - xy[-last], axis=1)
            print(f"{dv:6.2f} | " + " | ".join(f"{c:>22s}" for c in cells)
                  + f" | {up[sel_all].mean():4.0%} / {(up & (walked_last > 0.02))[sel_all].mean():4.0%}")
    if stand:   # how much the policy moves after the shove: largest command change from its value at the shove, 1 s
        qc = np.degrees(np.asarray(qc)); k0 = int(np.asarray(t_push)[0])
        dq = np.max(np.abs(qc[k0:k0 + 50] - qc[k0 - 1]), 0)              # (N, nj)
        grp = {g_: [i for i, j in enumerate(env.joints) if j.endswith(g_)] for g_ in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll")}
        for dv in dvs:
            sel = np.repeat([c[1] == dv for c in combos], a.n)
            for lab, m in (("up", sel & up), ("fell", sel & ~up)):
                if m.any():
                    print(f"  dv {dv:.2f} {lab:4s} (n {m.sum():3d}): command moved within 1 s (max |change| deg, mean over runs): "
                          + " ".join(f"{g_.split('_')[-1][:5]} {dq[m][:, ix].max(1).mean():4.1f}" for g_, ix in grp.items()))
    if "rec_n" in st.info:
        rn = np.asarray(st.info["rec_n"])
        print(f"recovery steps started per run: {rn.mean():.2f} (up runs {rn[up].mean() if up.any() else 0:.2f}, "
              f"fallen {rn[~up].mean() if (~up).any() else 0:.2f}); touchdowns after the shove {np.asarray(td).mean():.1f}")
    print(f"overall up {up.mean():.0%}" + (f", home {home.mean():.0%}" if stand else
          f", resumed {(up & (np.linalg.norm(xy[-1] - xy[-int(4.0 / cfg.ctrl_dt)], axis=1) > 0.02)).mean():.0%}, "
          f"supervisor triggers per run {nsup.mean():.2f}"))


if __name__ == "__main__":
    main()
