"""Round 17 (2026-11-16): gait quality in the GPU simulator: how hard the feet land, how much the body jolts after a
touchdown, how high the feet clear, and whether it stays up. Walking at --vx for 20 s, no shoves, deterministic policy
with the run-time action filter (as on the robot).

    scripts/gpu_python.sh scripts/gait_eval.py <run[:params]> [--sim robot|loose] [--n 64] [--set k=json ...]

Per touchdown (a board's lowest point reaching the floor): the board centre's downward speed over the 20 ms before
(mm/s), and the body's gyro jolt (roll/pitch rate change per 20 ms tick, deg/s) RMS over the 0.1 s after it, against
the jolt in the middle of the swing (0.1-0.3 s after a lift-off). Per swing: the lowest board point's peak height.
Robot comparison: scripts/robot17_analysis/landing.py gives the same jolt from the robot's log.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from brax.envs.wrappers import training as bw  # noqa: E402

from push_eval import LOOSE, ROBOT, load_policy  # noqa: E402
from slender_biped.mjx_env import SlenderBipedEnv, randomization_fn  # noqa: E402
from slender_biped.runcfg import env_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sim", choices=["robot", "loose", "train"], default="robot")
    ap.add_argument("--act-lpf", type=float, default=0.7)
    ap.add_argument("--vx", type=float, default=0.015)
    ap.add_argument("--T", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero", action="store_true", help="zero action: the reference played open loop")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--falls", action="store_true", help="round 23: why it falls. Per fall: when the tilt passed 10 deg "
                    "(gait phase, which foot was up, roll or pitch, which way), the sideways rock per cycle before it, and "
                    "the body's shake spectrum (dominant frequency) in walkers that stay up and in those that fall")
    ap.add_argument("--profile", action="store_true", help="round 18: the swinging board's height over the swing (by the "
                    "gait clock, 12 bins from lift-off to touchdown), and when it really lands (s after the scheduled touchdown)")
    a = ap.parse_args()
    W, p, d = load_policy(a.run)
    over = dict(step_err=(), reset_stagger=0.0, push_dv=(0.0, 0.0), push_interval_s=(1000.0, 1000.0), push_vel=0.0, act_lpf=a.act_lpf,
                cmd_resample_s=(1000.0, 1000.0), reset_hard_prob=0.0, reset_joint_noise=0.02, reset_tilt=0.02,
                cmd_vx=(a.vx, a.vx), cmd_zero_prob=0.0, cmd_wz=(0.0, 0.0), sup=False)
    if a.sim in ("robot", "loose"):
        over.update(ROBOT)
    if a.sim == "loose":
        over.update(hidden_flex=LOOSE)
    for kv in a.set:
        k, v = kv.split("=", 1)
        x = json.loads(v)
        over[k] = tuple(tuple(e) if isinstance(e, list) else e for e in x) if isinstance(x, list) else x
    cfg = env_config(d, **over)
    env = SlenderBipedEnv(cfg)
    if a.zero:
        W = lambda obs: jp.zeros(obs.shape[:-1] + (env.nj,))  # noqa: E731
    N = a.n
    nst = int(a.T / cfg.ctrl_dt)
    wenv = bw.DomainRandomizationVmapWrapper(env, functools.partial(
        randomization_fn, rng=jax.random.split(jax.random.PRNGKey(a.seed + 1), N)))
    st = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(a.seed), N))

    def body(carry, k):
        st, dead = carry
        st2 = wenv.step(st, W(st.obs["state"]))
        dead = dead | (st2.done > 0)
        low, centre = jax.vmap(env._tip_low)(st2.pipeline_state)
        return (st2, dead), (low, centre[..., 2], st2.pipeline_state.qvel[:, 3:5], dead, st2.pipeline_state.qpos[:, 0:2],
                             st2.info["phase"], st2.info.get("gate_held", jp.zeros(N, jp.int32)) > 0,
                             st2.pipeline_state.qpos[:, jnp_roll], st2.pipeline_state.qpos[:, 3:7])

    jnp_roll = jp.array([7 + env.joints.index("L_hip_roll"), 7 + env.joints.index("R_hip_roll")])
    (st, dead), (low, cz, w, dd, xy, ph, gh, hr, quat) = jax.jit(lambda c: jax.lax.scan(body, c, jp.arange(nst)))((st, jp.zeros(N, bool)))
    low, cz, w, dd, xy, ph, gh, hr, quat = map(np.asarray, (low, cz, w, dd, xy, ph, gh, hr, quat))
    w = np.degrees(w)
    jolt = np.r_[np.zeros((1, N)), np.linalg.norm(np.diff(w, axis=0), axis=2)]       # (T, N)
    contact = low < 0.0015
    vz = np.r_[np.zeros((1, N, 2)), np.diff(cz, axis=0)] / cfg.ctrl_dt
    k0 = int(2.0 / cfg.ctrl_dt)                                                      # skip the first 2 s
    td_v, td_j, mid_j, peaks, spread_lo, spread_pk = [], [], [], [], [], []
    for n in range(N):
        alive = ~dd[:, n]
        for s in range(2):
            c = contact[:, n, s]
            pk, up_at = 0.0, None
            for t in range(1, nst - 6):
                if not alive[t + 5]:
                    break
                if c[t - 1] and not c[t]:
                    up_at, pk = t, 0.0
                    if t > k0:            # round 21: legs apart (hip roll L - R) at the real lift-off (late shifters)
                        spread_lo.append(abs(np.degrees(hr[t, n, 0] - hr[t, n, 1])))
                elif not c[t]:
                    pk = max(pk, low[t, n, s])
                if not c[t - 1] and c[t] and up_at is not None and t > k0:
                    td_v.append(-min(vz[t - 1, n, s], vz[t, n, s], 0.0) * 1000)
                    td_j.append(np.sqrt(np.mean(jolt[t:t + 5, n] ** 2)))
                    if t - up_at > 12:
                        mid_j.append(np.sqrt(np.mean(jolt[up_at + 5:up_at + 15, n] ** 2)))
                    peaks.append(pk * 1000)
                    spread_pk.append(np.degrees(np.abs(hr[up_at:t, n, 0] - hr[up_at:t, n, 1])).max())
                    up_at = None
    up = ~dd[-1]
    dist = np.linalg.norm(xy[-1] - xy[k0], axis=1)
    td_v, td_j, mid_j, peaks = map(np.array, (td_v, td_j, mid_j, peaks))
    tag = (cfg.ref_table + ("" if not a.set else " " + " ".join(a.set)))
    print(f"gait {a.run} ({p.name}{', ZERO ACTION' if a.zero else ''}) sim {a.sim} vx {a.vx} n {N} {a.T:.0f}s | {tag}")
    if len(td_v) == 0 or len(peaks) == 0 or len(mid_j) == 0:     # round 23: e.g. the table played open loop falls at once
        print(f"  up {up.mean():4.0%}  walked {100 * np.median(dist[up]) if up.any() else 0:5.1f} cm  touchdowns {len(td_v)} (too few for landing stats)")
        if a.falls:
            falls(cfg, quat, w, low, ph, dd, k0)
        return
    print(f"  up {up.mean():4.0%}  walked {100 * np.median(dist[up]) if up.any() else 0:5.1f} cm (median of up)  "
          f"touchdowns {len(td_v)}  |  landing speed p50 {np.median(td_v):4.0f} p90 {np.percentile(td_v, 90):4.0f} mm/s  "
          f"jolt after td p50 {np.median(td_j):5.1f} p90 {np.percentile(td_j, 90):5.1f}  mid-swing {np.median(mid_j):5.1f} "
          f"(deg/s per tick)  clearance p10 {np.percentile(peaks, 10):4.1f} p50 {np.median(peaks):4.1f} mm")
    alive = ~dd[k0:]
    print(f"  gate holding (clock stopped) {100 * (gh[k0:] & alive).sum() / max(alive.sum(), 1):.1f} % of the walking time")
    # round 23: double support from the simulated contacts (both boards within 1.5 mm of the floor), per touchdown
    both = contact[k0:, :, 0] & contact[k0:, :, 1] & alive
    none = ~contact[k0:, :, 0] & ~contact[k0:, :, 1] & alive
    n_td = sum(int(((~contact[k0:-1, n, s]) & contact[k0 + 1:, n, s] & alive[1:, n]).sum()) for n in range(N) for s in range(2))
    ds = [run_len for n in range(N) for run_len in _runs(both[:, n])]
    print(f"  both feet down {100 * both.sum() / max(alive.sum(), 1):.0f} % of the time: {both.sum() * cfg.ctrl_dt / max(n_td, 1):.2f} s "
          f"per step (stretches p50 {np.median(ds) * cfg.ctrl_dt if ds else 0:.2f} s); a step every "
          f"{alive.sum() * cfg.ctrl_dt / max(n_td, 1):.2f} s; neither down {100 * none.sum() / max(alive.sum(), 1):.0f} %")
    if spread_lo:
        print(f"  legs apart at lift-off (hip roll L - R): p10 {np.percentile(spread_lo, 10):.1f} p50 {np.median(spread_lo):.1f} deg"
              + (f"; peak in the swing p50 {np.median(spread_pk):.1f} (lift-off / peak {np.median(spread_lo) / np.median(spread_pk):.0%})" if spread_pk else ""))
    if a.profile:
        profile(cfg, low, ph, dd, k0)
    if a.falls:
        falls(cfg, quat, w, low, ph, dd, k0)


def falls(cfg, quat, w, low, ph, dd, k0):
    """Round 23: where the falls start. Body roll (+ = toward the right) / pitch (+ = forward) from the base quaternion."""
    qw, qx, qy, qz = (quat[..., i] for i in range(4))
    roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1)))
    T, N = roll.shape
    dt = cfg.ctrl_dt; u = ph / (2 * np.pi) % 1.0; contact = low < 0.0015
    cyc = int(round(1.0 / (cfg.gait_hz * dt)))
    rows = []
    for n in range(N):
        if not dd[-1, n]:
            continue
        kd = int(np.argmax(dd[:, n]))
        tilt = np.hypot(roll[:, n] - roll[k0 - 25:k0, n].mean(), pitch[:, n] - pitch[k0 - 25:k0, n].mean())
        ks = kd
        while ks > 1 and tilt[ks - 1] > 10:
            ks -= 1
        r0, p0 = roll[ks, n] - roll[k0 - 25:k0, n].mean(), pitch[ks, n] - pitch[k0 - 25:k0, n].mean()
        dirn = ("right" if r0 > 0 else "left") if abs(r0) > abs(p0) else ("forward" if p0 > 0 else "back")
        up = "both down" if contact[ks, n].all() else ("L up" if not contact[ks, n, 0] else ("R up" if not contact[ks, n, 1] else "both up"))
        rock = [np.ptp(roll[max(k, 0):k + cyc, n]) for k in range(ks - 4 * cyc, ks, cyc) if k > k0]
        rows.append((ks * dt, u[ks, n], dirn, up, rock))
    if rows:
        print(f"  falls {len(rows)}: tilt past 10 deg at t p50 {np.median([r[0] for r in rows]):.1f} s")
        for d_ in ("left", "right", "forward", "back"):
            sel = [r for r in rows if r[2] == d_]
            if sel:
                print(f"    {d_:7s} {len(sel):2d}: gait phase u " + " ".join(f"{r[1]:.2f}" for r in sorted(sel, key=lambda r: r[1]))
                      + " | feet: " + ", ".join(f"{k} {sum(r[3] == k for r in sel)}" for k in ("L up", "R up", "both down", "both up") if any(r[3] == k for r in sel)))
        g = [r[4] for r in rows if len(r[4]) == 4]
        if g:
            g = np.array(g)
            print("    roll p2p per cycle, the 4 cycles before the tilt (median): " + " ".join(f"{x:.1f}" for x in np.median(g, 0)) + " deg")
    # shake: body angular rate spectrum after the first 2 s (alive stretch only), roll and pitch
    for lab, sel in (("up", ~dd[-1]), ("fell", dd[-1])):
        if not sel.any():
            continue
        S = 0
        for n in np.flatnonzero(sel):
            kd = int(np.argmax(dd[:, n])) if dd[-1, n] else T
            x = w[k0:max(kd - 25, k0 + 64), n]
            if len(x) < 64:
                continue
            x = x - x.mean(0)
            f = np.fft.rfftfreq(len(x), dt)
            P = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(1) / len(x)
            S = S + np.interp(np.linspace(0, 25, 251), f, P)
        if np.ndim(S):
            fq = np.linspace(0, 25, 251); gait = cfg.gait_hz
            band = (fq > 2.5) & (fq < 12)
            k = np.argmax(np.where(band, S, 0))
            print(f"    shake ({lab}): strongest above 2.5 Hz at {fq[k]:.1f} Hz, {S[k] / S[band].mean():.1f}x the 2.5-12 Hz mean; "
                  f"step rate {2 * gait:.2f} Hz")


def _runs(m):
    """Lengths (ticks) of the True stretches of a 1-D mask."""
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))


def profile(cfg, low, ph, dd, k0, B=12):
    """The swinging board's lowest point (mm) at 13 points from the table's lift-off to its touchdown (gait clock),
    median over swings; and the real touchdown (board within 1.5 mm) in s after the scheduled one."""
    st = np.load(ROOT / "models" / "ref" / f"{cfg.ref_table}.npz")["stance"][-1]
    n = len(st); ev = []
    for i in range(n):
        for leg in (0, 1):                              # stance columns: 0 = left, 1 = right
            if st[i - 1, leg] >= 0.5 > st[i, leg]:
                lo = i / n
                j = next(k for k in range(1, n) if st[(i + k) % n, leg] >= 0.5)
                ev.append((leg, lo, ((i + j) % n) / n))
    u = ph / (2 * np.pi)
    T, N = u.shape
    prof, late = [], []
    for nn in range(N):
        for leg, lo, td in ev:
            span = (td - lo) % 1.0
            for t in range(k0, T - 30):
                if dd[t + 30, nn]:
                    break
                if u[t, nn] != u[t + 1, nn] and (u[t, nn] - lo) % 1.0 > (u[t + 1, nn] - lo) % 1.0:
                    ks, row = t + 1, []
                    for b in range(B + 1):
                        target = (lo + span * b / B) % 1.0
                        k = next((k for k in range(ks, min(ks + 60, T)) if (u[k, nn] - target) % 1.0 < 0.02), None)
                        row.append(np.nan if k is None else low[k, nn, leg] * 1000)
                    prof.append(row)
                    ktd = next((k for k in range(ks, min(ks + 60, T - 1)) if (u[k, nn] - td) % 1.0 < 0.02), None)
                    if ktd is not None:
                        kl = next((k for k in range(ks + 5, min(ks + 80, T)) if low[k, nn, leg] < 0.0015), None)
                        if kl is not None:
                            late.append((kl - ktd) * cfg.ctrl_dt)
    P = np.nanmedian(np.array(prof), 0)
    print("  swing profile (lowest board point, mm, lift-off .. scheduled touchdown):", " ".join(f"{x:4.0f}" for x in P))
    if late:
        print(f"  lands vs scheduled touchdown: p10 {np.percentile(late, 10):+.2f} p50 {np.median(late):+.2f} p90 {np.percentile(late, 90):+.2f} s")


if __name__ == "__main__":
    main()
