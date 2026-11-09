"""Shift-and-step reference gaits for the F_board150w pads (2026-11-09).

Robot trials of the deterministic QuickStep (slender_biped/rock_walk.py) with a hip-roll weight shift
(`--det shift=0.10 release=1 fwd=0.03`) showed the shift and the swing cancel: the body roll stayed within ~0..+4 deg
through the swing, and the falls came after touchdown, when the released shift unwound. These tables give residual
RL policies that structure, slowly, with a full rebalance between steps:

* step table (`step`): per step, of duration T_step: settle (hold), shift (hip rolls: stance leg inward, swing leg
  outward, by `shift` rad; the legs lean like a parallelogram and the pelvis slides over the stance board), swing
  (lift bump, swing board forward by the step, stance board back by the same so the pelvis advances; the shift blends
  out over the swing and a further `release_t` s), settle. Legs alternate: right step in the first half of the cycle.
  Foot placement: the planar two-link leg (rock_walk.leg_ik) with the heel on the floor, so staggered boards tilt a
  little toe-up (front) / toe-down (rear) instead of shifting the pelvis (no "ready" pose).
* balance table (`balance`): shift, lift the right board and hold it up for `hold` s, lower, unshift, settle. The
  right leg only (the cycle repeats).

Rows over the forward command v: step length d = v * T_step (balance: v > 0 enables the pattern, v = 0 stands).

    .venv/bin/python -m slender_biped.shift_gait step --name b150L_shift --T-step 2.5 --vmax 0.016
    .venv/bin/python -m slender_biped.shift_gait balance --name b150L_bal1 --hold 2.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .rock_walk import leg_ik, leg_fk, BOARD_HALF

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "models" / "ref"
Q_NOM = np.radians([0.0, 0.0, 16.0, 32.0, 0.0, 0.0, 16.0, 32.0])


def _ease(x):
    x = min(1.0, max(0.0, x))
    return 0.5 - 0.5 * math.cos(math.pi * x)


def _cyc(x):
    x = min(1.0, max(0.0, x))
    return x - math.sin(2 * math.pi * x) / (2 * math.pi)


HEEL = False                                     # place the lowest board edge on the floor (else the board centre)


def foot_q(dx):
    """Hip pitch, knee placing the board centre dx ahead of the stance position at the stance height (HEEL: raised
    so the lowest edge of the tilted board is on the floor; that shortened both legs of a staggered stance and the
    robot sat back 5 deg on two board edges)."""
    hp0, kn0 = Q_NOM[2], Q_NOM[3]
    x0, z0, _ = leg_fk(hp0, kn0)
    hp, kn, dz = hp0, kn0, 0.0
    for _ in range(5 if HEEL else 1):
        hp, kn = leg_ik(x0 + dx, z0 + dz, hp, kn)
        dz = BOARD_HALF * abs(math.sin(leg_fk(hp, kn)[2]))
    return np.array([hp, kn])


def step_pose(t, T, d, p):
    """Joint targets at time t in a two-step cycle (right step in [0, T), left in [T, 2T)); stance flags."""
    leg = 1 if t < T else 0                         # swing leg of this half: 1 = right
    st_leg = 1 - leg
    tau = (t % T)
    t_sh0 = p["settle0"]
    t_sw0 = t_sh0 + p["shift_t"]
    t_sw1 = t_sw0 + p["swing_t"]
    q = Q_NOM.copy()
    stance = np.ones(2, bool)
    # fore-aft: before this step the swing foot is behind (-d/2), the stance foot ahead (+d/2); after: swapped
    prog = _cyc((tau - t_sw0) / p["swing_t"])
    x_sw = -d / 2 + d * prog
    x_st = d / 2 - d * prog
    q[4 * leg + 2:4 * leg + 4] = foot_q(x_sw)
    q[4 * st_leg + 2:4 * st_leg + 4] = foot_q(x_st)
    lift = 0.0
    if t_sw0 <= tau < t_sw1 and d > 0:
        b = math.sin(math.pi * (tau - t_sw0) / p["swing_t"]) ** 2
        q[4 * leg + 3] += p["lift_knee"] * b
        q[4 * leg + 2] += p["lift_hip"] * b
        stance[leg] = False
        lift = p["swing_h"] * b
    # weight shift: in over shift_t before the swing, out over the swing plus release_t
    if d > 0:
        a = _ease((tau - t_sh0) / p["shift_t"]) * (1.0 - _ease((tau - t_sw0) / (p["swing_t"] + p["release_t"])))
        q[4 * st_leg + 1] -= p["shift"] * a          # stance leg inward
        q[4 * leg + 1] += (p["shift"] + p["out"]) * a if tau < t_sw0 else p["shift"] * a
        if tau >= t_sw0:                             # the swing leg keeps `out` from lift-off on, landing wider
            q[4 * leg + 1] += p["out"] * _ease((tau - t_sw0) / p["swing_t"]) * (1.0 - _ease((tau - t_sw1) / max(p["release_t"], 1e-3)))
    z = np.zeros(2)
    z[leg] = lift
    return q, stance, z


def balance_pose(t, p, on):
    """Shift onto the left board, lift the right one, hold it up, lower, unshift."""
    q = Q_NOM.copy()
    stance = np.ones(2, bool)
    z = np.zeros(2)
    if not on:
        return q, stance, z
    t1 = p["settle0"]; t2 = t1 + p["shift_t"]; t3 = t2 + p["swing_t"]; t4 = t3 + p["hold"]; t5 = t4 + p["swing_t"]
    t6 = t5 + p["shift_t"]
    a = _ease((t - t1) / p["shift_t"]) * (1.0 - _ease((t - t5) / p["shift_t"]))
    up = _ease((t - t2) / p["swing_t"]) * (1.0 - _ease((t - t4) / p["swing_t"]))
    q[1] -= p["shift"] * a
    q[5] += p["shift"] * a
    q[7] += p["lift_knee"] * up
    q[6] += p["lift_hip"] * up
    if t2 + 0.5 * p["swing_t"] <= t < t4 + 0.5 * p["swing_t"]:
        stance[1] = False
        z[1] = p["swing_h"]
    return q, stance, z


def build(kind, p, vs, nphase):
    Q = np.zeros((len(vs), nphase, 8)); S = np.zeros((len(vs), nphase, 2)); Z = np.zeros((len(vs), nphase, 2))
    if kind == "step":
        Tc = 2 * p["T_step"]
    else:
        Tc = p["settle0"] + 2 * p["shift_t"] + 2 * p["swing_t"] + p["hold"] + p["settle1"]
    for iv, v in enumerate(vs):
        for ip in range(nphase):
            t = Tc * ip / nphase
            if kind == "step":
                q, st, z = step_pose(t, p["T_step"], v * p["T_step"], p)
            else:
                q, st, z = balance_pose(t, p, v > 0)
            Q[iv, ip], S[iv, ip], Z[iv, ip] = q, st, z
    return dict(q=Q, tip_z=Z, stance=S, v=np.asarray(vs, float), nphase=nphase, gait_hz=1.0 / Tc, ik_err=0.0,
                params=json.dumps(dict(p, kind=kind)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["step", "balance"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--T-step", type=float, default=2.5)
    ap.add_argument("--settle0", type=float, default=0.4)
    ap.add_argument("--shift-t", type=float, default=0.8)
    ap.add_argument("--swing-t", type=float, default=0.35)
    ap.add_argument("--release-t", type=float, default=0.4)
    ap.add_argument("--shift", type=float, default=0.10)
    ap.add_argument("--out", type=float, default=0.0)
    ap.add_argument("--lift-knee", type=float, default=0.3)
    ap.add_argument("--lift-hip", type=float, default=0.15)
    ap.add_argument("--swing-h", type=float, default=0.008)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--settle1", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=0.016)
    ap.add_argument("--nv", type=int, default=5)
    ap.add_argument("--nphase", type=int, default=256)
    a = ap.parse_args()
    p = dict(T_step=a.T_step, settle0=a.settle0, shift_t=a.shift_t, swing_t=a.swing_t, release_t=a.release_t,
             shift=a.shift, out=a.out, lift_knee=a.lift_knee, lift_hip=a.lift_hip, swing_h=a.swing_h, hold=a.hold,
             settle1=a.settle1)
    vs = np.linspace(0, a.vmax, a.nv) if a.kind == "step" else np.array([0.0, a.vmax])
    t = build(a.kind, p, vs, a.nphase)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(REF_DIR / f"{a.name}.npz", **t)
    print(f"{a.name}: {a.kind}, cycle {1 / t['gait_hz']:.2f} s (gait_hz {t['gait_hz']:.6f}), v rows {np.round(vs, 4)}, "
          f"step at vmax {100 * a.vmax * a.T_step:.1f} cm")


if __name__ == "__main__":
    main()
