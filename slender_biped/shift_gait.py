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


def _quint(x):
    x = min(1.0, max(0.0, x))
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


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
    # 2026-11-16 round 17: the foot can reach its fore-aft spot by place_frac of the swing and then set straight down
    prog = _cyc((tau - t_sw0) / (p["swing_t"] * p.get("place_frac", 1.0)))
    x_sw = -d / 2 + d * prog
    x_st = d / 2 - d * prog
    q[4 * leg + 2:4 * leg + 4] = foot_q(x_sw)
    q[4 * st_leg + 2:4 * st_leg + 4] = foot_q(x_st)
    lift = 0.0
    if t_sw0 <= tau < t_sw1 and d > 0:
        x = (tau - t_sw0) / p["swing_t"]
        r = p.get("rise_frac", 0.5)              # 2026-11-09: rise over r of the swing, lower over the rest (slow touchdown)
        sf, hv = p.get("set_frac", 0.0), p.get("hover", 0.0)
        if x < r:
            b = 0.5 - 0.5 * math.cos(math.pi * x / r)
        elif sf <= 0.0:
            b = 0.5 + 0.5 * math.cos(math.pi * (x - r) / (1 - r))
        elif x < 1 - sf:          # round 17 soft set-down: down to `hover` of the lift by 1 - set_frac of the swing ...
            b = hv + (1 - hv) * (0.5 + 0.5 * math.cos(math.pi * (x - r) / (1 - sf - r)))
        else:                     # ... then the last bit slowly (zero speed at both ends)
            b = hv * (0.5 + 0.5 * math.cos(math.pi * (x - (1 - sf)) / sf))
        if p.get("ik_swing"):             # round 18d: the walk table's swing (quintic up, forward by place_frac,
            hp0, kn0 = Q_NOM[2], Q_NOM[3]  # straight down through the IK) in place of the bump
            x0, z0, _ = leg_fk(hp0, kn0)
            r2, dr = p.get("ik_rise", 0.45), p.get("ik_drop", 0.45)
            b = _quint(x / r2) if x < r2 else (1.0 if x < 1 - dr else 1.0 - _quint((x - (1 - dr)) / dr))
            xr = 0.0                      # round 18f: swing-leg retraction while high: reach ik_reach ahead mid-swing,
            pf = p.get("place_frac", 1.0)  # back to the spot by place_frac (before the straight-down part ends)
            if p.get("ik_reach", 0.0) > 0 and x < pf:
                xr = p["ik_reach"] * (0.5 - 0.5 * math.cos(2 * math.pi * x / pf))
            q[4 * leg + 2:4 * leg + 4] = leg_ik(x0 + x_sw + xr, z0 + p["ik_h"] * b, hp0, kn0)
        else:
            q[4 * leg + 3] += p["lift_knee"] * b
            q[4 * leg + 2] += p["lift_hip"] * b
        stance[leg] = False
        lift = p["swing_h"] * b
    # weight shift: in over shift_t before the swing, out over the swing plus release_t
    if d > 0:
        if p.get("hold_shift"):                  # 2026-11-09 slow step: stay over the stance foot until touchdown
            a = _ease((tau - t_sh0) / p["shift_t"]) * (1.0 - _ease((tau - t_sw1) / p["release_t"]))
        elif p.get("rel0", 0.0) > 0:             # round 17: the release starts at rel0 of the swing (1 = touchdown)
            t_r = t_sw0 + p["rel0"] * p["swing_t"]
            a = _ease((tau - t_sh0) / p["shift_t"]) * (1.0 - _ease((tau - t_r) / (t_sw1 - t_r + p["release_t"])))
        else:
            a = _ease((tau - t_sh0) / p["shift_t"]) * (1.0 - _ease((tau - t_sw0) / (p["swing_t"] + p["release_t"])))
        q[4 * st_leg + 1] -= p["shift"] * a          # stance leg inward
        q[4 * leg + 1] += (p["shift"] + p["out"]) * a if tau < t_sw0 else p["shift"] * a
        if tau >= t_sw0:                             # the swing leg keeps `out` from lift-off on, landing wider
            q[4 * leg + 1] += p["out"] * _ease((tau - t_sw0) / p["swing_t"]) * (1.0 - _ease((tau - t_sw1) / max(p["release_t"], 1e-3)))
    q[[1, 5]] += p.get("splay", 0.0)                 # 2026-11-11: wider stance, both hips abducted (ankles keep boards flat)
    z = np.zeros(2)
    z[leg] = lift
    return q, stance, z


def _bump(x, p):
    """l9's (c_S9mid18) lift profile over the swing fraction x: rise 0.42, down to a 0.2 hover, set down over 0.28."""
    r, sf, hv = 0.42, 0.28, 0.2
    if x < r:
        return 0.5 - 0.5 * math.cos(math.pi * x / r)
    if x < 1 - sf:
        return hv + (1 - hv) * (0.5 + 0.5 * math.cos(math.pi * (x - r) / (1 - sf - r)))
    return hv * (0.5 + 0.5 * math.cos(math.pi * (x - (1 - sf)) / sf))


def sway(x, flat):
    """Round 18: the side-to-side weight shift over a cycle (x in [0, 1)): +1 over the left foot at x = 0.25 (mid right
    swing), -1 over the right at 0.75. A sinusoid (flat 0), maximally flattened at the turning points with a 3rd
    harmonic (flat 1: 2nd derivative 0 there) or 3rd + 5th (flat 2: 2nd and 4th derivatives 0), so the body moves
    continuously like a human centre of mass (no holds), yet stays close to fully over the stance foot in the swing."""
    ph = 2 * math.pi * x
    f1, f3, f5 = math.sin(ph), math.sin(3 * ph), math.sin(5 * ph)
    g1 = (f1 + f3 / 9.0) / (1.0 - 1.0 / 9.0)
    g2 = (f1 + f3 / 6.0 + 0.02 * f5) / (1.0 - 1.0 / 6.0 + 0.02)
    if flat <= 1.0:
        return (1.0 - flat) * f1 + flat * g1
    return (2.0 - flat) * g1 + (flat - 1.0) * g2


def walk_pose(t, T, v, p):
    """Round 18 (2026-11-16) continuous walk, two steps per cycle (right swing centred at T/2, left at 3T/2):
    * weight shift: `sway` (smooth, periodic: no settle, no hold, no jump at the step boundaries);
    * fore-aft: the pelvis advances at a constant v (each stance foot moves back at v the whole stance);
    * swing: the board centre follows a planned path through the leg IK: up over rise_frac of the swing, forward
      (zero speed at both ends) by place_frac, then straight down over drop_frac, with zero speed at lift-off and
      touchdown in both directions. The board is level at both ends (it tips in the air as the IK dictates)."""
    sw = p["swing_t"]
    Ls = v * (2 * T - sw)                                   # the pelvis's travel over one stance
    q = Q_NOM.copy()
    stance = np.ones(2, bool)
    z = np.zeros(2)
    if p.get("sway_ds"):          # round 18b: the whole side-to-side transfer in double support (quintic, through the
        m = p.get("ds_margin", 0.0)   # centre at speed), fully over the stance foot from lift-off to touchdown (+ m)
        a0 = 0.5 * T + sw / 2 + m     # after the right touchdown: from over the left (+1) to over the right (-1)
        D = T - sw - 2 * m
        tt = t % (2 * T)
        x1 = ((tt - a0) % (2 * T)) / D             # first transfer (+1 -> -1) starts at a0
        x2 = ((tt - a0 - T) % (2 * T)) / D         # second (-1 -> +1) starts a step later
        if x1 <= 1.0:
            s = 1.0 - 2.0 * _quint(x1)
        elif x2 <= 1.0:
            s = -1.0 + 2.0 * _quint(x2)
        else:
            s = -1.0 if ((tt - a0) % (2 * T)) < T else 1.0
    else:
        s = sway((t / (2 * T)) % 1.0, p.get("flat", 1.0))
    q[1] -= p["shift"] * s                                  # +s: over the left foot (left hip roll inward)
    q[5] += p["shift"] * s
    hp0, kn0 = Q_NOM[2], Q_NOM[3]
    x0, z0, _ = leg_fk(hp0, kn0)
    for leg, c in ((1, 0.5 * T), (0, 1.5 * T)):            # 1 = right (swings first), 0 = left
        tau = (t - (c + sw / 2)) % (2 * T)                  # time since this foot's touchdown
        if tau < 2 * T - sw:
            xf, h = Ls / 2 - v * tau, 0.0
        else:
            x = (tau - (2 * T - sw)) / sw
            xf = -Ls / 2 + Ls * _cyc(x / p.get("place_frac", 0.8))
            r, dr = p.get("rise_frac", 0.35), p.get("drop_frac", 0.4)
            if x < r:                                   # quintic: zero speed and acceleration at both ends
                b = _quint(x / r)
            elif x < 1 - dr:
                b = 1.0
            else:
                b = 1.0 - _quint((x - (1 - dr)) / dr)
            h = p["swing_h"] * b
            stance[leg] = False
            z[leg] = h
            if p.get("bump"):                     # round 18d: l9's swing instead (equal hip + knee bump on the
                xb = x / p["bump_sw"] if p["bump_sw"] < 1 else x     # floor path), to separate swing from structure
                q[4 * leg + 2:4 * leg + 4] = leg_ik(x0 + xf, z0, hp0, kn0)
                q[4 * leg + 2] += p["bump"] * _bump(xb, p)
                q[4 * leg + 3] += p["bump"] * _bump(xb, p)
                continue
        q[4 * leg + 2:4 * leg + 4] = leg_ik(x0 + xf, z0 + h, hp0, kn0)
    q[[1, 5]] += p.get("splay", 0.0)
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


def with_ankles(q):
    """8-joint pose -> 10 joints (ankle roll after each knee) for the ankle-roll design study: each ankle roll undoes
    its hip roll's offset from the stance pose, so the boards stay flat while the legs lean (the shift moves the
    centre of mass over the stance board instead of rolling the boards onto their edges)."""
    return np.r_[q[:4], -(q[1] - Q_NOM[1]), q[4:], -(q[5] - Q_NOM[5])]


def build(kind, p, vs, nphase):
    nj = 10 if p.get("ankle") else 8
    Q = np.zeros((len(vs), nphase, nj)); S = np.zeros((len(vs), nphase, 2)); Z = np.zeros((len(vs), nphase, 2))
    if kind in ("step", "walk"):
        Tc = 2 * p["T_step"]
    else:
        Tc = p["settle0"] + 2 * p["shift_t"] + 2 * p["swing_t"] + p["hold"] + p["settle1"]
    for iv, v in enumerate(vs):
        for ip in range(nphase):
            t = Tc * ip / nphase
            if kind == "step":
                q, st, z = step_pose(t, p["T_step"], v * p["T_step"], p)
            elif kind == "walk":
                q, st, z = walk_pose(t, p["T_step"], v, p)
            else:
                q, st, z = balance_pose(t, p, v > 0)
            Q[iv, ip], S[iv, ip], Z[iv, ip] = (with_ankles(q) if nj == 10 else q), st, z
    return dict(q=Q, tip_z=Z, stance=S, v=np.asarray(vs, float), nphase=nphase, gait_hz=1.0 / Tc, ik_err=0.0,
                params=json.dumps(dict(p, kind=kind)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["step", "balance", "walk"])
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
    ap.add_argument("--hold-shift", action="store_true", help="keep the weight shift through the swing, release after touchdown")
    ap.add_argument("--rise-frac", type=float, default=0.5, help="fraction of the swing spent lifting (the rest lowers)")
    ap.add_argument("--set-frac", type=float, default=0.0, help="round 17: last fraction of the swing spent setting the "
                    "foot down slowly from --hover (fraction of the lift)")
    ap.add_argument("--hover", type=float, default=0.0)
    ap.add_argument("--place-frac", type=float, default=1.0, help="fraction of the swing by which the foot reaches its "
                    "fore-aft spot (then it sets straight down)")
    ap.add_argument("--rel0", type=float, default=0.0, help="fraction of the swing at which the weight shift starts "
                    "to release (0: lift-off, as c_T12HH; 1: touchdown)")
    ap.add_argument("--flat", type=float, default=1.0, help="round 18 walk: sway flatness at the turning points (0 sine, 1, 2)")
    ap.add_argument("--ik-h", type=float, default=0.0, help="round 18d step: the IK straight-down swing, centre lift (m)")
    ap.add_argument("--ik-rise", type=float, default=0.45, help="round 18 step: with --ik-h, fraction of the swing rising")
    ap.add_argument("--ik-drop", type=float, default=0.45, help="round 18 step: with --ik-h, fraction of the swing lowering")
    ap.add_argument("--ik-reach", type=float, default=0.0, help="round 18f step: with --ik-h, reach this far ahead mid-swing and retract")
    ap.add_argument("--bump", type=float, default=0.0, help="round 18d walk: l9's equal hip + knee bump (rad) as the swing")
    ap.add_argument("--sway-ds", action="store_true", help="round 18b walk: transfer in double support, held through the swing")
    ap.add_argument("--ds-margin", type=float, default=0.0, help="s: the held shift extends this long past lift-off / touchdown")
    ap.add_argument("--drop-frac", type=float, default=0.4, help="round 18 walk: last fraction of the swing spent lowering")
    ap.add_argument("--ankle", action="store_true", help="10 joints: ankle rolls keep the boards flat during the shift")
    ap.add_argument("--splay", type=float, default=0.0, help="rad, both hip rolls abducted throughout: a wider stance "
                    "(~0.27 m per rad per foot; robot round 3: falls came with the boards drifting together)")
    a = ap.parse_args()
    p = dict(T_step=a.T_step, settle0=a.settle0, shift_t=a.shift_t, swing_t=a.swing_t, release_t=a.release_t,
             shift=a.shift, out=a.out, lift_knee=a.lift_knee, lift_hip=a.lift_hip, swing_h=a.swing_h, hold=a.hold,
             settle1=a.settle1, ankle=a.ankle, rise_frac=a.rise_frac,
             hold_shift=a.hold_shift, splay=a.splay)
    for k_, v_ in (("set_frac", a.set_frac), ("hover", a.hover), ("place_frac", a.place_frac), ("rel0", a.rel0)):
        if v_ != (1.0 if k_ == "place_frac" else 0.0):
            p[k_] = v_
    if a.ik_h > 0:
        p.update(ik_swing=True, ik_h=a.ik_h, ik_rise=a.ik_rise, ik_drop=a.ik_drop)
        if a.ik_reach > 0:
            p.update(ik_reach=a.ik_reach)
    if a.kind == "walk" and a.bump > 0:
        p.update(bump=a.bump, bump_sw=1.0)
    if a.kind == "walk":
        p.update(flat=a.flat, drop_frac=a.drop_frac, place_frac=a.place_frac, rise_frac=a.rise_frac)
        if a.sway_ds:
            p.update(sway_ds=True, ds_margin=a.ds_margin)
    vs = np.linspace(0, a.vmax, a.nv) if a.kind in ("step", "walk") else np.array([0.0, a.vmax])
    t = build(a.kind, p, vs, a.nphase)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(REF_DIR / f"{a.name}.npz", **t)
    print(f"{a.name}: {a.kind}, cycle {1 / t['gait_hz']:.2f} s (gait_hz {t['gait_hz']:.6f}), v rows {np.round(vs, 4)}, "
          f"step at vmax {100 * a.vmax * a.T_step:.1f} cm")


if __name__ == "__main__":
    main()
