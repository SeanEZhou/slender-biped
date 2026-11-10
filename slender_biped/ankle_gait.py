"""Reference gaits for the ankle-roll leg as designed (hardware/ankle_p1 v2, 2026-11-09): straight legs, flat feet.

Kinematics (models/hw_ank2_*): thigh 138 mm, knee to sole 150 mm, hip roll axis 34.2 mm above the hip pitch axis,
hips +-40 mm. Joint order per leg: hip yaw, hip roll, hip pitch, knee, ankle roll.

* Feet flat fore-aft without an ankle pitch: a flat foot needs a vertical shin, i.e. knee = hip pitch (thigh angle
  from vertical, pelvis level). A foot ahead of the hip has the knee bent the usual way, a foot behind it has the knee
  bent "forward" (negative). Both legs swing like a parallelogram: the thighs pendulum, the shins stay vertical.
* Weight shift: both legs lean by `lam` toward the stance side (hip rolls), the ankle rolls keep both feet flat; at
  lam = 0.124 rad the pelvis centre is over the stance foot.
* Swing-foot clearance by hip hike: the stance leg rolls the pelvis by `psi` toward the stance side (lowering the
  stance hip, raising the swing hip by 2 * 40 mm * sin psi), the swing leg stays parallel to the stance leg with its
  foot flat. The swing foot then moves at a constant height (the flat-foot parallelogram keeps both hips at the same
  height), and touchdown is just the hike going back down, slowly. An optional knee bump adds lift mid-swing.
* Per step (`step`): settle, shift, hike up (foot lifts), swing (pelvis rides forward over the stance foot: its thigh
  goes from +a to -a, the swing foot from -d/2 to +d/2 relative to the pelvis), hike down (slow), unshift, settle.
  Stagger d between the feet in double support; the pelvis advances d per step. Right step first.
* `balance`: shift onto the left foot, hike (right foot up), hold, lower, unshift, settle.

    .venv/bin/python -m slender_biped.ankle_gait step --name a2_step --vmax 0.008
    .venv/bin/python -m slender_biped.ankle_gait balance --name a2_bal
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "models" / "ref"
LT, LS, W_HIP = 0.138, 0.150, 0.040


def _ease(x):
    x = min(1.0, max(0.0, x))
    return 0.5 - 0.5 * math.cos(math.pi * x)


def leg_q(theta, alpha_abs, psi, side, bump=0.0, bump_hip=None):
    """Joint angles of one leg: thigh angle theta (fore-aft, shin vertical), leg roll alpha_abs (rotation about world
    +x of the whole leg), pelvis roll psi (about +x); foot flat. side +1 left, -1 right. bump: extra knee flexion
    (hip pitch takes half: the foot rises, pitched slightly toe-down)."""
    hr = side * (alpha_abs - psi)                 # left axis +x, right axis -x
    ar = -side * alpha_abs                        # foot roll = leg roll + ankle (signed) = 0
    return np.array([0.0, hr, theta + (bump / 2 if bump_hip is None else bump_hip), theta + bump, ar])


def step_pose(t, T, d, p):
    leg = 1 if t < T else 0                       # swing leg of this half: 1 = right
    tau = t % T
    sw_side = -1 if leg == 1 else 1               # +1 left
    t1 = p["settle0"]; t2 = t1 + p["shift_t"]; t3 = t2 + p["hike_t"]; t4 = t3 + p["swing_t"]
    t5 = t4 + p["lower_t"]; t6 = t5 + p["unshift_t"]
    on = d > 0
    s = (_ease((tau - t1) / p["shift_t"]) * (1 - _ease((tau - t5) / p["unshift_t"]))) if on else 0.0
    h = (_ease((tau - t2) / p["hike_t"]) * (1 - _ease((tau - t4) / p["lower_t"]))) if on else 0.0
    prog = _ease((tau - t3) / p["swing_t"]) if on else 0.0
    a = math.asin(min(d / 2 / LT, 0.9))
    th_st = a - 2 * a * prog                      # stance foot from +d/2 to -d/2 relative to the pelvis
    th_sw = -th_st
    # stance side: the swing leg's opposite; the legs lean toward it: leg roll about +x = -lam * st_side
    st_side = -sw_side
    alpha = -st_side * p["lam"] * s
    psi = -st_side * p["psi"] * h                 # pelvis rolls down on the stance side
    bump = p["bump"] * math.sin(math.pi * prog) ** 2 if on else 0.0
    lk, lh = p.get("lift_knee", 0.0) * h, p.get("lift_hip", 0.0) * h   # flamingo lift: knee flexion with the hike
    q = np.zeros(10)
    st_idx, sw_idx = (0, 5) if st_side == 1 else (5, 0)
    th0 = p.get("theta0", 0.0)                   # both feet a little ahead of the hips (the centre of mass sits behind)
    q[st_idx:st_idx + 5] = leg_q(th_st + th0, alpha, psi, st_side)
    q[sw_idx:sw_idx + 5] = leg_q(th_sw + th0, alpha, psi, sw_side, bump + lk, bump / 2 + lh)
    stance = np.ones(2, bool)
    z = np.zeros(2)
    if on and t2 + 0.5 * p["hike_t"] <= tau < t4 + 0.5 * p["lower_t"]:
        stance[leg] = False
    z[leg] = 2 * W_HIP * math.sin(p["psi"] * h)
    return q, stance, z


def balance_pose(t, p, on):
    th0 = p.get("theta0", 0.0)
    q = np.zeros(10)
    q[[2, 3, 7, 8]] = th0
    stance = np.ones(2, bool)
    z = np.zeros(2)
    if not on:
        return q, stance, z
    t1 = p["settle0"]; t2 = t1 + p["shift_t"]; t3 = t2 + p["hike_t"]; t4 = t3 + p["hold"]; t5 = t4 + p["lower_t"]
    s = _ease((t - t1) / p["shift_t"]) * (1 - _ease((t - t5) / p["unshift_t"]))
    h = _ease((t - t2) / p["hike_t"]) * (1 - _ease((t - t4) / p["lower_t"]))
    alpha, psi = -p["lam"] * s, -p["psi"] * h      # stance left
    q[0:5] = leg_q(th0, alpha, psi, 1)
    q[5:10] = leg_q(th0, alpha, psi, -1, (p["bump"] + p.get("lift_knee", 0.0)) * h, (p["bump"] / 2 + p.get("lift_hip", 0.0)) * h)
    if t2 + 0.5 * p["hike_t"] <= t < t4 + 0.5 * p["lower_t"]:
        stance[1] = False
    z[1] = 2 * W_HIP * math.sin(p["psi"] * h)
    return q, stance, z


def build(kind, p, vs, nphase):
    Q = np.zeros((len(vs), nphase, 10)); S = np.zeros((len(vs), nphase, 2)); Z = np.zeros((len(vs), nphase, 2))
    if kind == "step":
        Tc = 2 * p["T_step"]
        assert p["settle0"] + p["shift_t"] + p["hike_t"] + p["swing_t"] + p["lower_t"] + p["unshift_t"] <= p["T_step"] + 1e-9
    else:
        Tc = p["settle0"] + p["shift_t"] + 2 * p["hike_t"] + p["hold"] + p["lower_t"] + p["unshift_t"] + p["settle1"]
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
    ap.add_argument("--T-step", type=float, default=4.0)
    ap.add_argument("--settle0", type=float, default=0.3)
    ap.add_argument("--shift-t", type=float, default=0.9)
    ap.add_argument("--hike-t", type=float, default=0.4)
    ap.add_argument("--swing-t", type=float, default=0.8)
    ap.add_argument("--lower-t", type=float, default=0.6)
    ap.add_argument("--unshift-t", type=float, default=0.6)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--settle1", type=float, default=0.6)
    ap.add_argument("--lam", type=float, default=0.124)
    ap.add_argument("--psi", type=float, default=0.12)
    ap.add_argument("--bump", type=float, default=0.0)
    ap.add_argument("--lift-knee", type=float, default=0.0, help="rad of swing-knee flexion at full lift (flamingo)")
    ap.add_argument("--lift-hip", type=float, default=0.0, help="rad of swing-hip flexion at full lift")
    ap.add_argument("--theta0", type=float, default=0.0, help="rad: both thighs (= knees) forward: feet ahead of the hips")
    ap.add_argument("--vmax", type=float, default=0.008)
    ap.add_argument("--nv", type=int, default=5)
    ap.add_argument("--nphase", type=int, default=512)
    a = ap.parse_args()
    p = dict(T_step=a.T_step, settle0=a.settle0, shift_t=a.shift_t, hike_t=a.hike_t, swing_t=a.swing_t,
             lower_t=a.lower_t, unshift_t=a.unshift_t, hold=a.hold, settle1=a.settle1, lam=a.lam, psi=a.psi, bump=a.bump,
             theta0=a.theta0, lift_knee=a.lift_knee, lift_hip=a.lift_hip)
    vs = np.linspace(0, a.vmax, a.nv) if a.kind == "step" else np.array([0.0, a.vmax])
    t = build(a.kind, p, vs, a.nphase)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(REF_DIR / f"{a.name}.npz", **t)
    print(f"{a.name}: {a.kind}, cycle {1 / t['gait_hz']:.2f} s (gait_hz {t['gait_hz']:.6f}), v rows {np.round(vs, 4)}"
          + (f", stagger at vmax {100 * a.vmax * a.T_step:.1f} cm (foot moves {200 * a.vmax * a.T_step:.1f} cm)" if a.kind == "step" else ""))


if __name__ == "__main__":
    main()
