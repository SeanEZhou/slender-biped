"""Interleaved walking references (round 24, 2026-11-18): the weight shift and the step happen together.

The round-23 tables (shift_gait.py) shift the weight over the stance board, step, shift, step: a quasi-static walk with
a 66 mm side-to-side rock and 30 mm steps. Humans and Unitree's G1 interleave the two: the body sways continuously and
only a little, the rear foot lifts as the front one lands, and the stance foot carries the body forward the whole time.

Model (per step of T_step seconds, a cycle is two steps; the right foot swings in the first half):
* Forward: the pelvis moves at a constant speed v. A stance board moves back under it at v for its whole stance
  (double support included); a swing board moves forward with a minimum-jerk profile and a lift bump.
* Duty: each foot is down for `duty` of the cycle, so both feet are down (2 duty - 1) T_step per step
  (human ~0.6, G1's RL gait 0.55).
* Sideways: the linear inverted pendulum's periodic orbit around a pivot `zmp_in` inside each board centre
  (the ankle-roll servos can hold the centre of pressure inside the 50 mm board). The pelvis crosses the midline
  at speed in the middle of the double support and turns around in the middle of the single support: no hold.
  `sway` scales the orbit (1 = the pendulum's own).
* Soles: level in roll always (ankle roll); stance soles level in pitch (no ankle pitch: the shin keeps its angle, the
  thigh rotates, the pelvis bobs a millimetre); the swing sole's pitch is free but kept near level.

    .venv/bin/python -m slender_biped.stride_gait --name s_A --T-step 0.4 --duty 0.6 --zmp-in 0.012 --vmax 0.12
"""
from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np

from .config import MODEL_DIR

REF_DIR = MODEL_DIR / "ref"
MODEL = "hw_ank3_b150_m2_10dof"
LEGS = ((0, [1, 2, 3, 4]), (1, [6, 7, 8, 9]))           # joint indices: hip roll, hip pitch, knee, ankle roll


def _mj(x):
    x = min(1.0, max(0.0, x))
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def _ramp(v, p):
    """round 24: the gait start. Sway and lift grow linearly from 0 (standing) to full at ramp_v m/s; the command
    ramps up from 0 at every start, so the first steps come in small (0 = full at any speed above 0)."""
    rv = p.get("ramp_v", 0.0)
    return min(1.0, v / rv) if rv > 0 else 1.0


def foot_paths(u, v, p):
    """Board-centre offsets (dx, dy, dz) per leg in the pelvis frame from the stance pose, and the stance flags, at
    cycle fraction u. Right swing centred at u = 0.25, left at 0.75; double supports centred at 0 and 0.5."""
    T = p["T_step"]
    beta = p["duty"]
    sw = 1.0 - beta                                     # swing window, fraction of the cycle
    L = v * beta * 2 * T                                # board travel in stance = pelvis travel
    h = p["swing_h"] if v > 0 else 0.0
    out = np.zeros((2, 3))
    stance = np.ones(2, bool)
    prog = np.full(2, -1.0)                             # swing progress 0..1 (-1 in stance)
    sprog = np.full(2, 9.0)                             # stance progress 0..1 (9 in swing)
    for leg, centre in ((1, 0.25), (0, 0.75)):
        s_sw = (u - (centre - sw / 2)) % 1.0           # 0 at lift-off
        if s_sw < sw:
            s = s_sw / sw
            out[leg, 0] = -L / 2 + L * _mj(s)
            pk = p["lift_peak"]                         # lift peak at this fraction of the swing
            b = s / pk if s < pk else 1 - (s - pk) / (1 - pk)
            out[leg, 2] = h * (0.5 - 0.5 * math.cos(math.pi * b))
            stance[leg] = False
            prog[leg] = s
        else:
            s = (s_sw - sw) / beta                      # 0 at touchdown
            sprog[leg] = s
            out[leg, 0] = L / 2 - L * s
    # sideways: pendulum orbit about a pivot zmp_in inside the stance board's centre
    if (v > 0 or p.get("inplace")) and p["sway"] > 0:
        Tc = math.sqrt(p["com_h"] / 9.81)
        piv = p["width"] / 2 - p["zmp_in"]
        ymin = piv / math.cosh(T / (2 * Tc))
        tau = ((u % 0.5) / 0.5 - 0.5) * T              # time from mid single support
        y = (piv - ymin * math.cosh(tau / Tc)) * p["sway"] * _ramp(v, p)
        side = 1.0 if (u % 1.0) < 0.5 else -1.0       # right swings first: the pelvis goes left (+y)
        out[:, 1] = -side * y
    return out, stance, prog, sprog


def build(p, vs, nphase):
    m = mujoco.MjModel.from_xml_path(str(MODEL_DIR / f"{MODEL}.xml"))
    d = mujoco.MjData(m)
    meta = json.loads((MODEL_DIR / f"{MODEL}.json").read_text())
    q_nom = np.array(meta["nominal_q"])
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    tip = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_tip") for s in "LR"]
    size = m.geom_size[tip[0]]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def fk(q):
        d.qpos[:] = 0; d.qpos[2] = meta["stand_base_z"]; d.qpos[3] = 1; d.qpos[7:] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        return d.geom_xpos[tip].copy(), d.geom_xmat[tip].reshape(2, 3, 3).copy()

    def low(q):
        x, R = fk(q)
        return x[:, 2] - np.abs(size * R[:, 2, :]).sum(-1)

    x0, _ = fk(q_nom)
    p = dict(p, com_h=float(d.subtree_com[0][2]))     # whole-robot centre of mass, standing (0.344 m)
    EPS = 0.005                                        # rad: the soft |pitch| of the board's lowest point

    def resid(q, leg, tgt, z_t):
        """Four exact targets for one leg's four joints: board centre x, y; level roll; then a level board in pitch
        (z_t None: the height follows) or the board's lowest point at z_t (the pitch follows: toe-off, heel landing,
        a lifted board tips)."""
        x, R = fk(q)
        roll = math.atan2(R[leg][2, 1], R[leg][2, 2]); pitch = math.asin(max(-1.0, min(1.0, -R[leg][2, 0])))
        if z_t is None:
            last = pitch
        else:
            zt, wc = z_t if isinstance(z_t, tuple) else (z_t, 0.0)     # wc: weight of the centre (mid-swing) height
            zl = x[leg, 2] - size[2] - size[0] * (math.hypot(pitch, EPS) - EPS)
            last = (1 - wc) * zl + wc * (x[leg, 2] - size[2]) - zt
        return np.array([x[leg, 0] - tgt[leg, 0], x[leg, 1] - tgt[leg, 1], 0.3 * roll, last])

    def solve_leg(q, leg, tgt, z_t, n=40):
        idx = LEGS[leg][1]
        for _ in range(n):
            r0 = resid(q, leg, tgt, z_t)
            if np.abs(r0).max() < 1e-6:
                break
            J = np.zeros((4, 4))
            for j, i in enumerate(idx):
                qq = q.copy(); qq[i] += 1e-5
                J[:, j] = (resid(qq, leg, tgt, z_t) - r0) / 1e-5
            dq = np.linalg.solve(J.T @ J + 1e-9 * np.eye(4), -J.T @ r0)
            dq *= min(1.0, 0.15 / max(np.abs(dq).max(), 1e-12))          # at most ~9 deg per iteration
            q[idx] = np.clip(q[idx] + dq, lo[idx], hi[idx])
        return q, float(np.abs(resid(q, leg, tgt, z_t)[[0, 1, 3]]).max())


    def level_h(q, tgt, leg):
        qq, _ = solve_leg(q.copy(), leg, tgt, None)
        x, R = fk(qq)
        return float(x[leg, 2] - size[2])

    Q = np.zeros((len(vs), nphase, 10)); Z = np.zeros((len(vs), nphase, 2)); S = np.zeros((len(vs), nphase, 2))
    err = 0.0
    dsl = (2 * p["duty"] - 1) / 2                         # one double support, fraction of the cycle
    dt = 2 * p["T_step"] / nphase
    for iv, v in enumerate(vs):
        q = q_nom.copy()
        prev_st = np.ones(2, bool)
        for rep in range(2):                                 # second pass starts from the cycle's end (periodic)
            for ip in range(nphase):
                u = ip / nphase
                off, st, prog, sprog = foot_paths(u, v, p)
                tgt = x0.copy()
                xv = p.get("x_off_v", 0.0)               # round 24: the offset grows with speed (0 standing)
                tgt[:, 0] += off[:, 0] + p["x_off"] * (min(1.0, v / xv) if xv > 0 else 1.0)
                tgt[:, 1] = np.array([1.0, -1.0]) * p["width"] / 2 + off[:, 1]
                for leg in (0, 1):                           # a leg just landed: restart its solve from the stance pose
                    if st[leg] and not prev_st[leg]:
                        q[LEGS[leg][1]] = q_nom[LEGS[leg][1]]
                prev_st = st
                h = {leg: level_h(q, tgt, leg) for leg in (0, 1) if st[leg]}
                if st.all():                               # double support: the pelvis height passes rear -> front
                    front = int(np.argmin(sprog))
                    lam = min(1.0, sprog[front] * p["duty"] / max(dsl, 1e-6))
                    lam = 0.5 - 0.5 * math.cos(math.pi * lam)
                    floor = (1 - lam) * h[1 - front] + lam * h[front]
                    z_t = {0: floor, 1: floor}
                else:
                    sl = int(np.argmax(st)); floor = h[sl]
                    z_t = {sl: None}
                w = 0.0
                for leg in z_t:
                    q, wl = solve_leg(q, leg, tgt, z_t[leg])
                    w = max(w, wl)
                if rep:
                    err = max(err, w)
                    Q[iv, ip] = q
                    S[iv, ip] = st
        # swings: cubic Hermite in joint space from the lift-off pose to the touchdown pose (the stance joint
        # velocities at both ends), plus a lift bump on hip pitch and knee (no ankle pitch: a knee bend tips the
        # board toe-down, a hip bend swings it forward and up)
        bump_k, bump_h = (p["lift_knee"] * _ramp(v, p), p["lift_hip"] * _ramp(v, p)) if (v > 0 or p.get("inplace")) else (0.0, 0.0)
        for leg, idx in LEGS:
            st = S[iv, :, leg].astype(bool)
            if st.all():
                continue
            lo_i = [i for i in range(nphase) if st[i] and not st[(i + 1) % nphase]][0]      # last stance sample
            td_i = [i for i in range(nphase) if st[i] and not st[i - 1]][0]                   # first stance sample
            n = (td_i - lo_i) % nphase
            qa, qb = Q[iv, lo_i, idx], Q[iv, td_i, idx]
            va = (Q[iv, lo_i, idx] - Q[iv, lo_i - 1, idx]) / dt
            vb = (Q[iv, (td_i + 1) % nphase, idx] - Q[iv, td_i, idx]) / dt
            Tn = n * dt
            for k in range(1, n):
                s_ = k / n
                h00, h10, h01, h11 = 2*s_**3 - 3*s_**2 + 1, s_**3 - 2*s_**2 + s_, -2*s_**3 + 3*s_**2, s_**3 - s_**2
                qq = h00 * qa + h10 * Tn * va + h01 * qb + h11 * Tn * vb
                pk = p["lift_peak"]
                bb = s_ / pk if s_ < pk else 1 - (s_ - pk) / (1 - pk)
                bump = (0.5 - 0.5 * math.cos(math.pi * bb)) ** p.get("lift_pow", 1.5)
                qq[1] += bump_h * bump; qq[2] += bump_k * bump                 # idx order: roll, pitch, knee, ankle
                Q[iv, (lo_i + k) % nphase, idx] = qq
        if p["smooth"] > 0:                                  # circular Gaussian along the cycle
            k = np.arange(nphase); k = np.minimum(k, nphase - k)
            w = np.exp(-0.5 * (k / p["smooth"]) ** 2); w /= w.sum()
            Q[iv] = np.fft.irfft(np.fft.rfft(Q[iv], axis=0) * np.fft.rfft(w)[:, None], n=nphase, axis=0)
        for ip in range(nphase):
            lw = low(Q[iv, ip]); st = S[iv, ip].astype(bool)
            Z[iv, ip] = lw - lw[st].min()
    return dict(q=Q, tip_z=Z, stance=S, v=vs, nphase=nphase, gait_hz=1.0 / (2 * p["T_step"]), ik_err=err,
                params=json.dumps(dict(p, model=MODEL, kind="stride")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--T-step", type=float, default=0.4)
    ap.add_argument("--duty", type=float, default=0.6)
    ap.add_argument("--width", type=float, default=0.081, help="board centre to centre (m)")
    ap.add_argument("--zmp-in", type=float, default=0.012, help="sway pivot inside the board centre (m)")
    ap.add_argument("--sway", type=float, default=1.0)
    ap.add_argument("--swing-h", type=float, default=0.02)
    ap.add_argument("--lift-peak", type=float, default=0.45)
    ap.add_argument("--lift-knee", type=float, default=0.5, help="swing knee bump (rad)")
    ap.add_argument("--lift-hip", type=float, default=0.3, help="swing hip-pitch bump (rad)")
    ap.add_argument("--lift-pow", type=float, default=1.5, help="bump shape: (raised cosine) ** this")
    ap.add_argument("--x-off", type=float, default=0.0)
    ap.add_argument("--x-off-v", type=float, default=0.0, help="round 24: speed (m/s) at which the full --x-off applies, "
                    "linear from 0 standing (0 = the full offset at every speed). The fixed offset left the centre of mass "
                    "30 mm ahead of the boards when standing / starting: forward falls in the first second")
    ap.add_argument("--ramp-v", type=float, default=0.0, help="round 24: sway and lift full from this speed (m/s), "
                    "linear from 0 standing (the gait start)")
    ap.add_argument("--inplace", action="store_true", help="round 24: the zero-speed row steps in place with the full "
                    "sway and lift (for the env's start_u gait start) instead of standing")
    ap.add_argument("--vmax", type=float, default=0.12)
    ap.add_argument("--nv", type=int, default=5)
    ap.add_argument("--nphase", type=int, default=128)
    ap.add_argument("--smooth", type=float, default=1.0, help="circular Gaussian, samples")
    a = ap.parse_args()
    p = dict(T_step=a.T_step, duty=a.duty, width=a.width, zmp_in=a.zmp_in, sway=a.sway, swing_h=a.swing_h,
             lift_peak=a.lift_peak, x_off=a.x_off, x_off_v=a.x_off_v, ramp_v=a.ramp_v, inplace=a.inplace, smooth=a.smooth,
             lift_knee=a.lift_knee, lift_hip=a.lift_hip, lift_pow=a.lift_pow)
    t = build(p, np.linspace(0, a.vmax, a.nv), a.nphase)
    np.savez(REF_DIR / f"{a.name}.npz", **t)
    Q = t["q"][-1]
    qd = np.degrees(np.abs(np.diff(np.r_[Q, Q[:1]], axis=0)).max(0) * a.nphase * t["gait_hz"])
    print(f"{a.name}: ik err {1000 * t['ik_err']:.2f} mm; both down {(2 * a.duty - 1) * a.T_step:.3f} s per step; "
          f"peak deg/s at vmax: roll {qd[[1, 6]].max():.0f} pitch {qd[[2, 7]].max():.0f} knee {qd[[3, 8]].max():.0f} "
          f"ankle {qd[[4, 9]].max():.0f}")


if __name__ == "__main__":
    main()
