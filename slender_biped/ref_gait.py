"""Reference gait for imitation-shaped walking (round 4).

A kinematic walking pattern for the as-built leg, solved once by IK on the MuJoCo model and stored as a
lookup table over (forward speed command, gait phase):

* Foot paths in the pelvis frame. Stance: the tip moves back at the walking speed. Swing: a cycloid forward
  (zero speed at lift-off and touchdown) with a smooth height bump.
* Lateral weight shift from the linear inverted pendulum's periodic orbit (point contact: the pelvis sways
  toward the stance foot, peaking at mid-stance), scaled by `sway`.
* Step width `width` (centre-to-centre of the tips); the hips are 80 mm apart, so < 80 mm narrows the track
  ("catwalk" placement). Optional pelvis twist through the hip-yaw joints (`yaw_twist`).

Phase convention matches mjx_env: left stance while sin(phase) >= 0. With duty > 0.5 both tips are down
around the hand-over. At zero speed the reference is the standing pose (no stepping).

    .venv/bin/python -m slender_biped.ref_gait --model hw_p0_8dof --name ref3 --steps-per-s 3.0

The F_board150w tables (2026-11-08; L = legs-only model, F = with torso and head):
    ... --model hw_pad_board150w_legs_8dof --name b150L_flats --flat --smooth 2 --steps-per-s 2.4 --duty 0.6 \
        --swing-h 0.05 --width 0.085 --vmax 0.16 --nv 9                                  # walking
    ... --model hw_pad_board150w_legs_8dof --name b150L_settle --flat --nphase 96 --steps-per-s 1.6 --joint-swing \
        --swing-frac 0.48 --swing-h 0.01 --sway 0 --full-h-speed 0.005 --width 0.081 --vmax 0.08 --nv 9  # step-and-settle
    ... --model hw_pad_board150w_legs_8dof --name b150L_skiflat --flat --smooth 2 --steps-per-s 2.0 --duty 0.6 \
        --swing-h 0.008 --width 0.09 --vmax 0.12 --nv 9                                  # skiing
writes models/ref/<name>.npz and prints the joint-speed demand against the servos' no-load speed.
"""
from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from .config import MODEL_DIR, load_actuators

REF_DIR = MODEL_DIR / "ref"


def _foot_paths(v, phase, p):
    """Tip-centre offsets (dx, dy_sway, dz) per leg in the pelvis frame, plus the stance flag."""
    T = 2.0 / p["steps_per_s"]                       # gait cycle (two steps)
    if p.get("swing_frac", 0.0) > 0:
        return _settle_paths(v, phase, p, T)
    beta = p["duty"]
    L = v * beta * T                                 # tip travel during stance = pelvis travel
    h = p["swing_h"] * min(1.0, v / p["full_h_speed"]) if v > 0 else 0.0
    out = np.zeros((2, 3))
    stance = np.zeros(2, bool)
    for leg, centre in ((0, 1.5 * np.pi), (1, 0.5 * np.pi)):           # swing centred here (L: sin < 0)
        sw = (1 - beta) * 2 * np.pi
        start = centre - sw / 2
        psi = ((phase - start) % (2 * np.pi)) / (2 * np.pi)            # 0 at lift-off
        sig = 1 - beta
        if psi < sig:
            s = psi / sig
            out[leg, 0] = -L / 2 + L * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
            out[leg, 2] = h * (0.5 - 0.5 * np.cos(2 * np.pi * s)) ** p["lift_shape"]
        else:
            s = (psi - sig) / beta
            out[leg, 0] = L / 2 - L * s
            stance[leg] = True
    # lateral sway: LIPM periodic orbit, pelvis toward the stance foot (left stance: sin(phase) >= 0)
    if v > 0:
        Ts, Tc = T / 2, np.sqrt(p["com_h"] / 9.81)
        w2 = p["width"] / 2
        ymin = w2 / np.cosh(Ts / (2 * Tc))
        tau = ((phase % np.pi) / np.pi - 0.5) * Ts                    # time from mid-stance
        y_rel = w2 - ymin * np.cosh(tau / Tc)                        # pelvis offset toward the stance side
        side = 1.0 if np.sin(phase) >= 0 else -1.0                    # +y = left
        ramp = min(1.0, v / p["full_h_speed"])
        out[:, 1] = -side * y_rel * p["sway"] * ramp                  # feet move opposite to the pelvis
    return out, stance


def _settle_paths(v, phase, p, T):
    """Step-and-settle (2026-11-08): each half cycle is a quick swing (swing_frac of it) and then a pause with both
    boards down. The pelvis advances only while one foot is up: the stance foot sweeps back by the step S = v T / 2
    during the other leg's swing (same cycloid as the swing, so the body comes to rest at touchdown) and nothing moves
    while both boards are down, so a rock started by the step can die out before the next one. No lateral sway (the
    boards stay flat), and the full lift at any non-zero speed (a short step still clears the rug)."""
    S = v * T / 2
    h = p["swing_h"] * min(1.0, v / p["full_h_speed"]) if v > 0 else 0.0
    w = p["swing_frac"] / 2                                             # swing window, fraction of the cycle
    u = phase / (2 * np.pi)
    cyc = lambda s: s - np.sin(2 * np.pi * s) / (2 * np.pi)            # noqa: E731
    out = np.zeros((2, 3))
    stance = np.ones(2, bool)
    for leg, centre in ((0, 0.75), (1, 0.25)):                         # swing centred here (L: sin < 0)
        own = (u - (centre - w / 2)) % 1.0
        oth = (u - (centre + 0.5 - w / 2)) % 1.0
        if own < w:
            s = own / w
            out[leg, 0] = -S / 2 + S * cyc(s)
            out[leg, 2] = h * (0.5 - 0.5 * np.cos(2 * np.pi * s)) ** p["lift_shape"]
            stance[leg] = False
        elif oth < w:
            out[leg, 0] = S / 2 - S * cyc(oth / w)
        else:
            out[leg, 0] = S / 2 if own < 0.5 else -S / 2
    return out, stance


def build(model: str, steps_per_s=3.0, duty=0.55, swing_h=0.025, width=0.081, sway=1.0, yaw_twist_deg=0.0,
          lift_shape=1.0, full_h_speed=0.05, vmax=0.3, nv=13, nphase=64, flat=False, w_pitch=0.1, smooth=3.0, swing_frac=0.0,
          joint_swing=False, bump_knee=0.25, bump_hip=0.12, lean_deg=0.0):
    """flat (2026-11-08, long rigid soles without ankles): a stance sole stays level instead of the pelvis keeping its
    height. The leg has only two sagittal joints, so a level sole fixes the shin angle: the pelvis then bobs (the thigh
    swings about the knee) and the swing foot's height is measured from the stance sole. Single stance: the stance leg
    solves (x, y, sole pitch = 0), the swing leg (x, y, floor + lift). Double support: both legs weighted least squares
    on (x, y, floor, pitch), w_pitch metres per radian. Switching between those tasks kinks the joint paths (knee
    jumps ~11 deg in one sample), so the table is then smoothed around the cycle (circular Gaussian, `smooth`
    samples) and the tip heights recomputed."""
    m = mujoco.MjModel.from_xml_path(str(MODEL_DIR / f"{model}.xml"))
    d = mujoco.MjData(m)
    meta = json.loads((MODEL_DIR / f"{model}.json").read_text())
    joints = meta["joints"]
    q_nom = np.array(meta["nominal_q"])
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    dadr = np.array([m.jnt_dofadr[j] for j in jid])
    lo = np.array([m.jnt_range[j][0] for j in jid])
    hi = np.array([m.jnt_range[j][1] for j in jid])
    tip = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("L_tip", "R_tip")]
    size = m.geom_size[tip[0]]

    def fk(q):
        mujoco.mj_resetDataKeyframe(m, d, kid)
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        return d.geom_xpos[tip].copy()

    def tip_low(q):
        fk(q)
        R = d.geom_xmat[tip].reshape(2, 3, 3)
        return d.geom_xpos[tip][:, 2] - np.abs(size * R[:, 2, :]).sum(-1)

    p0 = fk(q_nom)
    hip_y = np.array([p0[0, 1], p0[1, 1]])
    p = dict(steps_per_s=steps_per_s, duty=duty, swing_h=swing_h, width=width, sway=sway,
             full_h_speed=full_h_speed, lift_shape=lift_shape, com_h=float(d.subtree_com[1][2]), swing_frac=swing_frac,
             joint_swing=joint_swing, bump_knee=bump_knee, bump_hip=bump_hip, lean_deg=lean_deg)
    vs = np.linspace(0, vmax, nv)
    phases = np.linspace(0, 2 * np.pi, nphase, endpoint=False)
    Q = np.zeros((nv, nphase, len(joints)))
    Z = np.zeros((nv, nphase, 2))
    S = np.zeros((nv, nphase, 2))
    err_max = 0.0
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    if flat:
        return _build_flat(m, d, fk, tip_low, tip, p0, hip_y, p, vs, phases, joints, qadr, dadr, lo, hi, q_nom, jacp,
                           jacr, width, yaw_twist_deg, full_h_speed, steps_per_s, w_pitch, model, smooth)
    for iv, v in enumerate(vs):
        q = q_nom.copy()
        for ip, ph in enumerate(phases):
            off, stance = _foot_paths(v, ph, p)
            # narrower track: move both tips toward the midline by (hip spacing - width) / 2
            tgt = p0.copy()
            tgt[:, 0] += off[:, 0]
            tgt[:, 1] = np.sign(hip_y) * width / 2 + off[:, 1]
            tgt[:, 2] += off[:, 2]
            yaw = np.radians(yaw_twist_deg) * np.cos(ph) * (v > 0) * min(1.0, v / full_h_speed)
            q[0], q[4] = yaw, yaw            # both hip-yaw joints: pelvis twist (R axis is mirrored)
            for _ in range(3 if ip else 30):
                for it in range(40):
                    x = fk(q)
                    e = (tgt - x)
                    if np.abs(e).max() < 1e-5:
                        break
                    for leg, idx in ((0, [1, 2, 3]), (1, [5, 6, 7])):
                        mujoco.mj_jac(m, d, jacp, None, x[leg], m.geom_bodyid[tip[leg]])
                        J = jacp[:, dadr[idx]]
                        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e[leg])
                        q[idx] = np.clip(q[idx] + dq, lo[idx], hi[idx])
            err_max = max(err_max, float(np.abs(tgt - fk(q)).max()))
            Q[iv, ip] = q
            Z[iv, ip] = tip_low(q)
            S[iv, ip] = stance
    Z -= Z[0, 0].min()                         # standing: lowest tip point = 0
    return dict(q=Q, tip_z=Z, stance=S, v=vs, nphase=nphase, gait_hz=steps_per_s / 2, ik_err=err_max,
                params=json.dumps(dict(p, yaw_twist_deg=yaw_twist_deg, model=model)))


def _build_flat(m, d, fk, tip_low, tip, p0, hip_y, p, vs, phases, joints, qadr, dadr, lo, hi, q_nom, jacp, jacr,
                width, yaw_twist_deg, full_h_speed, steps_per_s, w_pitch, model, smooth):
    nv, nphase = len(vs), len(phases)
    Q = np.zeros((nv, nphase, len(joints)))
    Z = np.zeros((nv, nphase, 2))
    S = np.zeros((nv, nphase, 2))
    err_max = 0.0
    legs = ((0, [1, 2, 3]), (1, [5, 6, 7]))
    z0 = p0[:, 2].copy()                                   # sole centre height in the stance pose (pelvis frame)
    size = m.geom_size[tip[0]]
    settle = p.get("swing_frac", 0.0) > 0
    swing_low, w_swing = settle, 0.02                      # step-and-settle: clearance of the board's lowest point
    def solve(q, tgt, stance, off, outer):
        floor = float(np.mean(fk(q)[stance, 2])) if stance.any() else z0.mean()
        for _ in range(outer):                             # outer: floor from the stance soles, inner: IK
            for it in range(60):
                x = fk(q)
                R = d.geom_xmat[tip].reshape(2, 3, 3)
                done = True
                for leg, idx in legs:
                    mujoco.mj_jac(m, d, jacp, jacr, x[leg], m.geom_bodyid[tip[leg]])
                    pitch = np.arcsin(np.clip(-R[leg][2, 0], -1, 1))
                    if stance[leg] and stance.all():                 # double support: x, y, floor, level
                        J = np.vstack([jacp[:, dadr[idx]], w_pitch * jacr[1:2, dadr[idx]]])
                        e = np.r_[tgt[leg, 0] - x[leg, 0], tgt[leg, 1] - x[leg, 1], floor - x[leg, 2],
                                  -w_pitch * pitch]
                    elif stance[leg]:                                # single stance: x, y, level (z free)
                        J = np.vstack([jacp[:2, dadr[idx]], w_pitch * jacr[1:2, dadr[idx]]])
                        e = np.r_[tgt[leg, 0] - x[leg, 0], tgt[leg, 1] - x[leg, 1], -w_pitch * pitch]
                    elif swing_low:                                  # swing: x, y, LOWEST point floor + lift, ~level
                        hl = size[0]                                 # board half length
                        sg = np.sign(pitch) if pitch != 0 else 1.0
                        Jz = jacp[2, dadr[idx]] - hl * sg * jacr[1, dadr[idx]]
                        J = np.vstack([jacp[:2, dadr[idx]], Jz, w_swing * jacr[1:2, dadr[idx]]])
                        e = np.r_[tgt[leg, 0] - x[leg, 0], tgt[leg, 1] - x[leg, 1],
                                  floor + off[leg, 2] - (x[leg, 2] - hl * abs(pitch)), -w_swing * pitch]
                    else:                                            # swing: x, y, floor + lift
                        J = jacp[:, dadr[idx]]
                        e = np.r_[tgt[leg, 0] - x[leg, 0], tgt[leg, 1] - x[leg, 1], floor + off[leg, 2] - x[leg, 2]]
                    if np.abs(e).max() > 1e-5 and not (swing_low and not stance[leg]):
                        done = False
                    dq = np.linalg.lstsq(J.T @ J + 1e-6 * np.eye(3), J.T @ e, rcond=None)[0]
                    q[idx] = np.clip(q[idx] + dq, lo[idx], hi[idx])
                if done:
                    break
            floor = float(np.mean(fk(q)[stance, 2])) if stance.any() else floor
        return q

    def targets(v, ph, c):
        off, stance = _foot_paths(v, ph, p)
        tgt = p0.copy()
        tgt[:, 0] += off[:, 0] + c
        tgt[:, 1] = np.sign(hip_y) * width / 2 + off[:, 1]
        return tgt, stance, off

    def settle_offset(v):
        """Step-and-settle: staggered boards are both level and down only with the pelvis shifted fore-aft over them
        (front knee bent, rear leg straighter: the two IK branches of a flat sole). Find that shift c."""
        from scipy.optimize import minimize_scalar

        def resid(c):
            tgt, stance, off = targets(v, np.pi, c)                # mid-pause
            qq = solve(q_nom.copy(), tgt, stance, off, 12)
            low = tip_low(qq)
            R = d.geom_xmat[tip].reshape(2, 3, 3)
            return abs(low[0] - low[1]) + 0.05 * np.abs(np.arcsin(np.clip(-R[:, 2, 0], -1, 1))).sum()
        r = minimize_scalar(resid, bounds=(-0.06, 0.06), method="bounded", options=dict(xatol=2e-4))
        return float(r.x), float(r.fun)

    shifts = []
    err_where = None
    if settle and p.get("joint_swing"):
        return _settle_joint(p, vs, phases, joints, q_nom, settle_offset, targets, solve, tip_low, model, steps_per_s)
    for iv, v in enumerate(vs):
        q = q_nom.copy()
        c = 0.0
        if settle:            # every row, v = 0 too: standing still is the walking pose (both thighs vertical)
            c, res = settle_offset(max(v, 1e-4))
            shifts.append((float(v), c, res))
        for ip, ph in enumerate(phases):
            tgt, stance, off = targets(v, ph, c)
            yaw = np.radians(yaw_twist_deg) * np.cos(ph) * (v > 0) * min(1.0, v / full_h_speed)
            q[0], q[4] = yaw, yaw
            q = solve(q, tgt, stance, off, 4 if ip else 12)
            x = fk(q)
            e_ = np.abs(tgt[:, :2] - x[:, :2]).max(-1)
            e_ = e_[stance].max() if settle and stance.any() else e_.max()  # settle: swing x,y is a compromise
            if e_ > err_max: err_where = (float(v), ip)
            err_max = max(err_max, float(e_))
            Q[iv, ip] = q
            S[iv, ip] = stance
        if smooth > 0:
            k = np.arange(nphase)
            k = np.minimum(k, nphase - k)
            w = np.exp(-0.5 * (k / smooth) ** 2)
            w /= w.sum()
            F = np.fft.rfft(w)
            Q[iv] = np.fft.irfft(np.fft.rfft(Q[iv], axis=0) * F[:, None], n=nphase, axis=0)
        for ip in range(nphase):
            low = tip_low(Q[iv, ip])
            st = S[iv, ip].astype(bool)
            Z[iv, ip] = low - (low[st].min() if st.any() else low.min())        # height above the stance sole
    return dict(q=Q, tip_z=Z, stance=S, v=vs, nphase=nphase, gait_hz=steps_per_s / 2, ik_err=err_max,
                params=json.dumps(dict(p, yaw_twist_deg=yaw_twist_deg, model=model, flat=True, settle_shift=shifts, err_where=err_where)))


def _settle_joint(p, vs, phases, joints, q_nom, settle_offset, targets, solve, tip_low, model, steps_per_s):
    """Step-and-settle in joint space (2026-11-08). The two pause poses (boards level and staggered by the step, from
    IK) are held while both boards are down; during a swing the stance leg interpolates between them (hip pitch and knee
    change by the same angle: a thigh rotation, its board stays level) and the swing leg too, plus a lift bump of
    knee `bump_knee` and hip pitch `bump_hip` rad (the servos' ~35 rad/s^2 profile limits what a 0.3 s swing can do:
    the open-loop IK lift of a level board from a nearly straight rear leg needed ~250 rad/s^2). The tip-height
    target (ref_tip reward) asks for `swing_h` of clearance at the board's lowest point independently, so the policy
    is paid for finding the lift (body roll onto the stance board does most of it)."""
    nv, nphase = len(vs), len(phases)
    Q = np.zeros((nv, nphase, len(joints)))
    Z = np.zeros((nv, nphase, 2))
    S = np.zeros((nv, nphase, 2))
    w = p["swing_frac"] / 2
    shifts = []
    cyc = lambda s: s - np.sin(2 * np.pi * s) / (2 * np.pi)                # noqa: E731
    for iv, v in enumerate(vs):
        c, res = settle_offset(max(v, 1e-4))
        shifts.append((float(v), c, res))
        pose = {}
        for u in (0.0, 0.5):                     # u = 0: L ahead (L swung last), u = 0.5: R ahead
            tgt, stance, off = targets(v, 2 * np.pi * u, c)
            pose[u] = solve(q_nom.copy(), tgt, stance, off, 12).copy()
        h = p["swing_h"] * min(1.0, v / p["full_h_speed"]) if v > 0 else 0.0
        bk = p["bump_knee"] * (h > 0)
        bh = p["bump_hip"] * (h > 0)
        for ip, ph in enumerate(phases):
            u = ph / (2 * np.pi)
            q = pose[0.0].copy() if (u < 0.25 - w / 2 or u >= 0.75 + w / 2) else pose[0.5].copy()
            st = np.ones(2, bool)
            for leg, centre, a, b in ((1, 0.25, 0.0, 0.5), (0, 0.75, 0.5, 0.0)):
                x = (u - (centre - w / 2)) % 1.0
                if x < w:
                    sp = x / w
                    q = pose[a] + (pose[b] - pose[a]) * cyc(sp)
                    bump = 0.5 - 0.5 * np.cos(2 * np.pi * sp)
                    k = 4 * leg
                    q[k + 3] += bk * bump
                    q[k + 2] += bh * bump
                    st[leg] = False
                    Z[iv, ip, leg] = h * bump
            q[[2, 6]] -= np.radians(p.get("lean_deg", 0.0))   # pelvis pitched back, boards and legs unchanged
            Q[iv, ip] = q
            S[iv, ip] = st
    return dict(q=Q, tip_z=Z, stance=S, v=vs, nphase=nphase, gait_hz=steps_per_s / 2, ik_err=0.0,
                params=json.dumps(dict(p, model=model, flat=True, settle_shift=shifts)))


def lookup(table, vx, phase, xp=np):
    """Bilinear lookup -> (q_ref[nj], tip_z_ref[2], stance_ref[2]). Works with numpy or jax.numpy as xp."""
    v, nph = table["v"], table["q"].shape[1]
    fv = xp.clip(vx / v[-1] * (len(v) - 1), 0.0, len(v) - 1.0)
    i0 = xp.clip(xp.floor(fv).astype(int), 0, len(v) - 2)
    wv = fv - i0
    fp = (phase % (2 * np.pi)) / (2 * np.pi) * nph
    j0 = xp.floor(fp).astype(int) % nph
    j1 = (j0 + 1) % nph
    wp = fp - xp.floor(fp)

    def bil(A):
        a = A[i0, j0] * (1 - wp) + A[i0, j1] * wp
        b = A[i0 + 1, j0] * (1 - wp) + A[i0 + 1, j1] * wp
        return a * (1 - wv) + b * wv

    return bil(table["q"]), bil(table["tip_z"]), bil(table["stance"]) > 0.5


def level_pitch(m, joints, q):
    """Base rotation about y (rad, as in an axis-angle quaternion) that puts both boards level for joint angles q:
    the reference's pelvis lean (lean_deg) is in the hip pitches, so a start pose needs the pelvis pitched to match."""
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
    tip = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("L_tip", "R_tip")]

    def board_pitch(a):
        d.qpos[:] = m.qpos0
        qb = np.zeros(4)
        mujoco.mju_axisAngle2Quat(qb, np.array([0.0, 1.0, 0.0]), a)
        d.qpos[3:7] = qb
        d.qpos[qadr] = np.asarray(q, float)
        mujoco.mj_kinematics(m, d)
        R = d.geom_xmat[tip].reshape(2, 3, 3)
        return float(np.mean(np.arcsin(np.clip(-R[:, 2, 0], -1, 1))))
    a = 0.0
    for _ in range(20):                                          # secant / Newton on a nearly linear function
        f0, f1 = board_pitch(a), board_pitch(a + 1e-4)
        if abs(f0) < 1e-7:
            break
        a -= f0 * 1e-4 / (f1 - f0)
    return a


def load(name, xp=np):
    z = np.load(REF_DIR / f"{name}.npz")
    return {k: xp.asarray(z[k]) for k in ("q", "tip_z", "stance", "v")} | {"gait_hz": float(z["gait_hz"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hw_p0_8dof")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps-per-s", type=float, default=3.0)
    ap.add_argument("--duty", type=float, default=0.55)
    ap.add_argument("--swing-h", type=float, default=0.025)
    ap.add_argument("--width", type=float, default=0.081)
    ap.add_argument("--sway", type=float, default=1.0)
    ap.add_argument("--yaw-twist", type=float, default=0.0)
    ap.add_argument("--lift-shape", type=float, default=1.0)
    ap.add_argument("--vmax", type=float, default=0.3)
    ap.add_argument("--nv", type=int, default=13)
    ap.add_argument("--flat", action="store_true", help="level stance soles, pelvis bobs (long soles, no ankles)")
    ap.add_argument("--swing-frac", type=float, default=0.0,
                    help="step-and-settle: fraction of each step spent swinging (rest: both boards down, nothing moves)")
    ap.add_argument("--full-h-speed", type=float, help="speed at which the full lift is reached (default 0.03 flat, 0.05)")
    ap.add_argument("--nphase", type=int, default=64)
    ap.add_argument("--joint-swing", action="store_true", help="step-and-settle swing built in joint space (see _settle_joint)")
    ap.add_argument("--bump-knee", type=float, default=0.25)
    ap.add_argument("--bump-hip", type=float, default=0.12)
    ap.add_argument("--lean", type=float, default=0.0, help="--joint-swing: pelvis pitched back this many deg (both hip "
                    "pitches minus it; the flat boards keep the legs' world angles), moving the centre of mass aft")
    ap.add_argument("--smooth", type=float, default=3.0, help="with --flat: circular smoothing, samples")
    a = ap.parse_args()
    t = build(a.model, a.steps_per_s, a.duty, a.swing_h, a.width, a.sway, a.yaw_twist, a.lift_shape,
              full_h_speed=a.full_h_speed or (0.03 if a.flat else 0.05), vmax=a.vmax, nv=a.nv, nphase=a.nphase,
              flat=a.flat, smooth=a.smooth, swing_frac=a.swing_frac,
              joint_swing=a.joint_swing, bump_knee=a.bump_knee, bump_hip=a.bump_hip,
              lean_deg=a.lean)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(REF_DIR / f"{a.name}.npz", **t)
    meta = json.loads((MODEL_DIR / f"{a.model}.json").read_text())
    act = load_actuators()
    w0 = np.array([act["catalog"][act["assignments"][meta["cfg"]["actuator_set"]][j.split("_", 1)[1]]]["noload_speed"]
                   for j in meta["joints"]])
    dt = 1 / (t["gait_hz"] * t["q"].shape[1])
    qd = np.abs(np.diff(np.concatenate([t["q"], t["q"][:, :1]], 1), axis=1)) / dt
    print(f"{a.name}: {a.steps_per_s} steps/s, IK max error {t['ik_err']*1000:.2f} mm")
    for iv in sorted({min(i, len(t["v"]) - 1) for i in (2, 6, 10, 12)}):
        v = t["v"][iv]
        step = v / a.steps_per_s if a.swing_frac else v * a.duty * 2 / a.steps_per_s
        print(f"  v={v:.3f}  step {step*100:4.1f} cm  peak |qd|/no-load "
              + " ".join(f"{x:.2f}" for x in (qd[iv].max(0) / w0)[:4])
              + f"  q range (deg) " + " ".join(f"{np.degrees(np.ptp(t['q'][iv][:, k])):.0f}" for k in range(4))
              + f"  swing tip z max {t['tip_z'][iv].max()*1000:.0f} mm")


if __name__ == "__main__":
    main()
