"""Round 14 (2026-11-15): recovery-step references, chosen by the capture point.

A recovery step, over one recovery cycle (u 0 -> 1, rec_T s), for a swing side A (L / R) and a step displacement
D = (dx, +-dy outward):
    u 0.00-0.08  weight shift: the pelvis moves `shift` toward the other foot B
    u 0.08-0.27  foot A swings by D (lift `lift`), lands
    u 0.27-0.45  the pelvis moves over the new support, then toward A
    u 0.45-0.64  foot B swings by D (the closing step): the feet are back in the stance layout, displaced by D
    u 0.64-0.80  the pelvis centres; 0.80-1.00 hold
The joint angles at u = 0 and u = 1 are the same stance (the walker's zero-speed pose), so a policy idles on it.
Joint angles by damped least-squares IK on the model (feet: pad centre position, board flat; pitch weighted low,
since without ankle pitch a fore-aft step tilts the boards a little), pelvis level at the stance height.

Table: q [2 sides, n_dx, n_dy, n_phase, nj], tip_z / stance [..., 2], grids dx, dy (dy = outward distance >= 0).

    .venv-hw/bin/python -m slender_biped.rec_step build --name rs1   # -> models/ref/rs1.npz
"""
from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np

from .config import MODEL_DIR

REF_DIR = MODEL_DIR / "ref"
G = 9.81


def _ss(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def plan(u, side, dx, dy, shift=0.03, lift=0.03):
    """Pelvis xy offset, foot A / B displacement (x, y, z), stance flags (A down, B down) at cycle fraction u.
    side +1: A = left (outward +y), -1: A = right (outward -y)."""
    D = np.array([dx, side * dy])
    yA = np.array([0.0, side])                      # toward A
    seg = lambda a, b: (u - a) / (b - a)            # noqa: E731
    S1 = -shift * yA
    S2 = 0.5 * D + shift * yA
    if u < 0.08:
        pel = S1 * _ss(seg(0, 0.08))
    elif u < 0.27:
        pel = S1 + (0.5 * D - S1) * _ss(seg(0.08, 0.27)) * 0.5
    elif u < 0.45:
        p0 = S1 + (0.5 * D - S1) * 0.5
        pel = p0 + (S2 - p0) * _ss(seg(0.27, 0.45))
    elif u < 0.64:
        pel = S2 + (D + 0.5 * shift * yA - S2) * _ss(seg(0.45, 0.64)) * 0.5
    elif u < 0.80:
        p0 = S2 + (D + 0.5 * shift * yA - S2) * 0.5
        pel = p0 + (D - p0) * _ss(seg(0.64, 0.80))
    else:
        pel = D.copy()
    def swing(a, b):
        t = seg(a, b)
        if t <= 0:
            return np.zeros(3), True
        if t >= 1:
            return np.r_[D, 0.0], True
        return np.r_[D * _ss(t), lift * math.sin(math.pi * t)], False
    fA, dnA = swing(0.08, 0.27)
    fB, dnB = swing(0.45, 0.64)
    return pel, fA, fB, (dnA, dnB)


class _IK:
    def __init__(self, model):
        m = mujoco.MjModel.from_xml_path(str(MODEL_DIR / f"{model}.xml"))
        self.m, self.d = m, mujoco.MjData(m)
        meta = json.loads((MODEL_DIR / f"{model}.json").read_text())
        self.joints = meta["joints"]
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in self.joints])
        self.dadr = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in self.joints])
        self.lo = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][0] for j in self.joints])
        self.hi = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)][1] for j in self.joints])
        self.tip = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("L_tip", "R_tip")]

    def fk(self, base_pos, base_quat, q):
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[0:3], d.qpos[3:7] = base_pos, base_quat
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d.geom_xpos[self.tip].copy(), d.geom_xmat[self.tip].reshape(2, 3, 3).copy()

    def solve(self, base_pos, base_quat, q0, P, R, iters=60, w_pitch=0.15):
        q = q0.copy()
        m, d = self.m, self.d
        for _ in range(iters):
            p, Rc = self.fk(base_pos, base_quat, q)
            rows, J = [], []
            for f in range(2):
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jacGeom(m, d, jp, jr, self.tip[f])
                e_p = P[f] - p[f]
                e_r = 0.5 * sum(np.cross(Rc[f][:, i], R[f][:, i]) for i in range(3))
                wpf = w_pitch[f] if np.ndim(w_pitch) else w_pitch
                W = np.array([1.0, wpf, 1.0]) * 0.05         # rotation error in m-equivalent; world y = board pitch
                rows += [e_p, W * e_r]
                J += [jp[:, self.dadr], W[:, None] * jr[:, self.dadr]]
            e, J = np.concatenate(rows), np.vstack(J)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(len(e)), e)
            q = np.clip(q + dq, self.lo, self.hi)
            if np.max(np.abs(e[:3])) < 2e-5 and np.max(np.abs(e[6:9])) < 2e-5:
                break
        p, _ = self.fk(base_pos, base_quat, q)
        return q, float(np.max(np.linalg.norm(p - P, axis=1)))


def build(name, model="hw_ank3_b150_m2_10dof", stance_ref="c_T12HH", dxs=(-0.05, -0.025, 0.0, 0.025, 0.05),
          dys=(0.0, 0.025, 0.05), nph=128, shift=0.03, lift=0.025, rec_T=1.8):
    from .ref_gait import level_pitch, load
    ik = _IK(model)
    q_st = np.asarray(load(stance_ref)["q"][0, 0], float)         # the walker's zero-speed pose
    a = level_pitch(ik.m, ik.joints, q_st)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), a)
    base0 = np.array([0.0, 0.0, 0.0])
    P0, R0 = ik.fk(base0, quat, q_st)
    Q = np.zeros((2, len(dxs), len(dys), nph, len(q_st)))
    TZ = np.zeros((2, len(dxs), len(dys), nph, 2))
    ST = np.zeros((2, len(dxs), len(dys), nph, 2))
    worst = 0.0
    for si, side in enumerate((+1, -1)):
        A, B = (0, 1) if side > 0 else (1, 0)
        for ix, dx in enumerate(dxs):
            for iy, dy in enumerate(dys):
                q = q_st.copy()
                for k in range(nph):
                    u = k / nph
                    pel, fA, fB, (dnA, dnB) = plan(u, side, dx, dy, shift, lift)
                    P = P0.copy()
                    P[A] += fA; P[B] += fB
                    q, err = ik.solve(base0 + np.r_[pel, 0.0], quat, q, P, R0)
                    worst = max(worst, err)
                    Q[si, ix, iy, k] = q
                    TZ[si, ix, iy, k, A], TZ[si, ix, iy, k, B] = fA[2], fB[2]
                    ST[si, ix, iy, k, A], ST[si, ix, iy, k, B] = float(dnA), float(dnB)
                Q[si, ix, iy, -1] = Q[si, ix, iy, 0]                 # wrap exactly to the stance
    out = REF_DIR / f"{name}.npz"
    np.savez(out, q=Q, tip_z=TZ, stance=ST, dx=np.array(dxs), dy=np.array(dys), rec_T=rec_T, shift=shift, lift=lift,
             stance_ref=stance_ref, model=model)
    jump = np.degrees(np.max(np.abs(np.diff(Q, axis=3)), axis=(0, 1, 2, 3)))
    print(f"{out}: {Q.shape}, worst foot position error {worst * 1000:.2f} mm, largest per-bin joint step (deg): "
          + " ".join(f"{j.split('_', 1)[1][:6]} {v:.1f}" for j, v in zip(ik.joints, jump)))
    rng = np.degrees(Q.max(3) - Q.min(3)).max(axis=(0, 1, 2))
    print("joint range over a step (deg, worst target): " + " ".join(f"{v:.0f}" for v in rng))
    return out


def load_table(name, xp=np):
    z = np.load(REF_DIR / f"{name}.npz")
    return ({k: xp.asarray(z[k]) for k in ("q", "tip_z", "stance", "dx", "dy")} | {"rec_T": float(z["rec_T"])}
            | {"gate_u": float(z["gate_u"]) if "gate_u" in z else -1.0})


# ---------------------------------------------------------------- round 15: decisive step, hold, return
# Segment durations (s): weight shift, A swings out by D, pelvis to the new support's middle, hold (the gate sits in its
# middle: the clock stops there until the body is quiet), pelvis over A, B closes fore-aft (by dx), pelvis over B,
# A retracts sideways (by dy), pelvis centres. Fore-aft the rear / front foot closes up (as people do: the stance ends
# displaced by dx), sideways the stepped foot comes back (the stance ends where it started across). A return swing's
# lift scales with its distance (no swing for a zero distance).
SEG15 = (0.08, 0.36, 0.24, 0.16, 0.20, 0.36, 0.20, 0.36, 0.20)


def plan2(u, side, dx, dy, shift=0.012, lift=0.05, lift_back=0.06, shift_ret=0.03, drop=0.012, seg=SEG15):
    """Round 15 step. Returns pelvis (x, y, z offset), foot A and B displacements (x, y, z), stance flags (A, B)."""
    D = np.array([dx, side * dy])
    Dx, Dy = np.array([dx, 0.0]), np.array([0.0, side * dy])
    yA = np.array([0.0, side])
    b = np.cumsum((0.0,) + tuple(seg)) / sum(seg)
    seg_t = lambda i: (u - b[i]) / (b[i + 1] - b[i])  # noqa: E731
    mag = min(1.0, float(np.hypot(dx, dy)) / 0.09)
    zd = -drop * mag
    lf = lift if dx >= 0 else lift + (lift_back - lift) * min(1.0, -dx / 0.08)
    lB = lift * min(1.0, abs(dx) / 0.03)
    lA = lift * min(1.0, dy / 0.03)
    #     0      shift        out        mid       hold      over A               close B     over B                retract   centre
    # both boards planted: the pelvis stays at the feet's fore-aft midpoint (with a level pelvis and no ankle pitch both
    # boards are flat only there); it moves fore-aft only while one foot swings
    mx = np.array([0.5 * dx, 0.0])
    P = [np.zeros(2), -shift * yA, mx + 0.35 * Dy, 0.5 * D, 0.5 * D, mx + Dy + shift_ret * yA, Dx + Dy + shift_ret * yA,
         Dx - shift_ret * yA, Dx - shift_ret * yA, Dx]
    Z = [0.0, 0.0, zd, zd, zd, zd, zd, zd, zd, 0.0]
    i = int(np.clip(np.searchsorted(b, u, side="right") - 1, 0, len(seg) - 1))
    t = _ss(seg_t(i))
    pel = P[i] + (P[i + 1] - P[i]) * t
    pz = Z[i] + (Z[i + 1] - Z[i]) * t

    def swing(k, frm, to, h):
        tt = float(np.clip(seg_t(k), 0.0, 1.0))
        xy = frm + (to - frm) * _ss(np.clip((tt - 0.1) / 0.8, 0.0, 1.0))      # lift first, land straight down
        return np.r_[xy, h * math.sin(math.pi * tt)], not (0.0 < tt < 1.0) or h <= 0.0
    if u < b[3]:
        fA, dnA = swing(1, np.zeros(2), D, lf)
    else:
        fA, dnA = swing(7, D, Dx, lA)
    fB, dnB = swing(5, np.zeros(2), Dx, lB)
    return np.r_[pel, pz], fA, fB, (dnA, dnB)


def build2(name, model="hw_ank3_b150_m2_10dof", stance_ref="c_T12HH", dxs=(-0.08, -0.04, 0.0, 0.04, 0.08),
           dys=(0.0, 0.04, 0.08), nph=192, w_pitch=(1.0, 1.5), **kw):
    """w_pitch: board pitch weight in the IK for a planted / a swinging foot (no ankle pitch: a lifting foot tilts)."""
    from .ref_gait import level_pitch, load
    ik = _IK(model)
    q_st = np.asarray(load(stance_ref)["q"][0, 0], float)
    a = level_pitch(ik.m, ik.joints, q_st)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 1.0, 0.0]), a)
    base0 = np.zeros(3)
    P0, R0 = ik.fk(base0, quat, q_st)
    seg = kw.get("seg", SEG15)
    rec_T = float(sum(seg))
    b = np.cumsum((0.0,) + tuple(seg)) / sum(seg)
    gate_u = 0.5 * (b[3] + b[4])
    sw_pitch = []
    Q = np.zeros((2, len(dxs), len(dys), nph, len(q_st)))
    TZ = np.zeros((2, len(dxs), len(dys), nph, 2))
    ST = np.ones((2, len(dxs), len(dys), nph, 2))
    worst, worst_pitch = 0.0, 0.0
    for si, side in enumerate((+1, -1)):
        A, B = (0, 1) if side > 0 else (1, 0)
        for ix, dx in enumerate(dxs):
            for iy, dy in enumerate(dys):
                q = q_st.copy()
                for k in range(nph):
                    u = k / (nph - 1)
                    pel, fA, fB, (dnA, dnB) = plan2(u, side, dx, dy, **kw)
                    P = P0.copy()
                    P[A] += fA; P[B] += fB
                    dn_ = np.zeros(2, bool); dn_[A], dn_[B] = dnA, dnB
                    hz = np.zeros(2); hz[A], hz[B] = fA[2], fB[2]          # flat near the floor, freer up high
                    wp_ = w_pitch[0] - (w_pitch[0] - w_pitch[1]) * np.clip(hz / 0.02, 0.0, 1.0)
                    q, err = ik.solve(base0 + pel, quat, q, P, R0, w_pitch=wp_, iters=80)
                    _, Rc = ik.fk(base0 + pel, quat, q)
                    pit = np.degrees(np.arcsin(np.clip(Rc[:, 2, 0], -1, 1)))
                    down = dn_
                    worst_pitch = max(worst_pitch, float(np.max(np.abs(pit[down]), initial=0.0)))
                    sw_pitch.append(float(np.max(np.abs(pit[~down]), initial=0.0)))
                    worst = max(worst, err)
                    Q[si, ix, iy, k] = q
                    TZ[si, ix, iy, k, A], TZ[si, ix, iy, k, B] = fA[2], fB[2]
                    ST[si, ix, iy, k, A], ST[si, ix, iy, k, B] = float(dnA), float(dnB)
                Q[si, ix, iy, -1] = Q[si, ix, iy, 0]
    out = REF_DIR / f"{name}.npz"
    np.savez(out, q=Q, tip_z=TZ, stance=ST, dx=np.array(dxs), dy=np.array(dys), rec_T=rec_T, gate_u=gate_u,
             stance_ref=stance_ref, model=model, kind="v2")
    jump = np.degrees(np.max(np.abs(np.diff(Q, axis=3)), axis=(0, 1, 2, 3)))
    print(f"{out}: {Q.shape}, rec_T {rec_T:.2f} s, gate at u {gate_u:.3f}; worst foot position error {worst * 1000:.1f} mm,"
          f" board pitch: planted worst {worst_pitch:.1f} deg, swinging worst {max(sw_pitch):.1f}; largest per-bin joint step (deg): "
          + " ".join(f"{j.split('_', 1)[1][:6]} {v:.1f}" for j, v in zip(ik.joints, jump)))
    rng = np.degrees(Q.max(3) - Q.min(3)).max(axis=(0, 1, 2))
    print("joint range over a step (deg, worst target): " + " ".join(f"{v:.0f}" for v in rng))
    return out


def lookup(t, cmd, u, xp=np):
    """cmd = (dx, dy outward, side +1 L / -1 R); u in [0, 1]. -> q_ref, tip_z_ref, stance_ref (bool)."""
    si = xp.where(cmd[2] < 0, 1, 0)
    dxs, dys = t["dx"], t["dy"]
    nx, ny, nph = dxs.shape[0], dys.shape[0], t["q"].shape[3]
    fx = xp.clip((cmd[0] - dxs[0]) / (dxs[-1] - dxs[0]) * (nx - 1), 0.0, nx - 1.0)
    fy = xp.clip((cmd[1] - dys[0]) / (dys[-1] - dys[0]) * (ny - 1), 0.0, ny - 1.0)
    ix = xp.clip(xp.floor(fx).astype(int), 0, nx - 2); wx = fx - ix
    iy = xp.clip(xp.floor(fy).astype(int), 0, ny - 2); wy = fy - iy
    fp = xp.clip(u, 0.0, 1.0) * (nph - 1)
    j0 = xp.clip(xp.floor(fp).astype(int), 0, nph - 2); wp = fp - j0

    def tri(A):
        A = A[si]
        def at(jx, jy):
            return A[jx, jy, j0] * (1 - wp) + A[jx, jy, j0 + 1] * wp
        a = at(ix, iy) * (1 - wy) + at(ix, iy + 1) * wy
        b = at(ix + 1, iy) * (1 - wy) + at(ix + 1, iy + 1) * wy
        return a * (1 - wx) + b * wx
    return tri(t["q"]), tri(t["tip_z"]), tri(t["stance"]) > 0.5


def capture(ang, w, h=0.34, gain=1.2, th=0.035, dx_max=0.05, dy_max=0.05, xp=np):
    """Capture point from the IMU (round 14): ang = (roll, pitch) relative to the standing attitude, w = their rates
    (low-passed). The centre of mass is ~h x angle off the stance, moving at h x rate: cp = h (angle + rate / omega).
    In the pelvis frame: forward x = h (pitch + ...), left y = -h (roll + ...). Returns (trigger, cmd (dx, dy, side))."""
    om = math.sqrt(G / h)
    cx = h * (ang[1] + w[1] / om)
    cy = -h * (ang[0] + w[0] / om)
    trig = xp.sqrt(cx * cx + cy * cy) > th
    side = xp.where(cy >= 0, 1.0, -1.0)
    cmd = xp.stack([xp.clip(gain * cx, -dx_max, dx_max), xp.clip(gain * xp.abs(cy), 0.0, dy_max), side])
    return trig, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "build2"])
    ap.add_argument("--name", default="rs1")
    ap.add_argument("--shift", type=float, default=0.03)
    ap.add_argument("--lift", type=float, default=0.025)
    ap.add_argument("--rec-T", type=float, default=1.8)
    a = ap.parse_args()
    if a.cmd == "build2":
        build2(a.name)
    else:
        build(a.name, shift=a.shift, lift=a.lift, rec_T=a.rec_T)


if __name__ == "__main__":
    main()
