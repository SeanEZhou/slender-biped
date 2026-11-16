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
                W = np.array([1.0, w_pitch, 1.0]) * 0.05     # rotation error in m-equivalent; world y = board pitch
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
    return {k: xp.asarray(z[k]) for k in ("q", "tip_z", "stance", "dx", "dy")} | {"rec_T": float(z["rec_T"])}


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
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--name", default="rs1")
    ap.add_argument("--shift", type=float, default=0.03)
    ap.add_argument("--lift", type=float, default=0.025)
    ap.add_argument("--rec-T", type=float, default=1.8)
    a = ap.parse_args()
    build(a.name, shift=a.shift, lift=a.lift, rec_T=a.rec_T)


if __name__ == "__main__":
    main()
