"""Quasi-static and inverse-dynamics joint-torque estimates for the free-base biped.

Contacts are point contacts at the tip `*_contact` sites (no moment transmission),
which is exact for foot A and the instantaneous rigid limit for rocker B.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares

G = 9.81


class Statics:
    def __init__(self, model: mujoco.MjModel, joints: list[str]):
        self.m = model
        self.m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        self.d = mujoco.MjData(self.m)
        self.joints = joints
        self.qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints])
        self.dadr = np.array([self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints])
        self.site = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_contact") for s in "LR"}
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def set(self, q: dict, qd: dict | None = None, qdd: dict | None = None):
        d = self.d
        d.qpos[:] = 0
        d.qpos[2] = 0.5
        d.qpos[3] = 1
        d.qvel[:] = 0
        d.qacc[:] = 0
        for i, j in enumerate(self.joints):
            d.qpos[self.qadr[i]] = q.get(j, 0.0)
            if qd:
                d.qvel[self.dadr[i]] = qd.get(j, 0.0)
        mujoco.mj_forward(self.m, d)  # overwrites qacc -> prescribe it afterwards
        d.qacc[:] = 0
        if qdd:
            for i, j in enumerate(self.joints):
                d.qacc[self.dadr[i]] = qdd.get(j, 0.0)

    def com(self):
        return self.d.subtree_com[self.pelvis].copy()

    def contact_pos(self, s):
        return self.d.site_xpos[self.site[s]].copy()

    def jac(self, s):
        jp = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, None, self.site[s])
        return jp

    def torques(self, stance: str):
        """Joint torques for the current (static or prescribed-acceleration) state.

        stance: 'LR' double support, 'L'/'R' single support, '' = base fixed in space
        (base wrench absorbed by an imaginary fixture; used for swing-leg dynamics).
        Returns (tau[joints], contact forces, base-wrench residual norm).
        """
        mujoco.mj_inverse(self.m, self.d)
        qf = self.d.qfrc_inverse.copy()
        if not stance:
            return qf[self.dadr], {}, 0.0
        J = np.concatenate([self.jac(s) for s in stance], 0)  # (3k, nv)
        f, *_ = np.linalg.lstsq(J[:, :6].T, qf[:6], rcond=None)
        resid = float(np.linalg.norm(J[:, :6].T @ f - qf[:6]))
        tau = qf[self.dadr] - (J.T @ f)[self.dadr]
        forces = {s: f[3 * i:3 * i + 3] for i, s in enumerate(stance)}
        return tau, forces, resid


def stance_pose(knee, ankle_joint=True, hip_roll=0.0, ankle_max=np.radians(28)):
    """Symmetric crouch pose: hip = knee/2, ankle = knee/2 (level tip) clamped to the ankle range."""
    q = {}
    for s in "LR":
        q[f"{s}_hip_pitch"] = knee / 2
        q[f"{s}_knee"] = knee
        if ankle_joint:
            q[f"{s}_ankle"] = min(knee / 2, ankle_max)
        q[f"{s}_hip_roll"] = hip_roll
    return q


def balance_single(st: Statics, q: dict, stance: str, swing_lift_knee=0.5):
    """Shift the stance leg (hip roll/pitch + ankle compensation) so the CoM is over the stance tip.

    The pelvis is kept level; the swing leg is lifted by flexing hip and knee.
    """
    sw = "R" if stance == "L" else "L"
    q = dict(q)
    q[f"{sw}_knee"] = q[f"{sw}_knee"] + swing_lift_knee
    q[f"{sw}_hip_pitch"] = q[f"{sw}_hip_pitch"] + swing_lift_knee / 2
    has_ankle = f"{stance}_ankle" in st.joints

    def res(x):
        qq = dict(q)
        # stance-leg adduction (negative roll) moves the pelvis over the foot
        qq[f"{stance}_hip_roll"] = -x[0]
        qq[f"{sw}_hip_roll"] = x[0]  # keep swing leg roughly vertical
        qq[f"{stance}_hip_pitch"] = q[f"{stance}_hip_pitch"] + x[1]
        if has_ankle:
            qq[f"{stance}_ankle"] = q[f"{stance}_ankle"] - x[1]
        st.set(qq)
        return (st.com() - st.contact_pos(stance))[:2]

    sol = least_squares(res, [0.1, 0.0], bounds=([-0.3, -0.5], [0.4, 0.5]))
    res(sol.x)
    qq = dict(q)
    qq[f"{stance}_hip_roll"] = -sol.x[0]
    qq[f"{sw}_hip_roll"] = sol.x[0]
    qq[f"{stance}_hip_pitch"] = q[f"{stance}_hip_pitch"] + sol.x[1]
    if has_ankle:
        qq[f"{stance}_ankle"] = q[f"{stance}_ankle"] - sol.x[1]
    st.set(qq)
    return qq, float(np.linalg.norm(sol.fun))
