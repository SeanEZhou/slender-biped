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


def build(model: str, steps_per_s=3.0, duty=0.55, swing_h=0.025, width=0.081, sway=1.0, yaw_twist_deg=0.0,
          lift_shape=1.0, full_h_speed=0.05, vmax=0.3, nv=13, nphase=64):
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
             full_h_speed=full_h_speed, lift_shape=lift_shape, com_h=float(d.subtree_com[1][2]))
    vs = np.linspace(0, vmax, nv)
    phases = np.linspace(0, 2 * np.pi, nphase, endpoint=False)
    Q = np.zeros((nv, nphase, len(joints)))
    Z = np.zeros((nv, nphase, 2))
    S = np.zeros((nv, nphase, 2))
    err_max = 0.0
    jacp = np.zeros((3, m.nv))
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
    a = ap.parse_args()
    t = build(a.model, a.steps_per_s, a.duty, a.swing_h, a.width, a.sway, a.yaw_twist, a.lift_shape)
    REF_DIR.mkdir(exist_ok=True)
    np.savez(REF_DIR / f"{a.name}.npz", **t)
    meta = json.loads((MODEL_DIR / f"{a.model}.json").read_text())
    act = load_actuators()
    w0 = np.array([act["catalog"][act["assignments"][meta["cfg"]["actuator_set"]][j.split("_", 1)[1]]]["noload_speed"]
                   for j in meta["joints"]])
    dt = 1 / (t["gait_hz"] * t["q"].shape[1])
    qd = np.abs(np.diff(np.concatenate([t["q"], t["q"][:, :1]], 1), axis=1)) / dt
    print(f"{a.name}: {a.steps_per_s} steps/s, IK max error {t['ik_err']*1000:.2f} mm")
    for iv in (2, 6, 10, 12):
        v = t["v"][iv]
        print(f"  v={v:.3f}  step {v*a.duty*2/a.steps_per_s*100:4.1f} cm  peak |qd|/no-load "
              + " ".join(f"{x:.2f}" for x in (qd[iv].max(0) / w0)[:4])
              + f"  q range (deg) " + " ".join(f"{np.degrees(np.ptp(t['q'][iv][:, k])):.0f}" for k in range(4))
              + f"  swing tip z max {t['tip_z'][iv].max()*1000:.0f} mm")


if __name__ == "__main__":
    main()
