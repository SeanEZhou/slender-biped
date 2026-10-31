"""CPU MuJoCo evaluation of trained policies (independent of the GPU training stack).

* Full contact set (models/<name>.xml: every body geom collides with the floor and the
  other leg), default solver iterations -- a sim-to-sim check against the lean MJX model.
* Servo model, observation layout, latency and randomization ranges mirror
  slender_biped/mjx_env.py; the policy MLP is re-implemented in NumPy from the Brax params.
* Deterministic seeds; every episode is recorded (no cherry-picking).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.special import expit

from .config import MODEL_DIR, ROOT, load_actuators


# ---------------------------------------------------------------- policy
class NumpyPolicy:
    """Deterministic Brax PPO policy: normalize -> MLP(swish) -> tanh(mean)."""

    def __init__(self, params_path, obs_key="state"):
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        norm, pol = params[0], params[1]
        mean, std = norm.mean, norm.std
        if isinstance(mean, dict):
            mean, std = mean[obs_key], std[obs_key]
        self.mean, self.std = np.asarray(mean), np.asarray(std)
        layers = pol["params"]
        self.W = [np.asarray(layers[k]["kernel"]) for k in sorted(layers, key=lambda s: int(s.split("_")[-1]))]
        self.b = [np.asarray(layers[k]["bias"]) for k in sorted(layers, key=lambda s: int(s.split("_")[-1]))]

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = x * expit(x)  # swish
        mu = x[: x.shape[-1] // 2]
        return np.tanh(mu)


# ---------------------------------------------------------------- env (numpy mirror)
def quat_rot_t(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


@dataclass
class EvalConfig:
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    kp: float = 8.0
    kd: float = 0.05
    slew: float = 8.0
    latency_sub: int = 2
    deadzone: float = 0.0015
    action_scale: float = 0.35
    strength_scale: float = 1.0
    gait_hz: float = 1.6
    obs_noise: float = 1.0
    dr: bool = False
    push_vel: float = 0.0
    push_interval_s: tuple = (2.0, 4.0)
    reset_joint_noise: float = 0.08
    reset_tilt: float = 0.04
    max_tilt: float = 0.7
    min_height_frac: float = 0.6
    friction: float | None = None      # override tip friction
    mass_scale: float | None = None    # uniform mass scale override
    slope_deg: float = 0.0             # floor pitch about Y (positive = uphill when walking +x)
    ref_table: str = ""                # reference gait (round 4): residual action base, see mjx_env
    ref_residual: bool = False
    tip_condim: int = 3                # 4 = pad torsional friction on (see mjx_env)
    obs_vel: str = ""                  # round-5 extra inputs and estimator noise, see mjx_env.EnvConfig
    obs_accel: bool = False
    obs_hist: int = 0
    odo_noise: float = 0.002
    odo_bias: float = 0.01
    odo_scale: float = 0.1
    odo_alpha: float = 0.15
    odo_contact: str = "sched"
    line_frame: str = "heading"        # see mjx_env.EnvConfig.line_frame
    obs_line: bool = False             # integrated odometry offset from the path (start line, shifted by the vy command)
    odo_flip: float = 0.03
    odo_src: str = "legs"              # see mjx_env.EnvConfig.odo_src
    flow_noise: float = 0.02
    acc_noise: float = 0.01
    acc_bias: float = 0.03

    @classmethod
    def from_train(cls, env_cfg: dict, **over):
        keep = {k: env_cfg[k] for k in cls.__dataclass_fields__ if k in env_cfg}
        keep.update(over)
        for k in ("push_interval_s",):
            if k in keep:
                keep[k] = tuple(keep[k])
        return cls(**keep)


class CpuBiped:
    def __init__(self, model: str, cfg: EvalConfig, seed=0):
        self.cfg = cfg
        self.meta = json.loads((MODEL_DIR / f"{model}.json").read_text())
        m = mujoco.MjModel.from_xml_path(str(MODEL_DIR / f"{model}.xml"))
        m.opt.timestep = cfg.sim_dt
        if cfg.slope_deg:
            fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            a = np.radians(cfg.slope_deg)
            m.geom_quat[fid] = [np.cos(a / 2), 0, -np.sin(a / 2), 0]
        self.m0_mass = m.body_mass.copy()
        self.m0_inertia = m.body_inertia.copy()
        self.m0_arm = m.dof_armature.copy()
        self.m0_fl = m.dof_frictionloss.copy()
        self.m0_ipos = m.body_ipos.copy()
        self.m = m
        self.d = mujoco.MjData(m)
        self.joints = self.meta["joints"]
        self.nj = len(self.joints)
        act = load_actuators()
        specs = [act["catalog"][act["assignments"][self.meta["cfg"]["actuator_set"]][j.split("_", 1)[1]]] for j in self.joints]
        self.ts0 = np.array([s["stall_torque"] for s in specs]) * cfg.strength_scale
        self.w00 = np.array([s["noload_speed"] for s in specs])
        self.rated = np.array([s.get("rated_torque") or np.nan for s in specs])
        self.q_nom = np.array(self.meta["nominal_q"])
        self.q_lo = np.array([m.jnt_range[m.actuator_trnid[i, 0]][0] for i in range(self.nj)])
        self.q_hi = np.array([m.jnt_range[m.actuator_trnid[i, 0]][1] for i in range(self.nj)])
        self.base_z = self.meta["stand_base_z"]
        g = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.tip_gid = np.array([g("L_tip"), g("R_tip")])
        size = m.geom_size[self.tip_gid].copy()
        sph = m.geom_type[self.tip_gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        size[sph] = size[sph][:, :1].repeat(3, 1)
        self.tip_semi = size
        self.tip_box = m.geom_type[self.tip_gid] == mujoco.mjtGeom.mjGEOM_BOX  # flat soles (control frame)
        self.tip_pairs = [i for i in range(m.npair)
                          if {mujoco.mj_id2name(m, 5, m.pair_geom1[i]), mujoco.mj_id2name(m, 5, m.pair_geom2[i])}
                          in ({"floor", "L_tip"}, {"floor", "R_tip"})]
        if cfg.tip_condim != 3:
            for i in self.tip_pairs:
                m.pair_dim[i] = cfg.tip_condim
        self.pf0 = m.pair_friction.copy()
        self.floor_gid = g("floor")
        self.foot_gids = set(self.tip_gid.tolist())
        self.knee_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shin") for s in "LR"]
        self.knee_r = float(m.geom_size[g("L_knee_col")][0])
        self.rng = np.random.default_rng(seed)
        # round-5 estimators draw from their own stream, so older evaluations keep their exact random sequence
        self.est_rng = np.random.default_rng([seed, 5])
        self.kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.ref = None
        if cfg.ref_table:
            from .ref_gait import load
            self.ref = load(cfg.ref_table)

    def tip_low(self):
        d = self.d
        c = d.geom_xpos[self.tip_gid]
        R = d.geom_xmat[self.tip_gid].reshape(2, 3, 3)
        h = np.where(self.tip_box, np.abs(self.tip_semi * R[:, 2, :]).sum(-1),
                     np.linalg.norm(self.tip_semi * R[:, 2, :], axis=-1))
        return c[:, 2] - h, c.copy()

    def reset(self, cmd=(0.0, 0.0, 0.0)):
        c, m, d, rng = self.cfg, self.m, self.d, self.rng
        # model randomization (same ranges as mjx_env.randomization_fn)
        m.body_mass[:] = self.m0_mass
        m.body_inertia[:] = self.m0_inertia
        m.body_ipos[:] = self.m0_ipos
        m.dof_armature[:] = self.m0_arm
        m.dof_frictionloss[:] = self.m0_fl
        m.pair_friction[:] = self.pf0
        self.servo = dict(strength=np.ones(self.nj), speed=np.ones(self.nj), kp=np.ones(self.nj),
                          latency=c.latency_sub, deadzone=np.full(self.nj, c.deadzone))
        if c.dr:
            fr = rng.uniform(0.5, 1.2)
            for i in self.tip_pairs:
                m.pair_friction[i, 0:2] = fr
            ms = rng.uniform(0.85, 1.15, m.nbody)
            ms[0] = 1
            m.body_mass[:] *= ms
            m.body_inertia[:] *= ms[:, None]
            m.body_ipos[1] += rng.uniform(-0.005, 0.005, 3)
            m.dof_armature[6:] *= rng.uniform(0.6, 1.6, m.nv - 6)
            m.dof_frictionloss[6:] *= rng.uniform(0.5, 2.0, m.nv - 6)
            self.servo = dict(strength=rng.uniform(0.8, 1.05, self.nj), speed=rng.uniform(0.8, 1.05, self.nj),
                              kp=rng.uniform(0.75, 1.25, self.nj), latency=int(rng.integers(0, 7)),
                              deadzone=rng.uniform(0.0, 0.015, self.nj))
        if c.friction is not None:
            for i in self.tip_pairs:
                m.pair_friction[i, 0:2] = c.friction
        if c.mass_scale is not None:
            m.body_mass[:] *= c.mass_scale
            m.body_inertia[:] *= c.mass_scale
        mujoco.mj_resetDataKeyframe(m, d, self.kid)
        qj = np.clip(self.q_nom + c.reset_joint_noise * rng.normal(size=self.nj), self.q_lo, self.q_hi)
        tilt = c.reset_tilt * rng.normal(size=2)
        q_rp = np.array([1.0, tilt[0] / 2, tilt[1] / 2, 0.0])
        d.qpos[3:7] = q_rp / np.linalg.norm(q_rp)
        d.qpos[7:] = qj
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        low, _ = self.tip_low()
        d.qpos[2] += -low.min() + 0.0005
        mujoco.mj_forward(m, d)
        self.prev_cmd = qj.copy()
        self.q_tgt = qj.copy()
        self.last_act = np.zeros(self.nj)
        self.phase = 0.0
        self.cmd = np.array(cmd, float)
        low, centre = self.tip_low()
        self.contact_prev = low < 0.0015
        self.push_timer = self._steps(c.push_interval_s)
        self.tau = np.zeros(self.nj)
        self.y_est = 0.0
        self.path_heading = yaw_of(d.qpos[3:7])
        self.odo_vw = np.zeros(2)
        self.odo_v, self.odo_p, self.odo_con_prev = np.zeros(2), centre - d.qpos[0:3], low < 0.0015
        er = self.est_rng
        self.odo_bias = c.odo_bias * er.normal(size=2)
        self.odo_scale_k = 1.0 + er.uniform(-c.odo_scale, c.odo_scale)
        self.acc_bias_v = er.uniform(-c.acc_bias, c.acc_bias, 3)
        self.acc, self.v_prev = np.array([0.0, 0.0, 1.0]), np.zeros(3)
        self.hist = np.zeros((c.obs_hist, 3 + 3 + 3 * self.nj))
        if not c.obs_hist:
            return self.obs()
        self.obs()
        self.hist[:] = self._frame
        return self.obs()

    def _steps(self, lohi):
        return int(self.rng.integers(int(lohi[0] / self.cfg.ctrl_dt), int(lohi[1] / self.cfg.ctrl_dt) + 1))

    def obs(self):
        c, d = self.cfg, self.d
        quat = d.qpos[3:7]
        g = quat_rot_t(quat, np.array([0, 0, -1.0]))
        w = d.qvel[3:6].copy()
        q = d.qpos[7:] - self.q_nom
        qd = d.qvel[6:].copy()
        n = c.obs_noise
        r = self.rng
        g = g + n * 0.03 * r.normal(size=3)
        w = w + n * 0.15 * r.normal(size=3)
        q = q + n * 0.005 * r.normal(size=self.nj)
        qd = qd + n * 0.3 * r.normal(size=self.nj)
        clock = np.array([np.sin(self.phase), np.cos(self.phase)])
        self._frame = np.concatenate([g, w, q, 0.1 * qd, self.last_act])
        extra = []
        if c.obs_vel == "oracle":
            yaw = yaw_of(quat)
            vw = d.qvel[0:2]
            extra.append([np.cos(yaw) * vw[0] + np.sin(yaw) * vw[1], -np.sin(yaw) * vw[0] + np.cos(yaw) * vw[1]])
        elif c.obs_vel == "odo":
            extra.append(self.odo_v)
        if c.obs_accel:
            extra.append(self.acc)
        if c.obs_line:
            extra.append([np.clip(self.y_est, -0.5, 0.5)])
        if c.obs_hist:
            extra.append(self.hist.ravel())
        return np.concatenate([np.asarray(e, float) for e in extra] + [self._frame, self.cmd, clock])

    def _update_estimators(self, phase_next, cmd_vx):
        """Stance-leg odometry and accelerometer, same equations and noise as mjx_env.step."""
        c, d, r = self.cfg, self.d, self.est_rng
        quat, v_world = d.qpos[3:7], d.qvel[0:3].copy()
        yaw = yaw_of(quat)
        cy, sy = np.cos(yaw), np.sin(yaw)
        to_head = lambda v: np.array([cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]])  # noqa: E731
        low, centre = self.tip_low()
        p_meas = centre - d.qpos[0:3] + c.odo_noise * r.normal(size=(2, 3))
        con = low < 0.0015
        if c.odo_contact == "sensed" or self.ref is None:
            st = ((con & self.odo_con_prev) ^ (r.uniform(size=2) < c.odo_flip)).astype(float)
        else:
            from .ref_gait import lookup
            st = lookup(self.ref, max(cmd_vx, 0.0), phase_next)[2].astype(float)
        self.odo_con_prev = con
        if st.sum() == 0:
            st = np.ones(2)
        v_leg = -(st[:, None] * (p_meas - self.odo_p)[:, :2]).sum(0) / (st.sum() * c.ctrl_dt)
        b = self.odo_bias
        if c.odo_src == "flow":
            self.odo_vw = (self.odo_scale_k * v_world[:2] + np.array([cy * b[0] - sy * b[1], sy * b[0] + cy * b[1]])
                           + c.flow_noise * r.normal(size=2))
            self.odo_v = self.odo_v + c.odo_alpha * (to_head(self.odo_vw) - self.odo_v)
        else:
            self.odo_v = self.odo_v + c.odo_alpha * (self.odo_scale_k * to_head(v_leg) + self.odo_bias - self.odo_v)
            self.odo_vw = self.odo_scale_k * v_leg + np.array([cy * b[0] - sy * b[1], sy * b[0] + cy * b[1]])
        self.odo_p = p_meas
        self.acc = (quat_rot_t(quat, (v_world - self.v_prev) / c.ctrl_dt + np.array([0.0, 0.0, 9.81])) / 9.81
                    + self.acc_bias_v + c.acc_noise * r.normal(size=3))
        self.v_prev = v_world

    def step(self, action):
        c, m, d, s_ = self.cfg, self.m, self.d, self.servo
        action = np.clip(action, -1, 1)
        q_base = self.q_nom
        if self.ref is not None and c.ref_residual:
            from .ref_gait import lookup
            phase_next = (self.phase + 2 * np.pi * c.gait_hz * c.ctrl_dt) % (2 * np.pi)
            q_base = lookup(self.ref, max(self.cmd[0], 0.0), phase_next)[0]
        q_cmd = np.clip(q_base + c.action_scale * action, self.q_lo, self.q_hi)
        ts = self.ts0 * s_["strength"]
        w0 = self.w00 * s_["speed"]
        kp = c.kp * s_["kp"]
        n_sub = int(round(c.ctrl_dt / c.sim_dt))
        log = dict(tau=[], qd=[], u_req=[])
        for i in range(n_sub):
            cmd = self.prev_cmd if i < s_["latency"] else q_cmd
            self.q_tgt = self.q_tgt + np.clip(cmd - self.q_tgt, -c.slew * c.sim_dt, c.slew * c.sim_dt)
            q, qd = d.qpos[7:], d.qvel[6:]
            e = self.q_tgt - q
            e = np.sign(e) * np.maximum(np.abs(e) - s_["deadzone"], 0)
            u_req = kp * e - c.kd * qd
            u = np.clip(u_req, -1, 1)
            tau = np.clip(ts * (u - qd / w0), -ts, ts)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            log["tau"].append(tau.copy())
            log["qd"].append(d.qvel[6:].copy())
            log["u_req"].append(u_req.copy())
        self.tau = tau
        self.prev_cmd = q_cmd
        self.last_act = action
        self.phase = (self.phase + 2 * np.pi * c.gait_hz * c.ctrl_dt) % (2 * np.pi)
        if c.push_vel > 0:
            self.push_timer -= 1
            if self.push_timer <= 0:
                d.qvel[0:2] += c.push_vel * self.rng.normal(size=2)
                self.push_timer = self._steps(c.push_interval_s)
        vy_cmd = self.cmd[1]
        self._update_estimators(self.phase, self.cmd[0])
        if c.line_frame == "path":
            h = self.path_heading
            self.y_est += (-np.sin(h) * self.odo_vw[0] + np.cos(h) * self.odo_vw[1] - vy_cmd) * c.ctrl_dt
        else:
            self.y_est += (self.odo_v[1] - vy_cmd) * c.ctrl_dt
        obs = self.obs()
        if c.obs_hist:
            self.hist = np.concatenate([self.hist[1:], self._frame[None]], 0)
        return obs, {k: np.array(v) for k, v in log.items()}

    def fallen(self):
        d = self.d
        g = quat_rot_t(d.qpos[3:7], np.array([0, 0, -1.0]))
        tilt2 = g[0] ** 2 + g[1] ** 2
        knee_z = d.xpos[self.knee_bid][:, 2]
        # any non-tip geom touching the floor also counts as a fall in the full-contact model
        body_floor = False
        for i in range(d.ncon):
            con = d.contact[i]
            pair = {con.geom1, con.geom2}
            if self.floor_gid in pair and not (pair & self.foot_gids):
                body_floor = True
                break
        return bool(tilt2 > np.sin(self.cfg.max_tilt) ** 2 or d.qpos[2] < self.cfg.min_height_frac * self.base_z
                    or np.any(knee_z < self.knee_r) or g[2] > 0 or body_floor)


# ---------------------------------------------------------------- episodes
def yaw_of(quat):
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run_episode(env: CpuBiped, policy, seconds, cmd_schedule=None, record=False, renderer=None, cam="track",
                heading_hold=False, heading_gain=1.0):
    """cmd_schedule: list of (t_start, (vx, vy, wz)); default = all zeros.

    heading_hold: outer loop (operator / joystick equivalent) that overrides the yaw-rate command with
    clip(gain * (yaw0 - yaw), +-0.3 rad/s) using the pelvis yaw (IMU estimate on hardware).
    """
    sched = cmd_schedule or [(0.0, (0.0, 0.0, 0.0))]
    obs = env.reset(sched[0][1])
    c = env.cfg
    n = int(seconds / c.ctrl_dt)
    start_xy = env.d.qpos[0:2].copy()
    yaw0 = yaw_of(env.d.qpos[3:7])
    taus, qds, ureqs, vx, vy, wz, cmds, z, tilt, contacts = [], [], [], [], [], [], [], [], [], []
    touchdowns, bounces, frames = 0, 0, []
    # stillness (after a 2 s settle): any lift-off of a whole tip/sole, horizontal tip travel, base speed
    settle = 2.0
    liftoffs, tip_travel, vbase2, n_settled = 0, 0.0, 0.0, 0
    tip_prev = None
    air = np.zeros(2)
    fall_t = None
    si = 0
    for k in range(n):
        t = k * c.ctrl_dt
        while si + 1 < len(sched) and t >= sched[si + 1][0]:
            si += 1
            env.cmd = np.array(sched[si][1], float)
            obs[-5:-2] = env.cmd
        if heading_hold:
            err = (yaw0 - yaw_of(env.d.qpos[3:7]) + np.pi) % (2 * np.pi) - np.pi
            env.cmd[2] = float(np.clip(heading_gain * err, -0.3, 0.3))
            obs[-5:-2] = env.cmd
        a = policy(obs)
        obs, log = env.step(a)
        taus.append(log["tau"])
        qds.append(log["qd"])
        ureqs.append(log["u_req"])
        d = env.d
        vb = quat_rot_t(d.qpos[3:7], d.qvel[0:3])
        vx.append(vb[0]); vy.append(vb[1]); wz.append(d.qvel[5]); cmds.append(env.cmd.copy())
        z.append(d.qpos[2])
        g = quat_rot_t(d.qpos[3:7], np.array([0, 0, -1.0]))
        tilt.append(float(np.degrees(np.arcsin(min(1.0, np.hypot(g[0], g[1]))))))
        low, _ = env.tip_low()
        con = low < 0.0015
        td = con & ~env.contact_prev
        _, tip_c = env.tip_low()
        if t >= settle:
            liftoffs += int(np.sum(~con & env.contact_prev))
            if tip_prev is not None:
                tip_travel += float(np.sum(np.linalg.norm(tip_c[:, :2] - tip_prev[:, :2], axis=1)))
            vbase2 += float(d.qvel[0] ** 2 + d.qvel[1] ** 2)
            n_settled += 1
        tip_prev = tip_c
        # a step = touchdown after >= 60 ms of air time; shorter airborne blips are bounces/chatter
        touchdowns += int(np.sum(td & (air >= 0.06 - 1e-9)))
        bounces += int(np.sum(td & (air < 0.06 - 1e-9)))
        air = np.where(con, 0.0, air + c.ctrl_dt)
        env.contact_prev = con
        contacts.append(con)
        if renderer is not None and k % 2 == 0:
            frames.append(renderer.render(d, cam))
        if env.fallen():
            fall_t = t + c.ctrl_dt
            break
    taus = np.concatenate(taus) if taus else np.zeros((0, env.nj))
    qds = np.concatenate(qds) if qds else np.zeros((0, env.nj))
    ureqs = np.concatenate(ureqs) if ureqs else np.zeros((0, env.nj))
    dur = len(vx) * c.ctrl_dt
    disp = env.d.qpos[0:2] - start_xy
    res = dict(
        survived_s=dur if fall_t is None else fall_t, fell=fall_t is not None,
        drift_m=float(np.linalg.norm(disp)),
        forward_m=float(disp @ np.array([np.cos(yaw0), np.sin(yaw0)])),   # along the initial heading
        heading_change_deg=float(np.degrees((yaw_of(env.d.qpos[3:7]) - yaw0 + np.pi) % (2 * np.pi) - np.pi)),
        touchdowns=touchdowns, bounces=bounces, steps_per_s=touchdowns / max(dur, 1e-6),
        both_contact_frac=float(np.mean([np.all(cc) for cc in contacts])) if contacts else 0.0,
        peak_tau_frac=(np.abs(taus) / env.ts0).max(0).tolist() if len(taus) else None,
        rms_tau=np.sqrt((taus ** 2).mean(0)).tolist() if len(taus) else None,           # N*m
        rms_tau_frac=(np.sqrt((taus ** 2).mean(0)) / env.ts0).tolist() if len(taus) else None,  # of stall
        rms_over_rated=(np.sqrt((taus ** 2).mean(0)) / env.rated).tolist() if len(taus) else None,
        sat_events=int((np.abs(ureqs) > 1).sum()), sat_frac=float((np.abs(ureqs) > 1).mean()) if len(ureqs) else 0.0,
        overspeed_events=int((np.abs(qds) > env.w00).sum()),
        speed_util=(np.abs(qds) / env.w00).max(0).tolist() if len(qds) else None,
        mean_vx=float(np.mean(vx)) if vx else 0.0,
        liftoffs_after_settle=liftoffs, tip_travel_m=tip_travel,
        base_speed_rms=float(np.sqrt(vbase2 / max(n_settled, 1))),
        step_length_m=float(disp @ np.array([np.cos(yaw0), np.sin(yaw0)])) / max(touchdowns, 1),
        energy_proxy=float(np.sum(taus ** 2) * c.sim_dt),
        z_std=float(np.std(z)) if z else 0.0, max_tilt_deg=float(max(tilt)) if tilt else 0.0,
    )
    if record:
        res["trace"] = dict(vx=vx, vy=vy, wz=wz, cmd=[x.tolist() for x in cmds], z=z, tilt=tilt,
                            tau=taus[::5].tolist(), qd=qds[::5].tolist())
    return res, frames


def load_run(run_dir):
    run_dir = ROOT / "runs" / run_dir if not str(run_dir).startswith("/") else run_dir
    cfg = json.loads((run_dir / "config.json").read_text())
    return cfg, run_dir
