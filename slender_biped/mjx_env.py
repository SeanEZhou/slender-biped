"""GPU (MJX) training environment for the slender biped, Brax-compatible.

Control: 50 Hz joint-position targets -> servo model (slender_biped/actuator.py equations,
re-implemented in JAX) evaluated at every 2 ms physics substep:
    u_req = kp * deadzone(q_tgt - q) - kd * qd ;  u = clip(u_req, -1, 1)
    tau   = clip(tau_stall * s_str * (u - qd / (omega0 * s_spd)), +-tau_stall * s_str)
with command latency (whole substeps), target slew limiting and a deadzone standing in
for gear backlash. Saturation (|u_req| > 1) and over-speed are logged as metrics and
penalized, never hidden.

Observations
  state (policy, hardware-plausible): IMU gravity direction + gyro, joint pos/vel,
      previous action, velocity command, gait clock. Noise added.
  privileged_state (critic only): state + true base lin. velocity, pelvis height,
      tip contacts/positions, randomization parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

from .config import MODEL_DIR, load_actuators


@dataclass
class EnvConfig:
    model: str = "baseline_A_8dof"
    task: str = "stand"               # stand | walk
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    solver_iterations: int = 4
    action_scale: float = 0.35         # rad per unit action around the nominal pose
    # servo
    kp: float = 8.0
    kd: float = 0.05
    slew: float = 8.0
    latency_sub: int = 2               # nominal command latency in physics substeps (4 ms)
    deadzone: float = 0.0015
    strength_scale: float = 1.0        # global torque scale (actuator-strength ablation)
    # domain randomization
    dr: bool = False
    dr_strength: tuple = (0.8, 1.05)
    dr_speed: tuple = (0.8, 1.05)
    dr_kp: tuple = (0.75, 1.25)
    dr_latency_sub: tuple = (0, 6)     # 0-12 ms
    dr_deadzone: tuple = (0.0, 0.015)  # up to ~0.9 deg backlash-like play
    obs_noise: float = 1.0             # multiplier on nominal sensor noise (0 = ideal sensors)
    # resets / disturbances
    reset_joint_noise: float = 0.08
    reset_tilt: float = 0.04
    push_interval_s: tuple = (2.0, 4.0)
    push_vel: float = 0.0              # m/s std of random base velocity kicks (0 = off)
    # commands
    cmd_vx: tuple = (0.0, 0.0)
    cmd_vy: tuple = (0.0, 0.0)
    cmd_wz: tuple = (0.0, 0.0)
    cmd_zero_prob: float = 1.0
    cmd_resample_s: tuple = (3.0, 6.0)
    heading_cmd: bool = False          # yaw-rate command = clip(k * heading error) (legged-gym style)
    heading_gain: float = 1.0
    gait_hz: float = 1.6               # clock frequency (full gait cycle)
    phase_when_still: bool = False     # apply gait-phase shaping at zero command too (stepping in place)
    # termination
    max_tilt: float = 0.7
    min_height_frac: float = 0.6
    # reward weights
    rw: dict = field(default_factory=lambda: dict(
        track_lin=1.0, track_ang=0.5, upright=1.0, height=0.5, lin_vel_z=-1.0, ang_vel_xy=-0.05,
        torque=-0.02, saturation=-0.5, overspeed=-0.5, action_rate=-0.02, joint_vel=-0.0005,
        over_rated=0.0, heading=0.0, stop=0.0, pose=-0.3, slip=-4.0, feet_air_time=0.0, contact_phase=0.0, swing_height=0.0,
        both_feet=0.0, touchdown=0.0, liftoff=0.0, stride=0.0, alive=0.5, termination=-20.0))
    rated_frac: float = 0.33           # sustained-torque cap as a fraction of stall (vendor rated ~1/3)
    track_sigma: float = 0.01          # velocity-tracking kernel width (m/s)^2
    track_ang_sigma: float = 0.05      # yaw-rate tracking kernel width (rad/s)^2
    heading_sigma: float = 0.1         # heading-hold kernel width rad^2 (only with heading_cmd)
    stop_sigma: float = 0.002          # zero-command "stand still" velocity kernel (m/s)^2
    height_sigma: float = 0.03
    swing_height: float = 0.025
    air_time_target: float = 0.2
    stride_target: float = 0.06        # m, forward tip travel per step rewarded up to this (stride term)


def _quat_rot_t(q, v):
    """Rotate world vector v into the body frame of quaternion q (w, x, y, z)."""
    w, x, y, z = q
    R = jp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ v


class SlenderBipedEnv(Env):
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        xml = (MODEL_DIR / f"{cfg.model}_train.xml")
        if not xml.exists():
            xml = MODEL_DIR / f"{cfg.model}.xml"
        self.meta = json.loads((MODEL_DIR / f"{cfg.model}.json").read_text())
        m = mujoco.MjModel.from_xml_path(str(xml))
        m.opt.timestep = cfg.sim_dt
        m.opt.iterations = cfg.solver_iterations
        m.opt.ls_iterations = 2 * cfg.solver_iterations
        self.mj_model = m
        self.sys = mjx.put_model(m)
        self.n_sub = int(round(cfg.ctrl_dt / cfg.sim_dt))
        self.nj = m.nu
        self.joints = self.meta["joints"]
        act = load_actuators()
        specs = [act["catalog"][act["assignments"][self.meta["cfg"]["actuator_set"]][j.split("_", 1)[1]]]
                 for j in self.joints]
        self.ts = jp.array([s["stall_torque"] for s in specs]) * cfg.strength_scale
        self.w0 = jp.array([s["noload_speed"] for s in specs])
        self.q_nom = jp.array(self.meta["nominal_q"])
        lo = np.array([m.jnt_range[m.actuator_trnid[i, 0]][0] for i in range(self.nj)])
        hi = np.array([m.jnt_range[m.actuator_trnid[i, 0]][1] for i in range(self.nj)])
        self.q_lo, self.q_hi = jp.array(lo), jp.array(hi)
        self.base_z = float(self.meta["stand_base_z"])
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.tip_gid = np.array([gid("L_tip"), gid("R_tip")])
        size = m.geom_size[self.tip_gid].copy()
        sphere = m.geom_type[self.tip_gid] == mujoco.mjtGeom.mjGEOM_SPHERE
        size[sphere] = size[sphere][:, :1].repeat(3, 1)
        self.tip_semi = jp.array(size)                      # semi-axes for the support function
        self.tip_box = jp.array(m.geom_type[self.tip_gid] == mujoco.mjtGeom.mjGEOM_BOX)  # flat soles (control frame)
        self.knee_bid = np.array([bid("L_shin"), bid("R_shin")])
        self.pelvis_bid = bid("pelvis")
        self.knee_r = float(m.geom_size[gid("L_knee_col")][0])
        self.left_idx = np.array([i for i, j in enumerate(self.joints) if j.startswith("L_")])
        self.right_idx = np.array([i for i, j in enumerate(self.joints) if j.startswith("R_")])
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.qpos0 = jp.array(m.key_qpos[kid])
        self._data0 = mjx.make_data(m)
        global TIP_PAIRS
        self.tip_pairs = np.array([i for i in range(m.npair)
                                   if {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom1[i]),
                                       mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[i])} in
                                   ({"floor", "L_tip"}, {"floor", "R_tip"})])
        assert len(self.tip_pairs) == 2, self.tip_pairs
        TIP_PAIRS = jp.array(self.tip_pairs)
        # obs sizes
        self._state_size = 3 + 3 + 3 * self.nj + 3 + 2
        self._priv_size = self._state_size + 3 + 1 + 2 + 6 + 2 + 4 + 3 * self.nj

    # ------------------------------------------------------------------ brax API
    @property
    def observation_size(self):
        return {"state": (self._state_size,), "privileged_state": (self._priv_size,)}

    @property
    def action_size(self):
        return self.nj

    @property
    def backend(self):
        return "mjx"

    # ------------------------------------------------------------------ helpers
    def _tip_low(self, data):
        """Lowest world point of each tip (support function along -z) and the tip centre."""
        c = data.geom_xpos[self.tip_gid]                       # (2,3)
        R = data.geom_xmat[self.tip_gid]                       # (2,3,3)
        h = jp.where(self.tip_box, jp.sum(jp.abs(self.tip_semi * R[:, 2, :]), axis=-1),
                      jp.linalg.norm(self.tip_semi * R[:, 2, :], axis=-1))
        return c[:, 2] - h, c

    def _sample_servo(self, rng):
        c = self.cfg
        k = jax.random.split(rng, 5)
        u = lambda kk, lohi, n=self.nj: jax.random.uniform(kk, (n,), minval=lohi[0], maxval=lohi[1])  # noqa: E731
        if not c.dr:
            return dict(strength=jp.ones(self.nj), speed=jp.ones(self.nj), kp=jp.ones(self.nj),
                        latency=jp.array(c.latency_sub, jp.int32), deadzone=jp.full(self.nj, c.deadzone))
        return dict(strength=u(k[0], c.dr_strength), speed=u(k[1], c.dr_speed), kp=u(k[2], c.dr_kp),
                    latency=jax.random.randint(k[3], (), c.dr_latency_sub[0], c.dr_latency_sub[1] + 1),
                    deadzone=u(k[4], c.dr_deadzone))

    def _sample_cmd(self, rng):
        c = self.cfg
        k = jax.random.split(rng, 4)
        cmd = jp.array([jax.random.uniform(k[0], minval=c.cmd_vx[0], maxval=c.cmd_vx[1]),
                        jax.random.uniform(k[1], minval=c.cmd_vy[0], maxval=c.cmd_vy[1]),
                        jax.random.uniform(k[2], minval=c.cmd_wz[0], maxval=c.cmd_wz[1])])
        zero = jax.random.uniform(k[3]) < c.cmd_zero_prob
        return jp.where(zero, jp.zeros(3), cmd)

    def _steps(self, rng, lohi):
        return jax.random.randint(rng, (), int(lohi[0] / self.cfg.ctrl_dt), int(lohi[1] / self.cfg.ctrl_dt) + 1)

    # ------------------------------------------------------------------ reset
    def reset(self, rng):
        c = self.cfg
        rng, k1, k2, k3, k4, k5, k6, k7 = jax.random.split(rng, 8)
        qpos = self.qpos0
        qj = self.q_nom + c.reset_joint_noise * jax.random.normal(k1, (self.nj,))
        qj = jp.clip(qj, self.q_lo, self.q_hi)
        tilt = c.reset_tilt * jax.random.normal(k2, (2,))
        yaw = jax.random.uniform(k3, minval=-np.pi, maxval=np.pi)
        # small-angle quaternion from roll/pitch then yaw
        q_rp = jp.array([1.0, tilt[0] / 2, tilt[1] / 2, 0.0])
        q_rp = q_rp / jp.linalg.norm(q_rp)
        q_y = jp.array([jp.cos(yaw / 2), 0, 0, jp.sin(yaw / 2)])
        w1, x1, y1, z1 = q_y
        w2, x2, y2, z2 = q_rp
        quat = jp.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
        qpos = qpos.at[3:7].set(quat).at[7:].set(qj)
        data = self._data0.replace(qpos=qpos, qvel=jp.zeros(self.sys.nv))
        data = mjx.kinematics(self.sys, data)
        low, _ = self._tip_low(data)
        qpos = qpos.at[2].add(-jp.min(low) + 0.0005)
        data = data.replace(qpos=qpos)
        data = mjx.forward(self.sys, data)
        servo = self._sample_servo(k4)
        cmd = self._sample_cmd(k5)
        low, centre = self._tip_low(data)
        info = dict(
            rng=rng, servo=servo, cmd=cmd, heading=yaw + cmd[2] * 2.0, cmd_timer=self._steps(k6, c.cmd_resample_s),
            push_timer=self._steps(k7, c.push_interval_s), last_act=jp.zeros(self.nj),
            prev_cmd=qj, q_tgt=qj, phase=jp.array(0.0), air_time=jp.zeros(2),
            contact_prev=low < 0.0015, tip_prev=centre, lift_pos=centre, step=jp.array(0, jp.int32),
        )
        metrics = {k: jp.array(0.0) for k in self._metric_keys()}
        obs = self._obs(data, info, jp.zeros(self.nj), jp.zeros((self.nj,)), rng)
        return State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    def _metric_keys(self):
        return ["r_" + k for k in self.cfg.rw] + ["sat_frac", "overspeed_frac", "peak_tau_frac", "touchdowns",
                                                   "both_contact", "vx", "height"]

    # ------------------------------------------------------------------ step
    def step(self, state: State, action):
        c, s_ = self.cfg, state.info["servo"]
        action = jp.clip(action, -1, 1)
        q_cmd = jp.clip(self.q_nom + c.action_scale * action, self.q_lo, self.q_hi)
        ts = self.ts * s_["strength"]
        w0 = self.w0 * s_["speed"]
        kp = c.kp * s_["kp"]
        dt = c.sim_dt

        def sub(carry, i):
            data, q_tgt, acc = carry
            cmd = jp.where(i < s_["latency"], state.info["prev_cmd"], q_cmd)
            q_tgt = q_tgt + jp.clip(cmd - q_tgt, -c.slew * dt, c.slew * dt)
            q = data.qpos[7:]
            qd = data.qvel[6:]
            e = q_tgt - q
            e = jp.sign(e) * jp.maximum(jp.abs(e) - s_["deadzone"], 0.0)
            u_req = kp * e - c.kd * qd
            u = jp.clip(u_req, -1, 1)
            tau = jp.clip(ts * (u - qd / w0), -ts, ts)
            data = mjx.step(self.sys, data.replace(ctrl=tau))
            acc = dict(tau2=acc["tau2"] + jp.sum((tau / self.ts) ** 2) / self.n_sub,
                       over_rated=acc["over_rated"] + jp.sum(jp.maximum(jp.abs(tau) / self.ts - c.rated_frac, 0.0) ** 2) / self.n_sub,
                       sat=jp.maximum(acc["sat"], jp.mean(jp.abs(u_req) > 1.0)),
                       over=jp.maximum(acc["over"], jp.mean(jp.abs(qd) > w0)),
                       peak=jp.maximum(acc["peak"], jp.max(jp.abs(tau) / self.ts)),
                       tau=tau)
            return (data, q_tgt, acc), None

        acc0 = dict(tau2=0.0, over_rated=0.0, sat=0.0, over=0.0, peak=0.0, tau=jp.zeros(self.nj))
        (data, q_tgt, acc), _ = jax.lax.scan(sub, (state.pipeline_state, state.info["q_tgt"], acc0),
                                            jp.arange(self.n_sub))

        info = dict(state.info)
        rng, k_cmd, k_ct, k_push, k_pt, k_obs = jax.random.split(info["rng"], 6)
        # pushes
        info["push_timer"] = info["push_timer"] - 1
        do_push = (info["push_timer"] <= 0) & (c.push_vel > 0)
        kick = c.push_vel * jax.random.normal(k_push, (2,))
        qvel = data.qvel.at[0:2].add(jp.where(do_push, kick, 0.0))
        data = data.replace(qvel=qvel)
        info["push_timer"] = jp.where(do_push, self._steps(k_pt, c.push_interval_s), info["push_timer"])

        # ---- kinematic quantities
        quat = data.qpos[3:7]
        g_body = _quat_rot_t(quat, jp.array([0.0, 0.0, -1.0]))
        v_world = data.qvel[0:3]
        v_body = _quat_rot_t(quat, v_world)
        w_body = data.qvel[3:6]
        z = data.qpos[2]
        low, centre = self._tip_low(data)
        contact = low < 0.0015
        tip_v = (centre - info["tip_prev"]) / c.ctrl_dt
        slip = jp.sum(jp.where(contact & info["contact_prev"], jp.sum(tip_v[:, :2] ** 2, -1), 0.0))
        touchdown = contact & ~info["contact_prev"]
        liftoff = ~contact & info["contact_prev"]
        lift_pos = jp.where(liftoff[:, None], centre, info["lift_pos"])
        air = info["air_time"] + c.ctrl_dt
        air_rew = jp.sum(jp.where(touchdown, jp.clip(air - c.air_time_target, -0.3, 0.3), 0.0))
        air = jp.where(contact, 0.0, air)
        info["phase"] = jp.mod(info["phase"] + 2 * jp.pi * c.gait_hz * c.ctrl_dt, 2 * jp.pi)
        # left stance when sin(phase) >= 0, right when < 0
        s = jp.sin(info["phase"])
        want_contact = jp.array([s >= 0, s < 0])
        cmd = info["cmd"]
        moving = jp.linalg.norm(cmd) > 1e-3
        stepping = moving | c.phase_when_still  # gait-phase shaping active
        swing_target = c.swing_height * jp.abs(s)
        swing_err = jp.sum(jp.where(~want_contact, (low - swing_target) ** 2, 0.0))

        yaw_cur = jp.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        head_err = jp.mod(info["heading"] - yaw_cur + jp.pi, 2 * jp.pi) - jp.pi
        # heading-frame velocity for command tracking (yaw-only frame)
        yaw_fwd = _quat_rot_t(quat, jp.array([1.0, 0.0, 0.0]))
        # stride: forward (heading-direction) travel of a tip between its lift-off and touchdown
        fwd = jp.array([jp.cos(yaw_cur), jp.sin(yaw_cur)])
        step_len = (centre[:, :2] - lift_pos[:, :2]) @ fwd
        stride = jp.sum(jp.where(touchdown, jp.clip(step_len, 0.0, c.stride_target) / c.stride_target, 0.0))
        rw = c.rw
        tilt2 = jp.sum(g_body[:2] ** 2)
        terms = dict(
            track_lin=jp.exp(-jp.sum((cmd[:2] - v_body[:2]) ** 2) / c.track_sigma),
            track_ang=jp.exp(-(cmd[2] - w_body[2]) ** 2 / c.track_ang_sigma),
            heading=jp.exp(-head_err ** 2 / c.heading_sigma),
            stop=jp.where(moving, 0.0, jp.exp(-jp.sum(v_body[:2] ** 2) / c.stop_sigma)),
            upright=jp.exp(-tilt2 / 0.02),
            height=jp.exp(-((z - self.base_z) / c.height_sigma) ** 2),
            lin_vel_z=v_world[2] ** 2,
            ang_vel_xy=jp.sum(w_body[:2] ** 2),
            torque=acc["tau2"],
            over_rated=acc["over_rated"],
            saturation=acc["sat"],
            overspeed=acc["over"],
            action_rate=jp.sum((action - info["last_act"]) ** 2),
            joint_vel=jp.sum(data.qvel[6:] ** 2),
            pose=jp.sum((data.qpos[7:] - self.q_nom) ** 2),
            slip=slip,
            feet_air_time=air_rew * stepping,
            contact_phase=jp.where(stepping, jp.mean(contact == want_contact), 0.0),
            swing_height=jp.where(stepping, swing_err, 0.0),
            both_feet=jp.where(moving, 0.0, jp.all(contact).astype(float)),
            touchdown=jp.sum(touchdown).astype(float) * (1 - moving),
            liftoff=jp.sum(liftoff).astype(float) * (1 - moving),
            stride=stride * moving,
            alive=1.0,
            termination=0.0,
        )
        # ---- termination
        knee_z = data.xpos[self.knee_bid][:, 2]
        fallen = (tilt2 > jp.sin(c.max_tilt) ** 2) | (z < c.min_height_frac * self.base_z) | \
                 jp.any(knee_z < self.knee_r) | (g_body[2] > 0)
        nan = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))
        done = (fallen | nan).astype(float)
        terms["termination"] = done
        reward = sum(rw[k] * v for k, v in terms.items())
        reward = jp.where(nan, -rw["alive"], reward)

        # ---- commands
        info["cmd_timer"] = info["cmd_timer"] - 1
        resample = info["cmd_timer"] <= 0
        new_cmd = self._sample_cmd(k_cmd)
        info["cmd"] = jp.where(resample, new_cmd, cmd)
        yaw_now = jp.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        if c.heading_cmd:
            # sampled wz becomes a heading offset; the command the policy sees is the corrective yaw rate
            info["heading"] = jp.where(resample, yaw_now + new_cmd[2] * 2.0, info["heading"])
            err = jp.mod(info["heading"] - yaw_now + jp.pi, 2 * jp.pi) - jp.pi
            info["cmd"] = info["cmd"].at[2].set(jp.clip(c.heading_gain * err, -0.3, 0.3))
        info["cmd_timer"] = jp.where(resample, self._steps(k_ct, c.cmd_resample_s), info["cmd_timer"])

        info.update(rng=rng, last_act=action, prev_cmd=q_cmd, q_tgt=q_tgt, air_time=air,
                    contact_prev=contact, tip_prev=centre, lift_pos=lift_pos, step=info["step"] + 1)
        metrics = dict(state.metrics)
        for k, v in terms.items():
            metrics["r_" + k] = rw[k] * v
        metrics.update(sat_frac=acc["sat"], overspeed_frac=acc["over"], peak_tau_frac=acc["peak"],
                       touchdowns=jp.sum(touchdown).astype(float), both_contact=jp.all(contact).astype(float),
                       vx=v_body[0], height=z)
        obs = self._obs(data, info, action, acc["tau"], k_obs)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

    # ------------------------------------------------------------------ observations
    def _obs(self, data, info, action, tau, rng):
        c = self.cfg
        quat = data.qpos[3:7]
        g_body = _quat_rot_t(quat, jp.array([0.0, 0.0, -1.0]))
        w_body = data.qvel[3:6]
        q = data.qpos[7:] - self.q_nom
        qd = data.qvel[6:]
        k = jax.random.split(rng, 4)
        n = c.obs_noise
        g_n = g_body + n * 0.03 * jax.random.normal(k[0], (3,))
        w_n = w_body + n * 0.15 * jax.random.normal(k[1], (3,))
        q_n = q + n * 0.005 * jax.random.normal(k[2], (self.nj,))
        qd_n = qd + n * 0.3 * jax.random.normal(k[3], (self.nj,))
        clock = jp.array([jp.sin(info["phase"]), jp.cos(info["phase"])])
        state = jp.concatenate([g_n, w_n, q_n, 0.1 * qd_n, action, info["cmd"], clock])
        low, centre = self._tip_low(data)
        v_body = _quat_rot_t(quat, data.qvel[0:3])
        tip_rel = jax.vmap(lambda p: _quat_rot_t(quat, p - data.qpos[0:3]))(centre).ravel()
        s_ = info["servo"]
        priv = jp.concatenate([
            jp.concatenate([g_body, w_body, q, 0.1 * qd, action, info["cmd"], clock]),
            v_body, data.qpos[2:3], (low < 0.0015).astype(float), tip_rel,
            jp.array([self.sys.pair_friction[self.tip_pairs[0], 0], jp.sum(self.sys.body_mass)]),
            jp.array([jp.mean(s_["strength"]), jp.mean(s_["speed"]), s_["latency"] * c.sim_dt * 100,
                      jp.mean(s_["deadzone"]) * 100]),
            tau / self.ts, s_["kp"], s_["deadzone"] * 100,
        ])
        return {"state": state, "privileged_state": priv}


TIP_PAIRS = jp.array([0, 1])  # set by SlenderBipedEnv from pair names (tip-floor pairs)


def randomization_fn(sys, rng, num_envs=None):
    """Per-env model randomization: tip friction, link masses/inertias, armature, gear friction."""
    def rand(key):
        k = jax.random.split(key, 5)
        fr = jax.random.uniform(k[0], minval=0.5, maxval=1.2)
        pair_friction = sys.pair_friction.at[TIP_PAIRS, 0:2].set(fr)
        ms = jax.random.uniform(k[1], sys.body_mass.shape, minval=0.85, maxval=1.15).at[0].set(1.0)
        body_mass = sys.body_mass * ms
        body_inertia = sys.body_inertia * ms[:, None]
        pel_off = jax.random.uniform(k[2], (3,), minval=-0.005, maxval=0.005)
        body_ipos = sys.body_ipos.at[1].add(pel_off)
        arm = sys.dof_armature.at[6:].multiply(jax.random.uniform(k[3], (sys.nv - 6,), minval=0.6, maxval=1.6))
        fl = sys.dof_frictionloss.at[6:].multiply(jax.random.uniform(k[4], (sys.nv - 6,), minval=0.5, maxval=2.0))
        return pair_friction, body_mass, body_inertia, body_ipos, arm, fl

    keys = rng if num_envs is None else jax.random.split(rng, num_envs)
    pf, bm, bi, bp, arm, fl = jax.vmap(rand)(keys)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({"pair_friction": 0, "body_mass": 0, "body_inertia": 0, "body_ipos": 0,
                                    "dof_armature": 0, "dof_frictionloss": 0})
    sys_v = sys.tree_replace({"pair_friction": pf, "body_mass": bm, "body_inertia": bi, "body_ipos": bp,
                              "dof_armature": arm, "dof_frictionloss": fl})
    return sys_v, in_axes
