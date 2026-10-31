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
        both_feet=0.0, touchdown=0.0, liftoff=0.0, stride=0.0, ref_q=0.0, ref_tip=0.0, ref_contact=0.0, width=0.0,
        track_vy=0.0, line=0.0, alive=0.5, termination=-20.0))
    rated_frac: float = 0.33           # sustained-torque cap as a fraction of stall (vendor rated ~1/3)
    track_sigma: float = 0.01          # velocity-tracking kernel width (m/s)^2
    track_ang_sigma: float = 0.05      # yaw-rate tracking kernel width (rad/s)^2
    heading_sigma: float = 0.1         # heading-hold kernel width rad^2 (only with heading_cmd)
    stop_sigma: float = 0.002          # zero-command "stand still" velocity kernel (m/s)^2
    height_sigma: float = 0.03
    swing_height: float = 0.025
    air_time_target: float = 0.2
    stride_target: float = 0.06        # m, forward tip travel per step rewarded up to this (stride term)
    # reference gait (round 4, slender_biped/ref_gait.py): imitation terms and optional residual action
    ref_table: str = ""                # models/ref/<name>.npz; "" = off
    ref_residual: bool = False         # q_cmd = q_ref(phase, vx cmd) + action_scale * action
    ref_sigma_q: float = 0.1           # joint-tracking kernel width (rad^2, summed over joints)
    ref_sigma_tip: float = 0.0002      # tip-height tracking kernel width (m^2, summed over tips)
    ref_yaw_weight: float = 1.0        # weight of the hip-yaw joints in ref_q (0 = free for steering)
    width_target: float = 0.08         # m, lateral tip separation (body frame) above which the width term penalizes
    tip_condim: int = 3                # tip-floor contact dims; 4 adds the pad's torsional (yaw) friction, which the
                                       # model specifies (0.004) but condim 3 ignores
    # sideways drift (round 5). Extra policy inputs go in FRONT of the 35-dim base state, so the base layout (and
    # obs[-5:-2] = cmd) is unchanged and train.py can widen an older checkpoint's first layer with zero rows.
    obs_vel: str = ""                  # "" | "oracle" (true heading-frame vx, vy) | "odo" (stance-leg odometry estimate)
    obs_accel: bool = False            # accelerometer specific force (body frame, in g) with per-episode bias
    obs_hist: int = 0                  # past control steps of [gravity, gyro, q, qd, action] stacked in front
    odo_noise: float = 0.002           # m, per-step tip-position noise from joint encoders (0.005 rad x ~0.35 m)
    odo_bias: float = 0.01             # m/s, per-episode velocity bias (calibration) std
    odo_scale: float = 0.1             # per-episode scale error (leg-length / kinematics) half-range
    odo_alpha: float = 0.15            # low-pass on the odometry velocity (per 20 ms step)
    odo_contact: str = "sched"         # which tips count as stance: "sched" (reference timing) | "sensed" (foot switch /
                                       # load estimate: contact on this and the previous step, odo_flip reading errors)
    odo_flip: float = 0.03
    odo_src: str = "legs"              # velocity source for obs_vel="odo" and obs_line: "legs" (stance-leg kinematics) |
                                       # "flow" (downward optical-flow sensor: ground velocity x scale error + bias + noise;
                                       # immune to foot slip)
    flow_noise: float = 0.02           # m/s per-sample noise of the flow sensor
    acc_noise: float = 0.01            # g, white noise per sample
    acc_bias: float = 0.03             # g, per-episode bias half-range per axis
    vy_avg_tau: float = 0.67           # s, low-pass on heading-frame velocity for track_vy (~ one gait cycle)
    vy_avg_sigma: float = 0.0004       # (m/s)^2 kernel of the cycle-averaged lateral-velocity reward
    vy_avg_mode: str = "ema"           # "ema" (tau above; leaves ~+-0.035 m/s of the gait sway) | "box" (mean over
                                       # exactly one gait cycle: the sway cancels)
    obs_line: bool = False             # policy input: sideways offset from the commanded path, integrated from the
                                       # odometry (odo_v) since the last command change (drifts with the odometry bias)
    line_frame: str = "heading"        # integrate the offset estimate with the odometry in the current-heading frame
                                       # ("heading": a heading error leaks forward speed into it) or projected on the
                                       # path normal using the IMU yaw ("path")
    line_sigma: float = 0.01           # m^2 kernel of the "line" reward on the TRUE offset from the commanded path
    mirror_aug: bool = False           # half the episodes run the mirror image: the policy sees mirrored inputs and its
                                       # action and the reference are mirrored back (exact left-right symmetry data)


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
        if cfg.tip_condim != 3:
            for i in range(m.npair):
                names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom1[i]),
                         mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, m.pair_geom2[i])}
                if names in ({"floor", "L_tip"}, {"floor", "R_tip"}):
                    m.pair_dim[i] = cfg.tip_condim
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
        self.ref = None
        if cfg.ref_table:
            from .ref_gait import load
            self.ref = load(cfg.ref_table, jp)
            self.ref_w = jp.array([cfg.ref_yaw_weight if j.endswith("hip_yaw") else 1.0 for j in self.joints])
            assert abs(self.ref["gait_hz"] - cfg.gait_hz) < 1e-6, (self.ref["gait_hz"], cfg.gait_hz)
        self._cycle_n = int(round(1.0 / (cfg.gait_hz * cfg.ctrl_dt)))  # control steps per gait cycle
        # obs sizes
        self._frame_size = 3 + 3 + 3 * self.nj                       # history frame: gravity, gyro, q, qd, action
        self._base_size = self._frame_size + 3 + 2
        self._extra_size = ({"": 0, "oracle": 2, "odo": 2}[cfg.obs_vel] + 3 * cfg.obs_accel + int(cfg.obs_line)
                            + cfg.obs_hist * self._frame_size)
        self._state_size = self._extra_size + self._base_size
        # critic extras: averaged velocity + odometry estimate (track_vy), true and estimated path offset (line)
        self._priv_extra = 6 if cfg.rw.get("line", 0.0) else 4 if cfg.rw.get("track_vy", 0.0) else 0
        self._priv_size = self._priv_extra + self._base_size + 3 + 1 + 2 + 6 + 2 + 4 + 3 * self.nj

    # ------------------------------------------------------------------ brax API
    @property
    def observation_size(self):
        return {"state": (self._state_size,), "privileged_state": (self._priv_size,)}

    def _swap(self, x):
        """Left/right joint swap. The hip-yaw and hip-roll axes of the right leg are mirrored in the model, so the
        mirror image of a joint vector is the plain swap (no sign changes)."""
        return jp.concatenate([x[self.right_idx], x[self.left_idx]])

    def extra_obs_stats(self, base_mean, base_std):
        """Initial normalizer (mean, std) for the extra front inputs when widening an older checkpoint."""
        c = self.cfg
        m, s = [], []
        if c.obs_vel:
            m += [0.12, 0.0]; s += [0.08, 0.05]
        if c.obs_accel:
            m += [0.0, 0.0, 1.0]; s += [0.2, 0.2, 0.2]
        if c.obs_line:
            m += [0.0]; s += [0.15]
        for _ in range(c.obs_hist):
            m += list(base_mean[:self._frame_size]); s += list(base_std[:self._frame_size])
        return np.array(m), np.array(s)

    def priv_extra_stats(self):
        return (np.array([0.12, 0.0, 0.12, 0.0, 0.0, 0.0])[:self._priv_extra],
                np.array([0.08, 0.05, 0.08, 0.05, 0.15, 0.15])[:self._priv_extra])

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
        rng, kb1, kb2, kb3, kb4 = jax.random.split(rng, 5)
        info = dict(
            rng=rng, servo=servo, cmd=cmd, heading=yaw + cmd[2] * 2.0, cmd_timer=self._steps(k6, c.cmd_resample_s),
            push_timer=self._steps(k7, c.push_interval_s), last_act=jp.zeros(self.nj),
            prev_cmd=qj, q_tgt=qj, phase=jp.array(0.0), air_time=jp.zeros(2),
            contact_prev=low < 0.0015, tip_prev=centre, lift_pos=centre, step=jp.array(0, jp.int32),
            # round 5: cycle-averaged heading-frame velocity (reward), odometry and accelerometer state, history
            line_p0=qpos[0:2], line_yc=jp.array(0.0), y_est=jp.array(0.0), y_true=jp.array(0.0),
            v_avg=jp.zeros(2), v_buf=jp.zeros((self._cycle_n, 2)), odo_v=jp.zeros(2), odo_p=centre - qpos[0:3],
            odo_bias=c.odo_bias * jax.random.normal(kb1, (2,)),
            odo_scale=1.0 + jax.random.uniform(kb2, minval=-c.odo_scale, maxval=c.odo_scale),
            acc_bias=jax.random.uniform(kb3, (3,), minval=-c.acc_bias, maxval=c.acc_bias),
            acc=jp.array([0.0, 0.0, 1.0]), v_prev=jp.zeros(3), hist=jp.zeros((c.obs_hist, self._frame_size)),
            mirror=(jax.random.uniform(kb4) < 0.5) & c.mirror_aug,
        )
        metrics = {k: jp.array(0.0) for k in self._metric_keys()}
        obs, frame = self._obs(data, info, jp.zeros(self.nj), jp.zeros((self.nj,)), rng)
        info["hist"] = jp.tile(frame, (c.obs_hist, 1))
        obs, _ = self._obs(data, info, jp.zeros(self.nj), jp.zeros((self.nj,)), rng)
        return State(data, obs, jp.array(0.0), jp.array(0.0), metrics, info)

    def _metric_keys(self):
        return ["r_" + k for k in self.cfg.rw] + ["sat_frac", "overspeed_frac", "peak_tau_frac", "touchdowns",
                                                   "both_contact", "vx", "height", "vy_avg_abs"]

    # ------------------------------------------------------------------ step
    def step(self, state: State, action):
        c, s_ = self.cfg, state.info["servo"]
        action = jp.clip(action, -1, 1)
        mir = state.info["mirror"]
        act_real = jp.where(mir, self._swap(action), action)
        phase_next = jp.mod(state.info["phase"] + 2 * jp.pi * c.gait_hz * c.ctrl_dt, 2 * jp.pi)
        if self.ref is not None:
            from .ref_gait import lookup
            # mirrored episode: the mirror image of the reference half a cycle later (identical for symmetric tables)
            q_ref, z_ref, st_ref = lookup(self.ref, jp.maximum(state.info["cmd"][0], 0.0),
                                          phase_next + jp.pi * mir, jp)
            q_ref = jp.where(mir, self._swap(q_ref), q_ref)
            z_ref = jp.where(mir, z_ref[::-1], z_ref)
            st_ref = jp.where(mir, st_ref[::-1], st_ref)
        q_base = q_ref if (self.ref is not None and c.ref_residual) else self.q_nom
        q_cmd = jp.clip(q_base + c.action_scale * act_real, self.q_lo, self.q_hi)
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
        rng, k_cmd, k_ct, k_push, k_pt, k_obs, k_odo, k_acc = jax.random.split(info["rng"], 8)
        cmd_vy_prev = info["cmd"][1]
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
        info["phase"] = phase_next
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
        # ---- round 5: cycle-averaged heading-frame velocity (reward), stance-leg odometry, accelerometer
        cy, sy = jp.cos(yaw_cur), jp.sin(yaw_cur)
        to_head = lambda v: jp.array([cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]])  # noqa: E731
        if c.vy_avg_mode == "box":
            info["v_buf"] = jp.concatenate([info["v_buf"][1:], to_head(v_world)[None]], 0)
            v_avg = jp.mean(info["v_buf"], 0)
        else:
            v_avg = info["v_avg"] + (c.ctrl_dt / c.vy_avg_tau) * (to_head(v_world) - info["v_avg"])
        # tip positions relative to the pelvis as the joint encoders + IMU attitude would give them (noisy FK); the
        # pelvis moves opposite to the scheduled stance tip(s). On hardware the stance schedule is the same table.
        p_meas = centre - data.qpos[0:3] + c.odo_noise * jax.random.normal(k_odo, (2, 3))
        if c.odo_contact == "sensed" or self.ref is None:
            flip = jax.random.uniform(jax.random.fold_in(k_odo, 1), (2,)) < c.odo_flip
            st_w = ((contact & info["contact_prev"]) ^ flip).astype(float)
        else:
            st_w = st_ref.astype(float)
        st_w = jp.where(jp.sum(st_w) > 0, st_w, jp.ones(2))
        v_leg = -jp.sum(st_w[:, None] * (p_meas - info["odo_p"])[:, :2], 0) / (jp.sum(st_w) * c.ctrl_dt)
        if c.odo_src == "flow":
            b = info["odo_bias"]
            v_flow = (info["odo_scale"] * v_world[:2] + jp.array([cy * b[0] - sy * b[1], sy * b[0] + cy * b[1]])
                      + c.flow_noise * jax.random.normal(jax.random.fold_in(k_odo, 2), (2,)))
            odo_v = info["odo_v"] + c.odo_alpha * (to_head(v_flow) - info["odo_v"])
        else:
            odo_v = info["odo_v"] + c.odo_alpha * (info["odo_scale"] * to_head(v_leg) + info["odo_bias"] - info["odo_v"])
        f_imu = (_quat_rot_t(quat, (v_world - info["v_prev"]) / c.ctrl_dt + jp.array([0.0, 0.0, 9.81])) / 9.81
                 + info["acc_bias"] + c.acc_noise * jax.random.normal(k_acc, (3,)))
        info.update(v_avg=v_avg, odo_v=odo_v, odo_p=p_meas, acc=f_imu, v_prev=v_world)
        # path: line through line_p0 along the commanded heading, shifted sideways by the integrated vy command
        hc, hs = jp.cos(info["heading"]), jp.sin(info["heading"])
        rel = data.qpos[0:2] - info["line_p0"]
        info["line_yc"] = info["line_yc"] + cmd_vy_prev * c.ctrl_dt
        y_true = -hs * rel[0] + hc * rel[1] - info["line_yc"]
        if c.line_frame == "path":
            b = info["odo_bias"]
            vw = info["odo_scale"] * v_leg + jp.array([cy * b[0] - sy * b[1], sy * b[0] + cy * b[1]])
            if c.odo_src == "flow":
                vw = v_flow
            info["y_est"] = info["y_est"] + (-hs * vw[0] + hc * vw[1] - cmd_vy_prev) * c.ctrl_dt
        else:
            info["y_est"] = info["y_est"] + (odo_v[1] - cmd_vy_prev) * c.ctrl_dt
        info["y_true"] = y_true
        # stride: forward (heading-direction) travel of a tip between its lift-off and touchdown
        fwd = jp.array([jp.cos(yaw_cur), jp.sin(yaw_cur)])
        step_len = (centre[:, :2] - lift_pos[:, :2]) @ fwd
        stride = jp.sum(jp.where(touchdown, jp.clip(step_len, 0.0, c.stride_target) / c.stride_target, 0.0))
        rw = c.rw
        tilt2 = jp.sum(g_body[:2] ** 2)
        if self.ref is not None:
            ref_terms = dict(
                ref_q=jp.exp(-jp.sum(self.ref_w * (data.qpos[7:] - q_ref) ** 2) / c.ref_sigma_q),
                ref_tip=jp.exp(-jp.sum((jp.maximum(low, 0.0) - z_ref) ** 2) / c.ref_sigma_tip),
                ref_contact=jp.mean(contact == st_ref))
        else:
            ref_terms = dict(ref_q=0.0, ref_tip=0.0, ref_contact=0.0)
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
            **ref_terms,
            width=jp.where(moving, jp.maximum(jp.abs(_quat_rot_t(quat, centre[0] - centre[1])[1]) - c.width_target, 0.0) ** 2
                           * 1e4, 0.0),
            track_vy=jp.exp(-(cmd[1] - v_avg[1]) ** 2 / c.vy_avg_sigma),
            line=jp.where(moving, jp.exp(-y_true ** 2 / c.line_sigma), 0.0),
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
        info["line_p0"] = jp.where(resample, data.qpos[0:2], info["line_p0"])
        info["line_yc"] = jp.where(resample, 0.0, info["line_yc"])
        info["y_est"] = jp.where(resample, 0.0, info["y_est"])
        info["y_true"] = jp.where(resample, 0.0, info["y_true"])

        info.update(rng=rng, last_act=action, prev_cmd=q_cmd, q_tgt=q_tgt, air_time=air,
                    contact_prev=contact, tip_prev=centre, lift_pos=lift_pos, step=info["step"] + 1)
        metrics = dict(state.metrics)
        for k, v in terms.items():
            metrics["r_" + k] = rw[k] * v
        metrics.update(sat_frac=acc["sat"], overspeed_frac=acc["over"], peak_tau_frac=acc["peak"],
                       touchdowns=jp.sum(touchdown).astype(float), both_contact=jp.all(contact).astype(float),
                       vx=v_body[0], height=z, vy_avg_abs=jp.abs(v_avg[1]))
        obs, frame = self._obs(data, info, action, acc["tau"], k_obs)
        if c.obs_hist:
            info["hist"] = jp.concatenate([info["hist"][1:], frame[None]], 0)
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
        mir = info["mirror"]
        fy = jp.where(mir, jp.array([1.0, -1.0, 1.0]), 1.0)      # polar vectors (gravity, velocity, acceleration)
        fw = jp.where(mir, jp.array([-1.0, 1.0, -1.0]), 1.0)     # axial vectors (angular velocity)
        fc = jp.where(mir, jp.array([1.0, -1.0, -1.0]), 1.0)     # command (vx, vy, wz)
        sw = lambda x: jp.where(mir, self._swap(x), x)  # noqa: E731
        clock = jp.where(mir, -clock, clock)                      # phase + pi: the other leg's half of the cycle
        cmd = info["cmd"] * fc
        g_n, w_n, q_n, qd_n = g_n * fy, w_n * fw, sw(q_n), sw(qd_n)
        frame = jp.concatenate([g_n, w_n, q_n, 0.1 * qd_n, action])   # action is already in the policy's frame
        extra = []
        if c.obs_vel == "oracle":
            yaw = jp.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
            vw = data.qvel[0:2]
            extra.append(jp.array([jp.cos(yaw) * vw[0] + jp.sin(yaw) * vw[1], -jp.sin(yaw) * vw[0] + jp.cos(yaw) * vw[1]])
                         * fy[:2])
        elif c.obs_vel == "odo":
            extra.append(info["odo_v"] * fy[:2])
        if c.obs_accel:
            extra.append(info["acc"] * fy)
        if c.obs_line:
            extra.append(jp.clip(info["y_est"], -0.5, 0.5)[None] * fy[1])
        if c.obs_hist:
            extra.append(info["hist"].ravel())
        state = jp.concatenate(extra + [frame, cmd, clock])
        low, centre = self._tip_low(data)
        v_body = _quat_rot_t(quat, data.qvel[0:3])
        tip_rel = jax.vmap(lambda p: _quat_rot_t(quat, p - data.qpos[0:3]))(centre) * fy
        tip_rel = jp.where(mir, tip_rel[::-1], tip_rel).ravel()
        con = (low < 0.0015).astype(float)
        s_ = info["servo"]
        priv = jp.concatenate([
            (jp.concatenate([info["v_avg"], info["odo_v"], jp.clip(jp.array([info["y_true"], info["y_est"]]), -0.5, 0.5)])
             * jp.concatenate([fy[:2], fy[:2], fy[1:2], fy[1:2]]))[:self._priv_extra],
            jp.concatenate([g_body * fy, w_body * fw, sw(q), 0.1 * sw(qd), action, cmd, clock]),
            v_body * fy, data.qpos[2:3], jp.where(mir, con[::-1], con), tip_rel,
            jp.array([self.sys.pair_friction[self.tip_pairs[0], 0], jp.sum(self.sys.body_mass)]),
            jp.array([jp.mean(s_["strength"]), jp.mean(s_["speed"]), s_["latency"] * c.sim_dt * 100,
                      jp.mean(s_["deadzone"]) * 100]),
            sw(tau / self.ts), sw(s_["kp"]), sw(s_["deadzone"]) * 100,
        ])
        return {"state": state, "privileged_state": priv}, frame


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
