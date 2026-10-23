"""Hobby smart-servo model: position command -> PWM duty -> torque through a DC-motor curve.

    u   = clip(kp * (q_cmd - q) - kd * qd, -1, 1)          # servo-internal P(D) on duty
    tau = tau_stall * (u - qd / omega_noload)               # linear torque-speed (back-EMF)
    tau = clip(tau, -I_lim/I_stall * tau_stall, ...)        # firmware current limit

Back-EMF makes the servo brake even at u = 0, and caps speed at omega_noload under
full duty -- exactly the behavior an ideal PD with a torque clip hides. The command
path adds latency, slew limiting, a deadband and 12-bit quantization. Nothing is
silently clipped: every step records the *requested* (unsaturated) duty and torque,
so violations can be counted afterwards.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ServoParams:
    tau_stall: np.ndarray
    omega0: np.ndarray
    tau_rated: np.ndarray          # nan where unknown
    kp: np.ndarray
    kd: np.ndarray
    current_limit_frac: np.ndarray
    deadband: float = 0.0015
    latency_steps: int = 0         # in physics steps
    slew: float = 8.0              # rad/s max target slew
    counts_per_rev: int = 4096

    @classmethod
    def from_config(cls, servo_specs: list[dict], servo_model: dict, dt: float):
        f = lambda k: np.array([s[k] if s.get(k) is not None else np.nan for s in servo_specs], float)  # noqa: E731
        n = len(servo_specs)
        return cls(
            tau_stall=f("stall_torque"), omega0=f("noload_speed"), tau_rated=f("rated_torque"),
            kp=np.full(n, servo_model["kp_duty_per_rad"]), kd=np.full(n, servo_model["kd_duty_per_radps"]),
            current_limit_frac=np.full(n, servo_model["current_limit_frac"]),
            deadband=servo_model["deadband_rad"], latency_steps=int(round(servo_model["latency_s"] / dt)),
            slew=servo_model["target_slew_radps"],
            counts_per_rev=int(min(s.get("resolution_counts", 4096) for s in servo_specs)),
        )


class ServoBank:
    def __init__(self, p: ServoParams, dt: float, strength_scale=None, speed_scale=None):
        self.p, self.dt = p, dt
        n = len(p.tau_stall)
        self.ts = p.tau_stall * (1.0 if strength_scale is None else strength_scale)
        self.w0 = p.omega0 * (1.0 if speed_scale is None else speed_scale)
        self.q_tgt = np.zeros(n)
        self.buf: deque = deque()
        self.n = n
        self.last = {}

    def reset(self, q0):
        self.q_tgt = np.array(q0, float)
        self.buf = deque([self.q_tgt.copy()] * (self.p.latency_steps + 1), maxlen=self.p.latency_steps + 1)

    def command(self, q_cmd):
        """Called at every physics step with the policy's current position command."""
        q = np.round(np.asarray(q_cmd) / (2 * np.pi / self.p.counts_per_rev)) * (2 * np.pi / self.p.counts_per_rev)
        self.buf.append(q)

    def torque(self, q, qd):
        q_cmd = self.buf[0]
        step = self.p.slew * self.dt
        self.q_tgt = self.q_tgt + np.clip(q_cmd - self.q_tgt, -step, step)
        err = self.q_tgt - q
        err = np.where(np.abs(err) < self.p.deadband, 0.0, err)
        u_req = self.p.kp * err - self.p.kd * qd
        u = np.clip(u_req, -1, 1)
        tau_req_ideal = self.ts * u_req  # what an unlimited-voltage servo would push (diagnostic)
        tau = self.ts * (u - qd / self.w0)
        lim = self.ts * self.p.current_limit_frac
        tau_out = np.clip(tau, -lim, lim)
        self.last = {
            "u_req": u_req, "u": u, "tau": tau_out, "tau_ideal": tau_req_ideal, "qd": qd,
            "duty_sat": np.abs(u_req) > 1.0, "current_lim": np.abs(tau) > lim,
            "over_speed": np.abs(qd) > self.w0,
        }
        return tau_out


def available_torque(tau_stall, omega0, qd, direction=1.0):
    """Max |torque| in `direction` (+1/-1) at joint speed qd under full duty, current-limited at stall."""
    return np.clip(tau_stall * (1.0 - direction * qd / omega0), 0.0, tau_stall)
