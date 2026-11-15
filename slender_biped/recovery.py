"""Round 13 (2026-11-15) recovery supervisor: walk -> stand and stabilize -> resume. Numpy; the training env
(mjx_env, EnvConfig.sup*) runs the same rule in JAX.

While walking, the IMU's roll and pitch relative to the moment the gait clock started, plus sup_tau x their rates
low-passed at sup_lpf Hz (the frame's 5 Hz ring stays out), predict the tilt. Past sup_roll / sup_pitch the walk stops:
the command drops to zero and the recovery controller (the walker itself at zero command, the stance pose, or another
policy) takes over for at least sup_min_s. Once the body has stayed within sup_quiet_tilt of the start attitude and
turned slower than sup_quiet_w for sup_quiet_s (or after sup_max_s), walking resumes from the clock's start with the
forward command ramping up again.

Thresholds from the robot's k5 logs (scripts/robot13_analysis/detector.py, 69 trials, 39 falls, 9 min of walking): k5's
normal steps already reach 10-11 deg predicted roll (p99). 12 / 12 deg at tau 0.1 s catches all 39 falls 0.2 s before
they pass 18 deg, with ~2.4 false stops per minute; 10 / 10 deg catches them 0.4 s earlier at ~14 false stops a minute.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def roll_of(g):
    return math.atan2(-float(g[1]), -float(g[2]))


def pitch_of(g):
    return math.atan2(float(g[0]), -float(g[2]))


@dataclass
class SupParams:
    roll: float = 0.21          # rad
    pitch: float = 0.21
    tau: float = 0.1            # s
    lpf: float = 3.0            # Hz
    quiet_tilt: float = 0.05    # rad
    quiet_w: float = 0.35       # rad/s
    quiet_s: float = 0.6
    min_s: float = 0.6
    max_s: float = 5.0

    @classmethod
    def from_env(cls, env: dict, **over):
        kw = {f: float(env[f"sup_{f}"]) for f in cls.__dataclass_fields__ if f"sup_{f}" in env}
        kw.update({k: v for k, v in over.items() if v is not None})
        return cls(**kw)


class Supervisor:
    def __init__(self, p: SupParams, dt: float):
        self.p, self.dt = p, dt
        self.ref = None
        self.wf = np.zeros(2)
        self.on, self.t, self.q = False, 0, 0
        self.n = 0
        self.pred = np.zeros(2)
        self.al = 1.0 - math.exp(-2 * math.pi * p.lpf * dt)

    def start(self, g):
        """The attitude the tilt is measured from (the robot standing as the clock starts)."""
        self.ref = np.array([roll_of(g), pitch_of(g)])

    def update(self, g, w, walking: bool):
        """Returns None, 'trigger' (stop walking now) or 'resume' (walk again from the clock's start)."""
        if self.ref is None:
            self.start(g)
        p = self.p
        ang = np.array([roll_of(g), pitch_of(g)]) - self.ref
        self.wf += self.al * (np.asarray(w[:2], float) - self.wf)
        self.pred = ang + p.tau * self.wf
        trig = walking and not self.on and (abs(self.pred[0]) > p.roll or abs(self.pred[1]) > p.pitch)
        quiet = bool(np.all(np.abs(ang) < p.quiet_tilt) and np.all(np.abs(self.wf) < p.quiet_w))
        if self.on:
            self.t += 1
            self.q = self.q + 1 if quiet else 0
            if (self.t >= int(p.min_s / self.dt) and self.q >= int(p.quiet_s / self.dt)) or self.t >= int(p.max_s / self.dt):
                self.on, self.t, self.q = False, 0, 0
                return "resume"
            return None
        if trig:
            self.on, self.t, self.q = True, 0, 0
            self.n += 1
            return "trigger"
        return None
