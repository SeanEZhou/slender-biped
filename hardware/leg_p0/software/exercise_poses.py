"""Scripted joint trajectories for the measuring exercises (exercises.py on the robot, scripts/exercises_sim.py in
simulation): one definition, so the robot and the simulator run exactly the same motion. Numpy only.

Joint order (10): L hip_yaw, hip_roll, hip_pitch, knee, ankle_roll, then the same for R. Bench sign conventions
(roll + = abduction on each leg, mirrored), so q_right = q_left is a symmetric pose.

* weight shift by s (rad): both legs lean like a parallelogram, the ankles keep the boards flat (slender_biped/
  shift_gait.py balance_pose + with_ankles). s > 0 moves the weight onto the LEFT board, s < 0 onto the right.
  The c_T12 gait uses s = 0.135 (7.7 deg). The bench limits hip-roll adduction to -10 deg, so 9.5 deg is the most a
  shift can do; in the model the centre of mass crosses the standing board's inner edge at ~5.5 deg and is 7 mm from
  its centre at 10 deg (scripts/exercises_sim.py).
* foot lift by h (m): hip pitch and knee both +u, which keeps the shin's angle and so the board flat, the knee rising
  by THIGH * (cos(hp0) - cos(hp0 + u)).
"""
from __future__ import annotations

import math

import numpy as np

Q_NOM = np.array([0.0, 0.0, 0.2792526803190927, 0.5585053606381855, 0.0] * 2)
JOINTS = [f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll")]
THIGH = 0.138                     # hip pitch axis to knee axis (m), models/hw_ank3_b150_legs_10dof.xml
HP0 = Q_NOM[2]


def ease(x):
    x = min(1.0, max(0.0, x))
    return 0.5 - 0.5 * math.cos(math.pi * x)


def shift_pose(s, q=None):
    q = Q_NOM.copy() if q is None else q.copy()
    q[1] -= s; q[4] += s          # left leg: hip roll in, ankle undoes it
    q[6] += s; q[9] -= s          # right leg: hip roll out, ankle undoes it
    return q


def lift_u(h):
    """Hip pitch = knee increment (rad) that raises the board by h (m), board kept flat (rigid legs)."""
    return math.acos(math.cos(HP0) - h / THIGH) - HP0


def lift_pose(q, side, u):
    i = 0 if side == "L" else 5
    q = q.copy()
    q[i + 2] += u; q[i + 3] += u
    return q


class ShiftRamp:
    """Weight shift ramped slowly toward one side (the board on that side takes the weight) up to s_max, then held."""

    def __init__(self, side="L", rate_deg=0.5, s_max_deg=9.5, settle=2.0):
        self.sg = 1.0 if side == "L" else -1.0
        self.rate, self.s_max, self.settle = math.radians(rate_deg), math.radians(s_max_deg), settle
        self.duration = settle + self.s_max / self.rate + 3.0

    def s(self, t):
        return self.sg * min(self.s_max, max(0.0, t - self.settle) * self.rate)

    def __call__(self, t):
        s = self.s(t)
        stage = "settle" if t < self.settle else ("ramp" if abs(s) < self.s_max else "hold")
        return shift_pose(s), stage, 0


class LiftReps:
    """Stand on one foot, repeatedly: shift onto the stance board, lift the other board by h, hold, lower in
    lower_t s (short = a hard landing), unshift. One rep per (h, lower_t) pair."""

    def __init__(self, side="R", shift_deg=9.5, reps=((0.010, 0.5),), shift_t=1.0, lift_t=0.5, hold=2.5, settle=1.0):
        self.side = side                                  # the foot that lifts
        self.s = math.radians(shift_deg) * (1.0 if side == "R" else -1.0)   # weight onto the other board
        self.reps = list(reps)
        self.shift_t, self.lift_t, self.hold, self.settle = shift_t, lift_t, hold, settle
        self.rep_T = [settle + shift_t + lift_t + hold + lt + 1.0 + shift_t + settle for _, lt in self.reps]
        self.duration = sum(self.rep_T)

    def __call__(self, t):
        k = 0
        while k < len(self.reps) - 1 and t >= self.rep_T[k]:
            t -= self.rep_T[k]; k += 1
        h, lt = self.reps[k]
        t1 = self.settle; t2 = t1 + self.shift_t; t3 = t2 + self.lift_t; t4 = t3 + self.hold; t5 = t4 + lt
        t6 = t5 + 1.0; t7 = t6 + self.shift_t
        a = ease((t - t1) / self.shift_t) * (1.0 - ease((t - t6) / self.shift_t))
        up = ease((t - t2) / self.lift_t) if t < t4 else 1.0 - ease((t - t4) / lt)
        q = lift_pose(shift_pose(self.s * a), self.side, lift_u(h) * up)
        stage = ("settle" if t < t1 else "shift" if t < t2 else "lift" if t < t3 else "hold" if t < t4 else
                 "lower" if t < t5 else "land" if t < t6 else "unshift" if t < t7 else "settle")
        return q, stage, k


def parse_reps(lifts_mm, lower_s):
    """Every combination of lift heights (mm) and lowering times (s), heights first."""
    return [(h / 1000.0, lt) for lt in lower_s for h in lifts_mm]
