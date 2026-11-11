"""Deterministic rocking walker for the F_board150w pads (2026-11-08): a non-learning baseline.

Without ankles a flat board fixes its shin's tilt, so with both boards flat the pelvis cannot shift sideways and the
centre of mass stays between the boards (inner edges at +-15.5 mm). A foot can then only lift in two ways:

* tip over the stance board's INNER edge (centre of mass outside the support: a fall toward the lifted foot that the
  swing foot has to catch; what the RL walkers do, and how the robot fell on its first step), or
* rock the whole robot onto the stance board's OUTER edge (+-65.5 mm). The other board lifts by itself and gravity
  rocks the robot back: the centre of mass never leaves the span between the outer edges, so recovery is automatic
  as long as the rock stays below ~14 deg (legs only; ~11 deg with torso and head).

This controller does the second: a toy-walker rocking gait. A cycle (two steps) has phase phi; sin(phi) > 0 rocks
toward the right board (left foot swings), sin(phi) < 0 toward the left. Joint targets are offsets from the stance pose:

* rocking drive: the leg on the side to rock AWAY from extends and the other shortens (knee, with hip pitch following
  at `hip_ratio`), plus differential hip roll; amplitudes a_len, a_roll, phase lead psi
* swing: while rocked onto the other board the unloaded leg flexes (lift) and swings forward by `step` (hip pitch),
  the stance leg sweeps back by the same amount so the pelvis travels forward
* feedback: IMU pitch and pitch rate on both hip pitches (keeps the pelvis upright fore-aft); IMU roll on the rocking
  amplitude (more drive when the rock is too small, less when too big), so the rock stays near `roll_target`

`RockWalker` has the stand policies' calling convention (35-dim observation in, 8 actions out) with an env dict whose
action_scale is 1.0 (actions are radians from the stance pose), so the robot runner can play it like a policy.
Joint order: L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee.
Signs (sim = bench): roll + = abduction, hip pitch + = leg forward, knee + = flexion.
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT = dict(
    T=0.9,            # s per cycle (two steps)
    a_len=0.10,       # rad of knee for the rocking leg-length drive
    a_roll=0.04,      # rad of differential hip roll for the rocking drive
    psi=0.6,          # rad: drive leads the swing by this much phase
    hip_ratio=0.5,    # hip pitch follows the knee by this ratio in length changes (flat-ish foot)
    lift=0.35,        # rad of knee flexion at mid-swing
    lift_c=0.5,       # swing window centre, fraction of the half cycle
    lift_w=0.6,       # swing window width, fraction of the half cycle
    step=0.0,         # rad of hip pitch: swing leg forward / stance leg back per step
    k_p=0.0, k_pd=0.0,  # pitch feedback on both hip pitches (rad per rad, rad per rad/s)
    roll_target=0.06,   # rad: wanted rock amplitude
    k_amp=0.0,          # rocking-drive gain on (roll_target - measured peak roll)
)
KEYS = list(DEFAULT)


def _window(x, c, w):
    """Smooth bump on [c - w/2, c + w/2] (x, c, w as fractions of a half cycle), 0 outside, 1 at the centre."""
    u = (x - (c - w / 2)) / max(w, 1e-6)
    return np.where((u > 0) & (u < 1), np.sin(np.pi * np.clip(u, 0, 1)) ** 2, 0.0)


class RockWalker:
    def __init__(self, params=None, ctrl_dt=0.02, start_ramp=1.5):
        self.p = dict(DEFAULT, **(params or {}))
        self.dt = ctrl_dt
        self.t = 0.0
        self.peak = 0.0
        self.amp = 1.0
        self.start_ramp = start_ramp
        self.env = dict(action_scale=1.0, ctrl_dt=ctrl_dt, gait_hz=1.0 / self.p["T"], task="rock_walk",
                        model="hw_pad_board150w_legs_8dof", ref_table="", ref_residual=False)

    def reset(self):
        self.t, self.peak, self.amp = 0.0, 0.0, 1.0

    def targets(self, g, w):
        """Joint offsets (rad) from the stance pose, given IMU gravity direction g and gyro w (pelvis frame)."""
        p = self.p
        phi = 2 * math.pi * self.t / p["T"]
        ramp = min(1.0, self.t / self.start_ramp) if self.start_ramp > 0 else 1.0
        pitch = math.asin(max(-1.0, min(1.0, g[0])))            # + = leaning forward
        roll = math.asin(max(-1.0, min(1.0, g[1])))             # + = leaning left
        # rock amplitude regulation: track the peak |roll| over each half cycle
        self.peak = max(self.peak, abs(roll))
        if int(phi / math.pi) != int((phi - 2 * math.pi * self.dt / p["T"]) / math.pi):
            self.amp = float(np.clip(self.amp + p["k_amp"] * (p["roll_target"] - self.peak), 0.3, 2.5))
            self.peak = 0.0
        a = ramp * self.amp
        drive = math.sin(phi + p["psi"])                       # > 0: rock right (left leg long, right leg short)
        dq = np.zeros(8)
        dk = a * p["a_len"] * drive
        dq[3] += -dk; dq[7] += +dk                            # left knee extends, right flexes
        dq[2] += -p["hip_ratio"] * dk; dq[6] += +p["hip_ratio"] * dk
        dq[1] += a * p["a_roll"] * drive; dq[5] += -a * p["a_roll"] * drive
        # swing: left leg in the first half cycle, right in the second
        half = (phi % (2 * math.pi)) / math.pi                 # 0..2
        for leg, h0 in ((0, 0.0), (1, 1.0)):
            x = half - h0
            if x < 0:
                x += 2.0
            s = float(_window(x, p["lift_c"], p["lift_w"])) if x < 1.0 else 0.0
            k = 4 * leg
            dq[k + 3] += ramp * p["lift"] * s
            dq[k + 2] += ramp * 0.5 * p["lift"] * s
            # forward swing during the window, slow sweep back for the rest of the cycle
            if x < 1.0:
                prog = np.clip((x - (p["lift_c"] - p["lift_w"] / 2)) / max(p["lift_w"], 1e-6), 0, 1)
                fw = -1 + 2 * (0.5 - 0.5 * math.cos(math.pi * prog))
            else:
                fw = 1 - 2 * (x - 1.0)
            dq[k + 2] += ramp * p["step"] * fw
        # fore-aft balance on both hip pitches
        fb = p["k_p"] * pitch + p["k_pd"] * w[1]
        dq[2] += fb; dq[6] += fb
        return dq

    def __call__(self, obs):
        g, w = obs[0:3], obs[3:6]
        dq = self.targets(g, w)
        self.t += self.dt
        return np.clip(dq, -1.0, 1.0)


# ---------------------------------------------------------------- simpler probes (scripts/det_walk.py, robot runner)
DT = 0.02


def _env(ctl):
    return dict(action_scale=1.0, ctrl_dt=DT, gait_hz=1.0, task=f"det_{type(ctl).__name__.lower()}",
                model="hw_pad_board150w_legs_8dof", ref_table="", ref_residual=False)


class Lift:
    P = dict(lift=0.6, t_up=1.5, t0=1.0)   # knee flexion (rad, hip pitch follows at half), seconds to lift, start time

    def __init__(self, p=None, ctrl_dt=DT): self.p = dict(self.P, **(p or {})); self.t = 0.0; self.env = _env(self)

    def __call__(self, obs):
        p = self.p
        s = float(np.clip((self.t - p["t0"]) / p["t_up"], 0, 1)); s = 0.5 - 0.5 * math.cos(math.pi * s)
        self.t += DT
        dq = np.zeros(8); dq[3] = p["lift"] * s; dq[2] = 0.5 * p["lift"] * s
        return dq


class Sine:
    P = dict(a=0.1, f=1.2)

    def __init__(self, p=None, ctrl_dt=DT): self.p = dict(self.P, **(p or {})); self.t = 0.0; self.env = _env(self)

    def __call__(self, obs):
        u = self.p["a"] * math.sin(2 * math.pi * self.p["f"] * self.t) * min(1.0, self.t / 1.0)
        self.t += DT
        return np.array([0, u, 0, 0, 0, -u, 0, 0])


class CPG:
    P = dict(f=1.2, A=0.06, a0=0.1, k=0.5)   # Hz, target roll amplitude (rad), start drive (rad), gain

    def __init__(self, p=None, ctrl_dt=DT): self.p = dict(self.P, **(p or {})); self.t = 0.0; self.env = _env(self); self.a = self.p["a0"]; self.peak = 0.0; self.half = 0

    def __call__(self, obs):
        p = self.p
        r = math.asin(max(-1, min(1, obs[1]))); self.peak = max(self.peak, abs(r))
        h = int(2 * p["f"] * self.t)
        if h != self.half:
            self.half = h
            if self.t > 1.0:
                self.a = float(np.clip(self.a + p["k"] * (p["A"] - self.peak), 0.03, 0.2))
            self.peak = 0.0
        u = self.a * math.sin(2 * math.pi * p["f"] * self.t)
        self.t += DT
        return np.array([0, u, 0, 0, 0, -u, 0, 0])


# planar leg (F_board150w models, stance pose hip 16 / knee 32 deg): hip pitch axis -> knee -> board centre
L_THIGH, L_SHIN = 0.1380, 0.1438
SHIN_OFS = math.radians(-0.2)          # board-centre direction from the knee = hip pitch - knee + this (pelvis level)
BOARD_HALF = 0.060                     # board half length (collision box)


def leg_fk(hp, kn):
    """Board centre relative to the hip pitch axis (x forward, z up) and the board's pitch change from level."""
    a_t, a_s = hp, hp - kn + SHIN_OFS
    x = L_THIGH * math.sin(a_t) + L_SHIN * math.sin(a_s)
    z = -L_THIGH * math.cos(a_t) - L_SHIN * math.cos(a_s)
    return x, z, (hp - kn) - math.radians(16 - 32)


def leg_ik(x, z, hp, kn):
    """Hip pitch and knee putting the board centre at (x, z); Newton from (hp, kn)."""
    for _ in range(30):
        fx, fz, _ = leg_fk(hp, kn)
        ex, ez = x - fx, z - fz
        if abs(ex) + abs(ez) < 1e-7:
            break
        e = 1e-6
        J = np.array([[(leg_fk(hp + e, kn)[0] - fx) / e, (leg_fk(hp, kn + e)[0] - fx) / e],
                      [(leg_fk(hp + e, kn)[1] - fz) / e, (leg_fk(hp, kn + e)[1] - fz) / e]])
        d = np.linalg.solve(J, [ex, ez])
        hp, kn = hp + d[0], kn + d[1]
    return hp, kn


class QuickStep:
    """One quick step from the stance pose, then hold (2026-11-08 deterministic step test). Nothing moves for t0 s;
    then the swing leg (right by default) lifts (knee and hip-pitch bump) and swings so its board centre lands `fwd`
    m ahead at the same height, heel first (the leg stays a two-link chain: reaching forward without moving the
    pelvis tilts the board toe-up, and its heel is placed on the floor), optionally rolled outward by `out` rad of
    hip roll. The other leg and the pelvis keep the stance pose. Then everything holds. The swing takes T s.
    k_roll: from lift-off on, the swing hip roll adds k_roll x the IMU roll (rad), to land the board flat although
    the body has tipped toward the lifted foot."""
    P = dict(fwd=0.03, out=0.06, T=0.3, lift_knee=0.3, lift_hip=0.15, t0=1.0, leg=1.0, k_roll=0.0,
             shift=0.0, t_shift=2.0, release=0.0)

    def __init__(self, p=None, ctrl_dt=DT):
        self.p = dict(self.P, **(p or {}))
        self.t = 0.0
        self.env = _env(self)
        hp0, kn0 = math.radians(16), math.radians(32)
        x0, z0, _ = leg_fk(hp0, kn0)
        hp, kn, dz = hp0, kn0, 0.0
        for _ in range(4):                       # heel on the floor: raise the centre by the tilt's drop
            hp, kn = leg_ik(x0 + self.p["fwd"], z0 + dz, hp, kn)
            dz = BOARD_HALF * abs(math.sin(leg_fk(hp, kn)[2]))
        self.d_land = np.array([hp - hp0, kn - kn0])
        self.board_pitch = leg_fk(hp, kn)[2]

    def __call__(self, obs):
        p = self.p
        # weight shift first (shift > 0): both hip rolls by `shift` rad, the stance leg's inward and the swing leg's
        # outward: the legs lean like a parallelogram (boards on their edges, pelvis level) and the centre of mass
        # slides over the stance board, over t_shift s after t0; the step starts when the shift is complete. During the
        # swing the shift is kept, or blended out by the fraction `release`
        t_step = p["t0"] + (p["t_shift"] if p["shift"] > 0 else 0.0)
        s = float(np.clip((self.t - t_step) / p["T"], 0.0, 1.0))
        a = float(np.clip((self.t - p["t0"]) / p["t_shift"], 0.0, 1.0)) if p["shift"] > 0 else 0.0
        a = (0.5 - 0.5 * math.cos(math.pi * a)) * (1.0 - p["release"] * (s - math.sin(2 * math.pi * s) / (2 * math.pi)))
        self.t += DT
        cyc = s - math.sin(2 * math.pi * s) / (2 * math.pi)
        bump = 0.5 - 0.5 * math.cos(2 * math.pi * s)
        k = 4 * int(p["leg"])
        dq = np.zeros(8)
        dq[k + 2] = self.d_land[0] * cyc + p["lift_hip"] * bump
        dq[k + 3] = self.d_land[1] * cyc + p["lift_knee"] * bump
        dq[k + 1] = p["out"] * cyc
        sw, st = 4 * int(p["leg"]), 4 * (1 - int(p["leg"]))
        dq[sw + 1] += p["shift"] * a                 # swing leg: abduct (outward)
        dq[st + 1] -= p["shift"] * a                 # stance leg: adduct (inward)
        if s > 0:                                # from lift-off on: swing hip roll follows the measured body roll
            roll = math.asin(max(-1.0, min(1.0, float(obs[1]))))          # + = leaning left (IMU gravity)
            dq[k + 1] += p["k_roll"] * roll
        return dq


class ShiftWalk:
    """Plays a shift-and-step reference table (slender_biped/shift_gait.py) open loop: hip-roll weight shift, quick
    step, settle, alternating legs (2026-11-09). The clock waits at each lift-off until the IMU gyro reads under
    `gate_w` rad/s (at most `gate_max` s), like the trained walkers. The forward speed ramps up at `slew` m/s^2 to
    `vx` (step length = vx x 2.5 s). `table`: 0 = b150L_shift."""
    P = dict(vx=0.012, slew=0.01, gate_w=0.3, gate_max=1.0, t0=1.0, table=0.0)
    TABLES = ("b150L_shift",)

    def __init__(self, p=None, ctrl_dt=DT):
        from pathlib import Path
        self.p = dict(self.P, **(p or {}))
        z = np.load(Path(__file__).resolve().parents[1] / "models" / "ref" / f"{self.TABLES[int(self.p['table'])]}.npz")
        self.ref = {k: np.asarray(z[k]) for k in ("q", "v")}
        self.hz = float(z["gait_hz"])
        st = np.asarray(z["stance"])[-1]
        n = st.shape[0]
        self.gates = [i / n - 0.005 for i in range(n) if st[i - 1].all() and not st[i].all()]
        self.t, self.phase, self.v, self.held = 0.0, 0.0, 0.0, 0
        self.q_nom = self.ref["q"][0, 0].copy()
        self.env = dict(_env(self), gait_hz=self.hz)

    def _lookup(self, v, ph):
        Q, vs = self.ref["q"], self.ref["v"]
        n = Q.shape[1]
        fv = np.interp(v, vs, np.arange(len(vs)))
        i0 = min(int(fv), len(vs) - 2); wv = fv - i0
        fp = (ph / (2 * math.pi)) * n
        j0 = int(math.floor(fp)) % n; j1 = (j0 + 1) % n; wp = fp - math.floor(fp)
        a = Q[i0, j0] * (1 - wp) + Q[i0, j1] * wp
        b = Q[i0 + 1, j0] * (1 - wp) + Q[i0 + 1, j1] * wp
        return a * (1 - wv) + b * wv

    def __call__(self, obs):
        p = self.p
        self.t += DT
        if self.t < p["t0"]:
            return np.zeros(8)
        self.v += float(np.clip(p["vx"] - self.v, -p["slew"] * DT, p["slew"] * DT))
        ph_new = (self.phase + 2 * math.pi * self.hz * DT) % (2 * math.pi)
        u0, u1 = self.phase / (2 * math.pi), ph_new / (2 * math.pi)
        cross = any(((u0 - g) % 1.0) > ((u1 - g) % 1.0) for g in self.gates)
        w = obs[3:5]
        hold = cross and float(w @ w) > p["gate_w"] ** 2 and self.held < int(p["gate_max"] / DT)
        self.held = self.held + 1 if hold else 0
        if not hold:
            self.phase = ph_new
        return self._lookup(self.v, self.phase) - self.q_nom


class RefWalk(ShiftWalk):
    """10-joint legs with ankle roll (2026-11-10): plays a reference table (slender_biped/shift_gait.py --ankle) with
    the settle gate, plus IMU feedback. Roll u = k_r x roll + k_w x roll rate goes onto both hip rolls (gain 1) and
    both ankle rolls (gain `ank`), with the signs that turn both legs the same way in the world; pitch v = k_p x pitch
    + k_pd x pitch rate onto both hip pitches. `table` indexes TABLES (default c_T12, 1.2 s steps). Roll / pitch from
    the gravity direction in the pelvis frame (obs[0:3]), rates from the gyro (obs[3:5]). Tuned by
    scripts/refwalk_opt.py (results/refwalk_<table>.json)."""
    P = dict(vx=0.02, slew=0.01, gate_w=0.3, gate_max=1.0, t0=1.0, table=0.0, k_r=0.0, k_w=0.0, ank=1.0, k_p=0.0,
             k_pd=0.0, lim=0.15)
    TABLES = ("c_T12", "b150A_shift", "b150A_slow1", "c_T08", "c_big")
    MODEL = "hw_ank3_b150_legs_10dof"

    def __init__(self, p=None, ctrl_dt=DT):
        super().__init__(p, ctrl_dt)
        self.nj = self.ref["q"].shape[-1]
        self.env = dict(self.env, model=self.MODEL)
        jn = [f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll")]
        self.i = {n: k for k, n in enumerate(jn)}

    def __call__(self, obs):
        p = self.p
        if self.t + DT < p["t0"]:
            self.t += DT
            return np.zeros(self.nj)
        q = super().__call__(obs)
        g, w = obs[0:3], obs[3:6]
        roll, pitch = math.atan2(g[1], -g[2]), math.atan2(g[0], -g[2])
        u = float(np.clip(p["k_r"] * roll + p["k_w"] * w[0], -p["lim"], p["lim"]))
        v = float(np.clip(p["k_p"] * pitch + p["k_pd"] * w[1], -p["lim"], p["lim"]))
        i = self.i
        q[i["L_hip_roll"]] += u; q[i["R_hip_roll"]] -= u
        q[i["L_ankle_roll"]] += p["ank"] * u; q[i["R_ankle_roll"]] -= p["ank"] * u
        q[i["L_hip_pitch"]] += v; q[i["R_hip_pitch"]] += v
        return q


CONTROLLERS = dict(lift=Lift, sine=Sine, cpg=CPG, rock=RockWalker, step=QuickStep, shiftwalk=ShiftWalk, refwalk=RefWalk)
