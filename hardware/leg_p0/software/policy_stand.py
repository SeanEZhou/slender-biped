"""Run a trained stand policy on the real robot: IMU (RP2040) + 8 servos (Feetech bus) -> policy at 50 Hz.

FIRST HARDWARE TEST OF A SIMULATION POLICY. Keep a hand next to the robot and on the power switch.

    python policy_stand.py ports                      # which serial port is the servo bus and which the IMU
    python policy_stand.py imu-calib                  # once per IMU mounting: zero = hip plate level
    python policy_stand.py imu-calib --axis roll      # redo one zero only (roll | pitch)
    python policy_stand.py imu-check                  # live gravity/gyro in pelvis axes (no motion)
    python policy_stand.py timing                     # how fast the bus + IMU loop can run (no motion)
    python policy_stand.py run --dry-run              # policy runs, servos stay limp: hold the robot and tilt it
    python policy_stand.py run --enable               # START: stance, then Enter engages the policy
                                                      # STOP: Enter / q / space (or Ctrl-C) -> holds the stance pose

The policy is a trained network from runs/ (default: hw_stand_push2, the as-built legs + pelvis model, no torso or
head). Its 35 inputs are rebuilt exactly as in training (slender_biped/mjx_env.py): gravity direction and gyro in the
pelvis frame, joint angles relative to the standing pose, 0.1 x joint speeds, the previous action, a zero velocity
command and the 1.6 Hz gait clock. Its 8 outputs are joint targets: q = stance + 0.35 x action (rad).

Safety: everything is clamped to the bench joint limits; the policy stops if the pelvis tilts past --max-tilt, the IMU
goes quiet or the bus fails (a few garbled reads in a row are ridden through on the previous reading). Every stop holds
the stance pose with the torque on, so the robot doesn't collapse; --limp-on-fault switches the torque off on faults.
The sim joint signs are the bench signs (yaw + toe out, roll + abduction, pitch + leg forward, knee + foot back),
so hardware degrees map straight to the policy's radians.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import serial
import yaml
from serial.tools import list_ports

import biped_bench as bb
import feetech_bench as fb
from scservo_sdk import COMM_SUCCESS, GroupSyncRead, GroupSyncWrite

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MOUNT_FILE = HERE / "imu_mount.yaml"
DEFAULT_RUN = "hw_stand_push2"
VID_SERVO, VID_IMU = 0x1A86, 0x2E8A           # CH343 bus adapter, RP2040


# ------------------------------------------------------------------ ports
def find_ports():
    servo = imu = None
    for p in list_ports.comports():
        if p.vid == VID_SERVO:
            servo = p.device
        elif p.vid == VID_IMU:
            imu = p.device
    return servo, imu


def open_bus(port):
    cfg = bb.merged_cfg()
    cfg["port"] = port or find_ports()[0] or cfg["port"]
    return fb.Bus(cfg)


# ------------------------------------------------------------------ policy
class _Stub:
    """Stands in for brax/jax classes when unpickling: the weights and normalizer inside are plain numpy."""
    def __init__(self, *a, **k):
        self.__dict__.update(k)

    def __setstate__(self, s):
        self.__dict__.update(s if isinstance(s, dict) else {"_state": s})


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split(".")[0] in ("numpy", "builtins", "collections", "copyreg", "_codecs"):
            return super().find_class(mod, name)
        return _Stub


class Policy:
    """Deterministic Brax PPO policy (same maths as slender_biped/evaluation.py NumpyPolicy), numpy only."""

    def __init__(self, run, params=None):
        d = ROOT / "runs" / run
        pkls = sorted(d.glob("params_*M.pkl")) or [d / "params.pkl"]
        self.path = d / params if params else pkls[-1]
        with open(self.path, "rb") as f:
            norm, pol = _Unpickler(f).load()[:2]
        self.mean = np.asarray(vars(norm)["mean"]["state"], float)
        self.std = np.asarray(vars(norm)["std"]["state"], float)
        layers = pol["params"]
        keys = sorted(layers, key=lambda s: int(s.split("_")[-1]))
        self.W = [np.asarray(layers[k]["kernel"], float) for k in keys]
        self.b = [np.asarray(layers[k]["bias"], float) for k in keys]
        self.env = json.loads((d / "config.json").read_text())["env"]
        self.n_in = self.W[0].shape[0]       # 6 + 3 x joints + 5: checked against the model's joints in cmd_run

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = x / (1.0 + np.exp(-x))          # swish
        return np.tanh(x[: x.shape[-1] // 2])


def ref_lookup(table, vx, phase):
    """Reference-gait joint targets at (forward speed, gait phase): slender_biped/ref_gait.py lookup, numpy only."""
    v, nph = table["v"], table["q"].shape[1]
    fv = float(np.clip(vx / v[-1] * (len(v) - 1), 0.0, len(v) - 1.0))
    i0 = int(np.clip(math.floor(fv), 0, len(v) - 2))
    wv = fv - i0
    fp = (phase % (2 * math.pi)) / (2 * math.pi) * nph
    j0 = int(math.floor(fp)) % nph
    j1 = (j0 + 1) % nph
    wp = fp - math.floor(fp)
    Q = table["q"]
    a = Q[i0, j0] * (1 - wp) + Q[i0, j1] * wp
    b = Q[i0 + 1, j0] * (1 - wp) + Q[i0 + 1, j1] * wp
    return a * (1 - wv) + b * wv


DET_NAMES = ("lift", "sine", "cpg", "rock", "step", "shiftwalk", "refwalk")


class Rocker:
    """Deterministic (non-learning) controllers from slender_biped/rock_walk.py, as tried in simulation with
    scripts/det_walk.py: lift (lift the left board and hold), sine (open-loop side-to-side rock on the hip rolls),
    cpg (the same rock, amplitude regulated on the IMU roll), rock (rock + swing + pitch feedback), step (one quick
    step with the right leg from the stance, then hold: fwd, out, T, lift_knee, lift_hip, t0, shift, release),
    shiftwalk (the shift-and-step reference b150L_shift played open loop: vx, slew, gate_w, gate_max), refwalk (10-joint
    legs with ankle roll: a reference table with IMU roll / pitch feedback on hips and ankles; table, vx, k_r, k_w, ank,
    k_p, k_pd; results/refwalk_<table>.json gains are loaded if present). Run name
    <controller> for legs only, <controller>_full for torso and head on; parameters with --det k=v (defaults otherwise;
    'rock' also reads results/rock_walk_<L|F>.json if present). Same call signature as Policy; actions are radians from
    the stance pose (action_scale 1). Each ramps itself in, so run them with --ramp 0."""

    def __init__(self, name, params=None):
        sys.path.insert(0, str(ROOT))
        from slender_biped.rock_walk import CONTROLLERS
        kind, full = name.removesuffix("_full"), name.endswith("_full")
        self.path = ROOT / "slender_biped" / "rock_walk.py"
        p = {}
        tuned = ROOT / "results" / f"rock_walk_{'F' if full else 'L'}.json"
        if kind == "rock" and tuned.exists():
            p = json.loads(tuned.read_text())["params"]
            self.path = tuned
        if kind == "refwalk":
            tab = CONTROLLERS[kind].TABLES[int((params or {}).get("table", 0))]
            tuned = ROOT / "results" / f"refwalk_{tab}.json"
            if tuned.exists():
                p = json.loads(tuned.read_text())["params"]
                self.path = tuned
        p.update(params or {})
        self.ctl = CONTROLLERS[kind](p)
        self.env = dict(self.ctl.env, model="hw_pad_board150w_8dof" if full else "hw_pad_board150w_legs_8dof")
        if getattr(self.ctl, "MODEL", None):           # refwalk: the 10-joint legs with ankle roll
            self.env["model"] = self.ctl.MODEL

    def __call__(self, obs):
        return self.ctl(obs)


class Balancer:
    """Non-learning stand controller (scripts/pd_balance.py): stance hold + IMU tilt feedback on the hips + a slow
    self-trim. Same call signature as Policy (35-dim observation in, action in policy units out)."""

    def __init__(self, key="volt_0.68_hw", ki_p=-0.5):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_balance
        self.pdb = pd_balance
        self.gains = dict(json.loads((ROOT / "results/pd_balance.json").read_text())[key]["gains"], ki_p=ki_p)
        self.state = {}
        self.path = ROOT / "results/pd_balance.json"
        self.env = json.loads((ROOT / "runs" / DEFAULT_RUN / "config.json").read_text())["env"]

    def __call__(self, obs):
        return self.pdb.controller(obs, self.gains, self.state, self.env["ctrl_dt"]) / self.env["action_scale"]


# ------------------------------------------------------------------ IMU
class Imu:
    """Background reader of the RP2040 stream (hardware/imu/imu_stream.py), mapped into the pelvis frame."""

    def __init__(self, port, R=None):
        self.ser = serial.Serial(port, 115200, timeout=0.05)
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.g_s = self.w_s = None
        self.t = 0.0
        self.seq = None
        self.gaps = 0
        self.i2c_errors = 0                          # firmware's count of retried I2C transfers (W lines)
        self.msgs = []
        self._stop = False
        self._buf = b""
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        if not self.wait(1.0):
            # not streaming: another script may still be running (e.g. imu_live.py left behind by mpremote), or the
            # board sits at the REPL. Interrupt it, leave raw REPL, then soft-reboot so main.py starts.
            self.ser.write(b"\r\x03\x03")
            time.sleep(0.3)
            self.ser.write(b"\x02")
            time.sleep(0.2)
            self.ser.write(b"\x04")
            if not self.wait(4.0):
                errs = [m for m in self.msgs if m.startswith("E ")]
                if errs:
                    sys.exit(f"IMU board is running but the sensor doesn't answer:\n  {errs[-1][2:]}")
                sys.exit(f"IMU on {port} is not streaming. Install the firmware:\n  mpremote connect {port} cp "
                         f"hardware/imu/imu_stream.py :main.py + reset")

    def wait(self, s):
        t0 = time.time()
        while time.time() - t0 < s:
            if self.g_s is not None:
                return True
            time.sleep(0.02)
        return False

    def _loop(self):
        while not self._stop:
            try:                                         # whatever has arrived, as soon as it arrives (low latency)
                self._buf += self.ser.read(max(1, self.ser.in_waiting))
            except serial.SerialException:
                return
            *lines, self._buf = self._buf.split(b"\n")
            for ln in lines:
                p = ln.decode(errors="ignore").split()
                if len(p) == 8 and p[0] == "I":
                    try:
                        v = [float(x) for x in p[1:7]]
                        seq = int(p[7])
                    except ValueError:
                        continue
                    if self.seq is not None and seq != self.seq + 1:
                        self.gaps += 1
                    self.seq = seq
                    self.g_s, self.w_s, self.t = np.array(v[:3]), np.array(v[3:]), time.time()
                elif p and p[0] in ("B", "E"):
                    self.msgs.append(" ".join(p))
                elif len(p) == 3 and p[:2] == ["W", "i2c"]:
                    self.i2c_errors = int(p[2])

    def latest(self):
        """(gravity unit vector, angular rate rad/s) in the pelvis frame, and the sample's age in s."""
        return self.R @ self.g_s, self.R @ self.w_s, time.time() - self.t

    def rebias(self):
        """Re-zero the gyro (hold still). Waits for the board's B reply, not a fixed time."""
        n = len(self.msgs)
        self.ser.write(b"b")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            if any(m.startswith("B ") for m in self.msgs[n:]):
                break
            time.sleep(0.02)
        else:
            print("  (IMU gave no gyro re-zero reply within 3 s)")
        t0 = time.time()                             # and the stream is flowing again
        while time.time() - self.t > 0.05 and time.time() - t0 < 1.0:
            time.sleep(0.01)

    def close(self):
        self._stop = True
        self.th.join(0.5)
        self.ser.close()


def load_mount():
    if not MOUNT_FILE.exists():
        sys.exit("no IMU mounting calibration yet: run `python policy_stand.py imu-calib` first")
    return np.array(yaml.safe_load(MOUNT_FILE.read_text())["R_sensor_to_pelvis"], float)


def tilt_deg(g):
    """Pelvis tilt from vertical (deg), and roll/pitch (deg; pitch + = leaning forward, roll + = leaning left)."""
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
    # g = R^T (0, 0, -1): leaning forward (about +y) gives +g_x, leaning left (about -x) gives +g_y
    return tilt, math.degrees(math.atan2(g[1], -g[2])), math.degrees(math.atan2(g[0], -g[2]))


# ------------------------------------------------------------------ servos
class Servos:
    """One sync read (position + speed of all 8) and one sync write (goal positions) per control step."""

    def __init__(self, bus):
        self.bus = bus
        self.joints = bus.cfg["joints"]
        self.names = list(self.joints)
        self.ids = [j["id"] for j in self.joints.values()]
        # position, speed, load, voltage, temperature (registers 56-63) in the same sync read (2026-11-11: voltage and
        # temperature logged per servo, to check for supply sag)
        self.rd = GroupSyncRead(bus.port, bus.pk, fb.ADDR["PRESENT_POS"], 8)
        self.load = np.zeros(len(self.joints)); self.volt = np.zeros(len(self.joints)); self.temp = np.zeros(len(self.joints))
        for i in self.ids:
            self.rd.addParam(i)
        self.set_accel(None)

    def set_accel(self, accel):
        """accel None: goal positions only. Otherwise every write also sends the acceleration register (41) and goal
        speed 0 (max) with each goal, in one packet from 41 to 47, the way Feetech's own WritePosEx does. 2026-11-07:
        with the acceleration written once, a new hip-pitch servo intermittently ran the slow default ramp although
        the register read back correctly."""
        self.accel = accel
        if accel is None:
            self.wr = GroupSyncWrite(self.bus.port, self.bus.pk, fb.ADDR["GOAL_POS"], 2)
        else:
            self.wr = GroupSyncWrite(self.bus.port, self.bus.pk, fb.ADDR["ACCEL"], 7)

    def read(self):
        """Joint angles (rad) and speeds (rad/s) in the policy's joint order and sign convention."""
        for _ in range(3):                              # retries: a reply now and then goes missing or garbles
            res = self.rd.txRxPacket()
            if res == COMM_SUCCESS:
                break
        else:
            raise IOError(f"sync read: {self.bus.pk.getTxRxResult(res)}")
        q, qd = np.zeros(len(self.joints)), np.zeros(len(self.joints))
        for k, (n, j) in enumerate(self.joints.items()):
            pos = self.rd.getData(j["id"], fb.ADDR["PRESENT_POS"], 2)
            spd = self.rd.getData(j["id"], fb.ADDR["PRESENT_SPEED"], 2)
            spd = -(spd & 0x7FFF) if spd & 0x8000 else spd
            q[k] = math.radians(fb.counts_to_deg(j, pos))
            qd[k] = math.radians(j["direction"] * spd / fb.COUNTS_PER_DEG)
            ld = self.rd.getData(j["id"], fb.ADDR["PRESENT_LOAD"], 2)
            self.load[k] = -(ld & 0x3FF) if ld & 0x400 else ld          # sign-magnitude, 0.1 % of max torque
            self.volt[k] = self.rd.getData(j["id"], fb.ADDR["PRESENT_VOLT"], 1) / 10.0
            self.temp[k] = self.rd.getData(j["id"], fb.ADDR["PRESENT_TEMP"], 1)
        return q, qd

    def write(self, q_rad):
        self.wr.clearParam()
        for (n, j), q in zip(self.joints.items(), q_rad):
            c = fb.deg_to_counts(j, math.degrees(q))
            goal = [c & 0xFF, (c >> 8) & 0xFF]
            acc = self.accel if self.accel is None or np.isscalar(self.accel) else int(self.accel[n])
            self.wr.addParam(j["id"], goal if acc is None else [acc] + goal + [0, 0, 0, 0])
        res = self.wr.txPacket()
        if res != COMM_SUCCESS:
            raise IOError(f"sync write: {self.bus.pk.getTxRxResult(res)}")

    def limits_rad(self):
        lo = np.radians([j["min_deg"] for j in self.joints.values()])
        hi = np.radians([j["max_deg"] for j in self.joints.values()])
        return lo, hi

    def torque(self, on):
        for i in self.ids:
            try:
                self.bus.torque(i, on)
            except IOError as e:
                print("  ", e)


# ------------------------------------------------------------------ keyboard
class Keys:
    """Non-blocking single keypresses (cbreak mode); restores the terminal on exit."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def get(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def wait(self, valid):
        while True:
            k = sys.stdin.read(1)
            if k in valid:
                return k

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ------------------------------------------------------------------ commands
def cmd_ports(a):
    servo, imu = find_ports()
    print(f"servo bus: {servo or 'NOT FOUND (attach the CH343 adapter with usbipd)'}")
    print(f"IMU:       {imu or 'NOT FOUND (attach the RP2040 with usbipd)'}")


def imu_port(a):
    p = a.imu_port or find_ports()[1]
    if not p:
        sys.exit("IMU (RP2040) not found: attach it with usbipd, or pass --imu-port")
    return p


def mean_gravity(imu, s=1.0):
    vs, t0 = [], time.time()
    while time.time() - t0 < s:
        vs.append(imu.g_s.copy())
        time.sleep(0.01)
    g = np.mean(vs, 0)
    return g / np.linalg.norm(g)


def axis_from_pair(g_up, g_tilt, axis):
    """One pelvis axis (sensor coordinates) from two gravity readings taken with the plate level along one direction.
    roll: plate level SIDEWAYS in both readings, the second leaning FORWARD -> both lie in the pelvis x-z plane, whose
    normal is the pelvis y axis (left). pitch: plate level FORE-AFT in both, the second leaning LEFT -> both lie in the
    y-z plane, whose normal is the pelvis x axis (forward). The upright reading need not be level the other way."""
    g1, g2 = g_up / np.linalg.norm(g_up), g_tilt / np.linalg.norm(g_tilt)
    n = np.cross(-g1, g2) if axis == "roll" else np.cross(g1, g2)     # roll: ey = z x x; pitch: ex = y x z
    return n / np.linalg.norm(n)


def mount_from_axes(ex, ey):
    """The nearest rotation whose x / y rows are ex / ey (they are measured separately, so only nearly orthogonal)."""
    M = np.vstack([ex, ey, np.cross(ex, ey) / np.linalg.norm(np.cross(ex, ey))])
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def cmd_imu_calib(a):
    old = np.array(yaml.safe_load(MOUNT_FILE.read_text())["R_sensor_to_pelvis"], float) if MOUNT_FILE.exists() else None
    if a.axis != "both" and old is None:
        sys.exit(f"no {MOUNT_FILE.name} yet to keep the other axis from: run `imu-calib` (both axes) first")
    imu = Imu(imu_port(a))
    print("IMU mounting calibration (sensor axes -> pelvis axes: x forward, y left, z up), against the HIP PLATE: the\n"
          "simulator's pelvis frame is the plate, level when both boards are flat. Nothing moves. Each axis needs a\n"
          "spirit level (or a phone level app) along ONE direction only, on the plate or a face parallel to it.\n")
    steps = [("roll", "SIDEWAYS (level across the plate, left to right)", "FORWARD (chest toward the floor)"),
             ("pitch", "FORE-AFT (level along the plate, front to back)", "to its LEFT (left side toward the floor)")]
    got, info = {}, {}
    for axis, level, lean in steps:
        if a.axis not in ("both", axis):
            continue
        print(f"--- {axis.upper()} zero: the plate level {level}; the other direction doesn't matter.")
        while True:
            input(f"  a. Robot upright, plate level {level.split(' (')[0]}. Hold still, Enter... ")
            g1 = mean_gravity(imu)
            input(f"  b. Keep it level {level.split(' (')[0]} and tilt the whole robot {lean} by 20-30 deg. Hold, Enter... ")
            g2 = mean_gravity(imu)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(g1 @ g2)))))
            if ang >= 10:
                break
            print(f"  only {ang:.1f} deg between the two readings: tilt further (20-30 deg). Again.")
        got[axis] = axis_from_pair(g1, g2, axis)
        info[f"{axis}_tilt_used_deg"] = round(ang, 1)
        info[f"{axis}_readings"] = [[round(float(x), 4) for x in g] for g in (g1, g2)]
    ex = got.get("pitch", old[0] if old is not None else None)
    ey = got.get("roll", old[1] if old is not None else None)
    skew = math.degrees(math.asin(max(-1.0, min(1.0, float(ex @ ey)))))
    R = mount_from_axes(ex, ey)
    if abs(skew) > 3:
        print(f"\nWARNING: the forward and left axes are {90 - skew:.1f} deg apart, not 90: a level was misread, the robot\n"
              f"moved, or (with --axis) the IMU moved since the saved axis. Redo it (both axes if unsure).")
    MOUNT_FILE.write_text(
        "# IMU mounting: v_pelvis = R_sensor_to_pelvis @ v_sensor. Written by `policy_stand.py imu-calib` "
        f"{time.strftime('%Y-%m-%d %H:%M')}.\n# Zero = the hip plate level (roll and pitch levelled separately). "
        "Redo it whenever the IMU board is moved.\n"
        + yaml.safe_dump({"R_sensor_to_pelvis": [[round(float(x), 5) for x in r] for r in R],
                          "axes_measured": a.axis, "axes_skew_deg": round(skew, 2), **info}, sort_keys=False))
    print(f"\nSaved {MOUNT_FILE.name} ({a.axis}; axes {90 - skew:.1f} deg apart):")
    for r, ax in zip(R, "xyz"):
        print(f"  pelvis {ax} = sensor ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})")
    if old is not None:
        _, r0, p0 = tilt_deg(old @ (R.T @ np.array([0.0, 0.0, -1.0])))
        print(f"vs the previous calibration: the plate level now reads roll {r0:+.1f}, pitch {p0:+.1f} deg in the old one")
    print("Check it with `python policy_stand.py imu-check` (plate level -> roll and pitch ~0).")
    imu.close()


def cmd_imu_check(a):
    imu = Imu(imu_port(a), load_mount())
    print("pelvis frame. Expect: upright ~ g (0, 0, -1); lean forward -> pitch +; lean left -> roll +;\n"
          "turn left (counter-clockwise from above) -> wz +.   Ctrl-C to stop.")
    try:
        while True:
            g, w, age = imu.latest()
            t, r, p = tilt_deg(g)
            print(f"\rg ({g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f})  w ({w[0]:+.2f} {w[1]:+.2f} {w[2]:+.2f}) rad/s  "
                  f"tilt {t:5.1f}  roll {r:+6.1f}  pitch {p:+6.1f} deg  age {age * 1000:3.0f} ms ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    imu.close()


def cmd_timing(a):
    bus = open_bus(a.servo_port)
    sv = Servos(bus)
    imu_p = a.imu_port or find_ports()[1]
    imu = Imu(imu_p) if imu_p else None
    t_rd, t_wr = [], []
    q, _ = sv.read()
    for _ in range(200):
        t0 = time.perf_counter()
        q, _ = sv.read()
        t1 = time.perf_counter()
        t_rd.append(t1 - t0)
        if a.write:                                    # torque stays off: rewriting the present angles moves nothing
            sv.write(q)
            t_wr.append(time.perf_counter() - t1)
    pct = lambda v, p: 1000 * float(np.percentile(v, p))  # noqa: E731
    print(f"sync read  (8 servos): median {pct(t_rd, 50):.1f} ms, 95th {pct(t_rd, 95):.1f} ms, max {pct(t_rd, 100):.1f} ms")
    if t_wr:
        print(f"sync write (8 servos): median {pct(t_wr, 50):.1f} ms, 95th {pct(t_wr, 95):.1f} ms")
    budget = 20.0 - pct(t_rd, 95) - (pct(t_wr, 95) if t_wr else 1.0)
    print(f"50 Hz step budget left after bus I/O (95th pct): {budget:.1f} ms "
          + ("OK" if budget > 3 else "TOO TIGHT: the loop will overrun"))
    if imu:
        s0, t0 = imu.seq, time.time()
        time.sleep(2.0)
        print(f"IMU: {(imu.seq - s0) / (time.time() - t0):.0f} samples/s, {imu.gaps} gaps")
        imu.close()


ACCEL_HELP = ("servo acceleration register while the policy/test runs, 0-254 (x100 counts/s^2; 254 ~ 2230 deg/s^2). "
              "Step tests 2026-11-06: 0 gives no ramp on the new left knee but a slow 180-530 deg/s^2 default ramp "
              "on the seven older servos, 0.2 s per 5 deg correction")
P_DEFAULT = 32                                      # factory position P gain (EEPROM 21) on every servo here
SERVO_GAIN_REGS = {"P": 21, "D": 22, "I": 23, "min_force": 24, "dead_cw": 26, "dead_ccw": 27}   # STS3215 EEPROM


def set_p(bus, sid, p):
    """Write the position P gain (EEPROM 21): unlock, write, re-lock, verify. The servo can drop its reply while it
    saves to EEPROM, so failures are checked by reading back (same pattern as feetech_bench zero)."""
    if not 1 <= p <= 254:
        sys.exit(f"P gain {p} out of range 1..254")
    for addr, val in ((fb.ADDR["LOCK"], 0), (SERVO_GAIN_REGS["P"], p), (fb.ADDR["LOCK"], 1)):
        try:
            bus.w1(sid, addr, val)
        except IOError:
            pass
        time.sleep(0.05)
    got = None
    for _ in range(5):                               # the first read right after an EEPROM save can come back garbled
        time.sleep(0.05)
        try:
            got = bus.r1(sid, SERVO_GAIN_REGS["P"])
        except IOError:
            continue
        if got == p:
            return
    sys.exit(f"id {sid}: P gain reads {got} after writing {p}")


def _step_metrics(t, q, q0, q1):
    """Delay to 10 %, time to 90 %, overshoot and final error for one step q0 -> q1 (deg)."""
    s = q1 - q0
    x = (q - q0) / s                                   # 0 -> 1 progress along the step
    t10 = next((ti for ti, xi in zip(t, x) if xi >= 0.1), float("nan"))
    t90 = next((ti for ti, xi in zip(t, x) if xi >= 0.9), float("nan"))
    tail = q[t >= t[-1] - 0.15]
    return dict(t10=t10, t90=t90, over=max(0.0, (np.max(x) - 1.0)) * 100, err=float(np.mean(tail) - q1))


def set_motion_regs(bus, ids, accel, names):
    """Write the acceleration and goal-speed registers on every servo and read them back. 2026-11-07: a new
    hip-pitch servo was intermittently left on the slow default ramp (~400 deg/s^2) in step tests, with the other
    one fast, swapping between runs; this shows whether the register write is what fails."""
    bad = []
    for i, n in zip(ids, names):
        for _ in range(3):
            try:
                bus.w1(i, fb.ADDR["ACCEL"], accel)
                bus.w2(i, fb.ADDR["GOAL_SPEED"], 0)
                if bus.r1(i, fb.ADDR["ACCEL"]) == accel and bus.r2(i, fb.ADDR["GOAL_SPEED"]) == 0:
                    break
            except IOError:
                pass
        else:
            bad.append(n)
    print(f"acceleration {accel} read back on all {len(ids)} servos" if not bad else
          f"WARNING: acceleration/speed registers did not read back on {bad}")
    return bad


def cmd_step_test(a):
    if not a.enable:
        sys.exit("REFUSED: the joints move; pass --enable (hand on the power switch)")
    bus = open_bus(a.servo_port)
    sv = Servos(bus)
    names = sv.names
    joints = a.joints or names
    bad = [j for j in joints if j not in names]
    if bad:
        sys.exit(f"unknown joints {bad}; choose from {names}")
    missing = [n for n, j in sv.joints.items() if bus.ping(j["id"]) is None]
    if missing:
        sys.exit(f"no answer from {missing}")
    q_nom = stance_rad(sv.names)
    lo, hi = sv.limits_rad()
    p_orig = {}
    off_default = [n for n, j in sv.joints.items() if bus.r1(j["id"], SERVO_GAIN_REGS["P"]) != P_DEFAULT]
    if off_default:
        print(f"WARNING: P gain is not {P_DEFAULT} on {off_default} (left over from an interrupted test?). "
              f"Reset with `python policy_stand.py reset-gains`.")
    log_path = Path(a.log or f"steptest_{a.mode}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    rows, results = [], []
    try:                                             # everything from the first EEPROM write on is undone in finally
        if a.p_gain is not None:                     # EEPROM register: set for the tested joints, restored below
            for jn in joints:
                sid = sv.joints[jn]["id"]
                p_orig[sid] = bus.r1(sid, SERVO_GAIN_REGS["P"])
                set_p(bus, sid, a.p_gain)
            print(f"P gain set to {a.p_gain} on {joints} for this test (originals {sorted(set(p_orig.values()))} "
                  "are restored at the end)")
        print("servo gain registers (STS3215 map):")
        for n, j in sv.joints.items():
            print(f"  {n:12s} " + "  ".join(f"{k} {bus.r1(j['id'], r)}" for k, r in SERVO_GAIN_REGS.items()))
        print(f"\nMoving to the stance pose. Mode '{a.mode}': " + ("robot held UP in the air, legs hanging free."
              if a.mode == "air" else "robot STANDING on its pads; steady it lightly by the pelvis."))
        for n, j in sv.joints.items():
            fb.goto(bus, n, j, math.degrees(q_nom[names.index(n)]))
        time.sleep(3.0)
        print(f"servo acceleration register for the test: {a.accel} ({a.accel * 100 / fb.COUNTS_PER_DEG:.0f} "
              "deg/s^2; 0 = firmware-dependent: no ramp on the new knee, a slow default ramp on the older servos)")
        set_motion_regs(bus, sv.ids, a.accel, sv.names)   # same settings as a policy run
        sv.set_accel(a.accel)
        for jn in joints:
            k = names.index(jn)
            amp = math.radians(a.amp)
            seq = [amp, 0.0, -amp, 0.0]
            for d in seq:
                tgt = q_nom.copy()
                tgt[k] = np.clip(q_nom[k] + d, lo[k], hi[k])
                q_before = sv.read()[0][k]
                sv.write(tgt)
                t0 = time.perf_counter()
                ts, qs = [], []
                while time.perf_counter() - t0 < a.hold:
                    q, _ = sv.read()
                    ts.append(time.perf_counter() - t0)
                    qs.append(q[k])
                    rows.append([jn, round(math.degrees(tgt[k] - q_nom[k]), 2), round(ts[-1], 4)]
                                + [round(math.degrees(v), 3) for v in q])
                ts, qs = np.array(ts), np.degrees(np.array(qs))
                q0, q1 = math.degrees(q_before), math.degrees(tgt[k])
                if abs(q1 - q0) > 1.0:
                    m = _step_metrics(ts, qs, q0, q1)
                    m.update(joint=jn, step=q1 - q0)
                    results.append(m)
            print(f"  {jn}: done")
    except KeyboardInterrupt:
        print("\nCtrl-C")
    finally:
        try:
            sv.set_accel(None)                       # back to goal-only writes: keep the gentle bench settings
            for i in sv.ids:
                bus.w1(i, fb.ADDR["ACCEL"], bus.cfg["acceleration"])
                bus.w2(i, fb.ADDR["GOAL_SPEED"], bus.cfg["speed_counts_per_s"])
            sv.write(q_nom)
        except IOError:
            sv.torque(False)
        for sid, p in p_orig.items():
            set_p(bus, sid, p)
        if p_orig:
            print(f"P gain restored to {sorted(set(p_orig.values()))}: now "
                  f"{[bus.r1(s, SERVO_GAIN_REGS['P']) for s in p_orig]}")
    with log_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "target_offset_deg", "t"] + [f"q_{n}_deg" for n in names])
        w.writerows(rows)
    print(f"\nstep response, mode '{a.mode}', {a.amp:g} deg steps (median over the steps of each joint):")
    print(f"  {'joint':12s} {'delay 10%':>10s} {'time 90%':>9s} {'overshoot':>9s} {'final err':>9s}")
    for jn in joints:
        r = [m for m in results if m["joint"] == jn]
        if not r:
            continue
        med = lambda key: float(np.nanmedian([m[key] for m in r]))  # noqa: E731
        never = sum(np.isnan(m["t90"]) for m in r)
        print(f"  {jn:12s} {med('t10') * 1000:8.0f}ms {med('t90') * 1000:7.0f}ms {med('over'):8.0f}% "
              f"{med('err'):+8.2f}°" + (f"   ({never}/{len(r)} steps never reached 90%)" if never else ""))
    print("simulator, same protocol, STANDING, 5 deg (the model the policy was trained on):\n"
          "  hip_yaw  delay 16 ms, 90% 124 ms | hip_roll delay 21 ms, 90% >350 ms (load) |\n"
          "  hip_pitch delay 26 ms, 90% >350 ms (load) | knee delay 14 ms, 90% 104 ms")
    print(f"raw samples: {log_path}. Robot is holding the stance pose: `python biped_bench.py off` to release.")


def cmd_reset_gains(a):
    """Put the position P gain back to the factory 32 on every servo (no motion)."""
    bus = open_bus(a.servo_port)
    for n, j in bus.cfg["joints"].items():
        p = bus.r1(j["id"], SERVO_GAIN_REGS["P"])
        if p != P_DEFAULT:
            set_p(bus, j["id"], P_DEFAULT)
            print(f"  {n}: P {p} -> {P_DEFAULT}")
    print("P gains:", {n: bus.r1(j["id"], SERVO_GAIN_REGS["P"]) for n, j in bus.cfg["joints"].items()})


def stance_rad(names=None):
    """The stance pose (hip pitch 16, knee 32, everything else 0) for the given joint names (default: the 8-joint legs)."""
    names = names or [f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee")]
    return np.radians([16.0 if n.endswith("hip_pitch") else 32.0 if n.endswith("knee") else 0.0 for n in names])


def kv_float(s):
    k, v = s.split("=")
    return k, float(v)


def cmd_run(a):
    if not (a.enable or a.dry_run):
        sys.exit("REFUSED: pass --dry-run (servos limp) or --enable (motion; keep a hand on the power switch)")
    pol = (Balancer() if a.run == "balancer" else Rocker(a.run, dict(kv_float(x) for x in a.det))
           if a.run.removesuffix("_full") in DET_NAMES else Policy(a.run, a.params))
    env = pol.env
    dt, hz, scale = env["ctrl_dt"], env["gait_hz"], env["action_scale"]
    meta = json.loads((ROOT / "models" / f"{env['model']}.json").read_text())
    q_nom = np.array(meta["nominal_q"], float)
    legs8 = [f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee")]
    legs10 = [f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_roll")]
    if meta["joints"] not in (legs8, legs10):
        sys.exit(f"unexpected joint order in {env['model']}: {meta['joints']}")
    nj = len(meta["joints"])
    n_hist = int(env.get("obs_hist", 0) or 0)            # round 7: past frames [g, gyro, q, 0.1 qd, action] in front
    if isinstance(pol, Policy) and pol.n_in != (6 + 3 * nj) * (n_hist + 1) + 5:
        sys.exit(f"{a.run}: policy takes {pol.n_in} inputs; this runner builds the {(6 + 3 * nj) * (n_hist + 1) + 5}-input "
                 f"observation for {nj} joints and {n_hist} history frames only (no velocity / line inputs)")
    if n_hist:
        print(f"observation history: {n_hist} past frames ({n_hist * dt * 1000:.0f} ms)")
    R = load_mount()
    if a.pitch_trim:              # rotate the pelvis frame about y so the reported pitch drops by pitch_trim
        c, s = math.cos(math.radians(a.pitch_trim)), math.sin(math.radians(a.pitch_trim))
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ R
        print(f"pitch trim {a.pitch_trim:+.1f} deg: the policy sees pitch reduced by that much")
    if a.roll_trim:               # rotate the pelvis frame about x so the reported roll drops by roll_trim
        c, s = math.cos(math.radians(a.roll_trim)), math.sin(math.radians(a.roll_trim))
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]]) @ R
        print(f"roll trim {a.roll_trim:+.1f} deg: the policy sees roll (+ = leaning left) reduced by that much")
    bus = open_bus(a.servo_port)
    sv = Servos(bus)
    if list(sv.joints) != meta["joints"]:
        sys.exit(f"bench joint order {list(sv.joints)} != policy joint order {meta['joints']}"
                 + ("\n  (10-joint policies need the ankle_roll entries in bench_config*.yaml; 8-joint policies need them "
                    "removed or commented out)" if len(sv.joints) != nj else ""))
    missing = [n for n, j in sv.joints.items() if bus.ping(j["id"]) is None]
    if missing:
        sys.exit(f"no answer from {missing}: connect both legs (and power)")
    imu = Imu(imu_port(a), R)
    lo, hi = sv.limits_rad()
    roll_idx = [sv.names.index("L_hip_roll"), sv.names.index("R_hip_roll")]
    both_in = math.radians(bb.BOTH_ADDUCT_LIMIT)
    if isinstance(pol, Balancer):
        print(f"non-learning balancer, gains {pol.gains}")
    if isinstance(pol, Rocker):
        print(f"deterministic controller {a.run}, parameters {pol.ctl.p}")
        if a.ramp is None or a.ramp > 0:
            print("  (it ramps itself in; --ramp 0 is enough)")
    print(f"policy {pol.path.relative_to(ROOT)} ({env['model']}, task {env['task']}), {1 / dt:.0f} Hz, "
          f"action scale {scale} rad, gait clock {hz} Hz")
    g, _, _ = imu.latest()
    print(f"IMU tilt now {tilt_deg(g)[0]:.1f} deg. Tilt cut-out at {a.max_tilt:.0f} deg.")
    # walking / skiing policies (2026-11-08): a forward-speed command, heading held on the integrated gyro yaw rate as
    # in training (yaw-rate command = clip(gain x heading error, +-0.3)), and the reference gait as the action base
    ref = None
    if env.get("ref_table") and env.get("ref_residual"):
        z = np.load(ROOT / "models" / "ref" / f"{env['ref_table']}.npz")
        ref = {k: np.asarray(z[k]) for k in ("q", "v")}
        print(f"reference gait {env['ref_table']} (residual), forward command {a.vx:.2f} m/s")
    elif a.vx:
        print(f"forward command {a.vx:.2f} m/s")
    cmd_v = np.array([a.vx, 0.0, 0.0])
    # step-and-settle walkers (2026-11-08) were trained starting in the reference's zero-speed "ready" pose with the
    # forward command ramping up: blend into that pose first (clock stopped, no policy action), then ramp vx
    slew = float(env.get("cmd_vx_slew", 0.0))
    ready = a.ready if a.ready is not None else (2.0 if env.get("reset_ref_pose") and ref is not None else 0.0)
    q_ready = ref_lookup(ref, 0.0, 0.0) if (ready > 0 and ref is not None) else q_nom
    # 2026-11-11: robot logs: with the 1 s blend and the clock running, the first right step (inside the blend, its weight
    # shift scaled down) landed inward (boards 81-88 mm apart) and the first left lift-off came with the pelvis 52-62 mm
    # inside the standing board: most of the day's near-falls 1.9 s after engaging. Ready-pose policies trained fully
    # engaged from the first step: no blend by default; any blend holds the clock until it ends.
    if a.ramp is None:
        a.ramp = 0.0 if ready > 0 else 1.0
    print(f"policy blend-in {a.ramp:.1f} s" + (" (gait clock held until it ends)" if a.ramp > 0 else ""))
    if a.gyro_lpf is None:
        a.gyro_lpf = float(env.get("gyro_lpf", 0.0) or 0.0)
        if a.gyro_lpf > 0:
            print(f"gyro low-pass {a.gyro_lpf:.1f} Hz on the policy's input (as trained)")
    elif a.gyro_lpf > 0:
        print(f"gyro low-pass {a.gyro_lpf:.1f} Hz on the policy's input (override)")
    if a.ready_act and ready > 0:
        print("ready blend goes to the pose the policy holds standing (gait start pose + its action, clock at 0)")
    if a.gyro_scale != 1.0:
        print(f"gyro x {a.gyro_scale:.2f} on the policy's input (not as trained)")
    lpf = a.act_lpf if a.act_lpf is not None else float(env.get("act_lpf", 0.0))   # action low-pass, as trained
    if lpf > 0:
        print(f"action low-pass {lpf:.2f}" + (" (override)" if a.act_lpf is not None else " (from the run's config)"))
    if ready > 0:
        print(f"ready pose first: {ready:.1f} s blend to the gait's start pose, then the policy; forward command "
              f"ramps at {slew:.2f} m/s^2" if slew else f"ready pose first: {ready:.1f} s")
    if slew > 0:
        cmd_v[0] = 0.0
    gate_u = tuple(env.get("settle_gate_u", ()))
    gate_w = float(env.get("settle_gate_w", 0.3))
    gate_n = int(float(env.get("settle_gate_max", 0.5)) / dt)
    gate_held = 0
    if gate_u:
        print(f"settle gate: the next step waits until the body turns slower than {gate_w:.2f} rad/s "
              f"(at most {gate_n * dt:.1f} s)")
    log_path = Path(a.log or f"stand_{a.run}_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    with Keys() as keys:
        if a.enable:
            print("\nMoving to the stance pose (hip 16, knee 32" + (", ankles 0" if nj == 10 else "") + ")...")
            try:
                for n, j in sv.joints.items():
                    fb.goto(bus, n, j, math.degrees(q_nom[sv.names.index(n)]))
                t0 = time.time()
                while time.time() - t0 < 5.0:
                    q, _ = sv.read()
                    if np.max(np.abs(q - q_nom)) < math.radians(2):
                        break
                    time.sleep(0.1)
                else:
                    print(f"  (not all joints reached the stance: worst {np.degrees(np.max(np.abs(q - q_nom))):.1f} deg off)")
            except KeyboardInterrupt:
                sv.torque(False)
                sys.exit("\nCtrl-C: torque off")
            # the fitted servo model assumes the fastest ramp the register allows (0 = a slow default on some units)
            set_motion_regs(bus, sv.ids, a.accel, sv.names)
            if a.accel_match:                       # 2026-11-11: each servo's ramp as in the run's training model
                pa = list(env.get("prof_acc", [0.0, 0.0])) + [None]
                acc_j = {}
                for n in sv.names:
                    r = pa[1] if n.endswith("knee") else (pa[2] if n.endswith("hip_pitch") and pa[2] else pa[0])
                    acc_j[n] = int(np.clip(round(math.degrees(r) / 8.79), 1, 254)) if r else a.accel
                print("acceleration register per joint (training model):",
                      " ".join(f"{n} {v}" for n, v in acc_j.items()))
                sv.set_accel(acc_j)
            else:
                sv.set_accel(a.accel)
            imu.rebias()                                # re-zero the gyro while it stands still
            print("\nRobot on its pads, standing? Keep a hand near it.\n"
                  "  Enter = ENGAGE the policy      q = abort (holds the stance pose)")
            if keys.wait("\nq") == "q":
                print("aborted: holding the stance pose (`python biped_bench.py home --enable` or `off` to release)")
                return
        else:
            sv.torque(False)
            print("\nDRY RUN: servos limp, nothing is sent. Hold the robot, tilt it, move the legs by hand and watch\n"
                  "the policy's targets. Any key stops.")

        print("ENGAGED. Enter / q / space = stop.\n" if a.enable else "")
        last_a = np.zeros(nj)
        w_al = 1.0 - math.exp(-2 * math.pi * a.gyro_lpf * dt) if a.gyro_lpf > 0 else 1.0
        w_f = None                                      # the gyro the policy sees (--gyro-lpf)
        hist = None                                     # past observation frames (n_hist)
        a_est = np.zeros(nj)                            # --ready-act: the policy's standing action, smoothed
        phase = 0.0
        yaw = 0.0                                       # heading change since engage (gyro yaw rate, integrated)
        stop_reason, overruns, k, stale = "user", 0, 0, 0
        bad_reads, read_fails, q_last, qd_last = 0, 0, None, None
        loop_ms = []
        t_next = time.perf_counter()
        f = log_path.open("w", newline="")
        w = csv.writer(f)
        w.writerow(["t", "loop_ms"] + [f"g{c}" for c in "xyz"] + [f"w{c}" for c in "xyz"]
                   + [f"q_{n}" for n in sv.names] + [f"qd_{n}" for n in sv.names]
                   + [f"a_{n}" for n in sv.names] + [f"cmd_{n}" for n in sv.names]
                   + [f"volt_{n}" for n in sv.names] + [f"temp_{n}" for n in sv.names] + [f"load_{n}" for n in sv.names])
        v_min, v_start = 99.0, None
        t_start = time.perf_counter()
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    q, qd = sv.read()
                    bad_reads = 0
                except IOError:
                    bad_reads += 1                      # 2026-11-11: a garbled sync read stopped a run after 2.4 s;
                    read_fails += 1                     # run on the last reading for up to 5 steps (0.1 s)
                    if bad_reads > 5 or k == 0:
                        raise
                    q, qd = q_last, qd_last
                q_last, qd_last = q, qd
                g, wv, age = imu.latest()
                if age > 0.03:
                    stale += 1                          # run on the last sample through a short dropout
                if age > a.imu_timeout:
                    stop_reason = f"IMU silent for {age * 1000:.0f} ms"
                    break
                tilt = tilt_deg(g)[0]
                if tilt > a.max_tilt:
                    stop_reason = f"tilt {tilt:.0f} deg > {a.max_tilt:.0f}"
                    break
                t_run = t0 - t_start
                if t_run < ready:                       # blend into the ready pose, nothing else
                    r = 0.5 - 0.5 * math.cos(math.pi * t_run / ready)
                    act = np.zeros(nj)
                    q_goal = q_ready
                    if a.ready_act and isinstance(pol, Policy):    # ... plus the action the policy holds there
                        w_f = wv.copy() if w_f is None else w_f + w_al * (wv - w_f)
                        frame = np.concatenate([g, a.gyro_scale * w_f, q - q_nom, 0.1 * qd, a_est])
                        if n_hist and hist is None:
                            hist = np.tile(frame, (n_hist, 1))
                        obs = np.concatenate(([hist.ravel()] if n_hist else []) + [frame, [0.0, 0.0, 0.0], [0.0, 1.0]])
                        if n_hist:
                            hist = np.vstack([hist[1:], frame])
                        a_est = a_est + 0.1 * (np.clip(pol(obs), -1.0, 1.0) - a_est)
                        act = a_est.copy()
                        q_goal = q_ready + r * scale * a_est
                    q_cmd = np.clip(q_nom + r * (q_goal - q_nom), lo, hi)
                else:
                    if slew > 0:
                        cmd_v[0] += float(np.clip(a.vx - cmd_v[0], -slew * dt, slew * dt))
                    if a.vx:
                        cmd_v[2] = float(np.clip(-a.heading_gain * yaw, -0.3, 0.3))
                    w_f = wv.copy() if w_f is None else w_f + w_al * (wv - w_f)
                    frame = np.concatenate([g, a.gyro_scale * w_f, q - q_nom, 0.1 * qd, last_a])
                    if n_hist and hist is None:
                        hist = np.tile(frame, (n_hist, 1))
                    obs = np.concatenate(([hist.ravel()] if n_hist else []) + [frame, cmd_v, [math.sin(phase), math.cos(phase)]])
                    if n_hist:
                        hist = np.vstack([hist[1:], frame])
                    act = np.clip(pol(obs), -1.0, 1.0)
                    if lpf > 0:
                        act = lpf * last_a + (1.0 - lpf) * act
                    ramp = min(1.0, (t_run - ready) / a.ramp) if a.ramp > 0 else 1.0
                    q_base = q_nom
                    if ref is not None:
                        q_base = q_ready + ramp * (ref_lookup(ref, cmd_v[0], phase + 2 * math.pi * hz * dt) - q_ready)
                    q_cmd = np.clip(q_base + ramp * scale * act, lo, hi)
                if q_cmd[roll_idx[0]] < 0 and q_cmd[roll_idx[1]] < 0:       # pelvis rule: legs collide if both
                    q_cmd[roll_idx] = np.maximum(q_cmd[roll_idx], both_in)  # swing inward by more than ~5 deg
                if a.enable:
                    sv.write(q_cmd)
                last_a = act
                if t_run >= ready + a.ramp:              # the clock starts once the policy is fully engaged
                    ph_new = (phase + 2 * math.pi * hz * dt) % (2 * math.pi)
                    if gate_u:                          # settle gate (as trained): hold the clock at a lift-off
                        u0, u1 = phase / (2 * math.pi), ph_new / (2 * math.pi)   # while the body still rotates
                        cross = any(((u0 - gu) % 1.0) > ((u1 - gu) % 1.0) for gu in gate_u)
                        hold = (cross and a.vx and float(wv[0] ** 2 + wv[1] ** 2) > gate_w ** 2 and gate_held < gate_n)
                        gate_held = gate_held + 1 if hold else 0
                        if hold:
                            ph_new = phase
                    phase = ph_new
                    yaw += float(-(wv @ g)) * dt              # yaw rate about the vertical (g points down)
                el = (time.perf_counter() - t0) * 1000
                loop_ms.append(el)
                w.writerow([round(t0 - t_start, 4), round(el, 2)] + list(np.round(g, 4)) + list(np.round(wv, 4))
                           + list(np.round(q, 4)) + list(np.round(qd, 3)) + list(np.round(act, 4))
                           + list(np.round(q_cmd, 4)) + list(sv.volt) + list(sv.temp.astype(int))
                           + list(sv.load.astype(int)))
                v_min = min(v_min, float(sv.volt.min()))
                if v_start is None:
                    v_start = float(np.median(sv.volt))
                if k % 10 == 0:
                    d = np.degrees(q_cmd - q_nom)
                    h = nj // 2
                    print(f"\r{t0 - t_start:6.1f} s  tilt {tilt:4.1f}  |a| max {np.max(np.abs(act)):.2f}  target-stance"
                          f" L[{''.join(f'{x:+4.0f}' for x in d[:h])}] R[{''.join(f'{x:+4.0f}' for x in d[h:])}] deg"
                          f"  loop {el:4.1f} ms  {sv.volt.min():.1f} V ", end="", flush=True)
                k += 1
                if keys.get() is not None:
                    break
                t_next += dt
                wait = t_next - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    overruns += 1
                    t_next = time.perf_counter()
        except KeyboardInterrupt:
            stop_reason = "Ctrl-C"
        except IOError as e:
            stop_reason = f"bus error: {e}"
        finally:
            f.close()
            print()
            if a.enable:
                if stop_reason in ("user", "Ctrl-C") or not a.limp_on_fault:
                    # 2026-11-11: faults (tilt, IMU silence, bus errors) also hold the stance pose with torque on, so
                    # the robot doesn't collapse; --limp-on-fault restores the old torque-off safety stop
                    if stop_reason not in ("user", "Ctrl-C"):
                        print(f"FAULT ({stop_reason}): holding the stance pose, torque ON. Hand on the power switch.")
                    held = False
                    for attempt in range(5):           # the bus may have just failed: retry the hold a few times
                        try:
                            sv.set_accel(None)
                            for i in sv.ids:
                                bus.w1(i, fb.ADDR["ACCEL"], bus.cfg["acceleration"])
                                bus.w2(i, fb.ADDR["GOAL_SPEED"], bus.cfg["speed_counts_per_s"])
                            sv.write(q_nom)
                            held = True
                            break
                        except IOError as e:
                            err = e
                            time.sleep(0.05)
                    else:
                        print(f"could not command the stance after 5 tries ({err}): torque off")
                        sv.torque(False)
                    if held:
                        print("STOPPED: holding the stance pose (torque ON). Release with `python biped_bench.py off` "
                              "(or `home --enable`).")
                else:
                    sv.torque(False)
                    print("SAFETY STOP: torque OFF (the robot goes limp).")
            imu.close()
    n = len(loop_ms)
    secs = n * dt
    if n and v_start is not None:
        print(f"servo supply: {v_start:.1f} V at the start, lowest {v_min:.1f} V on any servo during the run "
              f"(logged per servo with temperature and load)")
    if read_fails:
        print(f"bus: {read_fails} sync reads failed after retries (ran on the previous reading)")
    print(f"stopped by: {stop_reason}. {n} steps ({secs:.1f} s at 50 Hz), loop work median "
          f"{np.median(loop_ms) if n else 0:.1f} ms, {overruns} overruns, IMU gaps {imu.gaps}, stale IMU steps {stale}, "
          f"I2C retries {imu.i2c_errors}. Log: {log_path}")
    for m in imu.msgs:
        if m.startswith("E "):
            print(f"  IMU board: {m[2:]}")
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servo-port", help="default: auto-detect the CH343 adapter")
    ap.add_argument("--imu-port", help="default: auto-detect the RP2040")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ports")
    sp = sub.add_parser("imu-calib", help="IMU zero = the hip plate level; roll and pitch levelled separately")
    sp.add_argument("--axis", choices=["both", "roll", "pitch"], default="both",
                    help="redo only the roll zero (plate level sideways) or only the pitch zero (level fore-aft), "
                         "keeping the other from the saved calibration")
    sub.add_parser("imu-check")
    sp = sub.add_parser("timing", help="bus + IMU speed (no motion)")
    sp.add_argument("--write", action="store_true", help="also time sync writes (torque must be off: nothing moves)")
    sp = sub.add_parser("step-test", help="servo step response, one joint at a time (moves the joints)")
    sp.add_argument("--mode", choices=["air", "stand"], default="air", help="robot held in the air, or standing")
    sp.add_argument("--joints", nargs="+", help="default: all 8 (e.g. L_hip_pitch R_hip_pitch)")
    sp.add_argument("--amp", type=float, default=5.0, help="step size, deg")
    sp.add_argument("--hold", type=float, default=0.6, help="s recorded per step")
    sp.add_argument("--accel", type=int, default=254, help=ACCEL_HELP)
    sp.add_argument("--p-gain", type=int, help="temporarily set the servo P gain (EEPROM 21, default 32) on the "
                                               "tested joints; restored afterwards")
    sp.add_argument("--enable", action="store_true")
    sp.add_argument("--log", help="CSV path (default steptest_<mode>_<time>.csv)")
    sub.add_parser("reset-gains", help="servo P gains back to the factory 32 (no motion)")
    sp = sub.add_parser("run", help="run the stand policy")
    sp.add_argument("--run", default=DEFAULT_RUN, help=f"policy run in runs/ (default {DEFAULT_RUN}), "
                    "'balancer' for the non-learning stance balancer (scripts/pd_balance.py), or a deterministic controller "
                    "(slender_biped/rock_walk.py): lift / sine / cpg / rock, with _full for torso and head on")
    sp.add_argument("--det", nargs="*", default=[], metavar="K=V", help="deterministic controller parameters")
    sp.add_argument("--params", help="checkpoint file in the run (e.g. params_0010M.pkl; default the newest)")
    sp.add_argument("--enable", action="store_true", help="send the policy's targets to the servos")
    sp.add_argument("--dry-run", action="store_true", help="servos limp; compute and show the policy's targets")
    sp.add_argument("--max-tilt", type=float, default=25.0, help="deg from vertical: stop beyond this (hold the stance)")
    sp.add_argument("--limp-on-fault", action="store_true", help="on a fault (tilt, IMU silence, bus error) switch "
                    "the torque off instead of holding the stance pose (the behaviour before 2026-11-11)")
    sp.add_argument("--imu-timeout", type=float, default=0.2,
                    help="s without an IMU sample before the safety stop (shorter gaps run on the last sample)")
    sp.add_argument("--ramp", type=float, default=None,
                    help="s to blend the policy in from the plain stance (default 1; 0 for policies that start from the "
                         "gait's ready pose, which trained fully engaged from the first step). The gait clock waits at "
                         "0 until the blend ends")
    sp.add_argument("--accel-match", action="store_true",
                    help="set each servo's acceleration register to the run's training model (prof_acc: hip yaw / "
                         "roll / ankle ~46 = 400 deg/s^2, hip pitch and knee ~228) instead of --accel for all; the old "
                         "hip servos ignore it, the newer ones (ankles, knees) follow it")
    sp.add_argument("--ready-act", action="store_true",
                    help="during the --ready blend, run the policy with the clock at 0 and blend to the gait start pose plus "
                         "its (smoothed) action, so it doesn't snap the body forward when it engages (2026-11-13)")
    sp.add_argument("--gyro-lpf", type=float, default=None,
                    help="Hz: first-order low-pass on the gyro the policy sees (default: the run's gyro_lpf, else off). 2026-11-13: the "
                         "full-body shake is a loop: the frame rings at 4.5-5 Hz, the gyro sees it and the policy "
                         "answers it at the knees and hip rolls. 2-3 Hz cuts that loop's gain at 4.5 Hz to 0.4-0.55")
    sp.add_argument("--gyro-scale", type=float, default=1.0,
                    help="multiply the gyro the policy sees (1 = as trained); <1 lowers its gain round the shake loop")
    sp.add_argument("--act-lpf", type=float, default=None,
                    help="action low-pass a = k * a_prev + (1 - k) * a_policy (default: the run's act_lpf, else 0 = off)")
    sp.add_argument("--ready", type=float, help="s to blend into the gait's start pose before the policy engages "
                    "(default 2 for policies trained from that pose, step-and-settle walkers; else 0)")
    sp.add_argument("--pitch-trim", type=float, default=0.0,
                    help="deg subtracted from the IMU pitch (+ = the robot balances leaning forward by this much "
                         "more than the IMU zero says; 2026-11-06 quiet stands read +1.7..+3.4 vs +0.8 in sim)")
    sp.add_argument("--roll-trim", type=float, default=0.0,
                    help="deg subtracted from the IMU roll (+ = leaning left). 2026-11-09: standing still on the big "
                         "pads the IMU read +1.4..+3.3 deg in every trial; level the pelvis with a spirit level, and if "
                         "it is level, --roll-trim 2 (or re-run imu-calib)")
    sp.add_argument("--accel", type=int, default=254, help=ACCEL_HELP)
    sp.add_argument("--log", help="CSV path (default stand_<run>_<time>.csv)")
    sp.add_argument("--vx", type=float, default=0.0,
                    help="forward speed command, m/s, for walking / skiing policies (e.g. 0.08); 0 = stand")
    sp.add_argument("--heading-gain", type=float, default=1.0, help="yaw-rate command per rad of heading error")
    a = ap.parse_args()
    {"ports": cmd_ports, "imu-calib": cmd_imu_calib, "imu-check": cmd_imu_check, "timing": cmd_timing,
     "step-test": cmd_step_test, "reset-gains": cmd_reset_gains, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
