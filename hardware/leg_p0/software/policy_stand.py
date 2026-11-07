"""Run a trained stand policy on the real robot: IMU (RP2040) + 8 servos (Feetech bus) -> policy at 50 Hz.

FIRST HARDWARE TEST OF A SIMULATION POLICY. Keep a hand next to the robot and on the power switch.

    python policy_stand.py ports                      # which serial port is the servo bus and which the IMU
    python policy_stand.py imu-calib                  # once per IMU mounting: learn the sensor -> pelvis axes
    python policy_stand.py imu-check                  # live gravity/gyro in pelvis axes (no motion)
    python policy_stand.py timing                     # how fast the bus + IMU loop can run (no motion)
    python policy_stand.py run --dry-run              # policy runs, servos stay limp: hold the robot and tilt it
    python policy_stand.py run --enable               # START: stance, then Enter engages the policy
                                                      # STOP: Enter / q / space (or Ctrl-C) -> holds the stance pose

The policy is a trained network from runs/ (default: hw_stand_push2, the as-built legs + pelvis model, no torso or
head). Its 35 inputs are rebuilt exactly as in training (slender_biped/mjx_env.py): gravity direction and gyro in the
pelvis frame, joint angles relative to the standing pose, 0.1 x joint speeds, the previous action, a zero velocity
command and the 1.6 Hz gait clock. Its 8 outputs are joint targets: q = stance + 0.35 x action (rad).

Safety: everything is clamped to the bench joint limits; the policy stops (torque OFF, the robot goes limp) if the
pelvis tilts past --max-tilt, the IMU goes quiet for 100 ms or the bus fails; a user stop holds the stance pose.
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

    def __init__(self, run):
        d = ROOT / "runs" / run
        pkls = sorted(d.glob("params_*M.pkl")) or [d / "params.pkl"]
        self.path = pkls[-1]
        with open(self.path, "rb") as f:
            norm, pol = _Unpickler(f).load()[:2]
        self.mean = np.asarray(vars(norm)["mean"]["state"], float)
        self.std = np.asarray(vars(norm)["std"]["state"], float)
        layers = pol["params"]
        keys = sorted(layers, key=lambda s: int(s.split("_")[-1]))
        self.W = [np.asarray(layers[k]["kernel"], float) for k in keys]
        self.b = [np.asarray(layers[k]["bias"], float) for k in keys]
        self.env = json.loads((d / "config.json").read_text())["env"]
        n_in = self.W[0].shape[0]
        if n_in != 35:
            sys.exit(f"{run}: policy takes {n_in} inputs; this runner builds the 35-input stand observation only")

    def __call__(self, obs):
        x = (obs - self.mean) / self.std
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = x / (1.0 + np.exp(-x))          # swish
        return np.tanh(x[: x.shape[-1] // 2])


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

    def latest(self):
        """(gravity unit vector, angular rate rad/s) in the pelvis frame, and the sample's age in s."""
        return self.R @ self.g_s, self.R @ self.w_s, time.time() - self.t

    def rebias(self):
        self.ser.write(b"b")
        time.sleep(1.4)

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
        self.rd = GroupSyncRead(bus.port, bus.pk, fb.ADDR["PRESENT_POS"], 4)
        for i in self.ids:
            self.rd.addParam(i)
        self.wr = GroupSyncWrite(bus.port, bus.pk, fb.ADDR["GOAL_POS"], 2)

    def read(self):
        """Joint angles (rad) and speeds (rad/s) in the policy's joint order and sign convention."""
        res = self.rd.txRxPacket()
        if res != COMM_SUCCESS:
            res = self.rd.txRxPacket()                  # one retry: a reply now and then goes missing
            if res != COMM_SUCCESS:
                raise IOError(f"sync read: {self.bus.pk.getTxRxResult(res)}")
        q, qd = np.zeros(8), np.zeros(8)
        for k, (n, j) in enumerate(self.joints.items()):
            pos = self.rd.getData(j["id"], fb.ADDR["PRESENT_POS"], 2)
            spd = self.rd.getData(j["id"], fb.ADDR["PRESENT_SPEED"], 2)
            spd = -(spd & 0x7FFF) if spd & 0x8000 else spd
            q[k] = math.radians(fb.counts_to_deg(j, pos))
            qd[k] = math.radians(j["direction"] * spd / fb.COUNTS_PER_DEG)
        return q, qd

    def write(self, q_rad):
        self.wr.clearParam()
        for (n, j), q in zip(self.joints.items(), q_rad):
            c = fb.deg_to_counts(j, math.degrees(q))
            self.wr.addParam(j["id"], [c & 0xFF, (c >> 8) & 0xFF])
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


def cmd_imu_calib(a):
    imu = Imu(imu_port(a))
    print("IMU mounting calibration (sensor axes -> pelvis axes: x forward, y left, z up). Nothing moves.\n")
    input("1/2  Stand the robot UPRIGHT with the pelvis level (on its pads in the stance pose, or held level).\n"
          "     Hold it still and press Enter... ")
    g1 = mean_gravity(imu)
    input("2/2  Now tilt the whole robot FORWARD (chest toward the floor) by about 20-30 deg and hold it there.\n"
          "     Press Enter... ")
    g2 = mean_gravity(imu)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(g1 @ g2)))))
    if ang < 10:
        sys.exit(f"only {ang:.1f} deg between the two poses: tilt further (20-30 deg) and run it again")
    ez = -g1                                           # pelvis up, in sensor coordinates
    ex = g2 - (g2 @ g1) * g1                           # leaning forward moves gravity toward +x (pelvis frame)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    R = np.vstack([ex, ey, ez])                        # rows = pelvis axes in sensor coordinates
    MOUNT_FILE.write_text(
        "# IMU mounting: v_pelvis = R_sensor_to_pelvis @ v_sensor. Written by `policy_stand.py imu-calib` "
        f"{time.strftime('%Y-%m-%d %H:%M')}.\n# Redo it whenever the IMU board is moved.\n"
        + yaml.safe_dump({"R_sensor_to_pelvis": [[round(float(x), 5) for x in r] for r in R],
                          "forward_tilt_used_deg": round(ang, 1)}, sort_keys=False))
    print(f"\nforward tilt used: {ang:.1f} deg. Saved {MOUNT_FILE.name}:")
    for r, ax in zip(R, "xyz"):
        print(f"  pelvis {ax} = sensor ({r[0]:+.3f}, {r[1]:+.3f}, {r[2]:+.3f})")
    print("Check it with `python policy_stand.py imu-check`.")
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
    q_nom = stance_rad()
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
        for i in sv.ids:                               # same settings as a policy run
            bus.w1(i, fb.ADDR["ACCEL"], a.accel)
            bus.w2(i, fb.ADDR["GOAL_SPEED"], 0)
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


def stance_rad(n=8):
    return np.radians([0.0, 0.0, 16.0, 32.0] * 2)[:n]


def cmd_run(a):
    if not (a.enable or a.dry_run):
        sys.exit("REFUSED: pass --dry-run (servos limp) or --enable (motion; keep a hand on the power switch)")
    pol = Balancer() if a.run == "balancer" else Policy(a.run)
    env = pol.env
    dt, hz, scale = env["ctrl_dt"], env["gait_hz"], env["action_scale"]
    meta = json.loads((ROOT / "models" / f"{env['model']}.json").read_text())
    q_nom = np.array(meta["nominal_q"], float)
    if meta["joints"] != [f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee")]:
        sys.exit(f"unexpected joint order in {env['model']}: {meta['joints']}")
    R = load_mount()
    if a.pitch_trim:              # rotate the pelvis frame about y so the reported pitch drops by pitch_trim
        c, s = math.cos(math.radians(a.pitch_trim)), math.sin(math.radians(a.pitch_trim))
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ R
        print(f"pitch trim {a.pitch_trim:+.1f} deg: the policy sees pitch reduced by that much")
    bus = open_bus(a.servo_port)
    sv = Servos(bus)
    if list(sv.joints) != meta["joints"]:
        sys.exit(f"bench joint order {list(sv.joints)} != policy joint order {meta['joints']}")
    missing = [n for n, j in sv.joints.items() if bus.ping(j["id"]) is None]
    if missing:
        sys.exit(f"no answer from {missing}: connect both legs (and power)")
    imu = Imu(imu_port(a), R)
    lo, hi = sv.limits_rad()
    roll_idx = [sv.names.index("L_hip_roll"), sv.names.index("R_hip_roll")]
    both_in = math.radians(bb.BOTH_ADDUCT_LIMIT)
    if isinstance(pol, Balancer):
        print(f"non-learning balancer, gains {pol.gains}")
    print(f"policy {pol.path.relative_to(ROOT)} ({env['model']}, task {env['task']}), {1 / dt:.0f} Hz, "
          f"action scale {scale} rad, gait clock {hz} Hz")
    g, _, _ = imu.latest()
    print(f"IMU tilt now {tilt_deg(g)[0]:.1f} deg. Tilt cut-out at {a.max_tilt:.0f} deg.")
    log_path = Path(a.log or f"stand_{a.run}_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    with Keys() as keys:
        if a.enable:
            print("\nMoving to the stance pose (hip 16, knee 32)...")
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
            for i in sv.ids:                            # the policy's servo model has no speed/accel ramp: as fast as
                                                        # the register allows (0 means a slow default on old firmware)
                bus.w1(i, fb.ADDR["ACCEL"], a.accel)
                bus.w2(i, fb.ADDR["GOAL_SPEED"], 0)
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
        last_a = np.zeros(8)
        phase = 0.0
        stop_reason, overruns, k = "user", 0, 0
        loop_ms = []
        t_next = time.perf_counter()
        f = log_path.open("w", newline="")
        w = csv.writer(f)
        w.writerow(["t", "loop_ms"] + [f"g{c}" for c in "xyz"] + [f"w{c}" for c in "xyz"]
                   + [f"q_{n}" for n in sv.names] + [f"qd_{n}" for n in sv.names]
                   + [f"a_{n}" for n in sv.names] + [f"cmd_{n}" for n in sv.names])
        t_start = time.perf_counter()
        try:
            while True:
                t0 = time.perf_counter()
                q, qd = sv.read()
                g, wv, age = imu.latest()
                if age > 0.1:
                    stop_reason = f"IMU silent for {age * 1000:.0f} ms"
                    break
                tilt = tilt_deg(g)[0]
                if tilt > a.max_tilt:
                    stop_reason = f"tilt {tilt:.0f} deg > {a.max_tilt:.0f}"
                    break
                obs = np.concatenate([g, wv, q - q_nom, 0.1 * qd, last_a, np.zeros(3),
                                      [math.sin(phase), math.cos(phase)]])
                act = np.clip(pol(obs), -1.0, 1.0)
                ramp = min(1.0, (t0 - t_start) / a.ramp) if a.ramp > 0 else 1.0
                q_cmd = np.clip(q_nom + ramp * scale * act, lo, hi)
                if q_cmd[roll_idx[0]] < 0 and q_cmd[roll_idx[1]] < 0:       # pelvis rule: legs collide if both
                    q_cmd[roll_idx] = np.maximum(q_cmd[roll_idx], both_in)  # swing inward by more than ~5 deg
                if a.enable:
                    sv.write(q_cmd)
                last_a = act
                phase = (phase + 2 * math.pi * hz * dt) % (2 * math.pi)
                el = (time.perf_counter() - t0) * 1000
                loop_ms.append(el)
                w.writerow([round(t0 - t_start, 4), round(el, 2)] + list(np.round(g, 4)) + list(np.round(wv, 4))
                           + list(np.round(q, 4)) + list(np.round(qd, 3)) + list(np.round(act, 4))
                           + list(np.round(q_cmd, 4)))
                if k % 10 == 0:
                    d = np.degrees(q_cmd - q_nom)
                    print(f"\r{t0 - t_start:6.1f} s  tilt {tilt:4.1f}  |a| max {np.max(np.abs(act)):.2f}  target-stance"
                          f" L[{d[0]:+4.0f}{d[1]:+4.0f}{d[2]:+4.0f}{d[3]:+4.0f}] R[{d[4]:+4.0f}{d[5]:+4.0f}{d[6]:+4.0f}"
                          f"{d[7]:+4.0f}] deg  loop {el:4.1f} ms ", end="", flush=True)
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
                if stop_reason == "user" or stop_reason == "Ctrl-C":
                    try:
                        for i in sv.ids:
                            bus.w1(i, fb.ADDR["ACCEL"], bus.cfg["acceleration"])
                            bus.w2(i, fb.ADDR["GOAL_SPEED"], bus.cfg["speed_counts_per_s"])
                        sv.write(q_nom)
                        print("STOPPED: holding the stance pose. Release with `python biped_bench.py off` "
                              "(or `home --enable`).")
                    except IOError as e:
                        print(f"could not command the stance ({e}): torque off")
                        sv.torque(False)
                else:
                    sv.torque(False)
                    print("SAFETY STOP: torque OFF (the robot goes limp).")
            imu.close()
    n = len(loop_ms)
    secs = n * dt
    print(f"stopped by: {stop_reason}. {n} steps ({secs:.1f} s at 50 Hz), loop work median "
          f"{np.median(loop_ms) if n else 0:.1f} ms, {overruns} overruns, IMU gaps {imu.gaps}. Log: {log_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servo-port", help="default: auto-detect the CH343 adapter")
    ap.add_argument("--imu-port", help="default: auto-detect the RP2040")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ports")
    sub.add_parser("imu-calib")
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
    sp.add_argument("--run", default=DEFAULT_RUN, help=f"policy run in runs/ (default {DEFAULT_RUN}), or "
                    "'balancer' for the non-learning stance balancer (scripts/pd_balance.py)")
    sp.add_argument("--enable", action="store_true", help="send the policy's targets to the servos")
    sp.add_argument("--dry-run", action="store_true", help="servos limp; compute and show the policy's targets")
    sp.add_argument("--max-tilt", type=float, default=25.0, help="deg from vertical: torque off beyond this")
    sp.add_argument("--ramp", type=float, default=1.0, help="s to blend the policy in from the plain stance")
    sp.add_argument("--pitch-trim", type=float, default=0.0,
                    help="deg subtracted from the IMU pitch (+ = the robot balances leaning forward by this much "
                         "more than the IMU zero says; 2026-11-06 quiet stands read +1.7..+3.4 vs +0.8 in sim)")
    sp.add_argument("--accel", type=int, default=254, help=ACCEL_HELP)
    sp.add_argument("--log", help="CSV path (default stand_<run>_<time>.csv)")
    a = ap.parse_args()
    {"ports": cmd_ports, "imu-calib": cmd_imu_calib, "imu-check": cmd_imu_check, "timing": cmd_timing,
     "step-test": cmd_step_test, "reset-gains": cmd_reset_gains, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
