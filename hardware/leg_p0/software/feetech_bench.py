"""Minimal, deliberately cautious bench tool for the P0 leg (Feetech STS3215 on a TTL bus adapter).

Software stop is NOT an emergency stop: keep a hand on the hardware power switch during every motion test.

    pip install feetech-servo-sdk pyyaml          # provides the `scservo_sdk` module
    python feetech_bench.py scan                   # list IDs that answer (does not move anything)
    python feetech_bench.py read                   # live position/load/voltage/temperature of configured joints
    python feetech_bench.py set-id --old 1 --new 3 --i-have-only-one-servo-connected
    python feetech_bench.py torque-off             # relax every configured joint
    python feetech_bench.py move knee 10 --enable  # move ONE joint to +10 deg (from neutral), slowly
    python feetech_bench.py wave --amp 10 --cycles 2 --enable   # low-amplitude 4-joint test, logged to CSV

Every command that moves a motor requires --enable and refuses targets outside the configured limits.
Ctrl-C at any time disables torque on all configured joints.
"""
from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from pathlib import Path

import yaml

try:
    from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install feetech-servo-sdk pyyaml")

# STS3215 control table (Feetech STS/SMS series; same addresses as the LeRobot STS3215 table)
ADDR = dict(ID=5, BAUD=6, MIN_ANGLE=9, MAX_ANGLE=11, TORQUE_ENABLE=40, ACCEL=41, GOAL_POS=42, GOAL_TIME=44,
            GOAL_SPEED=46, LOCK=55, PRESENT_POS=56, PRESENT_SPEED=58, PRESENT_LOAD=60, PRESENT_VOLT=62,
            PRESENT_TEMP=63, MOVING=66, PRESENT_CURRENT=69)
COUNTS_PER_DEG = 4096 / 360.0
HERE = Path(__file__).resolve().parent


class Bus:
    def __init__(self, cfg):
        self.cfg = cfg
        self.port = PortHandler(cfg["port"])
        self.pk = PacketHandler(0)  # protocol_end 0 = STS/SMS
        try:
            opened = self.port.openPort()
        except Exception as e:  # serial.SerialException: wrong port name, no permission, adapter unplugged
            sys.exit(f"cannot open {cfg['port']}: {e}\n(set `port:` in bench_config.yaml; on Linux add yourself "
                     "to the dialout group)")
        if not opened:
            sys.exit(f"cannot open {cfg['port']}")
        if not self.port.setBaudRate(cfg["baudrate"]):
            sys.exit(f"cannot set baud {cfg['baudrate']}")

    def _ok(self, res, err, what):
        if res != COMM_SUCCESS:
            raise IOError(f"{what}: {self.pk.getTxRxResult(res)}")
        if err:
            print(f"  warning ({what}): {self.pk.getRxPacketError(err)}")

    def ping(self, sid):
        model, res, err = self.pk.ping(self.port, sid)
        return model if res == COMM_SUCCESS else None

    def r1(self, sid, a):
        v, res, err = self.pk.read1ByteTxRx(self.port, sid, a)
        self._ok(res, err, f"read id {sid} @{a}")
        return v

    def r2(self, sid, a):
        v, res, err = self.pk.read2ByteTxRx(self.port, sid, a)
        self._ok(res, err, f"read id {sid} @{a}")
        return v

    def w1(self, sid, a, v):
        res, err = self.pk.write1ByteTxRx(self.port, sid, a, int(v))
        self._ok(res, err, f"write id {sid} @{a}")

    def w2(self, sid, a, v):
        res, err = self.pk.write2ByteTxRx(self.port, sid, a, int(v))
        self._ok(res, err, f"write id {sid} @{a}")

    def torque(self, sid, on):
        self.w1(sid, ADDR["TORQUE_ENABLE"], 1 if on else 0)

    def present(self, sid):
        signed = lambda v, b: -(v & ~(1 << b)) if v & (1 << b) else v  # noqa: E731  sign-magnitude
        return dict(pos=self.r2(sid, ADDR["PRESENT_POS"]), speed=signed(self.r2(sid, ADDR["PRESENT_SPEED"]), 15),
                    load=signed(self.r2(sid, ADDR["PRESENT_LOAD"]), 10), volt=self.r1(sid, ADDR["PRESENT_VOLT"]) / 10,
                    temp=self.r1(sid, ADDR["PRESENT_TEMP"]))


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def deg_to_counts(j, deg):
    return int(round(j["center"] + j["direction"] * deg * COUNTS_PER_DEG))


def counts_to_deg(j, counts):
    return (counts - j["center"]) / (j["direction"] * COUNTS_PER_DEG)


def check_target(name, j, deg):
    if not (j["min_deg"] <= deg <= j["max_deg"]):
        sys.exit(f"REFUSED: {name} target {deg:.1f} deg outside limits [{j['min_deg']}, {j['max_deg']}]")


def install_stop(bus):
    def stop(*_):
        print("\nSTOP: disabling torque on all configured joints")
        for j in bus.cfg["joints"].values():
            try:
                bus.torque(j["id"], False)
            except Exception as e:  # keep going: try every joint
                print("  ", e)
        sys.exit(1)
    signal.signal(signal.SIGINT, stop)
    return stop


def cmd_scan(bus, a):
    found = [(sid, bus.ping(sid)) for sid in range(0, 254)]
    found = [(s, m) for s, m in found if m is not None]
    print("answering IDs:", found or "none")
    if len({s for s, _ in found}) != len(found):
        print("duplicate replies detected: connect servos one at a time and assign unique IDs")


def cmd_read(bus, a):
    for name, j in bus.cfg["joints"].items():
        try:
            p = bus.present(j["id"])
            print(f"{name:10s} id {j['id']}: {p['pos']:5d} counts = {counts_to_deg(j, p['pos']):7.1f} deg  "
                  f"load {p['load']:5d}  {p['volt']:.1f} V  {p['temp']} C")
        except IOError as e:
            print(f"{name:10s} id {j['id']}: no answer ({e})")


def cmd_set_id(bus, a):
    if not a.i_have_only_one_servo_connected:
        sys.exit("REFUSED: connect exactly one servo and pass --i-have-only-one-servo-connected")
    if bus.ping(a.old) is None:
        sys.exit(f"no servo answers at id {a.old}")
    bus.w1(a.old, ADDR["LOCK"], 0)           # unlock EEPROM
    bus.w1(a.old, ADDR["ID"], a.new)
    bus.w1(a.new, ADDR["LOCK"], 1)           # lock EEPROM
    print("new id answers:", bus.ping(a.new) is not None)


def cmd_torque_off(bus, a):
    for name, j in bus.cfg["joints"].items():
        bus.torque(j["id"], False)
        print("torque off:", name)


def goto(bus, name, j, deg):
    check_target(name, j, deg)
    bus.w1(j["id"], ADDR["ACCEL"], bus.cfg["acceleration"])
    bus.w2(j["id"], ADDR["GOAL_SPEED"], bus.cfg["speed_counts_per_s"])
    bus.torque(j["id"], True)
    bus.w2(j["id"], ADDR["GOAL_POS"], deg_to_counts(j, deg))


def cmd_move(bus, a):
    if not a.enable:
        sys.exit("REFUSED: motion needs --enable (keep a hand on the power switch)")
    j = bus.cfg["joints"][a.joint]
    install_stop(bus)
    goto(bus, a.joint, j, a.deg)
    t0 = time.time()
    while time.time() - t0 < 6.0:
        p = bus.present(j["id"])
        print(f"\r{a.joint}: {counts_to_deg(j, p['pos']):6.1f} deg  load {p['load']:5d}  {p['temp']} C", end="")
        time.sleep(0.1)
    print()


def cmd_wave(bus, a):
    if not a.enable:
        sys.exit("REFUSED: motion needs --enable (keep a hand on the power switch)")
    install_stop(bus)
    joints = bus.cfg["joints"]
    for name, j in joints.items():   # pre-check the whole trajectory before moving anything
        for s in (-1, 1):
            check_target(name, j, min(max(s * a.amp, j["min_deg"]), j["max_deg"]))
    log = Path(a.log)
    with log.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "joint", "target_deg", "measured_deg", "load", "volt", "temp"])
        t0 = time.time()
        period = 4.0
        while time.time() - t0 < a.cycles * period:
            t = time.time() - t0
            for k, (name, j) in enumerate(joints.items()):
                tgt = a.amp * math.sin(2 * math.pi * t / period + k * math.pi / 2)
                tgt = min(max(tgt, j["min_deg"]), j["max_deg"])
                goto(bus, name, j, tgt)
                p = bus.present(j["id"])
                w.writerow([round(t, 3), name, round(tgt, 2), round(counts_to_deg(j, p["pos"]), 2), p["load"], p["volt"], p["temp"]])
            time.sleep(0.05)
    for name, j in joints.items():
        goto(bus, name, j, min(max(0.0, j["min_deg"]), j["max_deg"]))
    print("logged to", log)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "bench_config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    sub.add_parser("read")
    sp = sub.add_parser("set-id")
    sp.add_argument("--old", type=int, required=True)
    sp.add_argument("--new", type=int, required=True)
    sp.add_argument("--i-have-only-one-servo-connected", action="store_true")
    sub.add_parser("torque-off")
    sp = sub.add_parser("move")
    sp.add_argument("joint")
    sp.add_argument("deg", type=float)
    sp.add_argument("--enable", action="store_true")
    sp = sub.add_parser("wave")
    sp.add_argument("--amp", type=float, default=10.0)
    sp.add_argument("--cycles", type=int, default=2)
    sp.add_argument("--log", default="wave_log.csv")
    sp.add_argument("--enable", action="store_true")
    a = ap.parse_args()
    bus = Bus(load_cfg(a.config))
    {"scan": cmd_scan, "read": cmd_read, "set-id": cmd_set_id, "torque-off": cmd_torque_off,
     "move": cmd_move, "wave": cmd_wave}[a.cmd](bus, a)


if __name__ == "__main__":
    main()
