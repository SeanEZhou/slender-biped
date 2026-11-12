"""Measuring exercises: small scripted experiments that measure what the simulator assumes (docs/measuring.md).

    python exercises.py hold  --enable                 # E1 tip-over angles by hand, E2 hip give by hand (stance held)
    python exercises.py shift --enable --side L        # E6 slow sideways weight shift to the hip limit (catch it!)
    python exercises.py lift  --enable --side R --lifts 2 4 6 8 12 --lower 0.5   # E4 minimum clearance
    python exercises.py lift  --enable --side R --lifts 10 --hold 60      # E3 one-foot tip-over angles by hand
    python exercises.py lift  --enable --side R --lifts 15 --lower 0.6 0.3 0.15 0.08 0.04   # E5 landing

Each writes ex_<name>_<time>.csv (the runner's columns plus stage, rep and mark) and a .json beside it with the
exercise's settings. Analyse on the PC: .venv/bin/python scripts/exercises_sim.py compare <csv>.

Keys while running (each key is logged in the `mark` column):
  hold : f / b / l / r = the robot balances by itself on its toes / heels / left edges / right edges (tilt test),
         p = pushing now, m = any other mark
  shift: u = the far board just came unloaded (the paper strip under it pulls out), t = it is tipping (returns fast)
  lift : c = the paper strip under the lifted board pulls out freely (clear), n = still pinned, s = scuff / catch,
         o / i = (long --hold) tilted by hand on one foot, it balances by itself over the standing board's outer /
         inner edge
  all  : Enter or q = stop (returns to the stance pose, torque stays on)

Safety as in the runner: joint limits, the both-legs-inward rule, a tilt cut-out (returns to the stance pose fast),
torque stays on at the end. Keep a hand next to the robot; the shift exercise is meant to end in a tip-over.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import biped_bench as bb
import exercise_poses as ep
import feetech_bench as fb
import policy_stand as ps


def setup(a):
    bus = ps.open_bus(a.servo_port)
    sv = ps.Servos(bus)
    if list(sv.joints) != ep.JOINTS:
        sys.exit(f"joint order {list(sv.joints)} != {ep.JOINTS}: the exercises need the 10-joint (ankle roll) config")
    missing = [n for n, j in sv.joints.items() if bus.ping(j["id"]) is None]
    if missing:
        sys.exit(f"no answer from {missing}: connect both legs (and power)")
    imu = ps.Imu(ps.imu_port(a), ps.load_mount())
    return bus, sv, imu


def to_stance(bus, sv):
    for n, j in sv.joints.items():
        fb.goto(bus, n, j, math.degrees(ep.Q_NOM[sv.names.index(n)]))
    t0 = time.time()
    while time.time() - t0 < 5.0:
        q, _ = sv.read()
        if np.max(np.abs(q - ep.Q_NOM)) < math.radians(2):
            return
        time.sleep(0.1)
    print(f"  (not all joints reached the stance: worst {np.degrees(np.max(np.abs(q - ep.Q_NOM))):.1f} deg off)")


def run(a, name, target, duration, stop_keys, fast_keys, info):
    if not a.enable:
        sys.exit("REFUSED: pass --enable (motion; keep a hand on the power switch)")
    bus, sv, imu = setup(a)
    lo, hi = sv.limits_rad()
    roll_idx = [sv.names.index("L_hip_roll"), sv.names.index("R_hip_roll")]
    both_in = math.radians(bb.BOTH_ADDUCT_LIMIT)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path(a.log or f"ex_{name}_{stamp}.csv")
    meta = dict(exercise=name, time=stamp, args={k: v for k, v in vars(a).items() if k not in ("servo_port", "imu_port")},
                **info)
    dt = 0.02
    with ps.Keys() as keys:
        print("\nMoving to the stance pose (hip 16, knee 32, ankles 0)...")
        try:
            to_stance(bus, sv)
        except KeyboardInterrupt:
            sv.torque(False)
            sys.exit("\nCtrl-C: torque off")
        sv.set_accel(a.accel)
        imu.rebias()
        print(f"\n{info['instructions']}\n  Enter = START      q = abort (holds the stance pose)")
        if keys.wait("\nq") == "q":
            print("aborted: holding the stance pose")
            return
        print("RUNNING. Enter / q = stop.\n")
        f = log_path.open("w", newline="")
        w = csv.writer(f)
        w.writerow(["t", "loop_ms"] + [f"g{c}" for c in "xyz"] + [f"w{c}" for c in "xyz"]
                   + [f"q_{n}" for n in sv.names] + [f"qd_{n}" for n in sv.names]
                   + [f"a_{n}" for n in sv.names] + [f"cmd_{n}" for n in sv.names]
                   + [f"volt_{n}" for n in sv.names] + [f"temp_{n}" for n in sv.names] + [f"load_{n}" for n in sv.names]
                   + ["stage", "rep", "mark"])
        marks = []
        stop_reason, ret = "end", None          # ret: (t_start, q_from, seconds) once returning to the stance
        q_cmd = ep.Q_NOM.copy()
        t_start = time.perf_counter(); t_next = t_start; k = 0
        try:
            while True:
                t0 = time.perf_counter(); t = t0 - t_start
                q, qd = sv.read()
                g, wv, age = imu.latest()
                if age > 0.2:
                    stop_reason = "IMU silent"
                    break
                tilt, roll, pitch = ps.tilt_deg(g)
                key = keys.get()
                mark = ""
                if key is not None:
                    if key in stop_keys and ret is None:
                        stop_reason, ret = "user", (t, q_cmd.copy(), a.ret_slow)
                    elif key in fast_keys and ret is None:
                        stop_reason, ret = f"key {key}", (t, q_cmd.copy(), a.ret_fast)
                    if key.strip():
                        mark = key
                        marks.append((round(t, 2), key, round(roll, 1), round(pitch, 1)))
                        print(f"\n  mark {key!r} at {t:.1f} s: roll {roll:+.1f}, pitch {pitch:+.1f} deg")
                if ret is None and a.max_tilt and tilt > a.max_tilt:
                    stop_reason, ret = f"tilt {tilt:.0f} deg", (t, q_cmd.copy(), a.ret_fast)
                if ret is None and t > duration:
                    stop_reason, ret = "end", (t, q_cmd.copy(), a.ret_slow)
                if ret is None:
                    q_cmd, stage, rep = target(t)
                else:
                    r = ep.ease((t - ret[0]) / ret[2])
                    q_cmd, stage, rep = ret[1] + r * (ep.Q_NOM - ret[1]), "return", -1
                q_cmd = np.clip(q_cmd, lo, hi)
                if q_cmd[roll_idx[0]] < 0 and q_cmd[roll_idx[1]] < 0:
                    q_cmd[roll_idx] = np.maximum(q_cmd[roll_idx], both_in)
                sv.write(q_cmd)
                el = (time.perf_counter() - t0) * 1000
                w.writerow([round(t, 4), round(el, 2)] + list(np.round(g, 4)) + list(np.round(wv, 4))
                           + list(np.round(q, 4)) + list(np.round(qd, 3)) + [0.0] * len(q)
                           + list(np.round(q_cmd, 4)) + list(sv.volt) + list(sv.temp.astype(int))
                           + list(sv.load.astype(int)) + [stage, rep, mark])
                if k % 10 == 0:
                    print(f"\r{t:6.1f} s  {stage:8s} rep {rep:2d}  roll {roll:+5.1f}  pitch {pitch:+5.1f} deg  "
                          f"{sv.volt.min():.1f} V ", end="", flush=True)
                k += 1
                if ret is not None and t - ret[0] > ret[2] + 0.5:
                    break
                t_next += dt
                wait = t_next - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    t_next = time.perf_counter()
        except KeyboardInterrupt:
            stop_reason = "Ctrl-C"
        except IOError as e:
            stop_reason = f"bus error: {e}"
        finally:
            f.close()
            meta.update(stop=stop_reason, marks=marks)
            log_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
            try:
                sv.set_accel(None)
                sv.write(ep.Q_NOM)
            except IOError:
                pass
            imu.close()
    print(f"\nstopped by: {stop_reason}; holding the stance pose (torque ON; `python biped_bench.py off` releases).")
    print(f"log {log_path} (+ .json); marks: {marks}")


def cmd_hold(a):
    info = dict(instructions=(
        "HOLD: the robot holds the stance pose; you move it. Leave it untouched, boards flat, for the first 2 s\n"
        "  (that reading is the zero the tip angles are measured from).\n"
        "  E1 tilt test: tilt it slowly by hand onto its toes until it stays there by itself (no force in your hand),\n"
        "     press f; the same on the heels (b), onto the left board's outer edge (l), the right's (r). 2-3 times each.\n"
        "  E2 give test: both boards flat on the floor, push the pelvis gently sideways left and right (press p while\n"
        "     pushing); then hold one board down and lift the other a few mm by the pelvis (press p)."))
    a.max_tilt = 0.0
    run(a, "hold", lambda t: (ep.Q_NOM.copy(), "hold", 0), a.seconds, "\nq", "", info)


def cmd_shift(a):
    tr = ep.ShiftRamp(a.side, a.rate, a.max_shift)
    info = dict(side=a.side, rate_deg_s=a.rate, max_shift_deg=a.max_shift, instructions=(
        f"SHIFT: the weight moves slowly onto the {a.side} board ({a.rate} deg/s of hip roll, up to {a.max_shift} deg),\n"
        f"  until it tips over that board's outer edge. Paper strip under the OTHER board: press u when it pulls out\n"
        f"  freely, t when the robot starts to tip (it returns to the stance fast). CATCH IT."))
    run(a, f"shift{a.side}", tr, tr.duration, "\nq", "t", info)


def cmd_lift(a):
    reps = ep.parse_reps(a.lifts, a.lower)
    tr = ep.LiftReps(a.side, a.shift, reps, hold=a.hold)
    stance = "L" if a.side == "R" else "R"
    info = dict(side=a.side, shift_deg=a.shift, reps=reps, hold_s=a.hold, instructions=(
        f"LIFT: {len(reps)} reps: shift {a.shift} deg onto the {stance} board, lift the {a.side} board, hold {a.hold} s, "
        f"lower, unshift.\n  Lifts (mm, rigid-leg value) / lowering (s): {[(round(h * 1000), lt) for h, lt in reps]}\n"
        f"  Paper strip under the {a.side} board: during each hold tug it, press c if it pulls out freely, n if pinned,\n"
        f"  s if the board scuffs or catches. Watch each landing. With a long --hold: tilt it by hand outward over the\n"
        f"  standing board until it balances by itself (o), then inward (i)."))
    run(a, f"lift{a.side}", tr, tr.duration, "\nq", "", info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servo-port")
    ap.add_argument("--imu-port")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--enable", action="store_true")
    common.add_argument("--log")
    common.add_argument("--accel", type=int, default=254, help=ps.ACCEL_HELP)
    common.add_argument("--max-tilt", type=float, default=12.0, help="deg: beyond this, return to the stance fast")
    common.add_argument("--ret-fast", type=float, default=0.3, help="s back to the stance after a tilt / t key")
    common.add_argument("--ret-slow", type=float, default=1.5, help="s back to the stance after Enter / the end")
    sp = sub.add_parser("hold", parents=[common], help="E1/E2: stance held, you tilt / push it")
    sp.add_argument("--seconds", type=float, default=300.0)
    sp = sub.add_parser("shift", parents=[common], help="E3: slow sideways weight shift until it tips")
    sp.add_argument("--side", choices=["L", "R"], default="L", help="the board that takes the weight")
    sp.add_argument("--rate", type=float, default=0.5, help="deg/s of hip roll")
    sp.add_argument("--max-shift", type=float, default=9.5, help="deg (the bench stops hip-roll adduction at -10)")
    sp = sub.add_parser("lift", parents=[common], help="E4/E5: stand on one foot, lift, hold, land")
    sp.add_argument("--side", choices=["L", "R"], default="R", help="the board that lifts")
    sp.add_argument("--shift", type=float, default=9.5, help="deg of weight shift onto the other board (gait: 7.7; limit 10)")
    sp.add_argument("--lifts", type=float, nargs="+", default=[2, 4, 6, 8, 12], help="mm (rigid-leg value)")
    sp.add_argument("--lower", type=float, nargs="+", default=[0.5], help="s to lower the board")
    sp.add_argument("--hold", type=float, default=2.5, help="s on one foot")
    a = ap.parse_args()
    {"hold": cmd_hold, "shift": cmd_shift, "lift": cmd_lift}[a.cmd](a)


if __name__ == "__main__":
    main()
