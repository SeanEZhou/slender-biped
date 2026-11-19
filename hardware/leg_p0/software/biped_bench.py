"""Two-leg bench control for the P0 biped on the P08 pelvis: both legs on ONE bus, moving together.

Loads bench_config.yaml (left, IDs 1-4) and bench_config_right.yaml (right, IDs 5-8) into one set of joints,
L_hip_yaw ... R_knee, and plays whole-body keyframe routines with every servo arriving at each pose together.
Same sign conventions on both legs (yaw + = toe out, roll + = abduction, pitch + = leg forward, knee + = foot
back); each config's `direction` handles the mirroring.

Software stop is NOT an emergency stop: keep a hand on the power switch.

    python biped_bench.py list                          # routines and what they do
    python biped_bench.py read                          # all 8 joints: angle, load, temperature (no motion)
    python biped_bench.py stand --enable                # both legs to the standing pose, then HOLD
    python biped_bench.py play squat --enable --speed 0.5
    python biped_bench.py play march --enable           # alternating knee lifts (on the ground or in the air)
    python biped_bench.py march --enable --current      # ...and measure motor current: summary + CSV log
    python biped_bench.py walk --enable                 # shorthand: a routine name alone means `play <name>`
    python biped_bench.py off                           # torque off on all 8
    python biped_bench.py trim --enable                 # stand, nudge the left hip pitch / knee, re-zero them

Every target is checked against each joint's limits, and every pose AND the motion between poses against the
pelvis collision rule (README "Two-leg bench"): both legs adducting together collide beyond ~5 deg.
Ctrl-C during a routine turns torque off on all 8 joints. After `stand` (or --stay), use `off`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import feetech_bench as fb

HERE = Path(__file__).resolve().parent
LEGS = {"L": HERE / "bench_config.yaml", "R": HERE / "bench_config_right.yaml"}
BOTH_ADDUCT_LIMIT = -4.0     # deg; README: both legs adducting together collide beyond ~5 deg (1 deg margin)

STANCE = dict(hip_pitch=16, knee=32)          # pad level (leg_p0 README)


def both(**kw):
    return {f"{s}_{k}": v for s in "LR" for k, v in kw.items()}


def left(**kw):
    return {f"L_{k}": v for k, v in kw.items()}


def right(**kw):
    return {f"R_{k}": v for k, v in kw.items()}


# keyframes: (pose, seconds); joints left out keep their previous target; every routine ends at neutral
# unless --stay. Routines marked AIR are for the robot held up off the desk; the rest work on the pads.
ROUTINES = {
    "home": ("all 8 joints back to neutral (straight legs) over 2.5 s, then torque off", []),
    "check": ("AIR  each joint of each leg +10 deg alone (L then R): verify every direction",
              [(p, 1.5) for side in (left, right) for k, a in
               (("hip_yaw", 10), ("hip_roll", 10), ("hip_pitch", 10), ("knee", 20))
               for p in (side(**{k: a}), side(**{k: 0}))]),
    "check_ankle": ("AIR  ankle_p1 legs: each ankle roll +10 deg alone (L then R): the foot's OUTER edge must go UP "
                    "(sim sign, same sense as hip roll + = abduction); if it goes down, flip that ankle's direction",
                    [(p, 1.5) for side in (left, right) for p in (side(ankle_roll=10), side(ankle_roll=0))]),
    "stance": ("both legs to hip 16 / knee 32 (pads level), then back to straight",
               [(both(**STANCE), 2.0), ({}, 2.0)]),
    "squat": ("from stance, 3 squats to hip 45 / knee 90 on both legs",
              [(both(**STANCE), 2.0)] + [(both(hip_pitch=45, knee=90), 2.5), (both(**STANCE), 2.5)] * 3),
    "sway": ("from stance, rock the pelvis side to side (one leg abducts while the other adducts) x3",
             [(both(**STANCE), 2.0)] + [({**left(hip_roll=6), **right(hip_roll=-6)}, 1.5),
                                        ({**left(hip_roll=-6), **right(hip_roll=6)}, 1.5)] * 3
             + [(both(hip_roll=0), 1.5)]),
    "turn": ("from stance, turn both feet left then right (one toe out, the other toe in) x2",
             [(both(**STANCE), 2.0)] + [({**left(hip_yaw=12), **right(hip_yaw=-12)}, 1.5),
                                        ({**left(hip_yaw=-12), **right(hip_yaw=12)}, 2.0)] * 2
             + [(both(hip_yaw=0), 1.5)]),
    "march": ("from stance, alternate knee lifts (hip 45 / knee 80) x4; works on the ground (stands on one leg)",
              [(both(**STANCE), 2.0)] + [(left(hip_pitch=45, knee=80), 1.0), (left(**STANCE), 1.0),
                                         (right(hip_pitch=45, knee=80), 1.0), (right(**STANCE), 1.0)] * 4),
    "walk": ("AIR  from stance, alternating swing-through gait: lift, reach, trade places x3",
             [(both(**STANCE), 2.0)] + [
                 ({**left(hip_pitch=35, knee=70), **right(hip_pitch=5, knee=20)}, 1.0),    # L lifts, R pushes back
                 ({**left(hip_pitch=30, knee=20), **right(hip_pitch=-5, knee=15)}, 0.8),   # L reaches forward
                 ({**right(hip_pitch=35, knee=70), **left(hip_pitch=5, knee=20)}, 1.0),    # R lifts, L pushes back
                 ({**right(hip_pitch=30, knee=20), **left(hip_pitch=-5, knee=15)}, 0.8),   # R reaches forward
             ] * 3 + [(both(**STANCE), 1.5)]),
}


def merged_cfg():
    cfgs = {s: fb.load_cfg(p) for s, p in LEGS.items()}
    for key in ("port", "baudrate"):
        if cfgs["L"][key] != cfgs["R"][key]:
            sys.exit(f"left and right configs disagree on `{key}`: both legs must be on one bus")
    cfg = dict(cfgs["L"])
    cfg["joints"] = {f"{s}_{n}": j for s, c in cfgs.items() for n, j in c["joints"].items()}
    ids = [j["id"] for j in cfg["joints"].values()]
    if len(set(ids)) != len(ids):
        sys.exit(f"duplicate servo IDs across the two configs: {ids}")
    return cfg


def check_pair(pose, where):
    lr, rr = pose["L_hip_roll"], pose["R_hip_roll"]
    if lr < 0 and rr < 0 and min(lr, rr) < BOTH_ADDUCT_LIMIT:
        sys.exit(f"REFUSED: {where}: both legs adducting (L roll {lr:.1f}, R roll {rr:.1f}); "
                 f"the legs collide beyond ~5 deg (limit {BOTH_ADDUCT_LIMIT:g} each)")


def build_frames(joints, name, speed, stay):
    keys = ROUTINES[name][1] + ([] if stay else [({n: 0.0 for n in joints}, 2.5)])
    frames, pose = [], {n: 0.0 for n in joints}
    for target, dur in keys:
        unknown = set(target) - set(joints)
        if unknown:
            sys.exit(f"routine {name}: unknown joints {sorted(unknown)}")
        pose = {**pose, **target}
        frames.append((dict(pose), dur / speed))
    return frames


def precheck(joints, frames, start):
    """Limits at every pose; the pair rule at every pose and along the straight-line motion between poses."""
    prev = start
    for k, (pose, _) in enumerate(frames):
        for n, deg in pose.items():
            fb.check_target(n, joints[n], deg)
        for s in range(11):                        # servos arrive together, so motion ~ linear interpolation
            t = s / 10
            check_pair({n: prev[n] + t * (pose[n] - prev[n]) for n in pose}, f"frame {k + 1}")
        prev = pose


def cmd_list(bus, a):
    for name, (desc, _) in ROUTINES.items():
        print(f"  {name:7s} {desc}")
    print("  (stand = stance and hold: `stand --enable`)")


def cmd_read(bus, a):
    fb.cmd_read(bus, a)


def cmd_off(bus, a):
    fb.cmd_torque_off(bus, a)


def cmd_play(bus, a):
    if not a.enable:
        sys.exit("REFUSED: motion needs --enable (keep a hand on the power switch)")
    if a.speed <= 0 or a.speed > 2:
        sys.exit("REFUSED: --speed must be in (0, 2]")
    joints = bus.cfg["joints"]
    missing = [n for n, j in joints.items() if bus.ping(j["id"]) is None]
    if missing:
        sys.exit(f"no answer from {missing}: connect both legs (and power) first")
    start = {n: fb.counts_to_deg(j, bus.r2(j["id"], fb.ADDR["PRESENT_POS"])) for n, j in joints.items()}
    frames = build_frames(joints, a.name, a.speed, a.stay)
    precheck(joints, frames, start)                # before anything moves
    if ROUTINES[a.name][0].startswith("AIR"):
        print(f"{a.name}: AIR routine, hold the robot up off the desk")
    stop = fb.install_stop(bus)
    a.demo = False
    try:
        fb._play(bus, a, joints, frames)
    except IOError as e:
        print(f"\nBUS ERROR after retries: {e}")
        stop()


def cmd_stand(bus, a):
    a.name, a.stay, a.hold = "stance", True, False
    cmd_play(bus, a)


TRIM_KEYS = (("a", "d", "A", "D"), ("j", "l", "J", "L"))   # per trimmed joint: -0.5, +0.5, -0.1, +0.1 deg
TRIM_HELP = {"L_hip_pitch": "THIGH back / forward", "L_knee": "SHIN forward / back (knee straightens / bends)",
             "hip_roll": "leg IN / OUT (+ = abduction)", "ankle_roll": "foot's outer edge DOWN / UP (+ = outward)"}


def _trim_help(n):
    return TRIM_HELP.get(n) or TRIM_HELP.get(n[2:], "- / +")


def cmd_trim(bus, a):
    """2026-11-15: stand, nudge joints from the keyboard until the robot stands right, then re-zero them so that pose
    reads as the stance. The new zeros live in the servos (EEPROM) and apply to every tool.
    2026-11-19: --joints picks which (up to 2; default the left hip pitch and knee), e.g. --joints L_hip_roll."""
    a.name, a.stay, a.hold, a.speed = "stance", True, False, 1.0
    cmd_play(bus, a)                                     # both legs to the stance and hold
    joints = bus.cfg["joints"]
    names = list(a.joints)
    for n in names:
        if n not in joints:
            sys.exit(f"unknown joint {n}; one of {sorted(joints)}")
    if len(names) > len(TRIM_KEYS):
        sys.exit(f"at most {len(TRIM_KEYS)} joints at a time")
    home = {n: float(STANCE.get(n[2:], 0.0)) for n in names}          # the stance value each joint should read
    tgt = dict(home)
    keys = {}
    print("\nTRIM (robot standing or held; torque on everywhere)")
    for n, ks in zip(names, TRIM_KEYS):
        keys.update({ks[0]: (n, -0.5), ks[1]: (n, +0.5), ks[2]: (n, -0.1), ks[3]: (n, +0.1)})
        print(f"  {ks[0]} / {ks[1]}   {n}: {_trim_help(n)}, 0.5 deg   ({ks[2]} / {ks[3]}: 0.1 deg)")
    print(f"  z       re-zero: this pose becomes the stance ({', '.join(f'{n} {home[n]:g}' for n in names)}); "
          "you'll be asked to hold the robot\n"
          "  q       quit without re-zeroing (keeps holding; `off` releases)")
    other = lambda n: ("R_" if n.startswith("L_") else "L_") + n[2:]          # noqa: E731
    shown = names + [other(n) for n in names if other(n) in joints and other(n) not in names]
    while True:
        rd = {n: fb.counts_to_deg(joints[n], bus.r2(joints[n]["id"], fb.ADDR["PRESENT_POS"])) for n in shown}
        print("\r  trim: " + "  ".join(f"{n} {tgt[n] - home[n]:+5.1f}" for n in names) + " deg   |  reads: "
              + "  ".join(f"{n} {rd[n]:+5.1f}" for n in shown) + "   ", end="", flush=True)
        k = fb.getch()
        if k in keys:
            n, dd = keys[k]
            new = tgt[n] + dd
            if abs(new - home[n]) > 8:
                print("\n  trim limited to +-8 deg from the stance", end="")
                continue
            tgt[n] = new
            fb.goto(bus, n, joints[n], new, speed=int(10 * fb.COUNTS_PER_DEG))
        elif k == "z":
            todo = [n for n in names if abs(tgt[n] - home[n]) > 1e-6]
            if not todo:
                print("\n  nothing trimmed yet", end="")
                continue
            print(f"\n\nHOLD THE ROBOT UP by the pelvis, legs unloaded and still, then press y "
                  f"({', '.join(todo)} go limp for ~0.5 s while the zero is written). Any other key: back.")
            if fb.getch() != "y":
                continue
            for n in todo:
                j = joints[n]
                print(f"{n}: trimmed {tgt[n] - home[n]:+.1f} deg")
                try:
                    fb.cmd_zero(bus, argparse.Namespace(id=j["id"], target=fb.deg_to_counts(j, home[n]), reset=False))
                except SystemExit as e:
                    # 2026-11-15: a (false) failure here used to exit with this joint limp and the robot dropped.
                    # Whatever happened, hold the joint where it is now, then stop.
                    bus.flush()
                    pos = bus.r2(j["id"], fb.ADDR["PRESENT_POS"])
                    bus.w2(j["id"], fb.ADDR["GOAL_POS"], pos)
                    bus.torque(j["id"], True)
                    print(f"\n{n}: {e}\nHOLDING it where it is (torque ON). Run `python feetech_bench.py info` and check its "
                          "offset before anything else moves; `off` releases.")
                    return
                fb.goto(bus, n, j, home[n], speed=int(10 * fb.COUNTS_PER_DEG))   # torque back on, same pose
            print("re-zeroed: the trimmed pose now reads as the stance. Holding it; `off` releases.\n"
                  "Note the new offsets above (also in the servos' EEPROM).")
            return
        elif k in ("q", "\x03"):
            print("\nquit without re-zeroing; still holding (run `off` to release)")
            return


def cmd_shift_zero(bus, a):
    """2026-11-15: apply a known trim to one joint's zero without needing the pose: after it, the angle that read
    (stance + trim) reads the stance. offset += direction * trim (counts). Torque on that servo goes off (it's an EEPROM
    write): have the robot lying down or held, not standing."""
    if not a.enable:
        sys.exit("REFUSED: needs --enable")
    j = bus.cfg["joints"][a.joint]
    sid = j["id"]
    old = fb.decode_offset(bus.r2(sid, fb.ADDR["OFFSET"]))
    new = old + int(round(j["direction"] * a.trim * fb.COUNTS_PER_DEG))
    print(f"{a.joint} (id {sid}): trim {a.trim:+.2f} deg -> offset {old} -> {new}")
    if not a.yes:
        print("robot lying down or held up (this servo goes limp)? y = write, anything else = cancel")
        if fb.getch() != "y":
            sys.exit("cancelled")
    bus.torque(sid, False)
    try:
        bus.w1(sid, fb.ADDR["LOCK"], 0)
        try:
            bus.w2(sid, fb.ADDR["OFFSET"], fb.encode_offset(new))
        except IOError as e:
            print(f"  ({e}; verifying by read-back)")
        time.sleep(0.1)
    finally:
        try:
            bus.w1(sid, fb.ADDR["LOCK"], 1)
        except IOError as e:
            print(f"  ({e})")
    time.sleep(0.4)
    bus.flush()
    for _ in range(6):
        try:
            got, lock = fb.decode_offset(bus.r2(sid, fb.ADDR["OFFSET"])), bus.r1(sid, fb.ADDR["LOCK"])
            if got == new and lock == 1:
                print(f"OK: offset {got}, EEPROM locked; reads {fb.counts_to_deg(j, bus.r2(sid, fb.ADDR['PRESENT_POS'])):.1f} deg now (torque off)")
                return
        except IOError:
            pass
        time.sleep(0.15)
    sys.exit(f"could not confirm: offset reads {got}, lock {lock}; run `feetech_bench.py info` before anything else")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("read")
    sub.add_parser("off", help="torque off on all 8 joints")
    sp = sub.add_parser("play", help="play a whole-body routine (see `list`)")
    sp.add_argument("name", choices=sorted(ROUTINES))
    sp.add_argument("--speed", type=float, default=1.0, help="tempo factor: 0.5 = half speed")
    sp.add_argument("--hold", action="store_true", help="keep torque on at neutral afterwards")
    sp.add_argument("--stay", action="store_true", help="skip the return to neutral and hold the last pose")
    sp.add_argument("--enable", action="store_true")
    sp = sub.add_parser("trim", help="stand, nudge joints from the keyboard, re-zero them (2026-11-15)")
    sp.add_argument("--enable", action="store_true")
    sp.add_argument("--joints", nargs="+", default=["L_hip_pitch", "L_knee"],
                    help="up to 2 joints to trim, e.g. --joints L_hip_roll (default: L_hip_pitch L_knee)")
    sp = sub.add_parser("shift-zero", help="apply a known trim (deg) to one joint's zero, e.g. `shift-zero L_knee --trim -2.5`")
    sp.add_argument("joint")
    sp.add_argument("--trim", type=float, required=True, help="the trim you dialled in (as `trim` printed it)")
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--enable", action="store_true")
    sp = sub.add_parser("stand", help="both legs to the standing pose and hold (release with `off`)")
    sp.add_argument("--speed", type=float, default=1.0)
    sp.add_argument("--enable", action="store_true")
    for sp in (sub.choices["play"], sub.choices["stand"], sub.choices["trim"]):
        sp.add_argument("--current", action="store_true",
                        help="measure motor current on all 8 servos during the motion (summary + CSV)")
        sp.add_argument("--current-log", help="CSV path for --current (default current_<routine>_<time>.csv)")
    argv = sys.argv[1:]
    if argv and argv[0] in ROUTINES:            # shorthand: `biped_bench.py squat ...` = `play squat ...`
        argv = ["play"] + argv
    a = ap.parse_args(argv)
    bus = fb.Bus(merged_cfg())
    {"list": cmd_list, "read": cmd_read, "off": cmd_off, "play": cmd_play, "stand": cmd_stand, "trim": cmd_trim, "shift-zero": cmd_shift_zero}[a.cmd](bus, a)


if __name__ == "__main__":
    main()
