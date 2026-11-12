"""E1 sideways tilt test in simulation, the way it is done by hand: the robot holds its stance pose, a horizontal
"hand" force at the pelvis tilts it slowly sideways (stiff PD on the pelvis roll, 1 deg/s), then lets go.

    .venv/bin/python scripts/tilt_test_sim.py [--flex hip_roll=5 ankle_roll=8] [--friction 1.0] [--floor-tc 0.02]

Reports: the tip angle (the roll at which the hand's force crosses zero: past it the robot falls by itself), the roll at
which the feet start slipping, the foot-centre distance before / after letting go from a given tilt, and the roll at
rest afterwards. Robot (2026-11-12, docs/measuring.md E1): tip 6-7 deg from 85 mm; slip from 4-5 deg; let go from ~7
deg -> 85 -> 60-70 mm and +2.2 deg / -0.7 deg roll at rest.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import exercises_sim as es  # noqa: E402


def feet(env):
    import mujoco
    ids = [mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_tip") for s in "LR"]
    return env.d.geom_xpos[ids].copy()


def tilt_run(model, flex=(), friction=None, floor_tc=0.02, side=+1, release_at=None, rate=1.0, max_deg=16.0,
             kp_hip_roll=2.5, volt=0.68, width=None):
    env = es.make_env(model, flex, volt=volt, kp_hip_roll=kp_hip_roll)
    if friction is not None or floor_tc != 0.02:
        env.cfg.friction = friction; env.cfg.floor_tc = floor_tc
        env.reset()
        env.servo["kp"] = env.servo["kp"] * np.array([kp_hip_roll if j.endswith("hip_roll") else 1.0 for j in env.joints])
    q = es.ep.Q_NOM.copy()
    if width:                                   # abduct both hip rolls by `width` deg, ankles keep the boards flat
        w = math.radians(width); q[1] += w; q[4] -= w; q[6] += w; q[9] -= w
    a = q - env.q_nom
    for _ in range(100):                        # settle 2 s
        env.step(a)
    f0 = feet(env); w0 = f0[0, 1] - f0[1, 1]
    r0, _ = es.true_tilt(env)
    out = dict(w0=1000 * w0, roll0=r0, tip=None, slip=None, curve=[])
    kp, kd = 300.0, 15.0                         # N/rad, N s/rad at the pelvis: a firm hand
    th_t = math.radians(r0); dt = env.cfg.ctrl_dt
    stance = 0 if side > 0 else 1                # tilting left (+) stands on the left board
    fs0 = f0[stance, :2].copy(); neg = 0; cand = None; released = False; t = 0.0
    while t < 40:
        roll, _ = es.true_tilt(env)
        if not released:
            th_t += side * math.radians(rate) * dt
            F = kp * (th_t - math.radians(roll)) + kd * env.d.qvel[3]     # roll rate = -omega_x
            env.d.xfrc_applied[1, 1] = F         # +y pushes the pelvis left: roll + (leaning left)
            if side * F < 0 and abs(roll - r0) > 1:           # the hand holds it back: past the tip, if it lasts 0.5 s
                neg = neg + 1 if neg else 1
                if neg == 1:
                    cand = roll - r0
                if neg == 25 and out["tip"] is None:
                    out["tip"] = cand
            else:
                neg = 0
            fs = feet(env)
            if out["slip"] is None and np.linalg.norm(fs[stance, :2] - fs0) > 0.002:
                out["slip"] = roll - r0
            if abs(roll - r0) >= (release_at if release_at else max_deg):
                released = True; t_rel = t
                env.d.xfrc_applied[1, :] = 0
        elif t > t_rel + 3:
            break
        if int(t / dt) % 25 == 0 and not released:
            out["curve"].append((round(roll - r0, 1), round(float(env.d.xfrc_applied[1, 1]), 2)))
        env.step(a)
        t += dt
        if abs(es.true_tilt(env)[0]) > 40:
            out["fell"] = True; break
    f1 = feet(env)
    out["w1"] = 1000 * (f1[0, 1] - f1[1, 1]); out["roll1"] = es.true_tilt(env)[0]
    out.setdefault("fell", abs(out["roll1"]) > 20)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=es.MODEL)
    ap.add_argument("--flex", nargs="*", default=[])
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--floor-tc", type=float, default=0.02)
    ap.add_argument("--release", type=float, default=6.5)
    a = ap.parse_args()
    fl = tuple((k, float(v)) for k, v in (x.split("=") for x in a.flex))
    for side in (+1, -1):
        r = tilt_run(a.model, fl, a.friction, a.floor_tc, side)
        r2 = tilt_run(a.model, fl, a.friction, a.floor_tc, side, release_at=a.release)
        f = lambda v: "none" if v is None else f"{v:+.1f}"            # noqa: E731
        print(f"{'left ' if side > 0 else 'right'} flex {dict(fl)} fr {a.friction} tc {a.floor_tc}: tip {f(r['tip'])} slip {f(r['slip'])} | "
              f"let go at {a.release}: width {r2['w0']:.0f} -> {r2['w1']:.0f} mm, roll {r2['roll0']:+.1f} -> {r2['roll1']:+.1f}"
              f"{' FELL' if r2['fell'] else ''}")


if __name__ == "__main__":
    main()
