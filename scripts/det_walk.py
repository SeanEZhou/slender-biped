"""Run the deterministic (non-learning) walker experiments on the F_board150w pads, to watch them or to measure them.

    .venv/bin/python scripts/det_walk.py <controller> [--floor-tc 0.3] [--view] [--gif out.gif] [--set k=v ...]

Controllers (joint offsets from the stance pose, radians; no learning anywhere):

  lift   the simplest step: slowly lift the left board 2 cm and hold it. Shows the core problem: the centre of mass is
         15.5 mm outside the right board's inner edge, so the robot tips toward the lifted foot.   params: lift, t_up
  sine   open-loop differential hip roll, a*sin(2 pi f t): the side-to-side rock that lifts a board by itself.
                                                                                                   params: a, f
  cpg    the same rock with its amplitude regulated on the IMU roll (more drive when the rock is small, less when big).
                                                                                                   params: f, A, a0, k
  step   one quick step with the right leg from the stance pose, then hold: lands `fwd` m ahead, heel first,
         optionally rolled outward by `out` rad. With shift > 0 the weight moves over the left board first (hip
         rolls: left inward, right outward, by `shift` rad over t_shift s; the legs lean like a parallelogram, the
         pelvis stays level) and `release` (0..1) blends the shift out during the swing.
                                     params: fwd, out, T, lift_knee, lift_hip, t0, leg, shift, t_shift, release
  rock   slender_biped/rock_walk.py: rock + swing the unloaded leg + pitch feedback (its DEFAULT params; --set any key).

Without --view / --gif it runs every floor from hard (0.004) to the softest rug (1.2) and prints a table. With --view it
opens the MuJoCo viewer (real time, or slower with --slow 2) on one floor; the episode keeps going after a "fall" so you
can see what happens. Floors (--floor-tc): 0.004 hard, 0.1 / 0.3 rug (board sinks ~1 mm), 0.8 (~6 mm), 1.2 (~12 mm).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run, quat_rot_t  # noqa: E402
from slender_biped.rock_walk import CONTROLLERS  # noqa: E402

BASE_RUN = {"hw_pad_board150w_legs_8dof": "pad_board150w_legs", "hw_pad_board150w_8dof": "pad_board150w"}
FLOORS = (0.004, 0.1, 0.3, 0.8, 1.2)
DT = 0.02


CTL = CONTROLLERS


def make_env(model, tc, seed, dr, imu=True):
    cfg, _ = load_run(BASE_RUN[model])
    over = dict(floor_tc=tc, action_scale=1.0, reset_joint_noise=0.0, reset_tilt=0.0, dr=dr, push_vel=0.0)
    if imu:
        over.update(imu_filter=True, imu_gate=0.15, gyro_range=17.45)
    return CpuBiped(model, EvalConfig.from_train(cfg["env"], **over), seed=seed)


def run(ctl, env, seconds, on_step=None, stop_on_fall=True):
    obs = env.reset((0.0, 0.0, 0.0))
    x0 = env.d.qpos[0].copy()
    c0 = env.tip_low()[1].copy()
    roll, lift, pitch = [], [], []
    fell_at = None
    for k in range(int(seconds / DT)):
        obs, _ = env.step(np.clip(ctl(obs), -1, 1))
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        roll.append(math.degrees(math.asin(np.clip(g[1], -1, 1))))
        pitch.append(math.degrees(math.asin(np.clip(g[0], -1, 1))))
        lift.append(1000 * env.tip_low()[0])
        if fell_at is None and env.fallen():
            fell_at = (k + 1) * DT
        if on_step:
            on_step(k, roll[-1], lift[-1], fell_at)
        if fell_at is not None and stop_on_fall:
            break
    lift = np.array(lift); late = slice(len(roll) // 3, None)
    return dict(fell_at=fell_at, roll_pp=float(np.ptp(roll[late])) if len(roll) > 5 else 0.0,
                roll_max=float(np.max(np.abs(roll))), liftL=float(lift[:, 0].max()), liftR=float(lift[:, 1].max()),
                fwd=float(env.d.qpos[0] - x0), pitch_max=float(np.max(np.abs(pitch))),
                board_dx=(1000 * (env.tip_low()[1][:, 0] - c0[:, 0])).round(1).tolist(),
                end_roll=roll[-1], end_pitch=pitch[-1])


def fmt(r, seconds):
    s = f"FELL at {r['fell_at']:4.1f} s" if r["fell_at"] else f"up {seconds:.0f} s     "
    return (f"{s}  |roll| max {r['roll_max']:4.1f} deg  rock p-p {r['roll_pp']:4.1f} deg  "
            f"lift L/R {r['liftL']:4.1f}/{r['liftR']:4.1f} mm  fwd {100 * r['fwd']:+5.1f} cm  |pitch| max {r['pitch_max']:4.1f}"
            f"  boards moved L/R {r['board_dx'][0]:+5.1f}/{r['board_dx'][1]:+5.1f} mm  end roll/pitch {r['end_roll']:+.1f}/{r['end_pitch']:+.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("controller", choices=list(CTL))
    ap.add_argument("--set", nargs="*", default=[], help="controller params, k=v")
    ap.add_argument("--model", default="hw_pad_board150w_legs_8dof", choices=list(BASE_RUN), help="legs only / full body")
    ap.add_argument("--floor-tc", type=float, help="one floor only (default: all five)")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", action="store_true", help="servo / mass randomization as in training")
    ap.add_argument("--no-imu", action="store_true", help="perfect orientation instead of the robot's IMU filter")
    ap.add_argument("--view", action="store_true", help="MuJoCo viewer (needs a display; WSLg works)")
    ap.add_argument("--slow", type=float, default=1.0, help="viewer slow-motion factor")
    ap.add_argument("--gif", help="write a GIF of the run (one floor)")
    a = ap.parse_args()
    params = {}
    for kv in a.set:
        k, v = kv.split("="); params[k] = float(v)
    mk = lambda: CTL[a.controller](params)  # noqa: E731

    if a.view or a.gif:
        tc = a.floor_tc if a.floor_tc is not None else 0.3
        env = make_env(a.model, tc, a.seed, a.dr, not a.no_imu)
        print(f"{a.controller} {params or '(defaults)'} on floor tc {tc}")
        def report(k, roll, lift, fell_at, _st={"said": False}):
            if k % 25 == 0:
                print(f"  t {k * DT:5.2f} s  roll {roll:+6.1f} deg  lift L {lift[0]:5.1f}  R {lift[1]:5.1f} mm", flush=True)
            if fell_at and not _st["said"]:
                _st["said"] = True
                print(f"  *** fall condition at {fell_at:.2f} s (roll {roll:+.1f} deg)", flush=True)
        if a.view:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(env.m, env.d) as v:
                v.cam.distance, v.cam.elevation, v.cam.azimuth = 0.9, -12, 180   # from the front: roll is visible
                t_wall = time.time()
                def step(k, roll, lift, fell_at):
                    report(k, roll, lift, fell_at)
                    v.cam.lookat[:] = [env.d.qpos[0], env.d.qpos[1], 0.15]
                    v.sync()
                    dt = t_wall + (k + 1) * DT * a.slow - time.time()
                    if dt > 0:
                        time.sleep(dt)
                    if not v.is_running():
                        raise SystemExit
                r = run(mk(), env, a.seconds, step, stop_on_fall=False)
        else:
            from slender_biped.render import Renderer, write_video
            import subprocess, tempfile, shutil, mujoco
            env.m.mat_reflectance[:] = 0.0
            rnd = Renderer(env.m, 480, 400); rnd.set_groups(shells=True)
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance, cam.elevation, cam.azimuth = 0.9, -12, 180
            frames = []
            def step(k, roll, lift, fell_at):
                report(k, roll, lift, fell_at)
                if k % 2 == 0:
                    cam.lookat[:] = [env.d.qpos[0], env.d.qpos[1], 0.15]
                    frames.append(rnd.render(env.d, cam))
            r = run(mk(), env, a.seconds, step, stop_on_fall=False)
            rnd.close()
            with tempfile.TemporaryDirectory() as tmp:
                mp4 = Path(tmp) / "c.mp4"; write_video(mp4, frames, fps=25)
                ff = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
                vf = "fps=25,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none"
                subprocess.run([ff, "-v", "error", "-y", "-i", str(mp4), "-vf", vf, a.gif], check=True)
            print("wrote", a.gif)
        print(fmt(r, a.seconds))
        return

    print(f"{a.controller} {params or '(defaults)'}  {a.model}  {a.seconds:.0f} s, seed {a.seed}"
          f"{', randomized' if a.dr else ''}{', perfect IMU' if a.no_imu else ''}")
    for tc in ([a.floor_tc] if a.floor_tc is not None else FLOORS):
        r = run(mk(), make_env(a.model, tc, a.seed, a.dr, not a.no_imu), a.seconds)
        print(f"  floor {tc:5.3f}: {fmt(r, a.seconds)}", flush=True)


if __name__ == "__main__":
    main()
