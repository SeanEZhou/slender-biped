"""Play a reference table open loop on a model (no learning): joint targets = the table at a fixed speed.

    .venv/bin/python scripts/ref_play.py <table> <model> [--v 0.008] [--seconds 16] [--floor-tc 0.004] [--gif out.gif]

Prints per 0.5 s: phase, roll / pitch (deg), lowest point of each foot (mm), pelvis x / y (mm); then falls and steps.
"""
import argparse, math, os, sys
from pathlib import Path
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from slender_biped.evaluation import CpuBiped, EvalConfig, load_run, quat_rot_t  # noqa: E402
from slender_biped.ref_gait import load, lookup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table"); ap.add_argument("model")
    ap.add_argument("--v", type=float, default=0.008)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--floor-tc", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--every", type=float, default=0.25)
    ap.add_argument("--gif")
    a = ap.parse_args()
    tab = load(a.table)
    cfg, _ = load_run("pad_board150w_legs")
    env = CpuBiped(a.model, EvalConfig.from_train(cfg["env"], floor_tc=a.floor_tc, action_scale=1.0, reset_joint_noise=0.0,
                                                  reset_tilt=0.0, dr=a.dr, push_vel=0.0, imu_filter=True, imu_gate=0.15,
                                                  gyro_range=17.45), seed=a.seed)
    env.reset((0.0, 0.0, 0.0))
    dt = env.cfg.ctrl_dt
    frames, rnd = [], None
    if a.gif:
        import mujoco
        from slender_biped.render import Renderer
        env.m.mat_reflectance[:] = 0.0
        rnd = Renderer(env.m, 480, 400); rnd.set_groups(shells=True)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = 0.9, -12, 150
    fell, maxtilt, x0 = None, 0.0, env.d.qpos[0]
    for k in range(int(a.seconds / dt)):
        ph = 2 * math.pi * tab["gait_hz"] * k * dt
        q, z, st = lookup(tab, a.v, ph)
        env.step(np.clip(np.asarray(q) - np.asarray(env.q_nom), -1, 1))
        g = quat_rot_t(env.d.qpos[3:7], np.array([0, 0, -1.0]))
        roll, pitch = math.degrees(math.asin(np.clip(g[1], -1, 1))), math.degrees(math.asin(np.clip(g[0], -1, 1)))
        maxtilt = max(maxtilt, abs(roll), abs(pitch))
        low, c = env.tip_low()
        if fell is None and env.fallen():
            fell = (k + 1) * dt
        if not a.quiet and k % max(1, int(a.every / dt)) == 0:
            print(f"t {k * dt:5.2f} u {(ph / 2 / math.pi) % 1:.3f} roll {roll:+5.1f} pitch {pitch:+5.1f} "
                  f"low L {1000 * low[0]:5.1f} R {1000 * low[1]:5.1f} (ref {1000 * z[0]:4.1f} {1000 * z[1]:4.1f}) "
                  f"pelvis x {1000 * (env.d.qpos[0] - x0):+6.1f} y {1000 * env.d.qpos[1]:+5.1f}", flush=True)
        if rnd is not None and k % 2 == 0:
            cam.lookat[:] = [env.d.qpos[0], env.d.qpos[1], 0.17]
            frames.append(rnd.render(env.d, cam))
    print(f"{a.table} on {a.model} v {a.v} tc {a.floor_tc}: {'FELL at %.2f s' % fell if fell else 'up'}, max tilt "
          f"{maxtilt:.1f} deg, forward {100 * (env.d.qpos[0] - x0):+.1f} cm")
    if rnd is not None:
        import subprocess, tempfile, shutil
        from slender_biped.render import write_video
        rnd.close()
        with tempfile.TemporaryDirectory() as tmp:
            mp4 = Path(tmp) / "c.mp4"; write_video(mp4, frames, fps=25)
            ff = shutil.which("ffmpeg") or __import__("imageio_ffmpeg").get_ffmpeg_exe()
            vf = "fps=25,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none"
            subprocess.run([ff, "-v", "error", "-y", "-i", str(mp4), "-vf", vf, a.gif], check=True)
        print("wrote", a.gif)


if __name__ == "__main__":
    main()
