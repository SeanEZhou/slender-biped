"""Render a policy rollout to a README-sized GIF on CPU (llvmpipe), seen from the side.

The seed is given explicitly (default 10000, the first evaluation seed), never cherry-picked.
The camera tracks the pelvis from the side of the initial heading; floor reflections are turned
off because llvmpipe renders them as noise.

    .venv/bin/python scripts/render_gif.py walk_A8_v6b_ft --test walk40 --seconds 12 --out assets/gifs/walk.gif
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# CPU-only: never touch the GPU from evaluation (training owns it). Params unpickling imports JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from slender_biped.render import Renderer, write_video  # noqa: E402  (sets GL env first)
from slender_biped.evaluation import CpuBiped, EvalConfig, NumpyPolicy, load_run, run_episode, yaw_of  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import TESTS  # noqa: E402


class SideTracker:
    """Renderer wrapper: free camera that follows the pelvis (low-pass) from a fixed side azimuth."""

    def __init__(self, renderer: Renderer, model, azimuth_deg, distance, elevation_deg):
        self.r = renderer
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.distance, self.cam.elevation = azimuth_deg, distance, elevation_deg
        self.look = None

    def render(self, data, _camera):
        p = data.xpos[self.bid].copy()
        p[2] = 0.16                      # fixed height: no bobbing
        self.look = p if self.look is None else 0.9 * self.look + 0.1 * p
        self.cam.lookat[:] = self.look
        return self.r.render(data, self.cam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--test", default="stand")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--params", default="params.pkl")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--side", type=float, default=90.0, help="camera azimuth relative to the initial heading")
    a = ap.parse_args()
    cfg, run_dir = load_run(a.run)
    t = TESTS[a.test]
    ec = EvalConfig.from_train(cfg["env"], push_vel=t.get("push", 0.0))
    env = CpuBiped(cfg["env"]["model"], ec, seed=a.seed)
    env.m.mat_reflectance[:] = 0.0
    rw = max(480, a.width)  # render at least at output width so the GIF is never upscaled
    r = Renderer(env.m, rw, rw * 5 // 12 * 2)  # even height for the h264 intermediate
    r.set_groups(shells=True)
    env.reset()
    yaw0 = np.degrees(yaw_of(env.d.qpos[3:7]))
    # run_episode resets with the same seed, so the initial state is the one measured here
    env = CpuBiped(cfg["env"]["model"], ec, seed=a.seed)
    env.m.mat_reflectance[:] = 0.0
    tracker = SideTracker(r, env.m, yaw0 + a.side, 0.95, -12)
    res, frames = run_episode(env, NumpyPolicy(run_dir / a.params), a.seconds or t["seconds"], t["sched"],
                              renderer=tracker, heading_hold=t.get("heading_hold", False))
    r.close()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "clip.mp4"
        write_video(mp4, frames, fps=int(1 / (2 * ec.ctrl_dt)))
        vf = (f"fps={a.fps},scale={a.width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=none:diff_mode=rectangle")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", vf, str(out)], check=True)
    print(out, f"{out.stat().st_size / 1e6:.2f} MB  survived {res['survived_s']:.1f}s fell={res['fell']} "
               f"touchdowns={res['touchdowns']} forward={res['forward_m']:.2f}m")


if __name__ == "__main__":
    main()
