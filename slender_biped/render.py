"""Offscreen rendering helpers.

Rendering is forced onto Mesa llvmpipe (CPU) so it never competes with GPU training jobs.
Set SLENDER_GL=gpu to let MuJoCo pick the default GL backend instead.
"""
from __future__ import annotations

import os

if os.environ.get("SLENDER_GL", "cpu") == "cpu":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402


class Renderer:
    def __init__(self, model, width=800, height=1000):
        self.m = model
        self.r = mujoco.Renderer(model, height, width)
        self.opt = mujoco.MjvOption()

    def set_groups(self, shells=True, envelopes=False, collision=False):
        g = self.opt.geomgroup
        g[:] = 0
        g[0] = shells
        g[1] = envelopes
        g[3] = collision
        g[2] = 1  # world/floor geoms default to group 0 anyway

    def render(self, data, camera, shell_alpha=None):
        saved = None
        if shell_alpha is not None:
            saved = self.m.mat_rgba.copy()
            for name in ("shell", "joint", "accent"):
                mid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
                self.m.mat_rgba[mid, 3] = shell_alpha
        self.r.update_scene(data, camera=camera, scene_option=self.opt)
        img = self.r.render().copy()
        if saved is not None:
            self.m.mat_rgba[:] = saved
        return img

    def close(self):
        self.r.close()


def ortho_scale(model, camera: str, height_px: int) -> float:
    """Metres per pixel for an orthographic camera (fovy = vertical extent in metres)."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    return float(model.cam_fovy[cid]) / height_px


def stand_data(model):
    d = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, d, kid)
    mujoco.mj_forward(model, d)
    return d


def write_video(path, frames, fps=30):
    import imageio.v2 as imageio
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1) as w:
        for f in frames:
            w.append_data(np.asarray(f))
