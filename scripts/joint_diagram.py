"""Joint-axis diagram and joint-motion atlas (one servo = one rotation axis).

    python scripts/joint_diagram.py [model_name]

Outputs in assets/renders/<model>/:
  joint_axes.png    - x-ray front/side views with each servo's rotation axis drawn as an arrow
  joint_motion.png  - every joint of the left leg swept between its limits (ghosted extremes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slender_biped.render import Renderer, ortho_scale, stand_data  # noqa: E402  (sets GL env first)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

W, H = 700, 1000
CAM_Z = 0.2
JOINT_STYLE = {  # colour, label, what it does
    "hip_yaw": ("#7b3fb5", "hip yaw", "twists the leg about the vertical (turning)"),
    "hip_roll": ("#1f77b4", "hip roll", "swings the leg sideways (weight shift)"),
    "hip_pitch": ("#d62728", "hip pitch", "swings the leg forward / back"),
    "knee": ("#2ca02c", "knee", "bends the shin back"),
    "ankle": ("#e69500", "ankle", "tilts the tip up / down"),
}


def add_arrow(scene, frm, to, rgba, width=0.0028):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(frm, float), np.asarray(to, float))
    scene.ngeom += 1


def hex_rgba(h, a=1.0):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a]


def axes_figure(m, d, meta, out):
    r = Renderer(m, W, H)
    r.set_groups(shells=True, envelopes=True)
    s = ortho_scale(m, "front_ortho", H)
    imgs, anchors = {}, {}
    saved = m.mat_rgba.copy()
    for name in ("shell", "joint", "accent"):
        m.mat_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name), 3] = 0.22
    for cam in ("front_ortho", "side_ortho"):
        r.r.update_scene(d, camera=cam, scene_option=r.opt)
        for j in meta["joints"]:
            jt = j.split("_", 1)[1]
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            p, a = d.xanchor[jid].copy(), d.xaxis[jid].copy()
            L = 0.06
            add_arrow(r.r.scene, p - a * L * 0.5, p + a * L, hex_rgba(JOINT_STYLE[jt][0]), width=0.0035)
            anchors[(cam, j)] = (p, a)
        imgs[cam] = r.r.render().copy()
    m.mat_rgba[:] = saved
    r.close()

    def px(cam, p):
        horiz = p[1] if cam == "front_ortho" else p[0]
        return W / 2 + horiz / s, H / 2 - (p[2] - CAM_Z) / s

    fig, axs = plt.subplots(1, 3, figsize=(15, 8.2), dpi=130, gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, cam, title in ((axs[0], "front_ortho", "FRONT"), (axs[1], "side_ortho", "SIDE (faces right)")):
        ax.imshow(imgs[cam][40:-30, 60:-60], extent=(60, W - 60, H - 30, 40))
        ax.set_xlim(60, W - 60)
        ax.set_ylim(H - 30, 40)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        for j in meta["joints"]:
            if not j.startswith("L_"):
                continue
            jt = j[2:]
            col, lab, _ = JOINT_STYLE[jt]
            p, a = anchors[(cam, j)]
            x, y = px(cam, p)
            # put labels outboard of the leg, staggered vertically
            dx = 150 if cam == "front_ortho" else 170
            ax.annotate(lab, (x, y), (x + dx, y - 10), color=col, fontsize=10, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.8), va="center")
    ax = axs[2]
    ax.axis("off")
    rows = [jt for jt in JOINT_STYLE if any(j.endswith(jt) for j in meta["joints"])]
    ax.text(0, 0.98, "One servo = one rotation axis", fontsize=13, fontweight="bold", va="top")
    ax.text(0, 0.93, "Arrows show each servo's axis (right-hand rule:\nthumb along the arrow, fingers = positive turn).",
            fontsize=9.5, va="top", color="#444")
    y = 0.82
    servo = meta["servos"]
    for jt in rows:
        col, lab, what = JOINT_STYLE[jt]
        sv = servo[jt]
        lim = meta["cfg"]["joint_limits_deg"][jt]
        ax.add_patch(plt.Rectangle((0, y - 0.012), 0.035, 0.024, color=col, transform=ax.transAxes))
        ax.text(0.05, y, f"{lab}", fontsize=11.5, fontweight="bold", color=col, va="center")
        ax.text(0.05, y - 0.04, what, fontsize=9.5, va="center")
        d_ = sv["dims"]
        ax.text(0.05, y - 0.075, f"{sv['model'].replace('_', ' ')} · {d_['L']*1000:.0f}×{d_['W']*1000:.0f}×{d_['H']*1000:.0f} mm · "
                f"{sv['mass']*1000:.0f} g\nstall {sv['stall_torque']:.2f} N·m · range {lim[0]}° to {lim[1]}°",
                fontsize=8.5, va="top", color="#555", family="monospace")
        y -= 0.165
    ax.text(0, max(y + 0.04, 0.02),
            "Hip yaw + roll + pitch axes meet near one point,\nso together they act like a ball joint (3 DoF).\n"
            "Knee (and ankle) are single hinges (1 DoF).\n\nA dot instead of an arrow = that axis points\nstraight at the viewer (e.g. hip roll seen from\nthe front, hip pitch and knee seen from the side).",
            fontsize=9.5, va="top", style="italic")
    fig.suptitle(f"{meta['name']}: servo rotation axes (orange boxes = servo cases inside the shells)", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def render_pose(m, d, r, cam, q_over):
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos
    for j, v in q_over.items():
        d2.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
    mujoco.mj_forward(m, d2)
    r.r.update_scene(d2, camera=cam, scene_option=r.opt)
    rgb = r.r.render().copy()
    r.r.enable_segmentation_rendering()
    r.r.update_scene(d2, camera=cam, scene_option=r.opt)
    seg = r.r.render().copy()
    r.r.disable_segmentation_rendering()
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mask = (seg[..., 0] >= 0) & (seg[..., 0] != floor) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    return rgb, mask


def motion_figure(m, d, meta, out):
    r = Renderer(m, 760, 800)
    r.set_groups(shells=True)
    views = {"hip_yaw": "front_ortho", "hip_roll": "front_ortho", "hip_pitch": "side_ortho",
             "knee": "side_ortho", "ankle": "side_ortho"}
    jts = [jt for jt in JOINT_STYLE if f"L_{jt}" in meta["joints"]]
    fig, axs = plt.subplots(1, len(jts), figsize=(4.2 * len(jts), 5.6), dpi=130)
    for ax, jt in zip(np.atleast_1d(axs), jts):
        j = f"L_{jt}"
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = m.jnt_range[jid]
        q0 = d.qpos[m.jnt_qposadr[jid]]
        lo_s, hi_s = q0 + 0.85 * (lo - q0), q0 + 0.85 * (hi - q0)
        if jt == "knee":
            lo_s = max(lo_s, 0.02)
        cam = views[jt]
        # yaw barely changes a straight leg's outline: show it with the leg lifted and knee bent
        extra = {"L_hip_pitch": 0.7, "L_knee": 1.5} if jt == "hip_yaw" else {}
        base, _ = render_pose(m, d, r, cam, extra)
        comp = base.astype(float)
        col = np.array(hex_rgba(JOINT_STYLE[jt][0])[:3]) * 255
        for v in (lo_s, hi_s):
            rgb, mask = render_pose(m, d, r, cam, {**extra, j: v})
            tint = 0.55 * rgb[mask] + 0.45 * col
            comp[mask] = 0.45 * comp[mask] + 0.55 * tint
        _, mask0 = render_pose(m, d, r, cam, extra)
        comp[mask0] = base[mask0]
        ax.imshow(comp.astype(np.uint8)[60:-40, 60:-60])
        ax.axis("off")
        ax.set_title(f"{JOINT_STYLE[jt][1]}  ({np.degrees(lo):.0f}° … {np.degrees(hi):.0f}°)",
                     color=JOINT_STYLE[jt][0], fontsize=11, fontweight="bold")
        ax.text(0.5, -0.02, JOINT_STYLE[jt][2] + f"\nview: {cam.replace('_', ' ')}" + (" · leg lifted, knee bent" if extra else ""), transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5)
    r.close()
    fig.suptitle(f"{meta['name']}: left-leg joints swept to ~85% of their limits "
                 "(solid = stance pose, tinted ghosts = the two extremes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "baseline_A_8dof"
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"{name}.xml"))
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text())
    d = stand_data(m)
    out = ROOT / "assets" / "renders" / name
    out.mkdir(parents=True, exist_ok=True)
    axes_figure(m, d, meta, out / "joint_axes.png")
    motion_figure(m, d, meta, out / "joint_motion.png")
    print(out / "joint_axes.png", out / "joint_motion.png")


if __name__ == "__main__":
    main()
