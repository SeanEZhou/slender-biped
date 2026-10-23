"""Appearance review gate: renders, dimensioned silhouette, packaging + mass report.

    python scripts/inspect_model.py                  # all generated models + comparison sheet
    python scripts/inspect_model.py baseline_B_10dof
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slender_biped.render import Renderer, ortho_scale  # noqa: E402  (sets GL env first)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from slender_biped.morphology import dump, link_inertials, mass_table, measure  # noqa: E402

RENDERS = ROOT / "assets" / "renders"
RESULTS = ROOT / "results"
W, H = 700, 1000
CAM_Z = 0.2


def mark_bad_envelopes(m, report):
    for p in report["packaging"]:
        if not p["ok"]:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, p["envelope"])
            m.geom_rgba[gid] = [1, 0.05, 0.05, 0.9]


def dimensioned(m, img_front, img_side, rep, path):
    s = ortho_scale(m, "front_ortho", H)
    fig, axs = plt.subplots(1, 2, figsize=(12, 8.6), dpi=130)

    def px_y(y):
        return W / 2 + y / s

    def px_z(z):
        return H / 2 - (z - CAM_Z) / s

    def hdim(ax, x0, x1, z, label, off=0):
        r = px_z(z) + off
        ax.annotate("", (px_y(x0), r), (px_y(x1), r), arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.3))
        ax.text(px_y((x0 + x1) / 2), r - 8, label, color="#c0392b", ha="center", va="bottom", fontsize=9,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    def vdim(ax, x, z0, z1, label, side="right"):
        c = px_y(x)
        ax.annotate("", (c, px_z(z0)), (c, px_z(z1)), arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.3))
        ax.text(c + (6 if side == "right" else -6), px_z((z0 + z1) / 2), label, color="#1f4e79",
                ha="left" if side == "right" else "right", va="center", fontsize=9, rotation=90,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))

    ax = axs[0]
    ax.imshow(img_front)
    hw = rep["hip_width_m"]
    hdim(ax, -hw / 2, hw / 2, rep["pelvis_origin_height_m"] + 0.002,
         f"hip width {hw*1000:.0f} mm ({100*rep['hip_width_ratio']:.1f}% of H)")
    vdim(ax, -0.085, 0, rep["height_body_m"], f"body height {rep['height_body_m']*1000:.0f} mm", "left")
    sw = rep["shin_max_width_front_m"]
    y_shin = rep["hip_spacing_m"] / 2
    hdim(ax, y_shin - sw / 2, y_shin + sw / 2, 0.12, f"shin {sw*1000:.0f} mm\n({100*rep['shin_width_ratio']:.1f}%)")
    tw = rep["thigh_max_width_m"]
    hdim(ax, -y_shin - tw / 2, -y_shin + tw / 2, rep["pelvis_origin_height_m"] - 0.115, f"thigh {tw*1000:.0f}")
    vdim(ax, 0.075, 0, rep["pelvis_origin_height_m"], f"pelvis origin {rep['pelvis_origin_height_m']*1000:.0f} mm")
    ax.set_title("FRONT (orthographic)", fontsize=11)
    ax.axis("off")

    ax = axs[1]
    ax.imshow(img_side)
    knee_z = rep["knee_z_m"]
    hip_z = rep["hip_pitch_z_m"]
    vdim(ax, 0.07, 0, knee_z, f"shin {rep['shin_length_m']*1000:.0f} (axis-tip)")
    vdim(ax, 0.07, knee_z, hip_z, f"thigh {rep['thigh_length_m']*1000:.0f}")
    fp = rep["footprint"]
    ax.text(W * 0.03, H * 0.97, f"tip: {fp['type']}  {fp['length_mm']:.0f} x {fp['width_mm']:.0f} mm, "
            f"sagittal R {fp['sagittal_radius_mm']:.0f} mm, rollable arc (+/-20deg) {fp['rollable_arc_mm_pm20deg']:.0f} mm",
            fontsize=8.5, color="black", bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.set_title("SIDE (orthographic, robot faces right)", fontsize=11)
    ax.axis("off")
    fig.suptitle(f"{rep['model']}  |  mass {rep['total_mass_kg']*1000:.0f} g  |  "
                 f"ortho scale {s*1000:.2f} mm/px  |  stance pose (hip 16 deg, knee 32 deg)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def inspect(name):
    m = mujoco.MjModel.from_xml_path(str(ROOT / "models" / f"{name}.xml"))
    meta = json.loads((ROOT / "models" / f"{name}.json").read_text())
    rep, d = measure(m, meta)
    for nm, key in (("L_shin", "knee_z_m"), ("L_thigh", "hip_pitch_z_m")):
        rep[key] = float(d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)][2])
    by_cat, by_body = mass_table(meta)
    rep["mass_by_category_kg"] = by_cat
    rep["mass_by_body_kg"] = by_body
    rep["links"] = link_inertials(m)
    mark_bad_envelopes(m, rep)

    out = RENDERS / name
    out.mkdir(parents=True, exist_ok=True)
    r = Renderer(m, W, H)
    imgs = {}
    r.set_groups(shells=True)
    for cam in ("front_ortho", "side_ortho", "back_ortho", "three_quarter", "three_quarter_back"):
        imgs[cam] = r.render(d, cam)
        Image.fromarray(imgs[cam]).save(out / f"{cam}.png")
    r.set_groups(shells=True, envelopes=True)
    for cam in ("front_ortho", "side_ortho"):
        Image.fromarray(r.render(d, cam, shell_alpha=0.22)).save(out / f"xray_{cam}.png")
    r.set_groups(shells=False, collision=True)
    for cam in ("front_ortho", "side_ortho"):
        Image.fromarray(r.render(d, cam)).save(out / f"collision_{cam}.png")
    r.close()
    dimensioned(m, imgs["front_ortho"], imgs["side_ortho"], rep, out / "silhouette_dimensioned.png")
    RESULTS.mkdir(exist_ok=True)
    dump(rep, RESULTS / f"morphology_{name}.json")
    return rep


def sheet(names, path, cams=("front_ortho", "side_ortho", "three_quarter")):
    """Grid: one row per camera, one column per model (same ortho scale across columns)."""
    fig, axs = plt.subplots(len(cams), len(names), figsize=(2.3 * len(names), 3.3 * len(cams)), dpi=140,
                            squeeze=False)
    for j, n in enumerate(names):
        for i, c in enumerate(cams):
            im = np.asarray(Image.open(RENDERS / n / f"{c}.png"))[60:-20, 110:-110]
            axs[i, j].imshow(im)
            axs[i, j].axis("off")
        axs[0, j].set_title(n, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    names = sys.argv[1:] or sorted(p.stem for p in (ROOT / "models").glob("*_*dof.xml"))
    reps = {}
    for n in names:
        rep = inspect(n)
        reps[n] = rep
        bad = [p["envelope"] for p in rep["packaging"] if not p["ok"]]
        print(f"{n:24s} m={rep['total_mass_kg']*1e3:5.0f}g H={rep['height_body_m']*1e3:4.0f}mm "
              f"hip={rep['hip_width_m']*1e3:4.0f}mm ({rep['hip_width_ratio']*100:4.1f}%) "
              f"shin={rep['shin_max_width_front_m']*1e3:3.0f}mm ({rep['shin_width_ratio']*100:4.1f}%) "
              f"thigh={rep['thigh_max_width_m']*1e3:3.0f}mm COMz={rep['com_height_m']*1e3:4.0f}mm "
              f"pack={'OK' if not bad else 'VIOLATION ' + ','.join(bad)} contacts={len(rep['stand_contacts'])}")
    if len(names) > 1:
        sheet(names, RENDERS / "gate_A_variants.png")
        dump({n: {k: v for k, v in r.items() if k not in ("links", "packaging", "stand_contacts")}
              for n, r in reps.items()}, RESULTS / "morphology_summary.json")


if __name__ == "__main__":
    main()
