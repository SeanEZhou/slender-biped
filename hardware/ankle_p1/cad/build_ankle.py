"""Build, check and render the ankle-roll shin and foot.

    .venv-cad/bin/python hardware/ankle_p1/cad/build_ankle.py            # everything
    .venv-cad/bin/python hardware/ankle_p1/cad/build_ankle.py export     # STL / 3MF / STEP (+ right-leg mirrors, print-ready)
    .venv-cad/bin/python hardware/ankle_p1/cad/build_ankle.py check      # interference over the ankle roll and knee range, mass
    .venv-cad/bin/python hardware/ankle_p1/cad/build_ankle.py render

Outputs: hardware/ankle_p1/{exports,renders}, hardware/ankle_p1/verification_report.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "leg_p0" / "cad"))

import numpy as np  # noqa: E402
from build123d import export_step, export_stl, Mesher, Compound, Location  # noqa: E402

import ankle  # noqa: E402
import build as leg_build  # noqa: E402  (leg_p0 renderer and overlap check)

ROOT = HERE.parent
EXP, REN = ROOT / "exports", ROOT / "renders"
STEMS = {"shin_ankle": "A01_shin_ankle", "foot_ankle": "A02_foot_ankle"}
# print orientation: the part direction that faces DOWN onto the bed
PRINT_DOWN = {"shin_ankle": (0, 1, 0),                                  # on its side, outer knee arm on the bed
              "foot_ankle": (math.sin(math.radians(ankle.ANKLE["fork_pitch"])), 0,
                             -math.cos(math.radians(ankle.ANKLE["fork_pitch"])))}   # sole on the bed, socket open upward
BED_CENTER = (128.0, 128.0)
STANCE = dict(ankle.REST)          # v2 rest pose: the leg straight


def export_all(items):
    import trimesh
    for d in ("stl", "stl_right_mirror", "3mf", "step", "stl/print_ready"):
        (EXP / d).mkdir(parents=True, exist_ok=True)
    for it in items:
        if not it.printable:
            continue
        stem = STEMS[it.name]
        export_stl(it.shape, str(EXP / "stl" / f"{stem}.stl"), tolerance=0.02, angular_tolerance=0.15)
        export_step(it.shape, str(EXP / "step" / f"{stem}.step"))
        m = Mesher()
        m.add_shape(it.shape, linear_deflection=0.02, angular_deflection=0.15)
        m.write(str(EXP / "3mf" / f"{stem}.3mf"))
        export_stl(ankle.leg.mirror_right(it.shape), str(EXP / "stl_right_mirror" / f"{stem}_RIGHT.stl"),
                   tolerance=0.02, angular_tolerance=0.15)
        down = PRINT_DOWN[it.name]
        for src, dn, out in ((EXP / "stl" / f"{stem}.stl", down, f"{stem}_print.stl"),
                             (EXP / "stl_right_mirror" / f"{stem}_RIGHT.stl", (down[0], -down[1], down[2]), f"{stem}_RIGHT_print.stl")):
            tm = trimesh.load(src)
            tm.apply_transform(trimesh.geometry.align_vectors(dn, (0, 0, -1)))
            c = tm.bounds.mean(axis=0)
            tm.apply_translation((BED_CENTER[0] - c[0], BED_CENTER[1] - c[1], -tm.bounds[0][2]))
            tm.export(EXP / "stl" / "print_ready" / out)
    for tag, q in (("rest", STANCE), ("old_stance", ankle.leg.STANCE)):
        posed = ankle.pose([i for i in items if i.kind in ("part", "servo", "horn")], q)
        export_step(Compound([i.shape for i in posed]), str(EXP / "step" / f"ankle_assembly_{tag}.step"))


def check_all(items):
    rep = {"ankle_axis_point_mm": [round(v, 2) for v in (ankle.A.X, ankle.A.Y, ankle.A.Z)],
           "ankle_axis_dir": list(ankle.AXIS[1]),
           "ankle_axis_from_knee_mm": {"x": round(ankle.A.X, 2), "z": round(ankle.A.Z - ankle.Z_KNEE, 2)},
           "ankle_axis_above_sole_mm": round(ankle.ANKLE["axis_h"], 2),
           "knee_to_sole_mm": ankle.ANKLE["leg_len"],
           "params": ankle.ANKLE}
    t = time.time()
    near = [i for i in items if i.name in ("thigh", "shin_ankle", "foot_ankle") or i.name.endswith(("ankle_roll", "_knee"))]
    for tag, q in (("rest", STANCE), ("old_stance", ankle.leg.STANCE)):
        rep[f"collisions_{tag}"] = leg_build.collisions(ankle.pose(near, q))
        print(tag, rep[f"collisions_{tag}"], flush=True)
    # ankle roll sweep (foot vs everything on the shin), in the rest pose
    roll = {}
    for sgn in (1, -1):
        free, first = 0, None
        for a in range(5, 46, 5):
            hits = leg_build.collisions(ankle.pose(near, dict(STANCE, ankle_roll=sgn * a)), moving_links={"foot"})
            if hits:
                first = (sgn * a, hits[:3])
                break
            free = sgn * a
        roll["pos" if sgn > 0 else "neg"] = {"collision_free_to": free, "first_hit": first}
        print("ankle roll", roll, flush=True)
    rep["ankle_roll_rom"] = roll
    # knee flexion with the new shin and foot (thigh vs shin/foot), ankle at 0 and at +-20
    knee = {}
    for ar in (0, 20, -20):
        free, first = 0, None
        for k in range(10, 131, 10):
            hits = leg_build.collisions(ankle.pose(near, {"knee": k, "ankle_roll": ar}), moving_links={"shin", "foot"})
            if hits:
                first = (k, hits[:3])
                break
            free = k
        knee[f"ankle_{ar}"] = {"collision_free_to": free, "first_hit": first}
        print("knee", ar, knee[f"ankle_{ar}"], flush=True)
    rep["knee_rom"] = knee
    # foot level at stance and ankle height
    foot = [i for i in items if i.name == "foot_ankle"][0]
    st = ankle.pose([foot], STANCE)[0].shape
    bottom = min(st.faces(), key=lambda f: f.center().Z)
    rep["sole_normal_at_rest"] = [round(v, 3) for v in bottom.normal_at()]
    dens = ankle.P["print"]["density_g_cm3"] * ankle.P["print"]["fill_factor"]
    old = {"shin": ankle.leg.part_shin().volume, "pad_F_board150w_g": 25.0}
    rep["mass_g"] = {"shin_ankle": round(items[0].shape.volume / 1000 * dens, 1),
                     "foot_ankle": round(items[1].shape.volume / 1000 * dens, 1),
                     "old_P06_shin": round(old["shin"] / 1000 * dens, 1), "old_F_board150w_pad": 25.0,
                     "servo": ankle.S["mass_g"], "horns_screws_est": 8.0,
                     "note": f"printed = solid volume x {ankle.P['print']['density_g_cm3']} g/cm3 x fill {ankle.P['print']['fill_factor']}"}
    m = rep["mass_g"]
    m["added_per_leg"] = round(m["shin_ankle"] + m["foot_ankle"] + m["servo"] + m["horns_screws_est"]
                               - m["old_P06_shin"] - m["old_F_board150w_pad"], 1)
    import trimesh
    rep["meshes"] = {}
    for f in sorted((EXP / "stl").glob("A*.stl")) + sorted((EXP / "stl" / "print_ready").glob("A*.stl")) + sorted((EXP / "stl_right_mirror").glob("A*.stl")):
        tm = trimesh.load(f)
        rep["meshes"][f.name] = {"watertight": bool(tm.is_watertight), "volume_cm3": round(tm.volume / 1000, 2),
                                 "bbox_mm": [round(v, 1) for v in tm.extents]}
    rep["check_seconds"] = round(time.time() - t, 1)
    (ROOT / "verification_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def render_views(posed, views, size=(700, 1000), center=(0.0, 0.03, -0.17)):
    """views: (name, (azimuth, elevation), distance m). Perspective only: MuJoCo 4's orthographic camera renders
    these scenes blank (leg_p0's renderer included)."""
    import tempfile
    import mujoco
    with tempfile.TemporaryDirectory() as tmp:
        m = mujoco.MjModel.from_xml_string(leg_build._scene_xml(posed, tmp))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, size[1], size[0])
        out = {}
        for name, (az, el), dist in views:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = center
            cam.azimuth, cam.elevation, cam.distance = az, el, dist
            m.vis.global_.fovy = 30
            r.update_scene(d, camera=cam)
            out[name] = r.render().copy()
        r.close()
    return out


def render_all(items):
    import imageio.v2 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REN.mkdir(parents=True, exist_ok=True)
    vis = [i for i in items if i.kind in ("part", "servo", "horn")]
    lower = [i for i in vis if i.link in ("thigh", "shin", "foot")]
    ctr = ankle.A
    # 1. whole leg in the stance pose: front, side, three-quarter
    st = ankle.pose(vis, STANCE)
    views = [("front", (180, -5), 1.0), ("side", (90, -5), 1.0), ("three_quarter", (135, -15), 1.0)]
    imgs = render_views(st, views, center=(0.03, 0.03, -0.2))
    fig, axs = plt.subplots(1, 3, figsize=(14, 7), dpi=110)
    for ax, (k, title) in zip(axs, (("front", "front"), ("side", "side (faces left)"), ("three_quarter", "three-quarter"))):
        ax.imshow(imgs[k]); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle("P0 leg with ankle roll, rest pose (hip 16°, knee 32°): A01 straight shin with a 16° roll fork, A02 foot with the servo socket", fontsize=11)
    fig.tight_layout(); fig.savefig(REN / "ankle_leg_rest.png"); plt.close(fig)
    # 2. close-ups of the ankle at roll -20 / 0 / +20 (front view, stance)
    fig, axs = plt.subplots(1, 3, figsize=(13, 5), dpi=110)
    for ax, ar in zip(axs, (-20, 0, 20)):
        posed = ankle.pose(lower, dict(STANCE, ankle_roll=ar))
        c = ankle.leg.link_transform(Location((ctr.X, ctr.Y, ctr.Z)) * __import__("build123d").Box(1, 1, 1), "shin", STANCE).center()
        im = render_views(posed, [("c", (180, -8), 0.32)], size=(600, 560), center=(c.X / 1000, c.Y / 1000, c.Z / 1000 + 0.01))["c"]
        ax.imshow(im); ax.axis("off"); ax.set_title(f"ankle roll {ar:+d}°", fontsize=10)
    fig.suptitle("Ankle roll, front view (rest pose): the foot holds the servo; the shin's fork turns on its main horn (front) and idler horn (back)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(REN / "ankle_roll_front.png"); plt.close(fig)
    # 3. ankle close-up, three-quarter from behind (connectors, case screws) and exploded
    c = ankle.leg.link_transform(Location((ctr.X, ctr.Y, ctr.Z)) * __import__("build123d").Box(1, 1, 1), "shin", STANCE).center()
    cc = (c.X / 1000, c.Y / 1000, c.Z / 1000 + 0.02)
    posed = ankle.pose(lower, STANCE)
    im1 = render_views(posed, [("c", (-40, -15), 0.38)], size=(700, 700), center=cc)["c"]
    expl = []
    for it in ankle.pose(lower, STANCE):
        dz = {"servo_ankle_roll": -30, "foot_ankle": -60, "casescrews_ankle_roll": -60, "keepout_ankle_roll": -60}.get(it.name, 0)
        if it.link == "thigh":
            continue
        expl.append(ankle.Item(it.name, it.shape.moved(Location((0, 0, dz))), it.link, it.printable, it.color, it.kind))
    im2 = render_views(expl, [("c", (150, -15), 0.5)], size=(700, 900), center=(cc[0], cc[1], cc[2] - 0.03))["c"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 7), dpi=110)
    axs[0].imshow(im1); axs[0].set_title("rear three-quarter: fork arm on the idler horn, connector window, M2 case screws", fontsize=10)
    axs[1].imshow(im2); axs[1].set_title("exploded: shin with fork / servo + horns / foot with socket", fontsize=10)
    for ax in axs:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(REN / "ankle_closeup.png"); plt.close(fig)
    print("renders in", REN)


def export_sim_meshes(items):
    """Visual meshes for the simulator (models/meshes/ankle2/{L,R}_{shin,foot}.stl): body frames, metres. Shin frame:
    origin on the knee axis, leg straight; foot frame: origin on the ankle axis."""
    import trimesh
    out = ROOT.parents[1] / "models" / "meshes" / "ankle2"
    out.mkdir(parents=True, exist_ok=True)
    groups = {"shin": [i for i in items if i.link == "shin" and i.kind in ("part", "horn")],
              "foot": [i for i in items if i.link == "foot" and i.kind == "servo"]}   # pad + socket: model primitives
    for g, its in groups.items():
        meshes = []
        for it in its:
            f = EXP / f"_tmp_{it.name}.stl"
            export_stl(it.shape, str(f), tolerance=0.1, angular_tolerance=0.3)
            meshes.append(trimesh.load(f)); f.unlink()
        m = trimesh.util.concatenate(meshes)
        org = (0.0, ankle.CY, ankle.Z_KNEE) if g == "shin" else (ankle.A.X, ankle.A.Y, ankle.A.Z)
        m.apply_translation([-v for v in org]); m.apply_scale(0.001)
        if g == "foot" and ankle.ANKLE["fork_pitch"]:     # the sim's foot frame is the pitched one: undo the pitch
            m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(-ankle.ANKLE["fork_pitch"]), [0, -1, 0]))
        m.export(out / f"L_{g}.stl")
        r = m.copy(); r.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0])); r.invert()
        r.export(out / f"R_{g}.stl")
        print(g, m.bounds.round(4).tolist())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    parts = ankle.build_items()
    if what in ("all", "export"):
        export_all(parts)
        print("exported to", EXP)
    if what == "simmesh":
        export_sim_meshes(parts)
    full = [i for i in ankle.leg_items()]
    if what in ("all", "check"):
        rep = check_all(parts + [i for i in full if i.link == "thigh" or i.name.endswith("_knee")])
        print(json.dumps({k: rep[k] for k in ("ankle_axis_from_knee_mm", "collisions_rest", "collisions_old_stance",
                                               "ankle_roll_rom", "knee_rom", "sole_normal_at_rest", "mass_g")}, indent=1, default=str))
    if what in ("all", "render"):
        render_all(full)
