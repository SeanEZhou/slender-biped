"""Build, export, render and check the P0 leg.

    .venv-cad/bin/python hardware/leg_p0/cad/build.py            # everything
    .venv-cad/bin/python hardware/leg_p0/cad/build.py export     # STL/3MF/STEP only
    .venv-cad/bin/python hardware/leg_p0/cad/build.py render     # renders only
    .venv-cad/bin/python hardware/leg_p0/cad/build.py check      # interference + ROM + mass report

Outputs go to hardware/leg_p0/{exports,renders} and hardware/leg_p0/verification_report.json.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from build123d import export_step, export_stl, Mesher, Compound  # noqa: E402

import leg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exports"
REN = ROOT / "renders"
PRINTED = ["pelvis_plate", "clamp_tongue", "yaw_bracket", "roll_bracket", "thigh", "shin", "pad"]
PART_NO = {n: f"P{i + 1:02d}" for i, n in enumerate(PRINTED)}
# print orientation: the part axis that faces DOWN onto the bed (README "Printing"); others are added as we print them
_PAD_TILT = np.radians(leg.P["pad"]["stance_pitch"])
PRINT_DOWN = {"pelvis_plate": (0, 0, 1),     # plate face down, cup walls up (the cup floor bridges ~26 mm)
              "clamp_tongue": (0, 0, -1),    # flat, ribs up, nut traps open on the bed
              "yaw_bracket": (1, 0, 0),      # front web on the bed
              "roll_bracket": (-1, 0, 0),    # pitch-cradle back face on the bed, arms flat: horn holes print round
              "thigh": (0, 1, 0),            # lying on its side, one fork arm on the bed: layers run along the
                                             # tube (strong in bending), all horn/case holes print round
              "shin": (0, 1, 0),             # same as the thigh: one fork arm on the bed, tube along the bed
              "pad": (np.sin(_PAD_TILT), 0, -np.cos(_PAD_TILT))}   # pitched contact face on the bed: no
                                             # supports, and the face that touches the ground is the smoothest
BED_CENTER = (128.0, 128.0)                  # Bambu A1 (256 x 256)


# ----------------------------------------------------------------------------- export
def export_print_ready(stems):
    """Copies of the left-leg STLs rotated into print orientation, sitting on the bed at its centre."""
    import trimesh
    d = EXP / "stl" / "print_ready"
    d.mkdir(parents=True, exist_ok=True)
    jobs = [(EXP / "stl" / f"{stems[n]}.stl", down, f"{stems[n]}_print.stl") for n, down in PRINT_DOWN.items()]
    # right leg: the mirrored part (Y -> -Y) prints the same way with the mirrored down direction
    jobs += [(EXP / "stl_right_mirror" / f"{stems[n]}_RIGHT.stl", (down[0], -down[1], down[2]), f"{stems[n]}_RIGHT_print.stl")
             for n, down in PRINT_DOWN.items() if (EXP / "stl_right_mirror" / f"{stems[n]}_RIGHT.stl").exists()
             and n != "pelvis_plate"]          # the single-leg desk fixture has no right-hand use
    for src, down, out in jobs:
        m = trimesh.load(src)
        m.apply_transform(trimesh.geometry.align_vectors(down, (0, 0, -1)))
        c = m.bounds.mean(axis=0)
        m.apply_translation((BED_CENTER[0] - c[0], BED_CENTER[1] - c[1], -m.bounds[0][2]))
        m.export(d / out)


def export_all(items):
    (EXP / "stl").mkdir(parents=True, exist_ok=True)
    (EXP / "stl_right_mirror").mkdir(parents=True, exist_ok=True)
    (EXP / "3mf").mkdir(parents=True, exist_ok=True)
    (EXP / "step").mkdir(parents=True, exist_ok=True)
    out = {}
    for it in items:
        if not it.printable:
            continue
        stem = f"{PART_NO[it.name]}_{it.name}"
        export_stl(it.shape, str(EXP / "stl" / f"{stem}.stl"), tolerance=0.02, angular_tolerance=0.15)
        export_step(it.shape, str(EXP / "step" / f"{stem}.step"))
        m = Mesher()
        m.add_shape(it.shape, linear_deflection=0.02, angular_deflection=0.15)
        m.write(str(EXP / "3mf" / f"{stem}.3mf"))
        if it.name not in ("clamp_tongue",):   # tongue is symmetric
            export_stl(leg.mirror_right(it.shape), str(EXP / "stl_right_mirror" / f"{stem}_RIGHT.stl"),
                       tolerance=0.02, angular_tolerance=0.15)
        out[it.name] = stem
    export_print_ready(out)
    # servo dummy (NOT printable) and the neutral assembly
    dummy = leg.servo_case() + leg.horns()[0] + leg.horns()[1]
    export_step(dummy, str(EXP / "step" / "NOT_PRINTABLE_servo_dummy_STS3215.step"))
    export_stl(dummy, str(EXP / "stl" / "NOT_PRINTABLE_servo_dummy_STS3215.stl"), tolerance=0.02)
    for tag, q in (("neutral", {}), ("stance", leg.STANCE)):
        posed = leg.pose([i for i in items if i.kind in ("part", "servo", "horn")], q)
        export_step(Compound([i.shape for i in posed]), str(EXP / "step" / f"assembly_{tag}.step"))
    return out


# ----------------------------------------------------------------------------- rendering (MuJoCo, CPU)
def _scene_xml(posed, tmp, extra_geoms=""):
    assets, geoms = [], []
    for k, it in enumerate(posed):
        if it.kind == "keepout":
            continue
        f = Path(tmp) / f"g{k}.stl"
        export_stl(it.shape, str(f), tolerance=0.05, angular_tolerance=0.3)
        assets.append(f'<mesh name="m{k}" file="{f}" scale="0.001 0.001 0.001"/>')
        c = " ".join(f"{v:.3f}" for v in it.color)
        geoms.append(f'<geom type="mesh" mesh="m{k}" rgba="{c}"/>')
    return f"""<mujoco><asset><texture name="sky" type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.90 0.91 0.93" width="64" height="64"/>{''.join(assets)}</asset>
<visual><global offwidth="1600" offheight="1600"/><quality shadowsize="0"/>
<headlight ambient="0.42 0.42 0.42" diffuse="0.55 0.55 0.55" specular="0.1 0.1 0.1"/></visual>
<worldbody><light pos="0.4 -0.6 0.8" dir="-0.4 0.6 -0.8" directional="true" diffuse="0.35 0.35 0.35"/>
{''.join(geoms)}{extra_geoms}</worldbody></mujoco>"""


def render_views(posed, views, size=(700, 1000), fovy=0.46, center=(0.0, 0.03, -0.17)):
    """views: list of (name, cam_pos_dir, up, ortho). Returns dict name -> image."""
    import mujoco
    with tempfile.TemporaryDirectory() as tmp:
        m = mujoco.MjModel.from_xml_string(_scene_xml(posed, tmp))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, size[1], size[0])
        out = {}
        for name, direction, ortho, scale in views:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = center
            az, el = direction
            cam.azimuth, cam.elevation, cam.distance = az, el, 1.2
            m.vis.global_.orthographic = 1 if ortho else 0
            m.vis.global_.fovy = scale if ortho else 30
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
    stance = leg.pose(vis, leg.STANCE)
    neutral = leg.pose(vis, {})
    views = [("front", (180, 0), True, 0.46), ("side", (90, 0), True, 0.46), ("three_quarter", (135, -15), False, 0.0)]
    for tag, posed in (("stance", stance), ("neutral", neutral)):
        imgs = render_views(posed, views)
        for k, im in imgs.items():
            iio.imwrite(REN / f"leg_{tag}_{k}.png", im)
    # dimensioned front/side silhouettes at a consistent scale (mm grid)
    imgs = render_views(stance, views[:2], size=(700, 1000), fovy=0.46)
    mm_per_px = 460.0 / 1000
    fig, axs = plt.subplots(1, 2, figsize=(10, 7.4), dpi=130)
    for ax, (k, title) in zip(axs, (("front", "FRONT (from +X)"), ("side", "SIDE (from +Y, faces left)"))):
        ax.imshow(imgs[k], extent=(-350 * mm_per_px, 350 * mm_per_px, -500 * mm_per_px, 500 * mm_per_px))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mm")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("mm (origin = view centre)")
    fig.suptitle("P0 leg, stance pose (hip pitch 16°, knee 32°): pad level; hip stack = exposed mechanism zone", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(REN / "leg_silhouette_dimensioned.png")
    plt.close(fig)
    # exploded view (neutral pose, links pulled apart along -Z and printed parts offset)
    offs = {"pelvis": 0, "yaw": -45, "roll": -95, "thigh": -150, "shin": -215}
    expl = []
    for it in neutral:
        dz = offs[it.link] + (-30 if it.name == "pad" else 0)
        dy = 0 if it.kind != "servo" else 0
        expl.append(leg.Item(it.name, it.shape.moved(leg.Location((0, dy, dz))), it.link, it.printable, it.color, it.kind))
    im = render_views(expl, [("exploded", (135, -12), False, 0)], size=(900, 1400), center=(0.0, 0.03, -0.25))["exploded"]
    iio.imwrite(REN / "leg_exploded.png", im)


def render_joints(items):
    """One close-up per joint: blue = link holding the servo CASE, orange = link bolted to the HORNS
    (main + idler = passive coaxial support), dark = servo, gold = horns. Neutral pose."""
    import imageio.v2 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange = (0.25, 0.48, 0.85, 1), (0.92, 0.47, 0.20, 1)
    cams = {"hip_yaw": (150, -25), "hip_roll": (60, -15), "hip_pitch": (150, -10), "knee": (150, -10)}
    names = {"pelvis": "pelvis plate", "yaw": "yaw bracket", "roll": "roll bracket", "thigh": "thigh", "shin": "shin"}
    fig, axs = plt.subplots(1, 4, figsize=(16, 5.2), dpi=120)
    for ax, (j, s) in zip(axs, leg.SERVOS.items()):
        sub = []
        for it in items:
            if it.kind == "keepout":
                continue
            if it.link not in (s["parent"], s["child"]) and it.kind == "part":
                continue
            if it.kind in ("servo", "horn") and not it.name.endswith(j):
                continue
            col = it.color
            if it.kind == "part":
                col = blue if it.link == s["parent"] else orange
            sub.append(leg.Item(it.name, it.shape, it.link, it.printable, col, it.kind))
        ctr = tuple(v / 1000 for v in leg.AXES[j][0])
        sub = [i for i in sub if i.name != "clamp_tongue"]
        im = render_views(sub, [("c", cams[j], True, 0.12)], size=(600, 600), center=ctr)["c"]
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"{j}\ncase held by: {names[s['parent']]} (blue)\nhorn + idler bolted to: {names[s['child']]} (orange)",
                     fontsize=9)
    fig.suptitle("Every joint: parent link grips the servo case; child U-bracket bolts to the main horn AND the idler horn "
                 "(the idler turns on the servo's rear bearing = passive coaxial support)", fontsize=10)
    fig.tight_layout()
    fig.savefig(REN / "joints_case_vs_horn.png")
    plt.close(fig)


def render_comparison(items):
    """Side-by-side at the same scale: CAD stance vs the simulated baseline_P30 leg."""
    import imageio.v2 as iio
    import mujoco
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vis = [i for i in items if i.kind in ("part", "servo", "horn")]
    cad = render_views(leg.pose(vis, leg.STANCE), [("front", (180, 0), True, 0.46), ("side", (90, 0), True, 0.46)])
    sim_xml = ROOT.parents[1] / "models" / "baseline_P30_8dof.xml"
    import re
    xml = re.sub(r'<camera [^>]*/>', '', sim_xml.read_text())        # sim cameras use a newer MuJoCo attribute
    cwd = os.getcwd()
    os.chdir(sim_xml.parent)
    m = mujoco.MjModel.from_xml_string(xml)
    os.chdir(cwd)
    m.mat_reflectance[:] = 0
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand"))
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 1000, 700)
    pel = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    sim = {}
    for k, (az, el) in (("front", (180, 0)), ("side", (90, 0))):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = (pel[0], 0.0, 0.17)
        cam.azimuth, cam.elevation, cam.distance = az, el, 1.2
        m.vis.global_.orthographic = 1
        m.vis.global_.fovy = 0.46
        r.update_scene(d, camera=cam)
        sim[k] = r.render().copy()
    r.close()
    fig, axs = plt.subplots(1, 4, figsize=(16, 6.8), dpi=120)
    for ax, im, t in zip(axs, (sim["front"], cad["front"], sim["side"], cad["side"]),
                         ("simulation (baseline_P30): front", "P0 CAD: front", "simulation: side", "P0 CAD: side")):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle("Same scale (0.46 mm/px). Sim thigh 138 / shin 150 mm kept; the CAD hip stack is taller and the knee carries a "
                 "direct-drive servo bulge.", fontsize=10)
    fig.tight_layout()
    fig.savefig(REN / "compare_sim_vs_cad.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- checks
def _overlap(a, b, tol=1.0):
    ba, bb = a.bounding_box(), b.bounding_box()
    if (ba.max.X < bb.min.X or bb.max.X < ba.min.X or ba.max.Y < bb.min.Y or bb.max.Y < ba.min.Y
            or ba.max.Z < bb.min.Z or bb.max.Z < ba.min.Z):
        return 0.0
    try:
        v = (a & b).volume
    except Exception:
        return float("nan")
    return v if v > tol else 0.0


def collisions(posed, moving_links=None):
    """Volume overlaps (> 1 mm^3) between items on different links (keep-outs included)."""
    hits = []
    for i, a in enumerate(posed):
        for b in posed[i + 1:]:
            if a.link == b.link:
                continue
            if a.kind == "keepout" and b.kind == "keepout":
                continue
            if moving_links is not None and not ((a.link in moving_links) ^ (b.link in moving_links)):
                continue
            v = _overlap(a.shape, b.shape)
            if v:
                hits.append((a.name, b.name, round(v, 1)))
    return hits


def rom_sweep(items, step=5):
    out = {}
    for j, (lo, hi) in leg.P["rom_targets"].items():
        link = [l for l, jj in leg.JOINT_OF_LINK.items() if jj == j][0]
        moving = leg.CHAIN[leg.CHAIN.index(link):]
        res = {}
        for sgn, lim in ((-1, lo), (1, hi)):
            free = 0
            first = None
            for a in range(step, int(abs(lim)) + step, step):
                ang = sgn * min(a, abs(lim))
                hits = collisions(leg.pose(items, {j: ang}), moving_links=set(moving))
                if hits:
                    first = (ang, hits[:3])
                    break
                free = ang
            res["neg" if sgn < 0 else "pos"] = {"target": lim, "collision_free_to": free, "first_hit": first}
        out[j] = res
        print(j, res, flush=True)
    return out


def mass_report(items):
    dens = leg.P["print"]["density_g_cm3"] * leg.P["print"]["fill_factor"]
    parts = {it.name: round(it.shape.volume / 1000 * dens, 1) for it in items if it.printable}
    servos = 4 * leg.S["mass_g"]
    return {"printed_g_estimate": parts, "printed_total_g": round(sum(parts.values()), 1), "servos_g": servos,
            "hardware_g_estimate": 25, "leg_on_fixture_total_g": round(sum(parts.values()) + servos + 25, 1),
            "note": f"printed mass = solid volume x {leg.P['print']['density_g_cm3']} g/cm3 x fill {leg.P['print']['fill_factor']} (rough)"}


def mesh_checks():
    import trimesh
    res = {}
    for f in (sorted((EXP / "stl").glob("P*.stl")) + sorted((EXP / "stl" / "print_ready").glob("*.stl"))
              + sorted((EXP / "stl_right_mirror").glob("*.stl"))):
        m = trimesh.load(f)
        res[f.name] = {"faces": int(len(m.faces)), "watertight": bool(m.is_watertight), "volume_cm3": round(m.volume / 1000, 2),
                       "bbox_mm": [round(v, 1) for v in m.extents]}
    return res


def check_all(items):
    rep = {"levels_mm": leg.report_levels(), "axes": {k: {"point": v[0], "dir": v[1]} for k, v in leg.AXES.items()}}
    t = time.time()
    for tag, q in (("neutral", {}), ("stance", leg.STANCE)):
        rep[f"collisions_{tag}"] = collisions(leg.pose(items, q))
        print(tag, rep[f"collisions_{tag}"], flush=True)
    rep["rom"] = rom_sweep(items)
    rep["mass"] = mass_report(items)
    rep["meshes"] = mesh_checks()
    pad = [i for i in items if i.name == "pad"][0]
    pad_st = leg.link_transform(pad.shape, "shin", leg.STANCE)
    bottom = min(pad_st.faces(), key=lambda f: f.center().Z)
    rep["pad_underside_normal_at_stance"] = [round(v, 3) for v in bottom.normal_at()]
    rep["check_seconds"] = round(time.time() - t, 1)
    (ROOT / "verification_report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    items = leg.build_items()
    if what in ("all", "export"):
        print(export_all(items))
    if what in ("all", "render"):
        render_all(items)
        render_joints(items)
        render_comparison(items)
    if what in ("all", "check"):
        rep = check_all(items)
        print(json.dumps({k: rep[k] for k in ("levels_mm", "collisions_neutral", "collisions_stance", "mass",
                                               "pad_underside_normal_at_stance")}, indent=1, default=str))
