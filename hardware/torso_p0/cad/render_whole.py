"""Blender render of the assembled robot, every torso piece in the same white.

    bl/bin/python hardware/torso_p0/cad/render_whole.py <legs_dir> <assembled_dir>     # torso.py --assembled <dir>
"""
import json
import math
import os
import sys
from pathlib import Path

# WSL: the apt libcuda finds no GPU (cuInit 100) and shadows the working WSL one, so put that first and restart
WSL_LIB = "/usr/lib/wsl/lib"
if Path(WSL_LIB).is_dir() and not os.environ.get("LD_LIBRARY_PATH", "").startswith(WSL_LIB):
    os.environ["LD_LIBRARY_PATH"] = WSL_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FLOOR = -367.3            # underside of the pads in stance
MM = 0.001


def srgb(c):
    return tuple(((v + 0.055) / 1.055) ** 2.4 if v > 0.04045 else v / 12.92 for v in c)


def make_mat(name, base, rough, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (*srgb(base), 1)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Coat Roughness"].default_value = 0.18
    return m


def setup_materials():
    m = base_materials()
    white = make_mat("white", (0.98, 0.98, 0.98), 0.30, coat=0.5)
    return dict(ivory=m, white=dict(m, ivory=white, shell=white, dark=white, pad=white,
                                    servo=make_mat("black", (0.025, 0.025, 0.028), 0.38)))


def base_materials():
    return dict(
        ivory=make_mat("ivory", (0.93, 0.91, 0.86), 0.42, coat=0.15),
        shell=make_mat("shell", (0.94, 0.92, 0.875), 0.30, coat=0.55),
        dark=make_mat("dark", (0.20, 0.21, 0.24), 0.50),
        servo=make_mat("servo", (0.11, 0.11, 0.125), 0.42),
        pad=make_mat("pad", (0.05, 0.05, 0.055), 0.75),
        gold=make_mat("gold", (0.86, 0.66, 0.30), 0.26, metal=1.0),
        seam=make_mat("seam", (0.25, 0.23, 0.22), 0.6),
    )


def setup_world():
    w = bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.56, 0.58, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.75


def add_floor(mat):
    me = bpy.data.meshes.new("floor")
    s = 30.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    me.materials.append(mat)
    ob = bpy.data.objects.new("floor", me)
    bpy.context.collection.objects.link(ob)
    ob.is_shadow_catcher = True
    return ob


def area_light(name, loc, target, size, power, color=(1, 1, 1)):
    d = bpy.data.lights.new(name, "AREA")
    d.shape = "RECTANGLE"
    d.size, d.size_y = size
    d.energy = power
    d.color = color
    ob = bpy.data.objects.new(name, d)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    aim(ob, target)
    return ob


def aim(ob, target):
    from mathutils import Vector
    dv = Vector(target) - Vector(ob.location)
    ob.rotation_euler = dv.to_track_quat("-Z", "Y").to_euler()


def setup_scene(samples, res, gpu=True):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.film_transparent = True
    sc.cycles.max_bounces = 6
    sc.unit_settings.system = "METRIC"
    if gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == backend]
                if devs:
                    for d in prefs.devices:
                        d.use = d.type == backend
                    sc.cycles.device = "GPU"
                    print("device:", backend, [d.name for d in devs])
                    return
            except Exception as e:      # no usable GPU backend: stay on CPU
                print("backend failed", backend, e)
        print("device: CPU")


def camera(az_deg, el_deg, dist, target, lens=85):
    cam = bpy.data.cameras.new("cam")
    cam.lens = lens
    cam.clip_start = 0.05
    cam.clip_end = 60
    ob = bpy.data.objects.new("cam", cam)
    bpy.context.collection.objects.link(ob)
    a, e = math.radians(az_deg), math.radians(el_deg)
    ob.location = (target[0] + dist * math.cos(e) * math.cos(a), target[1] + dist * math.cos(e) * math.sin(a), target[2] + dist * math.sin(e))
    aim(ob, target)
    bpy.context.scene.camera = ob
    return ob


def lights():
    # robot faces +X; key from front-left high, big soft fill, rim from behind for the silhouette
    tgt = (0, 0, 0.45)
    area_light("key", (2.2, 1.6, 2.2), tgt, (2.2, 2.2), 70, (1.0, 0.97, 0.93))
    area_light("fill", (1.8, -2.4, 0.9), tgt, (3.0, 2.0), 12, (0.93, 0.96, 1.0))
    area_light("rimL", (-1.5, 1.4, 2.6), tgt, (1.2, 2.6), 55, (0.95, 0.97, 1.0))
    area_light("rimR", (-1.5, -1.4, 2.6), tgt, (1.2, 2.6), 55, (1.0, 0.96, 0.92))


def render_to(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    # soft studio backdrop: vertical gradient plus a slight vignette
    yy, xx = np.mgrid[0:h, 0:w]
    t = yy / max(h - 1, 1)
    top, bot = np.array((122.0, 127.0, 135.0)), np.array((158.0, 161.0, 166.0))
    bg = top + (bot - top) * t[..., None]
    vig = 1 - 0.20 * (((xx / w - 0.5) * 1.6) ** 2 + ((yy / h - 0.5) * 1.3) ** 2)
    bg = Image.fromarray(np.clip(bg * vig[..., None], 0, 255).astype("uint8"), "RGB").convert("RGBA")
    Image.alpha_composite(bg, im).convert("RGB").save(path)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build_legs(legs_dir, mats, parent):
    man = json.loads((Path(legs_dir) / "manifest.json").read_text())
    for it in man:
        bpy.ops.wm.stl_import(filepath=str(Path(legs_dir) / it["file"]))
        ob = bpy.context.selected_objects[0]
        ob.scale = (MM, MM, MM)
        ob.parent = parent
        name = it["name"]
        if it["kind"] == "servo":
            mat = mats["servo"]
        elif it["kind"] == "horn":
            mat = mats["gold"]
        elif "pad" in name:
            mat = mats["pad"]
        elif "thigh" in name or "shin" in name:
            mat = mats["ivory"]
        else:
            mat = mats["dark"]
        ob.data.materials.append(mat)
        ob.data.shade_smooth()
        ob.data.set_sharp_from_angle(angle=math.radians(32))


def stl(path, mat, parent=None, sharp=30):
    bpy.ops.wm.stl_import(filepath=str(path))
    ob = bpy.context.selected_objects[0]
    ob.scale = (MM, MM, MM)
    ob.parent = parent
    ob.data.materials.append(mat)
    ob.data.shade_smooth()
    ob.data.set_sharp_from_angle(angle=math.radians(sharp))
    return ob


def scene(res, samples=96):
    clear_scene()
    mats = setup_materials()["white"]
    setup_world()
    add_floor(make_mat("floor", (0.5, 0.5, 0.5), 0.6))
    setup_scene(samples, res)
    lights()
    return mats


def main():
    legs, assembled = sys.argv[1], sys.argv[2]
    out = ROOT / "renders"
    mats = scene((800, 1300))
    root = bpy.data.objects.new("root", None)
    bpy.context.collection.objects.link(root)
    root.location = (0, 0, -FLOOR * MM)
    build_legs(legs, mats, root)
    for f in sorted(Path(assembled).glob("T0*.stl")):
        stl(f, mats["shell"], root)
    tz = (-30.0 - FLOOR) * MM
    for name, (az, el) in dict(front=(0, 2), three_quarter=(38, 6), side=(90, 2), rear=(150, 6)).items():
        camera(az, el, 2.25, (0, 0, tz))
        render_to(out / f"whole_{name}.png")
    ims = [Image.open(out / f"whole_{n}.png") for n in ("front", "three_quarter", "side", "rear")]
    w, h = ims[0].size
    sheet = Image.new("RGB", (w * 4, h))
    for i, im in enumerate(ims):
        sheet.paste(im, (i * w, 0))
    sheet.save(out / "torso_whole.png")
    for n in ("front", "three_quarter", "side", "rear"):
        (out / f"whole_{n}.png").unlink()


if __name__ == "__main__":
    main()
