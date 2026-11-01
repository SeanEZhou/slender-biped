"""Blender renders of the whole robot with the unpainted head, the head alone, and the hair cap lifted.

    bl/bin/python hardware/head_p0/cad/render_whole.py <legs_dir> <assembled_dir>     # torso.py and head.py --assembled <dir>
"""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("torso_render", ROOT.parent / "torso_p0" / "cad" / "render_whole.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
bpy = tr.bpy

VIEWS = dict(front=(0, 2), three_quarter=(38, 6), side=(90, 2), rear=(150, 6))
HEAD_Z = (310.0 - tr.FLOOR) * tr.MM
LIFT = 0.055


def sheet(paths, name):
    ims = [Image.open(p) for p in paths]
    w, h = ims[0].size
    out = Image.new("RGB", (w * len(ims), h))
    for i, im in enumerate(ims):
        out.paste(im, (i * w, 0))
    out.save(ROOT / "renders" / name)
    for p in paths:
        p.unlink()


def shots(views, dist, target, name, res):
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = res
    paths = []
    for n, (az, el) in views.items():
        tr.camera(az, el, dist, target)
        paths.append(ROOT / "renders" / f"_{n}.png")
        tr.render_to(paths[-1])
    sheet(paths, name)


def main():
    legs, assembled = sys.argv[1], Path(sys.argv[2])
    (ROOT / "renders").mkdir(exist_ok=True)
    mats = tr.scene((800, 1300))
    root = bpy.data.objects.new("root", None)
    bpy.context.collection.objects.link(root)
    root.location = (0, 0, -tr.FLOOR * tr.MM)
    tr.build_legs(legs, mats, root)
    for f in sorted(assembled.glob("T0*.stl")):
        tr.stl(f, mats["shell"], root)
    tr.stl(assembled / "H01_face.stl", mats["shell"], root, sharp=35)
    hair = tr.stl(assembled / "H02_hair.stl", mats["shell"], root, sharp=35)
    shots(VIEWS, 2.25, (0, 0, (-30.0 - tr.FLOOR) * tr.MM), "head_whole.png", (800, 1300))
    shots(VIEWS, 0.46, (-.012, 0, HEAD_Z), "head_views.png", (700, 700))
    hair.location.z = LIFT
    shots(dict(three_quarter=(38, 8), rear=(150, 8)), 0.62, (-.012, 0, HEAD_Z + LIFT / 2), "head_parts.png", (800, 800))


if __name__ == "__main__":
    main()
