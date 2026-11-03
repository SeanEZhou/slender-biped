"""Blender renders of head H2, unpainted: the turnaround, the bare face from several angles, the parts apart, and the whole robot.

    .venv-render/bin/python hardware/head_p1/versions/v4/cad/render.py <legs_dir> <assembled_dir> [concept.png]

<assembled_dir> holds torso.py --assembled and head.py --assembled output. With a concept board, also writes
renders/drafts/concept_compare.png (the board's turnaround above the CAD's).
"""
import importlib.util
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("torso_render", Path(__file__).resolve().parents[4] / "torso_p0" / "cad" / "render_whole.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
bpy = tr.bpy

OUT = ROOT / "renders"
HEAD_Z = (305.0 - tr.FLOOR) * tr.MM
TURN = dict(front=(0, 3), three_quarter=(45, 3), side=(90, 3), three_quarter_back=(135, 3), back=(180, 3))
FONT = ImageFont.truetype("DejaVuSans.ttf", 22)


def shots(views, dist, target, res, tag):
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = res
    ims = []
    for n, (az, el) in views.items():
        tr.camera(az, el, dist, target)
        p = OUT / f"_{tag}_{n}.png"
        tr.render_to(p)
        ims.append(Image.open(p).convert("RGB"))
        p.unlink()
    return ims


def strip(ims, path=None):
    w, h = ims[0].size
    g = Image.new("RGB", (w * len(ims), h))
    for i, im in enumerate(ims):
        g.paste(im, (i * w, 0))
    if path:
        g.save(path)
    return g


def main():
    legs, assembled = sys.argv[1], Path(sys.argv[2])
    mats = tr.scene((800, 1300))
    root = bpy.data.objects.new("root", None)
    bpy.context.collection.objects.link(root)
    root.location = (0, 0, -tr.FLOOR * tr.MM)
    tr.build_legs(legs, mats, root)
    body = [tr.stl(f, mats["shell"], root) for f in sorted(assembled.glob("T0*.stl"))]
    face = tr.stl(assembled / "H11_face.stl", mats["shell"], root, 60)
    hair = tr.stl(assembled / "H12_hair.stl", mats["shell"], root, 28)
    strip(shots(dict(front=(0, 2), three_quarter=(38, 6), side=(90, 2), rear=(150, 6)), 2.25, (0, 0, (-30.0 - tr.FLOOR) * tr.MM), (800, 1300), "whole"), OUT / "head_whole.png")
    turn = shots(TURN, .40, (-.012, 0, HEAD_Z), (560, 560), "turn")
    strip(turn, OUT / "head_views.png")
    for o in body:
        if "T06" not in o.name:
            o.hide_render = True
    hair.hide_render = True
    strip(shots(dict(front=(0, 2), quarter=(25, 4), three_quarter=(50, 4), side=(90, 2), low=(30, -22)), .26, (-.004, 0, (295.0 - tr.FLOOR) * tr.MM), (480, 480), "face"), OUT / "face_structure.png")
    hair.hide_render = False
    hair.location.z = .06
    strip(shots(dict(three_quarter=(40, 10), back=(140, 10)), .62, (-.012, 0, HEAD_Z + .03), (800, 800), "parts"), OUT / "head_parts.png")
    if len(sys.argv) > 3:
        con = Image.open(sys.argv[3]).convert("RGB").crop((8, 780, 838, 950))
        top = con.resize((2800, int(con.size[1] * 2800 / con.size[0])), Image.LANCZOS)
        bot = strip(turn)
        g = Image.new("RGB", (2800, top.size[1] + bot.size[1] + 70), "white")
        d = ImageDraw.Draw(g)
        d.text((8, 4), "concept", fill=(20, 20, 30), font=FONT)
        g.paste(top, (0, 32))
        d.text((8, top.size[1] + 38), "CAD, unpainted (stickers not shown)", fill=(20, 20, 30), font=FONT)
        g.paste(bot, (0, top.size[1] + 68))
        (OUT / "drafts").mkdir(exist_ok=True)
        g.save(OUT / "drafts" / "concept_compare.png")


if __name__ == "__main__":
    main()
