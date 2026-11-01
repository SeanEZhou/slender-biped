"""Paint previews for head H1: hair and eye colours, and robot-style face markings. Previews only; nothing here is printed.

    <bpy venv>/bin/python hardware/head_p0/cad/render_paint.py <legs_dir> <assembled_dir>     # torso.py --assembled <dir>
"""
import json
import math
import os
import sys
from pathlib import Path

WSL = "/usr/lib/wsl/lib"
if Path(WSL).is_dir() and not os.environ.get("LD_LIBRARY_PATH", "").startswith(WSL):
    os.environ["LD_LIBRARY_PATH"] = WSL + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mathutils import Vector  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import head as hd  # noqa: E402

LEGS, ASM = Path(sys.argv[1]), Path(sys.argv[2])
OUT = HERE.parent / "renders" / "paint"
MM = 0.001
INK = (.07, .06, .08)
SKIN = (.97, .97, .965)
BLUE, RED, GREEN, YELLOW = (.20, .45, .85), (.78, .13, .15), (.16, .62, .34), (.93, .77, .18)
WHITE, LIGHT_BLUE, BLONDE, PINK = (.95, .95, .94), (.58, .80, .95), (.95, .82, .50), (.95, .62, .74)
COLOURS = [(f"{hn} hair, {en} eyes", hc, ec)
           for hn, hc, eyes in (("white", WHITE, (("blue", BLUE), ("red", RED), ("green", GREEN))),
                                ("light blue", LIGHT_BLUE, (("red", RED), ("blue", BLUE), ("green", GREEN), ("yellow", YELLOW))),
                                ("blonde", BLONDE, (("blue", BLUE), ("green", GREEN))),
                                ("dark blue", (.10, .16, .38), (("dark blue", (.10, .18, .50)),)),
                                ("black", (.04, .04, .05), (("black", (.05, .05, .06)),)),
                                ("pink", PINK, (("blue", BLUE), ("green", GREEN))))
           for en, ec in eyes]
MARKS = ["none", "under-eye bars", "temple circuit", "cheek chevrons", "tear line"]
ACCENT = (.20, .55, .85)


def srgb(c):
    return tuple(((v + .055) / 1.055) ** 2.4 if v > .04045 else v / 12.92 for v in c)


_mats = {}


def mat(col, rough=.36, coat=.3):
    k = (tuple(np.round(col, 3)), rough, coat)
    if k not in _mats:
        m = bpy.data.materials.new(str(k))
        m.use_nodes = True
        p = m.node_tree.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = (*srgb(col), 1)
        p.inputs["Roughness"].default_value = rough
        p.inputs["Coat Weight"].default_value = coat
        _mats[k] = m
    return _mats[k]


def obj(V, F, m, sharp=None):
    me = bpy.data.meshes.new("m")
    me.from_pydata((np.asarray(V) * MM).tolist(), [], np.asarray(F).tolist())
    me.update()
    me.materials.append(m)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    if sharp:
        me.set_sharp_from_angle(angle=math.radians(sharp))
    me.update()
    o = bpy.data.objects.new("o", me)
    bpy.context.collection.objects.link(o)
    return o


class Paint:
    """Flat paint layers lying on the sculpted face."""

    def __init__(self, face):
        self.f, self.parts = face, []

    def poly(self, pts, col, eps=.0006):
        pts = np.asarray(pts)
        c = pts.mean(0)
        rs = np.linspace(0, 1, 8)
        P = np.array([c + r * (q - c) for r in rs for q in pts])
        n = len(pts)
        F = []
        for i in range(len(rs) - 1):
            for j in range(n - 1):
                a, b, d, e = i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j
                F += [(a, b, d), (a, d, e)]
        self.parts.append((np.c_[P, self.f.z(P[:, 0], P[:, 1]) + eps], np.array(F), col))

    def line(self, pts, w, col, eps=.0009):
        pts = np.asarray(pts, float)
        d = np.gradient(pts, axis=0)
        nn = np.c_[-d[:, 1], d[:, 0]]
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
        w = np.broadcast_to(np.asarray(w, float), (len(pts),))[:, None]
        P = np.vstack([pts - nn * w / 2, pts + nn * w / 2])
        k = len(pts)
        F = []
        for i in range(k - 1):
            F += [(i, i + 1, k + i + 1), (i, k + i + 1, k + i)]
        self.parts.append((np.c_[P, self.f.z(P[:, 0], P[:, 1]) + eps], np.array(F), col))

    def dot(self, x, y, r, col, eps=.0011):
        self.poly(ell(x, y, r, r), col, eps)


def ell(cx, cy, rx, ry, n=40):
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]


def eyes_and_mouth(p, iris):
    dark = tuple(v * .45 for v in iris)
    for s in (1, -1):
        cx, cy = s * hd.EYE[0], hd.EYE[1]
        t = np.linspace(0, 1, 40)
        top = np.c_[cx + s * (.0245 - .049 * t), cy + .0005 + .0098 * np.sin(np.pi * t) ** .55]
        bot = np.c_[cx + s * (-.0245 + .049 * t), cy - .001 - .0085 * np.sin(np.pi * t) ** .9]
        p.poly(np.vstack([top, bot]), (.98, .98, .98), .0005)
        ix, iy = cx - s * .001, cy - .0003
        c = ell(ix, iy, .0098, .0098)
        c[:, 1] = np.minimum(c[:, 1], cy + .0082)
        p.poly(c, dark, .0008)
        c = ell(ix, iy - .001, .0078, .0078)
        c[:, 1] = np.minimum(c[:, 1], cy + .0082)
        p.poly(c, iris, .0010)
        p.poly(ell(ix, iy, .0040, .0040), (.05, .05, .07), .0012)
        p.dot(ix - s * .0035, iy + .0038, .0018, (1, 1, 1), .0014)
        p.line(top, .0040 * np.sin(np.pi * np.clip(t, .02, .98)) ** .3 + .001, INK, .0016)
        bx = np.linspace(.016, .072, 12)
        p.line(np.c_[s * bx, 1.934 + .0028 * np.sin(np.pi * (bx - .016) / .056) - .003 * ((bx - .016) / .056)], .0024, INK, .0012)
    mx = np.linspace(-hd.MOUTH_W, hd.MOUTH_W, 20)
    p.line(np.c_[mx, hd.MOUTH_Y + hd.mouth_dy(mx)], .0012, (.50, .28, .30), .0008)


def marking(p, kind):
    """Original robot-style face markings."""
    col = ACCENT
    if kind == "under-eye bars":
        for s in (1, -1):
            for k in range(3):
                x = s * (.036 + k * .0065)
                p.line(np.c_[[x, x], [1.8855, 1.8855 - .0075 + .0016 * k]], .0028, col)
    if kind == "temple circuit":
        for s, yo in ((1, 0), (-1, 0)):
            pts = np.array([(.070, 1.925), (.066, 1.905), (.058, 1.885), (.058, 1.866)])
            q = np.linspace(0, 1, 24)
            xy = np.c_[np.interp(q, np.linspace(0, 1, 4), pts[:, 0]) * s, np.interp(q, np.linspace(0, 1, 4), pts[:, 1])]
            p.line(xy, .0011, col)
            p.dot(s * .058, 1.8645, .0022, col)
            p.line(np.c_[[s * .0635, s * .050], [1.895, 1.879]], .0009, col)
            p.dot(s * .0497, 1.878, .0016, col)
    if kind == "cheek chevrons":
        for s in (1, -1):
            for k in range(2):
                x0, y0 = s * (.052 + k * .0075), 1.868 - k * .0005
                p.line(np.c_[[x0 + s * .0048, x0, x0 + s * .0048], [y0 + .0055, y0, y0 - .0055]], .0015, col)
    if kind == "tear line":
        for s in (1, -1):
            x = s * .0385
            p.line(np.c_[[x, x], [1.8895, 1.862]], .0013, col)
            p.dot(x, 1.8595, .0021, col)


def aim(o, t):
    o.rotation_euler = (Vector(t) - Vector(o.location)).to_track_quat("-Z", "Y").to_euler()


def light(loc, t, size, power, col=(1, 1, 1)):
    d = bpy.data.lights.new("l", "AREA")
    d.shape = "RECTANGLE"
    d.size, d.size_y = size
    d.energy, d.color = power, col
    o = bpy.data.objects.new("l", d)
    bpy.context.collection.objects.link(o)
    o.location = loc
    aim(o, t)
    o.visible_camera = False


for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 48
sc.cycles.use_denoising = True
sc.cycles.denoiser = "OPENIMAGEDENOISE"
sc.view_settings.view_transform, sc.view_settings.look = "Standard", "None"
sc.render.image_settings.file_format = "PNG"
pr = bpy.context.preferences.addons["cycles"].preferences
for be in ("OPTIX", "CUDA"):
    pr.compute_device_type = be
    pr.get_devices()
    if [d for d in pr.devices if d.type == be]:
        for d in pr.devices:
            d.use = d.type == be
        sc.cycles.device = "GPU"
        break
w = bpy.data.worlds.new("w")
sc.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (.50, .53, .58, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = .42
H = (-.012, 0, .31)
light((.9, -.55, 1.0), H, (1, 1), 9, (1, .97, .93))
light((.9, .9, .35), H, (1.2, 1.2), 4, (.92, .96, 1))
light((-.9, .1, .8), H, (.9, .9), 6)
light((1.5, -.4, -.1), (0, 0, -.05), (1.6, 2.2), 11)
cam = bpy.data.cameras.new("c")
cam.clip_start, cam.clip_end, cam.lens = .02, 40, 85
co = bpy.data.objects.new("c", cam)
bpy.context.collection.objects.link(co)
sc.camera = co


def shot(path, az, el, dist, t, res):
    a, e = math.radians(az), math.radians(el)
    co.location = (t[0] + dist * math.cos(e) * math.cos(a), t[1] + dist * math.cos(e) * math.sin(a), t[2] + dist * math.sin(e))
    aim(co, t)
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


WH, BK, GO = mat(SKIN, .32, .45), mat((.03, .03, .033), .38, .1), mat((.86, .66, .3), .26, .1)
for m in json.loads((LEGS / "manifest.json").read_text()):
    t = trimesh.load(LEGS / m["file"])
    obj(t.vertices, t.faces, BK if m["kind"] == "servo" else GO if m["kind"] == "horn" else WH, 35)
for f in sorted(ASM.glob("T*.stl")):
    t = trimesh.load(f)
    obj(t.vertices, t.faces, WH, 35)
face = hd.Face(True)
fp, hp = trimesh.load(ASM / "H01_face.stl"), trimesh.load(ASM / "H02_hair.stl")       # head.py --assembled <dir>
face_ob = obj(fp.vertices, fp.faces, WH, 35)
hair_ob = obj(hp.vertices, hp.faces, mat(WHITE, .3, .3), 35)
C = (-.012, 0, .298)
Tmp = OUT / "views"
Tmp.mkdir(parents=True, exist_ok=True)


def variant(hair_col, iris, mark, tag, body=False):
    hair_ob.data.materials[0] = mat(hair_col, .28, .35)
    p = Paint(face)
    eyes_and_mouth(p, iris)
    marking(p, mark)
    obs = [obj(hd.scale_mm(v), f, mat(col, .4, .1)) for v, f, col in p.parts]
    shot(Tmp / f"{tag}_f.png", 0, 2, .40, C, (440, 440))
    shot(Tmp / f"{tag}_34.png", -35, 3, .40, C, (440, 440))
    if body:
        shot(Tmp / f"{tag}_body.png", -24, 4, 2.0, (-.01, 0, -.01), (420, 840))
    for o in obs:
        me = o.data
        bpy.data.objects.remove(o)
        bpy.data.meshes.remove(me)


for i, (n, hc, ic) in enumerate(COLOURS):
    variant(hc, ic, "none", f"c{i}", body=True)
for i, mk in enumerate(MARKS[1:]):
    variant(COLOURS[0][1], COLOURS[0][2], mk, f"m{i}")
F = ImageFont.truetype("DejaVuSans.ttf", 18)


def sheet(items, name, suffixes, size=(360, 360)):
    cols = len(suffixes)
    per = 3
    rows = int(np.ceil(len(items) / per))
    G = Image.new("RGB", (per * cols * size[0], rows * (size[1] + 28)), "white")
    d = ImageDraw.Draw(G)
    for k, (tag, label) in enumerate(items):
        x, y = (k % per) * cols * size[0], (k // per) * (size[1] + 28)
        d.text((x + 8, y + 5), label, fill=(20, 20, 30), font=F)
        for j, s in enumerate(suffixes):
            G.paste(Image.open(Tmp / (f"{tag}_{s}.png" if s else f"{tag}.png")).convert("RGB").resize(size, Image.LANCZOS), (x + j * size[0], y + 28))
    G.save(OUT / name)


sheet([(f"c{i}", c[0]) for i, c in enumerate(COLOURS)], "colour_schemes.png", ("f", "34"))
sheet([(f"m{i}", m) for i, m in enumerate(MARKS[1:])] + [("c0", "no markings")], "face_markings.png", ("f", "34"))
COLS = 7
G = Image.new("RGB", (COLS * 300, 2 * 640), "white")
d = ImageDraw.Draw(G)
for i, c in enumerate(COLOURS):
    x, y = (i % COLS) * 300, (i // COLS) * 640
    G.paste(Image.open(Tmp / f"c{i}_body.png").convert("RGB").resize((300, 600), Image.LANCZOS), (x, y + 40))
    d.text((x + 6, y + 10), c[0], fill=(20, 20, 30), font=ImageFont.truetype("DejaVuSans.ttf", 14))
G.save(OUT / "full_body.png")

# how the two printed parts go together
for o in bpy.data.objects:
    if o.type == "MESH" and o not in (face_ob, hair_ob):
        o.hide_render = True
ho = hair_ob
ho.data.materials[0] = mat(LIGHT_BLUE, .3, .3)
Sp = (("front", 0, 2, 0), ("side", -90, 2, 0), ("rear", 180, 2, 0), ("below_rear", 180, -48, 0), ("below_side", -115, -48, 0), ("lifted", -35, 8, .05))
for n, az, el, up in Sp:
    ho.location.z = up
    shot(Tmp / f"s_{n}.png", az, el, .46 if up else .40, (C[0], C[1], C[2] + up * .4), (440, 440))
sheet([(f"s_{n}", n.replace("_", " ")) for n, *_ in Sp], "parts.png", ("",))
