"""Renders of head H2 v3 with eye, brow and mouth stickers on the unpainted face and hair.

    .venv-render/bin/python hardware/head_p1/versions/v3/cad/render_stickers.py <legs_dir> <assembled_dir>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("torso_render", HERE.parents[3] / "torso_p0" / "cad" / "render_whole.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
bpy = tr.bpy
spec = importlib.util.spec_from_file_location("head_v3", HERE / "head.py")
hd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd)

OUT = HERE.parent / "renders"
FACE = hd.Face()
EX, EY = hd.EYE
INK, LASH = (.16, .10, .10), (.20, .12, .11)
IRIS, IRIS_D, PUPIL = (.96, .64, .16), (.58, .27, .06), (.24, .10, .04)
BROW, LIP = (.46, .34, .28), (.62, .32, .32)
FONT = ImageFont.truetype("DejaVuSans.ttf", 22)


class Sticker:
    """Flat colour layers lying on the face surface, each a hair above the last."""

    def __init__(self):
        self.parts = []

    def poly(self, pts, col, eps):
        pts = np.asarray(pts, float)
        c = pts.mean(0)
        rs = np.linspace(0, 1, 8)
        P = np.array([c + r * (q - c) for r in rs for q in pts])
        n, F = len(pts), []
        for i in range(len(rs) - 1):
            for j in range(n - 1):
                a, b, d, e = i * n + j, i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j
                F += [(a, b, d), (a, d, e)]
        self.parts.append((np.c_[P, FACE.z(P[:, 0], P[:, 1]) + eps], np.array(F), col))

    def line(self, pts, w, col, eps):
        pts = np.asarray(pts, float)
        d = np.gradient(pts, axis=0)
        nn = np.c_[-d[:, 1], d[:, 0]]
        nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-9
        w = np.broadcast_to(np.asarray(w, float), (len(pts),))[:, None]
        P = np.vstack([pts - nn * w / 2, pts + nn * w / 2])
        k, F = len(pts), []
        for i in range(k - 1):
            F += [(i, i + 1, k + i + 1), (i, k + i + 1, k + i)]
        self.parts.append((np.c_[P, FACE.z(P[:, 0], P[:, 1]) + eps], np.array(F), col))


def ell(cx, cy, rx, ry, n=48):
    t = np.linspace(0, 2 * np.pi, n)
    return np.c_[cx + rx * np.cos(t), cy + ry * np.sin(t)]


def eye_open(st, s):
    cx, cy = s * EX, EY
    t = np.linspace(0, 1, 48)
    top = np.c_[cx + s * (-.021 + .043 * t), cy + .0008 + .0118 * np.sin(np.pi * t) ** .62 + .0032 * t]
    bot = np.c_[cx + s * (-.021 + .043 * t), cy - .0012 - .0068 * np.sin(np.pi * t) ** .9 + .0026 * t]
    st.poly(np.vstack([top, bot[::-1]]), (.99, .98, .97), .0004)
    ty = lambda x: np.interp(s * (x - cx), s * (top[:, 0] - cx), top[:, 1])   # noqa: E731
    by = lambda x: np.interp(s * (x - cx), s * (bot[:, 0] - cx), bot[:, 1])   # noqa: E731

    def clip(p):
        return np.c_[p[:, 0], np.clip(p[:, 1], by(p[:, 0]) + .0003, ty(p[:, 0]) - .0003)]
    ix, iy = cx + s * .0012, cy - .0004
    st.poly(clip(ell(ix, iy, .0092, .0112)), IRIS_D, .0006)
    st.poly(clip(ell(ix, iy - .0012, .0076, .0094)), IRIS, .0008)
    st.poly(clip(ell(ix, iy - .0062, .0058, .0034)), (1.0, .82, .42), .0010)      # warm lower glow, as in the concept
    st.poly(clip(ell(ix, iy + .0012, .0036, .0055)), PUPIL, .0012)
    st.poly(ell(ix - s * .0034, iy + .0050, .0024, .0028), (1, 1, 1), .0016)
    st.poly(ell(ix + s * .0038, iy - .0042, .0011, .0011), (1, 1, 1), .0016)
    st.line(top, .0036 * np.sin(np.pi * np.clip(t, .03, .97)) ** .35 + .0008, LASH, .0018)
    fl = np.c_[cx + s * np.linspace(.0205, .0285, 10), cy + .0035 + np.linspace(.0006, .0062, 10)]
    st.line(fl, np.linspace(.0030, .0005, 10), LASH, .0018)                        # outer flick
    st.line(bot[22:46], np.linspace(.0006, .0011, 24), (.52, .36, .34), .0016)


def eye_closed(st, s, happy=True):
    cx, cy = s * EX, EY
    t = np.linspace(0, 1, 40)
    h = .0072 if happy else -.0045
    arc = np.c_[cx + s * (-.019 + .039 * t), cy + .0005 + h * np.sin(np.pi * t) + .0018 * t]
    st.line(arc, .0030 * np.sin(np.pi * np.clip(t, .04, .96)) ** .4 + .0008, LASH, .0018)


def brows(st, lift=0.0):
    for s in (1, -1):
        bx = np.linspace(.017, .060, 18)
        u = (bx - .017) / .043
        st.line(np.c_[s * bx, 1.9285 + .0034 * np.sin(np.pi * u ** .8) + lift * (1 - u) - .0012 * u], .0026 * (1 - .65 * u) + .0005, BROW, .0012)


def mouth(st, kind):
    mx = np.linspace(-.0105, .0105, 24)
    y0 = 1.8372
    if kind == "smile":
        st.line(np.c_[mx, y0 - .0016 + .0034 * (mx / .0105) ** 2], .0011, LIP, .0010)
    elif kind == "open":
        top = np.c_[mx, y0 + .0004 + .0012 * (mx / .0105) ** 2]
        bot = np.c_[mx[::-1], y0 - .0062 + .0060 * (mx[::-1] / .0105) ** 2]
        st.poly(np.vstack([top, bot]), (.45, .14, .16), .0010)
        st.poly(ell(0, y0 - .0045, .0045, .0016), (.90, .52, .55), .0012)
    elif kind == "smirk":
        u = np.linspace(-1, 1, 24)
        st.line(np.c_[u * .0095, y0 - .0010 + .0016 * u ** 2 + .0016 * np.clip(u, 0, None) ** 2], .0011, LIP, .0010)


EXPRESSIONS = {"neutral": dict(eyes=("open", "open"), mouth="smile", lift=0.0),
               "happy": dict(eyes=("happy", "happy"), mouth="open", lift=.0015),
               "playful wink": dict(eyes=("open", "happy"), mouth="smirk", lift=.0008)}


def stickers(name):
    e = EXPRESSIONS[name]
    st = Sticker()
    for s, kind in zip((-1, 1), e["eyes"]):          # her right, then her left
        eye_open(st, s) if kind == "open" else eye_closed(st, s)
    brows(st, e["lift"])
    mouth(st, e["mouth"])
    return st


_mats = {}


def flat(col):
    if col not in _mats:
        _mats[col] = tr.make_mat(f"ink{len(_mats)}", col, .55)
    return _mats[col]


def add(V, F, col, parent):
    me = bpy.data.meshes.new("sticker")
    me.from_pydata((hd.mm(V) * tr.MM).tolist(), [], np.asarray(F).tolist())
    me.update()
    me.materials.append(flat(col))
    me.shade_smooth()
    o = bpy.data.objects.new("sticker", me)
    bpy.context.collection.objects.link(o)
    o.parent = parent
    return o


def shot(az, el, dist, target, res, tag):
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = res
    tr.camera(az, el, dist, target)
    p = OUT / f"_{tag}.png"
    tr.render_to(p)
    im = Image.open(p).convert("RGB")
    p.unlink()
    return im


def label(im, text):
    g = Image.new("RGB", (im.size[0], im.size[1] + 34), "white")
    g.paste(im, (0, 34))
    ImageDraw.Draw(g).text((8, 5), text, fill=(20, 20, 30), font=FONT)
    return g


def row(ims):
    w, h = ims[0].size
    g = Image.new("RGB", (w * len(ims), h), "white")
    for i, im in enumerate(ims):
        g.paste(im, (i * w, 0))
    return g


def main():
    legs, assembled = sys.argv[1], Path(sys.argv[2])
    mats = tr.scene((800, 1300))
    root = bpy.data.objects.new("root", None)
    bpy.context.collection.objects.link(root)
    root.location = (0, 0, -tr.FLOOR * tr.MM)
    tr.build_legs(legs, mats, root)
    for f in sorted(assembled.glob("T0*.stl")):
        tr.stl(f, mats["shell"], root)
    tr.stl(assembled / "H11_face.stl", mats["shell"], root, 60)
    tr.stl(assembled / "H12_hair.stl", mats["shell"], root, 28)
    head = (-.010, 0, (300.0 - tr.FLOOR) * tr.MM)
    shots = {}
    for name in EXPRESSIONS:
        objs = [add(V, F, c, root) for V, F, c in stickers(name).parts]
        if name == "neutral":
            shots["neutral_q"] = label(shot(35, 4, .36, head, (640, 640), "q"), "neutral, three-quarter")
            shots["neutral_s"] = label(shot(60, 3, .36, head, (640, 640), "s"), "neutral, 60 degrees")
            body = shot(30, 5, 2.25, (0, 0, (-30.0 - tr.FLOOR) * tr.MM), (800, 1300), "body")
        shots[name] = label(shot(0, 3, .33, head, (640, 640), name), name)
        for o in objs:
            bpy.data.objects.remove(o)
    sheet = Image.new("RGB", (640 * 3, 674 * 2), "white")
    sheet.paste(row([shots["neutral"], shots["neutral_q"], shots["neutral_s"]]), (0, 0))
    sheet.paste(row([shots["happy"], shots["playful wink"]]), (0, 674))
    sheet.save(OUT / "stickers.png")
    body.save(OUT / "stickers_whole.png")


if __name__ == "__main__":
    main()
