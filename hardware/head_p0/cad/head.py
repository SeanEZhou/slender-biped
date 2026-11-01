"""Printable head H1: a face part that bolts over the T06 neck, and a hair cap that lifts straight off it.

    .venv-cad/bin/python hardware/head_p0/cad/head.py [--assembled <dir>]

Build units below are "u" (the head is about 0.27 u tall); SCALE turns them into mm in the robot frame (+X forward,
+Y left, +Z up). Hair is a signed-distance field: a round cap plus blended locks that all start at the crown.
Face features are relief only, to guide painting. The two parts meet on a vertical bore, so the outer shape is unchanged
apart from the seam. Outputs: exports/print_ready, head_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as ni
from scipy.interpolate import PchipInterpolator as PI
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "torso_p0" / "cad"))
import shape as sh  # noqa: E402
import torso as ts  # noqa: E402

SCALE = 365.0              # mm per u: head about 96 mm tall
CHIN_Z = 262.0             # robot-frame height of u = CHIN_U
CHIN_U = 1.806
CENTRE_U = -0.016          # neck axis in u
WALL = 2.4
COLLAR_R = 16.0            # outer collar around the neck socket: 2.4 mm wall over the 13.4 mm bore
COLLAR_TOP = 285.0
SLIDE = 0.15               # per side, as the torso sleeves
PASSAGE_R = 10.0           # keeps the neck's 18.5 mm clear circle open into the head
WINDOW_TOP = 1.962         # u: the face window stops here, below the face part's rim
FACE_TOP = 322.0           # open rim of the face part
BORE_SPLIT = 285.4         # the bore follows the face window below this, the face part's own outline above it
ROOF_SKIN = 4.0            # crown left above the flat top of the hair's roof
FIT = 0.25                 # per side between face part and hair cap
HAIR_SCREWS = (180, 90, -90)      # degrees from +X: horizontal and exposed, like the torso seams
HAIR_SCREW_Z = 284.0       # low on the hair, where it is thin enough to leave the heads near the surface
HAIR_GRIP = 6.6            # hair under the screw head; with the pilot this takes an M3x12
HAIR_SCREW_L = 12.0
HAIR_PILOT = 6.5           # leaves the screw tip about 1 mm short of the bottom
PAD_R = 4.5
HEAD_R = 3.2               # screw head clearance
DENSITY = 1.24e-3
BED = 256.0
H = 0.0024                 # hair field spacing, u


def scale_mm(v):
    """u -> robot frame mm. The neck axis sits at the X of T06's neck."""
    v = np.asarray(v)
    return np.c_[v[:, 2] * SCALE + NECK_X - CENTRE_U * SCALE, v[:, 0] * SCALE, (v[:, 1] - CHIN_U) * SCALE + CHIN_Z]


# ---------- face ----------
Y_T = [1.803, 1.815, 1.835, 1.862, 1.895, 1.935, 1.985, 2.025, 2.05]
A_T = [.015, .033, .054, .070, .078, .082, .082, .058, .012]
F_T = [.054, .062, .070, .077, .081, .081, .074, .040, .010]
B_T = [.010, .015, .030, .055, .075, .092, .098, .068, .012]
FN, FZ0 = 2.3, -.008
EYE = (.0385, 1.902)
MOUTH_Y, MOUTH_W, SMILE = 1.8352, .0125, .0026       # corners rise SMILE above the middle: about 0.95 mm


def mouth_dy(x):
    return SMILE * ((x / MOUTH_W) ** 2 - 2 / 3)


def gauss(cx, cy, rx, ry, h):
    return lambda x, y: h * np.exp(-((x - cx) / rx) ** 2 - ((y - cy) / ry) ** 2)


def smile(cy, rx, ry, h):
    """A lip or groove that follows the curve of the mouth."""
    return lambda x, y: h * np.exp(-(x / rx) ** 2 - ((y - cy - mouth_dy(x)) / ry) ** 2)


class FeatureMap:
    """Raised and engraved paint guides, rasterised to a height map (u)."""
    U, X0, Y0, N = .00025, -.095, 1.80, 760

    def __init__(self):
        self.h = np.zeros((self.N, self.N), "f4")

    def _px(self, pts):
        return [((x - self.X0) / self.U, (y - self.Y0) / self.U) for x, y in pts]

    def add(self, draw_fn, mm, blur=1.5):
        im = Image.new("F", (self.N, self.N), 0.0)
        draw_fn(ImageDraw.Draw(im))
        self.h += ni.gaussian_filter(np.array(im), blur) * (mm / SCALE)

    def poly(self, pts, mm, blur=1.5):
        self.add(lambda d: d.polygon(self._px(pts), fill=1.0), mm, blur)

    def line(self, pts, width_u, mm, blur=1.5):
        self.add(lambda d: d.line(self._px(pts), fill=1.0, width=max(1, int(width_u / self.U)), joint="curve"), mm, blur)

    def __call__(self, x, y):
        return ni.map_coordinates(self.h, [(y - self.Y0) / self.U, (x - self.X0) / self.U], order=1, mode="constant")


def features():
    m = FeatureMap()
    for s in (1, -1):
        cx, cy = s * EYE[0], EYE[1]
        t = np.linspace(0, 1, 40)
        top = np.c_[cx + s * (.0245 - .049 * t), cy + .0005 + .0098 * np.sin(np.pi * t) ** .55]
        a = np.linspace(0, 2 * np.pi, 60)
        ix, iy = cx - s * .001, cy - .0003
        iris = np.c_[ix + .0098 * np.cos(a), np.minimum(iy + .0098 * np.sin(a), cy + .0082)]
        m.poly(iris, 0.35)                                       # iris: a low disc, easy to paint up to
        m.poly(np.c_[ix + .0042 * np.cos(a), iy + .0042 * np.sin(a)], -0.25)     # pupil: a shallow dimple
        m.line(top, .0044, 0.65)                                 # upper lid
        bx = np.linspace(.016, .072, 14)
        m.line(np.c_[s * bx, 1.934 + .0028 * np.sin(np.pi * (bx - .016) / .056) - .003 * ((bx - .016) / .056)], .0026, 0.45)  # brow
    mx = np.linspace(-MOUTH_W, MOUTH_W, 20)
    m.line(np.c_[mx, MOUTH_Y + mouth_dy(mx)], .0021, -0.50)      # mouth line
    return m


class Face:
    def __init__(self, detail=True):
        y = np.linspace(Y_T[0], Y_T[-1], 400)
        sm = lambda v: ni.gaussian_filter1d(PI(Y_T, v)(y), 6, mode="nearest")  # noqa: E731
        self.ys, self._a, self._f, self._b = y, sm(A_T), sm(F_T), sm(B_T)
        self.relief = []
        if detail:
            for s in (1, -1):
                self.relief += [gauss(s * EYE[0], EYE[1], .03125, .01375, -.003), gauss(s * EYE[0], EYE[1] + .01265, .03, .00462, .002)]
            self.relief += [gauss(0, 1.866, .0085, .012, .0052), gauss(0, 1.894, .006, .03, .0024)]                    # nose
            self.relief += [smile(MOUTH_Y, .012, .0016, -.0009), smile(1.8307, .0084, .0034, .00128), smile(1.8386, .0096, .0026, .0008)]  # lips
            self.relief.append(features())

    def a(self, y):
        return np.interp(y, self.ys, self._a)

    def z(self, x, y):
        """Front surface, for placing paint previews."""
        a = self.a(y)
        u = np.clip(np.abs(x) / np.maximum(a, 1e-4), 0, 1)
        return FZ0 + np.interp(y, self.ys, self._f) * (1 - u ** FN) ** (1 / FN) + self.D(x, y)

    def D(self, x, y):
        return sum((fn(x, y) for fn in self.relief), 0 * x)

    def mesh(self):
        """Closed loft, with its points concentrated on the features."""
        front = np.linspace(-.95, .95, 300)
        back = np.linspace(.95, 2 * np.pi - .95, 90)[1:-1]
        az = np.r_[front, back]
        y = np.unique(np.r_[np.linspace(Y_T[0], 1.82, 14), np.linspace(1.82, 1.96, 190), np.linspace(1.96, Y_T[-1], 26)])
        Yg, T = np.meshgrid(y, az, indexing="ij")
        S, C = np.sin(T), np.cos(T)
        x = self.a(Yg) * np.sign(S) * np.abs(S) ** (2 / FN)
        depth = np.where(C > 0, np.interp(Yg, self.ys, self._f), np.interp(Yg, self.ys, self._b))
        z = FZ0 + depth * np.sign(C) * np.abs(C) ** (2 / FN) + np.where(C > 0, self.D(x, Yg), 0)
        nu, nv = Yg.shape
        V = np.stack([x, Yg, z], -1).reshape(-1, 3)
        F = []
        for i in range(nu - 1):
            for j in range(nv):
                a, b = i * nv + j, i * nv + (j + 1) % nv
                c, d = (i + 1) * nv + (j + 1) % nv, (i + 1) * nv + j
                F += [(a, b, c), (a, c, d)]
        lo, hi = len(V), len(V) + 1
        V = np.vstack([V, V[:nv].mean(0), V[-nv:].mean(0)])
        F += [(lo, (j + 1) % nv, j) for j in range(nv)] + [(hi, (nu - 1) * nv + j, (nu - 1) * nv + (j + 1) % nv) for j in range(nv)]
        m = trimesh.Trimesh(V, np.array(F), process=True)
        if m.volume < 0:
            m.invert()
        return m


# ---------- hair ----------
class Field:
    def __init__(self):
        self.xs, self.ys, self.zs = np.arange(-.16, .16, H), np.arange(1.70, 2.13, H), np.arange(-.19, .15, H)
        self.X, self.Y, self.Z = [a.astype("f4") for a in np.meshgrid(self.xs, self.ys, self.zs, indexing="ij")]
        self.d = np.full(self.X.shape, 1e3, "f4")

    def capsules(self, P, r, k):
        P, r = np.asarray(P, "f4"), np.asarray(r, "f4")
        pad = float(r.max()) + .012
        lo, hi = P.min(0) - pad, P.max(0) + pad
        ix = tuple(slice(np.searchsorted(g, lo[i]), np.searchsorted(g, hi[i])) for i, g in enumerate((self.xs, self.ys, self.zs)))
        X, Y, Z = self.X[ix], self.Y[ix], self.Z[ix]
        best = np.full(X.shape, 1e3, "f4")
        for i in range(len(P) - 1):
            a, ab = P[i], P[i + 1] - P[i]
            t = np.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / ((ab @ ab) + 1e-12), 0, 1)
            dd = np.sqrt((X - a[0] - t * ab[0]) ** 2 + (Y - a[1] - t * ab[1]) ** 2 + (Z - a[2] - t * ab[2]) ** 2) - (r[i] + (r[i + 1] - r[i]) * t)
            best = np.minimum(best, dd)
        a = self.d[ix]
        h = np.clip(.5 + .5 * (best - a) / k, 0, 1)
        self.d[ix] = best * (1 - h) + a * h - k * h * (1 - h)

    def cut(self, b, k):
        h = np.clip(.5 - .5 * (self.d + b) / k, 0, 1)
        self.d = self.d * (1 - h) + (-b) * h + k * h * (1 - h)


class Crown:
    """Round cap: an ellipsoid above the widest point, then the listed lower sections."""
    TOP, YC, A, BF, BB, N, Z0 = 2.066, 1.95, .104, .092, .114, 2.0, -.012
    LOW = [(1.90, .103, .09, .11), (1.865, .094, .083, .1), (1.838, .076, .068, .08), (1.822, .054, .05, .058)]

    def __init__(self):
        q = np.linspace(0, 1, 14)[1:]
        k = np.sqrt(np.maximum(1 - q * q, 1e-4))
        Y = np.r_[self.YC, self.YC + (self.TOP - self.YC) * q, [l[0] for l in self.LOW]]
        o = np.argsort(Y)
        self.gy = np.linspace(Y.min(), Y.max(), 300)
        self.y0, self.y1 = Y.min(), Y.max()
        self._a = PI(Y[o], np.r_[self.A, self.A * k, [l[1] for l in self.LOW]][o])(self.gy)
        self._f = PI(Y[o], np.r_[self.BF, self.BF * k, [l[2] for l in self.LOW]][o])(self.gy)
        self._b = PI(Y[o], np.r_[self.BB, self.BB * k, [l[3] for l in self.LOW]][o])(self.gy)

    def a(self, y):
        return np.interp(y, self.gy, self._a)

    def bf(self, y):
        return np.interp(y, self.gy, self._f)

    def bb(self, y):
        return np.interp(y, self.gy, self._b)

    def field(self, F):
        A, dz = self.a(F.Y), F.Z - self.Z0
        B = np.where(dz > 0, self.bf(F.Y), self.bb(F.Y))
        r = (np.abs(F.X) / np.maximum(A, 1e-3)) ** self.N + (np.abs(dz) / np.maximum(B, 1e-3)) ** self.N
        d = (r ** (1 / self.N) - 1) * np.minimum(A, B).clip(.02, .2)
        d = np.where((F.Y > self.y1) | (F.Y < self.y0), np.maximum(d, np.maximum(F.Y - self.y1, self.y0 - F.Y)), d)
        return np.minimum(d, .3).astype("f4")

    def pt(self, az, y, off=0.):
        S, C = np.sin(az), np.cos(az)
        y = np.clip(y, self.y0, self.y1)
        b = (self.bf(y) if C > 0 else self.bb(y)) + off
        return np.array(((self.a(y) + off) * np.sign(S) * np.abs(S) ** (2 / self.N), y, self.Z0 + b * np.sign(C) * np.abs(C) ** (2 / self.N)))


def lock(F, P, w, t, k):
    """Wide flat lock: three blended rows that meet at the crown and at a rounded tip."""
    n = len(P)
    q = np.linspace(0, 1, n)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    up = P - np.c_[0 * P[:, 0], P[:, 1], np.full(n, -.012)]
    up[:, 1] *= .25
    up /= np.linalg.norm(up, axis=1, keepdims=True) + 1e-9
    Sd = np.cross(T, up)
    Sd /= np.linalg.norm(Sd, axis=1, keepdims=True) + 1e-9
    wid = w * np.minimum(1, 1.08 - q ** 2.5) * np.minimum(1, q * 3.)
    rad = np.maximum(t * (1 - .45 * q ** 2), .0035) * np.minimum(1, .05 + q * 1.6)
    F.capsules(P, rad, k)
    side = Sd * (wid * .55 * (1 - .6 * q ** 3))[:, None]
    F.capsules(P + side, rad * .9, k)
    F.capsules(P - side, rad * .9, k)


def strand(F, c, az, y1, w, t, off, curl=0., n=26, k=.015):
    q = np.linspace(0, 1, n)
    P = np.array([c.pt(az, v, off) for v in c.y1 + (y1 - c.y1) * q])
    u = P - np.c_[0 * P[:, 0], P[:, 1], np.full(n, c.Z0)]
    u[:, 1] = 0
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    lock(F, P - u * (curl * q ** 3)[:, None], w, t, k)


FRINGE = [(-.066, 1.93, .014), (-.04, 1.945, .015), (.012, 1.885, .0075), (.044, 1.94, .015), (.067, 1.925, .014)]  # tip x, y, width


def hair():
    F, c = Field(), Crown()
    F.d = c.field(F)
    r = np.random.default_rng(4)
    for a in np.linspace(1.12, np.pi, 8):                # sides and back, tips at the hem
        for s in (1, -1):
            az = s * a
            y1 = 1.824 + r.uniform(-.01, .01)
            q = np.linspace(0, 1, 26)
            azs = az + s * r.uniform(0, .05) * q ** 1.5
            P = np.array([c.pt(i, j, -.0014) for i, j in zip(azs, c.y1 + (y1 - c.y1) * q)])
            u = P - np.c_[0 * P[:, 0], P[:, 1], np.full(26, c.Z0)]
            u[:, 1] = 0
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
            lock(F, P - u * (.006 * q ** 3)[:, None], .024, .0064, .015)
    for s, y in ((1, 1.80), (-1, 1.804)):                # face-framing strands
        strand(F, c, s * .92, y, .012, .0056, -.0006, curl=.004, n=30, k=.016)
    for x, _, _ in FRINGE:                               # fringe strands, lined up with the scallops
        strand(F, c, np.arcsin(np.clip(x / float(c.a(1.95)), -1, 1)), 1.905, .016, .0052, -.0016, n=20, k=.016)
    F.capsules([[0, c.y1 - .036, c.Z0], [0, c.y1 - .035, c.Z0]], [.035, .035], .012)
    # completely smooth top: the apex is a true ellipsoid, and the strands fade into it
    dz = F.Z - c.Z0
    B = np.where(dz > 0, c.BF, c.BB)
    ell = (np.sqrt((F.X / c.A) ** 2 + ((F.Y - c.YC) / (c.TOP - c.YC)) ** 2 + (dz / B) ** 2) - 1) * .1
    top = F.Y > c.YC + .04
    F.d = np.where(top, np.minimum(F.d, ell), F.d)
    q = np.clip(np.hypot(F.X, dz) / .06, 0, 1)
    lim = np.where(top, ell - .0058 - .012 * q * q * (3 - 2 * q), -9.).astype("f4")
    h = np.clip(.5 - .5 * (F.d - lim) / .005, 0, 1)
    F.d = (F.d * (1 - h) + lim * h + .005 * h * (1 - h)).astype("f4")
    # face window, its top edge cut into rounded scallops
    best = np.full(F.X.shape, 9., "f4")
    for xt, yt, wx in FRINGE:
        cc = yt + ((F.X - xt) / wx) ** 2 * .012
        h = np.clip(.5 + .5 * (best - cc) / .012, 0, 1)
        best = best * (1 - h) + cc * h - .012 * h * (1 - h)
    h = np.clip(.5 + .5 * (best - WINDOW_TOP) / .008, 0, 1)
    best = best * (1 - h) + WINDOW_TOP * h - .008 * h * (1 - h)
    F.cut(np.maximum(np.maximum(F.Y - best, (np.abs(F.X) - .073) * .7), .012 - F.Z).astype("f4"), .011)
    # hollow under the hem, around the neck
    rx, rz = .07, .076
    F.cut(np.maximum((np.sqrt((F.X / rx) ** 2 + ((F.Z + .016) / rz) ** 2) - 1) * rx, F.Y - 1.858).astype("f4"), .016)
    v, f, _, _ = measure.marching_cubes(F.d, 0.0, spacing=(H, H, H))
    v += np.array((F.xs[0], F.ys[0], F.zs[0]))
    m = trimesh.Trimesh(v, f[:, [0, 2, 1]], process=True)
    m = max(m.split(only_watertight=False), key=lambda p: len(p.faces))
    if m.volume < 0:
        m.invert()
    return m


# ---------- the neck interface, taken from the torso ----------
RIM = sh.solid(sh.BODY).bounds[1][2] - 0.5
NECK_X = float(np.mean(ts.ring(RIM, 0.0)[:, 0]))


def neck_socket():
    """Straight bore over the top SLEEVE_H of the neck, a 45 deg seat that stops on the rim, then the cable passage."""
    r = np.max([ts.radial(z, 0.0, NECK_X) for z in np.linspace(RIM - ts.SLEEVE_H, RIM, 9)], axis=0) + SLIDE
    r_rim = ts.radial(RIM, 0.0, NECK_X)
    z0 = RIM - ts.SLEEVE_H - 2.0
    seat = float((r - r_rim).min())                      # the seat first touches the neck's outer edge at rim height
    rings = [(z0, r), (RIM - seat, r), (RIM - seat + (r.max() - PASSAGE_R), r - (r.max() - PASSAGE_R))]
    bore = ts.loft(rings, NECK_X)
    return bore, ts.rod((NECK_X, 0, RIM), (0, 0, 1), PASSAGE_R, COLLAR_TOP + 15 - RIM), r_rim


def screw_holes():
    out = []
    for deg in ts.NECK_SCREWS:
        ph = np.radians(deg)
        er = np.array((np.cos(ph), np.sin(ph), 0.))
        out.append((np.array((NECK_X, 0, RIM - ts.SLEEVE_H / 2)), er))
    return out


def cavity(outer, keep_r=COLLAR_R + 1.0, pitch=0.8, roof=False):
    """Hollow interior: everything further than WALL from the skin, optionally under a 45 deg roof that prints without supports."""
    import manifold3d as m3
    bb = outer.bounds
    zs = np.arange(bb[0][2] + pitch / 2, bb[1][2], pitch)
    nx, ny = int(np.ceil((bb[1][0] - bb[0][0]) / pitch)) + 4, int(np.ceil((bb[1][1] - bb[0][1]) / pitch)) + 4
    x0, y0 = bb[0][0] - 2 * pitch, bb[0][1] - 2 * pitch
    mf = m3.Manifold(m3.Mesh(np.asarray(outer.vertices, "f4"), np.asarray(outer.faces, "u4")))
    SS = 4                                               # sub-voxels per voxel, so outlines don't fatten the solid
    solid = np.zeros((nx, ny, len(zs)), bool)
    for k, z in enumerate(zs):
        arr = np.zeros((nx * SS, ny * SS), bool)
        for loop in mf.slice(float(z)).to_polygons():
            im = Image.new("1", (ny * SS, nx * SS), 0)
            ImageDraw.Draw(im).polygon([((p[1] - y0) / pitch * SS, (p[0] - x0) / pitch * SS) for p in loop], fill=1)
            arr ^= np.array(im)
        solid[:, :, k] = arr.reshape(nx, SS, ny, SS).mean((1, 3)) >= 0.5       # voxel centres sit at (i + 1/2) pitch
    dist = ni.distance_transform_edt(np.pad(solid, 1))[1:-1, 1:-1, 1:-1] * pitch      # padded: the array's edge counts as outside
    X, Y = np.meshgrid(x0 + (np.arange(nx) + .5) * pitch, y0 + (np.arange(ny) + .5) * pitch, indexing="ij")
    cav = dist > WALL
    rr = np.hypot(X - NECK_X, Y)
    for k, z in enumerate(zs):
        if z < RIM + 5:                                  # leave the collar and the seat solid
            cav[:, :, k] &= rr >= keep_r
    fill = ~cav
    if roof:                                             # cap the cavity under a height map whose slope is at most one voxel per voxel
        nz = cav.shape[2]
        top = np.where(cav.any(2), nz - 1 - np.argmax(cav[:, :, ::-1], axis=2), 10 * nz)     # columns with no cavity set no limit
        cross = ni.generate_binary_structure(2, 1)
        for _ in range(nz + nx):
            new = np.minimum(top, ni.minimum_filter(top, footprint=cross, mode="nearest") + 1)
            if (new == top).all():
                break
            top = new
        fill = ~(cav & (np.arange(nz)[None, None, :] <= top[:, :, None]))
    lab, n = ni.label(~fill)
    if n > 1:                                            # only the volume the passage reaches: no sealed pockets
        ax = np.argmin(np.abs(x0 + (np.arange(nx) + .5) * pitch - NECK_X))
        ay = np.argmin(np.abs(y0 + (np.arange(ny) + .5) * pitch))
        col = lab[ax, ay, :]
        keep = col[col > 0][0] if (col > 0).any() else np.argmax(np.bincount(lab.ravel())[1:]) + 1
        fill = lab != keep
    field = ni.gaussian_filter((~fill).astype("f4"), 0.8)
    v, f, _, _ = measure.marching_cubes(np.pad(field, 2), 0.5, spacing=(pitch, pitch, pitch))
    v += np.array((x0 - 1.5 * pitch, y0 - 1.5 * pitch, zs[0] - 2 * pitch))
    c = trimesh.Trimesh(v, f[:, [0, 2, 1]], process=True)
    if c.volume < 0:
        c.invert()
    return c


def collar():
    z0 = RIM - ts.SLEEVE_H
    rings = [(z0, np.full(ts.PHI.shape, COLLAR_R)), (COLLAR_TOP, np.full(ts.PHI.shape, COLLAR_R))]
    return ts.loft(rings, NECK_X)


def hull_radius(pts, phi=None):
    """Radius around the neck axis of the convex outline of points (mm, X and Y), at each of ts.PHI."""
    from scipy.spatial import ConvexHull
    eq = ConvexHull(np.asarray(pts) - (NECK_X, 0)).equations
    phi = ts.PHI if phi is None else phi
    d = np.c_[np.cos(phi), np.sin(phi)]
    den = d @ eq[:, :2].T
    return np.min(np.where(den > 1e-9, -eq[:, 2] / np.maximum(den, 1e-9), 1e9), axis=1)


def bore_outline():
    """The opening through the hair hem: the old hem hollow at the back, the face window's width at the front."""
    t = np.linspace(0, 2 * np.pi, 180)
    ell = np.c_[NECK_X + .076 * SCALE * np.cos(t), .07 * SCALE * np.sin(t)]
    front = np.array([(NECK_X + d, s * .0735 * SCALE) for d in (10.2, 60.) for s in (1, -1)])
    return np.vstack([ell, front])


def prism(r, z0, z1):
    return ts.loft([(z0, r), (z1, r)], NECK_X)


def hair_screw_axes():
    """Crossing point on the bore and the outward direction of each hair screw."""
    q = hull_radius(bore_outline(), np.radians(HAIR_SCREWS))
    out = []
    for deg, r in zip(HAIR_SCREWS, q):
        er = np.array((np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.))
        out.append((np.array((NECK_X, 0, HAIR_SCREW_Z)) + er * (r - 0.05), er))
    return out


def screw_pads():
    """One pad per hair screw, on a 45 deg gusset down to the collar so it prints without supports."""
    pads = []
    q = prism(hull_radius(bore_outline()), RIM - 12, FACE_TOP)
    for c, d in hair_screw_axes():
        er = d[:2] / np.linalg.norm(d[:2])
        base = ts.rod(c + d * 2, -d, PAD_R, HAIR_PILOT + 4.0)
        foot = trimesh.creation.icosphere(subdivisions=1, radius=0.8).apply_translation((NECK_X + er[0] * 14.5, er[1] * 14.5, 266.5))
        pads.append(ts.inter(trimesh.convex.convex_hull(np.vstack([base.vertices, foot.vertices])), q))
    return pads


def build(detail=True):
    hair_m, face_m = hair(), Face(detail).mesh()
    hair_m = trimesh.Trimesh(scale_mm(hair_m.vertices), hair_m.faces, process=False)
    face_m = trimesh.Trimesh(scale_mm(face_m.vertices), face_m.faces, process=False)
    outer = ts.union(hair_m, face_m, collar())
    bore_r = hull_radius(bore_outline())
    low = RIM - 40
    # face part: closed at first, so the wall follows the skin right up through the rim, then opened at the rim
    face_in = ts.inter(face_m, prism(bore_r, low, RIM + 90))
    pads = screw_pads()
    shadow = np.vstack([ts.inter(face_in, prism(bore_r, low, FACE_TOP)).vertices[:, :2]] + [p.vertices[:, :2] for p in pads])
    face_r = hull_radius(shadow)
    body = ts.union(face_in, collar())
    top_cut = prism(np.full(ts.PHI.shape, 90.), FACE_TOP, RIM + 120)
    bore, passage, _ = neck_socket()
    neck_holes = [ts.rod(c + er * 8.0, er, ts.M3_CLEAR / 2, 10.0) for c, er in screw_holes()]
    pilots = [ts.rod(c + d * 1.0, -d, ts.M3_PILOT / 2, HAIR_PILOT + 1.0) for c, d in hair_screw_axes()]
    face_part = ts.union(ts.diff(body, cavity(body)), *pads)
    face_part = ts.diff(face_part, top_cut, bore, passage, *neck_holes, *pilots)
    # hair cap: everything outside the bore, hollowed; every roof inside is 45 deg, so the inside prints without supports
    upper_r = face_r + FIT
    seat = FACE_TOP - FIT + 0.05                         # the face rim stops on the roof's slope
    roof = [(seat + d, np.maximum(upper_r - d, 1.0)) for d in np.arange(0.0, min(upper_r.max(), hair_m.bounds[1][2] - ROOF_SKIN - seat), 1.0)]
    hair_part = ts.diff(hair_m, prism(bore_r + FIT, low, BORE_SPLIT + 0.01), prism(upper_r, BORE_SPLIT, seat + 0.01), ts.loft(roof, NECK_X), cavity(hair_m, keep_r=0.0, roof=True))
    holes, bosses = [], []
    for c, d in hair_screw_axes():
        bosses.append(ts.diff(ts.inter(ts.rod(c, d, HEAD_R + WALL, 40.0), hair_m), prism(bore_r + FIT, low, FACE_TOP)))
        holes += [ts.rod(c - d, d, ts.M3_CLEAR / 2, HAIR_GRIP + 2.0), ts.rod(c + d * HAIR_GRIP, d, HEAD_R, 40.0)]
    hair_part = ts.diff(ts.union(hair_part, *bosses), *holes)
    return face_part, hair_part, outer


def stl_safe(m, tol=0.003):
    """Collapse the sliver edges booleans leave, which STL's 32-bit floats would weld into non-manifold edges on saving."""
    import manifold3d as m3
    mf = m3.Manifold(m3.Mesh(np.asarray(m.vertices, "f4"), np.asarray(m.faces, "u4"))).set_tolerance(tol)
    r = mf.to_mesh()
    return trimesh.Trimesh(np.asarray(r.vert_properties)[:, :3].astype("f8"), np.asarray(r.tri_verts), process=False)


def largest(m):
    """Keep the main body. Returns it and the volume of whatever was dropped."""
    bits = m.split(only_watertight=False)
    best = max(bits, key=lambda p: p.volume)
    return best, float(sum(p.volume for p in bits) - best.volume)


def check(face, hair_p, outer):
    rep = dict(rim_z=round(float(RIM), 2), solid_outer_mass_g=round(outer.volume * DENSITY, 1))
    for name, m in (("face", face), ("hair", hair_p)):
        rep[name] = dict(watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)), faces=len(m.faces), bbox_mm=[round(float(v), 1) for v in m.extents],
                         fits_bed=bool(m.extents.max() <= BED), solid_mass_g=round(m.volume * DENSITY, 1), com_mm=[round(float(v), 1) for v in m.center_mass],
                         bottom_z=round(float(m.bounds[0][2]), 2), top_z=round(float(m.bounds[1][2]), 2))
    v = face.volume + hair_p.volume
    rep["total"] = dict(solid_mass_g=round(v * DENSITY, 1), com_mm=[round(float(x), 1) for x in (face.center_mass * face.volume + hair_p.center_mass * hair_p.volume) / v])
    rep["face_hair_overlap_mm3"] = round(float(ts.inter(face, hair_p).volume), 3)
    # the cap must come straight off: lift it in steps and look for any collision with the face part
    rep["lift_overlap_mm3"] = {}
    for dz in (0.5, 2, 5, 10, 20, 40, 70):
        rep["lift_overlap_mm3"][str(dz)] = round(float(ts.inter(face, hair_p.copy().apply_translation((0, 0, dz))).volume), 3)
    t06 = ts.build()["T06_shoulders"]
    rep["t06_overlap_mm3"] = dict(face=round(float(ts.inter(face, t06).volume), 2), hair=round(float(ts.inter(hair_p, t06).volume), 2))
    # screwdriver reach: a 6 mm cylinder straight out from each hole, starting at the screw head's seat
    rep["neck_driver_paths_mm3"], rep["hair_driver_paths_mm3"] = {}, {}
    for (c, er), deg in zip(screw_holes(), ts.NECK_SCREWS):
        path = ts.rod(c + er * (COLLAR_R + 0.4), er, 3.0, 60.0)
        rep["neck_driver_paths_mm3"][str(deg)] = round(float(ts.inter(face, path).volume + ts.inter(hair_p, path).volume), 2)
    for (c, d), deg in zip(hair_screw_axes(), HAIR_SCREWS):
        path = ts.rod(c + d * (HAIR_GRIP + 0.2), d, 3.0, 60.0)
        rep["hair_driver_paths_mm3"][str(deg)] = dict(head=round(float(ts.inter(face, path).volume + ts.inter(hair_p, path).volume), 2), t06=round(float(ts.inter(t06, path).volume), 2))
    # each hair screw, on its shared axis: a clear path through the cap, then a full bite in the pad
    rep["hair_screws"] = {}
    for (c, d), deg in zip(hair_screw_axes(), HAIR_SCREWS):
        tip = HAIR_GRIP - HAIR_SCREW_L
        core = ts.rod(c + d * tip, d, 1.2, HAIR_SCREW_L)
        thread = ts.rod(c + d * tip, d, 1.5, -tip)
        hole = ts.rod(c + d * tip, d, ts.M3_PILOT / 2, -tip)
        bite = ts.diff(thread, hole)
        ring = ts.diff(ts.rod(c + d * (HAIR_GRIP - .6), d, 2.75, .6), ts.rod(c + d * (HAIR_GRIP - 1), d, ts.M3_CLEAR / 2, 2.))
        rep["hair_screws"][str(deg)] = dict(core_vs_hair_mm3=round(float(ts.inter(core, hair_p).volume), 3), core_vs_face_mm3=round(float(ts.inter(core, face).volume), 3),
                                            thread_vs_hair_mm3=round(float(ts.inter(ts.rod(c + d * .3, d, 1.5, HAIR_GRIP - .3), hair_p).volume), 3),
                                            thread_bite_mm=round(float(ts.inter(bite, face).volume / bite.volume) * -tip, 2), tip_clear_mm=round(HAIR_PILOT + tip, 2),
                                            head_seat_on_hair=round(float(ts.inter(ring, hair_p).volume / ring.volume), 2))
    # clear path up the neck into the head
    rep["passage_overlap_mm3"] = round(float(ts.inter(face, ts.rod((NECK_X, 0, RIM - 1), (0, 0, 1), 18.5 / 2, 15.0)).volume), 2)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="", help="also write robot-frame copies for renders; not kept in the repo")
    a = ap.parse_args()
    face_part, hair_part, outer = build()
    face_part, dropped_f = largest(stl_safe(face_part))
    hair_part, dropped_h = largest(stl_safe(hair_part))
    out = ROOT / "exports" / "print_ready"
    out.mkdir(parents=True, exist_ok=True)
    rep = check(face_part, hair_part, outer)
    rep["dropped_mm3"] = dict(face=round(dropped_f, 2), hair=round(dropped_h, 2))
    for name, m in (("H01_face", face_part), ("H02_hair", hair_part)):
        if a.assembled:
            Path(a.assembled).mkdir(parents=True, exist_ok=True)
            m.export(Path(a.assembled) / f"{name}.stl")
        p = m.copy()
        p.apply_translation((BED / 2 - p.bounds.mean(axis=0)[0], BED / 2 - p.bounds.mean(axis=0)[1], -p.bounds[0][2]))
        p = stl_safe(p)
        p.export(out / f"{name}_print.stl")
        saved = trimesh.load(out / f"{name}_print.stl")
        rep[name.split("_")[1]]["saved_file"] = dict(watertight=bool(saved.is_watertight), volume=bool(saved.is_volume), bodies=len(saved.split(only_watertight=False)))
    (ROOT / "head_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
