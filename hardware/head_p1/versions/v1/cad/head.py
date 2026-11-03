"""Head H2 "Rebel Princess": a face with modelled bone structure, a hair cap with a crown braid, and two ear units.

    .venv-cad/bin/python hardware/head_p1/versions/v1/cad/head.py [--assembled <dir>]

Same frame, units and neck interface as head H1 (hardware/head_p0/cad/head.py): build units are "u", scaled into robot-frame
mm. Eyes, brows and mouth are stickers, so the face carries only shape: brow ridge, eye sockets, nose, cheekbones, chin.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage as ni
from scipy.interpolate import PchipInterpolator as PI

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("head_h1", Path(__file__).resolve().parents[4] / "head_p0" / "cad" / "head.py")
h1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1)
ts = h1.ts

SCALE, NECK_X = h1.SCALE, h1.NECK_X
DENSITY, BED = h1.DENSITY, h1.BED


def mm(v):
    return h1.scale_mm(np.asarray(v))


def to_mm(m):
    return trimesh.Trimesh(mm(m.vertices), m.faces, process=False)


# ---------- face: a lofted head with bone structure added as smooth height features on the front ----------
Y_T = [1.803, 1.815, 1.835, 1.862, 1.895, 1.935, 1.985, 2.025, 2.05]
A_T = [.012, .030, .051, .068, .077, .080, .080, .058, .012]       # half width
F_T = [.052, .061, .068, .075, .079, .080, .074, .040, .010]       # depth in front of the axis
B_T = [.010, .015, .030, .055, .075, .092, .098, .068, .012]       # depth behind it
FN, FZ0 = 2.4, -.008
EYE = (.0375, 1.901)


def bump(cx, cy, rx, ry, h, p=2.0, rot=0.0):
    """Smooth bump (h > 0) or dip (h < 0); p > 2 flattens its top, so a socket gets a floor a sticker can lie on."""
    c, s = np.cos(rot), np.sin(rot)

    def f(x, y):
        u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
        return h * np.exp(-(((u / rx) ** 2 + (v / ry) ** 2) ** (p / 2)))
    return f


def nose(x, y):
    """Bridge rising from between the eyes to a small rounded tip, then falling back sharply underneath."""
    top, tip = 1.912, 1.862
    t = np.clip((top - y) / (top - tip), 0, 1)
    h = .0072 * t ** 1.8
    w = .0042 + .0036 * t
    under = np.where(y < tip, np.exp(-((tip - y) / .0042) ** 2), 1.0)
    return h * np.exp(-(x / w) ** 2) * under


class Face:
    def __init__(self):
        y = np.linspace(Y_T[0], Y_T[-1], 500)
        sm = lambda v: ni.gaussian_filter1d(PI(Y_T, v)(y), 6, mode="nearest")  # noqa: E731
        self.ys, self._a, self._f, self._b = y, sm(A_T), sm(F_T), sm(B_T)
        ex, ey = EYE
        self.relief = [nose,
                       bump(0, 1.864, .0062, .005, .0022),                         # nose tip
                       bump(0, 1.84, .02, .013, .0016),                            # muzzle under the mouth sticker
                       bump(0, 1.814, .011, .008, .0016)]                          # chin
        for s in (1, -1):
            self.relief += [bump(s * ex, ey, .0265, .0175, -.0058, p=4, rot=s * .12),       # eye socket, flat floored
                            bump(s * (ex + .002), 1.929, .029, .0062, .0016, rot=-s * .1),  # brow ridge
                            bump(s * .0072, 1.8605, .0042, .003, .0013),                     # nose wings
                            bump(s * .051, 1.874, .022, .014, .0030),                       # cheekbone
                            bump(s * .073, 1.935, .012, .02, -.0024)]                       # temple

    def a(self, y):
        return np.interp(y, self.ys, self._a)

    def D(self, x, y):
        return sum((f(x, y) for f in self.relief), 0 * x)

    def z(self, x, y):
        a = self.a(y)
        u = np.clip(np.abs(x) / np.maximum(a, 1e-4), 0, 1)
        return FZ0 + np.interp(y, self.ys, self._f) * (1 - u ** FN) ** (1 / FN) + self.D(x, y)

    def mesh(self):
        front = np.linspace(-1.1, 1.1, 380)
        back = np.linspace(1.1, 2 * np.pi - 1.1, 90)[1:-1]
        az = np.r_[front, back]
        y = np.unique(np.r_[np.linspace(Y_T[0], 1.82, 20), np.linspace(1.82, 1.965, 260), np.linspace(1.965, Y_T[-1], 30)])
        Yg, T = np.meshgrid(y, az, indexing="ij")
        S, C = np.sin(T), np.cos(T)
        x = self.a(Yg) * np.sign(S) * np.abs(S) ** (2 / FN)
        depth = np.where(C > 0, np.interp(Yg, self.ys, self._f), np.interp(Yg, self.ys, self._b))
        z = FZ0 + depth * np.sign(C) * np.abs(C) ** (2 / FN) + self.D(x, Yg) * np.clip(C * 3, 0, 1)
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


# ---------- hair: tapered locks with a ridge, over a slim cap, under a crown braid ----------
class Env:
    """The surface the hair lies on: a dome, then down to the hem with a slight inward taper."""
    TOP, YC, A, BF, BB, Z0 = 2.074, 1.945, .104, .094, .110, -.012
    INWARD, WHORL = -.10, (.012, -.012)          # negative: the bob flares out toward the hem

    def k(self, y):
        y = np.asarray(y, float)
        up = np.sqrt(np.clip(1 - ((y - self.YC) / (self.TOP - self.YC)) ** 2, 0, 1))
        down = 1 - self.INWARD * (np.clip(self.YC - y, 0, None) / (self.YC - 1.80)) ** 1.5
        return np.where(y > self.YC, up, down)

    def pt(self, az, y, off=0.):
        az, y = np.asarray(az, float), np.asarray(y, float)
        k, s, c = self.k(y), np.sin(az), np.cos(az)
        b = np.where(c > 0, self.BF, self.BB)
        w = (np.clip(y - self.YC, 0, None) / (self.TOP - self.YC)) ** 2
        return np.stack([(self.A * k + off) * s + self.WHORL[0] * w, y, self.Z0 + (b * k + off) * c + self.WHORL[1] * w], -1)


ENV = Env()
CENTRE = np.array((0, 1.93, -.012))
RING = (1.977, .054)                 # crown braid height: RING[0] + RING[1] cos(az), so high at the front and low at the back


def ring_y(az):
    return RING[0] + RING[1] * np.cos(az)


def blade(P, W, T, seed=0, tip=2.2):
    """A closed lock along centreline P: four-sided with an off-centre ridge, narrowing to a point, its underside sunk into the cap."""
    n = len(P)
    q = np.linspace(0, 1, n)
    ridge = np.random.default_rng(seed).uniform(-.3, .3)
    Tg = np.gradient(P, axis=0)
    Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-9
    N = P - CENTRE
    N -= (N * Tg).sum(1, keepdims=True) * Tg
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    S = np.cross(Tg, N)
    w = W * np.minimum(1, .35 + q * 3.5) * (1 - np.clip((q - .18) / .82, 0, 1) ** tip) ** 1.15 + .0013
    t = 1.35 * T * np.minimum(1, .4 + q * 3) * (1 - .78 * q ** 1.6) + .0012
    ring = np.stack([P - S * w[:, None], P + S * (ridge * w)[:, None] + N * t[:, None], P + S * w[:, None], P - N * (.9 * t + .004)[:, None]], 1)
    V = ring.reshape(-1, 3)
    F = []
    for i in range(n - 1):
        for j in range(4):
            a, b = i * 4 + j, i * 4 + (j + 1) % 4
            F += [(a, b, b + 4), (a, b + 4, a + 4)]
    F += [(0, 2, 1), (0, 3, 2), (len(V) - 4, len(V) - 3, len(V) - 2), (len(V) - 4, len(V) - 2, len(V) - 1)]
    m = trimesh.Trimesh(V, np.array(F), process=False)
    if m.volume < 0:
        m.invert()
    return m


def lock(az0, y0, az1, y1, W=.016, T=.006, off=.003, flare=0., n=30, seed=0, tip=2.2, drop=1.0):
    """A lock from (az0, y0) to (az1, y1) over the hair surface; flare lifts its tip outward."""
    q = np.linspace(0, 1, n)
    y = y0 + (y1 - y0) * q ** drop
    s = q * q * (3 - 2 * q)
    return blade(ENV.pt(az0 + (az1 - az0) * s, y, off + flare * q ** 3), W, T, seed, tip)


def cap(hem=1.855, off=.0005):
    """The skull cap under the locks; a negative offset gives the hollow inside it."""
    e = ENV
    ys = np.r_[np.linspace(e.TOP - .0006, e.YC, 26)[:-1], np.linspace(e.YC, hem, 16)]
    az = np.linspace(-np.pi, np.pi, 96, endpoint=False)
    V = np.array([e.pt(a, y, off) for y in ys for a in az])
    nv, F = len(az), []
    for i in range(len(ys) - 1):
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            F += [(a, b, b + nv), (a, b + nv, a + nv)]
    k = (len(ys) - 1) * nv
    V = np.vstack([V, V[-nv:].mean(0), V[:nv].mean(0) + (0, .0002, 0)])
    F += [(len(V) - 2, k + j, k + (j + 1) % nv) for j in range(nv)] + [(len(V) - 1, (j + 1) % nv, j) for j in range(nv)]
    m = trimesh.Trimesh(V, np.array(F), process=False)
    if m.volume < 0:
        m.invert()
    return m


def locks():
    r = np.random.default_rng(7)
    L = []
    for i, az in enumerate(np.linspace(-np.pi, np.pi, 12, endpoint=False)):        # crown, from the whorl out to the braid
        L.append(lock(az, ENV.TOP - .002, az + .25, ring_y(az) - .006, .028, .0058, .002, seed=i, n=22))
    bangs = [(.42, -.12, 1.888, .027, 1.0), (.18, -.46, 1.912, .026, 1.1), (-.06, -.76, 1.895, .025, 1.2), (-.32, -.98, 1.868, .023, 1.2),
             (.56, .86, 1.882, .021, 1.4)]
    for i, (a0, a1, y1, W, dr) in enumerate(bangs):                                    # a few broad locks swept to her right, one ending between the eyes
        L.append(lock(a0, ring_y(a0) - .002, a1, y1, W, .0066, .006, flare=.004, seed=20 + i, drop=dr))
    for s in (1, -1):                                                                  # framing the face
        L += [lock(s * .95, ring_y(.95) - .003, s * 1.0, 1.828, .018, .0064, .005, flare=.012, seed=30 + s),
              lock(s * 1.1, ring_y(1.1) - .003, s * 1.1, 1.838, .019, .0066, .004, flare=.014, seed=32 + s),
              lock(s * .84, ring_y(.84) - .003, s * .80, 1.858, .013, .0058, .006, flare=.010, seed=34 + s),
              lock(s * .72, ring_y(.72) - .003, s * .74, 1.885, .011, .0055, .007, flare=.008, seed=36 + s)]
    layers = [(1.2, np.pi, 8, 1.838, .032, .0068, .002, .016), (1.32, np.pi - .1, 7, 1.862, .025, .0082, .005, .016), (1.5, np.pi - .25, 5, 1.892, .019, .0078, .008, .012)]
    for li, (a0, a1, count, y1, W, T, off, fl) in enumerate(layers):                   # sides and back: layered, tips flicking out
        for a in np.linspace(a0, a1, count):
            for s in (1, -1):
                if a == np.pi and s < 0:
                    continue
                sw = s * r.uniform(-.06, .06)
                L.append(lock(s * a - 2 * sw, ring_y(a) - .004, s * a + sw, y1 + r.uniform(-.016, .016), W * r.uniform(.85, 1.15), T, off, flare=fl * r.uniform(.6, 1.3), seed=int(r.integers(1e9))))
    q = np.linspace(0, 1, 18)                                                          # one flyaway on the crown
    p0 = ENV.pt(.3, 2.058, .002)
    L.append(blade(p0 + np.c_[.006 * q, .014 * np.sin(np.pi * q * .8), .022 * q], .0048, .0036, seed=99, tip=1.4))
    return L


def braid(pitch=.0125, off=.012):
    """Crown braid: alternating lobes along a ring around the head, each angled into the middle, which reads as a three-strand braid."""
    az = np.linspace(-np.pi, np.pi, 1441)
    P = ENV.pt(az, ring_y(az), off)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    sl = np.r_[0, np.cumsum(seg)]
    n = int(round(sl[-1] / pitch / 2)) * 2
    out = []
    for i in range(n):
        s0 = sl[-1] * i / n
        j = min(np.searchsorted(sl, s0), len(P) - 2)
        c, tg = P[j], P[j + 1] - P[j]
        tg /= np.linalg.norm(tg)
        nr = c - CENTRE
        nr -= (nr @ tg) * tg
        nr /= np.linalg.norm(nr)
        bn = np.cross(nr, tg)
        side = 1 if i % 2 else -1
        ang = np.radians(38) * -side
        ax = tg * np.cos(ang) + bn * np.sin(ang)
        wd = np.cross(nr, ax)
        R = np.c_[ax * .0185, wd * .0088, nr * .0068]
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        m.vertices = m.vertices @ R.T + c + bn * side * .0075
        out.append(m)
    return out


# ---------- ear units: a disc with a ring groove, a hub around the screw, and a lens pod pointing forward and down ----------
EAR_Z, EAR_X = 293.0, -1.0     # centre height and how far behind the neck axis
EAR_R, EAR_H = 11.5, 5.0       # disc radius and thickness
EAR_FLOOR = 3.0                # under the screw head
EAR_PAD = 7.5                  # from where the screw crosses the bore to the flat the disc sits on
EAR_SCREW_L, EAR_PILOT = 16.0, 6.5
POD_ANGLE = np.radians(-40)    # from forward, toward down
KEY_R = 7.6


def ear_frame(side):
    """Origin on the pad, and its in-plane forward, up and outward directions."""
    yb = .07 * SCALE * np.sqrt(1 - (EAR_X / (.076 * SCALE)) ** 2)
    c = np.array((NECK_X + EAR_X, side * (yb - .05), EAR_Z))
    w = np.array((0, side * 1., 0))
    return c, c + w * EAR_PAD, np.array((1., 0, 0)), np.array((0, 0, 1.)), w


def local(m, o, u, v, w):
    m = m.copy()
    m.vertices = o + m.vertices[:, :1] * u + m.vertices[:, 1:2] * v + m.vertices[:, 2:3] * w
    if m.volume < 0:
        m.invert()
    return m


def cyl(r, w0, w1, x=0., y=0., sections=96):
    return trimesh.creation.cylinder(radius=r, height=w1 - w0, sections=sections).apply_translation((x, y, (w0 + w1) / 2))


def ear(side):
    """One ear unit in the robot frame, plus the clearance it needs from the hair."""
    c, o, u, v, w = ear_frame(side)
    p = np.array((np.cos(POD_ANGLE), np.sin(POD_ANGLE)))
    arm = trimesh.creation.box(extents=(7.0, 4.5, 3.5)).apply_translation((12.5, 0, 1.75))
    arm.apply_transform(trimesh.transformations.rotation_matrix(POD_ANGLE, (0, 0, 1)))
    px, py = 16.5 * p
    body = ts.union(cyl(EAR_R, 0, EAR_H - 1.0), cyl(EAR_R - 1.0, EAR_H - 1.01, EAR_H), cyl(6.6, EAR_H - .01, EAR_H + 1.2), arm, cyl(3.1, 0, 6.0, px, py, 48))
    cuts = [cyl(10.6, EAR_H - 1.0, EAR_H + .1) if False else ts.diff(cyl(10.6, EAR_H - 1.0, EAR_H + .1), cyl(8.0, EAR_H - 1.5, EAR_H + .5)),
            cyl(h1.HEAD_R, EAR_FLOOR, EAR_H + 3), cyl(ts.M3_CLEAR / 2, -1, EAR_FLOOR + .1, sections=32),
            cyl(1.9, 4.8, 6.5, px, py, 48), cyl(1.65, -.1, 2.2, -KEY_R * p[0], -KEY_R * p[1], 32)]
    unit = ts.diff(body, *cuts)
    room = ts.union(cyl(EAR_R + .6, 0, 40), arm.copy().apply_scale(1.0).apply_translation((0, 0, 0)), cyl(3.7, 0, 40, px, py, 48),
                    trimesh.creation.box(extents=(8.2, 5.7, 40)).apply_translation((12.5, 0, 20)).apply_transform(trimesh.transformations.rotation_matrix(POD_ANGLE, (0, 0, 1))))
    key = cyl(1.4, -1.0, 1.8, -KEY_R * p[0], -KEY_R * p[1], 32)
    return dict(unit=local(unit, o, u, v, w), room=local(room, o, u, v, w), key=local(key, o, u, v, w), axis=(c, w), frame=(o, u, v, w))


def draft():
    face = to_mm(Face().mesh())
    bore_r = h1.hull_radius(h1.bore_outline())
    front = trimesh.creation.box(extents=(60, 200, 200)).apply_translation((NECK_X + 10 + 30, 0, 300))
    hairline = h1.CHIN_Z + (1.955 - h1.CHIN_U) * SCALE
    skull = ts.diff(to_mm(cap()), h1.prism(bore_r, h1.RIM - 40, h1.CHIN_Z + (1.90 - h1.CHIN_U) * SCALE), ts.inter(h1.prism(bore_r, h1.RIM - 40, hairline), front))
    hair = ts.union(skull, *[to_mm(m) for m in locks()], *[to_mm(m) for m in braid()])
    ears = [ear(1), ear(-1)]
    hair = ts.diff(hair, *[e["room"] for e in ears])
    return face, hair, ears


# ---------- print parts ----------
FACE_TOP = 322.0           # open rim of the face part, under the hair
BORE_SPLIT = 285.4         # the bore follows the face window below this, and only what lies above each height above it
LEVEL = 2.0


def back_axis():
    r = h1.hull_radius(h1.bore_outline(), np.radians([180]))[0]
    return np.array((NECK_X - r + .05, 0, 284.0)), np.array((-1., 0, 0))


def build():
    face_m, hair_m, ears = draft()
    bore_r = h1.hull_radius(h1.bore_outline())
    low = h1.RIM - 40
    axes = [back_axis()] + [e["axis"] for e in ears]
    h1.hair_screw_axes = lambda: axes                        # so H1's pad builder places these three
    face_in = ts.inter(face_m, h1.prism(bore_r, low, FACE_TOP + 40))
    body = ts.union(face_in, h1.collar())
    pads = h1.screw_pads()
    # the cap's bore at each height only has to clear what lies above that height
    pv = np.vstack([ts.inter(face_in, h1.prism(bore_r, low, FACE_TOP)).vertices] + [q.vertices for q in pads])
    rings = [(z, h1.hull_radius(pv[pv[:, 2] >= z][:, :2]) + h1.FIT) for z in np.arange(BORE_SPLIT, FACE_TOP, LEVEL)]
    seat = FACE_TOP - h1.FIT + .05
    rings.append((seat, h1.hull_radius(pv[pv[:, 2] >= FACE_TOP - 1][:, :2]) + h1.FIT))
    upper = ts.loft(rings, NECK_X)
    top_r = rings[-1][1]
    roof = ts.loft([(seat + d, np.maximum(top_r - d, 1.0)) for d in np.arange(0.0, min(top_r.max(), hair_m.bounds[1][2] - h1.ROOF_SKIN - seat), 1.0)], NECK_X)
    lower = h1.prism(bore_r + h1.FIT, low, BORE_SPLIT + .01)

    top_cut = h1.prism(np.full(ts.PHI.shape, 90.), FACE_TOP, h1.RIM + 120)
    bore, passage, _ = h1.neck_socket()
    neck = [ts.rod(c + er * 8.0, er, ts.M3_CLEAR / 2, 10.0) for c, er in h1.screw_holes()]
    pilots = [ts.rod(c + d, -d, ts.M3_PILOT / 2, h1.HAIR_PILOT + 1.0) for c, d in axes]
    face_part = ts.diff(ts.union(ts.diff(body, h1.cavity(body)), *pads), top_cut, bore, passage, *neck, *pilots)

    # hollow: the cap surface moved in by WALL, under a 45 deg cone so its roof prints without supports
    apex = hair_m.bounds[1][2] - h1.ROOF_SKIN
    cone = ts.loft([(z, np.full(ts.PHI.shape, max(apex - z, .5))) for z in np.arange(low, apex, 1.0)], NECK_X)
    void = ts.inter(to_mm(cap(hem=1.872, off=-h1.WALL / SCALE)), cone)
    hair_part = ts.diff(hair_m, lower, upper, roof, void)
    (cb, db), holes = axes[0], []
    bosses = [ts.diff(ts.inter(ts.rod(cb, db, h1.HEAD_R + h1.WALL, 40.0), hair_m), lower, upper)]
    holes += [ts.rod(cb - db, db, ts.M3_CLEAR / 2, h1.HAIR_GRIP + 2.0), ts.rod(cb + db * h1.HAIR_GRIP, db, h1.HEAD_R, 40.0)]
    for e in ears:
        c, d = e["axis"]
        bosses.append(ts.diff(ts.inter(ts.rod(c, d, EAR_R, EAR_PAD), hair_m), lower, upper))
        holes.append(ts.rod(c - d, d, ts.M3_CLEAR / 2, EAR_PAD + 2.0))
    hair_part = ts.diff(ts.union(hair_part, *bosses), *holes, *[e["room"] for e in ears])
    hair_part = ts.union(hair_part, *[e["key"] for e in ears])
    return dict(face=face_part, hair=hair_part, ear_L=ears[0]["unit"], ear_R=ears[1]["unit"]), ears, axes, hair_m


def frac(a, b):
    return float(ts.inter(a, b).volume / b.volume)


def check(parts, ears, axes):
    rep = {}
    for n, m in parts.items():
        rep[n] = dict(watertight=bool(m.is_watertight), bodies=len(m.split(only_watertight=False)), bbox_mm=[round(float(v), 1) for v in m.extents],
                      solid_mass_g=round(m.volume * DENSITY, 1), bottom_z=round(float(m.bounds[0][2]), 1), top_z=round(float(m.bounds[1][2]), 1))
    tot = sum(m.volume for m in parts.values())
    rep["total"] = dict(solid_mass_g=round(tot * DENSITY, 1), com_mm=[round(float(v), 1) for v in sum(m.center_mass * m.volume for m in parts.values()) / tot],
                        rim_z=round(float(h1.RIM), 1))
    names = list(parts)
    rep["overlap_mm3"] = {f"{a}/{b}": round(float(ts.inter(parts[a], parts[b]).volume), 3) for i, a in enumerate(names) for b in names[i + 1:]}
    t06 = ts.build()["T06_shoulders"]
    rep["t06_overlap_mm3"] = {n: round(float(ts.inter(m, t06).volume), 3) for n, m in parts.items()}
    rep["hair_lift_mm3"] = max(float(ts.inter(parts["face"], parts["hair"].copy().apply_translation((0, 0, dz))).volume) for dz in (.5, 2, 5, 10, 20, 40, 70))
    rep["ear_removal_mm3"] = {n: max(float(ts.inter(parts["hair"], parts[n].copy().apply_translation(e["axis"][1] * k)).volume) for k in (.5, 2, 5, 10, 30))
                              for n, e in (("ear_L", ears[0]), ("ear_R", ears[1]))}
    rep["screws"] = {}
    specs = [("back", axes[0], 12.0, h1.HAIR_GRIP, "hair")] + [(f"ear_{n}", e["axis"], EAR_SCREW_L, EAR_PAD + EAR_FLOOR, f"ear_{n}") for n, e in zip("LR", ears)]
    for name, (c, d), L, seat, on in specs:
        tip = seat - L
        core = ts.rod(c + d * tip, d, 1.2, L)
        bite = ts.diff(ts.rod(c + d * tip, d, 1.5, -tip), ts.rod(c + d * (tip - 1), d, ts.M3_PILOT / 2, -tip + 2))
        ring = ts.diff(ts.rod(c + d * (seat - .6), d, 2.75, .6), ts.rod(c + d * (seat - 1), d, ts.M3_CLEAR / 2, 2.))
        path = ts.rod(c + d * (seat + .2), d, 3.0, 60.0)
        rep["screws"][name] = dict(length_mm=L, core_hits_mm3=round(sum(float(ts.inter(core, m).volume) for m in parts.values()), 3),
                                   thread_mm=round(frac(parts["face"], bite) * -tip, 2), tip_clear_mm=round(h1.HAIR_PILOT + tip, 2),
                                   head_seat=round(frac(parts[on], ring), 2), driver_path_mm3=round(sum(float(ts.inter(path, m).volume) for m in [*parts.values(), t06]), 2))
    rep["neck_driver_paths_mm3"] = {str(deg): round(sum(float(ts.inter(ts.rod(c + er * (h1.COLLAR_R + .4), er, 3.0, 60.0), m).volume) for m in parts.values()), 2)
                                    for (c, er), deg in zip(h1.screw_holes(), ts.NECK_SCREWS)}
    rep["passage_overlap_mm3"] = round(float(ts.inter(parts["face"], ts.rod((NECK_X, 0, h1.RIM - 1), (0, 0, 1), 18.5 / 2, 15.0)).volume), 2)
    return rep


def print_pose(n, m, ears):
    """Upright for face and hair; ear units lie on their backs."""
    m = m.copy()
    if n.startswith("ear"):
        o, u, v, w = ears[0 if n == "ear_L" else 1]["frame"]
        R = np.c_[u, v, w]
        m.vertices = (m.vertices - o) @ R
        if m.volume < 0:
            m.invert()
    m.apply_translation((BED / 2 - m.bounds.mean(axis=0)[0], BED / 2 - m.bounds.mean(axis=0)[1], -m.bounds[0][2]))
    return h1.stl_safe(m)


FILES = dict(face="H11_face", hair="H12_hair", ear_L="H13_ear_left", ear_R="H13_ear_right")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="")
    a = ap.parse_args()
    parts, ears, axes, _ = build()
    parts = {n: h1.largest(h1.stl_safe(m))[0] for n, m in parts.items()}
    rep = check(parts, ears, axes)
    out = ROOT / "exports" / "print_ready"
    out.mkdir(parents=True, exist_ok=True)
    for n, m in parts.items():
        if a.assembled:
            Path(a.assembled).mkdir(parents=True, exist_ok=True)
            m.export(Path(a.assembled) / f"{FILES[n]}.stl")
        print_pose(n, m, ears).export(out / f"{FILES[n]}_print.stl")
        saved = trimesh.load(out / f"{FILES[n]}_print.stl")
        rep[n]["saved_file"] = dict(watertight=bool(saved.is_watertight), volume=bool(saved.is_volume), bodies=len(saved.split(only_watertight=False)),
                                    bbox_mm=[round(float(v), 1) for v in saved.extents])
    (ROOT / "head_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
