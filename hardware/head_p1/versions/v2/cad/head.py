"""Head H2 "Rebel Princess": a face with modelled bone structure and a hair cap with a crown braid.

    .venv-cad/bin/python hardware/head_p1/versions/v2/cad/head.py [--assembled <dir>]

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
RING = (1.977, .054)
SINK = h1.WALL / 2 / SCALE    # locks reach half way into the cap's wall                 # crown braid height: RING[0] + RING[1] cos(az), so high at the front and low at the back


def ring_y(az):
    return RING[0] + RING[1] * np.cos(az)


def blade(P, W, T, seed=0, tip=.65, sink=None):
    """A closed lock along centreline P: widest near its root and narrowing steadily to a point, so it reads as a triangle,
    with an off-centre ridge. Its underside reaches `sink` below the centreline: into the cap, but not through the cap's wall."""
    n = len(P)
    q = np.linspace(0, 1, n)
    ridge = np.random.default_rng(seed).uniform(-.3, .3)
    Tg = np.gradient(P, axis=0)
    Tg /= np.linalg.norm(Tg, axis=1, keepdims=True) + 1e-9
    N = P - CENTRE
    N -= (N * Tg).sum(1, keepdims=True) * Tg
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-9
    S = np.cross(Tg, N)
    w = W * np.minimum(1, .45 + q * 6) * (1 - np.clip((q - .06) / .94, 0, 1)) ** tip + .0018     # tips stay about 1.3 mm wide
    t = 1.35 * T * np.minimum(1, .4 + q * 5) * (1 - .8 * q ** 1.2) + .0016
    ring = np.stack([P - S * w[:, None], P + S * (ridge * w)[:, None] + N * t[:, None], P + S * w[:, None], P - N * (.9 * t + .004 if sink is None else np.broadcast_to(sink, t.shape))[:, None]], 1)
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


def lock(az0, y0, az1, y1, W=.016, T=.006, off=.003, flare=0., n=30, seed=0, tip=.65, drop=1.0):
    """A lock from (az0, y0) to (az1, y1) over the hair surface; flare lifts its tip outward."""
    q = np.linspace(0, 1, n)
    y = y0 + (y1 - y0) * q ** drop
    s = q * q * (3 - 2 * q)
    f0, f1 = np.random.default_rng(seed + 1).uniform(.55, 1.0, 2)    # each lock its own depth, so overlapping undersides never coincide
    return blade(ENV.pt(az0 + (az1 - az0) * s, y, off + flare * q ** 3), W, T, seed, tip, sink=off + flare * q ** 3 + SINK * (f0 + (f1 - f0) * q))


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
    for i, az in enumerate(np.linspace(-np.pi, np.pi, 11, endpoint=False)):        # crown, from the whorl out to the braid
        L.append(lock(az, ENV.TOP - .002, az + .25, ring_y(az) - .006, .036, .0058, .002, seed=i, n=22, tip=.8))
    # bangs: parted over her left eye, broad locks swept to her right, her left forehead left open
    bangs = [(.30, -.08, 1.886, .050, .0078, 1.0), (.06, -.46, 1.906, .048, .0076, 1.1), (-.22, -.80, 1.876, .046, .0074, 1.15),
             (.56, .80, 1.862, .040, .0072, 1.3), (.20, .04, 1.872, .016, .0056, 1.0)]
    for i, (a0, a1, y1, W, T, dr) in enumerate(bangs):
        L.append(lock(a0, ring_y(a0) - .002, a1, y1, W, T, .006, flare=.004, seed=20 + i, drop=dr, n=36))
    for s in (1, -1):                                                                  # framing the face: turned to the front, tips curling toward the chin
        L += [lock(s * .92, ring_y(.92) - .003, s * .80, 1.838, .046, .0095, .006, flare=.006, seed=30 + s, n=40),
              lock(s * 1.12, ring_y(1.12) - .003, s * 1.0, 1.834, .048, .0100, .005, flare=.010, seed=32 + s, n=40)]
    # sides and back: wide locks whose tips draw in toward the middle and down to a V at the nape
    for li, (a0, count, ylo, dy, W, T, off, fl) in enumerate([(1.3, 6, 1.842, .034, .056, .0100, .002, .018),
                                                                (1.5, 5, 1.862, .026, .046, .0100, .005, .012),
                                                                (1.75, 3, 1.893, .018, .036, .0078, .008, .008)]):
        for a in np.linspace(a0, np.pi, count):
            for s in (1, -1):
                if a == np.pi and s < 0:
                    continue
                u = (np.pi - a) / (np.pi - 1.3)
                y1 = ylo + dy * np.sin(np.pi * np.clip(u, 0, 1)) + r.uniform(-.005, .005)
                a1 = a + (np.pi - a) * .22
                L.append(lock(s * a, ring_y(a) - .004, s * a1, y1, W * r.uniform(.9, 1.1), T, off, flare=fl * r.uniform(.7, 1.2), seed=int(r.integers(1e9)), n=34))
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


# ---------- screws for the hair cap: back and both sides, straight in, heads recessed in the hair ----------
SIDE_Z, SIDE_X = 290.0, -6.0      # side screws sit 6 mm behind the neck axis, clear of where the cap stops at the front


def screw_axes():
    rb = h1.hull_radius(h1.bore_outline(), np.radians([180]))[0]
    yb = .07 * SCALE * np.sqrt(1 - (SIDE_X / (.076 * SCALE)) ** 2)      # the bore's edge 6 mm behind the axis
    out = [(np.array((NECK_X - rb + .05, 0, 284.0)), np.array((-1., 0, 0)))]
    for s in (1, -1):
        out.append((np.array((NECK_X + SIDE_X, s * (yb - .05), SIDE_Z)), np.array((0, s * 1., 0))))
    return out


def draft():
    face = to_mm(Face().mesh())
    bore_r = h1.hull_radius(h1.bore_outline())
    # the cap stops at the hairline across the whole front half, so no flat cut face shows beside the cheeks: the locks frame the face
    front = trimesh.creation.box(extents=(60, 200, 200)).apply_translation((NECK_X + 2 + 30, 0, 300))
    hairline = h1.CHIN_Z + (1.955 - h1.CHIN_U) * SCALE
    skull = ts.diff(to_mm(cap()), h1.prism(bore_r, h1.RIM - 40, h1.CHIN_Z + (1.90 - h1.CHIN_U) * SCALE), ts.inter(h1.prism(np.full(ts.PHI.shape, 80.), h1.RIM - 40, hairline), front))
    # mirrored locks can meet exactly on the centre plane, leaving zero-thickness contacts that break once saved as STL,
    # so each piece is nudged by a few hundredths of a millimetre
    r = np.random.default_rng(3)
    pieces = [to_mm(m).apply_translation(r.normal(0, .02, 3)) for m in [*locks(), *braid()]]
    hair = ts.union(skull, *pieces)
    return face, hair


# ---------- print parts ----------
FACE_TOP = 322.0           # open rim of the face part, under the hair
BORE_SPLIT = 285.4         # the bore follows the face window below this, and only what lies above each height above it
LEVEL = 2.0


def build():
    face_m, hair_m = draft()
    bore_r = h1.hull_radius(h1.bore_outline())
    low = h1.RIM - 40
    axes = screw_axes()
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
    bosses, holes = [], []
    for c, d in axes:
        bosses.append(ts.diff(ts.inter(ts.rod(c, d, h1.HEAD_R + h1.WALL, 40.0), hair_m), lower, upper))
        holes += [ts.rod(c - d, d, ts.M3_CLEAR / 2, h1.HAIR_GRIP + 2.0), ts.rod(c + d * h1.HAIR_GRIP, d, h1.HEAD_R, 40.0)]
    hair_part = ts.diff(ts.union(hair_part, *bosses), *holes)
    return dict(face=face_part, hair=hair_part), axes, hair_m


def frac(a, b):
    return float(ts.inter(a, b).volume / b.volume)


def check(parts, axes):
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
    rep["screws"] = {}
    specs = [(n, ax, 12.0, h1.HAIR_GRIP, "hair") for n, ax in zip(("back", "left", "right"), axes)]
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


def print_pose(m):
    """Upright, centred on the bed."""
    m = m.copy()
    m.apply_translation((BED / 2 - m.bounds.mean(axis=0)[0], BED / 2 - m.bounds.mean(axis=0)[1], -m.bounds[0][2]))
    return h1.stl_safe(m)


FILES = dict(face="H11_face", hair="H12_hair")


def separate(m, gap=.01):
    """Where two surfaces touch at a point (separate vertices at one position), pull each copy inward along its own normal,
    so the contact no longer welds into a non-manifold edge once rounded to STL's 32-bit floats."""
    m = m.copy()
    _, inv, cnt = np.unique(np.asarray(m.vertices, "f4"), axis=0, return_inverse=True, return_counts=True)
    shared = cnt[inv.ravel()] > 1
    if shared.any():
        v = np.array(m.vertices)
        v[shared] -= m.vertex_normals[shared] * gap
        m = trimesh.Trimesh(v, m.faces, process=False)
    return m


def save(m, path):
    """Export and reload, separating point contacts and cleaning until the file is one closed solid."""
    for _ in range(3):
        m = separate(m)
        m.export(path)
        back = trimesh.load(path)
        if back.is_volume and len(back.split(only_watertight=False)) == 1:
            return back
        m = h1.stl_safe(back)
    raise RuntimeError(f"{path} is not a single closed solid")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="")
    a = ap.parse_args()
    parts, axes, _ = build()
    parts = {n: h1.largest(h1.stl_safe(m))[0] for n, m in parts.items()}
    rep = check(parts, axes)
    out = ROOT / "exports" / "print_ready"
    out.mkdir(parents=True, exist_ok=True)
    for n, m in parts.items():
        if a.assembled:
            Path(a.assembled).mkdir(parents=True, exist_ok=True)
            save(m, Path(a.assembled) / f"{FILES[n]}.stl")
        saved = save(print_pose(m), out / f"{FILES[n]}_print.stl")
        rep[n]["saved_file"] = dict(watertight=bool(saved.is_watertight), volume=bool(saved.is_volume), bodies=len(saved.split(only_watertight=False)),
                                    bbox_mm=[round(float(v), 1) for v in saved.extents])
    (ROOT / "head_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
