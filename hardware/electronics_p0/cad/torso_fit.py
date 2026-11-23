"""Free space inside the T1 torso tubes, for checking that the electronics mount fits and can be assembled.

    .venv-cad/bin/python hardware/electronics_p0/cad/torso_fit.py      # once, or after the torso changes (~1-2 min)

Builds T03-T06 with hardware/torso_p0/cad/torso.py (leg CAD frame), slices each tube every 1 mm and stores its free
section (the section's convex hull minus its material) in torso_cache/ (not kept in the repo), with the tubes as
STLs for the renders. mount.py loads it.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TORSO_CAD = ROOT.parent / "torso_p0" / "cad"
CACHE = ROOT / "torso_cache"
TUBES = ("T03_pelvis_ring", "T04_midriff", "T05_chest", "T06_shoulders")
PLATE_TOP = 30.7


def main():
    sys.path.insert(0, str(TORSO_CAD))
    import manifold3d as m3d
    import torso
    parts = torso.build()
    CACHE.mkdir(exist_ok=True)
    out = {}
    for name in TUBES:
        tm = parts[name]
        tm.export(CACHE / f"{name}.stl")
        man = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(tm.vertices, np.float32),
                                    tri_verts=np.asarray(tm.faces, np.uint32)))
        z0, z1 = tm.bounds[:, 2]
        secs = {}
        for zp in range(int(np.ceil(z0 - PLATE_TOP)), int(np.floor(z1 - PLATE_TOP)) + 1):
            sl = man.slice(PLATE_TOP + zp + 0.01)
            if sl.is_empty():
                continue
            free = sl.hull() - sl
            secs[zp] = [np.asarray(p, float)[:: max(1, len(p) // 240)] for p in free.to_polygons()]
        out[name] = secs
        print(f"{name}: zp {min(secs)}..{max(secs)}")
    (CACHE / "free_sections.pkl").write_bytes(pickle.dumps(out))
    (CACHE / ".gitignore").write_text("*\n")


if __name__ == "__main__":
    main()
