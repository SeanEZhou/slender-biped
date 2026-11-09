"""Quick shaded renders of the pad exports (hardware/feet_p1/renders/pads.png), shown in the robot's frame."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pads  # noqa: E402

names = sys.argv[1:] or ["board120", "board90w", "disc70", "talon60"]
fig = plt.figure(figsize=(4 * len(names), 4))
for i, n in enumerate(names):
    tm = pads.build(n)
    if pads.DESIGNS[n][1] == "side":                       # back to standing on the sole for the picture
        tm.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    v = tm.vertices - tm.bounds.mean(0)
    ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
    tri = v[tm.faces]
    nrm = tm.face_normals
    shade = 0.35 + 0.65 * np.clip(nrm @ np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]), 0, 1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[shade * 0.95, shade * 0.75, shade * 0.8, np.ones_like(shade)],
                                         linewidths=0))
    r = np.abs(v).max()
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.view_init(elev=22, azim=-60)
    ax.set_axis_off()
    ax.set_title(n)
out = HERE.parent / "renders" / "pads.png"
plt.tight_layout()
plt.savefig(out, dpi=110)
print(out)
