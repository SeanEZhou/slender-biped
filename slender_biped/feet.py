"""Foot geoms: one or more primitives per foot (`L_tip`, `L_tip_1`, ...; scripts/make_pads.py), shared by the MJX env
and the CPU evaluator. Lowest point of each foot = min over its geoms of the support function along -z."""
from __future__ import annotations

import mujoco
import numpy as np

PAD_AREA_REF = 0.031209 * 0.018     # m^2, the original 31 x 18 mm pad: the rug-softness reference area
T_BOX, T_SPHERE, T_ELLIPSOID, T_CAPSULE, T_CYLINDER = 0, 1, 2, 3, 4
_TYPES = {int(mujoco.mjtGeom.mjGEOM_BOX): T_BOX, int(mujoco.mjtGeom.mjGEOM_SPHERE): T_SPHERE,
          int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): T_ELLIPSOID, int(mujoco.mjtGeom.mjGEOM_CAPSULE): T_CAPSULE,
          int(mujoco.mjtGeom.mjGEOM_CYLINDER): T_CYLINDER}


def foot_geoms(m):
    """(gid (2, K), type code (2, K), size (2, K, 3)); K geoms per foot, the first (`<S>_tip`) at the pad centre."""
    rows = []
    for S in "LR":
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{S}_tip")]
        k = 1
        while (g := mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{S}_tip_{k}")) >= 0:
            ids.append(g)
            k += 1
        rows.append(ids)
    assert len(rows[0]) == len(rows[1]) and min(rows[0] + rows[1]) >= 0, rows
    gid = np.array(rows)
    typ = np.vectorize(lambda g: _TYPES[int(m.geom_type[g])])(gid)
    return gid, typ, m.geom_size[gid].copy()


def foot_names(m):
    gid, _, _ = foot_geoms(m)
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in gid.ravel()}


def lowest(xp, pos, mat, typ, size):
    """Lowest world z of each foot. pos (2, K, 3), mat (2, K, 3, 3) world rotation, xp = numpy or jax.numpy."""
    rz = mat[..., 2, :]                                   # world-z component of each local axis
    h_box = xp.sum(xp.abs(size * rz), axis=-1)
    h_sph = size[..., 0]
    h_ell = xp.linalg.norm(size * rz, axis=-1)
    axz = xp.abs(rz[..., 2])
    h_cap = size[..., 0] + size[..., 1] * axz
    h_cyl = size[..., 1] * axz + size[..., 0] * xp.sqrt(xp.maximum(1.0 - axz ** 2, 0.0))
    h = xp.where(typ == T_BOX, h_box, xp.where(typ == T_SPHERE, h_sph, xp.where(
        typ == T_ELLIPSOID, h_ell, xp.where(typ == T_CAPSULE, h_cap, h_cyl))))
    return xp.min(pos[..., 2] - h, axis=-1)
