"""Export a body's collision geometry as a triangle mesh (for grasp synthesizers that want an OBJ)."""
from __future__ import annotations

import mujoco
import numpy as np


def body_subtree(model, bid):
    sub = {int(bid)}; changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in sub and b not in sub:
                sub.add(b); changed = True
    return sub


def geom_trimesh(model, g):
    """A trimesh of one geom in its own frame (box, sphere, capsule, cylinder, ellipsoid, mesh)."""
    import trimesh
    t = int(model.geom_type[g]); s = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * s)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(s[0]))
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=float(s[0]), height=2 * float(s[1]), count=(12, 12))
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(s[0]), height=2 * float(s[1]), sections=24)
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0); m.apply_scale(s); return m
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g]); v0, nv = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        f0, nf = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
        return trimesh.Trimesh(model.mesh_vert[v0:v0 + nv].copy(), model.mesh_face[f0:f0 + nf].copy(), process=False)
    return None


def body_mesh(model, data, body, collision_only=True, frame="body"):
    """Union (concatenation) of the geoms of `body` and its subtree, expressed in the body's frame
    (frame='body') or the world (frame='world'). Returns a trimesh."""
    import trimesh
    bid = body if isinstance(body, int) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"no body {body!r}")
    sub = body_subtree(model, bid)
    mujoco.mj_forward(model, data)
    pb, Rb = data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
    parts = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in sub:
            continue
        if collision_only and not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        m = geom_trimesh(model, g)
        if m is None:
            continue
        T = np.eye(4); T[:3, :3] = data.geom_xmat[g].reshape(3, 3); T[:3, 3] = data.geom_xpos[g]
        if frame == "body":
            Tb = np.eye(4); Tb[:3, :3] = Rb; Tb[:3, 3] = pb
            T = np.linalg.inv(Tb) @ T
        m = m.copy(); m.apply_transform(T); parts.append(m)
    if not parts:
        raise ValueError(f"body {model.body(bid).name!r} has no {'collision ' if collision_only else ''}geoms to export")
    return trimesh.util.concatenate(parts)
