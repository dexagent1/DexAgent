"""Distances to actual geom material, preserving gaps between separate components.

Collision meshes use MuJoCo's convex hull; visual-only meshes use their triangles.
A non-watertight visual mesh supplies a surface but cannot certify interior depth.
"""
from __future__ import annotations

import numpy as np


class MaterialSurface:
    def __init__(self, model, bodies):
        import mujoco
        import trimesh
        from scipy.spatial import ConvexHull
        from tool_library.common import scene_assert as SA
        self.model = model
        self.geoms = []
        self.uncovered = []
        self.meshes = {}
        paired = set(int(g) for g in np.r_[model.pair_geom1, model.pair_geom2])
        for g in range(model.ngeom):
            collidable = SA.geom_collidable(model, g) or g in paired
            if int(model.geom_bodyid[g]) not in bodies or not (collidable or SA.geom_visible(model, g)):
                continue
            kind = int(model.geom_type[g])
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(model.geom_dataid[g])
                va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
                fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
                vertices = np.asarray(model.mesh_vert[va:va+vn], float)
                faces = np.asarray(model.mesh_face[fa:fa+fn], int)
                if collidable:
                    try:
                        hull = ConvexHull(vertices)
                        faces = hull.simplices.copy()
                        tri = vertices[faces]
                        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
                        flip = np.einsum('ij,ij->i', normals, hull.equations[:, :3]) < 0
                        faces[flip] = faces[flip][:, [0, 2, 1]]
                        closed = True
                    except Exception:
                        closed = False
                else:
                    mesh = trimesh.Trimesh(vertices, faces, process=True)
                    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
                    closed = bool(mesh.is_watertight and mesh.is_winding_consistent)
                if not len(faces):
                    self.uncovered.append(model.geom(g).name or f'geom{g}')
                    continue
                self.meshes[g] = (vertices[faces], closed)
                if not closed:
                    self.uncovered.append(model.geom(g).name or f'geom{g}')
            elif kind not in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX,
                              mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER,
                              mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_PLANE):
                self.uncovered.append(model.geom(g).name or f'geom{g}')
                continue
            self.geoms.append(g)

    def distances(self, data, points):
        """Signed component-material distances and matching world surface points.

        The minimum signed distance detects entry into any material component;
        it never fills the empty convex envelope between disconnected walls.
        """
        import mujoco
        points = np.atleast_2d(np.asarray(points, float))
        best = np.full(len(points), np.inf)
        surface = np.full_like(points, np.nan)
        for g in self.geoms:
            R = data.geom_xmat[g].reshape(3, 3)
            local = (points-data.geom_xpos[g]) @ R
            kind = int(self.model.geom_type[g])
            size = np.asarray(self.model.geom_size[g], float)
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                triangles, closed = self.meshes[g]
                distances, closest = _mesh_distance(triangles, local, closed)
            else:
                distances, closest = _primitive_distance(kind, size, local)
            use = distances < best
            best[use] = distances[use]
            surface[use] = closest[use] @ R.T + data.geom_xpos[g]
        return best, surface


def _primitive_distance(kind, size, points):
    import mujoco
    T = mujoco.mjtGeom
    q = np.asarray(points, float)
    if kind == T.mjGEOM_BOX:
        closest = np.clip(q, -size, size)
        inside = np.all(np.abs(q) < size, axis=1)
        for i in np.flatnonzero(inside):
            axis = int(np.argmin(size-np.abs(q[i])))
            closest[i, axis] = size[axis] * (1 if q[i, axis] >= 0 else -1)
    elif kind in (T.mjGEOM_SPHERE, T.mjGEOM_CAPSULE):
        centre = np.zeros_like(q)
        if kind == T.mjGEOM_CAPSULE:
            centre[:, 2] = np.clip(q[:, 2], -size[1], size[1])
        offset = q-centre
        radius = np.linalg.norm(offset, axis=1)
        direction = offset/np.maximum(radius[:, None], 1e-15)
        direction[radius < 1e-15] = [1, 0, 0]
        closest = centre+size[0]*direction
        inside = radius < size[0]
    elif kind == T.mjGEOM_CYLINDER:
        radial = np.linalg.norm(q[:, :2], axis=1)
        direction = q[:, :2]/np.maximum(radial[:, None], 1e-15)
        direction[radial < 1e-15] = [1, 0]
        closest = q.copy()
        closest[:, :2] = direction*np.minimum(radial, size[0])[:, None]
        closest[:, 2] = np.clip(q[:, 2], -size[1], size[1])
        inside = (radial < size[0]) & (np.abs(q[:, 2]) < size[1])
        for i in np.flatnonzero(inside):
            if size[0]-radial[i] < size[1]-abs(q[i, 2]):
                closest[i, :2] = direction[i]*size[0]
            else:
                closest[i, 2] = size[1]*(1 if q[i, 2] >= 0 else -1)
    elif kind == T.mjGEOM_ELLIPSOID:
        closest = np.array([_ellipsoid_closest(size, p) for p in q])
        inside = np.linalg.norm(q/size, axis=1) < 1
    else:  # infinite collision plane
        closest = q.copy(); closest[:, 2] = 0
        inside = q[:, 2] < 0
    distances = np.linalg.norm(q-closest, axis=1)
    distances[inside] *= -1
    return distances, closest


def _ellipsoid_closest(size, point):
    from scipy.optimize import brentq
    a2 = size*size
    q = np.abs(point)
    amin = float(a2.min())
    small = np.isclose(a2, amin, rtol=1e-12, atol=0)
    def constraint(t):
        return float(np.sum(a2*q*q/(a2+t)**2)-1)
    if np.linalg.norm(q/size) >= 1:
        upper = max(float(np.linalg.norm(size*q)), float(a2.max()))
        while constraint(upper) > 0:
            upper *= 2
        lam = brentq(constraint, 0, upper)
        closest = a2*q/(a2+lam)
    else:
        other = ~small
        on_singular = np.zeros(3)
        on_singular[other] = a2[other]*q[other]/(a2[other]-amin)
        fill = 1-float(np.sum((on_singular[other]/size[other])**2))
        if np.all(q[small] == 0) and fill >= 0:
            closest = on_singular
            closest[int(np.flatnonzero(small)[0])] = np.sqrt(amin*fill)
        else:
            lower = np.nextafter(-amin, 0)
            lam = brentq(constraint, lower, 0)
            closest = a2*q/(a2+lam)
    return closest*np.where(point < 0, -1, 1)


def _mesh_distance(triangles, points, closed):
    import trimesh
    distances, closest = [], []
    for point in points:
        candidate = trimesh.triangles.closest_point(triangles, np.broadcast_to(point, (len(triangles), 3)))
        lengths = np.linalg.norm(candidate-point, axis=1)
        which = int(np.argmin(lengths)); distance = float(lengths[which])
        if closed and distance > 1e-10 and _mesh_inside(triangles, point):
            distance = -distance
        distances.append(distance); closest.append(candidate[which])
    return np.array(distances), np.array(closest)


def _mesh_inside(triangles, point):
    """Generalized winding number; inner cavity faces cancel outer faces.

    Unlike ray parity, consistently oriented overlapping solid components do
    not cancel each other. Closed, consistently wound input is required above.
    """
    v = triangles-point
    lengths = np.linalg.norm(v, axis=2)
    numerator = np.einsum('ij,ij->i', v[:, 0], np.cross(v[:, 1], v[:, 2]))
    denominator = (lengths.prod(axis=1)
                   + np.einsum('ij,ij->i', v[:, 0], v[:, 1])*lengths[:, 2]
                   + np.einsum('ij,ij->i', v[:, 1], v[:, 2])*lengths[:, 0]
                   + np.einsum('ij,ij->i', v[:, 2], v[:, 0])*lengths[:, 1])
    winding = np.sum(2*np.arctan2(numerator, denominator))/(4*np.pi)
    return abs(float(winding)) > .5
