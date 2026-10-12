"""Object visual penetration audit: objects that interpenetrate on the meshes the camera shows.

    from tool_library.verifiers.object_visual_penetration_audit import object_visual_penetration_audit
    object_visual_penetration_audit("scene.xml", "trajectory.npz")

Collision geometry is a simplification: convex wedges around a bowl, a cylinder for a cup, a
box for a toy. Physics keeps those apart, and every contact-based check (episode_physics_check,
penetration_check) reads them. The rendered meshes can still overlap: a cup whose collision
cylinder sits in a gap between the bowl's wedges shows half inside the bowl's wall, a toy box
shows its corner through the drawer's side. Stage 4 renders the visual meshes.
"""
from __future__ import annotations

TOOL = {
    "name": "object_visual_penetration_audit",
    "title": "Object visual penetration audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Object-object and object-table interpenetration measured on the VISUAL meshes (what Stage 4 renders), "
               "which collision hulls and wedges can hide.",
    "description": (
        "episode_physics_check and penetration_check measure interpenetration on collision geometry; this audit "
        "measures it on the drawn geometry. Scene objects are the top-level non-robot bodies (free objects and "
        "fixtures) with their subtrees, leaving out a subtree bolted to the robot by an equality (an arm named "
        "without the robot prefix); each body's visual geoms (groups 0-2, drawn, not colliding; a body without "
        "them uses its collision geoms) become solids: a primitive is tested analytically, a closed (watertight) mesh by "
        "a filled voxel grid of pitch_m (default 1.5 mm) with the depth taken as the distance to dense surface samples, "
        "and an open mesh (which encloses nothing) provides probe points and causes an explicit incomplete-coverage "
        "failure. On every "
        "stride-th frame, for every pair of objects whose bounding boxes overlap, the surface samples of one (about "
        "one per spacing_m squared) are moved into the other's body frames and tested against its solids. The depth of a pair is the "
        "min_points-th deepest point (single points at a sharp edge do not count); it fails above tol_m (default "
        "5 mm). Every object's lowest visual point may not sink more than table_tol_m (default 5 mm) into the "
        "tabletop over the table. Reports the worst frame per pair with its phase label and depth."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "tol_m", "type": "float", "description": "Allowed object-object visual interpenetration. Default 0.005."},
        {"name": "table_tol_m", "type": "float", "description": "Allowed sinking of a visual mesh into the tabletop. Default 0.005."},
        {"name": "spacing_m", "type": "float", "description": "Target surface sample spacing. Default 0.0025."},
        {"name": "min_points", "type": "int", "description": "Points that must be that deep. Default 3."},
        {"name": "stride", "type": "int", "description": "Check every n-th frame. Default 2."},
        {"name": "pitch_m", "type": "float", "description": "Voxel pitch of a closed mesh's inside test (also its depth resolution). Default 0.0015."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {objects, frame, label, depth_mm, bodies}; pairs {a|b: worst depth mm, frame}; table {object: deepest mm, frame}; open_meshes (geoms that could not be tested as containers)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "trimesh"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "penetration", "visual", "mesh", "rendering", "stage4"],
    "status": "tested",
    "entrypoint": "object_visual_penetration_audit",
}

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from tool_library.common import episode_audit as EA


def _area(m, g):
    t = int(m.geom_type[g]); s = np.asarray(m.geom_size[g], float)
    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(m.geom_dataid[g])
        v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        f0, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        V = np.asarray(m.mesh_vert[v0:v0 + nv], float); F = np.asarray(m.mesh_face[f0:f0 + nf], int)
        return float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()) if nf else 0.0
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        return 8 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2])
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return 4 * np.pi * s[0] ** 2
    if t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        return 2 * np.pi * s[0] * 2 * s[1] + 2 * np.pi * s[0] ** 2
    return 0.01


class _Solid:
    """Inside test and depth for one geom, in its body's frame: a primitive analytically, a closed mesh by a filled voxel
    grid (its pitch bounds the error) with depth = distance to dense surface samples. An open mesh cannot hold
    anything inside it and is used only as a probe surface."""

    def __init__(self, m, g, pitch, spacing):
        import trimesh
        self.R, self.t = EA.geom_in_body(m, g)
        self.type = int(m.geom_type[g]); self.size = np.asarray(m.geom_size[g], float)
        self.name = m.geom(g).name or f"geom{g}"
        n = int(np.clip(_area(m, g) / spacing ** 2, 300, 40000))
        P, N = EA.geom_surface_samples(m, g, n)
        self.surface = P @ self.R.T + self.t                      # body frame
        self.vox = None
        self.closed = True
        if self.type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[g])
            V = np.asarray(m.mesh_vert[int(m.mesh_vertadr[mid]):int(m.mesh_vertadr[mid]) + int(m.mesh_vertnum[mid])], float)
            F = np.asarray(m.mesh_face[int(m.mesh_faceadr[mid]):int(m.mesh_faceadr[mid]) + int(m.mesh_facenum[mid])], int)
            tm = trimesh.Trimesh(V, F, process=True)
            self.closed = bool(tm.is_watertight)
            if self.closed:
                self.vox = tm.voxelized(pitch).fill()
                self.tree = cKDTree(P)
        lo, hi = self.surface.min(0), self.surface.max(0)
        self.box = (lo - pitch, hi + pitch)

    def depth(self, Pb):
        """Depth inside this solid (0 outside) of body-frame points."""
        out = np.zeros(len(Pb))
        if not self.closed:
            return out
        near = np.all((Pb >= self.box[0]) & (Pb <= self.box[1]), axis=1)
        if not near.any():
            return out
        L = (Pb[near] - self.t) @ self.R                          # geom frame
        s = self.size
        T = mujoco.mjtGeom
        if self.type == int(T.mjGEOM_MESH):
            ins = self.vox.is_filled(L)
            d = np.zeros(len(L))
            if ins.any():
                d[ins] = self.tree.query(L[ins])[0]
        elif self.type == int(T.mjGEOM_BOX):
            q = s - np.abs(L)
            d = np.where(np.all(q > 0, axis=1), q.min(axis=1), 0.0)
        elif self.type == int(T.mjGEOM_SPHERE):
            d = np.clip(s[0] - np.linalg.norm(L, axis=1), 0, None)
        elif self.type == int(T.mjGEOM_CYLINDER):
            rr = s[0] - np.linalg.norm(L[:, :2], axis=1); hh = s[1] - np.abs(L[:, 2])
            d = np.where((rr > 0) & (hh > 0), np.minimum(rr, hh), 0.0)
        elif self.type == int(T.mjGEOM_CAPSULE):
            z = np.clip(L[:, 2], -s[1], s[1])
            d = np.clip(s[0] - np.linalg.norm(L - np.c_[np.zeros((len(L), 2)), z], axis=1), 0, None)
        elif self.type == int(T.mjGEOM_ELLIPSOID):
            k = np.linalg.norm(L / s, axis=1)
            d = np.where(k < 1, (1 - k) * s.min(), 0.0)
        else:
            d = np.zeros(len(L))
        out[near] = d
        return out


def object_visual_penetration_audit(scene, trajectory, frames=None, tol_m=0.005, table_tol_m=0.005, spacing_m=0.0025,
                                    min_points=3, stride=2, pitch_m=0.0015):
    EA.check_positive(tol_m=tol_m, table_tol_m=table_tol_m, spacing_m=spacing_m, min_points=min_points, stride=stride,
                      pitch_m=pitch_m)
    from tool_library.verifiers._recording import collision_recording
    ep = EA.Episode(scene, collision_recording(trajectory) if frames is None else trajectory, frames)
    m = ep.model
    tb = ep.table
    skip = {0} | ({tb["body"]} if tb else set())
    # bodies bolted to the robot by an equality (an arm named without the robot prefix) are robot, not scene objects
    bolted = set()
    for e in range(m.neq):
        b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if int(m.eq_type[e]) in (EA.TJ.WELD, EA.TJ.CONNECT) and (b1 in ep.robot_bodies) != (b2 in ep.robot_bodies):
            bolted.add(b2 if b1 in ep.robot_bodies else b1)
    groups = {}
    for b in range(1, m.nbody):
        if b in skip or b in ep.robot_bodies or int(m.body_parentid[b]) != 0:
            continue
        sub = EA.TJ.subtree(m, b)
        if sub & bolted and not any(o["body"] in sub for o in ep.objects.values()):
            continue
        bg = {}
        for bb in sub:
            gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bb and int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
            vis = [g for g in gs if EA.geom_is_visual(m, g)]
            use = vis or [g for g in gs if EA.geom_collides(m, g)]
            if use:
                bg[bb] = use
        if bg:
            groups[ep.body_name(b)] = bg
    if not groups:
        return EA.finish([], None, pairs={}, table={}, note="no scene objects with geometry")
    solids = {k: {b: [_Solid(m, g, pitch_m, spacing_m) for g in gs] for b, gs in bg.items()} for k, bg in groups.items()}
    open_meshes = sorted({sd.name for k in solids for b in solids[k] for sd in solids[k][b] if not sd.closed})
    probes = {}
    for k in solids:
        probes[k] = {}
        for b, sl in solids[k].items():
            P = np.vstack([sd.surface for sd in sl])
            probes[k][b] = P[::max(1, len(P) // 3000)]
    xp, xq = ep.kin()
    names = list(groups)
    pairs, tab, events = {}, {}, []
    for t in range(ep.a, ep.b + 1, int(stride)):
        R = {}
        world = {}
        for k in names:
            for b, Pb in probes[k].items():
                R[b] = EA.quat_to_mat(xq[t, b])
                world[(k, b)] = Pb @ R[b].T + xp[t, b]
        box = {k: (np.min([world[(k, b)].min(0) for b in probes[k]], axis=0), np.max([world[(k, b)].max(0) for b in probes[k]], axis=0))
               for k in names}
        if tb is not None:
            for k in names:
                if all(bb in ep.static_bodies for bb in groups[k]):
                    continue                          # a fixture welded to the world is part of the bench
                W = np.vstack([world[(k, b)] for b in probes[k]])
                inside = (W[:, 0] >= tb["xy"][0]) & (W[:, 0] <= tb["xy"][1]) & (W[:, 1] >= tb["xy"][2]) & (W[:, 1] <= tb["xy"][3])
                if not inside.any():
                    continue
                z = np.sort(W[inside, 2])
                depth = tb["top_z"] - z[min(min_points, len(z)) - 1]
                if depth * 1000 > tab.get(k, {"depth_mm": -1e9})["depth_mm"]:
                    tab[k] = {"depth_mm": round(1000 * depth, 2), "frame": t}
                if depth > table_tol_m:
                    events.append({"objects": [k, "table"], "frame": t, "label": ep.labels[t], "depth_mm": round(1000 * depth, 1)})
        for i, ka in enumerate(names):
            for kb in names[i + 1:]:
                (la, ha), (lb, hb) = box[ka], box[kb]
                if np.any(la > hb + 0.002) or np.any(lb > ha + 0.002):
                    continue
                best, bodies = 0.0, None
                for (k1, k2) in ((ka, kb), (kb, ka)):
                    for b1 in probes[k1]:
                        W = world[(k1, b1)]
                        for b2, sl in solids[k2].items():
                            local = (W - xp[t, b2]) @ R[b2] if b2 in R else (W - xp[t, b2]) @ EA.quat_to_mat(xq[t, b2])
                            dep = np.zeros(len(local))
                            for sd in sl:
                                dep = np.maximum(dep, sd.depth(local))
                            dep = np.sort(dep)[::-1]
                            if len(dep) >= min_points and dep[min_points - 1] > best:
                                best = float(dep[min_points - 1]); bodies = [ep.body_name(b1), ep.body_name(b2)]
                key = f"{ka}|{kb}"
                if best * 1000 > pairs.get(key, {"depth_mm": -1})["depth_mm"]:
                    pairs[key] = {"depth_mm": round(1000 * best, 2), "frame": t, "bodies": bodies}
                if best > tol_m:
                    events.append({"objects": [ka, kb], "frame": t, "label": ep.labels[t], "depth_mm": round(1000 * best, 1), "bodies": bodies})
    issues = ([f"visual penetration coverage unavailable for non-watertight meshes: {', '.join(open_meshes)}; "
               "repair the visual surface or provide a closed audited geometry"] if open_meshes else [])
    by = {}
    for e in events:
        by.setdefault(tuple(e["objects"]), []).append(e)
    for key, evs in sorted(by.items(), key=lambda kv: -max(e["depth_mm"] for e in kv[1])):
        w = max(evs, key=lambda e: e["depth_mm"])
        if key[1] == "table":
            issues.append(f"{key[0]}'s visible mesh sinks {w['depth_mm']} mm into the tabletop at frame {w['frame']} ({w['label']}; "
                          f"{len(evs)} checked frames): its collision geometry rests higher than its mesh; fit the collision "
                          f"geometry to the mesh bottom")
        else:
            issues.append(f"the visible meshes of {key[0]} and {key[1]} interpenetrate by {w['depth_mm']} mm at frame {w['frame']} "
                          f"({w['label']}; bodies {w.get('bodies')}; {len(evs)} checked frames) although their collision geometry "
                          f"does not: make the collision geometry follow the mesh (more wedges, a thinner wall) and re-execute")
    worst = max(events, key=lambda e: e["depth_mm"]) if events else None
    return EA.finish(issues, worst, pairs=pairs, table=tab, n_events=len(events), open_meshes=open_meshes[:20],
                     stride=int(stride), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(object_visual_penetration_audit(sys.argv[1], sys.argv[2]), indent=1))
