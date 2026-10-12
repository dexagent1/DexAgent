"""Collision-visual alignment check: what an object shows is what its physics touches."""
from __future__ import annotations

TOOL = {
    "name": "collision_visual_alignment_check",
    "title": "Collision-visual alignment check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "For every visible, non-colliding geom of every object, measures how far its surface lies outside the object's collision geoms and flags exposed visual parts that stick out of the physics by more than a tolerance (a collision primitive smaller than the mesh it stands for, a visual mesh moved off its collider).",
    "description": (
        "When a collision primitive is smaller than the visual mesh it stands for, the hand visibly sinks into the "
        "object (or the object into the table) with no contact at all: the Sharpa palm box once left the visible palm "
        "17 mm outside physics. Each object (the Stage 2 manifest's objects, else every top-level non-robot body) is "
        "posed at keyframe 0. For each visible geom that does not collide (group 0-2, alpha > 0), its surface points "
        "(mesh vertices, or samples of a primitive) are measured against the object's collision geoms: signed "
        "distance to each (primitives exactly, meshes as their convex hull, which is what MuJoCo collides), keeping "
        "the nearest. A point counts as exposed when it lies more than tol_m outside every collision geom AND more "
        "than tol_m outside the convex hull of all the object's collision geometry, so decorations enclosed by the "
        "object (the parts inside a computer case, a spring in a slot, a straw in a bottle) are not reported. Thin "
        "decorations (a geom whose smallest extent is below thin_m: wires, labels, stripes) are skipped. A geom "
        "fails when its worst exposed point is beyond tol_m and at least min_frac of its points are exposed; a small "
        "geom (largest extent below decoration_m) that no collision geom comes near is a loose decoration and only warned about."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json naming the objects. Default: top-level non-robot bodies."},
        {"name": "objects", "type": "list[str] | None", "description": "Only these objects (ids or body names)."},
        {"name": "tol_m", "type": "float", "description": "Allowed visual surface outside the physics, metres. Default 0.005."},
        {"name": "thin_m", "type": "float", "description": "Visual geoms thinner than this are decorations and skipped, metres. Default 0.004."},
        {"name": "min_frac", "type": "float", "description": "Smallest fraction of a geom's points that must be exposed for it to fail. Default 0.02."},
        {"name": "decoration_m", "type": "float", "description": "A geom smaller than this (largest extent) that no collision geom touches at all is a loose decoration (a plug, a sticker): reported as a warning, not a failure. Default 0.015."},
        {"name": "ignore", "type": "list[str] | None", "description": "Geom or body names (substrings) not to judge."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; warnings (loose decorations); geoms [{geom, body, object, n_points, max_out_m, exposed_frac, verdict}]; skipped (thin or no collision); worst (largest exposed distance, m)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "collision", "visual", "geometry", "alignment", "stage2"],
    "status": "tested",
    "entrypoint": "collision_visual_alignment_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA


def _min_extent(model, g):
    P = SA.geom_local_points(model, g)
    if len(P) < 2:
        return 0.0
    if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        # principal axes of the vertices: the smallest spread is the thickness of a flat or thin mesh
        Q = P - P.mean(axis=0)
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        return float(np.ptp(Q @ Vt.T, axis=0).min())
    return float((P.max(axis=0) - P.min(axis=0)).min())


def collision_visual_alignment_check(scene, recon=None, objects=None, tol_m=0.005, thin_m=0.004, min_frac=0.02, decoration_m=0.015, ignore=None):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    groups = SA.object_groups(model, recon, objects, ref=st["model"], cfg=st["cfg"])
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    ign = [str(x) for x in (ignore or [])]
    rows, skipped, issues, warnings = [], [], [], []
    worst = 0.0
    from scipy.spatial import ConvexHull
    seen = set()
    for grp in groups:
        if grp["root"] < 0:
            continue
        gs = SA.body_geoms(model, grp["bodies"])
        col = [g for g in gs if SA.geom_collidable(model, g)]
        vis = [g for g in gs if SA.geom_visible(model, g) and not SA.geom_collidable(model, g) and g not in seen]
        seen |= set(vis)
        if not vis:
            continue
        if not col:
            skipped.append({"object": grp["id"], "why": "no collision geometry at all (contact_exclusion_audit / hand_contact_pair_mask_check judge that)"})
            continue
        CP = np.vstack([p for p in (SA.geom_world_points(model, data, c) for c in col) if len(p)])
        try:
            hull = ConvexHull(CP).equations
        except Exception:                                    # noqa: BLE001 - flat collision geometry
            hull = None
        for g in vis:
            lab = SA.geom_label(model, g)
            body = model.body(int(model.geom_bodyid[g])).name
            if any(s in lab or s in body for s in ign):
                skipped.append({"geom": lab, "why": "ignored"})
                continue
            if _min_extent(model, g) < thin_m:
                skipped.append({"geom": lab, "why": f"thin (< {thin_m} m)"})
                continue
            P = SA.geom_world_points(model, data, g)
            if not len(P):
                continue
            D = np.min(np.stack([SA.signed_distance(model, data, c, P) for c in col]), axis=0)
            if hull is not None:
                H = (P @ hull[:, :3].T + hull[:, 3]).max(axis=1)
                exposed = (D > tol_m) & (H > tol_m)
            else:
                exposed = D > tol_m
            out = float(D[exposed].max()) if exposed.any() else float(max(0.0, min(D.max(), tol_m)))
            frac = float(exposed.mean())
            verdict = "ok"
            P_loc = SA.geom_local_points(model, g)
            size = float((P_loc.max(axis=0) - P_loc.min(axis=0)).max()) if len(P_loc) else 0.0
            if exposed.any() and frac >= min_frac and size < decoration_m and float(D.min()) > tol_m:
                verdict = "loose decoration"
                warnings.append(f"visual geom {lab} of {grp['id']} ({size * 1000:.0f} mm) has no collision geometry near it "
                                f"({float(D.min()) * 1000:.0f} mm away): a decoration the hand passes through")
            elif exposed.any() and frac >= min_frac:
                verdict = "exposed"
                worst = max(worst, out)
                issues.append(f"visual geom {lab} of {grp['id']} sticks {out * 1000:.1f} mm out of the object's collision geometry "
                              f"({frac * 100:.0f}% of its surface): the hand would sink into it with no contact; enlarge or add a collision "
                              f"geom that covers it (or make the visual match the collider)")
            rows.append({"geom": lab, "body": body, "object": grp["id"], "n_points": int(len(P)), "max_out_m": round(out, 5),
                         "exposed_frac": round(frac, 3), "verdict": verdict})
    return SA.report(issues, warnings=warnings, geoms=rows, skipped=skipped, worst=round(worst, 5))
