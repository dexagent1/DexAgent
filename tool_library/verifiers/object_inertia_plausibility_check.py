"""Object inertia plausibility check: each object body's inertia is the inertia of its own shape at its own mass."""
from __future__ import annotations

TOOL = {
    "name": "object_inertia_plausibility_check",
    "title": "Object inertia plausibility check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "For every object body with mass, compares the principal moments of inertia and the centre of mass with those of its geometry filled uniformly at the body's mass, flagging point masses, tiny inertia on big bodies, huge inertia on small ones and a centre of mass outside the body.",
    "description": (
        "A body's inertia decides how it turns in a grasp and how easily a push spins it. An explicit inertial with "
        "a point-mass diagonal (1e-9) lets a held object spin freely about any contact, a huge diagonal makes it a "
        "flywheel that no finger can turn, and a centre of mass placed outside the shape makes it tip from a light "
        "touch. For each object body (the Stage 2 manifest's bodies, else every non-robot body) heavier than "
        "min_mass_kg and carrying geometry, the expected inertia is computed from its collision geoms (all visible "
        "geoms when it has none: primitives exactly, meshes as their convex hull) at uniform density scaled to the "
        "body's mass, about the body's actual centre of mass. Each sorted principal moment of the body must be within "
        "[1/ratio_tol, ratio_tol] of the expected one, and the centre of mass must lie inside the geometry's "
        "bounding box grown by com_margin of its size. Jointed parts (a lid, a drawer) are judged as bodies of their "
        "own. Hollow shapes modelled with solid collision primitives get an inertia up to ~2x smaller than a thin "
        "shell, which the default tolerance of 4 accepts."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json naming the objects' bodies. Default: every non-robot body."},
        {"name": "objects", "type": "list[str] | None", "description": "Only these objects (ids or body names)."},
        {"name": "ratio_tol", "type": "float", "description": "Allowed factor between actual and expected principal moments. Default 4.0."},
        {"name": "com_margin", "type": "float", "description": "Allowed centre-of-mass excursion outside the geometry's box, as a fraction of its size. Default 0.1."},
        {"name": "min_mass_kg", "type": "float", "description": "Bodies lighter than this (spring coils, carriers) are skipped. Default 0.001."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; bodies [{body, object, mass_kg, inertia (sorted principal), expected, ratios, com_outside_m, verdict}]; skipped; worst (largest factor off)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "trimesh"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "inertia", "mass", "com", "physics", "stage2"],
    "status": "tested",
    "entrypoint": "object_inertia_plausibility_check",
}

import math

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA


def _prim_inertia(model, g):
    """(volume, centroid in geom frame, inertia tensor about the centroid per unit density, geom frame)."""
    t = int(model.geom_type[g]); s = np.array(model.geom_size[g], float)
    G = mujoco.mjtGeom
    if t == int(G.mjGEOM_BOX):
        a, b, c = 2 * s[:3]
        V = a * b * c
        return V, np.zeros(3), V / 12.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    if t == int(G.mjGEOM_SPHERE):
        V = 4 / 3 * math.pi * s[0] ** 3
        return V, np.zeros(3), np.eye(3) * 0.4 * V * s[0] ** 2
    if t == int(G.mjGEOM_ELLIPSOID):
        a, b, c = s[:3]
        V = 4 / 3 * math.pi * a * b * c
        return V, np.zeros(3), V / 5.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    if t == int(G.mjGEOM_CYLINDER):
        r, h = s[0], 2 * s[1]
        V = math.pi * r * r * h
        return V, np.zeros(3), V * np.diag([(3 * r * r + h * h) / 12, (3 * r * r + h * h) / 12, r * r / 2])
    if t == int(G.mjGEOM_CAPSULE):
        r, h = s[0], 2 * s[1]
        Vc, Vs = math.pi * r * r * h, 4 / 3 * math.pi * r ** 3
        Ixc = Vc * (3 * r * r + h * h) / 12
        Ixs = Vs * (0.4 * r * r + h * h / 4 + 3 * h * r / 8)
        return Vc + Vs, np.zeros(3), np.diag([Ixc + Ixs, Ixc + Ixs, Vc * r * r / 2 + Vs * 0.4 * r * r])
    if t == int(G.mjGEOM_MESH):
        import trimesh
        V = SA.mesh_vertices(model, g)
        try:
            h = trimesh.convex.convex_hull(V)
            vol = float(h.volume)
            if vol <= 0:
                return 0.0, np.zeros(3), np.zeros((3, 3))
            return vol, np.array(h.center_mass, float), np.array(h.moment_inertia, float)   # density 1: inertia per unit density
        except Exception:                                    # noqa: BLE001 - degenerate mesh
            return 0.0, np.zeros(3), np.zeros((3, 3))
    return 0.0, np.zeros(3), np.zeros((3, 3))


def _expected(model, b, geoms):
    """Expected inertia (3x3, body frame, about the actual COM) and the geometry box, at the body's mass."""
    parts = []
    for g in geoms:
        V, c, I = _prim_inertia(model, g)
        if V <= 0:
            continue
        R = SA_quat_mat(model.geom_quat[g])
        cb = model.geom_pos[g] + R @ c
        parts.append((V, cb, R @ I @ R.T))
    if not parts:
        return None, None
    Vt = sum(p[0] for p in parts)
    m = float(model.body_mass[b])
    rho = m / Vt
    com_act = np.array(model.body_ipos[b], float)
    I = np.zeros((3, 3))
    for V, cb, Ic in parts:
        d = cb - com_act
        I += rho * Ic + rho * V * (d @ d * np.eye(3) - np.outer(d, d))
    return I, Vt


def SA_quat_mat(q):
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(q, float))
    return M.reshape(3, 3)


def object_inertia_plausibility_check(scene, recon=None, objects=None, ratio_tol=4.0, com_margin=0.1, min_mass_kg=0.001):
    if ratio_tol <= 1.0:
        raise ValueError(f"ratio_tol must be > 1, got {ratio_tol}")
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    groups = SA.object_groups(model, recon, objects, ref=st["model"], cfg=st["cfg"])
    rows, skipped, issues = [], [], []
    worst = 1.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    seen = set()
    for grp in groups:
        if grp["root"] < 0:
            issues.append(f"object {grp['id']}: body {grp.get('missing')} is not in the scene")
            continue
        for b in sorted(grp["bodies"]):
            if b in seen:
                continue
            seen.add(b)
            name = model.body(b).name
            m = float(model.body_mass[b])
            gs = SA.body_geoms(model, [b])
            col = [g for g in gs if SA.geom_collidable(model, g)]
            use = col or [g for g in gs if SA.geom_visible(model, g)]
            if m < min_mass_kg or not use:
                skipped.append({"body": name, "mass_kg": round(m, 6), "why": "light" if m < min_mass_kg else "no geometry"})
                continue
            I_exp, Vt = _expected(model, b, use)
            if I_exp is None:
                skipped.append({"body": name, "mass_kg": round(m, 6), "why": "degenerate geometry"})
                continue
            e_exp = np.sort(np.linalg.eigvalsh(I_exp))
            e_act = np.sort(np.array(model.body_inertia[b], float))
            ratios = e_act / np.maximum(e_exp, 1e-15)
            # the box of the body's geometry in the body frame
            pts = []
            for g in use:
                P = SA.geom_local_points(model, g)
                if len(P):
                    pts.append(model.geom_pos[g] + P @ SA_quat_mat(model.geom_quat[g]).T)
            P = np.vstack(pts)
            lo, hi = P.min(axis=0), P.max(axis=0)
            pad = com_margin * np.maximum(hi - lo, 1e-4)
            c = np.array(model.body_ipos[b], float)
            outside = float(np.linalg.norm(np.maximum(lo - pad - c, 0) + np.maximum(c - hi - pad, 0)))
            verdict = "ok"
            off = float(max(ratios.max(), 1.0 / max(ratios.min(), 1e-12)))
            if off > ratio_tol:
                worst = max(worst, off)
                kind = "point-like" if ratios.min() < 1.0 / ratio_tol else "too large"
                verdict = kind
                issues.append(f"body {name} ({grp['id']}): principal inertia {SA.r6(e_act, 9)} vs {SA.r6(e_exp, 9)} expected from its shape at "
                              f"{m:.4g} kg ({'smallest' if kind == 'point-like' else 'largest'} factor {off:.3g}); "
                              f"let MuJoCo infer the inertia from the geometry (drop the explicit <inertial> diaginertia) or fix it")
            if outside > 0:
                verdict = "com outside" if verdict == "ok" else verdict + ", com outside"
                issues.append(f"body {name} ({grp['id']}): centre of mass {SA.r6(c, 4)} lies {outside * 1000:.1f} mm outside its geometry; "
                              f"put the <inertial> pos inside the object")
            rows.append({"body": name, "object": grp["id"], "mass_kg": round(m, 6), "inertia": SA.r6(e_act, 9), "expected": SA.r6(e_exp, 9),
                         "ratios": SA.r6(ratios, 3), "com_outside_m": round(outside, 5), "verdict": verdict})
    return SA.report(issues, bodies=rows, skipped=skipped, worst=round(worst, 3))
