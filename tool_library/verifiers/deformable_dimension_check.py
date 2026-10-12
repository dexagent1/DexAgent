"""Deformable dimension check: a rope is as long and as thick as the centreline measured in the video."""
from __future__ import annotations

TOOL = {
    "name": "deformable_dimension_check",
    "title": "Deformable dimension check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Measures each rope/cable object's length along its links and its radius at keyframe 0 and compares them with the measured centreline (a measure_rope result, or a length and radius), and, when the measurement has a centreline in the rig frame, how far the simulated rope lies from it.",
    "description": (
        "A rope built 30% short cannot make the knot the video makes, one built twice as thick does not fit the "
        "loop, and one laid 10 cm off the measured centreline starts where the hands do not reach. The rope objects "
        "are found as deformable_topology_check finds them (manifest deformables named rope/cable/..., else chains of "
        "linked bodies). Length: the polyline through the link centres in chain order plus half a link at each end "
        "(capsule/cylinder half-length; for other links half the spacing). Radius: the median capsule/cylinder/sphere "
        "radius of the links. Both are compared with the measurement: `measurement` is a measure_rope output (dict or "
        "json path with length_m, radius_m and optionally centreline_rig / points) or {length_m, radius_m}. The "
        "relative length error must be within length_tol and the radius error within radius_tol; with a rig-frame "
        "centreline, the mean distance from the link centres to it must be at most centreline_tol_m. With several "
        "ropes, `rope` names the one measured. A measurement measure_rope marked reliable: false (a coil, a crossing, "
        "an unjoined piece) fails the check with its reasons: its length is a lower bound."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "measurement", "type": "str | dict", "description": "measure_rope output (json path or dict) or {length_m, radius_m[, centreline_rig]}."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json naming the deformable objects."},
        {"name": "rope", "type": "str | None", "description": "The rope object to judge when the scene has several (id or root body)."},
        {"name": "length_tol", "type": "float", "description": "Allowed relative length error. Default 0.15."},
        {"name": "radius_tol", "type": "float", "description": "Allowed relative radius error. Default 0.35."},
        {"name": "centreline_tol_m", "type": "float", "description": "Allowed mean distance of the links from a rig-frame centreline, metres. Default 0.03."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; rope id; n_links; length_m {scene, measured, rel_err}; radius_m {scene, measured, rel_err}; centreline_mean_dist_m."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "deformable", "rope", "length", "radius", "measure_rope", "stage2"],
    "status": "tested",
    "entrypoint": "deformable_dimension_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA


def deformable_dimension_check(scene, measurement, recon=None, rope=None, length_tol=0.15, radius_tol=0.35, centreline_tol_m=0.03):
    meas, _ = SA.load_json(measurement)
    if not isinstance(meas, dict) or "length_m" not in meas:
        raise ValueError("measurement: give a measure_rope result (json or dict) with length_m (and radius_m)")
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    ropes = SA.rope_groups(model, recon, [rope] if rope else None, ref=st["model"], cfg=st["cfg"])
    if not ropes:
        return SA.report([f"no rope object{' ' + str(rope) if rope else ''} found in the scene to compare with the measurement"])
    if len(ropes) > 1 and rope is None:
        raise ValueError(f"rope: the scene has {len(ropes)} ropes ({[g['id'] for g, _ in ropes]}); name the measured one")
    g, links = ropes[0]
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    L = set(links)
    adj = {b: set() for b in links}
    for b in links:
        p = int(model.body_parentid[b])
        if p in L:
            adj[b].add(p); adj[p].add(b)
    order = SA.chain_order(links, adj)
    P = np.array([data.xpos[b] for b in order])
    radii, halves = [], []
    for b in order:
        for x in SA.body_geoms(model, [b]):
            t = int(model.geom_type[x])
            if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                radii.append(float(model.geom_size[x][0])); halves.append(float(model.geom_size[x][1]))
            elif t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                radii.append(float(model.geom_size[x][0]))
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(0)
    half = float(np.median(halves)) if halves else (float(np.median(steps)) / 2 if len(steps) else 0.0)
    length = float(steps.sum() + 2 * half)
    radius = float(np.median(radii)) if radii else None
    issues = []
    if meas.get("reliable") is False:
        # measure_rope flagged its own numbers (a coil, a crossing, a piece it could not join): comparing a scene with
        # them would pass a wrong rope or fail a right one
        issues.append("the measurement is marked unreliable by measure_rope (" + "; ".join(str(i) for i in meas.get("issues") or [])[:400]
                      + "): measure the rope on a frame where it lies untangled, or give length_m / radius_m known otherwise")
    ml = float(meas["length_m"])
    le = (length - ml) / ml if ml > 0 else float("inf")
    if abs(le) > length_tol:
        issues.append(f"rope {g['id']} is {length:.3f} m long in the scene but the video's centreline measures {ml:.3f} m "
                      f"({le * 100:+.0f}%, tolerance {length_tol * 100:.0f}%); rebuild it at the measured length")
    rr = None
    if meas.get("radius_m") is not None and radius is not None:
        mr = float(meas["radius_m"])
        rr = (radius - mr) / mr if mr > 0 else float("inf")
        if abs(rr) > radius_tol:
            issues.append(f"rope {g['id']} has radius {radius * 1000:.1f} mm in the scene but {mr * 1000:.1f} mm in the video "
                          f"({rr * 100:+.0f}%); rebuild it at the measured radius")
    cl = meas.get("centreline_rig") or meas.get("points")
    mean_d = None
    if cl is not None:
        C = np.asarray(cl, float)
        if C.ndim == 2 and C.shape[1] == 3 and len(C) >= 2:
            # distance of each link centre to the measured polyline
            def seg_d(p):
                a, b = C[:-1], C[1:]
                ab = b - a
                t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
                return float(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))
            mean_d = float(np.mean([seg_d(p) for p in P]))
            if mean_d > centreline_tol_m:
                issues.append(f"rope {g['id']} lies {mean_d * 100:.1f} cm (mean) from the measured centreline: lay it along the "
                              f"centreline (construct_rope centreline=...)")
    return SA.report(issues, rope=g["id"], n_links=len(links),
                     length_m={"scene": round(length, 4), "measured": round(ml, 4), "rel_err": round(le, 4)},
                     radius_m={"scene": None if radius is None else round(radius, 5), "measured": meas.get("radius_m"),
                               "rel_err": None if rr is None else round(rr, 4)},
                     centreline_mean_dist_m=None if mean_d is None else round(mean_d, 4))
