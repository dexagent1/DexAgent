"""Spatial, material-coordinate diagnostics for an executed rope state.

This tool does not prescribe a knot trajectory or certify task completion.
It makes the geometry that must guide a deformable plan observable.
"""
from __future__ import annotations

TOOL = {
    "name": "inspect_rope_state",
    "title": "Inspect current rope geometry and threading",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Read a cable's current recorded centreline, body/material coordinates, spatial crossings, loop apertures, endpoint relations, and observed end passages through a selected material loop.",
    "description": (
        "Use a physically executed prefix's scene and trajectory to plan the next deformable manipulation. "
        "A passing grasp, in_hand or no_slip proves contact, not topological progress. Returns the actual ordered "
        "centreline, cumulative arc length and segment bodies, projected crossings with world coordinates and "
        "over/under order, candidate near-planar loop apertures, and each end's plane distance and projected "
        "boundary clearance. Candidate loops are discovered in three orthogonal views; loop_bounds can instead "
        "select a material arc explicitly as [vertex index + fraction, vertex index + fraction]. These bounds "
        "track material, not a fixed world location. To inspect a completed thread attempt, provide history_start "
        "and explicit loop_bounds; every recorded frame through frame is checked for the selected endpoint's "
        "side change through the moving aperture. Crossings outside the aperture, excessive opening or "
        "non-planarity, and insufficient strand clearance are reported separately. All passage results are "
        "diagnostic candidates: interpolation cannot prove continuous collision clearance, a hand may not fit "
        "through an aperture its rope fits, and a passage alone is not a knot. topology reports the existing "
        "four-closure classifier, but final acceptance still requires rope_knot_check and the complete gate suite "
        "after release/settling. A single image or static loop is insufficient. This tool never changes scene or state."
    ),
    "inputs": [
        {"name": "scene", "type": "path | None", "description": "MuJoCo scene containing the cable; with no trajectory reads its home keyframe."},
        {"name": "trajectory", "type": "path | None", "description": "Recorded trajectory.npz; use the actual executed prefix, not planned wrist references."},
        {"name": "rope", "type": "str | None", "description": "Cable name prefix, required when the scene contains multiple cables."},
        {"name": "frame", "type": "int", "description": "Recorded frame, default -1 (last)."},
        {"name": "points", "type": "list | None", "description": "Optional standalone [N,3] centreline for geometric analysis; no simulation or provenance is inferred."},
        {"name": "radius_m", "type": "float | None", "description": "Cable radius for diagnostic clearance; default maximum collision capsule radius from the scene, or zero for points-only input."},
        {"name": "loop_bounds", "type": "list[float] | None", "description": "Explicit increasing pair of centreline parameters delimiting a material loop arc; omitted discovers candidates."},
        {"name": "history_start", "type": "int | None", "description": "First trajectory frame for passage inspection, requires explicit loop_bounds and a trajectory. Checks every recorded frame without decimation."},
        {"name": "endpoint", "type": "str", "description": "'first' or 'last' endpoint for passage tracking, default 'last'; must lie outside loop_bounds."},
        {"name": "endpoint_radius_m", "type": "float | None", "description": "Required radius for passage clearance, default rope radius. Supply a larger envelope for a cap or held tool; this is not a full hand collision screen."},
        {"name": "max_loops", "type": "int", "description": "Maximum discovered loop candidates, default 6 (1..20)."},
        {"name": "topology", "type": "bool", "description": "Include the existing four-closure topological classification, default true."},
    ],
    "outputs": [
        {"name": "state", "type": "dict", "description": "diagnostic_only, source, centreline, material_coordinates, endpoints, crossings, loops, passages, topology and limitations. No task-success verdict is produced."},
    ],
    "requirements": ["numpy", "scipy", "mujoco>=3.2", "sympy"],
    "simulators": ["MuJoCo"],
    "tags": ["rope", "deformable", "closed loop planning", "threading", "loop aperture", "topology"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "inspect_rope_state",
}

import numpy as np

from tool_library.common import knot as K


def _frame_index(frame, size):
    if isinstance(frame, bool) or int(frame) != frame:
        raise ValueError("frame indices must be integers")
    index = int(frame) if frame >= 0 else size + int(frame)
    if not 0 <= index < size:
        raise ValueError(f"frame {frame} outside trajectory of length {size}")
    return index


def inspect_rope_state(scene=None, trajectory=None, rope=None, frame=-1, points=None, radius_m=None,
                       loop_bounds=None, history_start=None, endpoint="last", endpoint_radius_m=None,
                       max_loops=6, topology=True):
    if endpoint not in ("first", "last"):
        raise ValueError("endpoint must be 'first' or 'last'")
    if isinstance(max_loops, bool) or int(max_loops) != max_loops or not 1 <= max_loops <= 20:
        raise ValueError("max_loops must be an integer in [1, 20]")
    if history_start is not None and (loop_bounds is None or trajectory is None):
        raise ValueError("history_start requires a trajectory and explicit material loop_bounds")
    if points is not None and (scene is not None or trajectory is not None):
        raise ValueError("give points or scene/trajectory, not both")
    body_names, history, times, frame_index = None, None, None, None
    if points is not None:
        P = K._polyline(points)
        radius = 0. if radius_m is None else float(radius_m)
        source = {"kind": "supplied_polyline", "physics_verified": False}
    else:
        if scene is None:
            raise ValueError("give scene (+ trajectory) or points")
        import mujoco
        from tool_library.common.trajectory import load_model, load_trajectory
        from tool_library.verifiers.rope_knot_check import _cable_bodies, _centreline
        model = load_model(scene)
        data = mujoco.MjData(model)
        pre, bodies = _cable_bodies(model, rope)
        body_names = [model.body(b).name for b in bodies]
        if trajectory is None:
            if model.nkey:
                mujoco.mj_resetDataKeyframe(model, data, 0)
            Q, fps = data.qpos[None].copy(), None
        else:
            Q, _, fps = load_trajectory(trajectory)
        if Q.ndim != 2 or Q.shape[1] != model.nq or not np.isfinite(Q).all():
            raise ValueError("trajectory must contain finite qpos with the scene's nq")
        frame_index = _frame_index(frame, len(Q))
        data.qpos[:] = Q[frame_index]
        mujoco.mj_kinematics(model, data)
        P = K._polyline(_centreline(model, data, bodies, pre))
        ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
               and model.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
        if radius_m is None and not ids:
            raise ValueError("no cable collision capsules found; give radius_m for diagnostic clearance")
        radius = float(max(model.geom_size[g, 0] for g in ids)) if radius_m is None else float(radius_m)
        source = {"kind": "recorded_state" if trajectory else "scene_home", "scene": str(scene),
                  "trajectory": None if trajectory is None else str(trajectory), "frame": frame_index,
                  "rope_prefix": pre, "physics_verified": False,
                  "note": "state geometry is read here; execution provenance and physical validity require the trusted rollout audits"}
        if history_start is not None:
            first = _frame_index(history_start, len(Q))
            if first >= frame_index:
                raise ValueError("history_start must precede frame")
            history = []
            for q in Q[first:frame_index + 1]:
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                history.append(_centreline(model, data, bodies, pre))
            times = {"first_frame": first, "last_frame": frame_index, "fps": fps}
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius_m must be finite and nonnegative")
    endpoint_radius = radius if endpoint_radius_m is None else float(endpoint_radius_m)
    if not np.isfinite(endpoint_radius) or endpoint_radius < 0:
        raise ValueError("endpoint_radius_m must be finite and nonnegative")
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    loops = ([dict(K.loop_geometry(P, loop_bounds, radius), id=0, discovery_projection="explicit material bounds")]
             if loop_bounds is not None else K.loop_candidates(P, radius, int(max_loops)))
    for loop in loops:
        loop["endpoints"] = {key: K.endpoint_loop_relation(P[idx], loop, radius, endpoint_radius)
                             for key, idx in (("first", 0), ("last", -1))}
        loop.pop("basis", None)
        loop.pop("boundary_projected", None)
    crossings = []
    for order in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        for c in K.crossings(P[:, order], closed=False):
            over, under = K.curve_point(P, c["s_over"]), K.curve_point(P, c["s_under"])
            crossings.append({"projection": "".join("xyz"[i] for i in order[:2]),
                              "view_from_positive_axis": "xyz"[order[2]],
                              "over_parameter": c["s_over"], "under_parameter": c["s_under"],
                              "over_point": over.tolist(), "under_point": under.tolist(),
                              "strand_gap_m": c["dz"] - 2 * radius, "sign": c["sign"]})
    passages = None
    if history is not None:
        passages = K.endpoint_passages(history, loop_bounds, endpoint, radius, endpoint_radius)
        passages.update(times)
        for event in passages["passage_candidates"] + passages["rejected_plane_crossings"]:
            event["frames"] = [f + times["first_frame"] for f in event["frames"]]
    return {"diagnostic_only": True, "source": source, "radius_m": radius,
            "centreline": P.tolist(), "length_m": float(lengths[-1]),
            "material_coordinates": [{"parameter": i, "arc_length_m": float(s),
                                      "body": body_names[i] if body_names and i < len(body_names) else None}
                                     for i, s in enumerate(lengths)],
            "endpoints": {"first": P[0].tolist(), "last": P[-1].tolist()},
            "crossings": crossings, "loops": loops, "passages": passages,
            "topology": K.check(P) if topology else None,
            "limitations": ["Local grasp/contact success does not establish knot or threading success.",
                            "Automatic loop discovery is incomplete; absence of candidates is not proof that no usable loop exists.",
                            "Apertures use a fitted plane and imaginary closing chord; their clearances exclude hands, table, and other obstacles.",
                            "Observed endpoint passages are diagnostic candidates between recorded states, not continuous collision or topological proof."]}
