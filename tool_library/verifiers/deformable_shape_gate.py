"""Deformable shape gate: a rope's final centreline against a target centreline or target end positions."""
from __future__ import annotations

TOOL = {
    "name": "deformable_shape_gate",
    "title": "Deformable shape gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A rope or cable's centreline on a frame (the last by default) against a target centreline (discrete Frechet "
               "distance) and/or target positions of its two ends.",
    "description": (
        "The goal predicate for a rope laid into a shape ('the rope runs left to right across the table', 'the brass end "
        "rests at the left'). The cable's segment bodies (construct_rope names them <prefix>B_first, <prefix>B_<i>, "
        "<prefix>B_last) give its centreline on the decision frame. The centreline and the target polyline are both "
        "resampled to n points by arc length and compared with the discrete Frechet distance, which respects the order "
        "along the rope (a rope folded back on itself does not match a straight target the way a point-cloud distance "
        "would); unless ordered, the rope may run either way along the target. planar (or a [k, 2] target) compares x, y "
        "only. target_ends {first, last} checks the end segments, swapped when that fits better and ordered is false. "
        "The issue names the largest deviation's location."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "rope", "type": "str | None", "description": "Cable name prefix (e.g. 'red_rope'); default the only cable."},
        {"name": "target_points", "type": "list[list[float]] | None", "description": "Target centreline [k, 3] (or [k, 2] planar), m."},
        {"name": "target_ends", "type": "dict | None", "description": "{'first': [x, y, z], 'last': [x, y, z]} (either may be left out), m."},
        {"name": "tol_m", "type": "float", "description": "Frechet distance tolerance, m. Default 0.03."},
        {"name": "ends_tol_m", "type": "float", "description": "End position tolerance, m. Default 0.03."},
        {"name": "ordered", "type": "bool", "description": "The rope's first segment must match the target's first point. Default false."},
        {"name": "planar", "type": "bool", "description": "Compare x, y only. Default false."},
        {"name": "frame", "type": "int", "description": "Frame to judge. Default -1 (the last)."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; frame; segments; frechet_m; end_errors_m; ends_swapped."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "rope", "deformable", "shape", "task gate"],
    "used_in": ["Knot the Rope"],
    "status": "tested",
    "entrypoint": "deformable_shape_gate",
}

from tool_library.common import task_gates as G


def deformable_shape_gate(scene, trajectory, rope=None, target_points=None, target_ends=None, tol_m=0.03, ends_tol_m=0.03,
                          ordered=False, planar=False, frame=-1, out_json=None):
    if target_points is None and target_ends is None:
        raise ValueError("target_points / target_ends: give a target centreline or end positions")
    if target_ends is not None and not (isinstance(target_ends, dict) and set(target_ends) <= {"first", "last"} and target_ends):
        raise ValueError("target_ends: {'first': [x, y, z], 'last': [x, y, z]}")
    ep, t, _ = G.open_window(scene, trajectory, frame=frame)
    key, bodies = G.cable_bodies(ep, rope)
    rep = G.gate_deformable_shape(ep, bodies, t, target_points=target_points, target_ends=target_ends, tol_m=float(tol_m),
                                  ends_tol_m=float(ends_tol_m), ordered=bool(ordered), planar=bool(planar), names=(key,))
    rep["rope"] = key
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(deformable_shape_gate(sys.argv[1], sys.argv[2], target_points=json.loads(sys.argv[3])), indent=1))
