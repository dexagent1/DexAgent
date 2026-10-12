"""Lift clearance gate: the object is lifted clear of its support."""
from __future__ import annotations

TOOL = {
    "name": "lift_clearance_gate",
    "title": "Lift clearance gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "The object is lifted at least min_height_m clear of its support (the rise of its lowest point, or its "
               "distance to the support when the support is not below it) and no longer touches the support, at the end "
               "of the window or on some frame.",
    "description": (
        "The goal predicate behind 'held clear of the table'. The support is what the object touched when the window "
        "started (any non-robot body within 2 mm) -- or, when it was already off its support then, what it touched on "
        "the last frame it rested on something (a cap freed by the last turn of an unscrew) -- or the bodies given in "
        "`support`; with none found it is the table. On the decision frame (the window's last with at='end', any frame "
        "with at='any') the object passes when it does not touch the support and either its lowest point (exact support "
        "functions of its collision geoms) rose at least min_height_m above where it rested, or its distance to the "
        "support is at least min_height_m (a cap pulled off a bottle held on its side moves along the bottle, not up). "
        "The issue gives both numbers."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The lifted object."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window; the reference is its first frame."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "min_height_m", "type": "float", "description": "Clearance needed, m. Default 0.01."},
        {"name": "support", "type": "str | list[str] | None", "description": "Support bodies (or 'table') instead of the automatic one."},
        {"name": "at", "type": "str", "description": "'end' (the window's last frame) or 'any'. Default end."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; frame; rise_m; clearance_m; touches_support; support; start_lowest_z; reference_frame."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "lift", "grasp", "task gate"],
    "status": "tested",
    "entrypoint": "lift_clearance_gate",
}

from tool_library.common import task_gates as G


def lift_clearance_gate(scene, trajectory, body, frames=None, subgoal=None, min_height_m=0.01, support=None, at="end", out_json=None):
    if at not in ("end", "any"):
        raise ValueError("at: 'end' or 'any'")
    if float(min_height_m) <= 0:
        raise ValueError("min_height_m: must be positive")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    A = ep.bodies(body, "body")
    S = None if support is None else ep.bodies(support, "support") - A
    rep = G.gate_lift_clearance(ep, A, t0, t1, min_height_m=float(min_height_m), support=S, at=at, names=(str(body),))
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(lift_clearance_gate(sys.argv[1], sys.argv[2], sys.argv[3]), indent=1))
