"""Rotation turns gate: a part turned at least N degrees about its axis, wraps counted."""
from __future__ import annotations

TOOL = {
    "name": "rotation_turns_gate",
    "title": "Rotation turns gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A part turned at least min_deg about its axis inside a window, counting full wraps (a screw cap, a knob, a "
               "box turned a quarter turn on the table), optionally in a given direction and until a thread runs out.",
    "description": (
        "The goal predicate behind 'the thread is fully disengaged' and 'the box lies turned a quarter turn'. For a hinge "
        "joint the turn is its qpos travel over the window (hinge values are unbounded, so several turns count fully). "
        "For a body the rotation about `axis` (in the body frame; default its z) is integrated frame by frame from the "
        "incremental rotations, so wraps past 180 deg are counted and a back-and-forth twist cancels; relative_to "
        "measures it against another body (a cap against its bottle while the bottle itself turns). direction +1 / -1 "
        "requires the sign. With release_joint and release_travel_m (a construct_bottle cap: its lift joint and thread "
        "pitch x turns) the gate passes only when the lift joint reached the travel inside the window, i.e. the thread "
        "ran out and the cap is free, which is what 'fully unscrewed' means; the turned angle is reported."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "min_deg", "type": "float", "description": "Required turn, deg."},
        {"name": "joint", "type": "str | None", "description": "The turning hinge joint."},
        {"name": "body", "type": "str | None", "description": "The turning body, instead of a joint."},
        {"name": "axis", "type": "list[float] | None", "description": "Turning axis in the body frame. Default [0, 0, 1]."},
        {"name": "relative_to", "type": "str | None", "description": "Measure the body's turn relative to this body."},
        {"name": "direction", "type": "int | None", "description": "+1 or -1 to require the sign of the turn."},
        {"name": "release_joint", "type": "str | None", "description": "Lift joint that reaches release_travel_m when the thread runs out."},
        {"name": "release_travel_m", "type": "float | None", "description": "Lift travel at which the part is free, m."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; source; total_deg (signed); peak_deg; required_deg; turns; release_travel_m, release_needed_m, released."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "rotation", "unscrew", "task gate"],
    "used_in": ["Open the Bottle"],
    "status": "tested",
    "entrypoint": "rotation_turns_gate",
}

from tool_library.common import task_gates as G


def rotation_turns_gate(scene, trajectory, min_deg, joint=None, body=None, axis=None, relative_to=None, direction=None,
                        release_joint=None, release_travel_m=None, frames=None, subgoal=None, out_json=None):
    if (joint is None) == (body is None):
        raise ValueError("joint / body: give exactly one")
    if direction not in (None, 1, -1):
        raise ValueError("direction: +1, -1 or None")
    if (release_joint is None) != (release_travel_m is None):
        raise ValueError("release_joint / release_travel_m: give both or neither")
    if float(min_deg) <= 0:
        raise ValueError("min_deg: must be positive")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    b = ep.body(body, "body") if body is not None else None
    r = ep.body(relative_to, "relative_to") if relative_to is not None else None
    rep = G.gate_rotation_turns(ep, t0, t1, float(min_deg), joint=joint, body=b, axis=axis, relative_to=r, direction=direction,
                                release_joint=release_joint, release_travel_m=release_travel_m)
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(rotation_turns_gate(sys.argv[1], sys.argv[2], float(sys.argv[3]), joint=sys.argv[4]), indent=1))
