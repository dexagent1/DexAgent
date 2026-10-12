"""Joint target gate: an articulation joint reaches its target inside a window and holds it to the window's end."""
from __future__ import annotations

TOOL = {
    "name": "joint_target_gate",
    "title": "Joint target gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "An articulation joint (drawer slide, lid hinge, latch) reaches a target value, a fraction of its range, or "
               "its open / closed state inside a window and stays there until the window ends.",
    "description": (
        "The goal predicate behind 'the drawer is slid out far enough', 'the front panel is flush with the housing' and "
        "'the flaps are hooked under the rim and stay there'. The joint's qpos column is read on every frame of the "
        "window. The target is a value (target, rad or m), a fraction of the joint's range from its lower limit, or a "
        "state: 'closed' is the joint's closed value -- 0 when the range contains it (the library's constructors close "
        "at 0), else the limit nearest 0 -- within tol (default 10 deg for a hinge, 10 mm for a slide); 'open' means at "
        "least open_fraction (0.5) of the way from closed to the far limit. The gate reports the first frame the target "
        "is met; with hold it also fails when the joint leaves the target before the window ends (a latch that springs "
        "back, a lid that reopens after the hand lets go), naming the frame. A joint without limits needs an explicit "
        "target."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "joint", "type": "str", "description": "Hinge or slide joint name."},
        {"name": "target", "type": "float | None", "description": "Target value, rad (hinge) or m (slide)."},
        {"name": "fraction", "type": "float | None", "description": "Target as a fraction of the range from the lower limit."},
        {"name": "state", "type": "str | None", "description": "'open' or 'closed' instead of a value."},
        {"name": "tol", "type": "float | None", "description": "Tolerance around a target, rad or m. Default 10 deg / 0.01 m."},
        {"name": "open_fraction", "type": "float", "description": "How far from closed toward the far limit counts as open. Default 0.5."},
        {"name": "hold", "type": "bool", "description": "Require the target to hold to the window end. Default true."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; joint; unit; start; end; target; open_threshold; tol; range; reached_frame; lost_frame; closest."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "articulation", "joint", "task gate"],
    "used_in": ["Place Toy in Drawer", "Giftbox Packing"],
    "status": "tested",
    "entrypoint": "joint_target_gate",
}

from tool_library.common import task_gates as G


def joint_target_gate(scene, trajectory, joint, target=None, fraction=None, state=None, tol=None, open_fraction=0.5,
                      hold=True, frames=None, subgoal=None, out_json=None):
    given = [x is not None for x in (target, fraction, state)]
    if sum(given) != 1:
        raise ValueError("target / fraction / state: give exactly one")
    if state is not None and state not in ("open", "closed"):
        raise ValueError("state: 'open' or 'closed'")
    if fraction is not None and not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("fraction: between 0 and 1")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    rep = G.gate_joint_target(ep, joint, t0, t1, target=target, fraction=fraction, state=state, tol=tol,
                              open_fraction=float(open_fraction), hold=bool(hold))
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(joint_target_gate(sys.argv[1], sys.argv[2], sys.argv[3], state=sys.argv[4]), indent=1))
