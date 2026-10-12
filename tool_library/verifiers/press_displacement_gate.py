"""Press displacement gate: a pressed part moved (or was seated) by at least d, optionally several times."""
from __future__ import annotations

TOOL = {
    "name": "press_displacement_gate",
    "title": "Press displacement gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A pressed part was displaced by at least min_m inside a window -- a slide joint's travel (a pump head, a "
               "button, a latch) or a body's motion along a direction (a brick pressed onto studs) -- through at least "
               "min_strokes separate presses.",
    "description": (
        "The goal predicate behind 'the pump head has been pressed through its stroke several times' and 'the brick is "
        "pressed onto the studs'. For a joint the displacement is |q - q(first frame)| on every frame of the window; for "
        "a body it is its motion along `direction` (world, default straight down), optionally measured in a reference "
        "body's frame so that a pressed part is judged against its base even when the whole assembly moves. peak "
        "judges the largest excursion (a button that springs back after the press); peak false judges the value at the "
        "window's end (a part that must stay seated). A stroke is an excursion that reaches min_m and returns below 30% "
        "of it (or the window ends); the gate fails when fewer than min_strokes strokes occurred, naming the count -- a "
        "single press where the goal says 'several times' is a miss."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "min_m", "type": "float", "description": "Required displacement, m (rad for a hinge)."},
        {"name": "joint", "type": "str | None", "description": "The pressed joint (slide or hinge)."},
        {"name": "body", "type": "str | None", "description": "The pressed body, instead of a joint."},
        {"name": "direction", "type": "list[float] | None", "description": "Press direction for a body, world. Default [0, 0, -1]."},
        {"name": "relative_to", "type": "str | None", "description": "Measure the body's motion in this body's frame."},
        {"name": "min_strokes", "type": "int", "description": "Separate presses required. Default 1."},
        {"name": "peak", "type": "bool", "description": "Judge the largest excursion (true) or the end value (false). Default true."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; source; displacement_m; end_displacement_m; required_m; strokes; required_strokes."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "press", "button", "pump", "task gate"],
    "status": "tested",
    "entrypoint": "press_displacement_gate",
}

from tool_library.common import task_gates as G


def press_displacement_gate(scene, trajectory, min_m, joint=None, body=None, direction=None, relative_to=None, min_strokes=1,
                            peak=True, frames=None, subgoal=None, out_json=None):
    if (joint is None) == (body is None):
        raise ValueError("joint / body: give exactly one")
    if float(min_m) <= 0:
        raise ValueError("min_m: must be positive")
    if int(min_strokes) < 1:
        raise ValueError("min_strokes: at least 1")
    if direction is not None and len(direction) != 3:
        raise ValueError("direction: a 3-vector")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    b = ep.body(body, "body") if body is not None else None
    r = ep.body(relative_to, "relative_to") if relative_to is not None else None
    rep = G.gate_press_displacement(ep, t0, t1, float(min_m), joint=joint, body=b, direction=direction, relative_to=r,
                                    min_strokes=int(min_strokes), peak=bool(peak), names=(str(joint or body),))
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(press_displacement_gate(sys.argv[1], sys.argv[2], float(sys.argv[3]), joint=sys.argv[4]), indent=1))
