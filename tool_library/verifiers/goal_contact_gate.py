"""Goal contact gate: two named bodies touch at the end of a window (and optionally throughout it)."""
from __future__ import annotations

TOOL = {
    "name": "goal_contact_gate",
    "title": "Goal contact gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Two named bodies (with everything attached below them) are in contact at the end of a window of an executed "
               "episode, optionally on a fraction of all its frames, or are apart (a goal 'touches', 'against', 'separated from').",
    "description": (
        "The goal predicate behind 'slide the brick until it touches the yellow brick', 'the sponge stays in contact with "
        "the table' and 'the cap is separated from the bottle'. The trajectory's qpos is replayed in the execution model "
        "of the scene it was executed in (hand hulls paired with every object, as the Stage 3 executor runs it); on the "
        "window's last frame every geom of the two subtrees is given a detection margin of tol_m, collision detection "
        "runs, and the pair touches when some contact between them is closer than tol_m (default 4 mm: reconstructed "
        "parts carry millimetre errors, while the misses this gate exists for are centimetres short). When they do not "
        "touch, the gap is measured up to 5 cm and reported, so a push that stopped 13 mm short says so. throughout "
        "requires contact on at least min_frac of the window's frames (every stride-th frame). apart inverts the test: "
        "no contact closer than apart_m at the end. Two parts of one articulated object that the scene excludes from "
        "collision (a lid and its box) are measured by the geometric distance of their primitive geoms instead. 'table' "
        "names the support surface. Correct episodes pass: a resting contact sits inside the margin on every frame; a "
        "displacement that stops short of the target body fails however far it moved."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz (qpos [T, nq], fps, labels, optional subgoal)."},
        {"name": "body_a", "type": "str", "description": "First body (its subtree counts)."},
        {"name": "body_b", "type": "str", "description": "Second body, or 'table'."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window; the end is the last frame."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames whose npz subgoal index matches."},
        {"name": "throughout", "type": "bool", "description": "Also require contact on min_frac of the window. Default false."},
        {"name": "min_frac", "type": "float", "description": "Fraction of frames for throughout. Default 1.0."},
        {"name": "tol_m", "type": "float", "description": "Contact distance that counts as touching, m. Default 0.004."},
        {"name": "apart", "type": "bool", "description": "Require the bodies to be apart at the end instead. Default false."},
        {"name": "apart_m", "type": "float | None", "description": "Smallest allowed distance for apart, m. Default tol_m."},
        {"name": "stride", "type": "int", "description": "Frame step for throughout. Default 2."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues (frame, gap, what to change); end_frame; touching_at_end; end_gap_m; touch_fraction and first_lost_frame for throughout; gap_m for apart."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "contact", "task gate"],
    "status": "tested",
    "entrypoint": "goal_contact_gate",
}

from tool_library.common import task_gates as G


def goal_contact_gate(scene, trajectory, body_a, body_b, frames=None, subgoal=None, throughout=False, min_frac=1.0,
                      tol_m=G.GOAL_TOUCH_TOL_M, apart=False, apart_m=None, stride=2, out_json=None):
    if float(tol_m) <= 0:
        raise ValueError("tol_m: must be positive")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    A = ep.bodies(body_a, "body_a")
    B = ep.bodies(body_b, "body_b") - A
    if not B:
        raise ValueError("body_b: it lies inside body_a's subtree")
    rep = G.gate_contact(ep, A, B, t0, t1, tol=float(tol_m), throughout=bool(throughout), min_frac=float(min_frac),
                         apart=bool(apart), apart_m=apart_m, stride=int(stride), names=(str(body_a), str(body_b)))
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(goal_contact_gate(*sys.argv[1:5]), indent=1))
