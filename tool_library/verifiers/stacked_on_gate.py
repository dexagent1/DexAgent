"""Stacked-on gate: A rests on B (or bridges several supports) at the end of a window."""
from __future__ import annotations

TOOL = {
    "name": "stacked_on_gate",
    "title": "Stacked-on gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A rests on B at the end of a window: supported by B's top (an A-B contact whose normal points up out of B), "
               "level within a tolerance, optionally yaw-aligned modulo a symmetry, and not supported by the table; several "
               "supports mean A bridges all of them.",
    "description": (
        "The goal predicate behind 'the pink brick rests on top of the sage green brick' and 'the cream brick bridges the "
        "two bricks'. On the decision frame (the window's last frame, or `frame`) the trajectory state is loaded in the "
        "execution model and collision detection runs with a 2 mm margin on A and each support. A is on a support when "
        "some contact between their subtrees has a normal pointing down from A into the support by at least normal_min "
        "(0.7, i.e. within 45 deg of vertical): A sits on the support's top, not against its side. Every support must "
        "carry A (bridging); A must not touch the table (unless allow_table); A's nearest body axis must be within "
        "level_tol_deg of vertical (a brick resting on a stud edge fails); with yaw_tol_deg, A's yaw relative to the first "
        "support, folded by yaw_sym_deg (90 for a square brick), must be within the tolerance. Two parts of one "
        "articulated object that the scene excludes from collision (a hinged lid on its box) are judged by the geometric "
        "gap of their primitive geoms and A's level. Fails with the gap to each support when A is elsewhere."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The stacked body A."},
        {"name": "support", "type": "str | list[str]", "description": "The supporting body B, or several bodies A must bridge."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last]; decided on the last frame."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "frame", "type": "int | None", "description": "Decide on this frame instead (negative counts from the end)."},
        {"name": "tol_m", "type": "float", "description": "Contact detection distance, m. Default 0.002."},
        {"name": "level_tol_deg", "type": "float", "description": "Allowed tilt of A from level. Default 10."},
        {"name": "normal_min", "type": "float", "description": "Minimum downward component of a supporting contact normal. Default 0.7."},
        {"name": "yaw_tol_deg", "type": "float | None", "description": "Yaw alignment tolerance to the first support; None skips. Default None."},
        {"name": "yaw_sym_deg", "type": "float", "description": "Rotational symmetry folded out of the yaw error. Default 90."},
        {"name": "allow_table", "type": "bool", "description": "Allow A to touch the table too. Default false."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; frame; supports {name: contacts, top_contacts, gap_m}; touches_table; level_err_deg; yaw_err_deg."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "stacking", "support", "task gate"],
    "status": "tested",
    "entrypoint": "stacked_on_gate",
}

from tool_library.common import task_gates as G


def stacked_on_gate(scene, trajectory, body, support, frames=None, subgoal=None, frame=None, tol_m=G.TOUCH_TOL_M,
                    level_tol_deg=10.0, normal_min=0.7, yaw_tol_deg=None, yaw_sym_deg=90.0, allow_table=False, out_json=None):
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal, frame)
    A = ep.bodies(body, "body")
    sups = support if isinstance(support, (list, tuple)) else [support]
    if not sups:
        raise ValueError("support: give at least one supporting body")
    Bs = []
    for s in sups:
        B = ep.bodies(s, "support") - A
        if not B:
            raise ValueError(f"support: {s!r} lies inside the stacked body's subtree")
        Bs.append(B)
    rep = G.gate_stacked_on(ep, A, Bs, t1, tol=float(tol_m), level_tol_deg=float(level_tol_deg), normal_min=float(normal_min),
                            allow_table=bool(allow_table), yaw_tol_deg=yaw_tol_deg, yaw_sym_deg=yaw_sym_deg,
                            names={"A": str(body), "B": [str(s) for s in sups]})
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(stacked_on_gate(*sys.argv[1:5]), indent=1))
