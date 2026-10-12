"""Held-through-window gate: the object stays in the hand for the whole window (no drop, no table support)."""
from __future__ import annotations

TOOL = {
    "name": "held_through_window_gate",
    "title": "Held through window gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "On every frame of a window the object is held: touched by the hand on at least two links with opposing "
               "contact normals and not supported by the table ('in_hand'), or touched by the hand and not moving "
               "('fixed', a steadying hold).",
    "description": (
        "The goal predicate behind 'the battery is held in the right fingers' and 'the bottle is carried over the mug'. "
        "Each frame of the window (every stride-th) is loaded in the execution model, whose hand hulls are paired with "
        "every object, and collision detection runs with a 1 mm margin on the hand and the object. A frame counts as "
        "held in mode 'in_hand' when hand links touch the object on at least min_links distinct bodies, two of the "
        "contact normals are at least opposing_deg apart (a pinch, a wrap, two hands squeezing a lid) and the object does "
        "not touch the table (allow_table for a rope whose rest still lies on the cloth); in mode 'fixed' when any hand "
        "link touches it, and over the window the object moves less than stay_tol_m / stay_tol_deg. The window passes "
        "when at least min_frac of the frames are held, the longest run of unheld frames is at most max_gap_s (a drop "
        "and re-catch fails) and the last frame is held. The hand is a side ('left', 'right'; both when None) or an "
        "explicit list of body names or name prefixes."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The held object (its subtree)."},
        {"name": "side", "type": "str | None", "description": "'left', 'right', or None for either hand."},
        {"name": "hand_bodies", "type": "list[str] | None", "description": "Body names or prefixes that make up the hand, instead of side."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "mode", "type": "str", "description": "'in_hand' or 'fixed'. Default in_hand."},
        {"name": "min_frac", "type": "float", "description": "Fraction of frames that must be held. Default 0.9."},
        {"name": "max_gap_s", "type": "float", "description": "Longest allowed run of unheld frames, s. Default 0.25."},
        {"name": "opposing_deg", "type": "float", "description": "Angle two contact normals must span. Default 100."},
        {"name": "min_links", "type": "int", "description": "Distinct hand links touching. Default 2."},
        {"name": "allow_table", "type": "bool", "description": "Allow table contact while held (a rope). Default false."},
        {"name": "stay_tol_m", "type": "float", "description": "Allowed motion in 'fixed' mode, m. Default 0.01."},
        {"name": "stay_tol_deg", "type": "float", "description": "Allowed rotation in 'fixed' mode, deg. Default 8."},
        {"name": "stride", "type": "int", "description": "Frame step. Default 1."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; mode; frames; held_fraction; longest_unheld_s; longest_unheld_from; end_links; end_opposing; end_on_table; moved_m and rotated_deg in fixed mode."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "grasp", "carry", "task gate"],
    "status": "tested",
    "entrypoint": "held_through_window_gate",
}

from tool_library.common import task_gates as G


def held_through_window_gate(scene, trajectory, body, side=None, hand_bodies=None, frames=None, subgoal=None, mode="in_hand",
                             min_frac=0.9, max_gap_s=0.25, opposing_deg=100.0, min_links=2, allow_table=False,
                             stay_tol_m=0.01, stay_tol_deg=8.0, stride=1, out_json=None):
    if mode not in ("in_hand", "fixed"):
        raise ValueError("mode: 'in_hand' or 'fixed'")
    if side not in (None, "left", "right", "both", "any"):
        raise ValueError("side: 'left', 'right' or None")
    if not 0.0 < float(min_frac) <= 1.0:
        raise ValueError("min_frac: in (0, 1]")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    A = ep.bodies(body, "body")
    H = ep.hand_set(side, hand_bodies) - A
    rep = G.gate_held(ep, A, t0, t1, H, mode=mode, min_frac=float(min_frac), max_gap_s=float(max_gap_s),
                      opposing_deg=float(opposing_deg), min_links=int(min_links), allow_table=bool(allow_table),
                      stay_tol_m=float(stay_tol_m), stay_tol_deg=float(stay_tol_deg), stride=int(stride), names=(str(body),))
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(held_through_window_gate(sys.argv[1], sys.argv[2], sys.argv[3], frames=[int(sys.argv[4]), int(sys.argv[5])]), indent=1))
