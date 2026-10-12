"""Rope severed gate: a cuttable rope's join was switched off inside the window and the halves came apart."""
from __future__ import annotations

TOOL = {
    "name": "rope_severed_gate",
    "title": "Rope severed gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A cuttable rope's join equality was active when the window started and inactive at its end (the per-frame "
               "eq_active recorded in the trajectory), and the two sides of the join are at least min_gap_m apart.",
    "description": (
        "The goal predicate behind 'the rope is separated into two pieces at the cut point'. A cut is run-time state that "
        "no qpos holds: the executor switches the rope's join (a CONNECT equality between two segment bodies, "
        "construct_cuttable_rope) off when the scissors' blades close on it and records eq_active [T, neq] and eq_names "
        "in the trajectory. The gate reads the join's state just before the window (or eq_active_start / the model's "
        "eq_active0 for a window starting at frame 0) and on the window's last frame, names the frame the join switched "
        "off, and measures the distance between the join's two anchors at the end (outcome.join_end_gap): a join switched "
        "off whose halves still sit together did not separate. joins names the equalities to judge; by default every "
        "rope join (a CONNECT/WELD between two non-robot bodies) that went from whole to cut in the window, and none "
        "doing so fails. A trajectory without eq_active fails with the instruction to record it."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz with eq_active [T, neq] and eq_names."},
        {"name": "joins", "type": "str | list[str] | None", "description": "Join equality name(s); default every rope join cut in the window."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "min_gap_m", "type": "float", "description": "Separation of the cut ends at the window's end, m. Default 0.003."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; recorded; joins {name: whole_at_start, whole_at_end, cut_frame, end_gap_m}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "rope", "cut", "scissors", "task gate"],
    "used_in": ["Scissor Cutting"],
    "status": "tested",
    "entrypoint": "rope_severed_gate",
}

from tool_library.common import task_gates as G


def rope_severed_gate(scene, trajectory, joins=None, frames=None, subgoal=None, min_gap_m=0.003, out_json=None):
    if float(min_gap_m) < 0:
        raise ValueError("min_gap_m: must not be negative")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    if joins is not None:
        import mujoco
        for j in ([joins] if isinstance(joins, str) else joins):
            if mujoco.mj_name2id(ep.model, mujoco.mjtObj.mjOBJ_EQUALITY, str(j)) < 0:
                raise ValueError(f"joins: no equality {j!r} in the scene")
    rep = G.gate_rope_severed(ep, t0, t1, joins=joins, min_gap_m=float(min_gap_m))
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(rope_severed_gate(sys.argv[1], sys.argv[2]), indent=1))
