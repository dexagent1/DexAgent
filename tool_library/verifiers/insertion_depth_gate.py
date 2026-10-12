"""Insertion depth gate: how far A went into B along an axis (slot, port, socket)."""
from __future__ import annotations

TOOL = {
    "name": "insertion_depth_gate",
    "title": "Insertion depth gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Depth of A along an insertion axis past B's entry plane (a drive in its slot, a plug in its port, a battery "
               "in its cell) reaches a minimum depth or a minimum fraction of A's length; without an axis, A is seated "
               "inside B's envelope and touches it.",
    "description": (
        "The goal predicate behind 'the drive sits in the top slot', 'the plug tip is fully inside the port' and 'the tray "
        "front is flush with the stack'. axis is the insertion direction in B's root frame; the entry plane is "
        "perpendicular to it through `entry` (a point in B's frame), by default B's outer face on the side A comes from "
        "(the extent of B's collision geoms along -axis). A's leading point, centre of mass and trailing point along the "
        "axis come from exact support functions of its collision geoms. With min_depth_m the chosen point (leading, com "
        "or trailing) must be at least that far past the entry plane (a negative value allows a part that stops just "
        "short: flush within that much); otherwise min_fraction of A's length along the axis must be past it. Without an "
        "axis the gate checks that A's centre lies inside B's collision extent in B's frame (more than half inserted "
        "along whichever axis it entered) and that A touches B; table contact is reported but allowed, since a slot with "
        "an open floor lets the part rest on the table."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The inserted body A."},
        {"name": "receiver", "type": "str", "description": "The body B it goes into (holder, case, socket)."},
        {"name": "axis", "type": "list[float] | None", "description": "Insertion direction in B's frame; None for the envelope test."},
        {"name": "entry", "type": "list[float] | None", "description": "A point of the entry plane in B's frame, m. Default B's outer face."},
        {"name": "min_depth_m", "type": "float | None", "description": "Required depth of `point` past the entry plane, m."},
        {"name": "min_fraction", "type": "float", "description": "Required fraction of A's length past the plane when no min_depth_m. Default 0.5."},
        {"name": "point", "type": "str", "description": "'leading', 'com' or 'trailing'. Default leading."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last]; decided on the last frame."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "frame", "type": "int | None", "description": "Decide on this frame instead."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; mode ('axis' | 'envelope'); depth_m; inserted_fraction; length_along_axis_m; axis_world; com_in_B, B_lo, B_hi, depth_inside_m in envelope mode; touches_B; touches_table."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "insertion", "slot", "task gate"],
    "used_in": ["Battery Insertion", "Computer Installation"],
    "status": "tested",
    "entrypoint": "insertion_depth_gate",
}

from tool_library.common import task_gates as G


def insertion_depth_gate(scene, trajectory, body, receiver, axis=None, entry=None, min_depth_m=None, min_fraction=0.5,
                         point="leading", frames=None, subgoal=None, frame=None, out_json=None):
    if point not in ("leading", "com", "trailing"):
        raise ValueError("point: 'leading', 'com' or 'trailing'")
    if axis is not None and (len(axis) != 3 or sum(float(a) ** 2 for a in axis) < 1e-12):
        raise ValueError("axis: a non-zero 3-vector in the receiver's frame")
    if entry is not None and axis is None:
        raise ValueError("entry: needs an axis")
    if not 0.0 < float(min_fraction) <= 1.0:
        raise ValueError("min_fraction: in (0, 1]")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal, frame)
    A = ep.bodies(body, "body")
    B = ep.bodies(receiver, "receiver") - A
    if not B:
        raise ValueError("receiver: it lies inside body's subtree")
    rep = G.gate_insertion_depth(ep, A, B, t1, axis=axis, entry=entry, min_depth_m=min_depth_m, min_fraction=float(min_fraction),
                                 point=point, names=(str(body), str(receiver)))
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(insertion_depth_gate(*sys.argv[1:5]), indent=1))
