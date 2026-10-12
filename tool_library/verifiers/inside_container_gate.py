"""Inside-container gate: A ends inside B's interior, resting on B's inner floor rather than its rim."""
from __future__ import annotations

TOOL = {
    "name": "inside_container_gate",
    "title": "Inside-container gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A's centre of mass is inside B's interior volume and A rests on B below its rim band (not on the rim, not on "
               "the table) at the end of a window; optionally A's whole footprint lies within B's.",
    "description": (
        "The goal predicate behind 'the cup stands on the inner bottom of the bowl' and 'the toy rests inside the tray'. "
        "On the decision frame B's frame is its root body's pose with the axis nearest to vertical taken as up; B's "
        "interior is the extent of B's collision geoms along those axes (exact support functions of boxes, cylinders, "
        "capsules, spheres, ellipsoids and mesh vertices). A passes when (1) its mass-weighted centre lies inside B's "
        "footprint (shrunk by xy_margin_m) and between B's floor and rim height (+ com_above_rim_m for a part taller "
        "than the container), (2) at least one A-B contact lies below the rim band (the top max(rim_band_min_m, "
        "rim_band_frac x depth) of B), so A rests on B's inner floor or walls and not balanced on the rim, and (3) A does "
        "not touch the table. entirely additionally requires A's footprint inside B's. A cup that ends on a bowl's rim "
        "fails (2) and usually (1); a toy dropped beside the tray fails (1) and (3)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The contained body A."},
        {"name": "container", "type": "str", "description": "The container body B (its subtree: a drawer's tray, a bowl)."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last]; decided on the last frame."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "frame", "type": "int | None", "description": "Decide on this frame instead."},
        {"name": "rim_band_frac", "type": "float", "description": "Rim band as a fraction of B's depth. Default 0.15."},
        {"name": "rim_band_min_m", "type": "float", "description": "Smallest rim band, m. Default 0.004."},
        {"name": "xy_margin_m", "type": "float", "description": "Shrink B's footprint by this much for the centre test, m. Default 0."},
        {"name": "com_above_rim_m", "type": "float", "description": "How far A's centre may sit above the rim, m. Default 0."},
        {"name": "entirely", "type": "bool", "description": "Require A's whole footprint inside B's. Default false."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; com_in_B; B_interior_lo/hi; com_inside_xy; com_inside_height; contacts_with_B; contacts_below_rim; rim_band_m; touches_table; entirely_inside_footprint."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "container", "placement", "task gate"],
    "used_in": ["Place Cup in Bowl", "Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "inside_container_gate",
}

from tool_library.common import task_gates as G


def inside_container_gate(scene, trajectory, body, container, frames=None, subgoal=None, frame=None, rim_band_frac=0.15,
                          rim_band_min_m=0.004, xy_margin_m=0.0, com_above_rim_m=0.0, entirely=False, out_json=None):
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal, frame)
    A = ep.bodies(body, "body")
    B = ep.bodies(container, "container") - A
    if not B:
        raise ValueError("container: it lies inside the contained body's subtree")
    rep = G.gate_inside_container(ep, A, B, t1, rim_band_frac=float(rim_band_frac), rim_band_min_m=float(rim_band_min_m),
                                  xy_margin_m=float(xy_margin_m), com_above_rim_m=float(com_above_rim_m),
                                  entirely=bool(entirely), names=(str(body), str(container)))
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(inside_container_gate(*sys.argv[1:5]), indent=1))
