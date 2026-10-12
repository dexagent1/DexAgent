"""Target-region gate: an object ends inside a table-frame region, optionally with an orientation tolerance."""
from __future__ import annotations

TOOL = {
    "name": "target_region_gate",
    "title": "Target region gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "An object ends inside a region of the table (a box, a polygon or a disc in the table frame), optionally "
               "within a height band and a yaw or upright tolerance, and stays there for the last hold_s of the window.",
    "description": (
        "The goal predicate behind 'the cap rests on the table to the right of the bottle' once the agent has written "
        "where 'right of the bottle' is. The table frame is the scene's world x, y with z measured from the tabletop "
        "(the top of the body named 'table', else 0). On the window's last frame the object's reference point (its body "
        "origin, or its mass-weighted centre with point='com') must lie inside the region polygon (even-odd test), its "
        "height above the table inside z_range when given, its yaw (body x axis about vertical) within yaw_tol_deg of "
        "yaw_target_deg folded by yaw_sym_deg, and its starting up axis within upright_tol_deg of vertical when given. "
        "With hold_s > 0 every frame of the last hold_s seconds must satisfy the same test. The issue names the distance "
        "to the region's centre."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The placed body."},
        {"name": "region", "type": "dict", "description": "{'box': [[xmin, ymin], [xmax, ymax]]} | {'polygon': [[x, y], ...]} | {'center': [x, y], 'radius': r}, metres in the table frame."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last]; decided at the last frame."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "z_range", "type": "list[float] | None", "description": "[min, max] height of the reference point above the tabletop, m."},
        {"name": "point", "type": "str", "description": "'origin' (body frame origin) or 'com'. Default origin."},
        {"name": "yaw_target_deg", "type": "float | None", "description": "Target yaw about vertical, deg."},
        {"name": "yaw_tol_deg", "type": "float | None", "description": "Yaw tolerance, deg. Default 15 when a target is given."},
        {"name": "yaw_sym_deg", "type": "float | None", "description": "Rotational symmetry folded out of the yaw error, deg."},
        {"name": "upright_tol_deg", "type": "float | None", "description": "Allowed tilt of the starting up axis, deg."},
        {"name": "hold_s", "type": "float", "description": "The object must stay in the region for the last hold_s seconds. Default 0."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; end {frame, xy, z_above_table, in_region, in_height, yaw_err_deg, tilt_deg, ok}; frames_evaluated."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "placement", "region", "task gate"],
    "status": "tested",
    "entrypoint": "target_region_gate",
}

from tool_library.common import task_gates as G


def target_region_gate(scene, trajectory, body, region, frames=None, subgoal=None, z_range=None, point="origin",
                       yaw_target_deg=None, yaw_tol_deg=None, yaw_sym_deg=None, upright_tol_deg=None, hold_s=0.0, out_json=None):
    if point not in ("origin", "com"):
        raise ValueError("point: 'origin' or 'com'")
    G.region_polygon(region)
    if z_range is not None and (len(z_range) != 2 or float(z_range[0]) > float(z_range[1])):
        raise ValueError("z_range: give [min, max]")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    A = ep.bodies(body, "body")
    rep = G.gate_target_region(ep, A, t0, t1, region, z_range=z_range, point=point, yaw_target_deg=yaw_target_deg,
                               yaw_tol_deg=yaw_tol_deg, yaw_sym_deg=yaw_sym_deg, upright_tol_deg=upright_tol_deg,
                               hold_s=float(hold_s), names=(str(body),))
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(target_region_gate(sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])), indent=1))
