TOOL = {
    "name": "handle_pinch_pose",
    "title": "Pinch pose on a bar or handle",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Solve a thumb-versus-finger pinch that straddles a bar-shaped feature (handle, rod, cord segment, scissors ring tube) at a chosen point along it.",
    "description": "A bar is a segment with a radius: a drawer handle, a pen, a battery, a rope segment or the tube of a scissors ring. Given its two end points and radius (or the name of a capsule/cylinder geom to read them from), the pad points are placed on opposite sides of the axis at parameter s along it, across a pinch direction that defaults to horizontal-perpendicular to the axis (so the pads come in from the sides, not from above, which is the configuration that can lift). The fingertip IK solver then puts the thumb on one side and the index (plus the middle, offset along the bar) on the other, with seeds back along the approach direction, and the idle fingers are curled. Optionally tilts the pinch direction about the bar's tangent, which is how a rope was pinched from above-and-beside in the knot task. The result is checked with grasp_straddle_test, whose numbers are returned alongside the pose.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}; the bar belongs to the last object"},
        {"name": "bar", "type": "dict or str", "description": "{p0, p1, radius} in world metres, or the name of a capsule/cylinder geom in the compiled scene"},
        {"name": "s", "type": "float", "description": "where along the bar (0 = p0, 1 = p1) to pinch (default 0.5)"},
        {"name": "fingers", "type": "list", "description": "pinching fingers, thumb first (default thumb, index, middle)"},
        {"name": "pinch_dir", "type": "xyz or None", "description": "direction from the thumb side to the finger side across the bar (default: horizontal, perpendicular to the bar)"},
        {"name": "tilt_deg", "type": "float", "description": "rotate the pinch direction about the bar tangent by this angle (default 0)"},
        {"name": "finger_spread_m", "type": "float", "description": "spacing of index and middle along the bar (default 0.014)"},
        {"name": "approach_dirs", "type": "list of xyz", "description": "seed approach directions (default: +z and the thumb-side direction)"},
        {"name": "iters / yaw_step_deg / pitches_deg", "type": "numbers", "description": "IK budget and seed density"},
        {"name": "time_budget_s", "type": "float | None", "description": "Stop proposing further IK seeds after this duration, allowing the current seed to finish. Returns ik_search with seeds tried and whether the budget was exhausted. Physical verification is still required."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "{side, pos, euler, finger_targets, grip_fingers, residual_mm, worst_residual_mm, overlap_mm, palm_facing, bar, straddle: {...}}"},
    ],
    "status": "tested",
    "entrypoint": "handle_pinch_pose",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["pinch", "handle", "bar", "grasp"],
    "used_in": ["Place Toy in Drawer", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Scissor Cutting"],
}


def handle_pinch_pose(scene, bar, s=0.5, fingers=("thumb", "index", "middle"), pinch_dir=None,
                      tilt_deg=0.0, finger_spread_m=0.014, approach_dirs=None, iters=320,
                      yaw_step_deg=30, pitches_deg=(-60, -30, 0, 30, 60), time_budget_s=None):
    from tool_library.common import hand as H

    model, data0, rig, ids = H.open_scene(scene)
    if isinstance(bar, str):
        bar = H.bar_from_geom(model, data0, bar)
    return H.bar_pinch_pose(model, data0, rig, ids, bar, s=s, fingers=fingers, pinch_dir=pinch_dir,
                            tilt_deg=tilt_deg, finger_spread_m=finger_spread_m,
                            approach_dirs=approach_dirs, iters=iters, yaw_step_deg=yaw_step_deg,
                            pitches_deg=pitches_deg, time_budget_s=time_budget_s)
