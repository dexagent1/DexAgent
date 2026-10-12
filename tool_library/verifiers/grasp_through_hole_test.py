TOOL = {
    "name": "grasp_through_hole_test",
    "title": "Finger-through-hole test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Check that a finger really passes through a loop (scissors ring, handle hole, bag strap) rather than resting across it.",
    "description": "A finger that has gone through a hole has its tip on the far side of the hole plane and its knuckle on the near side, with the segment between them crossing the plane inside the hole's ellipse. A finger lying across the loop, or merely reaching into it, satisfies neither. The hole is given as its centre, normal and two semi-axes in the world frame; for a ring geom these can be read off its pose. Kinematic, run on the posed hand before any physics; the scissors task used it on both the thumb ring and the finger ring.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets}"},
        {"name": "finger", "type": "str", "description": "finger that must be through the hole"},
        {"name": "centre", "type": "xyz (m, world)", "description": "hole centre"},
        {"name": "normal", "type": "xyz", "description": "hole plane normal; the tip must end on the -normal side"},
        {"name": "semi_axes", "type": "[a, b] m", "description": "ellipse semi-axes of the opening"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "tip beyond, knuckle before, crossing inside"},
        {"name": "tip_beyond_mm / knuckle_before_mm / crossing_r", "type": "numbers", "description": "signed distances to the plane and the normalised crossing radius (<= 1 is inside)"},
    ],
    "status": "tested",
    "entrypoint": "grasp_through_hole_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["scissors", "ring", "hook", "geometry"],
    "used_in": ["Scissor Cutting"],
}


def grasp_through_hole_test(scene, pose, finger, centre, normal, semi_axes):
    import mujoco
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose)
    r = H.through_hole_check(model, d, rig, finger, centre, normal, semi_axes)
    r["finger"] = finger
    return r
