TOOL = {
    "name": "grasp_straddle_test",
    "title": "Pinch straddle test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Check geometrically that a pinch has the bar between its two pads, on opposite sides and below the centre line, rather than both pads resting on top of it.",
    "description": "Four conditions on the two fingertip sites of a posed hand relative to a bar (axis point, direction, radius): the pad-to-pad direction is roughly perpendicular to the axis (|cos| < 0.5; two pads lined up along a cord are not either side of it); the pads are on opposite sides of the axis in the cross-section plane (a sign test); the axis passes through the middle 70% of the pad segment within 2 mm, with each pad within a pad radius of the bar; and neither pad is above the bar's centre line, because the contact force runs from the pad centre toward the object centre and pads riding high only press it down (measured on a cord: pads 3 mm high held 7 N, the hand rose 50 mm, the cord stayed). Purely kinematic, so it runs before any physics.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, grip_fingers}; the first two grip fingers are the pads"},
        {"name": "bar", "type": "dict or str", "description": "{p0, p1, radius} in world metres, or a capsule/cylinder geom name; defaults to pose['bar'] when the pose came from handle_pinch_pose"},
        {"name": "fingers", "type": "list", "description": "the two pads (default: pose grip_fingers[:2] or thumb, index)"},
        {"name": "max_rise_mm", "type": "float", "description": "how far above the centre line a pad may sit (default 1)"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "all four conditions hold"},
        {"name": "along_axis_cos / opposite_sides / along_segment / axis_to_segment_mm / pad_to_axis_mm / pad_rise_mm / below_centre_line", "type": "numbers", "description": "each condition's measurement"},
    ],
    "status": "tested",
    "entrypoint": "grasp_straddle_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["pinch", "geometry", "bar"],
    "used_in": ["Place Toy in Drawer", "Knot the Rope", "Scissor Cutting"],
}


def grasp_straddle_test(scene, pose, bar=None, fingers=None, max_rise_mm=1.0):
    import mujoco
    import numpy as np
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    bar = bar if bar is not None else pose.get("bar")
    if bar is None:
        raise ValueError("bar is required (none stored in the pose)")
    if isinstance(bar, str):
        bar = H.bar_from_geom(model, data0, bar)
    p0, p1 = np.asarray(bar["p0"], float), np.asarray(bar["p1"], float)
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose)
    fins = tuple(fingers or (list(pose.get("grip_fingers") or [])[:2] or ["thumb", "index"]))
    # the axis point nearest the pads, so a long bar is judged where it is held
    tips = np.mean([d.site_xpos[rig.fingertip_sites[f]] for f in fins if f in rig.fingertip_sites], axis=0)
    t = p1 - p0
    L = np.linalg.norm(t) + 1e-12
    s = float(np.clip(np.dot(tips - p0, t / L) / L, 0.0, 1.0))
    r = H.straddle_check(model, d, rig, p0 + s * t, t, float(bar["radius"]), fingers=fins,
                         max_rise_m=max_rise_mm / 1000.0)
    r["fingers"] = list(fins)
    r["along_bar"] = round(s, 3)
    return r
