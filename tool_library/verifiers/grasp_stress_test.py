TOOL = {
    "name": "grasp_stress_test",
    "title": "Grasp stress ladder",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Drive a posed hand through approach, close, lift, carry, turn and hold in one continuous physics rollout and fail at the first phase the object is nudged, dropped, slips or drifts.",
    "description": "Every phase goes through the hand's position actuators, never by teleporting onto the object. The open hand first settles back along its own retreat axis and descends to the pose (the object must not be nudged more than 20 mm); the fingers then ramp toward the press target and stop at the first configuration where the grip pads are on the object with more than 12 N (else the full press is used); after seating, the wrist lifts 60, 100 and 150 mm (each must raise the object more than half the commanded height), carries it 40 mm out and back along up, forward, back, left and right, turns 25 degrees about roll, pitch and yaw (object drift in the wrist frame below 30 mm), and holds still for 0.8 s (slip rate below 10 mm/s). Drift is measured in the wrist frame because a held object keeps a constant offset however far the hand travels. With maintain_N > 0 the grip fingers keep closing while the measured contact force is below that value, since a position servo bleeds off grip force as the object beds into the pads. Returns every phase's numbers and the name of the first failing phase. While the hand alone carries the object, the finger servos' reserve is judged as finger_servo_saturation_audit judges an episode (the smallest share of the finger torque caps with which the hand's contacts carry it, every executor frame): a grasp that holds only with a finger servo at 95% of its cap or more on 60% of the hold fails at 'servo_reserve' (a drive pinched 29 mm off its centre lifted with the thumb's abduction servo at 110% of its cap).",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}; the last object is the grasped one"},
        {"name": "pose", "type": "dict", "description": "{pos, euler, finger_targets, finger_targets_squeezed?, grip_fingers?} from the grasp skills"},
        {"name": "lift_heights_m", "type": "list", "description": "default (0.06, 0.10, 0.15)"},
        {"name": "shake_m / turn_deg / hold_s", "type": "numbers", "description": "carry amplitude (0.04), turn angle (25), hold duration (0.8)"},
        {"name": "nudge_max_mm / drift_max_mm / slip_max_mm_s", "type": "numbers", "description": "pass thresholds (20, 30, 10)"},
        {"name": "maintain_N", "type": "float", "description": "grip-force regulation target, 0 disables (default 0)"},
        {"name": "video_path", "type": "path", "description": "optional mp4 of the rollout"},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "every phase passed"},
        {"name": "first_fail", "type": "str or None", "description": "name of the first failing phase"},
        {"name": "phases", "type": "list of dicts", "description": "per-phase measurements: object_nudged_mm, contact_force_N, achieved_mm, drift_mm, slip_mm_s, ok"},
        {"name": "lifts / max_drift_mm / slip_mm_s / penetration_at_end_mm", "type": "numbers", "description": "summary measurements"},
    ],
    "status": "tested",
    "entrypoint": "grasp_stress_test",
    "requirements": ["mujoco>=3.1", "numpy", "imageio (video only)"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "stress-test", "lift", "slip"],
    "used_in": ["Place Cup in Bowl", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Open the Bottle", "Scissor Cutting"],
}


def grasp_stress_test(scene, pose, lift_heights_m=(0.06, 0.10, 0.15), shake_m=0.04, turn_deg=25.0,
                      hold_s=0.8, nudge_max_mm=20.0, drift_max_mm=30.0, slip_max_mm_s=10.0,
                      maintain_N=0.0, video_path=None):
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    record, finish = _recorder(model, video_path)
    r = H.stress_rollout(model, data0, rig, ids, pose, bid,
                         lift_heights=tuple(lift_heights_m), shake_m=shake_m, turn_deg=turn_deg,
                         hold_s=hold_s, nudge_max_m=nudge_max_mm / 1000.0,
                         drift_max_m=drift_max_mm / 1000.0, slip_max_mps=slip_max_mm_s / 1000.0,
                         maintain_N=maintain_N, record=record)
    r.pop("data", None)
    if finish:
        r["video"] = finish()
    return r


def _recorder(model, video_path, fps=30, every=None):
    """Frame grabber for the rollout videos (None when no path is given)."""
    if not video_path:
        return None, None
    import mujoco
    import numpy as np
    every = every or max(int(round(1.0 / (fps * model.opt.timestep))), 1)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.type = int(mujoco.mjtCamera.mjCAMERA_FREE)
    cam.lookat[:] = [0.0, 0.0, 0.12]
    cam.distance, cam.azimuth, cam.elevation = 0.7, 135, -20
    frames, count = [], {"n": 0}

    def record(m, d, label):
        count["n"] += 1
        if count["n"] % every:
            return
        renderer.update_scene(d, cam)
        frames.append(renderer.render().copy())

    def finish():
        import imageio
        if frames:
            imageio.mimwrite(video_path, np.asarray(frames), fps=fps, codec="libx264", quality=7)
        renderer.close()
        return video_path
    return record, finish
