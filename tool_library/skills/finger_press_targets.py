TOOL = {
    "name": "finger_press_targets",
    "title": "Finger press targets for a held object",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "From a grasp that holds an object with one finger resting on a sprung part (a spray pump head, a trigger, a push button), solve the finger joint targets that drive that finger's pad through the part's stroke along its joint, and write the hold, pressed and released poses the subgoal planner's hold motion executes as finger-only steps.",
    "description": (
        "The grasp pose (pose_from_points, generate_rigid_grasps ...) is placed in its hand scene with its closed finger "
        "targets. The pressing finger is taken out of the gripping fingers, so the executor's grip clamp "
        "(press_targets: the gripping fingers' flexion pushed depth_rad past touch) squeezes the body without driving "
        "the pressing finger into the part. The part's joint (slide or hinge) gives the press direction at the pad: "
        "along the slide axis, or along the hinge tangent at the pad, toward the joint limit farther from the current "
        "value. Damped-least-squares finger IK (only that finger's joints move, the wrist and the other fingers stay) "
        "drives the fingertip site to the pad position moved by stroke_m + overdrive_m along that direction. "
        "Three poses are written with the same wrist pose: `hold` (the grasp with the pressing finger out of the grip, "
        "its own open shape and press_depth_rad = depth_rad so the executor clamps exactly as the press was solved, every finger backed off without folding the pressing finger into the palm), "
        "`pressed` (finger_targets = the grip clamp with the pressing finger on its IK targets) and `released` "
        "(the grip clamp with the pressing finger back on its resting targets). Pressed and released carry "
        "finger_targets_open = the previous step's command and press_depth_rad 0, so a hold motion with approach_m 0 "
        "on them moves only the pressing finger. The IK is kinematic: the part does not move during the solve, the "
        "servo target simply lies past the part's rest so the finger pushes it; finger_press_in_hand_test proves the "
        "stroke in physics."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene (hand_scene_from_recon dict or its JSON path); the grasped object is the last object."},
        {"name": "pose", "type": "dict | str", "description": "Grasp pose {pos, euler, finger_targets, grip_fingers, ...} or its JSON path; the pressing finger rests on the part."},
        {"name": "joint", "type": "str", "description": "The part's slide or hinge joint (full name, or a unique suffix of it, e.g. 'pump_slide')."},
        {"name": "finger", "type": "str", "description": "Pressing finger: thumb | index | middle | ring | pinky. Default 'index'."},
        {"name": "stroke_m", "type": "float | None", "description": "Pad travel along the press direction; default the joint's range span (radians times the pad's lever for a hinge)."},
        {"name": "rest_offset_m", "type": "list | None", "description": "World shift [dx, dy, dz] of the pressing finger's resting pad before the press is solved (finger IK only), to move the pad off an edge or a protruding nozzle onto the middle of the pressing face. Default none."},
        {"name": "rest_point", "type": "list | None", "description": "World point [x, y, z] the pressing finger's tip site is moved to before the press is solved (finger IK only): a grasp that wraps every finger can free one onto the part's pressing face (e.g. the part's press site raised by the pad standoff). Exclusive with rest_offset_m. Default none."},
        {"name": "overdrive_m", "type": "float", "description": "Extra pad travel past the stroke so the servo keeps pushing at the stop. Default 0.003."},
        {"name": "depth_rad", "type": "float", "description": "Grip clamp depth of the gripping fingers (the executor's press_targets). Default 0.3."},
        {"name": "open_frac", "type": "float", "description": "How far the hold pose's own open shape backs every finger off. Default 0.55."},
        {"name": "iters", "type": "int", "description": "Finger IK iterations. Default 300."},
        {"name": "out_dir", "type": "str | None", "description": "Folder for hold.json, pressed.json, released.json (with `prefix`)."},
        {"name": "prefix", "type": "str", "description": "File name prefix in out_dir. Default 'press'."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "hold, pressed, released (pose dicts), paths {hold, pressed, released}, joint, finger, direction (world unit vector), stroke_m, pad_travel_m (achieved by the IK along the direction), residual_mm, grip_fingers, issues (e.g. the finger cannot reach the stroke)."},
    ],
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "grasp", "press", "pump", "trigger", "button", "finger", "in-hand"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "finger_press_targets",
}

import json
from pathlib import Path

import mujoco
import numpy as np


def _load(p):
    if isinstance(p, (str, Path)):
        return json.loads(Path(p).read_text())
    return dict(p)


def _find_joint(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
    if j >= 0:
        return j
    hits = [i for i in range(model.njnt) if (model.joint(i).name or "").endswith(str(name))]
    if len(hits) != 1:
        raise ValueError(f"joint: no unique joint named or ending with {name!r} ({len(hits)} matches)")
    return hits[0]


def finger_press_targets(scene, pose, joint, finger="index", stroke_m=None, overdrive_m=0.003, depth_rad=0.3,
                         open_frac=0.55, iters=300, out_dir=None, prefix="press", rest_offset_m=None, rest_point=None):
    from tool_library.common import hand as H
    pose = _load(pose)
    if "finger_targets" not in pose or "pos" not in pose or "euler" not in pose:
        raise ValueError("pose: needs pos, euler and finger_targets")
    if finger not in H.FINGERS:
        raise ValueError(f"finger: one of {H.FINGERS}, got {finger!r}")
    if overdrive_m < 0 or (stroke_m is not None and float(stroke_m) <= 0):
        raise ValueError("stroke_m must be > 0 and overdrive_m >= 0")
    model, data0, rig, ids = H.open_scene(scene)
    if finger not in rig.fingertip_sites:
        raise ValueError(f"finger: the hand has no fingertip site for {finger!r}")
    jid = _find_joint(model, joint)
    jtype = int(model.jnt_type[jid])
    if jtype not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError(f"joint: {model.joint(jid).name!r} is not a slide or hinge joint")
    if not bool(model.jnt_limited[jid]):
        raise ValueError(f"joint: {model.joint(jid).name!r} has no range; a press needs a stroke")
    lo, hi = (float(v) for v in model.jnt_range[jid])
    closed = {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range}
    grip = [f for f in (pose.get("grip_fingers") or [f_ for f_ in H.FINGERS if any(H.finger_of_joint(k) == f_ for k in closed)]) if f != finger]
    if not grip:
        raise ValueError("pose: no gripping finger is left once the pressing finger is taken out of the grip")
    squeezed = {k: float(v) for k, v in (pose.get("finger_targets_squeezed") or {}).items() if k in rig.finger_range}
    clamp = H.press_targets(rig, closed, grip, depth_rad=float(depth_rad), squeezed=squeezed)
    fkeys = [k for k in closed if H.finger_of_joint(k) == finger]
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose, closed)
    mujoco.mj_forward(model, d)
    rest_res = None
    if rest_offset_m is not None and rest_point is not None:
        raise ValueError("give rest_offset_m or rest_point, not both")
    if rest_offset_m is not None or rest_point is not None:
        t0 = np.asarray(d.site_xpos[rig.fingertip_sites[finger]], float).copy()
        goal0 = (t0 + np.asarray(rest_offset_m, float)) if rest_offset_m is not None else np.asarray(rest_point, float)
        if goal0.shape != (3,):
            raise ValueError("rest_offset_m / rest_point: give three numbers in metres")
        rest_res = H.finger_ik(model, d, rig, {finger: goal0}, iters=int(iters)).get(finger)
        for k in fkeys:
            closed[k] = float(d.qpos[rig.finger_qpos[k]])
    for k in fkeys:
        clamp[k] = closed[k]                       # the pressing finger rests where the grasp put it
    q = float(d.qpos[int(model.jnt_qposadr[jid])])
    sign = 1.0 if (hi - q) >= (q - lo) else -1.0
    body = int(model.jnt_bodyid[jid])
    R = d.xmat[body].reshape(3, 3)
    axis = R @ np.asarray(model.jnt_axis[jid], float)
    tip0 = np.asarray(d.site_xpos[rig.fingertip_sites[finger]], float).copy()
    if jtype == int(mujoco.mjtJoint.mjJNT_SLIDE):
        direction = sign * axis
        span = hi - lo
    else:
        anchor = np.asarray(d.xanchor[jid], float)
        t = np.cross(axis, tip0 - anchor)
        lever = float(np.linalg.norm(t))
        if lever < 1e-4:
            raise ValueError("joint: the pad lies on the hinge axis; pressing there turns nothing")
        direction = sign * t / lever
        span = (hi - lo) * lever
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    stroke = float(stroke_m) if stroke_m is not None else float(span)
    goal = tip0 + direction * (stroke + float(overdrive_m))
    res = H.finger_ik(model, d, rig, {finger: goal}, iters=int(iters))
    tip1 = np.asarray(d.site_xpos[rig.fingertip_sites[finger]], float)
    travel = float((tip1 - tip0) @ direction)
    pressed_f = {k: float(d.qpos[rig.finger_qpos[k]]) for k in fkeys}
    pressed_t = dict(clamp); pressed_t.update(pressed_f)
    issues = []
    if rest_res is not None and rest_res > 0.004:
        issues.append(f"the {finger} cannot rest where asked: its tip stays {rest_res * 1000:.1f} mm from the rest point; "
                      "choose a grasp that brings the finger's knuckle nearer the part")
    if travel < 0.5 * stroke:
        issues.append(f"the {finger} pad travels only {travel * 1000:.1f} mm of the {stroke * 1000:.1f} mm stroke along the joint; "
                      "rest the finger closer to the middle of its range or choose another finger")
    base = {k: v for k, v in pose.items() if k not in ("finger_targets", "finger_targets_open", "finger_targets_squeezed",
                                                      "grip_fingers", "press_depth_rad", "points", "normals", "targets")}
    opened = H.open_targets(rig, closed, float(open_frac))
    hold = dict(pose, finger_targets=dict(closed), grip_fingers=grip, finger_targets_open=opened, press_finger=finger,
                press_joint=model.joint(jid).name, press_depth_rad=float(depth_rad))
    pressed = dict(base, finger_targets=pressed_t, finger_targets_open=dict(clamp), grip_fingers=grip, press_depth_rad=0.0,
                   press_finger=finger, press_joint=model.joint(jid).name)
    released = dict(base, finger_targets=dict(clamp), finger_targets_open=dict(pressed_t), grip_fingers=grip, press_depth_rad=0.0,
                    press_finger=finger, press_joint=model.joint(jid).name)
    paths = {}
    if out_dir:
        od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
        for nm, p in (("hold", hold), ("pressed", pressed), ("released", released)):
            fp = od / f"{prefix}_{nm}.json"
            fp.write_text(json.dumps(p, indent=1, default=float))
            paths[nm] = str(fp)
    return {"hold": hold, "pressed": pressed, "released": released, "paths": paths, "joint": model.joint(jid).name,
            "finger": finger, "direction": [round(float(v), 4) for v in direction], "stroke_m": round(stroke, 4),
            "pad_travel_m": round(travel, 4), "residual_mm": round(float(res.get(finger, 0.0)) * 1000, 2),
            "grip_fingers": grip, "joint_value": round(q, 5),
            "rest_residual_mm": None if rest_res is None else round(float(rest_res) * 1000, 2), "issues": issues}
