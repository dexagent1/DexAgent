TOOL = {
    "name": "finger_press_in_hand_test",
    "title": "Finger press in hand test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Hold an object in a posed hand, lift it (and optionally tilt it), then press a sprung part of it (a spray pump head, a trigger, a button) with one finger several times; pass when the part travels through its stroke on every press, springs back on every release and the object stays put in the hand.",
    "description": (
        "Physics rollout in the floating-hand scene. The hand comes in with its open shape (the pose's "
        "finger_targets_open, else every finger backed off by open_frac) from approach_m back along its retreat axis "
        "(the pose's approach, else back along the fingers, as the executor does), closes over close_s to the grip clamp "
        "(press_targets of the gripping fingers, the executor's clamp; the pressing finger stays on its resting "
        "targets), seats, lifts the object by lift_m and, when `tilt` = {axis, deg} is given, turns the wrist about that "
        "world axis through the object's grip point. Then `cycles` times: the pressing finger ramps to the pressed "
        "targets (pressed_pose's finger targets for that finger) over press_s and holds, the joint value is read; it "
        "ramps back and holds, the joint value is read. The object's position in the wrist frame is tracked from "
        "the end of the lift/tilt. Verdicts: DROPPED (the object did not rise with the hand or lost the grip), STALLED "
        "(a press moved the joint less than min_travel_frac of its range: the finger cannot push the spring, or its "
        "push goes into the grip), STUCK (a release left the joint more than return_frac of its range away from rest), "
        "SLIPPED (the object drifted more than drift_max_mm in the hand while pumping), PUMPED. The finger servos and "
        "wrist are force-capped by the hand scene's defaults, so a press that only works with more force fails."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | str", "description": "Hand scene {hand_xml, side, objects, floor, grasp_body?} or its JSON path; the held object is the last object (or grasp_body)."},
        {"name": "pose", "type": "dict | str", "description": "Hold pose {pos, euler, finger_targets, grip_fingers (without the pressing finger), finger_targets_open?} (finger_press_targets' hold)."},
        {"name": "pressed_pose", "type": "dict | str", "description": "Pose whose finger_targets hold the pressing finger's pressed joint targets (finger_press_targets' pressed)."},
        {"name": "joint", "type": "str", "description": "The pressed part's slide or hinge joint (full name or unique suffix)."},
        {"name": "finger", "type": "str | None", "description": "Pressing finger; default pressed_pose['press_finger'], else 'index'."},
        {"name": "cycles", "type": "int", "description": "Press-release cycles. Default 3."},
        {"name": "lift_m", "type": "float", "description": "Lift before pumping. Default 0.06; 0 keeps the object on its support (a button on a resting box)."},
        {"name": "tilt", "type": "dict | None", "description": "{axis: world [x, y, z], deg: angle} wrist turn after the lift (pumping a tilted sprayer). Default none."},
        {"name": "press_s, release_s, dwell_s", "type": "float", "description": "Finger ramp times and the dwell at each end. Defaults 0.4, 0.4, 0.3."},
        {"name": "min_travel_frac", "type": "float", "description": "Required joint travel per press as a fraction of its range. Default 0.5."},
        {"name": "return_frac", "type": "float", "description": "Allowed joint offset from rest after a release, as a fraction of its range. Default 0.25."},
        {"name": "drift_max_mm", "type": "float", "description": "Allowed object drift in the wrist frame while pumping. Default 8."},
        {"name": "approach_m", "type": "float", "description": "How far back along the retreat axis the open hand starts. Default 0.06; 0 places it open at the grasp."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, verdict, issues, measured {joint, range, rest, presses: [{travel_mm, frac}], releases: [{offset_mm}], drift_mm_max, lift_achieved_mm, tilt_achieved_deg, fingers_at_end, contact_force_N, press_finger_on_part}."},
    ],
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "grasp", "press", "pump", "trigger", "button", "in-hand", "stress-test"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "finger_press_in_hand_test",
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


def finger_press_in_hand_test(scene, pose, pressed_pose, joint, finger=None, cycles=3, lift_m=0.06, tilt=None,
                              press_s=0.4, release_s=0.4, dwell_s=0.3, min_travel_frac=0.5, return_frac=0.25,
                              drift_max_mm=8.0, close_s=0.6, open_frac=0.55, depth_rad=0.3, approach_m=0.06):
    from scipy.spatial.transform import Rotation as Rot
    from tool_library.common import hand as H
    pose = _load(pose); pressed_pose = _load(pressed_pose)
    for nm, p in (("pose", pose), ("pressed_pose", pressed_pose)):
        if "finger_targets" not in p:
            raise ValueError(f"{nm}: needs finger_targets")
    if "pos" not in pose or "euler" not in pose:
        raise ValueError("pose: needs pos and euler")
    if int(cycles) < 1:
        raise ValueError("cycles must be >= 1")
    finger = finger or pressed_pose.get("press_finger") or pose.get("press_finger") or "index"
    if finger not in H.FINGERS:
        raise ValueError(f"finger: one of {H.FINGERS}, got {finger!r}")
    if tilt is not None and (not isinstance(tilt, dict) or "axis" not in tilt or "deg" not in tilt):
        raise ValueError("tilt: give {axis: [x, y, z], deg: angle} or None")
    scene = H.load_scene_spec(scene)
    model, data0, rig, ids = H.open_scene(scene)
    jid = _find_joint(model, joint)
    if int(model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError(f"joint: {model.joint(jid).name!r} is not a slide or hinge joint")
    lo, hi = (float(v) for v in model.jnt_range[jid])
    span = max(hi - lo, 1e-9)
    qa = int(model.jnt_qposadr[jid])
    bid = H.grasp_body(model, scene, ids)
    part_gids = H.subtree_geoms(model, model.body(int(model.jnt_bodyid[jid])).name)
    closed = {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range}
    fkeys = [k for k in rig.finger_range if H.finger_of_joint(k) == finger]
    grip = [f for f in (pose.get("grip_fingers") or H.FINGERS) if f != finger]
    clamp = H.press_targets(rig, closed, grip, depth_rad=float(depth_rad),
                            squeezed={k: float(v) for k, v in (pose.get("finger_targets_squeezed") or {}).items() if k in rig.finger_range})
    for k in fkeys:
        if k in closed:
            clamp[k] = closed[k]
    pressed = dict(clamp)
    for k in fkeys:
        if k in pressed_pose["finger_targets"]:
            pressed[k] = float(pressed_pose["finger_targets"][k])
    if all(abs(pressed.get(k, 0.0) - clamp.get(k, 0.0)) < 1e-6 for k in fkeys):
        issues = [f"pressed_pose moves no joint of the {finger}: nothing would press the part"]
    else:
        issues = []
    opened = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range} or H.open_targets(rig, closed, float(open_frac))
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos; d.qvel[:] = 0
    pos = np.asarray(pose["pos"], float); eul = np.asarray(pose["euler"], float)
    # the open hand arrives as the executor brings it, from approach_m back along its retreat axis: teleported open onto
    # the grip, its opened fingers started inside a free sprayer and kicked it 30 mm away before anything closed
    a_ax = H.approach_axis(model, d, rig, pose)
    pre = pos + a_ax * float(approach_m)
    H.apply_pose(model, d, rig, {"pos": pre.tolist(), "euler": eul.tolist()}, opened)
    H.step_hold(model, d, rig, pre, eul, opened, 0.3)
    if approach_m > 0:
        H.ramp(model, d, rig, pre, pos, eul, eul, opened, opened, 0.5)
        H.step_hold(model, d, rig, pos, eul, opened, 0.1)
    q_rest = float(d.qpos[qa])
    H.ramp(model, d, rig, pos, pos, eul, eul, opened, clamp, float(close_s))
    H.step_hold(model, d, rig, pos, eul, clamp, 0.4)
    z0 = float(d.xpos[bid][2])
    up = pos + np.array([0.0, 0.0, float(lift_m)])
    H.ramp(model, d, rig, pos, up, eul, eul, clamp, clamp, 0.6)
    H.step_hold(model, d, rig, up, eul, clamp, 0.3)
    lift_ach = float(d.xpos[bid][2]) - z0
    cur_p, cur_e = up.copy(), eul.copy()
    tilt_ach = None
    if tilt is not None and abs(float(tilt["deg"])) > 0:
        ax = np.asarray(tilt["axis"], float); ax /= max(np.linalg.norm(ax), 1e-12)
        piv = np.asarray(d.xpos[bid], float).copy()
        R0 = H.euler_to_mat(cur_e); n = max(1, int(np.ceil(abs(float(tilt["deg"])) / 5.0)))
        zb0 = np.asarray(d.xmat[bid], float).reshape(3, 3)[:, 2].copy()
        for i in range(n):
            Rs = Rot.from_rotvec(np.radians(float(tilt["deg"])) * (i + 1) / n * ax).as_matrix()
            p1 = piv + Rs @ (up - piv); e1 = H.nearest_euler_branch(H.mat_to_euler(Rs @ R0), cur_e)
            H.ramp(model, d, rig, cur_p, p1, cur_e, e1, clamp, clamp, 1.0 / n)
            cur_p, cur_e = p1, e1
        H.step_hold(model, d, rig, cur_p, cur_e, clamp, 0.3)
        zb1 = np.asarray(d.xmat[bid], float).reshape(3, 3)[:, 2]
        tilt_ach = float(np.degrees(np.arccos(np.clip(zb0 @ zb1, -1, 1))))
    held0 = H.finger_contacts(model, d, rig, ids["grasp_gids"])
    ref = H.object_in_wrist(d, rig, bid)
    rest_now = float(d.qpos[qa])
    presses, releases, drift_max, on_part = [], [], 0.0, []
    for _ in range(int(cycles)):
        H.ramp(model, d, rig, cur_p, cur_p, cur_e, cur_e, clamp, pressed, float(press_s))
        H.step_hold(model, d, rig, cur_p, cur_e, pressed, float(dwell_s))
        qp = float(d.qpos[qa])
        presses.append({"travel_mm": round(abs(qp - rest_now) * 1000, 2), "frac": round(abs(qp - rest_now) / span, 3)})
        tip_on = False
        for i in range(d.ncon):
            c = d.contact[i]; g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if a in rig.geom_ids and b in part_gids and f"_{finger}_" in (model.geom(a).name or ""):
                    tip_on = True
        on_part.append(tip_on)
        drift_max = max(drift_max, float(np.linalg.norm(H.object_in_wrist(d, rig, bid) - ref)))
        H.ramp(model, d, rig, cur_p, cur_p, cur_e, cur_e, pressed, clamp, float(release_s))
        H.step_hold(model, d, rig, cur_p, cur_e, clamp, float(dwell_s))
        qr = float(d.qpos[qa])
        releases.append({"offset_mm": round(abs(qr - rest_now) * 1000, 2), "frac": round(abs(qr - rest_now) / span, 3)})
        drift_max = max(drift_max, float(np.linalg.norm(H.object_in_wrist(d, rig, bid) - ref)))
    held1 = H.finger_contacts(model, d, rig, ids["grasp_gids"])
    lifted = float(lift_m) <= 0 or lift_ach > 0.5 * float(lift_m)     # lift_m 0: the object stays on its support (a button on a box)
    verdict = "PUMPED"
    if not lifted or held1["n_finger"] < 2:
        verdict = "DROPPED"
        issues.append(f"the object rose {lift_ach * 1000:.0f} of {lift_m * 1000:.0f} mm and {held1['n_finger']} fingers hold it at the end: "
                      "the grip does not hold the object; choose a grasp that survives grasp_hold_test first")
    elif any(p["frac"] < float(min_travel_frac) for p in presses):
        verdict = "STALLED"
        worst = min(p["frac"] for p in presses)
        issues.append(f"a press moved the joint only {worst * 100:.0f}% of its range (need {min_travel_frac * 100:.0f}%): rest the {finger} "
                      "pad on the part's pressing face, press deeper (finger_press_targets overdrive_m) or use a finger with more travel")
    elif any(r["frac"] > float(return_frac) for r in releases):
        verdict = "STUCK"
        issues.append(f"after a release the joint stayed {max(r['frac'] for r in releases) * 100:.0f}% of its range from rest: "
                      f"the resting {finger} pad keeps it pressed; rest the finger lighter on the part")
    elif drift_max * 1000 > float(drift_max_mm):
        verdict = "SLIPPED"
        issues.append(f"the object drifted {drift_max * 1000:.1f} mm in the hand while pumping (limit {drift_max_mm} mm): "
                      "the press pushes it through the grip; grip higher on the body or with more fingers opposite the press")
    ok = verdict == "PUMPED" and not issues
    return {"ok": bool(ok), "verdict": verdict, "issues": issues,
            "measured": {"joint": model.joint(jid).name, "range": [lo, hi], "rest_before_grip": round(q_rest, 5),
                         "rest": round(rest_now, 5), "presses": presses, "releases": releases,
                         "drift_mm_max": round(drift_max * 1000, 2), "lift_achieved_mm": round(lift_ach * 1000, 1),
                         "tilt_achieved_deg": None if tilt_ach is None else round(tilt_ach, 1),
                         "fingers_before": held0["fingers"], "fingers_at_end": held1["fingers"],
                         "contact_force_N": held1["force_N"], "press_finger_on_part": on_part}}
