"""Stress test for a wipe / scrub / slide: a flat hand presses a free pad onto its support and drags it along."""
from __future__ import annotations

TOOL = {
    "name": "flat_press_drag_test",
    "title": "Flat press-and-drag test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Lay a posed hand on a free object (sponge, cloth pad, eraser, card), press it down onto its support and drag the wrist sideways in physics; pass when the object slid along with the hand, stayed flat on the support and stayed under the pads.",
    "description": (
        "For wiping and scrubbing motions, where the hand does not close around the object but holds it against the "
        "table by friction. The hand is teleported above the pose along its approach axis with its posed fingers, "
        "descended onto the pose, pushed a further press_m into the object along its contact normal as the executor's wrist "
        "press goes (the pose's mean normal; the approach axis when it records none; default: the pose's "
        "press_travel_m when positive, else 4 mm) and then translated by travel_m along a horizontal direction in steps, "
        "with the wrist and finger servos of the hand rig in contact physics. The object (scene grasp_body, else the free "
        "body of the last scene object) must move at least min_fraction of the wrist travel along the direction, must not "
        "rise more than max_lift_mm or tilt more than max_tilt_deg (a pad that flips or rides up on the fingers is no "
        "wipe), and a hand geom must still touch it at the end. A hand that never touches the object reports STALLED, "
        "a hand that skates over it SKATED, an object that tips or lifts TIPPED."
    ),
    "inputs": [
        {"name": "scene", "type": "dict | path", "description": "Hand scene {hand_xml, side, objects:[{name, xml, pos, quat|yaw_deg}], floor, grasp_body?} or the JSON file hand_scene_from_recon writes."},
        {"name": "pose", "type": "dict | path", "description": "Hand pose {pos, euler, finger_targets, press_travel_m?, approach?} (flat_hand_pose, pose_from_points), or its JSON file."},
        {"name": "direction", "type": "list[float]", "description": "World direction of the drag; its vertical part is dropped."},
        {"name": "travel_m", "type": "float", "description": "Wrist travel along the direction. Default 0.04."},
        {"name": "press_m", "type": "float | None", "description": "Extra wrist travel into the object after contact, along the pose's mean contact normal (else back along its approach). Default: pose press_travel_m if > 0, else 0.004."},
        {"name": "approach_m", "type": "float", "description": "Start height above the pose along its approach axis. Default 0.05."},
        {"name": "seconds", "type": "float", "description": "Duration of the drag. Default 1.5."},
        {"name": "min_fraction", "type": "float", "description": "Required object travel / wrist travel. Default 0.6."},
        {"name": "max_lift_mm", "type": "float", "description": "Largest allowed rise of the object. Default 5."},
        {"name": "max_tilt_deg", "type": "float", "description": "Largest allowed tilt of the object. Default 15."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, verdict, measured {object, travel_m, object_moved_m, fraction, lateral_m, max_lift_mm, max_tilt_deg, contact_start, contact_end, peak_wrist_force_n}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "stress test", "wipe", "scrub", "slide", "press", "flat hand"],
    "status": "tested",
    "entrypoint": "flat_press_drag_test",
}

import mujoco
import numpy as np

from tool_library.common import hand as H


def _object_body(model, scene):
    gb = scene.get("grasp_body")
    if gb:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gb)
        if bid >= 0:
            return bid
    names = [o.get("name", "") for o in scene.get("objects", [])]
    for prefix in reversed(names):
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                b = int(model.jnt_bodyid[j])
                if (model.body(b).name or "").startswith(prefix):
                    return b
    raise ValueError("no free object body found: give scene['grasp_body'] or put the pad last in scene['objects']")


def _subtree(model, b0):
    out, st = set(), [b0]
    while st:
        b = st.pop(); out.add(b); st += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
    return out


def flat_press_drag_test(scene, pose, direction, travel_m=0.04, press_m=None, approach_m=0.05, seconds=1.5,
                         min_fraction=0.6, max_lift_mm=5.0, max_tilt_deg=15.0):
    from pathlib import Path
    from tool_library.common import grasp_robust as GR
    # a hand-scene dict or the JSON file hand_scene_from_recon writes, a pose dict or its JSON file, as the other grasp
    # tools take them
    if isinstance(scene, (str, Path)):
        if not Path(scene).exists():
            raise ValueError(f"scene: {scene} does not exist")
        scene = H.load_scene_spec(scene)
    if not isinstance(scene, dict):
        raise ValueError("scene must be a hand-scene dict or its JSON file (hand_scene_from_recon output)")
    if isinstance(pose, (str, Path)):
        pose = GR.load_pose(pose)
    if not isinstance(pose, dict):
        raise ValueError("pose must be a pose dict or its JSON file")
    for k in ("pos", "euler", "finger_targets"):
        if k not in pose:
            raise ValueError(f"pose is missing {k!r}")
    if travel_m <= 0:
        raise ValueError("travel_m must be positive; reverse `direction` instead")
    d_in = np.asarray(direction, float).ravel()
    if d_in.size not in (2, 3):
        raise ValueError(f"direction must be [dx, dy] or [dx, dy, dz], got {direction!r}")
    dvec = np.r_[d_in[:2], 0.0]                          # the vertical part, if any, is dropped
    if np.linalg.norm(dvec) < 1e-6:
        raise ValueError("direction must have a horizontal part")
    dvec /= np.linalg.norm(dvec)
    model, data0, rig, ids = H.open_scene(scene)
    ob = _object_body(model, scene)
    obj_bodies = _subtree(model, ob)
    obj_gids = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in obj_bodies}
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos; d.qvel[:] = 0
    mujoco.mj_forward(model, d)
    fingers = {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range}
    pos = np.asarray(pose["pos"], float); eul = np.asarray(pose["euler"], float)
    a = H.approach_axis(model, d, rig, pose)
    if press_m is None:
        pt = float(pose.get("press_travel_m") or 0.0)
        press_m = pt if pt > 0 else 0.004
    dt = float(model.opt.timestep)
    peak = [0.0]

    def touching():
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 in rig.body_ids and c.geom2 in obj_gids) or (b2 in rig.body_ids and c.geom1 in obj_gids):
                return True
        return False

    def run(p, secs):
        for _ in range(max(1, int(secs / dt))):
            H.hold_ctrl(d, rig, p, eul, fingers); mujoco.mj_step(model, d)
            peak[0] = max(peak[0], float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("x", "y", "z")])))

    rig.teleport(d, pos + a * approach_m, eul); rig.set_fingers(d, fingers); mujoco.mj_forward(model, d)
    run(pos + a * approach_m, 0.2)
    H.ramp(model, d, rig, pos + a * approach_m, pos, eul, eul, fingers, fingers, 0.5)
    # the press goes into the surface along the pose's contact normal, as the executor's wrist press does
    # (hand.executor_clamp); a pose without agreeing normals presses back along its approach
    n_in = -a
    nrm = [np.asarray(v, float) / max(np.linalg.norm(v), 1e-9) for v in (pose.get("normals") or {}).values()]
    if nrm and np.linalg.norm(np.mean(nrm, axis=0)) > 0.5:
        n_in = -np.mean(nrm, axis=0) / np.linalg.norm(np.mean(nrm, axis=0))
    p_press = pos + n_in * float(press_m)
    H.ramp(model, d, rig, pos, p_press, eul, eul, fingers, fingers, 0.3)
    run(p_press, 0.3)
    mujoco.mj_forward(model, d)
    x0 = d.xpos[ob].copy(); R0 = d.xmat[ob].reshape(3, 3).copy()
    c0 = touching()
    n = max(1, int(np.ceil(travel_m / 0.004)))
    lift, tilt = 0.0, 0.0
    for i in range(n):
        p_new = p_press + dvec * travel_m * (i + 1) / n
        for _ in range(max(1, int(seconds / n / dt))):
            H.hold_ctrl(d, rig, p_new, eul, fingers); mujoco.mj_step(model, d)
            peak[0] = max(peak[0], float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("x", "y", "z")])))
            lift = max(lift, float(d.xpos[ob][2] - x0[2]))
            Rr = R0.T @ d.xmat[ob].reshape(3, 3)
            tilt = max(tilt, float(np.degrees(np.arccos(np.clip(Rr[2, 2], -1, 1)))))
    run(p_new, 0.2)
    disp = d.xpos[ob] - x0
    moved = float(disp @ dvec); lateral = float(np.linalg.norm(disp[:2] - moved * dvec[:2]))
    frac = moved / travel_m
    c1 = touching()
    issues = []
    if not np.all(np.isfinite(d.qpos)):
        verdict = "DIVERGED: the simulation blew up"; issues.append(verdict)
    elif not c0 and not c1 and frac < 0.1:
        verdict = f"STALLED: the hand never touched the object (moved {moved*1000:.0f} mm)"
        issues.append(verdict + "; lower the pose onto the object's top face or raise press_m")
    elif frac < min_fraction:
        verdict = f"SKATED: object moved {moved*1000:.0f} of {travel_m*1000:.0f} mm ({100*frac:.0f}%)"
        issues.append(verdict + "; press deeper (press_m / a lower contact point) or put more of the hand on the object")
    elif lift * 1000 > max_lift_mm or tilt > max_tilt_deg:
        verdict = f"TIPPED: object rose {lift*1000:.1f} mm and tilted {tilt:.1f} deg while moving"
        issues.append(verdict + "; press nearer the object's centre, flatten the hand, or drag slower")
    else:
        verdict = f"DRAGGED: object followed {moved*1000:.0f} of {travel_m*1000:.0f} mm ({100*frac:.0f}%), flat (lift {lift*1000:.1f} mm, tilt {tilt:.1f} deg)"
    if verdict.startswith("DRAGGED") and not c1:
        issues.append("the hand lost contact with the object at the end of the drag; press deeper")
    return {"ok": not issues, "issues": issues, "verdict": verdict, "measured": {
        "object": model.body(ob).name, "travel_m": float(travel_m), "press_m": float(press_m),
        "object_moved_m": round(moved, 4), "fraction": round(frac, 3), "lateral_m": round(lateral, 4),
        "max_lift_mm": round(lift * 1000, 2), "max_tilt_deg": round(tilt, 2), "contact_start": bool(c0), "contact_end": bool(c1),
        "peak_wrist_force_n": round(peak[0], 1)}}
