"""Camera hand overlay check: the robot hands at the setup's home pose, seen through the video camera, cover the
hands of frame 0."""
from __future__ import annotations

TOOL = {
    "name": "camera_hand_overlay_check",
    "title": "Camera hand overlay check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Renders the robot hands at the setup's home keyframe through the registered video camera and compares them with the hand masks of frame 0 (overall IoU and per-hand centroid offsets): the paper's hand-overlay check that the camera is registered to the footage, for recordings that start with the hands at home.",
    "description": (
        "The bench recordings start with both hands resting at the setup's home pose (keyframe 'home': palm down at "
        "the bottom corners of the ego frame), and the station positions and home pose were fitted to them, so the "
        "rendered home hands must land on the hands in the first frame when the camera is right; a camera registered "
        "to other footage or with a wrong pitch shows as an offset. The hands are rendered by segmentation from the "
        "scene (default: the setup's scene.xml) at its home keyframe (or keyframe 0) through camera.json, split into "
        "left and right; the hand mask of frame 0 (any segmenter, e.g. masks/hands.png) is split into connected "
        "components, each assigned to the nearer rendered hand. The overall IoU must be at least min_iou and each "
        "matched hand's centroid must be within max_offset_frac of the image diagonal and, unless it overlaps well (IoU 0.6), "
        "its rendered area must be within area_band of the frame's (a wrong focal length or height scales the hands). When the "
        "check applies, an empty hands mask fails (a wrong mask path or a missed segmentation). A recording that does not "
        "start at home cannot be judged: start_at_home=None applies the check only when "
        "camera.json says source 'setup' (the bench camera); footage from other sessions (source given, estimate, "
        "egodex) starts wherever the person's hands were and is reported as not applicable unless start_at_home=True."
    ),
    "inputs": [
        {"name": "camera", "type": "str | dict", "description": "camera.json path or dict."},
        {"name": "hands_mask", "type": "str | ndarray", "description": "Binary mask of the hands in frame 0 (masks/hands.png)."},
        {"name": "scene", "type": "str | None", "description": "Scene whose robot is rendered. Default: the setup's scene.xml."},
        {"name": "setup", "type": "str | None", "description": "Setup folder (home keyframe name). Default: the repository's setup/."},
        {"name": "start_at_home", "type": "bool | None", "description": "Whether frame 0 shows the hands at the home pose. Default None: true for source 'setup' cameras."},
        {"name": "min_iou", "type": "float", "description": "Smallest overall IoU. Default 0.4."},
        {"name": "area_band", "type": "list[float]", "description": "Accepted robot/frame hand area ratio per hand when its IoU is below 0.6 (a wrong focal length or height scales the hands). Default [0.5, 2.0]."},
        {"name": "max_offset_frac", "type": "float", "description": "Largest centroid offset per hand, as a fraction of the image diagonal. Default 0.06."},
        {"name": "min_area_px", "type": "int", "description": "Mask components smaller than this (at the camera resolution) are noise. Default 200."},
        {"name": "max_side", "type": "int", "description": "Render at most this many pixels on the long side (masks are compared at that size). Default 960."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; applicable; iou; hands [{side, robot_px, human_px, iou, centroid_offset_px, offset_frac}]; image [w, h]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "opencv-python", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "camera", "registration", "hands", "overlay", "home", "stage2"],
    "status": "tested",
    "entrypoint": "camera_hand_overlay_check",
}

import math

import mujoco
import numpy as np

from tool_library.common import rigcam as RC
from tool_library.common import scene_assert as SA


def camera_hand_overlay_check(camera, hands_mask, scene=None, setup=None, start_at_home=None, min_iou=0.4, max_offset_frac=0.06,
                              min_area_px=200, max_side=960, area_band=(0.5, 2.0)):
    cam = SA.load_camera(camera)
    W0, H0 = int(cam["width"]), int(cam["height"])
    s = min(1.0, float(max_side) / max(W0, H0))
    W, H = int(round(W0 * s)), int(round(H0 * s))
    camr = RC.scaled(cam, W, H) if s < 1.0 else cam
    human = SA.load_mask(hands_mask, (W, H))
    src = str(cam.get("source", "")).lower()
    applicable = bool(start_at_home) if start_at_home is not None else (src == "setup")
    base = {"applicable": applicable, "image": [W, H], "camera_source": src or None}
    if not human.any():
        if applicable:
            return SA.report(["the hands mask of frame 0 is empty although frame 0 should show the hands at home: check the "
                              "mask path and the hand segmentation (or pass start_at_home=False when the recording does not "
                              "start at home)"], iou=None, hands=[], **base)
        return SA.report([], note="frame 0 shows no hands: the overlay cannot be judged", iou=None, hands=[], **base)
    if not applicable:
        return SA.report([], note=f"camera source '{src or 'unknown'}' is not the setup's bench camera: frame 0 need not show the "
                                   f"hands at home (pass start_at_home=True to force)", iou=None, hands=[], **base)
    st = SA.load_setup(setup)
    model, err = SA.try_load_model(scene if scene is not None else str(st["scene_path"]))
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], **base)
    data = mujoco.MjData(model)
    home = ((st["cfg"] or {}).get("home") or {}).get("keyframe", "home")
    k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, home)
    if k < 0 and model.nkey:
        k = 0
    if k < 0:
        return SA.report(["the scene has no keyframe: the robot's home pose is undefined"], **base)
    mujoco.mj_resetDataKeyframe(model, data, k)
    mujoco.mj_forward(model, data)
    robot = SA.robot_body_ids(model, st["model"])
    hands = SA.hand_body_ids(model, robot)
    if not hands:
        return SA.report(["no robot hands (bodies robot_<side>_<side>_*) found in the scene"], **base)
    seg = SA.render_segmentation(model, data, camr)
    gb = np.where(seg >= 0, model.geom_bodyid[np.clip(seg, 0, model.ngeom - 1)], -1)
    rob = {side: np.isin(gb, list(b)) for side, b in hands.items()}
    import cv2
    n, lab, stats, cents = cv2.connectedComponentsWithStats(human.astype(np.uint8))
    min_a = max(1, int(min_area_px * s * s))
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_a]
    rc = {side: (np.array(np.nonzero(m)[::-1]).mean(axis=1) if m.any() else None) for side, m in rob.items()}
    assign = {side: np.zeros_like(human) for side in rob}
    for i in comps:
        c = cents[i]
        cand = [(np.linalg.norm(c - rc[sd]), sd) for sd in rob if rc[sd] is not None]
        if cand:
            assign[min(cand)[1]] |= lab == i
    allr = np.zeros_like(human)
    for m in rob.values():
        allr |= m
    hum = np.zeros_like(human)
    for i in comps:
        hum |= lab == i
    iou = float((allr & hum).sum() / max((allr | hum).sum(), 1))
    diag = math.hypot(W, H)
    rows, issues = [], []
    for side, m in rob.items():
        hm = assign[side]
        row = {"side": side, "robot_px": int(m.sum() / (s * s)), "human_px": int(hm.sum() / (s * s))}
        if m.any() and hm.any():
            c_r = np.array(np.nonzero(m)[::-1]).mean(axis=1); c_h = np.array(np.nonzero(hm)[::-1]).mean(axis=1)
            off = float(np.linalg.norm(c_r - c_h))
            row.update({"iou": round(float((m & hm).sum() / max((m | hm).sum(), 1)), 3), "centroid_offset_px": round(off / s, 1),
                        "offset_frac": round(off / diag, 4)})
            ratio = float(m.sum()) / max(float(hm.sum()), 1.0)
            row["area_ratio"] = round(ratio, 3)
            if row["iou"] < 0.6 and not area_band[0] <= ratio <= area_band[1]:
                issues.append(f"the {side} robot hand at home covers {ratio:.2f}x the area of the {side} hand in frame 0: the camera's focal "
                              f"length or height does not match this footage")
            if off / diag > max_offset_frac:
                issues.append(f"the {side} robot hand at home lands {off / s:.0f} px ({off / diag * 100:.1f}% of the diagonal) from the {side} hand "
                              f"in frame 0: the camera is not registered to this footage (check its pitch, height and focal length)")
        rows.append(row)
    if iou < min_iou:
        issues.append(f"the robot hands at home overlap the hands of frame 0 with IoU {iou:.2f} (< {min_iou}): re-register the camera "
                      f"(register_table_camera) against this video")
    return SA.report(issues, iou=round(iou, 3), hands=rows, **base)
