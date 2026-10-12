"""Camera registration check: the video camera puts the table plane where the frame shows it."""
from __future__ import annotations

TOOL = {
    "name": "camera_registration_check",
    "title": "Camera registration check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Checks a registered video camera (camera.json): its intrinsics and extrinsics agree with each other, the scene's camera of the same name is the same camera, its height, pitch and roll are those of a camera over a table, and the table plane it implies lies under every object mask's contact line (below the horizon, in front of the camera, within range and on the table).",
    "description": (
        "Every Stage 2 placement is a back-projection through the video camera onto the table plane, so a camera "
        "registered to the wrong footage, with a stale focal length or with the scene's copy out of sync silently "
        "puts every object in the wrong place at the wrong size. The check reads camera.json (rigcam convention: fx, "
        "fy, cx, cy, R_cv, pos, quat_wxyz) and: (1) consistency - fovy matches fy and the height, fx = fy within 2%, "
        "the principal point within 10% of the image centre, R_cv a rotation, quat_wxyz the same rotation in MuJoCo's "
        "convention, and any recorded pitch/yaw/height_above_table equal to what R_cv and pos give; (2) the scene camera "
        "(scene_camera, default video_cam) has the same pose (pos_tol_m, angle_tol_deg) and fovy; (3) plausibility - "
        "height above the tabletop in height_band, optical axis pitched pitch_band below horizontal, |roll| at most "
        "max_roll_deg, and the optical axis meets the table within max_range_m; (4) with masks, each object mask's "
        "lowest pixels (where it stands on the table) lie below the table plane's horizon line, and back-projected "
        "onto z = 0 they land in front of the camera, within max_range_m and on the table (the scene's table geom, "
        "else setup.yaml's table) grown by table_margin_m. Masks may be a dict {id: path}, a list of paths or a "
        "folder (every PNG except those named hands*)."
    ),
    "inputs": [
        {"name": "camera", "type": "str | dict", "description": "camera.json path or dict."},
        {"name": "scene", "type": "str | mujoco.MjModel | None", "description": "Scene MJCF whose camera must equal camera.json and whose table bounds the contacts."},
        {"name": "scene_camera", "type": "str", "description": "Name of the scene's video camera. Default 'video_cam'."},
        {"name": "masks", "type": "dict | list | str | None", "description": "Object masks of frame 0: {id: png}, [png], or a masks/ folder."},
        {"name": "setup", "type": "str | None", "description": "Setup folder for the table extent when no scene is given. Default: the repository's setup/."},
        {"name": "height_band", "type": "list[float]", "description": "Camera height above the tabletop, metres. Default [0.15, 2.0]."},
        {"name": "pitch_band", "type": "list[float]", "description": "Optical axis below horizontal, degrees. Default [5, 89.9]."},
        {"name": "max_roll_deg", "type": "float", "description": "Largest roll about the optical axis. Default 20."},
        {"name": "max_range_m", "type": "float", "description": "Farthest table point a contact may back-project to. Default 3.0."},
        {"name": "pos_tol_m", "type": "float", "description": "Allowed scene-camera position difference. Default 0.005."},
        {"name": "angle_tol_deg", "type": "float", "description": "Allowed scene-camera orientation / fovy difference. Default 0.5."},
        {"name": "table_margin_m", "type": "float", "description": "Margin around the table for back-projected contacts. Default 0.05."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; camera {height_m, pitch_deg, yaw_deg, roll_deg, fovy_deg, axis_range_m}; consistency {...}; scene_camera {pos_diff_m, angle_diff_deg, fovy_diff_deg}; masks [{id, contact_v, horizon_v, contact_xy, range_m, on_table, verdict}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "camera", "registration", "table plane", "horizon", "stage2"],
    "status": "tested",
    "entrypoint": "camera_registration_check",
}

import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import rigcam as RC
from tool_library.common import scene_assert as SA


def _mask_items(masks):
    if masks is None:
        return []
    if isinstance(masks, dict):
        return [(str(k), v) for k, v in masks.items()]
    if isinstance(masks, (list, tuple)):
        return [(Path(str(p)).stem, p) for p in masks]
    d = Path(masks)
    if d.is_dir():
        return [(p.stem, p) for p in sorted(d.glob("*.png")) if not p.stem.lower().startswith("hand")]
    return [(d.stem, d)]


def _table_rect(model, setup_cfg):
    """(xmin, xmax, ymin, ymax) of the tabletop: the scene's table geom, else setup.yaml's table."""
    if model is not None:
        tb = SA.table_body_ids(model, setup_cfg)
        d = mujoco.MjData(model)
        mujoco.mj_kinematics(model, d)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) in tb and SA.geom_collidable(model, g) and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                P = SA.geom_world_points(model, d, g)
                return float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max())
    t = (setup_cfg or {}).get("table") or {}
    if "center" in t and "half_extents" in t:
        c, h = t["center"], t["half_extents"]
        return c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1]
    return None


def camera_registration_check(camera, scene=None, scene_camera="video_cam", masks=None, setup=None, height_band=(0.15, 2.0),
                              pitch_band=(5.0, 89.9), max_roll_deg=20.0, max_range_m=3.0, pos_tol_m=0.005, angle_tol_deg=0.5,
                              table_margin_m=0.05):
    issues = []
    try:
        cam = SA.load_camera(camera)
    except (ValueError, FileNotFoundError, KeyError) as e:
        return SA.report([f"camera.json cannot be read: {e}"])
    for k in ("width", "height", "fx", "fy", "cx", "cy", "R_cv", "pos"):
        if k not in cam:
            return SA.report([f"camera.json lacks '{k}': write it with register_table_camera"])
    W, H = int(cam["width"]), int(cam["height"])
    fx, fy, cx, cy = (float(cam[k]) for k in ("fx", "fy", "cx", "cy"))
    R = np.asarray(cam["R_cv"], float).reshape(3, 3)
    pos = np.asarray(cam["pos"], float)
    cons = {}
    fovy_f = math.degrees(2 * math.atan(0.5 * H / fy))
    cons["fovy_from_fy"] = round(fovy_f, 4)
    if "fovy_deg" in cam and abs(float(cam["fovy_deg"]) - fovy_f) > 0.1:
        issues.append(f"camera.json fovy_deg {cam['fovy_deg']:.3f} disagrees with fy {fy:.1f} px at height {H} px ({fovy_f:.3f} deg); "
                      f"regenerate camera.json from one focal length")
    if abs(fx - fy) / fy > 0.02:
        issues.append(f"camera.json fx {fx:.1f} and fy {fy:.1f} differ by more than 2%: the video's pixels are square; use one focal length")
    if abs(cx - W / 2) > 0.1 * W or abs(cy - H / 2) > 0.1 * H:
        issues.append(f"camera.json principal point ({cx:.0f}, {cy:.0f}) is far from the image centre ({W / 2:.0f}, {H / 2:.0f}); check the resolution it was written for")
    orth = float(np.abs(R.T @ R - np.eye(3)).max())
    cons["R_orthonormal_err"] = round(orth, 8)
    if orth > 1e-4 or np.linalg.det(R) < 0:
        issues.append(f"camera.json R_cv is not a rotation (orthonormality error {orth:.2g}, det {np.linalg.det(R):.3f}); rebuild it")
        return SA.report(issues, consistency=cons)
    if "quat_wxyz" in cam:
        q = Rot.from_matrix(R @ RC.CV_TO_MJ).as_quat()
        a = SA.quat_angle_deg([q[3], q[0], q[1], q[2]], cam["quat_wxyz"])
        cons["quat_vs_R_deg"] = round(a, 4)
        if a > 0.1:
            issues.append(f"camera.json quat_wxyz and R_cv disagree by {a:.2f} deg: the MuJoCo camera and the measurements use different poses; regenerate it")
    fwd = R[:, 2]
    pitch = math.degrees(math.asin(np.clip(-fwd[2], -1, 1)))
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    right = R[:, 0]
    fwd_h = np.array([fwd[0], fwd[1], 0.0]); fwd_h /= max(np.linalg.norm(fwd_h), 1e-9)
    left = np.array([-fwd_h[1], fwd_h[0], 0.0])
    roll = math.degrees(math.atan2(float(right @ np.array([0, 0, 1.0])), float(right @ (-left))))
    height = float(pos[2])
    for k, v, tol in (("pitch_deg", pitch, 0.5), ("yaw_deg", yaw, 0.5), ("height_above_table_m", height, 0.002)):
        if k in cam and isinstance(cam[k], (int, float)):
            dv = float(cam[k]) - v
            if k == "yaw_deg":
                dv = (dv + 180) % 360 - 180
            if abs(dv) > tol:
                issues.append(f"camera.json {k} {cam[k]:.3f} disagrees with its R_cv/pos ({v:.3f}); regenerate camera.json")
    axis_range = height / math.sin(math.radians(pitch)) if pitch > 0 else float("inf")
    info = {"height_m": round(height, 4), "pitch_deg": round(pitch, 3), "yaw_deg": round(yaw, 3), "roll_deg": round(roll, 3),
            "fovy_deg": round(fovy_f, 3), "axis_range_m": round(axis_range, 3) if math.isfinite(axis_range) else None}
    if not height_band[0] <= height <= height_band[1]:
        issues.append(f"the camera stands {height:.3f} m above the tabletop, outside {list(height_band)}: the table plane is misplaced "
                      f"(check the height/scale of the registration)")
    if not pitch_band[0] <= pitch <= pitch_band[1]:
        issues.append(f"the camera's optical axis is pitched {pitch:.1f} deg below horizontal, outside {list(pitch_band)}: it does not look down at a table")
    if abs(roll) > max_roll_deg:
        issues.append(f"the camera is rolled {roll:.1f} deg about its optical axis (> {max_roll_deg}): check the registration's up direction")
    if math.isfinite(axis_range) and axis_range > max_range_m:
        issues.append(f"the optical axis meets the table {axis_range:.2f} m away (> {max_range_m} m): the pitch or height is wrong")
    # the scene's copy of the camera
    sc = {}
    model = None
    if scene is not None:
        model, err = SA.try_load_model(scene)
        if model is None:
            issues.append(f"the scene does not compile: {err}")
        else:
            pose = SA.scene_camera_pose(model, scene_camera)
            if pose is None:
                issues.append(f"the scene has no camera '{scene_camera}'; add camera.json's camera to it (rigcam.mujoco_camera_xml)")
            else:
                p_s, R_s, fovy_s = pose
                Rm = R @ RC.CV_TO_MJ
                ang = math.degrees(math.acos(np.clip((np.trace(Rm.T @ R_s) - 1) / 2, -1, 1)))
                sc = {"pos_diff_m": round(float(np.linalg.norm(p_s - pos)), 5), "angle_diff_deg": round(ang, 4), "fovy_diff_deg": round(fovy_s - fovy_f, 4)}
                if sc["pos_diff_m"] > pos_tol_m or ang > angle_tol_deg or abs(fovy_s - fovy_f) > angle_tol_deg:
                    issues.append(f"the scene camera '{scene_camera}' is not camera.json's camera (position {sc['pos_diff_m'] * 1000:.1f} mm, "
                                  f"orientation {ang:.2f} deg, fovy {fovy_s - fovy_f:+.2f} deg apart): every silhouette check through it is "
                                  f"meaningless; write camera.json's camera into the scene")
    # contacts of the object masks
    rect = _table_rect(model, SA.load_setup(setup)["cfg"])
    rows = []
    for oid, mp in _mask_items(masks):
        try:
            m = SA.load_mask(mp, (W, H))
        except ValueError as e:
            raise ValueError(f"masks: {e}")
        cols = np.nonzero(m.any(axis=0))[0]
        if len(cols) == 0:
            rows.append({"id": oid, "verdict": "empty mask"})
            continue
        bottom = np.array([H - 1 - np.argmax(m[::-1, c]) for c in cols], float) + 0.5
        hv = np.array([SA.horizon_v(cam, c + 0.5) for c in cols])
        above = bottom <= hv + 2.0
        row = {"id": oid, "contact_v": round(float(bottom.max()), 1), "horizon_v": round(float(np.median(hv)), 1)}
        if above.all():
            row["verdict"] = "above the horizon"
            issues.append(f"object {oid}'s mask stands entirely above the table plane's horizon (row {row['horizon_v']:.0f}): the camera's "
                          f"pitch/roll cannot be right for this frame")
            rows.append(row)
            continue
        uv = np.c_[cols[~above] + 0.5, bottom[~above]]
        P = RC.pixel_on_plane(cam, uv, 0.0)
        rng = np.linalg.norm(P - pos, axis=1)
        c = np.median(P, axis=0)
        row.update({"contact_xy": SA.r6(c[:2], 4), "range_m": round(float(np.median(rng)), 3)})
        bad = []
        if float(np.median(rng)) > max_range_m:
            bad.append(f"its contact back-projects {np.median(rng):.2f} m from the camera (> {max_range_m})")
        if rect is not None:
            on = rect[0] - table_margin_m <= c[0] <= rect[1] + table_margin_m and rect[2] - table_margin_m <= c[1] <= rect[3] + table_margin_m
            row["on_table"] = bool(on)
            if not on:
                bad.append(f"its contact back-projects to ({c[0]:.3f}, {c[1]:.3f}) m, off the table x {rect[0]:.2f}..{rect[1]:.2f}, y {rect[2]:.2f}..{rect[3]:.2f}")
        row["verdict"] = "ok" if not bad else "off the table plane"
        if bad:
            issues.append(f"object {oid}: {'; '.join(bad)}: the camera does not put the table where the frame shows it")
        rows.append(row)
    return SA.report(issues, camera=info, consistency=cons, scene_camera=sc, masks=rows)
