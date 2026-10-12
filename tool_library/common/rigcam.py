"""A pinhole camera in the rig frame (origin between the robot mounts on the tabletop, x right, y into
the work, z up): intrinsics as fx, fy, cx, cy in pixels; extrinsics as the OpenCV camera axes (x right,
y down, z forward) expressed in the rig frame plus the camera position. One camera.json serves the
MuJoCo camera, the Isaac camera, and every pixel <-> table-plane measurement.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])     # MuJoCo cameras look along -z with y up


def camera_dict(width, height, fx, fy, cx, cy, R_cv, pos, extra=None) -> dict:
    """Build the canonical camera dictionary. R_cv: 3x3 whose columns are the OpenCV camera axes in the rig frame."""
    R_cv = np.asarray(R_cv, float).reshape(3, 3)
    pos = [float(v) for v in np.asarray(pos, float).ravel()]
    q = Rot.from_matrix(R_cv @ CV_TO_MJ).as_quat()
    cam = {"width": int(width), "height": int(height), "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
           "fovy_deg": float(math.degrees(2 * math.atan(0.5 * int(height) / float(fy)))),
           "pos": pos, "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
           "R_cv": R_cv.tolist(), "frame": "rig: origin between the robot mounts on the tabletop, x right, y into the work, z up"}
    fwd = R_cv[:, 2]
    cam["pitch_deg"] = float(math.degrees(math.asin(np.clip(-fwd[2], -1, 1))))
    cam["yaw_deg"] = float(math.degrees(math.atan2(fwd[1], fwd[0])))
    cam["height_above_table_m"] = pos[2]
    if extra:
        cam.update(extra)
    return cam


def camera_from_fovy(width, height, fovy_deg, pos, R_cv, cx=None, cy=None, extra=None) -> dict:
    f = 0.5 * int(height) / math.tan(math.radians(float(fovy_deg)) / 2)
    return camera_dict(width, height, f, f, int(width) / 2 if cx is None else cx, int(height) / 2 if cy is None else cy, R_cv, pos, extra)


def R_cv_from_angles(yaw_deg=90.0, pitch_deg=35.0, roll_deg=0.0) -> np.ndarray:
    """Camera axes from the optical axis direction: yaw about rig z from +x (90 = looking along +y, into the
    work), pitch below horizontal, roll about the optical axis."""
    yaw, pit, r = (math.radians(float(v)) for v in (yaw_deg, pitch_deg, roll_deg))
    fwd = np.array([math.cos(yaw) * math.cos(pit), math.sin(yaw) * math.cos(pit), -math.sin(pit)])
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    if abs(r) > 1e-12:
        c, s = math.cos(r), math.sin(r)
        right, up = c * right + s * up, -s * right + c * up
    return np.stack([right, -up, fwd], axis=1)              # OpenCV: x right, y down, z forward


def camera_from_mujoco(width, height, fovy_deg, pos, quat_wxyz, extra=None) -> dict:
    w, x, y, z = quat_wxyz
    R_mj = Rot.from_quat([x, y, z, w]).as_matrix()
    return camera_from_fovy(width, height, fovy_deg, pos, R_mj @ CV_TO_MJ, extra=extra)


def load(path) -> dict:
    cam = json.loads(Path(path).read_text())
    cam["R_cv"] = np.asarray(cam["R_cv"], float).reshape(3, 3).tolist()
    return cam


def save(cam: dict, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cam, indent=1))
    return str(path)


def K(cam) -> np.ndarray:
    return np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])


def project(cam, P_rig):
    """Rig points [N, 3] -> pixels [N, 2] and depth along the optical axis [N]."""
    R = np.asarray(cam["R_cv"]); p = np.asarray(cam["pos"])
    Pc = (np.atleast_2d(np.asarray(P_rig, float)) - p) @ R
    z = Pc[:, 2]
    uv = np.stack([cam["cx"] + cam["fx"] * Pc[:, 0] / z, cam["cy"] + cam["fy"] * Pc[:, 1] / z], axis=1)
    return uv, z


def pixel_on_plane(cam, uv, z_plane=0.0):
    """Pixels [N, 2] -> rig points on the horizontal plane z = z_plane. Raises when a pixel looks above the horizon."""
    R = np.asarray(cam["R_cv"]); p = np.asarray(cam["pos"])
    uv = np.atleast_2d(np.asarray(uv, float))
    d_cam = np.stack([(uv[:, 0] - cam["cx"]) / cam["fx"], (uv[:, 1] - cam["cy"]) / cam["fy"], np.ones(len(uv))], axis=1)
    d = d_cam @ R.T
    s = (float(z_plane) - p[2]) / d[:, 2]
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("some pixels do not hit the table plane (they look above the horizon): check the camera pitch and height")
    return p + d * s[:, None]


def scaled(cam, width, height) -> dict:
    """The same camera at another image resolution."""
    sx, sy = int(width) / cam["width"], int(height) / cam["height"]
    return camera_dict(width, height, cam["fx"] * sx, cam["fy"] * sy, cam["cx"] * sx, cam["cy"] * sy, cam["R_cv"], cam["pos"],
                       {k: v for k, v in cam.items() if k not in ("width", "height", "fx", "fy", "cx", "cy", "fovy_deg", "pos", "quat_wxyz", "R_cv", "frame", "pitch_deg", "yaw_deg", "height_above_table_m")})


def mujoco_camera_xml(cam, name="ego") -> str:
    q = cam["quat_wxyz"]; p = cam["pos"]
    res = f' resolution="{int(cam["width"])} {int(cam["height"])}"' if cam.get("width") and cam.get("height") else ""   # its image size: renderers keep the aspect
    return f'<camera name="{name}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}" fovy="{cam["fovy_deg"]:.4f}"{res}/>'


def to_tablecam(cam) -> tuple[dict, dict]:
    """The equivalent common.tablecam camera (table frame: forward, left, up with the origin on the tabletop
    under the camera) and the rigid map from that frame back to the rig frame: rig = foot + R_tr @ table."""
    from tool_library.common import tablecam as TC
    R = np.asarray(cam["R_cv"]); p = np.asarray(cam["pos"])
    fwd = R[:, 2]
    fwd_h = np.array([fwd[0], fwd[1], 0.0]); fwd_h /= max(np.linalg.norm(fwd_h), 1e-9)
    left = np.array([-fwd_h[1], fwd_h[0], 0.0])
    up = np.array([0.0, 0.0, 1.0])
    R_tr = np.stack([fwd_h, left, up], axis=1)                          # rig <- table
    pitch = math.degrees(math.asin(np.clip(-fwd[2], -1, 1)))
    # roll: the image-right axis' component along table up, measured in the pitched frame
    right = R[:, 0]
    roll = math.degrees(math.atan2(float(right @ up), float(right @ (-left))))
    f = 0.5 * (cam["fx"] + cam["fy"])
    tc = TC.camera(f, cam["width"], cam["height"], p[2], pitch, roll)
    foot = np.array([p[0], p[1], 0.0])
    return tc, {"foot": foot.tolist(), "R_tr": R_tr.tolist()}


def table_to_rig(frame_map: dict, P_table):
    return np.array(frame_map["foot"]) + np.atleast_2d(np.asarray(P_table, float)) @ np.asarray(frame_map["R_tr"]).T


def rig_to_table(frame_map: dict, P_rig):
    return (np.atleast_2d(np.asarray(P_rig, float)) - np.array(frame_map["foot"])) @ np.asarray(frame_map["R_tr"])
