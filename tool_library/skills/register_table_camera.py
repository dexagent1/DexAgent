"""Pin the video camera in the rig frame: from the setup, from given numbers, or from estimates."""
from __future__ import annotations

TOOL = {
    "name": "register_table_camera",
    "title": "Register table camera",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Writes camera.json, the video camera in the rig frame, from one of three sources: the setup's calibrated ego camera, given intrinsics and pose, or estimates (focal or fovy, pitch, roll, height above the table, optional yaw and foot position) such as GeoCalib's angles and a depth-plane height; also draws a table grid overlay to judge the registration.",
    "description": (
        "The rig frame is the setup's: origin between the robot mounts on the tabletop, x toward the right station, y "
        "into the work, z up. `source='setup'` reads cameras.ego from setup.yaml and rescales it to the video's "
        "resolution. `source='given'` takes fx, fy, cx, cy (or fovy_deg) with pos and quat_wxyz (MuJoCo convention) or "
        "R_cv (OpenCV axes in the rig frame). `source='estimate'` builds the pose from pitch_deg (below horizontal), "
        "roll_deg, height_m above the tabletop, yaw_deg (90 = looking straight into the work) and the camera's foot "
        "position foot_xy on the table (default: the setup camera's foot), with fovy_deg or focal_px (default: the "
        "setup's). Writes camera.json (fx, fy, cx, cy, fovy_deg, pos, quat_wxyz, R_cv) and, when a frame is given, "
        "camera_check.png: the frame with the rig axes, a 10 cm grid on the tabletop and the horizon drawn through the "
        "camera, so a wrong pitch, height or focal shows as a grid that does not lie on the table."
    ),
    "inputs": [
        {"name": "out_json", "type": "path", "description": "Where camera.json is written."},
        {"name": "source", "type": "str", "description": "'setup' | 'given' | 'estimate'. Default 'setup'."},
        {"name": "width", "type": "int", "description": "Video width in pixels (required unless frame is given)."},
        {"name": "height", "type": "int", "description": "Video height in pixels."},
        {"name": "setup_yaml", "type": "path | None", "description": "setup.yaml; default: the repository's setup/setup.yaml."},
        {"name": "fovy_deg", "type": "float | None", "description": "Vertical field of view (given/estimate)."},
        {"name": "focal_px", "type": "float | None", "description": "Focal length in pixels at the given resolution (alternative to fovy_deg)."},
        {"name": "intrinsics", "type": "dict | None", "description": "{fx, fy, cx, cy} for source='given'."},
        {"name": "pos", "type": "list[3] | None", "description": "Camera position in the rig frame (given)."},
        {"name": "quat_wxyz", "type": "list[4] | None", "description": "MuJoCo camera orientation (given); or give R_cv."},
        {"name": "R_cv", "type": "list[list] | None", "description": "3x3 OpenCV camera axes in the rig frame (given)."},
        {"name": "pitch_deg", "type": "float | None", "description": "Optical axis below horizontal (estimate)."},
        {"name": "roll_deg", "type": "float", "description": "Roll about the optical axis (estimate). Default 0."},
        {"name": "height_m", "type": "float | None", "description": "Camera height above the tabletop (estimate)."},
        {"name": "yaw_deg", "type": "float", "description": "Optical axis yaw from rig +x (estimate). Default 90."},
        {"name": "foot_xy", "type": "list[2] | None", "description": "Where the camera stands over the table, rig x, y (estimate). Default: the setup camera's foot."},
        {"name": "frame", "type": "path | None", "description": "A frame of the video to draw camera_check.png on."},
    ],
    "outputs": [
        {"name": "camera", "type": "dict", "description": "The camera.json content: width, height, fx, fy, cx, cy, fovy_deg, pos, quat_wxyz, R_cv, pitch_deg, yaw_deg, height_above_table_m, source."},
        {"name": "out_json / check_png", "type": "path", "description": "The written files."},
    ],
    "status": "tested",
    "entrypoint": "register_table_camera",
    "requirements": ["numpy", "scipy", "pyyaml", "opencv-python"],
    "tags": ["camera", "registration", "extrinsics", "intrinsics", "rig frame", "stage 2"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.register_table_camera camera.json --source setup --width 640 --height 480 --frame frame00000.jpg",
}

import json
import math
import sys
from pathlib import Path

import numpy as np

from tool_library.common import rigcam as RC

DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "setup" / "setup.yaml"


def _setup_camera(setup_yaml):
    import yaml
    s = yaml.safe_load(Path(setup_yaml or DEFAULT_SETUP).read_text())
    ego = s["cameras"]["ego"]
    return RC.camera_from_mujoco(ego["width"], ego["height"], ego["fovy_deg"], ego["pos"], ego["quat_wxyz"])


def draw_check(cam, frame_path, out_png):
    import cv2
    img = cv2.imread(str(frame_path))
    if img is None:
        raise FileNotFoundError(frame_path)
    H, W = img.shape[:2]
    if (W, H) != (cam["width"], cam["height"]):
        img = cv2.resize(img, (cam["width"], cam["height"]))
    def px(P):
        uv, z = RC.project(cam, P)
        return uv, z
    # 10 cm grid on the tabletop, x in [-0.5, 0.5], y in [0, 1.2]
    for x in np.arange(-0.5, 0.51, 0.1):
        uv, z = px([[x, y, 0.0] for y in np.linspace(0.0, 1.2, 25)])
        pts = [tuple(int(round(v)) for v in p) for p, zz in zip(uv, z) if zz > 0.05]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (0, 200, 255) if abs(x) > 1e-6 else (0, 0, 255), 1, cv2.LINE_AA)
    for y in np.arange(0.0, 1.21, 0.1):
        uv, z = px([[x, y, 0.0] for x in np.linspace(-0.5, 0.5, 21)])
        pts = [tuple(int(round(v)) for v in p) for p, zz in zip(uv, z) if zz > 0.05]
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, (0, 200, 255) if abs(y) > 1e-6 else (255, 0, 0), 1, cv2.LINE_AA)
    # horizon: points far along the plane
    uv, z = px([[x, 50.0, 0.0] for x in np.linspace(-60, 60, 7)])
    pts = [tuple(int(round(v)) for v in p) for p, zz in zip(uv, z) if zz > 0]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"fovy {cam['fovy_deg']:.1f}  pitch {cam['pitch_deg']:.1f}  h {cam['height_above_table_m']:.3f} m  foot ({cam['pos'][0]:.2f}, {cam['pos'][1]:.2f})",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "red: rig x axis (baseline)  blue: y = 0 (the mounts)  grid: 10 cm on the tabletop  cyan: horizon",
                (8, cam["height"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)
    return str(out_png)


def register_table_camera(out_json, source="setup", width=None, height=None, setup_yaml=None, fovy_deg=None, focal_px=None,
                          intrinsics=None, pos=None, quat_wxyz=None, R_cv=None, pitch_deg=None, roll_deg=0.0, height_m=None,
                          yaw_deg=90.0, foot_xy=None, frame=None):
    if frame is not None and (width is None or height is None):
        import cv2
        im = cv2.imread(str(frame))
        if im is None:
            raise FileNotFoundError(frame)
        height, width = im.shape[:2]
    base = _setup_camera(setup_yaml)
    if source == "setup":
        if width is None or height is None:
            width, height = base["width"], base["height"]
        cam = RC.scaled(base, width, height) if (int(width), int(height)) != (base["width"], base["height"]) else dict(base)
    elif source == "given":
        if width is None or height is None:
            raise ValueError("width and height are required for source='given'")
        if intrinsics:
            fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
        elif fovy_deg is not None or focal_px is not None:
            f = float(focal_px) if focal_px is not None else 0.5 * int(height) / math.tan(math.radians(float(fovy_deg)) / 2)
            fx = fy = f; cx, cy = int(width) / 2, int(height) / 2
        else:
            raise ValueError("source='given' needs intrinsics {fx, fy, cx, cy}, or fovy_deg, or focal_px")
        if pos is None:
            raise ValueError("source='given' needs pos")
        if R_cv is not None:
            R = np.asarray(R_cv, float).reshape(3, 3)
        elif quat_wxyz is not None:
            from scipy.spatial.transform import Rotation as Rot
            w, x, y, z = quat_wxyz
            R = Rot.from_quat([x, y, z, w]).as_matrix() @ RC.CV_TO_MJ
        else:
            raise ValueError("source='given' needs quat_wxyz or R_cv")
        cam = RC.camera_dict(width, height, fx, fy, cx, cy, R, pos)
    elif source == "estimate":
        if width is None or height is None:
            raise ValueError("width and height are required for source='estimate'")
        if pitch_deg is None or height_m is None:
            raise ValueError("source='estimate' needs pitch_deg and height_m (GeoCalib / fit_camera_to_table_plane give them)")
        if focal_px is not None:
            fov = math.degrees(2 * math.atan(0.5 * int(height) / float(focal_px)))
        elif fovy_deg is not None:
            fov = float(fovy_deg)
        else:
            fov = base["fovy_deg"]
        foot = foot_xy if foot_xy is not None else base["pos"][:2]
        R = RC.R_cv_from_angles(yaw_deg, pitch_deg, roll_deg)
        cam = RC.camera_from_fovy(width, height, fov, [float(foot[0]), float(foot[1]), float(height_m)], R)
    else:
        raise ValueError("source must be 'setup', 'given' or 'estimate'")
    cam["source"] = source
    out = Path(out_json)
    RC.save(cam, out)
    result = {"camera": cam, "out_json": str(out)}
    if frame is not None:
        result["check_png"] = draw_check(cam, frame, out.with_name("camera_check.png"))
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("out_json"); ap.add_argument("--source", default="setup")
    ap.add_argument("--width", type=int); ap.add_argument("--height", type=int); ap.add_argument("--setup-yaml")
    ap.add_argument("--fovy-deg", type=float); ap.add_argument("--focal-px", type=float)
    ap.add_argument("--pitch-deg", type=float); ap.add_argument("--roll-deg", type=float, default=0.0); ap.add_argument("--height-m", type=float)
    ap.add_argument("--yaw-deg", type=float, default=90.0); ap.add_argument("--foot-xy", type=float, nargs=2); ap.add_argument("--frame")
    a = ap.parse_args(argv)
    r = register_table_camera(a.out_json, a.source, a.width, a.height, a.setup_yaml, a.fovy_deg, a.focal_px, None, None, None, None,
                              a.pitch_deg, a.roll_deg, a.height_m, a.yaw_deg, a.foot_xy, a.frame)
    print(json.dumps({k: v for k, v in r.items() if k != "camera"} | {"camera": {k: r["camera"][k] for k in ("fovy_deg", "pos", "pitch_deg", "yaw_deg")}}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
