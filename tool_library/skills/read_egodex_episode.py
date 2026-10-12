"""One EgoDex episode -> frames, the camera in the rig frame, the table plane, and the hand joints."""
from __future__ import annotations

TOOL = {
    "name": "read_egodex_episode",
    "title": "Read EgoDex episode",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Opens an EgoDex episode (hdf5 + mp4): writes the first frame (or a strided set), the head camera as camera.json in the rig frame via the table plane found from the fingertip heights, the task text, and the 2D hand joints per frame.",
    "description": (
        "EgoDex stores the camera intrinsics (camera/intrinsic), the per-frame camera pose and 3D skeleton "
        "(transforms/*, ARKit world, gravity-aligned y-up) and the task description in the file attributes. The table "
        "plane is the horizontal plane through the low percentile of all fingertip heights over the episode minus a pad "
        "(fingertips touch the table during every task); the rig frame is built from it: z up, y the chosen frame's "
        "optical axis projected onto the plane, x = y cross z, origin so that the camera's foot lands at `cam_xy` (default "
        "(0.019, 0.05): the robot mounts 5 cm behind the camera foot, since a head camera 0.3 to 0.4 m up with a 72 degree "
        "lens would otherwise see the parked robot hands in the middle of the frame). Writes frame%05d.jpg files, camera.json "
        "(fx, fy, cx, cy, fovy_deg, pos, quat_wxyz, R_cv in the rig frame; the frame's own pose), episode.json (task, "
        "description, objects named by the dataset, n_frames, fps, table height, rig transform world<-rig, fingertip "
        "pixels at the frame) and hands.npz (every joint's 3D rig position and 2D pixel per frame)."
    ),
    "inputs": [
        {"name": "hdf5", "type": "path", "description": "The episode's .hdf5."},
        {"name": "mp4", "type": "path | None", "description": "The episode's .mp4; default: same stem as the hdf5."},
        {"name": "out_dir", "type": "path", "description": "Receives frames, camera.json, episode.json, hands.npz."},
        {"name": "frame", "type": "int", "description": "The frame whose camera pose defines the rig frame and camera.json. Default 0."},
        {"name": "stride", "type": "int", "description": "Also write every k-th frame; 0 writes only `frame`. Default 0."},
        {"name": "cam_xy", "type": "list[2]", "description": "Camera foot in the rig frame. Default [0.019, 0.05]."},
        {"name": "tip_percentile", "type": "float", "description": "Percentile of fingertip heights taken as the tabletop. Default 1.0."},
        {"name": "tip_pad_m", "type": "float", "description": "Fingertip centre height above the surface when touching. Default 0.008."},
    ],
    "outputs": [
        {"name": "frames", "type": "list[path]", "description": "Written frame files (the chosen frame first)."},
        {"name": "camera", "type": "dict", "description": "camera.json content in the rig frame."},
        {"name": "episode", "type": "dict", "description": "task, description, dataset objects, n_frames, fps, width, height, table_height_world, rig {R_world_rig, origin_world}, fingertips_px at the frame, hands_npz."},
    ],
    "status": "tested",
    "entrypoint": "read_egodex_episode",
    "requirements": ["h5py", "opencv-python", "numpy", "scipy"],
    "tags": ["egodex", "camera", "table plane", "hands", "stage 2"],
    "used_in": [],
    "cli": "python -m tool_library.skills.read_egodex_episode episode.hdf5 out/ --stride 30",
}

import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common import rigcam as RC

TIPS = ["ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip"]


def _attrs(f):
    out = {}
    for k, v in f.attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif hasattr(v, "tolist"):
            v = v.tolist()
        out[k] = v
    return out


def episode_description(attrs):
    """Select the description of the recorded direction of a reversible episode."""
    def text(value):
        value = value.decode().strip() if isinstance(value, bytes) else str(value or "").strip()
        # EgoDex writes the literal string "None" for an episode without a second (reverse) description
        return "" if value.lower() in ("none", "null", "nan") else value
    first, second = text(attrs.get("llm_description")), text(attrs.get("llm_description2"))
    selector = text(attrs.get("which_llm_description"))
    if selector:
        if selector not in ("1", "2"):
            raise ValueError(f"unknown which_llm_description={selector!r}; cannot choose the task direction")
        selected = first if selector == "1" else second
        if not selected:
            raise ValueError(f"which_llm_description={selector} names a missing description")
        return selected
    if first and second and first != second:
        raise ValueError("reversible EgoDex episode has two descriptions but no which_llm_description; task direction is ambiguous")
    if first or second:
        return first or second
    raise ValueError("EgoDex episode has no task description")


def read_egodex_episode(hdf5, out_dir, mp4=None, frame=0, stride=0, cam_xy=(0.019, 0.05), tip_percentile=1.0, tip_pad_m=0.008):
    import cv2
    import h5py
    hdf5 = Path(hdf5)
    mp4 = Path(mp4) if mp4 else hdf5.with_suffix(".mp4")
    if not mp4.exists():
        raise FileNotFoundError(f"no video next to {hdf5}: {mp4}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5, "r") as f:
        attrs = _attrs(f)
        description = episode_description(attrs)
        K = f["camera/intrinsic"][()].astype(float)
        Tc = f["transforms/camera"][()].astype(float)
        joints = {k: f[f"transforms/{k}"][()].astype(float) for k in f["transforms"].keys() if k != "camera"}
        conf = {k: f[f"confidences/{k}"][()] for k in f["confidences"].keys()} if "confidences" in f else {}
    n = len(Tc)
    if not 0 <= frame < n:
        raise ValueError(f"frame {frame} outside 0..{n - 1}")
    up = np.array([0.0, 1.0, 0.0])                       # ARKit gravity-aligned world
    # table plane from the fingertips
    ys = []
    for side in ("left", "right"):
        for t in TIPS:
            k = side + t
            if k in joints:
                y = joints[k][:, 1, 3]
                c = conf.get(k)
                if c is not None and (c > 0.5).sum() > 10:
                    y = y[c > 0.5]
                ys.append(y)
    if not ys:
        raise ValueError("no fingertip transforms in the file; cannot find the table plane")
    y_table = float(np.percentile(np.concatenate(ys), float(tip_percentile))) - float(tip_pad_m)
    # rig frame from the chosen frame's camera
    T = Tc[frame]; R = T[:3, :3]; p = T[:3, 3]
    fwd = R[:, 2]
    yv = fwd - (fwd @ up) * up
    if np.linalg.norm(yv) < 1e-6:
        raise ValueError("the camera looks straight down; the rig yaw is undefined at this frame, choose another")
    yv /= np.linalg.norm(yv)
    xv = np.cross(yv, up)
    R_wr = np.stack([xv, yv, up], axis=1)                 # world <- rig
    foot = p - ((p - y_table * up) @ up) * up
    origin = foot - float(cam_xy[0]) * xv - float(cam_xy[1]) * yv
    R_cv = R_wr.T @ R                                     # camera axes in rig (OpenCV)
    pos = R_wr.T @ (p - origin)
    # frames
    cap = cv2.VideoCapture(str(mp4))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = [frame] + ([i for i in range(0, min(n, nvid), int(stride)) if i != frame] if stride and stride > 0 else [])
    files = []
    for i in sorted(set(want)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = cap.read()
        if not ok:
            continue
        fp = out / f"frame{i:05d}.jpg"
        cv2.imwrite(str(fp), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        files.append(str(fp))
    cap.release()
    files.sort(key=lambda s: (not s.endswith(f"frame{frame:05d}.jpg"), s))
    sx, sy = W / (2 * K[0, 2]), H / (2 * K[1, 2])         # intrinsics are for the principal-point-centred image
    cam = RC.camera_dict(W, H, K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy, R_cv, pos,
                         {"source": "egodex", "frame": int(frame), "hdf5": str(hdf5), "mp4": str(mp4)})
    RC.save(cam, out / "camera.json")
    # hands: 3D rig positions and pixels for every joint and frame
    names = sorted(joints)
    P = np.stack([joints[k][:, :3, 3] for k in names], axis=1)          # (n, J, 3) world
    P_rig = (P - origin) @ R_wr                                          # (n, J, 3)
    uv = np.zeros((n, len(names), 2))
    for i in range(n):
        Ti = Tc[i]; Ri = Ti[:3, :3]; pi = Ti[:3, 3]
        Pc = (P[i] - pi) @ Ri
        z = np.where(np.abs(Pc[:, 2]) < 1e-6, 1e-6, Pc[:, 2])
        uv[i, :, 0] = cam["cx"] + cam["fx"] * Pc[:, 0] / z
        uv[i, :, 1] = cam["cy"] + cam["fy"] * Pc[:, 1] / z
    np.savez_compressed(out / "hands.npz", names=np.array(names), rig_xyz=P_rig.astype(np.float32), pixels=uv.astype(np.float32),
                        camera_world=Tc.astype(np.float32), R_world_rig=R_wr, origin_world=origin, y_table_world=y_table)
    tips = {}
    for side in ("left", "right"):
        for t in TIPS:
            k = side + t
            if k in names:
                j = names.index(k)
                tips[k] = [round(float(uv[frame, j, 0]), 1), round(float(uv[frame, j, 1]), 1), round(float(P_rig[frame, j, 2]), 4)]
    episode = {"task": attrs.get("task"), "description": description,
               "description1": attrs.get("llm_description"), "description2": attrs.get("llm_description2"),
               "which_llm_description": attrs.get("which_llm_description"),
               "dataset_objects": attrs.get("llm_objects"), "environment": attrs.get("environment"), "n_frames": int(n), "fps": fps,
               "width": W, "height": H, "frame": int(frame), "table_height_world": y_table,
               "rig": {"R_world_rig": R_wr.tolist(), "origin_world": origin.tolist(), "cam_xy": [float(cam_xy[0]), float(cam_xy[1])]},
               "fingertips_px_z": tips, "hands_npz": str(out / "hands.npz"), "camera_json": str(out / "camera.json"), "frames": files}
    (out / "episode.json").write_text(json.dumps(episode, indent=1))
    return {"frames": files, "camera": cam, "episode": episode}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("hdf5"); ap.add_argument("out_dir"); ap.add_argument("--mp4"); ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0)
    a = ap.parse_args(argv)
    r = read_egodex_episode(a.hdf5, a.out_dir, a.mp4, a.frame, a.stride)
    print(json.dumps({"frames": r["frames"][:3], "camera": {k: r["camera"][k] for k in ("fovy_deg", "pos", "pitch_deg")}, "task": r["episode"]["task"], "n_frames": r["episode"]["n_frames"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
