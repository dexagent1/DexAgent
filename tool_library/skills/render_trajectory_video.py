"""Play a qpos trajectory through the scene and record it from one or more cameras."""
from __future__ import annotations

TOOL = {
    "name": "render_trajectory_video",
    "title": "Render trajectory video",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Kinematic playback of a (N, nq) qpos trajectory rendered from one or several cameras into mp4 files and an optional side-by-side preview.",
    "description": (
        "Every delivered episode is checked by eye before anything else, and the review needs "
        "the same frames the robot data will carry: the video camera, the wrist cameras, an "
        "overview. This sets qpos frame by frame (no physics, so the recording is reproduced "
        "exactly), renders each requested camera offscreen and encodes one mp4 per camera plus "
        "a labelled side-by-side preview. A camera that carries a resolution (MJCF camera `resolution`, which "
        "stage_on_bench writes from camera.json) is rendered at that aspect, so the sim video of an EgoDex video camera "
        "is 16:9 like the human video instead of squashed into 640 x 480, and crops given as fractions of the frame match "
        "between the two. Accepts a trajectory npz with 'qpos' and 'fps', or an array."
    ),
    "inputs": [
        {"name": "scene", "type": "path | MjModel", "description": "Scene MJCF."},
        {"name": "trajectory", "type": "path | ndarray", "description": "npz with qpos (N,nq) [+ fps], or the array."},
        {"name": "out_dir", "type": "path", "description": "Receives <camera>.mp4 and preview.mp4."},
        {"name": "cameras", "type": "list[str | dict]", "description": "Camera names or free-camera dicts (see render_scene). Default: every camera in the model."},
        {"name": "fps", "type": "float | None", "description": "Output frame rate; the npz fps or 30 when None."},
        {"name": "stride", "type": "int", "description": "Render every k-th frame (1)."},
        {"name": "width, height", "type": "int | None", "description": "Frame size. Default: a scene camera that carries a resolution (the video camera stage_on_bench writes from camera.json carries the human video's width and height) is rendered at its aspect, 480 pixels high (854 x 480 for a 1920 x 1080 EgoDex camera); any other camera at 640 x 480. Give one of them to scale the camera's aspect, both to force a size."},
        {"name": "overlay_text", "type": "bool", "description": "Stamp the frame index on the preview (True)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "videos per camera, preview path, n_frames, fps, sizes {camera: [width, height]}."},
    ],
    "status": "tested",
    "entrypoint": "render_trajectory_video",
    "requirements": ["mujoco", "imageio[ffmpeg]", "EGL"],
    "tags": ["render", "video", "replay", "preview"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
    "cli": "MUJOCO_GL=egl python -m tool_library.skills.render_trajectory_video scene.xml trajectory.npz out/ ego,left_wrist",
}

import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common.render import Offscreen, load_model
from tool_library.common.video import VideoWriter


def _label(img, text):
    import cv2
    img = np.ascontiguousarray(img)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (0, 0), (w + 10, h + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (5, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


DEFAULT_SIZE = (640, 480)
DEFAULT_HEIGHT = 480       # a camera with a resolution is rendered at its aspect this many pixels high


def _even(x):
    return max(2, int(round(float(x) / 2.0)) * 2)


def frame_size(model, camera, width=None, height=None):
    """(width, height) to render `camera` at: both given, as given; else a named scene camera with a resolution keeps
    its aspect (height DEFAULT_HEIGHT, or scaled from the one size given); any other camera DEFAULT_SIZE."""
    if width is not None and height is not None:
        return int(width), int(height)
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) if isinstance(camera, str) else -1
    res = [int(v) for v in model.cam_resolution[cid]] if cid >= 0 and hasattr(model, "cam_resolution") else [1, 1]
    if res[0] > 1 and res[1] > 1:
        aspect = res[0] / res[1]
        if width is not None:
            return int(width), _even(int(width) / aspect)
        h = int(height) if height is not None else DEFAULT_HEIGHT
        return _even(h * aspect), h
    if width is not None:
        return int(width), _even(int(width) * DEFAULT_SIZE[1] / DEFAULT_SIZE[0])
    if height is not None:
        return _even(int(height) * DEFAULT_SIZE[0] / DEFAULT_SIZE[1]), int(height)
    return DEFAULT_SIZE


def render_trajectory_video(scene, trajectory, out_dir, cameras=None, fps=None, stride=1, width=None, height=None,
                            overlay_text=True):
    import mujoco
    recording = {}
    if isinstance(trajectory, (str, Path)):
        with np.load(trajectory, allow_pickle=False) as z:
            Q = np.asarray(z["qpos"], float)
            fps = fps if fps is not None else float(z["fps"]) if "fps" in z.files else 30.0
            recording = {key: z[key] for key in ("joint_names", "actuator_names", "qvel", "ctrl", "eq_active", "time_s") if key in z.files}
    else:
        Q = np.asarray(trajectory, float)
        fps = 30.0 if fps is None else fps
    if not np.isfinite(float(fps)) or float(fps) <= 0 or int(stride) != stride or stride < 1:
        raise ValueError("fps must be positive and finite; stride must be a positive integer")
    free = {f"_free{i}": c for i, c in enumerate(cameras or []) if isinstance(c, dict) and "pos" in c}
    model = load_model(scene, cameras=free) if free else load_model(scene)
    if cameras is None:
        cameras = [model.camera(i).name for i in range(model.ncam)] or [{"lookat": [0, 0, 0.2], "distance": 1.6, "azimuth": 120, "elevation": -30}]
    names = []
    for i, c in enumerate(cameras):
        names.append(c if isinstance(c, str) else (f"_free{i}" if "pos" in c else c))
    if Q.ndim != 2 or Q.shape[1] != model.nq or len(Q) == 0 or not np.isfinite(Q).all():
        raise ValueError(f"trajectory must be a finite nonempty (N, {model.nq}) qpos array")
    if "joint_names" in recording:
        from tool_library.common.episode import joint_names
        if list(recording["joint_names"].astype(str)) != joint_names(model):
            raise ValueError("recorded joint identity differs from the rendering scene")
    if "actuator_names" in recording and list(recording["actuator_names"].astype(str)) != [model.actuator(i).name for i in range(model.nu)]:
        raise ValueError("recorded actuator identity differs from the rendering scene")
    for key, count in (("qvel", model.nv), ("ctrl", model.nu), ("eq_active", model.neq)):
        if key in recording and (recording[key].shape != (len(Q), count) or not np.isfinite(recording[key]).all()):
            raise ValueError(f"recorded {key} has invalid shape or nonfinite values")
    if "time_s" in recording and (recording["time_s"].shape != (len(Q),) or not np.isfinite(recording["time_s"]).all()
                                      or np.any(np.diff(recording["time_s"]) <= 0)):
        raise ValueError("recorded time_s must be finite and strictly increasing")
    data = mujoco.MjData(model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = [str(n if isinstance(n, str) else f"view{i}") for i, n in enumerate(names)]
    sizes = {k: frame_size(model, n, width, height) for k, n in zip(keys, names)}
    # the offscreen buffer must hold the largest frame
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), max(w for w, _ in sizes.values()))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), max(h for _, h in sizes.values()))
    rends = {sz: Offscreen(model, sz[0], sz[1]) for sz in set(sizes.values())}
    out_fps = float(fps) / int(stride)
    videos = {k: str(out / f"{k.lstrip('_')}.mp4") for k in keys}
    preview = str(out / "preview.mp4") if len(keys) > 1 or overlay_text else None
    h_min = min(h for _, h in sizes.values())
    try:
        from contextlib import ExitStack
        with ExitStack() as stack:
            writers = {k: stack.enter_context(VideoWriter(path, out_fps)) for k, path in videos.items()}
            preview_writer = stack.enter_context(VideoWriter(preview, out_fps)) if preview else None
            for t in range(0, len(Q), int(stride)):
                data.qpos[:] = Q[t]
                for key in ("qvel", "ctrl", "eq_active"):
                    if key in recording:
                        getattr(data, key)[:] = recording[key][t]
                data.time = float(recording["time_s"][t]) if "time_s" in recording else t / float(fps)
                mujoco.mj_forward(model, data)
                tiles = []
                for k, n in zip(keys, names):
                    frame = rends[sizes[k]].rgb(data, n)
                    writers[k].append_data(frame)
                    if preview_writer is not None:
                        tile = _label(frame.copy(), f"{k.lstrip('_')}  t={t}") if overlay_text else frame
                        if tile.shape[0] != h_min:
                            import cv2
                            tile = cv2.resize(tile, (_even(tile.shape[1] * h_min / tile.shape[0]), h_min))
                        tiles.append(tile)
                if preview_writer is not None:
                    preview_writer.append_data(np.concatenate(tiles, 1))
    finally:
        for r in rends.values():
            r.close()
    n = len(range(0, len(Q), int(stride)))
    return {"videos": videos, "preview": preview, "n_frames": n, "fps": out_fps, "cameras": keys,
            "sizes": {k.lstrip("_"): list(sizes[k]) for k in keys}}


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(render_trajectory_video(a[0], a[1], a[2], a[3].split(",") if len(a) > 3 else None), indent=1))
