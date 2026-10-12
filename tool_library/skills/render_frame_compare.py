"""The Stage 2 receipt: video frame, simulation render through the same camera, and their blend, side by side."""
from __future__ import annotations

TOOL = {
    "name": "render_frame_compare",
    "title": "Render frame comparison",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Renders the scene through the registered video camera (added from camera.json when the scene lacks it) at its initial or a given state and writes a side-by-side image: the video frame, the render, and a 50/50 blend, optionally with a fourth panel from an Isaac Sim render.",
    "description": (
        "Loads the MuJoCo scene, adds the video camera from camera.json (or uses a named camera), renders RGB at the "
        "camera's resolution at the initial state (keyframe 0 when present) or `qpos`, and composes [frame | render | "
        "blend] with labels; with `isaac_png` a fourth panel shows the Isaac render and a fifth its blend. The blend is "
        "where a placement error shows as a double image. Returns the paths and the mean absolute pixel difference "
        "inside `mask` (the objects' pixels) as a rough fit number; silhouette_iou_check is the real measure."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "path", "description": "MuJoCo scene."},
        {"name": "camera", "type": "path | dict | str", "description": "camera.json, its dict, or a camera name in the scene."},
        {"name": "frame", "type": "path", "description": "The video frame to compare with."},
        {"name": "out_png", "type": "path", "description": "The composite image."},
        {"name": "qpos", "type": "list | path | None", "description": "State to render; default initial / keyframe 0."},
        {"name": "isaac_png", "type": "path | None", "description": "An Isaac Sim render of the same scene to add as extra panels."},
        {"name": "mask", "type": "path | None", "description": "Union mask of the objects for the difference number."},
        {"name": "labels", "type": "list[str] | None", "description": "Panel labels; default ['video', 'simulation', 'blend']."},
    ],
    "outputs": [
        {"name": "out_png / render_png", "type": "path", "description": "The composite and the bare render."},
        {"name": "mean_abs_diff", "type": "float | None", "description": "Mean |frame - render| over the mask, 0..255."},
    ],
    "status": "tested",
    "entrypoint": "render_frame_compare",
    "requirements": ["mujoco>=3.1", "numpy", "opencv-python"],
    "tags": ["render", "comparison", "frame 0", "camera", "stage 2"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.render_frame_compare scene.xml camera.json frame00000.jpg frame0_compare.png",
}

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from tool_library.common import rigcam as RC


def render_with_camera(scene_xml, camera, qpos=None, width=None, height=None):
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    if isinstance(camera, str) and not camera.endswith(".json") and not os.path.exists(camera):
        name = camera; cam = None
    else:
        cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
        c = spec.worldbody.add_camera(); c.name = "compare_video_cam"; c.pos = cam["pos"]; c.quat = cam["quat_wxyz"]; c.fovy = cam["fovy_deg"]
        name = "compare_video_cam"
    m = spec.compile(); d = mujoco.MjData(m)
    if qpos is not None:
        q = np.load(qpos) if isinstance(qpos, (str, Path)) else np.asarray(qpos, float); d.qpos[:len(q)] = q
    elif m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    W = int(width or (cam["width"] if cam else 640)); H = int(height or (cam["height"] if cam else 480))
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    with mujoco.Renderer(m, H, W) as r:
        r.update_scene(d, camera=name)
        return r.render().copy()


def render_frame_compare(scene_xml, camera, frame, out_png, qpos=None, isaac_png=None, mask=None, labels=None):
    import cv2
    img = cv2.imread(str(frame))
    if img is None:
        raise FileNotFoundError(frame)
    H, W = img.shape[:2]
    rgb = render_with_camera(scene_xml, camera, qpos, W, H)
    ren = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if ren.shape[:2] != (H, W):
        ren = cv2.resize(ren, (W, H))
    blend = (0.5 * img + 0.5 * ren).astype(np.uint8)
    panels = [img, ren, blend]; labs = list(labels or ["video", "simulation", "blend"])
    if isaac_png:
        isa = cv2.imread(str(isaac_png))
        if isa is not None:
            isa = cv2.resize(isa, (W, H)); panels += [isa, (0.5 * img + 0.5 * isa).astype(np.uint8)]; labs += ["isaac sim", "isaac blend"]
    tw = 480 if W > 480 else W; th = int(round(H * tw / W))
    tiles = []
    for p, l in zip(panels, labs):
        t = cv2.resize(p, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (10 + 11 * len(l), 24), (0, 0, 0), -1); cv2.putText(t, l, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    sheet = np.hstack(tiles)
    out = Path(out_png); out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    rp = out.with_name(out.stem + "_render.png"); cv2.imwrite(str(rp), ren)
    diff = None
    if mask is not None:
        mk = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if mk is not None:
            mk = mk[..., 0] if mk.ndim == 3 else mk
            if mk.shape != (H, W):
                mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
            sel = mk > 0
            diff = float(np.abs(img[sel].astype(float) - ren[sel].astype(float)).mean()) if sel.any() else None
    return {"out_png": str(out), "render_png": str(rp), "mean_abs_diff": None if diff is None else round(diff, 1), "panels": labs}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("scene_xml"); ap.add_argument("camera"); ap.add_argument("frame"); ap.add_argument("out_png")
    ap.add_argument("--qpos"); ap.add_argument("--isaac-png"); ap.add_argument("--mask")
    a = ap.parse_args(argv)
    print(json.dumps(render_frame_compare(a.scene_xml, a.camera, a.frame, a.out_png, a.qpos, a.isaac_png, a.mask), indent=1))


if __name__ == "__main__":
    sys.exit(main())
