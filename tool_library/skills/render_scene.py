"""One image of a scene from a named or free camera: RGB, segmentation or depth."""
from __future__ import annotations

TOOL = {
    "name": "render_scene",
    "title": "Render scene",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Offscreen MuJoCo render of a scene at a given qpos from a named camera or a free camera (pos+quat, look-at, or azimuth/elevation), as RGB, body segmentation or depth.",
    "description": (
        "The single most used look at a scene: what does it look like from the video camera, a "
        "wrist, or an overview? Loads the model, sets the configuration, runs forward "
        "kinematics and renders offscreen with EGL. Segmentation returns a per-pixel body id "
        "map (object masks for IoU against video masks); depth returns metres. Cameras may be a "
        "name defined in the MJCF or a dict for an ad-hoc free camera."
    ),
    "inputs": [
        {"name": "scene", "type": "path | MjModel", "description": "MJCF/MJB or a loaded model."},
        {"name": "camera", "type": "str | dict", "description": "Camera name, or {'pos','quat','fovy'} / {'lookat','distance','azimuth','elevation'}."},
        {"name": "out_path", "type": "path | None", "description": "PNG for rgb/segmentation preview; .npy for depth/labels when given."},
        {"name": "qpos", "type": "ndarray | None", "description": "Configuration; model default when None."},
        {"name": "width, height", "type": "int", "description": "Image size (640 x 480)."},
        {"name": "mode", "type": "str", "description": "'rgb' (default), 'segmentation' or 'depth'."},
    ],
    "outputs": [
        {"name": "image", "type": "ndarray", "description": "RGB uint8 (H,W,3); body-id int32 (H,W) with -1 background; or depth float32 metres."},
    ],
    "status": "tested",
    "entrypoint": "render_scene",
    "requirements": ["mujoco", "EGL (MUJOCO_GL=egl) or OSMesa"],
    "tags": ["render", "camera", "segmentation", "depth"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
    "cli": "MUJOCO_GL=egl python -m tool_library.skills.render_scene scene.xml ego out.png",
}

import sys

import numpy as np

from tool_library.common.render import Offscreen, load_model


def render_scene(scene, camera, out_path=None, qpos=None, width=640, height=480, mode="rgb"):
    import mujoco
    if isinstance(camera, dict) and "pos" in camera and not isinstance(scene, mujoco.MjModel):
        model = load_model(scene, cameras={"_view": camera})      # exact pose incl. roll
        camera = "_view"
    else:
        model = load_model(scene)
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = np.asarray(qpos, float)
    mujoco.mj_forward(model, data)
    r = Offscreen(model, width, height)
    try:
        if mode == "rgb":
            img = r.rgb(data, camera)
        elif mode == "segmentation":
            img = r.segmentation(data, camera)
        elif mode == "depth":
            img = r.depth(data, camera)
        else:
            raise ValueError(mode)
    finally:
        r.close()
    if out_path:
        from PIL import Image
        p = str(out_path)
        if p.endswith(".npy"):
            np.save(p, img)
        elif mode == "rgb":
            Image.fromarray(img).save(p)
        elif mode == "segmentation":
            from tool_library.common.video import PALETTE_COLORS
            vis = np.zeros(img.shape + (3,), np.uint8)
            for b in np.unique(img):
                if b >= 0:
                    vis[img == b] = PALETTE_COLORS[(int(b) % (len(PALETTE_COLORS) - 1)) + 1]
            Image.fromarray(vis).save(p)
        else:
            d = np.clip(img / max(float(np.nanmax(img)), 1e-6), 0, 1)
            Image.fromarray((255 * (1 - d)).astype(np.uint8)).save(p)
    return img


if __name__ == "__main__":
    a = sys.argv[1:]
    img = render_scene(a[0], a[1], a[2] if len(a) > 2 else None, mode=a[3] if len(a) > 3 else "rgb")
    print(img.shape, img.dtype)
