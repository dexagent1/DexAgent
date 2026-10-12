"""Look at a plan before executing it: the keyframes rendered as a sheet, with the plan's own collisions listed."""
from __future__ import annotations

TOOL = {
    "name": "preview_keyframes",
    "title": "Preview keyframes",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Render a subgoal's planned keyframes (no physics) as one labelled sheet from a chosen camera and list where the "
               "plan itself drives a hand through the other hand or through an object it is not handling, in seconds.",
    "description": (
        "A kinematic preflight. Takes the keyframes of plan_subgoal_keyframes (or its json), sets the scene to each "
        "keyframe's qpos, renders it from a named scene camera or a free camera looking at the handled object, tiles "
        "the frames with their phase labels, and runs the plan collision sweep on the interpolated reference "
        "(hand-hand and hand-vs-bystander interpenetration deeper than 4 mm, with frame and geom names). Use it to "
        "check an approach, a swing or a carry visually before spending a physics rollout, and to see why a rollout "
        "reported plan_collision."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "keyframes", "type": "list | str", "description": "Keyframes or the plan json written by plan_subgoal_keyframes."},
        {"name": "out_png", "type": "str", "description": "Sheet image to write."},
        {"name": "camera", "type": "str | None", "description": "Scene camera name (video_cam, ego, wrist_right ...); default a free camera looking at the handled object from the front-right."},
        {"name": "lookat_body", "type": "str | None", "description": "Body the free camera looks at (default the first object in the plan's phases)."},
        {"name": "allowed_bodies", "type": "list[str] | None", "description": "Bodies the hands may touch (the handled part, held objects) for the collision sweep."},
        {"name": "max_frames", "type": "int", "description": "At most this many keyframes on the sheet (evenly spaced). Default 12."},
        {"name": "width", "type": "int", "description": "Tile width. Default 320."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "out_png, n_keyframes, labels shown, plan_collisions [{frame, depth_mm, geoms, kind}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "preview", "render", "plan", "collision"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl"],
    "status": "tested",
    "entrypoint": "preview_keyframes",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E
from tool_library.skills.chain_subgoals import chain_subgoals


def preview_keyframes(scene, keyframes, out_png, camera=None, lookat_body=None, allowed_bodies=None, max_frames=12, width=320):
    from PIL import Image, ImageDraw
    plan = None
    if isinstance(keyframes, (str, Path)):
        plan = json.loads(Path(keyframes).read_text())
        keyframes = plan["keyframes"] if isinstance(plan, dict) else plan
    model, _ = E.build_execution_model(scene)
    if allowed_bodies is None and isinstance(plan, dict):
        exp = plan.get("expect", {})
        per = list((exp.get("hands") or {}).values()) or [exp]          # a two-hand plan: both hands' parts
        allowed_bodies = [b for e in per for b in [e.get("object")] + list(e.get("involved", []) or [])]
        allowed_bodies = [b for b in dict.fromkeys(allowed_bodies + list(exp.get("involved", []) or [])
                                                   + list((plan.get("state", {}).get("held") or {}).values())) if b]
        if lookat_body is None:
            lookat_body = next((e.get("object") for e in per if e.get("object")), None)
    data = mujoco.MjData(model)
    h = int(width * 3 / 4)
    r = mujoco.Renderer(model, h, int(width))
    idx = np.linspace(0, len(keyframes) - 1, min(int(max_frames), len(keyframes))).astype(int)
    tiles, labels = [], []
    for i in idx:
        kf = keyframes[int(i)]
        data.qpos[:] = np.asarray(kf["qpos"], float); mujoco.mj_forward(model, data)
        if camera and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera) >= 0:
            r.update_scene(data, camera=camera)
        else:
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            if lookat_body and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lookat_body) >= 0:
                cam.lookat[:] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, lookat_body)]
            else:
                cam.lookat[:] = [0.0, 0.4, 0.1]
            cam.distance, cam.azimuth, cam.elevation = 0.7, -60, -25
            r.update_scene(data, cam)
        img = Image.fromarray(r.render().copy()); dr = ImageDraw.Draw(img)
        lab = f"{int(i)}: {kf.get('label', '')}"[:60]
        dr.rectangle([0, 0, img.width, 14], fill=(0, 0, 0)); dr.text((3, 1), lab, fill=(255, 255, 255))
        tiles.append(img); labels.append(kf.get("label", ""))
    cols = 4 if len(tiles) > 4 else len(tiles); rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * int(width), rows * h), (30, 30, 30))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * int(width), (k // cols) * h))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True); sheet.save(out_png)
    ref = chain_subgoals(model, keyframes, fps=30.0, max_rate_deg_s=90.0, max_rate_m_s=0.4)
    hits = E.plan_collision_sweep(model, ref["qpos"], allowed_bodies=allowed_bodies or [])
    try:
        r.close()
    except Exception:
        pass
    return {"out_png": str(out_png), "n_keyframes": len(keyframes), "labels": labels, "plan_collisions": hits, "n_reference_frames": int(len(ref["qpos"]))}
