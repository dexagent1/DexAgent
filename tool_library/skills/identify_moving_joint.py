"""Which joint of an articulated object did the demonstrator move?"""
from __future__ import annotations

TOOL = {
    "name": "identify_moving_joint",
    "title": "Identify moving joint",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Ranks the slide/hinge joints of an articulated object by rendering each one opened from the video camera and matching edges against the real frame; also reports raw pixel change per projected part face.",
    "description": (
        "A four-drawer cabinet has four equally drivable prismatic joints and optimizing the "
        "wrong one looks like success. For every candidate joint this finds the part's front "
        "face (the geom near the front along the joint axis with the largest face across it), "
        "projects it into the video camera for a labelled region, then OPENS that joint alone "
        "to its range limit, renders the scene from the video camera and correlates the "
        "edge map with the real 'during' frame inside the object's footprint. Raw frame "
        "differencing is also reported but is only a prior: a sweeping arm inflates it. Both "
        "frames are saved with regions drawn so a vision model can confirm the winner."
    ),
    "inputs": [
        {"name": "scene", "type": "path | MjModel", "description": "MJCF with the articulated object and a camera matching the video."},
        {"name": "frame_before", "type": "path | ndarray", "description": "RGB frame with the part at rest."},
        {"name": "frame_during", "type": "path | ndarray", "description": "RGB frame with the part moved."},
        {"name": "camera", "type": "str | dict", "description": "Camera name in the scene, or {'pos','quat','fovy'}."},
        {"name": "joint_prefix", "type": "str", "description": "Only joints whose name starts with this."},
        {"name": "out_dir", "type": "path | None", "description": "Where to save annotated before/during PNGs and joint_id.json."},
        {"name": "qpos", "type": "ndarray | None", "description": "Rest configuration; model default when None."},
        {"name": "pad_px", "type": "int", "description": "Region padding in pixels (6)."},
    ],
    "outputs": [
        {"name": "ranking", "type": "dict", "description": "most_changed joint; ranked list with joint, body, range, bbox, edge_match, pixel_change; image paths."},
    ],
    "status": "tested",
    "entrypoint": "identify_moving_joint",
    "requirements": ["mujoco", "numpy", "pillow"],
    "tags": ["articulated", "joint", "drawer", "video evidence"],
    "used_in": ["Place Toy in Drawer"],
}

import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common.render import Offscreen, camera_dict, load_model, project
from tool_library.common.video import read_frame, write_json


def _front_face_corners(model, data, bid, axis):
    """Corners of the part's front panel: near the front along `axis`, largest face across it."""
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == bid]
    if not geoms:
        return [], None
    front = {g: float(np.asarray(data.geom_xpos[g]) @ axis) for g in geoms}
    front_at = max(front.values())
    best, best_area = None, -1.0
    for g in geoms:
        if front[g] < front_at - 0.05:
            continue
        lat = np.sort(np.asarray(model.geom_size[g], float))[::-1]
        area = float(lat[0] * max(lat[1], 1e-4))
        if area > best_area:
            best, best_area = g, area
    Rg = np.asarray(data.geom_xmat[best]).reshape(3, 3)
    sz = np.asarray(model.geom_size[best], float)
    cg = np.asarray(data.geom_xpos[best], float)
    return [cg + Rg @ (sz * (sx, sy, sz2)) for sx in (-1, 1) for sy in (-1, 1) for sz2 in (-1, 1)], best


def _grad(x):
    g = x.mean(2) if x.ndim == 3 else x
    return np.abs(np.diff(g, axis=0)[:, :-1]) + np.abs(np.diff(g, axis=1)[:-1, :])


def identify_moving_joint(scene, frame_before, frame_during, camera, joint_prefix="", out_dir=None, qpos=None, pad_px=6):
    import mujoco
    from PIL import Image, ImageDraw
    cam_name = camera
    if isinstance(camera, dict):
        model = load_model(scene, cameras={"_video": {"pos": camera["pos"], "quat": camera["quat"], "fovy": camera.get("fovy", 45.0)}})
        cam_name = "_video"
    else:
        model = load_model(scene)
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    A = read_frame(frame_before) if isinstance(frame_before, (str, Path)) else np.asarray(frame_before)
    B = read_frame(frame_during) if isinstance(frame_during, (str, Path)) else np.asarray(frame_during)
    H, W = A.shape[:2]
    cam = camera_dict(model, data, cam_name)
    cands = []
    for j in range(model.njnt):
        name = model.joint(j).name or f"joint{j}"
        if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            continue
        if joint_prefix and not name.startswith(joint_prefix):
            continue
        bid = int(model.jnt_bodyid[j])
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
            ax = np.asarray(data.xaxis[j], float)
        else:
            ax = np.asarray(data.xmat[bid], float).reshape(3, 3)[:, 1]
        ax = ax / (np.linalg.norm(ax) or 1.0)
        corners, g = _front_face_corners(model, data, bid, ax)
        c = {"joint": name, "jid": j, "body": model.body(bid).name, "bid": bid,
             "range": [round(float(x), 4) for x in model.jnt_range[j]], "bbox": None}
        if corners:
            uv, depth = project(np.asarray(corners), cam["pos"], cam["quat"], cam["fovy"], W, H)
            uv = uv[depth > 1e-6]
            if len(uv):
                c["bbox"] = [max(0, int(uv[:, 0].min()) - pad_px), max(0, int(uv[:, 1].min()) - pad_px),
                             min(W - 1, int(uv[:, 0].max()) + pad_px), min(H - 1, int(uv[:, 1].max()) + pad_px)]
                c["face_geom"] = model.geom(g).name or f"geom{g}"
        cands.append(c)
    assert cands, f"no slide/hinge joints{' with prefix ' + joint_prefix if joint_prefix else ''}"
    boxes = [c["bbox"] for c in cands if c["bbox"]]
    assert boxes, "no candidate part projects into the camera"
    foot = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    RW, RH = 480, 270
    sx, sy = RW / W, RH / H
    fx0, fy0, fx1, fy1 = int(foot[0] * sx), int(foot[1] * sy), int(foot[2] * sx), int(foot[3] * sy)

    def crop(img):
        return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((RW, RH)), float)[fy0:fy1 + 1, fx0:fx1 + 1]

    gr_real = _grad(crop(B))
    diff = np.abs(A.astype(float) - B.astype(float)).mean(2)
    frame_mean = float(diff.mean()) or 1e-6
    rend = Offscreen(model, RW, RH)
    d = mujoco.MjData(model)
    try:
        for c in cands:
            if not c["bbox"]:
                c["edge_match"] = c["pixel_change"] = None
                continue
            x0, y0, x1, y1 = c["bbox"]
            c["pixel_change"] = round(float(diff[y0:y1 + 1, x0:x1 + 1].mean()), 3)
            c["pixel_change_vs_frame"] = round(c["pixel_change"] / frame_mean, 3)
            d.qpos[:] = data.qpos
            d.qpos[int(model.jnt_qposadr[c["jid"]])] = float(model.jnt_range[c["jid"]][1])
            mujoco.mj_forward(model, d)
            gr = _grad(crop(rend.rgb(d, cam_name)))
            n0, n1 = min(gr.shape[0], gr_real.shape[0]), min(gr.shape[1], gr_real.shape[1])
            a1, a2 = gr[:n0, :n1].ravel(), gr_real[:n0, :n1].ravel()
            c["edge_match"] = None if a1.std() < 1e-6 or a2.std() < 1e-6 else round(float(np.corrcoef(a1, a2)[0, 1]), 4)
    finally:
        rend.close()
    scored = sorted([c for c in cands if c.get("edge_match") is not None], key=lambda c: -c["edge_match"])
    images = []
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for tag, img in (("before", A), ("during", B)):
            im = Image.fromarray(img.astype(np.uint8))
            dr = ImageDraw.Draw(im)
            for i, c in enumerate(scored):
                col = (255, 40, 40) if i == 0 else (60, 160, 255)
                dr.rectangle(c["bbox"], outline=col, width=5 if i == 0 else 3)
                dr.text((c["bbox"][0] + 6, max(0, c["bbox"][1] - 22)), f"{c['joint']} e={c['edge_match']}", fill=col)
            p = out / f"joint_id_{tag}.png"
            im.save(p)
            images.append(str(p))
    ranked = [{k: v for k, v in c.items() if k not in ("bid", "jid")} for c in scored]
    result = {"most_changed": scored[0]["joint"] if scored else None, "ranked": ranked, "images": images,
              "note": "edge_match = edge-map correlation between the real 'during' frame and a render with THAT joint opened "
                      "(higher is better). pixel_change is a prior only: a sweeping arm inflates it."}
    if out_dir:
        write_json(result, Path(out_dir) / "joint_id.json")
    return result


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(identify_moving_joint(a[0], a[1], a[2], a[3], out_dir=a[4] if len(a) > 4 else None), indent=1))
