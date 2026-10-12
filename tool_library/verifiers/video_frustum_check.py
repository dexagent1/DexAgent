"""Video frustum check: every Stage 1 object is where the video camera can see it at the scene's initial state."""
from __future__ import annotations

TOOL = {
    "name": "video_frustum_check",
    "title": "Video frustum check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Projects every Stage 1 object's geometry at keyframe 0 through the video camera (camera.json or a scene camera) and requires it in front of the camera with its centre and most of its surface inside the image, so an object placed off-screen, behind the camera or under the table is caught.",
    "description": (
        "Every object Stage 1 found was seen in the video's first frame or later, so its reconstruction must stand "
        "where the video camera sees it; an object placed outside the frustum (a sign flip in a placement, a pose "
        "fitted against the wrong camera, an object dropped below the table) silently breaks Stage 3's targets and "
        "Stage 4's composite. Objects come from scene_semantics.json (mapped to bodies through scene_recon.json, else "
        "by id), or every object when no Stage 1 file is given. Each object's visible geoms (collision geoms when it "
        "has none) are sampled at keyframe 0 and projected through the camera: the fraction of points in front of the "
        "camera and inside the image shrunk by margin_px must be at least min_inside_frac, and the centroid must "
        "project inside the image. Objects of type deformable that Stage 1 lists are checked the same way on all "
        "their bodies. A Stage 1 object with no body in the scene is reported."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "camera", "type": "str | dict", "description": "camera.json path / dict, or the name of the scene's video camera (then give width and height). Default 'video_cam'."},
        {"name": "semantic", "type": "str | dict | None", "description": "scene_semantics.json (Stage 1); default: every object of the scene."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json mapping Stage 1 ids to bodies."},
        {"name": "width", "type": "int | None", "description": "Image width when camera is a scene camera name. Default 640."},
        {"name": "height", "type": "int | None", "description": "Image height when camera is a scene camera name. Default 480."},
        {"name": "min_inside_frac", "type": "float", "description": "Smallest fraction of an object's surface points inside the image. Default 0.5."},
        {"name": "margin_px", "type": "float", "description": "Border of the image not counted as inside, pixels. Default 0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; objects [{id, inside_frac, behind_frac, centroid_px, verdict}]; camera {width, height, fovy_deg}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "camera", "frustum", "visibility", "stage1", "stage2"],
    "status": "tested",
    "entrypoint": "video_frustum_check",
}

import mujoco
import numpy as np

from tool_library.common import rigcam as RC
from tool_library.common import scene_assert as SA


def video_frustum_check(scene, camera="video_cam", semantic=None, recon=None, width=None, height=None, min_inside_frac=0.5, margin_px=0.0):
    if not 0.0 < min_inside_frac <= 1.0:
        raise ValueError(f"min_inside_frac must be in (0, 1], got {min_inside_frac}")
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    try:
        cam = SA.load_camera(camera, model, width, height)
    except ValueError as e:
        return SA.report([str(e)], objects=[])
    W, H = int(cam["width"]), int(cam["height"])
    if semantic is not None:
        mapping = SA.stage1_groups(model, semantic, recon, ref=st["model"], cfg=st["cfg"])
        items = list(mapping.items())
    else:
        items = [(str(g["id"]), g) for g in SA.object_groups(model, recon, ref=st["model"], cfg=st["cfg"])]
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    rows, issues = [], []
    for oid, g in items:
        if g is None or g["root"] < 0:
            issues.append(f"Stage 1 object {oid} has no body in the scene; reconstruct it (or name its root_body in scene_recon.json)")
            rows.append({"id": oid, "verdict": "missing"})
            continue
        gs = SA.body_geoms(model, g["bodies"])
        use = [x for x in gs if SA.geom_visible(model, x)] or [x for x in gs if SA.geom_collidable(model, x)]
        pts = [SA.geom_world_points(model, data, x) for x in use]
        pts = [p for p in pts if len(p)]
        if not pts:
            issues.append(f"object {oid} has no geometry to project")
            rows.append({"id": oid, "verdict": "no geometry"})
            continue
        P = np.vstack(pts)
        uv, z = RC.project(cam, P)
        front = z > 1e-4
        inside = front & (uv[:, 0] >= margin_px) & (uv[:, 0] <= W - margin_px) & (uv[:, 1] >= margin_px) & (uv[:, 1] <= H - margin_px)
        c_uv, c_z = RC.project(cam, P.mean(axis=0)[None])
        c_in = bool(c_z[0] > 1e-4 and 0 <= c_uv[0, 0] <= W and 0 <= c_uv[0, 1] <= H)
        frac = float(inside.mean())
        verdict = "ok"
        if not front.any():
            verdict = "behind the camera"
        elif frac < min_inside_frac or not c_in:
            verdict = "outside the frame"
        if verdict != "ok":
            issues.append(f"object {oid} is {verdict} of the video camera at keyframe 0 ({frac * 100:.0f}% of its surface in the "
                          f"{W}x{H} image, centre at pixel {SA.r6(c_uv[0], 1)}); place it where the first frame shows it")
        rows.append({"id": oid, "inside_frac": round(frac, 3), "behind_frac": round(float((~front).mean()), 3),
                     "centroid_px": SA.r6(c_uv[0], 1), "verdict": verdict})
    return SA.report(issues, objects=rows, camera={"width": W, "height": H, "fovy_deg": round(float(cam.get("fovy_deg", 0.0)), 3)})
