"""Task final state vs video gate: the scene's final state, seen through the video camera, against the video's last frame."""
from __future__ import annotations

TOOL = {
    "name": "task_final_state_vs_video_gate",
    "title": "Final state vs video gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Render each object's silhouette at the trajectory's final state through the video camera (a per-frame "
               "camera for a moving head camera) and compare it with the video's last-frame object mask (IoU per "
               "object); objects below min_iou ended somewhere else than in the demonstration.",
    "description": (
        "A whole-task check that needs no goal sentence: where the human left each object is where the robot should leave "
        "it, unless a declared deviation says otherwise. The scene is compiled with the video camera -- a camera named in "
        "the scene, a pinhole camera dict / JSON (width, height, fx, fy or fovy_deg, pos, quat_wxyz or R_cv, as Stage 2's "
        "camera.json), or a per-frame camera (a list indexed by video frame or {'frames': [{frame, pos, quat_wxyz, ...}]}) "
        "from which the camera of video_frame (default the last) is taken -- and rendered with MuJoCo's segmentation "
        "renderer at the masks' resolution on trajectory frame `frame` (default the last). Each object's silhouette is the "
        "union of the bodies in its subtree (scene_recon_json maps Stage 1 ids to root bodies; otherwise names resolve "
        "as exact, slug or a unique '_<name>' suffix). Pixels covered by the robot at that state (ignore_robot) and by "
        "occluder_mask (the human hands in the video frame) are removed from both silhouettes, since neither image shows "
        "the object there. IoU below min_iou fails the object and names it; between min_iou and warn_iou is a warning. "
        "Objects listed in `exclude` (an accepted deviation) are skipped; when no other mask names a body, nothing is "
        "judged and the gate fails (unless every mask was excluded, which is only noted). With out_dir it writes the render and one "
        "overlay per object. The camera's aspect ratio must match the masks'."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "masks", "type": "dict[str, str | ndarray] | str", "description": "Object -> binary mask of the video's last frame (PNG or array), or a folder of <object>.png."},
        {"name": "camera", "type": "str | dict | list", "description": "Scene camera name, camera dict / JSON, or per-frame camera list / JSON."},
        {"name": "video_frame", "type": "int | None", "description": "Video frame of a per-frame camera; default its last."},
        {"name": "frame", "type": "int", "description": "Trajectory frame to render. Default -1."},
        {"name": "min_iou", "type": "float", "description": "Failure threshold per object. Default 0.4."},
        {"name": "warn_iou", "type": "float", "description": "Warning threshold. Default 0.6."},
        {"name": "ignore_robot", "type": "bool", "description": "Drop the pixels the robot covers. Default true."},
        {"name": "occluder_mask", "type": "str | ndarray | None", "description": "Mask of video pixels that hide objects (hands); dropped from both."},
        {"name": "scene_recon_json", "type": "str | None", "description": "Stage 2 scene_recon.json for Stage 1 id -> bodies."},
        {"name": "exclude", "type": "list[str] | None", "description": "Objects not judged (accepted deviations)."},
        {"name": "out_dir", "type": "str | None", "description": "Write render.png and overlay_<object>.png here."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; notes; iou {object: value}; failures; warnings; bodies_used; unmatched_masks (folder masks that name no body, skipped); render_size; camera_frame; frame."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "video", "silhouette", "final state", "task gate"],
    "status": "tested",
    "entrypoint": "task_final_state_vs_video_gate",
}

import json
from pathlib import Path

import numpy as np

from tool_library.common import task_gates as G

_OCCLUDERS = ("hands", "hand", "arms", "arm", "person")


def task_final_state_vs_video_gate(scene, trajectory, masks, camera, video_frame=None, frame=-1, min_iou=0.4, warn_iou=0.6,
                                   ignore_robot=True, occluder_mask=None, scene_recon_json=None, exclude=None, out_dir=None,
                                   out_json=None):
    if not 0.0 <= float(min_iou) <= float(warn_iou) <= 1.0:
        raise ValueError("min_iou / warn_iou: 0 <= min_iou <= warn_iou <= 1")
    from_folder = isinstance(masks, (str, Path))
    if from_folder:
        d = Path(str(masks))
        if not d.is_dir():
            raise ValueError(f"masks: {masks} is not a folder (or pass a dict object -> mask)")
        files = sorted(d.glob("*.png"))
        masks = {f.stem: str(f) for f in files if f.stem not in _OCCLUDERS}
        if occluder_mask is None:
            occ = [f for f in files if f.stem in _OCCLUDERS]
            occluder_mask = str(occ[0]) if occ else None
    if not isinstance(masks, dict) or not masks:
        raise ValueError("masks: give at least one object mask")
    kind_cam = G.load_camera(camera, video_frame)
    ep, t, _ = G.open_window(scene, trajectory, frame=frame)
    loaded = {k: G.load_mask(v) for k, v in masks.items()}
    H, W = next(iter(loaded.values())).shape
    for k, m in loaded.items():
        if m.shape != (H, W):
            loaded[k] = G.load_mask(m, (W, H))
    if kind_cam[0] == "dict" and "width" in kind_cam[1] and "height" in kind_cam[1]:
        ca = float(kind_cam[1]["width"]) / float(kind_cam[1]["height"])
        if abs(ca - W / H) > 0.01 * ca:
            raise ValueError(f"masks: aspect {W}x{H} differs from the camera's {kind_cam[1]['width']}x{kind_cam[1]['height']}")
    occ = np.zeros((H, W), bool) if occluder_mask is None else G.load_mask(occluder_mask, (W, H))
    sil, rmodel = G.render_silhouettes(ep.scene_path, ep.Q[t], kind_cam, W, H)
    robot = np.zeros((H, W), bool)
    if ignore_robot:
        for b, msk in sil.items():
            if b.startswith("robot_"):
                robot |= msk
    drop = robot | occ
    recon = G.recon_objects(G.load_json(scene_recon_json, "scene_recon_json")) if scene_recon_json else {}
    skip = set(exclude or [])
    ious, used, failures, warnings, issues, unmatched = {}, {}, [], [], [], []
    for obj, gt in loaded.items():
        if obj in skip:
            continue
        names = []
        r = recon.get(obj)
        roots = ([r["root"]] if r and r.get("root") else []) + (r["bodies"] if r else [])
        for nm in roots or [obj]:
            try:
                names += [ep.name(b) for b in ep.subtree(ep.body(nm, "masks"))]
            except ValueError:
                pass
        if not names:
            if from_folder:                       # a folder may hold extra masks (a whole-bottle mask beside its parts)
                unmatched.append(obj)
                continue
            raise ValueError(f"masks: object {obj!r} names no body in the scene (pass scene_recon_json for Stage 1 ids)")
        names = sorted(set(names))
        rend = np.zeros((H, W), bool)
        for nm in names:
            if nm in sil:
                rend |= sil[nm]
        a, b = rend & ~drop, gt & ~drop
        inter, union = float((a & b).sum()), float((a | b).sum())
        iou = inter / union if union > 0 else 0.0
        ious[obj] = round(iou, 3)
        used[obj] = names
        if iou < float(min_iou):
            failures.append(obj)
            if not a.any():
                why = "it is not visible from the camera in the final state (moved out of view or hidden)"
            elif not b.any():
                why = "the video shows it nowhere in that frame (or it is fully occluded)"
            else:
                ca_, cb_ = np.argwhere(a).mean(0), np.argwhere(b).mean(0)
                why = f"its silhouette centre is {np.linalg.norm(ca_ - cb_):.0f} px from the video's"
            issues.append(f"{obj} ends somewhere else than in the demonstration's last frame (IoU {iou:.2f} < {min_iou}; {why}): "
                          "place it where the video leaves it, or declare the deviation")
        elif iou < float(warn_iou):
            warnings.append(obj)
    if out_dir:
        from PIL import Image
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        seg = np.zeros((H, W, 3), np.uint8)
        for k, (nm, msk) in enumerate(sorted(sil.items())):
            seg[msk] = [(53 * k) % 255, (97 * k) % 255, (151 * k) % 255]
        Image.fromarray(seg).save(od / "render.png")
        for obj in ious:
            rend = np.zeros((H, W), bool)
            for nm in used[obj]:
                if nm in sil:
                    rend |= sil[nm]
            ov = np.zeros((H, W, 3), np.uint8)
            ov[rend] = [255, 80, 80]
            ov[loaded[obj]] = ov[loaded[obj]] // 2 + np.array([0, 120, 255], np.uint8) // 2
            ov[drop] = [60, 60, 60]
            Image.fromarray(ov).save(od / f"overlay_{obj}.png")
    if not ious and not skip:
        raise ValueError("masks: no mask names an object of the scene (pass scene_recon_json for Stage 1 ids)")
    notes = []
    if not ious:                                  # everything left after `exclude` named no body: nothing was judged
        if all(obj in skip for obj in loaded):
            notes.append("every mask is excluded: nothing was judged")
        else:
            issues.append(f"no mask outside `exclude` names a body of the scene ({unmatched[:5]}): nothing was judged; "
                          "pass scene_recon_json for Stage 1 ids")
    rep = {"ok": not failures and not issues, "issues": issues, "notes": notes, "iou": ious, "failures": failures, "warnings": warnings, "bodies_used": used,
           "unmatched_masks": unmatched,
           "render_size": [W, H], "frame": int(t), "camera_frame": kind_cam[1].get("frame") if kind_cam[0] == "dict" else None,
           "excluded": sorted(skip)}
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import sys
    print(json.dumps(task_final_state_vs_video_gate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
