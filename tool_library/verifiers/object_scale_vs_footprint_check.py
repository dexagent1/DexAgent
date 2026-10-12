"""Object scale vs footprint check: each object is as big as its silhouette in the video says."""
from __future__ import annotations

TOOL = {
    "name": "object_scale_vs_footprint_check",
    "title": "Object scale vs footprint check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Renders every object at keyframe 0 through the registered video camera and compares the extent of its silhouette (width, height, square root of area) with the extent of its video mask (or of a measure_mask_footprint result): a uniform scale error such as the 0.78 a free-scale placement fit once returned shows as all three ratios off together. An object the recon declares hidden_at_frame0 is skipped.",
    "description": (
        "Silhouette IoU can stay above 0.8 while an object is 20% too small or too large, and a size error changes "
        "every grasp aperture and every fit. The scene is rendered by segmentation through camera.json (robot "
        "hidden, the other objects kept so occlusions match); pixels of the hand mask (ignore_mask, masks/hands.png "
        "by default when a masks folder is given) are removed from both silhouettes. For each object with a mask, the "
        "2nd-98th percentile horizontal and vertical pixel spans and the square root of the pixel area of the "
        "rendered and the video silhouettes give three size ratios (model / video); a ratio is taken as an interval "
        "allowing slack_px of segmentation noise on each span, and spans of a mask cut by the image border are not "
        "used. The object's scale estimate is the median ratio; it fails when the whole interval lies outside "
        "ratio_band. A deformable simulated in Isaac Sim (scene_recon entry with simulator 'isaac' and isaac.cord_json) has no "
        "MuJoCo body: its silhouette is its settled cord drawn as a tube of its radius along the link centres "
        "(scene_assert.cord_tube_mask), judged the same way. With `footprints` (measure_mask_footprint outputs, {id: dict or json path}) the video spans are "
        "the footprint's bbox_px. An object mostly covered by the hand mask, out of view, or with fewer than two usable "
        "spans (from masks) is reported as not judged. A mask (or a mask file of a masks folder, hands.png and "
        "ignore_mask aside) that matches no object fails, and so does a call in which no mask or footprint matches an "
        "object at all. The metric width of the object across the view is reported too."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "camera", "type": "str | dict", "description": "camera.json path or dict (the video camera)."},
        {"name": "masks", "type": "dict | str | None", "description": "{object id: mask png} or a masks/ folder (files named by object id)."},
        {"name": "footprints", "type": "dict | None", "description": "{object id: measure_mask_footprint output (dict or json path)} instead of masks."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json mapping object ids to bodies (and masks when masks is None)."},
        {"name": "ignore_mask", "type": "str | None", "description": "Hand mask whose pixels are ignored in both silhouettes. Default: hands.png in the masks folder."},
        {"name": "ratio_band", "type": "list[float]", "description": "Accepted model/video size ratio. Default [0.8, 1.3]."},
        {"name": "slack_px", "type": "float", "description": "Segmentation noise allowed on each span, pixels. Default 2."},
        {"name": "max_side", "type": "int", "description": "Render at most this many pixels on the long side. Default 1280."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; objects [{id, ratios {width, height, sqrt_area}, scale, interval, model_width_m, verdict}]; worst (scale furthest from 1); unmatched_masks (mask files of a masks folder that name no object)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "scale", "size", "mask", "footprint", "camera", "stage2"],
    "status": "tested",
    "entrypoint": "object_scale_vs_footprint_check",
}

from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import rigcam as RC
from tool_library.common import scene_assert as SA


def _spans(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    w = float(np.percentile(xs, 98) - np.percentile(xs, 2) + 1)
    h = float(np.percentile(ys, 98) - np.percentile(ys, 2) + 1)
    H, W = m.shape
    return {"w": w, "h": h, "a": float(np.sqrt(len(xs))), "cut_x": bool(xs.min() <= 1 or xs.max() >= W - 2),
            "cut_y": bool(ys.min() <= 1 or ys.max() >= H - 2)}


def object_scale_vs_footprint_check(scene, camera, masks=None, footprints=None, recon=None, ignore_mask=None, ratio_band=(0.8, 1.3),
                                    slack_px=2.0, max_side=1280):
    if not ratio_band[0] < 1.0 < ratio_band[1]:
        raise ValueError(f"ratio_band must bracket 1, got {ratio_band}")
    if masks is None and footprints is None and recon is None:
        raise ValueError("give masks (or a masks folder), footprints, or a recon manifest naming the masks")
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    cam0 = SA.load_camera(camera)
    W0, H0 = int(cam0["width"]), int(cam0["height"])
    s = min(1.0, float(max_side) / max(W0, H0))
    W, H = int(round(W0 * s)), int(round(H0 * s))
    cam = RC.scaled(cam0, W, H) if s < 1.0 else cam0
    rec, rec_dir = SA.load_json(recon) if recon is not None else (None, None)
    groups = {str(g["id"]): g for g in SA.object_groups(model, rec, ref=st["model"], cfg=st["cfg"])}
    isaac = SA.isaac_only_ids(rec)
    from tool_library.common.hidden_objects import is_hidden
    hidden = {str(o.get("id")) for o in (rec or {}).get("objects", []) if is_hidden(o)}
    items, unmatched = {}, []
    mdir = None
    if footprints is not None:
        for k, v in footprints.items():
            fp, _ = SA.load_json(v)
            if "bbox_px" not in fp:
                raise ValueError(f"footprints['{k}'] is not a measure_mask_footprint result (no bbox_px)")
            items[str(k)] = ("bbox", fp["bbox_px"])
    elif isinstance(masks, dict):
        items = {str(k): ("mask", v) for k, v in masks.items()}
    else:
        if masks is not None:
            mdir = Path(masks)
            ign_p = Path(ignore_mask).resolve() if ignore_mask is not None else None
            for p in sorted(mdir.glob("*.png")):
                if p.stem in groups or p.stem in isaac or p.stem in hidden:
                    items[p.stem] = ("mask", p)
                elif p.name != "hands.png" and p.resolve() != ign_p:
                    unmatched.append(p.name)
        elif rec is not None:
            for o in rec.get("objects", []):
                if o.get("mask") and o.get("id") is not None:
                    p = (rec_dir / o["mask"]) if rec_dir is not None else Path(o["mask"])
                    items[str(o["id"])] = ("mask", p)
                    mdir = mdir or p.parent
    ign = None
    if ignore_mask is not None:
        ign = SA.load_mask(ignore_mask, (W, H))
    elif mdir is not None and (mdir / "hands.png").exists():
        ign = SA.load_mask(mdir / "hands.png", (W, H))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    robot = SA.robot_body_ids(model, st["model"])
    try:
        seg = SA.render_segmentation(model, data, cam, hide_bodies=robot)
    except ValueError as e:
        return SA.report([str(e)])
    gb = np.where(seg >= 0, model.geom_bodyid[np.clip(seg, 0, model.ngeom - 1)], -1)
    right = np.asarray(cam["R_cv"], float)[:, 0].copy(); right[2] = 0; right /= max(np.linalg.norm(right), 1e-9)
    rows, issues = [], []
    worst = 1.0
    if unmatched:
        issues.append(f"mask file(s) {unmatched} in {mdir} match no object of the scene (name each mask by its object id, "
                      f"as in scene_recon.json)")
    if not items:
        issues.append("no object mask or footprint matched an object of the scene: nothing was compared")
    entries = {str(o.get("id")): o for o in (rec or {}).get("objects", [])}
    for oid, (kind, src) in items.items():
        cord = None
        if oid in isaac and oid not in groups:
            # a deformable simulated in Isaac Sim: its settled cord, drawn as a tube of its radius along the link centres
            cord = SA.isaac_cord_spec(entries.get(oid) or {}, rec_dir)
            if cord is None or not (cord.get("link_poses") or {}).get("centres"):
                rows.append({"id": oid, "verdict": "simulated in Isaac Sim (no cord specification to draw)"})
                continue
        if oid in hidden:
            # declared hidden at source frame 0 (a watch in a closed box): there is no frame-0 footprint to compare
            rows.append({"id": oid, "verdict": "hidden at frame 0 (declared): no frame-0 footprint"})
            continue
        if cord is not None:
            lp = cord["link_poses"]
            rm = SA.cord_tube_mask(cam, lp["centres"], cord["radius"], cord["segment_m"], size=(W, H), quats=lp.get("quats"))
        else:
            g = groups.get(oid)
            if g is None or g["root"] < 0:
                issues.append(f"mask {oid} has no object in the scene (name the object as in scene_recon.json)")
                continue
            rm = np.isin(gb, list(g["bodies"]))
        full = int(rm.sum())
        if ign is not None:
            rm &= ~ign
        if full and rm.sum() < max(50, 0.3 * full):
            # the hands of frame 0 cover most of where the object is rendered (held in the hand): its extent cannot be read
            rows.append({"id": oid, "verdict": "occluded by the hands", "visible_frac": round(float(rm.sum()) / full, 3)})
            continue
        sr = _spans(rm)
        if kind == "mask":
            vm = SA.load_mask(src, (W, H))
            if ign is not None:
                vm &= ~ign
            sv = _spans(vm)
        else:
            x0, y0, x1, y1 = (float(v) * s for v in src)
            sv = {"w": x1 - x0, "h": y1 - y0, "a": None, "cut_x": x0 <= 1 or x1 >= W - 2, "cut_y": y0 <= 1 or y1 >= H - 2}
        if sr is None or sv is None:
            # nothing to compare: an object out of view is video_frustum_check's finding, not a size error
            rows.append({"id": oid, "verdict": "not visible in the render" if sr is None else "empty mask"})
            continue
        ratios, lo, hi = {}, [], []
        sl = slack_px * s if s < 1.0 else slack_px
        for key, name, cut in (("w", "width", sv["cut_x"] or sr["cut_x"]), ("h", "height", sv["cut_y"] or sr["cut_y"]),
                               ("a", "sqrt_area", sv["cut_x"] or sv["cut_y"] or sr["cut_x"] or sr["cut_y"])):
            if sv.get(key) is None or cut or sv[key] <= sl:
                continue
            r = sr[key] / sv[key]
            ratios[name] = round(r, 3)
            lo.append(max(sr[key] - sl, 0.0) / (sv[key] + sl)); hi.append((sr[key] + sl) / max(sv[key] - sl, 1e-6))
        if cord is not None:
            model_w = float(np.ptp(np.asarray(cord["link_poses"]["centres"], float) @ right)) + 2.0 * float(cord["radius"])
        else:
            P = np.vstack([p for p in (SA.geom_world_points(model, data, x) for x in SA.body_geoms(model, g["bodies"]) if SA.geom_visible(model, x)) if len(p)] or [np.zeros((1, 3))])
            model_w = float(np.ptp(P @ right))
        if not ratios:
            rows.append({"id": oid, "ratios": {}, "verdict": "cut by the image border", "model_width_m": round(model_w, 4)})
            continue
        scale = float(np.median(list(ratios.values())))
        ilo, ihi = float(np.median(lo)), float(np.median(hi))
        verdict = "ok"
        if len(ratios) < 2 and sv.get("a") is not None:
            # one span alone (the others cut by the border) is mostly the view's foreshortening, not the object's size
            rows.append({"id": oid, "ratios": ratios, "scale": round(scale, 3), "verdict": "too little visible to judge", "model_width_m": round(model_w, 4)})
            continue
        if ihi < ratio_band[0] or ilo > ratio_band[1]:
            verdict = "too small" if ihi < ratio_band[0] else "too large"
            worst = max(worst, scale, 1.0 / max(scale, 1e-9))
            issues.append(f"object {oid} is {verdict} for its video silhouette: model/video size {scale:.2f} (width {ratios.get('width')}, "
                          f"height {ratios.get('height')}, sqrt area {ratios.get('sqrt_area')}); rebuild it at {1.0 / scale:.2f}x its current size "
                          f"(take the size from the mask widths, not from a free-scale fit)")
        rows.append({"id": oid, "ratios": ratios, "scale": round(scale, 3), "interval": [round(ilo, 3), round(ihi, 3)],
                     "model_width_m": round(model_w, 4), "verdict": verdict})
    return SA.report(issues, objects=rows, worst=round(worst, 3), image=[W, H], unmatched_masks=unmatched)
