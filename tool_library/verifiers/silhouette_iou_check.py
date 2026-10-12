"""Silhouette IoU check: the rendered scene lands where the video shows the objects."""
from __future__ import annotations

TOOL = {
    "name": "silhouette_iou_check",
    "title": "Silhouette IoU check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Renders each object's silhouette from the video camera and compares it with the "
               "object's video mask; low IoU on one object means a wrong pose, on every object a "
               "wrong camera.",
    "description": (
        "The image-space proof that a reconstruction is registered to its footage. The scene is "
        "rendered with MuJoCo's segmentation renderer from the named camera at the resolution "
        "of the masks, each object's body (or the union of the bodies that make up a "
        "constructed object: the named body with its whole subtree, or every body matched by name prefix) becomes a silhouette, and its "
        "intersection over union with the given video mask is computed. An object below "
        "`min_iou` (0.25) fails; below `warn_iou` (0.45) is reported. When two or more objects "
        "are checked and all of them fail at once, the verifier reports `camera_suspect`: the "
        "camera they are all seen through is wrong, and re-fitting poses cannot help. The "
        "render and the masks must be the same aspect ratio, so the render is sized to the "
        "mask. Thin objects (a cord, a ribbon, a pen lying down: a mask whose skeleton is at least 8 times "
        "longer than the mask is wide, or the ids given in `thin`) are scored by their centrelines instead: "
        "a segmentation mask of a 3 mm cord is several times wider than the cord, which caps the plain IoU "
        "near 0.3 for a perfect placement. Both skeletons are widened to a common tolerance (half the wider "
        "of the two widths plus 2 px) and their IoU is the object's score; the plain IoU is reported too. A group object "
        "(several separate pieces under one mask: a tower of loose bars, a fan of cards) is given in `groups` as the list of "
        "its pieces' root bodies; its silhouette is the union of every piece with its subtree, and it is scored by plain "
        "IoU (its mask is several pieces, not one elongated object) unless it is listed in `thin`. An object declared "
        "hidden at frame 0 (given in `hidden`: a watch inside a closed box) is not scored by IoU: with the robot and "
        "ignore_mask pixels left out, its render must show at most 0.2% of the image and at most 10% (25 px floor) of its "
        "own silhouette drawn without the other objects, its frame-0 mask at most 0.2% of the image, and drawn alone it "
        "must be in the view (covered, not out of frame or under the table)."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "masks", "type": "dict[str, ndarray | str]", "description": "Object name -> binary mask (bool/uint8 array or PNG path), all the same size."},
        {"name": "camera", "type": "str | dict", "description": "Named camera in the scene, or a free camera {lookat, distance, azimuth, elevation}."},
        {"name": "qpos", "type": "ndarray | None", "description": "State to render; the initial state when omitted."},
        {"name": "min_iou", "type": "float", "description": "Failure threshold per object. Default 0.25."},
        {"name": "warn_iou", "type": "float", "description": "Warning threshold. Default 0.45."},
        {"name": "thin", "type": "str | list[str]", "description": "'auto' (default): elongated masks are scored by centreline; or the object ids to score so; [] for plain IoU everywhere."},
        {"name": "ignore_mask", "type": "ndarray | str | None", "description": "Pixels to leave out of every silhouette and mask (the demonstrator's hands over an object): a binary mask or PNG path, sized like the masks."},
        {"name": "ignore_robot", "type": "bool", "description": "Leave the pixels the robot covers at the rendered pose out of both silhouettes (a hand parked over an object in the ego view is not a placement error). Default true."},
        {"name": "groups", "type": "dict[str, list[str]] | None", "description": "Mask key -> body names of a group object's pieces (scene_recon `parts`): its silhouette is their union. Default None."},
        {"name": "hidden", "type": "list[str] | None", "description": "Mask keys of objects declared hidden at frame 0 (scene_recon `hidden_at_frame0`): instead of an IoU, their render and their frame-0 mask must show (near) zero visible pixels while the object, drawn alone, lies in the view. Default None."},
        {"name": "out_dir", "type": "str | None", "description": "If given, writes render.png and one overlay per object there."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; iou: {object: score} (the centreline IoU for thin objects); plain_iou; thin: {object: "
                        "{width_px, render_width_px, elongation, tolerance_px}}; hidden: {object: {visible_px, bound_px, "
                        "unoccluded_px, mask_px, ok, reason}}; failures; warnings; camera_suspect; "
                        "bodies_used: which bodies formed each silhouette; render_size."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "silhouette", "iou", "camera", "registration"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Open the Bottle", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "silhouette_iou_check",
}

from pathlib import Path

import numpy as np

from tool_library.common import scene as S


THIN_ELONGATION = 8.0       # skeleton length / mask width above which a mask is a thin object


def _centreline_score(sil, gt):
    """(score, info) of two silhouettes compared by centreline, or (None, info) when gt is not elongated."""
    import cv2
    from skimage.morphology import skeletonize
    sk_g = skeletonize(gt)
    n = int(sk_g.sum())
    if n < 10:
        return None, {}
    dt_g = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 3)
    w_g = max(1.0, 2.0 * float(np.median(dt_g[sk_g])))
    sk_s = skeletonize(sil) if sil.any() else np.zeros_like(sil)
    w_s = 0.0
    if sk_s.any():
        dt_s = cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 3)
        w_s = 2.0 * float(np.median(dt_s[sk_s]))
    r = int(round(0.5 * max(w_g, w_s))) + 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    a = cv2.dilate(sk_g.astype(np.uint8), k) > 0
    b = cv2.dilate(sk_s.astype(np.uint8), k) > 0
    info = {"width_px": round(w_g, 1), "render_width_px": round(w_s, 1), "elongation": round(n / w_g, 1), "tolerance_px": r}
    return float(S.iou(b, a)), info


def _object_bodies(model, obj, group=None):
    """Names of the bodies that make up a mask key's object (a group's pieces, else the named body's subtree)."""
    names = []
    for nm in (group or [obj]):
        b = S.resolve_body(model, nm)
        if b >= 0:
            names += [model.body(x).name for x in sorted(S.subtree(model, b))]
    return list(dict.fromkeys(names))


def _hidden_report(model, data, W, H, camera, parts, sil, gt, ignore):
    """A hidden-at-frame-0 object: what of it the camera sees at this state (robot and ignore pixels left out), what
    its frame-0 mask shows, and whether, drawn without the other objects, it lies in the view at all."""
    from tool_library.common import hidden_objects as HO
    own = set(parts)
    keep = {0} | {b for b in range(model.nbody) if model.body(b).name in own or model.body(b).name.startswith("table")}
    alone = S.render_body_masks(model, data, W, H, camera, hide_bodies=[b for b in range(model.nbody) if b not in keep])
    unocc = np.zeros((H, W), bool)
    for b in parts:
        if b in alone:
            unocc |= alone[b]
    visible, mask_px, n_unocc = int((sil & ~ignore).sum()), int((gt & ~ignore).sum()), int(unocc.sum())
    bound = HO.visibility_bound(H, W, n_unocc)
    rep = {"visible_px": visible, "bound_px": round(bound, 1), "unoccluded_px": n_unocc, "mask_px": mask_px, "ok": True}
    if n_unocc == 0:
        rep.update(ok=False, reason="drawn alone it renders no pixels: it is out of the camera's view or inside the table, not covered "
                                    "by another object; place it where the video reveals it")
    elif visible > bound:
        rep.update(ok=False, reason=f"declared hidden at frame 0 but {visible} px of it are visible in the render (at most {bound:.0f}): "
                                    "the object that covers it in the video does not cover it here (a lid open, a wall missing, a wrong "
                                    "pose), or it is not hidden and needs a frame-0 mask")
    elif mask_px > HO.MAX_FRAME_FRACTION * H * W:
        rep.update(ok=False, reason=f"declared hidden at frame 0 but its frame-0 mask shows {mask_px} px: an object the video shows at "
                                    "frame 0 is scored by silhouette IoU, not declared hidden")
    return rep


def check_silhouette_iou(scene, masks, camera, qpos=None, min_iou=0.25, warn_iou=0.45, out_dir=None, ignore_robot=True,
                         thin="auto", ignore_mask=None, groups=None, hidden=None):
    model = S.load_model(scene)
    data = S.forward(model, qpos)
    loaded = {k: S.load_mask(v) for k, v in masks.items()}
    H, W = next(iter(loaded.values())).shape
    for k, m in loaded.items():
        if m.shape != (H, W):
            loaded[k] = S.load_mask(m, (W, H))
    body_masks = S.render_body_masks(model, data, W, H, camera)
    # pixels the robot covers at this pose say nothing about the objects (the video has the human's hands there, or
    # nothing): they are left out of both silhouettes, so a hand parked over an object does not fail its placement
    robot = np.zeros((H, W), bool)
    if ignore_robot:
        for b, msk in body_masks.items():
            if b.startswith("robot_"):
                robot |= msk
    if ignore_mask is not None:
        robot |= S.load_mask(ignore_mask, (W, H))
    ious, used, failures, warnings, covered, plain, thin_info = {}, {}, [], [], {}, {}, {}
    groups = {str(k): [str(b) for b in v] for k, v in (groups or {}).items() if v}
    hidden, hidden_rep = {str(k) for k in (hidden or [])}, {}
    for obj, gt in loaded.items():
        if obj in hidden:
            # declared hidden at frame 0: every body of the object, drawn or not, and a visibility test instead of an IoU
            parts = _object_bodies(model, obj, groups.get(obj))
            if not parts:
                ious[obj], used[obj] = 0.0, []
                failures.append(obj)
                continue
            sil = np.zeros((H, W), bool)
            for b in parts:
                if b in body_masks:
                    sil |= body_masks[b]
            hidden_rep[obj] = _hidden_report(model, data, W, H, camera, parts, sil, gt, robot)
            used[obj] = parts
            if not hidden_rep[obj]["ok"]:
                failures.append(obj)
            continue
        bid = S.resolve_body(model, obj)
        parts = []
        if obj in groups:
            # a group object: the union of its separate pieces, each with everything under it
            for nm in groups[obj]:
                pb = S.resolve_body(model, nm)
                if pb >= 0:
                    parts += [model.body(b).name for b in sorted(S.subtree(model, pb)) if model.body(b).name in body_masks]
            parts = list(dict.fromkeys(parts))
        elif bid >= 0 and model.body(bid).name in body_masks:
            # the named body and everything hinged or slid on it: a box is its base and its lid
            parts = [model.body(b).name for b in sorted(S.subtree(model, bid)) if model.body(b).name in body_masks]
        else:
            s = S.slug(obj)
            parts = [b for b in body_masks if b == s or b.startswith(f"{s}_") or b.endswith(f"_{s}")]
        if not parts:
            ious[obj], used[obj] = 0.0, []
            failures.append(obj)
            continue
        sil = np.zeros((H, W), bool)
        for b in parts:
            sil |= body_masks[b]
        if robot.any():
            covered[obj] = round(float((gt & robot).sum() / max(gt.sum(), 1)), 3)
            sil = sil & ~robot; gt = gt & ~robot
        v = S.iou(sil, gt)
        plain[obj] = round(v, 3)
        if (thin == "auto" and obj not in groups) or obj in (thin or []):
            score, info = _centreline_score(sil, gt)
            if score is not None and (obj in (thin or []) or info["elongation"] >= THIN_ELONGATION):
                v = score; thin_info[obj] = info
        ious[obj], used[obj] = round(v, 3), parts
        if v < min_iou:
            failures.append(obj)
        elif v < warn_iou:
            warnings.append(obj)
        if out_dir:
            from PIL import Image
            o = Path(out_dir)
            o.mkdir(parents=True, exist_ok=True)
            ov = np.zeros((H, W, 3), np.uint8)
            ov[gt] = (0, 160, 0)
            ov[sil] = (200, 0, 0)
            ov[gt & sil] = (220, 200, 0)
            Image.fromarray(ov).save(o / f"overlay_{S.slug(obj)}.png")
    if out_dir:
        from PIL import Image
        Image.fromarray(S.render_rgb(model, data, W, H, camera)).save(Path(out_dir) / "render.png")
    vals = list(ious.values())
    camera_suspect = len(vals) >= 2 and max(vals) < min_iou
    return {"ok": not failures, "iou": ious, "plain_iou": plain, "thin": thin_info, "hidden": hidden_rep, "failures": failures, "warnings": warnings,
            "camera_suspect": camera_suspect, "bodies_used": used, "render_size": [W, H], "robot_covered_fraction": covered}


silhouette_iou_check = check_silhouette_iou   # the library calls every tool by its own name
