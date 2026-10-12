"""Fit an object's table pose to its video mask when it is seen THROUGH a clear part
(a transparent drawer front, a clear box wall, a glass): transparent geoms do not occlude.

    from tool_library.skills.fit_placement_through_transparent import fit_placement_through_transparent
    r = fit_placement_through_transparent("scene.xml", "plushie_plushie", "masks/plushie.png", "video_cam",
                                          half_range={"x": 0.03, "y": 0.03, "yaw": 40})
    r["pose"], r["iou"], r["n_transparent_geoms"]

The silhouette renderers of the library treat every visible geom as opaque, so an
object resting inside a clear-fronted container renders fully hidden and its IoU is
0 whatever its pose. This tool renders the segmentation through the scene's named
camera with every geom whose rgba alpha is below `alpha_max` hidden (clear plastic,
glass), so the object's own silhouette is compared with the mask the camera saw
through the clear part, while opaque parts still occlude it.
"""
from __future__ import annotations

TOOL = {
    "name": "fit_placement_through_transparent",
    "title": "Fit placement through transparent parts",
    "type": "skill",
    "category": "Measurements",
    "summary": "Grid-then-refine fit of a free body's x, y, yaw (and optionally z) to its video mask through a scene camera, with transparent geoms (alpha below a threshold) made non-occluding, for objects seen through clear drawer fronts, walls or glass.",
    "description": (
        "Compiles the scene, hides every geom whose rgba alpha < alpha_max (and that does not "
        "belong to the fitted body) by moving it to geom group 5 and disabling that group, and "
        "renders MuJoCo segmentation through `camera` at render_scale. The body's pose is "
        "searched on a coarse grid over the given half ranges around its initial pose (keyframe 0 when the scene has one) "
        "(x, y in metres, yaw in degrees, optional z), then refined by shrinking the grid "
        "around the best cell `n_refine` times; the score is the IoU between the body's "
        "visible pixels and the mask, with pixels of `ignore_mask` excluded from both. The "
        "body must be a free body (its pose is set through the free joint's qpos). With "
        "out_xml, the body's pos/quat attributes and its free joint's pose in every keyframe are rewritten in a copy of "
        "the scene XML text. Returns the best pose, its IoU, the starting IoU, the number of transparent "
        "geoms hidden and an overlay PNG when asked."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "path", "description": "Scene MJCF with the camera and the body."},
        {"name": "body", "type": "str", "description": "Free body to place."},
        {"name": "mask", "type": "path", "description": "Binary mask PNG at the camera resolution."},
        {"name": "camera", "type": "str", "description": "Camera name in the scene."},
        {"name": "ignore_mask", "type": "path | None", "description": "Pixels to ignore (hands)."},
        {"name": "half_range", "type": "dict", "description": "Search half widths {'x': m, 'y': m, 'yaw': deg, 'z'?: m}. Default x 0.03, y 0.03, yaw 30."},
        {"name": "alpha_max", "type": "float", "description": "Geoms with alpha below this do not occlude. Default 0.6."},
        {"name": "steps", "type": "int", "description": "Grid points per parameter per round. Default 7."},
        {"name": "n_refine", "type": "int", "description": "Refinement rounds. Default 3."},
        {"name": "render_scale", "type": "float", "description": "Render resolution factor. Default 0.5."},
        {"name": "out_xml", "type": "path | None", "description": "Write the scene with the fitted pose here."},
        {"name": "out_png", "type": "path | None", "description": "Overlay of mask (green) and render (red)."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "x, y, z, yaw_deg of the body in the world frame."},
        {"name": "iou", "type": "float", "description": "Best IoU."},
        {"name": "iou_start", "type": "float", "description": "IoU at the initial pose."},
        {"name": "n_transparent_geoms", "type": "int", "description": "Geoms made non-occluding."},
        {"name": "visible_px", "type": "int", "description": "Rendered body pixels at the best pose (full resolution equivalent)."},
        {"name": "out_xml", "type": "str | None", "description": "Written scene."},
        {"name": "overlay_png", "type": "str | None", "description": "Overlay image."},
    ],
    "status": "tested",
    "entrypoint": "fit_placement_through_transparent",
    "requirements": ["numpy", "mujoco", "Pillow"],
    "tags": ["placement", "transparent", "occlusion", "silhouette", "iou", "drawer", "glass"],
    "used_in": [],
    "cli": "python -m tool_library.skills.fit_placement_through_transparent scene.xml body mask.png camera",
}

import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def fit_placement_through_transparent(scene_xml, body, mask, camera, ignore_mask=None, half_range=None,
                                      alpha_max=0.6, steps=7, n_refine=3, render_scale=0.5,
                                      out_xml=None, out_png=None):
    import mujoco
    from PIL import Image
    hr = {"x": 0.03, "y": 0.03, "yaw": 30.0}
    if half_range:
        bad = set(half_range) - {"x", "y", "yaw", "z"}
        if bad:
            raise ValueError(f"half_range: unknown keys {sorted(bad)}; use x, y, yaw, z")
        hr = dict(half_range)
    if steps < 3 or n_refine < 0 or not (0 < render_scale <= 1):
        raise ValueError("need steps >= 3, n_refine >= 0, 0 < render_scale <= 1")
    m = mujoco.MjModel.from_xml_path(str(scene_xml)); d = mujoco.MjData(m)
    try:
        bid = m.body(body).id
    except Exception:
        raise ValueError(f"body {body!r} not in scene")
    ja = m.body_jntadr[bid]
    if ja < 0 or m.jnt_type[ja] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"body {body!r} must have a free joint")
    try:
        cid = m.camera(camera).id
    except Exception:
        raise ValueError(f"camera {camera!r} not in scene")
    qa = m.jnt_qposadr[ja]
    mk = np.array(Image.open(mask).convert("L")) > 127
    H, W = mk.shape
    h, w = max(8, int(round(H * render_scale))), max(8, int(round(W * render_scale)))
    ign = None
    if ignore_mask:
        ign = np.array(Image.open(ignore_mask).convert("L").resize((w, h), Image.NEAREST)) > 127
    mks = np.array(Image.fromarray(mk.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
    sub = set()
    stack = [bid]
    while stack:
        b = stack.pop(); sub.add(b)
        stack += [c for c in range(m.nbody) if m.body_parentid[c] == b and c != b]
    hidden = 0
    for g in range(m.ngeom):
        if m.geom_rgba[g][3] < alpha_max and m.geom_bodyid[g] not in sub:
            m.geom_group[g] = 5; hidden += 1
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w); m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    from tool_library.common.render import segmentation_renderer
    r = segmentation_renderer(m, h, w)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1; opt.geomgroup[5] = 0
    if m.nkey:                                      # the scene's initial state: a staged scene keeps poses in its keyframe
        mujoco.mj_resetDataKeyframe(m, d, 0)
    q0 = d.qpos.copy()
    x0, y0, z0 = q0[qa:qa + 3]
    qw, qx, qy, qz = q0[qa + 3:qa + 7]
    yaw0 = math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    def score(p):
        d.qpos[:] = q0
        d.qpos[qa:qa + 3] = [p["x"], p["y"], p.get("z", z0)]
        a = math.radians(p["yaw"]); d.qpos[qa + 3:qa + 7] = [math.cos(a / 2), 0, 0, math.sin(a / 2)]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cid, scene_option=opt)
        seg = r.render()
        objid, objtype = seg[..., 0], seg[..., 1]
        geom_b = np.full(m.ngeom + 1, -1); geom_b[:m.ngeom] = m.geom_bodyid
        ids = np.where(objtype == int(mujoco.mjtObj.mjOBJ_GEOM), objid, m.ngeom)
        ids = np.clip(ids, 0, m.ngeom)
        vis = np.isin(geom_b[ids], list(sub)) & (objtype == int(mujoco.mjtObj.mjOBJ_GEOM))
        a_, b_ = vis, mks
        if ign is not None:
            a_, b_ = vis & ~ign, mks & ~ign
        return _iou(a_, b_), vis

    centre = {"x": x0, "y": y0, "yaw": yaw0}
    if "z" in hr:
        centre["z"] = z0
    start_iou, _ = score(centre)
    best, best_s = dict(centre), start_iou
    span = dict(hr)
    for _ in range(n_refine + 1):
        keys = list(span)
        grids = [np.linspace(best[k] - span[k], best[k] + span[k], steps) for k in keys]
        c_best, c_s = best, best_s
        for vals in itertools.product(*grids):
            p = dict(zip(keys, vals))
            s, _ = score(p)
            if s > c_s:
                c_best, c_s = p, s
        best, best_s = dict(c_best), c_s
        span = {k: v * 2.0 / (steps - 1) for k, v in span.items()}
    s, vis = score(best)
    pose = {"x": float(best["x"]), "y": float(best["y"]), "z": float(best.get("z", z0)), "yaw_deg": float(best["yaw"])}
    res = {"pose": pose, "iou": round(s, 4), "iou_start": round(start_iou, 4), "n_transparent_geoms": hidden,
           "visible_px": int(vis.sum() / render_scale ** 2), "out_xml": None, "overlay_png": None}
    if out_xml:
        txt = Path(scene_xml).read_text()
        pat = re.compile(r'(<body\s+name="' + re.escape(body) + r'")([^>]*)>')
        mt = pat.search(txt)
        if not mt:
            raise ValueError(f"body {body!r}: tag not found in the XML text for out_xml")
        attrs = re.sub(r'\s(pos|quat|euler|axisangle|xyaxes|zaxis)="[^"]*"', "", mt.group(2))
        a = math.radians(pose["yaw_deg"])
        new = f'{mt.group(1)} pos="{pose["x"]:.6f} {pose["y"]:.6f} {pose["z"]:.6f}" quat="{math.cos(a / 2):.6f} 0 0 {math.sin(a / 2):.6f}"{attrs}>'
        txt = txt[:mt.start()] + new + txt[mt.end():]
        # keyframes hold the free joint's pose explicitly: write the fitted pose into every key
        new7 = [pose["x"], pose["y"], pose["z"], math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]

        def _patch_key(mk_):
            vals = mk_.group(2).split()
            if len(vals) > qa + 6:
                vals[qa:qa + 7] = [f"{v:.6g}" for v in new7]
            return f'{mk_.group(1)}qpos="{" ".join(vals)}"'
        txt = re.sub(r'(<key\b[^>]*?)qpos="([^"]*)"', _patch_key, txt)
        Path(out_xml).parent.mkdir(parents=True, exist_ok=True)
        Path(out_xml).write_text(txt)
        res["out_xml"] = str(out_xml)
    if out_png:
        ov = np.zeros((h, w, 3), np.uint8); ov[mks, 1] = 255; ov[vis, 0] = 255
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ov).save(out_png); res["overlay_png"] = str(out_png)
    r.close()
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(fit_placement_through_transparent(a[0], a[1], a[2], a[3]), indent=1))
