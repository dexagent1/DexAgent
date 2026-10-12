"""Measure a box-shaped object in metres from its silhouette on a known table plane.

    from tool_library.skills.measure_box_on_table import measure_box_on_table
    res = measure_box_on_table(["masks/box_f000.png", "masks/box_f120.png"], "camera.json")
    res["length_m"], res["width_m"], res["views"][0]["height_m"], res["views"][0]["pos_rig"], res["mean_iou"]

    from tool_library.common.tablecam import camera            # or a table camera and ndarray masks
    cam = camera(focal_px=870, width=640, height=480, h_above_table_m=0.684, pitch_deg=35.9)
    res = measure_box_on_table([{"mask": mask_a}, {"mask": mask_b, "h_bounds": [0.02, 0.1]}], cam, fit_pitch=True)

The route for objects a 3D reconstructor cannot handle, transparent ones
above all, and a cheap cross-check for the rest: a rounded-rectangular
prism is fitted to the object's own outline, on the plane it rests on. Two
views at different yaws make the camera pitch identifiable (the depth axis
is foreshortened by 1/sin(pitch); a 3 degree error is 8% on one side of the
box and none on the other), so the pitch is refitted jointly. The per-view
silhouette IoU says whether to believe the millimetres.
"""
from __future__ import annotations

TOOL = {
    "name": "measure_box_on_table",
    "title": "Measure box on table",
    "type": "skill",
    "category": "Measurements",
    "summary": "Fit a rounded-rectangular prism (length, width, height, corner radius, pose) to one or more silhouettes of an object resting on a known table plane, refitting the camera pitch across views.",
    "description": (
        "Given a table camera (focal, image size, height above the table, pitch, roll) and "
        "one or more binary silhouettes of the same object at different yaws, optimizes "
        "[pitch, length, width, corner radius] plus per view [height, outward growth, forward, "
        "lateral, yaw] with Nelder-Mead restarts to maximize the IoU between the projected "
        "prism's convex hull and each silhouette. share_footprint ties length/width/radius "
        "across views (one object), heights may differ (open vs closed); h_bounds per view "
        "constrain the height, which the silhouette of a transparent object fixes weakly. "
        "Returns the dimensions, each view's pose on the table (forward, left, yaw), the "
        "refitted pitch and the per-view IoU. The search runs per view first (yaw grid, then a "
        "scaled Powell search) and jointly after. On synthetic silhouettes of a 20 x 14 x 6 cm "
        "box at 0.8 m the footprint comes back within about 1 cm; height and pitch trade off "
        "against each other under perspective, so seed the pitch from a calibration when one "
        "exists and read the IoU before trusting the millimetres. silhouette_from_edges in "
        "common/tablecam.py extracts the outline of a dark-edged object on a light surface. "
        "Masks may be PNG paths (non-zero = object) or arrays; the camera may be the Stage 2 "
        "camera.json (rig frame), its dict, or a common.tablecam camera. With a rig camera every "
        "view's pose is also returned in the rig frame (pos_rig of the box's bottom centre on the "
        "table, yaw_rig_deg), ready for placing the object. One view cannot fit the pitch: "
        "fit_pitch defaults to True with two or more views and False (trust the camera) with one."
    ),
    "inputs": [
        {"name": "views", "type": "list[str | {mask: str | ndarray [H, W], init?: dict, h_bounds?: [lo, hi], label?: str}]",
         "description": "Silhouettes of the one object, one per view (frame): a mask PNG path (non-zero pixels = object, as "
                        "segment_with_text_prompts writes them), or a dict with that path or a bool/uint8 array under 'mask', which may "
                        "also carry a per-view 'init' {height, yaw_deg, fwd, lat} (m, deg), 'h_bounds' [lo, hi] height bounds in m "
                        "([0.005, 0.40] when left out) and a 'label'. All masks must have the same size."},
        {"name": "camera", "type": "str | dict",
         "description": "The camera the masks were seen through: the Stage 2 camera.json path or its dict (rig frame: fx, fy, R_cv, "
                        "pos; then each view's rig-frame pose is returned too), or a common.tablecam.camera dict (focal_px, width, "
                        "height, h_above_table_m, pitch_deg, roll_deg). Its pitch is the starting pitch. A camera of another "
                        "resolution than the masks is rescaled to them."},
        {"name": "share_footprint", "type": "bool",
         "description": "One object seen in every view: length, width and corner radius are shared, heights may differ (open vs "
                        "closed). False fits each view's footprint on its own. Default True."},
        {"name": "fit_pitch", "type": "bool | None",
         "description": "Refit the camera pitch jointly with the box (needs two or more views at different yaws: one view cannot "
                        "separate a foreshortening error from a shorter box). Default None = True with two or more views, False "
                        "with one; True with one view raises."},
        {"name": "init", "type": "dict | None", "description": "Starting length, width, height, corner_r in metres. Default 0.20, 0.15, 0.06, 0.02."},
        {"name": "restarts", "type": "int", "description": "Random restarts of the joint search. Default 12."},
        {"name": "seed", "type": "int", "description": "Random seed of the restarts. Default 1."},
    ],
    "outputs": [
        {"name": "fit", "type": "dict", "description": "length_m, width_m, corner_r_m, pitch_deg, pitch_deg_start, pitch_fitted, mean_iou, views: [label, height_m, grow_m, fwd_m, left_m, yaw_deg, silhouette_iou, and with a rig camera pos_rig (bottom centre on the table) and yaw_rig_deg], x (raw parameters), camera_frame ('rig' | 'table')."},
    ],
    "requirements": ["numpy", "scipy", "opencv-python"],
    "tags": ["measurement", "silhouette", "box", "transparent", "table plane"],
    "used_in": ["Place Toy in Drawer", "Giftbox Packing", "Meal Box (extra sample)"],
    "status": "tested",
    "entrypoint": "measure_box_on_table",
}

import numpy as np

from tool_library.common import tablecam as TC


def prism_points(L, W, H, rad, fwd, lat, yaw_deg, n_corner=6):
    rad = float(np.clip(rad, 1e-4, min(L, W) / 2 - 1e-4))
    a, b = L / 2 - rad, W / 2 - rad
    pts = []
    for (cx, cy, t0) in ((a, b, 0.0), (-a, b, 90.0), (-a, -b, 180.0), (a, -b, 270.0)):
        for t in np.linspace(t0, t0 + 90.0, n_corner):
            pts.append((cx + rad * np.cos(np.radians(t)), cy + rad * np.sin(np.radians(t))))
    q = np.asarray(pts)
    th = np.radians(float(yaw_deg)); c, s = np.cos(th), np.sin(th)
    q = q @ np.array([[c, s], [-s, c]]) + np.array([fwd, lat])
    return np.concatenate([np.c_[q, np.zeros(len(q))], np.c_[q, np.full(len(q), H)]])


def hull_mask(uv, shape):
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.int32(np.round(cv2.convexHull(np.float32(uv)))), 1)
    return m.astype(bool)


def _iou(a, b):
    return float((a & b).sum()) / max(int((a | b).sum()), 1)


def _load_mask(m):
    """A mask PNG/JPG/NPY path or an array -> bool [H, W] (non-zero = object; a colour image is reduced over its
    colour channels, an alpha channel ignored)."""
    from pathlib import Path
    if isinstance(m, (str, Path)):
        p = Path(m)
        if not p.exists():
            raise ValueError(f"views: mask file {p} does not exist")
        if p.suffix == ".npy":
            m = np.load(p)
        else:
            from PIL import Image
            m = np.asarray(Image.open(p))
    a = np.asarray(m)
    if a.ndim == 3:
        a = a[..., :3].max(axis=2) if a.shape[2] >= 3 else a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"views: a mask must be a 2-D image, got shape {a.shape}")
    return a > (127 if a.dtype == np.uint8 and a.max() > 1 else 0)


def _shift(m, du, dv):
    """m moved by (du, dv) pixels, zero-filled (no wrap-around)."""
    out = np.zeros_like(m)
    H, W = m.shape
    ys, yd = (slice(0, H - dv), slice(dv, H)) if dv >= 0 else (slice(-dv, H), slice(0, H + dv))
    xs, xd = (slice(0, W - du), slice(du, W)) if du >= 0 else (slice(-du, W), slice(0, W + du))
    out[yd, xd] = m[ys, xs]
    return out


def _as_tablecam(camera, shape):
    """(common.tablecam camera, rig frame map or None) from a camera.json path / rig camera dict / table camera."""
    from pathlib import Path
    from tool_library.common import rigcam as RC
    if isinstance(camera, (str, Path)):
        if not Path(camera).exists():
            raise ValueError(f"camera: {camera} does not exist (give the Stage 2 camera.json)")
        camera = RC.load(camera)
    if isinstance(camera, (tuple, list)) and len(camera) == 2 and isinstance(camera[0], dict):
        camera = camera[0]                      # the (camera, frame_map) pair rigcam.to_tablecam returns
    if not isinstance(camera, dict):
        raise ValueError(f"camera: give a camera.json path, a rig camera dict or a common.tablecam camera, got {type(camera).__name__}")
    H, W = shape
    if "R_cv" in camera:
        cam = camera
        if (int(cam["width"]), int(cam["height"])) != (W, H):
            if abs(cam["width"] / cam["height"] - W / H) > 0.01:
                raise ValueError(f"camera: {cam['width']}x{cam['height']} has another aspect ratio than the {W}x{H} masks")
            cam = RC.scaled(cam, W, H)
        off = (float(cam["cx"]) - W / 2.0, float(cam["cy"]) - H / 2.0)
        tc, fmap = RC.to_tablecam(cam)
        return tc, fmap, off
    need = ("focal_px", "width", "height", "h_above_table_m", "pitch_deg")
    if not all(k in camera for k in need):
        raise ValueError(f"camera: a table camera needs {need} (common.tablecam.camera), a rig camera fx, fy, cx, cy, R_cv, pos; "
                         f"got keys {sorted(camera)[:12]}")
    if (int(camera["width"]), int(camera["height"])) != (W, H):
        raise ValueError(f"camera: the table camera is {camera['width']}x{camera['height']} but the masks are {W}x{H}")
    tc = dict(camera)
    tc.setdefault("roll_deg", 0.0)
    return tc, None, (0.0, 0.0)


def measure_box_on_table(views, camera, share_footprint=True, fit_pitch=None, init=None, restarts=12, seed=1):
    from scipy.optimize import minimize
    if isinstance(views, (str, dict)) or (hasattr(views, "ndim") and getattr(views, "ndim", 0) == 2):
        views = [views]                         # one view given bare
    views = [v if isinstance(v, dict) else {"mask": v} for v in views]
    if not views or any("mask" not in v for v in views):
        raise ValueError("views: give one entry per view, a mask path or {'mask': path | array, ...}")
    if fit_pitch is None:
        fit_pitch = len(views) >= 2
    if fit_pitch and len(views) < 2:
        raise ValueError("one view cannot separate a foreshortening error from a shorter box: give two views "
                         "at different yaws, or fit_pitch=False to trust the camera's pitch")
    masks = [_load_mask(v["mask"]) for v in views]
    shape, n = masks[0].shape, len(views)
    if any(m.shape != shape for m in masks):
        raise ValueError(f"views: masks of different sizes {[m.shape for m in masks]}; give masks of one video")
    for i, m in enumerate(masks):
        if m.sum() < 20:
            raise ValueError(f"views[{i}]: the mask has {int(m.sum())} object pixels; the object is not in it")
    camera, frame_map, pp_off = _as_tablecam(camera, shape)
    du, dv = int(round(pp_off[0])), int(round(pp_off[1]))
    if du or dv:
        # the table camera has its principal point at the image centre; moving each mask by the camera.json offset
        # puts every pixel on the ray it was seen along
        masks = [_shift(m, -du, -dv) for m in masks]
    p0 = float(camera["pitch_deg"])
    ini = dict(length=0.20, width=0.15, height=0.06, corner_r=0.02); ini.update(init or {})
    hlo = np.array([float((v.get("h_bounds") or (0.005, 0.40))[0]) for v in views])
    hhi = np.array([float((v.get("h_bounds") or (0.005, 0.40))[1]) for v in views])
    # a view's starting position: the back-projected centroid of its mask
    starts = []
    for m in masks:
        ys, xs = np.where(m)
        c = TC.plane_points([[xs.mean(), ys.mean()]], camera)[0]
        starts.append((float(c[0]), float(c[1])))

    def unpack(x):
        pitch, L, W, rad = float(x[0]), x[1], x[2], x[3]
        per = x[4:].reshape(n, 5) if share_footprint else x[4:].reshape(n, 8)
        return pitch, L, W, rad, per

    def masks_of(x):
        pitch, L, W, rad, per = unpack(x)
        cam = TC.camera(camera["focal_px"], camera["width"], camera["height"], camera["h_above_table_m"], pitch, camera["roll_deg"])
        out = []
        for i in range(n):
            if share_footprint:
                H, grow, fwd, lat, yaw = per[i]; Li, Wi, ri = L, W, rad
            else:
                H, grow, fwd, lat, yaw, Li, Wi, ri = per[i]
            out.append(hull_mask(TC.project(prism_points(Li + 2 * grow, Wi + 2 * grow, H, ri + grow, fwd, lat, yaw), cam), shape))
        return out

    def cost(x):
        pitch, L, W, rad, per = unpack(x)
        dims = [(L, W, rad)] if share_footprint else [tuple(r[5:8]) for r in per]
        for Li, Wi, ri in dims:
            if not (0.02 < Li < 0.80 and 0.02 < Wi < 0.80 and Wi <= Li and 0.001 < ri < min(Li, Wi) / 2):
                return 10.0
        if not (10 < pitch < 80) or (not fit_pitch and abs(pitch - p0) > 1e-9):
            return 10.0
        if not np.all((per[:, 0] > hlo) & (per[:, 0] < hhi) & (per[:, 1] >= -1e-6) & (per[:, 1] < 0.005)):
            return 10.0
        try:
            return float(n) - sum(_iou(a, b) for a, b in zip(masks_of(x), masks))
        except ValueError:
            return 10.0

    # Stage 1, each view on its own with the pitch fixed: a yaw grid (the local
    # search cannot cross a 90 degree yaw error) then a scaled Powell search
    # over the view's eight parameters from the best three yaws.
    def view_cost(i, y):
        H, grow, fwd, lat, yaw, L, W, rad = y
        if not (0.02 < L < 0.80 and 0.02 < W < 0.80 and 0.001 < rad < min(L, W) / 2 and hlo[i] < H < hhi[i]
                and -1e-6 <= grow < 0.005):
            return 10.0
        try:
            return 1.0 - _iou(hull_mask(TC.project(prism_points(L + 2 * grow, W + 2 * grow, H, rad + grow, fwd, lat, yaw), camera), shape), masks[i])
        except ValueError:
            return 10.0

    vstep = np.array([.006, .002, .025, .025, 5.0, .015, .015, .006])
    per_view = []
    for i in range(n):
        vi = dict(ini, **(views[i].get("init") or {}))
        fwd0, lat0 = vi.get("fwd", starts[i][0]), vi.get("lat", starts[i][1])
        grid = []
        for yaw in ([vi["yaw_deg"]] if "yaw_deg" in vi else np.arange(0.0, 180.0, 15.0)):
            y = np.array([vi["height"], 0.0, fwd0, lat0, yaw, vi["length"], vi["width"], vi["corner_r"]])
            grid.append((view_cost(i, y), y))
        grid.sort(key=lambda g: g[0])
        bv = None
        for c0, y in grid[:3]:
            r = minimize(lambda z: view_cost(i, y + z * vstep), np.zeros(8), method="Powell",
                         options=dict(maxiter=4000, maxfev=4000, xtol=1e-3, ftol=1e-6))
            if bv is None or r.fun < bv[0]:
                bv = (float(r.fun), y + r.x * vstep)
        y = bv[1]
        if y[6] > y[5]:                              # keep the long side as the length
            y[5], y[6], y[4] = y[6], y[5], (y[4] + 90.0) % 180.0
        per_view.append(y)

    # Stage 2, all views together with the pitch free and the footprint shared.
    x0 = [p0, float(np.mean([y[5] for y in per_view])), float(np.mean([y[6] for y in per_view])),
          float(np.mean([y[7] for y in per_view]))]
    for y in per_view:
        x0 += list(y[:5]) + ([] if share_footprint else list(y[5:8]))
    x0 = np.asarray(x0, float)
    step = np.asarray([2.0, .015, .015, .006] + ([.006, .002, .025, .025, 5.0] + ([] if share_footprint else [.015, .015, .006])) * n)

    def cost_n(z):
        return cost(x0 + z * step)

    best = (cost(x0), x0)
    rng = np.random.default_rng(seed)
    for k in range(int(restarts)):
        z0 = (best[1] - x0) / step + (rng.normal(0, 0.5, x0.size) if k else 0.0)
        if not fit_pitch:
            z0[0] = 0.0
        for method in ("Powell", "Nelder-Mead"):
            r = minimize(cost_n, z0, method=method, options=dict(maxiter=6000, maxfev=6000, xtol=1e-3, ftol=1e-7)
                         if method == "Powell" else dict(maxiter=6000, maxfev=6000, xatol=1e-4, fatol=1e-7))
            z0 = r.x
            if r.fun < best[0]:
                best = (float(r.fun), x0 + r.x * step)
    pitch, L, W, rad, per = unpack(best[1])
    ious = [_iou(a, b) for a, b in zip(masks_of(best[1]), masks)]
    out = {"length_m": round(float(L), 5), "width_m": round(float(W), 5), "corner_r_m": round(float(rad), 5),
           "pitch_deg": round(pitch, 3), "pitch_deg_start": p0, "pitch_fitted": bool(fit_pitch),
           "mean_iou": round(float(np.mean(ious)), 4), "views": [], "x": [float(v) for v in best[1]],
           "camera_frame": "rig" if frame_map is not None else "table"}
    if frame_map is not None:
        R_tr = np.asarray(frame_map["R_tr"])
        rig_yaw0 = float(np.degrees(np.arctan2(R_tr[1, 0], R_tr[0, 0])))      # the table frame's forward axis in the rig
    for i in range(n):
        row = {"label": views[i].get("label", f"view{i}"), "height_m": round(float(per[i, 0]), 5),
               "grow_m": round(float(per[i, 1]), 5), "fwd_m": round(float(per[i, 2]), 4), "left_m": round(float(per[i, 3]), 4),
               "yaw_deg": round(float(per[i, 4] % 180.0), 2), "silhouette_iou": round(ious[i], 4)}
        if not share_footprint:
            row.update(length_m=round(float(per[i, 5]), 5), width_m=round(float(per[i, 6]), 5), corner_r_m=round(float(per[i, 7]), 5))
        if frame_map is not None:
            from tool_library.common import rigcam as RC
            row["pos_rig"] = [round(float(v), 4) for v in RC.table_to_rig(frame_map, [[per[i, 2], per[i, 3], 0.0]])[0]]
            row["yaw_rig_deg"] = round(float((per[i, 4] + rig_yaw0) % 180.0), 2)
        out["views"].append(row)
    return out
