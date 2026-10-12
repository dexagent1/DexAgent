"""Where on the table an object stands and how big it is, from one mask and the registered camera."""
from __future__ import annotations

TOOL = {
    "name": "measure_mask_footprint",
    "title": "Measure mask footprint",
    "type": "skill",
    "category": "Measurements",
    "summary": "Back-projects an object's video mask through the registered camera onto the tabletop: the contact line from the mask's bottom edge (where the object meets the table), its width in metres, the apparent height, and the footprint centroid and axes from the full mask, as initial size and pose for a constructor and the placement fit.",
    "description": (
        "Two readings of one silhouette. The bottom edge of the mask (lowest mask pixel per column, cleaned of hand pixels "
        "with ignore_mask) lies on the table when the object stands on it: back-projected onto z = 0 it gives the near "
        "contact line, its centre, its span in metres (the object's width across the view) and the depth of the object. "
        "The apparent height is the mask's vertical pixel extent above that contact line converted to metres at the "
        "object's depth, an overestimate for objects deeper than tall. The full mask back-projected onto the plane gives a "
        "footprint polygon whose area, centroid, principal axes and yaw are right for flat objects (boards, boxes seen from "
        "above, lids) and stretched away from the camera for tall ones; `height_guess_m` lifts the plane for that projection "
        "when known. Returns everything in the rig frame with the pixel box and a debug overlay. Not a measurement of hidden "
        "extents: the far side of a tall object is not visible, so use the width and the contact line as the reliable numbers."
    ),
    "inputs": [
        {"name": "mask", "type": "path | ndarray", "description": "Binary mask PNG (non-zero = object) at the video resolution."},
        {"name": "camera", "type": "path | dict", "description": "camera.json from register_table_camera or read_egodex_episode."},
        {"name": "ignore_mask", "type": "path | ndarray | None", "description": "Pixels to ignore (hands, arms)."},
        {"name": "height_guess_m", "type": "float", "description": "Plane height used for the full-mask footprint projection (half the object's height is a good guess). Default 0."},
        {"name": "out_png", "type": "path | None", "description": "Debug overlay: contact line and footprint drawn on the frame (frame needed)."},
        {"name": "frame", "type": "path | None", "description": "The video frame for the overlay."},
    ],
    "outputs": [
        {"name": "contact", "type": "dict", "description": "center_xy (rig), left_xy, right_xy, width_m, depth_m (distance along the optical axis), n_pixels."},
        {"name": "apparent_height_m", "type": "float", "description": "Mask height above the contact line in metres at the object's depth."},
        {"name": "footprint", "type": "dict", "description": "centroid_xy, length_m and width_m along the principal axes, yaw_deg of the long axis, area_m2, polygon_xy (convex hull) from the full-mask plane projection."},
        {"name": "bbox_px", "type": "list[4]", "description": "x0, y0, x1, y1 of the mask."},
    ],
    "status": "tested",
    "entrypoint": "measure_mask_footprint",
    "requirements": ["numpy", "opencv-python"],
    "tags": ["mask", "footprint", "placement", "size", "table plane", "stage 2"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.measure_mask_footprint mask.png camera.json --frame frame00000.jpg --out-png footprint.png",
}

import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common import rigcam as RC


def _mask(m):
    if isinstance(m, (str, Path)):
        import cv2
        a = cv2.imread(str(m), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(m)
        if a.ndim == 3:
            a = a[..., 0]
        return a > 0
    return np.asarray(m) > 0


def measure_mask_footprint(mask, camera, ignore_mask=None, height_guess_m=0.0, out_png=None, frame=None):
    import cv2
    cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
    M = _mask(mask)
    if ignore_mask is not None:
        M = M & ~_mask(ignore_mask)
    H, W = M.shape
    if (W, H) != (cam["width"], cam["height"]):
        raise ValueError(f"mask is {W}x{H} but the camera is {cam['width']}x{cam['height']}; rescale one of them")
    ys, xs = np.nonzero(M)
    if len(xs) < 20:
        raise ValueError("the mask has fewer than 20 pixels after removing the ignore mask")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    # contact line: the lowest rows of the mask lie on the table; their centre gives the depth
    bottom_band = ys >= y1 - max(3, int(0.05 * (y1 - y0 + 1)))
    bl = np.stack([xs[bottom_band], ys[bottom_band]], 1)
    P = RC.pixel_on_plane(cam, bl.astype(float), 0.0)
    centre = P.mean(axis=0)
    _, depth = RC.project(cam, centre[None])
    f = 0.5 * (cam["fx"] + cam["fy"])
    # width: the horizontal pixel span of the lower fifth of the mask, converted at the contact depth
    lower = ys >= y1 - max(4, int(0.2 * (y1 - y0 + 1)))
    span_px = float(xs[lower].max() - xs[lower].min() + 1)
    width = float(span_px * depth[0] / f)
    left = RC.pixel_on_plane(cam, np.array([[xs[lower].min(), y1]], float), 0.0)[0]
    right = RC.pixel_on_plane(cam, np.array([[xs[lower].max(), y1]], float), 0.0)[0]
    # apparent height: pixel extent above the contact rows at the object's depth
    px_h = float(y1 - y0)
    app_h = float(px_h * depth[0] / f)
    # full-mask footprint on a plane at height_guess_m
    step = max(1, int(np.sqrt(len(xs) / 4000)))
    try:
        F = RC.pixel_on_plane(cam, np.stack([xs[::step], ys[::step]], 1).astype(float), float(height_guess_m))
    except ValueError:
        F = P
    hull = cv2.convexHull(F[:, :2].astype(np.float32)).reshape(-1, 2)
    area = float(cv2.contourArea(hull.astype(np.float32)))
    cen = F[:, :2].mean(axis=0)
    cov = np.cov((F[:, :2] - cen).T)
    w, v = np.linalg.eigh(cov)
    long_axis = v[:, 1]
    proj_l = (F[:, :2] - cen) @ long_axis
    proj_w = (F[:, :2] - cen) @ v[:, 0]
    length = float(np.percentile(proj_l, 98) - np.percentile(proj_l, 2))
    wid = float(np.percentile(proj_w, 98) - np.percentile(proj_w, 2))
    yaw = float(np.degrees(np.arctan2(long_axis[1], long_axis[0])))
    out = {"contact": {"center_xy": [round(float(centre[0]), 4), round(float(centre[1]), 4)], "left_xy": [round(float(left[0]), 4), round(float(left[1]), 4)],
                       "right_xy": [round(float(right[0]), 4), round(float(right[1]), 4)], "width_m": round(width, 4), "depth_m": round(float(depth[0]), 4), "n_pixels": int(len(bl))},
           "apparent_height_m": round(app_h, 4),
           "footprint": {"centroid_xy": [round(float(cen[0]), 4), round(float(cen[1]), 4)], "length_m": round(length, 4), "width_m": round(wid, 4),
                         "yaw_deg": round(yaw, 2), "area_m2": round(area, 5), "polygon_xy": np.round(hull, 4).tolist(), "plane_height_m": float(height_guess_m)},
           "bbox_px": [x0, y0, x1, y1], "mask_pixels": int(len(xs))}
    if out_png and frame is not None:
        img = cv2.imread(str(frame))
        img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img
        ov = img.copy(); ov[M] = (0.5 * ov[M] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        for x, y in bl[::max(1, len(bl) // 400)]:
            cv2.circle(ov, (int(x), int(y)), 1, (0, 255, 0), -1)
        uv, _ = RC.project(cam, np.c_[hull, np.full(len(hull), float(height_guess_m))])
        cv2.polylines(ov, [uv.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        uv, _ = RC.project(cam, centre[None]); cv2.drawMarker(ov, tuple(int(v) for v in uv[0]), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(ov, f"contact ({centre[0]:.3f}, {centre[1]:.3f}) width {width:.3f} m  height~{app_h:.3f} m  footprint {length:.3f}x{wid:.3f} m yaw {yaw:.0f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), ov)
        out["overlay_png"] = str(out_png)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("mask"); ap.add_argument("camera"); ap.add_argument("--ignore-mask"); ap.add_argument("--height-guess-m", type=float, default=0.0)
    ap.add_argument("--out-png"); ap.add_argument("--frame")
    a = ap.parse_args(argv)
    r = measure_mask_footprint(a.mask, a.camera, a.ignore_mask, a.height_guess_m, a.out_png, a.frame)
    r["footprint"].pop("polygon_xy", None)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.exit(main())
