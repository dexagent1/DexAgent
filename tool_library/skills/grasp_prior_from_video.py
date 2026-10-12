"""Human grasp prior: from which side, at what height and with which hand the human took the object."""
from __future__ import annotations

TOOL = {
    "name": "grasp_prior_from_video",
    "title": "Grasp prior from video",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Read the human's grasp off the video at the moment of contact: the approach direction on the table (from "
               "the hand mask toward the object mask, back-projected through the calibrated camera), whether the hand "
               "came over the top or from the side, the hand's lateral offset, and a ranked list of approach "
               "directions to give the grasp skills or to filter their candidates with.",
    "description": (
        "A human prior for pose selection. Given the object's mask (its footprint on the table at frame 0 or at the "
        "contact frame) and the acting hand's mask at the frame where the hand closes on the object (both at the "
        "video resolution; segment_with_text_prompts produces them), the skill back-projects both masks onto the "
        "table plane through camera.json, takes the object's footprint centroid and the hand's footprint centroid "
        "(the hand mask's lowest rows, where the fingers are, projected at the object's height), and reports the unit "
        "approach vector from the hand toward the object in the rig frame, the hand-object distance, the fraction of "
        "the hand mask that lies inside the object's silhouette (a large fraction means the hand came over the top), "
        "and the hand's angular position around the object. From these it ranks the eight horizontal approach "
        "directions plus 'top' and returns them as `approach_dirs` (the format pose_from_points / handle_pinch_pose "
        "take) and as a `family` hint (side wrap, top, pinch) for generate_rigid_grasps candidates: keep the "
        "candidates whose `approach` points within `tolerance_deg` of the human's. The prior is coarse (a hand mask on "
        "a table plane), which is what it is for: picking among verified candidates, never replacing the stress tests."
    ),
    "inputs": [
        {"name": "camera", "type": "str | dict", "description": "camera.json (rigcam) of the video camera."},
        {"name": "object_mask", "type": "str", "description": "Binary PNG of the object at the video resolution."},
        {"name": "hand_mask", "type": "str", "description": "Binary PNG of the acting hand at the contact frame (the frame where fingers close on the object)."},
        {"name": "object_height_m", "type": "float", "description": "Height at which the hand's footprint is projected (the grasp height on the object). Default 0.05."},
        {"name": "tolerance_deg", "type": "float", "description": "Angular tolerance for ranking approach directions. Default 45."},
        {"name": "top_fraction", "type": "float", "description": "Hand-inside-object silhouette fraction above which the grasp is read as from the top. Default 0.5."},
    ],
    "outputs": [
        {"name": "prior", "type": "dict", "description": "approach_xy (unit vector hand -> object), approach_angle_deg, distance_m, hand_over_object_fraction, from_top (bool), family hint, approach_dirs (ranked unit vectors incl. [0,0,1] when from the top), hand_centroid_xy, object_centroid_xy."},
    ],
    "requirements": ["numpy", "opencv-python"],
    "simulators": [],
    "tags": ["stage3", "human prior", "grasp", "approach", "video"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Place Cup in Bowl", "Multi-Cup Grasping"],
    "status": "tested",
    "entrypoint": "grasp_prior_from_video",
}

from pathlib import Path

import numpy as np

from tool_library.common import rigcam as RC


def _mask(p):
    import cv2
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(str(p))
    return m > 127


def grasp_prior_from_video(camera, object_mask, hand_mask, object_height_m=0.05, tolerance_deg=45.0, top_fraction=0.5):
    cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
    O, Hm = _mask(object_mask), _mask(hand_mask)
    if O.shape != Hm.shape:
        raise ValueError(f"masks differ in size: object {O.shape}, hand {Hm.shape}")
    H, W = O.shape
    if (W, H) != (cam["width"], cam["height"]):
        cam = RC.scaled(cam, W, H)
    oy, ox = np.nonzero(O); hy, hx = np.nonzero(Hm)
    if len(ox) < 20 or len(hx) < 20:
        raise ValueError("a mask has fewer than 20 pixels")
    # object footprint centroid from its lowest rows (on the table)
    band = oy >= oy.max() - max(3, int(0.08 * (oy.max() - oy.min() + 1)))
    Po = RC.pixel_on_plane(cam, np.stack([ox[band], oy[band]], 1).astype(float), 0.0)
    o_xy = Po[:, :2].mean(axis=0)
    # hand: the lowest fifth of the hand mask (fingers) projected at the grasp height
    hb = hy >= hy.max() - max(4, int(0.2 * (hy.max() - hy.min() + 1)))
    Ph = RC.pixel_on_plane(cam, np.stack([hx[hb], hy[hb]], 1).astype(float), float(object_height_m))
    h_xy = Ph[:, :2].mean(axis=0)
    v = o_xy - h_xy; dist = float(np.linalg.norm(v))
    inside = float((Hm & O).sum() / max(1, Hm.sum()))
    over = float(((Hm & O).sum()) / max(1, O.sum()))
    # the object's planar radius from its whole silhouette on the table plane
    step = max(1, int(np.sqrt(len(ox) / 3000)))
    Pf = RC.pixel_on_plane(cam, np.stack([ox[::step], oy[::step]], 1).astype(float), 0.0)[:, :2]
    radius = float(np.percentile(np.linalg.norm(Pf - Pf.mean(axis=0), axis=1), 90))
    # over the top: the hand's fingers project inside the object's footprint, or the masks overlap heavily
    from_top = bool(inside >= top_fraction or dist < max(0.03, 1.2 * radius))
    ang = float(np.degrees(np.arctan2(v[1], v[0]))) if dist > 1e-6 else 0.0
    u = (v / dist) if dist > 1e-6 else np.array([0.0, 1.0])
    dirs = []
    for k in range(8):
        a = np.radians(45 * k); d = np.array([np.cos(a), np.sin(a)])
        score = float(d @ u)
        dirs.append((score, [round(float(d[0]), 3), round(float(d[1]), 3), 0.0]))
    dirs.sort(key=lambda x: -x[0])
    ranked = [d for sc, d in dirs if sc >= np.cos(np.radians(tolerance_deg))]
    # approach_dirs for the pose solvers point FROM the object TOWARD the hand (the direction the hand retreats along)
    approach_dirs = [[-d[0], -d[1], 0.0] for d in ranked]
    if from_top:
        approach_dirs = [[0.0, 0.0, 1.0]] + approach_dirs
    family = "top" if from_top else "side"
    return {"approach_xy": [round(float(u[0]), 3), round(float(u[1]), 3)], "approach_angle_deg": round(ang, 1), "distance_m": round(dist, 4),
            "hand_over_object_fraction": round(inside, 3), "object_covered_fraction": round(over, 3), "object_radius_m": round(radius, 4), "from_top": from_top, "family": family,
            "approach_dirs": approach_dirs, "candidate_approach_tolerance_deg": float(tolerance_deg),
            "hand_centroid_xy": [round(float(h_xy[0]), 4), round(float(h_xy[1]), 4)], "object_centroid_xy": [round(float(o_xy[0]), 4), round(float(o_xy[1]), 4)],
            "note": "retreat/approach vectors in pose dicts point from the object toward the hand; a generate_rigid_grasps candidate matches when its `approach` is within the tolerance of one of approach_dirs"}
