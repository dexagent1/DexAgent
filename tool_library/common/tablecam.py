"""A camera looking down at a table: pixels <-> metres on the table plane.

The table frame is (forward, left, up) with the origin on the tabletop
directly below the camera. The camera sits h_above_table_m up, pitched
pitch_deg below horizontal and rolled roll_deg, with focal_px and a
principal point at the image centre. One image gives one ray per pixel;
the table plane is the second equation, so anything KNOWN to lie on a plane
parallel to the table can be measured in metres from a single view.
"""
from __future__ import annotations

import numpy as np


def camera(focal_px, width, height, h_above_table_m, pitch_deg, roll_deg=0.0):
    """Camera dictionary. Axes are the camera's x (right), y (down), z
    (forward) expressed in the table frame."""
    p, r = np.radians(float(pitch_deg)), np.radians(float(roll_deg))
    xc = np.array([0.0, -np.cos(r), np.sin(r)])          # image right = table -left (rolled)
    zc = np.array([np.cos(p), 0.0, -np.sin(p)])          # optical axis, pitched down
    xc = xc - np.dot(xc, zc) * zc
    xc /= np.linalg.norm(xc)
    return {"focal_px": float(focal_px), "width": int(width), "height": int(height),
            "h_above_table_m": float(h_above_table_m), "pitch_deg": float(pitch_deg), "roll_deg": float(roll_deg),
            "R_table_cam": np.stack([xc, np.cross(zc, xc), zc], axis=1)}


def project(P, cam):
    """Table-frame points [N, 3] -> pixels [N, 2]."""
    Q = (np.atleast_2d(np.asarray(P, float)) - np.array([0.0, 0.0, cam["h_above_table_m"]])) @ cam["R_table_cam"]
    f, cx, cy = cam["focal_px"], cam["width"] / 2.0, cam["height"] / 2.0
    return np.stack([cx + f * Q[:, 0] / Q[:, 2], cy + f * Q[:, 1] / Q[:, 2]], axis=1)


def plane_points(uv, cam, z_above_table=0.0):
    """Pixels [N, 2] -> table-frame points on the plane z = z_above_table."""
    uv = np.atleast_2d(np.asarray(uv, float))
    f, cx, cy = cam["focal_px"], cam["width"] / 2.0, cam["height"] / 2.0
    d = (cam["R_table_cam"] @ np.stack([(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f, np.ones(len(uv))])).T
    h = cam["h_above_table_m"] - float(z_above_table)
    s = -h / d[:, 2]
    if np.any(s <= 0):
        raise ValueError("some pixels do not hit the table plane (above the horizon): check pitch and height")
    return d * s[:, None] + np.array([0.0, 0.0, cam["h_above_table_m"]])


def optical_depth(P, cam):
    """Distance along the optical axis of table-frame points."""
    Q = (np.atleast_2d(np.asarray(P, float)) - np.array([0.0, 0.0, cam["h_above_table_m"]])) @ cam["R_table_cam"]
    return Q[:, 2]


def silhouette_from_edges(frame_rgb, roi, k=10, close=7):
    """Filled outline of a dark-edged object inside roi on a light surface:
    pixels k grey levels below the local median (31 px) are edges, closed
    and filled from the largest contour. The route for transparent objects."""
    import cv2
    x0, y0, x1, y1 = (int(v) for v in roi)
    sub = cv2.cvtColor(np.asarray(frame_rgb)[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    edge = (cv2.medianBlur(sub, 31).astype(int) - sub.astype(int)) > int(k)
    m = cv2.morphologyEx(edge.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError(f"nothing dark enough inside roi {roi} at k={k}")
    fill = np.zeros_like(m)
    cv2.drawContours(fill, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    full = np.zeros(np.asarray(frame_rgb).shape[:2], np.uint8)
    full[y0:y1, x0:x1] = fill
    return full.astype(bool)
