"""Tests for trace_rope_waypoints on synthetic cords drawn through known cameras."""
import sys

import cv2
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.common import tablecam as TC
from tool_library.common import rigcam as RC
from tool_library.skills.trace_rope_waypoints import trace_rope_waypoints

R_CORD = 0.003
fails = 0


def check(name, cond, msg=""):
    global fails
    if cond:
        print(f"[PASS] {name}")
    else:
        fails += 1
        print(f"[FAIL] {name} {msg}")


def draw(px, shape, width):
    m = np.zeros(shape, np.uint8)
    cv2.polylines(m, [np.round(px).astype(np.int32).reshape(-1, 1, 2)], False, 255, int(width), cv2.LINE_AA)
    return (m > 127).astype(np.uint8)


def length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def waypoints(px, k, rng, noise=2.0):
    idx = np.linspace(0, len(px) - 1, k).round().astype(int)
    return (px[idx] + rng.normal(0, noise, (k, 2))).tolist()


rng = np.random.default_rng(0)
cam = TC.camera(focal_px=700.0, width=640, height=480, h_above_table_m=0.5, pitch_deg=50.0)

# 1. S-curve on the table, no crossing
t = np.linspace(0, 1, 400)
P1 = np.stack([0.42 + 0.05 * np.sin(2 * np.pi * t), -0.12 + 0.24 * t, np.full_like(t, R_CORD)], 1)
px1 = TC.project(P1, cam)
w1 = 2 * R_CORD * 700 / float(np.median(TC.optical_depth(P1, cam)))
m1 = draw(px1, (480, 640), max(3, round(w1)))
r1 = trace_rope_waypoints(m1, cam, waypoints(px1, 8, rng))
e1 = abs(r1["length_m"] - length(P1)) / length(P1)
check("s_curve_length_and_no_crossing", r1["frame"] == "table" and e1 < 0.08 and not r1["crossings"] and r1["ok"],
      f"len {r1['length_m']} vs {length(P1):.4f}, crossings {len(r1['crossings'])}, issues {r1['issues']}")

# 2. a curl (prolate cycloid) that crosses itself once: one crossing, the later strand lifted by a diameter
th = np.linspace(-2.7, 2.7, 600)          # a prolate cycloid loops once about th = 0 (crossing near +-2.0)
a, b = 0.02, 0.05
P2 = np.stack([0.45 - b * np.cos(th), -0.10 + a * th - b * np.sin(th) + 0.05, np.full_like(th, R_CORD)], 1)
px2 = TC.project(P2, cam)
m2 = draw(px2, (480, 640), max(3, round(w1)))
r2 = trace_rope_waypoints(m2, cam, waypoints(px2, 16, rng))
z = np.asarray(r2["centreline"])[:, 2]
e2 = abs(r2["length_m"] - length(P2)) / length(P2)
lifted_ok = len(r2["crossings"]) == 1 and abs(z.max() - (R_CORD + r2["radius_m"] * 2.2)) < 0.25 * R_CORD + 0.002
check("curl_one_crossing_lifted", lifted_ok and e2 < 0.10 and r2["crossings"][0]["lifted"] == "b",
      f"crossings {len(r2['crossings'])}, zmax {z.max():.4f}, radius {r2['radius_m']}, len err {e2:.3f}")

# 3. the same S-curve with a stretch hidden (a hand over it): the gap is bridged straight
m3 = m1.copy()
mid = px1[180:230]
x0, y0 = mid.min(0).astype(int) - 6
x1, y1 = mid.max(0).astype(int) + 6
m3[y0:y1, x0:x1] = 0
wp3 = waypoints(px1, 8, rng) + []
wp3 = sorted(wp3 + [px1[205].tolist()], key=lambda q: int(np.argmin(np.linalg.norm(px1 - np.asarray(q), axis=1))))
r3 = trace_rope_waypoints(m3, cam, wp3, snap_px=4)
e3 = abs(r3["length_m"] - length(P1)) / length(P1)
check("occluded_gap_bridged", r3["n_bridged_legs"] >= 1 and e3 < 0.10,
      f"bridged {r3['n_bridged_legs']}, len err {e3:.3f}")

# 4. a rig-frame camera.json dict: rig output with build_isaac_scene points, the 'earlier' strand on top
R_cv = RC.R_cv_from_angles(yaw_deg=90.0, pitch_deg=40.0)
rc = RC.camera_from_fovy(640, 480, 60.0, [0.0, -0.1, 0.45], R_cv)
P4 = np.stack([0.0 + P2[:, 1], 0.25 + (P2[:, 0] - 0.45), P2[:, 2]], 1)
px4, d4 = RC.project(rc, P4)
w4 = 2 * R_CORD * rc["fx"] / float(np.median(d4))
m4 = draw(px4, (480, 640), max(3, round(w4)))
r4 = trace_rope_waypoints(m4, rc, waypoints(px4, 16, rng), over="earlier")
c4 = np.asarray(r4["points"])
e4 = abs(r4["length_m"] - length(P4)) / length(P4)
near = np.linalg.norm(c4[:, :2][:, None] - P4[None, :, :2], axis=2).min(1).mean()
check("rig_camera_points_earlier_on_top", r4["frame"] == "rig" and len(r4["crossings"]) == 1
      and r4["crossings"][0]["lifted"] == "a" and e4 < 0.10 and near < 0.004,
      f"frame {r4['frame']}, crossings {len(r4['crossings'])}, len err {e4:.3f}, mean offset {near:.4f}")

# 5. misuse fails loudly
bad = 0
for kwargs in ({"waypoints_px": [[10, 10]]}, {"waypoints_px": [[10, 10], [20, 20]], "bogus": 1},
               {"waypoints_px": [[10, 10], [9000, 20]]}):
    try:
        trace_rope_waypoints(m1, cam, **kwargs)
    except ValueError:
        bad += 1
check("misuse_raises_valueerror", bad == 3, f"{bad}/3 raised")

sys.exit(1 if fails else 0)
