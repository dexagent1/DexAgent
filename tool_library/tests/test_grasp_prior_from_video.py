"""grasp_prior_from_video: a hand footprint beside an object, both drawn through the video camera, gives the approach
from the hand toward the object; a hand drawn over the object reads as a grasp from the top.

    python tool_library/tests/test_grasp_prior_from_video.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import rigcam as RC  # noqa: E402
from tool_library.skills.grasp_prior_from_video import grasp_prior_from_video  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def disc(cam, centre, radius, path, z=0.0):
    """A filled footprint of `radius` metres around `centre` on the plane z, drawn through the camera."""
    ring = [np.array(centre) + radius * np.array([np.cos(a), np.sin(a), 0.0]) for a in np.linspace(0, 2 * np.pi, 48)]
    uv, _ = RC.project(cam, np.array(ring) + [0, 0, z])
    img = np.zeros((cam["height"], cam["width"]), np.uint8)
    cv2.fillPoly(img, [np.round(uv).astype(np.int32)], 255)
    cv2.imwrite(str(path), img)
    return str(path)


tmp = Path(tempfile.mkdtemp(prefix="dexagent_prior_", dir=os.environ.get("TMPDIR")))
cam = RC.camera_from_fovy(640, 480, 38.0, [0.0, -0.55, 0.55], RC.R_cv_from_angles(90.0, 40.0))
obj = disc(cam, [0.05, 0.10, 0.0], 0.04, tmp / "obj.png")
side = disc(cam, [0.20, 0.10, 0.0], 0.035, tmp / "hand_right.png", z=0.05)
r = grasp_prior_from_video(cam, obj, side, object_height_m=0.05)
ax, ay = r["approach_xy"]
check("a hand to the right of the object approaches along -x", ax < -0.9 and abs(ay) < 0.3 and not r["from_top"], f"{np.round(r['approach_xy'], 3)} family {r['family']}")
check("the ranked directions start with the human's, pointing from the object back to the hand (the retreat axis)",
      np.dot(np.asarray(r["approach_dirs"][0])[:2], [-ax, -ay]) > 0.9, str(r["approach_dirs"][:2]))
top = disc(cam, [0.05, 0.10, 0.0], 0.03, tmp / "hand_top.png", z=0.05)
r2 = grasp_prior_from_video(cam, obj, top, object_height_m=0.05)
check("a hand over the object reads as from the top", r2["from_top"] and r2["family"] == "top" and r2["approach_dirs"][0] == [0.0, 0.0, 1.0],
      f"distance {r2['distance_m']} m, {r2['family']}")
print("\nALL PASS" if not FAILS else f"\n{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
