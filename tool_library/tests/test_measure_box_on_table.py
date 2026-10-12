"""measure_box_on_table called the way the harness calls it: mask PNG paths and the Stage 2 camera.json.

    python tool_library/tests/test_measure_box_on_table.py

The runs could not call it: `fit_pitch` was not a declared input (one view then always raised), and views needed
ndarray masks plus a common.tablecam dict while callers have mask PNGs and a camera.json. Synthetic masks are the
projected outline of a known box through a rig camera (rigcam.project, independent of the table-camera maths the
tool uses), so the rig-frame pose it returns is checked against the truth.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tool_library.common import rigcam as RC  # noqa: E402
from tool_library.skills.measure_box_on_table import hull_mask, measure_box_on_table, prism_points  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_boxmeas_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def rig_camera(width=640, height=480, cx=None, cy=None):
    R = RC.R_cv_from_angles(yaw_deg=90.0, pitch_deg=36.0)
    return RC.camera_from_fovy(width, height, 38.0, [0.0, -0.25, 0.62], R, cx=cx, cy=cy)


def box_mask_png(cam, L, W, H, rad, x, y, yaw_deg, path):
    """The box's outline seen through the rig camera, written as a 0/255 PNG (the segmenters' format)."""
    P = prism_points(L, W, H, rad, 0.0, 0.0, yaw_deg)          # box frame = rig frame up to the (x, y) offset
    P[:, 0] += x
    P[:, 1] += y
    uv, _ = RC.project(cam, P)
    m = hull_mask(uv, (cam["height"], cam["width"]))
    Image.fromarray(m.astype(np.uint8) * 255).save(path)
    return str(path)


TRUTH = dict(L=0.20, W=0.14, H=0.06, rad=0.015)
POSES = [(0.05, 0.42, 20.0), (-0.08, 0.48, 75.0)]                # rig x, y, yaw (deg)
cam = rig_camera()
cam_json = RC.save(cam, TMP / "camera.json")
paths = [box_mask_png(cam, TRUTH["L"], TRUTH["W"], TRUTH["H"], TRUTH["rad"], x, y, yaw, TMP / f"box_{i}.png")
         for i, (x, y, yaw) in enumerate(POSES)]


def yaw_err(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def pose_ok(fit, poses, tol_m=0.012, tol_deg=4.0):
    errs = []
    for v, (x, y, yaw) in zip(fit["views"], poses):
        errs.append((round(float(np.hypot(v["pos_rig"][0] - x, v["pos_rig"][1] - y)), 4), round(yaw_err(v["yaw_rig_deg"], yaw), 2)))
    return all(e[0] < tol_m and e[1] < tol_deg for e in errs), errs


# 1. the harness's call: two mask PNG paths and the camera.json path, no fit_pitch given
fit = measure_box_on_table(paths, cam_json, restarts=2)
pok, perr = pose_ok(fit, POSES)
check("mask PNG paths + camera.json: footprint within 1 cm, rig poses within 12 mm / 4 deg, IoU > 0.95",
      abs(fit["length_m"] - TRUTH["L"]) < 0.01 and abs(fit["width_m"] - TRUTH["W"]) < 0.01 and fit["mean_iou"] > 0.95
      and pok and fit["pitch_fitted"] and fit["camera_frame"] == "rig",
      f"L {fit['length_m']} W {fit['width_m']} iou {fit['mean_iou']} pose err {perr}")

# 2. one view: fit_pitch defaults to False (trust the camera) instead of raising; explicit True still refuses
fit1 = measure_box_on_table([{"mask": paths[0], "h_bounds": [0.03, 0.10]}], cam_json, restarts=2)
pok1, perr1 = pose_ok(fit1, POSES[:1])
check("one view, fit_pitch left out: measured with the camera's pitch, footprint within 1.5 cm",
      not fit1["pitch_fitted"] and abs(fit1["length_m"] - TRUTH["L"]) < 0.015 and abs(fit1["width_m"] - TRUTH["W"]) < 0.015 and pok1,
      f"L {fit1['length_m']} W {fit1['width_m']} pose err {perr1}")
try:
    measure_box_on_table([paths[0]], cam_json, fit_pitch=True)
    check("one view with fit_pitch=True still raises ValueError", False)
except ValueError as e:
    check("one view with fit_pitch=True still raises ValueError", "two views" in str(e))

# 3. every parameter reaches the model's function schema (grouped rows were dropped from it)
from agent.library import call_tool, find_tool, load_index, tool_schema  # noqa: E402
index = load_index(ROOT / "tool_library")
from tool_library.skills.measure_box_on_table import TOOL  # noqa: E402  (the metadata index.json is built from)
schema = tool_schema(TOOL)["input_schema"]
want = {"views", "camera", "share_footprint", "fit_pitch", "init", "restarts", "seed"}
check("schema exposes every parameter, only views and camera required", want <= set(schema["properties"])
      and set(schema["required"]) == {"views", "camera"}, f"{sorted(schema['properties'])} required {schema['required']}")

# 4. end to end through the harness dispatch with JSON arguments (paths only)
r = call_tool(ROOT / "tool_library", index, "measure_box_on_table", json.loads(json.dumps(
    {"views": paths, "camera": cam_json, "restarts": 1, "share_footprint": True})))
check("call_tool with JSON arguments returns the fit", r.get("ok") and abs(r["result"]["length_m"] - TRUTH["L"]) < 0.015,
      str(r.get("error", ""))[:300] or f"L {r['result']['length_m']}")

# 5. camera.json written for another resolution and an off-centre principal point
cam_hi = rig_camera(1280, 960)
fit_hi = measure_box_on_table(paths, RC.save(cam_hi, TMP / "camera_1280.json"), restarts=2)
check("a 1280x960 camera.json is rescaled to the 640x480 masks", abs(fit_hi["length_m"] - TRUTH["L"]) < 0.01
      and pose_ok(fit_hi, POSES)[0], f"L {fit_hi['length_m']} {pose_ok(fit_hi, POSES)[1]}")
cam_off = rig_camera(cx=320 + 14, cy=240 - 9)
paths_off = [box_mask_png(cam_off, TRUTH["L"], TRUTH["W"], TRUTH["H"], TRUTH["rad"], x, y, yaw, TMP / f"off_{i}.png")
             for i, (x, y, yaw) in enumerate(POSES)]
fit_off = measure_box_on_table(paths_off, cam_off, restarts=2)
check("an off-centre principal point is honoured (masks moved onto their rays)", abs(fit_off["length_m"] - TRUTH["L"]) < 0.01
      and pose_ok(fit_off, POSES)[0], f"L {fit_off['length_m']} {pose_ok(fit_off, POSES)[1]}")

# 6. misuse is named
for bad, word in (([str(TMP / "missing.png")], "does not exist"), ([], "one entry per view")):
    try:
        measure_box_on_table(bad, cam_json)
        check(f"bad views raise ValueError ({word})", False)
    except ValueError as e:
        check(f"bad views raise ValueError ({word})", word in str(e), str(e)[:100])

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
