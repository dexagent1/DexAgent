"""register_camera_hand_overlay: the camera from the robot hands at home laid over hand masks.

Fixtures (tests/fixtures/register_camera_hand_overlay/):
  hands_setup_cam.png   the setup scene's home hands rendered (MuJoCo segmentation) through the setup's ego camera
  hands_moved_cam.png   the same through a camera 3 deg less pitched, 4 deg more yawed, 1 deg rolled, 5 cm right,
                        4 cm back, 3 cm higher (MOVED below)
  bench_bottle_frame95  a real bench frame (open_the_bottle, frame 95, hands resting) and its skin-colour hand mask:
                        the demonstrator's hands rest 10 cm inward, 8 cm forward and on the cloth, not at the robot's home
Run: MUJOCO_GL=egl python tool_library/tests/test_register_camera_hand_overlay.py
"""
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import human_prior as HP  # noqa: E402
from tool_library.common import rigcam as RC  # noqa: E402
from tool_library.skills.register_camera_hand_overlay import register_camera_hand_overlay  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "register_camera_hand_overlay"
FAILS = []


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def setup_camera():
    import yaml
    e = yaml.safe_load((ROOT / "setup" / "setup.yaml").read_text())["cameras"]["ego"]
    return RC.camera_from_mujoco(640, 480, e["fovy_deg"], e["pos"], e["quat_wxyz"])


def params(cam):
    return np.array(HP.camera_angles(cam) + (cam["pos"][2], cam["pos"][0], cam["pos"][1]))


def table_error_cm(est, ref):
    """Median distance (cm) between where the two cameras put the same pixels on the tabletop, over the work area."""
    G = np.array([[x, y, 0.0] for x in np.linspace(-0.3, 0.3, 7) for y in np.linspace(0.25, 0.7, 10)])
    uv, _ = RC.project(ref, G)
    Q = RC.pixel_on_plane(est, uv, 0.0)
    return float(np.median(np.linalg.norm(Q - G, axis=1)) * 100)


def rot_error_deg(a, b):
    Ra, Rb = np.asarray(a["R_cv"]), np.asarray(b["R_cv"])
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def blank_frame(d):
    import cv2
    p = Path(d) / "frame00000.jpg"
    cv2.imwrite(str(p), np.full((480, 640, 3), 200, np.uint8))
    return p


def recover(name, mask, truth, d, dp, prior_off):
    p = params(truth)
    init = {"pitch_deg": p[0] + dp[0], "yaw_deg": p[1] + dp[1], "roll_deg": p[2] + dp[2], "height_m": p[3] + dp[3],
            "foot_xy": [p[4] + dp[4], p[5] + dp[5]]}
    prior = {"pitch_deg": p[0] + prior_off[0], "roll_deg": p[2] + prior_off[1], "height_m": p[3] + prior_off[2]}
    cam_init = HP.camera_from_params(640, 480, truth["fx"], truth["fy"], 320, 240, init["pitch_deg"], init["yaw_deg"], init["roll_deg"],
                                     init["foot_xy"][0], init["foot_xy"][1], init["height_m"])
    t = time.time()
    r = register_camera_hand_overlay(blank_frame(d), FIX / mask, Path(d) / name / "camera.json", init=init, table_prior=prior)
    te0, te = table_error_cm(cam_init, truth), table_error_cm(r["camera"], truth)
    re = rot_error_deg(r["camera"], truth)
    pe = 100 * float(np.linalg.norm(np.asarray(r["camera"]["pos"]) - np.asarray(truth["pos"])))
    info = (f"IoU {r['iou']:.2f}, table error {te0:.1f} -> {te:.1f} cm, rotation {re:.1f} deg, position {pe:.1f} cm, "
            f"{time.time() - t:.0f} s")
    return r, te0, te, re, pe, info


def main():
    t0 = time.time()
    true = setup_camera()
    p = params(true)
    moved = HP.camera_from_params(640, 480, true["fx"], true["fy"], 320, 240, p[0] - 3, p[1] + 4, p[2] + 1, p[4] + 0.05, p[5] - 0.04, p[3] + 0.03)
    with tempfile.TemporaryDirectory() as d:
        # 1. the setup's own camera from a perturbed guess and a slightly wrong table prior
        r, te0, te, re, pe, info = recover("setup", "hands_setup_cam.png", true, d, [2.0, -3.0, 1.0, -0.03, 0.03, -0.04], [0.8, -0.5, 0.01])
        check("setup camera recovered from a perturbed guess", r["accepted"] and r["iou"] >= 0.85 and te <= 3.0 and te < te0 and re <= 3.0, info)
        cj = RC.load(Path(d) / "setup" / "camera.json")
        need = {"width", "height", "fx", "fy", "cx", "cy", "fovy_deg", "pos", "quat_wxyz", "R_cv", "pitch_deg", "yaw_deg", "height_above_table_m"}
        check("camera.json has register_table_camera's keys and a check image", need <= set(cj) and cj["source"] == "hand_overlay"
              and Path(r["check_png"]).exists())
        # 2. a different camera (the masks rendered through MOVED)
        r, te0, te, re, pe, info = recover("moved", "hands_moved_cam.png", moved, d, [-2.0, 3.0, -1.0, 0.03, -0.03, 0.04], [-0.6, 0.7, -0.01])
        check("a moved camera recovered", r["accepted"] and r["iou"] >= 0.85 and te <= 4.0 and te < te0, info)
        # 3. real bench frame: the demonstrator's hands do not rest at the home pose -> the overlay must not be trusted
        init = {"pitch_deg": p[0], "yaw_deg": p[1] + 2.0, "roll_deg": p[2], "height_m": p[3], "foot_xy": [p[4] - 0.02, p[5]]}
        r = register_camera_hand_overlay(FIX / "bench_bottle_frame95.jpg", FIX / "bench_bottle_frame95_hands.png", Path(d) / "bench" / "camera.json",
                                         init=init)
        cam_init = HP.camera_from_params(640, 480, true["fx"], true["fy"], 320, 240, init["pitch_deg"], init["yaw_deg"], init["roll_deg"],
                                         init["foot_xy"][0], init["foot_xy"][1], init["height_m"])
        rej = RC.load(r["rejected_json"]) if r["rejected_json"] else None
        check("real resting hands off the home pose: overlay rejected, initial guess kept",
              (not r["accepted"]) and r["iou"] < 0.7 and rej is not None and rot_error_deg(r["camera"], cam_init) < 1e-6
              and any("REJECTED" in w for w in r["warnings"]),
              f"IoU {r['iou']:.2f}; rejected estimate's table error {table_error_cm(rej, true):.1f} cm vs the guess's {table_error_cm(cam_init, true):.1f} cm" if rej else "")
    # 4. the gravity prior's angle conversion (estimate_gravity / fit_camera_to_table_plane give gravity_cam)
    errs = []
    for pit, rol in [(33.7, -1.6), (12.0, 4.0), (55.0, 0.0)]:
        R = RC.R_cv_from_angles(90.0, pit, rol)
        g = R.T @ np.array([0.0, 0.0, -1.0])
        a, b = HP.angles_from_gravity(g)
        errs.append(max(abs(a - pit), abs(b - rol)))
    check("gravity_cam -> pitch, roll", max(errs) < 0.05, f"worst {max(errs):.4f} deg")
    # 5. misuse
    with tempfile.TemporaryDirectory() as d:
        import cv2
        empty = Path(d) / "empty.png"
        cv2.imwrite(str(empty), np.zeros((480, 640), np.uint8))
        bad = 0
        for kw in ({"hand_mask": empty}, {"hand_mask": FIX / "hands_setup_cam.png", "hand_pose": "standing"},
                   {"hand_mask": FIX / "hands_setup_cam.png", "table_prior": {"pitch_deg": -30.0, "camera_height_m": 0.5}}):
            try:
                register_camera_hand_overlay(blank_frame(d), out_json=Path(d) / "c.json", **kw)
            except ValueError:
                bad += 1
        check("misuse raises ValueError (empty mask, unknown hand_pose, sign-ambiguous prior)", bad == 3, f"{bad}/3")
    print(f"{len(FAILS)} failed, {time.time() - t0:.0f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
