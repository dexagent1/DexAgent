"""Camera intrinsics end to end: fx != fy and an off-centre principal point are honoured by every step.

Stage 2's camera.json can carry fx != fy and (cx, cy) off the image centre. Stage 4 must put a world point on the same
pixel in the clean plate, in the wrist views' re-projected plates and in the Blender render, or the robot is pasted
in the wrong place. Checks, with independent projection maths (pixel (i, j) covers [i, i+1) x [j, j+1)):
  - config.intrinsics: a fovy-only camera keeps the old square-pixel, centred lens; fx/fy/cx/cy given for another
    image size are scaled; a size of another aspect ratio is refused;
  - export.table_mask / export.project: the ray through a pixel centre lands on the table point that projects back
    to that pixel centre;
  - export.reproject_plate: a dot drawn on the ego plate at a table point appears, in another camera, where that
    point projects;
  - render_cycles (when bpy is installed): small spheres at known points render with their centroid within 1 px of
    the projection, for an fx != fy off-centre camera and for a fovy-only one.

    cd stage4_robot_data_generation && python tests/test_camera_intrinsics.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from stage4 import export  # noqa: E402
from stage4 import kinematics as K  # noqa: E402
from stage4.config import intrinsics, resolved_camera  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="stage4_cam_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    FAILS += 0 if ok else 1


def look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """A camera at pos looking at target, MuJoCo convention (x right, y up, looking along -z): pos, quat_wxyz."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    fwd = (target - pos) / np.linalg.norm(target - pos)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    return {"pos": pos.tolist(), "quat_wxyz": K.mat2quat(np.stack([right, upv, -fwd], 1)).tolist()}


def pixel(cam, P, W, H):
    """Independent pinhole projection (OpenCV axes from the MuJoCo camera frame) -> continuous (u, v)."""
    fx, fy, cx, cy = intrinsics(cam, W, H)
    R = K.quat2mat(cam["quat_wxyz"])
    q = R.T @ (np.asarray(P, float) - np.asarray(cam["pos"], float))
    x, y, z = q[0], -q[1], -q[2]
    return cx + fx * x / z, cy + fy * y / z


def main():
    # 1. config.intrinsics
    f = (48 / 2) / math.tan(math.radians(38.0) / 2)
    ok_fovy = np.allclose(intrinsics({"fovy": 38.0}, 64, 48), (f, f, 32.0, 24.0))
    sc = intrinsics({"fx": 1394.0, "fy": 1487.0, "cx": 650.0, "cy": 500.0, "image_size": [1280, 1024]}, 320, 256)
    ok_scale = np.allclose(sc, (348.5, 371.75, 162.5, 125.0))
    try:
        intrinsics({"fx": 1394.0, "fy": 1487.0, "image_size": [1280, 1024]}, 640, 480)
        refused = False
    except ValueError as e:
        refused = "aspect" in str(e)
    check("intrinsics: fovy-only unchanged, fx/fy/cx/cy scaled to the output, another aspect refused",
          ok_fovy and ok_scale and refused, f"{sc}")

    # 2. table_mask and project agree with each other and with the independent projection
    W, H = 120, 90
    ego = {**look_at([0.05, -0.45, 0.55], [0.0, 0.35, 0.0]), "fx": 110.0, "fy": 130.0, "cx": 52.0, "cy": 50.0}
    table = {"top_z": 0.0, "center": [0.0, 0.3], "half_size": [1.0, 1.0]}
    on, pts = export.table_mask(ego, table, W, H)
    u, v, z = export.project(ego, pts[on], W, H)
    jj, ii = np.nonzero(on)
    err_rt = float(np.max(np.hypot(u - (ii + 0.5), v - (jj + 0.5))))
    P = np.array([0.08, 0.30, 0.0])
    up_, vp_, _ = export.project(ego, P, W, H)
    err_ind = float(np.hypot(*(np.array([up_, vp_]) - np.array(pixel(ego, P, W, H)))))
    check("table_mask rays and project round-trip to the pixel centres (fx != fy, off-centre principal point)",
          on.sum() > 1000 and err_rt < 1e-6 and err_ind < 1e-9, f"round trip {err_rt:.2e} px, vs independent {err_ind:.2e} px")

    # 3. reproject_plate: a dot at a table point on the ego plate lands where that point projects in another camera
    plate = np.full((H, W, 3), 60, np.uint8)
    ue, ve = pixel(ego, P, W, H)
    cv2.circle(plate, (int(round(ue * 16 - 8)), int(round(ve * 16 - 8))), 2 * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)
    view = {**look_at([0.35, -0.10, 0.40], [0.05, 0.32, 0.0]), "fovy": 70.0}
    img = export.reproject_plate(plate, resolved_camera(ego, W, H), view, table, np.array([0, 0, 0], np.uint8), W, H)
    w = np.clip(img[..., 0].astype(float) - 60, 0, None)
    w[w < 40] = 0
    yy, xx = np.mgrid[:H, :W]
    cu, cv_ = (w * xx).sum() / w.sum() + 0.5, (w * yy).sum() / w.sum() + 0.5
    uv_, vv_ = pixel(view, P, W, H)
    err_re = float(np.hypot(cu - uv_, cv_ - vv_))
    check("re-projected plate: a table point keeps its place in another view", err_re < 1.0,
          f"dot at ({cu:.2f}, {cv_:.2f}), expected ({uv_:.2f}, {vv_:.2f}): {err_re:.2f} px")

    # 4. Blender: spheres at known points render on their projections
    try:
        import bpy  # noqa: F401
    except ImportError:
        print("[SKIP] bpy not installed: Blender render not checked")
    else:
        markers = [np.array(p) for p in ([0.0, 0.35, 0.02], [0.18, 0.18, 0.02], [-0.20, 0.55, 0.06])]
        for name, cam in (("fx != fy, off-centre", ego), ("fovy only", {**ego, "fx": None, "fy": None, "fovy": 45.0})):
            cam = {k: v for k, v in cam.items() if v is not None}
            vdir = TMP / name.replace(" ", "_").replace(",", "").replace("!=", "ne")
            vdir.mkdir()
            T = np.tile(np.eye(4), (1, len(markers), 1, 1))
            for k, p in enumerate(markers):
                T[0, k, :3, 3] = p
            np.savez_compressed(vdir / "transforms.npz", T=T)
            rc = resolved_camera(cam, W, H)
            meta = {"geoms": [{"gid": k, "name": f"m{k}", "type": "sphere", "size": [0.012, 0, 0], "rgba": [1, 0, 0, 1],
                               "robot": False, "material": {"base": [1.0, 0.1, 0.1], "metallic": 0.0, "rough": 0.5}}
                              for k in range(len(markers))],
                    "table": {"top_z": -10.0, "center": [50.0, 50.0], "half_size": [0.1, 0.1]},
                    "width": W, "height": H, "fps": 30.0, "n_frames": 1, "samples": 16,
                    "lighting": {"sun_energy": 3.0, "sun_angle": 0.3, "sun_euler": [0.55, 0.15, 0.9], "world_strength": 0.55,
                                 "fill_energy": 6.0, "world_color": [0.5, 0.5, 0.5]},
                    "fill_pos": cam["pos"], "transforms": str(vdir / "transforms.npz"), "near_fade": [0.0, 0.0],
                    "camera": {"pos": cam["pos"], "quat_wxyz": cam["quat_wxyz"],
                               **{k: rc[k] for k in ("fx", "fy", "cx", "cy", "fovy")}, "clip_start": 0.01}}
            (vdir / "scene.json").write_text(json.dumps(meta))
            r = subprocess.run([sys.executable, "-m", "stage4.render_cycles", str(vdir)], cwd=HERE.parent,
                               capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            rgba = cv2.imread(str(vdir / "frames" / "f00000.png"), cv2.IMREAD_UNCHANGED)
            if r.returncode != 0 or rgba is None:
                check(f"Blender render ({name})", False, (r.stdout + r.stderr)[-400:])
                continue
            a = rgba[..., 3].astype(float) / 255.0
            n, lab = cv2.connectedComponents((a > 0.05).astype(np.uint8))
            errs = []
            for p in markers:
                ue_, ve_ = pixel(cam, p, W, H)
                best = None
                for c in range(1, n):
                    m = (lab == c) * a
                    cu_, cv2_ = (m * xx).sum() / m.sum() + 0.5, (m * yy).sum() / m.sum() + 0.5
                    d = math.hypot(cu_ - ue_, cv2_ - ve_)
                    best = d if best is None else min(best, d)
                errs.append(best if best is not None else float("inf"))
            check(f"Blender render ({name}): every marker within 1 px of its projection", n - 1 == len(markers) and max(errs) < 1.0,
                  f"{n - 1} blobs, errors {[round(e, 2) for e in errs]} px")
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
