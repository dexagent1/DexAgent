"""Smoke tests for the Perception & Grounding skills.

    python tool_library/tests/test_perception.py

Synthetic tests always run. Model-backed sections run only when their package
imports in the current interpreter and their checkpoints are found, otherwise
they are reported as SKIP. Because the models live in different environments,
the test may be run from several interpreters; the sections share one scratch
folder (DEXAGENT_TEST_SHARED, default $TMPDIR/dexagent_perception_shared) so
SAM 2 can pick up the detections SAM 3 wrote earlier.

    DEXAGENT_ASSETS_ROOT   root holding sam2_ckpt/ (default: this checkout's parent)
    DEXAGENT_TEST_VIDEO    a static-camera human video with a battery in it (default: the shipped battery_insertion clip)
    DEXAGENT_HAMER_ROOT    HaMeR checkout with _DATA/            (extract_hand_meshes)
    DEXAGENT_SAM3D_REPO    sam-3d-objects checkout with checkpoints/hf  (reconstruct_object_meshes)
    DEXAGENT_VDA_REPO      Video-Depth-Anything checkout with checkpoints/  (estimate_depth)
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.common.video import frame_files, load_label_map, read_frame, save_label_map, write_frames, write_video  # noqa: E402
from tool_library.skills.extract_frames import extract_frames  # noqa: E402
from tool_library.skills.fit_camera_to_table_plane import fit_camera_to_table_plane  # noqa: E402
from tool_library.skills.measure_object_color import measure_object_color  # noqa: E402
from tool_library.skills.probe_video import probe_video  # noqa: E402

ASSETS = Path(os.environ.get("DEXAGENT_ASSETS_ROOT", ROOT.parent))
FOOTAGE = Path(os.environ.get("DEXAGENT_TEST_VIDEO", ROOT / "stage1_semantic_understanding/examples/battery_insertion/video.mp4"))
SHARED = Path(os.environ.get("DEXAGENT_TEST_SHARED", Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "dexagent_perception_shared"))
SHARED.mkdir(parents=True, exist_ok=True)
TMP = Path(tempfile.mkdtemp(prefix="dexagent_percep_", dir=os.environ.get("TMPDIR")))
RESULTS = []


def ok(name, cond, note=""):
    RESULTS.append((name, bool(cond), note))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {note}", flush=True)


def skip(name, why):
    print(f"[SKIP] {name}: {why}", flush=True)


def has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ synthetic
def test_video_io():
    rng = np.random.default_rng(0)
    frames = [np.clip(rng.normal(120, 20, (120, 160, 3)), 0, 255).astype(np.uint8) for _ in range(15)]
    for t, f in enumerate(frames):
        f[40:80, 50 + 3 * t:90 + 3 * t] = (200, 40, 40)              # a red square drifting right
    vid = write_video(frames, TMP / "synthetic.mp4", 5.0)
    info = probe_video(vid, thumbnails=3, out_dir=TMP / "thumbs")
    ok("probe_video reads size/fps/frames", info["width"] == 160 and info["height"] == 120 and abs(info["fps"] - 5) < 0.1 and info["frames"] == 15,
       f"{info['width']}x{info['height']} @ {info['fps']} fps, {info['frames']} frames")
    ok("probe_video thumbnails", len(info.get("thumbnails", [])) == 3)
    meta = extract_frames(vid, TMP / "frames", stride=2)
    files = frame_files(TMP / "frames")
    ok("extract_frames stride 2 -> 8 frames named frame%05d", len(files) == 8 and Path(files[0]).name == "frame00000.jpg" and meta["source_frames"][1] == 2)
    masks = TMP / "masks"
    for t, f in enumerate(files):
        lab = np.zeros((120, 160), np.uint8)
        s = meta["source_frames"][t]
        lab[40:80, 50 + 3 * s:90 + 3 * s] = 1
        save_label_map(lab, masks / f"frame{t:05d}.png", 1)
    ok("label map round-trips", int(load_label_map(masks / "frame00000.png").max()) == 1)
    pal = measure_object_color(TMP / "frames", masks, label=1, k=2, n_frames=4)
    top = pal[0]
    ok("measure_object_color finds the red", top["rgb"][0] > 0.6 and top["rgb"][1] < 0.35 and top["rgb"][2] < 0.35, f"palette={pal}")


def test_table_plane():
    """Depth of a plane seen from a camera 0.6 m up, pitched 35 deg down, with an object blob on it."""
    H, W, f = 240, 320, 300.0
    pitch = np.radians(35.0)
    up_cam = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])          # world-up in OpenCV camera coords
    cam_h = 0.6
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1)
    denom = rays @ up_cam                                              # plane: up . X = -cam_h
    with np.errstate(divide="ignore"):
        z = np.where(denom < -1e-6, -cam_h / denom, np.nan)
    depth = z.astype(np.float32)
    depth[np.isnan(depth)] = 6.5                                      # above the horizon: far wall
    obj = np.zeros((H, W), np.uint8)
    obj[130:190, 130:200] = 1                                         # the object mask (depth irrelevant there)
    depth[obj > 0] -= 0.1
    np.savez(TMP / "depth.npz", depths=np.stack([depth] * 3).astype(np.float16), fps=5.0)
    for t in range(3):
        save_label_map(obj, TMP / "plane_masks" / f"frame{t:05d}.png", 1)
    r = fit_camera_to_table_plane(TMP / "depth.npz", TMP / "plane_masks", focal_px=f, labels=[1])
    ok("fit_camera_to_table_plane recovers pitch -35 and height 0.6", r is not None and abs(r["pitch_deg"] + 35) < 2.0 and abs(r["camera_height_m"] - 0.6) < 0.02,
       f"pitch={r and r['pitch_deg']} h={r and r['camera_height_m']} mode={r and r['mode']} inl={r and r['inlier_frac']}")
    bad_geo = {"up_cam": [0.0, -np.cos(np.radians(40)), np.sin(np.radians(40))], "pitch_deg": 40.0}   # looking UP: wrong side
    r2 = fit_camera_to_table_plane(TMP / "depth.npz", TMP / "plane_masks", focal_px=f, labels=[1], geocalib=bad_geo)
    ok("cross-check overrules a GeoCalib that points at the wrong side", r2["decision"]["source"] == "support_plane", f"{r2['decision']['disagreement_deg']} deg")
    good_geo = {"up_cam": up_cam.tolist(), "pitch_deg": -35.0}
    r3 = fit_camera_to_table_plane(TMP / "depth.npz", TMP / "plane_masks", focal_px=f, labels=[1], geocalib=good_geo)
    ok("cross-check keeps a GeoCalib that agrees", r3["decision"]["source"] == "geocalib")


# -------------------------------------------------------------- real footage
def real_frames():
    d = SHARED / "frames"
    if not frame_files(d):
        if not FOOTAGE.exists():
            return None
        extract_frames(FOOTAGE, d, stride=10, max_frames=12)
    return d


def test_sam3(frames):
    if not has("sam3"):
        return skip("detect_objects / segment_with_text_prompts", "sam3 not importable here")
    from tool_library.skills.detect_objects import detect_objects
    from tool_library.skills.segment_with_text_prompts import segment_with_text_prompts
    scan = detect_objects(frames, ["battery", "hand"], SHARED / "detect.json", frame_indices=[0, 5, 11], min_score=0.2)
    best = scan["summary"]["best_score_per_keyword"]
    ok("detect_objects finds the battery and a hand in the footage", best.get("battery", 0) > 0.3 and best.get("hand", 0) > 0.3, f"best={best}")
    meta = segment_with_text_prompts(frames, ["hand", "arm"], SHARED / "erase_masks", min_score=0.3)
    cov = meta.get("mean_coverage", meta.get("coverage"))
    ok("segment_with_text_prompts wrote one erase mask per frame", len(frame_files(SHARED / "erase_masks")) == len(frame_files(frames)), f"meta={ {k: v for k, v in meta.items() if k != 'per_frame'} }")


def test_sam2(frames):
    if not has("sam2"):
        return skip("track_objects", "sam2 not importable here")
    ckpt = ASSETS / "sam2_ckpt/sam2.1_hiera_large.pt"
    if not ckpt.exists():
        return skip("track_objects", f"no checkpoint at {ckpt}")
    det = SHARED / "detect.json"
    if not det.exists():
        return skip("track_objects", "no detect.json from the SAM 3 section yet (run this test in the sam3 env first)")
    from tool_library.skills.track_objects import track_objects
    r = track_objects(frames, det, SHARED / "track", checkpoint=ckpt, fps=3.0)
    n = len(frame_files(frames))
    labs = [int(load_label_map(p).max()) for p in frame_files(SHARED / "track" / "pred")]
    ok("track_objects: a label map per frame with the battery tracked through the clip", len(labs) == n and sum(l > 0 for l in labs) >= n - 2,
       f"objects={[o['prompt'] for o in r['objects']]} frames_with_mask={sum(l > 0 for l in labs)}/{n}")


def test_geocalib(frames):
    if not has("geocalib"):
        return skip("estimate_gravity", "geocalib not importable here")
    from tool_library.skills.estimate_gravity import estimate_gravity
    g = estimate_gravity(frames, SHARED / "gravity.json", max_frames=6)
    ok("estimate_gravity: egocentric camera looks down (pitch < -10 deg), unit up vector", g["pitch_deg"] < -10 and abs(np.linalg.norm(g["up_cam"]) - 1) < 1e-6,
       f"pitch={g['pitch_deg']:.1f} roll={g['roll_deg']:.1f} inliers={g['n_inliers']}/{g['n_frames']}")


def test_vda(frames):
    repo = Path(os.environ.get("DEXAGENT_VDA_REPO", ""))
    if not repo.exists() or not has("torch"):
        return skip("estimate_depth", "set DEXAGENT_VDA_REPO to a Video-Depth-Anything checkout with checkpoints/")
    from tool_library.skills.estimate_depth import estimate_depth
    try:
        r = estimate_depth(frames, SHARED / "depth.npz", repo, repo / "checkpoints/metric_video_depth_anything_vitl.pth", fps=3.0, max_frames=8,
                           preview_mp4=SHARED / "depth_preview.mp4")
    except ImportError as e:
        return skip("estimate_depth", str(e)[:120])
    ok("estimate_depth: metric range plausible for a tabletop (median 0.3-2 m)", 0.3 < r["median_m"] < 2.0, f"{r}")


def test_hamer(frames):
    root = Path(os.environ.get("DEXAGENT_HAMER_ROOT", ""))
    if root.exists():
        sys.path.insert(0, str(root))
    if not root.exists() or not has("hamer"):
        return skip("extract_hand_meshes", "set DEXAGENT_HAMER_ROOT and run in an env with hamer")
    from tool_library.skills.extract_hand_meshes import extract_hand_meshes
    r = extract_hand_meshes(frames, SHARED / "hands", hamer_root=root, fps=3.0)
    ok("extract_hand_meshes: hands found in most frames, npz + overlay written", r["detection_rate"] >= 0.4 and Path(r["hands_npz"]).exists(),
       f"rate={r['detection_rate']} L={r['hands_left']} R={r['hands_right']}")


def test_sam3d(frames):
    repo = Path(os.environ.get("DEXAGENT_SAM3D_REPO", ""))
    pred = SHARED / "track" / "pred"
    if not repo.exists():
        return skip("reconstruct_object_meshes", "set DEXAGENT_SAM3D_REPO to a sam-3d-objects checkout")
    if not frame_files(pred):
        return skip("reconstruct_object_meshes", "no tracked masks yet (SAM 2 section)")
    from tool_library.skills.reconstruct_object_meshes import reconstruct_object_meshes
    try:
        r = reconstruct_object_meshes(frames, pred, SHARED / "meshes", repo, labels={1: "battery"})
    except ImportError as e:
        return skip("reconstruct_object_meshes", str(e)[:120])
    glb = r["battery"]["glb"]
    ok("reconstruct_object_meshes: textured glb for the battery", glb and Path(glb).stat().st_size > 10_000, f"{r}")


def test_void(frames):
    """Opt-in (slow, loads a 5B video model): DEXAGENT_TEST_VOID=1 in an env with the VOID checkout."""
    if os.environ.get("DEXAGENT_TEST_VOID") != "1":
        return skip("erase_with_void", "set DEXAGENT_TEST_VOID=1 to run VOID (several minutes on a GPU)")
    masks = SHARED / "erase_masks"
    if not frame_files(masks):
        return skip("erase_with_void", "no erase masks yet (SAM 3 section)")
    from tool_library.skills.erase_with_void import erase_with_void
    r = erase_with_void(frames, masks, SHARED / "void", fps=3.0)
    ok("erase_with_void: clean plate + per-frame clean frames", Path(r["plate"]).exists() and len(frame_files(r["clean_dir"])) == len(frame_files(frames)),
       f"coverage={r['erase_coverage']} never_uncovered={r['never_uncovered']}")


def test_pose_fit(frames):
    needed = [SHARED / "depth.npz", SHARED / "track" / "pred", SHARED / "meshes" / "battery.glb"]
    if not all(p.exists() for p in needed) or not has("torch") or not has("trimesh"):
        return skip("fit_object_poses", "needs depth.npz, tracked masks and battery.glb from the other sections, plus torch+trimesh")
    from tool_library.skills.fit_object_poses import fit_object_poses
    r = fit_object_poses(frames, SHARED / "track" / "pred", SHARED / "meshes", SHARED / "depth.npz", ["battery"], SHARED / "poses",
                         focal_px=697.0, labels={"battery": 1}, gravity_json=SHARED / "gravity.json" if (SHARED / "gravity.json").exists() else None,
                         options={"max_frames": 8, "iters_init": 150, "iters_step": 60})
    ok("fit_object_poses: per-frame poses with IoU > 0.4", r["battery"]["iou_mean"] and r["battery"]["iou_mean"] > 0.4, f"{r}")


if __name__ == "__main__":
    test_video_io()
    test_table_plane()
    frames = real_frames()
    if frames is None:
        skip("real-footage sections", f"no footage at {FOOTAGE}")
    else:
        test_sam3(frames)
        test_sam2(frames)
        test_geocalib(frames)
        test_vda(frames)
        test_hamer(frames)
        test_sam3d(frames)
        test_void(frames)
        test_pose_fit(frames)
    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed; scratch {TMP}; shared {SHARED}")
    sys.exit(1 if failed else 0)
