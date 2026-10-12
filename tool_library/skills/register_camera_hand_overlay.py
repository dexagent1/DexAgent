"""The video camera in the rig frame from the robot hands at their home pose laid over the demonstrator's hands."""
from __future__ import annotations

TOOL = {
    "name": "register_camera_hand_overlay",
    "title": "Register camera by hand overlay",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Estimates the video camera's pitch, yaw, roll, height and foot position in the setup's rig frame by rendering the robot hands at the home keyframe through candidate cameras and maximising their overlap with the human hand masks, optionally held to a table-plane (gravity + depth) prior; writes camera.json in register_table_camera's format and a check image.",
    "description": (
        "The setup's home keyframe is the pose every demonstration starts from, so the human hands in a frame where "
        "they rest at the start and the robot hands at home should cover the same pixels. The skill renders the "
        "robot hands' visible meshes (no arm: at home it is outside the ego view) as projected convex hulls, which "
        "needs no OpenGL, and scores a camera by 1 - IoU with the human hand mask plus a symmetric chamfer term that "
        "keeps the score informative when the silhouettes do not overlap yet. Human pixels on the elbow side of each "
        "robot hand's wrist are ignored, because a hand-and-arm mask (SAM 3's 'hand' prompt returns the forearm too) "
        "has no robot counterpart. The search is a coarse grid over yaw and the foot position (x, y) at the initial "
        "pitch, roll and height, then Nelder-Mead over all six numbers from the best grid cells, inside a box around "
        "the initial guess. A table prior (pitch, roll from gravity; height from a depth plane: estimate_gravity, "
        "fit_camera_to_table_plane) adds a Gaussian penalty with the given sigmas. Accuracy depends on how close the "
        "demonstrator's hands are to the robot's home pose. With masks that match it (the home hands rendered through a "
        "known camera) the estimate lands within 1-3 degrees and 1-3 cm, and pixels on the table move by 0.4-1.3 cm "
        "(median) with a table prior, 1-5 cm without; the residual is a real ambiguity, not the optimiser: both hands lie "
        "at about one depth, so yaw trades against the sideways foot position and pitch against height at almost "
        "the same overlap. On the paper's bench footage the resting hands sit about 10 cm inward, 8 cm forward and on "
        "the cloth (10 cm below the home hands); the overlap then stays below IoU 0.6 and the estimate is worse than a "
        "reasonable initial guess. So the result is gated: below min_iou (0.7) the overlay is REJECTED, camera.json is "
        "the initial guess with the prior's pitch, roll and height, and the rejected estimate is kept for inspection. "
        "hand_pose='on_table' with fit_spread=True models hands resting on the cloth and spread differently from home. "
        "Writes camera.json (width, height, fx, fy, cx, cy, fovy_deg, pos, quat_wxyz, R_cv, pitch_deg, yaw_deg, "
        "height_above_table_m, plus source, accepted, the overlap and the search settings) and camera_check.png (table "
        "grid, human mask outline in red, robot hands at the written camera in blue)."
    ),
    "inputs": [
        {"name": "frame", "type": "path", "description": "A video frame where the demonstrator's hands rest at the start pose (frame 0 often shows no hands: pick the first frame they rest in)."},
        {"name": "hand_mask", "type": "path | ndarray", "description": "Binary hand mask at the frame's resolution (masks/hands.png), or a segment_with_text_prompts out_dir (the frame%05d.png matching `frame`, else its only mask)."},
        {"name": "out_json", "type": "path", "description": "Where camera.json is written; camera_check.png goes next to it."},
        {"name": "setup_dir", "type": "path | None", "description": "Setup folder with scene.xml (keyframe `home`) and setup.yaml; default the repository's setup/."},
        {"name": "fovy_deg", "type": "float | None", "description": "Vertical field of view of the video camera; or give intrinsics. Default: the initial camera's (the setup's ego camera)."},
        {"name": "intrinsics", "type": "dict | None", "description": "{fx, fy, cx, cy} in pixels at the frame's resolution."},
        {"name": "init", "type": "path | dict | None", "description": "Initial camera: a camera.json (register_table_camera's estimate) or {pitch_deg, yaw_deg, roll_deg, height_m, foot_xy}. Default the setup's ego camera."},
        {"name": "table_prior", "type": "path | dict | None", "description": "{pitch_deg (below horizontal), roll_deg, height_m} or the output of estimate_gravity / fit_camera_to_table_plane (gravity_cam, camera_height_m). Default none."},
        {"name": "prior_sigma", "type": "list[3]", "description": "Prior sigmas: pitch deg, roll deg, height m. Default [2.0, 1.5, 0.02]."},
        {"name": "xy_box_m", "type": "float", "description": "Search half-width of the camera foot around the initial guess. Default 0.15."},
        {"name": "yaw_box_deg", "type": "float", "description": "Search half-width of yaw. Default 15."},
        {"name": "hand_pose", "type": "str", "description": "'home': the keyframe as it is; 'on_table': each home hand lowered rigidly onto the tabletop (a resting human hand lies on the cloth). Default 'home'."},
        {"name": "fit_spread", "type": "bool", "description": "Also solve a mirrored inward/outward shift of the two hands (up to spread_box_m) and report it. Default false."},
        {"name": "spread_box_m", "type": "float", "description": "Limit of that shift. Default 0.12."},
        {"name": "keyframe", "type": "str", "description": "The robot keyframe the hands are rendered at. Default 'home'."},
        {"name": "work_width", "type": "int", "description": "Image width the score is computed at (speed vs precision). Default 160."},
        {"name": "min_iou", "type": "float", "description": "Overlap below which the estimate is rejected and the initial guess (with the prior) is written instead. Default 0.7."},
        {"name": "seed", "type": "int", "description": "Random seed of the refinement starts. Default 0."},
    ],
    "outputs": [
        {"name": "camera", "type": "dict", "description": "camera.json content (register_table_camera's keys) with source 'hand_overlay' (or 'hand_overlay_rejected_initial_guess'), accepted, overlap_iou, overlap_iou_init, params {pitch_deg, yaw_deg, roll_deg, height_m, x, y, spread_m}, prior, hand_pose."},
        {"name": "accepted", "type": "bool", "description": "The overlap reached min_iou and camera.json is the overlay's estimate."},
        {"name": "out_json / check_png / rejected_json", "type": "path", "description": "The written files; rejected_json holds the overlay's estimate when it was rejected."},
        {"name": "iou / iou_init / cost", "type": "float", "description": "Overlap of the robot hands with the (trimmed) human mask at the estimate and at the initial guess; the final score."},
        {"name": "warnings", "type": "list[str]", "description": "Low overlap (the hands do not rest at the home pose), a parameter at the edge of its search box."},
    ],
    "status": "tested",
    "entrypoint": "register_camera_hand_overlay",
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "opencv-python", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["camera", "registration", "extrinsics", "hands", "overlay", "home pose", "stage 2"],
    "cli": "python -m tool_library.skills.register_camera_hand_overlay frame00095.jpg masks/hands.png camera.json",
}

import json
import math
import sys
from pathlib import Path

import numpy as np

from tool_library.common import human_prior as HP
from tool_library.common import rigcam as RC

DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "setup"


def _read_mask(hand_mask, frame_path, W, H):
    import cv2
    if isinstance(hand_mask, np.ndarray):
        m = hand_mask
    else:
        p = Path(hand_mask)
        if p.is_dir():
            stem = Path(frame_path).stem
            cand = p / f"{stem}.png"
            pngs = sorted(p.glob("*.png"))
            if cand.exists():
                p = cand
            elif len(pngs) == 1:
                p = pngs[0]
            else:
                raise ValueError(f"hand_mask folder {p} has no {stem}.png and more than one mask; pass the mask file")
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(p)
    if m.ndim == 3:
        m = m[..., -1] if m.shape[2] == 4 else m.max(axis=2)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    m = m > 0
    if m.sum() < 50:
        raise ValueError("hand_mask has (almost) no pixels: segment the hands on a frame where they rest at the start pose")
    return m


def _init_params(init, base):
    """(pitch, yaw, roll, height, x, y) from a camera.json/dict or an angles dict."""
    if init is None:
        init = base
    if isinstance(init, (str, Path)):
        init = json.loads(Path(init).read_text())
    if "R_cv" in init:
        p, y, r = HP.camera_angles(init)
        return np.array([p, y, r, float(init["pos"][2]), float(init["pos"][0]), float(init["pos"][1])])
    try:
        foot = init.get("foot_xy", base["pos"][:2])
        return np.array([float(init["pitch_deg"]), float(init.get("yaw_deg", 90.0)), float(init.get("roll_deg", 0.0)),
                         float(init["height_m"]), float(foot[0]), float(foot[1])])
    except KeyError as e:
        raise ValueError(f"init needs a camera.json or pitch_deg and height_m (missing {e})") from None


def _prior(table_prior):
    if table_prior is None:
        return None
    if isinstance(table_prior, (str, Path)):
        table_prior = json.loads(Path(table_prior).read_text())
    tp = table_prior.get("plane", table_prior) if isinstance(table_prior, dict) else table_prior
    out = {}
    if "gravity_cam" in tp:
        out["pitch_deg"], out["roll_deg"] = HP.angles_from_gravity(tp["gravity_cam"])
    elif "pitch_deg" in tp:
        if "camera_height_m" in tp or "up_cam" in tp:
            raise ValueError("table_prior looks like estimate_gravity / fit_camera_to_table_plane output (pitch negative = "
                             "looking down) but has no gravity_cam; pass gravity_cam, or {pitch_deg (below horizontal), roll_deg, height_m}")
        out["pitch_deg"] = float(tp["pitch_deg"])
        if "roll_deg" in tp:
            out["roll_deg"] = float(tp["roll_deg"])
    for k in ("height_m", "camera_height_m"):
        if k in tp and tp[k] is not None:
            out["height_m"] = float(tp[k])
    if not out:
        raise ValueError("table_prior gives none of pitch_deg, roll_deg, height_m, gravity_cam, camera_height_m")
    return out


def register_camera_hand_overlay(frame, hand_mask, out_json, setup_dir=None, fovy_deg=None, intrinsics=None, init=None,
                                 table_prior=None, prior_sigma=(2.0, 1.5, 0.02), xy_box_m=0.15, yaw_box_deg=15.0,
                                 hand_pose="home", fit_spread=False, spread_box_m=0.12, keyframe="home", work_width=160,
                                 min_iou=0.7, seed=0):
    import cv2
    import yaml
    from scipy.optimize import minimize
    img = cv2.imread(str(frame))
    if img is None:
        raise FileNotFoundError(frame)
    H, W = img.shape[:2]
    if hand_pose not in ("home", "on_table"):
        raise ValueError("hand_pose must be 'home' or 'on_table'")
    setup = Path(setup_dir) if setup_dir else DEFAULT_SETUP
    s = yaml.safe_load((setup / "setup.yaml").read_text())
    ego = s["cameras"]["ego"]
    base = RC.camera_from_mujoco(ego["width"], ego["height"], ego["fovy_deg"], ego["pos"], ego["quat_wxyz"])
    # intrinsics at the frame's resolution
    if intrinsics is not None:
        fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
    else:
        if fovy_deg is None:
            src = init if isinstance(init, dict) and "fy" in init else (json.loads(Path(init).read_text()) if isinstance(init, (str, Path)) else None)
            if src is not None and "fy" in src:
                fovy_deg = math.degrees(2 * math.atan(0.5 * src["height"] / src["fy"]))
            else:
                fovy_deg = base["fovy_deg"]
        fx = fy = 0.5 * H / math.tan(math.radians(float(fovy_deg)) / 2)
        cx, cy = W / 2.0, H / 2.0
    p0 = _init_params(init, base)
    prior = _prior(table_prior)
    sig = [float(v) for v in prior_sigma]
    if prior:                                    # start from the prior where it speaks
        if "pitch_deg" in prior:
            p0[0] = prior["pitch_deg"]
        if "roll_deg" in prior:
            p0[2] = prior["roll_deg"]
        if "height_m" in prior:
            p0[3] = prior["height_m"]
    mask = _read_mask(hand_mask, frame, W, H)
    hulls = HP.robot_hand_hulls(setup / "scene.xml", keyframe=keyframe)
    box_lo = np.array([p0[0] - 12, p0[1] - yaw_box_deg, p0[2] - 6, max(p0[3] - 0.15, 0.05), p0[4] - xy_box_m, p0[5] - xy_box_m])
    box_hi = np.array([p0[0] + 12, p0[1] + yaw_box_deg, p0[2] + 6, p0[3] + 0.15, p0[4] + xy_box_m, p0[5] + xy_box_m])
    roots = {sd: v["root_pos"] for sd, v in hulls.items()}
    tips = {sd: v["root_pos"] + v["root_R"][:, 2] * 0.19 for sd, v in hulls.items()}
    drop = {sd: (-float(np.concatenate(v["parts"])[:, 2].min()) if hand_pose == "on_table" else 0.0) for sd, v in hulls.items()}

    def parts_of(q):
        spread = float(q[6]) if fit_spread else 0.0
        return HP.pose_hands(hulls, drop_to_table=hand_pose == "on_table", spread_m=spread)

    def make_score(width):
        """The score at a working resolution: 1 - IoU + chamfer (+ prior), outside the box a steep wall."""
        sc = float(width) / W
        w, h = int(round(W * sc)), int(round(H * sc))
        hm = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA) > 0
        ys, xs = np.nonzero(hm)
        pix = np.stack([xs, ys], axis=1).astype(float)
        dt_h = cv2.distanceTransform((~hm).astype(np.uint8), cv2.DIST_L2, 3)

        def cam_of(q):
            return HP.camera_from_params(w, h, fx * sc, fy * sc, cx * sc, cy * sc, q[0], q[1], q[2], q[4], q[5], q[3])

        def valid_human(cam, q):
            """Human pixels on the finger side of the nearest robot hand's wrist (forearm pixels have no robot twin)."""
            spread = float(q[6]) if fit_spread else 0.0
            segs = []
            for sd in roots:
                off = np.array([(1.0 if sd == "left" else -1.0) * spread, 0.0, drop[sd]])
                uv, z = RC.project(cam, np.array([roots[sd] + off, tips[sd] + off]))
                if (z <= 0.02).any():
                    return np.ones(len(pix), bool)
                segs.append(uv)
            dmin = tt = None
            for a, b in segs:
                ab = b - a
                t = ((pix - a) @ ab) / max(float(ab @ ab), 1e-6)
                d = np.linalg.norm(pix - (a + np.clip(t, 0, 1)[:, None] * ab), axis=1)
                if dmin is None:
                    dmin, tt = d, t
                else:
                    closer = d < dmin
                    dmin, tt = np.where(closer, d, dmin), np.where(closer, t, tt)
            return tt > -0.15                     # 15 % of the hand length past the wrist still counts (the wrist itself)

        def score(q, return_parts=False):
            q = np.asarray(q, float)
            if np.any(q[:6] < box_lo - 1e-9) or np.any(q[:6] > box_hi + 1e-9) or (fit_spread and abs(q[6]) > spread_box_m):
                out = 5.0 + float(np.sum(np.clip(box_lo - q[:6], 0, None) + np.clip(q[:6] - box_hi, 0, None)))
                return (out, 0.0) if return_parts else out
            cam = cam_of(q)
            r = HP.render_silhouette(cam, parts_of(q)) > 0
            keep = valid_human(cam, q)
            hv = np.zeros_like(hm)
            hv[ys[keep], xs[keep]] = True
            if r.sum() < 10 or hv.sum() < 10:
                return (3.0, 0.0) if return_parts else 3.0
            iou = float((r & hv).sum()) / float((r | hv).sum())
            dt_r = cv2.distanceTransform((~r).astype(np.uint8), cv2.DIST_L2, 3)
            cham = (float(dt_h[r].mean()) + float(dt_r[hv].mean())) / h
            c = 1.0 - iou + 0.5 * cham
            if prior:
                pen = 0.0
                if "pitch_deg" in prior:
                    pen += ((q[0] - prior["pitch_deg"]) / sig[0]) ** 2
                if "roll_deg" in prior:
                    pen += ((q[2] - prior["roll_deg"]) / sig[1]) ** 2
                if "height_m" in prior:
                    pen += ((q[3] - prior["height_m"]) / sig[2]) ** 2
                c += 0.02 * pen
            if fit_spread:
                c += 0.02 * (q[6] / 0.1) ** 2
            return (c, iou) if return_parts else c
        return score

    steps = np.array([3.0, 3.0, 1.5, 0.03, 0.03, 0.04] + ([0.03] if fit_spread else []))
    rng = np.random.default_rng(seed)

    def nm(score, st, scale=1.0, iters=600):
        simplex = [st]
        for i in range(len(st)):
            e = st.copy(); e[i] += scale * steps[i] * (1 if rng.random() < 0.5 else -1); simplex.append(e)
        return minimize(score, st, method="Nelder-Mead", options=dict(maxiter=iters, xatol=1e-4, fatol=1e-6, initial_simplex=np.array(simplex)))

    coarse, mid, fine = make_score(max(80, work_width // 2)), make_score(work_width), make_score(2 * work_width)
    q_init = np.r_[p0, [0.0]] if fit_spread else p0.copy()
    # 1. coarse grid over the planar placement (yaw, x, y) and the spread, at the initial pitch, roll, height
    cands = []
    spreads = [0.0] if not fit_spread else list(np.linspace(-0.5, 1.0, 4) * spread_box_m)
    for dyaw in np.linspace(-yaw_box_deg, yaw_box_deg, 7):
        for dx in np.linspace(-xy_box_m, xy_box_m, 5):
            for dy in np.linspace(-xy_box_m, xy_box_m, 7):
                for spr in spreads:
                    q = p0.copy(); q[1] += dyaw; q[4] += dx; q[5] += dy
                    if fit_spread:
                        q = np.r_[q, [spr]]
                    cands.append((coarse(q), q))
    cands.sort(key=lambda t: t[0])
    # 2. Nelder-Mead from the best cells and the initial guess, at low resolution
    res = [nm(coarse, q) for q in [q for _, q in cands[:5]] + [q_init]]
    best = min(res, key=lambda r: r.fun)
    # 3. basin hopping at the working resolution: restarts around the incumbent (the score has many shallow
    #    minima where one robot finger covers one human finger)
    best_q, best_c = best.x, mid(best.x)
    for k in range(6):
        st = best_q + rng.normal(0.0, 1.0, len(best_q)) * steps * (0.7 if k % 2 else 1.5)
        r = nm(mid, st, iters=400)
        if r.fun < best_c:
            best_q, best_c = r.x, r.fun
    r = nm(mid, best_q, scale=0.5, iters=500)
    if r.fun < best_c:
        best_q, best_c = r.x, r.fun
    # 4. polish at twice the working resolution
    r = nm(fine, best_q, scale=0.3, iters=400)
    q = r.x if r.fun < fine(best_q) else best_q
    score = fine
    c_init, iou_init = score(q_init, True)
    c_fin, iou_fin = score(q, True)
    names = ["pitch_deg", "yaw_deg", "roll_deg", "height_m", "x", "y"]
    as_params = lambda v: {"pitch_deg": round(float(v[0]), 3), "yaw_deg": round(float(v[1]), 3), "roll_deg": round(float(v[2]), 3),
                           "height_m": round(float(v[3]), 4), "x": round(float(v[4]), 4), "y": round(float(v[5]), 4),
                           "spread_m": round(float(v[6]), 4) if fit_spread else 0.0}
    params = as_params(q)
    accepted = bool(iou_fin >= float(min_iou))
    warnings = []
    if not accepted:
        warnings.append(f"REJECTED: the robot hands at {keyframe} cover the human hands with IoU {iou_fin:.2f} < min_iou {min_iou}: "
                        "the hands in this frame do not rest at the robot's start pose, and an overlay this poor moves the camera "
                        "by several degrees and centimetres in the wrong direction (measured on the bench footage). camera.json "
                        "is the initial guess (with the table prior's pitch, roll and height); the overlay's estimate is in "
                        "camera_overlay_rejected.json. Use a frame where the hands rest as the robot's home pose does, or "
                        "register the camera from the table plane (register_table_camera source='estimate')")
    for i, n in enumerate(names):
        if q[i] - box_lo[i] < 0.03 * (box_hi[i] - box_lo[i]) or box_hi[i] - q[i] < 0.03 * (box_hi[i] - box_lo[i]):
            warnings.append(f"{n} ended at the edge of its search box ({q[i]:.3f}); widen the box or improve the initial guess")
    meta = {"overlap_iou": round(iou_fin, 4), "overlap_iou_init": round(iou_init, 4), "params": params, "prior": prior,
            "hand_pose": hand_pose, "fit_spread": bool(fit_spread), "keyframe": keyframe, "min_iou": float(min_iou),
            "init_params": {n: round(float(v), 4) for n, v in zip(names, p0)}}
    cam_est = HP.camera_from_params(W, H, fx, fy, cx, cy, q[0], q[1], q[2], q[4], q[5], q[3])
    cam_est.update(meta, source="hand_overlay", accepted=accepted)
    out = Path(out_json)
    rejected_json = None
    if accepted:
        cam, q_used = cam_est, q
    else:
        rejected_json = RC.save(cam_est, out.with_name("camera_overlay_rejected.json"))
        cam = HP.camera_from_params(W, H, fx, fy, cx, cy, p0[0], p0[1], p0[2], p0[4], p0[5], p0[3])
        cam.update(meta, source="hand_overlay_rejected_initial_guess", accepted=False)
        q_used = q_init
    RC.save(cam, out)
    check = _draw(img, cam, mask, HP.render_silhouette(cam, parts_of(q_used)), out.with_name("camera_check.png"))
    return {"camera": cam, "accepted": accepted, "out_json": str(out), "check_png": check, "rejected_json": rejected_json,
            "iou": round(iou_fin, 4), "iou_init": round(iou_init, 4), "cost": round(float(c_fin), 5), "cost_init": round(float(c_init), 5),
            "params": params, "params_used": as_params(q_used), "warnings": warnings}


def _draw(img, cam, mask, sil, out_png):
    import cv2
    from tool_library.skills.register_table_camera import draw_check
    tmp = str(out_png)
    cv2.imwrite(tmp, img)
    draw_check(cam, tmp, tmp)
    im = cv2.imread(tmp)
    blue = np.zeros_like(im); blue[..., 0] = 255
    sel = sil > 0
    im[sel] = (0.55 * im[sel] + 0.45 * blue[sel]).astype(np.uint8)
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnt, -1, (0, 0, 255), 2)
    cv2.putText(im, f"hand overlay IoU {cam['overlap_iou']:.2f} (blue: robot hands at {cam['keyframe']}, red: human hands)",
                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(tmp, im)
    return tmp


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("frame"); ap.add_argument("hand_mask"); ap.add_argument("out_json")
    ap.add_argument("--fovy-deg", type=float); ap.add_argument("--init"); ap.add_argument("--table-prior")
    ap.add_argument("--hand-pose", default="home"); ap.add_argument("--fit-spread", action="store_true")
    a = ap.parse_args(argv)
    r = register_camera_hand_overlay(a.frame, a.hand_mask, a.out_json, fovy_deg=a.fovy_deg, init=a.init, table_prior=a.table_prior,
                                     hand_pose=a.hand_pose, fit_spread=a.fit_spread)
    print(json.dumps({k: v for k, v in r.items() if k != "camera"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
