"""Moving joint check: which joint of an articulated object did the video actually move?"""
from __future__ import annotations

TOOL = {
    "name": "moving_joint_check",
    "title": "Moving joint check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Opens each candidate joint in simulation, renders it from the video camera, and "
               "ranks the candidates by how well their edges match a video frame taken during "
               "the motion; optionally asserts that an expected joint ranks first.",
    "description": (
        "A four-drawer cabinet has four slide joints that are all equally drivable, and a "
        "trajectory optimized against the wrong one looks like success. Frame differencing "
        "cannot tell them apart because the arm sweeps across every drawer. This verifier "
        "tests hypotheses instead: for each candidate joint it sets the joint to its open "
        "limit, renders the scene from the camera registered to the video, and correlates the "
        "edge map of that render with the edge map of a real frame taken while the part was "
        "moving, inside the image footprint the candidates occupy. The joint whose rendering "
        "looks most like the video ranks first. With `expected` given, ok means that joint "
        "won; without it, ok means a clear winner exists (margin above `min_margin`; a single "
        "candidate needs a finite edge match)."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model, with a camera registered to the video."},
        {"name": "frame_during", "type": "str | ndarray", "description": "Video frame (path or RGB array) taken while the part is moving."},
        {"name": "camera", "type": "str", "description": "Name of the video-registered camera in the scene."},
        {"name": "joints", "type": "list[str] | str | None", "description": "Candidate joint names, a name prefix, or None for every hinge and slide joint."},
        {"name": "expected", "type": "str | None", "description": "The joint that should win, if known."},
        {"name": "frame_before", "type": "str | ndarray | None", "description": "Optional frame before the motion; if given, the still render is scored against it too and reported."},
        {"name": "min_margin", "type": "float", "description": "Required gap between the best and second correlation when no expected joint is given. Default 0.02."},
        {"name": "out_dir", "type": "str | None", "description": "If given, writes the candidate renders and the scored crop there."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; best: joint name; ranking: [{joint, edge_match, bbox}]; margin; footprint: the "
                        "scored image region; reason when a lone candidate has no score."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "articulated", "joint", "video", "grounding"],
    "used_in": ["Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "moving_joint_check",
}

from pathlib import Path

import numpy as np

from tool_library.common import scene as S


def _load_rgb(img):
    if isinstance(img, (str, Path)):
        from PIL import Image
        return np.asarray(Image.open(img).convert("RGB"), float)
    return np.asarray(img, float)


def _grad(x):
    g = x.mean(axis=2) if x.ndim == 3 else x
    return np.abs(np.diff(g, axis=0)[:, :-1]) + np.abs(np.diff(g, axis=1)[:-1, :])


def check_moving_joint(scene, frame_during, camera, joints=None, expected=None,
                       frame_before=None, min_margin=0.02, out_dir=None):
    model = S.load_model(scene)
    real = _load_rgb(frame_during)
    H, W = real.shape[:2]
    if isinstance(joints, str):
        cands = [j for j in range(model.njnt) if model.joint(j).name.startswith(joints)
                 and int(model.jnt_type[j]) in (S.HINGE, S.SLIDE)]
    elif joints:
        cands = [S.joint_id(model, j) for j in joints]
    else:
        cands = [j for j in range(model.njnt) if int(model.jnt_type[j]) in (S.HINGE, S.SLIDE)]
    if not cands:
        raise ValueError("no hinge or slide joints to test")

    # image footprint of the candidates: union of their parts, closed and open
    data0 = S.forward(model)
    masks0 = S.render_body_masks(model, data0, W, H, camera)
    foot = np.zeros((H, W), bool)
    renders, bboxes = {}, {}
    for j in cands:
        parts = S.subtree(model, int(model.jnt_bodyid[j]))
        d = S.forward(model)
        q = int(model.jnt_qposadr[j])
        lim = float(model.jnt_range[j][1]) if model.jnt_limited[j] else (
            d.qpos[q] + (0.15 if int(model.jnt_type[j]) == S.SLIDE else 1.2))
        d.qpos[q] = lim
        S.mujoco.mj_forward(model, d)
        m_open = S.render_body_masks(model, d, W, H, camera)
        m = np.zeros((H, W), bool)
        for b in parts:
            n = model.body(b).name
            m |= masks0.get(n, False) | m_open.get(n, False)
        foot |= m
        renders[j] = S.render_rgb(model, d, W, H, camera).astype(float)
        ys, xs = np.where(m)
        bboxes[j] = None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    ys, xs = np.where(foot)
    if not len(xs):
        raise ValueError("the candidate parts are not visible from this camera")
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    gr_real = _grad(real[y0:y1, x0:x1])
    ranking = []
    for j in cands:
        gr = _grad(renders[j][y0:y1, x0:x1])
        a, b = gr.ravel(), gr_real.ravel()
        score = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
        ranking.append({"joint": model.joint(j).name, "edge_match": round(score, 4), "bbox": bboxes[j]})
    ranking.sort(key=lambda r: -(r["edge_match"] if np.isfinite(r["edge_match"]) else -9))
    best = ranking[0]["joint"]
    margin = (ranking[0]["edge_match"] - ranking[1]["edge_match"]) if len(ranking) > 1 else float("inf")
    ok = (best == expected) if expected else bool(margin >= min_margin)
    reason = None
    if not expected and len(ranking) == 1 and not np.isfinite(ranking[0]["edge_match"]):
        # a lone candidate wins by default: it must at least produce a score (edges in both the render and the frame)
        ok, reason = False, (f"the only candidate {best!r} has no edge match score (a flat render or frame inside the "
                             f"footprint): nothing supports it; check the camera and the frame")
    rep = {"ok": ok, "best": best,
           "ranking": ranking, "margin": None if margin == float("inf") else round(float(margin), 4),
           "footprint": [x0, y0, x1, y1], "expected": expected}
    if reason:
        rep["reason"] = reason
    if frame_before is not None:
        gr0 = _grad(_load_rgb(frame_before)[y0:y1, x0:x1])
        still = _grad(S.render_rgb(model, data0, W, H, camera).astype(float)[y0:y1, x0:x1])
        rep["still_vs_before"] = round(float(np.corrcoef(still.ravel(), gr0.ravel())[0, 1]), 4)
    if out_dir:
        from PIL import Image
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        for j in cands:
            Image.fromarray(renders[j].astype(np.uint8)).save(o / f"open_{model.joint(j).name}.png")
        Image.fromarray(real[y0:y1, x0:x1].astype(np.uint8)).save(o / "real_crop.png")
    return rep


moving_joint_check = check_moving_joint   # the library calls every tool by its own name
