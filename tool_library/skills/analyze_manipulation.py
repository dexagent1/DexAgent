"""Which hand manipulates which object, when, and where it first touches.

    from tool_library.skills.analyze_manipulation import analyze_manipulation
    rep = analyze_manipulation(object_masks={"cup": masks_T_H_W},
                               hand_points={"right": {t: uv_N_2, ...}, "left": {...}},
                               object_positions={"cup": world_T_3})
    rep["objects"]["cup"]["manipulated_by"], rep["hands"]["right"]["touches"]["cup"]

The measured facts a manipulation plan is written from, computed from
per-frame object masks and per-frame hand pixel points (projected hand mesh
vertices or keypoints). Contact timing uses pixel proximity (4 px to the
mask), because silhouettes are exact where monocular hand depth is not. The
manipulating hand is the one that MOVES WITH the object step by step while
it moves, never merely the one nearest to it: a resting hand next to a
carried object logs contact frames too.
"""
from __future__ import annotations

TOOL = {
    "name": "analyze_manipulation",
    "title": "Analyze manipulation",
    "type": "skill",
    "category": "Measurements",
    "summary": "From per-frame object masks and hand pixel points: contact episodes per hand and object, the manipulating hand by co-motion, and the grasp region on the object.",
    "description": (
        "For every frame it builds a distance transform of each object mask and reads the "
        "nearest distance of each hand's pixel points to it; a hand touches when that is "
        "within contact_px (4) for at least sustain (3) consecutive sampled frames, giving "
        "first/last contact frames and the region on the object's bounding box where the "
        "first contact happens (as fractions and words such as 'top-left'). Which hand "
        "manipulates a moving object is decided by co-motion: over the object's motion "
        "window the per-step pixel displacement of the hand centroid and of the object "
        "centroid are compared step by step, summing the aligned component; proximity only "
        "breaks the tie when no hand moved with the object. Object world positions, when "
        "given, provide the motion window (6 cm, 4 frames), travel, start/end and lift. "
        "Inputs are arrays and dictionaries, so any detector, hand estimator and camera work."
    ),
    "inputs": [
        {"name": "object_masks", "type": "dict name -> ndarray [T, H, W] bool | callable(name, t)", "description": "Per-frame object masks."},
        {"name": "hand_points", "type": "dict side -> {frame: ndarray [N, 2]}", "description": "Pixel points of each hand per frame (projected mesh vertices, keypoints)."},
        {"name": "object_positions", "type": "dict name -> ndarray [T, 3] | None", "description": "World positions per frame (camera-motion compensated), for motion window and travel."},
        {"name": "contact_px, sustain, stride", "type": "float, int, int", "description": "Contact distance, persistence, frame stride. Defaults 4, 3, auto (~110 samples)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "objects: moved, motion_window, start/end, travel, lift, manipulated_by, hand_evidence; hands: touches per object with first/last contact frame, contact frames, grasp_region_on_object; look_at_frames."},
    ],
    "requirements": ["numpy", "scipy"],
    "tags": ["manipulation", "contact", "hand-object", "planning"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "analyze_manipulation",
}

import numpy as np

from tool_library.common import trajectory as TJ

MASK_DOWN = 4


def _region_words(u, v):
    return f"{'top' if v < 0.34 else ('bottom' if v > 0.66 else 'middle')}-" \
           f"{'left' if u < 0.34 else ('right' if u > 0.66 else 'center')}"


def analyze_manipulation(object_masks, hand_points, object_positions=None, contact_px=4.0, sustain=3,
                         stride=None, n_frames=None):
    from scipy.ndimage import distance_transform_edt
    get = object_masks if callable(object_masks) else (lambda name, t: object_masks[name][t])
    names = list(object_positions or []) if callable(object_masks) else list(object_masks)
    if n_frames is None:
        n_frames = min(([len(m) for m in object_masks.values()] if not callable(object_masks) else [])
                       + [len(p) for p in (object_positions or {}).values()] or [0])
        if not n_frames:
            n_frames = max(max(d) for d in hand_points.values()) + 1
    N = int(n_frames)
    stride = stride or max(1, N // 110)
    objects = {}
    for name in names:
        o = {}
        if object_positions and name in object_positions:
            W = np.asarray(object_positions[name], float)[:N]
            win = TJ.motion_window(W)
            start, end = np.median(W[:5], axis=0), np.median(W[-5:], axis=0)
            o.update(moved=win is not None, motion_window_frames=list(win) if win else None,
                     travel_m=round(float(np.linalg.norm(W.max(0) - W.min(0))), 3),
                     start_world=[round(float(x), 3) for x in start], end_world=[round(float(x), 3) for x in end],
                     start_to_end_m=round(float(np.linalg.norm(end - start)), 3),
                     max_lift_m=round(float(W[:, 2].max() - start[2]), 3))
        objects[name] = o
    dists, pix_at, hand_uv, obj_uv = {}, {}, {}, {}
    for t in range(0, N, stride):
        dt_maps = {}
        for name in names:
            try:
                m = np.asarray(get(name, t), bool)
            except (KeyError, IndexError):
                continue
            if not m.any():
                continue
            small = m[::MASK_DOWN, ::MASK_DOWN]
            dt_maps[name] = (distance_transform_edt(~small) * MASK_DOWN, m)
            ys, xs = np.where(m)
            obj_uv.setdefault(name, {})[t] = (float(xs.mean()), float(ys.mean()))
        for side, pts in hand_points.items():
            if t not in pts:
                continue
            uv = np.asarray(pts[t], float)
            hand_uv.setdefault(side, {})[t] = (float(uv[:, 0].mean()), float(uv[:, 1].mean()))
            for name, (dt, m) in dt_maps.items():
                H, Wd = dt.shape
                iu = np.clip((uv[:, 0] / MASK_DOWN).astype(int), 0, Wd - 1)
                iv = np.clip((uv[:, 1] / MASK_DOWN).astype(int), 0, H - 1)
                d = dt[iv, iu]
                dists.setdefault((side, name), []).append((t, float(d.min())))
                if d.min() <= contact_px:
                    ys, xs = np.where(m)
                    near = d <= contact_px
                    u_f = (float(np.median(uv[near, 0])) - xs.min()) / max(1, xs.max() - xs.min())
                    v_f = (float(np.median(uv[near, 1])) - ys.min()) / max(1, ys.max() - ys.min())
                    pix_at.setdefault((side, name), {})[t] = (u_f, v_f)
    hands = {s: {"touches": {}} for s in hand_points}
    for (side, name), seq in dists.items():
        ts = np.array([t for t, _ in seq]); ds = np.array([d for _, d in seq])
        touch = ds <= contact_px
        first = last = None; run = 0
        for i, tv in enumerate(touch):
            run = run + 1 if tv else 0
            if run >= sustain and first is None:
                first = int(ts[i - sustain + 1])
            if tv and run >= sustain:
                last = int(ts[i])
        if first is None:
            continue
        e = {"first_contact_frame": first, "last_contact_frame": last, "contact_frames": int(touch.sum()),
             "min_dist_px": round(float(ds.min()), 1)}
        reg = pix_at.get((side, name), {})
        nf = [reg[t] for t in sorted(reg) if t >= first][:3]
        if nf:
            u, v = float(np.median([p[0] for p in nf])), float(np.median([p[1] for p in nf]))
            e["grasp_region_on_object"] = {"u_frac": round(u, 2), "v_frac": round(v, 2), "words": _region_words(u, v)}
        hands[side]["touches"][name] = e
    for name, ob in objects.items():
        if not ob.get("moved"):
            # without world positions, use the mask centroid path for the window
            if "moved" not in ob and name in obj_uv and len(obj_uv[name]) > 4:
                ts = sorted(obj_uv[name]); C = np.array([obj_uv[name][t] for t in ts])
                d = np.linalg.norm(C - C[0], axis=1); idx = np.flatnonzero(d > 20.0)
                ob["moved"] = len(idx) >= 4
                ob["motion_window_frames"] = [ts[idx[0]], ts[idx[-1]]] if ob["moved"] else None
            if not ob.get("moved"):
                ob["manipulated_by"] = None
                continue
        w0, w1 = ob["motion_window_frames"]
        cand = {}
        for side in hands:
            if name not in hands[side]["touches"]:
                continue
            n = sum(1 for t, d in dists[(side, name)] if w0 - 5 <= t <= w1 + 5 and d <= contact_px)
            ts = sorted(t for t in hand_uv.get(side, {}) if w0 - 5 <= t <= w1 + 5 and t in obj_uv.get(name, {}))
            travel = co = align = 0.0
            if len(ts) >= 2:
                Hc = np.array([hand_uv[side][t] for t in ts]); Oc = np.array([obj_uv[name][t] for t in ts])
                dH, dO = np.diff(Hc, axis=0), np.diff(Oc, axis=0)
                nH, nO = np.linalg.norm(dH, axis=1), np.linalg.norm(dO, axis=1)
                travel = float(nH.sum())
                mov = nO > 1.0
                if mov.any():
                    cs = np.einsum("ij,ij->i", dH[mov], dO[mov]) / np.maximum(nH[mov] * nO[mov], 1e-9)
                    co = float((np.minimum(nH[mov], nO[mov]) * np.clip(cs, 0.0, None)).sum())
                    align = float(np.average(cs, weights=np.maximum(nO[mov], 1e-9)))
            cand[side] = {"contact_frames": n, "hand_travel_px": round(travel, 1),
                          "motion_align": round(align, 3), "co_motion": round(co, 1)}
        ob["hand_evidence"] = cand
        if cand:
            best = max(cand, key=lambda s: (cand[s]["co_motion"], cand[s]["contact_frames"]))
            if cand[best]["co_motion"] <= 0.0:
                best = max(cand, key=lambda s: cand[s]["contact_frames"])
        else:
            best = None
        ob["manipulated_by"] = best
    look = sorted({f for ob in objects.values() if ob.get("moved")
                   for f in (ob["motion_window_frames"][0], sum(ob["motion_window_frames"]) // 2, ob["motion_window_frames"][1])})
    return {"n_frames": N, "stride": stride, "objects": objects, "hands": hands, "look_at_frames": look}
