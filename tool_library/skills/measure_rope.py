"""Measure a rope lying on a table from one mask: centreline in metres, length, radius.

    from tool_library.common.tablecam import camera
    from tool_library.skills.measure_rope import measure_rope
    cam = camera(focal_px=870, width=640, height=480, h_above_table_m=0.684, pitch_deg=35.9)
    m = measure_rope(mask, cam)
    m["length_m"], m["radius_m"], m["centreline_table"], m["radius_cv"], m["self_crossings"]

The skeleton of a rope mask is not a path: end caps and every place the rope
touches itself sprout spurs. The rope is the LONGEST path through the
skeleton graph (double-sweep Dijkstra), tip to tip by construction. The
radius is the distance transform's ridge value along that path, which for a
cylinder is its radius at any viewing angle. Metric lift uses the table
plane raised by the rope's own radius, which is what is being solved, so it
iterates. A mask in several pieces (a segmenter that lost the cord under a
hand or across a glare) is put back together: each piece's centreline is
joined to the chain where its end continues the chain's end within
max_gap_px and 60 degrees, and the gaps are bridged straight. The result
carries its own sensors and says when not to believe it: `reliable` is False
with the reasons in `issues` when the cord crosses or lies on itself (the
skeleton then has junctions and the longest path cannot follow the cord
through them, so the length is a lower bound), when the centreline explains
too little of the mask (a coil whose turns touch), when the radius varies
along the cord, or when a piece could not be joined.
"""
from __future__ import annotations

TOOL = {
    "name": "measure_rope",
    "title": "Measure rope",
    "type": "skill",
    "category": "Measurements",
    "summary": "Rope mask on a table camera -> ordered centreline in metres, length, radius; joins a mask in pieces, and flags (reliable false + issues) a coiled, crossing or fragmentary cord instead of returning confident wrong numbers.",
    "description": (
        "Skeletonizes the mask, builds the 8-connected skeleton graph and takes the longest "
        "path through it as the tip-to-tip centreline (spurs from caps and self-contact are "
        "dropped by construction). Reads the per-point radius in pixels from the distance "
        "transform, lifts the centreline onto the table plane raised by the current radius "
        "estimate, converts radii to metres with the per-point optical depth, and iterates "
        "until the radius converges (a 2% length correction on a typical bench, silently wrong "
        "if skipped). Smooths with an end-held moving average (a plain convolution shortens a "
        "rope by half a window per tip), resamples to n_points, and reports length, radius, "
        "diameter, the centreline in the table frame and in pixels, the tips, the radius "
        "coefficient of variation over the middle 70% and the self-crossings (junctions of the "
        "spur-pruned skeleton). A mask in several pieces is joined: pieces smaller than "
        "min_fragment_px are dropped, each piece's longest skeleton path is chained to the cord "
        "where its end continues the chain's end (gap <= max_gap_px, both turns <= 60 degrees), "
        "and the gaps are bridged straight (bridged_gaps_px; they count in the length, not in the "
        "radius). reliable is False, with the reasons in issues, when the cord crosses itself, "
        "the centreline explains under 75% of the mask area (turns lying on each other), the "
        "skeleton has much length off the path, the radius varies more than 35% or a piece could "
        "not be joined; then length_m is a lower bound and radius_m is inflated by merged strands. "
        "The two tips are unordered unless start_near_px names the end to start from (the plug "
        "end, say); radius_m fixes the radius when it is known (a mask blurred wider than the cord)."
    ),
    "inputs": [
        {"name": "mask", "type": "ndarray [H, W] bool | path", "description": "Rope silhouette (any segmenter)."},
        {"name": "camera", "type": "dict | path", "description": "The table camera (common.tablecam.camera), or a camera.json / dict from register_table_camera or read_egodex_episode; then centreline_rig and points (rig frame, for build_isaac_scene) are returned too."},
        {"name": "n_points", "type": "int", "description": "Points of the resampled centreline. Default 120."},
        {"name": "smooth_win", "type": "int", "description": "End-held moving-average window, in skeleton pixels. Default 9."},
        {"name": "radius_init_m", "type": "float", "description": "Starting radius for the plane lift. Default 0.008."},
        {"name": "radius_m", "type": "float | None", "description": "The cord's known radius in metres (from a spec or a sharper frame): used for the plane lift and returned as radius_m, the mask's own estimate as radius_measured_m. Default None (measured)."},
        {"name": "start_near_px", "type": "list[2] | None", "description": "A pixel (u, v) near the end the centreline should start from (the plug end); tip 'a' is then that end. Default None (whichever end the skeleton search finds first)."},
        {"name": "max_gap_px", "type": "float | None", "description": "Largest gap bridged between mask pieces, in pixels. Default None = 15% of the image diagonal."},
        {"name": "min_fragment_px", "type": "int", "description": "Mask pieces with fewer pixels are noise and dropped. Default 30."},
    ],
    "outputs": [
        {"name": "measurement", "type": "dict", "description": "reliable, ok (= reliable), issues; length_m, radius_m, radius_measured_m, diameter_m, centreline_table [n, 3], centreline_px [n, 2], tip_px, radius_cv, self_crossings, coverage (share of the mask the centreline explains), fragments {found, joined, dropped_px}, bridged_gaps_px, bridged_length_m, mask_area_px, bbox_table_m; with a rig camera centreline_rig and points."},
    ],
    "requirements": ["numpy", "scipy", "scikit-image", "opencv-python"],
    "tags": ["measurement", "rope", "deformable", "centreline"],
    "used_in": ["Knot the Rope", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "measure_rope",
}

import heapq
from pathlib import Path

import numpy as np

from tool_library.common import tablecam as TC


def _skeleton(mask):
    from skimage.morphology import skeletonize
    return skeletonize(np.asarray(mask) > 0)


def _graph(sk):
    pts = np.argwhere(sk)
    idx = {(int(v), int(u)): k for k, (v, u) in enumerate(pts)}
    adj = [[] for _ in pts]
    for k, (v, u) in enumerate(pts):
        for dv in (-1, 0, 1):
            for du in (-1, 0, 1):
                if dv == du == 0:
                    continue
                j = idx.get((int(v) + dv, int(u) + du))
                if j is not None:
                    adj[k].append((j, 1.0 if abs(dv) + abs(du) == 1 else np.sqrt(2.0)))
    return pts, adj


def _prune(sk, max_len):
    """Remove end branches (end pixel to the first junction) shorter than max_len: the spurs a skeleton grows at a
    rope's round caps and along a blurred edge. Branches leading to another end (an isolated stroke) are kept."""
    sk = sk.copy()
    for _ in range(3):
        pts, adj = _graph(sk)
        if len(pts) < 3:
            break
        deg = np.array([len(a) for a in adj])
        removed = False
        for e in np.nonzero(deg == 1)[0]:
            path, prev, cur = [int(e)], -1, int(e)
            while True:
                nxt = [j for j, _ in adj[cur] if j != prev]
                if len(nxt) != 1 or deg[nxt[0]] > 2:
                    hit_junction = len(nxt) == 1 and deg[nxt[0]] > 2
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
                if len(path) > max_len:
                    hit_junction = False
                    break
            if hit_junction and len(path) <= max_len:
                for k in path:
                    sk[tuple(pts[k])] = False
                removed = True
        if not removed:
            break
    return sk


def _junction_clusters(sk):
    """Number of junction clusters (8-connected groups of pixels with more than two skeleton neighbours)."""
    import cv2
    pts, adj = _graph(sk)
    j = np.zeros(sk.shape, np.uint8)
    for k, a in enumerate(adj):
        if len(a) > 2:
            j[tuple(pts[k])] = 1
    if not j.any():
        return 0
    n, _ = cv2.connectedComponents(j, connectivity=8)
    return int(n - 1)


def _longest_path(pts, adj):
    def dij(src):
        D = np.full(len(pts), np.inf); P = np.full(len(pts), -1, int); D[src] = 0.0
        q = [(0.0, src)]
        while q:
            d, k = heapq.heappop(q)
            if d > D[k] + 1e-9:
                continue
            for j, w in adj[k]:
                if d + w < D[j] - 1e-9:
                    D[j], P[j] = d + w, k
                    heapq.heappush(q, (D[j], j))
        return D, P

    D, _ = dij(0)
    a = int(np.argmax(np.where(np.isfinite(D), D, -1.0)))
    D, P = dij(a)
    b = int(np.argmax(np.where(np.isfinite(D), D, -1.0)))
    path, k = [], b
    while k != -1:
        path.append(k); k = P[k]
    return pts[path[::-1]][:, ::-1].astype(float), float(D[b])


def centreline_px(rope_mask, spur_px=None):
    """Ordered tip-to-tip centreline (N, 2) pixels (u, v) of ONE connected mask: the longest path of the spur-pruned
    skeleton. Returns (centreline, self_crossings, skeleton_length_px, path_length_px)."""
    import cv2
    m = np.asarray(rope_mask) > 0
    if spur_px is None:
        dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
        spur_px = 2.0 * float(np.percentile(dt[dt > 0], 90)) + 2.0 if (dt > 0).any() else 3.0
    sk = _prune(_skeleton(m), int(round(spur_px)))
    pts, adj = _graph(sk)
    if len(pts) < 3:
        raise ValueError("skeleton too small: is the mask empty?")
    path, plen = _longest_path(pts, adj)
    sk_len = float(sum(w for a in adj for _, w in a)) / 2.0
    return path, _junction_clusters(sk), sk_len, plen


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _end_tangent(P, at_end, k):
    """Unit direction pointing OUT of polyline P at its end (at_end) or start."""
    k = int(max(1, min(k, len(P) - 1)))
    return _unit(P[-1] - P[-1 - k]) if at_end else _unit(P[0] - P[k])


def join_fragments(paths, max_gap_px, k_tan, max_turn_deg=60.0):
    """Chain fragment centreline paths into one cord, starting from the longest: repeatedly attach the piece whose end
    best continues the chain's head or tail (gap <= max_gap_px, the gap within max_turn_deg of both ends' directions),
    bridging the gap with straight pixels. Returns (chain [N, 2], bridge mask [N], gaps_px, unjoined paths)."""
    order = sorted(range(len(paths)), key=lambda i: -len(paths[i]))
    chain = np.asarray(paths[order[0]], float)
    bridge = np.zeros(len(chain), bool)
    left = [np.asarray(paths[i], float) for i in order[1:]]
    gaps = []
    cos_max = np.cos(np.radians(max_turn_deg))
    while left:
        best = None
        for i, f in enumerate(left):
            for frag in (f, f[::-1]):
                # tail: chain end -> fragment start
                g = frag[0] - chain[-1]; dg = float(np.linalg.norm(g))
                if 0 < dg <= max_gap_px:
                    c1 = float(_end_tangent(chain, True, k_tan) @ _unit(g))
                    c2 = float(_unit(g) @ -_end_tangent(frag, False, k_tan))
                    if c1 >= cos_max and c2 >= cos_max:
                        cost = dg * (3.0 - c1 - c2)
                        if best is None or cost < best[0]:
                            best = (cost, i, frag, "tail", dg)
                # head: fragment end -> chain start
                g = chain[0] - frag[-1]; dg = float(np.linalg.norm(g))
                if 0 < dg <= max_gap_px:
                    c1 = float(_end_tangent(frag, True, k_tan) @ _unit(g))
                    c2 = float(_unit(g) @ -_end_tangent(chain, False, k_tan))
                    if c1 >= cos_max and c2 >= cos_max:
                        cost = dg * (3.0 - c1 - c2)
                        if best is None or cost < best[0]:
                            best = (cost, i, frag, "head", dg)
        if best is None:
            break
        _, i, frag, side, dg = best
        n_b = max(0, int(np.ceil(dg)) - 1)
        if side == "tail":
            a, b = chain[-1], frag[0]
            br = np.array([a + (b - a) * t for t in np.linspace(0, 1, n_b + 2)[1:-1]]).reshape(-1, 2)
            chain = np.concatenate([chain, br, frag])
            bridge = np.concatenate([bridge, np.ones(len(br), bool), np.zeros(len(frag), bool)])
        else:
            a, b = frag[-1], chain[0]
            br = np.array([a + (b - a) * t for t in np.linspace(0, 1, n_b + 2)[1:-1]]).reshape(-1, 2)
            chain = np.concatenate([frag, br, chain])
            bridge = np.concatenate([np.zeros(len(frag), bool), np.ones(len(br), bool), bridge])
        gaps.append(round(dg, 1))
        left.pop(i)
    return chain, bridge, gaps, left


def resample(P, n):
    P = np.asarray(P, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], int(n))
    return np.stack([np.interp(t, s, P[:, i]) for i in range(P.shape[1])], axis=1)


def smooth_ends_held(P, win):
    P = np.asarray(P, float)
    if win < 3:
        return P
    k, pad = np.ones(win) / win, win // 2
    out = np.stack([np.convolve(np.pad(P[:, i], pad, mode="edge"), k, mode="valid")[:len(P)] for i in range(P.shape[1])], axis=1)
    out[0], out[-1] = P[0], P[-1]
    return out


# reliability thresholds, measured on synthetic cords (a clean S-curve: coverage 0.95-1.05, off-path skeleton under
# 5%, radius cv under 0.15; a 2.5-turn coil with touching turns: coverage ~0.4, several junctions)
MIN_COVERAGE = 0.75
MAX_OFF_PATH = 0.20
MAX_RADIUS_CV = 0.35


def measure_rope(mask, camera, n_points=120, smooth_win=9, radius_init_m=0.008, radius_m=None, start_near_px=None,
                 max_gap_px=None, min_fragment_px=30):
    """`camera` may also be a camera.json path or dict from register_table_camera / read_egodex_episode
    (fx, fy, R_cv, pos); the centreline is then returned in the rig frame."""
    frame_map = None
    if isinstance(camera, (str, Path)) or (isinstance(camera, dict) and "R_cv" in camera):
        from tool_library.common import rigcam as RC
        cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
        camera, frame_map = RC.to_tablecam(cam)
    return _measure_rope_table(mask, camera, n_points, smooth_win, radius_init_m, frame_map, radius_m, start_near_px,
                               max_gap_px, min_fragment_px)


def _measure_rope_table(mask, camera, n_points, smooth_win, radius_init_m, frame_map, radius_m=None, start_near_px=None,
                        max_gap_px=None, min_fragment_px=30):
    if isinstance(mask, (str, Path)):                # a mask PNG path, as the segmentation skills write it
        from PIL import Image
        arr = np.asarray(Image.open(mask))
        mask = arr[..., :3].max(axis=2) if arr.ndim == 3 else arr
    import cv2
    rope = (np.asarray(mask) > 0).astype(np.uint8)
    if rope.sum() < 3:
        raise ValueError("measure_rope: the mask is empty")
    issues = []
    dt = cv2.distanceTransform(rope, cv2.DIST_L2, 5)
    r_px0 = float(np.percentile(dt[dt > 0], 90))
    # pinholes (a sharp kink drawn by a segmenter, mask noise) make skeleton cycles that read as crossings; the
    # inside of a real loop is far larger than a few cord widths squared
    cnts, hier = cv2.findContours(rope, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is not None:
        for c, h in zip(cnts, hier[0]):
            if h[3] >= 0 and cv2.contourArea(c) < 4.0 * np.pi * r_px0 ** 2:
                cv2.drawContours(rope, [c], -1, 1, -1)
        dt = cv2.distanceTransform(rope, cv2.DIST_L2, 5)
    spur = 2.0 * r_px0 + 2.0
    # the mask's pieces: specks are noise, every other piece is part of the cord until shown otherwise
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(rope, connectivity=8)
    pieces = [i for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= int(min_fragment_px)]
    dropped_px = int(sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, n_lab) if i not in pieces))
    if not pieces:
        raise ValueError(f"measure_rope: no mask piece of at least {min_fragment_px} pixels")
    paths, crossings, sk_len, path_len = [], 0, 0.0, 0.0
    for i in pieces:
        try:
            cl_i, x_i, sl_i, pl_i = centreline_px(lab == i, spur_px=spur)
        except ValueError:
            continue
        paths.append(cl_i); crossings += x_i; sk_len += sl_i; path_len += pl_i
    if not paths:
        raise ValueError("skeleton too small: is the mask empty?")
    H, W = rope.shape
    gap_max = float(max_gap_px) if max_gap_px is not None else 0.15 * float(np.hypot(H, W))
    cl, is_bridge, gaps, unjoined = join_fragments(paths, gap_max, k_tan=max(3, int(round(3 * r_px0))))
    if start_near_px is not None:
        q = np.asarray(start_near_px, float)
        if np.linalg.norm(cl[-1] - q) < np.linalg.norm(cl[0] - q):
            cl, is_bridge = cl[::-1].copy(), is_bridge[::-1].copy()
    if crossings:
        issues.append(f"the cord crosses or lies on itself ({crossings} skeleton junction{'s' if crossings > 1 else ''}): the "
                      f"longest skeleton path cannot follow it through a crossing, so length_m is a lower bound")
    if unjoined:
        ul = int(sum(len(u) for u in unjoined))
        issues.append(f"{len(unjoined)} mask piece(s) ({ul} px of centreline) could not be joined to the cord within "
                      f"{gap_max:.0f} px and 60 degrees: they are left out of the length (raise max_gap_px if they are the cord)")
    off_path = (sk_len - path_len) / max(path_len, 1e-9)
    if off_path > MAX_OFF_PATH:
        issues.append(f"{100 * off_path:.0f}% of the skeleton's length is off the measured path (strands lying on each "
                      f"other, a tangle or a noisy mask)")
    rad_px = np.array([dt[int(round(v)), int(round(u))] for u, v in cl])
    measured = ~is_bridge
    mid_all = np.zeros(len(cl), bool)
    mid_all[int(0.15 * len(cl)):int(0.85 * len(cl))] = True
    mid = mid_all & measured & (rad_px > 0)
    if mid.sum() < 3:
        mid = measured & (rad_px > 0)
    r = float(radius_m) if radius_m is not None else float(radius_init_m)
    r_meas = r
    for _ in range(8):                           # the plane is the table raised by the radius
        P = TC.plane_points(cl, camera, z_above_table=r)
        rad_m = rad_px * TC.optical_depth(P, camera) / camera["focal_px"]
        r_meas = float(np.median(rad_m[mid]))
        if radius_m is not None:
            break
        if abs(r_meas - r) < 1e-6:
            r = r_meas; break
        r = r_meas
    P = TC.plane_points(cl, camera, z_above_table=r)
    rad_m = rad_px * TC.optical_depth(P, camera) / camera["focal_px"]
    cv_r = float(np.std(rad_m[mid]) / max(np.mean(rad_m[mid]), 1e-9))
    if cv_r > MAX_RADIUS_CV:
        issues.append(f"the radius varies {100 * cv_r:.0f}% along the cord (a bleeding mask or strands merged side by side): "
                      f"radius_m is not the cord's")
    # how much of the mask the centreline explains: a cord of length l and radius r covers about 2 r l + pi r^2 pixels
    r_px = float(np.median(rad_px[mid]))
    measured_len_px = float(np.linalg.norm(np.diff(cl, axis=0), axis=1)[measured[1:] & measured[:-1]].sum())
    coverage = (2 * r_px * measured_len_px + np.pi * r_px ** 2) / max(float(rope.sum() - dropped_px), 1.0)
    if coverage < MIN_COVERAGE:
        issues.append(f"the centreline explains only {100 * coverage:.0f}% of the mask: turns lie on or against each other "
                      f"(a coil) or the cord is tangled, so length_m is a lower bound and radius_m is inflated")
    Ps = smooth_ends_held(P, smooth_win)
    seg = np.linalg.norm(np.diff(Ps, axis=0), axis=1)
    bridged_m = float(seg[is_bridge[1:] | is_bridge[:-1]].sum()) if is_bridge.any() else 0.0
    Pr = resample(Ps, n_points)
    reliable = not issues
    result = {"reliable": reliable, "ok": reliable, "issues": issues,
            "length_m": round(float(seg.sum()), 4), "radius_m": round(r, 5), "radius_measured_m": round(r_meas, 5),
            "diameter_m": round(2 * r, 4),
            "centreline_table": np.round(Pr, 5).tolist(), "centreline_px": np.round(resample(cl, n_points), 2).tolist(),
            "tip_px": {"a": [round(float(x), 1) for x in cl[0]], "b": [round(float(x), 1) for x in cl[-1]]},
            "radius_cv": round(cv_r, 3), "self_crossings": int(crossings), "coverage": round(float(coverage), 3),
            "off_path_skeleton": round(float(off_path), 3),
            "fragments": {"found": len(paths), "joined": len(paths) - len(unjoined), "dropped_px": dropped_px},
            "bridged_gaps_px": gaps, "bridged_length_m": round(bridged_m, 4), "mask_area_px": int(rope.sum()),
            "bbox_table_m": {"fwd": [round(float(Ps[:, 0].min()), 4), round(float(Ps[:, 0].max()), 4)],
                             "left": [round(float(Ps[:, 1].min()), 4), round(float(Ps[:, 1].max()), 4)]}}
    if frame_map is not None:                       # the registered camera was given: also report the rig frame
        from tool_library.common import rigcam as RC
        result["centreline_rig"] = RC.table_to_rig(frame_map, Pr).round(5).tolist()
        result["points"] = result["centreline_rig"]   # ready for build_isaac_scene's deformables
    return result
