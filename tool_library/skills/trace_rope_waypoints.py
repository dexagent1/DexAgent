"""Trace a coiled or self-crossing cord's centreline from one mask, guided by ordered waypoints.

    from tool_library.skills.trace_rope_waypoints import trace_rope_waypoints
    r = trace_rope_waypoints("masks/cable.png", "camera.json",
                             waypoints_px=[[1396, 824], [1300, 832], ..., [1486, 952]])
    r["centreline"], r["length_m"], r["radius_m"], r["crossings"], r["points"]

measure_rope takes the longest skeleton path, which cannot follow a cord through the places where it crosses or lies on
itself, so it refuses coils. Here the order of the cord is given from outside: a list of approximate pixels along the
cord from one end to the other, read off the frame (one every few centimetres, and one on each side of every crossing).
Between consecutive waypoints the centreline is the cheapest path through the mask, where a step costs more the
farther it is from the mask's medial axis, so the route hugs the middle of the cord and, with waypoints close enough,
cannot jump to a neighbouring turn. A leg whose waypoints are not connected inside the mask (a hand covers the cord) is
bridged straight and reported. The pixel path is lifted onto the table plane raised by the cord radius, smoothed and
resampled; every plan-view crossing of the cord with itself is found and the strand on top is lifted by one cord
diameter (plus a margin) with a smooth bump, so a MuJoCo or PhysX cable laid along the result does not start
interpenetrated. The strand on top is the later one along the cord by default, or as given per crossing.
"""
from __future__ import annotations

TOOL = {
    "name": "trace_rope_waypoints",
    "title": "Trace rope along waypoints",
    "type": "skill",
    "category": "Measurements",
    "summary": "Coiled or self-crossing cord mask + ordered pixel waypoints along it -> centreline in metres routed through the mask's middle between the waypoints, with the crossings found and the upper strand lifted, ready for construct_rope / construct_plug_cable / build_isaac_scene.",
    "description": (
        "For cords measure_rope refuses (coils, loops, crossings). Snaps each waypoint to the nearest mask pixel within "
        "snap_px (a waypoint with no mask pixel that close stays where it is: an occluded stretch), builds an 8-connected "
        "pixel graph of the mask whose step cost is the step length times 1 + centre_weight x (1 - min(dt / r_px, 1)) with "
        "dt the distance transform and r_px the cord's half width, and routes each leg between consecutive waypoints by "
        "Dijkstra (scipy). Unconnected legs are bridged straight. A leg whose route is longer than max_detour times the "
        "straight distance plus 4 cord widths is reported as a possible jump to another turn (add a waypoint there). The "
        "radius is the median distance-transform value along the routed pixels converted to metres at each point's depth "
        "(or radius_m when given); the pixel path is lifted onto the plane z = radius (rig frame for a camera.json, table "
        "frame for a common.tablecam camera), smoothed with an end-held moving average of smooth_win pixels and resampled to "
        "n_points. Crossings are the plan-view intersections of non-adjacent segments; at each the strand named by `over` "
        "('later' default, 'earlier', or a list per crossing) is raised by 2 x radius x (1 + lift_margin) under a cosine "
        "bump of lift_span_m. Reports the 3D minimum gap between non-adjacent stretches after lifting (negative = the laid "
        "cord starts interpenetrated, an issue). Assumes the cord lies on the table plane except where it crosses itself."
    ),
    "inputs": [
        {"name": "mask", "type": "ndarray [H, W] | path", "description": "Binary cord silhouette at the camera resolution."},
        {"name": "camera", "type": "dict | path", "description": "camera.json (or its dict) in the rig frame, or a common.tablecam camera dict (then the result is in the table frame)."},
        {"name": "waypoints_px", "type": "list[[u, v]]", "description": "Ordered pixels along the cord from end a to end b (at least 2). Put one every few centimetres and one on each side of every crossing."},
        {"name": "n_points", "type": "int", "description": "Points of the resampled centreline. Default 120."},
        {"name": "radius_m", "type": "float | None", "description": "Known cord radius in metres; default None = measured from the mask."},
        {"name": "snap_px", "type": "float", "description": "Largest distance from a waypoint to the mask it is snapped across. Default 15."},
        {"name": "centre_weight", "type": "float", "description": "Extra step cost away from the mask's medial axis. Default 4.0."},
        {"name": "max_detour", "type": "float", "description": "A leg routed longer than this times its straight distance (plus 4 cord widths) is reported. Default 2.5."},
        {"name": "smooth_win", "type": "int", "description": "End-held moving-average window over the routed pixels. Default 9."},
        {"name": "lift_crossings", "type": "bool", "description": "Raise the upper strand at every crossing. Default True."},
        {"name": "over", "type": "str | list[str]", "description": "Which strand is on top at a crossing: 'later' (default), 'earlier', or one of them per crossing in order along the cord."},
        {"name": "lift_margin", "type": "float", "description": "Extra lift as a fraction of the cord diameter. Default 0.1."},
        {"name": "lift_span_m", "type": "float", "description": "Arc length of the cosine bump around each crossing. Default 0.04."},
    ],
    "outputs": [
        {"name": "trace", "type": "dict", "description": "ok, issues; frame ('rig' | 'table'); centreline [n, 3] metres (lifted at crossings); points (= centreline, rig frame only, for build_isaac_scene); centreline_px [n, 2]; length_m; radius_m; radius_measured_m; legs [{i, from_px, to_px, routed_px, straight_px, bridged, detour}]; crossings [{s_a_m, s_b_m, xy, lifted}]; min_self_gap_m; snapped [{i, moved_px, on_mask}]."},
    ],
    "requirements": ["numpy", "scipy", "opencv-python"],
    "tags": ["measurement", "rope", "cable", "deformable", "centreline", "coil", "crossing"],
    "used_in": ["EgoDex charge_uncharge_device"],
    "status": "tested",
    "entrypoint": "trace_rope_waypoints",
    "cli": "python -c \"from tool_library.skills.trace_rope_waypoints import trace_rope_waypoints; print(trace_rope_waypoints('mask.png', 'camera.json', [[u0, v0], [u1, v1]]))\"",
}

from pathlib import Path

import numpy as np

_DEFAULTS = dict(n_points=120, radius_m=None, snap_px=15.0, centre_weight=4.0, max_detour=2.5, smooth_win=9,
                 lift_crossings=True, over="later", lift_margin=0.1, lift_span_m=0.04)


def _load_mask(mask):
    if isinstance(mask, (str, Path)):
        from PIL import Image
        arr = np.asarray(Image.open(mask))
        mask = arr[..., :3].max(axis=2) if arr.ndim == 3 else arr
    m = np.asarray(mask) > 0
    if m.ndim != 2 or m.sum() < 3:
        raise ValueError("trace_rope_waypoints: `mask` must be a non-empty 2-D binary mask")
    return m


def _camera_ops(camera):
    """(lift(uv, z) -> [N,3], depth(P) -> [N], frame name)."""
    if isinstance(camera, (str, Path)) or (isinstance(camera, dict) and "R_cv" in camera):
        from tool_library.common import rigcam as RC
        cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
        lift = lambda uv, z: np.asarray(RC.pixel_on_plane(cam, np.asarray(uv, float), z_plane=z), float)
        depth = lambda P: np.asarray(RC.project(cam, np.asarray(P, float))[1], float)
        return lift, depth, float(cam["fx"]), "rig"
    if isinstance(camera, dict) and "focal_px" in camera:
        from tool_library.common import tablecam as TC
        lift = lambda uv, z: np.asarray(TC.plane_points(np.asarray(uv, float), camera, z_above_table=z), float)
        depth = lambda P: np.asarray(TC.optical_depth(np.asarray(P, float), camera), float)
        return lift, depth, float(camera["focal_px"]), "table"
    raise ValueError("trace_rope_waypoints: `camera` must be a camera.json path/dict (with R_cv) or a common.tablecam camera dict")


def _graph(m, dt, r_px, centre_weight):
    from scipy.sparse import coo_matrix
    H, W = m.shape
    idx = -np.ones(m.shape, np.int64)
    vs, us = np.nonzero(m)
    idx[vs, us] = np.arange(len(vs))
    pen = 1.0 + float(centre_weight) * (1.0 - np.minimum(dt / max(r_px, 1e-6), 1.0))
    rows, cols, wts = [], [], []
    for dv, du in ((0, 1), (1, 0), (1, 1), (1, -1)):
        v2, u2 = vs + dv, us + du
        ok = (v2 >= 0) & (v2 < H) & (u2 >= 0) & (u2 < W)
        ok[ok] = m[v2[ok], u2[ok]]
        a, b = idx[vs[ok], us[ok]], idx[v2[ok], u2[ok]]
        step = np.hypot(dv, du)
        w = step * 0.5 * (pen[vs[ok], us[ok]] + pen[v2[ok], u2[ok]])
        rows += [a, b]; cols += [b, a]; wts += [w, w]
    n = len(vs)
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    return G, idx, vs, us


def _smooth(P, win):
    P = np.asarray(P, float)
    if win < 3 or len(P) < win:
        return P
    k, pad = np.ones(win) / win, win // 2
    out = np.stack([np.convolve(np.pad(P[:, i], pad, mode="edge"), k, mode="valid")[:len(P)] for i in range(P.shape[1])], 1)
    out[0], out[-1] = P[0], P[-1]
    return out


def _resample(P, n):
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], int(n))
    return np.stack([np.interp(t, s, P[:, i]) for i in range(P.shape[1])], 1), s[-1]


def _seg_intersect(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    w = q1 - p1
    t = (w[0] * d2[1] - w[1] * d2[0]) / den
    s = (w[0] * d1[1] - w[1] * d1[0]) / den
    if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
        return t, s
    return None


def trace_rope_waypoints(mask, camera, waypoints_px, **kw):
    unknown = set(kw) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"trace_rope_waypoints: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(_DEFAULTS)}")
    p = dict(_DEFAULTS, **kw)
    wp = np.asarray(waypoints_px, float)
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
        raise ValueError("trace_rope_waypoints: `waypoints_px` must be a list of at least 2 [u, v] pixels")
    if int(p["n_points"]) < 4:
        raise ValueError("trace_rope_waypoints: `n_points` must be at least 4")
    import cv2
    from scipy.sparse.csgraph import dijkstra
    m = _load_mask(mask)
    H, W = m.shape
    lift, depth, focal, frame = _camera_ops(camera)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    r_px = max(float(np.percentile(dt[m], 90)), 1.0)
    G, idx, vs, us = _graph(m, dt, r_px, p["centre_weight"])
    issues, snapped, nodes = [], [], []
    for i, (u, v) in enumerate(wp):
        if not (0 <= u < W and 0 <= v < H):
            raise ValueError(f"trace_rope_waypoints: waypoint {i} {[u, v]} lies outside the {W}x{H} mask")
        d2 = (us - u) ** 2 + (vs - v) ** 2
        k = int(np.argmin(d2))
        dk = float(np.sqrt(d2[k]))
        if dk <= float(p["snap_px"]):
            nodes.append(k); snapped.append({"i": i, "moved_px": round(dk, 1), "on_mask": True})
        else:
            nodes.append(None); snapped.append({"i": i, "moved_px": 0.0, "on_mask": False})
    path, legs, on_mask_flags = [], [], []
    for i in range(len(wp) - 1):
        a, b = nodes[i], nodes[i + 1]
        pa = np.array([us[a], vs[a]], float) if a is not None else wp[i]
        pb = np.array([us[b], vs[b]], float) if b is not None else wp[i + 1]
        straight = float(np.linalg.norm(pb - pa))
        seg, bridged = None, True
        if a is not None and b is not None:
            D, P = dijkstra(G, directed=False, indices=a, return_predecessors=True)
            if np.isfinite(D[b]):
                chain, k = [], b
                while k != -9999 and k >= 0:
                    chain.append(k)
                    if k == a:
                        break
                    k = P[k]
                chain = chain[::-1]
                seg = np.stack([us[chain], vs[chain]], 1).astype(float)
                bridged = False
        if seg is None:
            n_b = max(2, int(np.ceil(straight)))
            seg = np.array([pa + (pb - pa) * t for t in np.linspace(0, 1, n_b)])
        routed = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else 0.0
        detour = routed / max(straight, 1e-9)
        legs.append({"i": i, "from_px": [round(float(x), 1) for x in pa], "to_px": [round(float(x), 1) for x in pb],
                     "routed_px": round(routed, 1), "straight_px": round(straight, 1), "bridged": bool(bridged),
                     "detour": round(detour, 2)})
        if not bridged and routed > float(p["max_detour"]) * straight + 8.0 * r_px:
            issues.append(f"leg {i} ({legs[-1]['from_px']} -> {legs[-1]['to_px']}) is routed {routed:.0f} px for a "
                          f"{straight:.0f} px gap: it may run along another turn; add a waypoint between them")
        if path:
            seg = seg[1:]
        path.append(seg)
        on_mask_flags.append(np.full(len(seg), not bridged))
    cl = np.concatenate(path)
    on_mask = np.concatenate(on_mask_flags)
    if len(cl) < 4:
        raise ValueError("trace_rope_waypoints: the routed path is shorter than 4 pixels; spread the waypoints along the cord")
    ui = np.clip(np.round(cl[:, 0]).astype(int), 0, W - 1)
    vi = np.clip(np.round(cl[:, 1]).astype(int), 0, H - 1)
    rad_px = dt[vi, ui]
    use = on_mask & (rad_px > 0)
    if p["radius_m"] is not None:
        r = float(p["radius_m"])
        P0 = lift(cl, r)
        r_meas = float(np.median(rad_px[use] * depth(P0)[use] / focal)) if use.sum() >= 3 else r
    else:
        r = 0.004
        for _ in range(8):
            P0 = lift(cl, r)
            r_new = float(np.median(rad_px[use] * depth(P0)[use] / focal)) if use.sum() >= 3 else r
            if abs(r_new - r) < 1e-6:
                r = r_new; break
            r = r_new
        r_meas = r
    P0 = lift(cl, r)
    Ps = _smooth(P0, int(p["smooth_win"]))
    Pr, length = _resample(Ps, int(p["n_points"]))
    px_r, _ = _resample(_smooth(cl, int(p["smooth_win"])), int(p["n_points"]))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Pr[:, :2], axis=0), axis=1))])
    seg_len = length / max(len(Pr) - 1, 1)
    gap_idx = max(2, int(np.ceil(4.0 * r / max(seg_len, 1e-9))) + 1)
    crossings = []
    for i in range(len(Pr) - 1):
        for j in range(i + gap_idx, len(Pr) - 1):
            hit = _seg_intersect(Pr[i, :2], Pr[i + 1, :2], Pr[j, :2], Pr[j + 1, :2])
            if hit is None:
                continue
            t, u = hit
            crossings.append({"s_a_m": float(s[i] + t * (s[i + 1] - s[i])), "s_b_m": float(s[j] + u * (s[j + 1] - s[j])),
                              "xy": [round(float(x), 4) for x in Pr[i, :2] + t * (Pr[i + 1, :2] - Pr[i, :2])]})
    crossings.sort(key=lambda c: c["s_a_m"])
    over = p["over"]
    if isinstance(over, str):
        if over not in ("later", "earlier"):
            raise ValueError("trace_rope_waypoints: `over` must be 'later', 'earlier' or a list of them")
        over = [over] * len(crossings)
    elif len(over) != len(crossings):
        raise ValueError(f"trace_rope_waypoints: `over` lists {len(over)} entries for {len(crossings)} crossings")
    dz = np.zeros(len(Pr))
    if p["lift_crossings"]:
        h = 2.0 * r * (1.0 + float(p["lift_margin"]))
        half = 0.5 * float(p["lift_span_m"])
        for c, o in zip(crossings, over):
            if o not in ("later", "earlier"):
                raise ValueError("trace_rope_waypoints: every `over` entry must be 'later' or 'earlier'")
            sc = c["s_b_m"] if o == "later" else c["s_a_m"]
            w = np.clip(1.0 - np.abs(s - sc) / max(half, 1e-9), 0.0, 1.0)
            dz = np.maximum(dz, h * 0.5 * (1.0 - np.cos(np.pi * w)))
            c["lifted"] = "b" if o == "later" else "a"
    Pl = Pr.copy()
    Pl[:, 2] = Pl[:, 2] + dz
    # nearest approach of non-adjacent stretches in 3D, minus the cord diameter
    gap = np.inf
    for i in range(len(Pl)):
        j0 = i + gap_idx
        if j0 >= len(Pl):
            break
        d = np.linalg.norm(Pl[j0:] - Pl[i], axis=1)
        gap = min(gap, float(d.min()) - 2.0 * r)
    gap = None if not np.isfinite(gap) else round(gap, 5)
    if gap is not None and gap < -0.25 * r:
        issues.append(f"two stretches of the laid cord interpenetrate by {-gap * 1000:.1f} mm (turns lying side by side "
                      f"closer than a diameter): move a waypoint off the neighbouring turn or lower n_points")
    n_bridged = sum(1 for l in legs if l["bridged"])
    if n_bridged:
        issues_note = f"{n_bridged} leg(s) bridged straight (not connected in the mask)"
    else:
        issues_note = None
    out = {"ok": not issues, "issues": issues, "frame": frame,
           "centreline": np.round(Pl, 5).tolist(), "centreline_px": np.round(px_r, 1).tolist(),
           "length_m": round(float(np.linalg.norm(np.diff(Pl, axis=0), axis=1).sum()), 4),
           "radius_m": round(r, 5), "radius_measured_m": round(r_meas, 5),
           "legs": legs, "crossings": crossings, "min_self_gap_m": gap, "snapped": snapped,
           "n_bridged_legs": n_bridged}
    if issues_note:
        out["note"] = issues_note
    if frame == "rig":
        out["points"] = out["centreline"]
    return out
