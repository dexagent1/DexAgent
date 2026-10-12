"""Is an open rope knotted? Topology of its centreline (used by rope_knot_check).

A knot is a topological property: close the two open ends far away and ask whether the closed curve is the unknot.
Counting crossings is not enough (a pile, a rope through the table and a loose tangle all have crossings). The open
centreline is closed four ways (over / under x radial / along the end tangents), each closure's planar diagram gives
signed over/under crossings. A nontrivial Alexander polynomial proves nontriviality, but the polynomial does not
uniquely identify a knot, and polynomial 1 does not prove the unknot. Named types additionally require a diagram
certificate after topology-preserving reduction. All four closures must agree. These are classifications under
the specified closure convention, not a complete recognizer for arbitrary open tangles. The rope and closures
must be embedded in 3-D, and a bounded deterministic view search must establish a regular planar diagram.
"""
from __future__ import annotations

from bisect import bisect_right

import numpy as np


# ----------------------------------------------------------------- geometry
def _seg_intersections(P: np.ndarray, closed: bool = True):
    """All pairwise crossings of the polyline's XY projection.

    Returns a list of dicts: i, j (segment indices), ui, uj (parameters in
    [0,1) along each segment), zi, zj (heights at the crossing), and the
    curve parameters si = i+ui, sj = j+uj.
    """
    P = np.asarray(P, float)
    n = len(P) - (0 if closed else 1)
    A = P[np.arange(n)]
    B = P[(np.arange(n) + 1) % len(P)]
    d = B - A
    out = []
    for i in range(n):
        di = d[i, :2]
        if np.linalg.norm(di) < 1e-12:
            continue
        for j in range(i + 2, n):
            if closed and i == 0 and j == n - 1:
                continue
            dj = d[j, :2]
            if np.linalg.norm(dj) < 1e-12:
                continue
            den = di[0] * dj[1] - di[1] * dj[0]
            if abs(den) < 1e-14:
                continue
            r = A[j, :2] - A[i, :2]
            ui = (r[0] * dj[1] - r[1] * dj[0]) / den
            uj = (r[0] * di[1] - r[1] * di[0]) / den
            if 0.0 <= ui < 1.0 and 0.0 <= uj < 1.0:
                zi = A[i, 2] + ui * d[i, 2]
                zj = A[j, 2] + uj * d[j, 2]
                out.append({"i": i, "j": j, "ui": float(ui), "uj": float(uj),
                            "zi": float(zi), "zj": float(zj),
                            "si": i + float(ui), "sj": j + float(uj),
                            "ti": d[i].copy(), "tj": d[j].copy()})
    return out


def crossings(P: np.ndarray, closed: bool = True, min_dz: float = 0.0):
    """Signed over/under crossings of the XY diagram of `P`.

    Each: over/under curve parameters, sign (+1 right-handed / -1 left),
    dz (height gap, metres).  `min_dz` drops near-level pairs (a flat
    tangent contact is not a crossing the sim has decided yet).
    """
    return _signed_crossings(_seg_intersections(P, closed), min_dz)


def _signed_crossings(intersections, min_dz=0.):
    out = []
    for c in intersections:
        dz = c["zi"] - c["zj"]
        if abs(dz) < min_dz:
            continue
        if dz >= 0:
            s_over, s_under, t_o, t_u = c["si"], c["sj"], c["ti"], c["tj"]
        else:
            s_over, s_under, t_o, t_u = c["sj"], c["si"], c["tj"], c["ti"]
        # right-hand rule, viewer on +z: positive when the under strand is
        # the over strand rotated counter-clockwise
        sgn = int(np.sign(t_o[0] * t_u[1] - t_o[1] * t_u[0])) or 1
        out.append({"s_over": float(s_over), "s_under": float(s_under),
                    "sign": sgn, "dz": float(abs(dz)),
                    "i": c["i"], "j": c["j"]})
    return out


def _projection_geometry(P, min_dz=0.):
    """Check the hypotheses of a regular closed polygonal knot diagram."""
    P = np.asarray(P, float)
    n = len(P)
    xy = P[:, :2]
    edges = np.roll(xy, -1, axis=0) - xy
    lengths2 = np.sum(edges * edges, axis=1)
    tolerance = max(1e-10, 1e-10 * float(np.ptp(P, axis=0).max()),
                    64 * np.finfo(float).eps * float(np.abs(P).max()))
    issues = []
    if np.any(lengths2 <= tolerance ** 2):
        return [], ['zero or unresolved projected edge']
    # A nonincident vertex cannot lie on an edge. This also detects collinear
    # edge overlaps and projected reversals, which crossing enumeration skips.
    delta = xy[:, None, :] - xy[None, :, :]
    position = np.clip(np.sum(delta * edges[None, :, :], axis=2) / lengths2, 0., 1.)
    distance2 = np.sum((delta - position[..., None] * edges) ** 2, axis=2)
    vertex, edge = np.indices((n, n))
    nonincident = (vertex != edge) & (vertex != (edge + 1) % n)
    if np.any(nonincident & (distance2 <= tolerance ** 2)):
        issues.append('nonincident projected vertex on edge or overlapping edges')
    # The crossing enumerator conservatively skips nearly parallel pairs.
    # They cannot be used in a certificate when their projected boxes overlap.
    lo = np.minimum(xy, np.roll(xy, -1, axis=0))
    hi = np.maximum(xy, np.roll(xy, -1, axis=0))
    for i in range(n):
        js = np.arange(i + 2, n)
        if i == 0:
            js = js[js != n - 1]
        if not len(js):
            continue
        overlap = np.all((hi[i] + tolerance >= lo[js]) & (hi[js] + tolerance >= lo[i]), axis=1)
        determinant = edges[i, 0] * edges[js, 1] - edges[i, 1] * edges[js, 0]
        if np.any(overlap & (np.abs(determinant) < 1e-14)):
            issues.append('unresolved nearly parallel projected segments')
            break
    raw = _seg_intersections(P, closed=True)
    if any(abs(c['zi'] - c['zj']) <= max(float(min_dz), tolerance) for c in raw):
        issues.append('unresolved crossing depth order')
    if raw:
        locations = np.array([xy[c['i']] + c['ui'] * edges[c['i']] for c in raw])
        gap2 = np.sum((locations[:, None, :] - locations[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(gap2, np.inf)
        if np.any(gap2 <= tolerance ** 2):
            issues.append('triple or coincident projected crossings')
    return raw, issues


def regular_diagram(P, min_dz=0.):
    """Choose the first regular view from a bounded deterministic sequence.

    Only the viewing basis changes; no vertex, closure or crossing is repaired
    or dropped. Selection depends on projection regularity, never on the knot
    type or polynomial obtained. An inconclusive bounded search fails closed.
    """
    P = np.asarray(P, float)
    views = [np.eye(3)]
    # Start with a small generic tilt: it resolves vertical closure legs while
    # usually retaining the diagram's simpler crossing layout. Larger fixed
    # alternatives handle geometries that remain degenerate in this view.
    for direction in ((.000137, .000271, 1.), (-.311, .173, 1.), (.419, -.233, 1.),
                      (.733, .449, 1.), (1., .377, .193), (.239, 1., .413),
                      (1.618, -1., .707)):
        normal = np.asarray(direction, float)
        normal /= np.linalg.norm(normal)
        reference = np.eye(3)[int(np.argmin(np.abs(normal)))]
        horizontal = np.cross(reference, normal)
        horizontal /= np.linalg.norm(horizontal)
        views.append(np.stack([horizontal, np.cross(normal, horizontal), normal]))
    attempts = []
    for i, view in enumerate(views):
        projected = P @ view.T
        raw, issues = _projection_geometry(projected, min_dz)
        attempts.append({'view': i, 'issues': issues})
        if not issues:
            return {'ok': True, 'points': projected, 'crossings': _signed_crossings(raw),
                    'view': i, 'view_matrix': view.tolist(), 'attempts': attempts}
    return {'ok': False, 'points': P, 'crossings': crossings(P, min_dz=min_dz),
            'view': None, 'view_matrix': None, 'attempts': attempts}


# ----------------------------------------------------- planning diagnostics
# These measurements are deliberately separate from check(): a visible loop,
# or an end crossing its fitted plane, is not evidence that a knot exists.
def _polyline(P):
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 4 or not np.isfinite(P).all():
        raise ValueError("centreline must contain at least four finite 3-D points")
    if np.any(np.linalg.norm(np.diff(P, axis=0), axis=1) < 1e-10):
        raise ValueError("consecutive centreline points must be distinct")
    return P


def embedding_issues(P, closed=False):
    """Reject nonembedded 3-D centrelines independently of their projection.

    This is a numerical topology check, not a rope-thickness clearance test.
    Distinct nonadjacent segments must not meet; adjacent segments may share
    their endpoint, but cannot retrace each other. In particular, a vertical
    overlap cannot disappear merely because its XY projection has zero length.
    """
    P = np.asarray(P, float)
    extent = float(np.ptp(P, axis=0).max())
    tolerance = max(1e-10, 1e-9 * extent,
                    32 * np.finfo(float).eps * float(np.abs(P).max()))
    A = P if closed else P[:-1]
    B = np.roll(P, -1, axis=0) if closed else P[1:]
    D = B - A
    lengths = np.linalg.norm(D, axis=1)
    issues = [{"kind": "zero_length_segment", "segments": [int(i)]}
              for i in np.flatnonzero(lengths <= tolerance)]
    if issues:
        return issues
    n = len(D)
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        # An ordinary corner shares one point. A reversal shares an interval.
        sine = np.linalg.norm(np.cross(D[i], D[j])) / (lengths[i] * lengths[j])
        if np.dot(D[i], D[j]) < 0 and min(lengths[i], lengths[j]) * sine <= tolerance:
            issues.append({"kind": "adjacent_backtracking", "segments": [i, j]})
    for i in range(n):
        indices = np.arange(i + 2, n)
        if closed and i == 0:
            indices = indices[indices != n - 1]
        if not len(indices):
            continue
        u, v, w = D[i], D[indices], A[i] - A[indices]
        a, c = lengths[i] ** 2, lengths[indices] ** 2
        b, d, e = v @ u, w @ u, np.sum(v * w, axis=1)
        # The convex distance problem on [0,1]^2 attains its minimum at an
        # interior stationary point or one of its four edges. Check all five
        # cases; cross products avoid cancellation for almost parallel lines.
        candidates = [
            w - np.clip(e / c, 0, 1)[:, None] * v,
            w + u - np.clip((e + b) / c, 0, 1)[:, None] * v,
            w + np.clip(-d / a, 0, 1)[:, None] * u,
            w + np.clip((b - d) / a, 0, 1)[:, None] * u - v,
        ]
        distance2 = np.min([np.sum(delta * delta, axis=1) for delta in candidates], axis=0)
        normal = np.cross(u, v)
        denominator = np.sum(normal * normal, axis=1)
        nonparallel = denominator > 0
        s, t = np.zeros(len(indices)), np.zeros(len(indices))
        s[nonparallel] = (-np.sum(np.cross(w, v) * normal, axis=1)[nonparallel]
                          / denominator[nonparallel])
        t[nonparallel] = (-np.sum(np.cross(w, u) * normal, axis=1)[nonparallel]
                          / denominator[nonparallel])
        interior = nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        delta = w[interior] + s[interior, None] * u - t[interior, None] * v[interior]
        distance2[interior] = np.minimum(distance2[interior], np.sum(delta * delta, axis=1))
        for k in np.flatnonzero(distance2 <= tolerance ** 2):
            issues.append({"kind": "nonadjacent_intersection", "segments": [i, int(indices[k])],
                           "distance_m": float(np.sqrt(max(0., distance2[k])))})
    return issues


def curve_point(P, parameter):
    """Material coordinate: integer vertex index plus fraction of next segment."""
    s = float(parameter)
    if not np.isfinite(s) or not 0 <= s <= len(P) - 1:
        raise ValueError(f"curve parameter must lie in [0, {len(P) - 1}]")
    i = min(int(s), len(P) - 2)
    return np.asarray(P[i]) + (s - i) * (np.asarray(P[i + 1]) - P[i])


def curve_arc(P, bounds):
    a, b = map(float, bounds)
    if not a < b:
        raise ValueError("loop_bounds must be an increasing pair of curve parameters")
    first, last = curve_point(P, a), curve_point(P, b)
    middle = P[np.arange(int(np.floor(a)) + 1, int(np.ceil(b)))]
    arc = np.vstack([first, middle, last])
    if len(arc) < 4:
        raise ValueError("loop_bounds must span at least three segments")
    return arc


def _polygon_measure(points, polygon):
    """Inside mask and exact distance to the line segments of a 2-D polygon."""
    X = np.atleast_2d(np.asarray(points, float))
    A, B = polygon, np.roll(polygon, -1, axis=0)
    D = B - A
    norm2 = np.sum(D * D, axis=1)
    t = np.sum((X[:, None, :] - A) * D, axis=2) / np.maximum(norm2, 1e-30)
    nearest = A + np.clip(t, 0, 1)[..., None] * D
    distance = np.min(np.linalg.norm(X[:, None, :] - nearest, axis=2), axis=1)
    inside = np.zeros(len(X), bool)
    for a, b in zip(A, B):
        crossed = (a[1] > X[:, 1]) != (b[1] > X[:, 1])
        if abs(b[1] - a[1]) > 1e-15:
            x_edge = a[0] + (X[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            inside ^= crossed & (X[:, 0] < x_edge)
    return inside, distance


def loop_geometry(P, bounds, radius=0.0, *, locate_aperture=True):
    """Fitted near-planar aperture of a specified material arc, for planning.

    The arc's ends are joined by an *imaginary* chord. Its gap, non-planarity,
    and self-intersections are exposed and invalidate passage measurements
    when large. Even a valid aperture is diagnostic, not topology evidence.
    """
    P = _polyline(P)
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    arc = curve_arc(P, bounds)
    origin = arc.mean(axis=0)
    _, singular, axes = np.linalg.svd(arc - origin, full_matrices=False)
    normal = axes[2]
    # A deterministic orientation prevents arbitrary SVD sign reversals in a
    # frame history. Temporal code also aligns to the preceding normal.
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    basis = np.stack([axes[0], np.cross(normal, axes[0])])
    xy = (arc - origin) @ basis.T
    gap = float(np.linalg.norm(arc[-1] - arc[0]))
    perimeter = float(np.linalg.norm(np.diff(arc, axis=0), axis=1).sum() + gap)
    area = .5 * abs(float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))))
    rms = float(np.sqrt(np.mean(((arc - origin) @ normal) ** 2)))
    gap_limit = max(4 * radius, .1 * perimeter)
    plane_limit = max(radius, .1 * np.sqrt(area / np.pi))
    simple = not _seg_intersections(np.column_stack([xy, np.zeros(len(xy))]), closed=True)
    issues = []
    if singular[1] < 1e-8 or area < 1e-10:
        issues.append("arc has no measurable aperture")
    if not simple:
        issues.append("projected boundary self-intersects")
    if gap > gap_limit:
        issues.append("arc endpoints are too far apart to treat the closing chord as a loop")
    if rms > plane_limit:
        issues.append("arc is too non-planar for a reliable plane-passage measurement")
    out = {"bounds": list(map(float, bounds)), "valid_aperture": not issues, "issues": issues,
           "origin": origin.tolist(), "normal": normal.tolist(), "basis": basis.tolist(),
           "area_m2": area, "perimeter_m": perimeter, "closure_gap_m": gap,
           "plane_rms_m": rms, "closure_gap_limit_m": gap_limit, "plane_rms_limit_m": plane_limit,
           "boundary": arc.tolist(), "boundary_projected": xy.tolist()}
    if locate_aperture and area > 1e-10 and simple:
        from scipy.optimize import minimize
        # Find a large interior clearance, not merely the polygon centroid
        # (which can be outside a concave aperture). Report the result as a
        # lower bound: this bounded search need not find the global maximum.
        gx, gy = np.meshgrid(np.linspace(xy[:, 0].min(), xy[:, 0].max(), 25),
                             np.linspace(xy[:, 1].min(), xy[:, 1].max(), 25))
        candidates = np.column_stack([gx.ravel(), gy.ravel()])
        inside, distances = _polygon_measure(candidates, xy)
        score = np.where(inside, distances, -distances)
        seed = candidates[int(np.argmax(score))]

        def objective(x):
            inside_, distance_ = _polygon_measure(x, xy)
            return float(-distance_[0] if inside_[0] else distance_[0])

        result = minimize(objective, seed, method="Nelder-Mead", options={"maxiter": 120, "xatol": 1e-7})
        centre = result.x if objective(result.x) < objective(seed) else seed
        interior, clearance = _polygon_measure(centre, xy)
        if interior[0]:
            out.update(aperture_center=(origin + centre @ basis).tolist(),
                       aperture_clear_radius_m=max(0., float(clearance[0]) - radius),
                       aperture_scope="lower-bound fitted-plane clearance from the selected rope boundary, including its imaginary closing chord; excludes hands and other obstacles")
    return out


def loop_candidates(P, radius=0.0, max_loops=8):
    """Near-closed arcs found from crossings in three orthogonal projections."""
    P = _polyline(P)
    candidates = []
    for order in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        for c in _seg_intersections(P[:, order], closed=False):
            bounds = [c["si"], c["sj"]]
            if bounds[1] - bounds[0] < 3:
                continue
            if any(np.max(np.abs(np.asarray(bounds) - old["bounds"])) < 1. for old in candidates):
                continue
            item = loop_geometry(P, bounds, radius, locate_aperture=False)
            item["discovery_projection"] = ["xyz"[i] for i in order[:2]]
            candidates.append(item)
    # An almost closed simple arc has no projected crossing, but has an
    # aperture. This includes a loop before its strands cross.
    whole = loop_geometry(P, [0., len(P) - 1.], radius, locate_aperture=False)
    if whole["valid_aperture"]:
        whole["discovery_projection"] = "near-closed endpoints"
        candidates.append(whole)
    candidates.sort(key=lambda x: (not x["valid_aperture"], -x["area_m2"]))
    result = []
    for i, item in enumerate(candidates[:int(max_loops)]):
        measured = loop_geometry(P, item["bounds"], radius)
        measured.update(id=i, discovery_projection=item["discovery_projection"])
        result.append(measured)
    return result


def endpoint_loop_relation(point, loop, rope_radius=0., endpoint_radius=0.):
    point = np.asarray(point, float)
    origin, normal, basis = (np.asarray(loop[k], float) for k in ("origin", "normal", "basis"))
    xy = (point - origin) @ basis.T
    inside, boundary = _polygon_measure(xy, np.asarray(loop["boundary_projected"]))
    return {"signed_plane_distance_m": float(np.dot(point - origin, normal)),
            "projection_inside": bool(inside[0]),
            "boundary_clearance_m": float(boundary[0]) - float(rope_radius) - float(endpoint_radius),
            "valid_aperture": bool(loop["valid_aperture"])}


def endpoint_passages(history, bounds, endpoint="last", radius=0., endpoint_radius=None):
    """Geometric candidates for an end passing a moving material loop.

    Checks *every supplied frame* and interpolates an observed plane crossing.
    A side change outside the polygon is not threading. Coarse recordings,
    non-planar loops, boundary contact, and deforming surfaces remain uncertain:
    callers must still use contact physics and the independent topology gate.
    """
    H = np.asarray(history, float)
    if H.ndim != 3 or H.shape[2] != 3 or len(H) < 2:
        raise ValueError("history must have shape [frames>=2, points>=4, 3]")
    if endpoint not in ("first", "last"):
        raise ValueError("endpoint must be 'first' or 'last'")
    idx = 0 if endpoint == "first" else H.shape[1] - 1
    if float(bounds[0]) <= idx <= float(bounds[1]):
        raise ValueError("tracked endpoint must lie outside loop_bounds")
    endpoint_radius = float(radius if endpoint_radius is None else endpoint_radius)
    events, rejected = [], []
    prior = None
    for frame, P in enumerate(H):
        loop = loop_geometry(P, bounds, radius, locate_aperture=False)
        normal = np.asarray(loop["normal"])
        if prior is not None and np.dot(normal, prior["normal"]) < 0:
            normal = -normal
        distance = float(np.dot(P[idx] - loop["origin"], normal))
        if abs(distance) < 1e-10:
            # A recorded point exactly on the plane must not erase a side
            # change across the next frame. Keep the last definite side.
            if prior is not None:
                prior["valid"] &= loop["valid_aperture"]
            continue
        if prior is not None and prior["distance"] * distance < 0:
            a, b = 0., 1.
            for _ in range(14):
                t = (a + b) / 2
                curve = H[prior["frame"]] * (1 - t) + P * t
                middle = loop_geometry(curve, bounds, radius, locate_aperture=False)
                middle_normal = np.asarray(middle["normal"])
                if np.dot(middle_normal, prior["normal"]) < 0:
                    middle_normal = -middle_normal
                d = float(np.dot(curve[idx] - middle["origin"], middle_normal))
                if d * prior["distance"] > 0:
                    a = t
                else:
                    b = t
            relation = endpoint_loop_relation(curve[idx], middle, radius, endpoint_radius)
            reliable = prior["valid"] and loop["valid_aperture"] and middle["valid_aperture"]
            item = {"frames": [prior["frame"], frame], "fraction": t, "point": curve[idx].tolist(),
                    "direction": "negative_to_positive" if distance > 0 else "positive_to_negative", **relation}
            if reliable and relation["projection_inside"] and relation["boundary_clearance_m"] > 0:
                events.append(item)
            else:
                item["reason"] = "invalid aperture" if not reliable else ("outside aperture" if not relation["projection_inside"] else "insufficient boundary clearance")
                rejected.append(item)
        prior = {"normal": normal, "distance": distance, "valid": loop["valid_aperture"], "frame": frame}
    return {"passage_candidates": events, "rejected_plane_crossings": rejected,
            "frames_checked": len(H), "endpoint": endpoint, "bounds": list(map(float, bounds)),
            "scope": "diagnostic geometry between recorded states only; neither continuous collision clearance nor a knot proof"}


# ------------------------------------------------------------- the closure
def closure(P: np.ndarray, mode: str = "over", how: str = "radial",
            height: float = 2.0, reach: float = 0.5) -> np.ndarray:
    """Close an open curve 'at infinity'.

    Each end is extended outward (radially from the curve's centroid, or
    along its own end tangent), lifted `height` above the curve's maximum z
    ('over') or below its minimum z ('under'), and joined. Height is positive
    clearance, not an absolute world coordinate: translating the rope must
    translate the artificial closure with it. The result is a closed
    polyline; the last vertex joins back to the first.
    """
    P = _polyline(P)
    if mode not in ('over', 'under') or how not in ('radial', 'tangent'):
        raise ValueError("closure mode must be over/under and how must be radial/tangent")
    if not np.isfinite(height) or height <= 0 or not np.isfinite(reach) or reach <= 0:
        raise ValueError("closure height and reach must be finite positive clearances")
    c = P[:, :2].mean(axis=0)
    R = float(np.linalg.norm(P[:, :2] - c, axis=1).max()) + float(reach)

    def ext(end, tan):
        if how == "tangent":
            dxy = np.asarray(tan, float)[:2]
        else:
            dxy = end[:2] - c
        if np.linalg.norm(dxy) < 1e-9:
            dxy = np.array([1.0, 0.0])
        dxy = dxy / np.linalg.norm(dxy)
        far = c + dxy * R
        return np.array([far[0], far[1], end[2]])

    zc = float(P[:, 2].max() + height) if mode == "over" else float(P[:, 2].min() - height)
    ea = ext(P[0], P[0] - P[min(3, len(P) - 1)])
    eb = ext(P[-1], P[-1] - P[max(len(P) - 4, 0)])
    top_a = np.array([ea[0], ea[1], zc])
    top_b = np.array([eb[0], eb[1], zc])
    return np.vstack([ea, P, eb, top_b, top_a])


# --------------------------------------------------------------- invariants
def _arcs_and_matrix(cr: list, n_total_param: float):
    """Alexander matrix rows from the crossing list (closed diagram)."""
    n = len(cr)
    unders = sorted(c["s_under"] for c in cr)

    def arc_of(s):
        return bisect_right(unders, s % n_total_param) % n

    rows = []
    for c in cr:
        k = arc_of(c["s_over"])
        # The incoming/outgoing arcs are adjacent in the sorted under-visit
        # order. A numerical +/- epsilon can jump across a nearby crossing.
        j_out = bisect_right(unders, c["s_under"]) % n
        i_in = (j_out - 1) % n
        rows.append((k, i_in, j_out, c["sign"]))
    return rows


def alexander(P_closed: np.ndarray, min_dz: float = 0.0, *, diagram=None) -> dict:
    """Alexander polynomial of a closed curve, normalised (lowest power
    removed, positive leading coefficient), plus the knot determinant.

    Direct XY computation is diagnostic unless its projection is regular.
    check() supplies the same validated diagram to this and the certificate.
    """
    import sympy as sp
    cr = crossings(P_closed, closed=True, min_dz=min_dz) if diagram is None else diagram
    n = len(cr)
    if n == 0:
        return {"n_crossings": 0, "poly": "1", "det": 1, "coeffs": [1]}
    visits = [c[key] for c in cr for key in ('s_over', 's_under')]
    if len(set(visits)) != len(visits) or not np.isfinite(visits).all():
        # Coincident visits are not a regular knot diagram. Preserve an
        # explicit degenerate diagnostic rather than constructing fake arcs.
        return {"n_crossings": n, "poly": "0", "det": 0, "coeffs": [0],
                "invalid_diagram": "crossing visits are coincident or nonfinite"}
    t = sp.symbols("t")
    rows = _arcs_and_matrix(cr, float(len(P_closed)))
    M = sp.zeros(n, n)
    for r, (k, i_in, j_out, sgn) in enumerate(rows):
        if k == i_in == j_out:
            # a Reidemeister-I kink on a single arc: the row is (1-t)+t-1=0
            M[r, k] += 0
            continue
        M[r, k] += 1 - t
        if sgn > 0:
            M[r, i_in] += t
            M[r, j_out] += -1
        else:
            M[r, i_in] += -1
            M[r, j_out] += t
    if n == 1:
        poly = sp.Integer(1)
    else:
        minor = M[1:, 1:]
        # Work in the exact polynomial domain. Expanding a nested Berkowitz
        # expression can take minutes for a modest tangle (~18 crossings),
        # even though its polynomial determinant has only a few terms.
        poly = sp.expand(minor.det(method="domain-ge"))
    poly = sp.Poly(poly, t) if poly != 0 else None
    if poly is None or poly.is_zero:
        # a degenerate diagram (split components from a bad closure)
        return {"n_crossings": n, "poly": "0", "det": 0, "coeffs": [0]}
    coeffs = poly.all_coeffs()[::-1]               # ascending powers
    while coeffs and coeffs[0] == 0:
        coeffs.pop(0)
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    if coeffs and coeffs[-1] < 0:
        coeffs = [-c for c in coeffs]
    cf = [int(c) for c in coeffs]
    npoly = sum(c * t ** k for k, c in enumerate(cf))
    det = abs(int(npoly.subs(t, -1)))
    return {"n_crossings": n, "poly": str(sp.expand(npoly)), "det": det,
            "coeffs": cf}


KNOWN = {
    (0,): "degenerate",
    (1,): "unknot",
    (1, -1, 1): "trefoil",
    (1, -3, 1): "figure-eight",
    (1, -1, 1, -1, 1): "cinquefoil (5_1)",
    (2, -3, 2): "three-twist (5_2)",
}


def classify(coeffs) -> str:
    """Polynomial resemblance only; different knots can share a polynomial."""
    return KNOWN.get(tuple(int(c) for c in coeffs), "other")


def diagram_certificate(cr, coeffs):
    """Conservative named-knot certificate from a single-component diagram.

    Remove Reidemeister-I curls and Reidemeister-II pairs whose over strand
    and under strand each meet both crossings consecutively, with opposite
    crossing signs. These are isotopies, not polynomial matching. An
    empty reduced diagram proves the unknot. There is only one nontrivial knot
    with at most three crossings and one additional knot with four; their
    polynomials distinguish these small diagrams. Larger unreduced diagrams
    remain unnamed, even when their polynomial resembles one of these knots.
    Failure to simplify does not establish non-equivalence.
    """
    visits = sorted([(c['s_over'], i, True) for i, c in enumerate(cr)]
                    + [(c['s_under'], i, False) for i, c in enumerate(cr)])
    word = [(i, over) for _, i, over in visits]
    removed = []
    while word:
        kink = next((word[i][0] for i in range(len(word))
                     if word[i][0] == word[(i + 1) % len(word)][0]), None)
        if kink is not None:
            removed.append({'move': 'I', 'crossings': [kink]})
            word = [entry for entry in word if entry[0] != kink]
            continue
        pair = None
        for i, (a, over) in enumerate(word):
            b, next_over = word[(i + 1) % len(word)]
            if a == b or over != next_over or cr[a]['sign'] != -cr[b]['sign']:
                continue
            other = [j for j, (k, is_over) in enumerate(word)
                     if k in (a, b) and is_over != over]
            if len(other) == 2 and (other[1] - other[0]) % len(word) in (1, len(word) - 1):
                pair = [a, b]
                break
        if pair is None:
            break
        removed.append({'move': 'II', 'crossings': pair})
        word = [entry for entry in word if entry[0] not in pair]
    remaining = len(word) // 2
    cf = tuple(int(c) for c in coeffs)
    kind = ('unknot' if remaining == 0 and cf == (1,) else
            'trefoil' if remaining == 3 and cf == (1, -1, 1) else
            'figure-eight' if remaining == 4 and cf == (1, -3, 1) else None)
    return {'type': kind, 'method': 'Reidemeister-I/II reduction and small-diagram classification',
            'input_crossings': len(cr), 'reduced_crossings': remaining,
            'removed_crossings': removed,
            'polynomial_type_hint': classify(coeffs)}


# ------------------------------------------------------------------ verdict
def check(P: np.ndarray, min_dz: float = 0.0, radius: float | None = None) -> dict:
    """The verdict for an OPEN rope centreline.

    Runs the four closures (over/under x radial/tangent).  `knot` is True
    only when ALL normalized closure polynomials agree on a non-trivial
    result. Degenerate diagrams and unresolved over/under crossings fail
    closed. `ends_free` is an informational projection diagnostic: outward
    extensions can cross a valid knot's diagram while remaining separated
    in 3-D. It does not describe whether a hand or support holds an end.
    """
    P = _polyline(P)
    if not np.isfinite(min_dz) or min_dz < 0:
        raise ValueError("min_dz must be finite and nonnegative")
    res, certificates, projections = {}, {}, {}
    geometry = {"open_rope": embedding_issues(P)}
    for mode in ("over", "under"):
        for how in ("radial", "tangent"):
            Q = closure(P, mode, how)
            key = f"{mode}/{how}"
            geometry[key] = embedding_issues(Q, closed=True)
            valid_embedding = not geometry[key] and not geometry['open_rope']
            selected = (regular_diagram(Q, min_dz) if valid_embedding else
                        {'ok': False, 'points': Q, 'crossings': crossings(Q, min_dz=min_dz),
                         'view': None, 'view_matrix': None,
                         'attempts': [{'view': None, 'issues': ['nonembedded 3-D curve; projection search skipped']}]})
            projections[key] = {k: selected[k] for k in ('ok', 'view', 'view_matrix', 'attempts')}
            res[key] = alexander(selected['points'], diagram=selected['crossings'])
            certificates[key] = diagram_certificate(selected['crossings'], res[key]['coeffs'])
            certificates[key]['regular_diagram'] = selected['ok']
            certificates[key]['valid_embedding'] = valid_embedding
            if not valid_embedding or not selected['ok']:
                certificates[key]['type'] = None
    types = {k: classify(v["coeffs"]) for k, v in res.items()}
    # Comparing coarse names is insufficient: many different polynomials
    # map to 'other'. In particular, an unresolved open tangle can yield
    # four 'other' names while its radial and tangent closures disagree.
    polynomial_agreement = len({tuple(v["coeffs"]) for v in res.values()}) == 1
    degenerate = [key for key, value in res.items() if value["coeffs"] == [0]]
    height_tolerance = max(float(min_dz), 1e-10, 1e-8 * float(np.ptp(P, axis=0).max()))
    unresolved = [{"segments": [c["i"], c["j"]], "height_gap_m": abs(c["zi"] - c["zj"])}
                  for c in _seg_intersections(P, closed=False) if abs(c["zi"] - c["zj"]) <= height_tolerance]
    invalid_embeddings = {key: value for key, value in geometry.items() if value}
    invalid_projections = [key for key, value in projections.items() if not value['ok']]
    agree = (polynomial_agreement and not degenerate and not unresolved
             and not invalid_embeddings and not invalid_projections)
    issues = []
    if not polynomial_agreement:
        issues.append("normalized closure polynomials disagree")
    if degenerate:
        issues.append(f"degenerate zero polynomial in closures: {degenerate}")
    if unresolved:
        issues.append("self-crossing over/under order is unresolved (intersecting or near-level centreline)")
    if invalid_embeddings:
        issues.append(f"3-D centreline is not embedded (self-intersection, retracing or degenerate segment): {list(invalid_embeddings)}")
    if invalid_projections:
        issues.append(f"no regular projection found within the bounded view search: {invalid_projections}")
    # are the ends free?  count crossings between the outward extension
    # segments and the rope (at the end's own height: a genuine crossing,
    # either way, means the end points back into the pile)
    ends_free = True
    for how in ("radial", "tangent"):
        Q = closure(P, "over", how)
        m = len(Q)
        for c in _seg_intersections(Q, closed=True):
            # segments 0 (ext a) and m-4 (ext b) are the extensions
            if c["i"] in (0, m - 4) or c["j"] in (0, m - 4):
                ends_free = False
    own = crossings(P, closed=False, min_dz=min_dz)
    nontrivial = agree and res['over/radial']['coeffs'] not in ([0], [1])
    certified = {v['type'] for v in certificates.values()}
    kind = (next(iter(certified)) if agree and len(certified) == 1 and None not in certified
            else 'other' if nontrivial else 'ambiguous')
    if agree and (len(certified) != 1 or None in certified):
        issues.append('polynomial agreement does not certify a named knot or the unknot; diagram reduction is inconclusive')
    verdict = {
        "knot": bool(nontrivial),
        "type": kind,
        "agree": bool(agree),
        "polynomial_agreement": bool(polynomial_agreement),
        "diagram_certificates": certificates,
        "issues": issues,
        "unresolved_crossings": unresolved,
        "embedding_issues": invalid_embeddings,
        "diagram_projections": projections,
        "ends_free": bool(ends_free),
        "closures": {k: {"type": types[k], "poly": v["poly"],
                         "det": v["det"], "n_crossings": v["n_crossings"]}
                     for k, v in res.items()},
        "n_self_crossings": len(own),
        "self_crossings": [{"s_over": round(c["s_over"], 2),
                            "s_under": round(c["s_under"], 2),
                            "sign": c["sign"], "dz_mm": round(c["dz"] * 1e3, 1)}
                           for c in own],
        "self_crossings_scope": "original XY projection diagnostic; certified topology uses the regular closure diagrams",
    }
    verdict["trefoil"] = bool(agree and kind == "trefoil")
    return verdict


# ---------------------------------------------- the crossing order, and a fast screen for planners
def seg_crossings_np(P: np.ndarray, closed: bool = False):
    """_seg_intersections vectorized (a planner screens thousands of predicted ropes): arrays i, j (segment indices),
    ui, uj (parameters in [0, 1)), zi, zj (heights at the crossing), ti, tj (segment vectors) for every crossing of
    the XY projection."""
    P = np.asarray(P, float)
    n = len(P) - (0 if closed else 1)
    A = P[np.arange(n)]
    B = P[(np.arange(n) + 1) % len(P)]
    d = B - A
    I, J = np.triu_indices(n, k=2)
    if closed and n > 2:
        keep = ~((I == 0) & (J == n - 1))
        I, J = I[keep], J[keep]
    di, dj = d[I, :2], d[J, :2]
    den = di[:, 0] * dj[:, 1] - di[:, 1] * dj[:, 0]
    r = A[J, :2] - A[I, :2]
    ok = (np.abs(den) >= 1e-14) & (np.linalg.norm(di, axis=1) >= 1e-12) & (np.linalg.norm(dj, axis=1) >= 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ui = (r[:, 0] * dj[:, 1] - r[:, 1] * dj[:, 0]) / den
        uj = (r[:, 0] * di[:, 1] - r[:, 1] * di[:, 0]) / den
    hit = ok & (ui >= 0.0) & (ui < 1.0) & (uj >= 0.0) & (uj < 1.0)
    I, J, ui, uj = I[hit], J[hit], ui[hit], uj[hit]
    zi = A[I, 2] + ui * d[I, 2]
    zj = A[J, 2] + uj * d[J, 2]
    return {"i": I, "j": J, "ui": ui, "uj": uj, "zi": zi, "zj": zj, "ti": d[I], "tj": d[J]}


def crossings_fast(P: np.ndarray, closed: bool = True, min_dz: float = 0.0):
    """crossings() on seg_crossings_np: the same signed over/under list, for screening (no regularity check)."""
    c = seg_crossings_np(P, closed)
    inter = [{"i": int(c["i"][k]), "j": int(c["j"][k]), "zi": float(c["zi"][k]), "zj": float(c["zj"][k]),
              "si": int(c["i"][k]) + float(c["ui"][k]), "sj": int(c["j"][k]) + float(c["uj"][k]),
              "ti": c["ti"][k], "tj": c["tj"][k]} for k in range(len(c["i"]))]
    return _signed_crossings(inter, min_dz)


def alexander_det(P_closed: np.ndarray, min_dz: float = 0.0) -> int:
    """|Alexander polynomial at t = -1| (the knot determinant) of a closed curve's XY diagram, numerically from the
    same matrix as alexander(): 1 unknot, 3 trefoil, 5 figure-eight. Microseconds instead of the symbolic determinant,
    for screening candidate ropes; a verdict is still check()'s (the determinant does not name a knot)."""
    cr = crossings_fast(P_closed, closed=True, min_dz=min_dz)
    n = len(cr)
    if n <= 1:
        return 1
    rows = _arcs_and_matrix(cr, float(len(P_closed)))
    M = np.zeros((n, n))
    for r, (k, i_in, j_out, _sgn) in enumerate(rows):
        if k == i_in == j_out:
            continue
        M[r, k] += 2.0                       # 1 - t at t = -1; both signs put -1 on the other two arcs
        M[r, i_in] += -1.0
        M[r, j_out] += -1.0
    return int(round(abs(np.linalg.det(M[1:, 1:]))))


_DET_TYPE = {1: "unknot", 3: "trefoil", 5: "figure-eight"}


def quick_type(P: np.ndarray, min_dz: float = 0.0) -> str:
    """check()'s four closures screened by their determinant: 'trefoil' when all four give 3 (and so on), 'ambiguous'
    when they disagree or a closure cannot be built, 'other' for any other agreed value. A candidate the screen calls
    a knot is then confirmed by check() (a determinant of 3 is also 8_19's, which no short sequence of moves ties)."""
    P = np.asarray(P, float)
    try:
        dets = {alexander_det(closure(P, mode, how), min_dz=min_dz) for mode in ("over", "under") for how in ("radial", "tangent")}
    except ValueError:
        return "ambiguous"
    if len(dets) > 1:
        return "ambiguous"
    return _DET_TYPE.get(dets.pop(), "other")


def crossing_order(P: np.ndarray, min_dz: float = 0.0) -> dict:
    """The crossings met walking along an OPEN rope from its first point: each crossing is labelled a, b, c ... in the
    order it is first met and each visit says whether the rope passes over (+) or under (-) there. The ORDER, not the
    count, decides the knot: 'a b c a b c' with alternating over/under is an overhand (trefoil), 'a b c c b a' is nested
    (a fold, the unknot). Returns code ('a+ b- c+ a- b+ c-'), letters ('abcabc'), over ('+-+-+-'), n_crossings,
    alternating (consecutive visits alternate over/under), interleaved (every two crossings are met a..b..a..b, as all
    three of a trefoil's are). A diagnostic of the XY diagram for planning and for checking a move; never a knot proof."""
    cr = crossings_fast(np.asarray(P, float), closed=False, min_dz=min_dz)
    visits = []
    for k, c in enumerate(cr):
        visits.append((c["s_over"], k, "+"))
        visits.append((c["s_under"], k, "-"))
    visits.sort()
    names, letters, over = {}, [], []
    for _, k, o in visits:
        if k not in names:
            names[k] = chr(ord("a") + len(names)) if len(names) < 26 else f"x{len(names)}"
        letters.append(names[k])
        over.append(o)
    alternating = all(over[i] != over[i + 1] for i in range(len(over) - 1))
    pos = {}
    for i, l in enumerate(letters):
        pos.setdefault(l, []).append(i)
    ch = list(pos.values())
    inter = all((a[0] < b[0] < a[1] < b[1]) or (b[0] < a[0] < b[1] < a[1]) for x, a in enumerate(ch) for b in ch[x + 1:])
    return {"code": " ".join(l + o for l, o in zip(letters, over)), "letters": "".join(letters), "over": "".join(over),
            "n_crossings": len(cr), "alternating": bool(alternating), "interleaved": bool(inter)}


# --------------------------------------------------------------- self-test
def _param_curve(kind: str, n: int = 400, hand: int = +1) -> np.ndarray:
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if kind == "trefoil":
        x = np.sin(u) + 2 * np.sin(2 * u)
        y = np.cos(u) - 2 * np.cos(2 * u)
        z = -np.sin(3 * u) * hand
    elif kind == "figure8":
        x = (2 + np.cos(2 * u)) * np.cos(3 * u)
        y = (2 + np.cos(2 * u)) * np.sin(3 * u)
        z = np.sin(4 * u)
    elif kind == "circle":
        x, y, z = np.cos(u), np.sin(u), 0 * u
    elif kind == "kink":               # unknot with one Reidemeister-I loop
        x = np.cos(u) + 0.9 * np.cos(2 * u)
        y = np.sin(u) + 0.9 * np.sin(2 * u)
        z = 0.3 * np.sin(u)
    else:
        raise ValueError(kind)
    return np.stack([x, y, z], 1) * 0.1


def selftest(verbose: bool = True) -> bool:
    ok = True
    want = {"trefoil": "trefoil", "figure8": "figure-eight",
            "circle": "unknot", "kink": "unknot"}
    for kind, exp in want.items():
        for hand in (+1, -1):
            Q = _param_curve(kind, hand=hand)
            a = alexander(Q)
            got = classify(a["coeffs"])
            ok &= (got == exp)
            if verbose:
                print(f"closed {kind:8s} hand {hand:+d}: {a['poly']:>14s} "
                      f"det {a['det']} -> {got} {'OK' if got == exp else 'FAIL'}")
    # OPEN overhand: a trefoil cut open, with the ends led outward
    Q = _param_curve("trefoil", n=400)
    # cut at a lobe apex (the outermost point): both ends then lead away
    # from the tangle, the way the working ends of a real overhand do
    k = int(np.argmax(np.linalg.norm(Q[:, :2] - Q[:, :2].mean(0), axis=1)))
    Q = np.roll(Q, -k, axis=0)
    open_curve = Q[4:-4]
    v = check(open_curve)
    ok &= v["trefoil"]
    if verbose:
        print(f"open overhand (cut trefoil): {v['type']} agree {v['agree']} "
              f"ends_free {v['ends_free']} -> {'OK' if v['trefoil'] else 'FAIL'}")
    # an open U with a crossing but no knot
    u = np.linspace(0, 1, 60)
    U = np.stack([np.cos(np.pi * u) * 0.3, np.sin(np.pi * u) * 0.3 * 1.3,
                  0 * u + 0.014], 1)
    v = check(U)
    ok &= (v["type"] == "unknot")
    if verbose:
        print(f"open U: {v['type']} -> {'OK' if v['type']=='unknot' else 'FAIL'}")
    return ok
