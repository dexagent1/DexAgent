"""Over-moves on a rope lying on the table: what each move does to the rope and its crossings, and a search for the
moves that tie a knot (used by plan_knot_moves).

The rope is its centreline: the cable's vertices in order (construct_rope: B_first, B_1 .. B_last, then the tip site) in
the table plane, the rest length of every segment, and for every pair of segments that cross in the plan view the one
on top. A move picks one link (the capsule between two consecutive vertices), lifts it, carries it along a path (through
optional via points) to a landing point and sets it down before the fingers open; another hand may pin one link.

What a move does (follow-the-leader with friction): the carried link translates with the hand; every segment the move
stretches past its length is pulled back to it, slack rope exerts nothing and does not move, a pinned link does not
move (a move that would drag it is refused), and the rope bends round the hand with a radius (vertices two apart stay
a 90-degree chord apart). Pure follow-the-leader, where each vertex is moved toward the one ahead of it, drags the far
rope along the hand's path; here a stretched segment is shortened by moving BOTH its ends (a projection), which is the
least motion that keeps the rope inextensible, as rope with friction on a table answers. Measured on this bench on an
executed 40 cm carry of a 1.55 m, 3.2 cm cable end across its other leg: follow-the-leader 0.20 m mean from the
executed rope (it pulled the far leg 22 cm), the projection 28 mm with the crossing order right. (common/ropesim.py,
the cable simulated alone, was 10 mm: plan_knot_moves checks its candidates there.)

Over and under: the carried link is lifted `lift_m`; the rope next to it hangs from the hand at about 45 degrees, so a
vertex at arc distance s from the carried link is lifted max(0, lift_m - s) while it moves. Rope that moved lies ON TOP
of rope that did not where they now cross (laid over it, drawn over it behind the hand, or dragged into it: a heavy
cable dragged against another rides over it in simulation); a segment that only slid keeps its order at the crossings
it already had, even when it slid THROUGH a crossing (a tail pulled through under a strand stays under it); of two moved
segments the one nearer the hand is on top. A move that would lift a segment lying under a resting strand (the strand
would come along) or leave two capsules overlapping without crossing (one piled on the other) is refused.

The knot of a predicted rope is read from its diagram: knot.quick_type screens (the determinant of check()'s four
closures) and knot.check confirms on the polyline whose heights encode the decided over/under at every crossing, so the
planner and rope_knot_check judge the same way.
"""
from __future__ import annotations

import math

import numpy as np

from tool_library.common import knot as K

GOALS = {"trefoil": "trefoil", "overhand": "trefoil", "figure-eight": "figure-eight", "figure_eight": "figure-eight"}


# ----------------------------------------------------------------------------- geometry helpers
def _cross_pairs(xy):
    """Plan-view crossings of an open polyline: arrays I < J (segment indices, J >= I + 2), UI, UJ."""
    P = np.zeros((len(xy), 3))
    P[:, :2] = xy
    c = K.seg_crossings_np(P, closed=False)
    return c["i"], c["j"], c["ui"], c["uj"]


def _seg_dist(xy, I, J):
    """Closest distance between segments I and J (arrays) of the polyline xy."""
    p1, q1 = xy[I], xy[I + 1]
    p2, q2 = xy[J], xy[J + 1]
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r)
    b = np.einsum("ij,ij->i", d1, d2)
    den = a * e - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(den > 1e-12, np.clip((b * f - c * e) / den, 0.0, 1.0), 0.0)
        t = np.where(e > 1e-12, (b * s + f) / e, 0.0)
    t = np.clip(t, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(a > 1e-12, np.clip((b * t - c) / a, 0.0, 1.0), 0.0)
    return np.linalg.norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]), axis=1)


def _point_seg_dist(xy, p, segs):
    a, b = xy[segs], xy[segs + 1]
    d = b - a
    L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
    t = np.clip(np.einsum("ij,ij->i", p[None] - a, d) / L2, 0.0, 1.0)
    return np.linalg.norm(a + d * t[:, None] - p[None], axis=1)


# ----------------------------------------------------------------------------- the rope
class Rope:
    """A rope on the table: vertices xy (n, 2), segment rest lengths seg (n - 1), radius r, table top z0, and `top`:
    {(i, j): the segment on top} for every crossing pair of segments i < j."""

    def __init__(self, xy, seg, radius, top=None, z0=0.0):
        self.xy = np.asarray(xy, float).copy()
        self.seg = np.asarray(seg, float).copy()
        self.r = float(radius)
        self.z0 = float(z0)
        self.top = dict(top or {})

    @classmethod
    def from_points(cls, P, radius, z0=0.0):
        """From a physical centreline (n, 3): the plan-view crossings with the strand higher at each on top."""
        P = np.asarray(P, float)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        c = K.seg_crossings_np(P, closed=False)
        top = {(int(i), int(j)): (int(i) if zi >= zj else int(j)) for i, j, zi, zj in zip(c["i"], c["j"], c["zi"], c["zj"])}
        return cls(P[:, :2], seg, radius, top, z0)

    def copy(self):
        return Rope(self.xy, self.seg, self.r, self.top, self.z0)

    @property
    def n(self):
        return len(self.xy)

    def centre(self, k):
        return 0.5 * (self.xy[k] + self.xy[k + 1])

    def heading(self, k):
        d = self.xy[k + 1] - self.xy[k]
        return float(math.atan2(d[1], d[0]))

    def arc(self):
        return np.concatenate([[0.0], np.cumsum(self.seg)])

    def crossings(self):
        """[{i, j, ui, uj, xy, top}] for the current geometry (top from the record; a pair not recorded gets i)."""
        I, J, UI, UJ = _cross_pairs(self.xy)
        out = []
        for i, j, ui, uj in zip(I.tolist(), J.tolist(), UI.tolist(), UJ.tolist()):
            xy = self.xy[i] + ui * (self.xy[i + 1] - self.xy[i])
            out.append({"i": i, "j": j, "ui": ui, "uj": uj, "xy": xy, "top": self.top.get((i, j), i)})
        return out

    def under_segments(self):
        """Segments that lie under another strand somewhere."""
        return {(c["j"] if c["top"] == c["i"] else c["i"]) for c in self.crossings()}

    def polyline(self):
        """The centreline in 3-D with heights that encode the decided over/under at every crossing (the over strand a rope
        diameter above the under strand right at the crossing), for knot.check / rope_knot_check."""
        base = self.z0 + self.r
        hi = base + 2.0 * self.r
        per = {}
        for c in self.crossings():
            per.setdefault(c["i"], []).append((c["ui"], c["top"] == c["i"]))
            per.setdefault(c["j"], []).append((c["uj"], c["top"] == c["j"]))
        pts = []
        for k in range(self.n - 1):
            a, b = self.xy[k], self.xy[k + 1]
            pts.append([a[0], a[1], base])
            xs = sorted(per.get(k, []))
            if not xs:
                continue
            us = [u for u, _ in xs]
            bounds = [0.0] + us + [1.0]
            for q, (u, is_top) in enumerate(xs):
                room = min(u - bounds[q], bounds[q + 2] - u)
                L = max(float(self.seg[k]), 1e-6)
                eps = min(0.4 * room, 0.002 / L)
                z = hi if is_top else base
                for uu in (u - eps, u + eps):
                    p = a + uu * (b - a)
                    pts.append([p[0], p[1], z])
        e = self.xy[-1]
        pts.append([e[0], e[1], base])
        # knot.check rejects coincident consecutive points (a crossing right at a vertex puts u - eps on it)
        out = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(np.subtract(p, out[-1])) > 1e-7:
                out.append(p)
        return np.asarray(out, float)

    def order(self):
        return K.crossing_order(self.polyline())

    def knot_type(self, confirm=True):
        """'trefoil' / 'unknot' / ... of the diagram: the fast screen, confirmed by knot.check when it finds a knot."""
        P = self.polyline()
        q = K.quick_type(P)
        if confirm and q not in ("unknot",):
            v = K.check(P)
            return v["type"] if v["agree"] else "ambiguous"
        return q


# ----------------------------------------------------------------------------- one move
def _follow_ftl(X, seg, k, pts, pinned, step_m, pin_tol_m, rope):
    """Pure follow-the-leader: each vertex moves straight toward the one ahead of it to its segment length while taut."""
    n = len(X)
    r_bend = max(4.0 * rope.r, 0.015)
    th = 2.0 * math.asin(min(1.0, float(np.median(rope.seg)) / (2.0 * r_bend)))
    cmax, smax = math.cos(th), math.sin(th)
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        steps = max(1, int(math.ceil(L / step_m)))
        dx, dy = (b - a) / steps
        for _ in range(steps):
            X[k][0] += dx; X[k][1] += dy
            X[k + 1][0] += dx; X[k + 1][1] += dy
            for rng, ref in ((range(k + 2, n), -1), (range(k - 1, -1, -1), +1)):
                for i in rng:
                    j = i + ref
                    vx, vy = X[i][0] - X[j][0], X[i][1] - X[j][1]
                    Lv = math.hypot(vx, vy)
                    s0 = seg[min(i, j)]
                    if Lv <= s0 + 1e-9:
                        break
                    if i in pinned:
                        if Lv - s0 > pin_tol_m:
                            return f"the move drags the pinned link (vertex {i})"
                        break
                    ux, uy = vx / Lv, vy / Lv
                    jp = j + ref if 0 <= j + ref < n else None
                    if jp is not None:
                        wx, wy = X[j][0] - X[jp][0], X[j][1] - X[jp][1]
                        Lw = math.hypot(wx, wy)
                        if Lw > 1e-9:
                            wx, wy = wx / Lw, wy / Lw
                            if ux * wx + uy * wy < cmax:
                                sgn = 1.0 if (wx * uy - wy * ux) >= 0 else -1.0
                                ux = wx * cmax - sgn * wy * smax
                                uy = wy * cmax + sgn * wx * smax
                    X[i][0] = X[j][0] + ux * s0; X[i][1] = X[j][1] + uy * s0
    return None


def _follow_projected(X, seg, k, pts, pinned, step_m, passes, pin_tol_m):
    """Follow-the-leader resolved as a projection: after each small step of the carried link, every segment stretched past
    its length is shortened back by moving BOTH its vertices, equally (the carried link and pinned links do not move), a
    few Gauss-Seidel sweeps outward from the hand. A slack segment exerts nothing. The result is the least total motion
    that keeps the rope inextensible, which is how rope lying on a table with friction answers a sideways carry: the
    nearby rope swings round, the far rope stays."""
    n = len(X)
    fixed = set(pinned) | {k, k + 1}
    order = list(range(k + 1, n - 1)) + list(range(k - 1, -1, -1))
    # the tightest bend: 90 degrees over two joints, i.e. vertices two apart at least the chord 2 s cos 45 apart
    # (never more than the rope already has: a bend the rope lies in now is not straightened by a move elsewhere)
    dmin = [min(math.sqrt(2.0) * 0.5 * (seg[i] + seg[min(i + 1, len(seg) - 1)]),
                math.hypot(X[min(i + 2, n - 1)][0] - X[i][0], X[min(i + 2, n - 1)][1] - X[i][1])) for i in range(len(seg))]
    legs = []
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        steps = max(1, int(math.ceil(L / step_m)))
        legs += [((b - a) / steps, passes)] * steps
    legs.append((np.zeros(2), 200))                  # the rope comes to rest once the hand has stopped
    for (dx, dy), npass in legs:
        if True:
            X[k][0] += dx; X[k][1] += dy
            X[k + 1][0] += dx; X[k + 1][1] += dy
            for _p in range(npass):
                worst = 0.0
                for i in order:
                    j = i + 1
                    vx, vy = X[j][0] - X[i][0], X[j][1] - X[i][1]
                    Lv = math.hypot(vx, vy)
                    ex = Lv - seg[i]
                    if ex <= 1e-7:
                        continue
                    fi, fj = i in fixed, j in fixed
                    if fi and fj:
                        if ex > pin_tol_m:
                            return f"the move drags a pinned link (segment {i})"
                        continue
                    f = ex / Lv
                    if fi:
                        X[j][0] -= vx * f; X[j][1] -= vy * f
                    elif fj:
                        X[i][0] += vx * f; X[i][1] += vy * f
                    else:
                        X[i][0] += 0.5 * vx * f; X[i][1] += 0.5 * vy * f
                        X[j][0] -= 0.5 * vx * f; X[j][1] -= 0.5 * vy * f
                    worst = max(worst, ex)
                # bending: two links apart stay at least a chord of the tightest bend apart (a thick rope does not fold
                # back on itself next to the hand)
                for i in order:
                    j = i + 2
                    if j >= n:
                        continue
                    vx, vy = X[j][0] - X[i][0], X[j][1] - X[i][1]
                    Lv = math.hypot(vx, vy)
                    lack = dmin[i] - Lv
                    if lack <= 1e-6 or Lv < 1e-9:
                        continue
                    fi, fj = i in fixed, j in fixed
                    if fi and fj:
                        continue
                    f = lack / Lv
                    if fi:
                        X[j][0] += vx * f; X[j][1] += vy * f
                    elif fj:
                        X[i][0] -= vx * f; X[i][1] -= vy * f
                    else:
                        X[i][0] -= 0.5 * vx * f; X[i][1] -= 0.5 * vy * f
                        X[j][0] += 0.5 * vx * f; X[j][1] += 0.5 * vy * f
                    worst = max(worst, lack)
                if worst < 2e-4:
                    break
    # the pinned links must not have been pulled off their places by the end
    for i in range(n - 1):
        if (i in pinned and i + 1 in pinned) and math.hypot(X[i + 1][0] - X[i][0], X[i + 1][1] - X[i][1]) - seg[i] > pin_tol_m:
            return f"the move drags a pinned link (segment {i})"
    if pinned:
        for i in order:
            if math.hypot(X[i + 1][0] - X[i][0], X[i + 1][1] - X[i][1]) - seg[i] > max(pin_tol_m, 0.003):
                return "the move drags a pinned link: the rope between it and the hand is too short"
    return None


MIN_LOOP_M = 0.18            # the smallest loop a predicted move may close (arc length between a crossing's two visits)
MIN_LOOP_RADII = 10.0        # ... and at least this many rope radii


def lay(rope, k, path, lift_m, pins=(), step_m=0.01, pin_tol_m=0.002, overlap_frac=0.75, model="projected", passes=4, min_loop_m=None):
    """Carry segment k (both its vertices, translated) along `path` (xy points its centre visits, the last is where it
    lands) with the hand `lift_m` above the table, other vertices following the leader; `pins` are segment indices whose
    two vertices may not move. Returns (Rope, info) or (None, reason)."""
    X = [list(map(float, p)) for p in rope.xy]
    n = len(X)
    seg = [float(s) for s in rope.seg]
    pinned = set()
    for p in pins or ():
        pinned |= {int(p), int(p) + 1}
    if k in pins or (k in pinned) or (k + 1 in pinned):
        return None, "the carried link is pinned"
    c0 = rope.centre(k)
    pts = [np.asarray(c0, float)] + [np.asarray(p, float)[:2] for p in path]
    start = [tuple(p) for p in X]
    if model == "ftl":
        err = _follow_ftl(X, seg, k, pts, pinned, step_m, pin_tol_m, rope)
    else:
        err = _follow_projected(X, seg, k, pts, pinned, step_m, passes, pin_tol_m)
    if err:
        return None, err
    Xa = np.asarray(X, float)
    X0 = np.asarray(start, float)
    disp = np.linalg.norm(Xa - X0, axis=1)
    arc = rope.arc()
    a_lead = np.minimum(np.abs(arc - arc[k]), np.abs(arc - arc[k + 1]))
    lift = np.where(disp > 0.002, np.maximum(0.0, float(lift_m) - a_lead), 0.0)
    lift[k] = lift[k + 1] = float(lift_m)
    moved_seg = np.maximum(disp[:-1], disp[1:]) > 0.002
    lift_seg = np.maximum(lift[:-1], lift[1:])
    lead_arc = np.minimum(a_lead[:-1], a_lead[1:])
    clear = 2.0 * rope.r
    # a lifted segment lying under a resting strand would lift that strand with it
    for c in rope.crossings():
        under = c["j"] if c["top"] == c["i"] else c["i"]
        over = c["top"]
        if lift_seg[under] >= clear and not moved_seg[over]:
            return None, f"segment {under} lies under segment {over} and would lift it along"
    # the strands the carried link passed over in the air: the rope that follows it along its path is drawn over them too
    passed = set()
    if float(lift_m) >= clear:
        lead = np.asarray(pts, float)
        for Xs in (X0, Xa):
            for q in range(len(lead) - 1):
                A = np.vstack([Xs, lead[q:q + 2]])
                c = K.seg_crossings_np(np.c_[A, np.zeros(len(A))], closed=False)
                last = len(Xs) - 1                   # the path segment's index in the stacked polyline
                for i, j in zip(c["i"].tolist(), c["j"].tolist()):
                    if j == last and i < len(Xs) - 1 and abs(i - k) > 1:
                        passed.add(int(i))
    new = Rope(Xa, rope.seg, rope.r, {}, rope.z0)
    I, J, UI, UJ = _cross_pairs(Xa)
    slid = np.ceil(np.maximum(disp[:-1], disp[1:]) / max(float(np.median(rope.seg)), 1e-6)).astype(int) + 1
    old_pairs = list(rope.top.items())

    def inherited(i, j):
        """A crossing a strand slid THROUGH: the rope moved along itself, so the crossing pair's segments changed but the
        strands did not; the nearest old crossing of the same two strands (within the distance each slid) gives the
        order."""
        best, bd = None, None
        for (a, b), t in old_pairs:
            for (x, y, tx) in ((a, b, t == a), (b, a, t == b)):
                if abs(x - i) <= slid[i] and abs(y - j) <= slid[j]:
                    dd = abs(x - i) + abs(y - j)
                    if bd is None or dd < bd:
                        best, bd = (i if tx else j), dd
        return best

    lead_path = np.asarray(pts, float)

    def on_path(i, ui):
        """The crossing lies on the hand's path: the follower was drawn over the other strand right where the carried link
        crossed it."""
        if float(lift_m) < clear:
            return False
        x = Xa[i] + ui * (Xa[i + 1] - Xa[i])
        if len(lead_path) < 2:
            return False
        return float(_point_seg_dist(lead_path, x, np.arange(len(lead_path) - 1)).min()) <= 3.0 * rope.r

    for i, j, ui in zip(I.tolist(), J.tolist(), UI.tolist()):
        prev = rope.top.get((i, j))
        if prev is None:
            prev = inherited(i, j)
        if on_path(i, ui) and (moved_seg[i] or moved_seg[j]):
            if moved_seg[i] and moved_seg[j]:
                new.top[(i, j)] = i if lead_arc[i] < lead_arc[j] else j
            else:
                new.top[(i, j)] = i if moved_seg[i] else j
            continue
        mi, mj = bool(moved_seg[i]), bool(moved_seg[j])
        li, lj = float(lift_seg[i]), float(lift_seg[j])
        if not mi and not mj:
            new.top[(i, j)] = prev if prev is not None else (i if li >= lj else j)
            continue
        if mi and mj:
            if max(li, lj) >= clear and abs(li - lj) > 1e-4:
                new.top[(i, j)] = i if li > lj else j            # the one nearer the hand is higher
            elif (j in passed) != (i in passed):
                new.top[(i, j)] = i if j in passed else j        # it followed the hand over the other
            elif prev is not None:
                new.top[(i, j)] = prev
            else:
                new.top[(i, j)] = i if lead_arc[i] < lead_arc[j] else j   # the rope nearer the hand rides over
            continue
        mv, st = (i, j) if mi else (j, i)
        lv = li if mi else lj
        if lv >= clear or st in passed or prev is None:
            # laid over it, drawn over it behind the hand, or dragged into it: the moving rope rides over a resting strand
            # (the cable simulation does so for a heavy rope dragged across another)
            new.top[(i, j)] = mv
        else:
            new.top[(i, j)] = prev
    # two capsules overlapping along their length without crossing: one piled on the other, which is not predicted (only
    # new overlaps of moved rope count: two resting strands that already touched keep doing so)
    nI, nJ = np.triu_indices(n - 1, k=2)
    near = set()
    for i, j in zip(I.tolist(), J.tolist()):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                near.add((min(i + di, j + dj), max(i + di, j + dj)))
    if len(nI):
        sel = moved_seg[nI] | moved_seg[nJ]
        nI, nJ = nI[sel], nJ[sel]
    if len(nI):
        lim = overlap_frac * 2.0 * rope.r
        dd = _seg_dist(Xa, nI, nJ)
        d0 = _seg_dist(X0, nI, nJ)
        bad = (dd < lim) & (d0 >= lim)
        for i, j, dv in zip(nI[bad].tolist(), nJ[bad].tolist(), dd[bad].tolist()):
            if (i, j) in near:
                continue
            return None, f"segments {i} and {j} would lie on each other ({1000 * dv:.0f} mm apart)"
    # a thick rope cannot close a loop tighter than a few diameters, nor stack crossings on one spot: a crossing whose two
    # visits are less than min_loop_m apart along the rope, or two crossings closer than 3 radii in the plan view, is a
    # pile the prediction cannot stand for (only those the move made: the rope's own are kept)
    new_cr = new.crossings()
    loop_min = min_loop_m if min_loop_m is not None else max(MIN_LOOP_M, MIN_LOOP_RADII * rope.r)
    arc_n = new.arc()
    old_pairs_set = set(rope.top)
    for c in new_cr:
        if (c["i"], c["j"]) in old_pairs_set or not (moved_seg[c["i"]] or moved_seg[c["j"]]):
            continue
        a_i = arc_n[c["i"]] + c["ui"] * new.seg[c["i"]]
        a_j = arc_n[c["j"]] + c["uj"] * new.seg[c["j"]]
        if abs(a_j - a_i) < loop_min:
            return None, f"segments {c['i']} and {c['j']} would cross in a loop {100 * abs(a_j - a_i):.0f} cm round (tighter than {100 * loop_min:.0f} cm)"
    for x in range(len(new_cr)):
        for y in range(x + 1, len(new_cr)):
            if np.linalg.norm(new_cr[x]["xy"] - new_cr[y]["xy"]) < 3.0 * rope.r:
                return None, "two crossings would lie on one spot (a pile)"
    info = {"moved_vertices": int((disp > 0.002).sum()), "max_disp_m": float(disp.max()), "path_m": float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))}
    return new, info


def clearance(rope, k, skip=2):
    """Distance from segment k's centre to the nearest other strand (segments more than `skip` links away)."""
    segs = np.array([s for s in range(rope.n - 1) if abs(s - k) > skip], int)
    if not len(segs):
        return 1.0
    return float(_point_seg_dist(rope.xy, rope.centre(k), segs).min())


def is_on_top(rope, k, span=1):
    """Segment k and its neighbours within `span` lie under no strand."""
    under = rope.under_segments()
    return not any(s in under for s in range(k - span, k + span + 1))


PICK_BEND_DEG = 50.0


def pickable_geometry(rope, k, clear_m=0.05, span=2):
    """Where a pinch can close on segment k: it and its next neighbours lie flat on the table (no crossing, over or under),
    no link within `span` rides over another strand (raised and sloping), the rope there is nearly straight (every joint
    bent less than PICK_BEND_DEG at either end of the link, so a link of a loop can be taken), and no other strand comes
    within clear_m of its centre. Measured on the bench: a pinch
    on a link sloping off a crossing, two links from where the rope rode over the other strand, ran the thumb into that
    raised link on the way down and closed on nothing."""
    n = rope.n - 1
    near1 = set(range(max(0, k - 1), min(n, k + 2)))
    near = set(range(max(0, k - span), min(n, k + span + 1)))
    for c in rope.crossings():
        if c["i"] in near1 or c["j"] in near1 or c["top"] in near:
            return False
    for v in (k, k + 1):                         # the joints at the two ends of the picked link
        if not 1 <= v <= rope.n - 2:
            continue
        a, b = rope.xy[v] - rope.xy[v - 1], rope.xy[v + 1] - rope.xy[v]
        cosang = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        if cosang < math.cos(math.radians(PICK_BEND_DEG)):
            return False
    return clearance(rope, k, skip=span + 1) >= clear_m


# ----------------------------------------------------------------------------- scoring
def progress(order):
    """How far a crossing order is toward a trefoil: the size of the largest set of pairwise interleaved crossings (all
    three of a trefoil's are), at most 3, and the number of the others. Three interleaved crossings count as three only
    when they alternate over and under along the rope (a trefoil's do); otherwise as two: a crossing laid the wrong way
    round cannot be turned over by laying rope (the strand under would have to be lifted with the one on it)."""
    letters = order["letters"]
    pos = {}
    for i, l in enumerate(letters):
        pos.setdefault(l, []).append(i)
    ch = [v for v in pos.values() if len(v) == 2]
    best = min(1, len(ch))
    m = len(ch)

    def inter(a, b):
        return (a[0] < b[0] < a[1] < b[1]) or (b[0] < a[0] < b[1] < a[1])
    over = order.get("over", "")
    for x in range(m):
        for y in range(x + 1, m):
            if inter(ch[x], ch[y]):
                best = max(best, 2)
                for z in range(y + 1, m):
                    if inter(ch[x], ch[z]) and inter(ch[y], ch[z]):
                        idx = sorted(ch[x] + ch[y] + ch[z])
                        signs = [over[i] for i in idx] if len(over) > max(idx) else []
                        if signs and all(signs[i] != signs[i + 1] for i in range(len(signs) - 1)):
                            best = 3
    return best, max(0, m - best)


def tail_room(rope, cap_m=0.25):
    """Free rope at the ends: the arc length from each end to its first crossing (the whole rope when there is none),
    each capped at cap_m, summed. A tail that must still pass over and under strands needs length; a move that pulls a
    tail longer through a crossing is progress the crossing order does not show."""
    arc = rope.arc()
    cr = rope.crossings()
    if not cr:
        return 2.0 * cap_m
    ss = [c["i"] + c["ui"] for c in cr] + [c["j"] + c["uj"] for c in cr]
    lo, hi = min(ss), max(ss)
    a_lo = float(np.interp(lo, np.arange(len(arc)), arc))
    a_hi = float(arc[-1] - np.interp(hi, np.arange(len(arc)), arc))
    return min(a_lo, cap_m) + min(a_hi, cap_m)


def margin(rope, move, goal, lift_m, offsets=(0.02,), dirs=8):
    """How many landing points around the planned one (offset by each of `offsets` in `dirs` directions) still give the
    goal: the move's tolerance to the predictor's error and the executor's placement error."""
    k, path = move["k"], move["path"]
    hits, tot = 0, 0
    for r in offsets:
        for a in np.linspace(0, 2 * np.pi, dirs, endpoint=False):
            p2 = [np.asarray(p, float) for p in path]
            p2[-1] = p2[-1] + r * np.array([np.cos(a), np.sin(a)])
            nr, _ = lay(rope, k, p2, lift_m, pins=move.get("pins") or ())
            tot += 1
            if nr is not None and nr.knot_type(confirm=False) == goal:
                hits += 1
    return hits / max(tot, 1)


def targets_around(c, radii, n_ang, bounds=None):
    out = []
    for r in radii:
        for a in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
            p = np.asarray(c, float) + r * np.array([np.cos(a), np.sin(a)])
            if bounds is not None and not (bounds[0] <= p[0] <= bounds[1] and bounds[2] <= p[1] <= bounds[3]):
                continue
            out.append(p)
    return out


def targets_over_strands(rope, k, offsets=(0.06, 0.11), max_dist=0.6, bounds=None, min_gap=6):
    """Landing points just past each other strand, seen from link k: for every segment at least `min_gap` links away
    (a loop of that size), its nearest point to the link, pushed `offsets` beyond it along the line from the link. A
    lay to one of them crosses that strand however far it lies (the rings of targets_around stop at their largest
    radius)."""
    c = rope.centre(k)
    segs = np.array([s for s in range(rope.n - 1) if abs(s - k) >= min_gap], int)
    if not len(segs):
        return []
    a, b = rope.xy[segs], rope.xy[segs + 1]
    d = b - a
    t = np.clip(np.einsum("ij,ij->i", c[None] - a, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12), 0.0, 1.0)
    q = a + d * t[:, None]
    out = []
    for qq in q:
        v = qq - c
        L = float(np.linalg.norm(v))
        if L < 1e-6 or L > max_dist:
            continue
        for off in offsets:
            p = qq + v / L * float(off)
            if bounds is not None and not (bounds[0] <= p[0] <= bounds[1] and bounds[2] <= p[1] <= bounds[3]):
                continue
            if all(np.linalg.norm(p - o) > 0.03 for o in out):
                out.append(p)
    return out


def search(rope, pickable, landable, goal="trefoil", max_moves=3, beam=8, lift_m=0.05, radii=(0.07, 0.12, 0.18, 0.25),
           n_ang=12, vias=True, pin_options=None, bounds=None, pick_clear_m=0.05, land_clear_m=0.045, prefer=None, max_results=5,
           max_crossings=6, time_budget_s=None, sides=("left", "right"), verbose=False):
    """Beam search over sequences of moves for `goal`. pickable(side, rope, k) -> list of pick records (each a dict with at
    least 'flip'; the caller's reach screen at the pick) or []; landable(side, rope, k, pick, path) -> bool (the reach
    screen along the carry and at the landing). pin_options(side, rope, k) -> [segment] pins the OTHER hand can hold.
    prefer: per move index, {side, k, target} the human did (a cost for leaving it). Returns plans best first: {moves,
    rope (predicted), order, cost, margin}."""
    import time
    goal = GOALS.get(goal, goal)
    t_start = time.time()
    sides = [s for s in ("left", "right") if s in set(sides)]
    root = {"rope": rope, "moves": [], "cost": 0.0}
    frontier = [root]
    found = []
    seen = set()
    stats = {"lays": 0, "refused": 0, "expanded": 0}
    best_partial = None
    for depth in range(int(max_moves)):
        children = []
        for node in frontier:
            rp = node["rope"]
            stats["expanded"] += 1
            pref = (prefer or [None] * 99)[len(node["moves"])] if prefer and len(node["moves"]) < len(prefer) else None
            for side in sides:
                for k in range(rp.n - 1):
                    if not pickable_geometry(rp, k, pick_clear_m):
                        stats["pick_geometry"] = stats.get("pick_geometry", 0) + 1
                        continue
                    picks = pickable(side, rp, k)
                    if not picks:
                        stats["pick_unreachable"] = stats.get("pick_unreachable", 0) + 1
                        continue
                    c = rp.centre(k)
                    paths = [[t] for t in targets_around(c, radii, n_ang, bounds)]
                    paths += [[t] for t in targets_over_strands(rp, k, bounds=bounds)]
                    if vias and min(k, rp.n - 2 - k) <= 2:
                        # a curved carry of an end: a straight rope's first loop, a tail swept round a crossing into a loop
                        # (one via to the side of the straight line)
                        for t in targets_around(c, radii[1:], max(4, n_ang // 2), bounds):
                            mid = 0.5 * (c + t)
                            nrm = np.array([-(t - c)[1], (t - c)[0]]) / max(np.linalg.norm(t - c), 1e-9)
                            for off in (0.6, -0.6):
                                v = mid + nrm * off * np.linalg.norm(t - c)
                                if bounds is None or (bounds[0] <= v[0] <= bounds[1] and bounds[2] <= v[1] <= bounds[3]):
                                    paths.append([v, t])
                    pin_sets = [()]
                    if pin_options is not None:
                        pin_sets += [(p,) for p in pin_options(side, rp, k)]
                    for path in paths:
                        for pins in pin_sets:
                            stats["lays"] += 1
                            nr, info = lay(rp, k, path, lift_m, pins=pins)
                            if nr is None:
                                stats["refused"] += 1
                                continue
                            if clearance(nr, k, skip=3) < land_clear_m:
                                stats["land_crowded"] = stats.get("land_crowded", 0) + 1
                                continue
                            od = nr.order()
                            if od["n_crossings"] > max_crossings:
                                continue
                            pick = next((p for p in picks if landable(side, rp, k, p, path)), None)
                            if pick is None:
                                stats["land_unreachable"] = stats.get("land_unreachable", 0) + 1
                                continue
                            mv = {"side": side, "k": int(k), "path": [np.asarray(p, float) for p in path], "pins": list(pins),
                                  "pick": pick, "order": od, "path_m": info["path_m"]}
                            cost = node["cost"] + 1.0 + 0.5 * info["path_m"] + (0.6 if pins else 0.0) + (0.4 * (len(path) - 1))
                            if pref:
                                if pref.get("side") and pref["side"] != side:
                                    cost += 0.5
                                if pref.get("k") is not None:
                                    cost += 0.02 * abs(int(pref["k"]) - int(k))
                                if pref.get("target") is not None:
                                    cost += 2.0 * float(np.linalg.norm(np.asarray(pref["target"], float)[:2] - np.asarray(path[-1])))
                            if node["moves"] and node["moves"][-1]["side"] != side:
                                cost += 0.1                      # a hand change costs a little (the other arm moves in)
                            child = {"rope": nr, "moves": node["moves"] + [mv], "cost": cost}
                            if od["n_crossings"] >= 3 and od["interleaved"] and nr.knot_type(confirm=False) == goal:
                                stats["screened_knots"] = stats.get("screened_knots", 0) + 1
                                kt = nr.knot_type(confirm=True)
                                if kt == goal:
                                    found.append(child)
                                else:
                                    stats.setdefault("unconfirmed", {}).setdefault(kt, 0)
                                    stats["unconfirmed"][kt] += 1
                                continue
                            key = (od["code"], tuple(np.round(nr.xy[::4] / 0.03).astype(int).ravel().tolist()))
                            if key in seen:
                                continue
                            seen.add(key)
                            prog, extra = progress(od)
                            child["score"] = 10.0 * prog - 4.0 * extra + 3.0 * tail_room(nr) - cost
                            children.append(child)
                if time_budget_s and time.time() - t_start > float(time_budget_s):
                    break
        if found and len(found) >= max_results:
            break
        children.sort(key=lambda c: c["score"], reverse=True)
        if children and (best_partial is None or children[0]["score"] > best_partial["score"]):
            best_partial = children[0]
        # a varied beam: at most a third of it (at least two) per crossing order, so near copies of one loop do not crowd out
        # the states one move further along another way
        per_code, frontier = {}, []
        cap = max(2, int(beam) // 3)
        for c in children:
            code = c["moves"][-1]["order"]["code"]
            if per_code.get(code, 0) >= cap:
                continue
            per_code[code] = per_code.get(code, 0) + 1
            frontier.append(c)
            if len(frontier) >= int(beam):
                break
        if verbose:
            print(f"depth {depth + 1}: {len(children)} children, {len(found)} found; frontier:",
                  [(round(c["score"], 2), c["moves"][-1]["order"]["code"], c["moves"][-1]["k"]) for c in frontier[:6]], flush=True)
        if not frontier:
            break
        if time_budget_s and time.time() - t_start > float(time_budget_s):
            stats["timed_out"] = True
            break
    found.sort(key=lambda c: c["cost"])
    out = []
    for f in found[:max(int(max_results) * 3, 6)]:
        m = margin(_replay_before(rope, f["moves"], lift_m), f["moves"][-1], goal, lift_m)
        out.append({"moves": f["moves"], "rope": f["rope"], "order": f["rope"].order(), "cost": round(f["cost"], 3), "margin": round(m, 2)})
    out.sort(key=lambda p: (p["cost"] - 1.5 * p["margin"]))
    if best_partial is not None:
        # the most progress any searched sequence made (largest interleaved, alternating crossing set, tail room, cost):
        # what to execute and re-plan from when no sequence within max_moves ties the knot
        stats["best_partial"] = {"moves": best_partial["moves"], "rope": best_partial["rope"], "order": best_partial["rope"].order(),
                                 "progress": progress(best_partial["rope"].order())[0], "score": round(best_partial["score"], 2)}
    return out[:int(max_results)], dict(stats, seconds=round(time.time() - t_start, 1))


def overhand_search(rope, pickable, landable, lift_m=0.05, bounds=None, sides=("left", "right"), end_links=(1, 7),
                    pick_clear_m=0.05, land_clear_m=0.035, keep=(40, 12), radii=(0.12, 0.18, 0.25, 0.32), n_ang=16,
                    max_results=4, time_budget_s=None, verbose=False):
    """The overhand as three over-moves, searched directly (the beam search's scoring keeps too few one-loop states to
    reach it): (1) a link near one end is laid across the rope's other strand: the loop, that end's side on top (a);
    (2) a link near the OTHER end is laid over the loop: its tail crosses the loop's first side on top (b, the tail
    over); (3) any link is laid over that tail: a later piece of the loop crosses it on top (c, the tail under). A rope
    that already lies in one loop (an executed first move) is continued from step 2. The
    crossing order after the three is 'a+ b- c+ a- b+ c-' (or its mirror): alternating, interleaved, a trefoil. Every
    move is predicted by lay(), its pinch and carry screened by the callbacks (as in search()), and kept only when its
    crossing order is the one the next move needs; at most keep[0] loops and keep[1] two-crossing states per loop are
    expanded. Returns (plans, stats) like search()."""
    import time
    t0 = time.time()
    sides = [s for s in ("left", "right") if s in set(sides)]
    stats = {"lays": 0, "refused": 0, "loops": 0, "tails": 0}
    found = []

    def moves_from(rp, ks, targets_of, need):
        out = []
        for side in sides:
            for k in ks:
                if not (0 <= k < rp.n - 1) or not pickable_geometry(rp, k, pick_clear_m):
                    continue
                picks = pickable(side, rp, k)
                if not picks:
                    continue
                for t in targets_of(rp, k):
                    stats["lays"] += 1
                    nr, info = lay(rp, k, [t], lift_m)
                    if nr is None:
                        stats["refused"] += 1
                        continue
                    if clearance(nr, k, skip=3) < land_clear_m:
                        continue
                    od = nr.order()
                    if not need(nr, od):
                        continue
                    pick = next((p for p in picks if landable(side, rp, k, p, [t])), None)
                    if pick is None:
                        continue
                    out.append(({"side": side, "k": int(k), "path": [np.asarray(t, float)], "pins": [], "pick": pick, "order": od,
                                 "path_m": info["path_m"]}, nr))
        return out

    def tg(rp, k):
        return targets_over_strands(rp, k, bounds=bounds) + targets_around(rp.centre(k), radii, n_ang, bounds)

    n = rope.n - 1
    od0 = rope.order()
    if od0["n_crossings"] == 1:
        # the loop is already there (an executed first move): its end on top is the one that was laid
        orders, given_loop = ((0,) if od0["over"] == "+-" else (1,)), True
    elif od0["n_crossings"] == 0:
        orders, given_loop = (0, 1), False
    else:
        return [], dict(stats, why=f"overhand_search starts from a straight rope or one loop, not '{od0['code']}'")
    for first in orders:
        e1 = range(end_links[0], end_links[1]) if first == 0 else range(n - end_links[1], n - end_links[0])
        e2 = range(n - end_links[1] - 2, n) if first == 0 else range(0, end_links[1] + 2)
        loops = [(None, rope)] if given_loop else moves_from(rope, list(e1), tg, lambda r, od: od["n_crossings"] == 1)

        def other_tail(r):
            # the free rope between the loop's crossing and the other end: what the tail has to cross the loop with
            c = r.crossings()[0]
            arc = r.arc()
            s_far = max(c["i"] + c["ui"], c["j"] + c["uj"]) if first == 0 else min(c["i"] + c["ui"], c["j"] + c["uj"])
            a = float(np.interp(s_far, np.arange(len(arc)), arc))
            return arc[-1] - a if first == 0 else a
        loops.sort(key=lambda x: (-round(other_tail(x[1]), 2), x[0]["path_m"] if x[0] else 0.0))
        stats["loops"] += len(loops)
        for mv1, r1 in loops[:keep[0]]:
            tails = moves_from(r1, list(e2), tg, lambda r, od: od["n_crossings"] == 2 and od["letters"] == "abab")
            tails.sort(key=lambda x: x[0]["path_m"])
            stats["tails"] += len(tails)
            for mv2, r2 in tails[:keep[1]]:
                def is_knot(r, od):
                    return od["n_crossings"] >= 3 and od["interleaved"] and r.knot_type(confirm=False) == "trefoil"
                for mv3, r3 in moves_from(r2, list(range(n)), tg, is_knot):
                    if r3.knot_type(confirm=True) != "trefoil":
                        continue
                    mvs = [m_ for m_ in (mv1, mv2, mv3) if m_ is not None]
                    cost = len(mvs) + 0.5 * sum(m_["path_m"] for m_ in mvs) + (0.1 if len({m_["side"] for m_ in mvs}) > 1 else 0.0)
                    found.append({"moves": mvs, "rope": r3, "order": r3.order(), "cost": round(cost, 3),
                                  "margin": round(clearance(r3, mv3["k"], skip=3), 3)})
                if len(found) >= max_results or (time_budget_s and time.time() - t0 > float(time_budget_s)):
                    break
            if len(found) >= max_results or (time_budget_s and time.time() - t0 > float(time_budget_s)):
                break
        if verbose:
            print(f"overhand_search from the {'first' if first == 0 else 'last'} end: {stats}, {len(found)} found", flush=True)
        if len(found) >= max_results or (time_budget_s and time.time() - t0 > float(time_budget_s)):
            break
    found.sort(key=lambda p: p["cost"])
    return found[:int(max_results)], dict(stats, seconds=round(time.time() - t0, 1))


def _replay_before(rope, moves, lift_m):
    """The predicted rope before the last move of a sequence."""
    rp = rope
    for mv in moves[:-1]:
        rp, _ = lay(rp, mv["k"], mv["path"], lift_m, pins=mv.get("pins") or ())
    return rp


def replay(rope, moves, lift_m):
    """The predicted rope after each move: [Rope]."""
    out, rp = [], rope
    for mv in moves:
        rp, info = lay(rp, mv["k"], mv["path"], lift_m, pins=mv.get("pins") or ())
        if rp is None:
            raise ValueError(f"replay: move {mv} refused: {info}")
        out.append(rp)
    return out
