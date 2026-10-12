"""Group objects: one Stage 1 object made of several separate pieces (a tower of loose bars, a pile, fan or pyramid of
cards, a handful of blocks taken one at a time).

Stage 1 marks such an object with `pieces` = {"count": N, "at_least": bool, "kind": "<singular noun>", "loose": bool}.
Stage 2's scene_recon.json entry lists the pieces' root bodies under `parts` (each piece is its own body, a free body
when the pieces are loose; `root_body` is one of them). An entry without `parts` is a single-body object and every
check treats it exactly as before.

    from tool_library.common import piece_groups as PG
    PG.recon_parts(entry)            # ['tower_c0s0', 'tower_c0s1', ...] or []
    PG.stage1_pieces(stage1_object)  # {'count': 22, 'at_least': False, 'kind': 'bar', 'loose': True} or None
"""
from __future__ import annotations

import re

import numpy as np


def stage1_pieces(obj):
    """Normalized Stage 1 `pieces` block of an object, or None (a single-piece object).

    `count` may be an integer or a string such as '>=3' / '3+' / 'at least 3' (then at_least is True)."""
    if not isinstance(obj, dict) or not isinstance(obj.get("pieces"), dict):
        return None
    p = obj["pieces"]
    count, at_least = p.get("count"), bool(p.get("at_least", False))
    if isinstance(count, str):
        m = re.fullmatch(r"\s*(>=|at least|>|more than)?\s*(\d+)\s*(\+)?\s*", count.lower())
        if m:
            at_least = at_least or bool(m.group(1) or m.group(3))
            count = int(m.group(2)) + (1 if m.group(1) in (">", "more than") else 0)
        else:
            count = None
    if isinstance(count, bool) or not isinstance(count, int):
        count = None
    kind = str(p.get("kind") or "").strip().lower() or None
    return {"count": count, "at_least": at_least, "kind": kind, "loose": bool(p.get("loose", True))}


def recon_parts(entry):
    """The declared piece root bodies of a scene_recon.json object entry ([] for a single-body object)."""
    if not isinstance(entry, dict):
        return []
    parts = entry.get("parts")
    if not isinstance(parts, (list, tuple)):
        return []
    out = []
    for p in parts:
        if isinstance(p, str) and p and p not in out:
            out.append(p)
    return out


def is_group(entry):
    return len(recon_parts(entry)) >= 2


def group_bodies(entry):
    """Root body first, then every other declared part: the bodies a group object's silhouette and contacts use."""
    parts = recon_parts(entry)
    rb = entry.get("root_body") if isinstance(entry, dict) else None
    return ([rb] if rb else []) + [p for p in parts if p != rb]


def count_matches(n, pieces):
    """Whether n pieces satisfy a normalized Stage 1 pieces block (True when it states no count)."""
    if not pieces or pieces.get("count") is None:
        return True
    return n >= pieces["count"] if pieces.get("at_least") else n == pieces["count"]


def count_phrase(pieces):
    if not pieces or pieces.get("count") is None:
        return "an unstated number of"
    return f"{'at least ' if pieces.get('at_least') else ''}{pieces['count']}"


def subtree(model, bid):
    out, stack = set(), [int(bid)]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
    return out


def body_extent(model, data, bid):
    """Sorted extents (m) of the collidable (else any) geoms of a body subtree, boxed in the body's own frame (so a
    tilted piece measures like a flat one)."""
    sub = subtree(model, bid)
    gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub]
    col = [g for g in gs if model.geom_contype[g] or model.geom_conaffinity[g]]
    gs = col or gs
    if not gs:
        return None
    Rb, pb = data.xmat[bid].reshape(3, 3), data.xpos[bid]
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in gs:
        # the geom's bounding box (aabb: centre offset + half sizes in the geom frame) carried into the body frame
        aabb = np.asarray(model.geom_aabb[g], float)
        c, h = aabb[:3], aabb[3:]
        R = data.geom_xmat[g].reshape(3, 3)
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
        w = (corners @ R.T + data.geom_xpos[g] - pb) @ Rb
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return sorted((hi - lo).tolist())


def dissimilar_parts(model, data, part_ids, max_ratio=3.0):
    """Parts whose size differs from the median part's by more than max_ratio in volume or longest extent: a group of
    pieces is several of the same thing, not an object and its cap. [(body name, ratio)]."""
    ext = {b: body_extent(model, data, b) for b in part_ids}
    ext = {b: e for b, e in ext.items() if e}
    if len(ext) < 2:
        return []
    vol = {b: max(e[0], 1e-4) * max(e[1], 1e-4) * max(e[2], 1e-4) for b, e in ext.items()}
    longest = {b: e[2] for b, e in ext.items()}
    mv, ml = float(np.median(list(vol.values()))), float(np.median(list(longest.values())))
    out = []
    for b in ext:
        rv = max(vol[b] / mv, mv / vol[b])
        rl = max(longest[b] / ml, ml / longest[b]) if ml > 0 and longest[b] > 0 else 1.0
        if rv > max_ratio ** 2 or rl > max_ratio:
            out.append((model.body(b).name, round(max(rv ** 0.5, rl), 2)))
    return out
