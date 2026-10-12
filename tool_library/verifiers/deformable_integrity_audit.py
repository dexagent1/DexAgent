"""Deformable integrity audit: ropes and cables stay physical over the whole episode.

    from tool_library.verifiers.deformable_integrity_audit import deformable_integrity_audit
    deformable_integrity_audit("scene.xml", "trajectory.npz")

A cable in MuJoCo is a chain of capsules (or boxes, for a flat ribbon): nested bodies joined by ball
joints (a cable composite) or free capsules held together by connect / weld constraints. Pulled hard, a constraint-linked
chain stretches; driven fast, two of its segments pass through each other (a knot that ties or
unties itself through the rope), a segment sinks into the table or a finger cuts through it. The
rigid-object audits do not look at a chain's segments.
"""
from __future__ import annotations

TOOL = {
    "name": "deformable_integrity_audit",
    "title": "Deformable integrity audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Cables, ropes and ribbons (capsule or box link chains) keep their length, never pass through themselves or "
               "another strand of the same cable (at a frame or between frames), and do not sink into the table or the hand.",
    "description": (
        "No other audit models a chain. Chains are found in the scene: runs of at least min_segments nested bodies "
        "each carrying one capsule (or cylinder), or one box (the flat ribbon links construct_multi_strand_cable builds), "
        "joined by ball joints (or two or three hinges), as a MuJoCo cable composite builds them, and runs of free capsule "
        "bodies linked by connect / weld equalities; a chain starts at a free end or at a branch, so each strand hanging off "
        "a multi-strand cable's junction is a chain of its own. A box link's axis is the one that runs toward its "
        "neighbour (a ribbon link may be wider than long) and its radius across the chain is its half-thickness. Per frame: (1) "
        "stretch: for constraint-linked segments, the gap between the linked anchors relative to the segment length "
        "may not exceed stretch_frac (default 0.1); (2) self-interpenetration, for segments at least two apart in a chain "
        "and for every pair of segments of two strands of one cable: the distance between two capsule axes may not fall "
        "below the sum of their radii by more than pen_m (default 2 mm, a rope pressed on itself in a knot compresses "
        "about 1 mm); two boxes may not overlap (separating-axis depth) by more than pen_m or half the thinner link's "
        "thickness, whichever is smaller (a ribbon lying through another overlaps by its whole thickness, two strips pressed "
        "together by a fingertip by 0.2 mm); (3) table: no segment's lowest point (a box's lowest corner) more than pen_m "
        "under the tabletop over the table; (4) hand: no hand collision geom more than hand_pen_m (default 4 mm) inside a "
        "segment (the scene as the bench executor runs it). Between frames: (5) self-crossing, on the same pairs as (2): for "
        "two segments that are close on consecutive frames, the side of each axis relative to the other flips (the signed "
        "volume of their four end points changes sign) while their closest points lie inside both segments at a distance "
        "under the sum of the radii: one passed through the other. A scene without a chain passes with a note. Reports the "
        "worst event per rule with frames, phase labels and segments (a pair of strands is named 'A x B')."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "min_segments", "type": "int", "description": "Shortest run of capsule or box links that counts as a chain. Default 4."},
        {"name": "stretch_frac", "type": "float", "description": "Allowed anchor gap of a constraint link, as a share of the segment length. Default 0.1."},
        {"name": "pen_m", "type": "float", "description": "Allowed self- and table interpenetration. Default 0.002."},
        {"name": "hand_pen_m", "type": "float", "description": "Allowed hand-segment interpenetration. Default 0.004."},
        {"name": "stride", "type": "int", "description": "Check every n-th state for rules 1-4 (rule 5 uses every pair). Default 1. Whole-episode audits use every recorded physics substep when available; explicit frames select video frames."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {rule, chain, frames, label, segments, value}; chains {name: kind, profile (capsule | box), segments, radius_m (a box's half-thickness), width_m (boxes), length_m, per-rule maxima}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "rope", "cable", "ribbon", "deformable", "knot", "tunnelling"],
    "status": "tested",
    "entrypoint": "deformable_integrity_audit",
}

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA

CAPS = (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
BOX = int(mujoco.mjtGeom.mjGEOM_BOX)


def _segment_geom(m, b):
    """The one capsule (or cylinder) of a link body, else its one box (a flat ribbon or strap link)."""
    gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b]
    caps = [g for g in gs if int(m.geom_type[g]) in CAPS]
    if len(caps) == 1:
        return caps[0]
    boxes = [g for g in gs if int(m.geom_type[g]) == BOX] if not caps else []
    return boxes[0] if len(boxes) == 1 else None


def _quat_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _box_axis(m, run, i, g):
    """Index of the box axis that runs along the chain at link i: the one best aligned with the line to the next link's
    origin (from the previous one for the last link). A ribbon link can be wider than it is long (a 18 mm strip cut in
    12 mm links), so the longest half-size is not the chain direction."""
    b = run[i]
    if i + 1 < len(run):
        v = np.asarray(m.body_pos[run[i + 1]], float)
    else:
        v = _quat_mat(m.body_quat[b]).T @ np.asarray(m.body_pos[b], float)
    v = _quat_mat(m.geom_quat[g]).T @ v
    if np.linalg.norm(v) < 1e-9:
        return int(np.argmax(m.geom_size[g][:3]))
    return int(np.argmax(np.abs(v)))


def _root_of(m, b):
    while int(m.body_parentid[b]) != 0:
        b = int(m.body_parentid[b])
    return b


def _find_chains(ep, min_segments):
    m = ep.model
    seg = {b: _segment_geom(m, b) for b in range(1, m.nbody) if b not in ep.robot_bodies}
    seg = {b: g for b, g in seg.items() if g is not None}
    linked = {}
    for b, g in seg.items():
        jt = [int(m.jnt_type[j]) for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]) + int(m.body_jntnum[b]))]
        p = int(m.body_parentid[b])
        if p in seg and (jt == [EA.BALL] or (len(jt) in (2, 3) and all(x == EA.HINGE for x in jt))):
            linked[b] = p
    children = {}
    for c, p in linked.items():
        children.setdefault(p, []).append(c)
    chains = []
    for b in seg:
        p = linked.get(b)
        if p is not None and len(children.get(p, [])) == 1:
            continue                                   # inside a run, not a chain start
        # a chain starts at a free end or at a branch: the strands of a multi-strand cable hang off one junction body,
        # each is a chain of its own
        run = [b]
        while len(children.get(run[-1], [])) == 1:
            run.append(children[run[-1]][0])
        if len(run) >= min_segments:
            chains.append({"kind": "kinematic", "bodies": run, "geoms": [seg[x] for x in run], "links": []})
    # free capsules linked by connect / weld equalities
    free = {b for b, g in seg.items() if any(int(m.jnt_type[j]) == EA.FREE for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]) + int(m.body_jntnum[b])))}
    eqlinks = {}
    for e in range(m.neq):
        if int(m.eq_type[e]) in (EA.TJ.WELD, EA.TJ.CONNECT):
            b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            if b1 in seg and b2 in seg:
                eqlinks.setdefault(b1, []).append((b2, e)); eqlinks.setdefault(b2, []).append((b1, e))
    used = set()
    for b in sorted(free):
        if b in used or len(eqlinks.get(b, [])) != 1:
            continue
        run, links, prev = [b], [], None
        while True:
            nxt = [x for x in eqlinks.get(run[-1], []) if x[0] != prev and x[0] not in run]
            if len(nxt) != 1:
                break
            prev = run[-1]
            run.append(nxt[0][0]); links.append(nxt[0][1])
        used |= set(run)
        if len(run) >= min_segments:
            chains.append({"kind": "constraint", "bodies": run, "geoms": [seg[x] for x in run], "links": links})
    for ch in chains:
        # per link: the axis along the chain (a capsule's own z; a box's axis toward its neighbour), its half-length, the
        # radius across it (a box's half-thickness: what a strand lying on another is separated by) and the box half-sizes
        ax, half, rad = [], [], []
        for i, g in enumerate(ch["geoms"]):
            sz = np.asarray(m.geom_size[g], float)
            if int(m.geom_type[g]) == BOX:
                k = _box_axis(m, ch["bodies"], i, g)
                ax.append(k); half.append(float(sz[k])); rad.append(float(min(sz[j] for j in range(3) if j != k)))
            else:
                ax.append(2); half.append(float(sz[1])); rad.append(float(sz[0]))
        ch["axis"], ch["half"], ch["r"] = ax, np.array(half), np.array(rad)
        ch["box"] = np.array([int(m.geom_type[g]) == BOX for g in ch["geoms"]])
        ch["root"] = _root_of(m, ch["bodies"][0])
    return chains


def _seg_ends(d, g, k, h):
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]
    return c - h * R[:, k], c + h * R[:, k]


def _box_overlap(C1, R1, h1, C2, R2, h2):
    """Penetration depth of oriented boxes (vectorized over rows: centres [K,3], rotations [K,3,3] with the box axes as
    columns, half-sizes [K,3]) by the separating-axis test over the 15 candidate axes: the smallest overlap of the two
    projections (positive: the boxes interpenetrate by that much; negative: they are apart)."""
    t = C2 - C1
    axes = [R1[:, :, i] for i in range(3)] + [R2[:, :, j] for j in range(3)]
    axes += [np.cross(R1[:, :, i], R2[:, :, j]) for i in range(3) for j in range(3)]
    best = np.full(len(C1), np.inf)
    for L in axes:
        n = np.linalg.norm(L, axis=1)
        ok = n > 1e-6
        Lu = L / np.where(ok, n, 1.0)[:, None]
        ra = sum(h1[:, i] * np.abs(np.einsum("ij,ij->i", R1[:, :, i], Lu)) for i in range(3))
        rb = sum(h2[:, j] * np.abs(np.einsum("ij,ij->i", R2[:, :, j], Lu)) for j in range(3))
        ov = ra + rb - np.abs(np.einsum("ij,ij->i", t, Lu))
        best = np.where(ok, np.minimum(best, ov), best)
    return best


def _closest(p1, q1, p2, q2):
    """Closest points of segments [p1,q1], [p2,q2] (vectorized over rows): (s, t, distance)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("ij,ij->i", d1, d1); e = np.einsum("ij,ij->i", d2, d2); f = np.einsum("ij,ij->i", d2, r)
    c = np.einsum("ij,ij->i", d1, r); b = np.einsum("ij,ij->i", d1, d2)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-12)
    t = np.clip(t, 0, 1)
    s = np.clip((b * t - c) / np.maximum(a, 1e-12), 0, 1)
    c1 = p1 + d1 * s[:, None]; c2 = p2 + d2 * t[:, None]
    return s, t, np.linalg.norm(c1 - c2, axis=1)


def deformable_integrity_audit(scene, trajectory, frames=None, min_segments=4, stretch_frac=0.1, pen_m=0.002, hand_pen_m=0.004,
                               stride=1):
    EA.check_positive(min_segments=min_segments, stretch_frac=stretch_frac, pen_m=pen_m, hand_pen_m=hand_pen_m, stride=stride)
    if frames is None:
        from tool_library.verifiers._recording import collision_recording
        trajectory = collision_recording(trajectory)
    ep = EA.Episode(scene, trajectory, frames)
    m = ep.model
    chains = _find_chains(ep, int(min_segments))
    if not chains:
        return EA.finish([], None, chains={}, note="no cable (capsule or box link chain) in the scene")
    d = mujoco.MjData(m)
    tb = ep.table
    events = {"stretch": [], "self": [], "table": [], "hand": [], "crossing": []}
    info = {}
    hand_bodies = set().union(*[h["bodies"] for h in ep.hands.values()]) if ep.hands else set()
    scan = EA.ContactScan(ep, margin=0.0) if hand_bodies else None
    for ch in chains:
        name = ep.body_name(ch["bodies"][0])
        r, L = ch["r"], 2 * ch["half"]
        info[name] = {"kind": ch["kind"], "profile": "box" if ch["box"].all() else ("capsule" if not ch["box"].any() else "mixed"),
                      "segments": len(ch["bodies"]), "radius_m": round(float(r.mean()), 4),
                      "length_m": round(float(L.sum() + (2 * r.mean() if ch["kind"] == "constraint" else 0)), 3),
                      "max_stretch_frac": 0.0, "max_self_pen_mm": 0.0, "max_table_pen_mm": 0.0, "max_hand_pen_mm": 0.0, "crossings": 0}
        if ch["box"].any():
            info[name]["width_m"] = round(float(np.mean([2 * max(m.geom_size[g][j] for j in range(3) if j != k)
                                                         for g, k in zip(ch["geoms"], ch["axis"])])), 4)
        ch["L"], ch["name"] = L, name
    # rules 2 and 5 run over each object's links together: within a chain on links at least two apart, and between the
    # chains of one object (the strands of a multi-strand cable hang off one junction) on every pair, since a strand
    # passing through another is the same failure as a rope passing through itself
    groups = {}
    for ci, ch in enumerate(chains):
        groups.setdefault(ch["root"], []).append(ci)
    G = []
    for root, cis in groups.items():
        segs = [(ci, i) for ci in cis for i in range(len(chains[ci]["geoms"]))]
        cidx = np.array([c for c, _ in segs]); sidx = np.array([i for _, i in segs])
        I, J = np.triu_indices(len(segs), 1)
        keep = (cidx[I] != cidx[J]) | (np.abs(sidx[I] - sidx[J]) >= 2)
        I, J = I[keep], J[keep]
        cat = lambda k: np.concatenate([np.asarray(chains[ci][k]) for ci in cis])
        G.append({"chains": cis, "geoms": [g for ci in cis for g in chains[ci]["geoms"]], "axis": [a_ for ci in cis for a_ in chains[ci]["axis"]],
                  "half": cat("half"), "r": cat("r"), "box": cat("box").astype(bool), "cidx": cidx, "sidx": sidx, "I": I, "J": J,
                  "bb": (cat("box").astype(bool)[I] & cat("box").astype(bool)[J])})
        # two flat links overlap at most by the thinner one's thickness (a 1.2 mm ribbon lying through another reads 1.2
        # mm), so a box pair fails beyond half the thinner thickness (two strips pressed together compress far less:
        # 0.23 mm measured under a fingertip press) when that is below pen_m
        rr = G[-1]["r"]
        G[-1]["thr"] = np.where(G[-1]["bb"], np.minimum(float(pen_m), np.minimum(rr[I], rr[J])), float(pen_m))
    prev_ends = {}

    def where(gr, k):
        """(chain name for an event on pair k, [segment, segment]) with each segment given in its own chain."""
        ca, cb = chains[gr["cidx"][gr["I"][k]]]["name"], chains[gr["cidx"][gr["J"][k]]]["name"]
        return (ca if ca == cb else f"{ca} x {cb}"), [int(gr["sidx"][gr["I"][k]]), int(gr["sidx"][gr["J"][k]])], (ca, cb)

    for t in ep.window:
        d.qpos[:] = ep.Q[t]
        active = ep.extras.get("eq_active")
        if active is not None:
            d.eq_active[:] = active[t]
        mujoco.mj_kinematics(m, d)
        judge = (t - ep.a) % int(stride) == 0
        for gi, gr in enumerate(G):
            E = np.array([np.r_[_seg_ends(d, g, k, h)] for g, k, h in zip(gr["geoms"], gr["axis"], gr["half"])])     # [n, 6]
            P0, P1 = E[:, :3], E[:, 3:]
            r = gr["r"]
            I, J = gr["I"], gr["J"]
            if judge:
                for ci in gr["chains"]:
                    ch = chains[ci]; name = ch["name"]
                    for k, e in enumerate(ch["links"]):
                        if not d.eq_active[e]:
                            continue  # A verified cut has released this join; it cannot stretch.
                        gap, _ang = EA.eq_residual(m, d, e)
                        fr = gap / max(float(ch["L"][k]), 1e-6)
                        info[name]["max_stretch_frac"] = max(info[name]["max_stretch_frac"], round(fr, 3))
                        if fr > stretch_frac:
                            events["stretch"].append({"chain": name, "frame": t, "label": ep.labels[t], "segments": [k, k + 1], "value": round(fr, 3)})
                if len(I):
                    s_, t_, dist = _closest(P0[I], P1[I], P0[J], P1[J])
                    pen = (r[I] + r[J]) - dist                 # capsules: axis distance against the radii
                    bb = np.where(gr["bb"])[0]
                    if len(bb):
                        # box links (ribbons): the boxes' own overlap, for the pairs whose bounding spheres meet
                        gi_, gj_ = np.array(gr["geoms"])[I[bb]], np.array(gr["geoms"])[J[bb]]
                        C1, C2 = d.geom_xpos[gi_], d.geom_xpos[gj_]
                        h1, h2 = m.geom_size[gi_][:, :3], m.geom_size[gj_][:, :3]
                        near = np.linalg.norm(C2 - C1, axis=1) < np.linalg.norm(h1, axis=1) + np.linalg.norm(h2, axis=1)
                        ov = np.full(len(bb), -np.inf)
                        if near.any():
                            ov[near] = _box_overlap(C1[near], d.geom_xmat[gi_[near]].reshape(-1, 3, 3), h1[near],
                                                    C2[near], d.geom_xmat[gj_[near]].reshape(-1, 3, 3), h2[near])
                        pen[bb] = ov
                    for ci in gr["chains"]:
                        mine = (gr["cidx"][I] == ci) | (gr["cidx"][J] == ci)
                        if mine.any():
                            nm_ = chains[ci]["name"]
                            info[nm_]["max_self_pen_mm"] = max(info[nm_]["max_self_pen_mm"], round(1000 * float(pen[mine].max()), 2))
                    k = int(np.argmax(pen / gr["thr"]))
                    nm, sg, _ = where(gr, k)
                    if pen[k] > gr["thr"][k]:
                        events["self"].append({"chain": nm, "frame": t, "label": ep.labels[t], "segments": sg, "value": round(1000 * float(pen[k]), 1)})
                if tb is not None:
                    low = np.minimum(P0[:, 2], P1[:, 2]) - r
                    for n_, (g, bx) in enumerate(zip(gr["geoms"], gr["box"])):
                        if bx:                                  # a box's lowest corner
                            low[n_] = d.geom_xpos[g][2] - float(np.abs(d.geom_xmat[g].reshape(3, 3)[2]) @ m.geom_size[g][:3])
                    over = (np.minimum(P0[:, 0], P1[:, 0]) >= tb["xy"][0]) & (np.maximum(P0[:, 0], P1[:, 0]) <= tb["xy"][1]) & \
                           (np.minimum(P0[:, 1], P1[:, 1]) >= tb["xy"][2]) & (np.maximum(P0[:, 1], P1[:, 1]) <= tb["xy"][3])
                    for ci in gr["chains"]:
                        sel = over & (gr["cidx"] == ci)
                        if not sel.any():
                            continue
                        name = chains[ci]["name"]
                        dep = tb["top_z"] - low[sel]
                        k = int(np.argmax(dep))
                        info[name]["max_table_pen_mm"] = max(info[name]["max_table_pen_mm"], round(1000 * float(dep[k]), 2))
                        if dep[k] > pen_m:
                            events["table"].append({"chain": name, "frame": t, "label": ep.labels[t],
                                                    "segments": [int(gr["sidx"][np.where(sel)[0][k]])], "value": round(1000 * float(dep[k]), 1)})
            # rule 5: crossings between this frame and the previous one
            if gi in prev_ends and len(I):
                Q0, Q1 = prev_ends[gi]
                v_prev = np.einsum("ij,ij->i", Q1[I] - Q0[I], np.cross(Q1[J] - Q0[J], Q0[J] - Q0[I]))
                v_now = np.einsum("ij,ij->i", P1[I] - P0[I], np.cross(P1[J] - P0[J], P0[J] - P0[I]))
                flip = np.where(np.sign(v_prev) * np.sign(v_now) < 0)[0]
                if len(flip):
                    M0, M1 = 0.5 * (Q0 + P0), 0.5 * (Q1 + P1)
                    s_, t_, dist = _closest(M0[I[flip]], M1[I[flip]], M0[J[flip]], M1[J[flip]])
                    inner = (s_ > 0.02) & (s_ < 0.98) & (t_ > 0.02) & (t_ < 0.98) & (dist < (r[I[flip]] + r[J[flip]]) * 0.5)
                    for k in np.where(inner)[0]:
                        nm, sg, (ca, cb) = where(gr, int(flip[k]))
                        for c_ in {ca, cb}:
                            info[c_]["crossings"] += 1
                        events["crossing"].append({"chain": nm, "frame": t, "label": ep.labels[t], "frames": [t - 1, t],
                                                   "segments": sg, "value": round(1000 * float(dist[k]), 1)})
            prev_ends[gi] = (P0, P1)
        if scan is not None and judge:
            for b1, b2, dist, *_ in scan.at(t):
                for bh, bo in ((b1, b2), (b2, b1)):
                    if bh in hand_bodies:
                        ch = next((c for c in chains if bo in c["bodies"]), None)
                        if ch is not None:
                            info[ch["name"]]["max_hand_pen_mm"] = max(info[ch["name"]]["max_hand_pen_mm"], round(-1000 * dist, 2))
                            if -dist > hand_pen_m:
                                events["hand"].append({"chain": ch["name"], "frame": t, "label": ep.labels[t], "segments": [ch["bodies"].index(bo)],
                                                       "value": round(-1000 * dist, 1), "hand_body": ep.body_name(bh)})
    advice = {"stretch": ("is stretched", "the constraint links give under load: pull more slowly or hold the rope closer to the load"),
              "self": ("passes into itself", "the rope is pressed through its own segments: loosen the pull at that crossing"),
              "table": ("sinks into the tabletop", "the rope is pressed into the table: lift the pinch or lower the pressing force"),
              "hand": ("has a finger inside it", "the fingers cut into the rope: pinch it with the pads, not by closing through it"),
              "crossing": ("passes through itself between frames", "the rope tunnels through its own segment: slow the motion at that "
                                                                   "crossing (more frames) or raise the cable's contact stiffness")}
    issues, worst = [], None
    for rule, evs in events.items():
        if not evs:
            continue
        w = max(evs, key=lambda e: e["value"])
        unit = "x segment length" if rule == "stretch" else "mm"
        what, fix = advice[rule]
        issues.append(f"{w['chain']} {what}: {w['value']} {unit} at frame {w['frame']} ({w['label']}), segments {w['segments']} "
                      f"({len(evs)} events); {fix}")
        if worst is None:
            worst = dict(w, rule=rule)
    return EA.finish(issues, worst, chains=info, n_events={k: len(v) for k, v in events.items()}, frames=[ep.a, ep.b],
                     recording_grid=ep.extras.get("recording_grid", "video_frames"), n_states_checked=len(ep.window))


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(deformable_integrity_audit(sys.argv[1], sys.argv[2]), indent=1))
