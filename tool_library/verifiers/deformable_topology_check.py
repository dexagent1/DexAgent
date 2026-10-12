"""Deformable topology check: a rope or cable is one connected chain with free ends, flexible joints and no
self-intersection at rest."""
from __future__ import annotations

TOOL = {
    "name": "deformable_topology_check",
    "title": "Deformable topology check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "For every rope/cable object (a chain of linked bodies), checks the topology of its link graph (number of strands and of free ends), that every link after the first is jointed to its neighbour (no fused segments), that no active equality pins a link to the world, the table, a mocap body or the robot at the initial state, that non-adjacent links can collide with each other (a knot needs self-contact) and that no two non-adjacent links interpenetrate at keyframe 0.",
    "description": (
        "The paper checks deformables for topology. A rope modelled as a MuJoCo chain (cable composite, construct_rope, "
        "construct_cuttable_rope) can go wrong silently: two cables meant to be joined left as two strands, a link "
        "without a joint that makes a stiff rod of part of the rope, an end welded to the table or to a gripper "
        "mocap at the start, self-collision masked off so the rope passes through itself and no knot can hold, or a "
        "centreline that crosses itself at rest. Rope objects are the Stage 2 manifest's deformable objects whose "
        "skill, id or root name says rope/cable/thread/string/cord/wire, else any top-level object whose bodies form "
        "a chain of at least min_links linked bodies. The link graph has the object's geometry-carrying bodies as "
        "nodes and parent-child links plus equality constraints between links active at keyframe 0 as edges: its "
        "connected components are strands, its degree-1 nodes ends (a closed loop has none). Expected: "
        "expected_strands strands and expected_ends ends (default 1 and 2, or the object's declared `topology` in scene_recon.json, "
        "which the report lists under `declared`). Self-contact: the fraction of "
        "non-adjacent link pairs whose contype/conaffinity allow contact and that no <exclude> removes must be "
        "positive when require_self_collision; interpenetration: capsule/cylinder/sphere links closer than -self_tol_m "
        "(segment distance minus radii) at keyframe 0 while at least min_gap_links apart along the chain."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json naming the deformable objects."},
        {"name": "objects", "type": "list[str] | None", "description": "Only these objects (ids or root body names)."},
        {"name": "expected_strands", "type": "int", "description": "Connected strands per rope object. Default 1; an object whose scene_recon.json entry has `topology: {strands, ends}` (a ribbon of three strands joined at one end: strands 1, ends 3) is held to that instead."},
        {"name": "expected_ends", "type": "int | None", "description": "Free ends per rope object. Default 2 per strand (0 for a loop)."},
        {"name": "require_self_collision", "type": "bool", "description": "Non-adjacent links must be able to touch. Default True."},
        {"name": "self_tol_m", "type": "float", "description": "Allowed interpenetration of non-adjacent links at rest, metres. Default 0.002."},
        {"name": "min_gap_links", "type": "int", "description": "Links this many apart along the chain count as non-adjacent. Default 2."},
        {"name": "min_links", "type": "int", "description": "Bodies a chain needs to count as a rope without a manifest. Default 5."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; ropes [{id, n_links, strands, ends, fused_links, pinned, self_collision_frac, interpenetrations, verdict}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "deformable", "rope", "cable", "topology", "stage2"],
    "status": "tested",
    "entrypoint": "deformable_topology_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA

def _hops(links, adj):
    """Graph distance (joints) between every pair of links: for a chain it is the index difference, for the strands of a
    ribbon it runs through their junction."""
    out = {}
    for s in links:
        dist, front = {s: 0}, [s]
        while front:
            nxt = []
            for x in front:
                for y in adj[x]:
                    if y not in dist:
                        dist[y] = dist[x] + 1; nxt.append(y)
            front = nxt
        out[s] = dist
    return out


def _declared_topology(recon):
    """{object id: {strands, ends}} from scene_recon.json entries that declare a topology other than one chain."""
    if recon is None:
        return {}
    if not isinstance(recon, dict):
        import json
        from pathlib import Path
        try:
            recon = json.loads(Path(recon).read_text())
        except (OSError, ValueError):
            return {}
    out = {}
    for o in recon.get("objects", []):
        topo = o.get("topology")
        if isinstance(topo, dict) and ("strands" in topo or "ends" in topo):
            out[o.get("id")] = {k: int(topo[k]) for k in ("strands", "ends") if k in topo}
    return out


def _box_depth(model, data, x, y):
    """Interpenetration depth (m, >= 0) of two geoms, one a box. mj_geomDistance returns 0 instead of the depth for two
    coincident convex shapes: centres closer than the thinner half-size count as fully overlapping."""
    d = float(mujoco.mj_geomDistance(model, data, x, y, 0.0, None))
    if d == 0.0:
        thin = min(float(np.min(model.geom_size[x][:3][model.geom_size[x][:3] > 0])), float(np.min(model.geom_size[y][:3][model.geom_size[y][:3] > 0])))
        if float(np.linalg.norm(data.geom_xpos[x] - data.geom_xpos[y])) < thin:
            return 2.0 * thin
    return max(0.0, -d)


def _segment(model, data, g):
    """(p0, p1, radius) of a capsule/cylinder geom's axis segment, or a sphere as a point; None otherwise."""
    t = int(model.geom_type[g]); s = model.geom_size[g]
    c = data.geom_xpos[g]; R = data.geom_xmat[g].reshape(3, 3)
    if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        return c - R[:, 2] * s[1], c + R[:, 2] * s[1], float(s[0])
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return c, c, float(s[0])
    return None


def _seg_dist(p1, q1, p2, q2):
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def deformable_topology_check(scene, recon=None, objects=None, expected_strands=1, expected_ends=None, require_self_collision=True,
                              self_tol_m=0.002, min_gap_links=2, min_links=5):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    robot = SA.robot_body_ids(model, st["model"])
    table = SA.table_body_ids(model, st["cfg"])
    ropes = SA.rope_groups(model, recon, objects, min_links, ref=st["model"], cfg=st["cfg"])
    declared = _declared_topology(recon)
    anchors = SA.declared_anchors(model, recon, robot, table)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    excl = set()
    for i in range(model.nexclude):
        sig = int(model.exclude_signature[i])
        excl.add(frozenset({sig >> 16, sig & 0xFFFF}))       # (body1 << 16) + body2
    rows, issues = [], []
    for g, links in ropes:
        oid = g["id"]
        n_before = len(issues)
        L = set(links)
        # chain order: follow parent links from the link whose parent is outside
        adj = {b: set() for b in links}
        for b in links:
            p = int(model.body_parentid[b])
            # a geometry-less intermediate body (a joint frame) still links its neighbours
            while p not in L and p > 0 and p in g["bodies"] and not SA.body_geoms(model, [p]):
                p = int(model.body_parentid[p])
            if p in L:
                adj[b].add(p); adj[p].add(b)
        pinned, anchored = [], []
        for e in range(model.neq):
            if int(model.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)):
                continue
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            if hasattr(model, "eq_objtype") and int(model.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE):
                b1, b2 = int(model.site_bodyid[b1]), int(model.site_bodyid[b2])
            if not bool(model.eq_active0[e]):
                continue
            if b1 in L and b2 in L:
                adj[b1].add(b2); adj[b2].add(b1)
            elif (b1 in L) != (b2 in L):
                other = b2 if b1 in L else b1
                if other in anchors.get(oid, {}):
                    anchored.append(model.body(other).name); continue    # declared: a cord end held under a fixed object
                if other == 0 or other in table or other in robot or int(model.body_mocapid[other]) >= 0 or other not in g["bodies"]:
                    pinned.append(f"{model.equality(e).name or f'equality {e}'}: {model.body(b1 if b1 in L else b2).name} -> "
                                  f"{model.body(other).name if other > 0 else 'world'}")
        # strands and ends
        seen, strands = set(), 0
        for b in links:
            if b in seen:
                continue
            strands += 1
            stack = [b]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                stack += list(adj[x] - seen)
        ends = sum(1 for b in links if len(adj[b]) == 1)
        want_strands = int((declared.get(oid) or {}).get("strands", expected_strands))
        want_ends = (declared.get(oid) or {}).get("ends", expected_ends if oid not in declared else None)
        want_ends = int(want_ends) if want_ends is not None else 2 * want_strands
        # fused links: a link with no joint of its own whose parent is another link
        fused = [model.body(b).name for b in links if int(model.body_parentid[b]) in L and int(model.body_jntnum[b]) == 0]
        # self-collision capability between non-adjacent links
        order = SA.chain_order(links, adj)
        hops = _hops(links, adj)                      # links min_gap_links+ joints apart along the graph are non-adjacent
        far = [(a, b) for i, a in enumerate(order) for b in order[i + 1:] if hops[a].get(b, 10 ** 9) >= min_gap_links]
        can, tot = 0, 0
        lg = {b: [x for x in SA.body_geoms(model, [b]) if SA.geom_collidable(model, x)] for b in links}
        for a, b in far:
            tot += 1
            if frozenset({a, b}) in excl:
                continue
            if any((int(model.geom_contype[x]) & int(model.geom_conaffinity[y])) or (int(model.geom_contype[y]) & int(model.geom_conaffinity[x]))
                   for x in lg[a] for y in lg[b]):
                can += 1
        frac = can / tot if tot else 1.0
        # interpenetration at rest
        segs = {b: [s for s in (_segment(model, data, x) for x in lg[b]) if s is not None] for b in links}
        boxes = {b: [x for x in lg[b] if int(model.geom_type[x]) == int(mujoco.mjtGeom.mjGEOM_BOX)] for b in links}
        inter = []
        for a, b in far:
            deepest = 0.0
            for (p1, q1, r1) in segs[a]:
                for (p2, q2, r2) in segs[b]:
                    deepest = max(deepest, -(_seg_dist(p1, q1, p2, q2) - r1 - r2))
            for x in boxes[a]:                   # flat links (ribbons): exact convex distance against the other link
                for y in boxes[b] + [g for g in lg[b] if g not in boxes[b]]:
                    deepest = max(deepest, _box_depth(model, data, x, y))
            for y in boxes[b]:
                for x in (g for g in lg[a] if g not in boxes[a]):
                    deepest = max(deepest, _box_depth(model, data, x, y))
            if deepest > self_tol_m:
                inter.append({"links": [model.body(a).name, model.body(b).name], "depth_m": round(deepest, 4)})
        verdict = "ok"
        if strands != want_strands:
            issues.append(f"rope {oid} has {strands} separate strand(s), expected {want_strands}: join its cables (an active weld/connect "
                          f"between the cut ends) or rebuild it as one chain")
        if ends != want_ends:
            issues.append(f"rope {oid} has {ends} free end(s), expected {want_ends}: its link graph is not a simple chain")
        if fused:
            issues.append(f"rope {oid}: {len(fused)} link(s) have no joint to their neighbour ({', '.join(fused[:4])}): that part of the rope "
                          f"is a rigid rod; give every link its ball joint")
        if pinned:
            issues.append(f"rope {oid} is held at the start by {len(pinned)} active equality constraint(s) ({'; '.join(pinned[:3])}): "
                          f"a rope must lie free; deactivate them (eq active=false) or remove them")
        if require_self_collision and tot and can == 0:
            issues.append(f"rope {oid}: no two non-adjacent links can collide (contype/conaffinity or <exclude> masks them): the rope passes "
                          f"through itself and no knot can hold; enable self-collision for links {min_gap_links}+ apart")
        if inter:
            worst = max(inter, key=lambda x: x["depth_m"])
            issues.append(f"rope {oid} intersects itself at keyframe 0: {len(inter)} non-adjacent link pair(s) overlap, worst "
                          f"{worst['links'][0]} / {worst['links'][1]} by {worst['depth_m'] * 1000:.1f} mm; lift the crossing over the other strand")
        if len(issues) > n_before:
            verdict = "bad topology"
        rows.append({"id": oid, "n_links": len(links), "strands": strands, "ends": ends, "declared": declared.get(oid), "anchored_to": anchored, "fused_links": fused, "pinned": pinned,
                     "self_collision_frac": round(frac, 3), "interpenetrations": inter[:10], "verdict": verdict})
    if objects and not ropes:
        # asked to check named objects and none was recognised: a silent pass would prove nothing
        issues.append(f"none of {list(objects)} is a rope or cable in the scene: give recon (scene_recon.json) so its "
                      f"deformable objects and their bodies are known, or the object ids it lists")
    return SA.report(issues, ropes=rows, note=None if ropes else "no rope or cable objects in the scene")
