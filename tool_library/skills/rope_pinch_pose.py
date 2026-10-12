TOOL = {
    "name": "rope_pinch_pose",
    "title": "Pinch pose on a rope segment",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Pick a point along a rope (a chain of capsule bodies) by arc length and solve a thumb-versus-finger pinch that straddles it, tilted about the rope tangent.",
    "description": "Reads the rope's capsule segments from the compiled scene (every capsule geom whose name starts with the rope prefix, ordered along the chain), walks the requested arc length to find the segment and its tangent, and solves a pinch across it with the same fingertip IK as handle_pinch_pose. The pinch direction starts horizontal-perpendicular to the tangent and is tilted about the tangent by tilt_deg so the pads come from above-and-beside; the pads must end below the rope's centre line to lift it, which the returned straddle check reports. For a knot the crossing order matters more than the count, so the caller picks arc lengths from the video's ferrules and crossings; this tool only places the hand.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor}; the rope is the last object"},
        {"name": "rope_prefix", "type": "str", "description": "name prefix of the rope's capsule geoms in the compiled scene (default: the last object's name + '_')"},
        {"name": "arc_m", "type": "float", "description": "arc length from the rope's first segment to the pinch point (metres)"},
        {"name": "arc_frac", "type": "float", "description": "alternative to arc_m: fraction of the rope's total length"},
        {"name": "fingers", "type": "list", "description": "pinching fingers, thumb first (default thumb, index)"},
        {"name": "tilt_deg", "type": "float", "description": "tilt of the pinch direction about the tangent (default 35, toward from-above)"},
        {"name": "iters / yaw_step_deg / pitches_deg", "type": "numbers", "description": "IK budget and seed density"},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "handle_pinch_pose output plus segment: {geom, index, arc_m, tangent}"},
        {"name": "rope", "type": "dict", "description": "{n_segments, length_m, radius_m}"},
    ],
    "status": "tested",
    "entrypoint": "rope_pinch_pose",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["rope", "pinch", "deformable", "grasp"],
    "used_in": ["Knot the Rope", "Scissor Cutting"],
}

import numpy as np


def rope_segments(model, data, prefix):
    """Capsule segments under `prefix`, ordered along the chain by following
    the kinematic tree (a rope made of several chains: chain after chain, each entered from its end nearest the
    previous one), as [{"geom", "p0", "p1", "radius", "body"}]."""
    import mujoco
    caps = [g for g in range(model.ngeom)
            if (model.geom(g).name or "").startswith(prefix)
            and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)]
    if not caps:
        boxes = [g for g in range(model.ngeom) if (model.geom(g).name or "").startswith(prefix)
                 and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
        raise ValueError(f"no capsule geoms start with '{prefix}'" + (
            f": its {len(boxes)} links are flat boxes (a ribbon or strip); pinch one link as a thin plate with handle_pinch_pose "
            f"(object = the link's body) where it arches clear of the table" if boxes else ""))
    # order by tree depth (a chain's bodies nest one inside the next), chain by chain: a rope built in parts (the two
    # halves of a cuttable rope, joined by an equality) has several chains at the same depths, which must not interleave
    def depth(b):
        k = 0
        while b > 0:
            b = int(model.body_parentid[b])
            k += 1
        return k

    def root(b):
        while int(model.body_parentid[b]) > 0:
            b = int(model.body_parentid[b])
        return b
    chains = {}
    for g in caps:
        chains.setdefault(root(int(model.geom_bodyid[g])), []).append(g)
    for r in chains:
        chains[r].sort(key=lambda g: (depth(int(model.geom_bodyid[g])), g))
    order = [chains.pop(min(chains))]
    while chains:
        # the next chain is the one with an end nearest the end of the rope so far, entered from that end
        tail = np.asarray(data.geom_xpos[order[-1][-1]], float)
        best = min(((float(np.linalg.norm(np.asarray(data.geom_xpos[c[i]], float) - tail)), r, i) for r, c in chains.items() for i in (0, -1)))
        nxt = chains.pop(best[1])
        order.append(nxt if best[2] == 0 else nxt[::-1])
    caps = [g for c in order for g in c]
    out = []
    for g in caps:
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        c = np.asarray(data.geom_xpos[g], float)
        h = float(model.geom_size[g][1])
        out.append({"geom": model.geom(g).name, "p0": c - R[:, 2] * h, "p1": c + R[:, 2] * h,
                    "radius": float(model.geom_size[g][0]), "body": int(model.geom_bodyid[g])})
    # orient every segment so p1 of one is next to p0 of the next
    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        if np.linalg.norm(cur["p1"] - prev["p1"]) < np.linalg.norm(cur["p0"] - prev["p1"]):
            cur["p0"], cur["p1"] = cur["p1"], cur["p0"]
    if len(out) > 1 and np.linalg.norm(out[0]["p1"] - out[1]["p0"]) > np.linalg.norm(out[0]["p0"] - out[1]["p0"]):
        out[0]["p0"], out[0]["p1"] = out[0]["p1"], out[0]["p0"]
    return out


def rope_pinch_pose(scene, arc_m=None, arc_frac=None, rope_prefix=None, fingers=("thumb", "index"),
                    tilt_deg=35.0, iters=320, yaw_step_deg=30, pitches_deg=(-60, -30, 0, 30, 60)):
    from tool_library.common import hand as H

    model, data0, rig, ids = H.open_scene(scene)
    prefix = rope_prefix or ids["grasp_prefix"]
    segs = rope_segments(model, data0, prefix)
    lengths = [float(np.linalg.norm(s["p1"] - s["p0"])) for s in segs]
    total = float(sum(lengths))
    if arc_m is None:
        if arc_frac is None:
            arc_frac = 0.5
        arc_m = float(arc_frac) * total
    arc_m = float(np.clip(arc_m, 0.0, total))
    acc, idx, local = 0.0, len(segs) - 1, 1.0
    for i, L in enumerate(lengths):
        if arc_m <= acc + L or i == len(segs) - 1:
            idx, local = i, (arc_m - acc) / max(L, 1e-9)
            break
        acc += L
    seg = segs[idx]
    bar = {"p0": seg["p0"].tolist(), "p1": seg["p1"].tolist(), "radius": seg["radius"]}
    pose = H.bar_pinch_pose(model, data0, rig, ids, bar, s=float(np.clip(local, 0.0, 1.0)),
                            fingers=fingers, tilt_deg=tilt_deg, iters=iters,
                            yaw_step_deg=yaw_step_deg, pitches_deg=pitches_deg)
    tangent = (seg["p1"] - seg["p0"]) / max(lengths[idx], 1e-9)
    pose["segment"] = {"geom": seg["geom"], "index": idx, "arc_m": arc_m, "tangent": tangent.tolist(),
                       "body": model.body(seg["body"]).name}
    pose["rope"] = {"n_segments": len(segs), "length_m": round(total, 4), "radius_m": seg["radius"]}
    return pose
