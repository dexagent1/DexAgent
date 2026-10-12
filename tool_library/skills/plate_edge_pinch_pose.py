"""Pinch a thin plate (lid, panel, board, cover) across one of its edges with the hand kept outside the plate."""
from __future__ import annotations

TOOL = {
    "name": "plate_edge_pinch_pose",
    "title": "Plate edge pinch pose",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Solve a thumb-versus-fingers pinch across the edge of a thin box geom (a lid, panel, board or cover) with "
               "the thumb pad on one face, the finger pads on the other and the palm and knuckles kept outside the edge.",
    "description": (
        "Reads the plate from a box geom in the hand scene: its thin axis is the face normal, `edge` names the local "
        "in-plane axis and sign of the edge to pinch, `s` the point along that edge (0..1) and `inset_m` how far in from "
        "the edge the pads sit. The thumb target lies on the `thumb_face` side ('up' = the face whose normal points up "
        "in the world, 'down' the other), the other grip fingers on the opposite face, spread along the edge by "
        "`finger_spread_m`; every target is a pad radius (+1 mm) off its face. Fingertip IK (the library's damped "
        "least squares with the collision-escape null-space task) is run from every wrist seed backed off along the "
        "outward edge normal (tilted up by `seed_up`), one seed at a time, and a solution is accepted only when its "
        "worst pad residual is under `max_residual_mm` and every hand link except the middle and distal phalanges stays "
        "at least `min_out_m` outside the edge, so the palm never lies over the plate, and the hand opened to `open_frac` at the pose (as planners open it before "
        "closing) keeps `open_floor_clear_m` off the floor, so a plate lying low over a table can still be reached. "
        "With `pivot` (a hinge line) the palm closest to the line wins, so turning the plate with the hand is mostly a "
        "rotation of the hand. With `finger_axis_max_deg` the fingers must point into the plate within that angle. Among accepted solutions the "
        "self-collision-free solutions (the visible finger meshes intersecting by at most 1 mm, closed and opened; the fingers with no target are folded and spread clear) win, then the smallest hand-object overlap, then the largest outside margin. The pose carries `approach` = the outward "
        "edge normal tilted up, so planners and stress tests bring the open hand in from the side of the edge rather "
        "than from above. Records surface points and outward normals separately from pad-centre targets, and anchors "
        "them to the materialized plate body so joint-following planners retain the actual contact reference. "
        "Assumes the plate is a box geom whose thinnest half-size is its thickness."
    ),
    "inputs": [
        {"name": "scene", "type": "dict", "description": "Hand scene {hand_xml, side, objects, floor, ...} (hand_scene_from_recon output)."},
        {"name": "geom", "type": "str", "description": "Name (or unique name suffix) of the plate's box geom in the hand scene."},
        {"name": "edge", "type": "str", "description": "Edge to pinch as a signed local in-plane axis: '+x', '-x', '+y', '-y' (or '+z'/'-z' when z is in-plane)."},
        {"name": "s", "type": "float", "description": "Position along the edge, 0..1 along the remaining in-plane local axis. Default 0.5."},
        {"name": "inset_m", "type": "float", "description": "Distance of the pads in from the edge. Default 0.014."},
        {"name": "thumb_face", "type": "str", "description": "'up' or 'down': world-up face or the other one for the thumb. Default 'up'."},
        {"name": "fingers", "type": "list", "description": "Grip fingers, thumb first. Default ['thumb', 'index', 'middle']."},
        {"name": "finger_spread_m", "type": "float", "description": "Spacing of the opposing fingers along the edge. Default 0.018."},
        {"name": "min_out_m", "type": "float", "description": "Required clearance of palm and proximal links outside the edge. Default 0.02."},
        {"name": "max_residual_mm", "type": "float", "description": "Worst accepted pad residual. Default 3."},
        {"name": "seed_up", "type": "float", "description": "Upward tilt of the seed and approach direction (added to the unit edge normal). Default 0.3."},
        {"name": "yaw_step_deg / pitches_deg / back_m / iters", "type": "numbers", "description": "Seed sweep density, seed back-off and IK iterations per seed (45, [-60,-30,0,30,60], 0.12, 200)."},
        {"name": "finger_axis_max_deg", "type": "float | None", "description": "Largest angle between the hand's finger axis (palm to middle knuckle) and the inward edge normal; e.g. 35 keeps the fingers pointing into the plate so the hand pivots with a hinged plate instead of swinging over it. Default None (any)."},
        {"name": "finger_dir", "type": "xyz | None", "description": "World direction the finger axis is compared with under finger_axis_max_deg (default: into the plate, minus the edge normal); use it to match the direction an arm can present the hand from."},
        {"name": "pivot", "type": "dict | None", "description": "{point, axis}: the hinge line the plate turns about (world). When given, solutions whose palm lies closest to that line win (after overlap), so the hand mostly rotates instead of swinging through an arc when the plate turns. Default None."},
        {"name": "max_pivot_dist_m", "type": "float | None", "description": "With pivot: reject solutions whose palm is farther than this from the hinge line. Default None."},
        {"name": "open_frac", "type": "float", "description": "Opening fraction the planners use before closing (H.open_targets). Default 0.55."},
        {"name": "open_floor_clear_m", "type": "float", "description": "Clearance every hand geom of the OPENED hand may reach into the floor at the pose: the deepest hand-floor contact distance must be at least this (negative = penetration). Default -0.001."},
        {"name": "out_json", "type": "str | None", "description": "Write the pose here."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "{side, pos, euler, finger_targets, finger_targets_squeezed, grip_fingers, points (surface), normals, targets (pad centres), anchor, approach, press_depth_rad, residual_mm, worst_residual_mm, overlap_mm, outside_m, edge_point, n_seeds, n_accepted, self_overlap_mm, n_self_collision_free}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "pinch", "plate", "lid", "edge", "panel"],
    "used_in": ["Giftbox Packing"],
    "status": "tested",
    "entrypoint": "plate_edge_pinch_pose",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H

SELF_OVERLAP_MM = 1.0          # visible finger-finger intersection a pose may have (closed and opened)

_AX = {"x": 0, "y": 1, "z": 2}


def plate_edge_pinch_pose(scene, geom, edge, s=0.5, inset_m=0.014, thumb_face="up", fingers=("thumb", "index", "middle"),
                          finger_spread_m=0.018, min_out_m=0.02, max_residual_mm=3.0, seed_up=0.3, yaw_step_deg=45,
                          pitches_deg=(-60, -30, 0, 30, 60), back_m=0.12, iters=200, open_frac=0.55, open_floor_clear_m=-0.001,
                          finger_axis_max_deg=None, finger_dir=None, pivot=None, max_pivot_dist_m=None, out_json=None):
    if isinstance(scene, (str, Path)):                 # a hand-scene json path is as good as the dict
        import json as _json
        scene = _json.loads(Path(scene).read_text())
    if not isinstance(scene, dict):
        raise ValueError("scene must be a hand-scene dict (hand_scene_from_recon output) or its json path")
    if not isinstance(edge, str) or len(edge) != 2 or edge[0] not in "+-" or edge[1] not in _AX:
        raise ValueError("edge must be one of '+x', '-x', '+y', '-y', '+z', '-z'")
    if thumb_face not in ("up", "down"):
        raise ValueError("thumb_face must be 'up' or 'down'")
    fingers = list(fingers)
    if len(fingers) < 2 or fingers[0] != "thumb":
        raise ValueError("fingers must start with 'thumb' and name at least one opposing finger")
    if not 0.0 <= float(s) <= 1.0:
        raise ValueError("s must lie in [0, 1]")
    model, d0, rig, ids = H.open_scene(scene)
    gids = [i for i in range(model.ngeom) if model.geom(i).name == geom] or \
           [i for i in range(model.ngeom) if model.geom(i).name.endswith(geom)]
    if len(gids) != 1:
        raise ValueError(f"geom {geom!r} matches {len(gids)} geoms; give a unique name")
    g = gids[0]
    if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError(f"geom {geom!r} is not a box")
    c = d0.geom_xpos[g].copy(); R = d0.geom_xmat[g].reshape(3, 3).copy(); hs = model.geom_size[g].copy()
    thin = int(np.argmin(hs)); ax = _AX[edge[1]]
    if ax == thin:
        raise ValueError(f"edge axis {edge[1]} is the plate's thickness axis; choose an in-plane axis")
    tax = [i for i in range(3) if i not in (thin, ax)][0]
    ne = (1.0 if edge[0] == "+" else -1.0) * R[:, ax]; tt = R[:, tax]; nf = R[:, thin]
    if nf[2] < 0:
        nf = -nf
    nth = nf if thumb_face == "up" else -nf
    p = c + ne * (hs[ax] - float(inset_m)) + tt * (float(s) - 0.5) * 2.0 * hs[tax] * 0.9
    standoff = {f: H.pad_radius(model, rig, f) + 0.001 for f in fingers}
    so = standoff.get("index", standoff[fingers[0]])
    pts, nrm = {"thumb": p + nth * hs[thin]}, {"thumb": nth}
    others = fingers[1:]
    for k, f in enumerate(others):
        off = (k - (len(others) - 1) / 2.0) * float(finger_spread_m)
        pts[f] = p - nth * hs[thin] + tt * off; nrm[f] = -nth
    targets = {f: pts[f] + nrm[f] * standoff[f] for f in pts}
    cen = np.mean(list(targets.values()), axis=0)
    app = ne + float(seed_up) * np.array([0.0, 0.0, 1.0]); app /= np.linalg.norm(app)
    seeds = H.seed_sweep(cen, [app], back_m=float(back_m), yaw_step_deg=yaw_step_deg, pitches_deg=tuple(pitches_deg))
    # links that carry collision geometry (the floating base's slide/hinge bodies have none)
    links = sorted({int(model.geom_bodyid[i]) for i in ids["hand_gids"]}
                   - {b for b in rig.body_ids if any(k in model.body(b).name for k in ("DP", "MP"))})
    piv = None
    if pivot is not None:
        if not isinstance(pivot, dict) or "point" not in pivot or "axis" not in pivot:
            raise ValueError("pivot must be {point: xyz, axis: xyz} (a hinge line in world coordinates)")
        pa = np.asarray(pivot["axis"], float); piv = (np.asarray(pivot["point"], float), pa / (np.linalg.norm(pa) + 1e-12))
    best = None; n_ok = 0; n_self_free = 0; tried = []
    for sd in seeds:
        worst, res, pen, q, face = H.solve_fingertip_ik(model, d0, rig, ids, targets, [sd], iters=int(iters))
        d = mujoco.MjData(model); d.qpos[:] = q; mujoco.mj_forward(model, d)
        outside = min(float((d.xpos[b] - p) @ ne) for b in links)
        palm_b = [b for b in links if model.body(b).name.endswith("C_MC")] or [links[0]]
        mid_b = [b for b in rig.body_ids if model.body(b).name.endswith("middle_MCP_VL")]
        fax_deg = 0.0
        if mid_b:
            fv = d.xpos[mid_b[0]] - d.xpos[palm_b[0]]
            want = -ne if finger_dir is None else np.asarray(finger_dir, float) / (np.linalg.norm(finger_dir) + 1e-12)
            fax_deg = float(np.degrees(np.arccos(np.clip(fv @ want / (np.linalg.norm(fv) + 1e-12), -1, 1))))
        if finger_axis_max_deg is not None and fax_deg > float(finger_axis_max_deg):
            tried.append((round(worst * 1000, 1), round(outside, 4), None, round(fax_deg)))
            continue
        tried.append((round(worst * 1000, 1), round(outside, 4)))
        if worst * 1000.0 > float(max_residual_mm) or outside < float(min_out_m):
            continue
        # the fingers with no target folded clear of the part, the floor and the other fingers (they keep the IK seed's
        # posture otherwise, and an equal ring/pinky fold crosses their meshes), as the closed hand the pose records
        H.curl_free_fingers_clear(model, d, rig, ids, set(pts))
        q = d.qpos.copy()
        self_closed = H.hand_self_overlap(model, d, rig)
        # the open hand (as planners and stress tests open it before closing) must not reach into the floor
        closed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
        rig.set_fingers(d, H.open_targets(rig, closed, float(open_frac), curl=[f for f in H.FINGERS if f not in pts])); mujoco.mj_forward(model, d)
        # finger-finger intersection of the visible meshes, closed and opened: the stress tests and the executor reject
        # more than about a millimetre, so a self-collision-free solution outranks every other
        self_mm = 1000.0 * max(self_closed, H.hand_self_overlap(model, d, rig))
        fclear = 1.0
        fg = int(ids.get("floor_gid", -1))
        if fg >= 0:                                  # the scene's own contact pairs: deepest hand-floor contact
            hg = set(int(x) for x in ids["hand_gids"])
            for i in range(d.ncon):
                cc = d.contact[i]
                if (cc.geom1 == fg and cc.geom2 in hg) or (cc.geom2 == fg and cc.geom1 in hg):
                    fclear = min(fclear, float(cc.dist))
        tried[-1] = tried[-1] + (round(fclear, 4),)
        if fclear < float(open_floor_clear_m):
            continue
        pdist = 0.0
        if piv is not None:
            v = d.xpos[palm_b[0]] - piv[0]; pdist = float(np.linalg.norm(v - (v @ piv[1]) * piv[1]))
            if max_pivot_dist_m is not None and pdist > float(max_pivot_dist_m):
                continue
        n_ok += 1
        n_self_free += self_mm <= SELF_OVERLAP_MM
        key = ((self_mm > SELF_OVERLAP_MM, round(max(pen, 0.004) * 500) / 500.0, round(pdist, 2), -outside) if piv is not None else
               (self_mm > SELF_OVERLAP_MM, round(max(pen, 0.004) * 500) / 500.0, -outside))
        if best is None or key < best[0]:
            best = (key, worst, res, pen, q.copy(), outside, fax_deg, pdist, self_mm)
    if best is None:
        raise ValueError(f"no seed reached the pads with the hand {min_out_m} m outside the edge; lower min_out_m, "
                         f"raise inset_m or max_residual_mm, or try the other thumb_face (best residual_mm "
                         f"{min(t[0] for t in tried)}, best outside_m {max(t[1] for t in tried)}, among them "
                         f"{sorted(tried, key=lambda t: -t[1])[:5]}; best open-hand floor distance {max((t[2] for t in tried if len(t) > 2 and t[2] is not None), default=None)}, "
                         f"smallest finger-axis angle {min((t[3] for t in tried if len(t) > 3), default=None)})")
    _, worst, res, pen, q, outside, fax_deg, pdist, self_mm = best
    d = mujoco.MjData(model); d.qpos[:] = q; mujoco.mj_forward(model, d)
    pose = H.pose_from_data(model, d, rig, rig.side)
    H.finger_ik(model, d, rig, {f: pts[f] + nrm[f] * (standoff[f] - 0.003) for f in pts})
    squeezed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
    pose.update({
        "grip_fingers": sorted(pts), "finger_targets_squeezed": squeezed, "press_depth_rad": 0.12,
        "approach": [float(v) for v in app],
        "residual_mm": {f: round(v * 1000, 2) for f, v in res.items()}, "worst_residual_mm": round(worst * 1000, 2),
        "overlap_mm": round(pen * 1000, 2), "outside_m": round(outside, 4), "finger_axis_deg": round(fax_deg, 1), "pivot_dist_m": round(pdist, 4), "standoff_m": float(so),
        "standoff_m_by_finger": {f: float(v) for f, v in standoff.items()},
        "targets": {f: [float(v) for v in t] for f, t in targets.items()}, "edge_point": [float(v) for v in p],
        "points": {f: [float(v) for v in t] for f, t in pts.items()},
        "normals": {f: [float(v) for v in t] for f, t in nrm.items()},
        "edge_normal": [float(v) for v in ne], "n_seeds": len(seeds), "n_accepted": n_ok,
        "self_overlap_mm": round(self_mm, 2), "n_self_collision_free": int(n_self_free),
    })
    # The object can have settled or have a joint_init different from an older
    # grasp_body_pose record. These points were solved against d0, so record
    # that materialized transform, not the potentially stale input metadata.
    body = int(model.geom_bodyid[g])
    if body > 0:
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, d0.xmat[body])
        pose["anchor"] = {"body": model.body(body).name, "pos": d0.xpos[body].tolist(),
                          "quat_wxyz": quat.tolist()}
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(pose, fh, indent=1)
        pose["out_json"] = str(out_json)
    return pose
