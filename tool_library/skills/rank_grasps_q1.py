TOOL = {
    "name": "rank_grasps_q1",
    "title": "Rank grasps by Ferrari-Canny Q1",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Close each candidate grasp on the object in physics, read the simulator's contact set, and rank the candidates by the Ferrari-Canny Q1 force-closure quality and a press-versus-grip ratio.",
    "description": "For every pose the hand is placed open, the fingers are driven to the closed (or squeezed) targets and the scene settles; the hand-object contacts (positions, normals, forces, friction) are read from MuJoCo and turned into a 6-D wrench hull with 8-edge friction cones, torques normalised by the object's radius. Q1 is the radius of the largest origin-centred ball inside that hull: Q1 > 0 means force closure. press_frac = |sum F| / sum |F| tells a press (all fingers pushing the same way, ~1) from a grip (~0). Candidates are sorted by Q1 descending, then press_frac ascending, then number of contacts. Static and cheap, so it is the right first filter before the driven stress tests; it does not replace them, because a servo-held grasp can have force closure at rest and still slip under motion.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, floor, grasp_body?}"},
        {"name": "poses", "type": "list of pose dicts", "description": "candidates with pos, euler, finger_targets (and optionally finger_targets_squeezed)"},
        {"name": "settle_s", "type": "float", "description": "settle time after closing (default 0.5)"},
        {"name": "use_squeezed", "type": "bool", "description": "close to finger_targets_squeezed when present (default True)"},
        {"name": "min_force_N", "type": "float", "description": "ignore contacts weaker than this (default 0.05)"},
    ],
    "outputs": [
        {"name": "ranked", "type": "list", "description": "[{index, q1, press_frac, n_contacts, fingers, total_force_N, force_closure}] best first"},
        {"name": "best", "type": "int or None", "description": "index of the best candidate with force closure"},
    ],
    "status": "tested",
    "entrypoint": "rank_grasps_q1",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "quality", "force-closure", "ferrari-canny"],
    "used_in": ["Place Cup in Bowl", "Battery Insertion", "Open the Bottle"],
}

import numpy as np


def rank_grasps_q1(scene, poses, settle_s=0.5, use_squeezed=True, min_force_N=0.05):
    import mujoco
    from tool_library.common import hand as H

    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    radius = float(np.max(H.object_extent(model, data0, ids["grasp_gids"])["size"])) / 2.0
    out = []
    for i, pose in enumerate(poses):
        d = mujoco.MjData(model)
        d.qpos[:] = data0.qpos
        closed = {k: float(v) for k, v in (pose.get("finger_targets") or {}).items() if k in rig.finger_range}
        target = {k: float(v) for k, v in (pose.get("finger_targets_squeezed") or {}).items()
                  if k in rig.finger_range} if use_squeezed else {}
        target = target or closed
        opened = H.open_targets(rig, closed, 0.55, curl=[f for f in H.FINGERS
                                                         if pose.get("grip_fingers") and f not in pose["grip_fingers"]])
        H.apply_pose(model, d, rig, pose, opened)
        H.step_hold(model, d, rig, pose["pos"], pose["euler"], opened, 0.15)
        H.ramp(model, d, rig, pose["pos"], pose["pos"], pose["euler"], pose["euler"], opened, target, 0.4)
        H.step_hold(model, d, rig, pose["pos"], pose["euler"], target, settle_s)
        cons = H.hand_object_contacts(model, d, rig, ids["grasp_gids"], min_force=min_force_N)
        com = np.asarray(d.xipos[bid], float)
        q1 = H.q1_metric(cons, com, radius=radius)
        pf = H.press_fraction(cons)
        fingers = sorted({f for c in cons for f in H.FINGERS if f"_{f}_" in (c["geom"] or "")})
        out.append({"index": i, "q1": round(q1, 4), "press_frac": round(pf, 3), "n_contacts": len(cons),
                    "fingers": fingers, "total_force_N": round(sum(c["force"] for c in cons), 2),
                    "force_closure": bool(q1 > 0.0),
                    "object_moved_mm": round(float(np.linalg.norm(d.xpos[bid] - data0.xpos[bid])) * 1000, 1)})
    ranked = sorted(out, key=lambda r: (-r["q1"], r["press_frac"], -r["n_contacts"]))
    best = next((r["index"] for r in ranked if r["force_closure"]), None)
    return {"ranked": ranked, "best": best}
