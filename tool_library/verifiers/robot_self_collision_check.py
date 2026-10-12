"""Does any VISIBLE part of the robot pass through itself, the other station, the table or an object along the episode?"""
from __future__ import annotations

TOOL = {
    "name": "robot_self_collision_check",
    "title": "Robot self-collision check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Replay the episode with every visible robot geom made collidable (its convex hull) and report frames where the "
               "two hands, a hand and an arm, two arm links that are not neighbours, an arm link and the table or an object, a "
               "hand and the table, or a hand's VISIBLE mesh and an object interpenetrate.",
    "description": (
        "The hands ship with contacts off and the arms carry no collision geoms at all, so physics never objects when "
        "the plan drives the left thumb through the right palm or an elbow through the table. This verifier compiles a "
        "copy of the scene in which every visual robot geom collides with everything (and the objects' collision geoms "
        "with the arms), replays the trajectory kinematically and runs collision detection on each frame. Contacts "
        "between a body and its parent or grandparent, and between the links of one finger, are ignored (they touch by "
        "construction). It fails on any interpenetration deeper than `tol_m` and lists the worst offending pair per "
        "category with its frame and label: hand-hand, hand-arm, arm-arm, arm-table, arm-object, hand-table, and "
        "hand-object (the hand's visual meshes against the objects' collision geoms, tolerance `hand_object_tol_m`: the "
        "physics touches objects with the hand's collision shapes, and what the camera shows must not sink into them)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "trajectory", "type": "str | list", "description": "Episode npz (qpos, labels) or a qpos array."},
        {"name": "tol_m", "type": "float", "description": "Allowed interpenetration. Default 0.003."},
        {"name": "hand_object_tol_m", "type": "float | None", "description": "Allowed visible hand-object interpenetration (the pads' soft contact plus the finger meshes' 2 mm margin over their capsules). Default 0.004; None skips hand-object."},
        {"name": "stride", "type": "int", "description": "Frame stride. Default 1; raw integrator substeps are used when recorded."},
        {"name": "baseline", "type": "str | None", "description": "Deprecated: relative-to-baseline acceptance is rejected; absolute limits apply."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, failed [categories], worst {category: {frame, label, depth_mm, geoms}}, n_frames_checked."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "collision", "self-collision", "arm"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "robot_self_collision_check",
}

import mujoco
import numpy as np

from tool_library.common import episode as E


def _collidable_copy(scene_xml):
    from tool_library.common.handgeom import add_hand_hulls
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    add_hand_hulls(spec)
    for g in spec.geoms:
        body = g.parent.name if g.parent is not None else ""
        if body.startswith("robot_") and g.type != mujoco.mjtGeom.mjGEOM_PLANE:
            if "_collision_hand_" in (g.name or "") or g.group <= 2:
                g.contype = 1; g.conaffinity = 1; g.condim = 1
    # drop the scene's hand pairs (everything collides now)
    for p in list(spec.pairs):
        spec.delete(p)
    for e in list(spec.excludes):
        spec.delete(e)
    return spec.compile()


def related_robot_bodies(model, b1, b2):
    """Only the assembled neighbouring links are exempt, never an entire hand."""
    p1, p2 = int(model.body_parentid[b1]), int(model.body_parentid[b2])
    if b1 == b2 or b1 in (p2, int(model.body_parentid[p2])) or b2 in (p1, int(model.body_parentid[p1])):
        return True
    n1, n2 = model.body(b1).name, model.body(b2).name
    for side in ("left", "right"):
        if not (n1.startswith(f"robot_{side}_") and n2.startswith(f"robot_{side}_")):
            continue
        def mount_part(b, n):
            if "_arm_" in n:
                return n.endswith("gripper") or any(model.body(c).name.endswith("gripper")
                    for c in range(model.nbody) if int(model.body_parentid[c]) == b)
            return "_base_" in n or n.endswith("_hand_C_MC") or n.endswith("_palm")
        if ("_arm_" in n1) != ("_arm_" in n2) and mount_part(b1, n1) and mount_part(b2, n2):
            return True
    return False


def redundant_robot_proxy_contact(model, g1, g2):
    """Conservative pad envelopes overlap at joints even with separated visible meshes.

    Ignore that duplicate robot-robot test only when the same body's actual
    visible geometry is also present in the collidable audit model. Its visual
    pairs are still tested, including every finger against the palm and fingers.
    """
    for geom in (g1, g2):
        if "collision_hand_" not in (model.geom(geom).name or ""):
            continue
        body = int(model.geom_bodyid[geom])
        if any(int(model.geom_bodyid[g]) == body and model.geom_group[g] <= 2 and model.geom_rgba[g, 3] > 0
               and "collision_hand_" not in (model.geom(g).name or "") for g in range(model.ngeom)):
            return True
    return False


def robot_self_collision_check(scene, trajectory, tol_m=0.003, stride=1, hand_object_tol_m=0.004, baseline=None):
    if baseline is not None:
        raise ValueError("baseline-relative collision acceptance is unsupported; every episode must meet the absolute limits")
    model = _collidable_copy(scene)
    from tool_library.verifiers._recording import collision_recording
    trajectory = collision_recording(trajectory)
    if isinstance(trajectory, (str,)):
        ep = E.load_episode(trajectory); Q, labels = ep["qpos"], ep["labels"]
    elif isinstance(trajectory, dict):
        Q, labels = trajectory["qpos"], trajectory["labels"]
    else:
        Q = np.asarray(trajectory, float); labels = [""] * len(Q)
    if Q.ndim != 2 or len(Q) < 2 or not np.isfinite(Q).all() or len(labels) != len(Q):
        raise ValueError("trajectory must contain at least two finite, labelled state frames")
    if Q.shape[1] != model.nq:
        raise ValueError(f"trajectory has {Q.shape[1]} columns, the scene has nq={model.nq}")
    data = mujoco.MjData(model)
    def kind(b):
        n = model.body(b).name
        if not n.startswith("robot_"):
            return "table" if n in ("table", "world") else "object"
        return "arm" if "_arm_" in n else "hand"
    def related(b1, b2):
        return related_robot_bodies(model, b1, b2)
    worst = {}
    n = 0
    for t in range(0, len(Q), max(1, int(stride))):
        data.qpos[:] = Q[t]; mujoco.mj_kinematics(model, data); mujoco.mj_collision(model, data); n += 1
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= -min(tol_m, hand_object_tol_m if hand_object_tol_m is not None else tol_m):
                continue
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            k1, k2 = kind(b1), kind(b2)
            if k1 in ("hand", "arm") and k2 in ("hand", "arm") and redundant_robot_proxy_contact(model, c.geom1, c.geom2):
                continue
            if "object" in (k1, k2) and "table" in (k1, k2) or (k1 == k2 == "object") or (k1 == k2 == "table"):
                continue
            if k1 == "hand" and k2 == "object" or k2 == "hand" and k1 == "object":
                # touching is the task; SINKING into the object is not, and the camera sees the visual mesh
                if hand_object_tol_m is None or c.dist >= -hand_object_tol_m:
                    continue
            elif c.dist >= -tol_m:
                continue
            if related(b1, b2):
                continue
            cat = "-".join(sorted([k1, k2]))
            depth = -float(c.dist)
            if cat not in worst or depth > worst[cat]["depth_m"]:
                worst[cat] = {"frame": t, "label": labels[t], "depth_mm": round(1000 * depth, 1), "depth_m": depth,
                              "geoms": [model.geom(c.geom1).name, model.geom(c.geom2).name]}
    for v in worst.values():
        v.pop("depth_m", None)
    return {"ok": not worst, "failed": sorted(worst), "worst": worst, "n_frames_checked": n}
