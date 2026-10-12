"""Free body policy check: what moves in the video is free in the scene, what stays put is fixed."""
from __future__ import annotations

TOOL = {
    "name": "free_body_policy_check",
    "title": "Free body policy check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Every Stage 1 object a hand moves (role manipulated or tool) must be a free body in the scene, not welded to the world, the table or the robot and not a mocap body; every object Stage 1 calls static must be fixed; containers and targets may be either and are reported.",
    "description": (
        "A manipulated object built without a free joint (or welded to the world by an active equality, or made a "
        "kinematic mocap body) can never be picked up, and an object parented to a robot body rides the hand "
        "without any grasp; a 'static' object left free can be knocked away and become a support the robot must "
        "balance. Stage 1 objects (scene_semantics.json) are mapped to bodies through scene_recon.json (else by id). "
        "An object is free when its root body or one of its ancestors below the world carries a free joint, and "
        "fixed otherwise; equality welds and connects active at keyframe 0 between the object and the world, the "
        "table, a fixed object or the robot make it held. Rules: role manipulated or tool -> free and not held (a "
        "part attached to a free object by a weld, such as a cap on its bottle's thread, is free when that object "
        "is); role static -> fixed unless static_may_be_free; container/target -> reported only. Deformable "
        "objects are judged on their root body; a deformable whose scene_recon.json entry declares `anchored_to` "
        "(a cord end held under a fixed object) may be welded to that fixed object's body."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "semantic", "type": "str | dict", "description": "scene_semantics.json (Stage 1) with each object's role."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json mapping Stage 1 ids to root bodies."},
        {"name": "static_may_be_free", "type": "bool", "description": "Accept free bodies for role 'static'. Default False."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; objects [{id, role, root, free, held_by, mocap, under_robot, verdict}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "free joint", "fixed", "role", "stage1", "stage2"],
    "status": "tested",
    "entrypoint": "free_body_policy_check",
}

import mujoco

from tool_library.common import scene_assert as SA


def _free_ancestor(model, b):
    while b > 0:
        for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b]) + int(model.body_jntnum[b])):
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return b
        b = int(model.body_parentid[b])
    return -1


def free_body_policy_check(scene, semantic, recon=None, static_may_be_free=False):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    robot = SA.robot_body_ids(model, st["model"])
    table = SA.table_body_ids(model, st["cfg"])
    sem = {str(o.get("id")): o for o in SA.semantic_objects(semantic)}
    if not sem:
        raise ValueError("semantic: no Stage 1 objects found (give scene_semantics.json)")
    mapping = SA.stage1_groups(model, semantic, recon, ref=st["model"], cfg=st["cfg"])
    active = {e for e in range(model.neq) if bool(model.eq_active0[e])}
    welds = []
    for e in active:
        if int(model.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)):
            continue
        b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if hasattr(model, "eq_objtype") and int(model.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE):
            b1, b2 = int(model.site_bodyid[b1]), int(model.site_bodyid[b2])
        welds.append((e, b1, b2))
    anchors = SA.declared_anchors(model, recon, robot, table)
    rows, issues = [], []
    for oid, o in sem.items():
        role = o.get("role")
        if oid not in mapping:
            rows.append({"id": oid, "role": role, "verdict": "simulated in Isaac Sim"})
            continue
        g = mapping.get(oid)
        if g is None or g["root"] < 0:
            rows.append({"id": oid, "role": role, "verdict": "missing"})
            issues.append(f"Stage 1 object {oid} ({role}) has no body in the scene")
            continue
        root = g["root"]
        fa = _free_ancestor(model, root)
        free = fa >= 0
        mocap = int(model.body_mocapid[root]) >= 0
        under_robot = any(b in robot for b in _ancestors(model, root))
        held = []
        for e, b1, b2 in welds:
            for a, other in ((b1, b2), (b2, b1)):
                if a in g["bodies"] and other not in g["bodies"]:
                    o_free = _free_ancestor(model, other) >= 0 if other > 0 else False
                    if other in anchors.get(oid, {}):
                        continue                       # a declared anchor: a cord end held under a fixed object
                    if other == 0 or other in table or other in robot or not o_free:
                        held.append(f"{model.equality(e).name or f'equality {e}'} -> {model.body(other).name if other > 0 else 'world'}")
        verdict = "ok"
        if role in ("manipulated", "tool"):
            if under_robot:
                verdict = "rides the robot"
                issues.append(f"{oid} ({role}) is nested under a robot body: it rides the hand without a grasp; attach it to the world with a free joint")
            elif mocap:
                verdict = "mocap"
                issues.append(f"{oid} ({role}) is a mocap body: it moves only when code moves it, never by contact; give it a free joint")
            elif not free:
                verdict = "fixed"
                issues.append(f"{oid} is {role} in the video (a hand moves it) but its body {model.body(root).name} has no free joint: "
                              f"the hand can never move it; give the asset a <freejoint/> (constructors take free=True)")
            elif held:
                verdict = "held"
                issues.append(f"{oid} is {role} in the video but an active equality holds it at keyframe 0 ({', '.join(held[:3])}); deactivate it")
        elif role == "static" and free and not static_may_be_free:
            verdict = "free but static"
            issues.append(f"{oid} is static in the video (nothing moves it) but its body {model.body(root).name} is free: weld it to the "
                          f"table (no free joint) so a brush of the hand cannot carry it away")
        rows.append({"id": oid, "role": role, "root": model.body(root).name, "free": free, "held_by": held, "mocap": mocap,
                     "anchored_to": {model.body(b).name: r for b, r in anchors.get(oid, {}).items()},
                     "under_robot": under_robot, "verdict": verdict})
    return SA.report(issues, objects=rows)


def _ancestors(model, b):
    out = []
    b = int(model.body_parentid[b])
    while b > 0:
        out.append(b)
        b = int(model.body_parentid[b])
    return out
