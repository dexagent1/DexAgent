"""Free joint object check: does each object that should move have exactly one free root, and each static one none?"""
from __future__ import annotations

TOOL = {
    "name": "free_joint_object_check",
    "title": "Free joint object check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Each object meant to move has exactly one free joint, on its root body; extra free bodies of an object are tied to it by "
               "an equality unless they are the declared separate pieces of a group object (`parts`); no free joint is nested below a "
               "top-level body; static objects, the table and the robot have none.",
    "description": (
        "An object that should be picked up but whose root has no free joint is welded to the world, and Stage 3 can never "
        "move it; a part that is a separate free body without an equality to its object (a cap that should be screwed on, a "
        "lid that should be hinged) lies loose and falls off; a free joint below another body does not compile (MuJoCo: 'free "
        "joint can only be used on top level', naming only the first), which an attach into the wrong parent produces; a "
        "fixture (a rack, a shelf, the table) with a free joint slides when touched. The check reads the scene through MjSpec "
        "so every nested free joint is listed even when the scene does not compile, then compiles it and, per object (from a "
        "scene_recon.json, or the top-level non-robot bodies), counts the free joints among its bodies. `moving` objects "
        "(default: the recon's rigid objects whose Stage 1 role is 'manipulated' or 'tool', read through the recon's "
        "scene_semantic link) must have one free joint on their root body; every further free body of an object must be "
        "connected to another of its bodies by an equality constraint, except the pieces a group object declares in its "
        "scene_recon.json entry under `parts` (a tower of loose bars, a pile of cards: Stage 1 `pieces`), which are separate "
        "free bodies by design; when Stage 1 says the pieces are loose, every declared part must have its own free joint, and "
        "a declared part that is not a body of the scene is reported. Undeclared extra free bodies still fail. `static` "
        "objects, the table and robot bodies must have no free joint."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "objects", "type": "str | dict | list | None", "description": "scene_recon.json path or dict, or a list of object ids / dicts; default: top-level non-robot bodies."},
        {"name": "moving", "type": "list[str] | None", "description": "Object ids that must move freely; default from the recon's types and Stage 1 roles (none without a recon)."},
        {"name": "static", "type": "list[str] | None", "description": "Object ids that must not have a free joint. Default []."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; nested_free: [{joint, body, parent}]; objects: {id: {root, root_free, n_free, free_bodies, untied_free_bodies, "
                        "parts, moving, static}}; worst."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "free joint", "objects", "static", "attach"],
    "status": "tested",
    "entrypoint": "free_joint_object_check",
    "cli": "python -m tool_library.verifiers.free_joint_object_check scene.xml [scene_recon.json]",
}

import json
import sys
from pathlib import Path

import mujoco

from tool_library.common import compile_checks as C

_MOVING_ROLES = ("manipulated", "tool")


def _nested_free(scene):
    """[{joint, body, parent}] for free joints whose body is not a child of the world, read through MjSpec (no compile)."""
    if isinstance(scene, mujoco.MjModel):
        return []
    p = C.scene_path(scene)
    try:
        with C.capture_mj_warnings():
            spec = mujoco.MjSpec.from_file(str(p)) if p is not None else mujoco.MjSpec.from_string(str(scene))
    except Exception:
        return []
    out = []
    for j in spec.joints:
        if j.type != mujoco.mjtJoint.mjJNT_FREE:
            continue
        b = j.parent
        par = b.parent if b is not None else None
        if par is not None and par.name != "world" and par != spec.worldbody:
            out.append({"joint": j.name or "(unnamed)", "body": b.name or "(unnamed)", "parent": par.name or "(unnamed)"})
    return out


def _stage1_objects(objects):
    """{id: Stage 1 object} through the recon's scene_semantic link ({} without one)."""
    _, base = C.load_objects_arg(objects)
    raw = objects
    if isinstance(objects, (str, Path)):
        raw = json.loads(Path(objects).read_text())
    link = raw.get("scene_semantic") if isinstance(raw, dict) else None
    if link and base is not None and (base / link).exists():
        sem = json.loads((base / link).read_text())
        return {o.get("id"): o for o in sem.get("objects", []) if isinstance(o, dict)}
    return {}


def _default_moving(objects):
    objs, base = C.load_objects_arg(objects)
    if not objs:
        return []
    roles = {k: o.get("role") for k, o in _stage1_objects(objects).items()}
    return [o["id"] for o in objs if (o.get("type") in (None, "rigid")) and roles.get(o["id"]) in _MOVING_ROLES]


def free_joint_object_check(scene, objects=None, moving=None, static=None, robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    nested = _nested_free(scene)
    issues = []
    if nested:
        issues.append(f"{len(nested)} free joint(s) sit below another body ({[(n['joint'], n['body'], n['parent']) for n in nested][:4]}): MuJoCo "
                      "allows a free joint only on a top-level body; attach the object to the world (worldbody or a frame), not under "
                      "another body, or replace the joint by the hinge/slide the part needs")
    model, err = C.compile_scene(scene)
    if model is None:
        rep = C.compile_failure("a free joint on a non-top-level body or more than 6 dofs in one body: see nested_free", err, objects={})
        rep["issues"] = issues + rep["issues"]
        rep["nested_free"] = nested
        return rep
    groups = C.object_groups(model, objects, robot_prefix, by_prefix=False)
    moving = list(_default_moving(objects) if moving is None else moving)
    static = list(static or [])
    unknown = [x for x in moving + static if x not in groups]
    if unknown:
        issues.append(f"moving/static name objects that are not in the scene's objects: {unknown}; objects: {sorted(groups)}")
    free_of = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == C.FREE:
            free_of.setdefault(int(model.jnt_bodyid[j]), []).append(j)
    eq_pairs = set()
    for e in range(model.neq):
        if int(model.eq_type[e]) in (C.WELD, C.CONNECT) and int(model.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_BODY):
            a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            eq_pairs |= {(a, b), (b, a)}
    rows = {}
    from tool_library.common.piece_groups import stage1_pieces
    s1 = _stage1_objects(objects) if objects is not None else {}
    declared = {str(o["id"]): list(o.get("parts") or []) for o in (C.load_objects_arg(objects)[0] or [])} if objects is not None else {}
    for oid, g in groups.items():
        root = g["root"]
        fb = sorted(b for b in g["bodies"] if b in free_of)
        root_free = root in free_of
        untied = []
        primary = root if root_free else (fb[0] if fb else None)
        # a group object's declared pieces are separate free bodies by design (piece_groups): never 'untied'
        parts = set(g.get("parts") or [])
        missing_parts = [n for n in declared.get(oid, []) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(n)) < 0]
        if missing_parts:
            issues.append(f"object '{oid}': declared parts {missing_parts[:4]} are not bodies of the scene; list the root body of each piece")
        pieces = stage1_pieces(s1.get(oid))
        if parts and pieces and pieces["loose"]:
            fixed_parts = [C.body_name(model, b) for b in sorted(parts) if b not in free_of]
            if fixed_parts:
                issues.append(f"object '{oid}': Stage 1 says its pieces are loose, but declared parts {fixed_parts[:4]} have no free "
                              "joint of their own: they cannot be taken off or topple; give each piece a <freejoint/>")
        for b in fb:
            if b == primary or b in parts:
                continue
            # a further free part must be held by an equality to some other body of the same object (or to the world/robot
            # when the object is meant to be attached); otherwise it lies loose
            if not any((b, x) in eq_pairs for x in g["bodies"] if x != b) and not any((b, x) in eq_pairs for x in (0,)):
                untied.append(C.body_name(model, b))
        row = {"root": C.body_name(model, root) if root >= 0 else None, "root_free": root_free, "n_free": len(fb),
               "free_bodies": [C.body_name(model, b) for b in fb], "untied_free_bodies": untied,
               "parts": [C.body_name(model, b) for b in sorted(parts)], "moving": oid in moving, "static": oid in static}
        rows[oid] = row
        if oid in moving and not root_free:
            where = f"; its free joint is on '{row['free_bodies'][0]}' instead" if fb else ""
            issues.append(f"object '{oid}' is meant to move but its root body '{row['root']}' has no free joint{where}: it is fixed to the "
                          "world and Stage 3 can never move it; give the root a <freejoint/>")
        if untied:
            issues.append(f"object '{oid}': parts {untied[:4]} are separate free bodies tied to nothing of the object: they lie loose and fall "
                          "off; weld them (or the right joint) to the object, or list them as objects of their own; when they are the "
                          "object's separate loose pieces (Stage 1 `pieces`: a tower of bars, a pile of cards), declare every piece's "
                          "root body in its scene_recon.json entry under `parts`")
        if oid in static and fb:
            issues.append(f"object '{oid}' is static but has free joints on {row['free_bodies'][:4]}: it slides when touched; drop the free joint")
    for b, js in free_of.items():
        nm = C.body_name(model, b)
        if nm in C.SUPPORT_BODIES or C.is_robot(nm, robot_prefix):
            issues.append(f"body '{nm}' ({'robot' if C.is_robot(nm, robot_prefix) else 'support'}) has a free joint: it must be fixed")
    first = next((r for r in rows.values() if (r["moving"] and not r["root_free"]) or r["untied_free_bodies"]), None)
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "nested_free": nested, "objects": rows,
                       "moving": moving, "static": static, "worst": first})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = free_joint_object_check(argv[0], objects=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
