"""Name uniqueness check: can every tool find every object part, and only its own, by name?"""
from __future__ import annotations

TOOL = {
    "name": "name_uniqueness_check",
    "title": "Name uniqueness check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Names are unique, every object part is named and claimed by an object of the recon, no object's '<id>_' prefix "
               "captures another asset's bodies, collidable object geoms are named, and the robot namespace holds the robot and only it.",
    "description": (
        "The library finds an object's parts by name: the recon's root body and body list, '<id>_' prefixes, and the library's "
        "resolver (exact, slug, or the single body ending in '_<id>'). Parts that none of these reach are invisible: an "
        "obstacle set built from the root body never saw an open lid or loose tips that were separate bodies, and the executor "
        "gives hand contact pairs only to NAMED collidable object geoms, so an unnamed geom is one the fingers pass through. "
        "The check compiles the scene (a repeated name is MuJoCo's compile error, reported as such) and fails: an unnamed "
        "sub-body of an object; a collidable object geom without a name that stands out of its body's named collision "
        "geometry (one buried inside it, like support feet under a can, is a note); a recon body missing from the scene or a "
        "recon root that does not exist; a body with colliding or visible geometry that no recon object claims (root subtree or listed bodies); an "
        "object that follows the '<id>_' naming whose prefix also matches bodies of an object from another asset (ids "
        "'battery' and 'battery_holder': a colour or silhouette check of the battery gathers the holder too; parts listed as "
        "their own object inside one asset, such as a cap of its bottle, share the prefix by design); an id the resolver maps "
        "to a body outside its object. For the robot it fails a non-robot part that uses the robot prefix and a body welded "
        "to the robot by an active equality without the prefix (an arm named 'arm_left_*' that every robot tool misses). "
        "Without a recon only unnamed parts and the robot namespace are judged. Names shared across element types are notes."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "objects", "type": "str | dict | list | None", "description": "scene_recon.json path or dict, or a list of object ids / {id, root_body, bodies}; default: the top-level non-robot bodies."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of every robot body, joint and geom. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; notes; objects: {id: {root, n_bodies, unnamed, missing_recon_bodies, resolves_to, "
                        "unnamed_collision_geoms}}; unclaimed_bodies; prefix_collisions; robot: {foreign_prefixed, unprefixed_welded}; worst."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "names", "prefix", "objects", "contract"],
    "status": "tested",
    "entrypoint": "name_uniqueness_check",
    "cli": "python -m tool_library.verifiers.name_uniqueness_check scene.xml [scene_recon.json]",
}

import json
import sys
from collections import defaultdict

import mujoco
import numpy as np

from tool_library.common import compile_checks as C
from tool_library.common import scene as S

_TYPES = (("body", mujoco.mjtObj.mjOBJ_BODY, "nbody"), ("joint", mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
          ("geom", mujoco.mjtObj.mjOBJ_GEOM, "ngeom"), ("site", mujoco.mjtObj.mjOBJ_SITE, "nsite"),
          ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"), ("equality", mujoco.mjtObj.mjOBJ_EQUALITY, "neq"),
          ("mesh", mujoco.mjtObj.mjOBJ_MESH, "nmesh"), ("material", mujoco.mjtObj.mjOBJ_MATERIAL, "nmat"),
          ("texture", mujoco.mjtObj.mjOBJ_TEXTURE, "ntex"), ("camera", mujoco.mjtObj.mjOBJ_CAMERA, "ncam"))


def _world_aabb(model, data, g):
    c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    R = data.geom_xmat[g].reshape(3, 3)
    ctr = data.geom_xpos[g] + R @ c
    ext = np.abs(R) @ h
    return ctr - ext, ctr + ext


def _stands_out(model, data, g, tol=0.001):
    """True unless geom g lies (within tol) inside the bounding box of its body's named collidable geoms."""
    b = int(model.geom_bodyid[g])
    named = [x for x in range(model.ngeom) if int(model.geom_bodyid[x]) == b and x != g and C.collidable(model, x)
             and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, x)]
    if not named:
        return True
    lo = np.min([_world_aabb(model, data, x)[0] for x in named], axis=0)
    hi = np.max([_world_aabb(model, data, x)[1] for x in named], axis=0)
    a, b2 = _world_aabb(model, data, g)
    return bool((a < lo - tol).any() or (b2 > hi + tol).any())


def _physical_or_visible(model, g):
    if C.collidable(model, g):
        return True
    alpha = float(model.geom_rgba[g][3])
    if int(model.geom_matid[g]) >= 0 and alpha == 1.0:
        alpha = float(model.mat_rgba[int(model.geom_matid[g])][3])
    return int(model.geom_group[g]) <= 2 and alpha > 0.0


def name_uniqueness_check(scene, objects=None, robot_prefix=C.ROBOT_PREFIX):
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        fix = ("two elements of one type share a name: rename one (attach assets with a distinct prefix)" if "repeated name" in str(err)
               else "fix the error MuJoCo names")
        return C.compile_failure(fix, err, objects={})
    issues, notes = [], []
    # names shared across types are legal in MuJoCo and only ambiguous to a reader
    owners = defaultdict(list)
    for kind, ot, cnt in _TYPES:
        for i in range(int(getattr(model, cnt))):
            nm = mujoco.mj_id2name(model, ot, i)
            if nm:
                owners[nm].append(kind)
    shared = {k: v for k, v in owners.items() if len(set(v)) > 1}
    if shared:
        ex = list(shared.items())[:5]
        notes.append(f"{len(shared)} names are used by more than one element type (legal; e.g. {ex})")
    # what the recon declares: each object's root subtree and listed bodies (tools work from these lists)
    groups = C.object_groups(model, objects, robot_prefix, by_prefix=False)
    by_id = objects is not None             # without a recon, object identity (and so the coverage/prefix rules) is unknown
    data = C.key_data(model)
    names = [C.body_name(model, b) for b in range(model.nbody)]
    rows = {}
    for oid, g in groups.items():
        bodies = sorted(g["bodies"])
        unnamed = [b for b in bodies if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)]
        missing = [bn for bn in g["declared_bodies"] if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(bn)) < 0]
        rb = S.resolve_body(model, oid)
        resolves_to = names[rb] if rb >= 0 else None
        ucol_all = [g2 for g2 in range(model.ngeom) if int(model.geom_bodyid[g2]) in g["bodies"] and C.collidable(model, g2)
                    and not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)]
        # an unnamed geom buried inside its body's named collision geometry (support feet under a can) cannot be reached
        # by a finger before the named geometry is: only the ones that stand out are a hole in the hand's contact pairs
        ucol = [x for x in ucol_all if _stands_out(model, data, x)]
        if len(ucol_all) > len(ucol):
            notes.append(f"object '{oid}': {len(ucol_all) - len(ucol)} unnamed collidable geoms lie within its named collision geometry (harmless)")
        row = {"root": names[g["root"]] if g["root"] >= 0 else None, "n_bodies": len(bodies), "unnamed": [f"body{b}" for b in unnamed],
               "missing_recon_bodies": missing, "resolves_to": resolves_to, "unnamed_collision_geoms": [f"geom{x}" for x in ucol]}
        rows[oid] = row
        if g["root"] < 0:
            issues.append(f"object '{oid}': neither a body named '{oid}' nor its recon root_body exists in the scene, so the object cannot "
                          f"be found by name: rebuild the recon from this scene (root_body = the object's top body)")
        if unnamed:
            issues.append(f"object '{oid}': {len(unnamed)} unnamed sub-bodies ({row['unnamed'][:4]}) are invisible to every name lookup; "
                          f"name them '<asset prefix>_<part>'")
        if missing:
            issues.append(f"object '{oid}': the recon lists bodies {missing[:5]} that are not in the scene: the recon is stale; rebuild it from this scene")
        if g["root"] >= 0 and rb >= 0 and rb not in g["bodies"]:
            issues.append(f"object '{oid}': the library's resolver maps '{oid}' to body '{resolves_to}', which belongs to something else, "
                          f"not to the object (root '{names[g['root']]}'): tools given the id act on the wrong body; rename so the id "
                          "resolves to the object's root")
        elif g["root"] >= 0 and rb < 0:
            notes.append(f"object '{oid}': the library's resolver finds no body for the id itself (tools must be given the root body "
                         f"'{names[g['root']]}')")
        if ucol:
            issues.append(f"object '{oid}': {len(ucol)} collidable geoms have no name ({row['unnamed_collision_geoms'][:4]}): the executor pairs "
                          "only named object geoms with the hands, so the fingers pass through these; name them")
    # coverage: a body with geometry that no object claims is invisible to every tool that works from the recon
    unclaimed = []
    if by_id:
        claimed = set().union(*[g["bodies"] for g in groups.values()]) if groups else set()
        robot_ids0 = C.robot_body_ids(model, robot_prefix)
        for b in range(1, model.nbody):
            if b in claimed or b in robot_ids0 or names[C.top_body(model, b)] in C.SUPPORT_BODIES or C.top_body(model, b) in robot_ids0:
                continue
            # a helper body (an invisible, non-colliding mocap grip) is nobody's part; a body that collides or is drawn is
            if any(int(model.geom_bodyid[x]) == b and int(model.geom_type[x]) != C.PLANE and _physical_or_visible(model, x)
                   for x in range(model.ngeom)):
                unclaimed.append(names[b])
        if unclaimed:
            issues.append(f"bodies {unclaimed[:6]} carry geometry but belong to no object of the recon: every tool working from the recon "
                          "(obstacle sets, clearances, Stage 4 randomization) misses them, the way separate lid and tip bodies of a tip box "
                          "were never tested against; list them under their object's 'bodies'")
    # prefix capture: an id-prefixed object whose '<id>_' also matches another asset's bodies
    collisions = []
    if by_id:
        objs_arg, _ = C.load_objects_arg(objects)
        asset = {o["id"]: (o.get("raw") or {}).get("asset_xml") for o in objs_arg}
        for a in groups:
            pre = f"{a}_"
            own = groups[a]["bodies"]
            if not own or not all(names[x].startswith(pre) or names[x] == a for x in own):
                continue                    # only objects that follow the '<id>_' convention are gathered by it
            for b in groups:
                if b == a or (asset.get(a) and asset.get(a) == asset.get(b)):
                    continue                # parts listed as their own object inside one asset share its prefix by design
                caught = [names[x] for x in groups[b]["bodies"] if names[x].startswith(pre) and x not in own]
                if caught:
                    collisions.append({"prefix_of": a, "captures": b, "bodies": caught[:5]})
                    issues.append(f"object '{a}' is found by the prefix '{pre}', which also matches {len(caught)} bodies of object '{b}' "
                                  f"({caught[:3]}): a prefix lookup for '{a}' (colour, silhouette and articulation checks gather parts so) "
                                  f"takes '{b}' too; attach one of them under a prefix that is not an extension of the other")
    # the robot namespace
    robot_ids = C.robot_body_ids(model, robot_prefix)
    foreign = []
    for oid, g in groups.items():
        foreign += [names[x] for x in g["bodies"] if x in robot_ids]
    unpref = set()
    for e in range(model.neq):
        if int(model.eq_type[e]) not in (C.WELD, C.CONNECT) or int(model.eq_objtype[e]) != int(mujoco.mjtObj.mjOBJ_BODY):
            continue
        b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if not bool(model.eq_active0[e]):
            continue                            # an inactive grasp weld to an object is not a robot part
        for x, y in ((b1, b2), (b2, b1)):
            if y in robot_ids and x not in robot_ids and x > 0:
                top = C.top_body(model, x)
                unpref.add(names[top] if top else names[x])
    if foreign:
        issues.append(f"object bodies {foreign[:5]} start with the robot prefix '{robot_prefix}': every robot tool will treat them as robot links; "
                      "rename them")
    if unpref:
        issues.append(f"bodies {sorted(unpref)[:5]} are welded (active equality) to the robot but do not start with '{robot_prefix}': robot "
                      f"tools (arm joints, robot parts, self-collision) miss them and object tools take them for objects; prefix them '{robot_prefix}'")
    first = next(iter([i for i in issues]), None)
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "notes": notes, "objects": rows,
                       "unclaimed_bodies": unclaimed, "prefix_collisions": collisions, "robot": {"foreign_prefixed": foreign, "unprefixed_welded": sorted(unpref)},
                       "worst": first})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = name_uniqueness_check(argv[0], objects=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
