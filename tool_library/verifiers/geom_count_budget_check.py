"""Geom count budget check: can a hand touch the scene's objects without each physics step paying for dozens of geoms?"""
from __future__ import annotations

TOOL = {
    "name": "geom_count_budget_check",
    "title": "Geom count budget check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Counts the collision geoms of each object and, more to the point, how many object collision geoms lie within one hand's "
               "reach of each other; a cluster above the budget (72 ring-wall geoms made each hand-scene step 15 ms and grasp search "
               "time out) fails with a simplification to apply.",
    "description": (
        "Every hand collision geom gets an explicit contact pair with every object collision geom in a hand scene and in the "
        "executor, so the cost of a step grows with the number of object geoms a hand can be touching at once: three 25-geom "
        "magnetic rings in a 6 cm cluster made each hand-scene step take 15 ms (552 contacts with the hand among them) and "
        "the grasp searches ran past 15 minutes, while a 100-geom meal box spread over 20 cm stayed under 4 ms. The check "
        "takes the collision geoms of the non-robot objects (an MJCF bench scene, or a hand-scene JSON / dict compiled the "
        "way the grasp tools compile it), counts them per object and by type, and for each geom counts the object geoms "
        "whose bounding sphere reaches into a sphere of `reach_m` (a hand's grasp envelope) around it. The largest such "
        "cluster above `max_local_geoms` fails, naming the objects in it and how to simplify them (fewer wall segments, a "
        "convex decomposition with a handful of hulls, primitives merged); `max_object_geoms` optionally caps each object "
        "too. With `measure=True` it also times mj_forward with a hand at the densest cluster (hand scene) or the paired "
        "execution model (bench scene); the time is reported, never judged, because it depends on the machine."
    ),
    "inputs": [
        {"name": "scene", "type": "str | dict", "description": "MJCF bench scene, or a hand-scene JSON path / dict ({hand_xml, side, objects: [...]})."},
        {"name": "objects", "type": "str | dict | list | None", "description": "scene_recon.json (or list) naming the objects of an MJCF scene; default: its top-level non-robot bodies."},
        {"name": "reach_m", "type": "float", "description": "Radius of a hand's reach around a geom, metres. Default 0.06."},
        {"name": "max_local_geoms", "type": "int", "description": "Largest number of object collision geoms within one reach. Default 48."},
        {"name": "max_object_geoms", "type": "int | None", "description": "Optional cap on one object's collision geoms. Default None."},
        {"name": "measure", "type": "bool", "description": "Also time mj_forward (reported only). Default False."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies. Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; objects: {id: {n_collision_geoms, by_type}}; densest: {count, center_geom, objects: {id: n}, reach_m}; "
                        "n_object_geoms; n_hand_pairs_estimate; timing_ms (when measured); worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "performance", "geoms", "contacts", "hand scene", "budget"],
    "status": "tested",
    "entrypoint": "geom_count_budget_check",
    "cli": "python -m tool_library.verifiers.geom_count_budget_check scene.xml",
}

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

_TYPES = {int(v): k.replace("mjGEOM_", "").lower() for k, v in mujoco.mjtGeom.__members__.items()}


def _load(scene, objects, robot_prefix):
    """(model, data, {object: [gids]}, hand geoms, hand rig or None, kind)."""
    spec = None
    if isinstance(scene, dict):
        spec = scene
    elif isinstance(scene, (str, Path)) and str(scene).lower().endswith(".json"):
        p = Path(scene)
        if not p.exists():
            raise ValueError(f"hand-scene file not found: {p}")
        spec = json.loads(p.read_text())
    if spec is not None:
        if "objects" not in spec or "hand_xml" not in spec:
            raise ValueError("a hand-scene dict needs 'hand_xml' and 'objects'")
        from tool_library.common import hand as H
        with C.capture_mj_warnings():
            model, data, rig, ids = H.open_scene(spec)
        groups = {}
        names = [o["name"] for o in spec["objects"]]
        for g in sorted(ids["obj_gids"]):
            bn = C.body_name(model, int(model.geom_bodyid[g]))
            owner = next((n for n in sorted(names, key=len, reverse=True) if bn == n or bn.startswith(f"{n}_")), bn)
            groups.setdefault(owner, []).append(int(g))
        return model, data, groups, sorted(int(g) for g in ids["hand_gids"]), rig, "hand_scene"
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        return None, err, None, None, None, "mjcf"
    data = C.key_data(model)
    paired = set(int(x) for x in model.pair_geom1) | set(int(x) for x in model.pair_geom2)
    grp = C.object_groups(model, objects, robot_prefix, by_prefix=objects is not None)
    groups, taken = {}, set()
    for oid, g in grp.items():
        gs = [x for x in range(model.ngeom) if int(model.geom_bodyid[x]) in g["bodies"] and x not in taken
              and int(model.geom_type[x]) != C.PLANE and (C.collidable(model, x) or x in paired)]
        taken |= set(gs)
        groups[oid] = gs
    hand = [x for x in range(model.ngeom) if "collision_hand_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, x) or "")]
    return model, data, groups, hand, None, "mjcf"


def geom_count_budget_check(scene, objects=None, reach_m=0.06, max_local_geoms=48, max_object_geoms=None, measure=False,
                            robot_prefix=C.ROBOT_PREFIX):
    if reach_m <= 0 or int(max_local_geoms) < 1:
        raise ValueError("reach_m must be > 0 and max_local_geoms >= 1")
    model, data, groups, hand, rig, kind = _load(scene, objects, robot_prefix)
    if model is None:
        return C.compile_failure("fix the error MuJoCo names", data, objects={})
    issues = []
    per = {}
    for oid, gs in groups.items():
        bt = {}
        for g in gs:
            t = _TYPES.get(int(model.geom_type[g]), str(int(model.geom_type[g])))
            bt[t] = bt.get(t, 0) + 1
        per[oid] = {"n_collision_geoms": len(gs), "by_type": bt}
        if max_object_geoms is not None and len(gs) > int(max_object_geoms):
            issues.append(f"object '{oid}' has {len(gs)} collision geoms (> {int(max_object_geoms)}): {bt}; simplify it")
    all_g = [(g, oid) for oid, gs in groups.items() for g in gs]
    densest = {"count": 0, "center_geom": None, "objects": {}, "reach_m": reach_m}
    if all_g:
        G = np.array([g for g, _ in all_g])
        P = np.asarray(data.geom_xpos[G], float)
        rb = np.asarray(model.geom_rbound[G], float)
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        inside = (D - rb[None, :]) < reach_m
        cnt = inside.sum(axis=1)
        i = int(np.argmax(cnt))
        members = {}
        for j in np.where(inside[i])[0]:
            members[all_g[j][1]] = members.get(all_g[j][1], 0) + 1
        densest = {"count": int(cnt[i]), "center_geom": C.geom_name(model, int(G[i])), "center_object": all_g[i][1],
                   "objects": members, "reach_m": reach_m}
        if cnt[i] > int(max_local_geoms):
            heavy = sorted(members.items(), key=lambda kv: -kv[1])
            tips = []
            for oid, n in heavy[:3]:
                bt = per[oid]["by_type"]
                main = max(bt.items(), key=lambda kv: kv[1])
                tips.append(f"'{oid}' ({n} here, mostly {main[1]} {main[0]}s): " +
                            ("fewer, larger segments (a ring wall of 24 boxes often works as 12-16; re-run the part's own mechanism test after merging: a thin cord escapes a coarse bead)" if main[0] in ("box", "capsule", "cylinder")
                             else "a convex decomposition with a handful of hulls" if main[0] == "mesh"
                             else "merge the small primitives"))
            issues.append(f"{int(cnt[i])} object collision geoms lie within {100 * reach_m:.0f} cm of '{densest['center_geom']}' "
                          f"(> {int(max_local_geoms)}), from {heavy}: a hand there is paired with all of them, so every hand-scene and "
                          f"executor step pays for {int(cnt[i])} x {max(len(hand), 1)} pairs; simplify " + "; ".join(tips))
    timing = None
    if measure:
        timing = _time_forward(model, data, kind, rig, scene, densest, groups)
    return C.jsonable({"ok": not issues, "compiled": True, "kind": kind, "issues": issues, "objects": per, "densest": densest,
                       "n_object_geoms": len(all_g), "n_hand_geoms": len(hand),
                       "n_hand_pairs_estimate": len(all_g) * len(hand) if hand else None, "timing_ms": timing,
                       "worst": densest if issues else None})


def _time_forward(model, data, kind, rig, scene, densest, groups, n=20):
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    if kind == "hand_scene" and rig is not None and densest.get("center_geom"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, densest["center_geom"])
        mujoco.mj_forward(model, d)
        rig.teleport(d, d.geom_xpos[g] + np.array([0.0, -0.03, 0.06]), [np.pi, 0.0, 0.0])
        m_use = model
    elif kind == "mjcf" and C.scene_path(scene) is not None:
        from tool_library.common import episode as E
        with C.capture_mj_warnings():
            m_use, _ = E.build_execution_model(str(scene))
        d = C.key_data(m_use)
    else:
        m_use = model
    with C.capture_mj_warnings():
        mujoco.mj_forward(m_use, d)
        t0 = time.perf_counter()
        for _ in range(n):
            mujoco.mj_forward(m_use, d)
        ms = (time.perf_counter() - t0) / n * 1000.0
    return {"forward_ms": ms, "ncon": int(d.ncon), "npair": int(m_use.npair), "where": "hand at the densest cluster" if kind == "hand_scene"
            else "execution model at the keyframe"}


def main(argv=None):
    rep = geom_count_budget_check((sys.argv[1:] if argv is None else argv)[0], measure=True)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
