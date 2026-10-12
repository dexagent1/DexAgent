"""Hand contact pair mask check: the hand touches exactly the object geoms that are meant to be touched."""
from __future__ import annotations

TOOL = {
    "name": "hand_contact_pair_mask_check",
    "title": "Hand contact pair mask check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "The hands collide only through explicit contact pairs; checks that the pairs a scene has, plus the ones the Stage 3 executor will add, respect each object geom's own contype/conaffinity intent (a magnet shell meant only for ring-ring contact must not become solid to the hand) and that every solid object geom gets a pair with each hand.",
    "description": (
        "The Sharpa hands ship with contype/conaffinity 0 and touch objects only through <pair>s: the executor "
        "(episode.build_execution_model) pairs every hand collision geom with every object geom whose contype or "
        "conaffinity meets the hand's contact bits (hand.hand_may_touch with HAND_CONTACT_BITS, MuJoCo's own rule), "
        "unless the geom or its class opts out with the suffix 'nohand'; it names unnamed geoms itself, so they are "
        "paired too. Two failures follow. A geom whose bits do not include the bits the table and arm use "
        "(the 'solid' bits, read from the table's collision geom, default 1) is meant for special contacts only, e.g. "
        "the EgoDex rings' magnet shell with its own bit; a hand pair makes that invisible shell solid to the fingers, "
        "which then grip a sphere around the ring instead of the ring. And a solid object geom that gets no pair "
        "(its bits miss the hand's, or it opts out) lets the hand pass straight through it. With include_executor_pairs "
        "(default) the executor's rule is applied on top of the scene's explicit pairs; with False only the explicit pairs count and every "
        "solid object geom must be paired with every collision geom of each hand. An unpaired solid geom that lies "
        "inside the paired geoms of its own body (within cover_tol_m: the support feet of a battery inside its can) "
        "is covered and needs no pair. Hand geoms are those whose name contains hand_geom_pattern."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "include_executor_pairs", "type": "bool", "description": "Add the pairs build_execution_model will add (object geoms the hand's contact bits meet and that do not opt out with 'nohand', x the hand's collision geoms). Default True."},
        {"name": "hand_geom_pattern", "type": "str", "description": "Substring naming the hands' collision geoms. Default '_collision_hand_'."},
        {"name": "solid_bits", "type": "int | None", "description": "contype/conaffinity bits of ordinary solid contact. Default: the table geom's bits (1 without a table)."},
        {"name": "sides", "type": "list[str] | None", "description": "Hands that must reach the objects. Default: every hand found."},
        {"name": "cover_tol_m", "type": "float", "description": "An unpaired solid geom lying within this distance inside the paired geoms of its body (support feet inside a can) needs no pair of its own, metres. Default 0.002."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; hands {side: n collision geoms}; solid_bits; pairs {explicit, executor}; special_paired [{geom, bits, source}]; unpaired [{geom, side, why}]; objects_geoms_checked."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "contact", "pair", "hand", "contype", "conaffinity", "magnet", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "hand_contact_pair_mask_check",
}

from pathlib import Path

import mujoco

from tool_library.common import scene_assert as SA


def _geom_classes(scene):
    """{geom name: default class name} from the scene's MjSpec (the compiled model keeps no classes)."""
    if not isinstance(scene, (str, Path)):
        return {}
    try:
        sp = mujoco.MjSpec.from_file(str(scene))
    except Exception:
        return {}
    return {g.name: (g.classname.name if g.classname is not None else "") for g in sp.geoms if g.name}


def hand_contact_pair_mask_check(scene, include_executor_pairs=True, hand_geom_pattern="_collision_hand_", solid_bits=None, sides=None,
                                 cover_tol_m=0.002):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    robot = SA.robot_body_ids(model, st["model"])
    table = SA.table_body_ids(model, st["cfg"])
    hands = {}
    for g in range(model.ngeom):
        nm = model.geom(g).name or ""
        if hand_geom_pattern in nm:
            side = "left" if f"{hand_geom_pattern}left" in nm else "right" if f"{hand_geom_pattern}right" in nm else "hand"
            hands.setdefault(side, set()).add(g)
    if sides is not None:
        unknown = [s for s in sides if s not in hands]
        if unknown:
            raise ValueError(f"sides {unknown}: the scene's hands are {sorted(hands)}")
        hands = {s: hands[s] for s in sides}
    if not hands:
        return SA.report([f"no hand collision geoms (names containing '{hand_geom_pattern}') in the scene; is the robot there?"], hands={})
    tbits = 0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) in table and SA.geom_collidable(model, g):
            tbits |= int(model.geom_contype[g]) | int(model.geom_conaffinity[g])
    bits = int(solid_bits) if solid_bits is not None else (tbits or 1)
    obj_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) not in robot and int(model.geom_bodyid[g]) not in table
                 and int(model.geom_bodyid[g]) != 0 and SA.geom_collidable(model, g)]
    all_hand = set().union(*hands.values())
    paired = {}                                             # object geom -> {side: set(sources)}
    n_exp = 0
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        for h, o in ((g1, g2), (g2, g1)):
            if h in all_hand and int(model.geom_bodyid[o]) not in robot:
                side = next(s for s, v in hands.items() if h in v)
                paired.setdefault(o, {}).setdefault(side, {"explicit": set(), "executor": False})["explicit"].add(h)
                n_exp += 1
    n_exec = 0
    if include_executor_pairs:
        # build_execution_model's rule: an object geom the hand's contact bits meet, unless it opts out ('nohand' name or
        # class); it names unnamed solid geoms itself, so a missing name no longer leaves a geom unpaired
        from tool_library.common.hand import HAND_CONTACT_BITS, _opted_out, hand_may_touch
        classes = _geom_classes(scene)
        for o in obj_geoms:
            nm = model.geom(o).name or ""
            if not hand_may_touch(model.geom_contype[o], model.geom_conaffinity[o], HAND_CONTACT_BITS) or _opted_out(nm, classes.get(nm)):
                continue
            for side in hands:
                paired.setdefault(o, {}).setdefault(side, {"explicit": set(), "executor": False})["executor"] = True
                n_exec += len(hands[side])
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def covered(o, side):
        """o lies inside the geoms of its body that this hand does touch."""
        b = int(model.geom_bodyid[o])
        mates = [g for g in obj_geoms if g != o and int(model.geom_bodyid[g]) == b and side in paired.get(g, {})]
        if not mates:
            return False
        P = SA.geom_world_points(model, data, o)
        if not len(P):
            return False
        import numpy as np
        D = np.min(np.stack([SA.signed_distance(model, data, g, P) for g in mates]), axis=0)
        return bool(D.max() <= cover_tol_m)

    special, unpaired = [], []
    for o in obj_geoms:
        ob = int(model.geom_contype[o]) | int(model.geom_conaffinity[o])
        info = paired.get(o, {})
        if not ob & bits:
            if info:
                src = sorted({("explicit" if v["explicit"] else "") or ("executor" if v["executor"] else "") for v in info.values()} - {""})
                special.append({"geom": SA.geom_label(model, o), "body": model.body(int(model.geom_bodyid[o])).name,
                                "bits": [int(model.geom_contype[o]), int(model.geom_conaffinity[o])], "source": src})
            continue
        for side, hg in hands.items():
            v = info.get(side)
            if v is None and covered(o, side):
                continue
            if v is None:
                why = "no pair with this hand (its contact bits miss the hand's, or it opts out)" if include_executor_pairs else "no pair with this hand"
                unpaired.append({"geom": SA.geom_label(model, o), "side": side, "why": why})
            elif not v["executor"] and len(v["explicit"]) < len(hg):
                unpaired.append({"geom": SA.geom_label(model, o), "side": side, "why": f"paired with {len(v['explicit'])} of the hand's {len(hg)} collision geoms"})
    issues = []
    for s in special[:8]:
        issues.append(f"object geom {s['geom']} has contype/conaffinity {s['bits'][0]}/{s['bits'][1]} (not the solid bit {bits}: it is meant only "
                      f"for contacts of its own kind, e.g. a magnet shell) but a hand pair ({'/'.join(s['source'])}) makes it solid to the "
                      f"fingers; keep it out of the hand pairs (name it or its class with the suffix 'nohand', or give it contact bits "
                      f"the hand's do not meet)")
    if len(special) > 8:
        issues.append(f"... and {len(special) - 8} more special-purpose geoms paired with the hand")
    by_geom = {}
    for u in unpaired:
        by_geom.setdefault((u["geom"], u["why"]), []).append(u["side"])
    for (g, why), ss in list(by_geom.items())[:8]:
        issues.append(f"solid object geom {g} has no contact with the {'/'.join(ss)} hand ({why}): the hand passes through it; "
                      + "give it the solid contact bits or add the hand pairs")
    if len(by_geom) > 8:
        issues.append(f"... and {len(by_geom) - 8} more solid object geoms the hands cannot touch")
    return SA.report(issues, hands={s: len(v) for s, v in hands.items()}, solid_bits=bits, pairs={"explicit": n_exp, "executor": n_exec},
                     special_paired=special, unpaired=unpaired, object_geoms_checked=len(obj_geoms))
