"""Articulation check: an object that must move has a joint that moves it."""
from __future__ import annotations

TOOL = {
    "name": "articulation_check",
    "title": "Articulation check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "An object declared articulated is built with at least one hinge or slide "
               "joint that has a usable range, not as a rigid free body.",
    "description": (
        "Stage 1 decides what moves; stage 2 must build it that way. A drawer added as a rigid "
        "mesh passes every geometric check and still cannot open. For each named object the "
        "verifier collects the bodies that belong to it (the body itself and everything whose "
        "name starts with its slug) and looks for hinge or slide joints on them. It fails when "
        "there is none, when the object is a free rigid body instead, or when the joint has a "
        "degenerate range (`min_travel`: 3 mm for a slide, 5 degrees for a hinge, summed over the object's joints of a kind so a sprung contact split into stages counts). It also "
        "drives each joint across its range and reports whether the moving part actually "
        "displaces, which catches a joint attached to the wrong link."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "objects", "type": "list[str]", "description": "Names of the objects that must be articulated."},
        {"name": "min_travel", "type": "float", "description": "Smallest joint range accepted: metres for slides, radians for hinges. Default 0.003 m / 0.087 rad."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; per object: joints [{joint, type, range, travel, moves_part_m}], "
                        "verdict in {articulated, no mechanism, rigid free body, degenerate range}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "articulated", "joint", "mechanism"],
    "used_in": ["Place Toy in Drawer", "Giftbox Packing", "Open the Bottle", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "articulation_check",
}

import numpy as np

from tool_library.common import scene as S


def _object_bodies(model, name):
    s = S.slug(name)
    out = set()
    for b in range(1, model.nbody):
        n = model.body(b).name
        if n == name or n == s or n.startswith(f"{s}_") or n.startswith(f"{name}_"):
            out.add(b)
    bid = S.resolve_body(model, name)
    if bid >= 0:
        out |= S.subtree(model, bid)
    return out


def check_articulation(scene, objects, min_travel=None):
    model = S.load_model(scene)
    per, ok = {}, True
    for name in objects:
        bodies = _object_bodies(model, name)
        rec = {"bodies": sorted(model.body(b).name for b in bodies), "joints": []}
        if not bodies:
            rec["verdict"] = "not in scene"
            per[name], ok = rec, False
            continue
        joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in bodies
                  and int(model.jnt_type[j]) in (S.HINGE, S.SLIDE)]
        if not joints:
            free = any(S.has_free_joint(model, b) for b in bodies)
            rec["verdict"] = "rigid free body" if free else "no mechanism"
            per[name], ok = rec, False
            continue
        good = False
        total = {"slide": 0.0, "hinge": 0.0}
        any_moves = False
        for j in joints:
            slide = int(model.jnt_type[j]) == S.SLIDE
            lo, hi = (float(x) for x in model.jnt_range[j])
            limited = bool(model.jnt_limited[j])
            travel = hi - lo if limited else float("inf")
            floor = (0.003 if slide else np.radians(5.0)) if min_travel is None else min_travel
            # drive the joint across its range and measure how far the part's
            # geometry moves (its body origin may sit on the hinge axis)
            b = int(model.jnt_bodyid[j])
            q = int(model.jnt_qposadr[j])
            part = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in S.subtree(model, b)]
            d = S.forward(model)
            p0 = d.geom_xpos[part].copy() if part else d.xpos[b][None].copy()
            d.qpos[q] = (lo if abs(d.qpos[q] - hi) < abs(d.qpos[q] - lo) else hi) if limited else d.qpos[q] + (0.1 if slide else 1.0)
            S.mujoco.mj_forward(model, d)
            p1 = d.geom_xpos[part] if part else d.xpos[b][None]
            moved = float(np.linalg.norm(p1 - p0, axis=1).max())
            entry = {"joint": model.joint(j).name, "type": "slide" if slide else "hinge",
                     "range": [lo, hi] if limited else None,
                     "travel": None if travel == float("inf") else round(travel, 4),
                     "moves_part_m": round(moved, 4)}
            entry["usable"] = bool(travel >= floor and moved > 1e-6)
            good |= entry["usable"]
            if moved > 1e-6:
                any_moves = True
                total["slide" if slide else "hinge"] += min(travel, 1e3)
            rec["joints"].append(entry)
        # several short joints in series (a sprung contact split into stages) add up to one mechanism
        if not good and any_moves:
            floor_s = 0.003 if min_travel is None else min_travel
            floor_h = np.radians(5.0) if min_travel is None else min_travel
            good = total["slide"] >= floor_s or total["hinge"] >= floor_h
        rec["total_travel"] = {k: round(v, 4) for k, v in total.items()}
        rec["verdict"] = "articulated" if good else "degenerate range"
        ok &= good
        per[name] = rec
    return {"ok": bool(ok), "objects": per}


articulation_check = check_articulation   # the library calls every tool by its own name
