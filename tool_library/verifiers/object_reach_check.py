"""Are the scene's objects where an arm can work on them? A placement check for Stage 2."""
from __future__ import annotations

TOOL = {
    "name": "object_reach_check",
    "title": "Object reach check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "For every free object in a bench scene, solves whether at least one station's arm can put a palm-down hand "
               "over it (and a fingers-in hand beside it) at grasp height; pose probes and conservative contact-distance bounds are reported separately, so a "
               "known impossible placement can be caught before Stage 3.",
    "description": (
        "The bench arms reach a band in front of their stations; a video whose objects lie farther out (another table, a "
        "camera farther back than the calibration assumed, a mis-scaled reconstruction) gives a scene every grasp of which "
        "fails the final arm-reach audit. This verifier takes the scene and, per free object root (or the bodies in "
        "`objects`), the object's horizontal position and height; for each station it solves the arm for a palm-down "
        "hand `hover_m` above the object's top with the fingers forward, and a level hand beside it with the fingers "
        "pointing at it, reporting the smaller residual. An object is reachable when one station gets within `pos_tol_m` "
        "/ `tilt_tol_deg`. A failed finite probe is inconclusive; only a violated conservative kinematic bound proves impossibility. A sampled pose witness may still need a turned "
        "pose; an unreachable one needs the scene (or the setup's stations) changed."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "objects", "type": "list[str] | None", "description": "Body names to check; default every free object root outside the robot."},
        {"name": "hover_m", "type": "float", "description": "Height of the palm-down wrist above the object's top. Default 0.06."},
        {"name": "pos_tol_m", "type": "float", "description": "Default 0.02."},
        {"name": "tilt_tol_deg", "type": "float", "description": "Default 20."},
        {"name": "out_json", "type": "str | None", "description": "Report."},
        {"name": "required_sides", "type": "dict | None", "description": "Optional body -> acting arm names. Every listed arm must reach that body; otherwise any arm suffices."},
        {"name": "requirement_mode", "type": "str", "description": "probe (default): require a sampled hand-pose IK witness; necessary: reject only objects outside a conservative kinematic contact bound, report unsuccessful pose probes as inconclusive. Neither mode certifies collision-free manipulation."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, verdict, objects {name: {pos, reachable, best_side, best_orientation, residual_mm, residual_deg, per_side {...}}}, unreachable [names]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage2", "reach", "placement", "workspace", "scene"],
    "used_in": ["Place Cup in Bowl", "Place Toy in Drawer", "Giftbox Packing", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "object_reach_check",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E


def _contact_bound(model, data, part, side):
    """Triangle-inequality upper bound for any hand contact, not a sampled IK claim."""
    from tool_library.common import armhand as AH
    base = model.body(f'robot_{side}_arm_root').id
    def length_to(body, stop):
        distance = 0.
        while body != stop and body > 0:
            distance += float(np.linalg.norm(model.body_pos[body]))
            for j in range(int(model.body_jntadr[body]), int(model.body_jntadr[body] + model.body_jntnum[body])):
                distance += 2 * float(np.linalg.norm(model.jnt_pos[j]))
                if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                    if not model.jnt_limited[j]:
                        return float('inf')
                    distance += float(np.abs(model.jnt_range[j]).max())
            body = int(model.body_parentid[body])
        return distance if body == stop else float('inf')
    arm = length_to(part['gripper'], base)
    ee = part['ee_body']
    hand = max((length_to(int(model.geom_bodyid[g]), ee) + float(np.linalg.norm(model.geom_pos[g])) + float(model.geom_rbound[g])
                for g in range(model.ngeom) if int(model.geom_bodyid[g]) in E.subtree_bodies(model, ee)), default=0.)
    mount = float(np.linalg.norm(AH.ArmSide(model, side).p_m))
    return data.xpos[base].copy(), arm + mount + hand


def object_reach_check(scene, objects=None, hover_m=0.06, pos_tol_m=0.02, tilt_tol_deg=20.0, out_json=None, required_sides=None, requirement_mode='probe'):
    if requirement_mode not in ('probe', 'necessary'):
        raise ValueError('requirement_mode must be probe or necessary')
    from tool_library.skills.optimize_pose_for_reach import _reach_solver
    from tool_library.verifiers.arm_workspace_map import named_orientation
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    parts = E.robot_parts(model)
    sides = [s for s, pt in parts.items() if pt.get("gripper", -1) >= 0]
    if objects is None:
        objects = [model.body(b).name for b, _ in E.object_free_joints(model).items()] if isinstance(E.object_free_joints(model), dict) else []
        if not objects:
            objects = [model.body(b).name for b in range(1, model.nbody) if int(model.body_parentid[b]) == 0 and not model.body(b).name.startswith(("robot_", "table", "world"))]
    if not sides:                                  # a scene without stations (before attach_arms): nothing to judge
        out = {"ok": False, "skipped": "no arm stations in the scene", "objects": {}, "unreachable": [],
               "verdict": "REACH NOT CHECKED: the scene has no arm stations (attach the setup's arms first)"}
        if out_json:
            Path(out_json).write_text(json.dumps(out, indent=1)); out["out_json"] = str(out_json)
        return out
    solvers = {s: _reach_solver(model, s) for s in sides}
    bounds = {s: _contact_bound(model, data, parts[s], s) for s in sides}
    rep, bad = {}, []
    for name in objects:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            rep[name] = {"error": "not in scene"}; bad.append(name); continue
        sub = E.subtree_bodies(model, bid)
        gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub and (model.geom_contype[g] | model.geom_conaffinity[g])]
        from tool_library.common import scene_assert as SA
        top = max((float(SA.geom_world_points(model, data, g)[:, 2].max()) for g in gids), default=float(data.xpos[bid][2]))
        c = data.xpos[bid].copy()
        per, best, feasible = {}, None, set()
        for s in sides:
            cands = {"palm_down": (np.array([c[0], c[1], top + hover_m]), named_orientation(s, "palm_down")),
                     "fingers_in": (np.array([c[0] + (0.17 if s == "right" else -0.17), c[1], max(top * 0.5, 0.04)]), named_orientation(s, "fingers_in"))}
            for oname, (p, R) in cands.items():
                e = solvers[s](p, R)
                r = {"residual_mm": round(1000 * e[0], 1), "residual_deg": round(float(np.degrees(e[1])), 1)}
                per[f"{s}:{oname}"] = r
                good = e[0] <= pos_tol_m and e[1] <= np.radians(tilt_tol_deg)
                if good:
                    feasible.add(s)
                rank = (not good, e[0] + .05 * e[1])
                if best is None or rank < (not (best[0] <= pos_tol_m and best[1] <= np.radians(tilt_tol_deg)), best[0] + .05 * best[1]):
                    best = (e[0], e[1], s, oname)
        required = set((required_sides or {}).get(name) or [])
        if required - {'left', 'right'}:
            raise ValueError(f"{name}: required_sides must contain left/right")
        witness = required.issubset(feasible) if required else bool(feasible)
        possible = {s for s, (base, radius) in bounds.items()
                    if any(float(np.linalg.norm(data.geom_xpos[g] - base)) - float(model.geom_rbound[g]) <= radius + pos_tol_m for g in gids)}
        necessary = required.issubset(possible) if required else bool(possible)
        ok = witness if requirement_mode == 'probe' else necessary
        rep[name] = {"pos": [round(float(v), 3) for v in c], "top_z": round(top, 3), "reachable": bool(ok), "best_side": best[2] if best else None,
                     "best_orientation": best[3] if best else None, "residual_mm": round(1000 * best[0], 1) if best else None,
                     "residual_deg": round(float(np.degrees(best[1])), 1) if best else None, "per_side": per,
                     "required_sides": sorted(required), "reachable_sides": sorted(feasible),
                     "pose_witness": bool(witness), "necessary_bound_passed": bool(necessary),
                     "status": 'pose_witness' if witness else 'inconclusive' if necessary else 'outside_kinematic_bound'}
        if not ok:
            bad.append(name)
    def _short(n):
        r = rep[n]
        if "pos" not in r:
            return f"{n} ({r.get('error', 'not measured')})"
        miss = f", {r['residual_mm']:.0f} mm short" if r.get("residual_mm") is not None else ""
        return f"{n} at y={r['pos'][1]:.2f} m ({r.get('best_side') or 'no'} arm{miss})"
    verdict = ("WORKSPACE PREREQUISITE OK; Stage 3 must still verify actual grasp poses and paths" if not bad else
               ("NO SAMPLED POSE WITNESS: " if requirement_mode == 'probe' else "OUTSIDE NECESSARY WORKSPACE BOUND: ")
               + ", ".join(_short(n) for n in bad) + "; check calibration and scale. Finite orientation probes cannot prove impossibility.")
    warnings = [f"{n}: sampled hand poses failed; contact feasibility is inconclusive, not certified unreachable"
                for n, row in rep.items() if row.get('status') == 'inconclusive']
    out = {"ok": not bad, "verdict": verdict, "objects": rep, "unreachable": bad,
           "warnings": warnings, "scope": requirement_mode}
    if out_json:
        Path(out_json).write_text(json.dumps(out, indent=1)); out["out_json"] = str(out_json)
    return out
