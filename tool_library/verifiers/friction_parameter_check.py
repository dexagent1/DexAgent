"""Friction parameter check: no object, table or contact pair is stickier (or more slippery) than real materials."""
from __future__ import annotations

TOOL = {
    "name": "friction_parameter_check",
    "title": "Friction parameter check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Checks the sliding, torsional and rolling friction of every object and table collision geom and of every explicit contact pair the setup does not have against plausible ranges, and the damping and friction loss of object free joints, so a 3.0 friction that makes any grasp hold (or a syrupy free joint that keeps a released object in place) is caught.",
    "description": (
        "MuJoCo combines two geoms' friction by taking the larger, so one sticky geom (friction 3.0) makes every "
        "contact it has hold; torsional and rolling friction are lengths (m): 1.0 means an object cannot spin in a "
        "pinch at all. Rules: each non-robot collidable geom's sliding friction within [min_slide, max_slide] "
        "(default 0.01-2.0; rubber on rubber is about 1-2), torsional at most max_torsional (0.05 m) and rolling at "
        "most max_rolling (0.01 m). Explicit <pair>s that the setup scene also has (the hand-table pairs) are left to "
        "contact_exclusion_audit; other pairs that touch a robot geom are capped at the executor's own hand-object "
        "pair friction (pair_cap, default [1.2, 1.2, 0.02, 0.001, 0.001]: hand.HAND_PAIR_FRICTION as "
        "build_execution_model writes it), pairs between non-robot geoms at the geom limits. Object free joints must have damping at most max_free_damping "
        "and no friction loss (both act on the object against the world, holding it wherever a hand leaves it). The "
        "table geom must keep the setup's friction."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "setup", "type": "str | None", "description": "Setup folder for the reference table and pairs. Default: the repository's setup/."},
        {"name": "min_slide", "type": "float", "description": "Lowest sliding friction accepted. Default 0.01."},
        {"name": "max_slide", "type": "float", "description": "Highest sliding friction accepted. Default 2.0."},
        {"name": "max_torsional", "type": "float", "description": "Highest torsional friction of a geom, metres. Default 0.05."},
        {"name": "max_rolling", "type": "float", "description": "Highest rolling friction of a geom, metres. Default 0.01."},
        {"name": "pair_cap", "type": "list[float]", "description": "Cap on the 5 friction values of robot pairs not in the setup. Default [1.2, 1.2, 0.02, 0.001, 0.001], the executor's own hand-object pair friction."},
        {"name": "max_free_damping", "type": "float", "description": "Highest damping on an object's free joint (N s/m). Default 0.01."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; geoms_checked; pairs_checked; bad_geoms [{geom, body, friction, why}]; bad_pairs; bad_joints; table {reference, scene}; worst (largest sliding friction seen)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "anti-tampering", "friction", "contact", "physics", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "friction_parameter_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA
from tool_library.common.hand import HAND_PAIR_FRICTION

# the hand-object pair friction build_execution_model writes (slide, slide, torsional, rolling, rolling)
EXECUTOR_PAIR = (HAND_PAIR_FRICTION[0], HAND_PAIR_FRICTION[0], HAND_PAIR_FRICTION[1], HAND_PAIR_FRICTION[2], HAND_PAIR_FRICTION[2])


def friction_parameter_check(scene, setup=None, min_slide=0.01, max_slide=2.0, max_torsional=0.05, max_rolling=0.01,
                             pair_cap=EXECUTOR_PAIR, max_free_damping=0.01):
    if not (0 <= min_slide < max_slide):
        raise ValueError(f"min_slide/max_slide must satisfy 0 <= min_slide < max_slide, got {min_slide}, {max_slide}")
    if len(pair_cap) != 5:
        raise ValueError(f"pair_cap needs 5 values (slide1 slide2 torsional roll1 roll2), got {pair_cap}")
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(setup)
    ref, cfg = st["model"], st["cfg"]
    robot = SA.robot_body_ids(model, ref)
    issues, bad_g, bad_p, bad_j = [], [], [], []
    n_g = 0
    worst = 0.0
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in robot or not SA.geom_collidable(model, g):
            continue
        n_g += 1
        f = [float(x) for x in model.geom_friction[g]]
        worst = max(worst, f[0])
        why = []
        if f[0] > max_slide:
            why.append(f"sliding {f[0]:.3g} > {max_slide}")
        if f[0] < min_slide:
            why.append(f"sliding {f[0]:.3g} < {min_slide}")
        if f[1] > max_torsional:
            why.append(f"torsional {f[1]:.3g} m > {max_torsional}")
        if f[2] > max_rolling:
            why.append(f"rolling {f[2]:.3g} m > {max_rolling}")
        if why:
            bad_g.append({"geom": SA.geom_label(model, g), "body": model.body(b).name, "friction": SA.r6(f), "why": why})
    # explicit pairs the setup does not have
    gname = lambda m, g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    ref_pairs = {frozenset({gname(ref, int(ref.pair_geom1[p])), gname(ref, int(ref.pair_geom2[p]))}) for p in range(ref.npair)}
    n_p = 0
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        if frozenset({gname(model, g1), gname(model, g2)}) in ref_pairs:
            continue
        n_p += 1
        f = [float(x) for x in model.pair_friction[p]]
        robot_pair = int(model.geom_bodyid[g1]) in robot or int(model.geom_bodyid[g2]) in robot
        cap = list(pair_cap) if robot_pair else [max_slide, max_slide, max_torsional, max_rolling, max_rolling]
        over = [i for i in range(5) if f[i] > cap[i] + 1e-12]
        low = f[0] < min_slide
        if over or low:
            bad_p.append({"pair": f"{SA.geom_label(model, g1)} - {SA.geom_label(model, g2)}", "friction": SA.r6(f), "cap": cap,
                          "why": (["above the cap " + str(cap)] if over else []) + ([f"sliding below {min_slide}"] if low else [])})
    # object free joints
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(model.jnt_bodyid[j]) in robot:
            continue
        d0 = int(model.jnt_dofadr[j])
        dmp = float(np.max(model.dof_damping[d0:d0 + 6])); fl = float(np.max(model.dof_frictionloss[d0:d0 + 6]))
        if dmp > max_free_damping or fl > 0:
            bad_j.append({"joint": model.joint(j).name, "damping": dmp, "frictionloss": fl})
    # the table keeps the setup's friction
    tab = {}
    for rg in range(ref.ngeom):
        if int(ref.geom_bodyid[rg]) in SA.table_body_ids(ref, cfg) and SA.geom_collidable(ref, rg):
            nm = gname(ref, rg)
            sg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm) if nm else -1
            if sg >= 0:
                tab[nm] = {"reference": SA.r6(ref.geom_friction[rg]), "scene": SA.r6(model.geom_friction[sg])}
                if SA.differs(model.geom_friction[sg], ref.geom_friction[rg], 1e-9, 1e-6):
                    issues.append(f"table geom {nm} friction is {tab[nm]['scene']} but {tab[nm]['reference']} in the setup; restore it")
    for x in bad_g[:10]:
        issues.append(f"geom {x['geom']} ({x['body']}) friction {x['friction']}: {', '.join(x['why'])}; use a plausible friction "
                      f"(sticky friction makes any grasp hold, near-zero friction makes none)")
    if len(bad_g) > 10:
        issues.append(f"... and {len(bad_g) - 10} more geoms with implausible friction")
    for x in bad_p[:8]:
        issues.append(f"contact pair {x['pair']} friction {x['friction']}: {', '.join(x['why'])}; lower it")
    for x in bad_j:
        issues.append(f"object free joint {x['joint']} has damping {x['damping']:.3g} / frictionloss {x['frictionloss']:.3g}: it holds the object "
                      f"against the world; remove them")
    return SA.report(issues, geoms_checked=n_g, pairs_checked=n_p, bad_geoms=bad_g, bad_pairs=bad_p, bad_joints=bad_j, table=tab,
                     worst=round(worst, 4))
