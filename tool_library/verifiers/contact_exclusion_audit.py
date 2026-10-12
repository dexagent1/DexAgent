"""Contact exclusion audit: nothing in the scene switches off contact between the robot and the objects or the table."""
from __future__ import annotations

TOOL = {
    "name": "contact_exclusion_audit",
    "title": "Contact exclusion audit",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Lists every way a scene can stop the robot from touching what it should touch (an <exclude> between a robot body and an object or the table, a missing or softened hand-table pair, an explicit robot pair with a gap or spongy solver parameters, object or table collision bits the arm cannot see, an object parented to or welded to a robot body) and fails on any.",
    "description": (
        "Contact is how DexAgent's objects move: a scene in which the hand passes through an object or the forearm "
        "through the table produces trajectories that look fine and cannot happen. The audit reads the compiled scene "
        "and the setup's reference (default: the repository's setup/): (1) every <exclude> body pair, failing when "
        "one side is a robot body and the other an object or the table, and when a robot-robot exclude is not in the "
        "reference; (2) the reference's explicit robot-table <pair>s (the hands collide only through pairs), which "
        "must all be present with the same condim, friction, solref, solimp, margin and gap; (3) every explicit pair "
        "between a robot geom and a non-robot geom, failing on a gap that exceeds the margin by more than gap_tol_m "
        "(penetration without force), a solref time constant above max_timeconst_s or a direct stiffness below "
        "min_stiffness, an impedance dmax below min_dmax, or condim 1 (frictionless); (4) the contype/conaffinity "
        "bits: each object geom that collides with the table must also collide with the robot arm's collision geoms, "
        "and the table geom must keep the reference's bits; (5) an object body whose parent is a robot body (MuJoCo "
        "filters parent-child contact and the object rides the hand) and an equality constraint between a robot body "
        "and an object (a glued object). Global contact switches (option flags) are physics_options_check's job."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "setup", "type": "str | None", "description": "Setup folder (or its setup.yaml / scene.xml). Default: the repository's setup/."},
        {"name": "gap_tol_m", "type": "float", "description": "Largest allowed gap - margin of a robot pair, metres. Default 0.001."},
        {"name": "max_timeconst_s", "type": "float", "description": "Largest allowed solref time constant of a robot pair. Default 0.1."},
        {"name": "min_stiffness", "type": "float", "description": "Smallest allowed direct stiffness (negative solref) of a robot pair, N/m. Default 1000."},
        {"name": "min_dmax", "type": "float", "description": "Smallest allowed solimp dmax of a robot pair. Default 0.5."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; excludes [{body1, body2, verdict}]; table_pairs {expected, present, changed}; robot_pairs_checked; bad_pairs; masked_geoms (object geoms the arm cannot collide with); parented_objects; robot_object_equalities."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "anti-tampering", "contact", "exclude", "pair", "contype", "conaffinity", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "contact_exclusion_audit",
}

import mujoco

from tool_library.common import scene_assert as SA

PAIR_FIELDS = (("pair_dim", "condim"), ("pair_friction", "friction"), ("pair_solref", "solref"), ("pair_solimp", "solimp"),
               ("pair_margin", "margin"), ("pair_gap", "gap"), ("pair_solreffriction", "solreffriction"))


def _exclude_bodies(model, i):
    sig = int(model.exclude_signature[i])
    return sig >> 16, sig & 0xFFFF          # MuJoCo encodes (body1 << 16) + body2


def _collide(model, g1, g2):
    return bool((int(model.geom_contype[g1]) & int(model.geom_conaffinity[g2])) or (int(model.geom_contype[g2]) & int(model.geom_conaffinity[g1])))


def contact_exclusion_audit(scene, setup=None, gap_tol_m=0.001, max_timeconst_s=0.1, min_stiffness=1000.0, min_dmax=0.5):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    from tool_library.verifiers._reference import load_reference
    st = load_reference(model, setup)
    ref, cfg = st["model"], st["cfg"]
    robot = SA.robot_body_ids(model, ref)
    table = SA.table_body_ids(model, cfg)
    ref_robot = SA.robot_body_ids(ref)
    bmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_BODY)
    objects = {b for b in range(1, model.nbody) if b not in robot and b not in table}
    issues = []
    bn = lambda b: model.body(b).name if b >= 0 else "world"
    # (1) excludes
    excl = []
    ref_rr = set()
    for i in range(ref.nexclude):
        a, b = _exclude_bodies(ref, i)
        if a in ref_robot and b in ref_robot:
            ref_rr.add(frozenset({bmap.get(a, -1), bmap.get(b, -1)}))
    for i in range(model.nexclude):
        a, b = _exclude_bodies(model, i)
        ra, rb = a in robot, b in robot
        verdict = "ok"
        if ra != rb and ((a in objects or b in objects) or (a in table or b in table) or a == 0 or b == 0):
            verdict = "robot-object" if (a in objects or b in objects) else "robot-table"
            issues.append(f"<exclude body1='{bn(a)}' body2='{bn(b)}'> switches off contact between the robot and "
                          f"{'an object' if verdict == 'robot-object' else 'the table'}; remove it")
        elif ra and rb and frozenset({a, b}) not in ref_rr:
            verdict = "robot-robot not in setup"
            issues.append(f"<exclude body1='{bn(a)}' body2='{bn(b)}'> is not in the setup: robot parts would pass through each other; remove it")
        excl.append({"body1": bn(a), "body2": bn(b), "verdict": verdict})
    # (2) the setup's robot-table pairs
    gname = lambda m, g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    scene_pairs = {}
    for p in range(model.npair):
        scene_pairs[frozenset({gname(model, int(model.pair_geom1[p])), gname(model, int(model.pair_geom2[p]))})] = p
    sgeoms = SA.names_of(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    expected, present, changed_pairs, missing_pairs = 0, 0, [], []
    ref_table = SA.table_body_ids(ref, cfg)
    for p in range(ref.npair):
        g1, g2 = int(ref.pair_geom1[p]), int(ref.pair_geom2[p])
        b1, b2 = int(ref.geom_bodyid[g1]), int(ref.geom_bodyid[g2])
        if not ((b1 in ref_robot and b2 in ref_table) or (b2 in ref_robot and b1 in ref_table)):
            continue
        expected += 1
        n1, n2 = gname(ref, g1), gname(ref, g2)
        mm = SA.match_names([n1, n2], sgeoms)
        key = frozenset({mm.get(n1) or n1, mm.get(n2) or n2})
        if key not in scene_pairs:
            missing_pairs.append(f"{n1} - {n2}")
            continue
        present += 1
        q = scene_pairs[key]
        for attr, field in PAIR_FIELDS:
            if SA.differs(getattr(model, attr)[q], getattr(ref, attr)[p], 1e-9, 1e-6):
                changed_pairs.append({"pair": f"{n1} - {n2}", "field": field, "reference": SA.r6(getattr(ref, attr)[p]), "scene": SA.r6(getattr(model, attr)[q])})
    if missing_pairs:
        issues.append(f"{len(missing_pairs)} of the setup's {expected} hand-table contact pairs are missing ({', '.join(missing_pairs[:4])}"
                      f"{' ...' if len(missing_pairs) > 4 else ''}): the hands pass through the table; restore the setup's <contact> pairs")
    for c in changed_pairs[:6]:
        issues.append(f"hand-table pair {c['pair']}: {c['field']} is {c['scene']} but {c['reference']} in the setup; restore it")
    # (3) explicit robot <-> non-robot pairs
    bad, n_rp = [], 0
    ref_keys = {frozenset({gname(ref, int(ref.pair_geom1[p])), gname(ref, int(ref.pair_geom2[p]))}) for p in range(ref.npair)}
    for p in range(model.npair):
        g1, g2 = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        r1, r2 = int(model.geom_bodyid[g1]) in robot, int(model.geom_bodyid[g2]) in robot
        if r1 == r2 or frozenset({gname(model, g1), gname(model, g2)}) in ref_keys:
            continue                                         # the setup's own pairs were compared in (2)
        n_rp += 1
        lab = f"{SA.geom_label(model, g1)} - {SA.geom_label(model, g2)}"
        why = []
        if float(model.pair_gap[p]) - float(model.pair_margin[p]) > gap_tol_m:
            why.append(f"gap {model.pair_gap[p]:.4g} m exceeds margin {model.pair_margin[p]:.4g} m (penetration without force)")
        sr = model.pair_solref[p]
        if sr[0] > max_timeconst_s:
            why.append(f"solref time constant {sr[0]:.4g} s is spongy (> {max_timeconst_s})")
        elif sr[0] < 0 and -sr[0] < min_stiffness:
            why.append(f"direct stiffness {-sr[0]:.4g} N/m is spongy (< {min_stiffness})")
        if float(model.pair_solimp[p][1]) < min_dmax:
            why.append(f"solimp dmax {model.pair_solimp[p][1]:.3g} < {min_dmax} (a weak contact)")
        if int(model.pair_dim[p]) == 1:
            why.append("condim 1 makes it frictionless")
        if why:
            bad.append({"pair": lab, "why": why})
    for b in bad[:8]:
        issues.append(f"contact pair {b['pair']}: {'; '.join(b['why'])}; use the setup's pair parameters")
    if len(bad) > 8:
        issues.append(f"... and {len(bad) - 8} more weakened robot pairs")
    # (4) collision bits
    arm_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot and SA.geom_collidable(model, g)
                 and "_collision_hand_" not in (model.geom(g).name or "")]
    table_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in table and SA.geom_collidable(model, g)]
    masked = []
    if arm_geoms and table_geoms:
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) not in objects or not SA.geom_collidable(model, g):
                continue
            if any(_collide(model, g, t) for t in table_geoms) and not any(_collide(model, g, a) for a in arm_geoms):
                masked.append(SA.geom_label(model, g))
    if masked:
        issues.append(f"{len(masked)} object geom(s) collide with the table but not with the robot arm ({', '.join(masked[:5])}"
                      f"{' ...' if len(masked) > 5 else ''}): give them the table's contype/conaffinity bits")
    rtg = [g for g in range(ref.ngeom) if int(ref.geom_bodyid[g]) in ref_table and SA.geom_collidable(ref, g)]
    for rg in rtg:
        nm = gname(ref, rg)
        sg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm) if nm else -1
        if sg >= 0 and (int(model.geom_contype[sg]), int(model.geom_conaffinity[sg])) != (int(ref.geom_contype[rg]), int(ref.geom_conaffinity[rg])):
            issues.append(f"table geom {nm} has contype/conaffinity {int(model.geom_contype[sg])}/{int(model.geom_conaffinity[sg])} but "
                          f"{int(ref.geom_contype[rg])}/{int(ref.geom_conaffinity[rg])} in the setup: the arm would pass through the table; restore them")
    # (5) objects parented to or glued to the robot
    parented = [f"{model.body(b).name} (child of {model.body(int(model.body_parentid[b])).name})" for b in range(1, model.nbody)
                if b not in robot and int(model.body_parentid[b]) in robot]
    # robot_body_ids folds the subtree of a robot body into the robot; find objects that are named like objects but sit there
    for b in range(1, model.nbody):
        nm = model.body(b).name
        if b in robot and not nm.startswith(SA.ROBOT_PREFIX) and bmap and b not in set(bmap.values()):
            parented.append(f"{nm} (inside robot body {model.body(int(model.body_parentid[b])).name})")
    if parented:
        issues.append(f"object bodies nested under the robot: {', '.join(parented[:5])}; MuJoCo filters parent-child contact and the "
                      f"object rides the hand: attach it to the world with its own joint")
    glued = []
    for e in range(model.neq):
        if int(model.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)):
            continue
        b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if hasattr(model, "eq_objtype") and int(model.eq_objtype[e]) == int(mujoco.mjtObj.mjOBJ_SITE):
            b1, b2 = int(model.site_bodyid[b1]), int(model.site_bodyid[b2])
        if (b1 in robot) != (b2 in robot) and ((b1 in objects) or (b2 in objects)):
            glued.append(f"{model.equality(e).name or f'equality {e}'} ({bn(b1)} - {bn(b2)}, active={bool(model.eq_active0[e])})")
    if glued:
        issues.append(f"equality constraints tie the robot to objects: {', '.join(glued[:5])}; objects must move by contact, remove them")
    return SA.report(issues, excludes=excl, table_pairs={"expected": expected, "present": present, "changed": changed_pairs, "missing": missing_pairs},
                     robot_pairs_checked=n_rp, bad_pairs=bad, masked_geoms=masked, parented_objects=parented,
                     robot_object_equalities=glued, setup=str(st["dir"]))
