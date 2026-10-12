"""Robot model integrity check: the scene's robot is the setup's robot, body for body and geom for geom."""
from __future__ import annotations

TOOL = {
    "name": "robot_model_integrity_check",
    "title": "Robot model integrity check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Compares every robot body, geom (collision and visual), joint and the hand-flange welds of a scene with the setup's reference robot and reports anything deleted, added, moved, shrunk, made lighter or made non-colliding.",
    "description": (
        "Loads the reference robot from a setup folder (default: the repository's setup/scene.xml + setup.yaml) and "
        "matches its robot bodies, joints and welds to the scene by name (exact, attach-prefix suffix, or the same name "
        "tokens without 'robot'). Per body it compares mass, principal inertia, centre of mass, gravity compensation (a "
        "proposal-scene hand may carry 1.0; rigid execution permits only the canonical value) and the pose relative to its parent (the station root bodies are "
        "left to stage_station_integrity_check). Per body it compares the ordered list of geoms: type, size, pose, mesh "
        "(vertex count and extents), contype/conaffinity, condim, group, friction, solref/solimp, margin/gap and "
        "priority, so a deleted or shrunk collision primitive, a hidden visual or a collision switched off is named. "
        "The hand hull geoms (handgeom.add_hand_hulls) may be absent because the executor adds them, but if present "
        "must equal the reference. Joint type, axis, anchor, damping, armature, frictionloss and stiffness are "
        "compared (ranges are robot_joint_range_check's job), and robot bodies or geoms the reference does not have "
        "are reported when they carry mass or collide. Assumes the robot is named as in the setup. Judge the scene a "
        "trajectory was executed on. A rigid runtime is compared with an independently compiled rigid robot from "
        "the trusted setup, including physical arm inertias and fixed hand mounts; proposal scenes retain their "
        "original setup reference."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF (Stage 2 scene.xml or a Stage 3 executed scene)."},
        {"name": "setup", "type": "str | None", "description": "Setup folder (or its setup.yaml / scene.xml) holding the reference robot. Default: the repository's setup/."},
        {"name": "pos_tol_m", "type": "float", "description": "Allowed difference of positions and sizes, metres. Default 1e-5."},
        {"name": "rel_tol", "type": "float", "description": "Allowed relative difference of masses, inertias and other parameters. Default 1e-4."},
        {"name": "angle_tol_deg", "type": "float", "description": "Allowed orientation difference, degrees. Default 0.01."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; n_bodies / n_geoms / n_joints compared; missing (robot entities absent from the scene); changed [{what, name, field, reference, scene}]; extra (robot bodies/geoms not in the reference); hulls_absent; worst (largest relative mass change)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "anti-tampering", "robot", "integrity", "setup", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "robot_model_integrity_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA

GEOM_FIELDS = (("geom_type", "type", 0.0, 0.0), ("geom_contype", "contype", 0.0, 0.0),
               ("geom_conaffinity", "conaffinity", 0.0, 0.0), ("geom_condim", "condim", 0.0, 0.0),
               ("geom_group", "group", 0.0, 0.0), ("geom_priority", "priority", 0.0, 0.0),
               ("geom_friction", "friction", 0.0, 1e-4), ("geom_solref", "solref", 0.0, 1e-4),
               ("geom_solimp", "solimp", 0.0, 1e-4), ("geom_margin", "margin", 1e-6, 0.0), ("geom_gap", "gap", 1e-6, 0.0))
JOINT_FIELDS = (("jnt_type", "type", 0.0, 0.0), ("dof_damping", "damping", 0.0, 1e-4), ("dof_armature", "armature", 0.0, 1e-4),
                ("dof_frictionloss", "frictionloss", 0.0, 1e-4), ("jnt_stiffness", "stiffness", 0.0, 1e-4))


def _geom_rows(model, b):
    gs = [g for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b]) + int(model.body_geomnum[b]))] if int(model.body_geomnum[b]) else []
    plain = [g for g in gs if not SA.is_hull_geom(model, g)]
    hulls = {model.geom(g).name: g for g in gs if SA.is_hull_geom(model, g)}
    return plain, hulls


def _cmp_geom(model, ref, g, rg, pos_tol, rel_tol, ang_tol, changed, label):
    for attr, field, at, rt in GEOM_FIELDS:
        a, b = getattr(model, attr)[g], getattr(ref, attr)[rg]
        if SA.differs(a, b, at, max(rt, rel_tol if rt else 0.0)):
            changed.append({"what": "geom", "name": label, "field": field, "reference": SA.r6(b), "scene": SA.r6(a)})
    if int(ref.geom_type[rg]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        if SA.differs(model.geom_size[g], ref.geom_size[rg], pos_tol, rel_tol):
            changed.append({"what": "geom", "name": label, "field": "size", "reference": SA.r6(ref.geom_size[rg]), "scene": SA.r6(model.geom_size[g])})
    else:
        sa, sb = SA.mesh_signature(model, g), SA.mesh_signature(ref, rg)
        if sa is None or sa[0] != sb[0] or SA.differs(sa[1], sb[1], pos_tol, rel_tol):
            changed.append({"what": "geom", "name": label, "field": "mesh", "reference": sb, "scene": sa})
    if SA.differs(model.geom_pos[g], ref.geom_pos[rg], pos_tol, 0.0):
        changed.append({"what": "geom", "name": label, "field": "pos", "reference": SA.r6(ref.geom_pos[rg]), "scene": SA.r6(model.geom_pos[g])})
    if SA.quat_angle_deg(model.geom_quat[g], ref.geom_quat[rg]) > ang_tol:
        changed.append({"what": "geom", "name": label, "field": "quat", "reference": SA.r6(ref.geom_quat[rg]), "scene": SA.r6(model.geom_quat[g])})


def robot_model_integrity_check(scene, setup=None, pos_tol_m=1e-5, rel_tol=1e-4, angle_tol_deg=0.01):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile, so its robot cannot be checked: {err}"], compiled=False)
    from tool_library.verifiers._reference import load_reference
    st = load_reference(model, setup)
    ref, cfg = st["model"], st["cfg"]
    roots = set(SA.station_root_names(cfg).values())
    bmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_BODY)
    jmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_JOINT)
    missing, changed, extra, hulls_absent = [], [], [], []
    n_geoms = 0
    worst = 0.0
    for rb, sb in bmap.items():
        name = ref.body(rb).name
        if sb < 0:
            missing.append(f"body {name}")
            continue
        rp = int(ref.body_parentid[rb])
        want_parent = 0 if rp == 0 else bmap.get(rp, -2)
        if int(model.body_parentid[sb]) != want_parent:
            changed.append({"what": "body", "name": name, "field": "parent", "reference": ref.body(rp).name,
                            "scene": model.body(int(model.body_parentid[sb])).name})
        rm, sm = float(ref.body_mass[rb]), float(model.body_mass[sb])
        if SA.differs(sm, rm, 1e-9, rel_tol):
            changed.append({"what": "body", "name": name, "field": "mass", "reference": SA.r6(rm), "scene": SA.r6(sm)})
            worst = max(worst, abs(sm - rm) / max(rm, 1e-9))
        if SA.differs(np.sort(model.body_inertia[sb]), np.sort(ref.body_inertia[rb]), 1e-12, rel_tol):
            changed.append({"what": "body", "name": name, "field": "inertia", "reference": SA.r6(ref.body_inertia[rb], 9), "scene": SA.r6(model.body_inertia[sb], 9)})
        if rm > 1e-9 and name not in roots and SA.differs(model.body_ipos[sb], ref.body_ipos[rb], pos_tol_m, 0.0):
            changed.append({"what": "body", "name": name, "field": "com", "reference": SA.r6(ref.body_ipos[rb]), "scene": SA.r6(model.body_ipos[sb])})
        gc_s, gc_r = float(model.body_gravcomp[sb]), float(ref.body_gravcomp[rb])
        is_hand = "_arm_" not in name
        if abs(gc_s - gc_r) > 1e-9 and not (not st["runtime"] and is_hand and abs(gc_s - 1.0) < 1e-9):
            changed.append({"what": "body", "name": name, "field": "gravcomp", "reference": gc_r, "scene": gc_s})
        if name not in roots:
            if SA.differs(model.body_pos[sb], ref.body_pos[rb], pos_tol_m, 0.0):
                changed.append({"what": "body", "name": name, "field": "pos", "reference": SA.r6(ref.body_pos[rb]), "scene": SA.r6(model.body_pos[sb])})
            if SA.quat_angle_deg(model.body_quat[sb], ref.body_quat[rb]) > angle_tol_deg:
                changed.append({"what": "body", "name": name, "field": "quat", "reference": SA.r6(ref.body_quat[rb]), "scene": SA.r6(model.body_quat[sb])})
        # geoms, in order (unnamed arm meshes have no other identity)
        rplain, rhull = _geom_rows(ref, rb)
        splain, shull = _geom_rows(model, sb)
        n_geoms += len(rplain)
        if len(rplain) != len(splain):
            rl = [SA.geom_label(ref, g) for g in rplain]; sl = [SA.geom_label(model, g) for g in splain]
            gone = [x for x in rl if x not in sl] or [f"{len(rplain) - len(splain)} geom(s)"] if len(rplain) > len(splain) else []
            added = [x for x in sl if x not in rl] if len(splain) > len(rplain) else []
            if gone:
                missing += [f"geom {x} (body {name})" for x in gone]
            if added:
                extra += [f"geom {x} (body {name})" for x in added]
            if not gone and not added:
                changed.append({"what": "body", "name": name, "field": "n_geoms", "reference": len(rplain), "scene": len(splain)})
            # compare the geoms that keep their name
            pairs = [(g, rg) for g in splain for rg in rplain if model.geom(g).name and model.geom(g).name == ref.geom(rg).name]
        else:
            pairs = list(zip(splain, rplain))
        for g, rg in pairs:
            _cmp_geom(model, ref, g, rg, pos_tol_m, rel_tol, angle_tol_deg, changed, SA.geom_label(ref, rg))
        for hn, rg in rhull.items():
            sn = next((k for k in shull if k.endswith(hn[len("robot_"):]) or k == hn), None)
            if sn is None:
                hulls_absent.append(hn)
            else:
                _cmp_geom(model, ref, shull[sn], rg, pos_tol_m, rel_tol, angle_tol_deg, changed, hn)
    # robot bodies the reference does not have
    mapped = {s for s in bmap.values() if s >= 0}
    for b in range(1, model.nbody):
        nm = model.body(b).name
        if b in mapped or not (nm.startswith(SA.ROBOT_PREFIX) or int(model.body_parentid[b]) in mapped):
            continue
        heavy = float(model.body_mass[b]) > 1e-9
        coll = any(SA.geom_collidable(model, g) for g in SA.body_geoms(model, [b]))
        if heavy or coll:
            extra.append(f"body {nm} ({'mass %.4g kg' % model.body_mass[b] if heavy else 'collides'})")
    # joints (ranges are robot_joint_range_check's)
    for rj, sj in jmap.items():
        name = ref.joint(rj).name
        if sj < 0:
            missing.append(f"joint {name}")
            continue
        for attr, field, at, rt in JOINT_FIELDS:
            arr_s, arr_r = getattr(model, attr), getattr(ref, attr)
            i_s = int(model.jnt_dofadr[sj]) if attr.startswith("dof_") else sj
            i_r = int(ref.jnt_dofadr[rj]) if attr.startswith("dof_") else rj
            if SA.differs(arr_s[i_s], arr_r[i_r], at, rt):
                changed.append({"what": "joint", "name": name, "field": field, "reference": SA.r6(arr_r[i_r]), "scene": SA.r6(arr_s[i_s])})
        if SA.differs(model.jnt_axis[sj], ref.jnt_axis[rj], 1e-6, 0.0):
            changed.append({"what": "joint", "name": name, "field": "axis", "reference": SA.r6(ref.jnt_axis[rj]), "scene": SA.r6(model.jnt_axis[sj])})
        if SA.differs(model.jnt_pos[sj], ref.jnt_pos[rj], pos_tol_m, 0.0):
            changed.append({"what": "joint", "name": name, "field": "pos", "reference": SA.r6(ref.jnt_pos[rj]), "scene": SA.r6(model.jnt_pos[sj])})
        rb_s = int(model.jnt_bodyid[sj]); rb_r = int(ref.jnt_bodyid[rj])
        if bmap.get(rb_r, -1) != rb_s:
            changed.append({"what": "joint", "name": name, "field": "body", "reference": ref.body(rb_r).name, "scene": model.body(rb_s).name})
    # the hand-flange welds (position-only attachment)
    for e in range(ref.neq):
        b1, b2 = int(ref.eq_obj1id[e]), int(ref.eq_obj2id[e])
        if int(ref.eq_type[e]) != int(mujoco.mjtEq.mjEQ_WELD) or b1 not in bmap or b2 not in bmap:
            continue
        s1, s2 = bmap[b1], bmap[b2]
        lab = f"weld {ref.body(b1).name} - {ref.body(b2).name}"
        hit = [k for k in range(model.neq) if int(model.eq_type[k]) == int(mujoco.mjtEq.mjEQ_WELD)
               and {int(model.eq_obj1id[k]), int(model.eq_obj2id[k])} == {s1, s2}]
        if not hit:
            missing.append(lab)
            continue
        k = hit[0]
        for attr, field in (("eq_data", "data"), ("eq_solref", "solref"), ("eq_solimp", "solimp")):
            if SA.differs(getattr(model, attr)[k], getattr(ref, attr)[e], 1e-6, 1e-4):
                changed.append({"what": "equality", "name": lab, "field": field, "reference": SA.r6(getattr(ref, attr)[e]), "scene": SA.r6(getattr(model, attr)[k])})
    issues = []
    if missing:
        issues.append(f"{len(missing)} robot part(s) of the setup are missing from the scene: {', '.join(missing[:8])}"
                      f"{' ...' if len(missing) > 8 else ''}; rebuild the scene from the setup's robot (setup/scene.xml) instead of editing it")
    for c in changed[:12]:
        issues.append(f"{c['what']} {c['name']}: {c['field']} is {c['scene']} in the scene but {c['reference']} in the setup; restore the setup's value"
                      + (" (a lighter or shrunk robot part changes what the hand can do)" if c["field"] in ("mass", "size", "mesh") else "")
                      + (" (a collision switched off lets the robot pass through)" if c["field"] in ("contype", "conaffinity") else ""))
    if len(changed) > 12:
        issues.append(f"... and {len(changed) - 12} more changed robot parameters (see 'changed')")
    if extra:
        issues.append(f"robot parts the setup does not have: {', '.join(extra[:8])}; remove them (an added collision geom or mass changes the robot)")
    return SA.report(issues, n_bodies=len(bmap), n_geoms=n_geoms, n_joints=len(jmap), missing=missing, changed=changed,
                     extra=extra, hulls_absent=hulls_absent, worst=round(worst, 6), setup=str(st["dir"]))
