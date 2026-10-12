"""Execution must preserve the objects and mechanisms supplied by Stage 2."""
from __future__ import annotations

TOOL = {
    "name": "execution_scene_integrity_check", "title": "Execution scene integrity check", "type": "verifier",
    "category": "Scene Assertions", "summary": "Compare execution objects, mechanisms and initial state with the independently compiled Stage 2 scene.",
    "description": "Permits only canonical rigid robot conversion, contact-pair construction and source-proven unspecified rigid contact defaults. Explicit compliance, object mass, inertia, geometry, friction, joints, actuators, tendons, constraints, contact exclusions and initial states must remain the same. The reference comes from the trusted Stage 2 input, never the submitted runtime scene.",
    "inputs": [{"name": "scene", "type": "str", "description": "Executed runtime MJCF."},
               {"name": "scene_recon_json", "type": "str", "description": "Trusted Stage 2 scene reconstruction manifest."},
               {"name": "trajectory", "type": "str | None", "description": "Runtime recording; its first object state must equal the trusted initial state."}],
    "outputs": [{"name": "report", "type": "dict", "description": "ok, issues, compared."}],
    "requirements": ["mujoco>=3.2", "numpy"], "simulators": ["MuJoCo"], "tags": ["integrity", "objects", "stage3"],
    "status": "tested", "entrypoint": "execution_scene_integrity_check",
}

import json
from pathlib import Path

import mujoco
import numpy as np


def execution_scene_integrity_check(scene, scene_recon_json, trajectory=None):
    from tool_library.common import episode as E, rigid_robot as RR
    source = Path(scene_recon_json).resolve()
    recon = json.loads(source.read_text())
    proposal = source.parent / recon.get("scene_xml", "scene.xml")
    canonical, _ = E.build_execution_model(str(proposal), scene_recon_json=source)
    ref = RR.compile_rigid(canonical).model
    got = mujoco.MjModel.from_xml_path(str(scene))
    issues, compared = [], 0

    def name(m, kind, i):
        return mujoco.mj_id2name(m, kind, int(i)) or f"unnamed[{int(i)}]"
    # a cord Isaac Sim simulates may be in the execution scene as its MuJoCo planning copy (build_cord_planning_scene);
    # its physics of record is the Isaac recording (isaac_recording_check), so the copy is not a Stage 2 object here
    from tool_library.common import isaac_cord as IC
    from tool_library.skills.execute_isaac_deformables import isaac_deformables
    copies = {n for _, s_ in isaac_deformables(source) for n in IC.body_names(s_["name"], s_["n_links"])}

    def robot(m, body):
        return m.body(int(body)).name.startswith("robot_") or m.body(int(body)).name in copies
    def eq(a, b, label):
        nonlocal compared
        compared += 1
        a, b = np.asarray(a), np.asarray(b)
        # MJCF export has six significant decimal places. These fixed numeric
        # serialization tolerances are far below geometric/task tolerances.
        same = a.shape == b.shape and (np.allclose(a, b, rtol=1e-5, atol=1e-7, equal_nan=False)
                                      if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)
                                      else np.array_equal(a, b))
        if not same:
            issues.append(f"{label} differs from the trusted Stage 2 runtime")
    def map_entities(m, kind, count, keep, label):
        out = {}
        for i in range(count):
            if keep(m, i):
                key = label(m, i) if label else name(m, kind, i)
                if key in out:
                    issues.append(f"ambiguous {kind.name} entity {key}")
                out[key] = i
        return out
    def rows(kind, count, keep, attrs, label=None):
        a = map_entities(got, kind, getattr(got, count), keep, label)
        b = map_entities(ref, kind, getattr(ref, count), keep, label)
        for key in sorted(set(a) ^ set(b)):
            issues.append(f"{kind.name} {key} {'added' if key in a else 'missing'} in the execution scene")
        pairs = [(key, a[key], b[key]) for key in sorted(set(a) & set(b))]
        for key, ga, rb in pairs:
            for attr in attrs:
                if hasattr(ref, attr):
                    eq(getattr(got, attr)[ga], getattr(ref, attr)[rb], f"{key}.{attr}")
        return pairs
    T = mujoco.mjtObj
    bodies = rows(T.mjOBJ_BODY, "nbody", lambda m, i: not robot(m, i),
                  ("body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_gravcomp", "body_mocapid"))
    for key, g, r in bodies:
        eq(got.body(int(got.body_parentid[g])).name, ref.body(int(ref.body_parentid[r])).name, f"{key}.parent")
    def geom_key(m, g):
        b = int(m.geom_bodyid[g])
        # The executor assigns names to otherwise unnamed object colliders.
        return (m.body(b).name, int(g) - int(m.body_geomadr[b]))
    geoms = rows(T.mjOBJ_GEOM, "ngeom", lambda m, i: not robot(m, m.geom_bodyid[i]),
                 ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "geom_condim",
                  "geom_group", "geom_priority", "geom_friction", "geom_solmix", "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_rgba"), geom_key)
    for key, g, r in geoms:
        if int(ref.geom_type[r]) == int(mujoco.mjtGeom.mjGEOM_MESH) and int(got.geom_type[g]) == int(ref.geom_type[r]):
            for attr, address, count in (("mesh_vert", "mesh_vertadr", "mesh_vertnum"), ("mesh_face", "mesh_faceadr", "mesh_facenum")):
                arrays = []
                for m, geom in ((got, g), (ref, r)):
                    mesh = int(m.geom_dataid[geom]); adr, n = int(getattr(m, address)[mesh]), int(getattr(m, count)[mesh])
                    arrays.append(getattr(m, attr)[adr:adr + n])
                eq(*arrays, f"{key}.{attr}")
        if int(ref.geom_type[r]) == int(mujoco.mjtGeom.mjGEOM_HFIELD) and int(got.geom_type[g]) == int(ref.geom_type[r]):
            arrays = []
            for m, geom in ((got, g), (ref, r)):
                h = int(m.geom_dataid[geom]); adr = int(m.hfield_adr[h]); n = int(m.hfield_nrow[h] * m.hfield_ncol[h])
                arrays.append(m.hfield_data[adr:adr + n])
            eq(*arrays, f"{key}.height_field")
        gm, rm = int(got.geom_matid[g]), int(ref.geom_matid[r])
        eq(name(got, T.mjOBJ_MATERIAL, gm) if gm >= 0 else "none",
           name(ref, T.mjOBJ_MATERIAL, rm) if rm >= 0 else "none", f"{key}.material")
        if gm >= 0 and rm >= 0:
            for attr in ("mat_rgba", "mat_texrepeat", "mat_texuniform", "mat_emission", "mat_specular", "mat_shininess", "mat_reflectance", "mat_metallic", "mat_roughness"):
                if hasattr(ref, attr):
                    eq(getattr(got, attr)[gm], getattr(ref, attr)[rm], f"{key}.{attr}")
    rows(T.mjOBJ_SITE, "nsite", lambda m, i: not robot(m, m.site_bodyid[i]), ("site_type", "site_size", "site_pos", "site_quat"))
    joints = rows(T.mjOBJ_JOINT, "njnt", lambda m, i: m.body(int(m.jnt_bodyid[i])).name not in copies,
                  ("jnt_type", "jnt_pos", "jnt_axis", "jnt_limited", "jnt_range", "jnt_stiffness", "jnt_margin", "jnt_solref", "jnt_solimp"))
    qfirst = vfirst = None
    if trajectory is not None:
        with np.load(str(trajectory), allow_pickle=False) as z:
            qfirst, vfirst = np.asarray(z["qpos"][0]), np.asarray(z["qvel"][0])
            if "initial_state" in z and "state_spec" in z:
                first, expected = mujoco.MjData(got), mujoco.MjData(ref)
                mujoco.mj_setState(got, first, z["initial_state"], int(z["state_spec"]))
                if ref.nkey:
                    mujoco.mj_resetDataKeyframe(ref, expected, 0)
                eq(first.time, expected.time, "recorded_initial_time")
                for attr in ("mocap_pos", "mocap_quat"):
                    eq(getattr(first, attr), getattr(expected, attr), f"recorded_initial_{attr}")
            elif got.nmocap or ref.nmocap:
                issues.append("initial complete simulator state is required to verify object mocap state")
    for key, g, r in joints:
        eq(got.body(int(got.jnt_bodyid[g])).name, ref.body(int(ref.jnt_bodyid[r])).name, f"{key}.body")
        jt = int(ref.jnt_type[r]); nq = 7 if jt == 0 else (4 if jt == 1 else 1); nv = 6 if jt == 0 else (3 if jt == 1 else 1)
        gq, rq = int(got.jnt_qposadr[g]), int(ref.jnt_qposadr[r]); gv, rv = int(got.jnt_dofadr[g]), int(ref.jnt_dofadr[r])
        for attr in ("dof_damping", "dof_armature", "dof_frictionloss", "dof_solref", "dof_solimp"):
            eq(getattr(got, attr)[gv:gv + nv], getattr(ref, attr)[rv:rv + nv], f"{key}.{attr}")
        eq(got.qpos0[gq:gq + nq], ref.qpos0[rq:rq + nq], f"{key}.qpos0")
        expected_q = (ref.key_qpos[0] if ref.nkey else ref.qpos0)[rq:rq + nq]
        expected_v = ref.key_qvel[0, rv:rv + nv] if ref.nkey else np.zeros(nv)
        eq((got.key_qpos[0] if got.nkey else got.qpos0)[gq:gq + nq], expected_q, f"{key}.home_qpos")
        eq(got.key_qvel[0, gv:gv + nv] if got.nkey else np.zeros(nv), expected_v, f"{key}.home_qvel")
        if qfirst is not None:
            eq(qfirst[gq:gq + nq], expected_q, f"{key}.recorded_initial_qpos")
            eq(vfirst[gv:gv + nv], expected_v, f"{key}.recorded_initial_qvel")
    # Tendons and equalities may bind objects to each other or the robot; all
    # must come from the canonical trusted construction, with identical targets.
    tendons = rows(T.mjOBJ_TENDON, "ntendon", lambda m, i: True,
                   ("tendon_stiffness", "tendon_damping", "tendon_frictionloss", "tendon_limited", "tendon_range",
                    "tendon_margin", "tendon_solref_lim", "tendon_solimp_lim", "tendon_solref_fri", "tendon_solimp_fri", "tendon_lengthspring"))
    def wraps(m, t):
        out = []
        for w in range(int(m.tendon_adr[t]), int(m.tendon_adr[t] + m.tendon_num[t])):
            typ, obj = int(m.wrap_type[w]), int(m.wrap_objid[w])
            kind = {1: T.mjOBJ_JOINT, 3: T.mjOBJ_SITE, 4: T.mjOBJ_GEOM, 5: T.mjOBJ_GEOM}.get(typ)
            out.append((typ, name(m, kind, obj) if kind and obj >= 0 else str(obj), float(m.wrap_prm[w])))
        return out
    for key, g, r in tendons:
        a, b = wraps(got, g), wraps(ref, r)
        eq([x[:2] for x in a], [x[:2] for x in b], f"{key}.wrap_targets")
        eq([x[2] for x in a], [x[2] for x in b], f"{key}.wrap_parameters")
    equalities = rows(T.mjOBJ_EQUALITY, "neq", lambda m, i: True,
                     ("eq_type", "eq_active0", "eq_data", "eq_solref", "eq_solimp"))
    for key, g, r in equalities:
        kind = {0: T.mjOBJ_BODY, 1: T.mjOBJ_BODY, 2: T.mjOBJ_JOINT, 3: T.mjOBJ_TENDON}.get(int(ref.eq_type[r]))
        for attr in ("eq_obj1id", "eq_obj2id"):
            ga, rb = int(getattr(got, attr)[g]), int(getattr(ref, attr)[r])
            eq(name(got, kind, ga) if kind and ga >= 0 else str(ga), name(ref, kind, rb) if kind and rb >= 0 else str(rb), f"{key}.{attr}")
    # An added actuator can apply otherwise unexplained forces to an object.
    actuators = rows(T.mjOBJ_ACTUATOR, "nu", lambda m, i: True,
                     ("actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_dynprm",
                      "actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_ctrllimited", "actuator_forcelimited", "actuator_actlimited",
                      "actuator_ctrlrange", "actuator_forcerange", "actuator_actrange"))
    for key, g, r in actuators:
        typ = int(ref.actuator_trntype[r]); kind = {0: T.mjOBJ_JOINT, 1: T.mjOBJ_JOINT, 3: T.mjOBJ_TENDON, 4: T.mjOBJ_SITE, 5: T.mjOBJ_BODY}.get(typ)
        for column in range(2):
            ga, rb = int(got.actuator_trnid[g, column]), int(ref.actuator_trnid[r, column])
            eq(name(got, kind, ga) if kind and ga >= 0 else str(ga), name(ref, kind, rb) if kind and rb >= 0 else str(rb), f"{key}.target{column}")
    def pair_key(m, i):
        return tuple(sorted((geom_key(m, int(m.pair_geom1[i])), geom_key(m, int(m.pair_geom2[i])))))
    rows(T.mjOBJ_PAIR, "npair", lambda m, i: True,
         ("pair_dim", "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap", "pair_friction"), pair_key)
    def excludes(m):
        return sorted(tuple(sorted((m.body(int(sig) >> 16).name, m.body(int(sig) & 65535).name))) for sig in m.exclude_signature)
    eq(excludes(got), excludes(ref), "contact exclusions")
    # Flex elements carry mechanics outside rigid-body/geom arrays. Canonical
    # conversion does not alter their ordering or topology.
    if ref.nflex or got.nflex:
        for attr in dir(ref):
            if attr.startswith("flex_") and isinstance(getattr(ref, attr), np.ndarray):
                eq(getattr(got, attr), getattr(ref, attr), attr)
    return {"ok": not issues, "issues": issues[:80], "n_issues": len(issues), "compared": compared}
