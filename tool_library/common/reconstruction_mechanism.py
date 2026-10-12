"""Independent, bounded Stage 2 component mobility evidence.

Stored reports are proposals, never certificates. The currently supported adapter
is articulated_joint_travel_test for a single-root, uncoupled, passive component.
Its ordinary 10 N / 1 Nm, 80% bidirectional travel criterion is only applied when
that verifier is explicitly claimed; it is not a universal test for springs,
threads, latches, detachable parts, or arbitrary mechanisms. Other required
mechanism claims remain unsupported and fail closed until they have an adapter.

A passing result proves mobility of a fixed-base component in isolation using
current scene physics. It does not prove robot reach, scene clearance, support,
the demonstrated initial state, or task success. Those require their own gates.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import mujoco
import numpy as np

SUPPORTED = {"articulated_joint_travel_test"}
SCOPE = "fixed-base component mobility; surrounding scene, robot feasibility and task outcome excluded"


def _same(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    return aa.shape == bb.shape and bool(np.isfinite(aa).all() and np.isfinite(bb).all()
                                         and np.allclose(aa, bb, atol=1e-7, rtol=1e-5))


def _children(model, root):
    out = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in out:
            out.add(b)
    return out


def _internal_contacts(model, bodies, prefix=""):
    """Object-local collision exceptions override per-geom collision parameters."""
    def body_name(b):
        return model.body(b).name[len(prefix):]
    pairs = {}
    for p in range(model.npair):
        a, b = int(model.pair_geom1[p]), int(model.pair_geom2[p])
        if int(model.geom_bodyid[a]) not in bodies or int(model.geom_bodyid[b]) not in bodies:
            continue
        names = tuple(sorted((model.geom(a).name[len(prefix):], model.geom(b).name[len(prefix):])))
        if names in pairs:
            raise ValueError("ambiguous duplicate internal contact pair")
        pairs[names] = {field: getattr(model, field)[p] for field in
                        ("pair_dim", "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap", "pair_friction")}
    excludes = set()
    for sig in model.exclude_signature:
        a, b = int(sig) >> 16, int(sig) & 0xffff
        if a in bodies and b in bodies:
            excludes.add(tuple(sorted((body_name(a), body_name(b)))))
    return pairs, excludes


def _binding(asset, scene, root_name):
    """Exact local component content; only attachment prefix/global pose may vary."""
    from tool_library.common.contact_defaults import _same_asset_geom
    roots = [b for b in range(1, asset.nbody) if int(asset.body_parentid[b]) == 0]
    if len(roots) != 1:
        raise ValueError("unsupported component: independent replay requires one asset root")
    ar = roots[0]
    an = asset.body(ar).name
    if not an or not (root_name == an or root_name.endswith("_" + an)):
        raise ValueError("recorded root_body must identify the asset's actual component root")
    prefix = root_name[:-len(an)]
    body_map = {}
    for a in range(1, asset.nbody):
        name = asset.body(a).name
        if not name:
            raise ValueError("unnamed asset body has no unambiguous scene binding")
        b = mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, prefix + name)
        if b < 0:
            raise ValueError(f"asset body {name!r} missing from scene")
        body_map[a] = b
    root = body_map[ar]
    if int(scene.body_parentid[root]) != 0:
        raise ValueError("unsupported component: root is nested under another scene body")
    if set(body_map.values()) != _children(scene, root):
        raise ValueError("asset and scene component body topology differ")
    scene_joints = set()
    for a, b in body_map.items():
        if a != ar and body_map.get(int(asset.body_parentid[a])) != int(scene.body_parentid[b]):
            raise ValueError("asset and scene body parent topology differ")
        fields = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_gravcomp")
        if a != ar:
            fields += ("body_pos", "body_quat")
        for field in fields:
            if not _same(getattr(asset, field)[a], getattr(scene, field)[b]):
                raise ValueError(f"asset/scene physics mismatch: {asset.body(a).name}.{field}")
        aj = [j for j in range(asset.njnt) if int(asset.jnt_bodyid[j]) == a]
        bj = [j for j in range(scene.njnt) if int(scene.jnt_bodyid[j]) == b]
        if len(aj) != len(bj):
            raise ValueError("asset and scene joint counts differ")
        for j, k in zip(aj, bj):
            if not asset.joint(j).name or scene.joint(k).name != prefix + asset.joint(j).name:
                raise ValueError("asset and scene joint names/order differ")
            scene_joints.add(k)
            for field in ("jnt_type", "jnt_axis", "jnt_pos", "jnt_limited", "jnt_range", "jnt_stiffness",
                          "jnt_solref", "jnt_solimp", "jnt_margin", "jnt_actfrclimited", "jnt_actfrcrange"):
                if not _same(getattr(asset, field)[j], getattr(scene, field)[k]):
                    raise ValueError(f"asset/scene physics mismatch: {asset.joint(j).name}.{field}")
            kind = int(asset.jnt_type[j])
            n = 6 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else 3 if kind == int(mujoco.mjtJoint.mjJNT_BALL) else 1
            da, db = int(asset.jnt_dofadr[j]), int(scene.jnt_dofadr[k])
            for field in ("dof_armature", "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp"):
                if not _same(getattr(asset, field)[da:da+n], getattr(scene, field)[db:db+n]):
                    raise ValueError(f"asset/scene physics mismatch: {asset.joint(j).name}.{field}")
            if kind in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                qa, qb = int(asset.jnt_qposadr[j]), int(scene.jnt_qposadr[k])
                for field in ("qpos0", "qpos_spring"):
                    if not _same(getattr(asset, field)[qa], getattr(scene, field)[qb]):
                        raise ValueError(f"asset/scene physics mismatch: {asset.joint(j).name}.{field}")
        ag = [g for g in range(asset.ngeom) if int(asset.geom_bodyid[g]) == a]
        bg = [g for g in range(scene.ngeom) if int(scene.geom_bodyid[g]) == b]
        if len(ag) != len(bg):
            raise ValueError("asset and scene geom counts differ")
        for g in ag:
            name = asset.geom(g).name
            if not name or mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_GEOM, prefix + name) < 0:
                raise ValueError("unnamed/missing geom has no unambiguous scene binding")
            if not _same_asset_geom(asset, scene, name, prefix + name, prefix):
                raise ValueError(f"asset/scene physics mismatch: geom {name}")
    ap, ae = _internal_contacts(asset, set(body_map))
    sp, se = _internal_contacts(scene, set(body_map.values()), prefix)
    if ae != se or set(ap) != set(sp):
        raise ValueError("asset/scene physics mismatch: internal contact pairs/exclusions")
    for pair in ap:
        if any(not _same(ap[pair][field], sp[pair][field]) for field in ap[pair]):
            raise ValueError("asset/scene physics mismatch: internal contact pair parameters")
    return root, set(body_map.values()), scene_joints


def _supported_component(asset, scene, bodies, joints):
    if asset.neq or asset.ntendon or asset.nu or asset.nflex or asset.nplugin:
        raise ValueError("unsupported component: coupled, actuated, flex or plugin mechanisms need a specialized independent adapter")
    if any(int(scene.body_mocapid[b]) >= 0 or int(scene.body_plugin[b]) >= 0 for b in bodies):
        raise ValueError("unsupported component: mocap or plugin body")
    for e in range(scene.neq):
        kind = int(scene.eq_objtype[e]); ids = {int(scene.eq_obj1id[e]), int(scene.eq_obj2id[e])}
        if (kind == int(mujoco.mjtObj.mjOBJ_BODY) and ids & bodies or
                kind == int(mujoco.mjtObj.mjOBJ_JOINT) and ids & joints):
            raise ValueError("unsupported component: scene equality couples this component")
    # Tendon routing can couple joints or sites on the object; no generic travel
    # interpretation is claimed for it. Reject only routes touching this object.
    for w in range(scene.nwrap):
        kind, obj = int(scene.wrap_type[w]), int(scene.wrap_objid[w])
        if (kind == int(mujoco.mjtWrap.mjWRAP_JOINT) and obj in joints or
                kind == int(mujoco.mjtWrap.mjWRAP_SITE) and int(scene.site_bodyid[obj]) in bodies or
                kind in (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER))
                and int(scene.geom_bodyid[obj]) in bodies):
            raise ValueError("unsupported component: scene tendon couples this component")
    for a in range(scene.nu):
        trn, obj = int(scene.actuator_trntype[a]), int(scene.actuator_trnid[a, 0])
        if (trn in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)) and obj in joints or
                trn == int(mujoco.mjtTrn.mjTRN_BODY) and obj in bodies or
                trn == int(mujoco.mjtTrn.mjTRN_SITE) and obj >= 0 and int(scene.site_bodyid[obj]) in bodies):
            raise ValueError("unsupported component: scene actuator drives this component")
    moving = [j for j in sorted(joints) if int(scene.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)]
    if not moving or len(moving) > 8:
        raise ValueError("unsupported component: expected 1–8 passive hinge/slide joints")
    for j in moving:
        if (int(scene.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
                or not scene.jnt_limited[j] or scene.jnt_stiffness[j] != 0):
            raise ValueError("unsupported component: unlimited, spring or non hinge/slide joints need a specialized independent adapter")
    if any(int(scene.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
           and int(scene.body_parentid[int(scene.jnt_bodyid[j])]) != 0 for j in joints):
        raise ValueError("unsupported component: independently free child part")
    if 6.0 * len(moving) / float(scene.opt.timestep) > 240000:
        raise ValueError("component replay exceeds the bounded 240000 integration-step budget")
    return [scene.joint(j).name for j in moving]


def _extract(scene_path, model, bodies, out):
    """Current compiled component, fixed at its root, without environment contacts."""
    from tool_library.skills.stage_on_bench import _absolute_assets
    spec = mujoco.MjSpec.from_file(str(scene_path))
    _absolute_assets(spec, Path(scene_path).parent)
    # Remove global references before removing their targets. Internal explicit
    # contacts/exclusions remain; unrelated robot contacts cannot enter the trial.
    keep_geoms = {model.geom(g).name for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}
    keep_bodies = {model.body(b).name for b in bodies}
    for collection in (spec.keys, spec.actuators, spec.sensors, spec.equalities, spec.tendons):
        for item in list(collection):
            spec.delete(item)
    for pair in list(spec.pairs):
        if pair.geomname1 not in keep_geoms or pair.geomname2 not in keep_geoms:
            spec.delete(pair)
    for exc in list(spec.excludes):
        if exc.bodyname1 not in keep_bodies or exc.bodyname2 not in keep_bodies:
            spec.delete(exc)
    for b in list(spec.worldbody.bodies):
        if b.name not in keep_bodies:
            spec.delete(b)
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
    for j in list(spec.joints):
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(j)  # the component fixture, not an execution constraint
    spec.compile()
    Path(out).write_text(spec.to_xml())


def independent_mechanism_trials(model, scene_path, semantic, manifest, base, tools, allow_missing_static=True):
    issues, trials = [], []
    entries = {str(o.get("id")): o for o in manifest.get("objects", []) if isinstance(o, dict)}
    for obj in semantic.get("objects", []):
        if obj.get("type") != "articulated":
            continue
        oid = str(obj.get("id")); rec = entries.get(oid)
        if rec is None and allow_missing_static and obj.get("role") == "static":
            continue
        row = {"id": oid, "ok": False, "independently_replayed": False, "scope": SCOPE,
               "supported_adapters": sorted(SUPPORTED), "verified_claims": []}
        trials.append(row)
        try:
            if not rec:
                raise ValueError("no reconstructed component entry")
            claimed = {k for k in (rec.get("verifiers") or {})
                       if (tools.get(k) or {}).get("category") == "Reconstruction Mechanism Tests"}
            row["claimed_mechanism_reports"] = sorted(claimed)
            unsupported = claimed - SUPPORTED
            row["unsupported_claims"] = sorted(unsupported)
            if unsupported:
                raise ValueError("no independent adapter for required mechanism report(s): " + ", ".join(sorted(unsupported)))
            if not claimed:
                raise ValueError("no supported required mechanism report was proposed")
            asset_path = (Path(base) / rec["asset_xml"]).resolve()
            asset = mujoco.MjModel.from_xml_path(str(asset_path))
            root, bodies, joints = _binding(asset, model, str(rec.get("root_body") or ""))
            joint_names = _supported_component(asset, model, bodies, joints)
            from agent.artifacts import model_digest
            row.update(scene_model_sha256=model_digest(model), asset_model_sha256=model_digest(asset),
                       asset_scene_physics_match=True, joints=joint_names,
                       fixed_root=model.body(root).name,
                       trial_defaults={"force_N": 10.0, "torque_Nm": 1.0, "seconds": 3.0,
                                       "min_fraction": 0.8, "root_slip_m": 0.005})
            with tempfile.TemporaryDirectory(prefix="dexagent_component_recheck_") as tmp:
                component = Path(tmp) / "component.xml"
                _extract(scene_path, model, bodies, component)
                row["component_model_sha256"] = model_digest(mujoco.MjModel.from_xml_path(str(component)))
                from tool_library.verifiers.articulated_joint_travel_test import articulated_joint_travel_test
                result = articulated_joint_travel_test(str(component), joints=joint_names, hold_root=False)
            row.update(independently_replayed=True, report=result, ok=result.get("ok") is True)
            if row["ok"]:
                row["verified_claims"] = sorted(claimed)
            else:
                raise ValueError("independent component mobility failed: " + str(result.get("verdict")))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            issues.append(f"$.objects[{oid}].verifiers: {row['error']}; embedded ok:true is not physical evidence")
    return {"ok": not issues, "issues": issues, "trials": trials,
            "coverage": "articulated components only; explicit supported claims independently replayed; other report categories are not certified here"}
