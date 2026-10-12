"""Validate a supported MuJoCo deformable model and its unforced resting state."""
from __future__ import annotations

TOOL = {
    "name": "deformable_rest_check", "title": "Deformable model and rest check",
    "type": "verifier", "category": "Scene Assertions",
    "summary": "Independently simulates every reconstructed cable at rest and checks all links for drift, penetration and invalid dynamics; rejects unsupported deformable representations.",
    "description": "The MuJoCo cable path (a cord that a subgoal cuts, built with construct_cuttable_rope, and older MuJoCo cable reconstructions) supports flexible cable chains, including declared branches and severable joins. It does not certify a rigid proxy as cloth or a volumetric soft body. Topology is checked separately. Deformables simulated in Isaac Sim (entries with simulator 'isaac') are left out here: isaac_deformable_rest_check re-measures them from their cord specification and settled poses.",
    "inputs": [
        {"name": "scene", "type": "str", "description": "Placed scene MJCF."},
        {"name": "recon", "type": "str | dict", "description": "Scene reconstruction manifest."},
        {"name": "seconds", "type": "float", "description": "Unforced settling duration. Default 1."},
        {"name": "max_mean_drift_m", "type": "float", "description": "Maximum mean link drift. Default .02."},
        {"name": "max_link_drift_m", "type": "float", "description": "Maximum single link drift. Default .05."},
    ],
    "outputs": [{"name": "report", "type": "dict", "description": "ok, issues, objects with independently measured link drift and penetration; scope=mujoco_cable_simulation."}],
    "entrypoint": "deformable_rest_check", "status": "tested",
    "requirements": ["mujoco>=3.2", "numpy"], "simulators": ["MuJoCo"],
    "tags": ["deformable", "cable", "rest", "stage2", "simulation"],
}

import json
from pathlib import Path
import mujoco
import numpy as np
from tool_library.common import scene_assert as SA


def deformable_rest_check(scene, recon, seconds=1.0, max_mean_drift_m=.02, max_link_drift_m=.05):
    if not np.isfinite(seconds) or seconds < .5:
        raise ValueError("at least 0.5 seconds of simulation is required")
    man = json.loads(Path(recon).read_text()) if isinstance(recon, (str, Path)) else recon
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    groups = {g['id']: (g, links) for g, links in SA.rope_groups(model, man, None, 5)}
    issues, rows, initial, all_links = [], {}, {}, set()
    for obj in man.get('objects', []):
        if obj.get('type') != 'deformable' or SA.is_isaac_only(obj):
            continue                                  # a cord simulated in Isaac Sim: isaac_deformable_rest_check
        oid = obj.get('id')
        declared = obj.get('simulation_model') or {}
        if declared.get('kind') != 'mujoco_cable':
            issues.append(f"{oid}: declare simulation_model.kind=mujoco_cable for a supported cable; cloth and volumetric soft bodies require a separately verified model")
        g, links = groups.get(oid, ({}, []))
        geoms = SA.body_geoms(model, links)
        physical = [x for x in geoms if SA.geom_collidable(model, x)]
        supported = {int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE)}
        flexible = sum(int(model.body_jntnum[b]) > 0 for b in links)
        if len(links) < 5 or flexible < len(links) - 1 or not physical or any(int(model.geom_type[x]) not in supported for x in physical):
            issues.append(f"{oid}: no supported flexible cable chain (links={len(links)}, jointed={flexible}); a rigid proxy is not a deformable simulation")
            continue
        initial[oid] = (links, data.xpos[links].copy())
        all_links.update(links)
        rows[oid] = {'links': len(links), 'mean_drift_m': 0., 'max_link_drift_m': 0.}
    penetration = 0.
    for _ in range(int(np.ceil(seconds / model.opt.timestep))):
        before = float(data.time)
        mujoco.mj_step(model, data)
        if data.time <= before or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(int(w.number) for w in data.warning):
            issues.append('rest simulation produced a warning, non-finite state or reset')
            break
        for oid, (links, positions) in initial.items():
            drift = np.linalg.norm(data.xpos[links] - positions, axis=1)
            rows[oid]['mean_drift_m'] = max(rows[oid]['mean_drift_m'], float(drift.mean()))
            rows[oid]['max_link_drift_m'] = max(rows[oid]['max_link_drift_m'], float(drift.max()))
        for c in data.contact[:data.ncon]:
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if a not in all_links and b not in all_links:
                continue
            if a == b or int(model.body_parentid[a]) == b or int(model.body_parentid[b]) == a:
                continue
            penetration = max(penetration, -float(c.dist))
    for oid, row in rows.items():
        if row['mean_drift_m'] > max_mean_drift_m or row['max_link_drift_m'] > max_link_drift_m:
            issues.append(f"{oid}: cable does not retain its reconstructed resting shape: {row}")
    if penetration > .003:
        issues.append(f"deformable rest penetration {penetration:.6f} m exceeds 0.003 m")
    return {'ok': not issues, 'issues': issues, 'objects': rows, 'max_penetration_m': penetration,
            'scope': 'mujoco_cable_simulation', 'isaac_verified': False}
