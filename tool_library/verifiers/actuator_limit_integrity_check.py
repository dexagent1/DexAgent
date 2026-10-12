"""Actuator limit integrity check: the robot's servos are as strong as the setup's, no stronger."""
from __future__ import annotations

TOOL = {
    "name": "actuator_limit_integrity_check",
    "title": "Actuator limit integrity check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Every robot actuator of a scene has the setup's transmission, gains, bias, control range, force range and gear, the robot joints keep their actuator force caps, and no extra actuator drives a robot joint.",
    "description": (
        "Stronger fingers or wrists make any grasp hold: a finger servo with ten times the gain, a widened force range "
        "or a second actuator on the same joint squeeze an object no real hand could. The reference actuators come from "
        "a setup folder (default: the repository's setup/scene.xml). Each reference robot actuator is matched to the "
        "scene by name; its target joint (by name), gain/bias/dynamics types and parameters, gear, ctrllimited/ctrlrange, "
        "forcelimited/forcerange and actlimited/actrange must equal the reference within a relative tolerance. Each "
        "robot joint's actuatorfrcrange (jnt_actfrclimited, jnt_actfrcrange) must equal the reference, and any scene "
        "actuator not in the reference whose transmission acts on a robot joint, body, site or tendon is reported as "
        "extra. The executor's own runtime caps (limit_hand_forces) are applied later and are not part of the scene."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "setup", "type": "str | None", "description": "Setup folder (or its setup.yaml / scene.xml). Default: the repository's setup/."},
        {"name": "rel_tol", "type": "float", "description": "Allowed relative difference of every parameter. Default 1e-6."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; n_actuators compared; changed [{actuator, field, reference, scene}]; missing; extra (actuators on robot joints the setup does not have); joint_caps_changed; worst (largest gain or force-range ratio scene/reference)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "anti-tampering", "robot", "actuator", "force", "gain", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "actuator_limit_integrity_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA

FIELDS = (("actuator_trntype", "transmission type"), ("actuator_gaintype", "gain type"), ("actuator_biastype", "bias type"),
          ("actuator_dyntype", "dynamics type"), ("actuator_gainprm", "gain"), ("actuator_biasprm", "bias"),
          ("actuator_dynprm", "dynamics"), ("actuator_gear", "gear"), ("actuator_ctrllimited", "ctrllimited"),
          ("actuator_ctrlrange", "ctrlrange"), ("actuator_forcelimited", "forcelimited"), ("actuator_forcerange", "forcerange"),
          ("actuator_actlimited", "actlimited"), ("actuator_actrange", "actrange"))


def _target(model, a):
    t = int(model.actuator_trntype[a]); i = int(model.actuator_trnid[a][0])
    objtype = {int(mujoco.mjtTrn.mjTRN_JOINT): mujoco.mjtObj.mjOBJ_JOINT, int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): mujoco.mjtObj.mjOBJ_JOINT,
               int(mujoco.mjtTrn.mjTRN_BODY): mujoco.mjtObj.mjOBJ_BODY, int(mujoco.mjtTrn.mjTRN_SITE): mujoco.mjtObj.mjOBJ_SITE,
               int(mujoco.mjtTrn.mjTRN_TENDON): mujoco.mjtObj.mjOBJ_TENDON}.get(t)
    return (objtype, i, mujoco.mj_id2name(model, objtype, i) if objtype is not None and i >= 0 else None)


def _ratio(a, b):
    a = np.abs(np.asarray(a, float)).max(); b = np.abs(np.asarray(b, float)).max()
    return float(a / b) if b > 1e-12 else (float("inf") if a > 1e-12 else 1.0)


def actuator_limit_integrity_check(scene, setup=None, rel_tol=1e-6):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    from tool_library.verifiers._reference import load_reference
    st = load_reference(model, setup)
    ref = st["model"]
    amap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_ACTUATOR)
    jmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_JOINT)
    robot_bodies = SA.robot_body_ids(model, ref)
    robot_joints = {j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in robot_bodies}
    changed, missing, extra, caps = [], [], [], []
    worst = 1.0
    for ra, sa in amap.items():
        name = ref.actuator(ra).name
        if sa < 0:
            missing.append(name)
            continue
        rt, st_ = _target(ref, ra), _target(model, sa)
        # the target must be the scene's counterpart of the reference target
        if rt[0] == mujoco.mjtObj.mjOBJ_JOINT:
            want = jmap.get(rt[1], -1)
            ok_t = st_[0] == mujoco.mjtObj.mjOBJ_JOINT and st_[1] == want
        else:
            ok_t = st_[2] == rt[2]
        if not ok_t:
            changed.append({"actuator": name, "field": "target", "reference": rt[2], "scene": st_[2]})
        for attr, field in FIELDS:
            a, b = getattr(model, attr)[sa], getattr(ref, attr)[ra]
            if SA.differs(a, b, 1e-9, rel_tol):
                changed.append({"actuator": name, "field": field, "reference": SA.r6(b), "scene": SA.r6(a)})
                if field in ("gain", "bias", "forcerange", "gear"):
                    worst = max(worst, _ratio(a, b))
    mapped = {s for s in amap.values() if s >= 0}
    for a in range(model.nu):
        if a in mapped:
            continue
        objtype, i, nm = _target(model, a)
        on_robot = ((objtype == mujoco.mjtObj.mjOBJ_JOINT and i in robot_joints)
                    or (objtype == mujoco.mjtObj.mjOBJ_BODY and i in robot_bodies)
                    or (objtype == mujoco.mjtObj.mjOBJ_SITE and int(model.site_bodyid[i]) in robot_bodies)
                    or (objtype == mujoco.mjtObj.mjOBJ_TENDON and (nm or "").startswith(SA.ROBOT_PREFIX))
                    or (model.actuator(a).name or "").startswith(SA.ROBOT_PREFIX))
        if on_robot:
            extra.append(f"{model.actuator(a).name or f'actuator {a}'} -> {nm}")
    for rj, sj in jmap.items():
        if sj < 0:
            continue
        if (bool(model.jnt_actfrclimited[sj]) != bool(ref.jnt_actfrclimited[rj])
                or SA.differs(model.jnt_actfrcrange[sj], ref.jnt_actfrcrange[rj], 1e-9, rel_tol)):
            caps.append({"joint": ref.joint(rj).name, "reference": [bool(ref.jnt_actfrclimited[rj])] + SA.r6(ref.jnt_actfrcrange[rj]),
                         "scene": [bool(model.jnt_actfrclimited[sj])] + SA.r6(model.jnt_actfrcrange[sj])})
            if model.jnt_actfrclimited[sj]:
                worst = max(worst, _ratio(model.jnt_actfrcrange[sj], ref.jnt_actfrcrange[rj]))
            elif ref.jnt_actfrclimited[rj]:
                worst = float("inf")
    issues = []
    for c in changed[:12]:
        issues.append(f"actuator {c['actuator']}: {c['field']} is {c['scene']} in the scene but {c['reference']} in the setup; "
                      f"restore the setup's value (a stronger or retargeted servo makes grasps hold that the robot cannot)")
    if len(changed) > 12:
        issues.append(f"... and {len(changed) - 12} more changed actuator parameters")
    if missing:
        issues.append(f"robot actuators of the setup missing from the scene: {', '.join(missing[:8])}")
    if extra:
        issues.append(f"actuators the setup does not have drive the robot: {', '.join(extra[:6])}; remove them (an extra servo on a joint doubles its strength)")
    for c in caps[:6]:
        issues.append(f"robot joint {c['joint']}: actuator force cap (limited, lo, hi) is {c['scene']} in the scene but {c['reference']} in the setup; restore it")
    return SA.report(issues, n_actuators=len(amap), changed=changed, missing=missing, extra=extra, joint_caps_changed=caps,
                     worst=("inf" if worst == float("inf") else round(worst, 4)), setup=str(st["dir"]))
