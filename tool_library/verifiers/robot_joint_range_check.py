"""Robot joint range check: every robot joint keeps the setup's limits."""
from __future__ import annotations

TOOL = {
    "name": "robot_joint_range_check",
    "title": "Robot joint range check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Every robot joint of a scene (arm, wrist and finger joints) is limited exactly as in the setup's reference robot; a widened, narrowed or unlimited range is reported with both values.",
    "description": (
        "A trajectory that only works because the scene's arm joints were given wider ranges than the real robot has "
        "(the hand-built giftbox twin widened its arm ranges to fit its trajectory) cannot be executed on the robot. "
        "The reference robot comes from a setup folder (default: the repository's setup/scene.xml, whose arm joints 2 "
        "and 3 have the documented lower limit -0.15 rad). Each reference robot joint is matched to the scene by name "
        "(exact, attach-prefix suffix or the same tokens without 'robot'); its 'limited' flag and both limits must "
        "equal the reference within `tol`. Missing robot joints are reported too. Only the robot is judged: object "
        "joints are checked by joint_range_check."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "setup", "type": "str | None", "description": "Setup folder (or its setup.yaml / scene.xml). Default: the repository's setup/."},
        {"name": "tol", "type": "float", "description": "Allowed difference per limit (rad or m). Default 1e-6."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; n_joints compared; changed [{joint, reference [lo, hi], scene [lo, hi], limited_ref, limited_scene, widened_by}]; missing; worst (largest widening, rad or m)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "anti-tampering", "robot", "joint", "range", "limits", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "robot_joint_range_check",
}

import mujoco

from tool_library.common import scene_assert as SA


def robot_joint_range_check(scene, setup=None, tol=1e-6):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    from tool_library.verifiers._reference import load_reference
    st = load_reference(model, setup)
    ref = st["model"]
    jmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_JOINT)
    changed, missing = [], []
    worst = 0.0
    for rj, sj in jmap.items():
        name = ref.joint(rj).name
        if sj < 0:
            missing.append(name)
            continue
        if int(ref.jnt_type[rj]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        lr, ls = bool(ref.jnt_limited[rj]), bool(model.jnt_limited[sj])
        rr, rs = [float(x) for x in ref.jnt_range[rj]], [float(x) for x in model.jnt_range[sj]]
        if lr != ls or (lr and (abs(rr[0] - rs[0]) > tol or abs(rr[1] - rs[1]) > tol)):
            widened = (max(0.0, rr[0] - rs[0]) + max(0.0, rs[1] - rr[1])) if ls else float("inf")
            worst = max(worst, widened)
            changed.append({"joint": name, "reference": [round(x, 6) for x in rr], "scene": [round(x, 6) for x in rs],
                            "limited_ref": lr, "limited_scene": ls, "widened_by": round(widened, 6) if widened != float("inf") else "unlimited"})
    issues = []
    for c in changed[:12]:
        if not c["limited_scene"] and c["limited_ref"]:
            issues.append(f"robot joint {c['joint']} is unlimited in the scene but limited to {c['reference']} in the setup; restore the setup's range")
        else:
            how = "widened" if c["widened_by"] not in (0.0, "unlimited") else "changed"
            issues.append(f"robot joint {c['joint']} range is {c['scene']} in the scene but {c['reference']} in the setup ({how}"
                          f"{' by %.4g' % c['widened_by'] if how == 'widened' else ''}); restore the setup's range: the real robot cannot go there")
    if len(changed) > 12:
        issues.append(f"... and {len(changed) - 12} more robot joints with changed ranges")
    if missing:
        issues.append(f"robot joints of the setup missing from the scene: {', '.join(missing[:8])}; rebuild the scene from the setup's robot")
    return SA.report(issues, n_joints=len(jmap), changed=changed, missing=missing,
                     worst=("unlimited" if worst == float("inf") else round(worst, 6)), setup=str(st["dir"]))
