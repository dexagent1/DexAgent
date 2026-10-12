"""Initial keyframe state check: the scene starts from a state it can hold, with the robot at the setup's home."""
from __future__ import annotations

TOOL = {
    "name": "initial_keyframe_state_check",
    "title": "Initial keyframe state check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Keyframe 0 exists with finite values of the right sizes, puts every robot joint at the setup's home pose (wrist positions moved with their station), and the objects are settled at it: simulated briefly with the robot held still, no object may drift, tilt or open beyond a few millimetres or degrees.",
    "description": (
        "Stage 3 and Stage 4 start every rollout from keyframe 0. A scene without a keyframe starts the robot at "
        "qpos0 (arms folded, hands at the origin) and the objects wherever the XML put them; a keyframe whose objects "
        "are not at rest (the hand-built cup example starts tilted and settles 47 mm) makes the first frames of every "
        "trajectory a settling transient that Stage 3 mistakes for its own action. The check: (1) at least one "
        "keyframe, with key_qpos (nq), key_ctrl (nu) and mocap entries finite; (2) every robot joint matched to the "
        "setup (default: the repository's setup/) equals the setup's home within tol_joint, the hands' wrist slides "
        "allowed to move by their station root's displacement from the setup (stage_station_integrity_check decides "
        "whether that displacement is legitimate); (3) servo targets in key_ctrl that differ from the joint values they "
        "drive are reported as a warning (the executor overwrites ctrl; a plain reset-and-step drives the hand to "
        "them), or as an issue with ctrl_must_match; (4) from keyframe 0 the scene is stepped for settle_s seconds "
        "with every robot joint clamped at its home value, and each object with a free joint must stay within "
        "drift_tol_m and tilt_tol_deg, each object hinge/slide within joint_tol of its keyframe value (half of it warns). "
        "A bad acceleration that makes MuJoCo reset the simulation (a part away from the weld or joint holding it) fails."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "setup", "type": "str | None", "description": "Setup folder holding the home keyframe. Default: the repository's setup/."},
        {"name": "settle_s", "type": "float", "description": "Seconds simulated with the robot held. Default 1.0."},
        {"name": "drift_tol_m", "type": "float", "description": "Largest object translation while settling, metres. Default 0.005."},
        {"name": "tilt_tol_deg", "type": "float", "description": "Largest object rotation while settling, degrees. Default 3."},
        {"name": "joint_tol", "type": "float", "description": "Largest object hinge (rad) / slide (m) change while settling: [hinge, slide] or one number. Default [0.25, 0.01] (a falling lid moves far more; a flap sagging onto its stop less); half of it is reported as a warning."},
        {"name": "tol_joint", "type": "float", "description": "Allowed robot joint difference from home, rad or m. Default 1e-4."},
        {"name": "ctrl_must_match", "type": "bool", "description": "Fail (not warn) when key_ctrl servo targets differ from the keyframe's joints. Default False."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; warnings; keyframe {n, name, nq, nu}; home {compared, off [{joint, scene, home}]}; ctrl_mismatch [{actuator, ctrl, joint_value}]; settle {object: {drift_m, tilt_deg}}; joints {joint: change}; worst (largest drift, m)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "keyframe", "initial state", "settle", "home", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "initial_keyframe_state_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA


def initial_keyframe_state_check(scene, setup=None, settle_s=1.0, drift_tol_m=0.005, tilt_tol_deg=3.0, joint_tol=(0.25, 0.01),
                                 tol_joint=1e-4, ctrl_must_match=False):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    from tool_library.verifiers._reference import load_reference
    st = load_reference(model, setup)
    ref, cfg = st["model"], st["cfg"]
    jt_hinge, jt_slide = (joint_tol, joint_tol) if isinstance(joint_tol, (int, float)) else (float(joint_tol[0]), float(joint_tol[1]))
    issues, warnings = [], []
    has_key = model.nkey > 0
    if not has_key:
        # judged from qpos0, where a scene without a keyframe starts; the robot's home cannot be judged
        issues.append("the scene has no keyframe: the robot starts at qpos0 and the objects wherever the XML put them; write keyframe 0 "
                      "with the setup's home pose and the objects at rest (as the setup's scene.xml has it)")
    kq = np.array(model.key_qpos[0] if has_key else model.qpos0, float)
    kc = np.array(model.key_ctrl[0] if has_key else np.zeros(model.nu), float)
    kinfo = {"n": int(model.nkey), "name": model.key(0).name if has_key else None, "nq": int(kq.size), "nu": int(kc.size)}
    if kq.size != model.nq or kc.size != model.nu:
        issues.append(f"keyframe 0 has {kq.size} qpos / {kc.size} ctrl values for a model with nq {model.nq} / nu {model.nu}")
    if has_key and not (np.isfinite(kq).all() and np.isfinite(kc).all() and np.isfinite(model.key_mpos[0]).all() and np.isfinite(model.key_mquat[0]).all()):
        issues.append("keyframe 0 holds non-finite values; rewrite it")
        return SA.report(issues, warnings=warnings, keyframe=kinfo, worst=None)
    # robot at home (wrist slides follow their station)
    robot = SA.robot_body_ids(model, ref)
    q_home = SA.ref_home_qpos(ref, cfg)
    home = {"compared": 0, "off": []}
    if q_home is not None and has_key:
        bmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_BODY)
        jmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_JOINT)
        shift = {}
        for side, rname in (SA.station_root_names(cfg) or {s: f"robot_{s}_arm_root" for s in ("left", "right")}).items():
            rb = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, rname)
            sb = bmap.get(rb, -1) if rb >= 0 else -1
            if sb >= 0:
                shift[side] = np.array(model.body_pos[sb], float) - np.array(ref.body_pos[rb], float)
        for rj, sj in jmap.items():
            if sj < 0 or int(ref.jnt_type[rj]) in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
                continue
            jn = ref.joint(rj).name
            want = float(q_home[int(ref.jnt_qposadr[rj])])
            for side, d in shift.items():
                for ax, k in (("x", 0), ("y", 1), ("z", 2)):
                    if jn.endswith(f"{side}_{side}_pos_{ax}"):
                        want += float(d[k])
            got = float(kq[int(model.jnt_qposadr[sj])])
            home["compared"] += 1
            if abs(got - want) > tol_joint:
                home["off"].append({"joint": jn, "scene": round(got, 6), "home": round(want, 6)})
        if home["off"]:
            o = home["off"]
            issues.append(f"keyframe 0 puts {len(o)} robot joint(s) off the setup's home: "
                          + ", ".join(f"{x['joint']} {x['scene']} (home {x['home']})" for x in o[:5]) + (" ..." if len(o) > 5 else "")
                          + "; copy the robot part of the setup's home keyframe")
    # servo targets vs the joints they drive
    mism = []
    for a in (range(model.nu) if has_key else []):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT) or int(model.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
            continue
        j = int(model.actuator_trnid[a][0])
        if int(model.jnt_bodyid[j]) not in robot:
            continue
        v = float(kq[int(model.jnt_qposadr[j])])
        if abs(float(kc[a]) - v) > 1e-3:
            mism.append({"actuator": model.actuator(a).name, "ctrl": round(float(kc[a]), 5), "joint_value": round(v, 5)})
    if mism:
        worst_c = max(abs(m["ctrl"] - m["joint_value"]) for m in mism)
        msg = (f"keyframe 0's ctrl gives {len(mism)} robot servo(s) targets away from the joints they drive (up to {worst_c:.3g}; "
               f"e.g. {mism[0]['actuator']} {mism[0]['ctrl']} vs {mism[0]['joint_value']}): a reset-and-step without setting ctrl drives the "
               f"hand there; set key ctrl to the keyframe's joint values")
        (issues if ctrl_must_match else warnings).append(msg)
    # settle with the robot held
    data = mujoco.MjData(model)
    if has_key:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    free = [int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
            and int(model.jnt_bodyid[j]) not in robot]
    obj_joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) not in robot
                  and int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
    p0 = {b: data.xpos[b].copy() for b in free}
    R0 = {b: data.xmat[b].reshape(3, 3).copy() for b in free}
    j0 = {j: float(data.qpos[int(model.jnt_qposadr[j])]) for j in obj_joints}
    settle, joints = {}, {}
    worst = 0.0
    try:
        SA.settle_robot_held(model, data, settle_s, robot)
    except FloatingPointError as e:
        issues.append(f"the simulation fails while settling from keyframe 0 ({e}): a part starts off the joint or weld that holds it "
                      f"(a cap away from its thread's carrier), or deep inside another object or the table; re-place it in keyframe 0")
        return SA.report(issues, warnings=warnings, keyframe=kinfo, home=home, ctrl_mismatch=mism, settle={}, joints={}, worst=None)
    from tool_library.common import scene as S
    for b in free:
        dp = float(np.linalg.norm(data.xpos[b] - p0[b]))
        ang = S.rotation_angle_deg(R0[b], data.xmat[b])
        settle[model.body(b).name] = {"drift_m": round(dp, 5), "tilt_deg": round(ang, 3)}
        worst = max(worst, dp)
        if dp > drift_tol_m or ang > tilt_tol_deg:
            issues.append(f"object {model.body(b).name} is not at rest in {'keyframe 0' if has_key else 'its initial state'}: it moves {dp * 1000:.1f} mm and turns {ang:.1f} deg in "
                          f"{settle_s:g} s with the robot still; place it resting on its support (upright, touching, not interpenetrating) "
                          f"and write that pose into keyframe 0")
    for j in obj_joints:
        dv = float(data.qpos[int(model.jnt_qposadr[j])]) - j0[j]
        tol = jt_hinge if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) else jt_slide
        joints[model.joint(j).name] = round(dv, 5)
        if tol / 2 < abs(dv) <= tol:
            warnings.append(f"object joint {model.joint(j).name} moves {dv:+.4f} while settling from keyframe 0 (its rest is a little away from the keyframe)")
        if abs(dv) > tol:
            issues.append(f"object joint {model.joint(j).name} moves {dv:+.4f} {'rad' if tol == jt_hinge else 'm'} while settling from keyframe 0: "
                          f"its keyframe value is not a rest state (a lid that falls, a drawer that slides); set the rest value in keyframe 0")
    return SA.report(issues, warnings=warnings, keyframe=kinfo, home=home, ctrl_mismatch=mism, settle=settle, joints=joints,
                     worst=round(worst, 5))
