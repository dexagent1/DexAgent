"""Close a posed hand on the handle of a part that slides on a prismatic joint (a drawer, a tray, a slider) and pull or push it along the joint axis in physics."""
from __future__ import annotations

TOOL = {
    "name": "grasp_slide_pull_test",
    "title": "Grasp slide-pull test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close the posed hand on a part that slides on a prismatic joint (drawer, tray, slider), translate the wrist along the joint axis by a set distance in physics, and pass when the part followed the hand, the pads stayed on it and no sibling joint of the same object was dragged along.",
    "description": (
        "The slide counterpart of grasp_turn_test. The hand scene is the floating-hand scene of hand_scene_from_recon "
        "(the last object holds the joint). The object's root is held still (as a brace or the bench would hold it); "
        "the open hand descends along the pose's approach axis onto the pose, closes to the pose's finger targets, "
        "presses to its squeezed targets, then the wrist is translated in steps along the joint axis in world frame "
        "(sign from `direction`: +1 moves the joint toward its upper limit) by `travel_m` with the fingers fixed. The "
        "verdict compares the joint's travel with the wrist's: PULLED when the fraction reaches `min_fraction` and at "
        "least one hand geom still touches the part at the end; SLIPPED / LOST / STALLED otherwise. Every other slide or "
        "hinge joint of the same object is watched: if one moves more than `sibling_tol` (m or rad) the hand caught the "
        "wrong part (a neighbouring drawer's handle) and the test fails with its name. Peak wrist force is reported "
        "against the hand's force cap."
    ),
    "inputs": [
        {"name": "scene", "type": "dict", "description": "Floating-hand scene {hand_xml, side, objects, floor, joint_init, grasp_body}; the last object holds the joint."},
        {"name": "pose", "type": "dict", "description": "Hand pose {pos, euler, finger_targets, finger_targets_squeezed?, grip_fingers?, approach?}."},
        {"name": "joint", "type": "str", "description": "Name of the slide joint to pull in the compiled hand scene (the bench name usually works; a suffix match is tried)."},
        {"name": "travel_m", "type": "float", "description": "Distance the wrist moves along the joint axis. Default 0.08."},
        {"name": "direction", "type": "int", "description": "+1 toward the joint's upper limit (open), -1 toward the lower. Default +1."},
        {"name": "seconds", "type": "float", "description": "Duration of the stroke. Default 1.5."},
        {"name": "min_fraction", "type": "float", "description": "Joint travel / wrist travel needed to pass. Default 0.8."},
        {"name": "sibling_tol", "type": "float", "description": "Largest allowed motion of any other joint of the same object. Default 0.01."},
        {"name": "hold_root", "type": "bool", "description": "Hold the object's free root still during the test. Default true."},
        {"name": "approach_m", "type": "float", "description": "Pre-grasp distance along the approach axis. Default 0.06."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, verdict, measured {travel_m, joint_moved_m, fraction, joint_start, joint_end, pads_start, pads_end, lost_at_m, siblings_moved {joint: value}, peak_wrist_force_n}."},
    ],
    "status": "tested",
    "entrypoint": "grasp_slide_pull_test",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "drawer", "slide", "prismatic", "pull", "stress-test"],
    "used_in": [],
}

import mujoco
import numpy as np

from tool_library.common import hand as H


def _find_joint(model, joint):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid >= 0:
        return jid
    cands = [j for j in range(model.njnt) if (model.joint(j).name or "").endswith(joint) or joint.endswith(model.joint(j).name or "\0")]
    if len(cands) == 1:
        return cands[0]
    # longest common suffix
    best, bl = -1, 0
    for j in range(model.njnt):
        n = model.joint(j).name or ""
        k = 0
        while k < min(len(n), len(joint)) and n[-1 - k] == joint[-1 - k]:
            k += 1
        if k > bl:
            best, bl = j, k
    if best >= 0 and bl >= 8:
        return best
    return -1


def grasp_slide_pull_test(scene, pose, joint, travel_m=0.08, direction=1, seconds=1.5, min_fraction=0.8, sibling_tol=0.01,
                          hold_root=True, approach_m=0.06):
    if travel_m <= 0:
        raise ValueError("travel_m must be positive; use direction=-1 to push")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    for k in ("pos", "euler", "finger_targets"):
        if k not in pose:
            raise ValueError(f"pose is missing {k!r}")
    model, data0, rig, ids = H.open_scene(scene)
    jid = _find_joint(model, joint)
    if jid < 0:
        raise ValueError(f"no joint {joint!r}; slide joints: {[model.joint(j).name for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE)]}")
    if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
        raise ValueError(f"joint {model.joint(jid).name!r} is not a slide joint; use grasp_turn_test for hinges")
    qa = int(model.jnt_qposadr[jid]); part = int(model.jnt_bodyid[jid])
    root = part
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])
    sub_root = set(); st = [root]
    while st:
        b = st.pop(); sub_root.add(b); st += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b]
    free_adr = None
    siblings = {}
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) not in sub_root:
            continue
        t = int(model.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE) and int(model.jnt_bodyid[j]) == root:
            free_adr = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
        elif j != jid and t in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            siblings[model.joint(j).name] = int(model.jnt_qposadr[j])
    sub = set(); st = [part]
    while st:
        b = st.pop(); sub.add(b); st += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b]
    part_gids = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub}

    d = mujoco.MjData(model); d.qpos[:] = data0.qpos; d.qvel[:] = 0
    q_free0 = d.qpos[free_adr[0]:free_adr[0] + 7].copy() if free_adr else None
    clamp = H.executor_clamp(rig, pose)
    closed, press, grip = clamp["closed"], clamp["press"], clamp["grip_fingers"]
    own_open = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    opened = own_open or H.approach_open_targets(rig, closed, float(pose.get("open_frac", .55)), grip, clamp["grip_posed"])
    pos = np.asarray(pose["pos"], float); eul = np.asarray(pose["euler"], float)
    dt = float(model.opt.timestep)
    peak = [0.0]

    def hold():
        if hold_root and free_adr is not None:
            d.qpos[free_adr[0]:free_adr[0] + 7] = q_free0; d.qvel[free_adr[1]:free_adr[1] + 6] = 0

    def pads():
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if b1 in rig.body_ids and c.geom2 in part_gids:
                hb = b1
            elif b2 in rig.body_ids and c.geom1 in part_gids:
                hb = b2
            else:
                continue
            out.add(H.finger_of_joint(model.body(hb).name.split(f"{rig.side}_")[-1]) or "palm")
        return sorted(out)

    def run(p, e, f, secs):
        for _ in range(max(1, int(secs / dt))):
            H.hold_ctrl(d, rig, p, e, f); mujoco.mj_step(model, d); hold()
            peak[0] = max(peak[0], float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("x", "y", "z")])))

    sib0 = {n: float(d.qpos[a]) for n, a in siblings.items()}
    a = H.approach_axis(model, d, rig, pose)
    rig.teleport(d, pos + a * approach_m, eul); rig.set_fingers(d, opened); mujoco.mj_forward(model, d)
    run(pos + a * approach_m, eul, opened, 0.3)
    H.ramp(model, d, rig, pos + a * approach_m, pos, eul, eul, opened, opened, 0.5); hold()
    H.ramp(model, d, rig, pos, pos, eul, eul, opened, closed, 0.4); hold()
    pressed_pos = pos + (np.asarray(clamp["wrist_press"], float) if clamp["wrist_press"] is not None else 0.)
    H.ramp(model, d, rig, pos, pressed_pos, eul, eul, closed, press, .3, lambda *_: hold())
    pos = pressed_pos
    run(pos, eul, press, .3)
    mujoco.mj_forward(model, d)
    q0 = float(d.qpos[qa]); pads0 = pads()
    axis = d.xmat[part].reshape(3, 3) @ model.jnt_axis[jid]
    axis = axis / np.linalg.norm(axis) * float(direction)
    n = max(1, int(np.ceil(travel_m / 0.005)))
    lost_at = None
    p_cur = pos.copy()
    for i in range(n):
        p_new = pos + axis * travel_m * (i + 1) / n
        for _ in range(max(1, int(seconds / n / dt))):
            H.hold_ctrl(d, rig, p_new, eul, press); mujoco.mj_step(model, d); hold()
            peak[0] = max(peak[0], float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("x", "y", "z")])))
        p_cur = p_new
        if lost_at is None and not pads():
            lost_at = round(travel_m * (i + 1) / n, 4)
    run(p_cur, eul, press, 0.3)
    moved = float(d.qpos[qa] - q0) * float(direction)
    frac = moved / travel_m
    pads_end = pads()
    sib_moved = {nm: round(abs(float(d.qpos[adr]) - sib0[nm]), 4) for nm, adr in siblings.items()}
    bad_sib = {k: v for k, v in sib_moved.items() if v > sibling_tol}
    issues = []
    if frac < min_fraction:
        if moved < 0.1 * travel_m:
            verdict = f"STALLED: the joint moved {moved*1000:.0f} mm while the wrist moved {travel_m*1000:.0f} mm"
            issues.append(verdict + "; the pads are not behind the handle or the hand caught another part: pick a pose whose fingers hook the handle on the far side")
        elif not pads_end:
            verdict = f"LOST: the hand came off the part after {lost_at} m (joint moved {moved*1000:.0f} mm)"
            issues.append(verdict + "; grip deeper behind the handle or use more fingers")
        else:
            verdict = f"SLIPPED: joint moved {moved*1000:.0f} of {travel_m*1000:.0f} mm ({100*frac:.0f}%)"
            issues.append(verdict + "; the pads skate along the pull: hook the handle instead of pinching its face")
    else:
        verdict = f"PULLED: the part followed {moved*1000:.0f} of {travel_m*1000:.0f} mm ({100*frac:.0f}%), held by {pads_end}"
    if not pads_end and frac >= min_fraction:
        issues.append("no hand geom touches the part at the end of the stroke")
    if bad_sib:
        issues.append(f"sibling joints dragged along: {bad_sib}; the hand catches a neighbouring part, move the pose away from it")
    return {"ok": not issues, "issues": issues, "verdict": verdict, "measured": {
        "travel_m": float(travel_m), "joint_moved_m": round(moved, 4), "fraction": round(frac, 3),
        "joint": model.joint(jid).name, "joint_start": round(q0, 4), "joint_end": round(float(d.qpos[qa]), 4),
        "pads_start": pads0, "pads_end": pads_end, "lost_at_m": lost_at, "siblings_moved": sib_moved,
        "peak_wrist_force_n": round(peak[0], 1)}}
