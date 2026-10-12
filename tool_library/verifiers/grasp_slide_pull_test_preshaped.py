"""Pull (or push) a slide joint with a hand that arrives already shaped: a hook or pinch on a handle that has to go
over the bar and down behind it, so it cannot approach with its fingers spread open."""
from __future__ import annotations

TOOL = {
    "name": "grasp_slide_pull_test_preshaped",
    "title": "Preshaped grasp slide-pull test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Drive a posed hand onto the handle of a sliding part with its own pre-grasp finger shape and approach path (a hook that drops in behind a bar), close, translate the wrist along the joint axis in physics, and pass when the part followed the hand and stayed in contact.",
    "description": (
        "A fork of grasp_slide_pull_test for grasps whose open hand cannot reach the pose: a finger hooked behind a drawer "
        "bar standing a few centimetres proud of the front, where the default 55% opening lays the finger on the housing "
        "instead of behind the bar. The pre-grasp finger shape is the pose's `finger_targets_open` when present, otherwise "
        "the closed shape opened by `open_frac` (0 = arrive closed, 0.55 = the default opening of the other tests); the "
        "fingers that do not grip stay curled. The wrist starts `approach_m` back along `approach_dir` (default the pose's "
        "own approach axis), visits the `approach_waypoints` (world offsets from the pose, as the subgoal planner takes "
        "them) and lands on the pose, then the fingers close to the pose's targets and press to its squeezed targets. "
        "The object's free root is held still (the bench, or the other hand, holds the housing). The wrist is then "
        "translated along the joint axis (sign from `direction`) by `travel_m` in steps with the fingers fixed. Verdict as "
        "in grasp_slide_pull_test: PULLED when joint travel / wrist travel reaches `min_fraction` and a hand geom still "
        "touches the part at the end; STALLED / LOST / SLIPPED otherwise. It also reports where the hand first touched "
        "the part during the approach and the deepest hand penetration into any object on the way, so a path that "
        "scrapes the housing is visible. Assumes the floating-hand scene convention of hand_scene_from_recon (the last "
        "object holds the joint)."
    ),
    "inputs": [
        {"name": "scene", "type": "dict", "description": "Floating-hand scene {hand_xml, side, objects, floor, joint_init, grasp_body}; the last object holds the joint."},
        {"name": "pose", "type": "dict", "description": "Hand pose {pos, euler, finger_targets, finger_targets_squeezed?, finger_targets_open?, grip_fingers?, approach?, approach_waypoints?}."},
        {"name": "joint", "type": "str", "description": "Slide joint to pull (bench name or a unique suffix)."},
        {"name": "travel_m", "type": "float", "description": "Distance the wrist moves along the joint axis. Default 0.08."},
        {"name": "direction", "type": "int", "description": "+1 toward the joint's upper limit, -1 toward the lower. Default +1."},
        {"name": "open_frac", "type": "float", "description": "Pre-grasp opening of the grip fingers when the pose has no finger_targets_open, 0..1. Default 0.15."},
        {"name": "approach_dir", "type": "xyz | None", "description": "World direction from the pose back to the pre-grasp. Default the pose's approach axis."},
        {"name": "approach_m", "type": "float", "description": "Pre-grasp distance along the approach direction. Default 0.06."},
        {"name": "approach_waypoints", "type": "list[xyz] | None", "description": "World offsets from the pose visited in order after the pre-grasp. Default the pose's own, else none."},
        {"name": "seconds", "type": "float", "description": "Duration of the stroke. Default 1.5."},
        {"name": "min_fraction", "type": "float", "description": "Joint travel / wrist travel needed to pass. Default 0.8."},
        {"name": "sibling_tol", "type": "float", "description": "Largest allowed motion of any other joint of the same object. Default 0.01."},
        {"name": "hold_root", "type": "bool", "description": "Hold the object's free root still. Default true."},
        {"name": "max_pen_mm", "type": "float", "description": "Deepest allowed hand penetration into any object during approach and close. Default 4."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, verdict, measured {travel_m, joint_moved_m, fraction, joint_start, joint_end, pads_start, pads_end, lost_at_m, siblings_moved, peak_wrist_force_n, approach_pen_mm, approach_pen_pair, joint_moved_during_approach_m, joint_moved_during_close_m, open_frac, approach_dir, n_waypoints}."},
    ],
    "status": "tested",
    "entrypoint": "grasp_slide_pull_test_preshaped",
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "drawer", "slide", "hook", "handle", "pull", "stress-test", "preshaped"],
    "used_in": ["Place Toy in Drawer"],
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
    best, bl = -1, 0
    for j in range(model.njnt):
        n = model.joint(j).name or ""
        k = 0
        while k < min(len(n), len(joint)) and n[-1 - k] == joint[-1 - k]:
            k += 1
        if k > bl:
            best, bl = j, k
    return best if (best >= 0 and bl >= 8) else -1


def _subtree(model, b0):
    out, st = set(), [b0]
    while st:
        b = st.pop(); out.add(b); st += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
    return out


def grasp_slide_pull_test_preshaped(scene, pose, joint, travel_m=0.08, direction=1, open_frac=0.15, approach_dir=None,
                                    approach_m=0.06, approach_waypoints=None, seconds=1.5, min_fraction=0.8,
                                    sibling_tol=0.01, hold_root=True, max_pen_mm=4.0):
    if travel_m <= 0:
        raise ValueError("travel_m must be positive; use direction=-1 to push")
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    if not (0.0 <= float(open_frac) <= 1.0):
        raise ValueError("open_frac must lie in [0, 1]")
    if approach_m < 0:
        raise ValueError("approach_m must be >= 0")
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
    sub_root = _subtree(model, root)
    free_adr, siblings = None, {}
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) not in sub_root:
            continue
        t = int(model.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE) and int(model.jnt_bodyid[j]) == root:
            free_adr = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
        elif j != jid and t in (int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE)):
            siblings[model.joint(j).name] = int(model.jnt_qposadr[j])
    part_gids = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in _subtree(model, part)}

    d = mujoco.MjData(model); d.qpos[:] = data0.qpos; d.qvel[:] = 0
    q_free0 = d.qpos[free_adr[0]:free_adr[0] + 7].copy() if free_adr else None
    clamp = H.executor_clamp(rig, pose)
    closed, press, grip = clamp["closed"], clamp["press"], clamp["grip_fingers"]
    if pose.get("finger_targets_open"):
        opened = dict(closed); opened.update({k: float(v) for k, v in pose["finger_targets_open"].items() if k in rig.finger_range})
    else:
        opened = H.approach_open_targets(rig, closed, float(open_frac), grip, clamp["grip_posed"]) if open_frac > 0 else dict(closed)
    pos = np.asarray(pose["pos"], float); eul = np.asarray(pose["euler"], float)
    dt = float(model.opt.timestep)
    peak = [0.0]; pen = [0.0, None]

    def hold():
        if hold_root and free_adr is not None:
            d.qpos[free_adr[0]:free_adr[0] + 7] = q_free0; d.qvel[free_adr[1]:free_adr[1] + 6] = 0

    def scan():
        hit = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            h1, h2 = b1 in rig.body_ids, b2 in rig.body_ids
            if h1 == h2:
                continue
            hg, og = (c.geom1, c.geom2) if h1 else (c.geom2, c.geom1)
            if og in part_gids:
                hit.add(H.finger_of_joint(model.body(int(model.geom_bodyid[hg])).name.split(f"{rig.side}_")[-1]) or "palm")
            if -c.dist * 1000 > pen[0]:
                pen[0] = -c.dist * 1000
                pen[1] = [model.geom(hg).name, model.geom(og).name or model.body(int(model.geom_bodyid[og])).name]
        return sorted(hit)

    def step(p, e, f):
        H.hold_ctrl(d, rig, p, e, f); mujoco.mj_step(model, d); hold()
        peak[0] = max(peak[0], float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("x", "y", "z")])))

    if approach_dir is None:
        a = H.approach_axis(model, d, rig, pose)
    else:
        a = np.asarray(approach_dir, float)
        if a.shape != (3,) or np.linalg.norm(a) < 1e-9:
            raise ValueError("approach_dir must be a non-zero xyz vector")
    a = a / np.linalg.norm(a)
    wps = approach_waypoints if approach_waypoints is not None else (pose.get("approach_waypoints") or [])
    path = [pos + a * approach_m] + [pos + np.asarray(w, float) for w in wps] + [pos]
    q_start = float(d.qpos[qa])
    rig.teleport(d, path[0], eul); rig.set_fingers(d, opened); mujoco.mj_forward(model, d)
    for _ in range(max(1, int(0.3 / dt))):
        step(path[0], eul, opened)
    pen[0], pen[1] = 0.0, None             # the teleport itself is not the approach
    for p0, p1 in zip(path[:-1], path[1:]):
        n = max(1, int(max(0.25, np.linalg.norm(p1 - p0) / 0.12) / dt))
        for i in range(n):
            u = (i + 1) / n; u = u * u * (3 - 2 * u)
            step(p0 + (p1 - p0) * u, eul, opened); scan()
    q_landed = float(d.qpos[qa])
    n = max(1, int(0.4 / dt))
    for i in range(n):
        u = (i + 1) / n
        step(pos, eul, {k: opened.get(k, v) + (v - opened.get(k, v)) * u for k, v in closed.items()}); scan()
    pressed_pos = pos + (np.asarray(clamp["wrist_press"], float) if clamp["wrist_press"] is not None else 0.)
    n = max(1, int(.3 / dt))
    for i in range(n):
        u = (i + 1) / n
        fingers = {k: closed.get(k, v) + u * (v - closed.get(k, v)) for k, v in press.items()}
        step(pos + u * (pressed_pos - pos), eul, fingers)
        scan()
    pos = pressed_pos
    for _ in range(max(1, int(.3 / dt))):
        step(pos, eul, press)
    mujoco.mj_forward(model, d)
    pads0 = scan()
    q0 = float(d.qpos[qa])
    axis = d.xmat[part].reshape(3, 3) @ model.jnt_axis[jid]
    axis = axis / np.linalg.norm(axis) * float(direction)
    sib0 = {nm: float(d.qpos[adr]) for nm, adr in siblings.items()}
    nseg = max(1, int(np.ceil(travel_m / 0.005)))
    lost_at, p_cur = None, pos.copy()
    for i in range(nseg):
        p_new = pos + axis * travel_m * (i + 1) / nseg
        for _ in range(max(1, int(seconds / nseg / dt))):
            step(p_new, eul, press)
        p_cur = p_new
        if lost_at is None and not scan():
            lost_at = round(travel_m * (i + 1) / nseg, 4)
    for _ in range(max(1, int(0.3 / dt))):
        step(p_cur, eul, press)
    mujoco.mj_forward(model, d)
    moved = float(d.qpos[qa] - q0) * float(direction)
    frac = moved / travel_m
    pads_end = scan()
    sib_moved = {nm: round(abs(float(d.qpos[adr]) - sib0[nm]), 4) for nm, adr in siblings.items()}
    bad_sib = {k: v for k, v in sib_moved.items() if v > sibling_tol}
    issues = []
    if frac < min_fraction:
        if moved < 0.1 * travel_m:
            verdict = f"STALLED: the joint moved {moved*1000:.0f} mm while the wrist moved {travel_m*1000:.0f} mm"
            issues.append(verdict + ("; no finger reached the part: change the approach path (approach_dir, approach_waypoints) or the pre-grasp shape (open_frac, finger_targets_open)" if not pads0 else "; the fingers touch the part but on the wrong side: hook the far side of the handle"))
        elif not pads_end:
            verdict = f"LOST: the hand came off the part after {lost_at} m (joint moved {moved*1000:.0f} mm)"
            issues.append(verdict + "; hook deeper behind the handle or curl the hooking finger more")
        else:
            verdict = f"SLIPPED: joint moved {moved*1000:.0f} of {travel_m*1000:.0f} mm ({100*frac:.0f}%)"
            issues.append(verdict + "; the pads skate along the pull: hook the handle instead of pressing its face")
    else:
        verdict = f"PULLED: the part followed {moved*1000:.0f} of {travel_m*1000:.0f} mm ({100*frac:.0f}%), held by {pads_end}"
    if not pads_end and frac >= min_fraction:
        issues.append("no hand geom touches the part at the end of the stroke")
    if bad_sib:
        issues.append(f"sibling joints dragged along: {bad_sib}; the hand catches a neighbouring part, move the pose away from it")
    pre_moved = q_landed - q_start
    close_moved = (q0 - q_landed) * float(direction)
    if abs(pre_moved) > 0.01:
        issues.append(f"the approach moved the joint by {pre_moved*1000:.0f} mm before the fingers closed: the path or the pre-grasp shape pushes the part; come in from another direction")
    if close_moved < -0.01:
        issues.append(f"closing the fingers drove the joint {-close_moved*1000:.0f} mm against the pull direction: the closing fingers push the part instead of hooking it")
    if pen[0] > max_pen_mm:
        issues.append(f"the hand sank {pen[0]:.1f} mm into {pen[1][1]} ({pen[1][0]}) on the way in: change the approach path or the pose")
    return {"ok": not issues, "issues": issues, "verdict": verdict, "measured": {
        "travel_m": float(travel_m), "joint_moved_m": round(moved, 4), "fraction": round(frac, 3),
        "joint": model.joint(jid).name, "joint_start": round(q0, 4), "joint_end": round(float(d.qpos[qa]), 4),
        "joint_moved_during_approach_m": round(pre_moved, 4), "joint_moved_during_close_m": round(close_moved, 4), "pads_start": pads0, "pads_end": pads_end,
        "lost_at_m": lost_at, "siblings_moved": sib_moved, "peak_wrist_force_n": round(peak[0], 1),
        "approach_pen_mm": round(pen[0], 2), "approach_pen_pair": pen[1], "open_frac": float(open_frac),
        "approach_dir": [round(float(x), 4) for x in a], "n_waypoints": len(wps)}}
