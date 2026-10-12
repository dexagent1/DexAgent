"""Holds-at-video-end check: the grasp survives the end of the clip in physics.

    from tool_library.verifiers.holds_at_video_end_check import check_holds_at_video_end
    check_holds_at_video_end("scene.xml", "trajectory.npz", body="cup",
                             holder="robot_right_right_hand_C_MC")

The last frame of a replicated clip often shows the object still in hand.
A kinematic trajectory cannot prove that it is held: fingers closed around
the object by pose are not a grasp. This gate takes the final frame, pins
every robot joint there, lets gravity act on the object for a second of
physics, and asks whether the object stays within a few millimetres of where
it was. The same test starting from a frame in the middle of a carry is a
grasp stress test at that moment.
"""
from __future__ import annotations

TOOL = {
    "name": "holds_at_video_end_check",
    "title": "Holds at video end check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "From the final frame, hold the robot still and run gravity: the object must stay in hand (or stay put where it was placed).",
    "description": (
        "Takes the trajectory's last frame (or any frame), freezes every robot joint by "
        "writing the pose back each step, and runs the scene's physics for settle_s seconds. "
        "The body passes when its displacement over that time stays under drop_tol (1 cm by "
        "default) and its rotation under 10 degrees; if the pose has it resting on something "
        "(placed, not held) the same test says whether the placement is stable. Any object "
        "that falls, slips out or is squeezed out by over-closed fingers fails. Reports the "
        "drift, how many robot bodies touch the object, and whether the object is in contact "
        "with the table at the start (held vs placed). Requires the scene's collision "
        "geometry and masses. The robot is pinned through its joints' qpos (every joint of "
        "a body with robot_prefix); a mocap-driven hand has no joints and stays at the "
        "model's mocap pose, not the frame's, so pose its mocap bodies in the scene first."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene with collision geometry."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "body", "type": "str", "description": "Object that must stay held or placed."},
        {"name": "holder", "type": "str | None", "description": "Hand root body; reported contact count uses its subtree."},
        {"name": "frame", "type": "int", "description": "Frame to test from. Default -1 (the end)."},
        {"name": "settle_s", "type": "float, seconds", "description": "Physics time. Default 1.0."},
        {"name": "drop_tol_m, rot_tol_deg", "type": "float", "description": "Defaults 0.01 m, 10 deg."},
        {"name": "robot_prefix", "type": "str", "description": "Bodies with this prefix are pinned. Default 'robot_'."},
        {"name": "noslip_iterations", "type": "int", "description": "No-slip solver passes during the settle, so soft-contact creep is not read as a drop. Default 10."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; drift_m, rot_deg, held (robot contacts at start), on_table, contacts at end, frame tested."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["task gate", "grasp", "physics", "drop test", "end state"],
    "used_in": ["Multi-Cup Grasping", "Giftbox Packing", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "holds_at_video_end_check",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ


def _robot_qcols(model, robot_prefix):
    cols, dofs = [], []
    for j in range(model.njnt):
        if model.body(model.jnt_bodyid[j]).name.startswith(robot_prefix):
            n = {TJ.FREE: 7, TJ.BALL: 4}.get(int(model.jnt_type[j]), 1)
            nd = {TJ.FREE: 6, TJ.BALL: 3}.get(int(model.jnt_type[j]), 1)
            cols += list(range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + n))
            dofs += list(range(int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + nd))
    return np.asarray(cols, int), np.asarray(dofs, int)


def _touching(model, data, ids_a, ids_b):
    out = set()
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if b1 in ids_a and b2 in ids_b:
            out.add(b1)
        elif b2 in ids_a and b1 in ids_b:
            out.add(b2)
    return out


def check_holds_at_video_end(scene, trajectory, body, holder=None, frame=-1, settle_s=1.0,
                             drop_tol_m=0.01, rot_tol_deg=10.0, robot_prefix="robot_", noslip_iterations=10,
                             rigid_armature=1e3):
    import copy
    model = copy.copy(TJ.load_model(scene))
    # soft contacts creep under a sustained tangential load (the friction
    # constraint only damps velocity); the no-slip post-pass removes that
    # solver drift so the test measures the grasp, not the integrator
    model.opt.noslip_iterations = max(int(model.opt.noslip_iterations), noslip_iterations)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    t = frame % len(Q)
    bid = TJ.body_id(model, body)
    adr = TJ.free_joint_qadr(model, bid)
    if adr is None:
        raise ValueError(f"'{body}' has no free joint; a fixed or articulated body cannot drop")
    data = mujoco.MjData(model)
    data.qpos[:] = Q[t]
    mujoco.mj_forward(model, data)
    rob_cols, rob_dofs = _robot_qcols(model, robot_prefix)
    q_rob = Q[t, rob_cols].copy()
    # the robot is a position source here: a huge armature on its dofs makes the
    # solver treat the posed fingers as rigid instead of letting them give way
    model.dof_armature[rob_dofs] += rigid_armature
    hold_ids = TJ.subtree(model, TJ.body_id(model, holder)) if holder else set(
        b for b in range(model.nbody) if model.body(b).name.startswith(robot_prefix))
    obj_ids = TJ.subtree(model, bid)
    table_ids = {b for b in range(model.nbody) if "table" in model.body(b).name.lower()} | {0}
    held0 = _touching(model, data, hold_ids, obj_ids)
    on_table0 = bool(_touching(model, data, table_ids, obj_ids))
    p0, R0 = data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
    # a weld between the robot and the object would hold it whatever the
    # fingers do: switch those off so the test is about contact
    disabled = []
    for e in range(model.neq):
        a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if (a in obj_ids) != (b in obj_ids) and (a in hold_ids or b in hold_ids or
                                                 model.body(a).name.startswith(robot_prefix) or
                                                 model.body(b).name.startswith(robot_prefix)):
            data.eq_active[e] = 0
            disabled.append(e)
    n = int(round(settle_s / model.opt.timestep))
    for _ in range(n):
        data.qpos[rob_cols] = q_rob
        data.qvel[rob_dofs] = 0.0
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    drift = float(np.linalg.norm(data.xpos[bid] - p0))
    rot = float(np.degrees(TJ.angle_between(R0, data.xmat[bid].reshape(3, 3))))
    rep = {"body": body, "frame": int(t), "label": labels[t], "settle_s": settle_s,
           "held_at_start": bool(held0), "robot_bodies_touching_start": len(held0),
           "on_table_at_start": on_table0,
           "robot_bodies_touching_end": len(_touching(model, data, hold_ids, obj_ids)),
           "drift_m": round(drift, 4), "rot_deg": round(rot, 2),
           "drop_tol_m": drop_tol_m, "rot_tol_deg": rot_tol_deg, "welds_disabled": disabled,
           "ok": bool(drift <= drop_tol_m and rot <= rot_tol_deg)}
    return rep


if __name__ == "__main__":
    rep = check_holds_at_video_end(sys.argv[1], sys.argv[2], sys.argv[3], holder=sys.argv[4] if len(sys.argv) > 4 else None)
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


holds_at_video_end_check = check_holds_at_video_end   # the library calls every tool by its own name
