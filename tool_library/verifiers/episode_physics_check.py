"""Is the executed episode physically plausible?"""
from __future__ import annotations

TOOL = {
    "name": "episode_physics_check",
    "title": "Episode physics check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Realism audit of an executed episode: finite state, no object faster than a hand can move things, no object "
               "jumping between frames, no interpenetration beyond a few millimetres between hands and objects or between "
               "objects, nothing left floating at the end, and the hands' servos never forced far off their commands.",
    "description": (
        "Replays the episode frame by frame (kinematics + collision detection on the physics scene with the hand-object "
        "contact pairs) and measures: per free object its peak speed and the largest single-frame displacement; the "
        "deepest hand-object/hand-table and object-object interpenetration on every frame; at the last frame every "
        "free object's support (contact with something below it, or its lowest point at the table within tolerance); "
        "and, when the episode carries the rollout diagnostics, the worst wrist tracking error and the frames where it "
        "exceeded the limit. Thresholds are generous for real manipulation and tight for artefacts: an object moving "
        "at 4 m/s, a 3 cm jump in one frame or a pad 1 cm inside a surface only happen when the simulation was "
        "forced. The report lists the offending frames and bodies."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "trajectory", "type": "str", "description": "Episode npz (qpos, labels, fps, optional diagnostics)."},
        {"name": "max_speed_mps", "type": "float", "description": "Default 4.0."},
        {"name": "max_jump_m", "type": "float", "description": "Largest allowed per-frame displacement of an object. Default 0.03."},
        {"name": "max_pen_m", "type": "float", "description": "Allowed interpenetration. Default 0.008."},
        {"name": "max_track_m", "type": "float", "description": "Allowed wrist tracking error. Default 0.03."},
        {"name": "stride", "type": "int", "description": "Frame stride for the collision replay. Default 2."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, failed, objects {name: {peak_speed, max_jump, jump_frame}}, penetration {hand_object_mm, object_object_mm, worst_frame, worst_pair}, floating_at_end, tracking {max_mm, frames_over}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "physics", "realism", "penetration"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "episode_physics_check",
}

import mujoco
import numpy as np

from tool_library.common import episode as E


def episode_physics_check(scene, trajectory, max_speed_mps=4.0, max_jump_m=0.03, max_pen_m=0.008, max_track_m=0.03, stride=2):
    model, _ = E.build_execution_model(scene)
    ep = E.load_episode(trajectory)
    Q, fps = ep["qpos"], ep["fps"]
    failed, rep = [], {}
    if not np.isfinite(Q).all():
        failed.append("non_finite_state")
    free = E.object_free_joints(model)
    data = mujoco.MjData(model)
    objs = {}
    for name, adr in free.items():
        P = Q[:, adr:adr + 3]
        step = np.linalg.norm(np.diff(P, axis=0), axis=1) if len(P) > 1 else np.zeros(1)
        speed = step * fps
        k = int(np.argmax(step)) if len(step) else 0
        objs[name] = {"peak_speed_mps": round(float(speed.max()), 3), "max_jump_m": round(float(step.max()), 4), "jump_frame": k + 1,
                      "travel_m": round(float(np.linalg.norm(P[-1] - P[0])), 4)}
        if speed.max() > max_speed_mps:
            failed.append(f"speed:{name}")
        if step.max() > max_jump_m:
            failed.append(f"jump:{name}")
    rep["objects"] = objs
    robot = {b for b in range(model.nbody) if model.body(b).name.startswith("robot_")}
    worst = {"hand_object_mm": 0.0, "object_object_mm": 0.0, "worst_frame": None, "worst_pair": None}
    for t in range(0, len(Q), max(1, int(stride))):
        data.qpos[:] = Q[t]; mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            r1, r2 = b1 in robot, b2 in robot
            depth = -float(c.dist) * 1000
            key = "hand_object_mm" if r1 != r2 else ("object_object_mm" if not (r1 or r2) else None)
            if key is None:
                continue
            if depth > worst[key]:
                worst[key] = round(depth, 2)
            if depth > max(worst["hand_object_mm"], worst["object_object_mm"]) - 1e-9:
                worst["worst_frame"] = t; worst["worst_pair"] = [model.geom(c.geom1).name, model.geom(c.geom2).name]
    rep["penetration"] = worst
    if worst["hand_object_mm"] > max_pen_m * 1000:
        failed.append("hand_object_penetration")
    if worst["object_object_mm"] > max_pen_m * 1000:
        failed.append("object_object_penetration")
    # floating at the end
    data.qpos[:] = Q[-1]; mujoco.mj_forward(model, data)
    floating = []
    for name in free:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name); sub = E.subtree_bodies(model, b)
        touching = E.contacts_at(model, data, sub, {x for x in range(model.nbody) if x not in sub})
        z = E.lowest_point(model, data, sub)
        if not touching and z > 0.012:
            floating.append({"body": name, "lowest_point_z": round(z, 4)})
    rep["floating_at_end"] = floating
    if floating:
        failed.append("floating_at_end")
    z = np.load(str(trajectory), allow_pickle=True)
    if "track_pos_m" in z.files:
        tp = np.asarray(z["track_pos_m"])
        over = [int(i) for i in np.where(tp > max_track_m)[0]]
        rep["tracking"] = {"max_mm": round(1000 * float(tp.max()), 1), "frames_over": over[:20], "n_over": len(over)}
        if over:
            failed.append("servo_tracking")
    if "wrist_force_n" in z.files:
        rep["peak_wrist_force_n"] = round(float(np.asarray(z["wrist_force_n"]).max()), 1)
    return {"ok": not failed, "failed": failed, **rep, "n_frames": int(len(Q))}
