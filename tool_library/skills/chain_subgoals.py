"""Stitch subgoal keyframes into one continuous, rate-limited joint trajectory.

    from tool_library.skills.chain_subgoals import chain_subgoals
    out = chain_subgoals("scene.xml", keyframes=[
            {"qpos": q_home, "hold": 10, "label": "home"},
            {"qpos": q_pregrasp, "transit": 40, "hold": 5, "label": "pre-grasp"},
            {"joints": {"robot_right_right_index_PIP": 1.2}, "transit": 15, "hold": 20, "label": "close"},
            {"qpos": q_lift, "transit": 30, "label": "lift"}],
          fps=30, max_rate_deg_s=90)
    out["qpos"], out["labels"]

A plan is a list of states with durations. Between two keyframes every joint
follows a quintic (zero velocity and acceleration at both ends), free-joint
quaternions are interpolated on the sphere, and a keyframe may set only some
joints (the rest hold their previous value). The result is then checked
against a joint-rate ceiling and transits that would break it are stretched
to the minimum legal duration, so the output is executable by construction.
"""
from __future__ import annotations

TOOL = {
    "name": "chain_subgoals",
    "title": "Chain subgoals",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Keyframes with hold and transit durations -> continuous qpos trajectory with quintic legs, slerped free joints, labels, and transits stretched to respect a joint-rate ceiling.",
    "description": (
        "Takes an ordered list of keyframes, each a full qpos or a partial {joint name: "
        "value} update, with hold frames (the pose is kept) and transit frames (the leg from "
        "the previous pose). Scalar joints follow a quintic ease with zero end velocity and "
        "acceleration; free-joint positions are eased the same way and quaternions are "
        "slerped; frames inherit the keyframe's label. When max_rate_deg_s is given, each "
        "transit is checked after construction and lengthened to the shortest duration that "
        "keeps every hinge under the ceiling (slides under max_rate_m_s), reporting which "
        "legs were stretched. Optionally writes trajectory.npz (qpos, labels, fps). Works on "
        "any scene: the chain is a time plan, not a robot."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene (for joint names, types and nq)."},
        {"name": "keyframes", "type": "list[dict]", "description": "Each: qpos [nq] or joints {name: value}; hold (int frames); transit (int frames, 0 for the first); label (str)."},
        {"name": "fps", "type": "float", "description": "Default 30."},
        {"name": "max_rate_deg_s, max_rate_m_s", "type": "float | None", "description": "Joint-rate ceilings used to stretch transits."},
        {"name": "q0", "type": "ndarray | None", "description": "Starting pose if the first keyframe is partial; default the scene's qpos0."},
        {"name": "out_path", "type": "str | None", "description": "Write trajectory.npz."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "qpos [T, nq], labels [T], fps, keyframe_frames (start frame of each keyframe), stretched legs, max_rate_deg_s achieved."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["planning", "keyframes", "interpolation", "subgoals"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "chain_subgoals",
}

import numpy as np

from tool_library.common import trajectory as TJ


def _quat_slerp(qa, qb, u):
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    if np.dot(qa, qb) < 0:
        qb = -qb
    d = np.clip(np.dot(qa, qb), -1.0, 1.0)
    if d > 0.9995:
        q = qa + u * (qb - qa)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * qa + np.sin(u * th) * qb) / np.sin(th)


def _leg(model, qa, qb, n):
    """n frames from qa (exclusive) to qb (inclusive), quintic ease."""
    u = TJ.smoothstep(np.arange(1, n + 1) / n)[:, None]
    Q = qa + u * (qb - qa)
    for j in range(model.njnt):
        if int(model.jnt_type[j]) in (TJ.FREE, TJ.BALL):
            a = int(model.jnt_qposadr[j]) + (3 if int(model.jnt_type[j]) == TJ.FREE else 0)
            for k in range(n):
                Q[k, a:a + 4] = _quat_slerp(qa[a:a + 4], qb[a:a + 4], float(u[k, 0]))
    return Q


def _max_rate(model, Q, fps):
    hinge = [int(model.jnt_qposadr[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == TJ.HINGE]
    slide = [int(model.jnt_qposadr[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == TJ.SLIDE]
    if len(Q) < 2:
        return 0.0, 0.0
    d = np.abs(np.diff(Q, axis=0)) * fps
    return (float(np.degrees(d[:, hinge].max())) if hinge else 0.0), (float(d[:, slide].max()) if slide else 0.0)


def chain_subgoals(scene, keyframes, fps=30.0, max_rate_deg_s=None, max_rate_m_s=None, q0=None, out_path=None):
    model = TJ.load_model(scene)
    q = np.array(model.qpos0 if q0 is None else q0, float).copy()
    poses, holds, transits, labels = [], [], [], []
    for i, kf in enumerate(keyframes):
        if "qpos" in kf:
            q = np.asarray(kf["qpos"], float).copy()
            if len(q) != model.nq:
                raise ValueError(f"keyframe {i}: qpos has {len(q)} values, the scene has nq={model.nq}")
        if "joints" in kf:
            for name, v in kf["joints"].items():
                j = TJ.joint_id(model, name)
                q[int(model.jnt_qposadr[j])] = float(v)
        poses.append(q.copy()); holds.append(int(kf.get("hold", 1)))
        transits.append(int(kf.get("transit", 0)) if i else 0); labels.append(str(kf.get("label", f"subgoal {i}")))
    stretched = []
    for i in range(1, len(poses)):               # the shortest legal transit for each leg
        if transits[i] <= 0:
            continue
        for _ in range(50):
            leg = _leg(model, poses[i - 1], poses[i], transits[i])
            rh, rs = _max_rate(model, np.vstack([poses[i - 1], leg]), fps)
            need = max((rh / max_rate_deg_s) if max_rate_deg_s else 0.0, (rs / max_rate_m_s) if max_rate_m_s else 0.0)
            if need <= 1.0:
                break
            new = int(np.ceil(transits[i] * need * 1.02))
            stretched.append({"leg": i, "label": labels[i], "from": transits[i], "to": new})
            transits[i] = new
    Q, L, starts = [], [], []
    for i in range(len(poses)):
        if i and transits[i] > 0:
            leg = _leg(model, poses[i - 1], poses[i], transits[i])
            Q.append(leg[:-1]); L += [labels[i]] * (transits[i] - 1)
        starts.append(sum(len(x) for x in Q))
        Q.append(np.repeat(poses[i][None], max(1, holds[i]), axis=0)); L += [labels[i]] * max(1, holds[i])
    Q = np.vstack(Q)
    rh, rs = _max_rate(model, Q, fps)
    res = {"qpos": Q, "labels": L, "fps": fps, "keyframe_frames": starts, "stretched": stretched,
           "max_rate_deg_s": round(rh, 2), "max_rate_m_s": round(rs, 4), "n_frames": len(Q)}
    if out_path:
        np.savez_compressed(out_path, qpos=Q, labels=np.array(L), fps=fps)
        res["out_path"] = str(out_path)
    return res
