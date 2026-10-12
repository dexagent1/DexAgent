"""Lift check: was the object really lifted, not slid or dropped?

    from tool_library.verifiers.lift_check import check_lift
    check_lift("scene.xml", "rollout.npz", body="cup", reference="demo.npz")

A grasp that closes on air lets the object stay on the table while the hand
rises; a weak one drops it a few centimetres up. Both leave the object's
highest point far below the demonstration's. The gate: the object's rise
above its start must reach lift_ratio (0.5) of the demonstration's rise,
judged only when the demonstration itself rose at least 4 cm, and the hand
must still be in contact for most of the frames the object is up.
"""
from __future__ import annotations

TOOL = {
    "name": "lift_check",
    "title": "Lift check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Object rise reaches a fraction of the demonstration's rise, and it stays in hand while up.",
    "description": (
        "Measures the body's height above its first frame along the trajectory and compares "
        "the peak with the demonstration's peak rise (or an explicit min_rise_m). The gate "
        "passes when peak rise >= lift_ratio x demonstration rise (default 0.5), and is "
        "skipped (ok with a note) when the demonstration never lifted the object more than "
        "lift_ref_floor (4 cm). When a holder body is given, the frames where the object is "
        "more than 1 cm up must have hand-object contact for at least carry_contact_frac "
        "(0.5) of them, measured with the scene's collision geometry; an object that rose "
        "because it was flung fails this part. Reports the frame and label of the peak."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "The rollout."},
        {"name": "body", "type": "str", "description": "The lifted object."},
        {"name": "reference", "type": "str | ndarray | None", "description": "Demonstration; defines the expected rise."},
        {"name": "min_rise_m", "type": "float | None", "description": "Explicit required rise instead of a reference."},
        {"name": "holder", "type": "str | None", "description": "Hand root body; enables the in-hand contact rule."},
        {"name": "lift_ratio, carry_contact_frac", "type": "float", "description": "Defaults 0.5 and 0.5."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; rise_m, reference_rise_m, required_rise_m, peak frame/label, frames_up, contact_frac_while_up."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["task gate", "lift", "grasp", "rollout"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Battery Insertion", "Computer Installation", "Open the Bottle"],
    "status": "tested",
    "entrypoint": "lift_check",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ


def contact_frames(model, Q, holder, held, frames):
    data = mujoco.MjData(model)
    hb, ob = TJ.subtree(model, TJ.body_id(model, holder)), TJ.subtree(model, TJ.body_id(model, held))
    out = np.zeros(len(frames), bool)
    for k, t in enumerate(frames):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 in hb and b2 in ob) or (b2 in hb and b1 in ob):
                out[k] = True
                break
    return out


def check_lift(scene, trajectory, body, reference=None, min_rise_m=None, holder=None,
               lift_ratio=None, carry_contact_frac=None, lift_ref_floor=None):
    g = TJ.GATES
    lift_ratio = g["lift_ratio_min"] if lift_ratio is None else lift_ratio
    carry_contact_frac = g["carry_contact_frac"] if carry_contact_frac is None else carry_contact_frac
    lift_ref_floor = g["lift_ref_floor_m"] if lift_ref_floor is None else lift_ref_floor
    model = TJ.load_model(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    bid = TJ.body_id(model, body)
    P, _ = TJ.pose_track(model, Q, bid)
    rise = P[:, 2] - P[0, 2]
    k = int(np.argmax(rise))
    rep = {"body": body, "rise_m": round(float(rise[k]), 4), "peak": {"frame": k, "label": labels[k]}}
    if reference is not None:
        Pr, _ = TJ.pose_track(model, TJ.as_trajectory(reference)[0], bid)
        ref_rise = float((Pr[:, 2] - Pr[0, 2]).max())
        rep["reference_rise_m"] = round(ref_rise, 4)
        if ref_rise < lift_ref_floor:
            rep.update(required_rise_m=0.0, note="demonstration never lifted the object; lift not judged", ok=True)
            return rep
        required = lift_ratio * ref_rise
    elif min_rise_m is not None:
        required = float(min_rise_m)
    else:
        raise ValueError("give a reference trajectory or min_rise_m")
    rep["required_rise_m"] = round(required, 4)
    ok = rise[k] >= required
    if holder is not None:
        # the in-hand rule covers the carry: frames where the object is up and
        # has not yet come to rest (an object placed on a shelf is up for the
        # rest of the episode and is not expected to be held there)
        step = np.linalg.norm(np.diff(P, axis=0), axis=1)
        moving = np.flatnonzero(step > 0.5e-3)
        t_rest = int(moving[-1]) + 1 if len(moving) else len(P) - 1
        up = np.flatnonzero((rise > 0.01) & (np.arange(len(P)) <= t_rest))
        rep["frames_up"] = int(len(up))
        rep["carry_ends"] = {"frame": t_rest, "label": labels[t_rest]}
        if len(up):
            cf = contact_frames(model, Q, holder, body, up)
            rep["contact_frac_while_up"] = round(float(cf.mean()), 3)
            rep["carry_contact_frac"] = carry_contact_frac
            ok &= cf.mean() >= carry_contact_frac
    rep["ok"] = bool(ok)
    return rep


if __name__ == "__main__":
    rep = check_lift(sys.argv[1], sys.argv[2], sys.argv[3], reference=sys.argv[4] if len(sys.argv) > 4 else None,
                     min_rise_m=None if len(sys.argv) > 4 else 0.05)
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


lift_check = check_lift   # the library calls every tool by its own name
