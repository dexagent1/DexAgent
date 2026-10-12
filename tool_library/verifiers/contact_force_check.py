"""Contact force check: is the object held with a force in the working range?

    from tool_library.verifiers.contact_force_check import check_contact_force
    check_contact_force("scene.xml", "trajectory.npz", holder="robot_right_right_hand_C_MC",
                        held="cup", frames=range(120, 300), min_n=0.5, max_n=40.0)

A kinematic trajectory that "grasps" by reaching around an object carries
nothing in physics. Posed into the scene frame by frame, MuJoCo's soft
contacts turn the finger-object penetration into a force; this check sums the
normal force between the holder's subtree (hand and fingers) and the held
body on each frame and asks that it stay inside [min_n, max_n] over the hold
window. Too little means an air grasp, too much means the fingers were
driven through the object.
"""
from __future__ import annotations

TOOL = {
    "name": "contact_force_check",
    "title": "Contact force check",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Normal force between a holder's subtree and a held body stays in a working range over the hold window (for a posed or stress-tested grasp; on frames of an executed episode, whose contacts carry penetration from the servo clamp, read finger_servo_saturation_audit and contact_impulse_spike_audit instead).",
    "description": (
        "For each frame of the window it writes the pose into the scene, runs a forward pass "
        "and sums the contact normal force over every contact between a geom of the holder's "
        "subtree (a hand with all its fingers) and a geom of the held body's subtree. The "
        "frame passes when the total is inside [min_n, max_n]; the window passes when at least "
        "hold_frac of its frames pass and the first and last frames do. Also reports how many "
        "distinct finger bodies touch (enclosure), the mean force, and the frame of the "
        "weakest hold. A bench scene is loaded as the executor runs it (its hand-object and hand-table contact pairs); "
        "a scene without robot hands uses its own collision geometry. The forces are those of "
        "the scene's solref/solimp, so compare them against the object's weight, which the "
        "report includes."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene with collision geometry on hand and object."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz or qpos array."},
        {"name": "holder", "type": "str", "description": "Root body of the hand (its subtree counts)."},
        {"name": "held", "type": "str", "description": "Body being held (its subtree counts)."},
        {"name": "frames", "type": "iterable[int] | list[str] | None", "description": "Hold window: frame indices or phase-label patterns. Default all frames."},
        {"name": "min_n, max_n", "type": "float, newtons", "description": "Working force range. Defaults 0.3 and 50."},
        {"name": "hold_frac", "type": "float", "description": "Fraction of window frames that must be in range. Default 0.9."},
        {"name": "min_bodies", "type": "int", "description": "Distinct holder bodies that must touch (2 = a pinch). Default 2."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; force per frame summary (min, mean, max N), frames in range, weakest frame and label, bodies in contact, object weight N, max_depth_mm (deepest hand-object penetration over the window: sub-millimetre for a seated grip; the posed force is that depth times the pair stiffness, so a wrap whose palm the wrist presses 0.7 mm in reads 60-90 N)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "contact", "force", "physics"],
    "used_in": ["Multi-Cup Grasping", "Giftbox Packing", "Battery Insertion", "Open the Bottle", "Scissor Cutting"],
    "status": "tested",
    "entrypoint": "contact_force_check",
}

import json
import sys

import mujoco
import numpy as np

from tool_library.common import trajectory as TJ


def contact_force_series(model, Q, holder, held, frames=None, rigid_armature=1e3, robot_prefix="robot_", depth=False):
    """Per frame: (total normal force N, number of distinct holder bodies touching[, deepest holder-held penetration m
    when depth=True]).
    Robot dofs get a huge armature so the posed fingers are rigid for the solver:
    otherwise a free hinge gives way and the force reads as the finger's inertia."""
    import copy
    model = copy.copy(model)
    for j in range(model.njnt):
        if model.body(model.jnt_bodyid[j]).name.startswith(robot_prefix):
            nd = {TJ.FREE: 6, TJ.BALL: 3}.get(int(model.jnt_type[j]), 1)
            a = int(model.jnt_dofadr[j])
            model.dof_armature[a:a + nd] += rigid_armature
    data = mujoco.MjData(model)
    hb = TJ.subtree(model, TJ.body_id(model, holder))
    ob = TJ.subtree(model, TJ.body_id(model, held))
    frames = range(len(Q)) if frames is None else list(frames)
    F, nb, f6 = np.zeros(len(frames)), np.zeros(len(frames), int), np.zeros(6)
    D = np.zeros(len(frames))
    for k, t in enumerate(frames):
        data.qpos[:] = Q[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        touching = set()
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 in hb and b2 in ob) or (b2 in hb and b1 in ob):
                mujoco.mj_contactForce(model, data, i, f6)
                F[k] += abs(f6[0])
                D[k] = max(D[k], -float(c.dist))
                touching.add(b1 if b1 in hb else b2)
        nb[k] = len(touching)
    return (F, nb, list(frames), D) if depth else (F, nb, list(frames))


def _load(scene):
    """A bench scene as the executor runs it: its hands ship without collision (contype 0) and touch objects only
    through the contact pairs build_execution_model adds, so a raw load reads 0 N on every grip."""
    if isinstance(scene, mujoco.MjModel):
        return scene
    try:
        from tool_library.common import episode as E
        return E.build_execution_model(str(scene))[0]
    except ValueError:                                 # no robot hands in it: the scene's own collision geometry
        return TJ.load_model(scene)


def check_contact_force(scene, trajectory, holder, held, frames=None, min_n=0.3, max_n=50.0,
                        hold_frac=0.9, min_bodies=2):
    model = _load(scene)
    Q, labels, fps = TJ.as_trajectory(trajectory)
    if frames is not None and len(frames) and isinstance(next(iter(frames)), str):
        frames = TJ.frames_matching(labels, list(frames))
        if not frames:
            raise ValueError("the phase patterns match no frame")
    F, nb, idx, D = contact_force_series(model, Q, holder, held, frames, depth=True)
    inrange = (F >= min_n) & (F <= max_n) & (nb >= min_bodies)
    k = int(np.argmin(F))
    weight = float(model.body_subtreemass[TJ.body_id(model, held)] * np.linalg.norm(model.opt.gravity))
    rep = {"frames": [int(idx[0]), int(idx[-1])], "n_frames": len(idx),
           "force_n": {"min": round(float(F.min()), 3), "mean": round(float(F.mean()), 3), "max": round(float(F.max()), 3)},
           "range_n": [min_n, max_n], "object_weight_n": round(weight, 3),
           "frac_in_range": round(float(inrange.mean()), 3), "hold_frac": hold_frac,
           "bodies_touching": {"min": int(nb.min()), "max": int(nb.max())},
           # the posed force turns penetration into newtons through the pair stiffness (0.3 mm of palm read ~37 N, 0.7 mm
           # ~63 N on a seated wrap): the depth says whether fingers were driven through the object
           "max_depth_mm": round(1000 * float(D.max()), 2),
           "weakest": {"frame": int(idx[k]), "label": labels[idx[k]], "force_n": round(float(F[k]), 3)},
           "ok": bool(inrange.mean() >= hold_frac and inrange[0] and inrange[-1])}
    return rep


if __name__ == "__main__":
    rep = check_contact_force(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print(json.dumps(rep, indent=1, default=str)); sys.exit(0 if rep["ok"] else 1)


contact_force_check = check_contact_force   # the library calls every tool by its own name
