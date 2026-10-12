"""Signed contact-wrench authority for hinge/slide search, not a physics verdict."""
from __future__ import annotations

import mujoco
import numpy as np


def joint_contact_directions(model, data, joint, target, moving_bodies, ignore_bodies=()):
    """Contact-normal opposition to a requested joint motion, not proof of cause.

    A support touching a stalled part is not necessarily obstructing its path:
    the desired motion may leave that support. Tangential friction is separate.
    Caller supplies current forward kinematics and collision data.
    """
    jid = int(joint)
    direction = float(np.sign(float(target) - data.qpos[int(model.jnt_qposadr[jid])]))
    dof = int(model.jnt_dofadr[jid])
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    rows = []
    for c in data.contact:
        if c.geom1 < 0 or c.geom2 < 0:
            continue
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (b1 in moving_bodies) == (b2 in moving_bodies):
            continue
        part, other = (b1, b2) if b1 in moving_bodies else (b2, b1)
        if other in ignore_bodies:
            continue
        force_direction = np.asarray(c.frame[:3]) * (-1. if part == b1 else 1.)
        mujoco.mj_jac(model, data, jp, jr, c.pos, part)
        effort = float(direction * (force_direction @ jp[:, dof]))
        ancestor, movable = other, False
        while ancestor > 0:
            movable = movable or bool(model.body_jntnum[ancestor])
            ancestor = int(model.body_parentid[ancestor])
        rows.append({"body": model.body(other).name or "world", "opposes_target": effort < -1e-8,
                     "normal_effort_per_unit": effort, "movable": bool(movable)})
    return rows


def friction_upper_bound(model, side, moving_bodies):
    """Conservative Coulomb coefficients for any hand/moving-part contact.

    Include explicit pair overrides and angular friction. Taking componentwise
    maxima can overestimate a particular contact, so a positive result is only
    possible authority. It cannot prove force closure or sufficient force.
    """
    hands = {g for g in range(model.ngeom)
             if f"_collision_hand_{side}_" in (model.geom(g).name or "")}
    paired = set(map(int, model.pair_geom1)) | set(map(int, model.pair_geom2))
    objects = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in moving_bodies
               and (model.geom_contype[g] or model.geom_conaffinity[g] or g in paired)}
    ids = sorted(hands | objects)
    bound = np.max(model.geom_friction[ids], axis=0) if ids else np.zeros(3)
    dimension = max((int(model.geom_condim[g]) for g in ids), default=1)
    for pair in range(model.npair):
        a, b = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
        if (a in hands and b in objects) or (b in hands and a in objects):
            f = model.pair_friction[pair]
            bound = np.maximum(bound, [max(f[0], f[1]), f[2], max(f[3], f[4])])
            dimension = max(dimension, int(model.pair_dim[pair]))
    if dimension < 3:
        bound[0] = 0.
    if dimension < 4:
        bound[1] = 0.
    if dimension < 6:
        bound[2] = 0.
    return bound


def contact_drive(model, data, joint, target, points, normals, friction=(0., 0., 0.)):
    """Support of unilateral contact wrench cones in the requested joint direction.

    Normals point OUT of the object; pressure on it therefore acts along -normal.
    Coefficients are signed generalized effort per newton of normal force
    (metres for hinges, dimensionless for slides). Multiple contacts may share
    an arbitrary nonnegative normal-force budget; its greatest possible effort
    is the maximum support over the contacts, not their unweighted sum.
    Passive forces, force limits, sticking, and actual contact are not tested.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(joint))
    if jid < 0 or int(model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError("contact drive requires an existing hinge or slide joint")
    p, n = np.asarray(points, float), np.asarray(normals, float)
    f = np.asarray(friction, float)
    if p.ndim != 2 or p.shape[1:] != (3,) or not len(p) or n.shape != p.shape or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError("points and normals must be matching nonempty finite xyz arrays")
    lengths = np.linalg.norm(n, axis=1)
    if np.any(lengths < 1e-12) or f.shape != (3,) or not np.isfinite(f).all() or np.any(f < 0):
        raise ValueError("normals must be nonzero and friction must have three finite nonnegative coefficients")
    if not np.isfinite(target):
        raise ValueError("target must be finite")
    n = n / lengths[:, None]
    delta = float(target) - float(data.qpos[int(model.jnt_qposadr[jid])])
    direction = float(np.sign(delta))
    mujoco.mj_forward(model, data)
    body, dof = int(model.jnt_bodyid[jid]), int(model.jnt_dofadr[jid])
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    pressure, cones = [], []
    for point, normal in zip(p, n):
        mujoco.mj_jac(model, data, jp, jr, point, body)
        v, w = direction * jp[:, dof], direction * jr[:, dof]
        a = float(-normal @ v)
        tangent = float(np.linalg.norm(v - normal * (normal @ v)))
        angular = f[1] * abs(float(normal @ w)) + f[2] * np.linalg.norm(w - normal * (normal @ w))
        pressure.append(a)
        cones.append(float(a + f[0] * tangent + angular))
    normal_best, cone_best = max(pressure), max(cones)
    eps = 1e-10
    status = ("no_motion" if abs(delta) <= eps else "normal_drives" if normal_best > eps
              else "friction_required" if cone_best > eps else "cannot_drive")
    return {"status": status, "can_drive": status in ("normal_drives", "friction_required", "no_motion"),
            "joint": str(joint), "direction": int(direction),
            "normal_effort_per_N": pressure, "cone_effort_upper_per_N": cones,
            "best_normal_effort_per_N": normal_best, "best_cone_effort_upper_per_N": cone_best,
            "friction_upper_bound": f.tolist(),
            "effort_units": "Nm/N" if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE) else "N/N",
            "assumption": "unilateral contact at the supplied surface points; conservative Coulomb wrench cone",
            "scope": "contact_direction_only", "execution_required": True}
