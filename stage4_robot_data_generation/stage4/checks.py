"""Feasibility checks every augmented trajectory must pass.

A sample is accepted only when all of these hold:
  reach         each end effector (and the arm bolted to it) reaches its
                re-anchored pose within tolerance, inside the joint limits;
  joint speed   no robot joint moves much faster than in the original episode;
  penetration   no contact is deeper than the same contact in the original
                episode plus a tolerance, so the new layout adds no collision;
  continuity    no object jumps between frames.
Contacts are evaluated on every visible geom, not only on the collision
geometry the scene defines: arm links are usually visual-only in a MuJoCo
scene, and an arm sweeping through the table or the other arm has to count.
"""
from __future__ import annotations

import mujoco
import numpy as np


def collision_model(scene_path, model, visual_groups):
    """The scene recompiled so that every visible geom collides, so the
    penetration check sees the arms, the hands and the objects as drawn.
    Collisions are switched on before compiling: that is when MuJoCo builds
    the convex hulls a mesh needs in order to collide at all."""
    from pathlib import Path
    assert Path(scene_path).suffix.lower() != ".mjb", (
        "the penetration check needs the scene as MJCF (.xml), not a compiled .mjb")
    spec = mujoco.MjSpec.from_file(str(scene_path))
    on = 0
    for g in spec.geoms:
        if (int(g.group) in visual_groups and g.rgba[3] > 0.0 and g.contype == 0
                and g.conaffinity == 0 and int(g.type) != int(mujoco.mjtGeom.mjGEOM_PLANE)):
            g.contype = g.conaffinity = 1
            on += 1
    m = spec.compile()
    assert (m.nq, m.ngeom, m.nbody) == (model.nq, model.ngeom, model.nbody), (
        "the collision copy of the scene does not match the scene")
    return m, on


def joined_bodies(model):
    """Body pairs bolted together by an equality constraint, such as a hand on
    its arm: their drawn shells overlap by design and are not a collision."""
    out = set()
    for e in range(model.neq):
        if model.eq_active0[e] and int(model.eq_type[e]) in (
                int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)):
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            out.add((min(b1, b2), max(b1, b2)))
    return out


def subtree(model, bid):
    out, stack = set(), [bid]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if model.body_parentid[c] == b and c != b]
    return out


def same_limb(model, sides):
    """Bodies of one arm-plus-hand assembly, per side: a hand is joined to its
    own arm by design, so contacts inside one limb are not checked. Contacts
    between the two limbs, and with everything else, are."""
    groups = []
    for s in sides.values():
        bodies = subtree(model, s["ee"].bid)
        b = s["ee"].bid
        while b != 0:
            bodies.add(b)
            b = model.body_parentid[b]
        if s["follower"]:
            b = s["follower"]["chain"].bid
            while b != 0:
                bodies.add(b)
                b = model.body_parentid[b]
        groups.append(bodies)
    return groups


def penetration_profile(model, Q, stride=1, limbs=()):
    """{(geom a, geom b): (deepest penetration in m, frame)} over a trajectory."""
    data = mujoco.MjData(model)
    joined = joined_bodies(model)
    out = {}
    for t in range(0, len(Q), stride):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= 0:
                continue
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (min(b1, b2), max(b1, b2)) in joined:
                continue
            if any(b1 in limb and b2 in limb for limb in limbs):
                continue                                   # one limb against itself
            key = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
            if -c.dist > out.get(key, (0.0, 0))[0]:
                out[key] = (float(-c.dist), t)
    return out


def geom_label(model, g):
    name = model.geom(g).name or f"geom{g}"
    return f"{model.body(model.geom_bodyid[g]).name}/{name}"


def check_penetration(model, Q, baseline, tol, limbs=()):
    prof = penetration_profile(model, Q, limbs=limbs)
    worst, where = 0.0, None
    for key, (depth, t) in prof.items():
        excess = depth - baseline.get(key, (0.0, 0))[0]
        if excess > worst:
            worst, where = excess, (key, t, depth)
    rep = {"new_penetration_m": round(worst, 5), "ok": worst <= tol}
    if where:
        (a, b), t, depth = where
        rep.update(pair=[geom_label(model, a), geom_label(model, b)], frame=int(t),
                   depth_m=round(depth, 5))
    return rep


def joint_speed(Q, qadr, fps):
    if len(Q) < 2:
        return 0.0
    return float(np.abs(np.diff(Q[:, qadr], axis=0)).max() * fps)


def check_joint_speed(Q, Q0, chains, fps, scale, floor):
    rep, ok = {}, True
    for name, qadr in chains.items():
        v, v0 = joint_speed(Q, qadr, fps), joint_speed(Q0, qadr, fps)
        limit = scale * v0 + floor
        rep[name] = {"max": round(v, 3), "original": round(v0, 3), "limit": round(limit, 3)}
        ok &= v <= limit
    rep["ok"] = bool(ok)
    return rep


def check_object_steps_max(Q, Q0, obj_qadr, tol):
    """Continuity for a trajectory of another length than the original (re-executed or retimed): an object's largest
    frame-to-frame motion may not exceed the original's largest by tol."""
    rep, ok = {}, True
    for name, adr in obj_qadr.items():
        step = np.linalg.norm(np.diff(Q[:, adr:adr + 3], axis=0), axis=1)
        step0 = np.linalg.norm(np.diff(Q0[:, adr:adr + 3], axis=0), axis=1)
        worst = float(step.max() - (step0.max() if len(step0) else 0.0)) if len(step) else 0.0
        rep[name] = round(worst, 5)
        ok &= worst <= tol
    rep["ok"] = bool(ok)
    return rep


def check_object_steps(Q, Q0, obj_qadr, tol):
    """An object's frame-to-frame motion may not exceed the original's by tol."""
    rep, ok = {}, True
    for name, adr in obj_qadr.items():
        step = np.linalg.norm(np.diff(Q[:, adr:adr + 3], axis=0), axis=1)
        step0 = np.linalg.norm(np.diff(Q0[:, adr:adr + 3], axis=0), axis=1)
        worst = float((step - step0).max()) if len(step) else 0.0
        rep[name] = round(worst, 5)
        ok &= worst <= tol
    rep["ok"] = bool(ok)
    return rep
