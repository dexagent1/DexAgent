TOOL = {
    "name": "joint_release_hold_test",
    "title": "Joint release hold test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Sets a hinge or slide joint to each of several positions, releases it under gravity with no actuation, and checks that it stays there (a lid that stays open, a drawer that stays half out).",
    "description": (
        "For every requested joint position the model is reset, the joint is set to that position with zero "
        "velocity, and the passive dynamics run for hold_s seconds with gravity. The joint must end within "
        "tol of where it was released, and a free root body must not slide more than root_slip_m. A model "
        "without a plane geom gets a floor at z=0 (add_floor) so a free base rests instead of falling, which "
        "would hide gravity torques on the joint. Positions are degrees for hinges and metres for slides. "
        "Positions outside the joint range are reported as issues, not clipped. Assumes the asset rests on "
        "the floor at its initial state (base at z=0)."
    ),
    "inputs": [
        {"name": "xml", "type": "path", "description": "MJCF of the asset (or a scene) containing the joint."},
        {"name": "joint", "type": "str", "description": "Name of the hinge or slide joint to test."},
        {"name": "positions", "type": "list[float]", "description": "Release positions: degrees (hinge) or metres (slide)."},
        {"name": "hold_s", "type": "float", "description": "Seconds of passive simulation per release. Default 3.0."},
        {"name": "tol", "type": "float", "description": "Allowed drift: degrees (hinge) or metres (slide). Default 3.0 for hinges, 0.005 for slides when None."},
        {"name": "root_slip_m", "type": "float", "description": "Allowed translation of a free root body. Default 0.01."},
        {"name": "add_floor", "type": "bool", "description": "Add a floor plane at z=0 when the model has no plane geom. Default True."},
        {"name": "keyframe", "type": "int | None", "description": "Keyframe to start from (a scene's home pose). Default None = qpos0."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "Every release position held within tol and the root did not slip."},
        {"name": "issues", "type": "list[str]", "description": "What failed and what parameter to change."},
        {"name": "trials", "type": "list[dict]", "description": "Per position: released, final, drift, root_slip_m, held."},
        {"name": "joint_type", "type": "str", "description": "hinge or slide."},
        {"name": "range", "type": "list[float]", "description": "Joint range in the reported units."},
    ],
    "status": "tested",
    "entrypoint": "joint_release_hold_test",
    "requirements": ["mujoco", "numpy"],
    "tags": ["hinge", "slide", "lid", "drawer", "hold", "friction", "articulated", "verifier"],
    "used_in": [],
}

import math

import mujoco
import numpy as np


def _load(xml, add_floor):
    spec = mujoco.MjSpec.from_file(str(xml))
    has_plane = any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms)
    if add_floor and not has_plane:
        g = spec.worldbody.add_geom()
        g.name = "_hold_test_floor"
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [2.0, 2.0, 0.1]
        g.pos = [0.0, 0.0, 0.0]
    return spec.compile()


def joint_release_hold_test(xml, joint, positions, hold_s=3.0, tol=None, root_slip_m=0.01,
                            add_floor=True, keyframe=None):
    if not positions:
        raise ValueError("positions is empty: give at least one release position")
    if hold_s <= 0:
        raise ValueError("hold_s must be positive")
    issues, trials = [], []
    try:
        m = _load(xml, add_floor)
    except Exception as e:  # a bad model is a verdict, not a crash
        return {"ok": False, "issues": [f"model does not compile: {e}"], "trials": [], "joint_type": None, "range": None}
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
        return {"ok": False, "issues": [f"joint '{joint}' not found; joints: {names}"], "trials": [],
                "joint_type": None, "range": None}
    jt = m.jnt_type[jid]
    if jt == mujoco.mjtJoint.mjJNT_HINGE:
        jtype, to_q, from_q = "hinge", math.radians, math.degrees
        tol = 3.0 if tol is None else float(tol)
        span_deg = math.degrees(float(m.jnt_range[jid][1] - m.jnt_range[jid][0])) if m.jnt_limited[jid] else 360.0
        if span_deg > 30.0 and all(abs(float(x)) <= 3.2 for x in positions) and any(abs(float(x)) > 0.0 for x in positions):
            # positions are degrees here; values all under 3.2 on a joint that spans tens of degrees are radians
            # by mistake (a lid tested at 1.6 degrees instead of 90 'passes' trivially)
            issues.append(f"positions {list(positions)} look like radians: this test takes DEGREES for a hinge "
                          f"(the joint spans {span_deg:.0f} deg); pass e.g. {[round(math.degrees(float(x)), 1) for x in positions]}")
    elif jt == mujoco.mjtJoint.mjJNT_SLIDE:
        jtype, to_q, from_q = "slide", float, float
        tol = 0.005 if tol is None else float(tol)
    else:
        return {"ok": False, "issues": [f"joint '{joint}' is not a hinge or slide"], "trials": [],
                "joint_type": "other", "range": None}
    adr = m.jnt_qposadr[jid]
    rng = [from_q(m.jnt_range[jid][0]), from_q(m.jnt_range[jid][1])] if m.jnt_limited[jid] else None
    # free root of the joint's tree, if any
    root = m.body_rootid[m.jnt_bodyid[jid]]
    free_adr = None
    for j in range(m.njnt):
        if m.jnt_bodyid[j] == root and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            free_adr = m.jnt_qposadr[j]
    d = mujoco.MjData(m)
    n = int(round(hold_s / m.opt.timestep))
    for pos in positions:
        pos = float(pos)
        if rng is not None and not (rng[0] - 1e-9 <= pos <= rng[1] + 1e-9):
            issues.append(f"release position {pos} is outside the joint range {rng}: widen the joint range "
                          f"or test inside it")
            trials.append({"released": pos, "final": None, "drift": None, "root_slip_m": None, "held": False})
            continue
        if keyframe is not None:
            mujoco.mj_resetDataKeyframe(m, d, int(keyframe))
        else:
            mujoco.mj_resetData(m, d)
        d.qpos[adr] = to_q(pos)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        p0 = d.qpos[free_adr:free_adr + 3].copy() if free_adr is not None else None
        # MuJoCo resets the state on a bad acceleration and carries on: the joint would then read its reset value
        bad0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        finite = True
        for _ in range(n):
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)) or int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > bad0:
                finite = False
                break
        if not finite:
            issues.append(f"simulation diverged after release at {pos} (non-finite, or MuJoCo reset it on a bad acceleration): "
                          f"check joint armature/damping")
            trials.append({"released": pos, "final": None, "drift": None, "root_slip_m": None, "held": False})
            continue
        final = from_q(d.qpos[adr])
        drift = final - pos
        slip = float(np.linalg.norm(d.qpos[free_adr:free_adr + 2] - p0[:2])) if p0 is not None else 0.0
        held = abs(drift) <= tol and slip <= root_slip_m
        if abs(drift) > tol:
            issues.append(f"{jtype} '{joint}' released at {pos:.3f} drifted to {final:.3f} ({drift:+.3f}) in "
                          f"{hold_s}s: raise its frictionloss (holding torque/force) or armature")
        if slip > root_slip_m:
            issues.append(f"free root slid {slip:.4f} m after release at {pos}: raise base friction or mass")
        trials.append({"released": pos, "final": round(final, 4), "drift": round(drift, 4),
                       "root_slip_m": round(slip, 5), "held": bool(held)})
    ok = not issues and all(t["held"] for t in trials)
    return {"ok": bool(ok), "issues": issues, "trials": trials, "joint_type": jtype, "range": rng, "tol": tol,
            "hold_s": hold_s}
