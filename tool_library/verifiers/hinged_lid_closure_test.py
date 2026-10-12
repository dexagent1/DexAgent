TOOL = {
    "name": "hinged_lid_closure_test",
    "title": "Hinged lid: rests open, closes, stays shut",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Three force-only trials on a box with a hinged lid (and optional skirt crease): the lid stays where frame 0 left it, a capped hinge torque swings it shut (then folds the skirt down), and released it stays seated shut under gravity.",
    "description": (
        "Loads a standalone asset MJCF (with its floor) and runs three trials from qpos0. REST: no forces for "
        "rest_s seconds; the lid and skirt must stay within tol_deg of their initial angles (an open lid lying "
        "folded back must not creep). CLOSE: a speed-limited torque on the lid joint toward closed_deg, capped at torque_factor "
        "times the lid subtree's gravity torque about the hinge (or torque_Nm), for close_s seconds, then a "
        "capped torque on the skirt joint toward skirt_closed_deg; both must reach their closed angle within "
        "tol_deg. HOLD: all torques removed for hold_s seconds; the lid and skirt must stay shut within tol_deg "
        "and the base must not slip more than root_slip_m at rest or after release. Torques act on the joints directly (qfrc_applied), so "
        "the trial judges the mechanism and its stops, not a grasp. Returns ok, issues and every measured angle."
    ),
    "inputs": [
        {"name": "xml", "type": "path", "description": "Standalone asset MJCF (built with include_floor on)."},
        {"name": "lid_joint", "type": "str", "description": "Name of the lid hinge joint."},
        {"name": "skirt_joint", "type": "str | None", "description": "Name of the skirt crease joint, or None. Default None."},
        {"name": "closed_deg", "type": "float", "description": "Lid joint value when shut. Default 0."},
        {"name": "skirt_closed_deg", "type": "float", "description": "Skirt joint value when folded down. Default 90."},
        {"name": "torque_Nm", "type": "float | None", "description": "Fixed closing torque; None = torque_factor x gravity torque. Default None."},
        {"name": "torque_factor", "type": "float", "description": "Multiple of the moving part's gravity torque. Default 3.0."},
        {"name": "rest_s", "type": "float", "description": "REST trial length, s. Default 2.0."},
        {"name": "close_s", "type": "float", "description": "Seconds per closing push. Default 3.0."},
        {"name": "hold_s", "type": "float", "description": "HOLD trial length, s. Default 3.0."},
        {"name": "tol_deg", "type": "float", "description": "Angle tolerance, deg. Default 3.0."},
        {"name": "root_slip_m", "type": "float", "description": "Allowed base translation during REST and HOLD (the pushes may shift a free base), m. Default 0.01."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; rest: initial/final angles and drift; close: reached angles and torques; hold: final angles and drift; root_slip_m."},
    ],
    "status": "tested",
    "entrypoint": "hinged_lid_closure_test",
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["lid", "hinge", "box", "close", "stays shut", "articulated", "mechanism test"],
    "used_in": ["Giftbox Packing"],
}

import math
from pathlib import Path

import numpy as np


def _grav_torque(m, d, jid):
    """Gravity torque magnitude of the joint's child subtree about the hinge axis."""
    b = m.jnt_bodyid[jid]
    mass = float(m.body_subtreemass[b])
    com = d.subtree_com[b]
    anchor = d.xanchor[jid]; axis = d.xaxis[jid]
    r = com - anchor
    tq = np.cross(r, np.array([0, 0, -9.81 * mass]))
    return abs(float(np.dot(tq, axis))) + 1e-4, mass


def hinged_lid_closure_test(xml, lid_joint, skirt_joint=None, closed_deg=0.0, skirt_closed_deg=90.0,
                            torque_Nm=None, torque_factor=3.0, rest_s=2.0, close_s=3.0, hold_s=3.0,
                            tol_deg=3.0, root_slip_m=0.01):
    import mujoco
    if not Path(xml).exists():
        raise ValueError(f"xml {xml!r} does not exist")
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    issues = []
    rep = {"ok": False, "issues": issues}
    try:
        jl = m.joint(lid_joint).id
    except KeyError:
        return dict(rep, issues=[f"lid_joint {lid_joint!r} not in the model; pass the lid hinge's name"])
    js = None
    if skirt_joint:
        try:
            js = m.joint(skirt_joint).id
        except KeyError:
            return dict(rep, issues=[f"skirt_joint {skirt_joint!r} not in the model"])
    for j in [jl] + ([js] if js is not None else []):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            return dict(rep, issues=[f"joint {m.joint(j).name} is not a hinge"])
    ql, qs = m.jnt_qposadr[jl], (m.jnt_qposadr[js] if js is not None else None)
    dl, ds = m.jnt_dofadr[jl], (m.jnt_dofadr[js] if js is not None else None)
    deg = lambda a: math.degrees(float(d.qpos[a]))
    roots = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    root = m.jnt_bodyid[jl]
    while m.body_parentid[root] != 0:
        root = m.body_parentid[root]
    mujoco.mj_forward(m, d)
    p_root0 = d.xpos[root].copy()
    lid0 = deg(ql); sk0 = deg(qs) if qs is not None else None

    def run(seconds, tgt_l=None, cap_l=0.0, tgt_s=None, cap_s=0.0):
        """Torque-capped, speed-limited push of each joint toward its target (None = no push). False when the
        simulation diverged: non-finite, or reset by MuJoCo on a bad acceleration (it then restarts from qpos0, and
        every later angle reads as if nothing had moved)."""
        n = int(seconds / m.opt.timestep)
        bad0 = int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        for _ in range(n):
            d.qfrc_applied[:] = 0.0
            for tgt, cap, qa, da in ((tgt_l, cap_l, ql, dl), (tgt_s, cap_s, qs, ds)):
                if tgt is None or da is None:
                    continue
                err = math.radians(tgt) - float(d.qpos[qa])
                d.qfrc_applied[da] = float(np.clip(cap * err / 0.03 - cap * float(d.qvel[da]) / 1.5, -cap, cap))
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)) or int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > bad0:
                d.qfrc_applied[:] = 0.0
                return False
        d.qfrc_applied[:] = 0.0
        return True

    # REST
    ok_fin = run(rest_s)
    rest = {"lid_init_deg": round(lid0, 2), "lid_after_deg": round(deg(ql), 2)}
    if qs is not None:
        rest.update(skirt_init_deg=round(sk0, 2), skirt_after_deg=round(deg(qs), 2))
    rest["lid_drift_deg"] = round(abs(deg(ql) - lid0), 2)
    if not ok_fin:
        issues.append("the simulation diverged in the REST trial (non-finite, or MuJoCo reset it on a bad acceleration); "
                      "check masses, armature and contacts")
    if rest["lid_drift_deg"] > tol_deg:
        issues.append(f"REST: lid crept {rest['lid_drift_deg']:.1f} deg from its initial angle; raise the lid "
                      f"frictionloss or set the initial angle where the lid rests on its support")
    if qs is not None and abs(deg(qs) - sk0) > tol_deg:
        issues.append(f"REST: skirt crept {abs(deg(qs) - sk0):.1f} deg; raise the skirt frictionloss/armature")
    # CLOSE
    mujoco.mj_forward(m, d)
    g_l, mass_l = _grav_torque(m, d, jl)
    tl = float(torque_Nm) if torque_Nm is not None else torque_factor * g_l
    slip_rest = float(np.linalg.norm((d.xpos[root] - p_root0)[:2]))
    ok_close = run(close_s, tgt_l=closed_deg, cap_l=tl)
    lid_closed = deg(ql)
    close = {"lid_torque_Nm": round(tl, 4), "lid_gravity_torque_Nm": round(g_l, 4),
             "lid_mass_kg": round(mass_l, 4), "lid_reached_deg": round(lid_closed, 2)}
    if abs(lid_closed - closed_deg) > tol_deg:
        issues.append(f"CLOSE: the lid only reached {lid_closed:.1f} deg (shut = {closed_deg}); the joint range "
                      f"or a collision blocks it, or the lid is too heavy for {tl:.3f} N m")
    if qs is not None:
        mujoco.mj_forward(m, d)
        g_s, _ = _grav_torque(m, d, js)
        ts = float(torque_Nm) if torque_Nm is not None else max(torque_factor * g_s,
                                                                 2.0 * float(m.dof_frictionloss[ds]) + 1e-3)
        ok_close = run(close_s, tgt_s=skirt_closed_deg, cap_s=ts) and ok_close
        close.update(skirt_torque_Nm=round(ts, 4), skirt_reached_deg=round(deg(qs), 2),
                     lid_with_skirt_deg=round(deg(ql), 2))
        if abs(deg(qs) - skirt_closed_deg) > tol_deg:
            issues.append(f"CLOSE: the skirt only reached {deg(qs):.1f} deg (down = {skirt_closed_deg}); "
                          f"its range or the front wall blocks it")
    if not ok_close:
        issues.append("the simulation diverged in the CLOSE trial (non-finite, or MuJoCo reset it on a bad acceleration); "
                      "the reached angles are not measurements: check the joint armature/damping and the stops' contacts")
    # HOLD
    lid_c, sk_c = deg(ql), (deg(qs) if qs is not None else None)
    mujoco.mj_forward(m, d)
    p_root1 = d.xpos[root].copy()
    ok_fin2 = run(hold_s)
    hold = {"lid_final_deg": round(deg(ql), 2), "lid_drift_deg": round(abs(deg(ql) - lid_c), 2)}
    if qs is not None:
        hold.update(skirt_final_deg=round(deg(qs), 2), skirt_drift_deg=round(abs(deg(qs) - sk_c), 2))
    if not ok_fin2:
        issues.append("the simulation diverged in the HOLD trial (non-finite, or MuJoCo reset it on a bad acceleration)")
    if abs(deg(ql) - closed_deg) > tol_deg:
        issues.append(f"HOLD: released, the lid opened to {deg(ql):.1f} deg; it does not stay shut "
                      f"(a springy hinge or a stop that does not hold)")
    if qs is not None and abs(deg(qs) - skirt_closed_deg) > tol_deg:
        issues.append(f"HOLD: released, the skirt swung to {deg(qs):.1f} deg; raise its frictionloss")
    mujoco.mj_forward(m, d)
    slip = max(slip_rest, float(np.linalg.norm((d.xpos[root] - p_root1)[:2])))
    if slip > root_slip_m:
        issues.append(f"the base slid {slip * 100:.1f} cm at rest or after release; it is not stable on its own")
    return {"ok": not issues, "issues": issues, "rest": rest, "close": close, "hold": hold,
            "root_body": m.body(root).name, "root_slip_m": round(slip, 4), "n_roots": len(roots)}


if __name__ == "__main__":
    import json, sys
    print(json.dumps(hinged_lid_closure_test(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or [None])), indent=1))
