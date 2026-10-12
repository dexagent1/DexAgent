"""Grasp turn test: does a pinched or wrapped part turn WITH the hand about a joint axis?"""
from __future__ import annotations

TOOL = {
    "name": "grasp_turn_test",
    "title": "Grasp turn test",
    "type": "verifier",
    "category": "Grasp Stress Tests",
    "summary": "Close the posed hand on a part that moves on a joint (a cap, a knob, a lid on a hinge; a drawer on a slide), "
               "drive the wrist along the joint's motion (an arc about a hinge, a translation along a slide) by a set "
               "amount in physics, and pass when the part followed the hand without the hand losing it.",
    "description": (
        "The stress test for a grasp whose job is torque. The hand scene is built with the object's root held fixed "
        "(a bottle steadied by the other hand, a cabinet on the floor) and the fingers are driven to the pose's closed "
        "and then pressed targets; the wrist is then servoed through an arc about the joint's axis (its world axis "
        "through the joint's world anchor, read from the model) in small steps, keeping the fingers' commands fixed and letting "
        "the hand rise with any thread coupling. It measures the joint angle reached against the angle the hand "
        "swept, the slip (sweep minus turn), the peak wrist force the arc needed, the pad contacts at the end, and "
        "whether any finger lost the part. Passes when the part turned at least `min_fraction` of the sweep and the "
        "hand still holds it at the end. Fails name the reason: SLIPPED (pads skate on the part), LOST (the hand came "
        "off), FLUNG (the part ran ahead of the hand: flicked, not held), STALLED (the joint did not move at all: wrong axis, joint blocked, or the pose presses the part "
        "sideways). Missing initial contact is reported as NO_CONTACT, not evidence of a blocked joint. Progress is "
        "signed along the requested direction; opposite motion cannot pass. Every simulated substep checks the same "
        "3 mm hand/object/self penetration limit as the other grasp stress tests, including grip pads, with visible "
        "support penetration checked at 60 Hz. Runs on the same scene dict the grasp skills take."
    ),
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects, grasp_body, joint_init, ...} as hand_scene_from_recon writes it."},
        {"name": "pose", "type": "dict", "description": "pos, euler, finger_targets[, finger_targets_squeezed, grip_fingers] from a grasp skill."},
        {"name": "joint", "type": "str", "description": "The hinge the part turns on, or the slide it travels on (in the hand scene's names, e.g. bottle_bottle_yaw)."},
        {"name": "sweep_deg", "type": "float", "description": "Angle the hand swings about a hinge axis. Default 90."},
        {"name": "sweep_m", "type": "float", "description": "Distance the hand travels along a slide axis. Default 0.08."},
        {"name": "direction", "type": "int", "description": "+1 or -1 about the joint axis. Default +1."},
        {"name": "seconds", "type": "float", "description": "Duration of the sweep. Default 2.0."},
        {"name": "min_fraction", "type": "float", "description": "Required turn / sweep. Default 0.8."},
        {"name": "max_fraction", "type": "float", "description": "Turn / sweep above which the part ran ahead of the hand (FLUNG, not held); a jump of more than three sweep steps in one step also counts. Default 1.25."},
        {"name": "hold_root", "type": "bool", "description": "Hold the object's free root during the test by making it 10^4 times heavier (default true)."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, verdict, measured {sweep_deg, turned_deg, fraction, slip_deg, lost_at_deg, pads_end, peak_wrist_force_n, peak_wrist_torque_nm, joint_before_close (the joint as the open hand arrives), joint_start (after the close, where the sweep is measured from), close_moved (deg or mm the close itself moved the part), joint_end}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "stress", "torque", "cap", "knob", "unscrew"],
    "used_in": ["Open the Bottle"],
    "status": "tested",
    "entrypoint": "grasp_turn_test",
}

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import hand as H


def grasp_turn_test(scene, pose, joint, sweep_deg=90.0, direction=1, seconds=2.0, min_fraction=0.8, hold_root=True,
                    out_json=None, sweep_m=0.08, max_fraction=1.25):
    import json
    from pathlib import Path
    model, data0, rig, ids = H.open_scene(scene)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        raise ValueError(f"no joint {joint!r}; joints: {[model.joint(j).name for j in range(model.njnt) if not model.joint(j).name.startswith(rig.prefix)]}")
    qa = int(model.jnt_qposadr[jid]); part = int(model.jnt_bodyid[jid])
    # the object's root (held during the test when it is free)
    root = part
    while int(model.body_parentid[root]) != 0:
        root = int(model.body_parentid[root])
    free_adr = None
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) == root and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_adr = (int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
    if hold_root and free_adr is not None:
        # hold the base physically: 10^4 times its mass and inertia (it rests on its support and the hand cannot move
        # it). Teleporting it back every step kicked the part and flung a dial 150 deg on a 40-deg sweep.
        model.body_mass[root] *= 1e4; model.body_inertia[root] *= 1e4
        mujoco.mj_setConst(model, mujoco.MjData(model))   # scratch data: mj_setConst resets the qpos it is given
    d = mujoco.MjData(model); d.qpos[:] = data0.qpos; d.qvel[:] = 0
    clamp = H.executor_clamp(rig, pose)
    closed, press, grip = clamp["closed"], clamp["press"], clamp["grip_fingers"]
    own_open = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    opened = own_open or H.approach_open_targets(rig, closed, float(pose.get("open_frac", 0.55)), grip, clamp["grip_posed"])
    pos = np.asarray(pose["pos"], float); eul = np.asarray(pose["euler"], float)
    dt = float(model.opt.timestep)
    sub = set()
    stack = [part]
    while stack:
        b = stack.pop(); sub.add(b)
        stack += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b]
    # bodies welded to the part belong to it (a cap is its own free body welded to the thread carrier)
    for e in range(model.neq):
        if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_WELD):
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            other = b2 if b1 in sub else (b1 if b2 in sub else None)
            if other is not None and other not in sub and other != 0:
                stack = [other]
                while stack:
                    b = stack.pop(); sub.add(b)
                    stack += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b]
    part_gids = {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub}
    peak = [0.0, 0.0]
    lost_at = None
    turned_hist = []
    monitor = H.GraspPenetrationMonitor(model, rig, ids)

    def held_step(m_, d_, lab_):                     # penetration is checked through every ramp, not only between them
        monitor(m_, d_, lab_)

    def pads_on_part():
        return sorted({H.finger_of_joint(model.body(int(model.geom_bodyid[c.geom1 if int(model.geom_bodyid[c.geom1]) in rig.body_ids else c.geom2])).name.split(f"{rig.side}_")[-1]) or "palm"
                       for i in range(d.ncon) for c in [d.contact[i]]
                       if (int(model.geom_bodyid[c.geom1]) in rig.body_ids and c.geom2 in part_gids) or (int(model.geom_bodyid[c.geom2]) in rig.body_ids and c.geom1 in part_gids)})

    def run(p, e, fingers, secs, label="hold"):
        for _ in range(max(1, int(secs / dt))):
            H.hold_ctrl(d, rig, p, e, fingers); mujoco.mj_step(model, d)
            monitor(model, d, label)
            fn = float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("x", "y", "z")]))
            tn = float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("rx", "ry", "rz")]))
            peak[0] = max(peak[0], fn); peak[1] = max(peak[1], tn)

    # approach from above the pose, close, press
    a = H.approach_axis(model, d, rig, pose)
    rig.teleport(d, pos + a * 0.06, eul); rig.set_fingers(d, opened); mujoco.mj_forward(model, d)
    monitor(model, d, "initial approach")
    run(pos + a * 0.06, eul, opened, 0.3, "settle")
    H.ramp(model, d, rig, pos + a * 0.06, pos, eul, eul, opened, opened, 0.5, held_step, label="approach")
    q_open = float(d.qpos[qa])                       # the part's state as the open hand arrives (joint_init kept)
    H.ramp(model, d, rig, pos, pos, eul, eul, opened, closed, 0.4, held_step, label="close")
    # Apply the executor's complete clamp through physical servo commands.
    # One-sided contacts sometimes need wrist travel instead of finger curl.
    pressed_pos = pos + (np.asarray(clamp["wrist_press"], float) if clamp["wrist_press"] is not None else 0.)
    H.ramp(model, d, rig, pos, pressed_pos, eul, eul, closed, press, .3, held_step, label="grip")
    pos = pressed_pos
    run(pos, eul, press, .3)
    q0 = float(d.qpos[qa]); pads0 = pads_on_part()
    # how far closing the hand moved the part before the sweep: pads closing on a free part push it along their
    # curl (a pinch closing on a free dial's rim turns it ~4.5 deg). Reported, not judged: the turn is measured from q0
    close_moved = (1000.0 * abs(q0 - q_open)) if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else float(np.degrees(abs(q0 - q_open)))
    # sweep: wrist arc about the actual hinge anchor (5 degree steps) or a translation along a
    # slide axis (5 mm steps), fingers fixed at press
    mujoco.mj_forward(model, d)
    is_slide = int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
    axis = d.xaxis[jid].copy(); axis /= np.linalg.norm(axis)
    centre = d.xanchor[jid].copy()
    total = float(sweep_m) if is_slide else abs(float(sweep_deg))
    n_steps = max(1, int(np.ceil(total / (0.005 if is_slide else 5.0))))
    step = (total if is_slide else np.radians(total)) / n_steps * (1 if direction >= 0 else -1)
    p_cur, e_cur = pos.copy(), eul.copy()
    R_cur = H.euler_to_mat(e_cur)
    swept = 0.0
    sign = 1. if direction >= 0 else -1.
    units_per_rad = 1000. if is_slide else 180. / np.pi
    for i in range(n_steps):
        if is_slide:
            p_new, R_new = p_cur + step * axis, R_cur
        else:
            Rd = Rot.from_rotvec(step * axis).as_matrix()
            p_new = centre + Rd @ (p_cur - centre); R_new = Rd @ R_cur
        e_new = H.nearest_euler_branch(H.mat_to_euler(R_new), e_cur)
        H.ramp(model, d, rig, p_cur, p_new, e_cur, e_new, press, press, seconds / n_steps, held_step, label="sweep")
        p_cur, e_cur, R_cur = p_new, e_new, R_new
        swept += abs(step) if is_slide else abs(np.degrees(step))
        turned_hist.append(round(float(sign * units_per_rad * (d.qpos[qa] - q0)), 1))
        if lost_at is None and not pads_on_part():
            lost_at = round(swept, 1)
        fn = float(np.linalg.norm([d.actuator_force[rig.wrist_act[k]] for k in ("x", "y", "z")]))
        peak[0] = max(peak[0], fn)
    run(p_cur, e_cur, press, 0.3)
    signed_delta = float(units_per_rad * (d.qpos[qa] - q0))
    turned = sign * signed_delta
    span = 1000 * total if is_slide else total
    unit = "mm" if is_slide else "deg"
    frac = turned / max(span, 1e-9)
    pads_end = pads_on_part()
    # the part must follow the hand, not run ahead of it: a part flicked by a slipping pad over-travels (a lid flung
    # 23 deg in one 5-deg step, a dial spun 150 deg on a 40-deg sweep) and is not held
    hist = np.asarray([0.0] + turned_hist, float)
    max_jump = float(np.max(np.abs(np.diff(hist)))) if len(hist) > 1 else 0.0
    step_span = (1000 * abs(step)) if is_slide else abs(np.degrees(step))
    flung = frac > float(max_fraction) or max_jump > 3.0 * step_span
    penetration = monitor.report()
    ok = frac >= float(min_fraction) and not flung and bool(pads_end) and len(pads_end) >= (1 if is_slide else 2) and penetration["ok"]
    verb = "PULLED" if is_slide else "TURNED"
    if not penetration["ok"]:
        verdict = (f"PENETRATION — contact depth {penetration['max_contact_mm']:.1f} mm / visible support depth "
                   f"{penetration['max_visual_floor_mm']:.1f} mm exceeded the 3 mm limit")
    elif turned < -max(.1, .1 * span):
        verdict = f"WRONG_DIRECTION — the part moved {-turned:.1f} {unit} against the requested joint direction"
    elif flung and frac >= float(min_fraction):
        verdict = (f"FLUNG — the part ran ahead of the hand: {turned:.0f} {unit} for a {span:.0f} {unit} sweep, up to {max_jump:.0f} {unit} in "
                   f"one {step_span:.0f}-{unit} step; the pads flick it instead of holding it (a seat deeper on the part, a firmer clamp, or "
                   f"more of the hand on it)")
    elif ok:
        verdict = f"{verb} — the part followed {turned:.0f} of {span:.0f} {unit} ({100*frac:.0f}%), held by {pads_end} at the end"
    elif not pads0:
        verdict = (f"NO_CONTACT — no hand contact with the moving part was established before the sweep "
                   f"(joint moved {turned:.0f} {unit}); this does not establish that the joint is blocked")
    elif turned < 0.1 * span:
        verdict = f"STALLED — the joint moved {turned:.0f} {unit} while the hand swept {span:.0f}; wrong axis, blocked joint, or the pose pushes the part sideways"
    elif not pads_end:
        verdict = f"LOST — the hand came off the part after {lost_at} {unit} of sweep (moved {turned:.0f} {unit})"
    else:
        verdict = f"SLIPPED — the pads skated: part moved {turned:.0f} of {span:.0f} {unit} ({100*frac:.0f}%)"
    rep = {"ok": bool(ok), "verdict": verdict, "failure_kind": None if ok else verdict.split(" —", 1)[0].lower(),
        "penetration": penetration, "measured": {
        "joint_type": "slide" if is_slide else "hinge", "sweep": float(span), "unit": unit,
        "sweep_deg": float(sweep_deg), "turned_deg": round(turned, 1), "fraction": round(frac, 3), "slip_deg": round(span - turned, 1),
        "requested_direction": int(sign), "signed_joint_delta": round(signed_delta, 3),
        "lost_at_deg": lost_at, "pads_start": pads0, "pads_end": pads_end, "turn_history_deg": turned_hist, "max_step_jump": round(max_jump, 1), "flung": bool(flung),
        "peak_wrist_force_n": round(peak[0], 1), "peak_wrist_torque_nm": round(peak[1], 2), "joint_start": round(q0, 4), "joint_end": round(float(d.qpos[qa]), 4),
        "joint_before_close": round(q_open, 4), "close_moved": round(close_moved, 2)}}
    if out_json:
        Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep
