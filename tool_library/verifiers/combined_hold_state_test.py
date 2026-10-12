"""Stress a combined rigid grasp from its actual recorded simulator state."""
from __future__ import annotations

TOOL = {
    "name": "combined_hold_state_test", "title": "Combined hold from recorded state",
    "type": "verifier", "category": "Grasp Stress Tests",
    "summary": "Test every object retained by one hand from an executed grasp_more state, with the real rigid arm and actual finger commands.",
    "description": (
        "Replays recorded controls to the specified frame without resetting object or hand coordinates. "
        "From that complete simulator state, the unchanged bounded arm/finger controller holds and moves each "
        "arm joint three degrees in both directions. Every rigid object must remain in hand, clear of external "
        "supports, within 20 mm and 15 degrees of its initial wrist-relative pose, with penetration at most 3 mm. "
        "Use the final frame of a successful grasp_more segment after lifting every object clear of its support. "
        "The manifest independently binds the frame and complete retained-object set to the trusted execution "
        "history. This tests the actual combined hold; multi_object_hold_test remains useful for candidate search "
        "before the first grasp. Flex objects and dynamic mechanism equalities are currently unsupported."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "The executed rigid scene_physics.xml."},
        {"name": "trajectory", "type": "str", "description": "Complete actual trajectory.npz with dense control/state recording."},
        {"name": "frame", "type": "int", "description": "Last frame of the grasp_more segment, with all held parts lifted."},
        {"name": "side", "type": "str", "description": "left or right."},
        {"name": "objects", "type": "list[str]", "description": "All objects still retained by this hand, in acquisition order."},
    ],
    "outputs": [{"name": "report", "type": "dict", "description": "ok, issues, prefix replay and per-object continuous stress measurements."}],
    "requirements": ["mujoco>=3.2", "numpy"], "simulators": ["MuJoCo"],
    "tags": ["grasp", "stress", "recorded state", "multiple objects"],
    "status": "tested", "entrypoint": "combined_hold_state_test",
}

import mujoco
import numpy as np


def _recorded_state(model, scene, trajectory, frame):
    """Reconstruct hidden simulator state by control replay, never pose injection."""
    from tool_library.common import episode as E, rigid_robot as RR
    from tool_library.common.evidence import file_digest
    keys = {"qpos", "qvel", "ctrl", "time_s", "initial_state", "state_spec", "scene_sha256",
            "robot_schema", "joint_names", "actuator_names", "substep_qpos", "substep_qvel",
            "substep_ctrl", "substep_time_s"}
    if model.neq:
        keys |= {"eq_active", "eq_active_start", "substep_eq_active", "substep_eq_active_before", "substep_eq_data"}
    if model.na:
        keys |= {"act", "substep_act"}
    with np.load(trajectory, allow_pickle=False) as source:
        if keys - set(source.files):
            raise ValueError("missing dense recorded state inputs: " + ", ".join(sorted(keys - set(source.files))))
        z = {key: source[key] for key in keys}
    if (str(z["scene_sha256"]) != file_digest(scene) or str(z["robot_schema"]) != RR.SCHEMA
            or list(z["joint_names"].astype(str)) != E.joint_names(model)
            or list(z["actuator_names"].astype(str)) != [model.actuator(i).name for i in range(model.nu)]):
        raise ValueError("recording does not identify the supplied rigid scene and coordinate layout")
    if isinstance(frame, bool) or int(frame) != frame or not 0 <= int(frame) < len(z["qpos"]):
        raise ValueError("frame must identify an actual recorded frame")
    frame = int(frame)
    dt = float(model.opt.timestep)
    T, N = len(z["qpos"]), len(z["substep_time_s"])
    shapes = {"qpos": (T, model.nq), "qvel": (T, model.nv), "ctrl": (T, model.nu), "time_s": (T,),
              "substep_qpos": (N, model.nq), "substep_qvel": (N, model.nv), "substep_ctrl": (N, model.nu),
              "substep_time_s": (N,), "initial_state": (mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION),)}
    if model.neq:
        shapes.update(eq_active=(T, model.neq), eq_active_start=(model.neq,), substep_eq_active=(N, model.neq),
                      substep_eq_active_before=(N, model.neq), substep_eq_data=(N, model.neq, 11))
    if model.na:
        shapes.update(act=(T, model.na), substep_act=(N, model.na))
    for key, shape in shapes.items():
        if z[key].shape != shape or not np.isfinite(z[key]).all():
            raise ValueError(f"invalid recorded {key}: expected finite {shape}")
    if int(z["state_spec"]) != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError("complete integration state is required")
    if not np.allclose(z["substep_time_s"], np.arange(1, N + 1) * dt, atol=1e-8, rtol=0):
        raise ValueError("recording omits or resets integration steps")
    steps = int(round(float(z["time_s"][frame]) / dt))
    if not 1 <= steps <= N or abs(steps * dt - z["time_s"][frame]) > 1e-8:
        raise ValueError("stress state must be a post-integration recorded frame")
    if model.neq and (not np.all(z["eq_active"][:frame + 1] == model.eq_active0)
                      or not np.all(z["substep_eq_active"][:steps] == model.eq_active0)
                      or not np.all(z["substep_eq_active_before"][:steps] == model.eq_active0)
                      or not np.allclose(z["substep_eq_data"][:steps], model.eq_data[None], atol=1e-12, rtol=0)):
        raise ValueError("dynamic mechanism equalities are unsupported in combined hold stress")
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, z["initial_state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
        raise ValueError("recording starts with unaccounted external forces")
    if not np.allclose(data.qpos, z["qpos"][0], atol=1e-12, rtol=0) or not np.allclose(data.qvel, z["qvel"][0], atol=1e-12, rtol=0):
        raise ValueError("complete initial state disagrees with frame zero")
    if model.neq and not np.array_equal(data.eq_active, model.eq_active0):
        raise ValueError("initial dynamic equality state is unsupported")
    limited = np.asarray(model.actuator_ctrllimited, bool)
    controls = z["substep_ctrl"][:steps, limited]
    lo, hi = model.actuator_ctrlrange[limited].T
    if np.any(controls < lo - 1e-9) or np.any(controls > hi + 1e-9):
        raise ValueError("recorded controls exceed actuator limits")
    q_error, v_error = 0., 0.
    for i in range(steps):
        data.ctrl[:] = z["substep_ctrl"][i]
        mujoco.mj_step(model, data)
        dq = float(np.max(np.abs(data.qpos - z["substep_qpos"][i]), initial=0))
        dv = float(np.max(np.abs(data.qvel - z["substep_qvel"][i]), initial=0))
        q_error, v_error = max(q_error, dq), max(v_error, dv)
        if not np.isfinite(dq + dv) or dq > 1e-8 or dv > 1e-7:
            raise ValueError(f"independent prefix replay diverged at step {i}: qpos {dq}, qvel {dv}")
        if model.na and not np.allclose(data.act, z["substep_act"][i], atol=1e-10, rtol=0):
            raise ValueError("actuator activation disagrees with control replay")
    if (not np.allclose(data.qpos, z["qpos"][frame], atol=1e-8, rtol=0)
            or not np.allclose(data.qvel, z["qvel"][frame], atol=1e-7, rtol=0)
            or not np.array_equal(data.ctrl, z["ctrl"][frame])):
        raise ValueError("selected frame is not the state and controls produced by replay")
    return data, {"steps": steps, "max_qpos_error": q_error, "max_qvel_error": v_error}


def _held_groups(model, objects):
    from tool_library.common import episode as E
    groups, bids = {}, {}
    for name in objects:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid <= 0 or name.startswith("robot_"):
            raise ValueError(f"invalid held body {name!r}")
        bodies = E.subtree_bodies(model, bid)
        joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in bodies]
        if (len(joints) != 1 or int(model.jnt_type[joints[0]]) != int(mujoco.mjtJoint.mjJNT_FREE)
                or int(model.jnt_bodyid[joints[0]]) != bid):
            raise ValueError(f"{name}: combined hold stress supports free rigid roots only")
        if any(bodies & previous for previous in groups.values()):
            raise ValueError("held objects must be disjoint rigid bodies")
        if np.any(model.body_gravcomp[list(bodies)] != 0):
            raise ValueError(f"{name}: object gravity compensation cannot certify a physical grasp")
        for eq in range(model.neq):
            if (model.eq_active0[eq] and int(model.eq_type[eq]) in (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT))
                    and (int(model.eq_obj1id[eq]) in bodies or int(model.eq_obj2id[eq]) in bodies)):
                raise ValueError(f"{name}: non-contact constraint cannot certify a physical grasp")
        groups[name], bids[name] = bodies, bid
    return groups, bids


def combined_hold_state_test(scene, trajectory, frame, side, objects):
    from tool_library.common import episode as E, held_contact as HC, rigid_robot as RR
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not RR.is_rigid(model) or model.nflex:
        raise ValueError("combined actual-state stress currently requires a rigid robot and rigid objects")
    if side not in ("left", "right") or len(objects) < 2 or len(set(objects)) != len(objects):
        raise ValueError("give one hand and every distinct held object (at least two)")
    parts = E.robot_parts(model)
    if side not in parts:
        raise ValueError("the requested hand is absent")
    pt = parts[side]
    hand = set(pt["rig"].body_ids)
    groups, bids = _held_groups(model, objects)
    data, replay = _recorded_state(model, str(scene), str(trajectory), frame)
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    ref = data.qpos.copy()
    # Use actual finger servo targets, not observed joint positions or an old
    # pose's proposed clamp. Arm targets initially hold the achieved FK.
    for item in parts.values():
        for aid, qadr in E.actuator_targets(model, item["rig"]):
            ref[qadr] = data.ctrl[aid]
    fps = 30.
    rows = [ref.copy() for _ in range(16)]
    phase_ends = [("initial hold", len(rows) - 1)]
    for joint, qa in zip(pt["arm_joints"], pt["arm_qadr"]):
        for sign in (1., -1.):
            target = ref.copy()
            target[qa] += sign * np.radians(3.)
            if model.jnt_limited[joint]:
                target[qa] = np.clip(target[qa], *model.jnt_range[joint])
            if abs(target[qa] - ref[qa]) < np.radians(.5):
                continue
            rows.extend(ref * (1 - u) + target * u for u in np.linspace(0, 1, 16)[1:])
            rows.extend(target.copy() for _ in range(4))
            rows.extend(target * (1 - u) + ref * u for u in np.linspace(0, 1, 16)[1:])
            phase_ends.append((f"{model.joint(joint).name} {sign * 3.:g} deg", len(rows) - 1))
    rows.extend(ref.copy() for _ in range(25))
    phase_ends.append(("final hold", len(rows) - 1))
    roll = RR.servo_rollout(model, np.asarray(rows), fps, state_start=state)
    issues = []
    if roll["unstable"] or roll["frames_simulated"] != len(rows):
        issues.append("stress simulation unstable or incomplete")
    Q = np.vstack([data.qpos[None], roll["substep_qpos"]])
    measured = {}
    for name, bid in bids.items():
        value = HC.measure_slip(model, Q, 1 / model.opt.timestep, 0, len(Q) - 1, bid,
                                pt["ee_body"], hand, groups[name])
        measured[name] = value
        if value["drift_m"] > .020 or value["turn_deg"] > HC.RIGID_SLIP_DEG:
            issues.append(f"{name}: combined hold drift exceeds 20 mm / 15 degrees")
    peak_pen = max(float(np.max(roll["hand_pen_m"], initial=0)), float(np.max(roll["object_pen_m"], initial=0)))
    if peak_pen > .003:
        issues.append("stress penetration exceeds 3 mm")
    speed = np.max(np.abs(roll["substep_qvel"][:, pt["arm_dadr"]]), initial=0)
    if speed > np.pi:
        issues.append("actual arm speed exceeds 180 degrees/s")
    union = set().union(*groups.values())
    probe = mujoco.MjData(model)
    support_events, missing_contact = [], []
    # Every recorded integration state is checked, including the exact start.
    for i, q in enumerate(Q):
        probe.qpos[:] = q
        mujoco.mj_forward(model, probe)
        touched = set()
        for c in probe.contact:
            if c.geom1 < 0 or c.geom2 < 0 or c.dist > .001:
                continue
            a, b = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            for name, bodies in groups.items():
                if (a in bodies) == (b in bodies):
                    continue
                other = b if a in bodies else a
                if other in hand:
                    touched.add(name)
                elif other not in union and len(support_events) < 8:
                    support_events.append({"sample": i, "object": name, "other": model.body(other).name})
        if set(objects) - touched and len(missing_contact) < 8:
            missing_contact.append({"sample": i, "objects": sorted(set(objects) - touched)})
    if support_events:
        issues.append("held objects contact external support during stress")
    if missing_contact:
        issues.append("a retained object loses hand contact during stress")
    from tool_library.verifiers.robot_self_collision_check import robot_self_collision_check
    collision = robot_self_collision_check(str(scene), Q, hand_object_tol_m=.003)
    if not collision["ok"]:
        issues.append("stress robot/visible-hand collision audit failed")
    return {"ok": not issues, "issues": issues, "objects": list(objects), "side": side, "frame": int(frame),
            "prefix_replay": replay, "measured": measured, "peak_penetration_mm": peak_pen * 1000.,
            "peak_arm_speed_deg_s": float(np.degrees(speed)), "support_events": support_events,
            "missing_contact": missing_contact, "stress_frames": len(rows), "stress_substeps": len(Q) - 1,
            "phases": [name for name, _ in phase_ends], "collision": collision}
