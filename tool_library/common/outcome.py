"""What a subgoal must have achieved, measured on the executed episode. The planner writes an `expect`
dict for each subgoal (rise, target position, joint value, turn, tilt, release, rest, untouched objects, a rope
severed, a press-fit part seated; per hand for a two-hand entry); `evaluate` reads the physics trajectory and says which
expectations held, with the numbers."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import episode as E
from tool_library.common import hand as H
from tool_library.common import held_contact as HC

# pos_tol_m caps the target tolerance; target_frac x the subgoal's planned displacement (its start to its target) and
# pos_tol_min_m bound it below that, so a short move must arrive (a 23 mm slide that moved a strip 4.8 mm passed the
# fixed 30 mm with 19 mm left)
DEFAULTS = {"pos_tol_m": 0.03, "target_frac": 0.5, "pos_tol_min_m": 0.003, "joint_tol_hinge_rad": np.radians(10), "joint_tol_slide_m": 0.01,
            "rest_window_s": 0.3, "rest_tol_m": 0.005, "support_tol_m": 0.012, "stay_tol_m": 0.02,
            "untouched_tol_m": 0.01, "untouched_tol_deg": 8.0, "max_hand_pen_m": 0.003, "max_obj_pen_m": 0.003,
            "max_track_m": 0.025, "max_track_deg": 15.0, "slip_tol_m": 0.02, "fraction": 0.8}


SLIP_WORDS = (": lift", ": carry", ": pour", ": tilt", ": set down")
REALISM = ("hand_penetration", "object_penetration", "servo_tracking")


def resolve_label_frames(expect, labels):
    """Frame references an expectation names by phase label, made relative to the subgoal's first frame (`labels` = the
    subgoal's frame labels, in order): pads_at_label -> pads_at_rel (the last frame of that phase), and for no_slip the
    first lift / carry / pour / tilt / set-down frame -> slip_from_rel (the hand is away from the object during the
    reach). A two-hand expectation is resolved per hand, on that hand's part of the two-hand labels. Fills `expect` in
    place and returns it."""
    if expect.get("hands"):
        for side, e in expect["hands"].items():
            resolve_label_frames(e, E.side_view(labels, side))
        return expect
    if expect.get("pads_at_label") and "pads_at" not in expect and "pads_at_rel" not in expect:
        lab = str(expect["pads_at_label"])
        idx = [i for i, l in enumerate(labels) if str(l).split(" #")[0] == lab]
        expect["pads_at_rel"] = idx[-1] if idx else len(labels) - 1
    if expect.get("_motion_no_slip"):
        expect["no_slip"] = True
        expect.pop("slip_from", None)
        expect.pop("slip_from_rel", None)
    if expect.get("no_slip") and "slip_from" not in expect and "slip_from_rel" not in expect:
        starts = [i for i, l in enumerate(labels) if any(w in str(l) for w in SLIP_WORDS)]
        expect["slip_from_rel"] = starts[0] if starts else 0
    st = expect.get("seated")
    if isinstance(st, dict) and st.get("phase") and "frames_rel" not in st:
        idx = [i for i, l in enumerate(labels) if str(l).split(" #")[0] == str(st["phase"])]
        if idx:
            expect["seated"] = dict(st, frames_rel=[idx[0], idx[-1]])
    return expect


def _pose_at(model, data, Q, t, bid):
    data.qpos[:] = Q[t]; mujoco.mj_kinematics(model, data)
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def _bid(model, name):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise ValueError(f"no body {name!r} in the scene")
    return int(b)


def evaluate(model, Q, fps, frames, expect, diagnostics=None, parts=None):
    """frames: the frame indices of the subgoal within Q (its end is frames[-1]); expect: see DEFAULTS and the
    planner. diagnostics: the servo_rollout dict (penetration and tracking per frame of Q) when available."""
    if expect.get("hands"):
        return _evaluate_hands(model, Q, fps, frames, expect, diagnostics, parts)
    cfg = {**DEFAULTS, **{k: v for k, v in expect.items() if k in DEFAULTS}}
    for key in ("max_hand_pen_m", "max_obj_pen_m", "slip_tol_m"):
        value = float(cfg[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        # Segment feedback uses the same hard penetration cap as final audits.
        # A plan may tighten that cap, never waive it with its own expectation.
        cfg[key] = min(value, DEFAULTS[key])
    # a scene without hands (a rope or a mechanism on its own) is judged on its objects only
    parts = parts or (E.robot_parts(model) if expect.get("hand") else {})
    data = mujoco.MjData(model)
    t0, t1 = int(frames[0]), int(frames[-1])
    checks, meas = {}, {}
    side = expect.get("hand")
    rig = parts[side]["rig"] if side in parts else None
    obj = expect.get("object")
    try:
        bid = _bid(model, obj) if obj else None
    except ValueError:
        if not expect.get("knot"):                # a knot names its cable by prefix ('red_rope'), not by a body
            raise
        bid = None
    sub = E.subtree_bodies(model, bid) if bid is not None else set()
    if bid is not None:
        p0, R0 = _pose_at(model, data, Q, t0, bid); p1, R1 = _pose_at(model, data, Q, t1, bid)
        meas["object_start"] = p0.round(4).tolist(); meas["object_end"] = p1.round(4).tolist()
        meas["object_moved_m"] = round(float(np.linalg.norm(p1 - p0)), 4)
        meas["object_rotated_deg"] = round(float(np.degrees(np.linalg.norm(Rot.from_matrix(R0.T @ R1).as_rotvec()))), 2)
        tilt = float(np.degrees(np.arccos(np.clip(R1[2, 2], -1, 1))))
        meas["object_tilt_end_deg"] = round(tilt, 2)
    # --- rise
    if "rise_m" in expect and bid is not None:
        zs = [_pose_at(model, data, Q, t, bid)[0][2] for t in range(t0, t1 + 1, max(1, (t1 - t0) // 60 or 1))]
        rise = float(p1[2] - p0[2]); peak = float(max(zs) - p0[2])
        meas["rise_m"] = round(rise, 4); meas["peak_rise_m"] = round(peak, 4)
        checks["rise"] = rise >= float(expect["rise_m"])
    # --- in hand at the end
    if expect.get("in_hand") and rig is not None and bid is not None:
        data.qpos[:] = Q[t1]; mujoco.mj_forward(model, data)
        cs = E.contacts_at(model, data, rig.body_ids, E.held_bodies(model, bid))   # a cable link: its neighbours count
        touching = {model.body(int(model.geom_bodyid[data.contact[i].geom1])).name for i, _ in cs} | \
                   {model.body(int(model.geom_bodyid[data.contact[i].geom2])).name for i, _ in cs}
        hand_links = sorted(b for b in touching if b in {model.body(x).name for x in rig.body_ids})
        meas["hand_links_touching"] = hand_links
        checks["in_hand"] = len(cs) > 0 and len(hand_links) >= int(expect.get("min_links", 2))
    # --- the pads touch the part from their planned side (a pad driven through a thin plate ends on the other side
    # and pushes the wrong way; a pad that never touches cannot lift)
    if expect.get("pads") and rig is not None and bid is not None:
        t = int(t0 + expect["pads_at_rel"]) if "pads_at_rel" in expect else int(expect.get("pads_at", t0)); t = max(t0, min(t, t1))
        # the planned contacts were written for the part where it stood when the subgoal was planned (its first
        # frame); a part nudged on approach carries them along: express them in the part's pose at the check frame
        data.qpos[:] = Q[t0]; mujoco.mj_kinematics(model, data)
        p0, R0 = data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
        data.qpos[:] = Q[t]; mujoco.mj_forward(model, data)
        pt_, Rt_ = data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()
        dR = Rt_ @ R0.T
        per = {}
        for f, spec0 in expect["pads"].items():
            spec = dict(spec0)
            spec["point"] = (dR @ (np.asarray(spec0["point"], float) - p0) + pt_).tolist()
            spec["normal"] = (dR @ np.asarray(spec0["normal"], float)).tolist()
            sid = rig.fingertip_sites.get(f)
            if sid is None:
                continue
            tip_body = int(model.site_bodyid[sid])
            n_out = np.asarray(spec["normal"], float); n_out /= max(np.linalg.norm(n_out), 1e-9)
            touching, side_ok = False, True
            for i in range(data.ncon):
                c = data.contact[i]
                b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if (b1 == tip_body and b2 in sub) or (b2 == tip_body and b1 in sub):
                    touching = True
                    n = np.asarray(c.frame[:3], float) * (1.0 if b1 == tip_body else -1.0)   # from the pad into the part
                    if float(n @ (-n_out)) < 0.3:
                        side_ok = False
            # geometric side as well: the pad centre must lie on the outward side of the planned contact plane
            pad = data.site_xpos[sid]
            geo = float((pad - np.asarray(spec["point"], float)) @ n_out)
            pad_r = H.pad_radius(model, rig, f) if hasattr(H, "pad_radius") else 0.008
            # the pad centre stays outside the surface by its radius less a soft-contact allowance
            near = geo - pad_r < 0.002                   # a pad posed one millimetre off its surface counts as placed
            per[f] = {"touching": bool(touching or near), "in_contact": touching, "side_ok": bool(side_ok and geo > pad_r - 0.004), "offset_mm": round(1000 * geo, 1)}
        meas["pads"] = per; meas["pads_frame"] = t
        n_touch = sum(1 for v in per.values() if v["touching"])
        need = int(expect.get("min_pads", max(1, (len(per) + 1) // 2)))
        checks["pads_touching"] = n_touch >= need and all(v["side_ok"] for v in per.values() if v["touching"])
    # --- slip: the object keeps its pose relative to the wrist over the window
    if expect.get("no_slip") and rig is not None and bid is not None:
        ee = parts[side]["ee_body"]
        ts = int(t0 + expect["slip_from_rel"]) if "slip_from_rel" in expect else int(expect.get("slip_from", t0)); ts = max(t0, min(ts, t1))
        held = HC.measure_slip(model, Q, fps, ts, t1, bid, ee,
                               set(getattr(rig, "body_ids", ())), E.held_bodies(model, bid), data=data)
        slip = held["drift_m"]
        meas["slip_in_hand_m"] = round(slip, 4)
        meas["slip_peak_frame"] = held["translation_peak_frame"]
        meas["slip_in_hand_deg"] = round(held["turn_deg"], 3)
        meas["slip_rotation_peak_frame"] = held["rotation_peak_frame"]
        meas["slip_held_body"] = held["body"]
        meas["slip_contact_point_local"] = held["contact_point_local"]
        meas["slip_turn_judged"] = held["turn_judged"]
        checks["no_slip"] = (slip <= cfg["slip_tol_m"]
                             and (not held["turn_judged"] or held["turn_deg"] <= HC.RIGID_SLIP_DEG))
    # --- target position
    if expect.get("target_pos") is not None and bid is not None:
        tgt = np.asarray(expect["target_pos"], float)
        axes = expect.get("target_axes", "xyz")
        d = p1 - tgt
        if axes == "xy":
            d = d[:2]
        err = float(np.linalg.norm(d))
        # the tolerance is at most target_frac of the whole planned displacement (all three axes: a set-down whose xy
        # target lies under the part is judged on its drop, a slide along the table on its length), never below
        # pos_tol_min_m, never above pos_tol_m
        planned = float(np.linalg.norm(tgt - p0))
        tol = min(float(cfg["pos_tol_m"]), max(float(cfg["target_frac"]) * planned, float(cfg["pos_tol_min_m"])))
        meas["target_err_m"] = round(err, 4)
        meas["planned_move_m"] = round(planned, 4)
        meas["target_tol_m"] = round(tol, 4)
        checks["at_target"] = err <= tol
    # --- released
    if expect.get("released") and rig is not None and bid is not None:
        data.qpos[:] = Q[t1]; mujoco.mj_forward(model, data)
        cs = E.contacts_at(model, data, rig.body_ids, sub)
        meas["hand_contacts_end"] = len(cs)
        checks["released"] = len(cs) == 0
    # --- at rest
    if expect.get("at_rest") and bid is not None:
        w = max(1, int(cfg["rest_window_s"] * fps))
        pa, _ = _pose_at(model, data, Q, max(t0, t1 - w), bid)
        mv = float(np.linalg.norm(p1 - pa))
        meas["moved_in_last_window_m"] = round(mv, 4)
        checks["at_rest"] = mv <= cfg["rest_tol_m"]
    if expect.get("released") and expect.get("at_rest") and bid is not None:
        # Match the task gate at the segment boundary. A dropped object can be
        # perfectly still at the end; that is insufficient for a placement.
        # Reuse the same contact-derived release time and seating limits rather
        # than a caller-provided phase label or a second set of tolerances.
        from tool_library.common import task_gates as TG
        episode = TG.Episode(model, qpos=Q, fps=fps)
        report = TG.gate_release_settle(episode, sub, t0, t1, names={"A": obj})
        checks["release_settle"] = bool(report["ok"])
        meas["release_settle"] = report
    # --- supported
    if expect.get("supported") and bid is not None:
        data.qpos[:] = Q[t1]; mujoco.mj_forward(model, data)
        others = {b for b in range(model.nbody) if b not in sub and b != 0}
        touching = E.contacts_at(model, data, sub, others)
        z_low = E.lowest_point(model, data, sub)
        meas["lowest_point_z"] = round(z_low, 4)
        meas["support_contacts"] = len(touching)
        on = expect.get("on")
        if on:
            onb = E.subtree_bodies(model, _bid(model, on))
            meas["on_contacts"] = len(E.contacts_at(model, data, sub, onb))
            checks["supported"] = meas["on_contacts"] > 0
        else:
            checks["supported"] = len(touching) > 0 or abs(z_low) <= cfg["support_tol_m"]
    # --- upright / tilt
    if expect.get("target_quat_wxyz") is not None:
        q = np.asarray(expect["target_quat_wxyz"], float)
        if bid is None or q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1., atol=1e-5):
            raise ValueError("target_quat_wxyz requires an object and a finite unit quaternion")
        target_R = Rot.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        angle = float(np.degrees(Rot.from_matrix(target_R.T @ R1).magnitude()))
        meas["target_rotation_error_deg"] = round(angle, 3)
        checks["at_target_orientation"] = angle <= 15.
    if expect.get("upright") and bid is not None:
        checks["upright"] = meas["object_tilt_end_deg"] <= float(expect.get("upright_tol_deg", 15.0))
    if "tilt_deg" in expect and bid is not None:
        tilts = []
        for t in range(t0, t1 + 1, max(1, (t1 - t0) // 60 or 1)):
            _, R = _pose_at(model, data, Q, t, bid)
            tilts.append(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
        meas["peak_tilt_deg"] = round(max(tilts), 2)
        checks["tilted"] = max(tilts) >= cfg["fraction"] * float(expect["tilt_deg"])
        if expect.get("tilted"):                       # a kept tilt: the object must END turned, not just pass through it
            end_tilt = tilts[-1]
            meas["end_tilt_deg"] = round(end_tilt, 2)
            checks["stays_tilted"] = end_tilt >= cfg["fraction"] * float(expect["tilted"]["deg"])
    # --- stays put (a held or steadied object)
    if expect.get("stays_put") and bid is not None:
        checks["stays_put"] = meas["object_moved_m"] <= cfg["stay_tol_m"]
    # --- joint value
    if expect.get("joint"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, expect["joint"])
        if j < 0:
            raise ValueError(f"no joint {expect['joint']!r}")
        a = int(model.jnt_qposadr[j]); hinge = int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        q0, q1 = float(Q[t0][a]), float(Q[t1][a])
        meas["joint_start"], meas["joint_end"] = round(q0, 4), round(q1, 4)
        if "joint_target" in expect:
            tgt = float(expect["joint_target"]); tol = cfg["joint_tol_hinge_rad"] if hinge else cfg["joint_tol_slide_m"]
            travel = tgt - q0
            frac = (q1 - q0) / travel if abs(travel) > 1e-9 else 1.0
            meas["joint_travel_fraction"] = round(float(frac), 3)
            checks["joint_reached"] = abs(q1 - tgt) <= tol or frac >= cfg["fraction"]
            if not checks["joint_reached"]:
                # A support contact at a stall does not by itself block the
                # desired motion. Check its normal against the joint Jacobian.
                qs = np.asarray(Q[t0:t1 + 1, a], float)
                tmax = t0 + int(np.argmax(np.abs(qs - q0)))
                data.qpos[:] = Q[tmax]; mujoco.mj_forward(model, data)
                part = int(model.jnt_bodyid[j]); psub = E.subtree_bodies(model, part)
                base_b = int(model.body_parentid[part])
                own_hand = set(rig.body_ids) if rig is not None else set()
                from tool_library.common.mechanism_contact import joint_contact_directions
                contacts = joint_contact_directions(model, data, j, tgt, psub,
                                                     own_hand | ({base_b} if base_b > 0 else set()))
                meas["joint_contacts_at_stall"] = {"frame": int(tmax), "contacts": contacts}
                blk = {c["body"] for c in contacts if c["opposes_target"]}
                if blk:
                    fixed = sorted({c["body"] for c in contacts if c["opposes_target"] and not c["movable"]})
                    meas["joint_blocked_by"] = {"frame": int(tmax), "bodies": sorted(blk)[:6], "fixed_bodies": fixed,
                                                "scope": "contact normal opposes target motion; cause not proved"}
                    meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + (
                        f"contact normals from {sorted(blk)[:4]} oppose the target joint motion at frame {tmax}; "
                        f"fixed bodies {fixed} must remain fixed. Check the contact and motion path before retrying")
                elif contacts:
                    meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + (
                        "contacts are present at the stall, but their normals do not oppose the requested joint motion; "
                        "check grasp force direction, friction and joint loads rather than inferring a blocked path from contact alone")
            # the part's base moved instead of the joint: the force went into tipping or dragging the whole object
            # (an empty box tips when its lid is lifted near the hinge): the fix is a hold with the other hand
            base = int(model.body_parentid[int(model.jnt_bodyid[j])])
            if base > 0:
                pa, Ra = _pose_at(model, data, Q, t0, base); pb, Rb = _pose_at(model, data, Q, t1, base)
                bmv = float(np.linalg.norm(pb - pa)); brot = float(np.degrees(np.linalg.norm(Rot.from_matrix(Ra.T @ Rb).as_rotvec())))
                meas["base_moved"] = {"body": model.body(base).name, "moved_m": round(bmv, 4), "rotated_deg": round(brot, 2)}
                if not checks["joint_reached"] and (bmv > cfg["untouched_tol_m"] or brot > cfg["untouched_tol_deg"]):
                    meas["hint"] = (f"the joint did not travel but its base {model.body(base).name} moved {1000 * bmv:.0f} mm / {brot:.0f} deg: the "
                                    "hand's force tipped or dragged the whole object instead of moving the part; hold the base with the other "
                                    "hand first (a hold subgoal), act farther from the hinge, or act in a direction the base cannot follow")
        if "turned_rad" in expect:
            turned = abs(q1 - q0)
            meas["turned_deg"] = round(float(np.degrees(turned)), 1)
            checks["turned"] = turned >= cfg["fraction"] * abs(float(expect["turned_rad"]))
        rb = expect.get("released_by")
        if rb:
            # a thread that ran out during this bite: the cap stops turning once it is free, so the turn is complete
            # when the lift joint reached the release travel (measured on qpos, so the check stands on the trajectory)
            lj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rb["joint"])
            if lj >= 0:
                la = int(model.jnt_qposadr[lj])
                lmax = float(np.max(np.abs(Q[t0:t1 + 1, la])))
                meas["release_travel_mm"] = round(1000 * lmax, 2)
                checks["thread_released"] = lmax >= float(rb["travel_m"]) - 1e-5
                if checks["thread_released"] and "turned" in checks and not checks["turned"]:
                    checks["turned"] = True; meas["turned_until_release"] = True
        if "joint_min_travel" in expect:
            checks["joint_moved"] = abs(q1 - q0) >= float(expect["joint_min_travel"])
    # --- a part turned on the table about the vertical (rotate): its yaw change, signed, against the requested angle
    if expect.get("yaw_turned_deg") is not None and bid is not None:
        pa, Ra = _pose_at(model, data, Q, t0, bid); pb, Rb = _pose_at(model, data, Q, t1, bid)
        dR = Rb @ Ra.T
        yaw = float(np.degrees(np.arctan2(dR[1, 0], dR[0, 0])))
        want = float(expect["yaw_turned_deg"])
        meas["yaw_turned_deg"] = round(yaw, 1)
        checks["yaw_turned"] = (np.sign(yaw) == np.sign(want)) and abs(yaw) >= cfg["fraction"] * abs(want)
    # --- a rope knotted at the end of the subgoal (tie): the topology of its centreline, not a crossing count
    if expect.get("knot"):
        from tool_library.verifiers.rope_knot_check import _cable_bodies, _centreline, _verdict
        want = "trefoil" if expect["knot"] is True else str(expect["knot"])
        import re as _re
        pre_ = bodies = None
        # the subgoal's object may name the cable ('red_rope') or one of its segments ('red_ropeB_12')
        for cand in ([obj, _re.sub(r"B_(first|last|\d+)$", "", obj)] if obj else []) + [None]:
            try:
                pre_, bodies = _cable_bodies(model, cand)
                break
            except ValueError:
                continue
        if bodies is None:
            raise ValueError("expect.knot: no cable (construct_rope bodies) in the scene")
        data.qpos[:] = Q[t1]; mujoco.mj_kinematics(model, data)
        v, hit = _verdict(_centreline(model, data, bodies, pre_), want)
        meas["knot"] = {"type": v["type"], "agree": v["agree"], "self_crossings": v["n_self_crossings"]}
        checks["knot"] = bool(hit)
    # --- the crossing order a rope move must leave (lay / drag, common/ropemoves.py): the planned one, extra curls dropped
    if expect.get("crossings"):
        from tool_library.common import ropemoves as RM
        RM.crossing_checks(model, data, Q, t1, expect, checks, meas)
    # --- a rope cut during the subgoal (severed): its join equality went from active to inactive inside the subgoal's
    # frames (the recorded equality state; a cut is run-time state no qpos holds) and the two cut ends are apart at the
    # end. expect["severed"] = a join equality name, a list of them, or True (any rope join that was cut).
    if expect.get("severed"):
        _severed(model, data, Q, t0, t1, expect, diagnostics, checks, meas)
    # --- displacement along a direction (push / pull)
    if expect.get("displacement") and bid is not None:
        dsp = expect["displacement"]; dvec = np.asarray(dsp["dir"], float); dvec /= max(np.linalg.norm(dvec), 1e-9)
        along = float(np.dot(p1 - p0, dvec))
        meas["displacement_along_m"] = round(along, 4)
        checks["displaced"] = along >= cfg["fraction"] * float(dsp["min_m"])
    # --- a press-fit part pushed home (press_seat): its root travelled along the press direction to the seat
    if expect.get("seated") and bid is not None:
        st = expect["seated"]
        ax = np.asarray(st.get("axis", [0.0, 0.0, -1.0]), float); ax = ax / max(np.linalg.norm(ax), 1e-9)
        tol = float(st.get("tol_m", 0.0015))
        if st.get("target") is not None:
            remaining = float((np.asarray(st["target"], float) - p1) @ ax)       # > 0: still short of the seat
        else:
            remaining = float(st.get("depth_m", 0.0)) - float((p1 - p0) @ ax)
        meas["seat_travel_mm"] = round(1000 * float((p1 - p0) @ ax), 2)
        meas["seat_remaining_mm"] = round(1000 * remaining, 2)
        if diagnostics is not None and diagnostics.get("wrist_force_n") is not None:
            # the wrist force during the press phase (the reach and the back-off accelerate the hand with the library's cap)
            a_, b_ = (t0 + int(st["frames_rel"][0]), t0 + int(st["frames_rel"][1])) if st.get("frames_rel") else (t0, t1)
            meas["peak_press_force_n"] = round(float(np.max(np.asarray(diagnostics["wrist_force_n"])[a_:b_ + 1])), 1)
        checks["seated"] = abs(remaining) <= tol
        if not checks["seated"]:
            meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + (
                f"the part stopped {1000 * remaining:.1f} mm short of its seat: the press did not overcome the fit; put the palm or the "
                "flat fingers squarely over the part (not a fingertip), raise force_n up to the 40 N wrist cap (never above), give "
                "travel_m the seat depth plus 5 mm, and check the part is aligned over its seat" if remaining > 0 else
                f"the part went {-1000 * remaining:.1f} mm past its seat: the seat target is wrong or the part was pushed through its support")
    # --- the thin-part motions' own expectations (common/nonprehensile.py: overhang, end_raised, on_support, table_clear)
    from tool_library.common import nonprehensile as NP
    if any(k in expect for k in NP.EXPECT_KEYS):
        NP.evaluate_extra(model, Q, t0, t1, expect, rig, checks, meas)
    # --- objects that must not move
    _untouched_checks(model, data, Q, t0, t1, expect.get("untouched", []) or [], cfg, checks, meas)
    # --- over the whole span: parts the other hand pins, two links one object is carried by, more objects in one hand
    if any(expect.get(k) for k in ("pinned", "spacing", "also_in_hand", "link_in_hand")):
        from tool_library.common import bimanual as B
        B.span_checks(model, data, Q, t0, t1, expect, parts, checks, meas, fps=fps)
    # --- physics realism from the rollout diagnostics
    if diagnostics is not None and "hand_pen_m" in diagnostics:
        hp = np.asarray(diagnostics["hand_pen_m"])[t0:t1 + 1]; op = np.asarray(diagnostics["object_pen_m"])[t0:t1 + 1]
        tp = np.asarray(diagnostics["track_pos_m"])[t0:t1 + 1]; tr = np.asarray(diagnostics["track_rot_deg"])[t0:t1 + 1]
        meas["max_hand_pen_mm"] = round(1000 * float(hp.max()), 2); meas["max_object_pen_mm"] = round(1000 * float(op.max()), 2)
        meas["max_track_pos_mm"] = round(1000 * float(tp.max()), 1); meas["max_track_rot_deg"] = round(float(tr.max()), 1)
        checks["hand_penetration"] = float(hp.max()) <= cfg["max_hand_pen_m"]
        checks["object_penetration"] = float(op.max()) <= cfg["max_obj_pen_m"]
        checks["servo_tracking"] = float(tp.max()) <= cfg["max_track_m"] and float(tr.max()) <= cfg["max_track_deg"]
        if not checks["servo_tracking"] and rig is not None:
            # name the common cause: the hand jammed against the table (or another body) at the worst frame
            tw = t0 + int(np.argmax(tp)); data.qpos[:] = Q[tw]; mujoco.mj_forward(model, data)
            hb = set(rig.body_ids); against = set()
            for i in range(data.ncon):
                c = data.contact[i]; b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                if (b1 in hb) != (b2 in hb):
                    other = b2 if b1 in hb else b1
                    against.add(model.body(other).name)
            meas["worst_track_frame"] = int(tw); meas["hand_against_at_worst"] = sorted(against)
            # the hand's wrist is a rot_x-rot_y-rot_z hinge chain: at rot_y near +-90 deg rot_x and rot_z align and the
            # servos lose an axis (a hold that passed every hand-scene test then drifts 50 deg); name it
            ry = (np.degrees(np.abs(np.asarray(Q[t0:t1 + 1, rig.wrist_qpos["ry"]], float)))
                  if "ry" in rig.wrist_qpos else np.array([]))
            if ry.size and float(ry.max()) > 75.0:
                meas["wrist_ry_max_deg"] = round(float(ry.max()), 1)
                meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + (
                    f"the {side} wrist reaches rot_y {float(ry.max()):.0f} deg, next to its Euler singularity (+-90): the servos lose an axis "
                    "there; turn the pose about its approach or contact axis (optimize_pose_for_reach) or pick one whose rot_y stays within +-75 deg")
            if "table" in against or "world" in against:
                meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + f"the {side} hand is jammed against the table at frame {tw} (tracking {1000 * float(tp.max()):.0f} mm): the pose or its pre-grasp opening puts fingers below the tabletop; give the pose a flat open shape (finger_targets_open) or raise it"
            elif against:
                meas["hint"] = (meas.get("hint", "") + " " if meas.get("hint") else "") + f"the {side} hand is pushed off its path by {sorted(against)[:3]} at frame {tw} (tracking {1000 * float(tp.max()):.0f} mm)"
    failed = [k for k, v in checks.items() if not v]
    return {"ok": not failed, "failed": failed, "checks": {k: bool(v) for k, v in checks.items()}, "measured": meas,
            "frames": [t0, t1]}


def _untouched_checks(model, data, Q, t0, t1, names, cfg, checks, meas):
    for name in names:
        try:
            ub = _bid(model, name)
        except ValueError:
            continue
        pa, Ra = _pose_at(model, data, Q, t0, ub); pb, Rb = _pose_at(model, data, Q, t1, ub)
        mv = float(np.linalg.norm(pb - pa)); rot = float(np.degrees(np.linalg.norm(Rot.from_matrix(Ra.T @ Rb).as_rotvec())))
        meas.setdefault("untouched", {})[name] = {"moved_m": round(mv, 4), "rotated_deg": round(rot, 2)}
        checks[f"untouched:{name}"] = mv <= cfg["untouched_tol_m"] and rot <= cfg["untouched_tol_deg"]


def _evaluate_hands(model, Q, fps, frames, expect, diagnostics, parts):
    """A two-hand entry (motion 'both'): each hand's expectations measured on the same frames (checks named
    '<side>:<check>'), the realism checks once (the rollout's diagnostics are per frame, worst hand), the bystanders
    neither hand may move, and hands_apart: no contact between the two hands on any frame."""
    parts = parts or E.robot_parts(model)
    shared = {k: v for k, v in expect.items() if k in DEFAULTS}
    cfg = {**DEFAULTS, **shared}
    checks, meas = {}, {"hands": {}}
    realism = {}
    for side, e in expect["hands"].items():
        e2 = {**shared, **dict(e)}
        e2.setdefault("hand", side)
        e2.pop("untouched", None)
        r = evaluate(model, Q, fps, frames, e2, diagnostics=diagnostics, parts=parts)
        for k, v in r["checks"].items():
            if k in REALISM:
                realism[k] = realism.get(k, True) and v
            else:
                checks[f"{side}:{k}"] = v
        meas["hands"][side] = r["measured"]
    for k in ("max_hand_pen_mm", "max_object_pen_mm", "max_track_pos_mm", "max_track_rot_deg"):
        vals = [mh[k] for mh in meas["hands"].values() if k in mh]
        if vals:
            meas[k] = max(vals)
    checks.update(realism)
    data = mujoco.MjData(model)
    t0, t1 = int(frames[0]), int(frames[-1])
    _untouched_checks(model, data, Q, t0, t1, expect.get("untouched", []) or [], cfg, checks, meas)
    if expect.get("pinned") or expect.get("spacing"):
        from tool_library.common import bimanual as B
        B.span_checks(model, data, Q, t0, t1, {k: expect[k] for k in ("pinned", "spacing", "slip_tol_m") if k in expect}, parts, checks, meas, fps=fps)
    if expect.get("hands_apart") and all(s in parts for s in ("left", "right")):
        lb, rb = set(parts["left"]["rig"].body_ids), set(parts["right"]["rig"].body_ids)
        touch = []
        for t in range(t0, t1 + 1):
            data.qpos[:] = Q[t]; mujoco.mj_forward(model, data)
            cs = [d_ for _, d_ in E.contacts_at(model, data, lb, rb) if d_ < 0.0]
            if cs:
                touch.append((t, round(-1000 * min(cs), 2)))
        meas["hands_touching_frames"] = len(touch)
        if touch:
            meas["hands_touching_first"] = {"frame": touch[0][0], "depth_mm": touch[0][1]}
        checks["hands_apart"] = not touch
    failed = [k for k, v in checks.items() if not v]
    return {"ok": not failed, "failed": failed, "checks": {k: bool(v) for k, v in checks.items()}, "measured": meas, "frames": [t0, t1]}


def join_end_gap(model, data, eq_name):
    """Distance (m) between the two sides of a CONNECT/WELD join at the current data (after kinematics): body1's
    anchor and body2's anchor. 0 while the rope is whole; the cut ends' separation once it is cut."""
    e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    if e < 0:
        return None
    b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
    p1 = data.xpos[b1] + data.xmat[b1].reshape(3, 3) @ model.eq_data[e][0:3]
    a2 = model.eq_data[e][3:6] if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_CONNECT) else np.zeros(3)
    p2 = data.xpos[b2] + data.xmat[b2].reshape(3, 3) @ a2
    return float(np.linalg.norm(p1 - p2))


def _severed(model, data, Q, t0, t1, expect, diagnostics, checks, meas):
    """The `severed` expectation (see evaluate): read the recorded equality state, find the frame each join was cut,
    and measure the cut ends' separation at the subgoal's last frame."""
    min_gap = float(expect.get("sever_gap_m", 0.003))
    ea = None if diagnostics is None else diagnostics.get("eq_active")
    names = None if diagnostics is None else diagnostics.get("eq_names")
    if ea is None or names is None:
        checks["severed"] = False
        meas["severed_hint"] = ("no equality state was recorded (eq_active): run the subgoal through execute_subgoal_plan with the "
                                "plan's scene_recon_json, whose rope entry carries `cut_joins` (construct_cuttable_rope)")
        return
    ea = np.asarray(ea).astype(bool); names = [str(x) for x in names]
    if t0 > 0:
        start = ea[t0 - 1]
    elif diagnostics.get("eq_active_start") is not None:
        start = np.asarray(diagnostics["eq_active_start"]).astype(bool)
    else:
        start = np.asarray(model.eq_active0).astype(bool)
    want = expect["severed"]
    if isinstance(want, str):
        want = [want]
    elif isinstance(want, (list, tuple)):
        want = [str(w) for w in want]
    else:                                        # True: every rope join (a CONNECT between two non-robot bodies) cut here
        want = []
        for i, nm in enumerate(names):
            e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, nm)
            if e < 0 or int(model.eq_type[e]) != int(mujoco.mjtEq.mjEQ_CONNECT):
                continue
            b1, b2 = model.body(int(model.eq_obj1id[e])).name, model.body(int(model.eq_obj2id[e])).name
            if not (b1.startswith("robot_") or b2.startswith("robot_")) and start[i] and not ea[t1, i]:
                want.append(nm)
    per = {}
    data.qpos[:] = Q[t1]; mujoco.mj_kinematics(model, data)
    for nm in want:
        if nm not in names:
            per[nm] = {"found": False}
            continue
        i = names.index(nm)
        col = ea[t0:t1 + 1, i]
        off = np.nonzero(~col)[0]
        gap = join_end_gap(model, data, nm)
        per[nm] = {"whole_at_start": bool(start[i]), "whole_at_end": bool(ea[t1, i]),
                   "cut_frame": int(t0 + off[0]) if len(off) and start[i] else None,
                   "end_gap_mm": None if gap is None else round(1000 * gap, 1)}
    meas["severed"] = per
    if not want:
        meas["severed_hint"] = "no rope join went from whole to cut during the subgoal"
    checks["severed"] = bool(want) and all(v.get("whole_at_start") and not v.get("whole_at_end") for v in per.values())
    checks["cut_ends_apart"] = bool(want) and all((v.get("end_gap_mm") or 0.0) >= 1000 * min_gap for v in per.values())
    if not checks["severed"] and want:
        meas["severed_hint"] = ("the join was not cut during this subgoal: the blades must close on it -- the join inside the "
                                "blades' span and wedge, in their plane, the hinge closing and a blade biting (sever_report "
                                "tells which gate held it)")
