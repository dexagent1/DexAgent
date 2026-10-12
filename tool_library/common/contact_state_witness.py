"""Conservative, internally generated physical witness for indirect object motion.

This establishes reconstruction/contact consistency, never robot task feasibility.
Only a previous semantic manipulation and measured fingertip tracking may drive a
limited articulation. No declared object poses or external trajectories are read.
"""
from __future__ import annotations

import numpy as np


def _under(model, body, root):
    while body:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return root == 0


def prior_support_witnesses(model, initial, sem, root_of, oid, target_sg, names,
                            xyz, fps, surface_points, final_frame, max_trials=6):
    """Yield bounded physical, time-aligned states; return failures as diagnostics.

    The deliberately narrow case is one directly contacting, previously moved
    articulated support. Other cases remain unverified by this helper.
    """
    import mujoco
    from scipy.spatial import cKDTree

    root = root_of[oid]
    rid = model.body(root).id
    target_bodies = {b for b in range(model.nbody) if _under(model, b, rid)}
    # The carried object must be passive; otherwise its own actuation, a weld,
    # or a mocap attachment could explain motion without support contacts.
    if any(model.body_mocapid[b] >= 0 or model.body_gravcomp[b] != 0 for b in target_bodies):
        return
    for actuator in range(model.nu):
        transmission = model.actuator_trntype[actuator]
        if transmission in (mujoco.mjtTrn.mjTRN_JOINT, mujoco.mjtTrn.mjTRN_JOINTINPARENT):
            joint = int(model.actuator_trnid[actuator, 0])
            if int(model.jnt_bodyid[joint]) in target_bodies:
                return
        elif transmission == mujoco.mjtTrn.mjTRN_BODY:
            if int(model.actuator_trnid[actuator, 0]) in target_bodies:
                return
        elif transmission == mujoco.mjtTrn.mjTRN_SITE:
            site = int(model.actuator_trnid[actuator, 0])
            if int(model.site_bodyid[site]) in target_bodies:
                return
        else:
            # Tendon/transmission dependency tracing is outside this narrow proof.
            return
    for equality in range(model.neq):
        if not initial.eq_active[equality]:
            continue
        kind = model.eq_type[equality]
        ids = (int(model.eq_obj1id[equality]), int(model.eq_obj2id[equality]))
        if kind in (mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT):
            if any(i in target_bodies for i in ids):
                return
        elif kind == mujoco.mjtEq.mjEQ_JOINT:
            if any(i >= 0 and int(model.jnt_bodyid[i]) in target_bodies for i in ids):
                return
        else:
            return
    col = {n: i for i, n in enumerate(names)}
    target_start = float(target_sg['time_window_s'][0])
    prior = []
    for sg in sem.get('subgoals', []):
        window = sg.get('time_window_s')
        if (sg is target_sg or sg.get('motion') not in {'open_joint', 'close_joint', 'push', 'pull'}
                or not window or float(window[0]) >= target_start):
            continue
        for ref in sg.get('object_refs', []):
            sid = str(ref).split(':')[0]
            if sid != oid and root_of.get(sid):
                prior.append((sg, sid))
    trials = 0
    for sg, sid in prior:
        support_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_of[sid])
        if support_root < 0:
            continue
        for jid in range(model.njnt):
            if (model.jnt_type[jid] not in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
                    or not model.jnt_limited[jid]
                    or not _under(model, int(model.jnt_bodyid[jid]), support_root)):
                continue
            moving = {b for b in range(model.nbody) if _under(model, b, int(model.jnt_bodyid[jid]))}
            # A direct physical contact must transmit this support's motion.
            paired = False
            for contact in initial.contact:
                if min(contact.geom) < 0 or contact.dist > 0.003:
                    continue
                b1, b2 = (int(model.geom_bodyid[g]) for g in contact.geom)
                if ((b1 in target_bodies and b2 in moving) or (b2 in target_bodies and b1 in moving)):
                    paired = True
                    break
            if not paired or target_bodies & moving:
                continue
            body_name = model.body(int(model.jnt_bodyid[jid])).name
            surface = surface_points(model, initial, body_name)
            if len(surface) < 4:
                continue
            tree = cKDTree(surface)
            start = max(0, int(np.ceil(float(sg['time_window_s'][0]) * fps)))
            end = min(final_frame, len(xyz) - 1, int(np.floor(float(sg['time_window_s'][1]) * fps)))
            if end <= start:
                continue
            sides = ('left', 'right') if sg.get('acting_hand') == 'both' else (sg.get('acting_hand'),)
            tips = [f'{side}{tip}' for side in sides for tip in
                    ('ThumbTip', 'IndexFingerTip', 'MiddleFingerTip', 'RingFingerTip', 'LittleFingerTip')
                    if f'{side}{tip}' in col]
            for tip in tips:
                points = xyz[start:end + 1, col[tip]]
                if not np.isfinite(points).all():
                    continue
                distances, _ = tree.query(points)
                near = np.flatnonzero(distances <= 0.02)
                if not len(near):
                    continue
                anchor_frame = start + int(near[0])
                # Contact at the end alone cannot establish an earlier motion.
                if end - anchor_frame < 2:
                    continue
                measured = xyz[anchor_frame:end + 1, col[tip]]
                anchor = measured[0]
                axis = initial.xaxis[jid].copy()
                origin = initial.xanchor[jid].copy()
                qadr = int(model.jnt_qposadr[jid])
                q0 = float(initial.qpos[qadr])
                hinge = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
                if hinge:
                    radii = measured - origin
                    radii -= np.outer(radii @ axis, axis)
                    if np.linalg.norm(radii[0]) < 0.025:
                        continue
                    a = radii[0] / np.linalg.norm(radii[0]); b = np.cross(axis, a)
                    delta = np.unwrap(np.arctan2(radii @ b, radii @ a))
                    predicted = (origin + np.outer(np.cos(delta), radii[0])
                                 + np.outer(np.sin(delta), np.cross(axis, radii[0]))
                                 + np.outer(np.full(len(delta), np.dot(anchor-origin, axis)), axis))
                else:
                    delta = (measured - anchor) @ axis
                    predicted = anchor + np.outer(delta, axis)
                # Measured motion must fit this articulation, not arbitrary travel.
                residual = np.linalg.norm(measured - predicted, axis=1)
                desired = q0 + delta
                lo, hi = model.jnt_range[jid]
                speed = np.max(np.abs(np.diff(desired))) * fps
                if (residual.max() > 0.02 or desired.min() < lo - 0.002 or desired.max() > hi + 0.002
                        or speed > (4.0 if hinge else 1.0) or np.ptp(desired) < (0.03 if hinge else 0.015)):
                    continue
                trials += 1
                if trials > max_trials:
                    return
                out = _replay(model, initial, jid, desired, anchor_frame, end,
                              final_frame, fps)
                if out["ok"]:
                    probe = mujoco.MjData(model)
                    maximum_gap = 0.0
                    for frame in range(anchor_frame, end+1):
                        probe.qpos[:] = out["_states"][frame]
                        mujoco.mj_forward(model, probe)
                        actual_surface = surface_points(model, probe, body_name)
                        actual_gap = float(cKDTree(actual_surface).query(xyz[frame, col[tip]])[0])
                        maximum_gap = max(maximum_gap, actual_gap)
                    out["max_physical_support_contact_gap_m"] = maximum_gap
                    if maximum_gap > 0.02:
                        out["ok"] = False
                        out["reason"] = "physical support surface does not follow measured fingertip contact"
                        out.pop("_states", None)
                out.update(support_object=sid, support_subgoal=sg.get('index'),
                           support_joint=model.joint(jid).name, tracking_fingertip=tip,
                           anchor_frame=anchor_frame,
                           first_measured_motion_frame=anchor_frame + int(np.flatnonzero(np.abs(delta) > (0.005 if hinge else 0.001))[0]),
                           max_tracking_residual_m=float(residual.max()),
                           scope='contact consistency witness, not robot execution feasibility')
                yield out


def _replay(model, initial, jid, desired, start, drive_end, final_frame, fps):
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_copyData(data, model, initial)
    data.time = 0.0
    data.xfrc_applied[:] = 0.0
    data.qfrc_applied[:] = 0.0
    qadr = int(model.jnt_qposadr[jid]); dof = int(model.jnt_dofadr[jid])
    hinge = model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    # Bounded external drive probes physical compatibility, not hand actuation.
    kp, kd, cap, tolerance = (40., 3., 5., 0.08) if hinge else (1000., 45., 30., 0.015)
    dt = float(model.opt.timestep)
    if dt <= 0 or final_frame/fps/dt > 15000:
        return {'ok': False, 'reason': 'physical witness step budget exceeded'}
    states, peak_error, peak_force, peak_motion = {}, 0.0, 0.0, 0.0
    first_q = float(initial.qpos[qadr])
    drive_frames = np.arange(start, drive_end + 1)
    for frame in range(final_frame + 1):
        target_time = frame/fps
        while data.time < target_time - 1e-10:
            source_frame = data.time*fps
            goal = float(np.interp(source_frame, drive_frames, desired))
            force = (float(np.clip(kp*(goal-data.qpos[qadr])-kd*data.qvel[dof], -cap, cap))
                     if start <= source_frame <= drive_end else 0.0)
            # Do not apply gravity compensation or any target-object force.
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof] = force
            peak_force = max(peak_force, abs(force))
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or any(w.number for w in data.warning):
                return {'ok': False, 'reason': 'unstable physical support replay'}
        expected = float(np.interp(frame, drive_frames, desired))
        if start <= frame <= drive_end:
            peak_error = max(peak_error, abs(float(data.qpos[qadr])-expected))
        if peak_error > tolerance:
            return {'ok': False, 'reason': 'bounded joint drive did not follow measured support motion',
                    'max_joint_tracking_error': peak_error, 'joint_force_cap': cap}
        if start <= frame <= drive_end:
            peak_motion = max(peak_motion, abs(float(data.qpos[qadr])-first_q))
        states[frame] = data.qpos.copy()
    if peak_motion < (0.03 if hinge else 0.01):
        return {'ok': False, 'reason': 'support did not physically move'}
    return {'ok': True, '_states': states, 'max_joint_tracking_error': peak_error,
            'joint_force_cap': cap, 'max_applied_joint_force': peak_force, 'max_joint_displacement': peak_motion,
            'driven_dofs': [dof], 'driven_frame_interval': [start, drive_end],
            'external_force_outside_contact': 0.0, 'object_state_writes': 0}
