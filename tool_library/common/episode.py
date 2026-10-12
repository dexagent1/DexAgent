"""Episode engine for Stage 3: run a kinematic reference through the scene's physics with the hands'
own servos (objects move only because a finger pushed them), fill the arm joints behind the hands by
IK, and measure what a subgoal achieved. Shared by the Stage 3 skills and verifiers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):  # headless offscreen rendering; MuJoCo rejects "egl" elsewhere
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import hand as H
from tool_library.common import plan as P
from tool_library.common import trajectory as TJ

SIDES = ("left", "right")


# ----------------------------------------------------------------------------- files
def save_episode(path, Q, labels, fps, **extra):
    arrs = {"qpos": np.asarray(Q, float), "labels": np.array([str(x) for x in labels]), "fps": float(fps)}
    for k, v in extra.items():
        if v is not None:
            arrs[k] = np.asarray(v)
    np.savez_compressed(str(path), **arrs)
    return str(path)


def load_episode(path):
    with np.load(str(path), allow_pickle=False) as z:
        key = "qpos" if "qpos" in z.files else "Q"
        out = {"qpos": np.asarray(z[key], float), "fps": float(z["fps"]) if "fps" in z.files else 30.0,
               "labels": [str(x) for x in z["labels"]] if "labels" in z.files else [""] * len(z[key])}
        for k in ("subgoal", "joint_names", "ctrl", "reference", "eq_active", "eq_names", "eq_active_start"):
            if k in z.files:
                out[k] = z[k]
    return out


def joint_names(model):
    from tool_library.common.rigid_robot import joint_identities
    identities = joint_identities(model)
    names = []
    for j in range(model.njnt):
        n = int(model.jnt_qposadr[j + 1] - model.jnt_qposadr[j]) if j + 1 < model.njnt else model.nq - int(model.jnt_qposadr[j])
        nm = identities[j]
        names += [nm] if n == 1 else [f"{nm}[{i}]" for i in range(n)]
    return names


def state_qpos(model, state, frame=-1):
    """Read a planning state by named coordinates, including rigid recordings.

    Runtime recordings omit the legacy six virtual wrist coordinates. Derive
    those coordinates from the recorded arm FK and fixed mount when a planning
    tool needs them. Arm, finger and object coordinates are never fitted or
    silently replaced by the scene's initial state. This adapter creates a
    planning representation; it does not certify or alter a physics recording.
    """
    names, recorded_scene_sha, recorded_path = None, None, None
    if isinstance(state, (str, bytes, Path)) or hasattr(state, '__fspath__'):
        with np.load(state, allow_pickle=False) as z:
            recorded_path = Path(state).resolve()
            q = np.asarray(z['qpos'] if 'qpos' in z.files else z['Q'], float)[int(frame)].copy()
            if 'joint_names' in z.files:
                names = [str(v) for v in z['joint_names']]
            if 'scene_sha256' in z.files:
                recorded_scene_sha = str(z['scene_sha256'])
    else:
        q = np.asarray(state, float).copy()
    if q.ndim != 1 or not np.isfinite(q).all():
        raise ValueError('state must be a finite qpos vector')
    if names is None:
        if q.shape != (model.nq,):
            raise ValueError(f'unnamed state has {q.size} coordinates, scene has {model.nq}; supply a named recording')
        return q
    if len(names) != len(q) or len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError('recording joint_names must uniquely identify every qpos column')
    from tool_library.common import rigid_robot as RR
    target = joint_names(model)
    virtual = {f'robot_{s}_{s}_{k}' for s in SIDES for k in RR.WRIST_SUFFIXES}
    missing, extra = set(target) - set(names), set(names) - set(target)
    if missing - virtual or extra - virtual:
        raise ValueError(f'recorded and scene joints differ: missing {sorted(missing - virtual)}, extra {sorted(extra - virtual)}')
    qout = np.asarray(model.key_qpos[0] if model.nkey else model.qpos0, float).copy()
    values = dict(zip(names, q))
    for i, name in enumerate(target):
        if name in values:
            qout[i] = values[name]
    if missing:
        parts = robot_parts(model)
        d = mujoco.MjData(model)
        if not recorded_scene_sha:
            raise ValueError('rigid state conversion requires scene_sha256 and the matching recorded scene')
        runtime_data = None
        if recorded_scene_sha:
            from tool_library.common.evidence import file_digest
            paths = [directory / name for directory in (recorded_path.parent, recorded_path.parent.parent)
                     for name in ('scene_physics.xml', 'scene.xml')]
            source_path = next((p for p in paths if p.is_file() and file_digest(p) == recorded_scene_sha), None)
            if source_path is None:
                raise ValueError('rigid state conversion requires the matching recorded scene beside the episode or its parent')
            runtime_model = mujoco.MjModel.from_xml_path(str(source_path))
            if set(joint_names(runtime_model)) != set(names) or not RR.is_rigid(runtime_model):
                raise ValueError('recorded scene does not match the rigid state joint identities')
            runtime_data = mujoco.MjData(runtime_model)
            runtime_data.qpos[:] = [values[n] for n in joint_names(runtime_model)]
            mujoco.mj_kinematics(runtime_model, runtime_data)
        for side in SIDES:
            wanted = {f'robot_{side}_{side}_{k}' for k in RR.WRIST_SUFFIXES}
            if not missing & wanted:
                continue
            if missing & wanted != wanted or side not in parts:
                raise ValueError(f'{side}: only a complete rigid-to-proposal wrist conversion is allowed')
            d.qpos[:] = qout
            mujoco.mj_kinematics(model, d)
            hand = runtime_data.body(f'robot_{side}_{side}_base_yaw')
            p, R = hand.xpos.copy(), hand.xmat.reshape(3, 3).copy()
            P.set_wrist(model, d, parts[side]['rig'], p, R)
            qout = d.qpos.copy()
    return qout


# ----------------------------------------------------------------------------- robot structure
def robot_parts(model):
    """Per side: hand rig, hand root body, arm joints (ids), gripper (arm tip) body, welds between them."""
    out = {}
    rigs = P.scene_rigs(model)
    for side, rig in rigs.items():
        arm_j = [j for j in range(model.njnt) if model.joint(j).name.startswith(f"robot_{side}_arm_")]
        grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{side}_arm_gripper")
        welds = []
        for e in range(model.neq):
            if int(model.eq_type[e]) not in (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)):
                continue
            b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
            if b1 in rig.body_ids or b2 in rig.body_ids or b1 == grip or b2 == grip:
                welds.append(e)
        yaw_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{side}_{side}_base_yaw")
        out[side] = {"rig": rig, "arm_joints": arm_j, "arm_qadr": [int(model.jnt_qposadr[j]) for j in arm_j],
                     "arm_dadr": [int(model.jnt_dofadr[j]) for j in arm_j], "gripper": int(grip), "welds": welds,
                     "ee_body": int(yaw_body) if yaw_body >= 0 else int(rig.root_body)}
    return out


def hand_qadr(model, rig):
    """qpos addresses of the hand's own joints (wrist + fingers)."""
    adr = list(rig.wrist_qpos.values()) + list(rig.finger_qpos.values())
    return sorted(int(a) for a in adr)


def actuator_targets(model, rig):
    """[(actuator id, qpos address)] for every actuator of the hand (each drives one joint)."""
    pairs = []
    for a in list(rig.wrist_act.values()) + list(rig.finger_act.values()):
        j = int(model.actuator_trnid[a][0])
        pairs.append((int(a), int(model.jnt_qposadr[j])))
    return pairs


def object_free_joints(model):
    """{body name: (qpos adr)} for every free joint outside the robot."""
    out = {}
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            b = model.body(int(model.jnt_bodyid[j])).name
            if not b.startswith("robot_"):
                out[b] = int(model.jnt_qposadr[j])
    return out


# ----------------------------------------------------------------------------- grip force balance
class GripBalancer:
    """Grip force regulation for position-servoed fingers with torque caps. A hand that pinches a cap with two
    fingers against one thumb pushes the cap sideways with the difference once the servos saturate; a human
    balances the forces. Every `every` steps the balancer reads the contact forces each finger puts on the
    objects, low-pass filters them (contact forces spike on impact), forms the hand's net force, and when the
    grip is opposed (the side pushing against the net force carries at least 30% of the side pushing along it)
    backs off the fingers that push along the net, never beyond `max_back_rad` from their planned command;
    offsets relax back when the net force is small. A wrap pushing an object into the palm or a single-finger
    push is left alone, and the offsets reset when the planned finger commands change (open, regrasp)."""

    def __init__(self, model, parts, gain=0.0015, every=10, tol_n=1.0, max_back_rad=0.3, alpha=0.15, relax=0.0005):
        self.model, self.parts = model, parts
        self.gain, self.every, self.tol, self.max_back, self.alpha, self.relax = gain, every, tol_n, max_back_rad, alpha, relax
        self.offset = {s: {k: 0.0 for k in pt["rig"].finger_act} for s, pt in parts.items()}
        self.ema = {s: {} for s in parts}
        self.last_cmd = {s: None for s in parts}
        self.f6 = np.zeros(6)
        self.robot_bids = {b for b in range(model.nbody) if model.body(b).name.startswith("robot_")}
        self.finger_of_body = {}
        for s, pt in parts.items():
            for b in pt["rig"].body_ids:
                nm = model.body(b).name.split(f"{pt['rig'].side}_")[-1]
                self.finger_of_body[b] = (s, H.finger_of_joint(nm))
        self.n = 0
        self.log = {s: [] for s in parts}

    def reset_if_changed(self, side, cmd):
        last = self.last_cmd[side]
        if last is None or np.max(np.abs(cmd - last)) > 0.05:
            self.offset[side] = {k: 0.0 for k in self.offset[side]}
            self.ema[side] = {}
        self.last_cmd[side] = cmd.copy()

    def step(self, data):
        self.n += 1
        if self.n % self.every:
            return
        m = self.model
        forces = {s: {} for s in self.parts}
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.robot_bids, b2 in self.robot_bids
            if r1 == r2:
                continue
            hb, ob = (b1, b2) if r1 else (b2, b1)
            if m.body(ob).name in ("table", "world") or hb not in self.finger_of_body:
                continue
            side, fin = self.finger_of_body[hb]
            mujoco.mj_contactForce(m, data, i, self.f6)
            fw = np.asarray(c.frame, float).reshape(3, 3).T @ self.f6[:3]
            fw = fw if r1 else -fw                                   # force the hand link exerts on the object
            forces[side][fin or "palm"] = forces[side].get(fin or "palm", np.zeros(3)) + fw
        for side, per in forces.items():
            ema = self.ema[side]
            for f in set(ema) | set(per):
                ema[f] = (1 - self.alpha) * ema.get(f, np.zeros(3)) + self.alpha * per.get(f, np.zeros(3))
            fingers = [f for f in ema if f not in ("palm", None) and np.linalg.norm(ema[f]) > 0.3]
            rig = self.parts[side]["rig"]
            net = sum(ema.values()) if ema else np.zeros(3)
            mag = float(np.linalg.norm(net))
            act = False
            if len(fingers) >= 2 and mag > self.tol:
                u = net / mag
                strong = sum(max(0.0, float(ema[f] @ u)) for f in fingers)
                weak = sum(max(0.0, -float(ema[f] @ u)) for f in ema)
                if weak >= 0.3 * strong:
                    act = True
                    for f in fingers:
                        along = float(ema[f] @ u)
                        if along <= 0:
                            continue
                        for k in rig.finger_act:
                            if H.finger_of_joint(k) == f and H.joint_role(k) in ("flex", "pinky_arch"):
                                self.offset[side][k] = float(np.clip(self.offset[side][k] - self.gain * min(along, 15.0), -self.max_back, 0.0))
                    self.log[side].append((round(mag, 2), {f: round(float(np.linalg.norm(v)), 1) for f, v in ema.items()}))
            if not act:                                              # relax toward the planned command
                for k in self.offset[side]:
                    if self.offset[side][k] < 0:
                        self.offset[side][k] = min(0.0, self.offset[side][k] + self.relax)


def m_qadr(model, act):
    return model.jnt_qposadr[int(model.actuator_trnid[act][0])]


# ----------------------------------------------------------------------------- physics execution
REBASE_RY_DEG = 60.0      # a segment whose wrist rot_y passes this is servoed in a re-based wrist frame


def _wrist_rebase(model, parts, Qref, q_start):
    """Per side whose reference comes near the wrist chain's Euler singularity (rot_y +-90 deg: rot_x and rot_z
    align and the servos lose an axis; a hold at rot_y 85-94 deg drifted 50 deg), the rotation B that centres the
    segment: the chordal mean of its reference orientations. The rot_x body's frame is set to B for the segment, so
    the servoed angles stay near zero; the scene's own frame (identity) is what every other tool and the saved
    trajectory use."""
    out = {}
    for side, pt in parts.items():
        rig = pt["rig"]
        ix = [rig.wrist_qpos[k] for k in ("rx", "ry", "rz")]
        ry = np.abs(np.concatenate([np.asarray(Qref, float)[:, ix[1]], [float(q_start[ix[1]])]]))
        if float(np.degrees(ry.max())) <= REBASE_RY_DEG:
            continue
        jx = [j for j in range(model.njnt) if int(model.jnt_qposadr[j]) == ix[0]][0]
        body = int(model.jnt_bodyid[jx])
        if np.abs(model.body_quat[body] - np.array([1.0, 0.0, 0.0, 0.0])).max() > 1e-9 or np.abs(model.body_pos[body]).max() > 1e-12:
            continue                                   # a wrist chain with its own offset frame: leave it alone
        M = sum(H.euler_to_mat(np.asarray(Qref, float)[t, ix]) for t in range(len(Qref)))
        U, _, Vt = np.linalg.svd(M)
        B = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
        out[side] = {"body": body, "B": B, "ix": ix, "dof": [int(model.jnt_dofadr[j]) for j in range(model.njnt) if int(model.jnt_qposadr[j]) in ix]}
    return out


def _to_base(Q, ix, B, start):
    """Wrist angles of rows Q re-expressed in the frame B (nearest branch, continuous from `start`)."""
    out = np.array(Q, float, copy=True)
    prev = np.asarray(start, float)
    for t in range(len(out)):
        e = H.mat_to_euler(B.T @ H.euler_to_mat(out[t, ix]))
        e = H.nearest_euler_branch(e, prev)
        out[t, ix] = e; prev = e
    return out


def servo_rollout(model, Qref, fps, qpos_start=None, wrist_kp=10000.0, arm="home", record_contacts=True,
                  ctrl_lead_s=0.0, settle_s=0.0, progress=None, eq_inactive=(), rules=(), finger_force_nm=H.FINGER_FORCE_NM,
                  wrist_force_n=H.WRIST_FORCE_N, wrist_torque_nm=H.WRIST_TORQUE_NM, finger_kp=H.FINGER_KP, balance_grip=False, latched=None,
                  qvel_start=None, eq_active_start=None, state_start=None, mechanism_state_start=None):
    """Integrate bounded motor and finger controls; record actual physical states.

    A bench with arms always executes the rigid flange-hand runtime model.
    Floating wrist references are only proposals: their arm FK must already
    agree, and the adapter maps named physical joints to the runtime. Object
    columns are ignored as commands; objects move through contact/mechanisms.
    ``arm={side: [T,n_arm]}`` supplies motor references, never state overrides.
    `rules` are the scene's run-time mechanisms, applied every physics step: screw threads that let their cap go
    ({engaged, lift_joint, travel_m, frac}), scissor cuts ({kind: 'cut', ...}, cut_rules_live) and connector latches
    ({kind: 'latch', eq, site_a, site_b, seat_m, release_N}, latch_rules_live / latch_step). `latched` ({eq: relpose})
    hands over the latches the previous segment left engaged; the result's `latched` is this segment's, and
    `latch_events` lists every engage / release.
    Returns Q [T, nq] (what happened), tracking and contact diagnostics per frame."""
    from tool_library.common import rigid_robot as RR
    parts = robot_parts(model)
    if any(pt["arm_joints"] for pt in parts.values()):
        rigid = RR.is_rigid(model)
        adapter = None if rigid else RR.compile_rigid(model)
        runtime = model if rigid else adapter.model
        qr = np.asarray(Qref, float).copy()
        if isinstance(arm, dict):
            for side, values in arm.items():
                if side not in parts or np.asarray(values).shape != (len(qr), len(parts[side]["arm_qadr"])):
                    raise ValueError("arm reference does not match the named station's trajectory")
                qr[:, parts[side]["arm_qadr"]] = values
        start = np.asarray(qr[0] if qpos_start is None else qpos_start, float)
        if adapter is not None:
            adapter.check_reference(qr)
            adapter.check_reference(start)
        out = RR.servo_rollout(runtime, qr if rigid else adapter.to_runtime_qpos(qr), fps,
                              qpos_start=start if rigid else adapter.to_runtime_qpos(start), qvel_start=qvel_start,
                              wrist_kp=wrist_kp, record_contacts=record_contacts, ctrl_lead_s=ctrl_lead_s,
                              settle_s=settle_s, progress=progress, eq_inactive=eq_inactive, rules=rules,
                              finger_force_nm=finger_force_nm, wrist_force_n=wrist_force_n, wrist_torque_nm=wrist_torque_nm,
                              finger_kp=finger_kp, balance_grip=balance_grip, latched=latched, eq_active_start=eq_active_start,
                              state_start=state_start, mechanism_state_start=mechanism_state_start)
        if adapter is not None:
            # Compatibility view for semantic/planning tools only. The physical
            # state, controls and XML below are the authoritative episode.
            out["runtime_qpos"] = out["qpos"].copy()
            out["runtime_qvel"] = out["qvel"].copy()
            out["runtime_ctrl"] = out["ctrl"].copy()
            out["runtime_scene_xml"] = adapter.xml
            out["runtime_scene_sha256"] = adapter.scene_sha256
            out["runtime_joint_names"] = np.array(joint_names(runtime))
            out["runtime_actuator_names"] = np.array([runtime.actuator(i).name for i in range(runtime.nu)])
            out["qpos"] = adapter.to_proposal_qpos(out["runtime_qpos"], seed=start)
            out["final_qpos"] = out["qpos"][-1].copy()
        return out
    q_start = np.asarray(Qref[0] if qpos_start is None else qpos_start, float)
    rebase = _wrist_rebase(model, parts, Qref, q_start)
    if not rebase:
        return _servo_rollout(model, Qref, fps, q_start, wrist_kp, arm, record_contacts, ctrl_lead_s, settle_s, progress, eq_inactive,
                              rules, finger_force_nm, wrist_force_n, wrist_torque_nm, finger_kp, balance_grip, latched)
    Qr = np.asarray(Qref, float)
    q0 = q_start.copy()
    for side, rb in rebase.items():                    # segments start at rest (a fresh MjData), so only angles convert
        q0 = _to_base(q0[None], rb["ix"], rb["B"], np.zeros(3))[0]
        Qr = _to_base(Qr, rb["ix"], rb["B"], q0[rb["ix"]])
    saved = {side: model.body_quat[rb["body"]].copy() for side, rb in rebase.items()}
    try:
        for side, rb in rebase.items():
            qb = np.zeros(4); mujoco.mju_mat2Quat(qb, rb["B"].reshape(-1)); model.body_quat[rb["body"]] = qb
        out = _servo_rollout(model, Qr, fps, q0, wrist_kp, arm, record_contacts, ctrl_lead_s, settle_s, progress, eq_inactive,
                             rules, finger_force_nm, wrist_force_n, wrist_torque_nm, finger_kp, balance_grip, latched)
    finally:
        for side, rb in rebase.items():
            model.body_quat[rb["body"]] = saved[side]
    Q = np.asarray(out["qpos"], float)
    for side, rb in rebase.items():                    # back to the scene's own wrist frame
        prev = q_start[rb["ix"]]
        for t in range(len(Q)):
            e = H.mat_to_euler(rb["B"] @ H.euler_to_mat(Q[t, rb["ix"]]))
            e = H.nearest_euler_branch(e, prev); Q[t, rb["ix"]] = e; prev = e
    out["qpos"] = Q; out["final_qpos"] = Q[-1].copy()
    out["wrist_rebased"] = {side: {"rot_y_max_deg": round(float(np.degrees(np.abs(np.asarray(Qref, float)[:, rb["ix"][1]]).max())), 1)} for side, rb in rebase.items()}
    return out


def _servo_rollout(model, Qref, fps, qpos_start, wrist_kp, arm, record_contacts, ctrl_lead_s, settle_s, progress, eq_inactive,
                   rules, finger_force_nm, wrist_force_n, wrist_torque_nm, finger_kp, balance_grip, latched=None):
    parts = robot_parts(model)
    if any(pt["arm_joints"] for pt in parts.values()):
        raise ValueError("floating-hand helper cannot execute an arm; use rigid servo_rollout")
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos_start, float)
    mujoco.mj_forward(model, data)
    gains, force_sched = {}, {}
    for side, pt in parts.items():
        gains[side] = H.set_wrist_gain(model, pt["rig"], kp=float(wrist_kp)) if wrist_kp else None
        # wrist_force_n may differ per hand ({side: cap}) and per frame ({side: [T, 3]}): a hand seating a part pushes with
        # its own cap during the press only, while the other hand lifts with the library's
        wf_side = wrist_force_n.get(side, H.WRIST_FORCE_N) if isinstance(wrist_force_n, dict) else wrist_force_n
        if wf_side is not None and np.ndim(wf_side) == 2:
            force_sched[side] = np.asarray(wf_side, float)
            wf_side = force_sched[side][0].tolist()
        gains[f"{side}_limits"] = H.limit_hand_forces(model, pt["rig"], finger_force_nm, wf_side, wrist_torque_nm, finger_kp=finger_kp)
    arm_path = arm if isinstance(arm, dict) else None
    if arm == "home" or arm_path is not None:
        for pt in parts.values():
            for e in pt["welds"]:
                data.eq_active[e] = 0
    inactive = set()
    for nm in eq_inactive or ():                      # constraints an earlier segment released (a thread that ran out)
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, nm)
        if e >= 0:
            data.eq_active[e] = 0; inactive.add(nm)
    live_rules = []
    has_cut_rules = any(r.get("kind") == "cut" for r in rules or ())
    live_cuts = cut_rules_live(model, rules, inactive)   # a rope cut by scissors (scissors.try_sever every step)
    for r in rules or ():                             # {"engaged": eq name, "lift_joint": name, "travel_m": x, "frac": 0.96}
        if r.get("kind") in ("cut", "latch"):
            continue
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, r["engaged"])
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, r["lift_joint"])
        if e >= 0 and j >= 0 and r["engaged"] not in inactive:
            live_rules.append((e, int(model.jnt_qposadr[j]), float(r.get("frac", 0.96)) * float(r["travel_m"]), r["engaged"]))
    released = []
    arm_q = {s: data.qpos[pt["arm_qadr"]].copy() for s, pt in parts.items()}
    acts = {s: actuator_targets(model, pt["rig"]) for s, pt in parts.items()}
    balancer = GripBalancer(model, parts) if balance_grip else None
    finger_qadr = {s: np.array([int(model.jnt_qposadr[int(model.actuator_trnid[a][0])]) for a in pt["rig"].finger_act.values()]) for s, pt in parts.items()}
    hand_bodies = set().union(*[pt["rig"].body_ids for pt in parts.values()])
    T = len(Qref)
    dt = float(model.opt.timestep)
    n_sub = max(1, int(round(1.0 / (float(fps) * dt))))
    lead = int(round(float(ctrl_lead_s) * fps))
    Q = np.zeros((T, model.nq))
    track = np.zeros((T, 2))                        # wrist position / orientation tracking error per frame (worst side)
    wf = np.zeros((T, 2))                           # peak wrist servo force (N) and torque (N m) over the frame, worst side
    pen = np.zeros(T)                               # deepest hand-object or hand-table interpenetration (m)
    obj_pen = np.zeros(T)                           # deepest object-object interpenetration (m)
    contacts = []                                   # per frame: set of (hand side, body) pairs in contact
    hand_force = np.zeros((T, 2))                   # net contact force each hand puts on the objects (N): left, right
    # the equality state in force after each frame: a cut is run-time state that no qpos carries, so a scene with
    # cut joins records it (scenes without are left exactly as they were)
    rec_eq = model.neq > 0                          # every equality's state per frame (cuts, threads that run out): a replay
                                                     # from a fresh MjData otherwise re-activates what the rollout released
    eq_hist = np.zeros((T, model.neq), np.int8) if rec_eq else None
    eq_start = data.eq_active.astype(np.int8).copy() if rec_eq else None
    # connector latches: taken over from the previous segment, or engaged here when the plug already sits in its port
    # (after eq_start, which is the fresh MjData's state a replay starts from)
    live_latches = latch_rules_live(model, rules)
    latch_events = latch_begin(model, data, live_latches, inactive, latched, [], 0) if live_latches else []
    f6 = np.zeros(6)
    robot_bids = {b for b in range(model.nbody) if model.body(b).name.startswith("robot_")}
    def apply_ctrl(t_cmd):
        t_cmd = min(t_cmd, T - 1)
        for s, pairs in acts.items():
            for a, qa in pairs:
                data.ctrl[a] = Qref[t_cmd][qa]
    apply_ctrl(0)
    unstable = None; T_done = T
    if settle_s > 0:
        for _ in range(int(settle_s / dt)):
            mujoco.mj_step(model, data)
            if live_latches:
                latch_step(model, data, live_latches, inactive, latch_events, 0)
            for s, pt in parts.items():
                data.qpos[pt["arm_qadr"]] = arm_q[s]; data.qvel[pt["arm_dadr"]] = 0.0
    for t in range(T):
        for side, F in force_sched.items():          # a per-frame wrist force cap (the press phase of press_seat)
            row = F[min(t, len(F) - 1)]
            for i_, ax in enumerate(("x", "y", "z")):
                model.actuator_forcerange[parts[side]["rig"].wrist_act[ax]] = [-float(row[i_]), float(row[i_])]
        for k in range(n_sub):
            # command: linear interpolation between reference frames, optionally leading the plan
            u = (k + 1) / n_sub
            t0 = min(t + lead, T - 1); t1 = min(t0 + 1, T - 1)
            for s, pairs in acts.items():
                for a, qa in pairs:
                    data.ctrl[a] = (1 - u) * Qref[t0][qa] + u * Qref[t1][qa]
            if balancer is not None:
                for s, pt in parts.items():
                    balancer.reset_if_changed(s, Qref[t1][finger_qadr[s]])
                    for k_, a in pt["rig"].finger_act.items():
                        data.ctrl[a] += balancer.offset[s][k_]
            mujoco.mj_step(model, data)
            if balancer is not None:
                balancer.step(data)
            for s, pt in parts.items():
                wa = pt["rig"].wrist_act
                fn = float(np.linalg.norm([data.actuator_force[wa[k]] for k in ("x", "y", "z")]))
                tn = float(np.linalg.norm([data.actuator_force[wa[k]] for k in ("rx", "ry", "rz")]))
                wf[t, 0] = max(wf[t, 0], fn); wf[t, 1] = max(wf[t, 1], tn)
            if arm == "home" or arm_path is not None:
                for s, pt in parts.items():
                    if arm_path is not None and s in arm_path:
                        A = arm_path[s]
                        data.qpos[pt["arm_qadr"]] = (1 - u) * A[min(t0, len(A) - 1)] + u * A[min(t1, len(A) - 1)]
                    else:
                        data.qpos[pt["arm_qadr"]] = arm_q[s]
                    data.qvel[pt["arm_dadr"]] = 0.0
            for e, qa, thr, nm in live_rules:         # a thread that has travelled its length lets the cap go
                if data.eq_active[e] and abs(float(data.qpos[qa])) >= thr:
                    data.eq_active[e] = 0; inactive.add(nm); released.append({"equality": nm, "frame": t})
            if live_cuts:                             # the blades closed on a join: the rope parts there
                cut_step(model, data, live_cuts, inactive, released, t)
            if live_latches:                          # a plug reached its port floor, or was pulled out
                latch_step(model, data, live_latches, inactive, latch_events, t)
        mujoco.mj_forward(model, data)
        Q[t] = data.qpos
        if rec_eq:
            eq_hist[t] = data.eq_active
        # instability guard: a body faster than any manipulation, a non-finite state, or MuJoCo's own
        # bad-acceleration reset (which silently returns the scene to qpos0) ends the segment
        vmax = float(np.max(np.abs(data.qvel))) if model.nv else 0.0
        badacc = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0
        if not np.isfinite(data.qpos).all() or vmax > 25.0 or badacc:
            unstable = {"frame": t, "max_qvel": vmax if np.isfinite(vmax) else None, "bad_qacc_reset": badacc}
            Q[t:] = Q[t - 1] if t else Q[t]
            if rec_eq:
                eq_hist[t:] = data.eq_active
            T_done = t
            break
        # tracking error of the wrist frames against the reference
        worst = [0.0, 0.0]
        for s, pt in parts.items():
            rig = pt["rig"]
            pr = np.array([Qref[t][rig.wrist_qpos[k]] for k in ("x", "y", "z")]); pa = rig.wrist_pos(data)
            er = np.array([Qref[t][rig.wrist_qpos[k]] for k in ("rx", "ry", "rz")]); ea = rig.wrist_euler(data)
            Rr, Ra = H.euler_to_mat(er), H.euler_to_mat(ea)
            ang = np.degrees(np.linalg.norm(Rot.from_matrix(Rr.T @ Ra).as_rotvec()))
            worst = [max(worst[0], float(np.linalg.norm(pr - pa))), max(worst[1], float(ang))]
        track[t] = worst
        if record_contacts:
            cs, dmax, omax = set(), 0.0, 0.0
            net = {s: np.zeros(3) for s in parts}
            for i in range(data.ncon):
                c = data.contact[i]
                b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
                r1, r2 = b1 in robot_bids, b2 in robot_bids
                d = float(c.dist)
                if r1 != r2:
                    other = b2 if r1 else b1
                    hb = b1 if r1 else b2
                    side = next((s for s, pt in parts.items() if hb in pt["rig"].body_ids), None)
                    if side:
                        cs.add((side, model.body(other).name))
                        if model.body(other).name not in ("table", "world"):
                            mujoco.mj_contactForce(model, data, i, f6)
                            fw = np.asarray(c.frame, float).reshape(3, 3).T @ f6[:3]
                            net[side] += fw if r1 else -fw      # force the hand exerts on the object
                    if -d > 0.005:
                        # the same collider artifact reaches hand hulls against meshes: a deep reading is re-measured
                        # on the geom pair (a true penetration keeps its depth)
                        d = max(d, H.geom_distance(model, data, int(c.geom1), int(c.geom2), 0.01))
                    dmax = max(dmax, -d)
                elif not r1 and not r2:
                    if -d > 0.005:
                        # MuJoCo's box-box collider now and then emits one contact as deep as a box edge for two
                        # nearly aligned faces (measured: 80 mm on one frame between two bars squeezed end to end, 2.2 mm by
                        # the exact distance): a deep reading is re-measured on the geom pair
                        d = max(d, H.geom_distance(model, data, int(c.geom1), int(c.geom2), 0.01))
                    omax = max(omax, -d)
            pen[t], obj_pen[t] = dmax, omax
            contacts.append(sorted(cs))
            for si, s_ in enumerate(("left", "right")):
                if s_ in net:
                    hand_force[t, si] = float(np.linalg.norm(net[s_]))
        if progress and t % max(1, T // 10) == 0:
            progress(t, T)
    if unstable is not None:
        contacts += [contacts[-1] if contacts else []] * (T - len(contacts))
    out = {"qpos": Q, "unstable": unstable, "frames_simulated": T_done, "track_pos_m": track[:, 0], "track_rot_deg": track[:, 1], "hand_pen_m": pen,
           "wrist_force_n": wf[:, 0], "wrist_torque_nm": wf[:, 1], "hand_object_force_n": hand_force,
           "object_pen_m": obj_pen, "contacts": contacts, "gains": gains, "final_qpos": Q[-1].copy(),
           "eq_inactive": sorted(inactive), "released": released,
           "grip_balance": None if balancer is None else {s: {"final_offsets_rad": {k: round(v, 3) for k, v in o.items() if abs(v) > 1e-4}, "events": len(balancer.log[s])} for s, o in balancer.offset.items()}}
    for side in force_sched:                         # the shared model leaves the rollout with the library's caps
        for ax in ("x", "y", "z"):
            model.actuator_forcerange[parts[side]["rig"].wrist_act[ax]] = [-float(H.WRIST_FORCE_N), float(H.WRIST_FORCE_N)]
    # engaged latches go to the next segment explicitly (latched); a pulled-out one is in eq_inactive (disarmed); the
    # latch welds' eq_data are put back as the scene has them
    out["latched"] = latch_end(model, live_latches) if live_latches else {}
    out["latch_events"] = latch_events
    if rec_eq:
        # eq_active[t] = the state in force after frame t; eq_active_start = before the first frame
        out.update({"eq_active": eq_hist, "eq_active_start": eq_start, "eq_names": equality_names(model),
                    "cuts": [r for r in released if "cut" in r]})
    return out


def cut_rules_live(model, rules, inactive=()):
    """The cut rules of a rollout as [(cutter, join, try_sever kwargs, scene prefix)]: one entry per rope join that is in
    the model and still whole. A rule is {"kind": "cut", "cutter": scissors cutter with its scene `prefix`
    (scissors.cutter_from_recon), "joins": [{eq, body, anchor_local, diameter_m}], "gates": "cord" (default: the gates
    scaled to the cord as scissors_cut_test uses them) | "tool" (the scissors' own), "sever_kwargs": {...} overrides}."""
    from tool_library.common import scissors as SC
    out = []
    for r in rules or ():
        if r.get("kind") != "cut":
            continue
        for j in r.get("joins") or ():
            e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, j["eq"])
            if e < 0 or j["eq"] in set(inactive or ()):
                continue
            c, kw = (SC.cord_gates(r["cutter"], j.get("diameter_m", r["cutter"].get("cut_gap_m", 0.0035)))
                     if r.get("gates", "cord") == "cord" else (r["cutter"], {}))
            out.append((c, j, dict(kw, **(r.get("sever_kwargs") or {})), str(r["cutter"].get("prefix", ""))))
    return out


def cut_step(model, data, live_cuts, inactive, released, frame):
    """One physics step of the cut rules: for every whole join near the blades, scissors.try_sever switches its equality
    off when the six measured gates hold (in span, in wedge, in plane, on the blade line, closing, biting). A cut adds
    the equality to `inactive` and {equality, frame, cut} to `released`. Returns the joins cut on this step."""
    from tool_library.common import scissors as SC
    done = []
    for c, j, kw, pre in live_cuts:
        if j["eq"] in inactive or not SC.join_near_blades(model, data, c, j, pre):
            continue
        cut = SC.try_sever(model, data, c, [j], pre, **kw)
        if cut:
            inactive.add(j["eq"])
            released.append({"equality": j["eq"], "frame": int(frame), "cut": cut[0]})
            done.append(cut[0])
    return done


def equality_names(model):
    """Every equality's name in id order (unnamed ones as 'eq<id>'): the columns of a recorded eq_active."""
    return [model.equality(e).name or f"eq{e}" for e in range(model.neq)]


# ----------------------------------------------------------------------------- connector latches
# A USB-C plug is held in its receptacle by spring latches, not by friction alone. A gram-scale friction fit holds or
# not depending on the contact model (its softness, the noslip iterations, the timestep; a squeezed soft contact can
# throw the plug out), and it lets go at whatever force the friction gives. The latch makes the hold and the pull-out
# force properties of the connector: an inactive weld between the port body and the plug body that the rollout
# switches on when the plug tip has bottomed out on the port floor and off when the plug is pulled out harder than the
# latch holds.
LATCH_SEAT_M = 0.0015          # tip-to-floor distance that counts as seated
LATCH_RELEASE_N = 15.0         # pull-out force of the latch (USB-C extraction force is 8-20 N)
LATCH_ALIGN_DEG = 20.0         # the plug and port axes must agree this well to engage
LATCH_TAU_S = 0.005            # low-pass of the pull-out load (a one-step contact spike does not unplug it)
# a latch clicks where the plug bottoms out: inside seat_m it engages once the tip no longer advances faster than this
# (a plug pushed in at 0.4 m/s otherwise latched on the first step inside seat_m, 0.7-1.5 mm short of the floor, and
# the weld then held it there against the push)
LATCH_SEAT_SPEED_M_S = 0.02
# ... and once it no longer presses on the port harder than this along the axis: soft contact lets a pushed plug sink
# into the port floor (0.2-0.8 mm under 3 N), and a latch that caught it there kept the push as a preload (measured:
# a 3 N push locked in, so a 12 N pull unplugged a 15 N latch); a hand that is still pressing is let go of first
LATCH_SEAT_FORCE_N = 1.0
# the latch weld: nearly hard, so an engaged gram-scale plug moves hundredths of a millimetre under a cord's weight
# and well under a millimetre before a 15 N pull lets it go (measured on a 6 g plug at dt 0.002: MuJoCo's default weld
# gives 3.3 mm/N, solref 0.004 with solimp 0.95 still 0.14 mm/N, this one 0.005 mm/N); 0.004 s is two of the bench's
# 0.002 s steps, the shortest time constant MuJoCo keeps stable
LATCH_WELD_SOLREF = [0.004, 1.0]
LATCH_WELD_SOLIMP = [0.998, 0.999, 0.001, 0.5, 2.0]


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], float)


def _quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def weld_relpose_now(model, data, e):
    """The relpose (pos 3, quat 4) that makes body-body weld e exactly satisfied at the current kinematics: the weld
    point (eq_data anchor, in body2's frame) expressed in body1's frame, and body2's orientation relative to body1.
    MuJoCo's weld residual is (x1 + R1 relpos) - (x2 + R2 anchor) and neg(q2) q1 relquat, so with these values both
    vanish and switching the weld on does not move anything."""
    b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
    anchor = np.asarray(model.eq_data[e][0:3], float)
    R1 = data.xmat[b1].reshape(3, 3); R2 = data.xmat[b2].reshape(3, 3)
    pos = R1.T @ (data.xpos[b2] + R2 @ anchor - data.xpos[b1])
    quat = _quat_mul(_quat_conj(data.xquat[b1]), data.xquat[b2])
    return np.concatenate([pos, quat / np.linalg.norm(quat)])


def latch_seated_eq_data(model, site_a, site_b, roll_deg=0.0, torquescale=1.0):
    """eq_data (anchor 3, relpose 7, torquescale 1) of a weld body1 = the body of site_b (the port), body2 = the body
    of site_a (the plug) holding the plug SEATED: the weld point is the plug tip site, held at the port floor site, and
    the tip site's frame coincides with the floor site's frame turned roll_deg about its z axis (the insertion axis;
    a reversible plug seats at 0 or 180)."""
    sa = site_a if isinstance(site_a, (int, np.integer)) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_a)
    sb = site_b if isinstance(site_b, (int, np.integer)) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_b)
    if sa < 0 or sb < 0:
        raise ValueError(f"latch_seated_eq_data: site {site_a if sa < 0 else site_b!r} is not in the model")
    qa, qb = np.asarray(model.site_quat[sa], float), np.asarray(model.site_quat[sb], float)
    h = np.radians(float(roll_deg)) / 2.0
    qroll = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
    # T_plug = T_port T_b Rz(roll) T_a^-1  ->  q_rel = qb qroll qa^-1
    qrel = _quat_mul(_quat_mul(qb, qroll), _quat_conj(qa))
    return np.concatenate([np.asarray(model.site_pos[sa], float), np.asarray(model.site_pos[sb], float),
                           qrel / np.linalg.norm(qrel), [float(torquescale)]])


def latch_rules_from_objects(objects):
    """Latch rules from Stage 2 manifest object entries: each entry's `latch` (one dict or a list, as
    add_connector_latch's scene_fields write it: {eq, site_a, site_b, seat_m, release_N, ...})."""
    rules = []
    for o in objects or ():
        if not isinstance(o, dict):
            continue
        lt = o.get("latch")
        for x in (lt if isinstance(lt, list) else [lt] if isinstance(lt, dict) else []):
            if not isinstance(x, dict) or not (x.get("eq") and x.get("site_a") and x.get("site_b")):
                continue
            r = {"kind": "latch", "eq": str(x["eq"]), "site_a": str(x["site_a"]), "site_b": str(x["site_b"]),
                 "seat_m": float(x.get("seat_m", LATCH_SEAT_M)), "release_N": float(x.get("release_N", LATCH_RELEASE_N))}
            for k in ("align_deg", "rearm_m", "tau_s", "seat_speed_m_s", "seat_force_N"):
                if x.get(k) is not None:
                    r[k] = float(x[k])
            rules.append(r)
    return rules


def latch_rules_live(model, rules):
    """The latch rules of a rollout as per-latch state dicts. A rule is {"kind": "latch", "eq": weld name, "site_a":
    plug tip site, "site_b": port floor site, "seat_m": 0.0015, "release_N": 15, "align_deg": 20, "rearm_m":
    max(2 seat_m, seat_m + 2 mm), "tau_s": 0.005, "seat_speed_m_s": 0.02, "seat_force_N": 1.0}. The weld must be
    body-body with site_a on one of its bodies and site_b on the other. A rule whose equality or sites are not in the
    model is skipped (a scene without that part)."""
    out = []
    W, B = int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtObj.mjOBJ_BODY)
    dt = float(model.opt.timestep)
    for r in rules or ():
        if r.get("kind") != "latch":
            continue
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, str(r["eq"]))
        sa = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(r["site_a"]))
        sb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, str(r["site_b"]))
        if e < 0 or sa < 0 or sb < 0:
            continue
        if int(model.eq_type[e]) != W or int(model.eq_objtype[e]) != B:
            raise ValueError(f"latch rule: equality {r['eq']!r} must be a body-body weld (add_connector_latch writes one)")
        b1, b2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        ba, bb = int(model.site_bodyid[sa]), int(model.site_bodyid[sb])
        if {ba, bb} != {b1, b2}:
            raise ValueError(f"latch rule {r['eq']!r}: site_a {r['site_a']!r} and site_b {r['site_b']!r} must sit on the weld's two bodies "
                             f"({model.body(b1).name}, {model.body(b2).name})")
        seat = float(r.get("seat_m", LATCH_SEAT_M))
        tau = float(r.get("tau_s", LATCH_TAU_S))
        out.append({"e": int(e), "name": str(r["eq"]), "sa": int(sa), "sb": int(sb), "plug_is_body2": ba == b2, "ba": ba, "bb": bb,
                    "seat_force": float(r.get("seat_force_N", LATCH_SEAT_FORCE_N)),
                    "seat": seat, "release": float(r.get("release_N", LATCH_RELEASE_N)),
                    "align_cos": float(np.cos(np.radians(float(r.get("align_deg", LATCH_ALIGN_DEG))))),
                    "rearm": float(r.get("rearm_m", max(2.0 * seat, seat + 0.002))), "alpha": dt / (tau + dt),
                    "seat_speed": float(r.get("seat_speed_m_s", LATCH_SEAT_SPEED_M_S)), "dt": dt,
                    "data0": np.array(model.eq_data[e], float), "state": "armed", "pull": 0.0, "pull_peak": 0.0, "gap_prev": None})
    return out


def latch_gap(model, data, L):
    """(distance between the plug tip and the port floor sites, cosine between their z axes) at the current data."""
    pa, pb = data.site_xpos[L["sa"]], data.site_xpos[L["sb"]]
    za, zb = data.site_xmat[L["sa"]].reshape(3, 3)[:, 2], data.site_xmat[L["sb"]].reshape(3, 3)[:, 2]
    return float(np.linalg.norm(pa - pb)), float(za @ zb)


def latch_press(model, data, L):
    """How hard the plug presses on the port along the insertion axis (N): the contact forces between the plug's body
    and the port's body, on the plug, against the insertion direction (a plug pushed into the floor gets it back)."""
    u_in = data.site_xmat[L["sb"]].reshape(3, 3)[:, 2]
    f6 = np.zeros(6)
    press = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if {b1, b2} != {L["ba"], L["bb"]}:
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        fw = np.asarray(c.frame, float).reshape(3, 3).T @ f6[:3]      # the force geom1 exerts on geom2
        f_plug = fw if b2 == L["ba"] else -fw
        press -= float(f_plug @ u_in)
    return press


def latch_pull(model, data, L):
    """Pull-out load the engaged latch carries (N): the weld force on the plug along the port's insertion axis (the
    floor site's z). The weld's first three rows are its world-frame position force f, applied +f to body1 and -f to
    body2; an outward load on the plug is balanced by an inward weld force, so the pull is that force along the
    insertion axis. Compression (a hand still pressing the plug home) and side loads (the port walls take those on a
    real connector) are not pull."""
    e = L["e"]
    ne = int(data.ne)
    if ne <= 0:
        return 0.0
    typ = data.efc_type[:ne]; ids = data.efc_id[:ne]
    rows = np.flatnonzero((typ == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) & (ids == e))
    if len(rows) < 3:
        return 0.0
    f = np.asarray(data.efc_force[rows[:3]], float)
    f_plug = -f if L["plug_is_body2"] else f
    u_in = data.site_xmat[L["sb"]].reshape(3, 3)[:, 2]
    return float(f_plug @ u_in)


def _latch_engage(model, data, L, events, frame, gap):
    mujoco.mj_kinematics(model, data)            # after mj_step the poses lag the integrated qpos by one step
    rel = weld_relpose_now(model, data, L["e"])
    model.eq_data[L["e"]][3:10] = rel
    data.eq_active[L["e"]] = 1
    L["state"], L["pull"], L["pull_peak"] = "engaged", 0.0, 0.0
    events.append({"equality": L["name"], "frame": int(frame), "event": "engage", "gap_mm": round(1000.0 * gap, 3),
                   "relpose": [round(float(v), 7) for v in rel]})


def latch_begin(model, data, live, inactive, latched=None, events=None, frame=0):
    """Start of a rollout (after mj_forward). Each latch takes its state: a latch in `latched` ({eq: relpose 7}, the
    engaged latches the previous segment handed over) is switched on with that relpose; one in `inactive` (pulled out
    by an earlier segment) is disarmed until its tip has left the seat (rearm_m); one the scene starts active
    (eq_active0, a plug that starts in its port) is engaged with the scene's relpose; every other latch is armed, and
    an armed latch whose tip is already seated (and not pressed harder than seat_force_N) engages here, before the first
    step, with the current relative pose.
    That last rule also carries an engaged latch across segments when a caller does not pass `latched` (each segment
    starts from a fresh MjData, where the weld is off): it re-engages at once where the plug sits, so nothing snaps,
    and only the weld's compliance under load (hundredths of a millimetre with the weld add_connector_latch writes) is
    taken as the new seat."""
    events = [] if events is None else events
    latched = latched or {}
    for L in live:
        e, nm = L["e"], L["name"]
        if nm in latched:
            model.eq_data[e][3:10] = np.asarray(latched[nm], float)
            data.eq_active[e] = 1; L["state"] = "engaged"; inactive.discard(nm)
        elif nm in inactive:
            data.eq_active[e] = 0; L["state"] = "disarmed"
        elif data.eq_active[e]:
            L["state"] = "engaged"
        else:
            L["state"] = "armed"
        L["pull"] = L["pull_peak"] = 0.0
    for L in live:
        gap, c = latch_gap(model, data, L)
        L["gap_prev"] = gap
        if L["state"] == "armed" and gap <= L["seat"] and c >= L["align_cos"] and latch_press(model, data, L) <= L["seat_force"]:
            _latch_engage(model, data, L, events, frame, gap)
    return events


def latch_step(model, data, live, inactive, events, frame):
    """One physics step of the latch rules (call after mj_step). An armed latch whose plug tip is within seat_m of
    the port floor (axes within align_deg), no longer advancing faster than seat_speed_m_s (it has bottomed out, or
    stopped) and not pressed on the port harder than seat_force_N (latch_press: the push that seated it is not locked
    in) engages: the weld's relpose is set to the bodies' current relative pose and the weld switched on, so it never
    snaps. An engaged latch whose low-passed pull-out load (latch_pull) exceeds release_N switches off (a pull-out),
    goes into `inactive` (so a segment that ends before the plug is out starts the next one disarmed) and stays
    disarmed until the tip is rearm_m from the floor, where it is armed again and leaves `inactive`. Events go to `events` as {equality, frame, event: engage |
    release, gap_mm | pull_n}. Returns the events of this step."""
    n0 = len(events)
    for L in live:
        e, nm = L["e"], L["name"]
        if L["state"] == "engaged":
            if not data.eq_active[e]:
                L["state"] = "armed"
                continue
            p = latch_pull(model, data, L)
            L["pull"] += L["alpha"] * (p - L["pull"])
            L["pull_peak"] = max(L["pull_peak"], L["pull"])
            if L["pull"] > L["release"]:
                data.eq_active[e] = 0; L["state"] = "disarmed"; inactive.add(nm)
                events.append({"equality": nm, "frame": int(frame), "event": "release", "pull_n": round(float(L["pull"]), 2)})
            continue
        gap, c = latch_gap(model, data, L)
        prev, L["gap_prev"] = L["gap_prev"], gap
        if L["state"] == "disarmed":
            if gap > L["rearm"]:
                L["state"] = "armed"; inactive.discard(nm)
            continue
        advancing = prev is not None and (prev - gap) / L["dt"] > L["seat_speed"]
        if gap <= L["seat"] and c >= L["align_cos"] and not advancing and latch_press(model, data, L) <= L["seat_force"]:
            _latch_engage(model, data, L, events, frame, gap)
    return events[n0:]


def latch_end(model, live):
    """End of a rollout: {eq: relpose} of the engaged latches (the next segment's `latched`), and every latch weld's
    eq_data put back as the scene has it (the model is shared between segments and replays)."""
    out = {L["name"]: [float(v) for v in model.eq_data[L["e"]][3:10]] for L in live if L["state"] == "engaged"}
    for L in live:
        model.eq_data[L["e"]] = L["data0"]
    return out


# ----------------------------------------------------------------------------- kinematic collision sweep
def plan_collision_sweep(model, Qref, allowed_bodies=(), stride=3, tol_m=0.004, max_hits=6):
    """Frames of a kinematic reference where the plan itself drives a hand through something: the other hand,
    or an object that is not the part being handled (`allowed_bodies`: the manipulated part's subtree, a cable link's neighbours, and the
    objects the hands hold, whose contact is the point). Returns [{frame, depth_mm, geoms, kind}] with kind
    'hand-hand' or 'hand-object'."""
    data = mujoco.MjData(model)
    left = {g for g in range(model.ngeom) if "_collision_hand_left_" in model.geom(g).name}
    right = {g for g in range(model.ngeom) if "_collision_hand_right_" in model.geom(g).name}
    hands = left | right
    allowed = set()
    for b in allowed_bodies or ():
        bid = b if isinstance(b, int) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if bid >= 0:
            allowed |= held_bodies(model, bid)
    robot = {b for b in range(model.nbody) if model.body(b).name.startswith("robot_")}
    hits = []
    for t in range(0, len(Qref), stride):
        data.qpos[:] = Qref[t]
        mujoco.mj_forward(model, data)
        worst = None
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist >= -tol_m:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in left and g2 in right) or (g1 in right and g2 in left):
                kind = "hand-hand"
            elif (g1 in hands) != (g2 in hands):
                if allowed_bodies is None:
                    continue
                other = g2 if g1 in hands else g1
                ob = int(model.geom_bodyid[other])
                if ob in robot or ob in allowed or model.body(ob).name in ("table", "world"):
                    continue
                kind = "hand-object"
            else:
                continue
            if worst is None or c.dist < worst[1]:
                worst = (t, float(c.dist), model.geom(g1).name, model.geom(g2).name, kind)
        if worst:
            hits.append({"frame": worst[0], "depth_mm": round(-1000 * worst[1], 1), "geoms": [worst[2], worst[3]], "kind": worst[4]})
    hits.sort(key=lambda h: -h["depth_mm"])
    return hits[:max_hits]


def hand_hand_sweep(model, Qref, stride=3, tol_m=0.004):
    """Hand-hand interpenetrations of a kinematic reference (see plan_collision_sweep)."""
    return [(h["frame"], -h["depth_mm"] / 1000, h["geoms"][0], h["geoms"][1]) for h in plan_collision_sweep(model, Qref, allowed_bodies=None, stride=stride, tol_m=tol_m) if h["kind"] == "hand-hand"]


# ----------------------------------------------------------------------------- arms
def fill_arms(model, Q, sides=None, rest=None, cfg=None, fps=30.0, max_arm_rate_deg_s=95.0, smooth_window=7):
    """Put every arm behind its hand on every frame: the gripper body is fitted to the hand's end-effector
    body pose (the pose the weld attaches) by the continuous arm fitter, then the joint path is rate-limited
    (a real arm has a speed limit; a branch flip of the fitter would otherwise turn a link 300 deg/s) and
    lightly smoothed. Returns (Q, report per side) with the residuals measured AFTER the limiting."""
    parts = robot_parts(model)
    Q = np.array(Q, float)
    data = mujoco.MjData(model)
    report = {}
    for side, pt in parts.items():
        if sides and side not in sides:
            continue
        if pt["gripper"] < 0 or not pt["arm_joints"]:
            report[side] = {"skipped": "no arm"}
            continue
        ch = TJ.Chain(model, pt["gripper"], pt["arm_joints"])
        T = len(Q)
        # the weld is position-only: the flange's orientation relative to the hand frame is whatever the home
        # pose has (a fixed mount rotation); the arm is fitted to hand pose x that mount rotation
        data.qpos[:] = model.key_qpos[0] if model.nkey else model.qpos0
        mujoco.mj_kinematics(model, data)
        Rg0 = data.xmat[pt["gripper"]].reshape(3, 3).copy(); Ry0 = data.xmat[pt["ee_body"]].reshape(3, 3).copy()
        R_mount = Ry0.T @ Rg0                     # gripper rotation expressed in the hand frame
        p_mount = Ry0.T @ (data.xpos[pt["gripper"]] - data.xpos[pt["ee_body"]])
        pos, rot = np.zeros((T, 3)), np.zeros((T, 3, 3))
        for t in range(T):
            data.qpos[:] = Q[t]
            mujoco.mj_kinematics(model, data)
            Ry = data.xmat[pt["ee_body"]].reshape(3, 3)
            pos[t] = data.xpos[pt["ee_body"]] + Ry @ p_mount; rot[t] = Ry @ R_mount
        seed = Q[0][pt["arm_qadr"]] if rest is None else np.asarray(rest[side], float)
        Qa, rp, rt, rr, resets = TJ.fit_arm_to_poses(model, ch, pos, rot, seed=seed, cfg=cfg)
        if max_arm_rate_deg_s:
            Qa = TJ.rate_limit(Qa, np.radians(float(max_arm_rate_deg_s)) / float(fps))
        if smooth_window and len(Qa) > smooth_window:
            Qa = TJ.savgol(Qa, int(smooth_window) | 1, 2)
        Qa = np.clip(Qa, ch.lo, ch.hi)
        for t in range(T):                                      # residuals of the delivered path
            rp[t], rt[t], rr[t] = TJ._arm_residual(model, data, ch, Qa[t], pos[t], rot[t])
        Q[:, pt["arm_qadr"]] = Qa
        step = np.degrees(np.abs(np.diff(Qa, axis=0)).max()) * fps if T > 1 else 0.0
        report[side] = {"pos_err_mm": {"max": round(float(rp.max()), 2), "mean": round(float(rp.mean()), 2)}, "peak_joint_speed_deg_s": round(float(step), 1),
                        "tilt_err_deg": {"max": round(float(rt.max()), 2), "mean": round(float(rt.mean()), 2)},
                        "roll_err_deg": {"max": round(float(rr.max()), 2)}, "branch_resets": resets,
                        "worst_frame": int(np.argmax(rp)),
                        "limit_margin_deg": round(float(np.degrees(min(np.min(Qa - ch.lo), np.min(ch.hi - Qa)))), 2)}
    return Q, report


# ----------------------------------------------------------------------------- reach of a plan
def keyframe_reach(model, keyframes, side, pos_tol_m=0.02, tilt_tol_deg=20.0):
    """Can the station's arm follow the hand through these keyframes? IK of the arm chain at every keyframe's
    wrist pose (position plus the home-pose mount rotation), warm-started along the sequence. Returns the worst
    residuals and the labels of the keyframes the arm cannot reach."""
    parts = robot_parts(model)
    if side not in parts or parts[side]["gripper"] < 0:
        return None
    pt = parts[side]
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0] if model.nkey else model.qpos0; mujoco.mj_kinematics(model, data)
    Rg0 = data.xmat[pt["gripper"]].reshape(3, 3).copy(); Ry0 = data.xmat[pt["ee_body"]].reshape(3, 3).copy()
    R_mount = Ry0.T @ Rg0; p_mount = Ry0.T @ (data.xpos[pt["gripper"]] - data.xpos[pt["ee_body"]])
    ch = TJ.Chain(model, pt["gripper"], pt["arm_joints"])
    home = data.qpos[pt["arm_qadr"]].copy()
    rng = np.random.default_rng(0)
    out = []
    q_prev = home.copy()
    # the keyframes and the midpoint of every transit between them: a hand that swings through an orientation the
    # wrist cannot produce fails between two reachable keyframes
    samples = []
    for i, kf in enumerate(keyframes):
        if i > 0 and float(kf.get("transit", 0.0)) > 0:
            qa, qb = np.asarray(keyframes[i - 1]["qpos"], float), np.asarray(kf["qpos"], float)
            samples.append({"keyframe": i - 0.5, "label": f"{kf.get('label', '')} (mid-transit)", "qpos": 0.5 * (qa + qb)})
        samples.append({"keyframe": i, "label": kf.get("label", ""), "qpos": np.asarray(kf["qpos"], float)})
    for kf in samples:
        i = kf["keyframe"]
        data.qpos[:] = np.asarray(kf["qpos"], float); mujoco.mj_kinematics(model, data)
        Ry = data.xmat[pt["ee_body"]].reshape(3, 3).copy(); py = data.xpos[pt["ee_body"]].copy()
        pt_, Rt = py + Ry @ p_mount, Ry @ R_mount
        best = None
        for k in range(8):
            data.qpos[pt["arm_qadr"]] = q_prev if k == 0 else np.clip(home + rng.normal(0, 0.4, len(home)), ch.lo, ch.hi)
            err = ch.solve(data, pt_, Rt, iters=100, rot_weight=0.3)
            if best is None or err[0] + 0.05 * err[1] < best[0] + 0.05 * best[1]:
                best = (err[0], err[1], data.qpos[pt["arm_qadr"]].copy())
            if best[0] <= pos_tol_m and best[1] <= np.radians(tilt_tol_deg):
                break
        q_prev = best[2]
        out.append({"keyframe": i, "label": kf["label"], "pos_err_mm": round(1000 * float(best[0]), 1), "rot_err_deg": round(float(np.degrees(best[1])), 1),
                    "ok": bool(best[0] <= pos_tol_m and best[1] <= np.radians(tilt_tol_deg))})
    kf_out = [o for o in out if float(o["keyframe"]).is_integer()]
    mid_out = [o for o in out if not float(o["keyframe"]).is_integer()]
    bad = [o for o in kf_out if not o["ok"]]
    bad_mid = [o for o in mid_out if not o["ok"]]
    # the keyframes decide; a transit whose midpoint the arm cannot hold exactly is a warning (the arm fill measures
    # the real error later: the first reach out of the home pose crosses the arm's near band, where a palm-down hand
    # is a centimetre or two off for a few frames)
    return {"ok": not bad, "worst_pos_err_mm": max(o["pos_err_mm"] for o in kf_out), "worst_rot_err_deg": max(o["rot_err_deg"] for o in kf_out),
            "unreachable": [{"keyframe": o["keyframe"], "label": o["label"], "pos_err_mm": o["pos_err_mm"], "rot_err_deg": o["rot_err_deg"]} for o in bad][:8],
            "transit_warnings": [{"keyframe": o["keyframe"], "label": o["label"], "pos_err_mm": o["pos_err_mm"], "rot_err_deg": o["rot_err_deg"]} for o in bad_mid][:8],
            "n_keyframes": len(keyframes), "n_samples": len(out)}


def insert_reach_vias(model, keyframes, side, pos_tol_m=0.02, tilt_tol_deg=20.0, max_vias=3):
    """Route transits the arm cannot follow through a via keyframe. For each transit whose midpoint the arm cannot hold
    (keyframe_reach's transit warnings), try a few intermediate wrist poses (the midpoint raised, the midpoint with the
    orientation of either end, both) and insert the one with the smallest residual, splitting the transit time. The
    first reach out of the home pose crosses the arm's near band this way instead of dragging the arm behind the hand.
    Returns (keyframes, inserted [...])."""
    from tool_library.common import plan as P
    from tool_library.common import hand as H
    rigs = P.scene_rigs(model)
    if side not in rigs:
        return keyframes, []
    rig = rigs[side]
    wq = [rig.wrist_qpos[k] for k in ("x", "y", "z", "rx", "ry", "rz")]
    rep = keyframe_reach(model, keyframes, side, pos_tol_m, tilt_tol_deg)
    if not rep or not rep.get("transit_warnings"):
        return keyframes, []
    out = list(keyframes); inserted = []
    for w in sorted(rep["transit_warnings"], key=lambda x: -x["pos_err_mm"])[:max_vias]:
        i = int(np.ceil(float(w["keyframe"])))             # the transit ends at keyframe i
        j = next((k for k, kf in enumerate(out) if kf is keyframes[i]), None)
        if j is None or j == 0:
            continue
        qa, qb = np.asarray(out[j - 1]["qpos"], float), np.asarray(out[j]["qpos"], float)
        pa, pb = qa[wq[:3]], qb[wq[:3]]; ea, eb = qa[wq[3:]], qb[wq[3:]]
        mid = 0.5 * (qa + qb)
        cands = []
        for dz in (0.0, 0.06, 0.12):
            for eul, tag in ((None, "slerp"), (ea, "start orientation"), (eb, "end orientation")):
                q = mid.copy(); q[wq[:3]] = 0.5 * (pa + pb) + np.array([0, 0, dz])
                if eul is not None:
                    q[wq[3:]] = eul
                cands.append((q, f"{tag}, +{dz:.2f} m"))
        best = None
        for q, tag in cands:
            mini = [dict(out[j - 1]), {"qpos": q.tolist(), "transit": 1.0, "hold": 0.0, "label": out[j]["label"]}, dict(out[j])]
            r = keyframe_reach(model, mini, side, pos_tol_m, tilt_tol_deg)
            worst = max([r["worst_pos_err_mm"]] + [t["pos_err_mm"] for t in r.get("transit_warnings", [])])
            if best is None or worst < best[0]:
                best = (worst, q, tag)
        if best is None or best[0] >= w["pos_err_mm"]:
            continue
        t = float(out[j].get("transit", 0.0))
        via = {"qpos": best[1].tolist(), "transit": 0.5 * t, "hold": 0.0, "label": out[j]["label"], "via": True}
        out[j]["transit"] = 0.5 * t
        out.insert(j, via)
        inserted.append({"before": out[j + 1]["label"], "how": best[2], "worst_mm_after": round(best[0], 1), "worst_mm_before": w["pos_err_mm"]})
    return out, inserted


# ----------------------------------------------------------------------------- measurements on an episode
def body_track(model, Q, body, stride=1):
    bid = body if isinstance(body, int) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"no body {body!r}")
    data = mujoco.MjData(model)
    idx = list(range(0, len(Q), stride))
    pos, rot = np.zeros((len(idx), 3)), np.zeros((len(idx), 3, 3))
    for i, t in enumerate(idx):
        data.qpos[:] = Q[t]; mujoco.mj_kinematics(model, data)
        pos[i] = data.xpos[bid]; rot[i] = data.xmat[bid].reshape(3, 3)
    return pos, rot


def subtree_bodies(model, bid):
    sub = {int(bid)}; changed = True
    while changed:
        changed = False
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in sub and b not in sub:
                sub.add(b); changed = True
    return sub


# a pinch on a cable link (a ball-jointed chain: composite or cable-plugin ropes) straddles its neighbours as soon as the
# cable bends: the link counts as held through its parents within this many links (its children are in its subtree)
CABLE_LINK_SPAN = 2


def held_bodies(model, bid, span=CABLE_LINK_SPAN):
    """The bodies a hand holding body `bid` may touch: its subtree and, for a link of a ball-jointed chain (a cable),
    also its parent links within `span`. Hinge or slide parts (a lid, a drawer) get their subtree only."""
    sub = subtree_bodies(model, bid)
    x = int(bid)
    for _ in range(int(span)):
        if int(model.body_jntnum[x]) != 1 or int(model.jnt_type[int(model.body_jntadr[x])]) != int(mujoco.mjtJoint.mjJNT_BALL):
            break
        par = int(model.body_parentid[x])
        if par <= 0:
            break
        sub.add(par); x = par
    return sub


def contacts_at(model, data, bodies_a, bodies_b):
    """Contacts between two body sets at the current data (after mj_forward)."""
    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (b1 in bodies_a and b2 in bodies_b) or (b2 in bodies_a and b1 in bodies_b):
            out.append((i, float(c.dist)))
    return out


def hand_contacts(model, data, rig, body):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return contacts_at(model, data, rig.body_ids, subtree_bodies(model, bid))


def lowest_point(model, data, bodies):
    """Lowest collision surface in world z, using actual transformed geometry.

    ``geom_rbound`` is a bounding sphere, so subtracting it from the centre
    reports fictitious penetration for wide or tilted objects and can make a
    floating object appear supported. Share the exact placement extent helper.
    """
    return bodies_extent(model, data, bodies)[0]


# ----------------------------------------------------------------------------- two-hand labels
# A frame where both hands act at once (a plan entry with motion 'both') carries both phases in one label:
# 'left: lift box_a | right: lift box_b'. Every reader that needs one hand's phase takes its part.
def combine_label(by_side, order=SIDES):
    """One label for a frame where both hands act: their '<side>: <phase>' labels joined by ' | '."""
    return " | ".join(str(by_side[s]) for s in order if s in by_side)


def split_label(label):
    """{side: '<side>: <phase>'} of a label; a one-hand label gives its own side only."""
    out = {}
    for part in str(label).split(" | "):
        side = part.split(":", 1)[0].strip()
        if side in SIDES:
            out[side] = part
    return out


def side_view(labels, side):
    """The frame labels as one hand sees them: its own part of every two-hand label; other labels unchanged."""
    return [split_label(l).get(side, str(l)) if " | " in str(l) else str(l) for l in labels]


# ----------------------------------------------------------------------------- exact geometry (placement)
# rbound is a bounding SPHERE: on a flat 2x2 stud brick it put the top 3-4 cm too high and the bottom 2 cm too low, so a
# brick placed `on` another hovered 5.7 cm above it. Placement reads the geoms' real extents instead.
def collision_geoms(model, bodies):
    """Geoms of the body set that take part in contacts (contype/conaffinity set, or named in an explicit pair),
    planes excluded; all of the set's non-plane geoms when none collides (a visual-only asset)."""
    bodies = {int(b) for b in bodies}
    paired = {int(g) for g in model.pair_geom1} | {int(g) for g in model.pair_geom2}
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies and int(model.geom_type[g]) != plane]
    col = [g for g in gs if model.geom_contype[g] or model.geom_conaffinity[g] or g in paired]
    return col or gs


def geom_extent(model, data, g, u):
    """(lowest, highest) of geom g along the world unit vector u at the current data (after kinematics): exact for
    spheres, capsules, cylinders, boxes and ellipsoids in any orientation and for meshes (their vertices); other types
    fall back to the bounding sphere."""
    u = np.asarray(u, float)
    c = data.geom_xpos[g]; R = data.geom_xmat[g].reshape(3, 3); s = model.geom_size[g]
    a = R.T @ u                                   # u in the geom's frame
    t = int(model.geom_type[g]); T = mujoco.mjtGeom
    if t == int(T.mjGEOM_SPHERE):
        h = s[0]
    elif t == int(T.mjGEOM_CAPSULE):
        h = abs(a[2]) * s[1] + s[0]
    elif t == int(T.mjGEOM_CYLINDER):
        h = abs(a[2]) * s[1] + s[0] * np.sqrt(max(0.0, 1.0 - a[2] ** 2))
    elif t == int(T.mjGEOM_BOX):
        h = float(np.abs(a) @ s)
    elif t == int(T.mjGEOM_ELLIPSOID):
        h = float(np.sqrt(((a * s) ** 2).sum()))
    elif t == int(T.mjGEOM_MESH) and int(model.geom_dataid[g]) >= 0:
        mid = int(model.geom_dataid[g]); v0 = int(model.mesh_vertadr[mid]); n = int(model.mesh_vertnum[mid])
        z = model.mesh_vert[v0:v0 + n] @ a
        base = float(c @ u)
        return base + float(z.min()), base + float(z.max())
    else:
        h = float(model.geom_rbound[g])
    base = float(c @ u)
    return base - float(h), base + float(h)


def bodies_extent(model, data, bodies, u=(0.0, 0.0, 1.0), geoms=None):
    """(lowest, highest) of the body set's collision geoms along u (default world z), exact (geom_extent)."""
    gs = collision_geoms(model, bodies) if geoms is None else list(geoms)
    if not gs:
        return np.inf, -np.inf
    ext = [geom_extent(model, data, g, u) for g in gs]
    return min(e[0] for e in ext), max(e[1] for e in ext)


def bodies_rect(model, data, bodies):
    """World xy rectangle (xmin, xmax, ymin, ymax) the body set's collision geoms cover from above."""
    x = bodies_extent(model, data, bodies, (1.0, 0.0, 0.0)); y = bodies_extent(model, data, bodies, (0.0, 1.0, 0.0))
    return float(x[0]), float(x[1]), float(y[0]), float(y[1])


def _ray_down(model, data, g, pnt):
    """Distance along -z from pnt to geom g, or -1 (no hit)."""
    vec = np.array([0.0, 0.0, -1.0])
    t = int(model.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
        return float(mujoco.mj_rayMesh(model, data, g, pnt, vec))
    if t in (int(mujoco.mjtGeom.mjGEOM_HFIELD), int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_SDF)):
        return -1.0
    return float(mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], pnt, vec, t))


def support_top(model, data, bodies, rect=None, grid_m=0.004, max_n=15):
    """Highest point of the body set's collision surface under a world xy rectangle (xmin, xmax, ymin, ymax): rays
    cast straight down on a grid over the rectangle (spacing grid_m, at most max_n x max_n) onto the set's own geoms,
    plus the exact tops of its primitive geoms whose centres lie inside the rectangle (small features a ray grid can
    pass between: the studs of a brick). Without a rectangle, or when nothing of the set lies under it, the set's
    exact highest point. Returns (z, info {method, n_rays, n_hits, n_features})."""
    gs = collision_geoms(model, bodies)
    if not gs:
        planes = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in {int(b) for b in bodies}
                  and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE)]
        if planes:                                  # a floor plane: its height
            return max(float(data.geom_xpos[g][2]) for g in planes), {"method": "plane", "n_rays": 0, "n_hits": 0, "n_features": 0}
        raise ValueError("support_top: the body set has no geoms")
    ext_z = [geom_extent(model, data, g, (0.0, 0.0, 1.0)) for g in gs]
    whole = max(e[1] for e in ext_z)
    if rect is None:
        return whole, {"method": "whole", "n_rays": 0, "n_hits": 0, "n_features": 0}
    x0, x1, y0, y1 = (float(v) for v in rect)
    pad = 0.001
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    nx = int(np.clip(np.ceil((x1 - x0) / grid_m) + 1, 3, max_n)); ny = int(np.clip(np.ceil((y1 - y0) / grid_m) + 1, 3, max_n))
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    xy_ext = {g: (geom_extent(model, data, g, (1.0, 0.0, 0.0)), geom_extent(model, data, g, (0.0, 1.0, 0.0))) for g in gs}
    z_start = whole + 0.05
    best, hits = -np.inf, 0
    prim = {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}
    for x in xs:
        for y in ys:
            pnt = np.array([x, y, z_start])
            top = -np.inf
            for g in gs:
                (ax0, ax1), (ay0, ay1) = xy_ext[g]
                if x < ax0 or x > ax1 or y < ay0 or y > ay1:
                    continue
                dist = _ray_down(model, data, g, pnt)
                if dist >= 0:
                    top = max(top, z_start - dist)
            if np.isfinite(top):
                hits += 1; best = max(best, top)
    feat = 0
    for g, (lo, hi) in zip(gs, ext_z):
        cx, cy = data.geom_xpos[g][0], data.geom_xpos[g][1]
        if int(model.geom_type[g]) in prim and x0 <= cx <= x1 and y0 <= cy <= y1:
            feat += 1; best = max(best, hi)
    if not np.isfinite(best):
        return whole, {"method": "whole (nothing of the support under the footprint)", "n_rays": nx * ny, "n_hits": 0, "n_features": 0}
    return float(best), {"method": "footprint", "n_rays": nx * ny, "n_hits": hits, "n_features": feat}


def hand_hand_clearance(model, Q, stride=2, cutoff_m=0.05, frames=None):
    """Smallest distance between the left and the right hand's collision geoms over frames of Q (kinematics only):
    {min_m, frame, geoms}; min_m is cutoff_m when the hands never come within it, negative when they overlap. None
    when the scene has not two hands."""
    left = [g for g in range(model.ngeom) if "_collision_hand_left_" in (model.geom(g).name or "")]
    right = [g for g in range(model.ngeom) if "_collision_hand_right_" in (model.geom(g).name or "")]
    if not left or not right:
        return None
    Q = np.asarray(Q, float)
    data = mujoco.MjData(model)
    rl, rr = model.geom_rbound[left], model.geom_rbound[right]
    best = {"min_m": float(cutoff_m), "frame": None, "geoms": None}
    idx = list(range(0, len(Q), max(1, int(stride)))) if frames is None else list(frames)
    if len(Q) and (len(Q) - 1) not in idx and frames is None:
        idx.append(len(Q) - 1)
    for t in idx:
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        cl, cr = data.geom_xpos[left], data.geom_xpos[right]
        D = np.linalg.norm(cl[:, None, :] - cr[None, :, :], axis=2) - rl[:, None] - rr[None, :]
        for i, j in zip(*np.nonzero(D < float(cutoff_m))):
            dist = H.geom_distance(model, data, left[i], right[j], float(cutoff_m))      # false zeros removed
            if dist < best["min_m"]:
                best = {"min_m": dist, "frame": int(t), "geoms": [model.geom(left[i]).name, model.geom(right[j]).name]}
    best["min_m"] = round(best["min_m"], 5)
    return best


def frames_of(labels, prefix=None, subgoal=None, subgoals=None):
    if subgoals is not None and subgoal is not None:
        return [t for t, s in enumerate(subgoals) if int(s) == int(subgoal)]
    return [t for t, l in enumerate(labels) if prefix is None or str(l).startswith(prefix)]


# ----------------------------------------------------------------------------- execution model
# The physics every Stage 3 episode (and Stage 4's replay of it) runs in: the bench's own options (setup/scene.xml
# leaves the timestep at MuJoCo's 0.002 and integrates implicitfast) plus what build_execution_model adds. A mechanism
# test that is meant to describe the episode runs with these (plug_port_retention_test).
EXECUTION_PHYSICS = {"timestep": 0.002, "integrator": "implicitfast", "cone": "elliptic", "noslip_iterations": 5, "impratio": 5.0}


def apply_execution_options(spec):
    """The contact options build_execution_model gives a scene (MjSpec or MjModel): elliptic cones, at least
    EXECUTION_PHYSICS noslip iterations and impratio. Timestep and integrator stay the scene's."""
    opt = spec.option if hasattr(spec, "option") else spec.opt
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.noslip_iterations = max(int(opt.noslip_iterations), int(EXECUTION_PHYSICS["noslip_iterations"]))
    if opt.impratio < EXECUTION_PHYSICS["impratio"]:
        opt.impratio = float(EXECUTION_PHYSICS["impratio"])
    return spec


def build_execution_model(scene_xml, pair_solref=(0.006, 1.0), friction=None, hand_hand=True, out_xml=None,
                          scene_recon_json=None):
    """The bench scene with explicit contact pairs between every hand collision geom and every object geom
    (the hands ship with contacts off and touch only what they are paired with), plus left-right hand pairs.
    Returns (model, n_pairs). With out_xml the paired scene is written for re-simulation."""
    from pathlib import Path
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    from tool_library.common.contact_defaults import apply_scene_defaults
    apply_scene_defaults(spec, scene_xml, scene_recon_json)
    from tool_library.common import rigid_robot as RR
    rigid_input = any((t.name or "") == "dexagent_robot_schema" for t in spec.texts)
    if friction is None:                        # the grasp tools' contact (hand.HAND_PAIR_FRICTION): a pose behaves the same
        from tool_library.common.hand import HAND_PAIR_FRICTION
        friction = HAND_PAIR_FRICTION
    from tool_library.common.handgeom import add_hand_hulls
    add_hand_hulls(spec)                    # scenes written before the hulls existed get them here
    # the arm carries the hand's own weight: without this the wrist servo sags 1.7 mm under the hand's mass, the
    # executed hand leaves the flange of the planned arm and the exact arm fit chases that sag near singularities
    for b in spec.bodies:
        nm = b.name or ""
        if not rigid_input and nm.startswith("robot_") and "_arm_" not in nm:
            b.gravcomp = 1.0
    from tool_library.common.hand import hand_may_touch, _opted_out
    hands = {"left": [], "right": []}
    objs, table = [], []
    n_unnamed = {}
    for g in spec.geoms:
        nm = g.name or ""
        body = g.parent.name if g.parent is not None else ""
        if "_collision_hand_" in nm:
            side = "left" if "_collision_hand_left_" in nm else "right"
            hands[side].append(nm)
        elif body and not body.startswith("robot_") and body not in ("world", "table", "camrig"):
            # the hand touches what the geom's own collision bits allow (a magnet shell in its own bit is not solid to
            # the fingers) unless the geom opts out ('nohand'); an unnamed solid geom gets a name so it is paired too
            cls = g.classname.name if getattr(g, "classname", None) is not None else ""
            if not hand_may_touch(g.contype, g.conaffinity) or _opted_out(nm, cls):
                continue
            if not nm:
                k = n_unnamed.get(body, 0); n_unnamed[body] = k + 1
                g.name = nm = f"{body}__geom{k}"
            objs.append(nm)
        elif body == "table" and nm and (g.contype or g.conaffinity):
            table.append(nm)
        elif body == "world" and nm and g.type == mujoco.mjtGeom.mjGEOM_PLANE and (g.contype or g.conaffinity):
            table.append(nm)
    # The floating scene is a planning representation. Remove its obsolete
    # position-only welds; rigid_robot replaces them with a fixed body tree
    # before any arm-equipped model executes physical controls.
    for e in list(spec.equalities):
        b1, b2 = e.name1 or "", e.name2 or ""
        if ("_arm_" in b1 and b2.startswith("robot_")) or ("_arm_" in b2 and b1.startswith("robot_")):
            spec.delete(e)
    existing = {(p.geomname1, p.geomname2) for p in spec.pairs} | {(p.geomname2, p.geomname1) for p in spec.pairs}
    n = 0
    def add(a, b):
        nonlocal n
        if (a, b) in existing:
            return
        pr = spec.add_pair(); pr.name = f"exec__{a}__{b}"; pr.geomname1, pr.geomname2 = a, b
        pr.solref = list(pair_solref); pr.solimp = [0.99, 0.999, 0.001, 0.5, 2]
        pr.friction = [friction[0], friction[0], friction[1], friction[2], friction[2]]; pr.condim = 6
        existing.add((a, b)); n += 1
    for side, hg in hands.items():
        for h in hg:
            for o in objs:
                add(h, o)
            for o in table:                  # the hands rest on, and cannot sink into, the table
                add(h, o)
    if hand_hand:
        for h in hands["left"]:
            for k in hands["right"]:
                add(h, k)
    # contact physics as the grasp tools have it: elliptic friction cones (with pyramidal cones the same grasps
    # seat with no finger force at all) and a few no-slip iterations so a held object does not creep out of a
    # clamp under sustained load (measured: a cup slid 47 mm during a 10 cm lift without them)
    # a seated lid with a strap pressed over its catch and a hand on it overflowed the default arena (17 MB) and
    # stopped the simulation; the hand scenes already run with 256 MB
    if int(spec.memory) < 256 * 2 ** 20:
        spec.memory = 256 * 2 ** 20
    apply_execution_options(spec)
    H.add_hand_self_contacts(spec)
    model = spec.compile()
    RR.register_model(model, spec, Path(scene_xml).resolve().parent)
    if out_xml:
        from pathlib import Path
        Path(out_xml).write_text(RR._SPECS[model].to_xml())
    return model, n
