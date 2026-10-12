"""Keyframe planning on a bench scene: the hands' world poses, rigid attachment of a grasped object or
part, and the per-motion keyframe recipes (pick, carry, place, joint motions, screw bites, press, tilt).
A keyframe is a full qpos of the scene with hold/transit frame counts and a label; chain_subgoals or
rollout_keyframes_physics turn a list of them into a trajectory."""
from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):  # headless offscreen rendering; MuJoCo rejects "egl" elsewhere
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import hand as H

SIDES = ("left", "right")


def scene_rigs(model, prefix_fmt="robot_{side}_"):
    """{side: HandRig} for the hands found in a scene (bench naming robot_<side>_<side>_pos_x ...)."""
    rigs = {}
    for side in SIDES:
        prefix = prefix_fmt.format(side=side)
        if (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{side}_pos_x") >= 0
                or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{side}_base_yaw") >= 0):
            rigs[side] = H.HandRig.build(model, prefix, side)
    if not rigs:
        raise ValueError("no hand with pos_x..rot_z wrist joints found in the scene")
    return rigs


def root_frame(model, data, rig):
    """World pose of the frame the wrist joints are expressed in (the hand root body's parent frame)."""
    b = rig.root_body
    # the root body carries pos_x: its own frame before its joints = parent's frame * body offset
    parent = int(model.body_parentid[b])
    Rp = data.xmat[parent].reshape(3, 3) if parent > 0 else np.eye(3)
    pp = data.xpos[parent] if parent > 0 else np.zeros(3)
    Rb = Rot.from_quat([model.body_quat[b][1], model.body_quat[b][2], model.body_quat[b][3], model.body_quat[b][0]]).as_matrix()
    return pp + Rp @ model.body_pos[b], Rp @ Rb


def wrist_world(model, data, rig):
    """(p, R) of the hand's wrist frame in the world, from its six joints and the root frame."""
    if rig.rigid_wrist:
        return data.xpos[rig.root_body].copy(), data.xmat[rig.root_body].reshape(3, 3).copy()
    p0, R0 = root_frame(model, data, rig)
    q = np.array([data.qpos[rig.wrist_qpos[k]] for k in ("x", "y", "z", "rx", "ry", "rz")])
    return p0 + R0 @ q[:3], R0 @ H.euler_to_mat(q[3:])


def wrist_joints_from_world(model, data, rig, p, R, prev_euler=None):
    """Six wrist joint values that place the wrist frame at world (p, R)."""
    p0, R0 = root_frame(model, data, rig)
    e = H.mat_to_euler(R0.T @ np.asarray(R, float))
    if prev_euler is not None:
        e = H.nearest_euler_branch(e, np.asarray(prev_euler, float))
    return np.concatenate([R0.T @ (np.asarray(p, float) - p0), e])


def set_wrist(model, data, rig, p, R, prev_euler=None):
    q = wrist_joints_from_world(model, data, rig, p, R, prev_euler)
    for k, v in zip(("x", "y", "z", "rx", "ry", "rz"), q):
        data.qpos[rig.wrist_qpos[k]] = float(v)
    return q


def body_pose(model, data, name):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise ValueError(f"no body named {name!r}")
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()


def free_joint_of(model, body_name):
    """The free joint on body_name or on an ancestor, else None."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    while b > 0:
        for j in range(model.njnt):
            if int(model.jnt_bodyid[j]) == b and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return j
        b = int(model.body_parentid[b])
    return None


def set_free_body(model, data, jid, p, R):
    adr = int(model.jnt_qposadr[jid])
    q = Rot.from_matrix(R).as_quat()
    data.qpos[adr:adr + 3] = p; data.qpos[adr + 3:adr + 7] = [q[3], q[0], q[1], q[2]]


def relative_transform(pa, Ra, pb, Rb):
    """T_a^-1 * T_b: pose of b expressed in frame a."""
    return Ra.T @ (pb - pa), Ra.T @ Rb


def compose(pa, Ra, p_rel, R_rel):
    return pa + Ra @ p_rel, Ra @ R_rel


def pose_from_dict(pose):
    return np.asarray(pose["pos"], float), H.euler_to_mat(np.asarray(pose["euler"], float))


def retreat_axis(pose, R):
    """Unit world vector along which the open hand backs away from the object before an approach: the
    pose's `approach` when a grasp skill recorded one (outward from the object), else minus the hand's local
    z (the palm faces +z), kept at least slightly upward so no approach starts below the table."""
    a = pose.get("approach")
    if a is not None:
        a = np.asarray(a, float)
        return a / max(np.linalg.norm(a), 1e-9)
    v = -np.asarray(R, float)[:, 2]
    if v[2] < 0.2:
        v = v - np.array([0, 0, v[2] - 0.2])
    return v / max(np.linalg.norm(v), 1e-9)


class Planner:
    """Builds keyframes on a scene. State: a working MjData whose qpos is the current keyframe; the
    attachments of carried objects to a hand; the list of keyframes written so far."""

    def __init__(self, model, qpos0=None, fps=30.0):
        self.model = model
        self.data = mujoco.MjData(model)
        if qpos0 is not None:
            self.data.qpos[:] = np.asarray(qpos0, float)
        elif model.nkey:
            mujoco.mj_resetDataKeyframe(model, self.data, 0)
        mujoco.mj_forward(model, self.data)
        self.rigs = scene_rigs(model)
        self.fps = float(fps)
        self.keyframes = []
        self.phases = []
        self.attached = {}          # side -> {"joint": free jid, "body": name, "rel": (p_rel, R_rel)}
        self.extra = {}             # side -> [attachments of the further objects the same hand holds (a second cup)]
        self.coupled = {}           # side -> [other sides that hold the same object and must move with it]
        self.prev_euler = {s: np.array([self.data.qpos[r.wrist_qpos[k]] for k in ("rx", "ry", "rz")]) for s, r in self.rigs.items()}

    # -------------------------------------------------------------- bookkeeping
    def frames(self, seconds):
        return max(1, int(round(float(seconds) * self.fps)))

    def push(self, label, transit_s=0.0, hold_s=0.0, side=None, objects=(), anchor=None):
        mujoco.mj_forward(self.model, self.data)
        kf = {"qpos": self.data.qpos.copy().tolist(), "transit": self.frames(transit_s) if self.keyframes else 0,
              "hold": max(1, self.frames(hold_s)), "label": label}
        self.keyframes.append(kf)
        self.phases.append({"label": label, "hand": side, "objects": list(objects), "anchor": anchor, "keyframe": len(self.keyframes) - 1})
        return kf

    def wrist(self, side):
        return wrist_world(self.model, self.data, self.rigs[side])

    def fingers(self, side, targets):
        self.rigs[side].set_fingers(self.data, {k: float(v) for k, v in targets.items() if k in self.rigs[side].finger_qpos})

    def move_wrist(self, side, p, R):
        q = set_wrist(self.model, self.data, self.rigs[side], p, R, self.prev_euler[side])
        self.prev_euler[side] = q[3:]
        self._carry(side)
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- attachment
    def attach(self, side, body):
        jid = free_joint_of(self.model, body)
        if jid is None:
            raise ValueError(f"{body!r} has no free joint: a kinematic carry needs a free body (parts on joints are moved with move_joint)")
        # a part on a hinge or slide below that free root (a lid, a drawer) is not carried with its whole object
        root = int(self.model.jnt_bodyid[jid])
        b = self.model.body(body).id if isinstance(body, str) else int(body)
        while b > 0 and b != root:
            if any(int(self.model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
                   for j in range(int(self.model.body_jntadr[b]), int(self.model.body_jntadr[b]) + int(self.model.body_jntnum[b]))):
                raise ValueError(f"{body!r} turns or slides on a joint of {self.model.body(root).name!r}: it moves with move_joint, "
                                 f"not as a carried body")
            b = int(self.model.body_parentid[b])
        pw, Rw = self.wrist(side)
        pb, Rb = body_pose(self.model, self.data, self.model.body(int(self.model.jnt_bodyid[jid])).name)
        self.attached[side] = {"joint": int(jid), "body": body, "root_body": self.model.body(int(self.model.jnt_bodyid[jid])).name,
                               "rel": relative_transform(pw, Rw, pb, Rb)}

    def attach_extra(self, side, body):
        """One more object carried by the same hand (bimanual.grasp_more): it rides the wrist like the primary."""
        prev = self.attached.pop(side, None)
        try:
            self.attach(side, body)
            ex = self.attached.pop(side)
        finally:
            if prev is not None:
                self.attached[side] = prev
        self.extra.setdefault(side, [])
        self.extra[side] = [e for e in self.extra[side] if e["root_body"] != ex["root_body"]] + [ex]

    def detach_body(self, side, body):
        """Let go of one of the objects a hand carries (the primary or an extra)."""
        a = self.attached.get(side)
        if a and a["body"] == body:
            self.attached.pop(side, None)
        self.extra[side] = [e for e in self.extra.get(side, []) if e["body"] != body]

    def detach(self, side):
        self.attached.pop(side, None)
        self.extra.pop(side, None)

    def _carry(self, side):
        for ex in self.extra.get(side, []):          # further objects in the same hand ride the wrist too
            pw, Rw = wrist_world(self.model, self.data, self.rigs[side])
            set_free_body(self.model, self.data, ex["joint"], *compose(pw, Rw, *ex["rel"]))
        a = self.attached.get(side)
        if a:
            pw, Rw = wrist_world(self.model, self.data, self.rigs[side])
            p, R = compose(pw, Rw, *a["rel"])
            mujoco.mj_forward(self.model, self.data)
            p0, R0 = body_pose(self.model, self.data, a["root_body"])
            set_free_body(self.model, self.data, a["joint"], p, R)
            # the other hands on the same object move with it (a two-handed carry or lid flip)
            for co in self.coupled.get(side, []):
                if co == side or co not in self.rigs:
                    continue
                pc, Rc = wrist_world(self.model, self.data, self.rigs[co])
                dR = R @ R0.T
                q = set_wrist(self.model, self.data, self.rigs[co], p + dR @ (pc - p0), dR @ Rc, self.prev_euler[co])
                self.prev_euler[co] = q[3:]

    # -------------------------------------------------------------- recipes (each returns the labels it added)
    def grasp_lift(self, side, pose, object_body, approach_m=0.08, open_frac=0.55, lift_m=0.08, label=None,
                   t_reach=1.2, t_descend=0.8, t_close=0.5, t_lift=0.8):
        rig = self.rigs[side]
        p, R = pose_from_dict(pose)
        closed = {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range}
        grip = list(pose.get("grip_fingers") or [])
        opened = H.open_targets(rig, closed, open_frac, curl=[f for f in H.FINGERS if grip and f not in grip])
        press = H.press_targets(rig, closed, grip)
        a = retreat_axis(pose, R)
        pre = p + a * float(approach_m)
        lab = label or f"{side}: reach for {object_body}"
        self.fingers(side, opened); self.move_wrist(side, pre, R); self.push(lab, t_reach, 0.1, side, [object_body], anchor=object_body)
        self.move_wrist(side, p, R); self.push(f"{side}: descend on {object_body}", t_descend, 0.1, side, [object_body], anchor=object_body)
        self.fingers(side, closed); self.push(f"{side}: close on {object_body}", t_close, 0.2, side, [object_body], anchor=object_body)
        self.fingers(side, press); self.push(f"{side}: grip {object_body}", 0.3, 0.3, side, [object_body], anchor=object_body)
        self.attach(side, object_body)
        self.move_wrist(side, p + np.array([0, 0, float(lift_m)]), R); self.push(f"{side}: lift {object_body}", t_lift, 0.2, side, [object_body], anchor=object_body)
        return {"closed": closed, "opened": opened, "grasp_pos": p.tolist()}

    def carry_to(self, side, target_pos, target_R=None, via_height=None, label=None, t=1.5, anchor=None):
        """Carry the attached object so that the body the subgoal names lands at target_pos (and target_R when given):
        the object's root for a whole object, the held segment for a rope."""
        a = self.attached.get(side)
        if not a:
            raise ValueError(f"{side} hand carries nothing: grasp_lift first")
        pw, Rw = self.wrist(side)
        p_rel, R_rel = a["rel"]
        pb_now, Rb_now = compose(pw, Rw, p_rel, R_rel)
        if a.get("body") and a["body"] != a.get("root_body"):
            # the subgoal names a part of a free tree (a rope segment, a sub-body): that part lands at the target,
            # the root keeps its offset from it (the outcome check measures the named body, not the root)
            mujoco.mj_forward(self.model, self.data)
            ph, Rh = body_pose(self.model, self.data, a["body"])
            q_rel = relative_transform(ph, Rh, pb_now, Rb_now)
            Rh_t = Rh if target_R is None else np.asarray(target_R, float)
            pb_t, Rb_t = compose(np.asarray(target_pos, float), Rh_t, *q_rel)
        else:
            Rb_t = Rb_now if target_R is None else np.asarray(target_R, float)
            pb_t = np.asarray(target_pos, float)
        # wrist pose that puts the object there: T_w = T_obj_t * T_rel^-1
        Rw_t = Rb_t @ R_rel.T; pw_t = pb_t - Rw_t @ p_rel
        lab = label or f"{side}: carry {a['body']}"
        if via_height is not None:
            mid = pw.copy(); mid[2] = max(mid[2], float(via_height)); self.move_wrist(side, mid, Rw)
            self.push(lab + " (up)", t * 0.3, 0.0, side, [a["body"]], anchor=anchor)
            mid2 = pw_t.copy(); mid2[2] = max(mid2[2], float(via_height)); self.move_wrist(side, mid2, Rw_t)
            self.push(lab, t * 0.5, 0.0, side, [a["body"]], anchor=anchor)
            self.move_wrist(side, pw_t, Rw_t); self.push(lab + " (down)", t * 0.3, 0.1, side, [a["body"]], anchor=anchor)
        else:
            self.move_wrist(side, pw_t, Rw_t); self.push(lab, t, 0.1, side, [a["body"]], anchor=anchor)
        return {"wrist_target": pw_t.tolist(), "object_target": pb_t.tolist()}

    def release(self, side, opened, retreat_m=0.08, label=None, t_open=0.5, t_retreat=0.8, retreat_dir=(0, 0, 1), anchor=None):
        """Open and back away. `anchor` is what the let-go belongs to for Stage 4 (the goal object, or None for
        the world); the released object itself is never the anchor of its own release."""
        a = self.attached.get(side)
        body = a["body"] if a else None
        self.fingers(side, opened); self.detach(side)
        self.push(label or f"{side}: let go" + (f" of {body}" if body else ""), t_open, 0.3, side, [body] if body else [], anchor=anchor)
        pw, Rw = self.wrist(side)
        d = np.asarray(retreat_dir, float); d /= max(np.linalg.norm(d), 1e-9)
        self.move_wrist(side, pw + d * float(retreat_m), Rw); self.push(f"{side}: retreat", t_retreat, 0.2, side, [body] if body else [], anchor=anchor)

    def move_joint(self, side, joint, target, pose=None, steps=None, label=None, t=1.5, object_body=None, sub_deg=10.0, sub_m=0.02, co_sides=(),
                   hand_follows="rigid", follow_point=None, grip_fingers=None):
        """Drive a hinge or slide joint of an object to `target` while the hand follows the moving part (the part's
        pose is read by forward kinematics at each step). When `pose` is given the hand first goes there (an open
        hand reaching the handle, then closing).

        `hand_follows`: 'rigid' = the wrist rides the part (a pinch that turns with a lid); 'level' = the wrist keeps
        its orientation while the point of contact follows the part: the centroid of the grip fingers' pads
        (`grip_fingers`, default the pose's, else every fingertip) or an explicit world `follow_point` rides the part
        rigidly and the wrist is placed so the pads stay there. Fingertips that lift a lid from below or hook a drawer
        work this way; an arm cannot roll its hand 100 degrees with a lid, and a level hand stays in reach."""
        m = self.model; d = self.data
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            raise ValueError(f"no joint {joint!r}")
        adr = int(m.jnt_qposadr[jid]); part = m.body(int(m.jnt_bodyid[jid])).name
        if pose is not None:
            rig = self.rigs[side]
            p, R = pose_from_dict(pose)
            closed = {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range}
            own_open = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
            opened = own_open or H.open_targets(rig, closed, float(pose.get("open_frac", 0.55)))
            a = retreat_axis(pose, R)
            self.fingers(side, opened); self.move_wrist(side, p + a * 0.06, R); self.push(f"{side}: reach for {part}", 2.0, 0.1, side, [object_body or part], anchor=object_body or part)
            self.move_wrist(side, p, R); self.push(f"{side}: descend on {part}", 0.6, 0.1, side, [object_body or part], anchor=object_body or part)
            self.fingers(side, closed); self.push(f"{side}: close on {part}", 0.4, 0.3, side, [object_body or part], anchor=object_body or part)
            if grip_fingers is None:
                grip_fingers = pose.get("grip_fingers")
        q0 = float(d.qpos[adr]); pw, Rw = self.wrist(side)
        if steps is None:
            hinge = int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE)
            steps = max(1, int(np.ceil(abs(float(target) - q0) / (np.radians(sub_deg) if hinge else sub_m))))
        pp0, Rp0 = body_pose(m, d, part)
        level = hand_follows == "level"
        if level:
            # the point that rides the part: the pads of the grip fingers (or all fingertips), or a given point
            rig = self.rigs[side]; mujoco.mj_forward(m, d)
            if follow_point is not None:
                c0 = np.asarray(follow_point, float)
            else:
                fins = [f for f in (grip_fingers or rig.fingertip_sites) if f in rig.fingertip_sites]
                c0 = np.mean([d.site_xpos[rig.fingertip_sites[f]] for f in fins], axis=0)
            off_w = c0 - pw                              # constant while the hand keeps its orientation
            p_rel, R_rel = relative_transform(pp0, Rp0, c0, Rw)
        else:
            p_rel, R_rel = relative_transform(pp0, Rp0, pw, Rw)
        co_rel, co_off, co_R0 = {}, {}, {}
        for co in co_sides or ():                      # other hands on the part ride it the same way
            if co != side and co in self.rigs:
                pc, Rc = self.wrist(co); co_R0[co] = Rc.copy()
                if level:
                    crig = self.rigs[co]; cc = np.mean([d.site_xpos[s] for s in crig.fingertip_sites.values()], axis=0)
                    co_off[co] = cc - pc; co_rel[co] = relative_transform(pp0, Rp0, cc, Rc)
                else:
                    co_rel[co] = relative_transform(pp0, Rp0, pc, Rc)
        lab = label or f"{side}: move {joint}"
        Rw0 = Rw.copy()
        for i in range(1, int(steps) + 1):
            d.qpos[adr] = q0 + (float(target) - q0) * i / steps
            mujoco.mj_forward(m, d)
            pp, Rp = body_pose(m, d, part)
            p_new, R_new = compose(pp, Rp, p_rel, R_rel)
            if level:
                p_new, R_new = p_new - off_w, Rw0
            self.move_wrist(side, p_new, R_new)
            for co, rel in co_rel.items():
                pc_new, Rc_new = compose(pp, Rp, *rel)
                if level:
                    pc_new, Rc_new = pc_new - co_off[co], co_R0[co]
                q = set_wrist(m, d, self.rigs[co], pc_new, Rc_new, self.prev_euler[co]); self.prev_euler[co] = q[3:]
            self.push(lab, t / steps, 0.0 if i < steps else 0.2, side, [object_body or part], anchor=object_body or part)
        return {"joint": joint, "from": q0, "to": float(target), "part": part, "hand_follows": hand_follows}

    def screw(self, side, yaw_joint, lift_joint, turns, pitch_m, pose=None, bite_deg=90.0, unscrew=True, label=None,
              t_bite=0.8, t_regrasp=0.6, object_body=None, sub_deg=15.0, regrasp_before=False, bite_index=1, regrasp_pose=None,
              regrasp_lift_m=0.0):
        """Turn a cap through `turns` in bites of `bite_deg`: the hand turns with the cap about the yaw joint's
        axis while the lift joint follows the thread coupling; between bites it opens, swings back around the
        axis and closes again. With `pose` the hand first approaches and closes; with `regrasp_before` (a bite
        planned from the state the previous one reached) it starts with the open swing-back and the re-close.
        The direction comes from the thread coupling lift = k * yaw: unscrewing raises the cap."""
        m = self.model; d = self.data
        jy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, yaw_joint); jl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, lift_joint)
        if jy < 0 or jl < 0:
            raise ValueError("yaw_joint / lift_joint not in the scene")
        ay, al = int(m.jnt_qposadr[jy]), int(m.jnt_qposadr[jl]); part = m.body(int(m.jnt_bodyid[jy])).name
        rig = self.rigs[side]
        who = object_body or part
        k = None
        for e in range(m.neq):
            if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT) and {int(m.eq_obj1id[e]), int(m.eq_obj2id[e])} == {jy, jl}:
                c = float(m.eq_data[e][1])
                k = c if int(m.eq_obj1id[e]) == jl else (1.0 / c if abs(c) > 1e-12 else None)
        if k is None:
            k = float(pitch_m) / (2 * np.pi)
        sign = (1.0 if k > 0 else -1.0) * (1.0 if unscrew else -1.0)
        bite = np.radians(float(bite_deg)); sub = np.radians(float(sub_deg))
        lab = label or f"{side}: {'unscrew' if unscrew else 'screw'} {who}"

        def lift_off(dist, label_):
            """Move the open hand along the part's axis (off the cap before the turn-back, onto it after): fingers that
            stay on the cap's top edge while turning back drag it back with them."""
            if abs(float(dist)) < 1e-6:
                return
            pp, Rp = body_pose(m, d, part)
            pw, Rw = self.wrist(side)
            self.move_wrist(side, pw + Rp[:, 2] * float(dist), Rw)
            self.push(label_, 0.25, 0.05, side, [who], anchor=who)

        def swing(angle, label_):
            """Rotate the wrist about the part's axis (through its origin) by `angle` in small steps."""
            pp, Rp = body_pose(m, d, part); axis = Rp[:, 2]; n_sub = max(1, int(np.ceil(abs(angle) / sub)))
            Rd = Rot.from_rotvec(angle / n_sub * axis).as_matrix()
            for i in range(n_sub):
                pw, Rw = self.wrist(side)
                self.move_wrist(side, pp + Rd @ (pw - pp), Rd @ Rw)
                self.push(label_, t_regrasp / n_sub, 0.1 if i == n_sub - 1 else 0.0, side, [who], anchor=who)

        if pose is not None:
            p, R = pose_from_dict(pose)
            closed = {kk: float(v) for kk, v in pose["finger_targets"].items() if kk in rig.finger_range}
            own_open = {kk: float(v) for kk, v in (pose.get("finger_targets_open") or {}).items() if kk in rig.finger_range}
            opened = own_open or H.open_targets(rig, closed, float(pose.get("open_frac", 0.55)))    # the pose's own opening, when it has one
            squeezed = {kk: float(v) for kk, v in (pose.get("finger_targets_squeezed") or {}).items() if kk in rig.finger_range}
            press = H.press_targets(rig, closed, list(pose.get("grip_fingers") or []), squeezed=squeezed)
            a = retreat_axis(pose, R)
            self.fingers(side, opened); self.move_wrist(side, p + a * 0.06, R); self.push(f"{side}: reach for {who}", 2.0, 0.1, side, [who], anchor=who)
            self.move_wrist(side, p, R); self.push(f"{side}: descend on {who}", 0.6, 0.1, side, [who], anchor=who)
            self.fingers(side, closed); self.push(f"{side}: close on {who}", 0.4, 0.2, side, [who], anchor=who)
            self.fingers(side, press); self.push(f"{side}: grip {who}", 0.3, 0.2, side, [who], anchor=who)
            closed = press
        else:
            closed = {kk: float(d.qpos[a_]) for kk, a_ in rig.finger_qpos.items()}
            rp = regrasp_pose or {}
            own_open = {kk: float(v) for kk, v in (rp.get("finger_targets_open") or {}).items() if kk in rig.finger_range}
            opened = own_open or H.open_targets(rig, closed, float(rp.get("open_frac", 0.55)))
            if regrasp_before:
                self.fingers(side, opened); self.push(f"{side}: open for regrasp", 0.3, 0.1, side, [who], anchor=who)
                lift_off(regrasp_lift_m, f"{side}: open for regrasp")
                swing(-sign * bite, f"{side}: turn back")
                lift_off(-regrasp_lift_m, f"{side}: turn back")
                if regrasp_pose is not None:
                    # a fresh pinch each bite: the original pose re-anchored to where the part is now, so the
                    # drift of the pads during the last bite does not accumulate
                    p, R = pose_from_dict(regrasp_pose)
                    closed = {kk: float(v) for kk, v in regrasp_pose["finger_targets"].items() if kk in rig.finger_range}
                    squeezed = {kk: float(v) for kk, v in (regrasp_pose.get("finger_targets_squeezed") or {}).items() if kk in rig.finger_range}
                    press = H.press_targets(rig, closed, list(regrasp_pose.get("grip_fingers") or []), depth_rad=float(regrasp_pose.get("press_depth_rad", H.PRESS_DEPTH_RAD)), squeezed=squeezed)
                    own_open = {kk: float(v) for kk, v in (regrasp_pose.get("finger_targets_open") or {}).items() if kk in rig.finger_range}
                    opened = own_open or H.open_targets(rig, closed, float(regrasp_pose.get("open_frac", 0.55)))   # the pose's own opening
                    self.fingers(side, opened); self.move_wrist(side, p, R); self.push(f"{side}: realign on {who}", 0.5, 0.1, side, [who], anchor=who)
                    self.fingers(side, closed); self.push(f"{side}: regrasp {who}", 0.4, 0.2, side, [who], anchor=who)
                    self.fingers(side, press); self.push(f"{side}: grip {who}", 0.3, 0.2, side, [who], anchor=who)
                    closed = press
                else:
                    self.fingers(side, closed); self.push(f"{side}: regrasp {who}", 0.4, 0.2, side, [who], anchor=who)
        total = abs(float(turns)) * 2 * np.pi
        done = 0.0; n = int(bite_index) - 1
        while done < total - 1e-6:
            step = min(bite, total - done)
            pw, Rw = self.wrist(side); pp0, Rp0 = body_pose(m, d, part); p_rel, R_rel = relative_transform(pp0, Rp0, pw, Rw)
            # the hand rides the turning part through small angular steps: joint-space interpolation between
            # keyframes is a straight chord, and one keyframe per bite would drag the cap across its own axis
            n_sub = max(1, int(np.ceil(step / sub)))
            for i in range(n_sub):
                d.qpos[ay] += sign * step / n_sub; d.qpos[al] += k * sign * step / n_sub
                mujoco.mj_forward(m, d); pp, Rp = body_pose(m, d, part)
                self.move_wrist(side, *compose(pp, Rp, p_rel, R_rel))
                self.push(f"{lab} bite {n + 1}", t_bite / n_sub, 0.1 if i == n_sub - 1 else 0.0, side, [who], anchor=who)
            done += step; n += 1
            if done < total - 1e-6:                              # more bites in this call: regrasp in between
                self.fingers(side, opened); self.push(f"{side}: open for regrasp", 0.3, 0.1, side, [who], anchor=who)
                lift_off(regrasp_lift_m, f"{side}: open for regrasp")
                swing(-sign * step, f"{side}: turn back")
                lift_off(-regrasp_lift_m, f"{side}: turn back")
                self.fingers(side, closed); self.push(f"{side}: regrasp {who}", 0.4, 0.2, side, [who], anchor=who)
        return {"turns": float(turns), "bites": n - int(bite_index) + 1, "part": part, "closed": closed, "opened": opened}

    def push_along(self, side, pose, direction, distance_m, label=None, t=1.0, back=True, object_body=None):
        """Place the hand at pose and translate it along direction by distance (a press, a push, a slide), then back."""
        rig = self.rigs[side]
        p, R = pose_from_dict(pose)
        self.fingers(side, {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range})
        dvec = np.asarray(direction, float); dvec /= max(np.linalg.norm(dvec), 1e-9)
        self.move_wrist(side, p - dvec * 0.04, R); self.push(f"{side}: approach for push", 0.8, 0.1, side, [object_body] if object_body else [], anchor=object_body)
        self.move_wrist(side, p, R); self.push(f"{side}: contact", 0.4, 0.1, side, [object_body] if object_body else [], anchor=object_body)
        self.move_wrist(side, p + dvec * float(distance_m), R); self.push(label or f"{side}: push", t, 0.3, side, [object_body] if object_body else [], anchor=object_body)
        if back:
            self.move_wrist(side, p - dvec * 0.04, R); self.push(f"{side}: back off", 0.6, 0.1, side, [object_body] if object_body else [], anchor=object_body)

    def rotate_push(self, side, pose, pivot, angle_deg, label=None, t=2.0, sub_deg=8.0, back=True, object_body=None, approach_m=0.04,
                    press=None):
        """Turn a part on the table about the vertical through `pivot` by pushing it: the hand comes onto its contact
        (pose), presses into it by `press` (a world displacement of the wrist: a flat hand on a lid top that only
        touches it drags nothing), then sweeps around the pivot turning with the part (the contact rides the face), then
        backs off."""
        rig = self.rigs[side]
        objs = [object_body] if object_body else []
        p, R = pose_from_dict(pose)
        self.fingers(side, {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range})
        a = retreat_axis(pose, R)
        self.move_wrist(side, p + a * float(approach_m), R); self.push(f"{side}: approach for rotate", 0.8, 0.1, side, objs, anchor=object_body)
        self.move_wrist(side, p, R); self.push(f"{side}: contact", 0.4, 0.1, side, objs, anchor=object_body)
        if press is not None and float(np.linalg.norm(press)) > 1e-6:
            self.move_wrist(side, p + np.asarray(press, float), R); self.push(f"{side}: press for rotate", 0.4, 0.1, side, objs, anchor=object_body)
        piv = np.asarray(pivot, float)
        n_sub = max(1, int(np.ceil(abs(float(angle_deg)) / float(sub_deg))))
        Rs = Rot.from_rotvec(np.radians(float(angle_deg)) / n_sub * np.array([0.0, 0.0, 1.0])).as_matrix()
        for i in range(n_sub):
            pw_i, Rw_i = self.wrist(side)
            q = pw_i - piv
            self.move_wrist(side, piv + Rs @ q, Rs @ Rw_i)
            self.push(label or f"{side}: rotate", float(t) / n_sub, 0.2 if i == n_sub - 1 else 0.0, side, objs, anchor=object_body)
        if back:
            pw, Rw = self.wrist(side)
            a2 = Rot.from_rotvec(np.radians(float(angle_deg)) * np.array([0.0, 0.0, 1.0])).as_matrix() @ a
            self.move_wrist(side, pw + a2 * float(approach_m), Rw); self.push(f"{side}: back off", 0.6, 0.1, side, objs, anchor=object_body)

    def tilt(self, side, axis, angle_deg, pivot=None, label=None, t=1.2, hold_s=1.0, sub_deg=10.0, return_level=True):
        """Rotate the wrist (and whatever it carries) about a world axis through `pivot` (default the wrist); with
        `return_level` false the object stays turned (a cell laid down into its slot, a card turned on edge)."""
        pw, Rw = self.wrist(side)
        ax = np.asarray(axis, float); ax /= max(np.linalg.norm(ax), 1e-9)
        Rd = Rot.from_rotvec(np.radians(float(angle_deg)) * ax).as_matrix()
        piv = pw if pivot is None else np.asarray(pivot, float)
        objs = [self.attached[side]["body"]] if side in self.attached else []
        n_sub = max(1, int(np.ceil(abs(float(angle_deg)) / float(sub_deg))))
        Rs = Rot.from_rotvec(np.radians(float(angle_deg)) / n_sub * ax).as_matrix()
        for i in range(n_sub):
            pw_i, Rw_i = self.wrist(side)
            self.move_wrist(side, piv + Rs @ (pw_i - piv), Rs @ Rw_i); self.push(label or f"{side}: tilt", t / n_sub, hold_s if i == n_sub - 1 else 0.0, side, objs)
        if not return_level:
            return
        for i in range(n_sub):
            pw_i, Rw_i = self.wrist(side)
            self.move_wrist(side, piv + Rs.T @ (pw_i - piv), Rs.T @ Rw_i); self.push(f"{side}: level", t / n_sub, 0.2 if i == n_sub - 1 else 0.0, side, objs)

    def hold(self, side, pose, label=None, t_reach=1.0, t_close=0.4, object_body=None):
        rig = self.rigs[side]
        p, R = pose_from_dict(pose)
        closed = {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range}
        opened = H.open_targets(rig, closed, 0.55); a = retreat_axis(pose, R)
        self.fingers(side, opened); self.move_wrist(side, p + a * 0.06, R); self.push(f"{side}: reach to hold {object_body or ''}".strip(), t_reach, 0.1, side, [object_body] if object_body else [], anchor=object_body)
        self.move_wrist(side, p, R); self.push(f"{side}: descend to hold", 0.6, 0.1, side, [object_body] if object_body else [], anchor=object_body)
        self.fingers(side, closed); self.push(label or f"{side}: hold {object_body or ''}".strip(), t_close, 0.3, side, [object_body] if object_body else [], anchor=object_body)
        return {"closed": closed, "opened": opened}

    def go_home(self, side, home_qpos, label=None, t=2.0):
        rig = self.rigs[side]
        for k, a in rig.wrist_qpos.items():
            self.data.qpos[a] = home_qpos[a]
        for k, a in rig.finger_qpos.items():
            self.data.qpos[a] = home_qpos[a]
        self.prev_euler[side] = np.array([self.data.qpos[rig.wrist_qpos[k]] for k in ("rx", "ry", "rz")])
        self.detach(side)
        self.push(label or f"{side}: return home", t, 0.3, side, [])
