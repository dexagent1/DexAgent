"""One Stage 1 subgoal -> keyframes on the scene, the phases they form, and what must be true afterwards.
The motion vocabulary is Stage 1's (grasp_lift, carry, place, release, hold, push, pull, press, open_joint,
close_joint, unscrew, screw, insert, pour, tilt, regrasp, home) plus press_seat (a press-fit part pushed home by the
wrist), both (two hands, two parts, one rollout) and the bimanual motions of common/bimanual.py: pin (one hand steadies a
part while the other acts), lift_together / carry_together (two hands on one cable) and grasp_more (a second object in a
hand that already holds one), and the cable moves of common/ropemoves.py: lay and drag. Hand poses come from the grasp skills
(pose dicts with pos, euler, finger_targets[, finger_targets_squeezed, grip_fingers, approach]). track_reference follows a
dense tracking reference frame by frame (common/tracking.py: the output of Stage 3's method modules)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import hand as H
from tool_library.common import plan as P
from tool_library.common import episode as E
from tool_library.common import bimanual as B
from tool_library.common import held_contact as HC

MOTIONS = ("grasp_lift", "carry", "place", "release", "hold", "push", "pull", "press", "press_seat", "open_joint", "close_joint",
           "unscrew", "screw", "insert", "pour", "tilt", "regrasp", "rotate", "cut", "home", "wait", "other", "both",
           "pin", "lift_together", "carry_together", "grasp_more", "track_reference")
SIDES = ("left", "right")
WRIST_KP_REF = 10000.0       # the executor's default wrist servo stiffness: the press overshoot that saturates the force cap
SEAT_MAX_TRAVEL_M = 0.05     # a press_seat reference never goes deeper than this past its contact


def expand_subgoals(subgoals):
    """Split motions that are executed closed-loop into their steps: unscrew / screw become one entry per bite
    (index k, step i), so each bite is planned from the state the previous bite really reached."""
    out = []
    for sg in subgoals:
        if sg.get("motion") in ("unscrew", "screw") and not sg.get("single_shot"):
            turns = abs(float(sg["turns"])); bite = float(sg.get("bite_deg", 90.0))
            n = max(1, int(np.ceil(turns * 360.0 / bite - 1e-6)))
            for i in range(n):
                step = dict(sg)
                step.update({"motion": sg["motion"] + "_bite", "step": i + 1, "steps": n,
                             "bite_turns": min(bite / 360.0, turns - i * bite / 360.0), "first": i == 0, "last": i == n - 1})
                if i > 0:
                    # later bites re-grasp with the pose bite 1 used: one pose, never the candidate list
                    rp = sg.get("pose")
                    rp = rp[0] if isinstance(rp, list) and rp else (load_pose(rp) if isinstance(rp, (str, Path)) else rp)
                    step["regrasp_pose"] = rp; step.pop("pose", None)
                if i < n - 1:
                    step.pop("lift_after_m", None)
                out.append(step)
        else:
            out.append(dict(sg))
    return out


def load_pose(pose):
    if isinstance(pose, (str, Path)):
        d = json.loads(Path(pose).read_text())
        if isinstance(d, dict) and "pose" in d and "pos" not in d:
            d = d["pose"]
        if isinstance(d, dict) and "poses" in d:
            d = d["poses"][0]
        if isinstance(d, list):
            d = d[0]
        return d
    return pose


def reanchor_pose(model, data, pose, anchor, mode="full"):
    """Move a hand pose rigidly with its part: the pose was solved with `anchor['body']` at anchor pos/quat;
    the part is now where the scene has it. Returns a new pose dict (pos, euler, approach transformed).
    mode: 'full' (default) follows the part's whole motion; 'yaw' its translation and its turn about the vertical
    only (a part that rolled about its own symmetry axis, a battery or a bottle, keeps the hand where it was posed);
    'translate' its translation only; 'none' leaves the pose as solved."""
    if not anchor or mode == "none":
        return pose
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor["body"])
    if b < 0:
        return pose
    mujoco.mj_forward(model, data)
    p_now, R_now = data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()
    q = anchor["quat_wxyz"]; R_then = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); p_then = np.asarray(anchor["pos"], float)
    dR = R_now @ R_then.T
    if mode == "translate":
        dR = np.eye(3)
    elif mode == "yaw":
        yaw = float(Rot.from_matrix(dR).as_euler("ZYX")[0])
        dR = Rot.from_euler("z", yaw).as_matrix()
    dp = p_now - dR @ p_then
    if np.linalg.norm(dp) < 1e-6 and np.linalg.norm(Rot.from_matrix(dR).as_rotvec()) < 1e-6:
        return pose
    p, R = P.pose_from_dict(pose)
    out = dict(pose)
    out["pos"] = (dR @ p + dp).tolist(); out["euler"] = H.mat_to_euler(dR @ R).tolist()
    if pose.get("approach") is not None:
        out["approach"] = (dR @ np.asarray(pose["approach"], float)).tolist()
    if pose.get("points"):
        out["points"] = {f: (dR @ np.asarray(v, float) + dp).tolist() for f, v in pose["points"].items()}
    if pose.get("normals"):
        out["normals"] = {f: (dR @ np.asarray(v, float)).tolist() for f, v in pose["normals"].items()}
    out["reanchored"] = {"body": anchor["body"], "mode": mode, "shift_m": round(float(np.linalg.norm(p_now - p_then)), 4),
                         "turn_deg": round(float(np.degrees(np.linalg.norm(Rot.from_matrix(dR).as_rotvec()))), 2)}
    return out


_ANCHOR_CACHE = {}


def _hand_scene_anchor(hs_path, hs):
    """The grasp body's pose in the hand scene as it actually compiles (its objects and joint_init), not the pose
    recorded when the file was written: a hand scene whose joint_init was edited afterwards (a lid set to another
    angle) keeps the old record, and re-anchoring to it would turn every pose by the difference."""
    key = None
    if hs_path is not None:
        try:
            key = (str(Path(hs_path).resolve()), Path(hs_path).stat().st_mtime)
        except OSError:
            key = None
    if key is not None and key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[key]
    rec = hs.get("grasp_body_pose")
    out = rec
    try:
        from tool_library.common import hand as H
        model, data, rig, ids = H.open_scene(hs)
        gb = H.grasp_body(model, hs, ids)
        q = Rot.from_matrix(data.xmat[gb].reshape(3, 3)).as_quat()
        live = {"body": model.body(gb).name, "pos": data.xpos[gb].round(6).tolist(), "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])]}
        if rec and rec.get("body") == live["body"]:
            out = live
        elif not rec:
            out = live
    except Exception:                                  # an unreadable hand scene: keep the record
        out = rec
    if key is not None:
        _ANCHOR_CACHE[key] = out
    return out


def pose_anchor(sg, pose):
    """Where the pose's part was when the pose was solved: pose['anchor'], else the hand scene the subgoal names (its
    grasp body as the hand scene compiles)."""
    if isinstance(pose, dict) and pose.get("anchor"):
        return pose["anchor"]
    hs = sg.get("hand_scene")
    if hs:
        path = None
        if isinstance(hs, (str, Path)):
            path = hs
            hs = json.loads(Path(hs).read_text())
        return _hand_scene_anchor(path, hs)
    return None


def _object_top(model, data, bodies):
    """Highest point of the body set's collision geometry (exact extents; bounding spheres put a flat brick's top
    centimetres too high)."""
    return float(E.bodies_extent(model, data, bodies)[1])


def _object_bottom(model, data, bodies):
    """Lowest point of the body set's collision geometry (exact extents)."""
    return float(E.bodies_extent(model, data, bodies)[0])


def _half_height(model, data, bodies):
    lo, hi = E.bodies_extent(model, data, bodies)
    return 0.5 * float(hi - lo)


def _whole_joints(model, adrs):
    """The qpos addresses of every joint that has at least one address in `adrs` (a free body moves as all 7)."""
    adrs = {int(a) for a in adrs}
    out = set()
    for j in range(model.njnt):
        a0 = int(model.jnt_qposadr[j])
        a1 = int(model.jnt_qposadr[j + 1]) if j + 1 < model.njnt else int(model.nq)
        rng = set(range(a0, a1))
        if rng & adrs:
            out |= rng
    return out


def free_objects(model):
    return [b for b in E.object_free_joints(model)]


def _blade_length(model, body_id, default=0.06):
    """How far a scissors half's blade reaches along its body's +x (construct_scissors: box geoms named *_blade*)."""
    far = 0.0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) == int(body_id) and "_blade" in (model.geom(g).name or ""):
            far = max(far, float(model.geom_pos[g][0] + model.geom_size[g][0]))
    return far if far > 0 else float(default)


def _support_margin(points, centre, axis):
    """Signed distance (m) of `centre` inside the convex hull of `points` projected on the plane normal to `axis`
    (negative: outside, the distance to the hull; a single point or a segment counts as outside)."""
    from scipy.spatial import ConvexHull
    axis = np.asarray(axis, float) / max(np.linalg.norm(axis), 1e-12)
    u = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    Pp = np.array([[(np.asarray(q, float) - centre) @ u, (np.asarray(q, float) - centre) @ v] for q in points]).reshape(-1, 2)
    if len(Pp) == 0:
        return -1.0
    if len(Pp) >= 3 and np.linalg.matrix_rank(Pp - Pp.mean(0), tol=2e-4) >= 2:
        try:
            hull = ConvexHull(Pp)
            inside = float(-np.max(hull.equations[:, 2]))
            if inside >= 0.0:
                return inside
            best = 1e9                                  # outside: distance to the hull's boundary
            V = Pp[hull.vertices]
            for i in range(len(V)):
                A, B = V[i], V[(i + 1) % len(V)]
                t = np.clip(-A @ (B - A) / max((B - A) @ (B - A), 1e-12), 0.0, 1.0)
                best = min(best, float(np.linalg.norm(A + t * (B - A))))
            return -best
        except Exception:
            pass
    i, j = np.unravel_index(np.argmax(np.linalg.norm(Pp[:, None] - Pp[None], axis=2)), (len(Pp), len(Pp)))
    A, B = Pp[i], Pp[j]
    t = np.clip(-A @ (B - A) / max((B - A) @ (B - A), 1e-12), 0.0, 1.0)
    return -float(np.linalg.norm(A + t * (B - A)))


def press_contacts(model, data, rig, p, R, fingers, part_gids, dvec, start_back_m=0.015, band_m=0.001, iters=8):
    """Where a posed hand pushed along `dvec` meets a part: lowered (kinematically, fingers as posed) from start_back_m
    behind the pose until its first collision geom touches the part's collision geoms. Returns (offset past the pose
    along dvec in m (negative: before the pose), [(contact point, hand geom, part geom)] within band_m of the first
    touch), or (None, []) when it never meets the part within 4 cm."""
    dvec = np.asarray(dvec, float) / max(np.linalg.norm(dvec), 1e-12)
    hand = sorted(rig.collision_gids)
    part = sorted(int(g) for g in part_gids)
    ft = np.zeros(6)

    def gaps(pp):
        P.set_wrist(model, data, rig, pp, R)
        rig.set_fingers(data, fingers)
        mujoco.mj_kinematics(model, data)
        out = []
        for h in hand:
            for g in part:
                if np.linalg.norm(data.geom_xpos[h] - data.geom_xpos[g]) > model.geom_rbound[h] + model.geom_rbound[g] + 0.045:
                    continue
                dist = H.geom_distance(model, data, h, g, 0.045, ft)
                out.append((dist, 0.5 * (ft[:3] + ft[3:]), h, g))
        return out

    q = np.asarray(p, float) - dvec * float(start_back_m)
    moved = -float(start_back_m)
    gs = gaps(q)
    for _ in range(int(iters)):
        if not gs:
            return None, []
        dm = min(x[0] for x in gs)
        if abs(dm) < 2e-4 or moved > 0.04:
            break
        step = dm if dm > 0 else dm          # a negative distance backs the hand out again
        q = q + dvec * step; moved += step
        gs = gaps(q)
    if not gs or moved > 0.04:
        return None, []
    dm = min(x[0] for x in gs)
    return moved, [(x[1].copy(), x[2], x[3]) for x in gs if x[0] <= dm + float(band_m)]


def press_balance(model, data, rig, p, R, fingers, part_gids, avoid_gids, dvec, centre,
                  pitches_deg=(0.0, -5.0, -10.0, -15.0, 5.0, 10.0), shifts_m=(0.0, -0.006, 0.006, -0.012, 0.012)):
    """The pose a compliant hand would settle into on a press: a point contact transmits no torque, so a press whose
    first contacts do not surround the part's centre (a fingertip on a stud brick's corner stud) tips the part and jams
    it (measured: 1.9 deg, 1 mm short of the seat at 23 N). Candidates rock the hand about the line across its fingers
    (local y) through its first contacts and shift it in the press plane; each is lowered onto the part
    (press_contacts) and scored by how far the part's centre lies inside its contacts' hull, with no hand geom more than
    1 mm into anything but the part. Returns {pos, R, pitch_deg, shift_m, margin_mm, offset_m, n_contacts, original:
    {margin_mm, offset_m}}; the original pose when nothing improves its margin by 1 mm."""
    dvec = np.asarray(dvec, float) / max(np.linalg.norm(dvec), 1e-12)
    centre = np.asarray(centre, float)
    p = np.asarray(p, float); R = np.asarray(R, float)
    d = mujoco.MjData(model); d.qpos[:] = data.qpos
    off0, c0 = press_contacts(model, d, rig, p, R, fingers, part_gids, dvec)
    if off0 is None:
        return None
    m0 = _support_margin([c[0] for c in c0], centre, dvec)
    piv = np.mean([c[0] for c in c0], axis=0)
    u = R[:, 1] - dvec * (R[:, 1] @ dvec)
    u = u / max(np.linalg.norm(u), 1e-9)
    w = np.cross(dvec, u)
    hand = sorted(rig.collision_gids)
    avoid = sorted(int(g) for g in avoid_gids)

    def clear(pp, RR, off):
        # the hand at its first contact and 2 mm past it must not go into the table or a neighbour
        for extra in (0.0, 0.002):
            P.set_wrist(model, d, rig, pp + dvec * (off + extra), RR); rig.set_fingers(d, fingers); mujoco.mj_kinematics(model, d)
            for h in hand:
                for g in avoid:
                    if np.linalg.norm(d.geom_xpos[h] - d.geom_xpos[g]) > model.geom_rbound[h] + model.geom_rbound[g]:
                        continue
                    if H.geom_distance(model, d, h, g, 0.01) < -0.001:
                        return False
        return True

    best = {"pos": p, "R": R, "pitch_deg": 0.0, "shift_m": [0.0, 0.0], "margin_mm": round(1000 * m0, 2), "offset_m": round(float(off0), 5),
            "n_contacts": len(c0), "score": (min(m0, 0.003), 0.0)}
    if m0 < 0.0:
        for dp in pitches_deg:
            Rd = Rot.from_rotvec(np.radians(float(dp)) * R[:, 1]).as_matrix()
            Rc = Rd @ R; pc = piv + Rd @ (p - piv)
            for su in shifts_m:
                for sw in shifts_m:
                    if dp == 0.0 and su == 0.0 and sw == 0.0:
                        continue
                    pp = pc + u * float(su) + w * float(sw)
                    off, cs = press_contacts(model, d, rig, pp, Rc, fingers, part_gids, dvec)
                    if off is None:
                        continue
                    mg = _support_margin([c[0] for c in cs], centre, dvec)
                    change = abs(float(dp)) / 5.0 + 1000.0 * float(np.hypot(su, sw)) / 6.0
                    score = (min(mg, 0.003), -change)
                    if score > best["score"] and mg > m0 + 0.001 and clear(pp, Rc, off):
                        best = {"pos": pp, "R": Rc, "pitch_deg": float(dp), "shift_m": [float(su), float(sw)], "margin_mm": round(1000 * mg, 2),
                                "offset_m": round(float(off), 5), "n_contacts": len(cs), "score": score}
    best.pop("score")
    best["original"] = {"margin_mm": round(1000 * m0, 2), "offset_m": round(float(off0), 5), "n_contacts": len(c0)}
    return best


class SubgoalPlanner:
    """Wraps plan.Planner with the grip memory and the expectation writer."""

    def __init__(self, model, qpos_plan, state=None, fps=30.0, rates=(90.0, 0.4)):
        self.model = model
        self.pl = P.Planner(model, qpos0=qpos_plan, fps=fps)
        self.rates = (float(rates[0]), float(rates[1]))     # the executor's joint-rate ceilings (a two-hand entry chains with them)
        self._rollout = {}
        self._target_info = None
        self.state = {"held": {}, "grip": {}, **(state or {})}
        for side, body in list(self.state["held"].items()):
            if body:
                try:
                    self.pl.attach(side, body)
                except ValueError:
                    pass
        B.attach_extras(self)                           # a hand holding two objects carries both

    # ------------------------------------------------------------------ helpers
    def _pose(self, sg):
        """The subgoal's hand pose, re-anchored to where its part is now. Grip settings given on the subgoal
        (open_frac, finger_targets_open, press_depth_rad, press_travel_m) override the pose's own."""
        pose = dict(load_pose(sg["pose"]))
        for k in ("open_frac", "finger_targets_open", "press_depth_rad", "press_travel_m"):
            if sg.get(k) is not None:
                pose[k] = sg[k]
        out = reanchor_pose(self.model, self.pl.data, pose, pose_anchor(sg, pose), mode=sg.get("reanchor", "full"))
        if out.get("reanchored"):
            self.state.setdefault("reanchored", []).append({"index": sg.get("index"), **out["reanchored"]})
        return out

    def _grip_from_pose(self, side, pose):
        rig = self.pl.rigs[side]
        # the clamp (fingers past touch, or for a one-sided press whose flexion would lift the pads, the wrist pressing
        # along the contact normal) as the grasp stress tests drive it (hand.executor_clamp)
        c = H.executor_clamp(rig, pose)
        closed, grip = c["closed"], c["grip_fingers"]
        own_open = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
        opened = own_open or H.approach_open_targets(rig, closed, float(pose.get("open_frac", 0.55)), grip, c["grip_posed"])
        g = {"closed": closed, "press": c["press"], "opened": opened, "own_open": bool(own_open), "wrist_press": c["wrist_press"],
             "grip_fingers": grip, "grip_posed": c["grip_posed"]}
        self.state["grip"][side] = g
        return g

    def _grip(self, side):
        g = self.state["grip"].get(side)
        if g is None:
            rig = self.pl.rigs[side]
            closed = {k: float(self.pl.data.qpos[a]) for k, a in rig.finger_qpos.items()}
            g = {"closed": closed, "press": H.press_targets(rig, closed), "opened": H.open_targets(rig, closed, 0.55)}
            self.state["grip"][side] = g
        return g

    def _untouched(self, involved):
        held = {b for b in self.state["held"].values() if b}
        held |= {e["body"] for lst in (self.state.get("in_hand") or {}).values() for e in (lst or [])}   # every object a hand holds
        m = self.model
        inv = set()
        for name in involved:
            if name:
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
                if b >= 0:
                    inv |= {m.body(x).name for x in E.subtree_bodies(m, b)}
                    r = b                                 # the root the part belongs to
                    while int(m.body_parentid[r]) != 0:
                        r = int(m.body_parentid[r])
                    inv |= {m.body(x).name for x in E.subtree_bodies(m, r)}
        # bodies welded to an involved body move with it (a cap on its thread carrier)
        changed = True
        while changed:
            changed = False
            for e in range(m.neq):
                if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_WELD):
                    continue
                n1, n2 = m.body(int(m.eq_obj1id[e])).name, m.body(int(m.eq_obj2id[e])).name
                if (n1 in inv) != (n2 in inv) and not (n1.startswith("robot_") or n2.startswith("robot_")):
                    other = n2 if n1 in inv else n1
                    inv |= {m.body(x).name for x in E.subtree_bodies(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, other))}
                    changed = True
        # objects resting on or in an involved body ride with it (a toy inside the drawer being opened):
        # anything whose collision geometry is within a centimetre of the involved bodies
        d = self.pl.data; mujoco.mj_forward(m, d)
        inv_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in inv}
        inv_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in inv_ids and (m.geom_contype[g] or m.geom_conaffinity[g])]
        riding = set()
        for name in free_objects(m):
            if name in inv or name in held:
                continue
            ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            sub = E.subtree_bodies(m, ob)
            geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub and (m.geom_contype[g] or m.geom_conaffinity[g])]
            close = False
            for g in geoms:
                for ig in inv_geoms:
                    if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[ig]) > m.geom_rbound[g] + m.geom_rbound[ig] + 0.02:
                        continue
                    if H.geom_distance(m, d, g, ig, 0.01) < 0.01:
                        close = True; break
                if close:
                    break
            if close:
                riding.add(name)
        return [b for b in free_objects(m) if b not in inv and b not in held and b not in riding]

    def _floor_clear_open(self, side, pose, g):
        """The widest opening (the pose's own open_frac, else 0.55, down to 0.15) whose fingers stay above the table
        at the pose: a lid lying 3 cm above the table leaves no room for fingers opened wide. A pose that carries an
        open_frac (a flat pinch stress-tested from 20% open) is never opened wider than that. A one-sided wrist press
        keeps its grip fingers as posed at every opening (hand.approach_open_targets)."""
        m, d = self.model, self.pl.data
        rig = self.pl.rigs[side]
        table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_table")
        if table < 0:
            return g["opened"]
        saved = d.qpos.copy()
        best = g["opened"]
        top = float(pose["open_frac"]) if pose.get("open_frac") is not None else 0.55
        for frac in [top] + [f for f in (0.45, 0.35, 0.25, 0.15) if f < top - 1e-9]:
            opened = H.approach_open_targets(rig, g["closed"], frac, pose.get("grip_fingers"), g.get("grip_posed", False))
            p_, R_ = P.pose_from_dict(pose)
            P.set_wrist(m, d, rig, p_, R_, self.pl.prev_euler[side]); rig.set_fingers(d, opened); mujoco.mj_forward(m, d)
            worst = min(H.geom_distance(m, d, g_, table, 0.05) for g_ in rig.collision_gids)
            best = opened
            if worst > -0.002:
                break
        d.qpos[:] = saved; mujoco.mj_forward(m, d)
        return best

    @staticmethod
    def _approach_timing(sg):
        """Expose contact speeds so an agent can act on impulse feedback."""
        times = {}
        for key in ("t_reach", "t_descend", "t_close", "t_grip"):
            if key in sg:
                value = float(sg[key])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be a finite positive duration in seconds")
                times[key] = value
        return times

    def _approach_pose(self, side, pose, object_body, approach_m=0.08, t_reach=2.0, t_descend=0.8, t_close=0.5, label=None, t_grip=0.3):
        """Open hand to the pre-grasp, down to the pose, close. Writes the grip memory."""
        g = self._grip_from_pose(side, pose)
        if not g.get("own_open"):                      # a pose may carry its own open shape (a hook keeps the index straight)
            g["opened"] = self._floor_clear_open(side, pose, g)
        self.state["grip"][side] = g
        p, R = P.pose_from_dict(pose)
        a = P.retreat_axis(pose, R)
        self.pl.fingers(side, g["opened"]); self.pl.move_wrist(side, p + a * float(approach_m), R)
        self.pl.push(label or f"{side}: reach for {object_body}", t_reach, 0.1, side, [object_body], anchor=object_body)
        # optional waypoints between the pre-grasp and the pose: world offsets from the pose, visited in order
        # (over a bar and then down behind it; around a wall)
        for i, off in enumerate(pose.get("approach_waypoints") or []):
            self.pl.move_wrist(side, p + np.asarray(off, float), R)
            self.pl.push(f"{side}: approach {object_body} ({i + 1})", 0.6, 0.0, side, [object_body], anchor=object_body)
        self.pl.move_wrist(side, p, R); self.pl.push(f"{side}: descend on {object_body}", t_descend, 0.1, side, [object_body], anchor=object_body)
        self.pl.fingers(side, g["closed"]); self.pl.push(f"{side}: close on {object_body}", t_close, 0.2, side, [object_body], anchor=object_body)
        if g.get("wrist_press"):
            self.pl.move_wrist(side, p + np.asarray(g["wrist_press"], float), R)   # a flat press: the wrist leans in
        self.pl.fingers(side, g["press"]); self.pl.push(f"{side}: grip {object_body}", t_grip, 0.3, side, [object_body], anchor=object_body)
        # the pads' planned contacts (point + outward normal) so the outcome can check each pad touches from its side
        pts = pose.get("points") or pose.get("targets")
        if pose.get("normals") and pts:
            self._pads = {"pads": {f: {"point": list(map(float, pts[f])), "normal": list(map(float, pose["normals"][f]))}
                                   for f in pose["normals"] if f in pts}, "pads_at_label": f"{side}: grip {object_body}"}
        else:
            self._pads = None
        return g

    def _relative_extent(self, object_body, axis, target_R=None):
        """Exact extent relative to the root after a proposed rigid rotation.

        Transform the query axis, not the scene state. This includes mesh and
        descendant geometry without altering the actual contact start state.
        """
        m, d = self.model, self.pl.data
        ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, object_body)
        if ob < 0:
            raise ValueError(f"no body {object_body!r}")
        u = np.asarray(axis, float)
        if target_R is not None:
            delta = np.asarray(target_R) @ d.xmat[ob].reshape(3, 3).T
            u = delta.T @ u
        bounds = np.asarray(E.bodies_extent(m, d, E.subtree_bodies(m, ob), u=u))
        return bounds - float(d.xpos[ob] @ u)

    def _footprint_at(self, object_body, target_xy, target_R=None):
        """The carried part's rectangle at its requested destination orientation."""
        x0, x1 = self._relative_extent(object_body, [1, 0, 0], target_R)
        y0, y1 = self._relative_extent(object_body, [0, 1, 0], target_R)
        return (x0 + target_xy[0], x1 + target_xy[0], y0 + target_xy[1], y1 + target_xy[1])

    def _target_from(self, sg, object_body):
        """World target position for the object root from `target_pos`, or `on`/`over`/`into` + offsets. Heights come
        from the bodies' real collision geometry: the support's highest surface under the carried part's footprint (rays
        on the support's own geoms plus the exact tops of small primitives such as studs) and the carried part's own
        lowest point."""
        m, d = self.model, self.pl.data
        self._target_info = None
        if sg.get("target_pos") is not None:
            return np.asarray(sg["target_pos"], float), None
        target_R = self._target_R(sg, object_body)
        if sg.get("into"):
            # into a container: the carried part's lowest point lands on the highest surface of the container under the
            # part's footprint at its centre: the inner floor (or the sloped wall where the part reaches it), not the rim
            ref = sg["into"]
            rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ref)
            if rb < 0:
                raise ValueError(f"no body {ref!r} for into")
            rsub = E.subtree_bodies(m, rb)
            mujoco.mj_forward(m, d)
            off = np.asarray(sg.get("offset", [0, 0, 0]), float)
            c = d.xpos[rb].copy() + np.array([off[0], off[1], 0.0])
            floor, info = E.support_top(m, d, rsub, self._footprint_at(object_body, c[:2], target_R))
            lowest_rel = self._relative_extent(object_body, [0, 0, 1], target_R)[0]
            z = floor - lowest_rel + float(sg.get("clearance_m", 0.003))
            self._target_info = {"support": ref, "support_top_z": round(float(floor), 5), "part_bottom_below_root_m": round(float(-lowest_rel), 5), **info}
            return np.array([c[0], c[1], z + off[2]]), ref
        ref = sg.get("on") or sg.get("over")
        if ref:
            rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ref)
            if rb < 0:
                raise ValueError(f"no body {ref!r} for on/over")
            rsub = E.subtree_bodies(m, rb)
            mujoco.mj_forward(m, d)
            base = d.xpos[rb].copy()
            off = np.asarray(sg.get("offset", [0, 0, 0]), float)
            tx, ty = float(base[0] + off[0]), float(base[1] + off[1])
            lowest_rel = self._relative_extent(object_body, [0, 0, 1], target_R)[0]
            top, info = E.support_top(m, d, rsub, self._footprint_at(object_body, (tx, ty), target_R))
            if sg.get("on"):
                z = top - lowest_rel + 0.002
            else:
                # a travel height clears the whole support (its rim, a stud, a handle), not only what lies under the part
                top = max(top, _object_top(m, d, rsub))
                z = top - lowest_rel + float(sg.get("height_m", 0.08))
            self._target_info = {"support": ref, "support_top_z": round(float(top), 5), "part_bottom_below_root_m": round(float(-lowest_rel), 5), **info}
            return np.array([tx, ty, z + off[2]]), ref
        if sg.get("target") == "start" and sg.get("start_pos") is not None:
            return np.asarray(sg["start_pos"], float), None
        raise ValueError("carry/place/insert need target_pos, on, over, into, or target='start' with start_pos")

    # ------------------------------------------------------------------ motions
    def run(self, sg):
        motion = sg.get("motion", "other")
        if motion == "both":
            return self.run_both(sg)
        if motion in ("lift_together", "carry_together"):
            return B.run_together(self, sg)
        self._rollout = {}
        self._target_info = None
        side = sg.get("side") or sg.get("acting_hand")
        if side not in self.pl.rigs and motion not in ("wait", "other", "track_reference"):
            raise ValueError(f"subgoal {sg.get('index')}: side must be one of {list(self.pl.rigs)}, got {side!r}")
        obj = sg.get("object")
        withs = sg.get("with") or []
        withs = [withs] if isinstance(withs, str) else list(withs)     # one body or a list of them
        bad = [w for w in withs if not isinstance(w, str) or mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, w) < 0]
        if bad:
            raise ValueError(f"subgoal {sg.get('index')}: `with` names no body of the scene: {bad} (give body names, one or a list)")
        before = len(self.pl.keyframes)
        fn = getattr(self, f"m_{motion}", None)
        if fn is None:
            raise ValueError(f"unknown motion {motion!r}; one of {MOTIONS}")
        if side in self.pl.rigs:
            B.check_pinning_side(self, sg, side, motion)
            if obj:
                B.make_primary(self, side, obj)            # one of several objects in the hand named: the motion is about it
        # keyframe 0 is the start state itself, so the first leg is a real transit from where the hands are
        start = self.pl.push("start", 0.0, 1 / self.pl.fps, side, [])
        self.pl.phases.pop()
        self._pads = None
        pins = None if sg.get("_in_both") else B.pins_for(self, B.acting_sides(sg))   # where the other hand's pins hold their parts now
        expect = fn(sg, side, obj) or {}
        if side in self.pl.rigs and not expect.get("released"):
            B.also_in_hand_expect(self, side, expect)      # the other objects of a hand holding several stay in it
        if len(self.pl.keyframes) > before + 1:
            start["label"] = self.pl.keyframes[before + 1]["label"]
        if self._pads and expect.get("pads") is None:
            expect.update(self._pads)
        if self._target_info:
            expect.setdefault("placement", self._target_info)
        expect.setdefault("hand", side)
        if obj and "object" not in expect:
            expect["object"] = obj
        involved = [obj, sg.get("on"), sg.get("over")] + withs + list(expect.get("involved", []) or [])
        expect["involved"] = [b for b in involved if b]
        expect.setdefault("untouched", self._untouched(involved))
        for k in ("pos_tol_m", "max_hand_pen_m", "max_obj_pen_m", "max_track_m", "fraction", "slip_tol_m", "stay_tol_m"):
            if k in sg:
                expect[k] = sg[k]
        if pins:
            B.add_pin_expect(self, sg, expect, B.acting_sides(sg), pins)   # parts the other hand pins stay put on every frame
        expect = HC.merge_motion_expectations(expect, sg.get("expect"))
        out = {"keyframes": self.pl.keyframes[before:], "phases": self.pl.phases[before:], "expect": expect, "state": self.state}
        if self._rollout:
            out["rollout"] = dict(self._rollout)          # physics settings the motion needs (press_seat: the wrist force cap)
        return out

    def _lift_offsets(self, sg, lift):
        """The lift's way out as wrist offsets from the grip: straight up by default, along `lift_dir` (a part under an
        overhang drawn out toward the operator while it rises), or through `lift_path` ([[dx, dy, dz], ...] offsets
        from the grip, visited in order: out first, then up)."""
        if sg.get("lift_path"):
            return [np.asarray(v, float) for v in sg["lift_path"]]
        if sg.get("lift_dir") is not None:
            v = np.asarray(sg["lift_dir"], float); v = v / max(np.linalg.norm(v), 1e-9)
            return [v * lift]
        return [np.array([0.0, 0.0, lift])]

    def _lift(self, side, obj, offsets, t_each):
        pw, Rw = self.pl.wrist(side)
        for i, off in enumerate(offsets):
            self.pl.move_wrist(side, pw + off, Rw)
            self.pl.push(f"{side}: lift {obj}", t_each, 0.3 if i == len(offsets) - 1 else 0.05, side, [obj], anchor=obj)

    def m_grasp_lift(self, sg, side, obj):
        lift = float(sg.get("lift_m", 0.08))
        offsets = self._lift_offsets(sg, lift)
        duration = float(sg.get("duration_s", 1.0 if self.state["held"].get(side) == obj and sg.get("pose") is None
                                else max(0.8, 0.4 * len(offsets))))
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("grasp_lift duration_s must be finite positive seconds")
        if self.state["held"].get(side) == obj and sg.get("pose") is None:
            self.pl.attach(side, obj); self._couple(side, obj)   # already in hand: lift only (with any other hand on it)
            self.pl.fingers(side, self._grip(side)["press"])
            self._lift(side, obj, offsets, duration / len(offsets))
        else:
            pose = self._pose(sg)
            g = self._approach_pose(side, pose, obj, approach_m=float(sg.get("approach_m", 0.08)), **self._approach_timing(sg))
            self.pl.attach(side, obj); self._couple(side, obj)   # a hand already holding it (a two-hand squeeze) lifts too
            self.pl.fingers(side, g["press"])
            self._lift(side, obj, offsets, duration / len(offsets))
        self.state["held"][side] = obj
        rise = float(offsets[-1][2])
        exp = {"rise_m": max(0.005, 0.5 * rise), "in_hand": True, "no_slip": True}
        if self.pl.coupled.get(side):
            exp["min_links"] = 1                        # each hand of a two-hand squeeze touches with one face
        return exp

    def m_hold(self, sg, side, obj):
        pose = self._pose(sg)
        self._approach_pose(side, pose, obj, approach_m=float(sg.get("approach_m", 0.06)), label=f"{side}: reach to hold {obj}", **self._approach_timing(sg))
        self.pl.phases[-1]["label"] = self.pl.keyframes[-1]["label"] = f"{side}: hold {obj}"
        if self._pads:
            self._pads["pads_at_label"] = f"{side}: hold {obj}"
        self.state["held"][side] = obj
        flat = bool(pose.get("grip_fingers")) and "thumb" not in (pose.get("grip_fingers") or [])
        return {"in_hand": True, "stays_put": True, "min_links": 1 if flat else 2}   # a flat press holds with one face

    def m_pin(self, sg, side, obj):
        """Steady a part (or a cable link or end) for the other hand: bimanual.plan_pin."""
        return B.plan_pin(self, sg, side, obj)

    def m_grasp_more(self, sg, side, obj):
        """Take a second object with the free fingers of a hand that holds one: bimanual.plan_grasp_more."""
        return B.plan_grasp_more(self, sg, side, obj)

    def _couple(self, side, obj):
        self.pl.coupled[side] = [s_ for s_, b in self.state["held"].items() if s_ != side and b == obj]

    def _target_R(self, sg, obj):
        """Target orientation of a carried object: `target_euler_deg` [rx, ry, rz] (absolute, intrinsic XYZ) or
        `turn` {axis, deg} applied to its current orientation; None keeps the orientation (the default)."""
        if sg.get("target_euler_deg") is not None:
            return Rot.from_euler("XYZ", np.radians(np.asarray(sg["target_euler_deg"], float))).as_matrix()
        if sg.get("turn"):
            ax = np.asarray(sg["turn"]["axis"], float); ax /= max(np.linalg.norm(ax), 1e-9)
            ob = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj); mujoco.mj_forward(self.model, self.pl.data)
            return Rot.from_rotvec(np.radians(float(sg["turn"]["deg"])) * ax).as_matrix() @ self.pl.data.xmat[ob].reshape(3, 3)
        return None


    def m_carry(self, sg, side, obj):
        obj = obj or self.state["held"].get(side)
        if self.state["held"].get(side) != obj:
            raise ValueError(f"carry: {side} hand does not hold {obj!r} (held: {self.state['held']})")
        self.pl.attach(side, obj); self._couple(side, obj)
        self.pl.fingers(side, self._grip(side)["press"])
        tgt, ref = self._target_from(sg, obj)
        via = sg.get("via_height")
        if via is None and sg.get("via_rise_m") is not None:     # rise this far above the higher end on the way
            pw, _ = self.pl.wrist(side)
            via = float(max(pw[2], tgt[2])) + float(sg["via_rise_m"])
        R_t = self._target_R(sg, obj)
        self.pl.carry_to(side, tgt, target_R=R_t, via_height=via, t=float(sg.get("duration_s", 1.5)), anchor=ref, label=f"{side}: carry {obj}" + (" (free)" if sg.get("arm_path") == "joint" else ""))
        exp = {"target_pos": tgt.tolist(), "in_hand": True, "no_slip": True, "target_axes": sg.get("target_axes", "xyz")}
        if R_t is not None:
            quat = Rot.from_matrix(R_t).as_quat()
            exp["target_quat_wxyz"] = np.r_[quat[3], quat[:3]].tolist()
        if self.pl.coupled.get(side):
            exp["min_links"] = 1                        # a two-hand carry: each hand touches with one face
        return exp

    def m_insert(self, sg, side, obj):
        exp = self.m_carry(sg, side, obj)
        exp["pos_tol_m"] = float(sg.get("pos_tol_m", 0.015))
        if sg.get("release"):
            exp.update(self._release(side, obj, sg, anchor=sg.get("on") or sg.get("over")))
            # let go in the slot: the part ends out of the hand, held by the slot (the carry's in-hand checks would fail
            # by construction)
            exp.pop("in_hand", None); exp.pop("no_slip", None); exp.pop("min_links", None)
            exp.setdefault("supported", True)
        return exp

    def m_place(self, sg, side, obj):
        obj = obj or self.state["held"].get(side)
        if self.state["held"].get(side) != obj:
            raise ValueError(f"place: {side} hand does not hold {obj!r}")
        self.pl.attach(side, obj); self._couple(side, obj)
        first_phase = len(self.pl.phases)
        self.pl.fingers(side, self._grip(side)["press"])
        tgt, ref = self._target_from(sg, obj)
        via = sg.get("via_height")
        if via is None:
            pw, _ = self.pl.wrist(side)
            via = max(pw[2], tgt[2]) + float(sg.get("carry_rise_m", sg.get("clearance_m", 0.05)))
        R_t = self._target_R(sg, obj)
        if R_t is None:
            mujoco.mj_kinematics(self.model, self.pl.data)
            R_t = P.body_pose(self.model, self.pl.data, obj)[1]
        self.pl.carry_to(side, tgt + np.array([0, 0, float(sg.get("hover_m", 0.015))]), target_R=R_t, via_height=via, t=float(sg.get("duration_s", 1.5)), anchor=ref, label=f"{side}: carry {obj}" + (" (free)" if sg.get("arm_path") == "joint" else ""))
        # the set-down, the let-go and the retreat belong to the destination (the goal object or the world),
        # not to the carried object: that is what Stage 4 anchors them to when the layout changes
        down_s = float(sg.get("set_down_s", 1.0))
        if not np.isfinite(down_s) or down_s <= 0:
            raise ValueError("set_down_s must be positive and finite")
        self.pl.carry_to(side, tgt, t=down_s, anchor=ref, label=f"{side}: set down {obj}")
        quat = Rot.from_matrix(R_t).as_quat()
        exp = {"target_pos": tgt.tolist(), "target_axes": sg.get("target_axes", "xy"), "supported": True, "at_rest": True,
               "target_quat_wxyz": np.r_[quat[3], quat[:3]].tolist()}
        if "upright" in sg:
            exp["upright"] = bool(sg["upright"])
        if ref:
            exp["on"] = ref if sg.get("on") else None
        if sg.get("release", True):
            exp.update(self._release(side, obj, sg, anchor=ref))
        # Placement needs enough effort to support the load, but not the full
        # pushing force of the hand. A small pose error at first table contact
        # otherwise becomes a large sustained press while the fingers open.
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj)
        mass = float(self.model.body_subtreemass[bid])
        weight = mass * float(np.linalg.norm(self.model.opt.gravity))
        default_force = min(float(H.WRIST_FORCE_N), max(10., weight + 5.))
        force = float(sg.get("placement_force_n", default_force))
        if not np.isfinite(force) or not 0 < force <= H.WRIST_FORCE_N:
            raise ValueError("placement_force_n must be positive and within the configured wrist force limit")
        contact_phases = [p["label"] for p in self.pl.phases[first_phase:]
                          if any(word in p["label"] for word in ("(down)", "set down", "let go"))]
        self._rollout.setdefault("wrist_force_n", {})[side] = {"cap": [force] * 3, "phases": contact_phases}
        return exp

    def _release(self, side, obj, sg, anchor=None):
        duration = float(sg.get("duration_s", 1.3)) if sg.get("motion") == "release" else 1.3
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("release duration_s must be finite and positive")
        scale = duration / 1.3
        timing = {}
        for key, default in (("t_open", .5 * scale), ("t_retreat", .8 * scale), ("settle_s", .5)):
            value = float(sg.get(key, default))
            if not np.isfinite(value) or value < 0 or (key != "settle_s" and value == 0):
                raise ValueError(f"{key} must be finite and {'nonnegative' if key == 'settle_s' else 'positive'}")
            timing[key] = value
        B.ease_off_pin(self, side)                         # a pressing hand backs out of its press first; the pin ends
        named = sg.get("object") and sg.get("motion") in ("release", "place", "insert")
        one = B.release_one(self, side, obj, dict(sg, **timing), anchor=anchor) if named else None
        if one is not None:                                # one of several objects in the hand: only its fingers open
            if timing["settle_s"]:
                self.pl.push(f"{side}: settle after release of {obj}", 0., timing["settle_s"], side, [obj], anchor=anchor)
            return one
        g = self._grip(side)
        self.pl.release(side, g["opened"], retreat_m=float(sg.get("retreat_m", 0.08)), retreat_dir=sg.get("retreat_dir", (0, 0, 1)),
                        t_open=timing["t_open"], t_retreat=timing["t_retreat"], anchor=anchor)
        # Record real settling physics inside this segment. A downstream wait
        # cannot satisfy a release outcome that already stopped execution.
        if timing["settle_s"]:
            self.pl.push(f"{side}: settle after release of {obj}", 0., timing["settle_s"], side, [obj], anchor=anchor)
        self.state["held"][side] = None
        (self.state.get("in_hand") or {}).pop(side, None)
        if (self.state.get("together") or {}).get("bodies", {}).get(side):
            self.state.pop("together", None)
        return {"released": True, "at_rest": True}

    def m_release(self, sg, side, obj):
        obj = obj or self.state["held"].get(side)
        exp = self._release(side, obj, sg)
        exp["supported"] = True
        return exp

    def _joint(self, sg, side, obj, default_label):
        joint = sg["joint"]
        target = float(sg["target"])
        pose = self._pose(sg) if sg.get("pose") else None
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0:
            raise ValueError(f"no joint {joint!r}")
        part = self.model.body(int(self.model.jnt_bodyid[jid])).name
        if pose is not None:
            self._approach_pose(side, pose, part, approach_m=float(sg.get("approach_m", 0.06)), **self._approach_timing(sg))
        else:
            self.pl.fingers(side, self._grip(side)["press"])
        co = [s_ for s_, b in self.state["held"].items() if s_ != side and b in (part, obj)]
        follow = sg.get("follow_point")
        if follow is None and pose is not None and pose.get("points"):
            follow = np.mean([np.asarray(v, float) for v in pose["points"].values()], axis=0).tolist()   # the contact itself rides the part
        self.pl.move_joint(side, joint, target, steps=sg.get("steps"), t=float(sg.get("duration_s", 1.5)),
                           label=f"{side}: {default_label} {part}", object_body=obj or part, co_sides=co,
                           hand_follows=sg.get("hand_follows", "rigid"), follow_point=follow,
                           grip_fingers=(pose or {}).get("grip_fingers") if pose else None)
        self.state["held"][side] = part if sg.get("keep_hold", True) else None
        if sg.get("release"):
            self._release(side, part, sg)
        return {"joint": joint, "joint_target": target, "object": part, "in_hand": not sg.get("release"), "min_links": 1}

    def m_open_joint(self, sg, side, obj):
        return self._joint(sg, side, obj, "open")

    def m_close_joint(self, sg, side, obj):
        return self._joint(sg, side, obj, "close")

    def m_pull(self, sg, side, obj):
        if sg.get("joint"):
            return self._joint(sg, side, obj, "pull")
        return self._push_like(sg, side, obj, "pull")

    def m_rotate(self, sg, side, obj):
        """Turn a part on the table about the vertical by pushing one of its faces around its centre (or `pivot`):
        `angle_deg` (positive = counterclockwise seen from above), `duration_s`, `back`. A second hand may hold the
        far end as a pivot with a hold subgoal first. Measured: a flat hand pressed on a box top (press_travel_m 6 mm)
        turned it 66-79 % of the sweep (the box lags the hand); press harder or sweep past the angle you need. A push on
        a side face turns it well only near the END of a long face (the push then points into the face)."""
        pose = self._pose(sg)
        g = self._grip_from_pose(side, pose)
        m, d = self.model, self.pl.data
        ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
        mujoco.mj_forward(m, d)
        pivot = np.asarray(sg["pivot"], float) if sg.get("pivot") is not None else d.xpos[ob].copy()
        ang = float(sg["angle_deg"])
        # the press the pose carries (press_travel_m): the grip's wrist press for a one-sided contact, else along the
        # pose's approach into the part
        press = g.get("wrist_press")
        travel = float(pose.get("press_travel_m") or 0.0)
        if press is None and travel > 0:
            p_, R_ = P.pose_from_dict(pose)
            press = (-P.retreat_axis(pose, R_) * travel).tolist()
        self.pl.rotate_push(side, pose, pivot, ang, label=f"{side}: rotate {obj}", t=float(sg.get("duration_s", 2.0)),
                            back=bool(sg.get("back", True)), object_body=obj, press=press)
        return {"yaw_turned_deg": ang, "upright": bool(sg.get("upright", True))}

    def m_push(self, sg, side, obj):
        if sg.get("joint"):
            return self._joint(sg, side, obj, "push")
        return self._push_like(sg, side, obj, "push")

    def m_press(self, sg, side, obj):
        if sg.get("joint"):
            return self._joint(sg, side, obj, "press")
        return self._push_like(sg, side, obj, "press")

    def _push_like(self, sg, side, obj, word):
        pose = self._pose(sg)
        rig = self.pl.rigs[side]
        self._grip_from_pose(side, pose)
        self.pl.push_along(side, pose, sg["direction"], float(sg["distance_m"]), label=f"{side}: {word} {obj}", back=bool(sg.get("back", True)), object_body=obj)
        exp = {"displacement": {"dir": list(sg["direction"]), "min_m": float(sg.get("min_m", 0.5 * float(sg["distance_m"])))}}
        if sg.get("joint"):
            exp["joint"] = sg["joint"]
        return exp

    def m_press_seat(self, sg, side, obj):
        """Push a press-fit part home (a brick onto studs, a peg into its hole, a cap onto a marker) with the WRIST, not
        the fingers. Measured: a fingertip press moved a perched stud brick 0 mm -- the finger servos' 0.8 N m torque
        cap gives a few newtons at the pad, below the clutch -- while the wrist servo delivers up to its 40 N cap. The
        pose is a palm or a flat set of fingers on the part (flat_hand_pose contact='pad': straight fingers rest on
        their extension stops, so the load goes through the hand's structure); the fingers stay as posed. The hand
        comes down along the pose's approach onto the part, then its wrist reference goes `travel_m` past the contact
        (where the posed hand, lowered kinematically, first meets the part; the pose is first settled -- rocked about the
        line across its fingers, shifted in the press plane -- until those first contacts surround the part's centre: a
        point contact transmits no torque, so a fingertip on a stud brick's corner stud tipped it 1.9 deg and jammed it)
        along `direction` (default straight down): the servo error saturates at the wrist force cap, which is set to
        `force_n` on the wrist axis nearest the direction (default and ceiling 40 N, the existing cap: never above), so
        the part is pushed with at most that force and never further than travel_m; it holds `hold_s`, then backs off.
        The seat target is `seat_depth_m` (the part's travel), `seat_z` (its root's world z when seated) or `seat_on` +
        `seat_offset_m` (seated = that body's origin z + the offset, e.g. construct_stud_brick's seat_offset_m). The
        hand must not hold anything (a press through a grip slips: place with release first). Expect: seated (the
        part's root within seat_tol_m of the target along the press direction; default 30 % of the needed travel, between
        0.3 and 1 mm), released, at rest."""
        m, d = self.model, self.pl.data
        if not obj:
            raise ValueError("press_seat: give `object` (the part to seat)")
        ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj)
        if ob < 0:
            raise ValueError(f"press_seat: no body {obj!r}")
        if self.state["held"].get(side):
            raise ValueError(f"press_seat: the {side} hand still holds {self.state['held'][side]!r}; let go first (place with release): "
                             "a press through a grip slips")
        asked = float(sg.get("force_n", H.WRIST_FORCE_N))
        if not asked > 0:
            raise ValueError("press_seat: force_n must be > 0")
        force = min(asked, float(H.WRIST_FORCE_N))     # the wrist's own cap is the ceiling
        dvec = np.asarray(sg.get("direction", [0.0, 0.0, -1.0]), float)
        if np.linalg.norm(dvec) < 1e-9:
            raise ValueError("press_seat: direction must be a non-zero vector")
        dvec = dvec / np.linalg.norm(dvec)
        mujoco.mj_forward(m, d)
        p0 = d.xpos[ob].copy()
        if sg.get("seat_depth_m") is not None:
            depth = float(sg["seat_depth_m"])
        elif sg.get("seat_z") is not None or sg.get("seat_on"):
            if abs(dvec[2]) < 0.5:
                raise ValueError("press_seat: seat_z / seat_on need a press direction with a vertical component; give seat_depth_m")
            if sg.get("seat_on"):
                sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sg["seat_on"])
                if sb < 0:
                    raise ValueError(f"press_seat: no body {sg['seat_on']!r} for seat_on")
                zt = float(d.xpos[sb][2]) + float(sg.get("seat_offset_m", 0.0))
            else:
                zt = float(sg["seat_z"])
            depth = (zt - float(p0[2])) / float(dvec[2])
        else:
            raise ValueError("press_seat needs seat_depth_m (how far the part must travel), seat_z (its root's world z when seated) "
                             "or seat_on + seat_offset_m")
        depth = max(0.0, depth)
        if sg.get("travel_m") is not None and not 0.0 < float(sg["travel_m"]) <= SEAT_MAX_TRAVEL_M:
            raise ValueError(f"press_seat: travel_m must be in (0, {SEAT_MAX_TRAVEL_M}] m, got {sg['travel_m']}")
        pose = self._pose(sg)
        rig = self.pl.rigs[side]
        p, R = P.pose_from_dict(pose)
        g = self._grip_from_pose(side, pose)
        posed = dict(g["closed"])
        # the press goes through where the hand really meets the part: lowered kinematically onto it, and settled (rocked
        # about the line across its fingers, shifted in the press plane) until its first contacts surround the part's
        # centre, as a compliant hand settles; a point contact off the centre tips a press-fit part and jams it
        part_b = E.subtree_bodies(m, ob)
        part_g = [gg for gg in range(m.ngeom) if int(m.geom_bodyid[gg]) in part_b and (m.geom_contype[gg] or m.geom_conaffinity[gg])]
        rig_b = set(rig.body_ids)
        avoid_g = [gg for gg in range(m.ngeom) if int(m.geom_bodyid[gg]) not in part_b and int(m.geom_bodyid[gg]) not in rig_b
                   and not (m.body(int(m.geom_bodyid[gg])).name or "").startswith("robot_")
                   and (m.geom_contype[gg] or m.geom_conaffinity[gg])
                   and not (sg.get("seat_on") and int(m.geom_bodyid[gg]) in E.subtree_bodies(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sg["seat_on"])))]
        bal = None
        if part_g and sg.get("settle", True):
            bal = press_balance(m, d, rig, p, R, posed, part_g, avoid_g, dvec, d.xipos[ob].copy())
        contact_off = 0.0
        if bal is not None:
            p, R = np.asarray(bal["pos"], float), np.asarray(bal["R"], float)
            contact_off = float(bal["offset_m"])
        # the travel is counted from the first contact, not from the pose: a pose 5 mm above where its fingers meet the
        # part pushed with 23 N where 40 N was planned
        travel = float(sg["travel_m"]) if sg.get("travel_m") is not None else float(np.clip(contact_off + depth + force / WRIST_KP_REF + 0.002, 0.002, SEAT_MAX_TRAVEL_M))
        g.update({"press": posed, "wrist_press": None})          # the fingers stay as posed: the wrist presses
        if not g.get("own_open"):
            g["opened"] = posed
        self.state["grip"][side] = g
        a = P.retreat_axis(pose, R)
        approach_m = float(sg.get("approach_m", 0.05))
        self.pl.fingers(side, g["opened"]); self.pl.move_wrist(side, p + a * approach_m, R)
        self.pl.push(f"{side}: reach for {obj}", float(sg.get("t_reach", 2.0)), 0.1, side, [obj], anchor=obj)
        self.pl.move_wrist(side, p, R); self.pl.push(f"{side}: descend on {obj}", 0.8, 0.1, side, [obj], anchor=obj)
        if g["opened"] != posed:
            self.pl.fingers(side, posed); self.pl.push(f"{side}: close on {obj}", 0.4, 0.1, side, [obj], anchor=obj)
        self.pl.move_wrist(side, p + dvec * travel, R)
        self.pl.push(f"{side}: press {obj} down", float(sg.get("press_s", 1.0)), float(sg.get("hold_s", 0.8)), side, [obj], anchor=obj)
        self.pl.move_wrist(side, p + a * approach_m, R); self.pl.push(f"{side}: back off", 0.8, 0.2, side, [obj], anchor=obj)
        self.state["held"][side] = None
        # the force cap: force_n on the wrist slide axis nearest the press direction (the others keep the library cap and
        # hold the hand on line); a direction between axes caps every axis at force_n
        R0 = P.root_frame(m, d, rig)[1]
        loc = R0.T @ dvec
        k = int(np.argmax(np.abs(loc)))
        if abs(loc[k]) >= np.cos(np.radians(25.0)):
            caps = [float(H.WRIST_FORCE_N)] * 3; caps[k] = force
        else:
            caps = [force] * 3
        # during the press only: a lower cap on the way there would leave the hand lagging its reach
        self._rollout["wrist_force_n"] = {side: {"cap": caps, "phases": [f"{side}: press {obj} down"]}}
        # the seat tolerance: at most 1 mm and at most 30 % of the travel the part needs (a stud brick perches 1.7 mm high:
        # a 1.5 mm tolerance passed a press that moved it 0.7 mm)
        tol = float(sg["seat_tol_m"]) if sg.get("seat_tol_m") is not None else float(np.clip(0.3 * depth, 0.0003, 0.001))
        exp = {"seated": {"axis": dvec.round(6).tolist(), "depth_m": round(depth, 5), "target": (p0 + dvec * depth).round(6).tolist(),
                          "tol_m": round(tol, 5), "phase": f"{side}: press {obj} down"},
               "object": obj, "released": True, "at_rest": True,
               "press": {"force_n": force, "travel_m": round(travel, 5), **({"force_clamped_from_n": asked} if asked > force else {}),
                         **({"settle": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bal.items() if k not in ("pos", "R")}} if bal is not None else {})}}
        if sg.get("seat_on"):
            exp["involved"] = [sg["seat_on"]]
        if sg.get("upright", True):
            exp["upright"] = True
        return exp

    def _screw(self, sg, side, obj, unscrew):
        pose = self._pose(sg) if sg.get("pose") else None
        if pose is not None:
            self._grip_from_pose(side, pose)
        r = self.pl.screw(side, sg["yaw_joint"], sg["lift_joint"], float(sg["turns"]), float(sg["pitch_m"]), pose=pose,
                          bite_deg=float(sg.get("bite_deg", 90.0)), unscrew=unscrew, object_body=obj, t_bite=float(sg.get("t_bite", 0.8)),
                          regrasp_lift_m=float(sg.get("regrasp_lift_m", 0.006)))
        part = r["part"]                     # the body carrying the yaw joint (a thread carrier inside the bottle)
        cap = obj or part                    # the body the hand holds
        self.state["held"][side] = cap
        exp = {"joint": sg["yaw_joint"], "turned_rad": float(sg["turns"]) * 2 * np.pi, "object": cap, "in_hand": True, "involved": [part]}
        if sg.get("lift_after_m"):
            self.pl.attach(side, cap)
            pw, Rw = self.pl.wrist(side)
            self.pl.move_wrist(side, pw + np.array([0, 0, float(sg["lift_after_m"])]), Rw); self.pl.push(f"{side}: lift {cap}", 0.8, 0.3, side, [cap], anchor=cap)
            exp["rise_m"] = 0.5 * float(sg["lift_after_m"])
        return exp

    def m_unscrew(self, sg, side, obj):
        return self._screw(sg, side, obj, True)

    def m_screw(self, sg, side, obj):
        return self._screw(sg, side, obj, False)

    def _screw_bite(self, sg, side, obj, unscrew):
        """One bite of a screw motion (from expand_subgoals): the approach on the first bite, the bite, then
        the regrasp (open, swing back, close) unless it is the last."""
        first = sg.get("first", True)
        pose = self._pose(sg) if sg.get("pose") else None            # bites after the first carry the original pose as `regrasp_pose`
        regrasp_pose = None
        if not first and sg.get("regrasp_pose"):
            rp = dict(sg); rp["pose"] = sg["regrasp_pose"]
            regrasp_pose = self._pose(rp)
        if pose is not None:
            self._grip_from_pose(side, pose)
        r = self.pl.screw(side, sg["yaw_joint"], sg["lift_joint"], float(sg["bite_turns"]), float(sg["pitch_m"]), pose=pose if first else None,
                          bite_deg=float(sg.get("bite_deg", 90.0)), unscrew=unscrew, object_body=obj, t_bite=float(sg.get("t_bite", 0.8)),
                          regrasp_before=not first, bite_index=int(sg.get("step", 1)), regrasp_pose=regrasp_pose,
                          regrasp_lift_m=float(sg.get("regrasp_lift_m", 0.006)))
        part = r["part"]; cap = obj or part
        self.state["held"][side] = cap
        exp = {"joint": sg["yaw_joint"], "turned_rad": float(sg["bite_turns"]) * 2 * np.pi, "object": cap, "in_hand": True, "involved": [part]}
        if sg.get("last") and sg.get("lift_after_m"):
            self.pl.attach(side, cap)
            pw, Rw = self.pl.wrist(side)
            self.pl.move_wrist(side, pw + np.array([0, 0, float(sg["lift_after_m"])]), Rw); self.pl.push(f"{side}: lift {cap}", 0.8, 0.3, side, [cap], anchor=cap)
            exp["rise_m"] = 0.5 * float(sg["lift_after_m"]); exp["in_hand"] = True
        return exp

    def m_unscrew_bite(self, sg, side, obj):
        return self._screw_bite(sg, side, obj, True)

    def m_screw_bite(self, sg, side, obj):
        return self._screw_bite(sg, side, obj, False)

    def m_pour(self, sg, side, obj):
        return self.m_tilt(sg, side, obj)

    def m_tilt(self, sg, side, obj):
        obj = obj or self.state["held"].get(side)
        if self.state["held"].get(side) != obj:
            raise ValueError(f"tilt/pour: {side} hand does not hold {obj!r}")
        self.pl.attach(side, obj); self._couple(side, obj)
        self.pl.fingers(side, self._grip(side)["press"])
        m, d = self.model, self.pl.data
        ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, obj); osub = E.subtree_bodies(m, ob)
        ref = sg.get("over")
        if ref:
            tgt, _ = self._target_from(sg, obj)
            # bring the object's top (its mouth) over the target, not its root
            mujoco.mj_forward(m, d)
            top_z = _object_top(m, d, osub)
            tgt = tgt + np.array([0, 0, top_z - d.xpos[ob][2]]) * 0  # root target; mouth offset handled by the pivot below
            self.pl.carry_to(side, tgt, via_height=sg.get("via_height"), t=float(sg.get("duration_s", 1.5)), anchor=ref, label=f"{side}: carry {obj} over {ref}")
        mujoco.mj_forward(m, d)
        angle = float(sg.get("angle_deg", 70.0))
        axis = sg.get("axis")
        if axis is None:
            # tilt about the horizontal axis perpendicular to the direction from the object to the target (or to the hand's +x)
            if ref:
                rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ref)
                v = d.xpos[rb] - d.xpos[ob]; v[2] = 0
            else:
                v = np.array(sg.get("toward", [0, 1, 0]), float)
            v = v / max(np.linalg.norm(v), 1e-9)
            axis = np.cross(np.array([0, 0, 1.0]), v)      # mouth leans toward the target
            axis = -axis
        pivot = sg.get("pivot")
        if pivot is None:
            pivot = d.xpos[ob].copy(); pivot[2] = _object_top(m, d, osub)      # keep the mouth where it is
        keep = bool(sg.get("keep", False))           # a tilt that stays turned (a cell laid into its slot, a sprayer aimed)
        self.pl.tilt(side, axis, angle, pivot=pivot, t=float(sg.get("tilt_s", 1.2)), hold_s=float(sg.get("hold_s", 1.5)),
                     label=f"{side}: pour {obj}" if sg.get("motion") == "pour" else f"{side}: tilt {obj}", return_level=not keep)
        return {"tilt_deg": angle, "in_hand": True}

    def m_cut(self, sg, side, obj):
        """Cut a rope with scissors this hand holds, or just shut them. The blades are closed by the FINGERS: the
        gripping fingers of the holding grip are driven `close_depth_rad` (default 0.6) further toward the thumb, which
        squeezes the two rings together (measured on the Sharpa hand with the thumb outside one ring and index + middle
        outside the other: 30 deg -> shut in 0.5 s, 3-18 N at the rings), or to explicit `finger_targets_close`. With a
        `join` (the rope join's equality name, scene_recon cut_joins[].eq) or a `cut_point` (world xyz) the held scissors
        are first carried, keeping their orientation, so the open blades come around the rope from beyond the tips
        (`approach_m` past them, default 0.03) and the join sits on the blades' bisector at `cut_s_frac` of the blade
        length (default 0.5), in their plane; `position: false` skips that. The cut itself is not planned: the
        executor's cut rule switches the join off only when scissors.try_sever measures the blades closing on it.
        Expectations: severed (the join, or any rope join without one), the scissors still in hand; without a join or
        cut_point the scissors must shut (joint_reached on the hinge's closed end). `joint` names the hinge (default:
        the hinge in the held tool), `close_s` / `hold_s` the close and the dwell (defaults 1.2, 0.6), `via` world
        positions of the pivot visited before the approach.

        Measured on the bench (Sharpa hand, the scissor-cutting sample's pair): a grip whose PALM lies over the blades
        (the thumb-on-the-upper-ring pinch comes from the blade side) pushes a vertical strand ahead of it to the tips,
        where the blades must close below ~10 deg to bite an 8 mm cord; and pressing the fingers a fixed angle deeper
        rolls the pads over the ring tubes and tilts the tool ~10 deg out of the cut plane. Closing targets solved from
        the tool's closed geometry (finger IK of each pad to its ring contact rotated shut) keep the plane flat
        (`finger_targets_close`)."""
        m, d = self.model, self.pl.data
        held = self.state["held"].get(side)
        tool = obj or held
        if not tool or held is None:
            raise ValueError(f"cut: the {side} hand holds nothing (held: {self.state['held']}); grasp the scissors first")
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tool)
        if tb < 0:
            raise ValueError(f"cut: no body {tool!r}")
        jid = P.free_joint_of(m, tool)
        root = m.body(int(m.jnt_bodyid[jid])).name if jid is not None else tool
        rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root)
        if sg.get("joint"):
            hj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, sg["joint"])
        else:
            sub = E.subtree_bodies(m, rb)
            hj = next((j for j in range(m.njnt) if int(m.jnt_bodyid[j]) in sub and int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)), -1)
        if hj < 0:
            raise ValueError(f"cut: no hinge in {root!r} (give `joint`)")
        hinge = m.joint(hj).name; ha = int(m.jnt_qposadr[hj]); ub = int(m.jnt_bodyid[hj])
        lo_, hi_ = (float(v) for v in m.jnt_range[hj])
        closed_q = lo_ if bool(m.jnt_limited[hj]) else 0.0
        self.pl.attach(side, root)
        mujoco.mj_forward(m, d)
        # the rope join: where it is now, and the rope bodies it belongs to (involved, not bystanders)
        join, J, rope = sg.get("join"), None, []
        if join:
            e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, join)
            if e < 0:
                raise ValueError(f"cut: no equality {join!r} (the rope's cut_joins[].eq in scene_recon.json)")
            b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            J = d.xpos[b1] + d.xmat[b1].reshape(3, 3) @ m.eq_data[e][0:3]
            for b in (b1, b2):
                fj = P.free_joint_of(m, m.body(b).name)
                rope.append(m.body(int(m.jnt_bodyid[fj])).name if fj is not None else m.body(b).name)
        elif sg.get("cut_point") is not None:
            J = np.asarray(sg["cut_point"], float)
        rope_lab = (sg.get("rope") or (rope[0] if rope else "the rope"))
        if J is not None and sg.get("position", True):
            # the blades' frame now: pivot = the tool root's origin, each blade along its body's +x, bisector between
            Rl, Ru = d.xmat[rb].reshape(3, 3), d.xmat[ub].reshape(3, 3)
            bis = Rl[:, 0] + Ru[:, 0]; bis /= max(np.linalg.norm(bis), 1e-9)
            L = float(sg.get("blade_length_m") or _blade_length(m, rb))
            s = float(sg.get("cut_s_frac", 0.5)) * L
            target = np.asarray(J, float) - s * bis
            pre = np.asarray(J, float) - (L + float(sg.get("approach_m", 0.03))) * bis
            for i, v in enumerate(sg.get("via") or []):
                self.pl.carry_to(side, np.asarray(v, float), t=float(sg.get("via_s", 1.0)), anchor=None, label=f"{side}: carry {root} ({i + 1})")
            self.pl.carry_to(side, pre, t=float(sg.get("duration_s", 1.5)), anchor=rope[0] if rope else None,
                             label=f"{side}: bring the open blades beside {rope_lab}")
            self.pl.carry_to(side, target, t=float(sg.get("slide_s", 1.2)), anchor=rope[0] if rope else None,
                             label=f"{side}: slide {rope_lab} into the blades")
        g = self._grip(side)
        rig = self.pl.rigs[side]
        if sg.get("finger_targets_close"):
            close_t = {k: float(v) for k, v in sg["finger_targets_close"].items() if k in rig.finger_range}
            close_t = {**g["press"], **close_t}
        else:
            close_t = H.press_targets(rig, g["press"], g.get("grip_fingers") or [], depth_rad=float(sg.get("close_depth_rad", 0.6)))
        self.pl.fingers(side, close_t)
        d.qpos[ha] = closed_q                     # the plan shows them shut; the physics decides where they really stop
        self.pl.push(f"{side}: close the blades" + (f" on {rope_lab}" if J is not None else ""), float(sg.get("close_s", 1.2)),
                     float(sg.get("hold_s", 0.6)), side, [root] + rope, anchor=rope[0] if rope else root)
        g = dict(g, press=close_t)                # a hand that has shut the scissors keeps squeezing them shut
        self.state["grip"][side] = g
        exp = {"object": root, "in_hand": True, "min_links": 1, "involved": rope}
        if J is not None:
            exp["severed"] = [join] if join else True
            exp["joint"] = hinge                  # recorded: how far the blades closed
        else:
            exp.update({"joint": hinge, "joint_target": closed_q})
        return exp

    def m_regrasp(self, sg, side, obj):
        if self.state["held"].get(side):
            self._release(side, self.state["held"][side], sg)
        return self.m_grasp_lift(sg, side, obj)

    def m_home(self, sg, side, obj):
        home = np.asarray(sg["home_qpos"], float) if sg.get("home_qpos") is not None else (self.model.key_qpos[0] if self.model.nkey else self.model.qpos0)
        if self.state["held"].get(side):
            self._release(side, self.state["held"][side], sg)
        self.pl.go_home(side, home, t=float(sg.get("duration_s", 1.5)))
        return {"object": None}

    def m_wait(self, sg, side, obj):
        self.pl.push(f"{side or 'both'}: wait", 0.0, float(sg.get("duration_s", 0.5)), side, [])
        return {"object": None}

    def m_other(self, sg, side, obj):
        return self.m_wait(sg, side, obj)

    def m_track_reference(self, sg, side, obj):
        """Follow a dense tracking reference (common/tracking.py; the output of Stage 3's method modules): every row of
        `reference` in `frames` [first, last] becomes a keyframe one frame after the previous (transit 1, hold 1: one
        executed frame per row; a row farther than the joint-rate ceilings allows is stretched as any leg is). The
        first row is reached from the current state in `approach_s` (default from the wrists' distance, 1/30 to 3 s).
        The rows set the hands' wrists and fingers and the planned object poses; the arm columns are left to the arm
        coupling, unless the entry says arms: 'reference' (the rows' arms hold their hands: an executed attempt's or a
        method's refined commands), when the executor runs the rows' arms as given (tracking.couple_as_given) and the
        rows one per frame without stretching (they are already timed). Expected (tracking.tracking_expect): the entry's object (else the part the reference moves most)
        ends where the reference ends it, turned like it when the reference turns it more than 20 deg (an in-hand
        rotation), its articulated joint at the reference's end value; the other parts the reference moves are
        involved, the rest untouched. The entry's `expect` adds the expectations of the motion the rows came from."""
        from tool_library.common import tracking as TR
        ref = TR.load_reference(sg["reference"], self.model)
        Q, labels = ref["qpos"], ref["labels"]
        if abs(ref["fps"] - self.pl.fps) > 1e-6:
            raise ValueError(f"track_reference: the reference runs at {ref['fps']} fps, the executor at {self.pl.fps}")
        a, b = (int(v) for v in (sg.get("frames") or (0, len(Q) - 1)))
        if not 0 <= a <= b < len(Q):
            raise ValueError(f"track_reference: frames {[a, b]} outside the reference's {len(Q)} rows")
        given = sg.get("arms") == "reference"         # the rows' arms hold their hands: the executor runs them as given
        cols = np.arange(self.model.nq) if given else TR.planning_columns(self.model)
        if given:
            self._rollout["arms"] = "reference"
        d = self.pl.data
        mujoco.mj_forward(self.model, d)
        hands = [s_ for s_ in self.pl.rigs]
        wrist = lambda q, r: np.asarray(q)[[r.wrist_qpos[k] for k in ("x", "y", "z")]]
        dist = max([float(np.linalg.norm(wrist(Q[a], self.pl.rigs[s_]) - wrist(d.qpos, self.pl.rigs[s_]))) for s_ in hands] or [0.0])
        approach = float(sg["approach_s"]) if sg.get("approach_s") is not None else float(np.clip(dist / 0.15, 1.0 / self.pl.fps, 3.0))
        n0 = len(self.pl.phases)
        for k in range(a, b + 1):
            d.qpos[cols] = Q[k, cols]
            for s_, r in self.pl.rigs.items():
                self.pl.prev_euler[s_] = np.array([d.qpos[r.wrist_qpos[x]] for x in ("rx", "ry", "rz")])
            self.pl.push(labels[k], approach if k == a else 1.0 / self.pl.fps, 0.0, side if side in self.pl.rigs else None, [obj] if obj else [])
        # one phase per run of a label (the rows are frames, not phases)
        merged = self.pl.phases[:n0]
        for p in self.pl.phases[n0:]:
            if merged[n0:] and merged[-1]["label"] == p["label"]:
                continue
            merged.append(p)
        self.pl.phases[:] = merged
        expect = TR.tracking_expect(self.model, Q, (a, b), obj=obj)
        expect["tracking"] = {"reference": ref["path"], "frames": [a, b], "source": ref["source"]}
        return expect

    # ------------------------------------------------------------------ two hands, two parts, one rollout
    def run_both(self, sg):
        """A plan entry {motion: 'both', hands: {left: {...entry...}, right: {...entry...}}, sync}: a human's simultaneous
        two-hand action on two DIFFERENT parts (pull two rings apart, set two bricks down side by side at once). Each
        hand's entry is planned on its own from the same start state (its own pose, re-anchoring and grip memory), then
        the two plans are laid on one time line so both hands move in ONE rollout: `sync` 'start' (default; both begin
        together, the shorter plan holds its last pose), 'end' (both finish together, the shorter waits at its start) or
        'keyframes' (the k-th keyframes of the two plans are reached together; needs equal keyframe counts). Each frame's
        label joins the two hands' phases ('left: lift a | right: lift b'). The expectations are each hand's own
        (`hands`), plus the bystanders neither plan touches (`untouched`) and `hands_apart` (the hands never touch each
        other); `hand_clearance_m` (default 0.005) is the planned distance the executor enforces between the hands
        before any physics. The two hands may not act on the same part: a two-hand action on one part is a hold
        followed by the acting hand's entry (which couples them)."""
        hands = sg.get("hands")
        if not isinstance(hands, dict) or sorted(hands) != list(SIDES):
            raise ValueError(f"subgoal {sg.get('index')}: motion 'both' needs `hands` with one entry for 'left' and one for 'right', "
                             f"got {sorted(hands) if isinstance(hands, dict) else hands!r}")
        for s_ in SIDES:
            if s_ not in self.pl.rigs:
                raise ValueError(f"subgoal {sg.get('index')}: motion 'both' needs a {s_} hand in the scene")
        sync = str(sg.get("sync", "start"))
        if sync not in ("start", "end", "keyframes"):
            raise ValueError(f"subgoal {sg.get('index')}: sync must be 'start', 'end' or 'keyframes', got {sync!r}")
        m = self.model
        mujoco.mj_forward(m, self.pl.data)
        q0 = self.pl.data.qpos.copy()
        res, subs = {}, {}
        for s_ in SIDES:
            sub = dict(hands[s_])
            if (sub.get("side") or sub.get("acting_hand") or s_) != s_:
                raise ValueError(f"both: the '{s_}' entry names side {sub.get('side') or sub.get('acting_hand')!r}")
            if sub.get("motion") in (None, "both"):
                raise ValueError(f"both: the '{s_}' entry needs its own motion (grasp_lift, carry, place, hold, press_seat ...)")
            sub["side"] = s_
            sub.setdefault("index", sg.get("index"))
            sub["_in_both"] = True                          # the pins in force are checked once, on the merged entry
            sp = SubgoalPlanner(m, q0, state=copy.deepcopy(self.state), fps=self.pl.fps, rates=self.rates)
            res[s_] = sp.run(sub); subs[s_] = sub
        objs = {s_: res[s_]["expect"].get("object") or subs[s_].get("object") for s_ in SIDES}
        acted = {s_: {b for b in (objs[s_], res[s_]["state"]["held"].get(s_)) if b} for s_ in SIDES}
        same = acted["left"] & acted["right"]
        if same:
            raise ValueError(f"both: the two hands act on the same part {sorted(same)}; a two-hand action on one part is a hold entry "
                             "followed by the acting hand's entry (the executor couples the hands that hold it)")
        # the qpos each plan moves: its hand and the parts it carries, turns or presses (whole joints: a free body's 7)
        owned = {}
        for s_ in SIDES:
            K = np.array([k["qpos"] for k in res[s_]["keyframes"]], float)
            moved = set(np.nonzero(np.abs(K - q0[None]).max(axis=0) > 1e-9)[0].tolist())
            owned[s_] = sorted(_whole_joints(m, moved | set(E.hand_qadr(m, self.pl.rigs[s_]))))
        clash = set(owned["left"]) & set(owned["right"])
        if clash:
            names = E.joint_names(m)
            raise ValueError(f"both: both hands' plans move {sorted({names[a].split('[')[0] for a in clash})[:4]}: the parts the two hands "
                             "act on must be separate (a part resting on the other hand's part rides with it)")
        merged, phases = (self._merge_keyframes(q0, res, owned) if sync == "keyframes" else self._merge_dense(q0, res, owned, sync))
        expS = {s_: dict(res[s_]["expect"]) for s_ in SIDES}
        unt_r = set(expS["right"].pop("untouched", None) or [])
        unt = [b for b in (expS["left"].pop("untouched", None) or []) if b in unt_r]
        involved = []
        for s_ in SIDES:
            involved += [b for b in [objs[s_]] + list(expS[s_].get("involved") or []) if b and b not in involved]
        exp = {"hands": expS, "sync": sync, "objects": objs, "involved": involved, "untouched": unt, "hands_apart": True,
               "hand_clearance_m": float(sg.get("hand_clearance_m", 0.005))}
        for k in ("pos_tol_m", "max_hand_pen_m", "max_obj_pen_m", "max_track_m", "fraction", "slip_tol_m", "stay_tol_m"):
            if k in sg:
                exp[k] = sg[k]
        B.add_pin_expect(self, sg, exp, B.acting_sides(sg))
        exp = HC.merge_motion_expectations(exp, sg.get("expect"))
        st = copy.deepcopy(self.state)
        for s_ in SIDES:
            rs = res[s_]["state"]
            st["held"][s_] = (rs.get("held") or {}).get(s_)
            if (rs.get("grip") or {}).get(s_) is not None:
                st["grip"][s_] = rs["grip"][s_]
            for key in ("pinned", "in_hand"):              # each hand's own pin and objects come from its own plan
                if s_ in (rs.get(key) or {}):
                    rec = copy.deepcopy(rs[key][s_])
                    old_ = (st.get(key) or {}).get(s_)
                    if key == "pinned" and isinstance(old_, dict) and old_.get("body") == rec.get("body") and old_.get("pos") is not None:
                        rec["pos"], rec["quat"] = old_["pos"], old_["quat"]      # the pin's reference pose is kept
                    st.setdefault(key, {})[s_] = rec
                elif s_ in (st.get(key) or {}):
                    st[key].pop(s_)
            for r_ in rs.get("reanchored", []) or []:
                if r_ not in st.get("reanchored", []):
                    st.setdefault("reanchored", []).append(r_)
        self.state = st
        self.pl.keyframes = merged; self.pl.phases = phases
        self.pl.data.qpos[:] = np.asarray(merged[-1]["qpos"], float); mujoco.mj_forward(m, self.pl.data)
        out = {"keyframes": merged, "phases": phases, "expect": exp, "state": self.state,
               "side_keyframes": {s_: res[s_]["keyframes"] for s_ in SIDES}}
        caps = {}
        for s_ in SIDES:
            wf = (res[s_].get("rollout") or {}).get("wrist_force_n")
            if isinstance(wf, dict):
                caps.update({k: v for k, v in wf.items() if k == s_})
            elif wf is not None:
                caps[s_] = wf
        if caps:
            out["rollout"] = {"wrist_force_n": caps}
        return out

    @staticmethod
    def _side_phase(phases, kf, side):
        """The phase a side's keyframe kf belongs to (keyframe 0 is the start state, labelled as the first phase)."""
        best = None
        for p_ in phases:
            if p_.get("keyframe", 0) <= max(kf, 1):
                best = p_
        best = best or (phases[0] if phases else {})
        return {"hand": side, "objects": list(best.get("objects") or []), "anchor": best.get("anchor")}

    def _merge_keyframes(self, q0, res, owned):
        """sync 'keyframes': keyframe k of the merged plan = both hands' keyframe k; each leg takes the longer of the two."""
        KL, KR = res["left"]["keyframes"], res["right"]["keyframes"]
        if len(KL) != len(KR):
            raise ValueError(f"both: sync 'keyframes' needs the same number of keyframes on both hands (left {len(KL)}, right {len(KR)}); "
                             "use sync 'start' or 'end'")
        merged, phases = [], []
        for i, (a, b) in enumerate(zip(KL, KR)):
            q = q0.copy()
            q[owned["left"]] = np.asarray(a["qpos"], float)[owned["left"]]
            q[owned["right"]] = np.asarray(b["qpos"], float)[owned["right"]]
            lab = E.combine_label({"left": a["label"], "right": b["label"]})
            merged.append({"qpos": q.tolist(), "transit": max(int(a.get("transit", 0)), int(b.get("transit", 0))) if i else 0,
                           "hold": max(int(a.get("hold", 1)), int(b.get("hold", 1))), "label": lab})
            if i == 1 or (i > 1 and lab != merged[-2]["label"]):
                for s_ in SIDES:
                    phases.append({"label": lab, **self._side_phase(res[s_]["phases"], i, s_), "keyframe": i})
        return merged, phases

    def _merge_dense(self, q0, res, owned, sync):
        """sync 'start' / 'end': each hand's plan interpolated on its own (the executor's quintic legs and joint-rate
        ceilings), the two frame streams aligned at their start or end, merged frame by frame and written as one keyframe
        per frame (runs of identical frames as holds), which chain_subgoals reproduces exactly."""
        from tool_library.skills.chain_subgoals import chain_subgoals
        m = self.model
        stream = {}
        for s_ in SIDES:
            K = res[s_]["keyframes"]
            r = chain_subgoals(m, K, fps=self.pl.fps, max_rate_deg_s=self.rates[0], max_rate_m_s=self.rates[1])
            n = len(r["qpos"]); kf = np.zeros(n, int); prev_end = -1
            for i, st in enumerate(r["keyframe_frames"]):
                end = int(st) + max(1, int(K[i].get("hold", 1))) - 1
                kf[prev_end + 1:end + 1] = i; prev_end = end
            stream[s_] = (np.asarray(r["qpos"], float), list(r["labels"]), kf)
        T = max(len(v[0]) for v in stream.values())
        Q = np.repeat(q0[None], T, axis=0)
        labs = {s_: [None] * T for s_ in SIDES}; info = {s_: [None] * T for s_ in SIDES}
        for s_ in SIDES:
            Qs, Ls, kf = stream[s_]; n = len(Qs); pad = T - n
            idx = (list(range(n)) + [n - 1] * pad) if sync == "start" else ([0] * pad + list(range(n)))
            wait = ([False] * n + [True] * pad) if sync == "start" else ([True] * pad + [False] * n)
            Q[:, owned[s_]] = Qs[np.asarray(idx)][:, owned[s_]]
            held0 = (self.state.get("held") or {}).get(s_)
            for t in range(T):
                # a hand waiting for the other keeps its anchor: before its plan starts, what it holds (or the world); after
                # its plan ends, its last phase's (Stage 4 may change a hand's anchor only through a blend)
                if wait[t] and sync == "end":
                    ph = {"hand": s_, "objects": [held0] if held0 else [], "anchor": held0}
                else:
                    ph = self._side_phase(res[s_]["phases"], int(kf[idx[t]]), s_)
                labs[s_][t] = f"{s_}: wait" if wait[t] else Ls[idx[t]]
                info[s_][t] = ph
        comb = [E.combine_label({s_: labs[s_][t] for s_ in SIDES}) for t in range(T)]
        merged, phases = [], []
        t = 0
        while t < T:
            u = t
            while u + 1 < T and comb[u + 1] == comb[t] and np.array_equal(Q[u + 1], Q[t]):
                u += 1
            merged.append({"qpos": Q[t].tolist(), "transit": 1 if merged else 0, "hold": u - t + 1, "label": comb[t]})
            if t == 0 or comb[t] != comb[t - 1]:
                for s_ in SIDES:
                    phases.append({"label": comb[t], **info[s_][t], "keyframe": len(merged) - 1})
            t = u + 1
        return merged, phases


# thin and flat parts (common/nonprehensile.py): slide, edge_pinch (overhang_grasp), tilt_up (pivot), hook, strip_pinch
from tool_library.common import nonprehensile as _NP  # noqa: E402
MOTIONS = MOTIONS + tuple(m for m in _NP.install(SubgoalPlanner) if m not in MOTIONS)
# a cable on the table (common/ropemoves.py): lay (lift a link clear of the strands, set it down over them), drag (slide it)
from tool_library.common import ropemoves as _RM  # noqa: E402
MOTIONS = MOTIONS + tuple(m for m in _RM.install(SubgoalPlanner) if m not in MOTIONS)
