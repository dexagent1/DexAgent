"""Shared engine of the grasp robustness verifiers: closing rotation, pose noise, mass and friction margins, carry
profile replay, torque disturbance, contact opposition, clean release, support dependence and two-hand clearance.

Every rollout here drives the hand the way the Stage 3 executor drives it (SubgoalPlanner._approach_pose, _lift and
_release in common/subgoal.py), so a verdict in the hand scene means the same on the bench:

* the hand scene is hand.open_scene: finger servos capped at 0.8 N m with a 0.05 kg m^2 reflected inertia, wrist
  servos capped at 40 N / 3 N m, explicit hand-object pairs with elliptic friction cones and no-slip iterations, the
  floating hand gravity-compensated as the bench arm carries it;
* the open hand starts on its pre-grasp (approach_m back along the executor's retreat axis, plan.retreat_axis) with the
  executor's open shape (the pose's own finger_targets_open, else open_targets at open_frac, narrowed until the fingers
  clear the floor), visits the pose's approach_waypoints (0.6 s each), descends in 0.8 s, closes to the pose's closed targets in 0.5 s (+0.2 s), then to the
  executor's clamp in 0.3 s (+0.3 s): press_targets past touch (torque-capped, so the clamp force is the cap), or for a
  one-sided grip (no thumb) a few millimetres of pad travel, with the wrist leaning in when the pads would curl away;
* every leg is a quintic ease on the wrist joints and finger targets, stretched to the executor's 90 deg/s joint-rate
  ceiling, as chain_subgoals interpolates keyframes;
* the hand touches what it touches on the bench: its pairs with visual-only object geoms are switched off (the bench
  pairs it with collision geoms only), the floor gets the bench table's stiffer contact, and the hand scene runs at the
  bench scene's own timestep when it names one.

The wrist frame used for drift is the hand body that carries the rot_z joint (the frame the fingers are mounted in);
hand.object_in_wrist reads the translation base, which never rotates, so it measures a world-axes offset instead.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import hand as H
from tool_library.common import plan as P

REPO = Path(__file__).resolve().parents[2]
REBASE_RY_DEG = 60.0       # the executor re-bases the wrist chain past this pitch (episode.REBASE_RY_DEG)
T_DESCEND, T_DESCEND_HOLD = 0.8, 0.1
T_CLOSE, T_CLOSE_HOLD = 0.5, 0.2
T_PRESS, T_PRESS_HOLD = 0.3, 0.3
T_LIFT, T_LIFT_HOLD = 0.8, 0.3
T_OPEN, T_OPEN_HOLD = 0.5, 0.3
T_RETREAT, T_RETREAT_HOLD = 0.8, 0.2


def default_hand_xml(side: str) -> Path:
    return REPO / "setup" / "robot" / f"sharpa_{side}.xml"


def smoothstep(u):
    u = float(np.clip(u, 0.0, 1.0))
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def rot_deg(Ra, Rb) -> float:
    """Geodesic angle between two rotation matrices, degrees."""
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(np.asarray(Ra, float).T @ np.asarray(Rb, float)).as_rotvec())))


def r2(x, n=2):
    return None if x is None else round(float(x), n)


def complete(report: dict, keys) -> dict:
    """The report with every key a verifier documents present (None when this run did not measure it: a grasp that
    never lifted has no slip, a part on a fixed asset has no lift), so a caller can always index it."""
    out = {k: None for k in keys}
    out.update(report)
    out.setdefault("note", None)
    return out


# ----------------------------------------------------------------------------- inputs
def load_scene(scene, side=None, hand_xml=None, grasp_body=None) -> dict:
    """A hand-scene description from a dict, a JSON file (hand_scene_from_recon's out_json, also {"scene": {...}}), or
    an objects MJCF (.xml: one asset, its first free body is grasped, the hand is setup/robot/sharpa_<side>.xml)."""
    if isinstance(scene, dict):
        spec = copy.deepcopy(scene)
    elif isinstance(scene, (str, Path)):
        p = Path(scene)
        if not p.exists():
            raise ValueError(f"scene: {p} does not exist")
        if p.suffix.lower() == ".xml":
            sd = side or "right"
            spec = {"hand_xml": str(hand_xml or default_hand_xml(sd)), "side": sd, "floor": True,
                    "objects": [{"name": p.stem, "xml": str(p.resolve()), "pos": [0.0, 0.0, 0.0]}]}
        else:
            try:
                spec = H.load_scene_spec(p)
            except json.JSONDecodeError as e:
                raise ValueError(f"scene: {p} is not a hand-scene JSON ({e})") from None
    else:
        raise ValueError("scene must be a hand-scene dict, a hand-scene JSON path, or an objects MJCF path")
    if not isinstance(spec, dict) or not spec.get("objects"):
        raise ValueError("scene: the hand scene needs a non-empty 'objects' list")
    if side and not spec.get("side"):
        spec["side"] = side
    if hand_xml:
        spec["hand_xml"] = str(hand_xml)
    if not spec.get("hand_xml"):
        spec["hand_xml"] = str(default_hand_xml(spec.get("side", "right")))
    if not Path(spec["hand_xml"]).exists():
        raise ValueError(f"scene: hand_xml {spec['hand_xml']} does not exist")
    if grasp_body:
        spec["grasp_body"] = grasp_body
    if "timestep" not in spec and spec.get("bench_scene") and Path(spec["bench_scene"]).exists():
        # the bench's own integration step (a screw-thread bench runs at 0.25 ms, a latch box at 1 ms): the same contact
        # physics as the bench, not the hand scene's 2 ms default
        import re
        m = re.search(r'<option\b[^>]*\btimestep="([^"]+)"', Path(spec["bench_scene"]).read_text())
        if m:
            spec["timestep"] = float(m.group(1))
    return spec


def load_pose(pose, name="pose") -> dict:
    """A grasp pose from a dict or a JSON file ({pos, euler, finger_targets, ...}; a ranked list or {"poses": [...]} gives
    its first entry, {"pose": {...}} its inner pose), as the executor's subgoal.load_pose reads it."""
    d = pose
    if isinstance(pose, (str, Path)):
        p = Path(pose)
        if not p.exists():
            raise ValueError(f"{name}: {p} does not exist")
        d = json.loads(p.read_text())
    if isinstance(d, dict) and "pos" not in d and isinstance(d.get("poses"), list):
        d = d["poses"]
    if isinstance(d, list):
        d = d[0] if d else None
    if isinstance(d, dict) and "pos" not in d and isinstance(d.get("pose"), dict):
        d = d["pose"]
    if not isinstance(d, dict) or d.get("pos") is None or d.get("euler") is None:
        raise ValueError(f"{name}: needs 'pos' and 'euler' (a pose from the grasp skills)")
    if not d.get("finger_targets"):
        raise ValueError(f"{name}: needs 'finger_targets' (the closed finger joint targets)")
    return copy.deepcopy(d)


def disable_pair(model, k):
    """Switch an explicit contact pair off on a compiled model. There is no enable flag, and neither knob alone is enough:
    a negative margin stops plane, box and capsule pairs but a mesh-mesh pair still reports a penetrating contact, and
    a large gap does not stop a plane-mesh pair (both measured with MuJoCo 3.10). With the margin below any distance and
    the impedance at its floor the pair can push with no more than 1e-4 of a normal contact, frictionless."""
    model.pair_margin[k] = -1.0
    model.pair_gap[k] = 0.0
    model.pair_solimp[k] = [1e-4, 1e-4, 0.001, 0.5, 2.0]
    model.pair_friction[k] = [0.0, 0.0, 0.0, 0.0, 0.0]


def subtree(model, bid) -> set:
    out, stack = set(), [int(bid)]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
    return out


def free_root(model, bid):
    """The body carrying the free joint at or above `bid` (what a lift carries), else None (a part on a fixed asset)."""
    b = int(bid)
    while b > 0:
        for j in range(model.njnt):
            if int(model.jnt_bodyid[j]) == b and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                return b
        b = int(model.body_parentid[b])
    return None


# ----------------------------------------------------------------------------- the case: one hand scene
class Case:
    """One compiled hand scene with the grasped body and the wrist frame resolved."""

    def __init__(self, spec: dict, bench_pairs=True, **over):
        self.spec = spec
        self.model, self.data0, self.rig, self.ids = H.open_scene(spec, **over)
        self.bid = int(H.grasp_body(self.model, spec, self.ids))
        self.root = free_root(self.model, self.bid)
        self.dt = float(self.model.opt.timestep)
        m, rig = self.model, self.rig
        jz = int(m.actuator_trnid[rig.wrist_act["rz"]][0])
        jx = int(m.actuator_trnid[rig.wrist_act["rx"]][0])
        self.wrist_body = int(m.jnt_bodyid[jz])
        self.rx_body = int(m.jnt_bodyid[jx])
        self.B = np.eye(3)
        self.lifted = subtree(m, self.root) if self.root is not None else subtree(m, self.bid)
        # free bodies welded onto what the hand lifts ride with it: a cap welded onto its bottle's thread carrier is part
        # of the bottle the hand wraps (the thumb may press on the cap). A body welded the other way round (the carrier a
        # grasped cap is welded to) is not free, so a grasped cap does not take its bottle with it.
        self.attached = set()
        if self.root is not None:
            for e in range(m.neq):
                if int(m.eq_type[e]) != int(mujoco.mjtEq.mjEQ_WELD) or not self.data0.eq_active[e]:
                    continue
                b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
                for a, b in ((b1, b2), (b2, b1)):
                    if a in self.lifted and b not in self.lifted and b > 0 and b not in self.rig.body_ids and free_root(m, b) == b:
                        self.attached |= subtree(m, b)
            self.lifted |= self.attached
        self.lifted_gids = {g for g in self.ids["obj_gids"] if int(m.geom_bodyid[g]) in self.lifted}
        self.mass = float(sum(m.body_mass[b] for b in self.lifted))
        self.n_visual_pairs_off = self._bench_pairs() if bench_pairs else 0
        if bench_pairs:
            self._bench_table()
        # what the hand can touch on the grasped part (and what is welded onto it): its collision geoms (all of its geoms
        # when it has none)
        part = subtree(m, self.bid) | self.attached
        coll = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in part and (m.geom_contype[g] or m.geom_conaffinity[g])}
        self.touch_gids = coll or {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in part}

    def _bench_table(self, settle_s=0.4):
        """Give the floor the bench table's contact (setup/scene.xml's default geom class: solref 0.005, solimp 0.99 0.999
        0.0005) and re-settle the objects on it. The hand scene's floor ships MuJoCo's softer 0.02 / 0.9 0.95: measured on an
        upright AA cell standing on its three terminal bumps, the softer floor let the bumps sink and the clamp tipped the
        cell over in the hand scene, while the same commands held it upright on the bench."""
        m, fl = self.model, self.ids.get("floor_gid", -1)
        if fl < 0:
            return
        m.geom_solref[fl] = BENCH_TABLE["solref"]
        m.geom_solimp[fl] = BENCH_TABLE["solimp"]
        m.geom_friction[fl] = BENCH_TABLE["friction"]
        d = self.data0
        held = []
        for jn, v in (self.spec.get("joint_init") or {}).items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if j >= 0 and int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                held.append((int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]), float(np.atleast_1d(v)[0])))
        for _ in range(int(settle_s / self.dt)):
            mujoco.mj_step(m, d)
            for adr, dof, v in held:
                d.qpos[adr] = v; d.qvel[dof] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def _bench_pairs(self) -> int:
        """Switch off the hand's pairs with visual-only object geoms. build_hand_scene pairs the hand with every geom under
        an object's prefix, so a hand scene grips a cup by the convex hull of its visual shell; the bench pairs the hand with
        collision geoms only (episode.build_execution_model), and that is what the executor's hand touches. Objects that
        have no collision geom at all keep their pairs."""
        m = self.model
        hand = self.rig.geom_ids
        fl = self.ids.get("floor_gid", -1)
        has_coll = {}
        for g in range(m.ngeom):
            if g in hand or g == fl:
                continue
            r = int(m.body_rootid[int(m.geom_bodyid[g])])
            has_coll[r] = has_coll.get(r, False) or bool(m.geom_contype[g] or m.geom_conaffinity[g])
        off = 0
        for k in range(m.npair):
            g1, g2 = int(m.pair_geom1[k]), int(m.pair_geom2[k])
            o = g2 if g1 in hand else (g1 if g2 in hand else None)
            if o is None or o == fl or m.geom_contype[o] or m.geom_conaffinity[o]:
                continue
            if has_coll.get(int(m.body_rootid[int(m.geom_bodyid[o])]), False):
                disable_pair(m, k)
                off += 1
        return off

    # -- wrist chain basis: the executor re-bases a segment that pitches past 60 deg (episode._wrist_rebase)
    def set_basis(self, rotations) -> bool:
        Rs = [np.asarray(R, float) for R in rotations]
        if not Rs:
            return False
        ry = max(abs(float(H.mat_to_euler(R)[1])) for R in Rs)
        m = self.model
        if np.degrees(ry) <= REBASE_RY_DEG:
            return False
        if np.abs(m.body_quat[self.rx_body] - np.array([1.0, 0, 0, 0])).max() > 1e-9 or np.abs(m.body_pos[self.rx_body]).max() > 1e-12:
            return False
        M = sum(Rs)
        U, _, Vt = np.linalg.svd(M)
        B = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, B.reshape(-1))
        m.body_quat[self.rx_body] = q
        self.B = B
        return True

    def euler(self, R, prev=None):
        e = H.mat_to_euler(self.B.T @ np.asarray(R, float))
        return H.nearest_euler_branch(e, prev) if prev is not None else e

    def fresh(self):
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data0.qpos
        d.qvel[:] = 0.0
        d.ctrl[:] = self.data0.ctrl
        mujoco.mj_forward(self.model, d)
        return d

    def place(self, d, p, R, fingers):
        """Teleport the hand (initialisation only) and tell the servos to hold it there."""
        e = self.euler(R)
        self.rig.teleport(d, p, e)
        self.rig.set_fingers(d, fingers)
        H.hold_ctrl(d, self.rig, p, e, fingers)
        mujoco.mj_forward(self.model, d)
        return e

    def wrist(self, d):
        return np.asarray(d.xpos[self.wrist_body], float).copy(), np.asarray(d.xmat[self.wrist_body], float).reshape(3, 3).copy()

    def obj(self, d, bid=None):
        b = self.bid if bid is None else bid
        return np.asarray(d.xpos[b], float).copy(), np.asarray(d.xmat[b], float).reshape(3, 3).copy()

    def obj_in_wrist(self, d):
        pw, Rw = self.wrist(d)
        po, Ro = self.obj(d)
        return Rw.T @ (po - pw), Rw.T @ Ro

    def contacts(self, d):
        return H.finger_contacts(self.model, d, self.rig, self.touch_gids)


# ----------------------------------------------------------------------------- the executor's grip
def executor_grip(case: Case, pose: dict) -> dict:
    """Closed / clamp / open finger targets as SubgoalPlanner._grip_from_pose and _floor_clear_open make them."""
    rig, m = case.rig, case.model
    c = H.executor_clamp(rig, pose)
    closed, press, grip, one_sided = c["closed"], c["press"], c["grip_fingers"], c["one_sided"]
    wrist_press = None if c["wrist_press"] is None else np.asarray(c["wrist_press"], float)
    own_open = {k: float(v) for k, v in (pose.get("finger_targets_open") or {}).items() if k in rig.finger_range}
    opened = own_open
    if not opened:
        # the widest opening whose fingers stay above the table at the pose (the executor's _floor_clear_open)
        top = float(pose["open_frac"]) if pose.get("open_frac") is not None else 0.55
        fl = case.ids.get("floor_gid", -1)
        d = case.fresh()
        for frac in [top] + [f for f in (0.45, 0.35, 0.25, 0.15) if f < top - 1e-9]:
            opened = H.approach_open_targets(rig, closed, frac, grip, c["grip_posed"])
            if fl < 0:
                break
            case.place(d, np.asarray(pose["pos"], float), H.euler_to_mat(pose["euler"]), opened)
            worst = min(H.hand_geom_distance(m, d, g, fl, 0.05) for g in rig.collision_gids)
            if worst > -0.002:
                break
    return {"closed": closed, "press": press, "opened": opened, "wrist_press": wrist_press, "grip_fingers": grip,
            "one_sided": one_sided}


# ----------------------------------------------------------------------------- driven legs
FPS = 30.0                 # the executor's plan frame rate
SEAT_WINDOW_S = 0.1        # the clamp's last 0.1 s, over which the seated grip is read
BENCH_TABLE = {"solref": [0.005, 1.0], "solimp": [0.99, 0.999, 0.0005, 0.5, 2.0], "friction": [1.0, 0.005, 0.0001]}
MAX_RATE_DEG_S = 90.0      # the executor's joint-rate ceiling (execute_subgoal_plan max_rate_deg_s, chain_subgoals)
MAX_RATE_M_S = 0.4


def leg_seconds(seconds, p0, p1, e0, e1, f0, f1):
    """A leg's duration as chain_subgoals stretches it: a quintic peaks at 1.875 x its mean rate, and no hinge may pass
    90 deg/s nor a slide 0.4 m/s (measured on the bench: a 0.5 s close of a 0.6 rad finger travel runs in 0.7 s)."""
    dh = [abs(float(b) - float(a)) for a, b in zip(e0, e1)] + [abs(float(v) - float(f0.get(k, v))) for k, v in f1.items()]
    ds = float(np.max(np.abs(np.asarray(p1, float) - np.asarray(p0, float))))
    need = 1.875 * max(max(dh, default=0.0) / np.radians(MAX_RATE_DEG_S), ds / MAX_RATE_M_S) * 1.02
    return max(float(seconds), need)


def leg(case: Case, d, p0, p1, R0, R1, f0, f1, seconds, on_step=None, e_prev=None):
    """Quintic leg on the wrist joints and the finger targets, through the servos, stretched to the executor's joint-rate
    ceiling. Returns the end Euler triple."""
    m, rig = case.model, case.rig
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    e0 = case.euler(R0, e_prev)
    e1 = case.euler(R1, e0)
    seconds = leg_seconds(seconds, p0, p1, e0, e1, f0, f1)
    # as the executor runs a plan: the quintic is sampled at the plan's frame rate (chain_subgoals) and the servo targets
    # move linearly between frames, n_sub physics steps per frame (episode.servo_rollout)
    n_sub = max(1, int(round(1.0 / (FPS * case.dt))))
    nf = max(int(round(float(seconds) * FPS)), 1)
    for j in range(nf):
        ua, ub = smoothstep(j / nf), smoothstep((j + 1) / nf)
        for k in range(n_sub):
            u = ua + (ub - ua) * (k + 1) / n_sub
            fg = {q: f0.get(q, v) + (v - f0.get(q, v)) * u for q, v in f1.items()}
            H.hold_ctrl(d, rig, p0 + (p1 - p0) * u, e0 + (e1 - e0) * u, fg)
            mujoco.mj_step(m, d)
            if on_step is not None:
                on_step(d)
    return e1


def hold(case: Case, d, p, R, fingers, seconds, on_step=None, e_prev=None):
    e = case.euler(R, e_prev)
    for _ in range(max(int(round(float(seconds) / case.dt)), 0)):
        H.hold_ctrl(d, case.rig, p, e, fingers)
        mujoco.mj_step(case.model, d)
        if on_step is not None:
            on_step(d)
    return e


def close_like_executor(case: Case, pose: dict, approach=True, approach_m=0.08, settle_s=0.3, d=None, on_close=None) -> dict:
    """Pre-grasp -> descend -> close -> clamp, as the executor drives a grasp. Tracks the grasped body from the moment
    the fingers start to close (after the open-hand descent) to the end of the clamp hold, and when each finger first
    touches it. Returns the live MjData, the grip, the wrist command the clamp holds, and the closing record."""
    m, rig = case.model, case.rig
    d = case.fresh() if d is None else d
    g = executor_grip(case, pose)
    p = np.asarray(pose["pos"], float)
    R = H.euler_to_mat(pose["euler"])
    a = P.retreat_axis(pose, R)
    if approach:
        pre = p + a * float(approach_m)
        case.place(d, pre, R, g["opened"])
        hold(case, d, pre, R, g["opened"], settle_s)
        # the pose's own approach waypoints (world offsets from the pose, 0.6 s each), as _approach_pose visits them: over a
        # bar and down behind it, in from the side of an upright cell
        cur = pre
        for off in pose.get("approach_waypoints") or []:
            nxt = p + np.asarray(off, float)
            leg(case, d, cur, nxt, R, R, g["opened"], g["opened"], 0.6)
            cur = nxt
        leg(case, d, cur, p, R, R, g["opened"], g["opened"], T_DESCEND)
    else:
        case.place(d, p, R, g["opened"])
        hold(case, d, p, R, g["opened"], settle_s)
    hold(case, d, p, R, g["opened"], T_DESCEND_HOLD)
    p_ref, R_ref = case.obj(d)
    nudge = float(np.linalg.norm(p_ref - case.obj(case.data0)[0]))
    rec = {"t": 0.0, "max_rot": 0.0, "max_shift": 0.0, "first_touch_s": {}, "series": []}
    touch_ids = {}
    for gid in rig.geom_ids:
        nm = m.geom(gid).name or ""
        for f in H.FINGERS:
            if f"_{f}_" in nm:
                touch_ids[gid] = f
    for gid in rig.geom_ids:
        if gid not in touch_ids:
            touch_ids[gid] = "palm"
    tg = case.touch_gids
    every = max(int(round(0.02 / case.dt)), 1)

    def track(dd):
        rec["t"] += case.dt
        po, Ro = case.obj(dd)
        rot, sh = rot_deg(R_ref, Ro), float(np.linalg.norm(po - p_ref))
        rec["max_rot"], rec["max_shift"] = max(rec["max_rot"], rot), max(rec["max_shift"], sh)
        k = int(round(rec["t"] / case.dt))
        if k % every == 0:
            rec["series"].append((round(rec["t"], 3), round(rot, 2), round(sh * 1000, 2)))
            for i in range(dd.ncon):
                c = dd.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                hg = g1 if g1 in touch_ids else (g2 if g2 in touch_ids else None)
                og = g2 if hg == g1 else g1
                if hg is not None and og in tg and c.dist < 0.0005:
                    rec["first_touch_s"].setdefault(touch_ids[hg], round(rec["t"], 3))
        if on_close is not None:
            on_close(dd)

    e = leg(case, d, p, p, R, R, g["opened"], g["closed"], T_CLOSE, track)
    hold(case, d, p, R, g["closed"], T_CLOSE_HOLD, track, e)
    p_cmd = p + (np.asarray(g["wrist_press"], float) if g["wrist_press"] is not None else 0.0)
    e = leg(case, d, p, p_cmd, R, R, g["closed"], g["press"], T_PRESS, track, e)
    e = hold(case, d, p_cmd, R, g["press"], T_PRESS_HOLD - SEAT_WINDOW_S, track, e)
    # the seated grip is read over the last 0.1 s of the clamp hold: a torque-capped pad chatters on and off a light object
    # from one step to the next, and a single-step reading once saw a thumb that was on the box 90% of the time as absent
    samples = []
    every = max(int(round(SEAT_WINDOW_S / 5 / case.dt)), 1)

    def seat_track(dd):
        track(dd)
        rec["n_seat"] = rec.get("n_seat", 0) + 1
        if rec["n_seat"] % every == 0:
            samples.append(case.contacts(dd))

    hold(case, d, p_cmd, R, g["press"], SEAT_WINDOW_S, seat_track, e)
    po, Ro = case.obj(d)
    return {"data": d, "grip": g, "pos": p_cmd, "R": R, "euler": e, "ref": (p_ref, R_ref), "nudge_mm": nudge * 1000,
            "floor_force_N": hand_floor_force(case, d),
            "rot_deg": rot_deg(R_ref, Ro), "shift_mm": float(np.linalg.norm(po - p_ref)) * 1000,
            "max_rot_deg": rec["max_rot"], "max_shift_mm": rec["max_shift"] * 1000,
            "first_touch_s": rec["first_touch_s"], "series": rec["series"], "seated": merge_contact_samples(samples) if samples else case.contacts(d)}


def merge_contact_samples(samples) -> dict:
    """finger_contacts readings over a short window -> the fingers on the object in at least half of them, mean force."""
    from collections import Counter
    cnt = Counter(f for s_ in samples for f in s_["fingers"])
    fingers = sorted(f for f, c in cnt.items() if c * 2 >= len(samples))
    return {"n_finger": len(fingers), "fingers": fingers, "force_N": round(float(np.mean([s_["force_N"] for s_ in samples])), 1)}


def contacts_over(case: Case, run: dict, seconds=0.1, samples=5, min_force=0.02) -> list:
    """Hand-object contacts gathered over a short hold of the clamp (contact forces divided by the number of samples, so
    sums are mean forces): a reading that does not depend on one step of a chattering pad."""
    d = run["data"]
    out = []
    every = max(int(round(float(seconds) / samples / case.dt)), 1)
    n = {"i": 0}

    def grab(dd):
        n["i"] += 1
        if n["i"] % every == 0:
            out.append(hand_object_contacts(case, dd, min_force=min_force))

    hold(case, d, run["pos"], run["R"], run["grip"]["press"], seconds, grab, run.get("euler"))
    k = max(len(out), 1)
    merged = []
    for cs in out:
        for c in cs:
            merged.append(dict(c, force=c["force"] / k))
    return merged


def lift_after(case: Case, run: dict, lift_m=0.08, seconds=T_LIFT, hold_s=T_LIFT_HOLD, on_step=None) -> dict:
    """The executor's lift: the wrist straight up by lift_m with the clamp held. Updates run["pos"]."""
    d = run["data"]
    z0 = float(d.xpos[case.bid][2])
    p1 = run["pos"] + np.array([0.0, 0.0, float(lift_m)])
    e = leg(case, d, run["pos"], p1, run["R"], run["R"], run["grip"]["press"], run["grip"]["press"], seconds, on_step, run.get("euler"))
    hold(case, d, p1, run["R"], run["grip"]["press"], hold_s, on_step, e)
    run["pos"], run["euler"] = p1, e
    rose = float(d.xpos[case.bid][2]) - z0
    return {"commanded_mm": r2(lift_m * 1000, 1), "rose_mm": r2(rose * 1000, 1), "ok": bool(rose > 0.5 * float(lift_m))}


def hold_measure(case: Case, run: dict, seconds=0.8, on_step=None) -> dict:
    """Hold the wrist still with the clamp; drift and turn of the object in the wrist frame over the hold."""
    d = run["data"]
    pa, Ra = case.obj_in_wrist(d)
    hold(case, d, run["pos"], run["R"], run["grip"]["press"], seconds, on_step, run.get("euler"))
    pb, Rb = case.obj_in_wrist(d)
    cn = case.contacts(d)
    drift = float(np.linalg.norm(pb - pa))
    return {"drift_mm": r2(drift * 1000), "turn_deg": r2(rot_deg(Ra, Rb)), "slip_mm_s": r2(drift * 1000 / max(seconds, 1e-6)),
            "fingers": cn["fingers"], "contact_force_N": cn["force_N"]}


# ----------------------------------------------------------------------------- perturbations
def perturb_pose(pose: dict, dpos, rotvec, pivot=None) -> dict:
    """The pose with the wrist translated by dpos (m) and turned by rotvec (rad, world axes) about `pivot` (default the
    wrist origin, where a calibration error of the arm's flange acts). The approach vector turns with the hand."""
    out = copy.deepcopy(pose)
    p = np.asarray(pose["pos"], float)
    Rm = H.euler_to_mat(pose["euler"])
    D = Rot.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    c = p if pivot is None else np.asarray(pivot, float)
    out["pos"] = (c + D @ (p - c) + np.asarray(dpos, float)).tolist()
    out["euler"] = H.nearest_euler_branch(H.mat_to_euler(D @ Rm), np.asarray(pose["euler"], float)).tolist()
    if pose.get("approach") is not None:
        out["approach"] = (D @ np.asarray(pose["approach"], float)).tolist()
    return out


def random_perturbations(n, pos_mm, rot_deg_, seed=0):
    """n wrist errors at the stated magnitudes in random directions, in antithetic pairs (+e, -e): a hand 2 mm too low
    and 2 mm too high behave differently, and a pair covers both."""
    rng = np.random.default_rng(int(seed))
    out = []
    while len(out) < int(n):
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        w = rng.normal(size=3); w /= np.linalg.norm(w)
        dp, dr = u * float(pos_mm) / 1000.0, w * np.radians(float(rot_deg_))
        out.append((dp, dr))
        if len(out) < int(n):
            out.append((-dp, -dr))
    return out


def scale_grasped(case: Case, mass_scale=1.0, friction_scale=1.0) -> dict:
    """Heavier and slipperier: every body the lift carries gets mass and inertia x mass_scale; the grasped geoms and
    their hand pairs get sliding and torsional friction x friction_scale."""
    m = case.model
    for b in case.lifted:
        m.body_mass[b] *= float(mass_scale)
        m.body_inertia[b] *= float(mass_scale)
    tg = set(case.touch_gids)
    for g in tg:
        m.geom_friction[g][:2] *= float(friction_scale)
    n_pairs = 0
    for k in range(m.npair):
        if int(m.pair_geom1[k]) in tg or int(m.pair_geom2[k]) in tg:
            m.pair_friction[k][:3] *= float(friction_scale)
            n_pairs += 1
    case.mass = float(sum(m.body_mass[b] for b in case.lifted))
    return {"mass_kg": r2(case.mass, 4), "pairs_scaled": n_pairs, "geoms_scaled": len(tg)}


def isolate_grasped(case: Case, d, keep_hand_floor=True, release_equalities=False) -> dict:
    """Take away everything the grasped object could rest on: the floor and every other object stop colliding with it
    (and with the hand, unless they are the floor), the other objects float in place (gravity-compensated, at rest).
    With release_equalities, equality constraints that tie the carried bodies to anything else are switched off too (a
    cap whose thread weld the bench released before the carry). Only the hand can hold the object afterwards. Works on the
    live model and data; the case is spent afterwards."""
    m = case.model
    released = []
    if release_equalities:
        for e in range(m.neq):
            b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            if int(m.eq_type[e]) in (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT)) and \
                    ((b1 in case.lifted) != (b2 in case.lifted)) and d.eq_active[e]:
                d.eq_active[e] = 0
                released.append(m.equality(e).name or f"eq{e}")
    lifted, lg = case.lifted, case.lifted_gids
    others = {g for g in case.ids["obj_gids"] if g not in lg}
    # any other object geom, including visual ones that never collided, so nothing can catch the object
    others |= {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in lifted and g not in case.rig.geom_ids and g != case.ids.get("floor_gid", -1)}
    fl = case.ids.get("floor_gid", -1)
    for g in others | ({fl} if fl >= 0 else set()):
        m.geom_contype[g] = 0
        m.geom_conaffinity[g] = 0
    off = 0
    for k in range(m.npair):
        g1, g2 = int(m.pair_geom1[k]), int(m.pair_geom2[k])
        if (g1 in others or g2 in others) or (not keep_hand_floor and fl in (g1, g2)):
            disable_pair(m, k)
            off += 1
    robot = case.rig.body_ids
    for b in range(1, m.nbody):
        if b not in lifted and b not in robot:
            m.body_gravcomp[b] = 1.0
    for j in range(m.njnt):
        b = int(m.jnt_bodyid[j])
        if b not in lifted and b not in robot:
            dof, nd = int(m.jnt_dofadr[j]), (6 if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) else
                                             3 if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_BALL) else 1)
            d.qvel[dof:dof + nd] = 0.0
    return {"pairs_disabled": off, "geoms_disabled": len(others) + (1 if fl >= 0 else 0), "equalities_released": released}


def hand_takes_weight(case: Case, run: dict, seconds=0.5, on_step=None, isolate=True, keep_hand_floor=True, release_equalities=False):
    """Hand the object's weight to the hand without a jolt: the carried bodies are made weightless (gravity-compensated),
    everything else stops touching them (isolate_grasped), and their weight is ramped back over `seconds` with the
    wrist holding still. A grasp that carries the object takes the load quasi-statically, as a slow lift off the table
    would load it; a pose that relied on the support lets the object slide out as the weight comes back. Returns the
    isolation record."""
    m, d = case.model, run["data"]
    for b in case.lifted:
        m.body_gravcomp[b] = 1.0
    iso = isolate_grasped(case, d, keep_hand_floor=keep_hand_floor, release_equalities=release_equalities) if isolate else {}
    n = max(int(round(float(seconds) / case.dt)), 1)
    e = run.get("euler")
    for i in range(n):
        g = 1.0 - (i + 1) / n
        for b in case.lifted:
            m.body_gravcomp[b] = g
        e = hold(case, d, run["pos"], run["R"], run["grip"]["press"], case.dt, on_step, e)
    for b in case.lifted:
        m.body_gravcomp[b] = 0.0
    return iso


def object_size_m(case: Case, d) -> float:
    ext = H.object_extent(case.model, d, case.lifted_gids or case.ids["grasp_gids"])
    return float(np.max(ext["size"]))


def hand_object_contacts(case: Case, d, min_force=0.05) -> list:
    return H.hand_object_contacts(case.model, d, case.rig, case.touch_gids, min_force=min_force)


def link_of_geom(name: str) -> str:
    for f in H.FINGERS:
        if f"_{f}_" in (name or ""):
            return f
    return "palm"


def hand_floor_force(case: Case, d) -> float:
    """Normal force (N) between the hand and the floor: fingers stopped on the table instead of the object."""
    fl = case.ids.get("floor_gid", -1)
    if fl < 0:
        return 0.0
    buf, tot = np.zeros(6), 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if fl not in (g1, g2) or c.exclude:
            continue
        other = g2 if g1 == fl else g1
        if other in case.rig.geom_ids:
            mujoco.mj_contactForce(case.model, d, i, buf)
            tot += abs(float(buf[0]))
    return round(tot, 2)


def support_force(case: Case, d) -> float:
    """Normal force (N) on the lifted body set from anything that is not the hand (floor, other objects)."""
    m = case.model
    lg = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in case.lifted}
    buf = np.zeros(6)
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in lg) == (g2 in lg):
            continue
        other = g2 if g1 in lg else g1
        if other in case.rig.geom_ids or c.exclude:
            continue
        mujoco.mj_contactForce(m, d, i, buf)
        tot += abs(float(buf[0]))
    return tot


# ----------------------------------------------------------------------------- wrist paths (carry replay)
def _contiguous_runs(idx):
    runs, cur = [], []
    for t in idx:
        if cur and t != cur[-1] + 1:
            runs.append(cur); cur = []
        cur.append(t)
    if cur:
        runs.append(cur)
    return runs


def _bench_wrist_poses(bench_model, Q, side):
    rigs = P.scene_rigs(bench_model)
    if side not in rigs:
        raise ValueError(f"side: the bench scene has no {side} hand (hands: {sorted(rigs)})")
    rig = rigs[side]
    d = mujoco.MjData(bench_model)
    Ps, Rs = [], []
    for q in Q:
        d.qpos[:] = q
        mujoco.mj_kinematics(bench_model, d)
        p, R = P.wrist_world(bench_model, d, rig)
        Ps.append(p); Rs.append(R)
    return np.asarray(Ps), np.asarray(Rs)


def wrist_path(carry, side, frames=None, phase="carry", subgoal=None, bench_scene=None, fps=None) -> dict:
    """World wrist path {pos [T,3], R [T,3,3], fps, frames, source} of one hand over a carry, from
    * a trajectory npz (qpos, labels, fps; optional subgoal and joint_names): `frames` [first, last], else the first
      contiguous run of frames labelled '<side>: <phase>...' (within `subgoal` when given);
    * a keyframes JSON ({"keyframes": [{qpos, transit, hold, label}]} or the list) chained with quintic legs at `fps`
      (default 30), then selected the same way;
    * a wrist-path JSON {"wrist": [[x, y, z, rx, ry, rz], ...], "fps": 30} (world pose per frame, intrinsic XYZ).
    Trajectory and keyframe qpos are read on the bench scene (`bench_scene`) when given, else by joint name."""
    src = Path(carry) if isinstance(carry, (str, Path)) else None
    if src is None and isinstance(carry, dict) and "wrist" in carry:
        obj = carry
    elif src is not None and src.suffix.lower() == ".json":
        if not src.exists():
            raise ValueError(f"carry: {src} does not exist")
        obj = json.loads(src.read_text())
    elif src is not None and src.suffix.lower() == ".npz":
        if not src.exists():
            raise ValueError(f"carry: {src} does not exist")
        obj = None
    else:
        raise ValueError("carry must be a trajectory .npz, a keyframes .json, or a wrist-path .json/dict")
    if isinstance(obj, dict) and "wrist" in obj:
        W = np.asarray(obj["wrist"], float)
        if W.ndim != 2 or W.shape[1] != 6 or len(W) < 2:
            raise ValueError("carry: 'wrist' must be a list of at least two [x, y, z, rx, ry, rz] rows")
        sel = list(range(len(W))) if frames is None else list(range(int(frames[0]), int(frames[1]) + 1))
        W = W[sel]
        return {"pos": W[:, :3], "R": np.asarray([H.euler_to_mat(e) for e in W[:, 3:]]), "fps": float(obj.get("fps", fps or 30.0)),
                "frames": [sel[0], sel[-1]], "source": "wrist path"}
    labels, subg, names = None, None, None
    if obj is None:
        z = np.load(str(src), allow_pickle=True)
        key = "qpos" if "qpos" in z.files else ("Q" if "Q" in z.files else None)
        if key is None:
            raise ValueError(f"carry: {src} has no qpos array")
        Q = np.asarray(z[key], float)
        labels = [str(x) for x in z["labels"]] if "labels" in z.files else [""] * len(Q)
        f_ = float(z["fps"]) if "fps" in z.files else 30.0
        subg = np.asarray(z["subgoal"]) if "subgoal" in z.files else None
        names = [str(x) for x in z["joint_names"]] if "joint_names" in z.files else None
        source = "trajectory"
    else:
        kfs = obj.get("keyframes") if isinstance(obj, dict) else obj
        if not isinstance(kfs, list) or not kfs or "qpos" not in kfs[0]:
            raise ValueError("carry: a keyframes JSON needs keyframes with full qpos")
        f_ = float(fps or 30.0)
        rows, labels = [], []
        prev = np.asarray(kfs[0]["qpos"], float)
        for i, kf in enumerate(kfs):
            q = np.asarray(kf["qpos"], float)
            n = int(kf.get("transit", 0)) if i else 0
            for k in range(1, n):
                rows.append(prev + smoothstep(k / n) * (q - prev)); labels.append(str(kf.get("label", "")))
            for _ in range(max(1, int(kf.get("hold", 1)))):
                rows.append(q.copy()); labels.append(str(kf.get("label", "")))
            prev = q
        Q = np.asarray(rows)
        source = "keyframes"
    if fps is not None:
        f_ = float(fps)
    if frames is not None:
        a, b = int(frames[0]), int(frames[1])
        if not (0 <= a < b < len(Q)):
            raise ValueError(f"frames: {frames} is outside the {len(Q)} frames of the carry source")
        sel = list(range(a, b + 1))
    else:
        want = f"{side}: {phase}"
        idx = [t for t, lab in enumerate(labels) if str(lab).startswith(want)
               and (subgoal is None or subg is None or int(subg[t]) == int(subgoal))]
        runs = _contiguous_runs(idx)
        if not runs:
            raise ValueError(f"frames: no frame of the carry source is labelled '{want}...'" + (f" in subgoal {subgoal}" if subgoal is not None else "")
                             + "; pass frames=[first, last]")
        sel = runs[0]
        if sel[0] > 0:
            sel = [sel[0] - 1] + sel                       # start from the frame before the carry: the lifted hold
    Qs = Q[sel]
    if bench_scene is not None and Path(bench_scene).exists():
        bm = mujoco.MjModel.from_xml_path(str(bench_scene))
        if bm.nq != Q.shape[1]:
            raise ValueError(f"bench_scene: nq={bm.nq} does not match the carry source's {Q.shape[1]} columns")
        Ps, Rs = _bench_wrist_poses(bm, Qs, side)
    elif names is not None:
        cols = []
        for jn in ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"):
            c = [i for i, nm in enumerate(names) if nm.endswith(f"_{side}_{jn}") and "_arm_" not in nm]
            if not c:
                raise ValueError(f"carry: no column for the {side} wrist joint {jn}; pass bench_scene")
            cols.append(c[0])
        W = Qs[:, cols]
        Ps, Rs = W[:, :3], np.asarray([H.euler_to_mat(e) for e in W[:, 3:]])
    else:
        raise ValueError("bench_scene: needed to read the wrist from this carry source (no joint_names)")
    return {"pos": np.asarray(Ps, float), "R": np.asarray(Rs, float), "fps": f_, "frames": [sel[0], sel[-1]], "source": source,
            "labels": [labels[t] for t in (sel[0], sel[-1])] if labels else None}


def path_kinematics(path: dict) -> dict:
    """Peak linear speed and acceleration and peak angular speed of a wrist path (finite differences)."""
    Pp, Rs, f = path["pos"], path["R"], float(path["fps"])
    if len(Pp) < 3:
        return {"peak_speed_m_s": 0.0, "peak_acc_m_s2": 0.0, "peak_ang_speed_deg_s": 0.0, "turned_deg": 0.0}
    v = np.diff(Pp, axis=0) * f
    a = np.diff(v, axis=0) * f
    w = [rot_deg(Rs[i], Rs[i + 1]) * f for i in range(len(Rs) - 1)]
    return {"peak_speed_m_s": r2(np.linalg.norm(v, axis=1).max(), 3), "peak_acc_m_s2": r2(np.linalg.norm(a, axis=1).max(), 2),
            "peak_ang_speed_deg_s": r2(max(w), 1), "turned_deg": r2(max(rot_deg(Rs[0], R) for R in Rs), 1)}


# ----------------------------------------------------------------------------- two hands
def two_hand_model(left_xml, right_xml):
    """Both hands floating in one model (hulls added as on the bench) for kinematic hand-hand distances.
    Returns (model, {"left": rig, "right": rig})."""
    from tool_library.common.handgeom import add_hand_hulls
    sp = mujoco.MjSpec.from_string('<mujoco model="two_hands"><compiler angle="radian"/><worldbody/></mujoco>')
    for side, xml in (("left", left_xml), ("right", right_xml)):
        h = mujoco.MjSpec.from_file(str(xml))
        add_hand_hulls(h)
        sp.attach(h, prefix=f"hand{side[0]}_", frame=sp.worldbody.add_frame())
    model = sp.compile()
    rigs = {s: H.HandRig.build(model, f"hand{s[0]}_", s) for s in ("left", "right")}
    return model, rigs


def geom_distance(model, d, g, h, distmax=0.25) -> float:
    """mj_geomDistance with its false zeros removed: MuJoCo 3.10 returns an exact 0.0 for some separated box-box and
    hull-hull pairs (measured: two palms 123 mm apart read 0.0 at isolated wrist positions 0.2 mm apart, a correct
    122.8 mm on either side; a palm hull and a thumb hull 109 mm apart read 0.0 for one micrometre nudge in three). A
    real touch reads a small non-zero number, so an exact zero is re-measured with the first geom nudged by 1 micrometre
    along each axis both ways and the smallest non-zero reading is kept."""
    dist = float(mujoco.mj_geomDistance(model, d, int(g), int(h), float(distmax), None))
    if dist != 0.0:
        return dist
    keep = np.array(d.geom_xpos[g], float)
    vals = []
    for k in range(6):
        d.geom_xpos[g] = keep + np.eye(3)[k % 3] * (1e-6 if k < 3 else -1e-6)
        vals.append(float(mujoco.mj_geomDistance(model, d, int(g), int(h), float(distmax), None)))
    d.geom_xpos[g] = keep
    nz = [v for v in vals if v != 0.0]
    return min(nz) if nz else 0.0


def hand_hand_distance(model, d, rig_a, rig_b, distmax=0.25):
    """Smallest distance (m, negative = penetration) between the two hands' collision geoms and the closest pair."""
    best, pair = float(distmax), None
    P_ = np.asarray(d.geom_xpos, float)
    Rb = np.asarray(model.geom_rbound, float)
    for g in rig_a.collision_gids:
        for h in rig_b.collision_gids:
            if float(np.linalg.norm(P_[g] - P_[h])) - Rb[g] - Rb[h] > best:
                continue
            dist = geom_distance(model, d, g, h, distmax)
            if dist < best:
                best, pair = dist, (model.geom(g).name, model.geom(h).name)
    return best, pair
