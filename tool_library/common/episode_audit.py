"""Shared machinery of the whole-episode audits (category Episode Audits).

Every audit judges one delivered Stage 3 episode: the scene it was executed in (MJCF) and the trajectory
(npz with `qpos [T, nq]`, `fps`, per-frame `labels` '<side>: <phase>' or, with both hands acting at once,
'left: <phase> | right: <phase>', optional `subgoal`). This module
loads the pair once, validates it, finds the robot's hands, arms and fingers, the free objects and the
table, replays kinematics and contacts, parses phase labels and builds the report every audit returns
(`ok`, `issues` naming frame / phase / bodies and what to change, `worst`, measurements).

Nothing here decides a verdict: thresholds live in the audits.
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys
from pathlib import Path

if sys.platform.startswith("linux"):  # headless offscreen rendering; MuJoCo rejects "egl" elsewhere
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from tool_library.common import trajectory as TJ

ROBOT_PREFIX = "robot_"
SIDES = ("left", "right")
FREE, BALL, SLIDE, HINGE = TJ.FREE, TJ.BALL, TJ.SLIDE, TJ.HINGE

# '<side>: <phase text> #<subgoal tag>' (the tag is optional; '#14.2' is the second hand's branch of subgoal 14)
LABEL_RE = re.compile(r"^(?P<side>[A-Za-z][A-Za-z_ ]*?)\s*:\s*(?P<phase>.*?)\s*(?:#(?P<tag>\d+(?:\.\d+)*))?\s*$")
LABEL_SIDES = ("left", "right", "both")

# phase vocabulary of the Stage 3 executor (armhand.TRANSIT_WORDS / CONTACT_WORDS) and of hand-built examples
CARRY_WORDS = ("lift", "carry", "transport", "raise")
HOLD_WORDS = CARRY_WORDS + ("grip", "hold", "set down", "lower", "place", "pour", "insert", "turn", "twist")
RELEASE_WORDS = ("let go", "release", "open the hand", "drop", "open hand", "ungrip")
AWAY_WORDS = ("retreat", "withdraw", "back away", "back off", "return home", "move away", "hold clear", "fold back")
ROT_TASK_WORDS = ("turn", "twist", "unscrew", "screw", "rotate", "flip", "swing", "spin", "pour", "tilt", "wipe", "fold")


# ----------------------------------------------------------------------------- report helpers
def jsonable(x):
    """Plain Python types all the way down (numpy scalars and arrays, tuples, sets)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def r3(x, n=3):
    return None if x is None or not np.isfinite(float(x)) else round(float(x), n)


def finish(issues, worst=None, **measured):
    """The report every audit returns: ok iff no issue."""
    rep = {"ok": not issues, "issues": list(issues), "worst": worst}
    rep.update(measured)
    return jsonable(rep)


def check_positive(**kw):
    """ValueError naming the parameter when a tolerance is not a positive number."""
    for k, v in kw.items():
        try:
            ok = v is not None and float(v) > 0 and np.isfinite(float(v))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{k} must be a positive number (got {v!r})")


# ----------------------------------------------------------------------------- labels
def _parse_part(text):
    m = LABEL_RE.match(str(text).strip())
    if not m or not m.group("phase"):
        return None
    return {"side": m.group("side").strip().lower(), "phase": m.group("phase").strip(), "tag": m.group("tag")}


def label_parts(label):
    """[{'side', 'phase', 'tag'}] of a label. A label is '<side>: <phase> #k', or several such parts joined by ' | '
    when both hands act at once ('left: hold the box #3 | right: press the lid #3'). A part after ' | ' that does
    not start with left / right / both is telemetry and ignored. [] when the label does not have that form."""
    parts = [p for p in str(label).split("|")]
    out = []
    for i, p in enumerate(parts):
        q = _parse_part(p)
        if q is None or (i > 0 and q["side"] not in LABEL_SIDES):
            if i == 0:
                return []
            continue
        out.append(q)
    return out


def parse_label(label):
    """The first part of a label ({'side', 'phase', 'tag'}), None when it does not have the '<side>: <phase>' form."""
    p = label_parts(label)
    return p[0] if p else None


def side_part(label, side):
    """The part of a label that speaks for `side` (its own part, else a 'both' part), or None."""
    parts = label_parts(label)
    for p in parts:
        if p["side"] == side:
            return p
    for p in parts:
        if p["side"] == "both":
            return p
    return None


def label_side(label):
    """The side a label speaks for: its single part's side, 'both' for a two-hand label, None when malformed."""
    p = label_parts(label)
    if not p:
        return None
    sides = {q["side"] for q in p}
    return p[0]["side"] if len(sides) == 1 else "both"


def phase_text(label, side=None):
    """Lower-case phase text: of `side`'s part when given, else of all parts joined."""
    if side is not None:
        p = side_part(label, side)
        return p["phase"].lower() if p else ""
    p = label_parts(label)
    return " | ".join(q["phase"] for q in p).lower() if p else str(label).lower()


def has_word(label, words, side=None):
    t = phase_text(label, side)
    return any(w in t for w in words)


def phase_runs(labels):
    """[(label, first, last)] of runs of identical labels."""
    return TJ.phase_runs(list(labels))


def side_runs(labels, side):
    """[(part text '<side>: <phase> #k', first, last)] of runs of frames whose label speaks for `side` with the same
    part (its own part or a 'both' part); frames that do not speak for it end a run."""
    seq = []
    for lab in labels:
        p = side_part(lab, side)
        seq.append(None if p is None else f"{p['side']}: {p['phase']}" + (f" #{p['tag']}" if p["tag"] else ""))
    out, start = [], None
    for i in range(len(seq) + 1):
        cur = seq[i] if i < len(seq) else None
        if start is not None and (i == len(seq) or cur != seq[start]):
            out.append((seq[start], start, i - 1)); start = None
        if start is None and cur is not None:
            start = i
    return out


def objects_named(label, names):
    """Object names mentioned in a label (longest first; a body name or its first token)."""
    t = str(label).lower()
    hit = [n for n in names if n.lower() in t]
    if not hit:
        hit = [n for n in names if n.split("_")[0].lower() in t.split()]
    return sorted(hit, key=len, reverse=True)


# ----------------------------------------------------------------------------- the episode
class Episode:
    """A scene + trajectory pair, validated, with lazily computed structure and kinematics."""

    def __init__(self, scene, trajectory, frames=None, robot_prefix=ROBOT_PREFIX):
        if scene is None:
            raise ValueError("scene: give the MJCF path the episode was executed in")
        if isinstance(scene, mujoco.MjModel):
            self.scene_path = None
        else:
            if not Path(str(scene)).exists():
                raise ValueError(f"scene: no file {scene}")
            self.scene_path = str(scene)
        self.model = TJ.load_model(scene)
        self.robot_prefix = robot_prefix
        self.extras = {}
        if trajectory is None:
            raise ValueError("trajectory: give the episode npz (qpos, fps, labels)")
        if isinstance(trajectory, (str, Path)):
            p = Path(str(trajectory))
            if not p.exists():
                raise ValueError(f"trajectory: no file {trajectory}")
            z = np.load(str(p), allow_pickle=True)
            key = "qpos" if "qpos" in z.files else ("Q" if "Q" in z.files else None)
            if key is None:
                raise ValueError(f"trajectory: {trajectory} has no 'qpos' array (keys {z.files})")
            Q = np.asarray(z[key], float)
            labels = [str(x) for x in z["labels"]] if "labels" in z.files else [""] * len(Q)
            fps = float(z["fps"]) if "fps" in z.files else 30.0
            for k in z.files:
                if k not in (key, "labels", "fps"):
                    self.extras[k] = z[k]
            self.traj_path = str(p)
        elif isinstance(trajectory, dict):
            Q = np.asarray(trajectory["qpos"], float)
            labels = [str(x) for x in trajectory.get("labels", [""] * len(Q))]
            fps = float(trajectory.get("fps", 30.0))
            self.extras = {k: v for k, v in trajectory.items() if k not in ("qpos", "labels", "fps")}
            self.traj_path = None
        else:
            Q, labels, fps = TJ.as_trajectory(trajectory)
            self.traj_path = None
        Q = np.atleast_2d(Q)
        if Q.ndim != 2 or Q.shape[1] != self.model.nq:
            raise ValueError(f"trajectory has {Q.shape[1] if Q.ndim == 2 else '?'} qpos columns, the scene has "
                             f"nq={self.model.nq}: give the scene the episode was executed in")
        if len(labels) != len(Q):
            raise ValueError(f"trajectory: {len(labels)} labels for {len(Q)} frames")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"trajectory: fps must be positive (got {fps})")
        self.Q, self.labels, self.fps, self.T = Q, list(labels), float(fps), len(Q)
        sg = self.extras.get("subgoal")
        self.subgoal = None if sg is None or len(sg) != self.T else np.asarray(sg)
        self.a, self.b = frame_window(frames, self.T)
        self._cache = {}

    # ---- frames
    @property
    def window(self):
        return range(self.a, self.b + 1)

    def info(self, t):
        t = int(t)
        return {"frame": t, "label": self.labels[t] if 0 <= t < self.T else None,
                "time_s": round(t / self.fps, 3)}

    # ---- structure
    def body_name(self, b):
        return self.model.body(int(b)).name or f"body{int(b)}"

    def geom_name(self, g):
        return self.model.geom(int(g)).name or f"geom{int(g)}"

    @property
    def robot_bodies(self):
        if "robot" not in self._cache:
            self._cache["robot"] = {b for b in range(self.model.nbody) if self.body_name(b).startswith(self.robot_prefix)}
        return self._cache["robot"]

    @property
    def static_bodies(self):
        """World and every body welded to it with no degree of freedom in its chain (table, walls, fixtures)."""
        if "static" not in self._cache:
            m = self.model
            st = set()
            for b in range(m.nbody):
                c, free = b, True
                while c > 0:
                    if m.body_jntnum[c] > 0 or m.body_mocapid[c] >= 0:
                        free = False
                        break
                    c = int(m.body_parentid[c])
                if free:
                    st.add(b)
            self._cache["static"] = st
        return self._cache["static"]

    @property
    def objects(self):
        """{name: dict(body, qadr, dadr, bodies, mass)} of the free bodies outside the robot."""
        if "objects" not in self._cache:
            m, out = self.model, {}
            for j in range(m.njnt):
                if int(m.jnt_type[j]) != FREE:
                    continue
                b = int(m.jnt_bodyid[j])
                nm = self.body_name(b)
                if nm.startswith(self.robot_prefix):
                    continue
                sub = TJ.subtree(m, b)
                out[nm] = {"body": b, "qadr": int(m.jnt_qposadr[j]), "dadr": int(m.jnt_dofadr[j]), "bodies": sub,
                           "mass": float(m.body_subtreemass[b])}
            self._cache["objects"] = out
        return self._cache["objects"]

    def object_of_body(self, b):
        for nm, o in self.objects.items():
            if int(b) in o["bodies"]:
                return nm
        return None

    @property
    def mech_joints(self):
        """Hinge / slide joints outside the robot (drawers, lids, caps on threads): {name: (jid, qadr, type)}."""
        if "mech" not in self._cache:
            m, out = self.model, {}
            for j in range(m.njnt):
                if int(m.jnt_type[j]) in (HINGE, SLIDE) and not self.body_name(m.jnt_bodyid[j]).startswith(self.robot_prefix):
                    out[m.joint(j).name or f"joint{j}"] = (j, int(m.jnt_qposadr[j]), int(m.jnt_type[j]))
            self._cache["mech"] = out
        return self._cache["mech"]

    @property
    def hands(self):
        """{side: dict(bodies, root, palm, finger_qadr, finger_joints, wrist_qadr, arm_joints, arm_qadr, flange,
        finger_act)} for the bench hands (Sharpa naming via episode.robot_parts) or, for other robots, a
        generic grouping of the robot bodies by the side word in their names."""
        if "hands" not in self._cache:
            self._cache["hands"] = _find_hands(self)
        return self._cache["hands"]

    def kind(self, b):
        """'robot' | 'static' (world, table, fixtures) | 'object' for a body id."""
        b = int(b)
        if b in self.robot_bodies:
            return "robot"
        if b in self.static_bodies:
            return "static"
        return "object"

    @property
    def table(self):
        if "table" not in self._cache:
            self._cache["table"] = table_info(self.model)
        return self._cache["table"]

    @property
    def gravity(self):
        return np.asarray(self.model.opt.gravity, float)

    # ---- kinematics over every frame (cheap: no collision)
    def kin(self):
        """(xpos [T, nbody, 3], xquat [T, nbody, 4]) for every frame."""
        if "kin" not in self._cache:
            m = self.model
            d = mujoco.MjData(m)
            xp = np.zeros((self.T, m.nbody, 3))
            xq = np.zeros((self.T, m.nbody, 4))
            xi = np.zeros((self.T, m.nbody, 3))
            for t in range(self.T):
                d.qpos[:] = self.Q[t]
                mujoco.mj_kinematics(m, d)
                xp[t] = d.xpos
                xq[t] = d.xquat
                xi[t] = d.xipos
            self._cache["kin"] = (xp, xq)
            self._cache["xipos"] = xi
        return self._cache["kin"]

    def com_track(self, bodies):
        """Mass-weighted centre of mass [T, 3] of a body set (an object's subtree)."""
        self.kin()
        xi = self._cache["xipos"]
        bl = sorted(int(b) for b in bodies)
        w = np.array([float(self.model.body_mass[b]) for b in bl])
        if w.sum() <= 0:
            return xi[:, bl[0]]
        return np.einsum("tbk,b->tk", xi[:, bl], w) / w.sum()

    def body_pos(self, b):
        return self.kin()[0][:, int(b)]

    def body_quat(self, b):
        return self.kin()[1][:, int(b)]

    # ---- models for contact queries / simulation
    def contact_model(self, margin=0.0):
        """The scene as the bench executor runs it (explicit hand-object and hand-table pairs, see
        episode.build_execution_model) when it has bench hands, else its own collision geometry; every
        geom and pair margin raised to `margin` so near contacts are reported."""
        key = ("cm", round(float(margin), 6))
        if key not in self._cache:
            base = self._cache.get("cm_base")
            if base is None:
                from tool_library.common import rigid_robot as RR
                if RR.is_rigid(self.model):
                    # Runtime contacts belong to the recorded physical scene;
                    # rebuilding a planning model can change their coverage.
                    base = self.model
                elif self.scene_path is not None and self.hands and any(h.get("bench") for h in self.hands.values()):
                    from tool_library.common import episode as E
                    try:
                        base = E.build_execution_model(self.scene_path)[0]
                    except Exception as e:
                        # no silent fallback: the raw scene's bench hands do not collide, so every contact
                        # reading would come back empty and a contact audit would pass without looking
                        raise RuntimeError(f"contact audit: the bench execution model of {self.scene_path} could not "
                                           f"be built ({type(e).__name__}: {e}); the raw scene's hands do not "
                                           "collide, so its contacts cannot be audited") from e
                if base is None:
                    base = self.model
                self._cache["cm_base"] = base
            m = copy_model(base)
            if margin > 0:
                m.geom_margin[:] = np.maximum(m.geom_margin, margin)
                if m.npair:
                    m.pair_margin[:] = np.maximum(m.pair_margin, margin)
            self._cache[key] = m
        return self._cache[key]


def copy_model(m):
    import copy
    return copy.copy(m)


def frame_window(frames, T):
    """(first, last) inclusive from frames=[first, last] (None = whole episode); ValueError on misuse."""
    if frames is None:
        return 0, T - 1
    try:
        a, b = [int(x) for x in frames]
    except (TypeError, ValueError):
        raise ValueError("frames must be [first, last] (inclusive frame indices) or None")
    if a < 0:
        a += T
    if b < 0:
        b += T
    if not (0 <= a <= b < T):
        raise ValueError(f"frames [{a}, {b}] outside the trajectory (0..{T - 1})")
    return a, b


def _find_hands(ep):
    m = ep.model
    out = {}
    try:
        from tool_library.common import episode as E
        parts = E.robot_parts(m)
    except Exception:
        parts = {}
    for side, pt in parts.items():
        rig = pt["rig"]
        fq = sorted(int(v) for v in rig.finger_qpos.values())
        # Proposal wrists have six virtual coordinates. A physical wrist is a
        # fixed child of the flange and has no independent wrist coordinates.
        rz = ([j for j in range(m.njnt) if int(m.jnt_qposadr[j]) == int(rig.wrist_qpos["rz"])]
              if "rz" in rig.wrist_qpos else [])
        wrist_body = int(m.jnt_bodyid[rz[0]]) if rz else int(rig.root_body)
        fj = sorted(int(m.actuator_trnid[a][0]) for a in rig.finger_act.values())
        out[side] = {"bench": True, "bodies": set(rig.body_ids), "root": wrist_body, "base": int(rig.root_body),
                     "palm": int(rig.palm_body) if rig.palm_body >= 0 else wrist_body,
                     "finger_qadr": fq, "finger_joints": fj, "finger_act": {int(m.actuator_trnid[a][0]): int(a) for a in rig.finger_act.values()},
                     "wrist_qadr": [int(rig.wrist_qpos[k]) for k in ("x", "y", "z", "rx", "ry", "rz") if k in rig.wrist_qpos],
                     "arm_joints": list(pt["arm_joints"]), "arm_qadr": list(pt["arm_qadr"]),
                     "flange": int(pt["gripper"]) if pt["gripper"] >= 0 else -1}
    if out:
        return out
    # a generic robot: group bodies by the side word in their names; arm = bodies with '_arm_' / 'arm' in the name
    groups = {}
    for b in ep.robot_bodies:
        nm = ep.body_name(b)
        side = "left" if "left" in nm else ("right" if "right" in nm else "robot")
        groups.setdefault(side, set()).add(b)
    for side, bodies in groups.items():
        arm_b = {b for b in bodies if "_arm" in ep.body_name(b)}
        hand_b = bodies - arm_b
        if not hand_b:
            continue
        base = min(hand_b, key=lambda b: _depth(m, b))
        moving = [b for b in hand_b if m.body_jntnum[b] > 0]
        root = min(moving, key=lambda b: (_depth(m, b), b)) if moving else base
        fj, fq, wq, aj, aq, fact = [], [], [], [], [], {}
        for j in range(m.njnt):
            if int(m.jnt_type[j]) not in (HINGE, SLIDE):
                continue
            b = int(m.jnt_bodyid[j])
            nm = m.joint(j).name or ""
            if b in arm_b:
                aj.append(j); aq.append(int(m.jnt_qposadr[j]))
            elif b in hand_b:
                if any(k in nm for k in ("pos_", "rot_")) or b in (root, base):
                    wq.append(int(m.jnt_qposadr[j]))
                else:
                    fj.append(j); fq.append(int(m.jnt_qposadr[j]))
        for a in range(m.nu):
            if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and int(m.actuator_trnid[a][0]) in fj:
                fact[int(m.actuator_trnid[a][0])] = a
        flange = max(arm_b, key=lambda b: _depth(m, b)) if arm_b else -1
        out[side] = {"bench": False, "bodies": hand_b, "root": root, "base": base, "palm": root, "finger_qadr": fq, "finger_joints": fj,
                     "finger_act": fact, "wrist_qadr": wq, "arm_joints": aj, "arm_qadr": aq, "flange": flange}
    return out


def _depth(m, b):
    d = 0
    while b > 0:
        b = int(m.body_parentid[b]); d += 1
    return d


def table_info(model, name="table"):
    """{'body', 'top_z', 'xy': [xmin, xmax, ymin, ymax]} of the table body (its geoms' world bounds), or of a
    ground plane (infinite footprint); None when the scene has neither."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    if bid >= 0 and any(int(model.geom_bodyid[g]) == bid for g in range(model.ngeom)):
        lo, hi = TJ.body_bounds(model, d, {bid}, TJ.mesh_bounds_cache(model))
        return {"body": int(bid), "top_z": float(hi[2]), "xy": [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]}
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) and int(model.geom_bodyid[g]) == 0:
            return {"body": 0, "top_z": float(d.geom_xpos[g][2]), "xy": [-np.inf, np.inf, -np.inf, np.inf]}
    return None


# ----------------------------------------------------------------------------- rotations and derivatives
def quat_angle(q1, q2):
    """Rotation angle (rad) between unit quaternions, row-wise."""
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    dots = np.abs(np.sum(q1 * q2, axis=-1)) / (np.linalg.norm(q1, axis=-1) * np.linalg.norm(q2, axis=-1) + 1e-12)
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def quat_to_mat(q):
    q = np.asarray(q, float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q / (np.linalg.norm(q) + 1e-12))
    return R.reshape(3, 3)


def central_accel(P, fps):
    """Second difference (m/s^2 or rad/s^2) at frames 1..T-2, shape [T-2, ...]."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return np.zeros((0,) + P.shape[1:])
    return (P[2:] - 2 * P[1:-1] + P[:-2]) * fps * fps


def runs_of(mask):
    """[(first, last)] of True runs in a boolean array."""
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1)); start = None
    return out


def moving_median(x, half):
    x = np.asarray(x, float)
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


# ----------------------------------------------------------------------------- contacts
class ContactScan:
    """Per-frame contacts of an episode on its contact model (margins raised to `margin`): body pairs with their
    smallest signed distance, optionally the contact points, normals and forces."""

    def __init__(self, ep, margin=0.002, model=None):
        self.ep = ep
        self.margin = float(margin)
        self.m = model if model is not None else ep.contact_model(margin)
        self.d = mujoco.MjData(self.m)

    def at(self, t, forces=False):
        """[(b1, b2, dist, g1, g2, i)] at frame t (b1, b2 body ids, dist in m; i indexes self.d.contact)."""
        m, d = self.m, self.d
        d.qpos[:] = self.ep.Q[int(t)]
        d.qvel[:] = 0.0
        if forces:
            mujoco.mj_forward(m, d)
        else:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > self.margin:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            out.append((int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2]), float(c.dist), g1, g2, i))
        return out

    def touching(self, t, set_a, set_b, tol=None):
        """Smallest distance between two body sets at frame t (inf when nothing within the margin)."""
        tol = self.margin if tol is None else tol
        best = np.inf
        for b1, b2, dist, *_ in self.at(t):
            if (b1 in set_a and b2 in set_b) or (b2 in set_a and b1 in set_b):
                best = min(best, dist)
        return best if best <= tol else np.inf


# ----------------------------------------------------------------------------- objects and hands over time
def object_track(ep, name):
    """(P [T, 3], quat [T, 4]) of a free object's body over the episode (world frame)."""
    o = ep.objects[name]
    return ep.body_pos(o["body"]), ep.body_quat(o["body"])


def relative_pose(ep, body, ref):
    """Pose of `body` in the frame of `ref` per frame: (p [T, 3], R [T, 3, 3])."""
    xp, xq = ep.kin()
    p = np.zeros((ep.T, 3)); R = np.zeros((ep.T, 3, 3))
    for t in range(ep.T):
        Rr = quat_to_mat(xq[t, ref]); Rb = quat_to_mat(xq[t, body])
        p[t] = Rr.T @ (xp[t, body] - xp[t, ref])
        R[t] = Rr.T @ Rb
    return p, R


def held_windows(ep, words=CARRY_WORDS, min_frames=3, infer=True, margin=0.003, merge=False, exclude=()):
    """[(side, object name, first, last)] of runs of a hand's phase (its own part of the label, or a 'both' part) that
    say it carries (lift / carry ...). The object is the free object named in that part; when it names none (and
    infer=True) it is the object that touches that hand on most sampled frames of the run (at least 60%).
    merge=True joins consecutive runs of the same side and object (a carry split into up / across / down)."""
    names = list(ep.objects)
    out = []
    scan = None
    for side in ep.hands:
        for part, a, b in side_runs(ep.labels, side):
            if b < ep.a or a > ep.b or not has_word(part, words) or (exclude and has_word(part, exclude)):
                continue
            a, b = max(a, ep.a), min(b, ep.b)
            if b - a + 1 < min_frames:
                continue
            objs = objects_named(part, names)
            obj = objs[0] if objs else None
            if obj is None and infer and names:
                if scan is None:
                    scan = ContactScan(ep, margin)
                fr = np.unique(np.linspace(a, b, min(7, b - a + 1)).astype(int))
                cnt = {}
                for t in fr:
                    for n2 in names:
                        if scan.touching(t, ep.hands[side]["bodies"], ep.objects[n2]["bodies"]) < np.inf:
                            cnt[n2] = cnt.get(n2, 0) + 1
                if cnt:
                    best = max(cnt, key=cnt.get)
                    if cnt[best] >= 0.6 * len(fr):
                        obj = best
            if obj is not None:
                out.append((side, obj, a, b))
    if merge and out:
        out.sort(key=lambda w: (w[0], w[1], w[2]))
        merged = [list(out[0])]
        for w in out[1:]:
            m = merged[-1]
            if w[0] == m[0] and w[1] == m[1] and w[2] <= m[3] + 1:
                m[3] = max(m[3], w[3])
            else:
                merged.append(list(w))
        out = [tuple(m) for m in merged]
    out.sort(key=lambda w: w[2])
    return out


# ----------------------------------------------------------------------------- geometry
def geom_is_visual(m, g):
    """A drawn geom (group 0-2, not transparent) that does not collide: what the camera shows."""
    return int(m.geom_group[g]) <= 2 and float(m.geom_rgba[g][3]) > 0 and not (int(m.geom_contype[g]) or int(m.geom_conaffinity[g]))


def geom_collides(m, g):
    if int(m.geom_contype[g]) or int(m.geom_conaffinity[g]):
        return True
    return any(int(m.pair_geom1[i]) == g or int(m.pair_geom2[i]) == g for i in range(m.npair))


def geom_local_points(m, g, n_prim=12):
    """Points on a geom's surface in its own frame: mesh vertices, or samples of a primitive's surface."""
    t = int(m.geom_type[g])
    s = np.asarray(m.geom_size[g], float)
    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(m.geom_dataid[g])
        v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        return np.asarray(m.mesh_vert[v0:v0 + nv], float)
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        g1 = np.linspace(-1, 1, 5)
        pts = []
        for ax in range(3):
            for sgn in (-1, 1):
                for u in g1:
                    for v in g1:
                        p = np.zeros(3); p[ax] = sgn
                        o = [i for i in range(3) if i != ax]
                        p[o[0]] = u; p[o[1]] = v
                        pts.append(p * s)
        return np.unique(np.array(pts), axis=0)
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        u = _fib_sphere(4 * n_prim)
        return u * s[0]
    if t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        r, h = s[0], s[1]
        ang = np.linspace(0, 2 * np.pi, n_prim, endpoint=False)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        pts = [np.c_[ring, np.full(len(ring), z)] for z in np.linspace(-h, h, 5)]
        if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            pts += [np.array([[0, 0, h + r]]), np.array([[0, 0, -h - r]])]
        else:
            pts += [np.c_[ring * 0.5, np.full(len(ring), z)] for z in (-h, h)] + [np.array([[0, 0, h], [0, 0, -h]])]
        return np.vstack(pts)
    if t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return _fib_sphere(4 * n_prim) * s
    return np.zeros((1, 3))


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def geom_world_points(m, d, g, local=None):
    P = geom_local_points(m, g) if local is None else local
    R = np.asarray(d.geom_xmat[g]).reshape(3, 3)
    return P @ R.T + np.asarray(d.geom_xpos[g])


# ----------------------------------------------------------------------------- simulation from a state
def settle(ep, qpos, seconds=1.0, robot="pin", model=None):
    """Simulate the scene from `qpos` for `seconds` with every robot dof held where it is (robot='pin') and
    return {object name: (moved m, rotated deg, final position)} plus the final qpos. The model is the
    contact model (hands collide through the executor's pairs)."""
    m = model if model is not None else ep.contact_model(0.0)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(qpos, float)
    d.qvel[:] = 0.0
    radr, rdadr = robot_dofs(ep, m)
    q_robot = d.qpos[radr].copy()
    for a in range(m.nu):                              # servos hold the pose they start in
        if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            j = int(m.actuator_trnid[a][0])
            d.ctrl[a] = d.qpos[int(m.jnt_qposadr[j])]
    mujoco.mj_kinematics(m, d)
    for e in range(m.neq):
        b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        if b1 in ep.robot_bodies and b2 in ep.robot_bodies:
            d.eq_active[e] = 0                         # the arm-hand placeholders: the arm does not drag the hand
        elif int(m.eq_type[e]) in (TJ.WELD, TJ.CONNECT):
            # a constraint the episode had released (a cap off its thread) must not snap back on
            gap, ang = eq_residual(m, d, e)
            if gap > 0.003 or ang > 3.0:
                d.eq_active[e] = 0
    mujoco.mj_forward(m, d)
    steps = int(round(float(seconds) / m.opt.timestep))
    start = {n: (d.xpos[o["body"]].copy(), d.xquat[o["body"]].copy()) for n, o in ep.objects.items()}
    peak = {n: [0.0, 0.0] for n in ep.objects}
    bad = None
    for k in range(steps):
        mujoco.mj_step(m, d)
        if robot == "pin" and len(radr):
            d.qpos[radr] = q_robot
            d.qvel[rdadr] = 0.0
        if not np.isfinite(d.qpos).all():
            bad = k
            break
        if k % 10 == 9:
            for n, o in ep.objects.items():
                p0, q0 = start[n]
                peak[n][0] = max(peak[n][0], float(np.linalg.norm(d.xpos[o["body"]] - p0)))
                peak[n][1] = max(peak[n][1], float(np.degrees(quat_angle(q0, d.xquat[o["body"]]))))
    mujoco.mj_forward(m, d)
    out = {}
    for n, o in ep.objects.items():
        p0, q0 = start[n]
        out[n] = {"moved_m": float(np.linalg.norm(d.xpos[o["body"]] - p0)) if bad is None else np.inf,
                  "rotated_deg": float(np.degrees(quat_angle(q0, d.xquat[o["body"]]))) if bad is None else np.inf,
                  "peak_moved_m": peak[n][0], "peak_rotated_deg": peak[n][1],
                  "start": p0.tolist(), "end": d.xpos[o["body"]].tolist()}
    return out, d.qpos.copy(), bad


def robot_dofs(ep, m=None):
    """(qpos addresses, dof addresses) of every robot joint."""
    m = m if m is not None else ep.model
    qa, da = [], []
    for j in range(m.njnt):
        if (m.body(int(m.jnt_bodyid[j])).name or "").startswith(ep.robot_prefix):
            n = {FREE: 7, BALL: 4}.get(int(m.jnt_type[j]), 1)
            nd = {FREE: 6, BALL: 3}.get(int(m.jnt_type[j]), 1)
            qa += list(range(int(m.jnt_qposadr[j]), int(m.jnt_qposadr[j]) + n))
            da += list(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + nd))
    return np.asarray(qa, int), np.asarray(da, int)


def robot_scalar_joints(ep):
    """[(jid, qadr, name, group)] of every robot hinge/slide; group is '<side>_arm', '<side>_wrist' or
    '<side>_fingers' (generic robots: 'robot_*')."""
    m = ep.model
    out = []
    owner = {}
    for s, h in ep.hands.items():
        for j in h["arm_joints"]:
            owner[j] = f"{s}_arm"
        for j in h["finger_joints"]:
            owner[j] = f"{s}_fingers"
        for j in range(m.njnt):
            if int(m.jnt_qposadr[j]) in h["wrist_qadr"]:
                owner[j] = f"{s}_wrist"
    for j in range(m.njnt):
        if int(m.jnt_type[j]) not in (HINGE, SLIDE):
            continue
        b = int(m.jnt_bodyid[j])
        if b not in ep.robot_bodies:
            continue
        out.append((j, int(m.jnt_qposadr[j]), m.joint(j).name or f"joint{j}", owner.get(j, "robot")))
    return out


# ----------------------------------------------------------------------------- inputs that are files
def read_json(x, what):
    """A dict from a dict or a json path; ValueError naming the parameter otherwise."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    import json
    p = Path(str(x))
    if not p.exists():
        raise ValueError(f"{what}: no file {x}")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise ValueError(f"{what}: {x} is not valid json ({e})")


def setup_scene_path():
    """The repository's default setup scene (setup/scene.xml next to tool_library/), or None."""
    p = Path(__file__).resolve().parents[2] / "setup" / "scene.xml"
    return str(p) if p.exists() else None


def match_fn(patterns):
    pats = list(patterns or [])
    return lambda name: any(fnmatch.fnmatchcase(name, p) for p in pats)


# ----------------------------------------------------------------------------- equality constraints
def equality_engagement(ep, tol_m=0.003, tol_deg=3.0):
    """[{eq, name, bodies (b1, b2), kind, engaged [T] bool}] for every weld / connect equality. `kind` is
    'robot' (both bodies robot: the arm-hand bolt), 'robot-object' (a hand holding an object by a constraint),
    'object' (a mechanism: a cap on its thread, two rope halves) or 'world'. `engaged` comes from a recorded
    eq_active [T, neq] when the trajectory carries one, else from the constraint's own residual (the two
    frames sit where the constraint puts them: within tol_m and, for a weld, tol_deg)."""
    m = ep.model
    rec = ep.extras.get("eq_active")
    rec = np.asarray(rec) if rec is not None and np.ndim(rec) == 2 and len(rec) == ep.T and np.shape(rec)[1] == m.neq else None
    out = []
    xp, xq = ep.kin()
    d0 = mujoco.MjData(m)
    mujoco.mj_resetData(m, d0)
    mujoco.mj_kinematics(m, d0)
    for e in range(m.neq):
        kind_e = int(m.eq_type[e])
        if kind_e not in (TJ.WELD, TJ.CONNECT):
            continue
        b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
        r1, r2 = b1 in ep.robot_bodies, b2 in ep.robot_bodies
        s1, s2 = b1 in ep.static_bodies, b2 in ep.static_bodies
        kind = "robot" if (r1 and r2) else ("robot-object" if (r1 or r2) else ("world" if (s1 or s2) else "object"))
        dat = np.asarray(m.eq_data[e], float)
        if rec is not None:
            eng = rec[:, e].astype(bool)
        else:
            gap = np.zeros(ep.T); ang = np.zeros(ep.T)
            if kind_e == TJ.CONNECT:
                a1, a2 = dat[0:3], dat[3:6]
                for t in range(ep.T):
                    p1 = xp[t, b1] + quat_to_mat(xq[t, b1]) @ a1
                    p2 = xp[t, b2] + quat_to_mat(xq[t, b2]) @ a2
                    gap[t] = np.linalg.norm(p1 - p2)
            else:
                rel_p, Rrel = _weld_rel(m, d0, b1, b2, dat)
                anchor = dat[0:3]
                for t in range(ep.T):
                    R1 = quat_to_mat(xq[t, b1]); R2 = quat_to_mat(xq[t, b2])
                    p1 = xp[t, b1] + R1 @ rel_p                  # the weld point as body 1 carries it
                    p2 = xp[t, b2] + R2 @ anchor                 # the weld point as body 2 carries it
                    gap[t] = np.linalg.norm(p1 - p2)
                    ang[t] = np.degrees(TJ.angle_between(R1 @ Rrel, R2))
            eng = (gap <= tol_m) & (ang <= tol_deg)
        out.append({"eq": e, "name": m.equality(e).name or f"eq{e}", "bodies": (b1, b2), "kind": kind, "engaged": eng})
    return out


def object_links(ep, engagement=None):
    """{object name: [(frame mask [T], partner object name or 'robot' or 'world')]} of the equality constraints
    that hold an object (engaged frames only)."""
    eng = engagement if engagement is not None else equality_engagement(ep)
    links = {n: [] for n in ep.objects}
    for e in eng:
        b1, b2 = e["bodies"]
        for ba, bb in ((b1, b2), (b2, b1)):
            oa = ep.object_of_body(ba)
            if oa is None:
                continue
            if bb in ep.robot_bodies:
                partner = "robot"
            elif bb in ep.static_bodies:
                partner = "world"
            else:
                partner = ep.object_of_body(bb) or "world"
            if partner == oa:
                continue
            links[oa].append((e["engaged"], partner, e["name"]))
    return links


# ----------------------------------------------------------------------------- derived models
def model_from_spec(ep, edit):
    """Compile a copy of the scene after `edit(spec)`; needs the scene as a file."""
    if ep.scene_path is None:
        raise ValueError("scene: this audit needs the scene as an MJCF path")
    spec = mujoco.MjSpec.from_file(ep.scene_path)
    edit(spec)
    return spec.compile()


def no_robot_contact_model(ep):
    """The scene with every robot geom out of collision (and every contact pair that names one removed):
    what the objects do when the robot is not there to prop them."""
    pref = ep.robot_prefix

    def edit(spec):
        robot_geoms = set()
        for g in spec.geoms:
            body = g.parent.name if g.parent is not None else ""
            if body.startswith(pref):
                g.contype = 0; g.conaffinity = 0
                if g.name:
                    robot_geoms.add(g.name)
        for p in list(spec.pairs):
            if p.geomname1 in robot_geoms or p.geomname2 in robot_geoms or \
                    (p.geomname1 or "").startswith(pref) or (p.geomname2 or "").startswith(pref):
                spec.delete(p)
    return model_from_spec(ep, edit)


def visual_robot_model(ep):
    """The scene with every visible robot geom colliding as its convex hull (the hands' hull geoms added), the
    objects' and the table's own collision geoms, contact pairs and excludes removed: what the camera shows,
    made collidable for kinematic sweeps."""
    pref = ep.robot_prefix

    def edit(spec):
        from tool_library.common.handgeom import add_hand_hulls
        try:
            add_hand_hulls(spec)
        except Exception:
            pass
        for g in spec.geoms:
            body = g.parent.name if g.parent is not None else ""
            if body.startswith(pref) and g.type != mujoco.mjtGeom.mjGEOM_PLANE:
                if "_collision_hand_" in (g.name or "") or g.group <= 2:
                    g.contype = 1; g.conaffinity = 1; g.condim = 1
                else:
                    g.contype = 0; g.conaffinity = 0
        for p in list(spec.pairs):
            spec.delete(p)
        for x in list(spec.excludes):
            spec.delete(x)
    return model_from_spec(ep, edit)


def interp_qpos(m, q0, q1, u):
    """qpos between two frames: scalar joints linearly, free / ball rotations by slerp."""
    q = (1 - u) * np.asarray(q0, float) + u * np.asarray(q1, float)
    for j in range(m.njnt):
        t = int(m.jnt_type[j]); a = int(m.jnt_qposadr[j])
        if t == FREE:
            a += 3
        if t in (FREE, BALL):
            qa, qb = np.asarray(q0[a:a + 4], float), np.asarray(q1[a:a + 4], float)
            if np.dot(qa, qb) < 0:
                qb = -qb
            ang = np.arccos(np.clip(abs(np.dot(qa, qb)) / (np.linalg.norm(qa) * np.linalg.norm(qb) + 1e-12), 0, 1))
            if ang < 1e-9:
                qi = qa
            else:
                qi = (np.sin((1 - u) * ang) * qa + np.sin(u * ang) * qb) / np.sin(ang)
            q[a:a + 4] = qi / (np.linalg.norm(qi) + 1e-12)
    return q


# ----------------------------------------------------------------------------- mesh surfaces
def mesh_surface_samples(m, g, n=1500, rng=None):
    """(points [n, 3], normals [n, 3]) on a mesh geom's surface in the geom frame, area-weighted, plus the
    vertices with their face-averaged normals."""
    rng = np.random.default_rng(0) if rng is None else rng
    mid = int(m.geom_dataid[g])
    v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    f0, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    V = np.asarray(m.mesh_vert[v0:v0 + nv], float)
    F = np.asarray(m.mesh_face[f0:f0 + nf], int)
    if nf == 0:
        return V, np.zeros_like(V)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    fn = cr / (2 * area[:, None] + 1e-15)
    k = rng.choice(nf, size=int(n), p=area / area.sum()) if area.sum() > 0 else rng.integers(0, nf, int(n))
    r1, r2 = rng.random(len(k)), rng.random(len(k))
    s1 = np.sqrt(r1)
    P = (1 - s1)[:, None] * a[k] + (s1 * (1 - r2))[:, None] * b[k] + (s1 * r2)[:, None] * c[k]
    vn = np.zeros_like(V)
    for i in range(3):
        np.add.at(vn, F[:, i], cr)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-15
    return np.vstack([P, V]), np.vstack([fn[k], vn])


def primitive_surface_samples(m, g, n=600):
    """(points, outward normals) on a primitive geom's surface in its frame."""
    t = int(m.geom_type[g]); s = np.asarray(m.geom_size[g], float)
    rng = np.random.default_rng(1)
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        u = _fib_sphere(n); return u * s[0], u
    if t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        u = _fib_sphere(n); P = u * s; N = u / (s ** 2); return P, N / np.linalg.norm(N, axis=1, keepdims=True)
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        pts, nrm = [], []
        areas = np.array([s[1] * s[2], s[0] * s[2], s[0] * s[1]])
        for ax in range(3):
            o = [i for i in range(3) if i != ax]
            k = max(4, int(n * areas[ax] / areas.sum() / 2))
            for sg in (-1, 1):
                P = np.zeros((k, 3)); P[:, ax] = sg * s[ax]
                P[:, o[0]] = rng.uniform(-s[o[0]], s[o[0]], k); P[:, o[1]] = rng.uniform(-s[o[1]], s[o[1]], k)
                Nn = np.zeros((k, 3)); Nn[:, ax] = sg
                pts.append(P); nrm.append(Nn)
        return np.vstack(pts), np.vstack(nrm)
    if t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        r, h = s[0], s[1]
        k = n // 2
        th = rng.uniform(0, 2 * np.pi, k); z = rng.uniform(-h, h, k)
        P1 = np.c_[r * np.cos(th), r * np.sin(th), z]; N1 = np.c_[np.cos(th), np.sin(th), np.zeros(k)]
        if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            u = _fib_sphere(n - k); cap = u * r; cap[:, 2] += np.sign(u[:, 2]) * h
            return np.vstack([P1, cap]), np.vstack([N1, u])
        k2 = (n - k) // 2
        rr = r * np.sqrt(rng.random(2 * k2)); tt = rng.uniform(0, 2 * np.pi, 2 * k2)
        zz = np.r_[np.full(k2, h), np.full(k2, -h)]
        P2 = np.c_[rr * np.cos(tt), rr * np.sin(tt), zz]; N2 = np.c_[np.zeros((2 * k2, 2)), np.sign(zz)]
        return np.vstack([P1, P2]), np.vstack([N1, N2])
    return np.zeros((1, 3)), np.array([[0, 0, 1.0]])


def geom_surface_samples(m, g, n=1500):
    if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        return mesh_surface_samples(m, g, n)
    return primitive_surface_samples(m, g, max(200, n // 3))


def geom_in_body(m, g):
    """(R, p) of a geom in its body's frame."""
    return quat_to_mat(m.geom_quat[g]), np.asarray(m.geom_pos[g], float)


# ----------------------------------------------------------------------------- object contact timeline
def adhesion_force(model):
    """{body id: the largest pull its adhesion actuators can exert, N}: actuators with a body transmission (MuJoCo's
    adhesion), force gain * ctrl + bias over the control range, capped by the force range."""
    out = {}
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_BODY):
            continue
        lo, hi = (model.actuator_ctrlrange[a] if model.actuator_ctrllimited[a] else (0.0, 1.0))
        g, b0 = float(model.actuator_gainprm[a][0]), float(model.actuator_biasprm[a][0])
        f = max(abs(g * float(lo) + b0), abs(g * float(hi) + b0))
        if model.actuator_forcelimited[a]:
            f = min(f, float(max(abs(model.actuator_forcerange[a][0]), abs(model.actuator_forcerange[a][1]))))
        b = int(model.actuator_trnid[a][0])
        out[b] = out.get(b, 0.0) + f
    return out


def object_contact_timeline(ep, margin=0.003, normals=False):
    """Per frame, per free object: {'robot': {side or 'robot': dist}, 'static': dist, 'objects': {name: dist},
    'normals': [(unit normal pushing INTO the object, friction, partner kind)], 'adhesion': [the pull an adhesion
    actuator of either body can add at that contact, N (0 for most), one per normal]} for every contact of the object's
    subtree within `margin` (contact model: the executor's hand pairs). Cached on the episode."""
    key = ("timeline", round(float(margin), 6), bool(normals))
    if key in ep._cache:
        return ep._cache[key]
    scan = ContactScan(ep, margin)
    owner = {}
    for n, o in ep.objects.items():
        for b in o["bodies"]:
            owner[b] = n
    side_of = {}
    for s, h in ep.hands.items():
        for b in h["bodies"]:
            side_of[b] = s
    adh = adhesion_force(ep.model) if normals else {}
    out = []
    for t in range(ep.T):
        rows = {n: {"robot": {}, "static": np.inf, "objects": {}, "normals": [], "adhesion": []} for n in ep.objects}
        for b1, b2, dist, g1, g2, i in scan.at(t):
            for bo, bx, sign in ((b1, b2, -1.0), (b2, b1, 1.0)):
                n = owner.get(bo)
                if n is None or owner.get(bx) == n:
                    continue
                r = rows[n]
                k = ep.kind(bx)
                if k == "robot":
                    s = side_of.get(bx, "robot")
                    r["robot"][s] = min(r["robot"].get(s, np.inf), dist)
                elif k == "static":
                    r["static"] = min(r["static"], dist)
                else:
                    other = owner.get(bx, ep.body_name(bx))
                    r["objects"][other] = min(r["objects"].get(other, np.inf), dist)
                if normals:
                    c = scan.d.contact[i]
                    nv = sign * np.asarray(c.frame[:3], float)
                    r["normals"].append((nv, float(c.friction[0]) if c.dim > 1 else 0.0, k))
                    r["adhesion"].append(adh.get(int(bo), 0.0) + adh.get(int(bx), 0.0))
        out.append(rows)
    ep._cache[key] = out
    return out


def any_contact(row):
    return bool(row["robot"]) or np.isfinite(row["static"]) or bool(row["objects"])


def robot_connected(timeline, t, name, links=None):
    """True when the object at frame t touches the robot directly or through a chain of touching objects (or an
    engaged constraint to an object that does)."""
    seen, stack = {name}, [name]
    while stack:
        x = stack.pop()
        row = timeline[t][x]
        if row["robot"]:
            return True
        nxt = list(row["objects"].keys())
        if links is not None:
            for mask, partner, _ in links.get(x, []):
                if mask[t]:
                    if partner == "robot":
                        return True
                    if partner != "world":
                        nxt.append(partner)
        for y in nxt:
            if y in timeline[t] and y not in seen:
                seen.add(y); stack.append(y)
    return False


def contact_within(row, tol, kinds=("robot", "static", "objects")):
    """True when the object's timeline row has a contact of the given kinds within `tol` metres."""
    if "robot" in kinds and any(v <= tol for v in row["robot"].values()):
        return True
    if "static" in kinds and row["static"] <= tol:
        return True
    if "objects" in kinds and any(v <= tol for v in row["objects"].values()):
        return True
    return False


def joint_names_by_qadr(m):
    """{qpos address: joint name} (multi-dof joints by their first address)."""
    return {int(m.jnt_qposadr[j]): (m.joint(j).name or f"joint{j}") for j in range(m.njnt)}


def _weld_rel(m, d0, b1, b2, dat):
    """(weld point in body 1's frame, body 2's orientation relative to body 1) of a weld. MuJoCo's weld data is
    [anchor in body 2 (3), the same point in body 1 (3), relative orientation quat (4), torquescale]; a zero quaternion
    means 'as at qpos0' (d0: kinematics at qpos0)."""
    rel_p, rel_q = np.asarray(dat[3:6], float), np.asarray(dat[6:10], float)
    if np.linalg.norm(rel_q) < 1e-9:
        R1, R2 = d0.xmat[b1].reshape(3, 3), d0.xmat[b2].reshape(3, 3)
        Rrel = R1.T @ R2
        rel_p = R1.T @ (d0.xpos[b2] + R2 @ np.asarray(dat[0:3], float) - d0.xpos[b1])
    else:
        Rrel = quat_to_mat(rel_q)
    return rel_p, Rrel


def eq_residual(m, d, e):
    """(gap m, angle deg) of a weld / connect equality at the current kinematics (d after mj_kinematics)."""
    b1, b2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
    dat = np.asarray(m.eq_data[e], float)
    R1, R2 = d.xmat[b1].reshape(3, 3), d.xmat[b2].reshape(3, 3)
    if int(m.eq_type[e]) == TJ.CONNECT:
        return float(np.linalg.norm(d.xpos[b1] + R1 @ dat[0:3] - d.xpos[b2] - R2 @ dat[3:6])), 0.0
    d0 = mujoco.MjData(m)
    mujoco.mj_resetData(m, d0); mujoco.mj_kinematics(m, d0)
    rel_p, Rrel = _weld_rel(m, d0, b1, b2, dat)
    p1 = d.xpos[b1] + R1 @ rel_p
    p2 = d.xpos[b2] + R2 @ dat[0:3]
    return float(np.linalg.norm(p1 - p2)), float(np.degrees(TJ.angle_between(R1 @ Rrel, R2)))
