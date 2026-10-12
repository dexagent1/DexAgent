"""Dense tracking references: what Stage 3's method modules exchange and what the executor follows.

A tracking reference is the whole episode as robot commands on the bench, one row per video frame:

    save_reference(path, model, Q [T, nq], labels [T], fps, segment [T], segments, source)
    ref = load_reference(path, model)        # {qpos, labels, fps, segment, segments, source, ...}

`Q` is in the bench scene's own qpos layout (the planning scene execute_subgoal_plan takes): the hands' wrists and
fingers are the commands, the objects' columns are where the reference wants them (targets, never commands). When
`arms_coupled` is set the arm columns hold the hands on their flanges in every row (an executed attempt's commands, a
method's refined commands) and the executor runs them as given; otherwise the arms are coupled to the hands as for any
plan. `segment[t]` names the plan entry a frame belongs to, `segments[k]` its Stage 1 indices
(and the motion, side and object when the source knew them). write_plan turns a reference into an execute_subgoal_plan
plan of `track_reference` entries, one per contiguous segment: the executor follows the frames, couples the arms, runs
the physics and checks each entry like any other (common/subgoal.py, m_track_reference), so a method module's
trajectory reaches Stage 4 through the same execution, packaging and gates as an agent's.

Sources: reference_from_attempt (the agent's best execute_subgoal_plan attempt in a Stage 3 workspace), the method
modules' own outputs. BenchTracker puts a reference on the bench's rigid runtime the way the executor will run it (home
start, joint-rate ceilings, arms coupled), rolls command chunks out with the executor's own servo rollout from any saved
integration state, and scores them against the reference's object poses: the simulator of sampling_mpc_refinement and
reinforcement_learning_tracking (RuntimeSim is its picklable half, for worker processes).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import episode as E

SCHEMA = "dexagent.tracking_reference.v1"
_SUFFIX = re.compile(r"(\s+#\d+(?:\.\d+)?)+$")
FLEX = re.compile(r"_(?:CMC_FE|MCP_FE|PIP|DIP|IP)$")          # the flexion joints a grip closes (abduction and opposition stay)
AH_CFG = {"project_pos_m": 0.006, "project_rot_deg": 4.0, "max_speed_deg_s": 90.0}   # execute_subgoal_plan's defaults
RATES = (90.0, 0.4)                                            # the executor's joint-rate ceilings (deg/s, m/s)
MOVED_M, MOVED_DEG, MOVED_JOINT = 0.005, 5.0, 0.03              # what counts as the reference moving a part
TURNED_DEG = 20.0                                              # an object turned more than this is judged on its orientation
FORCE_SOFT_N = 12.0                                            # hand-object force the costs start to penalize (the spike audit allows 15 N)


def clean_label(label) -> str:
    """A frame label without the executor's segment suffixes ('left: lift cup #2' -> 'left: lift cup')."""
    return _SUFFIX.sub("", str(label)).strip()


# ------------------------------------------------------------------------------------------------- the file --
def save_reference(path, model, Q, labels, fps, segment, segments, source, arms_coupled=False, **extra) -> str:
    Q = np.asarray(Q, float)
    if Q.ndim != 2 or Q.shape[1] != model.nq or not len(Q) or not np.isfinite(Q).all():
        raise ValueError(f"a tracking reference is [T, nq={model.nq}] finite rows, got {Q.shape}")
    segment = np.asarray(segment, int)
    if len(labels) != len(Q) or len(segment) != len(Q):
        raise ValueError("labels and segment need one entry per frame")
    if segment.min() < 0 or segment.max() >= len(segments):
        raise ValueError("segment ids must index `segments`")
    for s in segments:
        if not s.get("stage1_indices") or any(type(i) is not int for i in s["stage1_indices"]):
            raise ValueError("every segment needs integer stage1_indices")
    labels = [clean_label(l) for l in labels]
    if any(": " not in l for l in labels):
        raise ValueError("labels read '<side>: <phase>' (or '<left part> | <right part>')")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, schema=SCHEMA, qpos=Q, labels=np.array(labels), fps=float(fps), segment=segment,
                        segments=json.dumps(segments), joint_names=np.array(E.joint_names(model)), source=str(source),
                        arms_coupled=bool(arms_coupled), **{k: np.asarray(v) for k, v in extra.items()})
    return str(Path(path).resolve())


def load_reference(path, model=None) -> dict:
    with np.load(str(path), allow_pickle=False) as z:
        if "schema" not in z.files or str(z["schema"]) != SCHEMA:
            raise ValueError(f"{path} is not a tracking reference ({SCHEMA})")
        ref = {"qpos": z["qpos"].astype(float), "labels": [str(l) for l in z["labels"]], "fps": float(z["fps"]),
               "segment": z["segment"].astype(int), "segments": json.loads(str(z["segments"])),
               "joint_names": [str(n) for n in z["joint_names"]], "source": str(z["source"]),
               "arms_coupled": bool(z["arms_coupled"]) if "arms_coupled" in z.files else False,
               "path": str(Path(path).resolve())}
    if model is not None and ref["joint_names"] != E.joint_names(model):
        raise ValueError(f"{path} was written for another scene: its qpos columns are not this scene's")
    return ref


def runs(segment):
    """[(segment id, first frame, last frame)] of the contiguous runs of `segment`."""
    segment = np.asarray(segment, int)
    out, a = [], 0
    for t in range(1, len(segment) + 1):
        if t == len(segment) or segment[t] != segment[a]:
            out.append((int(segment[a]), a, t - 1)); a = t
    return out


# --------------------------------------------------------------------------------------------- the columns --
def arm_columns(model):
    return sorted(int(a) for pt in E.robot_parts(model).values() for a in (pt.get("arm_qadr") or []))


def planning_columns(model):
    """The columns a reference commands in the planning scene: everything but the arms (the coupling solves those)."""
    arm = set(arm_columns(model))
    return np.array([c for c in range(model.nq) if c not in arm], int)


def tracked_parts(model, Q, frames=None):
    """What the reference moves over `frames` (default all): free bodies {name: (qpos address)} that move more than
    MOVED_M or turn more than MOVED_DEG, and non-robot hinge/slide joints {name: address} that travel MOVED_JOINT."""
    Q = np.asarray(Q, float)
    a, b = (0, len(Q) - 1) if frames is None else (int(frames[0]), int(frames[1]))
    bodies, joints = {}, {}
    for name, adr in E.object_free_joints(model).items():
        p = Q[a:b + 1, adr:adr + 3]
        q = Q[a:b + 1, adr + 3:adr + 7]
        turn = Rot.from_quat(q[:, [1, 2, 3, 0]]).inv() * Rot.from_quat(np.tile(q[0, [1, 2, 3, 0]], (len(q), 1)))
        if np.linalg.norm(p - p[0], axis=1).max() > MOVED_M or np.degrees(turn.magnitude()).max() > MOVED_DEG:
            bodies[name] = int(adr)
    for j in range(model.njnt):
        if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        nm = model.joint(j).name or ""
        if nm.startswith("robot_") or (model.body(int(model.jnt_bodyid[j])).name or "").startswith("robot_"):
            continue
        adr = int(model.jnt_qposadr[j])
        if np.abs(Q[a:b + 1, adr] - Q[a, adr]).max() > MOVED_JOINT:
            joints[nm] = adr
    return bodies, joints


def _root_of(model, body):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body))
    while b > 0 and int(model.body_parentid[b]) != 0:
        b = int(model.body_parentid[b])
    return model.body(b).name if b > 0 else None


def tracking_expect(model, Q, frames, obj=None):
    """What a track_reference entry must achieve: the entry's object (else the part the reference moves most) ends
    where the reference ends it (target_pos), turned like it when the reference turns it more than TURNED_DEG
    (target_quat_wxyz: an in-hand rotation is judged on its orientation), its articulated joint at the reference's end
    value; every other part the reference moves is involved (not expected untouched)."""
    Q = np.asarray(Q, float)
    a, b = int(frames[0]), int(frames[1])
    bodies, joints = tracked_parts(model, Q, (a, b))
    free = E.object_free_joints(model)

    def joint_part(name):
        return _root_of(model, model.body(int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])).name)
    e = {"involved": sorted((set(bodies) | {joint_part(j) for j in joints}) - {None})}
    if obj is None and bodies:
        obj = max(bodies, key=lambda n: float(np.linalg.norm(Q[b, bodies[n]:bodies[n] + 3] - Q[a, bodies[n]:bodies[n] + 3])))
    if obj is None and joints:
        obj = joint_part(next(iter(joints)))
    if obj is None or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(obj)) < 0:
        return e
    e["object"] = obj
    root = _root_of(model, obj)
    adr = free.get(obj, free.get(root))
    if adr is not None and (obj in bodies or root in bodies):
        e["target_pos"] = Q[b, adr:adr + 3].round(5).tolist()
        e["target_axes"] = "xyz"
        q0, q1 = Q[a, adr + 3:adr + 7], Q[b, adr + 3:adr + 7]
        turn = (Rot.from_quat(q0[[1, 2, 3, 0]]).inv() * Rot.from_quat(q1[[1, 2, 3, 0]])).magnitude()
        if np.degrees(turn) > TURNED_DEG:
            e["target_quat_wxyz"] = (q1 / np.linalg.norm(q1)).round(6).tolist()
    sub = E.subtree_bodies(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root or obj))
    own = [(n, adr_) for n, adr_ in joints.items() if int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) in sub]
    if own:
        n, adr_ = max(own, key=lambda x: abs(Q[b, x[1]] - Q[a, x[1]]))
        e["joint"], e["joint_target"] = n, round(float(Q[b, adr_]), 5)
    return e


# -------------------------------------------------------------------------------------------------- sources --
def _attempt_dirs(root):
    root = Path(root)
    found = []
    for d in [root] + sorted(p for p in root.rglob("*") if p.is_dir() and "methods" not in p.relative_to(root).parts[:1]):
        if (d / "trajectory_proposal.npz").exists() or (d / "trajectory_hands.npz").exists():
            if (d / "outcomes.json").exists():
                found.append(d)
    return found


def _attempt_score(d):
    """(Stage 1 subgoals covered, plan entries passed, recency) of an executor folder."""
    try:
        outs = json.loads((Path(d) / "outcomes.json").read_text())
    except (OSError, ValueError):
        return (0, 0, 0.0)
    covered = {int(i) for o in outs if o.get("frames") for i in (o.get("stage1_indices") or [o.get("index")]) if i is not None}
    passed = sum(1 for o in outs if o.get("ok") and o.get("frames") and not o.get("dry_run"))
    return (len(covered), passed, (Path(d) / "outcomes.json").stat().st_mtime)


def reference_from_attempt(path, model, out_path, sem=None):
    """The agent's best physical attempt as a tracking reference: of the execute_subgoal_plan folders under `path` (a
    Stage 3 workspace or one folder) the one whose executed entries cover the most Stage 1 subgoals, then pass the most,
    then the newest; its commanded reference (trajectory_proposal.npz `reference`: wrists, fingers, coupled arms, the
    planned object poses) with one segment per executed plan entry (its Stage 1 indices, motion, side, object from
    outcomes.json). Entries that never reached physics are not in it. Writes `out_path`; returns (path, info)."""
    p = Path(path)
    dirs = [p] if (p / "outcomes.json").exists() else _attempt_dirs(p)
    if not dirs:
        raise FileNotFoundError(f"{p} holds no physical execute_subgoal_plan attempt (trajectory_proposal.npz with outcomes.json)")
    best = max(dirs, key=_attempt_score)
    src = best / "trajectory_proposal.npz" if (best / "trajectory_proposal.npz").exists() else best / "trajectory_hands.npz"
    ep = E.load_episode(src)
    Q = np.asarray(ep.get("reference", ep["qpos"]), float)
    if Q.shape[1] != model.nq:
        raise ValueError(f"{src}: {Q.shape[1]} columns but the scene has nq={model.nq}; pass the scene the attempt ran in")
    outs = json.loads((best / "outcomes.json").read_text())
    segs, segment = [], np.full(len(Q), -1, int)
    for o in outs:
        fr = o.get("frames")
        if not fr or o.get("dry_run"):
            continue
        a, b = int(fr[0]), min(int(fr[1]), len(Q) - 1)
        side = o.get("side") if o.get("side") in ("left", "right") else "both"
        obj = o.get("object") if isinstance(o.get("object"), str) else None
        segs.append({"stage1_indices": [int(i) for i in (o.get("stage1_indices") or [o["index"]])], "motion": o.get("motion"),
                     "side": side, "object": obj, "ok": bool(o.get("ok")), "tag": o.get("tag")})
        segment[a:b + 1] = len(segs) - 1
    if not segs:
        raise ValueError(f"{best}: no executed entry has frames")
    last = len(segment) - 1 - int(np.argmax(segment[::-1] >= 0))
    Q, labels, segment = Q[:last + 1], [str(l) for l in ep["labels"][:last + 1]], segment[:last + 1]
    for t in range(len(segment)):                     # a frame before the first entry belongs to it
        if segment[t] < 0:
            segment[t] = segment[t - 1] if t else int(segment[segment >= 0][0])
    out = save_reference(out_path, model, Q, labels, ep["fps"], segment, segs, source=f"agent attempt {best}", arms_coupled=True)
    covered = sorted({i for s in segs for i in s["stage1_indices"]})
    missing = sorted({int(g["index"]) for g in (sem or {}).get("subgoals", [])} - set(covered))
    return out, {"attempt": str(best), "frames": int(len(Q)), "entries": len(segs), "entries_passed": sum(s["ok"] for s in segs),
                 "stage1_covered": covered, "stage1_missing": missing}


def best_reference(workspace, model, out_path, given=None, sem=None):
    """The reference a method starts from: `given` (an earlier method's tracking reference) when there is one, else the
    agent's best attempt in the workspace."""
    if given:
        ref = load_reference(given, model)
        return ref["path"], {"source": ref["source"], "frames": int(len(ref["qpos"]))}
    return reference_from_attempt(workspace, model, out_path, sem=sem)


def scene_for(workspace, scene_recon_json, scene=None):
    """The scene the agent's attempts ran in: `scene`, the stations scene place_arm_stations wrote, the manifest's
    scene_executed, else Stage 2's."""
    if scene:
        return Path(scene).resolve()
    ws = Path(workspace)
    pj = ws / "place_arm_stations.json"
    if pj.exists():
        try:
            sp = Path(json.loads(pj.read_text()).get("scene", ""))
            # absolute, relative to the workspace, beside it, or relative to the run folder (a reused example's record)
            for cand in ([sp] if sp.is_absolute() else [ws / sp, ws / sp.name] + ([ws.parents[2] / sp] if len(ws.parents) > 2 else [])):
                if str(sp) and cand.is_file():
                    return cand.resolve()
        except ValueError:
            pass
    recon = json.loads(Path(scene_recon_json).read_text())
    return (Path(scene_recon_json).parent / recon.get("scene_xml", "scene.xml")).resolve()


def couple_as_given(model, Q, labels, fps, parts=None):
    """The rows' own arm columns as the arm references, every hand put on its flange, in the shape armhand.couple
    returns: for a reference whose arms already hold its hands (an executed attempt's commands, a method's refined
    commands), so the executor runs exactly what the method optimized. A joint step above the coupling's jump_rad is
    reported as a branch jump, which the executor refuses as it would after couple."""
    from tool_library.common import armhand as AH
    parts = parts or E.robot_parts(model)
    Q = np.array(Q, float)
    arms, report = {}, {"ok": True, "infeasible": [], "retimed_s": 0.0, "given": True}
    for side, pt in parts.items():
        if pt["gripper"] < 0 or not pt["arm_joints"]:
            continue
        S = AH.ArmSide(model, side, parts=parts)
        A = Q[:, pt["arm_qadr"]].copy()
        for t in range(len(Q)):
            Q[t] = S.put_hand(Q[t], A[t])
        step = np.abs(np.diff(A, axis=0)).max(axis=1) if len(A) > 1 else np.zeros(0)
        jumps = np.where(step > AH.DEFAULTS["jump_rad"])[0] + 1
        report[side] = {"mount": S.mount_source, "frames": int(len(Q)), "given": True, "branch_jumps": [int(j) for j in jumps[:8]],
                        "ik_residual_mm_max": 0.0, "ik_residual_deg_max": 0.0, "projected_frames": 0, "free_leg_frames": 0,
                        "peak_speed_deg_s": round(float(np.degrees(step.max()) * fps), 1) if step.size else 0.0}
        if len(jumps):
            report["ok"] = False
        arms[side] = A
    return {"qpos": Q, "arm": arms, "labels": list(labels), "src_index": np.arange(len(Q), dtype=float), "report": report,
            "legs": {}, "arm_start_used": {s: A[0].copy() for s, A in arms.items()}}


# ----------------------------------------------------------------------------------------------- the plan --
def _stage1(sem, k):
    return next((g for g in (sem or {}).get("subgoals", []) if int(g.get("index", -1)) == int(k)), {})


def _object_body(model, sem, recon, k):
    """The body a Stage 1 subgoal acts on: its first object ref's root body in the reconstruction."""
    refs = _stage1(sem, k).get("object_refs") or []
    roots = {o.get("id"): o.get("root_body", o.get("id")) for o in (recon or {}).get("objects", [])}
    for r in refs:
        for name in (roots.get(r), r):
            if name and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)) >= 0:
                return str(name)
    return None


def _busier_hand(model, Q, a, b):
    from tool_library.common import plan as P
    rigs = P.scene_rigs(model)
    travel = {s: float(np.abs(np.diff(Q[a:b + 1][:, [r.wrist_qpos[k] for k in ("x", "y", "z")]], axis=0)).sum()) for s, r in rigs.items()}
    return max(travel, key=travel.get) if travel else "right"


def write_plan(plan_path, reference, model, sem=None, recon=None, scene_recon_json=None, method=None) -> dict:
    """An execute_subgoal_plan plan that follows `reference`: one `track_reference` entry per contiguous segment, with
    its Stage 1 indices, the acting side and object (the segment's, else Stage 1's; a two-hand segment is filed under the
    hand that moves more, both hands still follow the rows), the frames, and the expectations of
    the segment's own motion with the reference's numbers (spider_bridge.stage1_expect: a grasp rises and holds, a place
    ends released and at rest...) on top of the generic tracking ones the executor writes. Returns the plan dict."""
    from tool_library.common import spider_bridge as SB
    ref = load_reference(reference, model)
    Q = ref["qpos"]
    plan_path = Path(plan_path)
    rel = Path(ref["path"])
    try:
        rel = rel.relative_to(plan_path.parent.resolve())
    except ValueError:
        pass
    entries = []
    for sid, a, b in runs(ref["segment"]):
        seg = ref["segments"][sid]
        s1 = [int(i) for i in seg["stage1_indices"]]
        g = _stage1(sem, s1[0])
        side = seg.get("side") if seg.get("side") in ("left", "right") else g.get("acting_hand")
        both = side not in ("left", "right")
        if both:                                      # a two-hand segment is filed under the hand that moves more
            side = _busier_hand(model, Q, a, b)
        obj = seg.get("object") or _object_body(model, sem, recon, s1[0])
        e = {"index": s1[0], "stage1_indices": s1, "motion": "track_reference", "side": side, "reference": str(rel),
             "frames": [int(a), int(b)], "source_motion": seg.get("motion") or g.get("motion")}
        if both:
            e["both_hands"] = True
        if ref["arms_coupled"]:
            e["arms"] = "reference"                       # the rows' arms are run as given (couple_as_given)
        if obj:
            e["object"] = obj
        if method:
            e["method"] = method
        motion = seg.get("motion") or g.get("motion")
        if not both and obj and motion:
            try:
                ex = SB.stage1_expect({"motion": motion, "object_refs": g.get("object_refs")}, model, Q, [a, b], side, obj)
            except Exception:                             # an expectation the reference cannot give is left to the generic one
                ex = None
            if ex:
                e["expect"] = {k: v for k, v in ex.items() if k in ("rise_m", "in_hand", "no_slip", "released", "at_rest", "supported",
                                                                      "upright", "target_axes")}
        entries.append(e)
    plan = {"subgoals": entries, "reference": str(rel), "source": ref["source"]}
    if scene_recon_json:
        plan["scene_recon_json"] = str(Path(scene_recon_json).resolve())
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=1))
    return plan


def coverage(reference, model, sem):
    """Stage 1 subgoals a reference covers and misses."""
    ref = load_reference(reference, model)
    covered = sorted({int(i) for s in ref["segments"] for i in s["stage1_indices"]})
    want = sorted(int(g["index"]) for g in (sem or {}).get("subgoals", []))
    return {"covered": covered, "missing": [k for k in want if k not in covered]}


# ------------------------------------------------------------------------------------------ the simulation --
class RuntimeSim:
    """The picklable half of BenchTracker: the rigid runtime scene (its XML file), the base commands in runtime
    coordinates, the frames' targets and the residual layout. Rollouts call rigid_robot.servo_rollout, the executor's
    own physics, from a complete integration state."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.model = mujoco.MjModel.from_xml_path(spec["runtime_xml"])
        self.fps = float(spec["fps"])
        self.base = np.asarray(spec["base"], float)
        self.T = len(self.base)
        self.rules = spec.get("rules") or []
        self.bodies = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in spec["bodies"]}
        self.joints = {n: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in spec["joints"]}
        free = {int(self.model.jnt_bodyid[j]): int(self.model.jnt_dofadr[j]) for j in range(self.model.njnt)
                if int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)}
        self.body_dof = {n: free.get(b) for n, b in self.bodies.items()}      # a tracked part's free-joint velocity address
        self.tpos = np.asarray(spec["target_pos"], float).reshape(self.T, len(self.bodies), 3)
        self.tquat = np.asarray(spec["target_quat"], float).reshape(self.T, len(self.bodies), 4)
        self.tjoint = np.asarray(spec["target_joint"], float).reshape(self.T, len(self.joints))
        self.sides = list(spec["sides"])
        self.arm_cols = {s: np.asarray(spec["arm_cols"][s], int) for s in self.sides}
        self.flex_cols = {s: np.asarray(spec["flex_cols"][s], int) for s in self.sides}
        lo, hi = self.model.jnt_range[:, 0], self.model.jnt_range[:, 1]
        jnt_of_col = np.full(self.model.nq, -1)
        for j in range(self.model.njnt):
            jnt_of_col[int(self.model.jnt_qposadr[j])] = j
        self.flex_lo = {s: np.array([lo[jnt_of_col[c]] for c in self.flex_cols[s]]) for s in self.sides}
        self.flex_hi = {s: np.array([hi[jnt_of_col[c]] for c in self.flex_cols[s]]) for s in self.sides}
        self.data = mujoco.MjData(self.model)
        self.weights = {"pos": 10.0, "rot": 0.5, "joint": 2.0, "pen": 100.0, "force": 0.05, **(spec.get("weights") or {})}

    # the residual: per side six arm joint offsets (rad) and one grip offset (rad along every flexion joint)
    @property
    def dim(self):
        return 7 * len(self.sides)

    def apply(self, commands, residual):
        """commands [n, nq] runtime rows, residual [n, dim]: the commands with the residual added (fingers clipped)."""
        out = np.array(commands, float)
        residual = np.asarray(residual, float).reshape(len(out), self.dim)
        for i, s in enumerate(self.sides):
            r = residual[:, 7 * i:7 * i + 7]
            out[:, self.arm_cols[s]] += r[:, :6]
            out[:, self.flex_cols[s]] = np.clip(out[:, self.flex_cols[s]] + r[:, 6:7], self.flex_lo[s], self.flex_hi[s])
        return out

    def rollout(self, t0, t1, commands, state=None, mechanism=None):
        """Physics from frame t0 (the integration `state` there; the base's first row at rest when None) through frame
        t1 under `commands` [t1 - t0 + 1, nq] (row 0 is frame t0's command). Returns {qpos, qvel, final_state,
        mechanism_final, unstable, hand_pen_m}."""
        from tool_library.common import rigid_robot as RR
        kw = {"state_start": np.asarray(state, float)} if state is not None else {"qpos_start": self.base[0]}
        r = RR.servo_rollout(self.model, np.asarray(commands, float), self.fps, rules=self.rules, mechanism_state_start=mechanism,
                             record_contacts=False, **kw)
        return {"qpos": r["qpos"], "qvel": r["qvel"], "final_state": r["final_state"], "mechanism_final": r["mechanism_final"],
                "unstable": r["unstable"], "hand_pen_m": r["hand_pen_m"], "hand_object_force_n": r["hand_object_force_n"]}

    def errors(self, t0, qpos):
        """Per frame of `qpos` (frames t0, t0+1, ...): position error [n, nb] (m), rotation error [n, nb] (rad) of the
        tracked bodies and joint error [n, nj] against the reference."""
        n = len(qpos)
        ep, er, ej = np.zeros((n, len(self.bodies))), np.zeros((n, len(self.bodies))), np.zeros((n, len(self.joints)))
        d, m = self.data, self.model
        for i in range(n):
            t = min(t0 + i, self.T - 1)
            d.qpos[:] = qpos[i]; mujoco.mj_kinematics(m, d)
            for k, (name, bid) in enumerate(self.bodies.items()):
                ep[i, k] = np.linalg.norm(d.xpos[bid] - self.tpos[t, k])
                q = d.xquat[bid]; qt = self.tquat[t, k]
                er[i, k] = 2.0 * np.arccos(np.clip(abs(float(np.dot(q, qt))), 0.0, 1.0))
            for k, (name, adr) in enumerate(self.joints.items()):
                ej[i, k] = abs(float(qpos[i][adr]) - self.tjoint[t, k])
        return ep, er, ej

    def cost(self, t0, roll, residual=None, w_u=0.0):
        """Tracking cost of a rollout from frame t0: object position (per m), rotation (per rad), joint error, hand
        penetration past 3 mm, hand-object force past FORCE_SOFT_N (the gates' contact spike audit allows 15 N), an
        unstable integration, and optionally the residual's size."""
        w = self.weights
        if roll["unstable"]:
            return 1e3
        ep, er, ej = self.errors(t0, roll["qpos"])
        c = w["pos"] * ep.sum() + w["rot"] * er.sum() + w["joint"] * ej.sum()
        c += w["pen"] * float(np.clip(np.asarray(roll["hand_pen_m"]) - 0.003, 0, None).sum())
        c += w["force"] * float(np.clip(np.asarray(roll["hand_object_force_n"]) - FORCE_SOFT_N, 0, None).sum())
        if residual is not None and w_u:
            c += w_u * float(np.square(residual).sum())
        return float(c) / max(1, len(roll["qpos"]))

    def summary(self, qpos):
        ep, er, ej = self.errors(0, qpos)
        out = {"frames": int(len(qpos)), "bodies": {}, "joints": {}}
        for k, name in enumerate(self.bodies):
            out["bodies"][name] = {"mean_mm": round(1000 * float(ep[:, k].mean()), 1), "max_mm": round(1000 * float(ep[:, k].max()), 1),
                                   "end_mm": round(1000 * float(ep[-1, k]), 1), "end_deg": round(float(np.degrees(er[-1, k])), 1)}
        for k, name in enumerate(self.joints):
            out["joints"][name] = {"end": round(float(ej[-1, k]), 4), "max": round(float(ej[:, k].max()), 4)}
        out["worst_mean_mm"] = max([v["mean_mm"] for v in out["bodies"].values()] or [0.0])
        out["worst_end_mm"] = max([v["end_mm"] for v in out["bodies"].values()] or [0.0])
        return out


class BenchTracker:
    """A tracking reference on the bench as the executor will run it: from the Stage 2 home state (key 0, at rest) to
    the reference's first frame, then frame by frame under the executor's joint-rate ceilings (chain_subgoals), the arms
    as given when the reference's arms hold its hands (couple_as_given), else coupled (armhand.couple with the
    executor's settings and arm planner), in the rigid runtime scene. `sim` is the
    RuntimeSim over those commands; `export` writes optimized runtime commands back as a tracking reference."""

    def __init__(self, scene, reference, work_dir, scene_recon_json=None, plan_arm_paths=True):
        from tool_library.common import armhand as AH
        from tool_library.common import rigid_robot as RR
        from tool_library.skills.chain_subgoals import chain_subgoals
        self.work = Path(work_dir); self.work.mkdir(parents=True, exist_ok=True)
        self.scene = str(Path(scene).resolve())
        model, _ = E.build_execution_model(self.scene, out_xml=str(self.work / "scene_proposal.xml"), scene_recon_json=scene_recon_json)
        self.model = model
        self.runtime = RR.compile_rigid(model)
        self.runtime_xml = self.work / "scene_physics.xml"
        self.runtime_xml.write_text(self.runtime.xml)
        self.ref = ref = load_reference(reference, model)
        self.fps = fps = ref["fps"]
        d = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        mujoco.mj_forward(model, d)
        q_home = d.qpos.copy()
        given = bool(ref["arms_coupled"])                # the rows' arms hold their hands: run them as given
        cols = np.arange(model.nq) if given else planning_columns(model)
        Qr = ref["qpos"]
        q_first = q_home.copy(); q_first[cols] = Qr[0, cols]
        parts = E.robot_parts(model)
        rigs = {s: pt["rig"] for s, pt in parts.items()}
        dist = max([float(np.linalg.norm(q_first[[r.wrist_qpos[k] for k in ("x", "y", "z")]] - q_home[[r.wrist_qpos[k] for k in ("x", "y", "z")]]))
                    for r in rigs.values()] or [0.0])
        approach = max(1, int(round(np.clip(dist / 0.15, 1.0 / fps, 3.0) * fps)))
        kfs = [{"qpos": q_home.tolist(), "hold": 1, "label": ref["labels"][0]}]
        for t in range(len(Qr)):
            q = q_home.copy() if t == 0 else np.asarray(kfs[-1]["qpos"], float)
            q[cols] = Qr[t, cols]
            kfs.append({"qpos": q.tolist(), "transit": approach if t == 0 else 1, "hold": 1, "label": ref["labels"][t]})
        # rows whose arms hold their hands are already timed: one per frame, unstretched, as the executor runs them
        ch = chain_subgoals(model, kfs, fps=fps, max_rate_deg_s=None if given else RATES[0], max_rate_m_s=None if given else RATES[1])
        if given:
            cp = couple_as_given(model, ch["qpos"], ch["labels"], fps, parts)
        else:
            arm_start = {s: q_home[pt["arm_qadr"]].copy() for s, pt in parts.items() if pt["gripper"] >= 0 and pt["arm_joints"]}
            planner = None
            if plan_arm_paths:
                from tool_library.common import armplan as AP
                planner = AP.ArmPlanner(self.scene)
            cp = AH.couple(model, ch["qpos"], fps, arm_start=arm_start, labels=ch["labels"], keyframe_frames=ch["keyframe_frames"],
                           cfg=AH_CFG, parts=parts, max_frames=max(3000, 2 * len(ch["qpos"])), planner=planner)
        self.couple_report = {"ok": cp["report"]["ok"], "infeasible": cp["report"]["infeasible"][:4], "retimed_s": cp["report"].get("retimed_s"),
                              "stretched": len(ch["stretched"]), "arms": "as given" if given else "coupled"}
        self.Qc, self.labels = np.asarray(cp["qpos"], float), list(cp["labels"])
        # which reference row each command frame stands for, mapped exactly as its label was: the coupling's retiming
        # labels a frame by the chain frame below it (floor of its source index), and chain_subgoals labels a frame by
        # the keyframe its leg leads to (keyframe 0 is home, keyframe k is row k - 1), so segments and labels agree at
        # every boundary
        kf = np.asarray(ch["keyframe_frames"], int)
        chain_frame = np.floor(np.asarray(cp["src_index"], float) + 1e-9).astype(int)
        self.ref_row = np.clip(np.searchsorted(kf, chain_frame, side="left") - 1, 0, len(Qr) - 1).astype(int)
        self.segment = ref["segment"][self.ref_row]
        self.Qrt = self.runtime.to_runtime_qpos(self.Qc)
        rt = self.runtime.model
        bodies, joints = tracked_parts(model, Qr)
        # targets: the tracked parts' poses in the reference, on the command time line
        dm = mujoco.MjData(model)
        T = len(self.Qc)
        tpos, tquat = np.zeros((T, len(bodies), 3)), np.zeros((T, len(bodies), 4))
        tjoint = np.zeros((T, len(joints)))
        bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in bodies]
        for t in range(T):
            dm.qpos[:] = self.Qc[t]; mujoco.mj_kinematics(model, dm)
            for k, b in enumerate(bids):
                tpos[t, k] = dm.xpos[b]; tquat[t, k] = dm.xquat[b]
            for k, adr in enumerate(joints.values()):
                tjoint[t, k] = self.Qc[t, adr]
        rparts = E.robot_parts(rt)
        sides = [s for s, pt in rparts.items() if pt["arm_qadr"]]
        flex = {}
        for s in sides:
            rig = rparts[s]["rig"]
            cols_ = [int(q) for _, q in E.actuator_targets(rt, rig)]
            names = {int(rt.jnt_qposadr[j]): rt.joint(j).name for j in range(rt.njnt)}
            flex[s] = [c for c in cols_ if FLEX.search(names.get(c, ""))]
        from tool_library.skills.execute_subgoal_plan import cut_rules_from_recon, latch_rules_from_recon, thread_rules_from_recon
        rules = []
        if scene_recon_json:
            rules = (thread_rules_from_recon(scene_recon_json) + cut_rules_from_recon(scene_recon_json)
                     + latch_rules_from_recon(scene_recon_json))
        self.spec = {"runtime_xml": str(self.runtime_xml), "fps": fps, "base": self.Qrt, "rules": rules,
                     "bodies": list(bodies), "joints": list(joints), "target_pos": tpos, "target_quat": tquat, "target_joint": tjoint,
                     "sides": sides, "arm_cols": {s: rparts[s]["arm_qadr"] for s in sides}, "flex_cols": flex}
        self.sim = RuntimeSim(self.spec)

    def export(self, path, commands, source, **extra) -> str:
        """Runtime commands [T, nq] (the base with a residual) as a tracking reference in the planning scene's layout,
        on the command time line: wrists from the arms' FK, fingers and arms as commanded, the objects' targets kept."""
        commands = np.asarray(commands, float)
        Q = self.runtime.to_proposal_qpos(commands, seed=self.Qc[0])
        free = E.object_free_joints(self.model)
        for adr in free.values():                     # the targets stay the reference's (runtime object columns are states)
            Q[:, adr:adr + 7] = self.Qc[:len(Q), adr:adr + 7]
        segs = self.ref["segments"]
        return save_reference(path, self.model, Q, self.labels[:len(Q)], self.fps, self.segment[:len(Q)], segs, source,
                              arms_coupled=True, **extra)


# ------------------------------------------------------------------------------------------ worker processes --
_WORKER = {}


def worker_init(spec):
    """Process-pool initializer: one RuntimeSim per worker."""
    _WORKER["sim"] = RuntimeSim(spec)


def worker_sim():
    return _WORKER["sim"]


def make_pool(spec, workers):
    """A process pool whose workers each hold a RuntimeSim of `spec` (None for one worker: run in this process)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    workers = int(workers or 1)
    if workers <= 1:
        worker_init(spec)
        return None
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=worker_init, initargs=(spec,))


def default_workers(cfg_workers=None):
    import os
    if cfg_workers:
        return max(1, int(cfg_workers))
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
    return max(1, min(8, n - 1))


def elapsed_min(t0):
    return (time.time() - t0) / 60.0
