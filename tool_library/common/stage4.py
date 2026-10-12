"""Access to the Stage 4 robot-data-generation package that ships next to this library.

The photoreal rendering tools (clean plate, Blender bundle, Cycles render,
compositing) reuse the systematic Stage 4 implementation in
``DexAgent/stage4_robot_data_generation/stage4`` rather than duplicating it.
Set DEXAGENT_STAGE4_DIR to point at another checkout.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

STAGE4_ROOT = Path(os.environ.get("DEXAGENT_STAGE4_DIR", Path(__file__).resolve().parents[2] / "stage4_robot_data_generation"))


def stage4_module(name: str):
    """`import stage4.<name>` from the Stage 4 package root."""
    if not (STAGE4_ROOT / "stage4" / f"{name}.py").exists():
        raise ImportError(f"stage4 package not found at {STAGE4_ROOT} (set DEXAGENT_STAGE4_DIR)")
    if str(STAGE4_ROOT) not in sys.path:
        sys.path.insert(0, str(STAGE4_ROOT))
    return importlib.import_module(f"stage4.{name}")


def load_runtime_episode(episode_dir, scene_recon_json=None):
    """Load the executor's real scene/state pair, never its floating-wrist proposal."""
    directory = Path(episode_dir)
    return load_runtime_pair(directory / "scene_physics.xml", directory / "trajectory.npz", scene_recon_json)


def load_runtime_pair(scene, trajectory, scene_recon_json=None):
    """Validate a portable runtime scene and its complete integration recording."""
    import hashlib
    import mujoco
    from tool_library.common import rigid_robot as R
    from tool_library.verifiers.physics_recording_check import physics_recording_check
    scene, trajectory = Path(scene), Path(trajectory)
    if not scene.is_file() or not trajectory.is_file():
        raise ValueError("Stage 4 needs scene_physics.xml and trajectory.npz from a rigid robot execution")
    model = mujoco.MjModel.from_xml_path(str(scene))
    if not R.is_rigid(model):
        raise ValueError("Stage 4 refuses a proposal scene: hands must be fixed children of the arm flanges")
    with np.load(trajectory, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if str(np.asarray(arrays.get("robot_schema", "")).item()) != R.SCHEMA:
        raise ValueError("trajectory is missing the rigid robot execution schema")
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    if str(np.asarray(arrays.get("scene_sha256", "")).item()) != digest:
        raise ValueError("runtime scene does not match the scene hash recorded by the executor")
    qpos = np.asarray(arrays.get("qpos"), float)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos):
        raise ValueError("runtime trajectory qpos does not match scene_physics.xml")
    for key, width in (("qvel", model.nv), ("ctrl", model.nu), ("reference", model.nq), ("eq_active", model.neq)):
        values = np.asarray(arrays.get(key), float)
        if values.shape != (len(qpos), width) or not np.isfinite(values).all():
            raise ValueError(f"runtime trajectory {key} is missing, non-finite or has the wrong shape")
    for key, width in (("initial_qpos", model.nq), ("initial_qvel", model.nv)):
        values = np.asarray(arrays.get(key), float)
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError(f"runtime trajectory {key} is missing or invalid")
    if not np.isfinite(qpos).all():
        raise ValueError("runtime trajectory contains non-finite positions")
    recording = physics_recording_check(str(scene), str(trajectory), scene_recon_json=scene_recon_json)
    if not recording.get("ok"):
        raise ValueError("invalid runtime dynamics recording: " + "; ".join(recording.get("issues", [])))
    return scene, trajectory, model, arrays


def copy_runtime_scene(scene, destination):
    """Copy an already portable runtime scene byte-for-byte with its assets.

    Publication must not reserialize the model after simulation. Stage 4 task
    handoffs flatten includes and make assets relative before replay begins.
    """
    import shutil
    import xml.etree.ElementTree as ET
    scene, destination = Path(scene).resolve(), Path(destination)
    root = ET.fromstring(scene.read_bytes())
    if root.findall(".//include"):
        raise ValueError("runtime scene has includes; create a portable Stage 4 handoff before replay")
    compiler = root.find("compiler")
    directories = {"mesh": compiler.get("meshdir", "") if compiler is not None else "",
                   "texture": compiler.get("texturedir", "") if compiler is not None else ""}
    dependencies = []
    for asset in root.findall("./asset/*"):
        if not asset.get("file"):
            continue
        relative = Path(directories.get(asset.tag, "")) / asset.get("file")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("runtime scene assets must be portable relative paths; regenerate the Stage 4 handoff")
        source = scene.parent / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        dependencies.append((source, destination.parent / relative))
    destination.parent.mkdir(parents=True, exist_ok=True)
    for source, target in dependencies:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    if scene != destination.resolve():
        shutil.copy2(scene, destination)
    return destination


class Cfg(dict):
    """Minimal stand-in for stage4.config.Config: a dict with .root/.out/.path()."""

    def __init__(self, d, root, out):
        super().__init__(d)
        self.root, self.out = Path(root), Path(out)

    def path(self, key):
        return self.path_of(self.get(key))

    def path_of(self, value):
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.root / p


def defaults():
    return stage4_module("config").DEFAULTS


# ----------------------------------------------------------------------------- Stage 3 -> Stage 4 inference
import json as _json
import re as _re

CARRY_WORDS = ("lift", "carry", "set down", "insert", "pour", "tilt", "lower")
LETGO_WORDS = ("let go", "release", "relax")


def side_text(label, side=None):
    """What one hand does in a phase label: 'left: carry cup (up) #3' -> 'carry cup (up)'. A two-hand label
    'left: X | right: Y #4' gives X for the left hand and Y for the right one; without a side (or for a 'both: ...'
    label) the first part's text."""
    text = _re.sub(r"\s+#[\d.]+$", "", str(label)).strip()
    parts = [x.strip() for x in text.split(" | ")]
    if side:
        for x in parts:
            if x.startswith(f"{side}: "):
                return x[len(side) + 2:]
    return parts[0].split(": ", 1)[-1]


def phase_verb(label, side=None):
    """The lower-case action of one hand in a phase label (see side_text)."""
    return side_text(label, side).lower()


def names_object(phase, obj):
    """Does a phase act on obj? Its `objects` list when it has one, else obj as a word in its hand's part of the
    label."""
    objs = phase.get("objects")
    if isinstance(objs, list) and objs:
        return obj in objs
    text = side_text(phase.get("label", ""), phase.get("hand"))
    return _re.search(rf"(?<![\w]){_re.escape(obj)}(?![\w])", text) is not None


def object_roles(phases, objects, moved=None, min_travel_m=0.02):
    """Carrier and release of every object, read from the phases that act on THAT object.

    carrier: the hand whose own lift/carry/set-down phase names the object (the first such hand; a second hand
    that carries it later is reported); release: that hand's first let-go phase naming the object after its first
    carry phase. A phase that is merely anchored to the object (the hand that stacks another brick on it) is
    neither. An object that travels less than min_travel_m in the trajectory has no carrier (it is a destination
    or a support). Returns ({object: {"carrier", "release"}}, warnings)."""
    roles, warnings = {}, []
    for obj in objects:
        if moved is not None and moved.get(obj, 1.0) < min_travel_m:
            continue
        first = None
        for i, p in enumerate(phases):
            if p.get("hand") and names_object(p, obj) and phase_verb(p["label"], p["hand"]).startswith(CARRY_WORDS):
                first = i
                break
        if first is None:
            continue
        side = phases[first]["hand"]
        others = sorted({p["hand"] for p in phases if p.get("hand") and p["hand"] != side and names_object(p, obj)
                         and phase_verb(p["label"], p["hand"]).startswith(CARRY_WORDS)})
        if others:
            warnings.append(f"{obj}: carried by {side} first and later by {others}; Stage 4 follows one carrier ({side})")
        rel = next((p["label"] for p in phases[first + 1:] if p.get("hand") == side and names_object(p, obj)
                    and phase_verb(p["label"], side).startswith(LETGO_WORDS)), None)
        roles[obj] = {"carrier": side, "release": rel}
    return roles, warnings


def object_rides(phases, objects, moved=None, min_travel_m=0.02):
    """Every carry of every object, in order: {object: [{"side", "release"}, ...]}.

    The first carry is object_roles' (the first hand whose own lift/carry/set-down phase names the object, released
    by that hand's first let-go naming it). After a release, the next carry is the next phase (of either hand) that
    lifts, carries or sets down the object again, with that hand's next let-go naming it (a part put down by one hand
    and picked up by the other, or handed over: the receiving hand carries it once the giving hand lets go). A carry
    with no let-go ends the list (the object stays in that hand). Objects that travel less than min_travel_m are left
    out, as in object_roles."""
    out = {}
    for obj in objects:
        if moved is not None and moved.get(obj, 1.0) < min_travel_m:
            continue
        rides, i = [], 0
        while i < len(phases):
            first = next((k for k in range(i, len(phases)) if phases[k].get("hand") and names_object(phases[k], obj)
                          and phase_verb(phases[k]["label"], phases[k]["hand"]).startswith(CARRY_WORDS)), None)
            if first is None:
                break
            side = phases[first]["hand"]
            rel = next((k for k in range(first + 1, len(phases)) if phases[k].get("hand") == side and names_object(phases[k], obj)
                        and phase_verb(phases[k]["label"], side).startswith(LETGO_WORDS)), None)
            rides.append({"side": side, "release": phases[rel]["label"] if rel is not None else None})
            if rel is None:
                break
            i = rel + 1
        if rides:
            out[obj] = rides
    return out


def fixed_bases(model, candidates):
    """The candidate roots that are fixed to the world (a child of the world with no joint of its own): a cabinet, a
    holder bolted to the table, the base of an articulated object whose parts move on hinges or slides. Stage 4 moves
    such a base as a whole (the model's body pose); its parts keep their joint values."""
    out = []
    for c in candidates:
        b = mujoco_body(model, c)
        if b is not None and int(model.body_parentid[b]) == 0 and int(model.body_jntnum[b]) == 0:
            out.append(c)
    return out


def mujoco_body(model, name):
    import mujoco
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name)) if name else -1
    return int(b) if b >= 0 else None


# ----------------------------------------------------------------------------- the executable plan Stage 4 re-executes
def _resolve(path, bases):
    p = Path(str(path))
    if p.is_absolute():
        return p if p.exists() else None
    for b in bases:
        if b is not None and (Path(b) / p).exists():
            return (Path(b) / p).resolve()
    return p.resolve() if p.exists() else None


def _poses_from(v, bases):
    """Hand poses named by a plan field: a pose dict, a JSON file (a pose, a ranked list, {grasps|poses: [...]},
    {pose: ...}), or a list of those."""
    if v is None:
        return []
    if isinstance(v, list):
        return [p for x in v for p in _poses_from(x, bases)]
    if isinstance(v, dict):
        if "pos" in v and "euler" in v:
            return [v]
        for k in ("grasps", "poses"):
            if isinstance(v.get(k), list):
                return _poses_from(v[k], bases)
        return _poses_from(v.get("pose"), bases) if isinstance(v.get("pose"), dict) else []
    p = _resolve(v, bases)
    if p is None:
        return []
    try:
        return _poses_from(_json.loads(p.read_text()), bases)
    except Exception:
        return []


def _with_anchor(sg, pose):
    """The pose with the object pose it was solved against (`anchor`), read the way the executor reads it (the pose's
    own anchor, else the grasp body of the hand scene the entry names): the plan then needs no hand scene."""
    if not isinstance(pose, dict) or pose.get("anchor"):
        return pose
    try:
        from tool_library.common.subgoal import pose_anchor
        anc = pose_anchor(sg, pose)
    except Exception:
        anc = None
    return dict(pose, anchor=anc) if anc else pose


def _inline_entry(e, used, ep_dir, tag, bases, extra=None, notes=None):
    """One executed entry with its hand poses inlined: the pose that ran first (pose_used), then the other candidates
    the plan offered, each anchored; hand scenes resolved to absolute paths (dropped when missing)."""
    e = {k: v for k, v in e.items() if not str(k).startswith("_")}
    extra = extra or {}
    if e.get("motion") == "both" and isinstance(e.get("hands"), dict):
        used = used if isinstance(used, dict) else {}
        hands = {}
        for s_, h in e["hands"].items():
            u = used.get(s_) or (Path(ep_dir) / str(tag) / f"pose_used_{s_}.json")
            hands[s_] = _inline_entry(dict(h, side=h.get("side", s_)), u, ep_dir, None, bases,
                                      extra=(extra.get("hands") or {}).get(s_), notes=notes)
        e["hands"] = hands
        return e
    hs = e.get("hand_scene") or extra.get("hand_scene")
    if isinstance(hs, str):
        hp = _resolve(hs, bases)
        if hp is not None:
            e["hand_scene"] = str(hp)
        else:
            e.pop("hand_scene", None)
    cands = _poses_from(e.get("pose"), bases) + _poses_from(extra.get("pose"), bases) + _poses_from(e.get("pose_candidates"), bases)
    up = None
    if isinstance(used, (str, Path)):
        up = _resolve(used, bases) or (Path(ep_dir) / str(tag) / "pose_used.json" if tag else None)
    if up is None and tag:
        up = Path(ep_dir) / str(tag) / "pose_used.json"
    first = _poses_from(str(up), bases) if up is not None and Path(up).exists() else []
    if e.get("pose") is not None or first or cands:
        seen, poses = [], []
        for c in first + cands:
            key = _json.dumps(c, sort_keys=True, default=float)
            if key not in seen:
                seen.append(key)
                poses.append(_with_anchor(e, c))
        if not poses and notes is not None:
            notes.append(f"entry {e.get('index')} {e.get('motion')}: its pose {e.get('pose')!r} could not be read")
        if poses:
            e["pose"] = poses[0] if len(poses) == 1 else poses[:48]
        e.pop("pose_candidates", None)
    if e.get("regrasp_pose") is not None:
        rp = _poses_from(e["regrasp_pose"], bases)
        if rp:
            e["regrasp_pose"] = _with_anchor(e, rp[0])
    return e


def executable_plan(ep_dir, manifest=None, plan=None, n_frames=None):
    """The executed Stage 3 plan as Stage 4 re-executes it on a drawn layout: (entries, notes).

    One entry per executed plan entry of the execution folder (outcomes.json, in order): the variant that ran (its
    segment's keyframes.json `subgoal`, else the matching entry of `plan` or of the manifest's subgoals), with every
    hand pose inlined (the pose that ran first, then the other candidates the plan offered) and anchored to the object
    pose it was solved against, and `stage3_frames` [first, last]: its window in the delivered trajectory. Returns
    (None, notes) when the folder cannot give the whole plan (no outcomes.json, a segment with no record of its entry,
    windows that do not add up to the trajectory)."""
    ep_dir = Path(ep_dir)
    notes = []
    oc_p = ep_dir / "outcomes.json"
    if not oc_p.exists():
        return None, [f"no executable plan handed off: {oc_p} is missing (the trajectory was not written by "
                      "execute_subgoal_plan)"]
    outcomes = _json.loads(oc_p.read_text())
    man_p = Path(manifest) if manifest else None
    plan_p = Path(plan) if isinstance(plan, (str, Path)) else None
    bases = [ep_dir, ep_dir.parent, man_p.parent if man_p else None, plan_p.parent if plan_p else None, Path.cwd()]
    pool = []
    if plan is not None:
        pv = _json.loads(plan_p.read_text()) if plan_p else plan
        pool = list(pv.get("subgoals", []) if isinstance(pv, dict) else pv)
    elif man_p is not None and man_p.exists():
        pool = [sg["plan"] for sg in _json.loads(man_p.read_text()).get("subgoals", [])
                if isinstance(sg, dict) and isinstance(sg.get("plan"), dict) and not sg.get("skipped")]
    entries, t, j = [], 0, 0
    for r in outcomes:
        n = int(r.get("n_frames") or 0)
        if not n:
            continue
        sig = (r.get("index"), r.get("motion"))
        k = next((k for k in range(j, len(pool)) if (pool[k].get("index"), pool[k].get("motion")) == sig), None)
        match = None
        if k is not None:
            match, j = pool[k], k + 1
        kf = ep_dir / str(r.get("tag")) / "keyframes.json" if r.get("tag") else None
        e = None
        if kf is not None and kf.exists():
            try:
                e = _json.loads(kf.read_text()).get("subgoal")
            except Exception:
                e = None
        e = e or match
        if not isinstance(e, dict):
            return None, notes + [f"no record of the plan entry of segment {r.get('tag')} (index {r.get('index')}, "
                                  f"{r.get('motion')}): pass the plan the executor ran (`plan`)"]
        e = _inline_entry(e, r.get("pose_used"), ep_dir, r.get("tag"), bases, extra=match, notes=notes)
        if r.get("turn_used") and not e.get("turn"):
            e["turn"] = r["turn_used"]
        e["stage3_frames"] = [t, t + n - 1]
        t += n
        entries.append(e)
    if not entries:
        return None, notes + ["the execution folder holds no executed segment"]
    if n_frames is not None and t != int(n_frames):
        return None, notes + [f"the execution folder's segments add up to {t} frames, the trajectory has {n_frames}: "
                              "it was edited after execution, so its plan cannot be re-executed as is"]
    return entries, notes


def leave_start_frames(phases, labels, carriers):
    """{object: first frame it is carried away from where it started}: its carrier's first carry / set-down phase
    naming it (a grasp and a lift still sit on the start, a carry heads elsewhere). Objects without a carrier stay
    on their start (a pushed object moves with it) and get no entry."""
    out = {}
    for obj, side in carriers.items():
        if not side:
            continue
        lab = next((p["label"] for p in phases if p.get("hand") == side and names_object(p, obj)
                    and phase_verb(p["label"], side).startswith(("carry", "set down", "insert", "lower", "pour", "tilt"))), None)
        if lab is not None:
            out[obj] = next((t for t, l in enumerate(labels) if l == lab), None)
    return out


def root_body(model, body):
    """The scene body an object hangs from (the child of the world), or None for a missing body."""
    import mujoco
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body)) if body else -1
    if b < 0:
        return None
    while int(model.body_parentid[b]) != 0:
        b = int(model.body_parentid[b])
    return model.body(b).name


def _subtree(model, bid):
    out, grow = {int(bid)}, True
    while grow:
        grow = False
        for b in range(model.nbody):
            if int(model.body_parentid[b]) in out and b not in out:
                out.add(b)
                grow = True
    return out


def resting_support(model, qpos, obj, candidates):
    """Which candidate body carries obj in the state qpos, and whether obj sits down inside it.

    Looks at the contacts between obj's subtree and each candidate's subtree whose normal is mostly vertical
    (a support, not a side touch) and picks the candidate with the most of them; `inside` when obj's lowest point is
    more than 5 mm below the candidate's top and its centre lies over the candidate. Returns (body, kind) or (None, None)."""
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    ob = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    if ob < 0:
        return None, None
    osub = _subtree(model, ob)
    subs = {c: _subtree(model, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c)) for c in candidates
            if c != obj and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c) >= 0}
    count = {}
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (b1 in osub) == (b2 in osub) or float(c.dist) > 0.002:
            continue
        other = b2 if b1 in osub else b1
        # the contact normal points from geom1 to geom2: the push on the object is upward when the other body is
        # below it (a brick stacked ON the object pushes it down and is not its support)
        up = float(c.frame[2]) * (1.0 if b2 in osub else -1.0)
        if up < 0.5:
            continue
        hit = next((name for name, sub in subs.items() if other in sub), None)
        key = hit or "__other__"                       # the table, or a body that is not a candidate
        count[key] = count.get(key, 0) + 1
    if not count:
        return None, None
    best = max(count, key=count.get)
    if best == "__other__":
        return None, None

    olo, _ = _world_aabb(model, data, osub)
    _slo, shi = _world_aabb(model, data, subs[best])
    centre = data.xpos[ob]
    inside = (olo[2] < shi[2] - 0.005) and _slo[0] <= centre[0] <= shi[0] and _slo[1] <= centre[1] <= shi[1] \
        and _over_hollow(model, data, subs[best], centre, shi[2], ob)
    return best, ("inside" if inside else "rests_on")


def _world_aabb(model, data, sub):
    """World axis-aligned box around the geoms of a body set (from each geom's own box)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in sub or int(model.geom_type[g]) == 0:     # planes have no box
            continue
        c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        R = data.geom_xmat[g].reshape(3, 3)
        pts = data.geom_xpos[g] + (c + signs * h) @ R.T
        lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
    return lo, hi


def _over_hollow(model, data, sub, centre, top, exclude):
    """A ray down over the object's centre meets the support more than 1 cm below the support's top: a container."""
    import mujoco
    gid = np.zeros(1, np.int32)
    groups = np.ones(6, np.uint8)
    z0 = top + 0.02
    for _ in range(12):
        dist = mujoco.mj_ray(model, data, np.array([centre[0], centre[1], z0]), np.array([0.0, 0.0, -1.0]), groups, 1,
                             int(exclude), gid)
        if dist < 0:
            return False
        if int(model.geom_bodyid[int(gid[0])]) in sub:
            return (z0 - dist) < top - 0.01
        z0 = z0 - dist - 0.001
    return False


def plan_entry(plan):
    """A manifest entry's plan as one dict: a plan given as a list (bites, alternatives) yields its first entry."""
    if isinstance(plan, dict):
        return plan
    if isinstance(plan, list):
        return next((x for x in plan if isinstance(x, dict)), {})
    return {}


def infer_relations(model, subgoals, phases, roles, randomized, q_end=None):
    """What every released object rests on or in, for Stage 4's `rests_on` / `inside`.

    Sources, strongest first: the object's last place/insert plan entry of the manifest (`into` -> inside, `on` ->
    rests_on, or the expectation's `on`); the anchor of its release phase (the planner anchors a set-down and a
    let-go to the goal body); the contacts holding it up in the final state q_end. Only relations to another
    randomized object are returned (a body that never moves needs none). Returns {object: {kind: body, "source"}}."""
    out = {}
    for obj, r in roles.items():
        if not r.get("release"):
            continue
        rel = None
        for sg in subgoals or []:
            p = plan_entry(sg.get("plan"))
            ex = sg.get("expect") if isinstance(sg.get("expect"), dict) else {}
            if (p.get("object") or sg.get("object")) != obj:
                continue
            motion = str(p.get("motion") or sg.get("motion") or "")
            if motion not in ("place", "insert", "release"):
                continue
            if p.get("into"):
                rel = ("inside", p["into"], "plan")
            elif p.get("on"):
                rel = ("rests_on", p["on"], "plan")
            elif ex.get("on"):
                rel = ("rests_on", ex["on"], "plan")
        if rel is None:
            ph = next((p for p in phases if p.get("label") == r["release"]), None)
            a = (ph or {}).get("anchor")
            if a and a not in ("world", obj):
                rel = ("rests_on", a, "release anchor")
        support = kind = None
        if q_end is not None:
            support, kind = resting_support(model, q_end, obj, [o for o in randomized if o != obj])
        if rel is None and support:
            rel = (kind, support, "final contacts")
        if rel is None:
            continue
        body = root_body(model, rel[1])
        if body is None or body == obj or body not in randomized:
            continue
        k = rel[0]
        if k == "rests_on" and kind == "inside" and support == body:
            k = "inside"                              # the plan said on, the final state says in
        out[obj] = {k: body, "source": rel[2]}
    return out


def set_down_neighbours(model, scene_xml, Q, phases, labels, objs, randomized, clearance_m=0.01, visual_groups=(0, 1, 2), stride=2):
    """{object: {"next_to": body, "gap_m", "source"}} for carried objects put down right beside another randomized object.

    A part set down at a world target a few millimetres from another part (a brick slid next to a brick, a cup beside
    a plate) has a goal relative to that part, whatever its target coordinates say: when the neighbour is moved and the
    goal is not, the carrying hand (or the part) runs into it. For every carried, released object without a resting
    relation, the frames of its final carrier's set-down (the phases from its last carry or set-down naming the object
    through the release and the retreat after it) are checked: when the part or that hand (every visible geom, as
    Stage 4's penetration check sees them) comes within clearance_m of another randomized object that neither rides
    along nor rests on it, the closest such object is its neighbour. objs: the randomize.objects block (carrier,
    release; lists allowed)."""
    import mujoco
    checks = stage4_module("checks")
    from tool_library.common.episode import robot_parts
    cm = None
    out = {}
    for nm, o in objs.items():
        if any(o.get(k) for k in ("rests_on", "inside", "next_to")) or not o.get("carrier") or not o.get("release"):
            continue
        side = o["carrier"][-1] if isinstance(o["carrier"], list) else o["carrier"]
        rel = o["release"][-1] if isinstance(o["release"], list) else o["release"]
        r = next((i for i, p in enumerate(phases) if p.get("label") == rel and p.get("hand") == side), None)
        if r is None:
            continue
        lo = r
        while lo - 1 >= 0 and (phases[lo - 1].get("hand") != side or (names_object(phases[lo - 1], nm) and phase_verb(
                phases[lo - 1]["label"], side).startswith(("carry", "set down", "lower", "insert", "let go", "release", "relax")))):
            lo -= 1
        hi = r
        while hi + 1 < len(phases) and (phases[hi + 1].get("hand") != side or phase_verb(phases[hi + 1]["label"], side).startswith(
                ("retreat", "back off", "lift off", "withdraw"))):
            hi += 1
        labs = {p["label"] for p in phases[lo:hi + 1] if p.get("hand") == side}
        frames = [t for t, lab in enumerate(labels) if lab in labs][::max(1, int(stride))]
        if not frames:
            continue
        if cm is None:
            cm = checks.collision_model(scene_xml, model, list(visual_groups))[0]
            try:
                parts = robot_parts(cm)
            except ValueError:                        # a scene without the bench's hands: the parts alone are checked
                parts = {}
        hand_bodies = _subtree(cm, int(parts[side]["rig"].root_body)) if side in parts else set()
        mine = _subtree(cm, mujoco_body(cm, nm))
        riders = {k for k, oo in objs.items() if oo.get("attached_to") == nm or oo.get("rests_on") == nm or oo.get("inside") == nm
                  or (oo.get("release") == o.get("release") and oo.get("carrier") == o.get("carrier"))}     # it rides along
        geoms_a = [g for g in range(cm.ngeom) if int(cm.geom_bodyid[g]) in hand_bodies | mine and int(cm.geom_type[g]) != 0
                   and (cm.geom_contype[g] or cm.geom_conaffinity[g])]
        others = _collision_geoms(cm, [k for k in randomized if k != nm and k not in riders])
        data = mujoco.MjData(cm)
        best = (float(clearance_m), None)
        for t in frames:
            data.qpos[:] = Q[t]
            mujoco.mj_kinematics(cm, data)
            for k, gb in others.items():
                d = _gap(cm, data, geoms_a, gb, best[0])
                if d < best[0]:
                    best = (d, k)
        if best[1] is not None:
            out[nm] = {"next_to": best[1], "gap_m": round(best[0], 4), "source": f"set down within {1000 * clearance_m:.0f} mm of it"}
    return out


def physics_rules(scene_recon_json):
    """The run-time rules a physics replay needs (screw threads that let a cap go after their travel, scissor cuts,
    connector latches), from a Stage 2 manifest: the same rules the Stage 3 executor derives."""
    from pathlib import Path as _P
    man = _json.loads(_P(scene_recon_json).read_text())
    rules = []
    objs = [o for o in man.get("objects", []) if isinstance(o, dict)]
    for o in objs:
        eqs, js = o.get("equalities") or {}, o.get("joints") or {}
        if not isinstance(eqs, dict) or not isinstance(js, dict):
            continue
        eq, lj, pr = eqs.get("engaged"), js.get("lift"), o.get("params") or {}
        if eq and lj and pr.get("thread_pitch") and pr.get("thread_turns"):
            rules.append({"engaged": eq, "lift_joint": lj, "travel_m": float(pr["thread_pitch"]) * float(pr["thread_turns"]), "frac": 0.96})
        elif eq and lj and pr.get("release_travel_m"):
            rules.append({"engaged": eq, "lift_joint": lj, "travel_m": float(pr["release_travel_m"]), "frac": 0.96})
    joins = [dict(j) for o in objs for j in (o.get("cut_joins") or []) if isinstance(j, dict) and j.get("eq") and j.get("body")]
    if joins:
        from tool_library.common import scissors as SC
        for c in (SC.cutter_from_recon(o) for o in objs):
            if c:
                rules.append({"kind": "cut", "cutter": c, "joins": joins, "gates": "cord"})
    from tool_library.common.episode import latch_rules_from_objects
    rules += latch_rules_from_objects(objs)
    return rules


def _collision_geoms(model, objects):
    """{object: [collidable geom ids of its subtree]} (planes excluded)."""
    import mujoco
    out = {}
    for o in objects:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o)
        if b < 0:
            continue
        sub = _subtree(model, b)
        out[o] = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub and int(model.geom_type[g]) != 0
                  and (model.geom_contype[g] or model.geom_conaffinity[g])]
    return out


def _gap(model, data, ga, gb, distmax):
    """Smallest surface distance between two geom sets (m), at most distmax; bounding spheres filter the pairs.
    (Contact margins are no substitute: MuJoCo's box-box collider reports nothing for separated boxes.)"""
    import mujoco
    best = float(distmax)
    ft = np.zeros(6)
    for a in ga:
        pa, ra = data.geom_xpos[a], float(model.geom_rbound[a])
        for b in gb:
            if np.linalg.norm(data.geom_xpos[b] - pa) - ra - float(model.geom_rbound[b]) >= best:
                continue
            d = float(mujoco.mj_geomDistance(model, data, a, b, best, ft))
            best = min(best, max(d, 0.0))
    return best


def interaction_groups(model, Q, objects, release_frames=None, clearance_m=0.01, stride=3):
    """Objects that must keep their relative placement: pairs whose geometry comes within clearance_m of each other
    while both still sit where they started (before either is carried away: release_frames gives that frame per
    object, see leave_start_frames), such as two bricks touching in the starting pile or a brick lifted past its
    neighbour. Moved independently, such a pair collides in most new
    layouts; moved together it keeps the clearance the trajectory proved. Returns a list of groups (lists of object
    names, each with two or more members) and {pair: (closest m, frame)}."""
    import mujoco
    geoms = _collision_geoms(model, objects)
    names = list(geoms)
    T = len(Q)
    until = {o: T for o in names}
    for o, t_rel in (release_frames or {}).items():
        if o in until and t_rel is not None:
            until[o] = int(t_rel)
    data = mujoco.MjData(model)
    close = {}
    frames = sorted(set(range(0, T, max(1, int(stride)))) | {0})
    for t in frames:
        live = [o for o in names if t < until[o]]
        if len(live) < 2:
            continue
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                d = _gap(model, data, geoms[a], geoms[b], clearance_m)
                if d < clearance_m and d < close.get((a, b), (np.inf, 0))[0]:
                    close[(a, b)] = (d, t)
    parent = {o: o for o in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in close:
        parent[find(a)] = find(b)
    groups = {}
    for o in names:
        groups.setdefault(find(o), []).append(o)
    return [g for g in groups.values() if len(g) > 1], close


def body_span(model, q0, root):
    """Largest distance between two body origins of a root's subtree in the state q0 (a cable's length on the table)."""
    import mujoco
    b = mujoco_body(model, root)
    if b is None:
        return 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_kinematics(model, data)
    P = data.xpos[sorted(_subtree(model, b))]
    return float(np.max(np.linalg.norm(P[:, None] - P[None], axis=2))) if len(P) > 1 else 0.0


def plan_radius(model, q0, root):
    """How far the object's geometry reaches from its root body's origin in the table plane (m), in the state q0: a
    yaw of the object about its origin by a radians moves its farthest point by about radius x a."""
    import mujoco
    b = mujoco_body(model, root)
    if b is None:
        return 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_kinematics(model, data)
    lo, hi = _world_aabb(model, data, _subtree(model, b))
    if not np.all(np.isfinite(lo)):
        return 0.0
    c = data.xpos[b][:2]
    corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    return float(np.max(np.linalg.norm(corners - c, axis=1)))


def origin_distance(model, q0, a, b):
    """Distance in the table plane between two bodies' origins in the state q0 (0 for a missing body)."""
    import mujoco
    ia, ib = mujoco_body(model, a), mujoco_body(model, b)
    if ia is None or ib is None:
        return 0.0
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_kinematics(model, data)
    return float(np.linalg.norm(data.xpos[ia][:2] - data.xpos[ib][:2]))


def copy_model(model):
    import copy as _copy
    return _copy.deepcopy(model)


def start_clearances(model, q0, objects, groups=(), max_m=0.06):
    """{object: distance (m) to the nearest other object outside its group in the state q0}, capped at max_m."""
    import mujoco
    gid = {o: i for i, g in enumerate(groups) for o in g}
    geoms = _collision_geoms(model, objects)
    data = mujoco.MjData(model)
    data.qpos[:] = q0
    mujoco.mj_kinematics(model, data)
    out = {o: float(max_m) for o in geoms}
    names = list(geoms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in gid and gid.get(a) == gid.get(b):
                continue
            d = _gap(model, data, geoms[a], geoms[b], max_m)
            out[a], out[b] = min(out[a], d), min(out[b], d)
    return out


def default_ranges(clearance_m, max_shift_m=0.03, min_shift_m=0.005, max_yaw_deg=15.0, min_yaw_deg=3.0):
    """Randomization ranges an object can take without meeting its neighbours at the start: half its clearance
    (the neighbour may move toward it by as much), between min_shift_m and max_shift_m; the yaw range scales with it."""
    r = float(np.clip(0.5 * clearance_m, min_shift_m, max_shift_m))
    yaw = float(np.clip(max_yaw_deg * r / max_shift_m, min_yaw_deg, max_yaw_deg))
    return {"dx": [-round(r, 4), round(r, 4)], "dy": [-round(r, 4), round(r, 4)], "dyaw_deg": [-round(yaw, 1), round(yaw, 1)]}


# ----------------------------------------------------------------------------- Stage 4 draws, as the gates run them
def load_task(task_dir):
    """(cfg, Episode) of a Stage 4 task folder, with Stage 4's own loader. Raises AssertionError / ValueError with
    Stage 4's message when the config is not usable."""
    from pathlib import Path as _P
    cfg_p = _P(task_dir) / "stage4.json"
    if not cfg_p.exists():
        raise FileNotFoundError(f"{cfg_p} is missing")
    cfg = stage4_module("config").load_config(cfg_p)
    return cfg, stage4_module("randomize").Episode(cfg)


def rejection_detail(rep, ep):
    """One line per failed kinematic check of an adapt() report: what failed, where."""
    out = {}
    pre = rep.get("prescreen")
    if isinstance(pre, dict) and not pre.get("ok", True):
        side, r = max(((s, r) for s, r in pre.items() if isinstance(r, dict)), key=lambda x: x[1]["pos_m"] + 0.01 * x[1]["rot_deg"])
        fr = r.get("frame")
        out["reach"] = (f"the {side} hand's anchored keyframes miss by {1000 * r['pos_m']:.1f} mm / {r['rot_deg']:.1f} deg at frame "
                        f"{fr} ('{ep.labels[fr] if fr is not None else ''}'), before the full adaptation")
    if "ik" in rep and not rep["ik"]["ok"]:
        side, r = max(((s, r) for s, r in rep["ik"].items() if isinstance(r, dict)), key=lambda x: x[1]["pos_m"] + 0.01 * x[1]["rot_deg"])
        out["ik"] = (f"the {side} hand misses its re-anchored pose by {1000 * r['pos_m']:.1f} mm / {r['rot_deg']:.1f} deg at frame "
                     f"{r['frame']} ('{ep.labels[r['frame']]}')")
    if "joint_speed" in rep and not rep["joint_speed"]["ok"]:
        name, r = max(((k, v) for k, v in rep["joint_speed"].items() if isinstance(v, dict)), key=lambda x: x[1]["max"] - x[1]["limit"])
        out["joint_speed"] = f"{name} moves {r['max']} rad/s, limit {r['limit']} (original {r['original']})" + (
            " even after retiming" if rep.get("retime") else "")
    if "object_steps" in rep and not rep["object_steps"]["ok"]:
        name, v = max(((k, v) for k, v in rep["object_steps"].items() if k != "ok"), key=lambda x: x[1])
        out["object_steps"] = f"{name} jumps {1000 * v:.1f} mm more in one frame than in the original"
    if "penetration" in rep and not rep["penetration"]["ok"]:
        p = rep["penetration"]
        out["penetration"] = (f"{p.get('pair')} {1000 * p['new_penetration_m']:.1f} mm deeper than in the original at frame "
                              f"{p.get('frame')} ('{ep.labels[p['frame']] if p.get('frame') is not None and p['frame'] < len(ep.labels) else ''}')")
    return out


HINTS = {
    "penetration": "narrow the randomize ranges of the objects in that pair, or move objects that touch or pass close "
                   "to each other together (attached_to)",
    "object_steps": "the object jumps between its carrier and where it rests: declare what it is put on or in "
                    "(rests_on / inside) and anchor its carry and set-down to that body",
    "ik": "the arm cannot follow the moved object: narrow the ranges (or the yaw range) of the object it handles there",
    "reach": "the arm cannot reach the moved object's anchored keyframes: narrow the ranges (or the yaw range) of the "
             "object it handles there, or move the Stage 3 grasp into the middle of the arm's workspace",
    "joint_speed": "the hand has to cover the displacement in too short a blend even after retiming: narrow the ranges "
                   "or blend over a longer phase",
    "physics": "the layout passes the kinematic checks but not the physics replay: narrow the ranges of the objects named "
               "in the failed checks, or check the grasp at the new pose; with the Stage 3 plan handed off "
               "(write_stage4_config writes plan.json) Stage 4 re-executes such layouts closed-loop",
    "reexecute": "the Stage 3 plan, re-executed on the layout, failed too: see the row's reexecute.why",
}


def run_draws(task_dir, draws=10, seed=None, physics_replay=False, stop_after=None, log=None, reexecute=False):
    """Draw layouts of a task folder exactly as Stage 4's randomize step does (same sampler with its adaptation, the
    same seeds, reach pre-screen, retiming and checks; the physics replay and the re-execution fallback when asked)
    without writing samples. Returns {accepted, draws, rows, rejections {check: count}, examples {check: first
    detail}, methods, seconds_per_draw, physics_seconds, relations, warnings, physics_original}."""
    import tempfile
    import time as _t
    from collections import Counter
    cfg, ep = load_task(task_dir)
    if seed is not None:
        cfg["seed"] = int(seed)
    quiet = log or (lambda *_: None)
    work = Path(tempfile.mkdtemp(prefix="stage4_draws_")) if reexecute else None
    rz = stage4_module("randomize").Randomizer(ep, cfg, log=quiet, physics_replay=bool(physics_replay),
                                                reexecute=bool(reexecute), workdir=work)
    rows, n_ok, t_kin, t_phys = [], 0, 0.0, 0.0
    try:
        for attempt in range(int(draws)):
            res = rz.attempt(attempt)
            r0 = res["row"]
            t_kin += float(r0.get("kinematic_seconds") or 0.0)
            t_phys += float(r0.get("physics_seconds") or 0.0)
            rep = res["warp"]["report"] if res.get("method") == "reexecuted" else res["report"]
            row = {"attempt": attempt, "ok": bool(res["ok"]), "failed": list(r0["failed"]), "layout": r0["layout"],
                   "method": r0.get("method"), "detail": rejection_detail(rep, ep)}
            for k in ("physics_seconds", "retime", "sampling", "reexecute", "warp_failed", "blamed"):
                if r0.get(k) is not None:
                    row[k] = r0[k]
            if r0.get("physics_failed"):
                row["detail"]["physics"] = "; ".join(r0["physics_failed"][:4])
            if r0.get("reexecute") and not res["ok"]:
                row["detail"]["reexecute"] = str(r0["reexecute"].get("why"))
            n_ok += int(row["ok"])
            rows.append(row)
            if stop_after is not None and n_ok >= int(stop_after):
                break
    finally:
        if work is not None:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
    cnt = Counter(f for r in rows for f in r["failed"])
    examples = {}
    for r in rows:
        for k, v in r["detail"].items():
            examples.setdefault(k, f"draw {r['attempt']}: {v}")
    rel = {n: f"{o['relation'] or 'rests with'} {o['rest']}" for n, o in ep.objects.items() if o["rests_from"] is not None}
    return {"accepted": n_ok, "draws": len(rows), "rows": rows, "rejections": dict(cnt), "examples": examples,
            "methods": dict(Counter(r["method"] for r in rows if r["ok"])),
            "seconds_per_draw": round(t_kin / max(1, len(rows)), 2),
            "physics_seconds": round(t_phys, 1), "relations": rel, "warnings": list(ep.warnings),
            "physics_original": None if rz.phys is None else rz.phys.baseline,
            "reexecution": rz.reexec is not None, "reexecution_note": rz.reexec_note}
