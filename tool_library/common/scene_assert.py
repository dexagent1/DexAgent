"""Shared helpers for the Scene Assertions verifiers that judge a reconstructed scene (Stage 2) and guard the robot
and the physics against tampering (Stage 2 and Stage 3 finish).

"The robot as it should be" is a setup folder: `setup/scene.xml` + `setup/setup.yaml` of this repository by default,
or any folder laid out the same way. Robot bodies, joints, geoms and actuators are matched to the reference by exact
name, then by name suffix (an attach prefix in front), then by the multiset of name tokens without `robot`
(`arm_right_link1` == `robot_right_arm_link1`, the naming of the hand-built twins).

Objects are the bodies that are neither robot, table nor helper (an empty mocap camera rig); they are grouped by the
Stage 2 manifest (scene_recon.json) when one is given, else by top-level body.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

if sys.platform.startswith("linux"):  # headless offscreen rendering; MuJoCo rejects "egl" elsewhere
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

SETUP_DIR = Path(__file__).resolve().parents[2] / "setup"
ROBOT_PREFIX = "robot_"
HELPER_BODIES = ("world", "camrig")
GEOM = mujoco.mjtGeom
OBJ = mujoco.mjtObj
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)          # MuJoCo's geom rgba when none is given


# ----------------------------------------------------------------------------------------------------------- files
def load_json(path_or_dict):
    """A dict, or a JSON file path -> (dict, folder it came from or None)."""
    if path_or_dict is None:
        return None, None
    if isinstance(path_or_dict, dict):
        return path_or_dict, None
    p = Path(path_or_dict)
    if not p.exists():
        raise ValueError(f"{p} does not exist")
    return json.loads(p.read_text()), p.parent


def try_load_model(scene):
    """(model, None) or (None, error message) for a scene that does not compile."""
    from tool_library.common import scene as S
    try:
        return S.load_model(scene), None
    except Exception as e:                                   # noqa: BLE001 - a broken scene is a verdict, not a crash
        return None, f"{type(e).__name__}: {e}"


def scene_dir(scene):
    if isinstance(scene, (str, Path)) and not str(scene).lstrip().startswith("<"):
        return Path(scene).resolve().parent
    return None


def report(issues, **meas):
    """The verifier report: ok, issues, then the measurements."""
    out = {"ok": not issues, "issues": list(issues)}
    out.update(meas)
    return out


def r6(x, n=6):
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r6(v, n) for v in x]
    return round(float(x), n)


# ----------------------------------------------------------------------------------------------------------- setup
def setup_paths(setup=None):
    """A setup folder, its setup.yaml or its scene.xml -> (folder, yaml path or None, scene path)."""
    p = Path(setup) if setup is not None else SETUP_DIR
    if p.is_dir():
        d = p
        y = d / "setup.yaml"
        y = y if y.exists() else None
        sc = d / "scene.xml"
        if y is not None:
            try:
                import yaml
                sc = d / (yaml.safe_load(y.read_text()) or {}).get("scene", "scene.xml")
            except Exception:                                # noqa: BLE001
                pass
    elif p.suffix.lower() in (".yaml", ".yml"):
        d, y = p.parent, p
        import yaml
        sc = d / ((yaml.safe_load(p.read_text()) or {}).get("scene", "scene.xml"))
    else:
        d, sc = p.parent, p
        y = d / "setup.yaml"
        y = y if y.exists() else None
    if not Path(sc).exists():
        raise ValueError(f"setup: no reference scene at {sc}; give a setup folder with scene.xml (+ setup.yaml)")
    return d, y, Path(sc)


@lru_cache(maxsize=8)
def _setup_cached(scene_path: str, yaml_path: str):
    model = mujoco.MjModel.from_xml_path(scene_path)
    cfg = {}
    if yaml_path:
        import yaml
        cfg = yaml.safe_load(Path(yaml_path).read_text()) or {}
    return model, cfg


def load_setup(setup=None):
    """{'dir', 'yaml_path', 'scene_path', 'model' (reference MjModel), 'cfg' (setup.yaml dict, {} without one)}."""
    d, y, sc = setup_paths(setup)
    model, cfg = _setup_cached(str(Path(sc).resolve()), str(Path(y).resolve()) if y else "")
    return {"dir": d, "yaml_path": y, "scene_path": sc, "model": model, "cfg": cfg}


def ref_home_qpos(ref_model, cfg=None):
    """The reference home keyframe's qpos (keyframe named in setup.yaml home.keyframe, else 'home', else 0)."""
    name = ((cfg or {}).get("home") or {}).get("keyframe", "home")
    k = mujoco.mj_name2id(ref_model, OBJ.mjOBJ_KEY, name)
    if k < 0:
        k = 0 if ref_model.nkey else -1
    return None if k < 0 else np.array(ref_model.key_qpos[k], float)


# ----------------------------------------------------------------------------------------------------------- names
def tokens(name):
    return Counter(t for t in re.split(r"[^A-Za-z0-9]+", str(name)) if t and t.lower() != "robot")


def names_of(model, objtype, n):
    return [mujoco.mj_id2name(model, objtype, i) or "" for i in range(n)]


def match_names(ref_names, scene_names):
    """{ref name: scene name or None}: exact, then a unique scene name ending in the ref name, then a unique scene
    name with the same tokens (ignoring 'robot')."""
    sset = set(n for n in scene_names if n)
    used = set()
    out = {}
    tok_index = {}
    for n in sset:
        tok_index.setdefault(frozenset(tokens(n).items()), []).append(n)
    for r in ref_names:
        if not r:
            continue
        if r in sset:
            out[r] = r; used.add(r); continue
        cands = [n for n in sset if n.endswith("_" + r) and n not in used]
        if len(cands) == 1:
            out[r] = cands[0]; used.add(cands[0]); continue
        cands = [n for n in tok_index.get(frozenset(tokens(r).items()), []) if n not in used]
        out[r] = cands[0] if len(cands) == 1 else None
        if out[r]:
            used.add(out[r])
    return out


def robot_ref_ids(ref, objtype, n):
    """Ids of the reference entities whose name (or, for geoms, whose body's name) starts with robot_."""
    if objtype == OBJ.mjOBJ_GEOM:
        return [g for g in range(ref.ngeom) if ref.body(int(ref.geom_bodyid[g])).name.startswith(ROBOT_PREFIX)]
    return [i for i in range(n) if (mujoco.mj_id2name(ref, objtype, i) or "").startswith(ROBOT_PREFIX)]


def robot_map(model, ref, objtype):
    """{ref id: scene id or -1} for the reference robot bodies / joints / actuators."""
    n_ref = {OBJ.mjOBJ_BODY: ref.nbody, OBJ.mjOBJ_JOINT: ref.njnt, OBJ.mjOBJ_ACTUATOR: ref.nu}[objtype]
    n_sc = {OBJ.mjOBJ_BODY: model.nbody, OBJ.mjOBJ_JOINT: model.njnt, OBJ.mjOBJ_ACTUATOR: model.nu}[objtype]
    rids = robot_ref_ids(ref, objtype, n_ref)
    rnames = [mujoco.mj_id2name(ref, objtype, i) for i in rids]
    m = match_names(rnames, names_of(model, objtype, n_sc))
    out = {}
    for i, nm in zip(rids, rnames):
        s = m.get(nm)
        out[i] = mujoco.mj_name2id(model, objtype, s) if s else -1
    return out


def robot_body_ids(model, ref=None):
    """Scene body ids that belong to the robot: named robot_*, matched to a reference robot body, or below one."""
    ids = {b for b in range(1, model.nbody) if model.body(b).name.startswith(ROBOT_PREFIX)}
    if ref is not None:
        ids |= {s for s in robot_map(model, ref, OBJ.mjOBJ_BODY).values() if s >= 0}
    changed = True
    while changed:
        changed = False
        for b in range(1, model.nbody):
            if b not in ids and int(model.body_parentid[b]) in ids:
                ids.add(b); changed = True
    return ids


def hand_body_ids(model, robot_ids=None):
    """{side: body ids of that hand} (bench naming robot_<side>_<side>_*, or any robot body with '_<side>_<side>_')."""
    out = {}
    for side in ("left", "right"):
        key = f"{side}_{side}_"
        out[side] = {b for b in range(1, model.nbody) if key in model.body(b).name
                     and (robot_ids is None or b in robot_ids)}
    return {s: v for s, v in out.items() if v}


def table_body_ids(model, cfg=None):
    names = {"table"}
    tb = ((cfg or {}).get("table") or {}).get("body")
    if tb:
        names.add(tb)
    return {b for b in range(1, model.nbody) if model.body(b).name in names}


def subtree(model, bid):
    out, stack = set(), [int(bid)]
    while stack:
        b = stack.pop()
        out.add(b)
        stack += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
    return out


def body_geoms(model, bids):
    bids = set(bids)
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bids]


def _helper(model, b):
    """An empty mocap or geometry-free, joint-free top-level body (a camera rig, a frame)."""
    if model.body(b).name in HELPER_BODIES:
        return True
    sub = subtree(model, b)
    has_geom = any(int(model.geom_bodyid[g]) in sub for g in range(model.ngeom))
    has_joint = any(int(model.jnt_bodyid[j]) in sub for j in range(model.njnt))
    return not has_geom and not has_joint


def object_groups(model, recon=None, objects=None, ref=None, cfg=None):
    """[{'id', 'root' (body id), 'bodies' (set of body ids), 'type', 'role'?}] for the scene's objects.

    With a scene_recon manifest (dict or path) its objects are used (root_body + bodies, ids as in the manifest);
    else every top-level body that is not robot, table or helper is one object with its subtree. `objects` keeps
    only the named ones (manifest ids or body names)."""
    from tool_library.common import scene as S
    from tool_library.common import piece_groups as PG
    robot = robot_body_ids(model, ref)
    table = table_body_ids(model, cfg)
    groups = []
    rec, _ = load_json(recon) if recon is not None else (None, None)
    if rec is not None:
        for o in rec.get("objects", []):
            rb = o.get("root_body")
            bid = S.resolve_body(model, rb) if rb else -1
            if bid < 0 and is_isaac_only(o):
                continue                                     # simulated in Isaac Sim only (a soft rope): not a MuJoCo body
            if bid < 0:
                groups.append({"id": o.get("id", rb), "root": -1, "bodies": set(), "type": o.get("type"), "missing": rb})
                continue
            bodies = set()
            for n in o.get("bodies") or []:
                b = S.resolve_body(model, n)
                if b >= 0:
                    bodies.add(b)
            bodies |= subtree(model, bid)
            # a group object (piece_groups): each declared piece with everything under it
            parts = [b for b in (S.resolve_body(model, n) for n in PG.recon_parts(o)) if b >= 0]
            for b in parts:
                bodies |= subtree(model, b)
            groups.append({"id": o.get("id", rb), "root": bid, "bodies": bodies, "type": o.get("type"), "parts": parts})
    else:
        for b in range(1, model.nbody):
            if int(model.body_parentid[b]) != 0 or b in robot or b in table or _helper(model, b):
                continue
            groups.append({"id": model.body(b).name, "root": b, "bodies": subtree(model, b), "type": None})
    if objects is not None:
        want = {str(x) for x in objects}
        groups = [g for g in groups if str(g["id"]) in want or (g["root"] >= 0 and model.body(g["root"]).name in want)]
        found = {str(g["id"]) for g in groups} | {model.body(g["root"]).name for g in groups if g["root"] >= 0}
        for w in sorted(want - found):
            bid = S.resolve_body(model, w)
            if bid >= 0:
                groups.append({"id": w, "root": bid, "bodies": subtree(model, bid), "type": None})
            else:
                groups.append({"id": w, "root": -1, "bodies": set(), "type": None, "missing": w})
    return groups


def is_isaac_only(o):
    """A manifest object that build_isaac_scene simulates in Isaac Sim instead of MuJoCo."""
    return str(o.get("skill", "")) == "build_isaac_scene" or str(o.get("simulator", "")).lower() in ("isaac", "isaac sim", "isaacsim")


def isaac_only_ids(recon):
    rec, _ = load_json(recon) if recon is not None else (None, None)
    return {str(o.get("id")) for o in (rec or {}).get("objects", []) if is_isaac_only(o)}


def isaac_cord_spec(o, base):
    """The isaac_cord specification (with its settled link poses) of an Isaac-only deformable entry, or None when the
    entry names no loadable isaac.cord_json."""
    from tool_library.common import isaac_cord as IC
    cj = (o.get("isaac") or {}).get("cord_json")
    if not cj:
        return None
    p = Path(cj) if Path(cj).is_absolute() or base is None else Path(base) / cj
    try:
        return IC.load_spec(p)
    except (OSError, ValueError, KeyError):
        return None


def cord_tube_mask(cam, centres, radius, segment_m=None, size=None, quats=None):
    """Silhouette [H, W] of a cord drawn through a camera.json camera (OpenCV axes, rig frame) as a tube of `radius`
    along its link centres, extended to the cord's two ends (half a link along the end links' axes when `quats` are
    given, else along the end tangents); each piece is as thick as the cord's projected diameter at its depth.
    `size` (width, height) rescales the camera to a mask's resolution."""
    import cv2
    from tool_library.common import rigcam as RC
    W, H = (int(size[0]), int(size[1])) if size is not None else (int(cam["width"]), int(cam["height"]))
    if (W, H) != (int(cam["width"]), int(cam["height"])):
        cam = RC.scaled(cam, W, H)
    C = np.asarray(centres, float)
    h = 0.5 * float(segment_m) if segment_m else 0.5 * float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    if quats is not None:
        from tool_library.common import isaac_cord as IC
        x0, x1 = IC.quat_to_mat(quats[0])[:, 0], IC.quat_to_mat(quats[-1])[:, 0]
    else:
        x0, x1 = C[1] - C[0], C[-1] - C[-2]
        x0, x1 = x0 / max(np.linalg.norm(x0), 1e-12), x1 / max(np.linalg.norm(x1), 1e-12)
    P = np.vstack([C[0] - h * x0, C, C[-1] + h * x1])
    uv, z = RC.project(cam, P)
    img = np.zeros((H, W), np.uint8)
    for i in range(len(P) - 1):
        if z[i] <= 1e-3 or z[i + 1] <= 1e-3:
            continue
        w = max(1, int(round(2.0 * float(radius) * float(cam["fx"]) / float(0.5 * (z[i] + z[i + 1])))))
        a, b = tuple(int(v) for v in np.round(uv[i])), tuple(int(v) for v in np.round(uv[i + 1]))
        cv2.line(img, a, b, 255, w)
        cv2.circle(img, b, max(w // 2, 1), 255, -1)
    return img > 0


def semantic_objects(semantic):
    sem, _ = load_json(semantic)
    return [] if sem is None else list(sem.get("objects", []))


def stage1_groups(model, semantic, recon=None, ref=None, cfg=None):
    """{stage1 object id: object group or None}: the manifest's object with the same id, else a body that resolves
    from the id (slug, '_<id>' suffix)."""
    from tool_library.common import scene as S
    groups = {str(g["id"]): g for g in object_groups(model, recon, ref=ref, cfg=cfg)} if recon is not None else {}
    isaac = isaac_only_ids(recon)
    out = {}
    for o in semantic_objects(semantic):
        oid = str(o.get("id"))
        if oid in isaac and oid not in groups:
            continue                                         # lives in the Isaac Sim scene, not in this MJCF
        g = groups.get(oid)
        if g is None:
            bid = S.resolve_body(model, oid)
            if bid < 0:
                cands = [b for b in range(1, model.nbody) if model.body(b).name.startswith(oid + "_")
                         and int(model.body_parentid[b]) == 0]
                bid = cands[0] if len(cands) == 1 else -1
            g = {"id": oid, "root": bid, "bodies": subtree(model, bid) if bid >= 0 else set(), "type": o.get("type")} if bid >= 0 else None
        out[oid] = g
    return out


# ----------------------------------------------------------------------------------------------------------- geoms
def geom_collidable(model, g):
    return bool(model.geom_contype[g] or model.geom_conaffinity[g])


def effective_rgba(model, g):
    """The geom's colour as renderers see it: its own rgba, or its material's when the geom keeps the default."""
    rgba = np.array(model.geom_rgba[g], float)
    mid = int(model.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, DEFAULT_RGBA):
        rgba = np.array(model.mat_rgba[mid], float)
    return rgba


def geom_visible(model, g, groups=(0, 1, 2)):
    return int(model.geom_group[g]) in groups and effective_rgba(model, g)[3] > 0.0 and int(model.geom_type[g]) != int(GEOM.mjGEOM_PLANE)


def mesh_vertices(model, g):
    mid = int(model.geom_dataid[g])
    a, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
    return np.array(model.mesh_vert[a:a + n], float)


def _sphere_dirs(n=42):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def geom_local_points(model, g, n=42):
    """Points on the geom's surface in its own frame (mesh: its vertices)."""
    t = int(model.geom_type[g]); s = np.array(model.geom_size[g], float)
    if t == int(GEOM.mjGEOM_MESH):
        return mesh_vertices(model, g)
    if t == int(GEOM.mjGEOM_BOX):
        c = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
        return c * s[:3]
    if t == int(GEOM.mjGEOM_SPHERE):
        return _sphere_dirs(n) * s[0]
    if t == int(GEOM.mjGEOM_ELLIPSOID):
        return _sphere_dirs(n) * s[:3]
    if t in (int(GEOM.mjGEOM_CAPSULE), int(GEOM.mjGEOM_CYLINDER)):
        r, h = s[0], s[1]
        ang = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        pts = [np.c_[ring, np.full(12, z)] for z in (-h, 0.0, h)]
        if t == int(GEOM.mjGEOM_CAPSULE):
            pts.append(np.array([[0, 0, h + r], [0, 0, -h - r]]))
        else:
            pts.append(np.array([[0, 0, h], [0, 0, -h]]))
        return np.vstack(pts)
    return np.zeros((0, 3))


def geom_world_points(model, data, g, n=42):
    P = geom_local_points(model, g, n)
    if len(P) == 0:
        return P
    return data.geom_xpos[g] + P @ data.geom_xmat[g].reshape(3, 3).T


_HULLS = {}


def mesh_hull(model, g):
    """(halfspace equations [k,4] of the convex hull in the geom frame (n.x + b <= 0 inside), hull volume)."""
    import hashlib
    from scipy.spatial import ConvexHull
    V = np.ascontiguousarray(mesh_vertices(model, g))
    key = hashlib.sha1(V.tobytes()).hexdigest()
    if key not in _HULLS:
        try:
            h = ConvexHull(V)
            _HULLS[key] = (h.equations.copy(), float(h.volume))
        except Exception:                                    # noqa: BLE001 - a degenerate (flat) mesh
            _HULLS[key] = (None, 0.0)
        if len(_HULLS) > 2048:
            _HULLS.pop(next(iter(_HULLS)))
    return _HULLS[key]


def geom_volume(model, g):
    """Volume of the solid MuJoCo collides (meshes: their convex hull), m^3."""
    if int(model.geom_type[g]) == int(GEOM.mjGEOM_MESH):
        return mesh_hull(model, g)[1]
    from tool_library.common import scene as S
    return float(S.geom_volume(model, g))


def signed_distance(model, data, g, P):
    """Signed distance (m, negative inside) of world points P to geom g. Exact for primitives; for a mesh the
    distance to its convex hull's supporting planes (exact inside, a lower bound outside)."""
    P = np.atleast_2d(np.asarray(P, float))
    R = data.geom_xmat[g].reshape(3, 3)
    q = (P - data.geom_xpos[g]) @ R
    t = int(model.geom_type[g]); s = np.array(model.geom_size[g], float)
    if t == int(GEOM.mjGEOM_SPHERE):
        return np.linalg.norm(q, axis=1) - s[0]
    if t == int(GEOM.mjGEOM_BOX):
        d = np.abs(q) - s[:3]
        out = np.linalg.norm(np.maximum(d, 0), axis=1)
        return out + np.minimum(d.max(axis=1), 0)
    if t == int(GEOM.mjGEOM_CAPSULE):
        z = np.clip(q[:, 2], -s[1], s[1])
        return np.linalg.norm(q - np.c_[np.zeros((len(q), 2)), z], axis=1) - s[0]
    if t == int(GEOM.mjGEOM_CYLINDER):
        dr = np.linalg.norm(q[:, :2], axis=1) - s[0]
        dz = np.abs(q[:, 2]) - s[1]
        d = np.c_[dr, dz]
        return np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
    if t == int(GEOM.mjGEOM_ELLIPSOID):
        k = np.linalg.norm(q / s[:3], axis=1)
        return (k - 1.0) * s[:3].min()
    if t == int(GEOM.mjGEOM_MESH):
        eq, _ = mesh_hull(model, g)
        if eq is None:
            return np.full(len(q), np.inf)
        return (q @ eq[:, :3].T + eq[:, 3]).max(axis=1)
    if t == int(GEOM.mjGEOM_PLANE):
        return q[:, 2]
    return np.full(len(q), np.inf)


# ----------------------------------------------------------------------------------------------------------- camera
def load_camera(camera, model=None, width=None, height=None):
    """A camera.json path / dict (rigcam convention), or the name of a world camera in `model` (needs width and
    height) -> rigcam camera dict."""
    from tool_library.common import rigcam as RC
    if isinstance(camera, dict):
        cam = dict(camera)
        cam["R_cv"] = np.asarray(cam["R_cv"], float).reshape(3, 3).tolist()
        return cam
    if isinstance(camera, (str, Path)) and str(camera).endswith(".json"):
        return RC.load(camera)
    if model is not None and isinstance(camera, str):
        cid = mujoco.mj_name2id(model, OBJ.mjOBJ_CAMERA, camera)
        if cid < 0:
            raise ValueError(f"camera '{camera}' is neither a camera.json nor a camera of the scene")
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        R = d.cam_xmat[cid].reshape(3, 3)
        from scipy.spatial.transform import Rotation as Rot
        q = Rot.from_matrix(R).as_quat()
        return RC.camera_from_mujoco(int(width or 640), int(height or 480), float(model.cam_fovy[cid]),
                                     d.cam_xpos[cid].tolist(), [q[3], q[0], q[1], q[2]])
    raise ValueError(f"camera: give a camera.json path, a camera dict or a scene camera name, got {camera!r}")


def scene_camera_pose(model, name):
    """(pos, R_mj 3x3, fovy) of a named scene camera at the model's reference configuration, or None."""
    cid = mujoco.mj_name2id(model, OBJ.mjOBJ_CAMERA, name)
    if cid < 0:
        return None
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    return np.array(d.cam_xpos[cid]), np.array(d.cam_xmat[cid]).reshape(3, 3), float(model.cam_fovy[cid])


def render_segmentation(model, data, cam, hide_bodies=(), max_side=None):
    """Geom-id image [H, W] (-1 = background) of the scene through the rigcam camera `cam` at its resolution (or
    scaled so its longer side is max_side). The first world camera of the model is retargeted for the render and
    restored; a model without one cannot be rendered this way (ValueError). Bodies in hide_bodies are not drawn."""
    from tool_library.common import rigcam as RC
    W, H = int(cam["width"]), int(cam["height"])
    if max_side and max(W, H) > max_side:
        s = max_side / float(max(W, H))
        cam = RC.scaled(cam, int(round(W * s)), int(round(H * s)))
        W, H = int(cam["width"]), int(cam["height"])
    cid = next((c for c in range(model.ncam) if int(model.cam_bodyid[c]) == 0), -1)
    if cid < 0:
        raise ValueError("the scene has no world camera to render through; add one (e.g. the setup's `ego`)")
    from scipy.spatial.transform import Rotation as Rot
    old = (model.cam_pos[cid].copy(), model.cam_quat[cid].copy(), float(model.cam_fovy[cid]), int(model.cam_mode[cid]),
           model.vis.global_.offwidth, model.vis.global_.offheight)
    hidden = []
    try:
        R_mj = np.asarray(cam["R_cv"], float) @ RC.CV_TO_MJ
        q = Rot.from_matrix(R_mj).as_quat()
        model.cam_pos[cid] = np.asarray(cam["pos"], float)
        model.cam_quat[cid] = [q[3], q[0], q[1], q[2]]
        model.cam_fovy[cid] = math.degrees(2 * math.atan(0.5 * H / float(cam["fy"])))
        model.cam_mode[cid] = int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), W)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), H)
        hide = set(hide_bodies)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) in hide:
                hidden.append((g, float(model.geom_rgba[g][3]), int(model.geom_group[g])))
                model.geom_group[g] = 5
        mujoco.mj_forward(model, data)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        from tool_library.common.render import segmentation_renderer
        with segmentation_renderer(model, H, W) as r:
            r.update_scene(data, camera=cid, scene_option=opt)
            seg = r.render().copy()
        ids = seg[..., 0].astype(int)
        types = seg[..., 1].astype(int)
        ids[types != int(OBJ.mjOBJ_GEOM)] = -1
        # MuJoCo's principal point is the image centre; shift when camera.json's is not
        du, dv = int(round(float(cam["cx"]) - W / 2.0)), int(round(float(cam["cy"]) - H / 2.0))
        if du or dv:
            # a shift, not np.roll: the uncovered border is no geom (-1), not ids wrapped from the far edge
            shifted = np.full_like(ids, -1)
            h, w = ids.shape
            if abs(dv) < h and abs(du) < w:
                shifted[max(dv, 0):h + min(dv, 0), max(du, 0):w + min(du, 0)] = \
                    ids[max(-dv, 0):h + min(-dv, 0), max(-du, 0):w + min(-du, 0)]
            ids = shifted
        return ids
    finally:
        model.cam_pos[cid], model.cam_quat[cid], model.cam_fovy[cid], model.cam_mode[cid] = old[0], old[1], old[2], old[3]
        model.vis.global_.offwidth, model.vis.global_.offheight = old[4], old[5]
        for g, _, grp in hidden:
            model.geom_group[g] = grp


def load_mask(mask, size=None):
    """PNG path / array -> bool [H, W], optionally resized (nearest) to (width, height)."""
    from tool_library.common import scene as S
    if isinstance(mask, (str, Path)) and not Path(mask).exists():
        raise ValueError(f"mask {mask} does not exist")
    return S.load_mask(mask, size)


def contact_measure(mask, cam, ignore=None, min_cols=3):
    """The measure_mask_footprint reading of one silhouette: per column the lowest mask pixel (where the object
    meets the table), back-projected onto z = 0. Returns {'width_m', 'center_xy', 'depth_m', 'apparent_height_m',
    'bottom_v', 'n_cols'} or None (empty mask, or a contact above the horizon)."""
    from tool_library.common import rigcam as RC
    m = np.asarray(mask, bool)
    if ignore is not None:
        ig = np.asarray(ignore, bool)
    cols = np.nonzero(m.any(axis=0))[0]
    if len(cols) < min_cols:
        return None
    H = m.shape[0]
    bottom = np.array([H - 1 - np.argmax(m[::-1, c]) for c in cols])
    top = np.array([np.argmax(m[:, c]) for c in cols])
    keep = np.ones(len(cols), bool)
    if ignore is not None:
        # a column whose bottom pixel touches a hand is occluded there: its contact is not the object's
        keep &= ~np.array([ig[min(b + 1, H - 1), c] or ig[b, c] for b, c in zip(bottom, cols)])
    if keep.sum() < min_cols:
        keep[:] = True
    uv = np.c_[cols[keep] + 0.5, bottom[keep] + 0.5]
    try:
        P = RC.pixel_on_plane(cam, uv, 0.0)
    except ValueError:
        return None
    # width across the view: the spread of the contact points along the image's horizontal on the table
    R = np.asarray(cam["R_cv"], float)
    right = R[:, 0].copy(); right[2] = 0.0
    right /= max(np.linalg.norm(right), 1e-9)
    s = P @ right
    centre = P.mean(axis=0)
    _, z = RC.project(cam, centre[None])
    depth = float(z[0])
    v_top = float(top.min()); v_bot = float(bottom[keep].max())
    return {"width_m": float(s.max() - s.min()), "center_xy": [float(centre[0]), float(centre[1])], "depth_m": depth,
            "apparent_height_m": float((v_bot - v_top) * depth / float(cam["fy"])), "bottom_v": v_bot, "n_cols": int(keep.sum())}


def horizon_v(cam, u):
    """Image row of the table plane's horizon (vanishing line of z = const) at column u; -inf when the camera looks
    straight down."""
    R = np.asarray(cam["R_cv"], float)
    # a pixel ray d = R @ [(u-cx)/fx, (v-cy)/fy, 1] is horizontal when its z component is 0
    a = R[2, 0] / cam["fx"]; b = R[2, 1] / cam["fy"]; c = R[2, 2]
    if abs(b) < 1e-12:
        return -math.inf
    return cam["cy"] - (a * (u - cam["cx"]) + c) / b


# ----------------------------------------------------------------------------------------------------------- physics
def robot_dofs(model, robot_ids):
    """(qpos addresses, dof addresses) of every joint on a robot body."""
    qa, da = [], []
    for j in range(model.njnt):
        if int(model.jnt_bodyid[j]) in robot_ids:
            t = int(model.jnt_type[j])
            nq = {0: 7, 1: 4}.get(t, 1); nv = {0: 6, 1: 3}.get(t, 1)
            qa += list(range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + nq))
            da += list(range(int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + nv))
    return np.array(qa, int), np.array(da, int)


def settle_robot_held(model, data, seconds, robot_ids):
    """Step physics for `seconds` with every robot joint clamped at its current value (the robot stands still at its
    home pose whatever its servos' keyframe targets say), returns the data. Raises FloatingPointError on divergence."""
    qa, da = robot_dofs(model, robot_ids)
    q_hold = data.qpos[qa].copy()
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            j = int(model.actuator_trnid[a][0])
            if int(model.jnt_bodyid[j]) in robot_ids and int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE):
                data.ctrl[a] = data.qpos[int(model.jnt_qposadr[j])]
    bad0 = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    t0 = float(data.time)
    for _ in range(int(round(seconds / model.opt.timestep))):
        t_prev = float(data.time)
        mujoco.mj_step(model, data)
        if int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > bad0:
            # MuJoCo resets the state to qpos0 on a bad acceleration and carries on: every later number would lie
            raise FloatingPointError(f"MuJoCo reset the simulation on a bad acceleration {t_prev - t0 + model.opt.timestep:.4f} s after the start")
        data.qpos[qa] = q_hold
        data.qvel[da] = 0.0
    if not np.isfinite(data.qpos).all():
        raise FloatingPointError("the simulation diverged")
    mujoco.mj_forward(model, data)
    return data


# ----------------------------------------------------------------------------------------------------------- compare
def differs(a, b, abs_tol=1e-6, rel_tol=1e-6):
    """True when arrays a and b differ beyond abs_tol + rel_tol * |b| anywhere."""
    a = np.atleast_1d(np.asarray(a, float)); b = np.atleast_1d(np.asarray(b, float))
    if a.shape != b.shape:
        return True
    return bool(np.any(np.abs(a - b) > abs_tol + rel_tol * np.abs(b)))


def quat_angle_deg(q1, q2):
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12); q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    return float(np.degrees(2 * np.arccos(np.clip(abs(float(q1 @ q2)), 0.0, 1.0))))


def geom_label(model, g):
    nm = model.geom(g).name
    return nm if nm else f"{model.body(int(model.geom_bodyid[g])).name}[geom {g - int(model.body_geomadr[int(model.geom_bodyid[g])])}]"


def is_hull_geom(model, g):
    from tool_library.common.handgeom import is_hull
    return is_hull(model.geom(g).name)


def mesh_signature(model, g):
    """(vertex count, AABB extents) of a mesh geom's mesh: equal meshes have equal signatures."""
    if int(model.geom_type[g]) != int(GEOM.mjGEOM_MESH):
        return None
    V = mesh_vertices(model, g)
    return int(len(V)), (V.max(axis=0) - V.min(axis=0)).round(6).tolist()


def station_root_names(cfg):
    out = {}
    for side, st in ((cfg or {}).get("stations") or {}).items():
        if isinstance(st, dict) and st.get("base_body"):
            out[side] = st["base_body"]
    return out


# ----------------------------------------------------------------------------------------------------------- ropes
ROPE_WORDS = re.compile(r"rope|cable|thread|string|cord|wire|chain|lace")


def declared_anchors(model, recon, robot=(), table=()):
    """{object id: {body id: reason}} for deformables whose scene_recon.json entry declares `anchored_to`
    ({"body": name, "reason": text} or a list of them): a cord end held under a fixed object (a thread under a tape
    dispenser). Only bodies of fixed scene objects count: never the world, the table, the robot or a mocap body."""
    rec, _ = load_json(recon) if recon is not None else (None, None)
    out = {}
    for o in (rec or {}).get("objects", []):
        decl = o.get("anchored_to")
        if not decl or o.get("type") != "deformable":
            continue
        for d in (decl if isinstance(decl, list) else [decl]):
            name = d.get("body") if isinstance(d, dict) else str(d)
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
            if b <= 0 or b in set(robot) or b in set(table) or int(model.body_mocapid[b]) >= 0:
                continue
            x, free = b, False
            while x > 0:
                if any(int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) for j in range(model.njnt) if int(model.jnt_bodyid[j]) == x):
                    free = True; break
                x = int(model.body_parentid[x])
            if not free:
                out.setdefault(str(o.get("id")), {})[b] = (d.get("reason") if isinstance(d, dict) else "") or ""
    return out


def rope_groups(model, recon=None, objects=None, min_links=5, ref=None, cfg=None):
    """Object groups that are ropes: a manifest deformable named like a rope, else a chain of >= min_links bodies."""
    rec, _ = load_json(recon) if recon is not None else (None, None)
    groups = object_groups(model, rec, objects, ref=ref, cfg=cfg)
    out = []
    info = {str(o.get("id")): o for o in (rec or {}).get("objects", [])}
    for g in groups:
        if g["root"] < 0:
            continue
        o = info.get(str(g["id"]), {})
        named = bool(ROPE_WORDS.search(" ".join([str(o.get("skill", "")), str(g["id"]), model.body(g["root"]).name]).lower()))
        links = chain_links(model, g["bodies"])
        if rec is not None and o:
            # a manifest deformable is a rope when named like one, when it declares a strand topology, or when its links
            # form long strands whatever its name (a "ribbon" of three strands); a pad or plush of a few jointed parts is not
            strands_like = len(links) >= min_links and is_strand_tree(model, links)
            if o.get("type") == "deformable" and (named or o.get("topology") or strands_like) and len(links) >= 2:
                out.append((g, links))
        elif len(links) >= min_links and is_chain(model, links):
            out.append((g, links))
    return out


def chain_links(model, bodies):
    """The object's bodies that carry collision (else any) geometry, in tree order."""
    bs = sorted(bodies)
    col = [b for b in bs if any(geom_collidable(model, g) for g in body_geoms(model, [b]))]
    return col or [b for b in bs if body_geoms(model, [b])]


def is_strand_tree(model, links):
    """The links hang together as one tree (each link's parent is a link, but one) with few free ends for their number:
    strands of a rope, cable or ribbon (ends <= max(2, n/5)), not a grid of pad cells or a star of parts."""
    s = set(links)
    if sum(1 for b in links if int(model.body_parentid[b]) not in s) != 1:
        return False
    kids = {b: 0 for b in links}
    for b in links:
        if int(model.body_parentid[b]) in s:
            kids[int(model.body_parentid[b])] += 1
    root = next(b for b in links if int(model.body_parentid[b]) not in s)
    ends = sum(1 for b in links if kids[b] == 0) + (1 if kids[root] == 1 else 0)
    return ends <= max(2, len(links) // 5)


def is_chain(model, links):
    s = set(links)
    kids = {b: [c for c in links if int(model.body_parentid[c]) == b] for b in links}
    return all(len(v) <= 1 for v in kids.values()) and sum(1 for b in links if int(model.body_parentid[b]) not in s) <= 2


def chain_order(links, adj):
    """Links in chain order starting from an end (any order for a loop or a tree)."""
    if not links:
        return []
    start = next((b for b in links if len(adj[b]) == 1), links[0])
    out, seen, stack = [], set(), [start]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x); out.append(x)
        stack += sorted(adj[x] - seen)
    out += [b for b in links if b not in seen]
    return out
