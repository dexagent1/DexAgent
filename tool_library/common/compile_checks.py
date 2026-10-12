"""Helpers shared by the scene compile checks: compiling with the error kept, the MJCF file references, the
robot / object partition of a bench scene, keyframe states, a stable model fingerprint (also runnable in another
interpreter), MuJoCo's measured margin/gap semantics, position servos, mesh arrays and warning capture.

Nothing here assumes a particular robot or object: the robot is every body whose name starts with
`robot_prefix` (the bench convention, setup/setup.yaml `robot.body_prefix`), and objects are given by a
Stage 2 scene_recon.json (root body + bodies per object) or, without one, are the scene's top-level bodies
that are not the robot, the table, the world or a geom-less mocap rig.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROBOT_PREFIX = "robot_"
SUPPORT_BODIES = ("world", "table")
WELD = int(mujoco.mjtEq.mjEQ_WELD)
CONNECT = int(mujoco.mjtEq.mjEQ_CONNECT)
FREE = int(mujoco.mjtJoint.mjJNT_FREE)
BALL = int(mujoco.mjtJoint.mjJNT_BALL)
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
PLANE = int(mujoco.mjtGeom.mjGEOM_PLANE)


# ----------------------------------------------------------------------------- loading
def scene_path(scene):
    """The scene's file path when it is one (not an XML string or a compiled model), else None."""
    if isinstance(scene, mujoco.MjModel):
        return None
    s = str(scene)
    if s.lstrip().startswith("<"):
        return None
    return Path(s)


def compile_scene(scene):
    """(model, error): compile an MJCF path / XML string / MjModel; the error text instead of an exception."""
    if isinstance(scene, mujoco.MjModel):
        return scene, None
    p = scene_path(scene)
    if p is not None and not p.exists():
        return None, f"scene file not found: {p}"
    try:
        with capture_mj_warnings():
            if p is None:
                return mujoco.MjModel.from_xml_string(str(scene)), None
            if p.suffix.lower() == ".mjb":
                return mujoco.MjModel.from_binary_path(str(p)), None
            return mujoco.MjModel.from_xml_path(str(p)), None
    except Exception as exc:       # MuJoCo raises ValueError with the parser / compiler message
        return None, f"{type(exc).__name__}: {exc}"


def require_scene(scene, name="scene"):
    """Misuse guard: a scene argument must be a path, an XML string or a compiled model."""
    if scene is None or not isinstance(scene, (str, os.PathLike, mujoco.MjModel)):
        raise ValueError(f"{name} must be an MJCF path, an XML string or a mujoco.MjModel, got {type(scene).__name__}")


def check_params(given, allowed, tool):
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise ValueError(f"{tool}: unknown parameter(s) {unknown}; valid: {sorted(allowed)}")


def name_of(model, objtype, i, kind):
    nm = mujoco.mj_id2name(model, objtype, int(i))
    return nm if nm else f"{kind}{int(i)}"


def body_name(model, b):
    return name_of(model, mujoco.mjtObj.mjOBJ_BODY, b, "body")


def geom_name(model, g):
    return name_of(model, mujoco.mjtObj.mjOBJ_GEOM, g, "geom")


def joint_name(model, j):
    return name_of(model, mujoco.mjtObj.mjOBJ_JOINT, j, "joint")


def actuator_name(model, a):
    return name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a, "actuator")


def eq_name(model, e):
    """An equality's name, or 'eq<id>' (the convention of a recorded eq_active's columns)."""
    nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, int(e))
    return nm if nm else f"eq{int(e)}"


# ----------------------------------------------------------------------------- warnings
@contextlib.contextmanager
def capture_mj_warnings(sink=None):
    """Route MuJoCo's text warnings into `sink` (a list) instead of stderr and MUJOCO_LOG.TXT in the cwd."""
    sink = [] if sink is None else sink
    try:
        prev = mujoco.get_mju_user_warning()
    except Exception:
        prev = None
    try:
        mujoco.set_mju_user_warning(lambda msg: sink.append(str(msg)))
        installed = True
    except Exception:
        installed = False
    try:
        yield sink
    finally:
        if installed:
            try:
                mujoco.set_mju_user_warning(prev)
            except Exception:
                pass


def warning_counts(data):
    """{warning name: count} for the MuJoCo warnings raised on `data` so far (non-zero ones)."""
    out = {}
    for nm, val in mujoco.mjtWarning.__members__.items():
        if nm == "mjNWARNING":
            continue
        n = int(data.warning[int(val)].number)
        if n:
            out[nm.replace("mjWARN_", "")] = n
    return out


# ----------------------------------------------------------------------------- structure
def top_body(model, b):
    """The ancestor of body b that is a child of the world (b itself when it is one; 0 for the world)."""
    b = int(b)
    while b > 0 and int(model.body_parentid[b]) != 0:
        b = int(model.body_parentid[b])
    return b


def subtree_ids(model, root):
    out = {int(root)}
    for b in range(int(root) + 1, model.nbody):
        p = int(model.body_parentid[b])
        if p in out:
            out.add(b)
    return out


def is_robot(name, robot_prefix=ROBOT_PREFIX):
    return bool(robot_prefix) and str(name).startswith(robot_prefix)


def robot_body_ids(model, robot_prefix=ROBOT_PREFIX):
    return {b for b in range(1, model.nbody) if is_robot(body_name(model, b), robot_prefix)}


def load_objects_arg(objects):
    """objects -> list of {"id", "root_body", "bodies", "type"} from a scene_recon.json path, its dict, or a list of
    ids / root body names / such dicts. None stays None. The recon's folder is returned too (for its stage-1 link)."""
    if objects is None:
        return None, None
    base = None
    if isinstance(objects, (str, os.PathLike)):
        p = Path(objects)
        if not p.exists():
            raise ValueError(f"objects: file not found: {p}")
        base = p.parent
        objects = json.loads(p.read_text())
    if isinstance(objects, dict):
        objects = objects.get("objects", [])
    if not isinstance(objects, (list, tuple)):
        raise ValueError("objects must be a scene_recon.json path or dict, or a list of object ids / dicts")
    out = []
    for o in objects:
        if isinstance(o, str):
            out.append({"id": o, "root_body": None, "bodies": [], "type": None, "parts": []})
        elif isinstance(o, dict) and ("id" in o or "root_body" in o):
            from tool_library.common.piece_groups import recon_parts
            out.append({"id": o.get("id") or o.get("root_body"), "root_body": o.get("root_body"),
                        "bodies": list(o.get("bodies") or []), "type": o.get("type"), "raw": o,
                        "parts": recon_parts(o)})            # a group object's separate pieces (piece_groups)
        else:
            raise ValueError(f"objects: entry {o!r} is neither an id nor a dict with 'id'/'root_body'")
    return out, base


def default_objects(model, robot_prefix=ROBOT_PREFIX):
    """Top-level bodies that are scene objects: not the robot, the table or the world, and carrying geometry or
    joints somewhere in their subtree (a geom-less mocap camera rig is not an object)."""
    out = []
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) != 0:
            continue
        nm = body_name(model, b)
        if is_robot(nm, robot_prefix) or nm in SUPPORT_BODIES:
            continue
        sub = subtree_ids(model, b)
        has_geom = any(int(model.geom_bodyid[g]) in sub for g in range(model.ngeom))
        has_jnt = any(int(model.jnt_bodyid[j]) in sub for j in range(model.njnt))
        if not (has_geom or has_jnt):
            continue
        out.append({"id": nm, "root_body": nm, "bodies": [body_name(model, x) for x in sorted(sub)], "type": None})
    return out


def object_groups(model, objects=None, robot_prefix=ROBOT_PREFIX, by_prefix=True):
    """{object id: {"root": body id or -1, "bodies": set of body ids, "type", "declared_bodies", "parts"}}.

    With a recon, an object's bodies are its listed bodies plus every body under its root, under each declared part of
    a group object (`parts`: its separate pieces, body ids in "parts") and (by_prefix) every body whose name starts with
    '<id>_' (constructed objects prefix their parts so); without one, each top-level subtree."""
    objs, _ = load_objects_arg(objects)
    if objs is None:
        objs = default_objects(model, robot_prefix)
    names = [body_name(model, b) for b in range(model.nbody)]
    out = {}
    for o in objs:
        oid = str(o["id"])
        root = -1
        for cand in [o.get("root_body"), oid, re.sub(r"[^a-z0-9]+", "_", oid.lower()).strip("_")]:
            if cand:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(cand))
                if bid >= 0:
                    root = bid
                    break
        bodies = set()
        if root >= 0:
            bodies |= subtree_ids(model, root)
        for bn in o.get("bodies") or []:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(bn))
            if bid >= 0:
                bodies |= subtree_ids(model, bid)
        parts = []
        for pn in o.get("parts") or []:
            pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(pn))
            if pid >= 0:
                parts.append(int(pid))
                bodies |= subtree_ids(model, pid)
        if by_prefix:
            pre = f"{oid}_"
            bodies |= {b for b in range(1, model.nbody) if names[b].startswith(pre)}
        out[oid] = {"root": int(root), "bodies": bodies, "type": o.get("type"),
                    "declared_bodies": list(o.get("bodies") or []), "raw": o.get("raw"), "parts": parts}
    return out


def collidable(model, g):
    return bool(model.geom_contype[g] or model.geom_conaffinity[g])


def can_collide(model, g1, g2):
    """MuJoCo's contype/conaffinity filter and its same-body / parent-child exclusion (explicit pairs aside)."""
    if not ((model.geom_contype[g1] & model.geom_conaffinity[g2]) or (model.geom_contype[g2] & model.geom_conaffinity[g1])):
        return False
    b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
    if b1 == b2:
        return False
    w1, w2 = int(model.body_weldid[b1]), int(model.body_weldid[b2])
    if w1 == w2:
        return False
    if not (int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)):
        p1, p2 = int(model.body_weldid[int(model.body_parentid[w1])]), int(model.body_weldid[int(model.body_parentid[w2])])
        if (p1 == w2 or p2 == w1) and w1 != 0 and w2 != 0:
            return False
    return True


# ----------------------------------------------------------------------------- keyframes
def key_data(model, key=0):
    """MjData at keyframe `key` (index or name), else at qpos0, kinematics done."""
    data = mujoco.MjData(model)
    if model.nkey > 0 and key is not None:
        k = key
        if isinstance(key, str):
            k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key)
            if k < 0:
                raise ValueError(f"keyframe '{key}' is not in the scene")
        if not 0 <= int(k) < model.nkey:
            raise ValueError(f"keyframe index {k} out of range (nkey={model.nkey})")
        mujoco.mj_resetDataKeyframe(model, data, int(k))
    mujoco.mj_forward(model, data)
    return data


def servo_actuators(model):
    """Position servos on a single joint: [{"act", "joint", "dof", "qadr", "kp", "kv"}]. A position servo is an
    affine-bias actuator whose bias stiffness cancels its gain (biasprm[1] = -gainprm[0]); kv = -biasprm[2]."""
    out = []
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) not in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)):
            continue
        if int(model.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_AFFINE):
            continue
        kp = float(model.actuator_gainprm[a][0]) * float(model.actuator_gear[a][0]) ** 2
        if kp <= 0 or abs(float(model.actuator_biasprm[a][1]) + float(model.actuator_gainprm[a][0])) > 1e-6 * max(1.0, kp):
            continue
        j = int(model.actuator_trnid[a][0])
        if int(model.jnt_type[j]) not in (HINGE, SLIDE):
            continue
        out.append({"act": a, "joint": j, "dof": int(model.jnt_dofadr[j]), "qadr": int(model.jnt_qposadr[j]),
                    "kp": kp, "kv": -float(model.actuator_biasprm[a][2]) * float(model.actuator_gear[a][0]) ** 2})
    return out


def adhesion_bodies(model):
    """Body ids driven by an adhesion-style actuator (body transmission)."""
    return {int(model.actuator_trnid[a][0]) for a in range(model.nu)
            if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_BODY)}


# ----------------------------------------------------------------------------- files
_FILE_ATTRS = {
    "mesh": ("file",), "texture": ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"),
    "hfield": ("file",), "skin": ("file",), "include": ("file",), "model": ("file",),
}


def mjcf_file_refs(path, _seen=None, _root_dir=None, _dirs=None):
    """Every file an MJCF references, following <include>: [{"tag", "name", "attr", "file", "resolved", "source"}].
    Resolution follows MuJoCo: an absolute path is used as is; a relative mesh / hfield / skin path is looked up in
    meshdir, a texture path in texturedir (assetdir sets both), each relative to the main model's folder when not
    absolute; <include> and <model> files are relative to the main model's folder."""
    path = Path(path)
    root_dir = Path(_root_dir) if _root_dir else path.parent
    dirs = dict(_dirs or {"mesh": None, "texture": None})
    seen = _seen if _seen is not None else set()
    out = []
    if str(path.resolve()) in seen:
        return out
    seen.add(str(path.resolve()))
    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        return [{"tag": "parse_error", "name": "", "attr": "", "file": str(path), "resolved": str(path), "source": str(path),
                 "error": str(exc)}]
    for comp in tree.getroot().iter("compiler"):
        for k in ("assetdir", "meshdir", "texturedir"):
            v = comp.get(k)
            if v is None:
                continue
            if k in ("assetdir", "meshdir"):
                dirs["mesh"] = v
            if k in ("assetdir", "texturedir"):
                dirs["texture"] = v
            out.append({"tag": "compiler", "name": k, "attr": k, "file": v,
                        "resolved": str(Path(v) if os.path.isabs(v) else root_dir / v), "source": str(path)})
    for el in tree.getroot().iter():
        tag = el.tag if isinstance(el.tag, str) else ""
        for attr in _FILE_ATTRS.get(tag, ()):
            f = el.get(attr)
            if not f:
                continue
            if os.path.isabs(f):
                res = Path(f)
            elif tag in ("include", "model"):
                res = root_dir / f
            else:
                d = dirs["texture"] if tag == "texture" else dirs["mesh"]
                base = root_dir if d is None else (Path(d) if os.path.isabs(d) else root_dir / d)
                res = base / f
            out.append({"tag": tag, "name": el.get("name") or "", "attr": attr, "file": f, "resolved": str(res), "source": str(path)})
            if tag == "include" and res.exists():
                out += mjcf_file_refs(res, seen, root_dir, dirs)
    return out


# ----------------------------------------------------------------------------- fingerprint
# Run verbatim in-process (exec) and in other interpreters (python -c), so both sides compute the same digest.
SUMMARY_SCRIPT = r'''
import hashlib, json, sys
import numpy as np
import mujoco

SIZES = ("nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite", "ncam", "nlight", "nmesh", "nmeshvert", "nmeshface",
         "ntex", "ntexdata", "nmat", "npair", "nexclude", "neq", "ntendon", "nsensor", "nkey", "nmocap", "nflex", "nhfield")
ARRAYS = ("body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
          "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_range", "jnt_limited", "qpos0", "dof_armature", "dof_damping",
          "dof_frictionloss", "geom_type", "geom_bodyid", "geom_size", "geom_pos", "geom_quat", "geom_contype",
          "geom_conaffinity", "geom_condim", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
          "geom_rgba", "mesh_vert", "mesh_face", "tex_width", "tex_height", "actuator_trntype", "actuator_trnid",
          "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "eq_type", "eq_obj1id",
          "eq_obj2id", "eq_data", "eq_active0", "eq_solref", "pair_geom1", "pair_geom2", "pair_margin", "pair_gap",
          "key_qpos", "key_ctrl", "key_mpos", "key_mquat")
SMALL = ("body_mass", "body_inertia", "body_pos", "jnt_range", "qpos0", "dof_armature", "dof_damping", "geom_size",
         "geom_margin", "geom_gap", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "eq_data", "key_qpos")
OPT = ("timestep", "impratio", "integrator", "cone", "jacobian", "solver", "iterations", "noslip_iterations")


def _digest(a):
    a = np.asarray(a)
    if a.dtype.kind == "f":
        a = np.round(a.astype(np.float64), 9) + 0.0
    return hashlib.sha256(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


def summarize(path, keep_small=True):
    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        return {"ok": False, "version": mujoco.__version__, "error": f"{type(exc).__name__}: {exc}"}
    sizes = {k: int(getattr(m, k)) for k in SIZES if hasattr(m, k)}
    arrays = {k: _digest(getattr(m, k)) for k in ARRAYS if hasattr(m, k)}
    opt = {k: float(getattr(m.opt, k)) for k in OPT if hasattr(m.opt, k)}
    small = {}
    if keep_small:
        for k in SMALL:
            if hasattr(m, k):
                small[k] = np.round(np.asarray(getattr(m, k), float), 9).ravel().tolist()
        small["geom_type"] = np.asarray(m.geom_type, float).tolist()
        # world bounds of every geom at qpos0, from the mesh vertices themselves for meshes: invariant to how a version
        # chooses the mesh frame, so only a real change of geometry or placement shows
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        wb = np.zeros((m.ngeom, 6))
        for g in range(m.ngeom):
            R = d.geom_xmat[g].reshape(3, 3)
            if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                k = int(m.geom_dataid[g])
                V = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]] @ R.T + d.geom_xpos[g]
                wb[g] = np.concatenate([V.min(0), V.max(0)])
            else:
                c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                ctr, ext = d.geom_xpos[g] + R @ c, np.abs(R) @ h
                wb[g] = np.concatenate([ctr - ext, ctr + ext])
        small["geom_world_bounds"] = np.round(wb, 9).ravel().tolist()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]
    whole = hashlib.sha256(json.dumps([sizes, arrays, opt], sort_keys=True).encode()).hexdigest()[:20]
    return {"ok": True, "version": mujoco.__version__, "sizes": sizes, "arrays": arrays, "opt": opt, "small": small,
            "body_names": names, "narena": int(getattr(m, "narena", 0)), "hash": whole}


if __name__ == "__main__":
    print("SUMMARY_JSON " + json.dumps(summarize(sys.argv[1])))
'''

_NS = {}


def summarize(path, keep_small=True):
    """The model fingerprint of an MJCF file in this interpreter (same code as compile_summary_in)."""
    if not _NS:
        exec(compile(SUMMARY_SCRIPT, "<compile_checks.summary>", "exec"), _NS)
    return _NS["summarize"](path, keep_small)


def compile_summary_in(python, scene, timeout_s=300.0, cwd=None):
    """Compile `scene` in another Python interpreter and return its fingerprint dict (or {"ok": False, "error"})."""
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    try:
        r = subprocess.run([str(python), "-c", SUMMARY_SCRIPT, str(Path(scene).resolve())], capture_output=True, text=True,
                           timeout=float(timeout_s), env=env, cwd=cwd)
    except FileNotFoundError:
        return {"ok": False, "error": f"interpreter not found: {python}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"compile in {python} timed out after {timeout_s} s"}
    for line in r.stdout.splitlines()[::-1]:
        if line.startswith("SUMMARY_JSON "):
            return json.loads(line[len("SUMMARY_JSON "):])
    tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
    return {"ok": False, "error": f"no summary from {python} (exit {r.returncode}): {' | '.join(tail)}"}


def diff_summaries(a, b, rtol=1e-6, atol=1e-9):
    """Differences between two fingerprints: [(field, detail)] for sizes, options and the small arrays."""
    out = []
    for k in sorted(set(a.get("sizes", {})) | set(b.get("sizes", {}))):
        va, vb = a.get("sizes", {}).get(k), b.get("sizes", {}).get(k)
        if va != vb:
            out.append((k, f"{va} vs {vb}"))
    for k in sorted(set(a.get("opt", {})) & set(b.get("opt", {}))):
        if not np.isclose(a["opt"][k], b["opt"][k], rtol=rtol, atol=atol):
            out.append((f"opt.{k}", f"{a['opt'][k]} vs {b['opt'][k]}"))
    for k in sorted(set(a.get("small", {})) & set(b.get("small", {}))):
        x, y = np.asarray(a["small"][k], float), np.asarray(b["small"][k], float)
        if x.shape != y.shape:
            out.append((k, f"length {x.size} vs {y.size}"))
            continue
        if x.size and not np.allclose(x, y, rtol=rtol, atol=atol):
            i = int(np.argmax(np.abs(x - y)))
            out.append((k, f"max difference {abs(x[i] - y[i]):.3g} at entry {i} ({x[i]:.6g} vs {y[i]:.6g})"))
    return out


# ----------------------------------------------------------------------------- contact semantics
_MARGIN_SEM = {}


def margin_semantics():
    """How this MuJoCo combines geom margins and what gap does, measured on two spheres (cached):
    {"combine": "sum"|"max", "gap_band": True when a contact is detected up to margin+gap but pushes only inside
    margin (MuJoCo 3.x), False when gap subtracts from the active distance (classic), "version"}."""
    if _MARGIN_SEM:
        return dict(_MARGIN_SEM)

    def probe(ma, ga, mb, gb, dist):
        xml = (f'<mujoco><worldbody><geom name="a" type="sphere" size="0.01" margin="{ma}" gap="{ga}"/>'
               f'<body pos="{0.02 + dist} 0 0"><freejoint/><geom name="b" type="sphere" size="0.01" mass="0.01" '
               f'margin="{mb}" gap="{gb}"/></body></worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return [(float(c.dist), float(c.includemargin)) for c in d.contact[:d.ncon]]

    c = probe(0.001, 0.0, 0.002, 0.0, 0.0025)
    combine = "sum" if c and abs(c[0][1] - 0.003) < 1e-6 else "max"
    margin_ab = 0.003 if combine == "sum" else 0.002
    c2 = probe(0.001, 0.001, 0.002, 0.001, margin_ab + 0.001)       # beyond margin, inside margin + gap
    gap_band = bool(c2) and abs(c2[0][1] - margin_ab) < 1e-6
    _MARGIN_SEM.update({"combine": combine, "gap_band": gap_band, "version": mujoco.__version__})
    return dict(_MARGIN_SEM)


def pair_distances(m1, g1, m2, g2, sem=None):
    """(active distance, detection distance) in metres for two geoms' margins and gaps under `sem`."""
    sem = sem or margin_semantics()
    margin = (m1 + m2) if sem["combine"] == "sum" else max(m1, m2)
    gap = (g1 + g2) if sem["combine"] == "sum" else max(g1, g2)
    if sem["gap_band"]:
        return margin, margin + gap
    return margin - gap, margin


# ----------------------------------------------------------------------------- meshes
def mesh_arrays(model, k):
    va, vn = int(model.mesh_vertadr[k]), int(model.mesh_vertnum[k])
    fa, fn = int(model.mesh_faceadr[k]), int(model.mesh_facenum[k])
    return np.asarray(model.mesh_vert[va:va + vn], float), np.asarray(model.mesh_face[fa:fa + fn], int)


def open_edges(F):
    """Edges not shared by exactly two faces (boundary or non-manifold): 0 for a closed surface."""
    if len(F) == 0:
        return 0
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    return int((cnt != 2).sum())


def mesh_volumes(V, F):
    """(signed volume, legacy-style unsigned volume) of a triangle mesh, m^3: the signed one is the true enclosed
    volume of a closed outward-wound mesh; the unsigned sum of |tetra| from the centroid over-counts an open or
    non-convex mesh the way MuJoCo's legacy inertia does."""
    if len(F) == 0 or len(V) < 4:
        return 0.0, 0.0
    c = V.mean(axis=0)
    a, b, cc = V[F[:, 0]] - c, V[F[:, 1]] - c, V[F[:, 2]] - c
    t = np.einsum("ij,ij->i", a, np.cross(b, cc)) / 6.0
    return float(t.sum()), float(np.abs(t).sum())


# ----------------------------------------------------------------------------- robot hold for simulation
def robot_kinematic_dofs(model, robot_prefix=ROBOT_PREFIX):
    """qpos/dof addresses of robot joints that no actuator drives (the arms, driven kinematically on the bench)."""
    driven = {int(model.actuator_trnid[a][0]) for a in range(model.nu)
              if int(model.actuator_trntype[a]) in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT))}
    qa, da = [], []
    for j in range(model.njnt):
        if j in driven or not is_robot(joint_name(model, j), robot_prefix):
            continue
        t = int(model.jnt_type[j])
        nq = {FREE: 7, BALL: 4}.get(t, 1)
        nv = {FREE: 6, BALL: 3}.get(t, 1)
        qa += list(range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + nq))
        da += list(range(int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + nv))
    return np.array(qa, int), np.array(da, int)


def jsonable(x):
    """Round floats and convert numpy types so a report is JSON-serializable."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if not np.isfinite(v) else round(v, 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, Path):
        return str(x)
    return x


def compile_failure(tool_issue, error, **extra):
    """The report of a check whose scene did not compile."""
    rep = {"ok": False, "compiled": False, "compile_error": error,
           "issues": [f"the scene does not compile ({error}); {tool_issue}"]}
    rep.update(extra)
    return rep


def default_pythons():
    """Interpreters named by DEXAGENT_PYTHONS (os.pathsep- or comma-separated), else just this one."""
    env = os.environ.get("DEXAGENT_PYTHONS", "")
    items = [x.strip() for x in re.split(r"[,%s]" % re.escape(os.pathsep), env) if x.strip()]
    return items or [sys.executable]
