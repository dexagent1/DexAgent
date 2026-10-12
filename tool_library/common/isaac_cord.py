"""Cords simulated in Isaac Sim (PhysX), and their kinematic MuJoCo mirror.

DexAgent simulates deformable objects in Isaac Sim and everything else (the robot, the table, rigid objects) in MuJoCo.
A cord (rope, cable, string) is the segmented cable of Lim et al., "Planar Robot Casting with Real2Sim2Real
Self-Supervised Learning" (arXiv:2111.04814): identical capsule rigid bodies on D6 ball joints (translations locked),
one PhysX articulation, laid along its measured centreline, its bending resisted by joint drives toward the laid (or a
straight) rest pose. This module holds what the two simulators share:

  * the cord specification (`cord_spec`): geometry and every PhysX coefficient, resolved once in Stage 2 and saved
    next to the Isaac scene (`<name>_cord.json`), so Stage 2's settling, Stage 3's episode and Stage 4's replays
    simulate the same asset;
  * the USD builders used inside Isaac Sim (`build_cord`, `GraspRig`); pxr exists only in Isaac Sim's Python after
    SimulationApp starts, so they take the modules as an argument;
  * the MIRROR: a MuJoCo cable with the cord's links, named as construct_rope names a cable (`<name>B_first`,
    `<name>B_1` ..., `<name>B_last`, ball joints), which MuJoCo never steps. Its qpos is written from the Isaac
    recording frame by frame (`mirror_qpos`), so every MuJoCo gate that reads object states (the Stage 1 goal gates,
    the episode audits, deformable_integrity_audit) measures the Isaac cord with its own code.

Numpy only at import time.
"""
from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SCHEMA = "dexagent.isaac_cord.v1"

# Every PhysX coefficient of a cord. `thin_cord` is build_isaac_scene's original model (8 mm links, heavily damped:
# settles a 1-6 mm cord laid on a table); `segmented_rope` is the coefficient set fitted on a real 27.7 mm velvet rope
# with a settle / drag-springback / arch-release probe suite and frozen (laid-loop drift 1 mm, dragged-end springback
# 6 mm, no flight over 85-113 s of manipulation).
PRESETS = {
    "thin_cord": {
        "segment_m": 0.008, "density_kg_m3": 1100.0, "capsule_trim_m": 0.0015, "rest_pose": "laid",
        "bend_stiffness": 0.004, "bend_damping": 0.002, "max_joint_velocity_deg_s": 800.0,
        "joint_friction_static": 0.0015, "joint_friction_dynamic": 0.0012,
        "linear_damping": 3.0, "angular_damping": 4.0, "max_depenetration_velocity": 0.05, "max_linear_velocity": 2.0,
        "sleep_threshold": 0.0, "contact_offset_m": 0.001, "rest_offset_m": 0.0,
        "static_friction": 0.8, "dynamic_friction": 0.7, "restitution": 0.0,
        "table_static_friction": 1.2, "table_dynamic_friction": 1.0, "table_friction_combine": None, "table_contact_offset_m": 0.002,
        "self_collision": False, "solver_position_iterations": None, "solver_velocity_iterations": None,
    },
    "segmented_rope": {
        "segment_m": 0.027353, "density_kg_m3": 1100.0, "capsule_trim_m": 0.008, "rest_pose": "laid",
        "bend_stiffness": 0.05, "bend_damping": 0.02, "max_joint_velocity_deg_s": 1500.0,
        "joint_friction_static": 0.02, "joint_friction_dynamic": 0.016,
        "linear_damping": 0.2, "angular_damping": 1.0, "max_depenetration_velocity": 0.5, "max_linear_velocity": 4.0,
        "sleep_threshold": 0.0, "contact_offset_m": 0.002, "rest_offset_m": 0.0,
        "static_friction": 0.6, "dynamic_friction": 0.558, "restitution": 0.0,
        # combined with max, the sticky table wins over the slippery rope-on-rope contact (a rope does not tow itself)
        "table_static_friction": 1.8, "table_dynamic_friction": 1.674, "table_friction_combine": "max", "table_contact_offset_m": 0.004,
        "self_collision": True, "solver_position_iterations": 48, "solver_velocity_iterations": 4,
    },
}
# How a hand holds a cord: a kinematic anchor follows the palm and a compliant D6 drive weld joins it to the one
# nearest link (a hand squeezing 18 g capsules by contact is a solver fight, not a grasp; a rigid weld on a tightening
# knot explodes). A release fades the weld out instead of switching it off: every rope jump measured sat at a hard
# release, as stored tension snapped free. While a release fades the anchor stays where the fingers opened (an anchor
# that follows the departing hand drags the rope along), and the cord's body damping is raised for a moment.
GRASP_DEFAULTS = {
    "weld_translational_stiffness": 600.0, "weld_translational_damping": 2.0,
    "weld_rotational_stiffness": 2.0, "weld_rotational_damping": 0.05,
    "capture_m": 0.03, "slip_m": 0.05, "fade_frames": 21, "calm_frames": 18, "calm_damping": [2.5, 6.0],
}
MAX_LINKS = 80          # links per articulation PhysX settles reliably (thin cords explode from ~120)
LEGACY_KEYS = {"bend_stiffness": "bend_stiffness", "bend_damping": "bend_damping", "joint_friction": "joint_friction_static",
               "density": "density_kg_m3"}


# --------------------------------------------------------------------------------------------- the specification --
def polyline_length(P) -> float:
    P = np.asarray(P, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def cord_spec(cord: dict, n_links: int | None = None) -> dict:
    """A cord dict ({name, points, radius} plus optional preset, physics overrides, n_links, segment_m, rgb, cap_rgb,
    cap_length_a_m, cap_length_b_m, fix_start, fix_end, grasp) resolved to the full specification every simulation of
    it uses. The legacy top-level keys of build_isaac_scene (bend_stiffness, bend_damping, joint_friction, density)
    still apply."""
    c = copy.deepcopy(dict(cord))
    for k in ("name", "points", "radius"):
        if k not in c:
            raise ValueError(f"cord {c.get('name', '?')!r} needs {k}")
    P = np.asarray(c["points"], float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2 or not np.isfinite(P).all():
        raise ValueError(f"cord {c['name']!r}: points must be N >= 2 finite [x, y, z] rows")
    preset = str(c.get("preset") or "thin_cord")
    if preset not in PRESETS:
        raise ValueError(f"cord {c['name']!r}: unknown preset {preset!r} (one of {sorted(PRESETS)})")
    phys = dict(PRESETS[preset])
    for old, new in LEGACY_KEYS.items():
        if c.get(old) is not None:
            phys[new] = float(c[old])
    if c.get("joint_friction") is not None:
        phys["joint_friction_dynamic"] = 0.8 * float(c["joint_friction"])
    if c.get("self_collision") is not None:
        phys["self_collision"] = bool(c["self_collision"])
    unknown = set(c.get("physics") or {}) - set(phys)
    if unknown:
        raise ValueError(f"cord {c['name']!r}: unknown physics keys {sorted(unknown)} (known: {sorted(phys)})")
    phys.update(c.get("physics") or {})
    grasp = dict(GRASP_DEFAULTS)
    unknown = set(c.get("grasp") or {}) - set(grasp)
    if unknown:
        raise ValueError(f"cord {c['name']!r}: unknown grasp keys {sorted(unknown)} (known: {sorted(grasp)})")
    grasp.update(c.get("grasp") or {})
    L = polyline_length(P)
    if L <= 0:
        raise ValueError(f"cord {c['name']!r}: its points have zero length")
    k = n_links or c.get("n_links")
    if not k:
        seg = float(c.get("segment_m") or phys["segment_m"])
        k = min(max(4, int(round(L / seg))), MAX_LINKS)
    k = int(k)
    R = float(c["radius"])
    rgb = c.get("rgb") or c.get("rgba") or c.get("color") or [0.7, 0.2, 0.2]
    rgb = [float(v) for v in list(rgb)[:3]]
    if max(rgb) > 1.0:
        rgb = [v / 255.0 for v in rgb]
    out = {"schema": SCHEMA, "name": str(c["name"]), "preset": preset, "points": np.round(P, 6).tolist(), "radius": R,
           "length_m": L, "n_links": k, "segment_m": L / k, "rgb": rgb,
           "cap_rgb": [float(v) for v in c["cap_rgb"][:3]] if c.get("cap_rgb") else None,
           "cap_length_a_m": float(c.get("cap_length_a_m") or 0.0), "cap_length_b_m": float(c.get("cap_length_b_m") or 0.0),
           "fix_start": bool(c.get("fix_start")), "fix_end": bool(c.get("fix_end")),
           "cast_shadows": bool(c["cast_shadows"]) if c.get("cast_shadows") is not None else R >= 0.003,
           "physics": phys, "grasp": grasp}
    if out["segment_m"] <= phys["capsule_trim_m"] + 1e-4:
        out["physics"]["capsule_trim_m"] = 0.2 * out["segment_m"]
    if c.get("link_poses"):
        out["link_poses"] = c["link_poses"]
    return out


def link_mass(spec) -> float:
    return float(spec["physics"]["density_kg_m3"]) * math.pi * spec["radius"] ** 2 * spec["segment_m"]


def cap_links(spec) -> tuple[int, int]:
    """How many links at each end carry the end-cap colour (cap lengths snapped to whole links)."""
    seg = spec["segment_m"]
    a = int(round(spec.get("cap_length_a_m", 0.0) / seg)) if spec.get("cap_length_a_m") else 0
    b = int(round(spec.get("cap_length_b_m", 0.0) / seg)) if spec.get("cap_length_b_m") else 0
    return a, b


def save_spec(spec, path) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(spec, indent=1))
    return str(path)


def load_spec(path) -> dict:
    s = json.loads(Path(path).read_text())
    if s.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not an Isaac cord specification ({SCHEMA})")
    return s


# --------------------------------------------------------------------------------------------------- geometry --
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_x_to(d):
    """(w, x, y, z) rotating +X onto d."""
    d = np.asarray(d, float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([1.0, 0, 0, 0])
    d = d / n
    c = float(d[0])
    if c > 1 - 1e-9:
        return np.array([1.0, 0, 0, 0])
    if c < -1 + 1e-9:
        return np.array([0.0, 0, 0, 1])
    ax = np.cross([1.0, 0, 0], d)
    ax /= np.linalg.norm(ax)
    th = math.acos(c)
    return np.array([math.cos(th / 2), *(math.sin(th / 2) * ax)])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def lay_poses(points, n_links, radius=0.0):
    """Resample a centreline into n_links equal links: centres [n, 3] and orientations [n, 4] (x along the link). A link
    is lifted to rest on z = 0 at least (radius above the table) when radius is given."""
    P = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.linspace(0.0, s[-1], int(n_links) + 1)
    path = np.stack([np.interp(want, s, P[:, i]) for i in range(3)], axis=1)
    centres = 0.5 * (path[:-1] + path[1:])
    if radius:
        centres[:, 2] = np.maximum(centres[:, 2], float(radius) + 0.0005)
    quats = np.stack([quat_x_to(t) for t in np.diff(path, axis=0)])
    return centres, quats


def frames_from_centreline(C, segment_m=None):
    """Link orientations from link centres alone (x along the chain, roll by parallel transport): what a centreline
    recording without orientations gives the mirror. Each link is aimed from the joint its predecessor ends at to its
    own recorded centre, so the mirror's chain reproduces the centres without accumulating error along the cord. A
    capsule is symmetric about its axis, so the roll is immaterial."""
    C = np.asarray(C, float)
    n = len(C)
    seg = float(segment_m) if segment_m else float(np.median(np.linalg.norm(np.diff(C, axis=0), axis=1)))
    T = np.zeros_like(C)
    T[0] = C[1] - C[0]
    T[0] /= max(np.linalg.norm(T[0]), 1e-12)
    joint = C[0] + 0.5 * seg * T[0]
    for i in range(1, n):
        v = C[i] - joint
        T[i] = v / max(np.linalg.norm(v), 1e-12)
        joint = joint + seg * T[i]
    Q = np.zeros((n, 4))
    up = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(up, T[0]); y /= np.linalg.norm(y)
    for i in range(n):
        x = T[i]
        y = y - np.dot(y, x) * x
        if np.linalg.norm(y) < 1e-9:
            y = np.cross([0.0, 0.0, 1.0], x)
        y /= np.linalg.norm(y)
        Q[i] = mat_to_quat(np.stack([x, y, np.cross(x, y)], axis=1))
    return Q


# ------------------------------------------------------------------------------------------- the MuJoCo mirror --
def body_names(name, n_links):
    """construct_rope's body names for a cable of n_links links."""
    return [f"{name}B_first"] + [f"{name}B_{i}" for i in range(1, n_links - 1)] + [f"{name}B_last"]


# a planning copy is stepped by MuJoCo only to give the robot's motion something to pick up (lay, drag, pin act on its
# links); its MuJoCo response is replaced by the Isaac recording, so it only needs to hang together, not to be the rope
PROXY_JOINT = {"damping": "0.01", "armature": "0.0001", "frictionloss": "0.002"}


def mirror_bodies_xml(spec, proxy=False) -> ET.Element:
    """The mirror cable as an MJCF <body> tree (root free joint, ball joints, one capsule per link along +x). proxy=True
    gives the ball joints some damping so MuJoCo can step it as a planning copy."""
    name, n, seg, R = spec["name"], int(spec["n_links"]), float(spec["segment_m"]), float(spec["radius"])
    names = body_names(name, n)
    ca, cb = cap_links(spec)
    rope_rgba = " ".join(f"{v:.4g}" for v in spec["rgb"]) + " 1"
    cap_rgba = " ".join(f"{v:.4g}" for v in (spec.get("cap_rgb") or spec["rgb"])) + " 1"
    dens = f"{spec['physics']['density_kg_m3']:.6g}"
    root = parent = None
    for i in range(n):
        b = ET.Element("body", name=names[i], pos=("0 0 0" if i == 0 else f"{seg:.6g} 0 0"))
        tag = "first" if i == 0 else ("last" if i == n - 1 else str(i))
        if i == 0:
            ET.SubElement(b, "joint", name=f"{name}J_{tag}", type="free", group="3", damping="0", armature="0")
        else:
            ET.SubElement(b, "joint", name=f"{name}J_{tag}", type="ball", pos="0 0 0", group="3",
                          **(PROXY_JOINT if proxy else {"damping": "0", "armature": "0"}))
        cap = i < ca or i >= n - cb
        ET.SubElement(b, "geom", name=f"{name}G{i}", type="capsule", size=f"{R:.6g} {seg / 2:.6g}", pos=f"{seg / 2:.6g} 0 0",
                      quat="0.707107 0 -0.707107 0", density=dens, rgba=cap_rgba if cap else rope_rgba, condim="4", group="0")
        if parent is None:
            root = b
        else:
            parent.append(b)
        parent = b
    return root


def mirror_qpos(centres, quats, segment_m):
    """The mirror cable's qpos from link poses (centres [n, 3], orientations [n, 4] wxyz with x along each link): the
    root's free joint at the first link's start, then each ball joint's rotation relative to its parent link."""
    C, Q = np.asarray(centres, float), np.asarray(quats, float)
    x0 = quat_to_mat(Q[0])[:, 0]
    q = [*(C[0] - 0.5 * float(segment_m) * x0), *Q[0]]
    for i in range(1, len(C)):
        r = qmul(qconj(Q[i - 1]), Q[i])
        q.extend(r / np.linalg.norm(r))
    return np.asarray(q, float)


def mirror_nq(n_links) -> int:
    return 7 + 4 * (int(n_links) - 1)


def cable_qpos_slice(model, spec):
    """(first, last + 1) qpos addresses of a cord's cable (a mirror or a planning copy) in a MuJoCo model, or None."""
    names = body_names(spec["name"], spec["n_links"])
    try:
        ids = [model.body(n).id for n in names]
    except KeyError:
        return None
    adr = [int(model.jnt_qposadr[model.body_jntadr[i]]) for i in ids]
    if any(int(model.body_jntnum[i]) != 1 for i in ids) or adr != sorted(adr):
        raise ValueError(f"{spec['name']}: the cable in the scene is not a chain of {spec['n_links']} links with one joint each")
    a, b = adr[0], adr[-1] + 4
    if b - a != mirror_nq(spec["n_links"]):
        raise ValueError(f"{spec['name']}: the cable in the scene has {b - a} qpos, a cord of {spec['n_links']} links has {mirror_nq(spec['n_links'])}")
    return a, b


def mirror_centres(model, data, spec):
    """Link centres of the mirror cable in a MuJoCo model at the current state."""
    ids = [model.body(n).id for n in body_names(spec["name"], spec["n_links"])]
    return np.array([data.xpos[i] + 0.5 * spec["segment_m"] * data.xmat[i].reshape(3, 3)[:, 0] for i in ids])


def add_mirrors(scene_xml, specs, out_xml, initial=None, proxy=False) -> dict:
    """scene_xml with the mirror cables appended to its worldbody (their qpos after the scene's own), every keyframe
    extended with the cables' initial qpos (specs' link_poses, else laid along points). Relative asset references are
    kept as they are and the files are copied next to out_xml at the same relative paths (runtime scenes name their
    meshes by content, assets/<sha256>.stl), so the mirror travels with its folder like the scene it extends. Returns
    {path, nq_scene, slices {name: [a, b]}, assets [relative paths]}."""
    import shutil
    src = Path(scene_xml).resolve()
    out_dir = Path(out_xml).resolve().parent
    tree = ET.parse(src)
    root = tree.getroot()
    comp = root.find("compiler")
    base = src.parent
    assets = []
    for el in root.iter():
        if el.tag not in ("mesh", "texture", "hfield", "skin"):
            continue
        sub = ((comp.get("meshdir" if el.tag == "mesh" else "texturedir") or comp.get("assetdir")) if comp is not None else None) or ""
        for attr in ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback"):
            f = el.get(attr)
            if not f or Path(f).is_absolute() or Path(sub).is_absolute():
                continue
            rel = Path(sub) / f
            s_, d_ = base / rel, out_dir / rel
            if s_.resolve() != d_.resolve():
                d_.parent.mkdir(parents=True, exist_ok=True)
                if not d_.exists():
                    shutil.copy2(s_, d_)
            assets.append(str(rel))
    wb = root.find("worldbody")
    import mujoco
    nq0 = mujoco.MjModel.from_xml_path(str(src)).nq
    slices, extra = {}, []
    a = nq0
    for s in specs:
        wb.append(mirror_bodies_xml(s, proxy=proxy))
        lp = (initial or {}).get(s["name"]) or s.get("link_poses")
        if lp is not None:
            C, Q = np.asarray(lp["centres"], float), np.asarray(lp["quats"], float)
        else:
            C, Q = lay_poses(s["points"], s["n_links"], s["radius"])
        q = mirror_qpos(C, Q, s["segment_m"])
        extra.extend(q.tolist())
        slices[s["name"]] = [a, a + len(q)]
        a += len(q)
    nv_extra = sum(6 + 3 * (int(s["n_links"]) - 1) for s in specs)
    kf = root.find("keyframe")
    if kf is not None:
        for key in kf.findall("key"):
            if key.get("qpos") is not None:
                key.set("qpos", key.get("qpos").strip() + " " + " ".join(f"{v:.8g}" for v in extra))
            if key.get("qvel") is not None:
                key.set("qvel", key.get("qvel").strip() + " " + " ".join(["0"] * nv_extra))
    out = Path(out_xml)
    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out)
    m = mujoco.MjModel.from_xml_path(str(out))
    if m.nq != a:
        raise RuntimeError(f"mirror scene has nq={m.nq}, expected {a}")
    return {"path": str(out), "nq_scene": int(nq0), "slices": slices, "assets": sorted(set(assets))}


# --------------------------------------------------------------------------------------- inside Isaac Sim (USD) --
def build_cord(stage, usd, root_path, spec, mat_cache, link_poses=None, table_material=None, articulation=True):
    """Author the cord on a live stage. `usd` = {Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf} (pxr);
    mat_cache(rgb, roughness, metallic) -> a visual material. With articulation=False the links are authored under
    root_path without an articulation root of their own: the caller's enclosing articulation (several strands tied to
    one junction body) holds them, with its own self-collision and solver settings. Returns {links, joints,
    drive_stiffness_attrs, body_apis, centres, quats}."""
    UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf = (usd[k] for k in ("UsdGeom", "UsdPhysics", "UsdShade", "PhysxSchema", "Gf", "Sdf"))
    ph = spec["physics"]
    n, seg, R = int(spec["n_links"]), float(spec["segment_m"]), float(spec["radius"])
    if link_poses is not None:
        centres, quats = np.asarray(link_poses["centres"], float), np.asarray(link_poses["quats"], float)
    else:
        centres, quats = lay_poses(spec["points"], n, R)
    mass = link_mass(spec)
    cyl = max(seg - float(ph["capsule_trim_m"]), 0.001)

    def xform(prim, pos, quat):
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
        xf.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in quat]))

    phys_mat = UsdShade.Material.Define(stage, f"{root_path}_phys")
    pm = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
    pm.CreateStaticFrictionAttr(float(ph["static_friction"]))
    pm.CreateDynamicFrictionAttr(float(ph["dynamic_friction"]))
    pm.CreateRestitutionAttr(float(ph["restitution"]))
    rope_mat = mat_cache(spec["rgb"], 0.95, 0.0)
    cap_mat = mat_cache(spec.get("cap_rgb") or spec["rgb"], 0.35, 0.85 if spec.get("cap_rgb") else 0.0)
    ca, cb = cap_links(spec)

    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    if articulation:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        art = PhysxSchema.PhysxArticulationAPI.Apply(root)
        art.CreateEnabledSelfCollisionsAttr(bool(ph["self_collision"]))
        if ph.get("solver_position_iterations"):
            art.CreateSolverPositionIterationCountAttr(int(ph["solver_position_iterations"]))
        if ph.get("solver_velocity_iterations"):
            art.CreateSolverVelocityIterationCountAttr(int(ph["solver_velocity_iterations"]))
    links, body_apis = [], []
    for i in range(n):
        lp = f"{root_path}/link_{i:03d}"
        xf = UsdGeom.Xform.Define(stage, lp)
        xform(xf.GetPrim(), centres[i], quats[i])
        cap = UsdGeom.Capsule.Define(stage, lp + "/geom")
        cap.CreateAxisAttr("X"); cap.CreateHeightAttr(cyl); cap.CreateRadiusAttr(R)
        if not spec.get("cast_shadows", True):
            cap.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim())
        pc.CreateContactOffsetAttr(float(ph["contact_offset_m"])); pc.CreateRestOffsetAttr(float(ph["rest_offset_m"]))
        UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(cap_mat if (i < ca or i >= n - cb) else rope_mat)
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim())
        prb.CreateMaxDepenetrationVelocityAttr(float(ph["max_depenetration_velocity"]))
        prb.CreateLinearDampingAttr(float(ph["linear_damping"])); prb.CreateAngularDampingAttr(float(ph["angular_damping"]))
        prb.CreateSleepThresholdAttr(float(ph["sleep_threshold"])); prb.CreateMaxLinearVelocityAttr(float(ph["max_linear_velocity"]))
        UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(mass)
        links.append(xf.GetPrim()); body_apis.append(prb)
    joints, drive_attrs = [], []
    for i in range(n - 1):
        jp = UsdPhysics.Joint.Define(stage, f"{root_path}/joints/j_{i:03d}")
        prim = jp.GetPrim()
        jp.CreateBody0Rel().SetTargets([links[i].GetPath()]); jp.CreateBody1Rel().SetTargets([links[i + 1].GetPath()])
        jp.CreateLocalPos0Attr(Gf.Vec3f(seg / 2, 0, 0)); jp.CreateLocalPos1Attr(Gf.Vec3f(-seg / 2, 0, 0))
        # rest pose: the laid shape (the cord remembers its lay) or straight (it remembers nothing but stays stiff)
        rel = qmul(qconj(quats[i]), quats[i + 1]) if ph["rest_pose"] == "laid" else np.array([1.0, 0, 0, 0])
        jp.CreateLocalRot0Attr(Gf.Quatf(*[float(v) for v in rel])); jp.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        for ax in ("transX", "transY", "transZ"):        # a ball joint: low > high locks the translation
            lim = UsdPhysics.LimitAPI.Apply(prim, ax); lim.CreateLowAttr(1.0); lim.CreateHighAttr(-1.0)
        for ax in ("rotX", "rotY", "rotZ"):
            drv = UsdPhysics.DriveAPI.Apply(prim, ax); drv.CreateTypeAttr("force")
            drive_attrs.append(drv.CreateStiffnessAttr(float(ph["bend_stiffness"])))
            drv.CreateDampingAttr(float(ph["bend_damping"])); drv.CreateTargetPositionAttr(0.0)
        PhysxSchema.PhysxJointAPI.Apply(prim)
        # PhysxJointAxisAPI has no Python binding: raw-authored. maxJointVelocity is in DEGREES/s
        for ax in ("rotX", "rotY", "rotZ"):
            prim.AddAppliedSchema(f"PhysxJointAxisAPI:{ax}")
            for attr, val in ((f"physxJointAxis:{ax}:staticFrictionEffort", ph["joint_friction_static"]),
                              (f"physxJointAxis:{ax}:dynamicFrictionEffort", ph["joint_friction_dynamic"]),
                              (f"physxJointAxis:{ax}:maxJointVelocity", ph["max_joint_velocity_deg_s"])):
                prim.CreateAttribute(attr, Sdf.ValueTypeNames.Float).Set(float(val))
        joints.append(prim)
    for end, idx in (("fix_start", 0), ("fix_end", n - 1)):
        if spec.get(end):
            fj = UsdPhysics.FixedJoint.Define(stage, f"{root_path}/joints/{end}")
            fj.CreateBody1Rel().SetTargets([links[idx].GetPath()])
    return {"links": links, "joints": joints, "drive_stiffness_attrs": drive_attrs, "body_apis": body_apis,
            "centres": centres, "quats": quats, "mass": mass}


def link_poses_usd(usd, links):
    """(centres [n, 3], quats [n, 4] wxyz) of the links on the live stage."""
    cache = usd["UsdGeom"].XformCache()
    C, Q = [], []
    for lk in links:
        T = cache.GetLocalToWorldTransform(lk)
        t = T.ExtractTranslation(); q = T.ExtractRotationQuat(); im = q.GetImaginary()
        C.append([t[0], t[1], t[2]]); Q.append([q.GetReal(), im[0], im[1], im[2]])
    return np.asarray(C, float), np.asarray(Q, float)


class GraspRig:
    """Kinematic hand anchors and compliant welds on cords: weld()
    joins an anchor to one link at their current relative pose, release() fades the weld out over fade_frames,
    update(frame) runs the fades."""

    def __init__(self, stage, usd, root="/World/anchors", weld_root="/World/welds"):
        self.stage, self.usd, self.root, self.weld_root = stage, usd, root, weld_root
        self.anchors, self._ops, self._fades, self._n = {}, {}, [], 0

    def add_anchor(self, name, pos=(0.0, 0.0, 1.0), quat=(1, 0, 0, 0)):
        U = self.usd
        xf = U["UsdGeom"].Xform.Define(self.stage, f"{self.root}/{name}")
        rb = U["UsdPhysics"].RigidBodyAPI.Apply(xf.GetPrim()); rb.CreateKinematicEnabledAttr(True)
        U["UsdPhysics"].MassAPI.Apply(xf.GetPrim()).CreateMassAttr(1.0)
        self.anchors[name] = xf.GetPrim()
        self.move_anchor(name, pos, quat)
        return xf.GetPrim()

    def move_anchor(self, name, pos, quat):
        U, prim = self.usd, self.anchors[name]
        ops = self._ops.get(name)
        if ops is None:
            xf = U["UsdGeom"].Xformable(prim); xf.ClearXformOpOrder()
            ops = self._ops[name] = (xf.AddTranslateOp(), xf.AddOrientOp())
        ops[0].Set(U["Gf"].Vec3d(*[float(v) for v in pos]))
        ops[1].Set(U["Gf"].Quatf(*[float(v) for v in quat]))

    def weld(self, anchor, link_prim, gains, scale=1.0):
        U = self.usd
        cache = U["UsdGeom"].XformCache()
        rel = cache.GetLocalToWorldTransform(link_prim) * cache.GetLocalToWorldTransform(self.anchors[anchor]).GetInverse()
        jp = U["UsdPhysics"].Joint.Define(self.stage, f"{self.weld_root}/w{self._n:03d}")
        self._n += 1
        prim = jp.GetPrim()
        jp.CreateBody0Rel().SetTargets([self.anchors[anchor].GetPath()]); jp.CreateBody1Rel().SetTargets([link_prim.GetPath()])
        t = rel.ExtractTranslation(); q = rel.ExtractRotationQuat(); qi = q.GetImaginary()
        jp.CreateLocalPos0Attr(U["Gf"].Vec3f(t[0], t[1], t[2])); jp.CreateLocalRot0Attr(U["Gf"].Quatf(q.GetReal(), qi[0], qi[1], qi[2]))
        jp.CreateLocalPos1Attr(U["Gf"].Vec3f(0, 0, 0)); jp.CreateLocalRot1Attr(U["Gf"].Quatf(1, 0, 0, 0))
        jp.CreateExcludeFromArticulationAttr(True)
        attrs = []
        for axes, k, d in ((("transX", "transY", "transZ"), gains["weld_translational_stiffness"], gains["weld_translational_damping"]),
                           (("rotX", "rotY", "rotZ"), gains["weld_rotational_stiffness"], gains["weld_rotational_damping"])):
            for ax in axes:
                drv = U["UsdPhysics"].DriveAPI.Apply(prim, ax); drv.CreateTypeAttr("force")
                sa, da = drv.CreateStiffnessAttr(float(k) * scale), drv.CreateDampingAttr(float(d) * scale)
                drv.CreateTargetPositionAttr(0.0)
                attrs.append((sa, float(k) * scale, da, float(d) * scale))
        return {"prim": prim, "attrs": attrs}

    def off(self, weld):
        U = self.usd
        j = U["UsdPhysics"].Joint(weld["prim"])
        (j.GetJointEnabledAttr() or j.CreateJointEnabledAttr(True)).Set(False)

    def release(self, weld, frame, fade_frames):
        self._fades.append((weld, int(frame), max(int(fade_frames), 1)))

    def update(self, frame):
        done = []
        for entry in self._fades:
            weld, t0, dur = entry
            u = (frame - t0) / float(dur)
            if u >= 1.0:
                self.off(weld); done.append(entry); continue
            s = (1.0 - u) ** 2                                 # quadratic bleed-off: a hand opening
            for sa, s0, da, d0 in weld["attrs"]:
                sa.Set(float(s0 * s)); da.Set(float(d0 * max(s, 0.05)))
        for entry in done:
            self._fades.remove(entry)
