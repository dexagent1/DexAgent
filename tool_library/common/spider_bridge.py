"""DexAgent's bench <-> SPIDER (facebookresearch/spider, physics-informed dexterous retargeting).

SPIDER optimizes a hand trajectory by annealed sampling (DIAL-MPC) in MuJoCo Warp so that the objects follow a
reference under contact physics. It runs in its own environment from its own checkout; nothing of it is copied here.
This module is the adapter on our side:

* the reference, in the bench's terms: the tracking reference comes from `tracking.best_reference` (an earlier method's
  reference, else `tracking.reference_from_attempt`: the commanded reference of the agent's best executed attempt);
  `human_keypoints` turns the demonstrator's hand joints (EgoDex hands.npz or HaMeR) into SPIDER's keypoints, the
  objects following the reference or a fitted track;
* SPIDER's dataset folder: the setup's own Sharpa hand MJCF is already a SPIDER robot (same joints, sites, actuators),
  given our actuation limits (`write_robot_xml`); the scene holds the hand(s), the table as the floor, one free object
  per hand, the other objects as fixed obstacles and our contact parameters (`write_scene`); the keypoints and/or the
  kinematic trajectory at SPIDER's reference rate;
* the mandatory preparation (`prep_keypoints`): ground the manipulated object on its support, put the fingertips that
  touch it on its surface (one pad radius out, so the links are not driven into it) and re-derive the thumb at the
  antipode of the finger contacts through the object centre (a human thumb is the least reliable joint and without
  opposition the fingers only push);
* the runs (`run_spider_ik`, `run_spider_mjwp`) as subprocesses of the configured interpreter;
* the result back on the bench (`read_mjwp`, `bench_command_from_spider`): the executed wrist plus the servo push SPIDER's
  wrist applied, re-expressed for the bench's stiffer wrist servo so the same force results, and the finger commands.

The skill (physics_informed_retargeting) runs these, checks each result once on the bench's runtime
(tracking.BenchTracker) and writes the tracking reference and the plan for execute_subgoal_plan; the gates are
execute_subgoal_plan's and the finish verifiers'.

SPIDER's file contract (processed/<dataset>/<robot>/<embodiment>/<task>/...): scene.xml, scene_eq.xml, task_info.json,
<data_id>/trajectory_kinematic.npz -> run_mjwp.py -> <data_id>/trajectory_mjwp.npz; keypoints for its IK under
processed/<dataset>/mano/<embodiment>/<task>/<data_id>/trajectory_keypoints.npz. Its qpos is [right hand 28, left
hand 28, right object 7, left object 7] (the hands present), its ctrl the hands' 28 position servos each.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E
from tool_library.common import hand as H
from tool_library.common import plan as P

DATASET = "dexagent"
ROBOT = "sharpa"
FINGERS = H.FINGERS
SIDE_ORDER = ("right", "left")           # SPIDER's order of hands and of their objects in qpos
WRIST_KEYS = ("x", "y", "z", "rx", "ry", "rz")
WRIST_SUFFIX = ("pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z")
# The manipulated object's free joint keeps the bench's own armature, damping and friction loss (zero on Stage 2's
# objects). SPIDER's generate_xml gives every object armature 1e-4: nine times the rotational inertia of the example's 30 g
# cup, so the cup SPIDER kept upright in a two-finger pinch turned 90 degrees out of the same pinch on the bench; and on an
# 11 g ring SPIDER's keypoint IK (a stiff weld from the object to its keypoint) diverged on the first steps with any armature
# (1e-4 or 1e-2), while it was stable with the bench's zero.
PAIR_SOLREF = (0.006, 1.0)               # the bench's hand-object pair (episode.build_execution_model)
PAIR_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
BENCH_WRIST_KP = 10000.0                 # the bench's wrist servo stiffness during execution (execute_subgoal_plan wrist_kp)
STATIC_MARGIN_M = 0.15                   # obstacles farther than this from the hands' and objects' paths are left out


class SpiderError(RuntimeError):
    """A SPIDER stage that did not produce its output; the message carries the tail of its log."""


# ------------------------------------------------------------------------------------------------- bench layout --
def hand_columns(model):
    """{side: {wrist [6 qadr: x y z rx ry rz], fingers [22 qadr], finger_names [...], rig}} of the bench hands. The
    fingers are in the hand's joint order, which is SPIDER's actuator order (both come from the same hand MJCF)."""
    out = {}
    for side, rig in P.scene_rigs(model).items():
        names = sorted(rig.finger_qpos, key=lambda k: rig.finger_qpos[k])
        out[side] = {"wrist": [int(rig.wrist_qpos[k]) for k in WRIST_KEYS], "fingers": [int(rig.finger_qpos[k]) for k in names],
                     "finger_names": names, "rig": rig}
    return out


def _subtree(model, bid):
    out, todo = [], [int(bid)]
    while todo:
        b = todo.pop()
        out.append(b)
        todo += [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
    return out


def object_roots(model):
    """Root bodies of the scene's objects (children of the world that are not the robot, the table or a camera rig)."""
    out = []
    for b in range(1, model.nbody):
        nm = model.body(b).name or ""
        if int(model.body_parentid[b]) != 0 or nm.startswith("robot_") or nm in ("table", "camrig"):
            continue
        if any(int(model.geom_bodyid[g]) in _subtree(model, b) for g in range(model.ngeom)):
            out.append(nm)
    return out


def body_track(model, Q, body):
    """[T, 3] world positions and [T, 4] quaternions (w, x, y, z) of a body along Q."""
    d = mujoco.MjData(model)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if b < 0:
        raise ValueError(f"no body {body!r} in the scene")
    pos, quat = np.zeros((len(Q), 3)), np.zeros((len(Q), 4))
    for t, q in enumerate(Q):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        pos[t], quat[t] = d.xpos[b], d.xquat[b]
    return pos, quat


def infer_roles(model, Q, labels, sides=None, objects=None, move_tol_m=0.01):
    """Which hands act and which free object each one manipulates, read off the reference: a hand acts when its wrist
    travels more than move_tol_m or a finger moves 0.1 rad; its object is the free body its phase labels name most often
    among those that move (else the one that moves most near that hand). `sides` / `objects` ({side: body}) override.
    Returns {sides, objects {side: body}, statics [object roots that are not manipulated], travel {body: m}}."""
    Q = np.asarray(Q, float)
    cols = hand_columns(model)
    free = E.object_free_joints(model)
    travel = {}
    for b, a in free.items():
        p = Q[:, a:a + 3]
        travel[b] = float(np.linalg.norm(p - p[0], axis=1).max())
    if sides is None:
        sides = []
        for s, c in cols.items():
            W, F = Q[:, c["wrist"][:3]], Q[:, c["fingers"]]
            if np.linalg.norm(W - W[0], axis=1).max() > move_tol_m or np.abs(F - F[0]).max() > 0.1:
                sides.append(s)
    sides = [s for s in SIDE_ORDER if s in sides]
    objects = dict(objects or {})
    for s in sides:
        if s in objects:
            continue
        counts = {}
        for lab in labels:
            part = E.split_label(lab).get(s, "")
            for b in free:
                if re.search(rf"(?<![\w]){re.escape(b)}(?![\w])", part):
                    counts[b] = counts.get(b, 0) + 1
        moving = {b: n for b, n in counts.items() if travel[b] > move_tol_m}
        if moving:
            objects[s] = max(moving, key=moving.get)
        elif counts:
            objects[s] = max(counts, key=counts.get)
        else:                                       # nothing named: the moving free body closest to this hand's path
            d = mujoco.MjData(model)
            rig = cols[s]["rig"]
            best, best_d = None, np.inf
            for b, a in free.items():
                if travel[b] <= move_tol_m:
                    continue
                dmin = np.inf
                for t in range(0, len(Q), max(1, len(Q) // 60)):
                    d.qpos[:] = Q[t]; mujoco.mj_kinematics(model, d)
                    dmin = min(dmin, float(np.linalg.norm(Q[t, a:a + 3] - d.xpos[rig.root_body])))
                if dmin < best_d:
                    best, best_d = b, dmin
            if best is None:
                raise ValueError(f"the {s} hand acts but no free object moves in the reference: SPIDER needs an object to track")
            objects[s] = best
    for s, b in objects.items():
        if b not in free:
            raise ValueError(f"{b!r} has no free joint: the SPIDER backend manipulates free rigid bodies only")
    manip = set(objects.values())
    statics = [b for b in object_roots(model) if b not in manip]
    return {"sides": sides, "objects": {s: objects[s] for s in sides}, "statics": statics, "travel": {b: round(v, 4) for b, v in travel.items()}}


# ------------------------------------------------------------------------------------------------ SPIDER robot --
def robot_xml_name(sides):
    sides = [s for s in SIDE_ORDER if s in sides]
    return "bimanual.xml" if len(sides) == 2 else f"{sides[0]}.xml"


def write_robot_xml(setup_dir, sides, out_xml, match_execution=True, wrist_kp=BENCH_WRIST_KP, sim_dt=0.01):
    """The setup's Sharpa hand(s) as one SPIDER robot MJCF (right first). The hand files already follow SPIDER's robot
    contract (virtual wrist pos_x..rot_z, `<side>_palm`, `<side>_<finger>_tip`, track_/trace_ sites, collision_hand_
    geoms, one position actuator per joint). match_execution gives SPIDER the bench's actuation (episode.servo_rollout
    + hand.limit_hand_forces): finger servos capped at 0.8 N m with armature 0.05, the wrist servo as stiff as the bench's
    (wrist_kp) and capped at 40 N / 3 N m, the hand's weight carried (gravcomp). A grasp SPIDER finds with stronger
    fingers would not transfer, and with the asset's ten times softer wrist SPIDER's hand lagged its command by up to
    13 mm: replayed on the bench's stiff wrist the same pinch of the cup example opened at the lift. The wrist's
    rotational reflected inertia stays above wrist_kp * sim_dt**2 (an explicit spring at SPIDER's 10 ms step is unstable
    below it); the bench's own 0.05 is for its 2 ms step."""
    setup_dir = Path(setup_dir)
    sides = [s for s in SIDE_ORDER if s in sides]
    if not sides:
        raise ValueError("no hand to write")
    roots = []
    for s in sides:
        src = setup_dir / "robot" / f"sharpa_{s}.xml"
        if not src.exists():
            raise FileNotFoundError(f"{src}: the setup folder has no Sharpa {s} hand")
        r = ET.parse(src).getroot()
        comp = r.find("compiler")
        md = (src.parent / (comp.get("meshdir", "") if comp is not None else "")).resolve()
        for m in r.iter("mesh"):                   # absolute files: the merged robot is read from anywhere
            if m.get("file") and not os.path.isabs(m.get("file")):
                m.set("file", str(md / m.get("file")))
        if comp is not None and "meshdir" in comp.attrib:
            del comp.attrib["meshdir"]
        roots.append(r)
    base = roots[0]
    for r in roots[1:]:
        for tag in ("asset", "worldbody", "actuator", "contact", "sensor", "equality", "tendon"):
            src_el = r.find(tag)
            if src_el is None:
                continue
            dst = base.find(tag)
            if dst is None:
                dst = ET.SubElement(base, tag)
            have = {(c.tag, c.get("name")) for c in dst}
            for c in list(src_el):
                if c.get("name") is not None and (c.tag, c.get("name")) in have:
                    continue                      # a shared asset (skybox texture)
                dst.append(c)
    if match_execution:
        wrist_rot = {f"{s}_{k}" for s in sides for k in ("rot_x", "rot_y", "rot_z")}
        wrist_pos = {f"{s}_{k}" for s in sides for k in ("pos_x", "pos_y", "pos_z")}
        rot_armature = max(H.WRIST_ROT_ARMATURE, float(wrist_kp) * float(sim_dt) ** 2)
        for j in base.iter("joint"):
            nm = j.get("name") or ""
            if nm in wrist_rot:
                j.set("armature", repr(rot_armature))
            elif nm and nm not in wrist_pos:
                j.set("armature", repr(H.FINGER_ARMATURE))
        for a in base.find("actuator"):
            jn = a.get("joint") or ""
            cap = H.WRIST_FORCE_N if jn in wrist_pos else H.WRIST_TORQUE_NM if jn in wrist_rot else H.FINGER_FORCE_NM
            a.set("forcelimited", "true")
            a.set("forcerange", f"{-cap!r} {cap!r}")
            if jn in wrist_pos or jn in wrist_rot:          # critically damped by the class's dampratio at compile time
                a.set("kp", repr(float(wrist_kp)))
        for b in base.find("worldbody").iter("body"):
            b.set("gravcomp", "1")
    Path(out_xml).parent.mkdir(parents=True, exist_ok=True)
    ET.indent(base, space="  ")
    Path(out_xml).write_text(ET.tostring(base, encoding="unicode"))
    return str(out_xml)


# ------------------------------------------------------------------------------------------------ SPIDER scene --
def _fmt(v):
    return " ".join(repr(float(x)) for x in np.ravel(v))


def _mat_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def _write_obj(path, V, F):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for v in V:
            fh.write(f"v {v[0]:.7g} {v[1]:.7g} {v[2]:.7g}\n")
        for f in F:
            fh.write(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")


def _collides(model, g):
    return bool(H.hand_may_touch(int(model.geom_contype[g]), int(model.geom_conaffinity[g])))


def _geoms_in_frame(model, data, bodies, R0, p0, mesh_dir, tag, visual=True):
    """Geom elements of `bodies` expressed in the frame (R0, p0) (world), meshes written as .obj under mesh_dir.
    Returns ([(xml geom attrs, collides)], [mesh asset attrs])."""
    geoms, meshes = [], []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies:
            continue
        col = _collides(model, g)
        if not col and (not visual or int(model.geom_group[g]) > 2):
            continue
        Rg = data.geom_xmat[g].reshape(3, 3)
        pos = R0.T @ (data.geom_xpos[g] - p0)
        quat = _mat_quat(R0.T @ Rg)
        gt = int(model.geom_type[g])
        name = f"{tag}_g{g}"
        att = {"name": name, "pos": _fmt(pos), "quat": _fmt(quat), "rgba": _fmt(model.geom_rgba[g]),
               "group": "3" if col else "1", "contype": "0", "conaffinity": "0", "density": "0", "mass": "0"}
        if gt == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g])
            va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            f = Path(mesh_dir) / f"{name}.obj"
            _write_obj(f, model.mesh_vert[va:va + vn], model.mesh_face[fa:fa + fn])
            meshes.append({"name": f"mesh_{name}", "file": str(f.resolve())})
            att.update(type="mesh", mesh=f"mesh_{name}")
        elif gt == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        else:
            att.update(type=mujoco.mjtGeom(gt).name[len("mjGEOM_"):].lower(), size=_fmt(model.geom_size[g]))
        if col:
            att.update(friction=_fmt(model.geom_friction[g]), condim=str(int(model.geom_condim[g])),
                       solref=_fmt(model.geom_solref[g]), solimp=_fmt(model.geom_solimp[g]))
        geoms.append((att, col))
    return geoms, meshes


def _combined_inertia(model, data, bodies, R0, p0):
    """Mass, centre and principal inertia (diag, quat) of rigidly joined bodies in the frame (R0, p0)."""
    m_tot, c = 0.0, np.zeros(3)
    for b in bodies:
        m = float(model.body_mass[b])
        m_tot += m; c += m * (R0.T @ (data.xipos[b] - p0))
    if m_tot <= 0:
        return 1e-3, np.zeros(3), np.full(3, 1e-7), np.array([1.0, 0, 0, 0])
    c /= m_tot
    I = np.zeros((3, 3))
    for b in bodies:
        m = float(model.body_mass[b])
        Rb = R0.T @ data.ximat[b].reshape(3, 3)
        Ib = Rb @ np.diag(model.body_inertia[b]) @ Rb.T
        r = R0.T @ (data.xipos[b] - p0) - c
        I += Ib + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    w, V = np.linalg.eigh(I)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    return m_tot, c, np.maximum(w, 1e-9), _mat_quat(V)


def _sub(el, tag, **att):
    return ET.SubElement(el, tag, {k: str(v) for k, v in att.items()})


def write_scene(model, q0, roles, robot_xml, scene_dir, mesh_dir, floor_z=0.0, contact_pos=None, track_bounds=None, object_armature=None):
    """SPIDER's scene.xml and scene_eq.xml for the bench state q0. The hands come from robot_xml; the table top is
    the floor plane (`floor`); each acting hand's object is `<side>_object` with a free joint (its body frame is the
    bench body's, so its 7 qpos are the bench's), geometry and combined inertia; a second hand that holds the same
    object, or none, gets SPIDER's far-away gravity-compensated stand-in; the other objects near the paths
    (track_bounds [lo, hi] world box, widened by STATIC_MARGIN_M) are fixed obstacles at their pose in q0. Contacts are
    explicit pairs with the bench's parameters: hand-object and hand-hand as the execution model pairs them, hand-table
    as the bench's table pairs, object-table/object-object/object-obstacle by MuJoCo's own combination of the geoms.
    contact_pos {side: [5, 3]} places the object track sites (object frame); object_armature overrides the bench's own
    free-joint armature. Returns paths and the layout."""
    scene_dir, mesh_dir = Path(scene_dir), Path(mesh_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    d = mujoco.MjData(model)
    d.qpos[:] = np.asarray(q0, float)
    mujoco.mj_forward(model, d)
    sides = [s for s in SIDE_ORDER if s in roles["sides"]]
    root = ET.parse(robot_xml).getroot()
    root.set("model", "dexagent_spider")
    for el in root.findall("option"):
        root.remove(el)
    opt = ET.Element("option", {"timestep": "0.01", "integrator": "implicitfast", "cone": "elliptic", "impratio": "5"})
    root.insert(1, opt)
    asset = root.find("asset")
    wb = root.find("worldbody")
    tg = next((g for g in range(model.ngeom) if model.body(int(model.geom_bodyid[g])).name == "table" and _collides(model, g)), None)
    floor_att = {"name": "floor", "type": "plane", "size": "0 0 0.05", "pos": f"0 0 {floor_z:.6f}", "rgba": "0.85 0.85 0.85 1",
                 "contype": "0", "conaffinity": "0", "group": "0"}
    if tg is not None:
        floor_att.update(friction=_fmt(model.geom_friction[tg]), solref=_fmt(model.geom_solref[tg]), solimp=_fmt(model.geom_solimp[tg]))
    _sub(wb, "geom", **floor_att)
    hand_geoms = {s: [g.get("name") for g in root.iter("geom") if (g.get("name") or "").startswith(f"collision_hand_{s}_")] for s in sides}
    obj_geoms, obstacle_geoms = {}, {}
    mesh_assets = []
    layout = {"sides": sides, "objects": {}, "statics": [], "dummy": []}
    held = {}
    for s in sides:
        b = roles["objects"][s]
        if b in held.values():                      # both hands on one object: SPIDER's stand-in for the second
            continue
        held[s] = b
    contact_pos = contact_pos or {}
    for s in sides:
        body = _sub(wb, "body", name=f"{s}_object")
        if s in held:
            bname = held[s]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            sub = _subtree(model, bid)
            if any(int(model.body_jntnum[c]) for c in sub if c != bid):
                raise ValueError(f"{bname} is articulated: the SPIDER backend manipulates free rigid bodies only")
            p0, R0 = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()
            body.set("pos", _fmt(p0)); body.set("quat", _fmt(d.xquat[bid]))
            ja = int(model.jnt_dofadr[int(model.body_jntadr[bid])])
            arm = float(model.dof_armature[ja:ja + 6].max()) if object_armature is None else float(object_armature)
            _sub(body, "joint", name=f"{s}_object_joint", type="free", armature=repr(arm),
                 damping=repr(float(model.dof_damping[ja:ja + 6].max())), frictionloss=repr(float(model.dof_frictionloss[ja:ja + 6].max())))
            m, c, I, iq = _combined_inertia(model, d, set(sub), R0, p0)
            _sub(body, "inertial", pos=_fmt(c), quat=_fmt(iq), mass=repr(m), diaginertia=_fmt(I))
            gs, ms = _geoms_in_frame(model, d, set(sub), R0, p0, mesh_dir / bname, f"{s}_object")
            mesh_assets += ms
            for att, col in gs:
                _sub(body, "geom", **att)
            obj_geoms[s] = [a["name"] for a, col in gs if col]
            layout["objects"][s] = {"body": bname, "nq_adr": None}
        else:
            body.set("pos", "0.5 0.5 0.5"); body.set("gravcomp", "1")
            _sub(body, "joint", name=f"{s}_object_joint", type="free", armature="1.0", damping="0", frictionloss="1.0")
            _sub(body, "geom", name=f"{s}_object_mass", type="sphere", size="0.1", density="10", group="3", contype="0", conaffinity="0")
            obj_geoms[s] = []
            layout["dummy"].append(s)
        _sub(body, "site", name=f"{s}_object", type="box", size="0.01 0.02 0.03", rgba="1 0 0 1", group="3")
        _sub(body, "site", name=f"trace_{s}_object", size="0.01", rgba="0 1 0 1", group="4")
        cp = np.asarray(contact_pos.get(s, np.zeros((5, 3))), float).reshape(5, 3)
        for i, f in enumerate(FINGERS):
            _sub(body, "site", name=f"track_object_{s}_{f}_tip", pos=_fmt(cp[i]), size="0.01", rgba="0 1 0 1", group="4")
    lo = hi = None
    if track_bounds is not None:
        lo = np.asarray(track_bounds[0], float) - STATIC_MARGIN_M
        hi = np.asarray(track_bounds[1], float) + STATIC_MARGIN_M
    for bname in roles["statics"]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        sub = set(_subtree(model, bid))
        if lo is not None:
            cg = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub and _collides(model, g)]
            if cg:
                pts = np.array([d.geom_xpos[g] for g in cg]); r = np.array([model.geom_rbound[g] for g in cg])
                if ((pts + r[:, None]) < lo).any(axis=1).all() or ((pts - r[:, None]) > hi).any(axis=1).all():
                    continue
        body = _sub(wb, "body", name=f"static_{bname}")
        gs, ms = _geoms_in_frame(model, d, sub, np.eye(3), np.zeros(3), mesh_dir / bname, f"static_{bname}")
        mesh_assets += ms
        for att, col in gs:
            _sub(body, "geom", **att)
        obstacle_geoms[bname] = [a["name"] for a, col in gs if col]
        layout["statics"].append(bname)
    for s in sides:
        for f in FINGERS:
            for kind in ("hand", "object"):
                mb = _sub(wb, "body", name=f"ref_{kind}_{s}_{f}_tip", mocap="true")
                _sub(mb, "site", name=f"ref_{kind}_{s}_{f}_tip", size="0.02", rgba="0 1 0 1", group="4")
    for ma in mesh_assets:
        _sub(asset, "mesh", **ma)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    hf = H.HAND_PAIR_FRICTION
    hand_pair = {"friction": _fmt([hf[0], hf[0], hf[1]]), "condim": "4", "solref": _fmt(PAIR_SOLREF), "solimp": _fmt(PAIR_SOLIMP)}
    tp = H.TABLE_PAIR
    table_pair = {"friction": _fmt(list(tp["friction"])[:3]), "condim": str(tp["condim"]), "solref": _fmt(tp["solref"])}
    all_obj = [g for s in sides for g in obj_geoms[s]] + [g for gl in obstacle_geoms.values() for g in gl]
    n = 0

    def pair(a, b, att):
        nonlocal n
        _sub(contact, "pair", name=f"p{n}", geom1=a, geom2=b, **att)
        n += 1

    geom_att = {g.get("name"): g.attrib for g in root.iter("geom") if g.get("name")}

    def mixed(a, b):                                # MuJoCo's default combination of two geoms' contact parameters
        A, B = geom_att[a], geom_att[b]
        fa = np.array([float(x) for x in A.get("friction", "1 0.005 0.0001").split()])
        fb = np.array([float(x) for x in B.get("friction", "1 0.005 0.0001").split()])
        f = np.maximum(fa, fb)
        return {"friction": _fmt([f[0], f[0], f[1]]), "condim": str(max(int(A.get("condim", 3)), int(B.get("condim", 3)), 3)),
                "solref": A.get("solref", "0.02 1")}
    for s in sides:
        for h in hand_geoms[s]:
            for o in all_obj:
                pair(h, o, hand_pair)
            pair(h, "floor", table_pair)
    if len(sides) == 2:
        for h in hand_geoms["right"]:
            for k in hand_geoms["left"]:
                pair(h, k, hand_pair)
    objs_free = [s for s in sides if obj_geoms[s]]
    for i, s in enumerate(objs_free):
        for o in obj_geoms[s]:
            pair(o, "floor", mixed(o, "floor"))
            for gl in obstacle_geoms.values():
                for k in gl:
                    pair(o, k, mixed(o, k))
            for s2 in objs_free[i + 1:]:
                for k in obj_geoms[s2]:
                    pair(o, k, mixed(o, k))
    ET.indent(root, space="  ")
    scene = scene_dir / "scene.xml"
    scene.write_text(ET.tostring(root, encoding="unicode"))
    # scene_eq.xml (SPIDER anneals over track->ref equalities when num_dyn > 1): the same scene plus those constraints
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    for site in [x.get("name") for x in root.iter("site") if (x.get("name") or "").startswith("track_")]:
        _sub(eq, "connect", name=f"{site}_equality_constraint", site1=site, site2=site.replace("track", "ref", 1),
             solref="0.02 1", solimp="0 1 100 0.5 2")
    scene_eq = scene_dir / "scene_eq.xml"
    scene_eq.write_text(ET.tostring(root, encoding="unicode"))
    m = mujoco.MjModel.from_xml_path(str(scene))
    layout["nq"], layout["nv"], layout["nu"], layout["npair"] = int(m.nq), int(m.nv), int(m.nu), int(m.npair)
    return {"scene": str(scene), "scene_eq": str(scene_eq), "layout": layout}


# ------------------------------------------------------------------------------------- trajectory conversions --
def spider_columns(spider_model, sides):
    """{side: {wrist [6], fingers [22], finger_names, object [7 qadr]}} in SPIDER's scene."""
    out = {}
    for s in sides:
        w = [int(spider_model.jnt_qposadr[mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_{k}")]) for k in WRIST_SUFFIX]
        fn, fa = [], []
        for j in range(spider_model.njnt):
            nm = spider_model.joint(j).name or ""
            if nm.startswith(f"{s}_") and nm[len(s) + 1:] not in WRIST_SUFFIX and "object" not in nm:
                fn.append(nm[len(s) + 1:]); fa.append(int(spider_model.jnt_qposadr[j]))
        oj = mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_object_joint")
        oa = int(spider_model.jnt_qposadr[oj])
        out[s] = {"wrist": w, "fingers": fa, "finger_names": fn, "object": list(range(oa, oa + 7))}
    return out


def bench_to_spider(model, Q, spider_model, layout, dz=0.0):
    """Bench qpos rows -> SPIDER qpos rows (hands by joint name, each held object's free joint; a stand-in keeps the
    scene's start pose). dz lifts everything (the grounding shift)."""
    Q = np.asarray(Q, float)
    bc, sc = hand_columns(model), spider_columns(spider_model, layout["sides"])
    free = E.object_free_joints(model)
    S = np.tile(spider_model.qpos0, (len(Q), 1))
    for s in layout["sides"]:
        S[:, sc[s]["wrist"]] = Q[:, bc[s]["wrist"]]
        S[:, sc[s]["wrist"][2]] += dz
        name_to_bench = dict(zip(bc[s]["finger_names"], bc[s]["fingers"]))
        for nm, a in zip(sc[s]["finger_names"], sc[s]["fingers"]):
            S[:, a] = Q[:, name_to_bench[nm]]
        ob = layout["objects"].get(s)
        if ob is not None:
            a = free[ob["body"]]
            S[:, sc[s]["object"]] = Q[:, a:a + 7]
            S[:, sc[s]["object"][2]] += dz
    return S


def spider_to_bench(model, Qbase, spider_model, layout, S, dz=0.0, hands_only=False):
    """SPIDER rows -> bench rows on top of Qbase (same length): the acting hands' joints and (unless hands_only) each
    held object's free joint, the grounding shift removed."""
    out = np.array(Qbase, float, copy=True)
    S = np.asarray(S, float)
    bc, sc = hand_columns(model), spider_columns(spider_model, layout["sides"])
    free = E.object_free_joints(model)
    for s in layout["sides"]:
        out[:, bc[s]["wrist"]] = S[:, sc[s]["wrist"]]
        out[:, bc[s]["wrist"][2]] -= dz
        name_to_spider = dict(zip(sc[s]["finger_names"], sc[s]["fingers"]))
        for nm, a in zip(bc[s]["finger_names"], bc[s]["fingers"]):
            out[:, a] = S[:, name_to_spider[nm]]
        ob = layout["objects"].get(s)
        if ob is not None and not hands_only:
            a = free[ob["body"]]
            out[:, a:a + 7] = S[:, sc[s]["object"]]
            out[:, a + 2] -= dz
    return out


def quat_columns(model):
    """qpos addresses of every free/ball joint's quaternion block."""
    out = []
    for j in range(model.njnt):
        t = int(model.jnt_type[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            out.append(int(model.jnt_qposadr[j]) + 3)
        elif t == int(mujoco.mjtJoint.mjJNT_BALL):
            out.append(int(model.jnt_qposadr[j]))
    return out


def resample(Q, t_src, t_dst, quat_adr=()):
    """Rows of Q at times t_src -> times t_dst: linear for joints, normalized blend (sign-aligned) for quaternions."""
    Q = np.asarray(Q, float)
    t_src, t_dst = np.asarray(t_src, float), np.clip(np.asarray(t_dst, float), t_src[0], t_src[-1])
    i1 = np.clip(np.searchsorted(t_src, t_dst, side="right"), 1, len(t_src) - 1)
    i0 = i1 - 1
    w = ((t_dst - t_src[i0]) / np.maximum(t_src[i1] - t_src[i0], 1e-12))[:, None]
    out = (1 - w) * Q[i0] + w * Q[i1]
    for a in quat_adr:
        q0, q1 = Q[i0, a:a + 4], Q[i1, a:a + 4]
        sgn = np.where((q0 * q1).sum(axis=1, keepdims=True) < 0, -1.0, 1.0)
        q = (1 - w) * q0 + w * sgn * q1
        out[:, a:a + 4] = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return out


def qvel_of(spider_model, S, dt):
    V = np.zeros((len(S), spider_model.nv))
    for t in range(1, len(S)):
        mujoco.mj_differentiatePos(spider_model, V[t], float(dt), S[t - 1], S[t])
    V[0] = V[1] if len(S) > 1 else 0.0
    return V


def keypoints_from_spider_qpos(spider_model, S, sides):
    """SPIDER's keypoint arrays from SPIDER qpos rows (forward kinematics of the scene's own hands): per side the palm
    site pose [T, 7], the fingertip sites [T, 5, 7] (identity orientation) and the object pose [T, 7]."""
    d = mujoco.MjData(spider_model)
    T = len(S)
    out = {}
    sc = spider_columns(spider_model, sides)
    for s in SIDE_ORDER:
        out[f"qpos_wrist_{s}"] = np.tile([0, 0, 0, 1.0, 0, 0, 0], (T, 1))
        out[f"qpos_finger_{s}"] = np.tile([0, 0, 0, 1.0, 0, 0, 0], (T, 5, 1)).astype(float)
        out[f"qpos_obj_{s}"] = np.tile([0, 0, 0, 1.0, 0, 0, 0], (T, 1))
    for t in range(T):
        d.qpos[:] = S[t]
        mujoco.mj_kinematics(spider_model, d)
        for s in sides:
            ps = mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_palm")
            out[f"qpos_wrist_{s}"][t, :3] = d.site_xpos[ps]
            out[f"qpos_wrist_{s}"][t, 3:] = _mat_quat(d.site_xmat[ps])
            for i, f in enumerate(FINGERS):
                st = mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_{f}_tip")
                out[f"qpos_finger_{s}"][t, i, :3] = d.site_xpos[st]
            out[f"qpos_obj_{s}"][t] = S[t, sc[s]["object"]]
    return out


# ------------------------------------------------------------------------------------------ the human's hands --
def human_keypoints(kp_by_side, hand_xml_by_side, scale="auto"):
    """The demonstrator's hands as the Sharpa's keypoints. kp_by_side {side: {'wrist': [T, 3], finger: [T, 4, 3]}}
    (human_prior.keypoints_from_egodex / keypoints_from_hamer, rig frame). Each frame's anatomical hand frame is
    matched to the Sharpa's (human_prior.sharpa_reference), which gives the robot palm pose; the fingertips are scaled
    about the knuckle centre by the ratio of the Sharpa's to the human's mean finger length ('auto') so the human's
    reach is kept in the robot's proportions. Returns {side: {palm [T, 7], tips [T, 5, 3], root [T, 4, 4]}}."""
    from tool_library.common import human_prior as HP
    out = {}
    for side, kp in kp_by_side.items():
        ref = HP.sharpa_reference(hand_xml_by_side[side], side)
        m = mujoco.MjModel.from_xml_path(str(hand_xml_by_side[side]))
        dd = mujoco.MjData(m)
        mujoco.mj_forward(m, dd)
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_base_yaw")
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
        R0, p0 = dd.xmat[root].reshape(3, 3), dd.xpos[root]
        R_rp = R0.T @ dd.site_xmat[palm].reshape(3, 3)
        p_rp = R0.T @ (dd.site_xpos[palm] - p0)
        T = len(kp["wrist"])
        L_robot = np.mean([HP.chain_length(ref["kp"][f]) for f in FINGERS])
        palm_pose, tips, roots = np.zeros((T, 7)), np.zeros((T, 5, 3)), np.zeros((T, 4, 4))
        for t in range(T):
            k = HP.frame_at(kp, t)
            R_h, c_h = HP.anatomical_frame(k, side)
            R_root = R_h @ ref["R_anat"].T
            p_root = c_h - R_root @ ref["c_anat"]
            roots[t] = np.eye(4); roots[t][:3, :3] = R_root; roots[t][:3, 3] = p_root
            palm_pose[t, :3] = p_root + R_root @ p_rp
            palm_pose[t, 3:] = _mat_quat(R_root @ R_rp)
            s_ = float(scale) if scale != "auto" else L_robot / max(np.mean([HP.chain_length(k[f]) for f in FINGERS]), 1e-6)
            for i, f in enumerate(FINGERS):
                tips[t, i] = c_h + s_ * (np.asarray(k[f], float)[-1] - c_h)
        out[side] = {"palm": palm_pose, "tips": tips, "root": roots}
    return out


def keypoints_npz(T, sides, palm=None, tips=None, objects=None):
    """SPIDER's trajectory_keypoints arrays from per-side palm poses [T, 7], tips [T, 5, 3] and object poses [T, 7]."""
    out = {}
    for s in SIDE_ORDER:
        out[f"qpos_wrist_{s}"] = np.tile([0, 0, 0, 1.0, 0, 0, 0], (T, 1))
        out[f"qpos_finger_{s}"] = np.tile([0, 0, 0, 1.0, 0, 0, 0], (T, 5, 1)).astype(float)
        out[f"qpos_obj_{s}"] = np.tile([0, 0, 0, 1.0, 0, 0, 0], (T, 1))
    for s in sides:
        if palm and s in palm:
            out[f"qpos_wrist_{s}"] = np.asarray(palm[s], float)
        if tips and s in tips:
            out[f"qpos_finger_{s}"][:, :, :3] = np.asarray(tips[s], float)
        if objects and s in objects:
            out[f"qpos_obj_{s}"] = np.asarray(objects[s], float)
    return out


# ------------------------------------------------------------------------------------------------- preparation --
def object_surface(spider_model, side, n=4000, seed=0):
    """(points [n, 3], normals [n, 3]) sampled on the collision geometry of `<side>_object`, in its body frame."""
    import trimesh
    b = mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    parts = []
    for g in range(spider_model.ngeom):
        if int(spider_model.geom_bodyid[g]) != b or int(spider_model.geom_group[g]) != 3:
            continue
        gt, sz = int(spider_model.geom_type[g]), spider_model.geom_size[g]
        if gt == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(spider_model.geom_dataid[g])
            va, vn = int(spider_model.mesh_vertadr[mid]), int(spider_model.mesh_vertnum[mid])
            mesh = trimesh.Trimesh(spider_model.mesh_vert[va:va + vn].copy(), process=False).convex_hull
        elif gt == int(mujoco.mjtGeom.mjGEOM_BOX):
            mesh = trimesh.creation.box(extents=2 * sz)
        elif gt == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=sz[0])
        elif gt == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            mesh = trimesh.creation.cylinder(radius=sz[0], height=2 * sz[1], sections=32)
        elif gt == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            mesh = trimesh.creation.capsule(radius=sz[0], height=2 * sz[1])
        elif gt == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0); mesh.apply_scale(sz)
        else:
            continue
        Tm = np.eye(4)
        Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, spider_model.geom_quat[g])
        Tm[:3, :3] = Rg.reshape(3, 3); Tm[:3, 3] = spider_model.geom_pos[g]
        mesh.apply_transform(Tm)
        parts.append(mesh)
    if not parts:
        raise ValueError(f"{side}_object has no collision geometry to put fingertips on")
    mesh = trimesh.util.concatenate(parts)
    pts, fi = trimesh.sample.sample_surface(mesh, int(n), seed=seed)
    return np.asarray(pts, float), np.asarray(mesh.face_normals[fi], float)


def _pose_mat(q7):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q7[3:], float))
    return R.reshape(3, 3), np.asarray(q7[:3], float)


def support_height(spider_model, side, q7):
    """(lowest z of `<side>_object` at pose q7, the height of what lies under it: the floor or a fixed obstacle)."""
    d = mujoco.MjData(spider_model)
    sc = spider_columns(spider_model, [side])
    d.qpos[:] = spider_model.qpos0
    d.qpos[sc[side]["object"]] = q7
    mujoco.mj_forward(spider_model, d)
    b = mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
    pts, _ = object_surface(spider_model, side, n=1500)
    R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
    W = pts @ R.T + p
    zlow = float(W[:, 2].min())
    low = W[W[:, 2] < zlow + 0.004]
    floor = float(spider_model.geom_pos[mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")][2])
    top = floor
    gid = np.zeros(1, np.int32)
    gg = np.ones(6, np.uint8)                       # every geom group
    for x in low[:: max(1, len(low) // 12)]:
        dist = mujoco.mj_ray(spider_model, d, np.array([x[0], x[1], zlow + 0.002]), np.array([0, 0, -1.0]), gg, 1, b, gid)
        if dist >= 0:
            top = max(top, zlow + 0.002 - dist)
    return zlow, top


def contact_window(obj_pos, fps, move_tol_m=0.005, lead_s=0.6, lag_s=0.4):
    """[first, last] frames of the hand-object interaction: from lead_s before the object first leaves its start by
    move_tol_m to lag_s after it last moves; None when it never moves."""
    P_ = np.asarray(obj_pos, float)
    off = np.linalg.norm(P_ - P_[0], axis=1) > move_tol_m
    if not off.any():
        return None
    step = np.r_[0.0, np.linalg.norm(np.diff(P_, axis=0), axis=1)]
    moving = np.where(step > move_tol_m / 10.0)[0]
    a = int(np.argmax(off))
    b = int(moving.max()) if len(moving) else len(P_) - 1
    return [max(0, a - int(lead_s * fps)), min(len(P_) - 1, b + int(lag_s * fps))]


def prep_keypoints(kp, spider_model, sides, fps, contact_m=0.02, pad_m=None, ground=True, move_tol_m=0.005):
    """The mandatory preparation of SPIDER keypoints, in place on a copy (sample-agnostic: the object, the contact window
    and the contacts are measured, never given):
      ground   rigid z-shift of hands and objects so each manipulated object's lowest point starts on what lies under it
               (the floor, or a fixed obstacle); the shift is the largest over the hands' objects and is reported (dz);
      contacts in the window where the object moves (contact_window), a fingertip within contact_m of the object's
               surface is put ON it: the nearest surface point plus one pad radius along its normal (the tip site is on
               the pad's centre line; a tip target on the surface drives the finger links into the object);
      thumb    where a finger touches, the thumb is re-derived at the antipode of the finger-contact centroid through
               the object centre (nearest surface point to that ray, plus the pad radius).
    Returns (kp, report {dz, window, contact_frames, opposition_frames, per_finger})."""
    kp = {k: np.array(v, float, copy=True) for k, v in kp.items()}
    T = len(kp["qpos_wrist_right"])
    rep = {"dz": 0.0, "sides": {}}
    if ground:
        dzs = []
        for s in sides:
            zlow, top = support_height(spider_model, s, kp[f"qpos_obj_{s}"][0])
            if abs(top - zlow) < 0.10:                   # further off is not a grounding error but a wrong reference
                dzs.append(top - zlow)
        dz = max(dzs, key=abs) if dzs else 0.0
        for s in SIDE_ORDER:
            kp[f"qpos_wrist_{s}"][:, 2] += dz
            kp[f"qpos_finger_{s}"][:, :, 2] += dz
            kp[f"qpos_obj_{s}"][:, 2] += dz
        rep["dz"] = round(float(dz), 5)
    from scipy.spatial import cKDTree
    for s in sides:
        pad = pad_m
        if pad is None:
            g = mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_GEOM, f"collision_hand_{s}_index_0")
            pad = float(spider_model.geom_size[g][0]) if g >= 0 else 0.007
        pts, nrm = object_surface(spider_model, s)
        tree = cKDTree(pts)
        objp = kp[f"qpos_obj_{s}"]
        win = contact_window(objp[:, :3], fps, move_tol_m=move_tol_m)
        contact = np.zeros((T, 5))
        cpos = [[] for _ in FINGERS]
        opp = 0
        if win is not None:
            b = mujoco.mj_name2id(spider_model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_object")
            com = spider_model.body_ipos[b]
            for t in range(win[0], win[1] + 1):
                R, p = _pose_mat(objp[t])
                tips = kp[f"qpos_finger_{s}"][t, :, :3]
                local = (tips - p) @ R
                dist, idx = tree.query(local)
                surf = {}
                for i in range(5):
                    if dist[i] < contact_m:
                        surf[i] = idx[i]
                if not any(i > 0 for i in surf):
                    continue
                for i, k in surf.items():
                    if i == 0:
                        continue
                    contact[t, i] = 1
                    loc = pts[k] + pad * nrm[k]
                    kp[f"qpos_finger_{s}"][t, i, :3] = p + R @ loc
                    cpos[i].append(pts[k])
                c = np.mean([pts[surf[i]] for i in surf if i > 0], axis=0)
                u = com - c
                if np.linalg.norm(u) > 1e-6:
                    u /= np.linalg.norm(u)
                    rel = pts - com
                    along = rel @ u
                    perp = np.linalg.norm(rel - along[:, None] * u, axis=1)
                    cand = np.where(along > 0)[0]
                    if len(cand):
                        k = cand[np.argmin(perp[cand])]
                        kp[f"qpos_finger_{s}"][t, 0, :3] = p + R @ (pts[k] + pad * nrm[k])
                        contact[t, 0] = 1
                        cpos[0].append(pts[k])
                        opp += 1
        kp[f"contact_{s}"] = contact
        kp[f"contact_pos_{s}"] = np.array([np.mean(c, axis=0) if c else np.zeros(3) for c in cpos])
        rep["sides"][s] = {"window": win, "contact_frames": int((contact[:, 1:].sum(axis=1) > 0).sum()),
                           "opposition_frames": int(opp), "pad_m": pad,
                           "per_finger": {f: int(contact[:, i].sum()) for i, f in enumerate(FINGERS)}}
    for s in SIDE_ORDER:
        kp.setdefault(f"contact_{s}", np.zeros((T, 5)))
        kp.setdefault(f"contact_pos_{s}", np.zeros((5, 3)))
    return kp, rep


# ------------------------------------------------------------------------------------------------ SPIDER runs --
def dataset_paths(ds_dir, embodiment, task, data_id=0):
    base = Path(ds_dir) / "processed" / DATASET
    return {"assets": base / "assets", "robot_xml": base / "assets" / "robots" / ROBOT / robot_xml_name([embodiment] if embodiment != "bimanual" else SIDE_ORDER),
            "mano_task": base / "mano" / embodiment / task, "mano_trial": base / "mano" / embodiment / task / str(data_id),
            "robot_task": base / ROBOT / embodiment / task, "trial": base / ROBOT / embodiment / task / str(data_id)}


def embodiment_of(sides):
    sides = [s for s in SIDE_ORDER if s in sides]
    return "bimanual" if len(sides) == 2 else sides[0]


def check_spider(spider_dir, spider_python):
    """Raise an error naming what is missing when the SPIDER checkout or its interpreter is not usable."""
    if not spider_dir or not Path(spider_dir, "examples", "run_mjwp.py").exists():
        raise FileNotFoundError(f"{spider_dir!r} is not a SPIDER checkout (examples/run_mjwp.py missing); clone "
                                "github.com/facebookresearch/spider and set models.spider_dir in dexagent.yaml")
    if not spider_python or not Path(spider_python).exists():
        raise FileNotFoundError(f"SPIDER python {spider_python!r} not found; create its environment (mujoco, mujoco_warp, "
                                "warp-lang, torch with CUDA, hydra-core, loguru, tyro) and set environments.spider")


def _env(spider_dir):
    tmp = Path(tempfile.gettempdir())
    env = dict(os.environ)
    # SPIDER's own package first; this library's modules stay out of its interpreter
    env["PYTHONPATH"] = str(Path(spider_dir).resolve())
    # the compile caches go where TMPDIR points (warp, triton and torch.compile write to the home folder or /tmp otherwise)
    env.setdefault("WARP_CACHE_PATH", str(tmp / "warp_cache"))
    env.setdefault("TRITON_CACHE_DIR", str(tmp / "triton_cache"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(tmp / "torchinductor_cache"))
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"                   # its progress lines reach the log as they are printed
    return env


def _run(cmd, cwd, env, log, timeout):
    with open(log, "w") as fh:
        fh.write(" ".join(str(c) for c in cmd) + "\n\n"); fh.flush()
        r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=float(timeout))
    return r.returncode


def _tail(log, n=25):
    try:
        lines = Path(log).read_text(errors="replace").replace("\r", "\n").splitlines()
    except OSError:
        return ""
    return "\n".join(l for l in lines[-n:] if l.strip())


def run_spider_ik(spider_dir, spider_python, ds_dir, embodiment, task, ref_dt, sim_dt=0.005, log=None, timeout=3600):
    """SPIDER's keypoint IK (spider/preprocess/ik.py): mocap welds pull the palm, the fingertips and the object onto the
    keypoints. Writes <trial>/trajectory_kinematic.npz (two frames shorter: its moving-average filter). Raises when the IK
    diverged (MuJoCo's bad-acceleration reset leaves the hands at the origin: a diverging object weld did that with a
    nonzero object armature, see write_scene)."""
    paths = dataset_paths(ds_dir, embodiment, task)
    log = log or paths["trial"] / "ik.log"
    cmd = [spider_python, Path(spider_dir) / "spider" / "preprocess" / "ik.py", "--dataset-dir", Path(ds_dir).resolve(),
           "--dataset-name", DATASET, "--robot-type", ROBOT, "--embodiment-type", embodiment, "--task", task, "--data-id", "0",
           "--no-show-viewer", "--ref-dt", repr(float(ref_dt)), "--sim-dt", repr(float(sim_dt))]
    paths["trial"].mkdir(parents=True, exist_ok=True)
    code = _run(cmd, spider_dir, _env(spider_dir), log, timeout)
    out = paths["trial"] / "trajectory_kinematic.npz"
    if code != 0 or not out.exists():
        raise SpiderError(f"SPIDER IK failed (exit {code}):\n{_tail(log)}")
    q = np.load(out)["qpos"]
    sides = SIDE_ORDER if embodiment == "bimanual" else (embodiment,)
    for i, s in enumerate(sides):                     # each hand's wrist at the origin on most frames: the IK was reset
        w = q[:, 28 * i:28 * i + 3]
        if (np.linalg.norm(w, axis=1) < 0.02).mean() > 0.5:
            raise SpiderError(f"SPIDER IK diverged: the {s} wrist stayed at the origin (MuJoCo reset the simulation):\n{_tail(log)}")
    return str(out)


# run_mjwp.py settings for the bench's scenes: re-planned every 0.2 s; njmax_per_env 350 overflows with a Sharpa hand and a
# mesh object (>= 351 needed), and the initial-state sanity check fails even on SPIDER's own shipped example, so it is off.
# sim_dt is SPIDER's 10 ms here; physics_informed_retargeting runs it at the bench's own step (2 ms), because MuJoCo stiffens a
# contact no faster than twice the step and a grasp SPIDER held in the softer contact did not hold on the bench.
MJWP_DEFAULTS = {"sim_dt": 0.01, "ctrl_dt": 0.2, "improvement_threshold": 0.0, "improvement_check_steps": 2, "final_noise_scale": 0.01,
                 "pos_rew_scale": 1.0, "rot_rew_scale": 0.3,
                 "nconmax_per_env": 500, "njmax_per_env": 1200, "sanity_check_seconds": 0.0, "save_video": True, "save_info": True,
                 "show_viewer": False, "viewer": "", "rerun_spawn": False, "save_rerun": False, "save_metrics": False, "wait_on_finish": False}
# Sampling regimes, measured on a thumb-index pinch of a 30 g cup that the bench's physics executes from the plan:
#   refine   the agent's kinematic plan, which the physics nearly executes already: noise of 0.01-0.02 rad on the fingers and
#            1.5-3 mm on the wrist around it, the hand held to it (wrist weight 2, finger weight 0.3), a 1.2 s horizon so a
#            sample sees whether the object rises with the hand. The exploratory settings (0.2-0.4 rad, finger weight 0.003,
#            0.6 s horizon) let the fingers drift 0.3 rad off the pinch before it closed: the cup was never touched; with
#            0.05 rad and the 1.2 s horizon it was lifted and carried, then dropped above the bowl.
#   explore  the same with 0.025-0.05 rad and 2.5-5 mm, for a plan whose contacts need larger corrections;
#   robust   refine scored by its worst case over three perturbed copies of the scene (SPIDER's domain randomization:
#            every contact pair's margin -2..+2 mm, the object shifted -3..+3 mm), three times the cost. Meant for commands
#            that hold only in MuJoCo Warp's exact contact (the cup example's 2 ms refine solution, replayed open loop in CPU
#            MuJoCo in SPIDER's own scene, dropped the cup where the bench did); on that example it held less (1 of 3
#            subgoals against refine's 2), so it is not in the default list;
#   retarget the human's hands, far from a feasible grasp: the wide exploration of the earlier Sharpa runs (0.2-0.4 rad,
#            temperature 1, the wrist loosely held, 1024 samples, 16 iterations).
_HOLD = {"horizon": 1.2, "knot_dt": 0.2, "num_samples": 512, "max_num_iterations": 8, "temperature": 0.1,
         "first_ctrl_noise_scale": 0.5, "last_ctrl_noise_scale": 1.0, "base_pos_rew_scale": 2.0, "base_rot_rew_scale": 0.6, "joint_rew_scale": 0.3}
MJWP_PROFILES = {
    "refine": {**_HOLD, "joint_noise_scale": 0.02, "pos_noise_scale": 0.003, "rot_noise_scale": 0.003},
    "explore": {**_HOLD, "joint_noise_scale": 0.05, "pos_noise_scale": 0.005, "rot_noise_scale": 0.005},
    "robust": {**_HOLD, "joint_noise_scale": 0.02, "pos_noise_scale": 0.003, "rot_noise_scale": 0.003,
               "num_dr": 3, "pair_margin_range": [-0.002, 0.002], "xy_offset_range": [-0.003, 0.003]},
    "retarget": {"horizon": 0.6, "knot_dt": 0.1, "num_samples": 1024, "max_num_iterations": 16, "temperature": 1.0,
                 "first_ctrl_noise_scale": 2.0, "last_ctrl_noise_scale": 4.0, "joint_noise_scale": 0.1, "pos_noise_scale": 0.01,
                 "rot_noise_scale": 0.01, "base_pos_rew_scale": 0.1, "base_rot_rew_scale": 0.03, "joint_rew_scale": 0.003},
}


def mjwp_settings(profile="refine", overrides=None):
    """run_mjwp.py settings: the defaults, the profile ('refine' | 'explore' | 'robust' | 'retarget'), then the caller's overrides."""
    if profile not in MJWP_PROFILES:
        raise ValueError(f"profile must be one of {sorted(MJWP_PROFILES)}")
    return {**MJWP_DEFAULTS, **MJWP_PROFILES[profile], **(overrides or {})}


def _default_keys(spider_dir):
    import yaml
    keys = set()
    for f in ("default.yaml",):
        p = Path(spider_dir, "examples", "config", f)
        if p.exists():
            keys |= set((yaml.safe_load(p.read_text()) or {}).keys())
    return keys


def run_spider_mjwp(spider_dir, spider_python, ds_dir, embodiment, task, overrides=None, profile="refine", device="cuda:0", log=None,
                    timeout=7200):
    """SPIDER's physics optimization (examples/run_mjwp.py: DIAL-MPC sampling in MuJoCo Warp) on <trial>/trajectory_kinematic.npz
    with mjwp_settings(profile, overrides). Returns the path of <trial>/trajectory_mjwp.npz."""
    paths = dataset_paths(ds_dir, embodiment, task)
    log = log or paths["trial"] / "mjwp.log"
    cfg = mjwp_settings(profile, overrides)
    known = _default_keys(spider_dir)
    args = [f"dataset_dir={Path(ds_dir).resolve()}", f"dataset_name={DATASET}", f"robot_type={ROBOT}", f"embodiment_type={embodiment}",
            f"task={task}", "data_id=0", f"device={device}"]
    for k, v in cfg.items():
        if isinstance(v, bool):
            val = "true" if v else "false"
        elif isinstance(v, str):
            val = f"'{v}'"
        else:
            val = repr(v)
        args.append(f"{'' if k in known else '+'}{k}={val}")
    cmd = [spider_python, Path(spider_dir) / "examples" / "run_mjwp.py"] + args
    out = paths["trial"] / "trajectory_mjwp.npz"
    if out.exists():
        out.unlink()
    code = _run(cmd, spider_dir, _env(spider_dir), log, timeout)
    if code != 0 or not out.exists():
        raise SpiderError(f"SPIDER run_mjwp failed (exit {code}):\n{_tail(log)}")
    return str(out)


def read_mjwp(path, sim_dt):
    """{qpos [N, nq], ctrl [N, nu], t [N]} of SPIDER's executed trajectory (its per-tick records flattened)."""
    z = np.load(str(path), allow_pickle=True)
    Q = np.asarray(z["qpos"], float); C = np.asarray(z["ctrl"], float)
    Q = Q.reshape(-1, Q.shape[-1]); C = C.reshape(-1, C.shape[-1])
    if "time" in z.files:
        t = np.asarray(z["time"], float).reshape(-1)
    else:
        t = (np.arange(len(Q)) + 1) * float(sim_dt)
    return {"qpos": Q, "ctrl": C, "t": t}


def force_equivalent_command(q_exec, ctrl, kp_spider, kp_bench):
    """The bench command that makes a servo of stiffness kp_bench push as hard as SPIDER's did: SPIDER's position servo
    pushed kp_spider * (ctrl - q); the bench servo pushes the same at q + kp_spider / kp_bench * (ctrl - q). With equal
    stiffness (write_robot_xml gives SPIDER the bench's servos) this is SPIDER's own command; a softer SPIDER wrist would
    give the executed wrist plus that fraction of its servo lead."""
    r = np.asarray(kp_spider, float) / np.maximum(np.asarray(kp_bench, float), 1e-9)
    return np.asarray(q_exec, float) + r * (np.asarray(ctrl, float) - np.asarray(q_exec, float))


def bench_command_from_spider(model, Qbase, spider_model, layout, res, t_frames, dz=0.0, wrist_kp_bench=BENCH_WRIST_KP, finger_kp_bench=None):
    """Bench reference rows at times t_frames from SPIDER's result: each acting hand's columns are the force-equivalent
    commands (wrist and fingers), the held objects SPIDER's executed poses (a record: the bench's rollout ignores
    object columns). Qbase supplies every other column (the idle hand, the arms, the obstacles)."""
    S = resample(res["qpos"], res["t"], t_frames, quat_adr=quat_columns(spider_model))
    C = resample(res["ctrl"], res["t"], t_frames)
    sc = spider_columns(spider_model, layout["sides"])
    cmd = S.copy()
    for s in layout["sides"]:
        for adr in sc[s]["wrist"] + sc[s]["fingers"]:
            j = [jj for jj in range(spider_model.njnt) if int(spider_model.jnt_qposadr[jj]) == adr][0]
            a = [aa for aa in range(spider_model.nu) if int(spider_model.actuator_trnid[aa][0]) == j][0]
            kp_s = float(spider_model.actuator_gainprm[a][0])
            is_wrist = adr in sc[s]["wrist"]
            kp_b = float(wrist_kp_bench) if is_wrist else float(finger_kp_bench or kp_s)
            cmd[:, adr] = force_equivalent_command(S[:, adr], C[:, a], kp_s, kp_b)
    out = spider_to_bench(model, Qbase, spider_model, layout, cmd, dz=dz)
    executed = spider_to_bench(model, Qbase, spider_model, layout, S, dz=dz)
    return out, executed


# ------------------------------------------------------------------------------------- expectations from Stage 1 --
def stage1_expect(sg1, model, Q, frames, side, obj, free_bodies=None):
    """The expectation of one Stage 1 subgoal, built from its motion and object refs, with the target numbers the
    reference reached (where the object ends): grasp_lift -> risen (half the reference's rise, 2-5 cm) and held without
    slip; hold -> held; carry -> at the reference's end position, held; place -> at its end position (x, y), released,
    at rest, supported, upright when it started upright; release -> released, at rest; push / pull / slide -> at its
    end position (x, y). Objects Stage 1 does not name are expected untouched. Returns None for a motion with no such
    expectation (joint motions, screwing, cutting...). tracking.write_plan uses it for every method module's plan."""
    motion = str(sg1.get("motion", ""))
    t0, t1 = int(frames[0]), int(frames[-1])
    pos, quat = body_track(model, np.asarray(Q)[[t0, t1]], obj)
    free = set(free_bodies if free_bodies is not None else E.object_free_joints(model))
    refs = set(sg1.get("object_refs") or [])
    named = {b for b in free if b == obj or any(b.startswith(f"{r}_") or b == r for r in refs)}
    e = {"hand": side, "object": obj, "involved": sorted(named | {obj}), "untouched": sorted(free - named - {obj})}
    R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, quat[0])
    upright0 = float(np.degrees(np.arccos(np.clip(R0.reshape(3, 3)[2, 2], -1, 1)))) < 30.0
    if motion == "grasp_lift":
        P_, _ = body_track(model, np.asarray(Q)[t0:t1 + 1], obj)
        rise = float(P_[:, 2].max() - P_[0, 2])
        e.update(rise_m=round(float(np.clip(0.5 * rise, 0.02, 0.05)), 3), in_hand=True, no_slip=True)
    elif motion == "hold":
        e.update(in_hand=True, no_slip=True)
    elif motion == "carry":
        e.update(target_pos=pos[1].round(5).tolist(), target_axes="xyz", in_hand=True, no_slip=True)
    elif motion == "place":
        e.update(target_pos=pos[1].round(5).tolist(), target_axes="xy", released=True, at_rest=True, supported=True)
        if upright0:
            e["upright"] = True
    elif motion == "release":
        e.update(released=True, at_rest=True)
    elif motion in ("push", "pull", "slide"):
        e.update(target_pos=pos[1].round(5).tolist(), target_axes="xy")
    else:
        return None
    return e
