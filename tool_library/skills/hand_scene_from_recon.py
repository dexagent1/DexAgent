"""Hand scene from a Stage 2 reconstruction: the floating-hand scene every grasp skill and stress test takes,
with the objects where the bench has them now."""
from __future__ import annotations

TOOL = {
    "name": "hand_scene_from_recon",
    "title": "Hand scene from reconstruction",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Build the floating-hand scene dict the grasp skills and stress tests take (hand_xml, objects, grasp_body, "
               "joint_init) from a Stage 2 scene_recon.json, with every object at its current bench pose.",
    "description": (
        "Grasp poses are solved on a scene with one floating hand and the objects on a floor at z=0; the bench's "
        "world frame is that same frame (table top at z=0, hand wrist joints equal to the world pose), so a pose "
        "solved here is used on the bench unchanged. The skill reads the manifest's objects (asset XML, root body), "
        "takes each root's pose from the bench scene's keyframe or from a given state (a trajectory frame, so a "
        "later subgoal is planned with the objects where the previous one left them), carries the objects' internal "
        "joint values over as joint_init, puts the target object last and names the grasped part as grasp_body (a "
        "cap, a drawer front, a lid), and picks the hand MJCF of the requested side from the setup folder. Returns the "
        "scene dict, also written to out_json for the next tool calls. It records the exact named bench_state so grasp "
        "and arm screens use the same object configuration by default; this is input provenance, not physical success evidence."
    ),
    "inputs": [
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 scene_recon.json (paths inside are relative to its folder)."},
        {"name": "side", "type": "str", "description": "'left' or 'right'."},
        {"name": "target", "type": "str", "description": "Object id from the manifest, or a body name in the bench scene (a part), that the hand will act on."},
        {"name": "grasp_body", "type": "str | None", "description": "Body of the part to grasp when it is not the object's root (default: the target's root, or the target itself when it names a part)."},
        {"name": "state", "type": "str | list | None", "description": "A trajectory.npz (last frame, or frame given by `frame`) or a qpos list giving the objects' current poses; default the scene's keyframe 0."},
        {"name": "frame", "type": "int", "description": "Frame of the state npz to use. Default -1 (last)."},
        {"name": "hand_xml", "type": "str | None", "description": "Hand MJCF; default <setup>/robot/sharpa_<side>.xml."},
        {"name": "setup_dir", "type": "str | None", "description": "Setup folder (default the repository's setup/)."},
        {"name": "include", "type": "list[str] | None", "description": "Object ids to include as obstacles (default all)."},
        {"name": "out_json", "type": "str | None", "description": "Write the scene dict here."},
    ],
    "outputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects:[{name, xml, pos, quat}], floor, grasp_body, grasp_body_pose (the part's pose the poses are solved against; the planner re-anchors a pose to where the part is at execution), joint_init, bench_scene, target, held_in_air (objects a hand holds in the air at this state -- above the table, touching nothing else, a hand's collision geometry within 3 mm -- marked hold: they keep their pose in the hand scene instead of falling while it settles, and the grasp stress tests treat them as a handover: held where they are until the tested hand has closed, then let go)}; pass it to generate_rigid_grasps, pose_from_points, handle_pinch_pose, push_pose_from_point, the grasp tests."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "grasp", "scene", "hand"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl"],
    "status": "tested",
    "entrypoint": "hand_scene_from_recon",
}

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import episode as E


def _scene_state_signature(hand_scene):
    fields = {"objects": [{key: obj.get(key) for key in ("name", "xml", "pos", "quat")}
                          for obj in hand_scene.get("objects", [])],
              "joint_init": hand_scene.get("joint_init", {})}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def record_bench_state(hand_scene, model, qpos, source, frame=None):
    """Record coordinate provenance and detect later edits to the extracted object state."""
    return {"schema": "dexagent.grasp_bench_state.v1", "joint_names": E.joint_names(model),
            "qpos": np.asarray(qpos, float).tolist(), "source": str(source), "frame": frame,
            "hand_scene_signature": _scene_state_signature(hand_scene)}


def _changed_object_joints(model, a, b, ignore_joints=()):
    changed = []
    for j in range(model.njnt):
        body = model.body(int(model.jnt_bodyid[j])).name
        if body.startswith("robot_") or model.joint(j).name in ignore_joints:
            continue
        start = int(model.jnt_qposadr[j])
        stop = int(model.jnt_qposadr[j + 1]) if j + 1 < model.njnt else model.nq
        x, y = a[start:stop], b[start:stop]
        same = np.allclose(x, y, atol=1e-7, rtol=0)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_BALL):
            same = same or np.allclose(x, -y, atol=1e-7, rtol=0)
        elif int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            same = np.allclose(x[:3], y[:3], atol=1e-7, rtol=0) and (
                np.allclose(x[3:], y[3:], atol=1e-7, rtol=0) or np.allclose(x[3:], -y[3:], atol=1e-7, rtol=0))
        if not same:
            changed.append(body)
    return changed


def resolve_bench_state(hand_scene, model, state=None):
    """Use the bench state from which a hand scene was extracted.

    This state is input provenance, not evidence that a motion was executed.
    An explicit state may update robot coordinates, but changing objects while
    retaining an old hand scene would test contact and reach in different worlds.
    """
    if isinstance(hand_scene, (str, Path)):
        hand_scene = json.loads(Path(hand_scene).read_text())
    saved = (hand_scene or {}).get("bench_state")
    if not saved:
        return None if state is None else E.state_qpos(model, state)
    if saved.get("hand_scene_signature") and saved["hand_scene_signature"] != _scene_state_signature(hand_scene):
        raise ValueError("hand scene object placement or joint_init changed after bench_state was recorded; "
                         "rebuild hand_scene_from_recon from the intended state")
    names = saved.get("joint_names", [])
    values = np.asarray(saved.get("qpos", []), float)
    target = E.joint_names(model)
    if (values.shape != (len(names),) or not np.isfinite(values).all()
            or len(set(names)) != len(names) or set(names) != set(target)):
        raise ValueError("hand scene bench_state does not match the bench's named coordinates; rebuild hand_scene_from_recon")
    named = dict(zip(names, values))
    q = np.asarray([named[name] for name in target], float)
    if state is None:
        return q
    explicit = E.state_qpos(model, state)
    changed = _changed_object_joints(model, q, explicit)
    if changed:
        raise ValueError("hand scene and requested bench state have different object states for "
                         + ", ".join(changed[:5])
                         + "; rebuild hand_scene_from_recon with the same state before grasp/reach search")
    return explicit


def _asset_root_name(asset_xml):
    """Name of the asset's first body under its worldbody (its root)."""
    try:
        m = mujoco.MjModel.from_xml_path(str(asset_xml))
        for b in range(1, m.nbody):
            if int(m.body_parentid[b]) == 0:
                return m.body(b).name
    except Exception:
        return None
    return None


def _root_local(asset_xml, root_name):
    """Pose of the asset's root body inside its own XML (frame -> body)."""
    m = mujoco.MjModel.from_xml_path(str(asset_xml))
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if b < 0:
        b = 1
    return m.body_pos[b].copy(), m.body_quat[b].copy()


def _held_in_air(model, data, bodies, grip_m=0.003):
    """A free object that touches nothing but the robot, hangs above the table and has a robot hand on it (a hand
    collision geom within grip_m: the bench scene's hands collide only through the executor's pairs, so they never show
    up as contacts here): something holds it right now. A part merely staged a hair above its support (a brick dropped
    0.5 mm over the studs it will perch on) touches nothing either, but no hand holds it: it is left to settle."""
    bodies = set(bodies)
    if not any(int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and int(model.jnt_bodyid[j]) in bodies for j in range(model.njnt)):
        return False
    from tool_library.common import hand as H
    hand = [g for g in range(model.ngeom) if "_collision_hand_" in (model.geom(g).name or "")]
    part = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]
    near = False
    for h in hand:
        for g in part:
            if np.linalg.norm(data.geom_xpos[h] - data.geom_xpos[g]) > model.geom_rbound[h] + model.geom_rbound[g] + grip_m:
                continue
            if H.geom_distance(model, data, h, g, grip_m + 0.001) <= grip_m:
                near = True
                break
        if near:
            break
    if not near:
        return False
    lo = min((float(data.geom_xpos[g][2] - model.geom_rbound[g]) for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies),
             default=0.0)
    if lo < 0.01:
        return False
    for c in data.contact[:data.ncon]:
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if (b1 in bodies) != (b2 in bodies):
            other = b2 if b1 in bodies else b1
            if not (model.body(other).name or "").startswith("robot_"):
                return False                           # resting on something
    return True


def hand_scene_from_recon(scene_recon_json, side, target, grasp_body=None, state=None, frame=-1, hand_xml=None,
                          setup_dir=None, include=None, out_json=None):
    man_p = Path(scene_recon_json).resolve()
    ws = man_p.parent
    man = json.loads(man_p.read_text())
    scene_xml = (ws / man.get("scene_xml", "scene.xml")).resolve()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    if state is None:
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
    elif isinstance(state, (str, Path)):
        data.qpos[:] = E.state_qpos(model, state, frame)
    else:
        data.qpos[:] = np.asarray(state, float)
    mujoco.mj_forward(model, data)
    repo = Path(__file__).resolve().parents[2]
    setup = Path(setup_dir) if setup_dir else repo / "setup"
    hand_xml = str(Path(hand_xml).resolve()) if hand_xml else str(setup / "robot" / f"sharpa_{side}.xml")
    if not Path(hand_xml).exists():
        raise FileNotFoundError(f"hand MJCF not found: {hand_xml}")
    objs = man.get("objects", [])
    by_id = {o["id"]: o for o in objs}
    # which manifest object owns the target (an id, a root body, or any body of an asset)
    tgt_obj, part = None, None
    if target in by_id:
        tgt_obj = by_id[target]
    else:
        for o in objs:
            if target == o.get("root_body") or target in (o.get("bodies") or []) or target in (o.get("parts") or []):
                tgt_obj, part = o, target
                break
        if tgt_obj is None:
            for o in objs:
                if target.startswith(f"{o['id']}_"):
                    tgt_obj, part = o, target
                    break
    if tgt_obj is None:
        raise ValueError(f"target {target!r} is neither an object id nor a body of the manifest ({[o['id'] for o in objs]})")
    grasp_body = grasp_body or part or tgt_obj.get("root_body")
    # objects sharing an asset (a cap listed separately from its bottle) are one attachment
    seen, ordered = set(), []
    others = [o for o in objs if o is not tgt_obj and (include is None or o["id"] in include)]
    for o in others + [tgt_obj]:
        ax = str((ws / o["asset_xml"]).resolve())
        if ax in seen or o.get("type") == "deformable" and not Path(ax).exists():
            continue
        seen.add(ax); ordered.append(o)
    if str((ws / tgt_obj["asset_xml"]).resolve()) not in {str((ws / o["asset_xml"]).resolve()) for o in ordered}:
        ordered.append(tgt_obj)
    # the asset attachment prefix on the bench: root_body = <prefix><asset root>; the prefix is the id used by stage_on_bench
    out_objs, joint_init, held_in_air = [], {}, []
    for o in ordered:
        root = o.get("root_body") or ""
        ax = (ws / o["asset_xml"]).resolve()
        # the attachment prefix: root_body = <prefix>_<a body of the asset>. The asset's own body names decide it
        # (an asset staged as 'meal_box' whose dish is 'mealbox_body' gives 'meal_box', not the head before the first '_');
        # a manifest id that heads the root body is the fallback
        prefix = None
        try:
            am = mujoco.MjModel.from_xml_path(str(ax))
            for b in range(1, am.nbody):
                bn = am.body(b).name
                if bn and root.endswith("_" + bn):
                    cand = root[: -len(bn) - 1]
                    if prefix is None or len(cand) < len(prefix):
                        prefix = cand
        except Exception:
            prefix = None
        if prefix is None:
            prefix = root.split("_")[0] if root else o["id"]
            for cand in sorted({o2["id"] for o2 in objs}, key=len, reverse=True):
                if root.startswith(cand + "_"):
                    prefix = cand
                    break
        rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
        if rb < 0:
            raise ValueError(f"root body {root!r} of {o['id']} is not in {scene_xml.name}")
        # the asset's own root may not be the manifest's root body (a carrier body above it, or a part listed as its own
        # object, such as a cap that is a free body welded onto its bottle): place the asset by ITS root in the bench
        top = rb
        asset_root = _asset_root_name(ax)
        cand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_{asset_root}") if asset_root else -1
        if cand >= 0:
            top = cand
        else:
            while int(model.body_parentid[top]) != 0 and model.body(int(model.body_parentid[top])).name.startswith(prefix + "_"):
                top = int(model.body_parentid[top])
        pw, Rw = data.xpos[top].copy(), data.xmat[top].reshape(3, 3).copy()
        local_name = model.body(top).name[len(prefix) + 1:]
        pl, ql = _root_local(ax, local_name)
        Rl = Rot.from_quat([ql[1], ql[2], ql[3], ql[0]]).as_matrix()
        Rf = Rw @ Rl.T; pf = pw - Rf @ pl
        qf = Rot.from_matrix(Rf).as_quat()
        entry = {"name": prefix, "xml": str(ax), "pos": pf.round(6).tolist(), "quat": [float(qf[3]), float(qf[0]), float(qf[1]), float(qf[2])]}
        if _held_in_air(model, data, E.subtree_bodies(model, top)):
            entry["hold"] = True                       # held by a hand at this moment: the hand scene keeps it there
            held_in_air.append(o["id"])
        out_objs.append(entry)
        # every joint the asset owns on the bench: the part's hinges and slides, a cable's ball joints, and the free poses
        # of its other free bodies (a lid that is a second free body of the same asset, a cap welded onto its bottle)
        owned = {b for b in range(model.nbody) if model.body(b).name.startswith(prefix + "_")} | E.subtree_bodies(model, top)
        for j in range(model.njnt):
            jb = int(model.jnt_bodyid[j])
            if jb in owned and int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                joint_init[model.joint(j).name] = float(data.qpos[int(model.jnt_qposadr[j])])
            elif jb in owned and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE) and jb != top:
                a = int(model.jnt_qposadr[j]); joint_init[model.joint(j).name] = [float(x) for x in data.qpos[a:a + 7]]
            elif jb in owned and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_BALL):
                # a cable's link joints: the rope lies as the bench has it, not straight as its asset was built
                a = int(model.jnt_qposadr[j]); joint_init[model.joint(j).name] = [float(x) for x in data.qpos[a:a + 4]]
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, grasp_body)
    anchor = None
    if gb >= 0:
        q = Rot.from_matrix(data.xmat[gb].reshape(3, 3)).as_quat()
        anchor = {"body": grasp_body, "pos": data.xpos[gb].round(6).tolist(), "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])]}
    scene = {"hand_xml": hand_xml, "side": side, "objects": out_objs, "floor": True, "grasp_body": grasp_body,
             "joint_init": joint_init, "bench_scene": str(scene_xml), "target": tgt_obj["id"], "target_root": tgt_obj.get("root_body"),
             "grasp_body_pose": anchor, "held_in_air": held_in_air}
    scene["bench_state"] = record_bench_state(scene, model, data.qpos,
        str(Path(state).resolve()) if isinstance(state, (str, Path)) else ("scene_keyframe" if state is None else "supplied_qpos"),
        int(frame) if isinstance(state, (str, Path)) else None)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(scene, indent=1))
        scene["out_json"] = str(out_json)
    return scene
