"""Carry the executed state of an object's internal joints (cable, hinge, slide) into a floating-hand scene."""
from __future__ import annotations

TOOL = {
    "name": "hand_scene_object_state",
    "title": "Hand scene with the object's executed joint state",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Copy the current values of an object's joints (the root's free joint, the ball joints of a cable, a lid hinge, a drawer slide) "
               "from a frame of an executed trajectory into a floating-hand scene's joint_init, so grasp solvers and stress "
               "tests see the object in its present shape instead of its built default.",
    "description": (
        "A floating-hand scene dict places each object by its root pose only; every internal joint starts at the asset's "
        "default. For a rigid object that is enough, but a rope whose root is placed while its cable joints are at zero is "
        "a straight cable, and an articulated box has its lid shut, so a pinch solved or stress-tested on any part away "
        "from the root is solved against geometry that is not there. This skill opens the bench scene the trajectory was "
        "executed in, takes the qpos at the requested frame, collects every joint whose name starts with one of the given "
        "prefixes (free joints as 7 values, so a rope's root lands where the bench has it; ball joints as 4 quaternion values, "
        "hinge and slide as 1; include_free=false leaves the free joints to the hand scene's own placement), and writes them "
        "into the hand scene's joint_init under the same names (the hand scene compiles objects with the bench's joint "
        "names). With the root pose already set from the same state, the object then has its executed shape. Optionally "
        "it also replaces the scene's grasp_body. Named recorded coordinates are mapped into the bench model. If the hand "
        "scene has bench-state provenance, that snapshot is updated too; changes to uncopied object joints are rejected "
        "so obstacles cannot silently retain stale states. Rebuild hand_scene_from_recon to update every object. "
        "It returns the updated dict and, when asked, writes it as JSON. It does "
        "not compile the hand scene; the grasp tool that consumes it does."
    ),
    "inputs": [
        {"name": "hand_scene", "type": "dict | str", "description": "Floating-hand scene dict or JSON path (hand_xml, side, objects, joint_init, ...)."},
        {"name": "scene", "type": "str", "description": "Bench MJCF; named trajectory coordinates are mapped by identity, unnamed trajectories must match its nq."},
        {"name": "trajectory", "type": "str", "description": "npz with qpos [T, nq]."},
        {"name": "prefixes", "type": "list[str]", "description": "Joint-name prefixes of the object(s) whose joints are copied (e.g. the object's id + '_')."},
        {"name": "frame", "type": "int", "description": "Frame of the trajectory (default -1, the last)."},
        {"name": "grasp_body", "type": "str | None", "description": "Replace the scene's grasp_body with this body (optional)."},
        {"name": "out_json", "type": "str | None", "description": "Write the updated scene here (optional)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "hand_scene (the updated dict), n_joints copied, joints (names), frame, out_json."},
    ],
    "status": "tested",
    "entrypoint": "hand_scene_object_state",
    "tags": ["hand scene", "deformable", "rope", "articulated", "state", "grasp"],
}

import json
from pathlib import Path

from tool_library.common import episode as E
from tool_library.skills.hand_scene_from_recon import resolve_bench_state, record_bench_state, _changed_object_joints


def hand_scene_object_state(hand_scene, scene, trajectory, prefixes, frame=-1, grasp_body=None, out_json=None, include_free=True):
    import mujoco
    if isinstance(hand_scene, (str, Path)):
        hand_scene = json.loads(Path(hand_scene).read_text())
    if not isinstance(hand_scene, dict) or "objects" not in hand_scene:
        raise ValueError("hand_scene: expected a floating-hand scene dict (or JSON path) with 'objects'")
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    if not prefixes:
        raise ValueError("prefixes: give at least one joint-name prefix")
    model = mujoco.MjModel.from_xml_path(str(scene))
    try:
        q = E.state_qpos(model, trajectory, int(frame))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"trajectory: {exc}") from exc
    width = {int(mujoco.mjtJoint.mjJNT_BALL): 4, int(mujoco.mjtJoint.mjJNT_HINGE): 1, int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_FREE): 7}
    out = dict(hand_scene)
    ji = dict(out.get("joint_init") or {})
    names = []
    for j in range(model.njnt):
        n = model.joint(j).name
        t = int(model.jnt_type[j])            # enum members and numpy ints compare differently across MuJoCo versions
        # the free joints too (a cable's first segment, a second free body): the ball joints alone place the rope's
        # shape around wherever the hand scene put its root, which was 0.5 m off the bench after a carry
        if not n or (t == int(mujoco.mjtJoint.mjJNT_FREE) and not include_free) or not any(n.startswith(p) for p in prefixes):
            continue
        a = int(model.jnt_qposadr[j])
        k = width[t]
        v = q[a:a + k].tolist()
        ji[n] = v if k > 1 else v[0]
        names.append(n)
    if not names:
        raise ValueError(f"prefixes: no joint in the scene starts with {prefixes}")
    out["joint_init"] = ji
    if hand_scene.get("bench_state"):
        original = resolve_bench_state(hand_scene, model)
        changed = _changed_object_joints(model, original, q, ignore_joints=names)
        if changed:
            raise ValueError("trajectory changes object joints outside the copied prefixes: " + ", ".join(changed[:5])
                             + "; rebuild hand_scene_from_recon from this trajectory to update all objects")
        out["bench_state"] = record_bench_state(out, model, q, Path(trajectory).resolve(), int(frame))
    if grasp_body:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, grasp_body) < 0:
            raise ValueError(f"grasp_body: no body '{grasp_body}' in the scene")
        out["grasp_body"] = grasp_body
    res = {"hand_scene": out, "n_joints": len(names), "joints": names, "frame": int(frame)}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=1))
        res["out_json"] = str(out_json)
    return res
