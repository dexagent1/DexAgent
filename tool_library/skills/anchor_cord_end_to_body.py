"""Anchor one end of a cord (a chain of link bodies) to a fixed object's body in a scene: an active weld whose relative
pose is read off the scene's keyframe, so the initial state already satisfies it.
"""
from __future__ import annotations

TOOL = {
    "name": "anchor_cord_end_to_body",
    "title": "Anchor a cord end to a fixed object",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Welds one end link of a cord (rope, thread, cable) to the body of a FIXED scene object (a tape dispenser, a clamp, a post) with an active weld whose relative pose is taken from keyframe 0, refusing the world, the table, the robot and mocap bodies.",
    "description": (
        "Loads the scene, resets it to keyframe `key` (or qpos0 when it has none), and reads the pose of cord_body "
        "relative to anchor_body. It refuses an anchor that is the world body, a mocap body, a body that can move "
        "(any joint in its chain to the world), a robot body (name starting with robot_prefix) or a table/floor "
        "(name containing one of forbidden_words): only a fixed object's own body can hold a cord end. The cord body "
        "must be able to move. It then adds <weld body1=anchor_body body2=cord_body relpose=...> (relpose = the cord link's "
        "pose in the anchor body's frame at the keyframe, so it is never taken from qpos0; the weld point is body2's "
        "origin, i.e. the cord end itself, so the light link cannot swing about the fixed object's distant origin) to the MJCF's <equality> "
        "section, stiff by default (solref [2 timestep, 1], solimp 0.99/0.999 over 0.1 mm, so a light end link gives "
        "only a fraction of a millimetre under a hand's pull), writes out_xml and re-compiles it: the equality's "
        "residual at the keyframe must be below 1e-6 m (and 1e-4 rad for a weld), otherwise it raises. Keyframes are "
        "untouched (an equality adds no state). Anchor the cable's ROOT link (the body carrying its free joint) when "
        "possible: a weld on a link deep in a nested MuJoCo cable acts through every joint above it and gives "
        "millimetres under a pull where one on the root gives microns; the result lists that as a warning. Assumes the scene is one MJCF file (as stage_on_bench writes it)."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "path", "description": "The scene MJCF holding both bodies."},
        {"name": "out_xml", "type": "path", "description": "Where to write the anchored scene (may be scene_xml itself)."},
        {"name": "cord_body", "type": "str", "description": "The cord's end link body to anchor."},
        {"name": "anchor_body", "type": "str", "description": "The fixed object's body that holds it."},
        {"name": "name", "type": "str | None", "description": "Equality name; default '<cord_body>_anchor'."},
        {"name": "key", "type": "int", "description": "Keyframe whose state defines the relative pose. Default 0"},
        {"name": "solref", "type": "list[2] | None", "description": "Equality solref; None gives [2 x timestep, 1]. Default None"},
        {"name": "solimp", "type": "list[5]", "description": "Equality solimp. Default [0.99, 0.999, 0.0001, 0.5, 2]"},
        {"name": "robot_prefix", "type": "str", "description": "Body-name prefix of the robot (refused as an anchor). Default robot_"},
        {"name": "forbidden_words", "type": "list[str]", "description": "Anchor names containing any of these are refused (the table, the floor). Default ['table', 'floor', 'ground']"},
    ],
    "outputs": [
        {"name": "anchored", "type": "dict", "description": "out_xml, equality (name), kind ('weld'), cord_body, anchor_body, relpose [x, y, z, qw, qx, qy, qz] (the cord link's pose in the anchor body's frame), residual_m, residual_rad, key, warnings (e.g. the anchored link is not the cable's root)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["rope", "cord", "thread", "cable", "anchor", "weld", "equality", "clamp", "deformable", "scene"],
    "used_in": ["EgoDex thread_unthread_bead_necklace"],
    "status": "tested",
    "entrypoint": "anchor_cord_end_to_body",
    "cli": "python -m tool_library.skills.anchor_cord_end_to_body scene.xml out.xml cord_body anchor_body",
}

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

_ALLOWED = {"name", "key", "solref", "solimp", "robot_prefix", "forbidden_words"}


def _fmt(v):
    return " ".join(f"{float(x):.9g}" for x in v)


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _state(model, key):
    data = mujoco.MjData(model)
    if model.nkey > key:
        mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    return data


def _residual(model, data, eq_id):
    """Largest positional and angular violation of one equality at the current state."""
    pos, ang = 0.0, 0.0
    rows =[i for i in range(data.nefc) if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and data.efc_id[i] == eq_id]
    vals = np.abs(data.efc_pos[rows]) if rows else np.zeros(0)
    if len(vals):
        pos = float(vals[:3].max())
        ang = float(vals[3:].max()) if len(vals) > 3 else 0.0
    return pos, ang, len(rows)


def anchor_cord_end_to_body(scene_xml, out_xml, cord_body, anchor_body, **kw):
    unknown = sorted(set(kw) - _ALLOWED)
    if unknown:
        raise ValueError(f"anchor_cord_end_to_body: unknown parameter(s) {unknown}; allowed: {sorted(_ALLOWED)}")
    kind = "weld"
    key = int(kw.get("key", 0))
    prefix = str(kw.get("robot_prefix", "robot_"))
    words = [w.lower() for w in kw.get("forbidden_words", ["table", "floor", "ground"])]
    scene_xml, out_xml = Path(scene_xml), Path(out_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cord_body)
    ab = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body)
    if cb < 0:
        raise ValueError(f"anchor_cord_end_to_body: cord_body {cord_body!r} is not a body of the scene")
    if ab < 0:
        raise ValueError(f"anchor_cord_end_to_body: anchor_body {anchor_body!r} is not a body of the scene")
    if ab == 0:
        raise ValueError("anchor_cord_end_to_body: anchor_body is the world; anchor to a fixed object's body instead")
    if model.body_mocapid[ab] >= 0:
        raise ValueError(f"anchor_cord_end_to_body: anchor_body {anchor_body!r} is a mocap body; anchor to a fixed object")
    if model.body_weldid[ab] != 0:
        raise ValueError(f"anchor_cord_end_to_body: anchor_body {anchor_body!r} can move (it has a joint in its chain); "
                         "only a fixed object can anchor a cord end")
    if anchor_body.startswith(prefix) or any(w in anchor_body.lower() for w in words):
        raise ValueError(f"anchor_cord_end_to_body: anchor_body {anchor_body!r} is the robot, the table or the floor; "
                         "anchor to the object that holds the cord in the video")
    if model.body_weldid[cb] == 0:
        raise ValueError(f"anchor_cord_end_to_body: cord_body {cord_body!r} cannot move; give the cord's end link")
    name = kw.get("name") or f"{cord_body}_anchor"
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name) >= 0:
        raise ValueError(f"anchor_cord_end_to_body: an equality named {name!r} already exists; pass another name")
    warnings = []
    depth, x = 0, cb
    while int(model.body_parentid[x]) != 0 and model.body_weldid[int(model.body_parentid[x])] != 0:
        x = int(model.body_parentid[x]); depth += 1
    if depth:
        warnings.append(f"{cord_body} is {depth} links below its cable's root "
                        f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, x)!r}: a weld there acts through every joint "
                        "above it and gives more under a pull; build the cord with its anchored end as the root link")
    data = _state(model, key)
    p1, q1 = data.xpos[cb].copy(), data.xquat[cb].copy()
    p2, q2 = data.xpos[ab].copy(), data.xquat[ab].copy()
    R2 = data.xmat[ab].reshape(3, 3)
    rel_p = R2.T @ (p1 - p2)                    # cord link origin in the anchor body's frame
    rel_q = _qmul(_qconj(q2), q1)
    rel_q /= np.linalg.norm(rel_q)
    dt = float(model.opt.timestep)
    solref = kw.get("solref") or [2.0 * dt, 1.0]
    solimp = kw.get("solimp", [0.99, 0.999, 0.0001, 0.5, 2])
    tree = ET.parse(str(scene_xml))
    root = tree.getroot()
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    attrs = {"name": name, "body1": anchor_body, "body2": cord_body, "solref": _fmt(solref), "solimp": _fmt(solimp),
             "active": "true"}
    attrs["relpose"] = _fmt(list(rel_p) + list(rel_q))
    ET.SubElement(eq, "weld", attrs)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_xml), encoding="unicode")
    m2 = mujoco.MjModel.from_xml_path(str(out_xml))
    if m2.nq != model.nq or m2.nkey != model.nkey:
        raise RuntimeError("anchor_cord_end_to_body: the anchored scene changed its state size; nothing should")
    d2 = _state(m2, key)
    eid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    pos, ang, nrows = _residual(m2, d2, eid)
    if nrows == 0:
        raise RuntimeError(f"anchor_cord_end_to_body: the equality {name!r} produced no constraint rows (inactive?)")
    if pos > 1e-6 or ang > 1e-4:
        raise RuntimeError(f"anchor_cord_end_to_body: the new {kind} is violated at keyframe {key} by {pos:.2e} m / "
                           f"{ang:.2e} rad; the relative pose convention or the keyframe is wrong")
    return {"out_xml": str(out_xml), "equality": name, "kind": kind, "cord_body": cord_body, "anchor_body": anchor_body,
            "relpose": [round(float(x), 9) for x in list(rel_p) + list(rel_q)], "residual_m": pos, "residual_rad": ang,
            "key": key, "warnings": warnings}


if __name__ == "__main__":
    import json, sys as _sys
    a = _sys.argv[1:]
    print(json.dumps(anchor_cord_end_to_body(a[0], a[1], a[2], a[3]), indent=1))
