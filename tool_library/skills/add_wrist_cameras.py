"""Bolt wrist cameras (and fixed cameras) onto a scene without moving a single body."""
from __future__ import annotations

TOOL = {
    "name": "add_wrist_cameras",
    "title": "Add wrist cameras",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Derives a wrist-camera pose for each hand from its own fingertip sites (overhand palm view or behind-the-palm view) and adds the cameras plus optional fixed cameras, leaving nq and every body pose unchanged.",
    "description": (
        "Hard-coded camera offsets do not transfer between hands: on one dexterous hand the two "
        "sides' local frames differed by a 90 deg rotation, not a mirror. So the frame is read "
        "from the fingertips: the mean finger direction is the hand axis, the index-to-pinky row "
        "is the image horizontal, and up points away from the thumb (back of hand). 'overhand' "
        "puts the lens on the palm looking out along the fingers, as on the real rig footage; "
        "'behind' puts it behind the palm looking into the grasp volume. Cameras and the "
        "optional mount geometry are jointless and non-colliding, so a qpos recorded before the "
        "cameras existed replays frame for frame."
    ),
    "inputs": [
        {"name": "scene", "type": "path", "description": "Scene MJCF."},
        {"name": "out_xml", "type": "path", "description": "Output MJCF with the cameras."},
        {"name": "hands", "type": "dict", "description": "{side: {'body': hand root body, 'fingertips': {'thumb','index','middle','ring','pinky': site names}}}."},
        {"name": "mode", "type": "str", "description": "'overhand' (default) or 'behind'."},
        {"name": "fovy", "type": "float", "description": "Vertical field of view in degrees (100 overhand, 70 behind)."},
        {"name": "fixed", "type": "dict | None", "description": "{name: {'pos', 'target' | 'quat', 'fovy'}} world cameras to add as well."},
        {"name": "draw_mount", "type": "bool", "description": "Add a small visual housing at the camera (default False)."},
        {"name": "params", "type": "dict", "description": "Framing: overhand u=0.05 ef=0.055 k=0.45 reach=0.6; behind back=0.155 side=0.048 out=0.012."},
    ],
    "outputs": [
        {"name": "info", "type": "dict", "description": "out_xml, camera names, per-camera local pos/quat, nq unchanged check."},
    ],
    "status": "tested",
    "entrypoint": "add_wrist_cameras",
    "requirements": ["mujoco >= 3.2 (MjSpec)"],
    "tags": ["camera", "wrist", "view augmentation"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
}

import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common.render import lookat_quat


def derive_wrist_pose(model, data, body, fingertips, mode="overhand", **p):
    """eye/target/up in the hand body's local frame, read off its fingertip sites."""
    bid = model.body(body).id
    R = np.asarray(data.xmat[bid], float).reshape(3, 3)
    o = np.asarray(data.xpos[bid], float)
    loc = {k: R.T @ (np.asarray(data.site_xpos[model.site(s).id], float) - o) for k, s in fingertips.items()}
    fing = np.mean([loc[k] for k in ("index", "middle", "ring", "pinky")], axis=0)
    f = fing / np.linalg.norm(fing)
    if mode == "overhand":
        u, ef, k, reach = p.get("u", 0.05), p.get("ef", 0.055), p.get("k", 0.45), p.get("reach", 0.6)
        row = loc["index"] - loc["pinky"]
        row -= f * float(row @ f)
        row /= np.linalg.norm(row)
        up = np.cross(row, f)
        up /= np.linalg.norm(up)
        if float(up @ (loc["thumb"] - fing)) > 0:     # must face away from the thumb: back of hand
            up = -up
        eye = -up * u + f * ef
        ax = f - up * k
        ax /= np.linalg.norm(ax)
        return dict(eye=eye, target=eye + ax * reach, up=up)
    back, side, out = p.get("back", 0.155), p.get("side", 0.048), p.get("out", 0.012)
    thumb = loc["thumb"]
    nrm = thumb - fing
    nrm -= f * float(nrm @ f)
    nrm /= np.linalg.norm(nrm)
    s_dir = np.cross(f, nrm)
    centre = 0.5 * (fing + thumb)
    return dict(eye=centre - f * back + s_dir * side + nrm * out, target=centre, up=-nrm)


def add_wrist_cameras(scene, out_xml, hands, mode="overhand", fovy=None, fixed=None, draw_mount=False, params=None):
    import mujoco
    fovy = float(fovy if fovy is not None else (100.0 if mode == "overhand" else 70.0))
    m0 = mujoco.MjModel.from_xml_path(str(scene))
    d0 = mujoco.MjData(m0)
    mujoco.mj_forward(m0, d0)
    spec = mujoco.MjSpec.from_file(str(scene))
    bodies = {b.name: b for b in spec.bodies}
    cams = {}
    for side, h in hands.items():
        pose = derive_wrist_pose(m0, d0, h["body"], h["fingertips"], mode, **(params or {}))
        body = bodies[h["body"]]
        cam = body.add_camera()
        cam.name = f"{side}_wrist"
        cam.pos = [float(v) for v in pose["eye"]]
        cam.quat = [float(v) for v in lookat_quat(pose["eye"], pose["target"], pose["up"])]
        cam.fovy = fovy
        cams[cam.name] = {"body": h["body"], "pos": list(cam.pos), "quat": list(cam.quat), "fovy": fovy}
        if draw_mount:
            g = body.add_geom()
            g.name, g.type = f"{side}_wristcam_housing", mujoco.mjtGeom.mjGEOM_BOX
            g.size, g.pos, g.quat = [0.012, 0.016, 0.009], list(cam.pos), list(cam.quat)
            g.rgba, g.contype, g.conaffinity, g.group, g.mass = [0.1, 0.1, 0.11, 1], 0, 0, 2, 1e-6
    for name, c in (fixed or {}).items():
        cam = spec.worldbody.add_camera()
        cam.name = name
        cam.pos = [float(v) for v in c["pos"]]
        cam.quat = [float(v) for v in (c["quat"] if "quat" in c else lookat_quat(c["pos"], c["target"], c.get("up", (0, 0, 1))))]
        cam.fovy = float(c.get("fovy", 45.0))
        cams[name] = {"body": "world", "pos": list(cam.pos), "quat": list(cam.quat), "fovy": cam.fovy}
    model = spec.compile()
    assert model.nq == m0.nq, "adding cameras changed nq"
    Path(out_xml).parent.mkdir(parents=True, exist_ok=True)
    Path(out_xml).write_text(spec.to_xml())
    return {"out_xml": str(out_xml), "cameras": cams, "mode": mode, "nq": int(model.nq)}


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(add_wrist_cameras(a[0], a[1], json.loads(Path(a[2]).read_text())), indent=1))
