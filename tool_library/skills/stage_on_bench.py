"""Place assets on a robot bench in the rig frame spanned by the two arm bases."""
from __future__ import annotations

TOOL = {
    "name": "stage_on_bench",
    "title": "Stage on bench",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Attaches asset MJCFs into an existing robot world at poses given in the rig frame (origin between the two arm bases on the table top, x along the baseline, y into the work).",
    "description": (
        "A fitted trajectory is expressed relative to the robot, so objects are staged relative "
        "to the robot too: the rig frame has its origin at the midpoint of the two arm roots "
        "projected to the table top, x from the left base to the right base, y into the "
        "workspace and z up. Each asset is attached through a frame at origin + R_rig @ (x, y, z) "
        "with its orientation in the rig frame (yaw only by default; roll_deg / pitch_deg for an "
        "object lying on its side or flipped, composed as R = Rz(yaw) Ry(pitch) Rx(roll), the "
        "convention fit_object_placement reports; or a full quat), with a name prefix, so the bench's joints and qpos layout are "
        "unchanged for every body that was already there (a recorded qpos still replays). "
        "The rig frame can also be passed explicitly for single-arm or custom benches."
    ),
    "inputs": [
        {"name": "world_xml", "type": "path", "description": "Robot world MJCF."},
        {"name": "assets", "type": "list[dict]", "description": "[{'name', 'xml', 'x', 'y', 'z'=0, 'yaw_deg'=0, 'pitch_deg'=0, 'roll_deg'=0 | 'quat'=[w,x,y,z], 'joints'={joint: value}}] in rig coordinates (metres, degrees). The orientation is R = Rz(yaw) Ry(pitch) Rx(roll) in the rig frame (roll_deg 180 lays a phone whose body frame has its back up screen up), or 'quat' (w, x, y, z) for any rotation; give one or the other. 'joints' sets the asset's initial joint values (an open lid, a pulled drawer) in the scene's keyframes, joint names as in the asset (the prefix is added). Other keys are refused."},
        {"name": "out_xml", "type": "path", "description": "Output MJCF."},
        {"name": "arm_roots", "type": "tuple[str, str]", "description": "Body names of the (left, right) arm bases used to derive the rig frame. Default ('robot_left_arm_root', 'robot_right_arm_root')."},
        {"name": "table_top_z", "type": "float", "description": "Height of the rig origin (table surface). Default 0.0."},
        {"name": "rig", "type": "dict | None", "description": "Explicit {'origin': [x,y,z], 'R': 3x3 columns x,y,z} overriding the derivation. The DexAgent setup scene's world frame IS the rig frame: pass origin [0,0,0] and the identity."},
        {"name": "cameras", "type": "dict | None", "description": "{name: camera.json path or dict} cameras to write into the world (for example {'video_cam': 'camera.json'}): pose, fovy and, when the dict has width and height, the camera's resolution (its image size, so renderers keep its aspect)."},
    ],
    "outputs": [
        {"name": "info", "type": "dict", "description": "out_xml, origin, R (3x3), per asset world_pos/world_quat and new body names, nq before/after."},
    ],
    "status": "tested",
    "entrypoint": "stage_on_bench",
    "requirements": ["mujoco >= 3.2 (MjSpec)"],
    "tags": ["bench", "rig frame", "attach", "staging"],
    "used_in": ["Multi-Cup Grasping", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
}

import json
import re
import sys
from pathlib import Path

import numpy as np

from tool_library.common.render import mat_to_quat


def rig_frame(model, data, arm_roots=("robot_left_arm_root", "robot_right_arm_root"), table_top_z=0.0):
    """(origin, R) with origin at the base midpoint on the table top, x left->right, y into the work."""
    pl = np.asarray(data.xpos[model.body(arm_roots[0]).id], float)
    pr = np.asarray(data.xpos[model.body(arm_roots[1]).id], float)
    x = pr - pl
    x[2] = 0.0
    x /= np.linalg.norm(x)
    z = np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    origin = 0.5 * (pr + pl)
    origin[2] = float(table_top_z)
    return origin, np.stack([x, y, z], 1)


def _strip_lights(xml_path, work_dir):
    """A standalone asset carries its own lights and a floor plane for previews; on the bench the world provides both."""
    src = Path(xml_path)
    txt = src.read_text()
    txt2 = re.sub(r"[ \t]*<light\b[^>]*/>\s*\n?", "", txt)
    txt2 = re.sub(r"[ \t]*<geom\b[^>]*type=\"plane\"[^>]*/>\s*\n?", "", txt2)
    if txt2 == txt:
        return src
    txt2 = re.sub(r'(meshdir|texturedir)="([^"]*)"', lambda m: f'{m.group(1)}="{(src.parent / m.group(2)).resolve()}"', txt2)
    dst = Path(work_dir) / f"{src.stem}_bench_attach.xml"
    dst.write_text(txt2)
    return dst


def _absolute_assets(spec, base_dir):
    """Mesh and texture files relative to the spec's own folder become absolute, so the written scene loads from anywhere."""
    base = Path(base_dir).resolve()
    mesh_dir = Path(spec.meshdir) if getattr(spec, "meshdir", "") else None
    tex_dir = Path(spec.texturedir) if getattr(spec, "texturedir", "") else None
    for m in spec.meshes:
        f = getattr(m, "file", "") or ""
        if f and not Path(f).is_absolute():
            root = base / mesh_dir if (mesh_dir and not mesh_dir.is_absolute()) else (mesh_dir or base)
            m.file = str((root / f).resolve())
    for t in spec.textures:
        f = getattr(t, "file", "") or ""
        if f and not Path(f).is_absolute():
            root = base / tex_dir if (tex_dir and not tex_dir.is_absolute()) else (tex_dir or base)
            t.file = str((root / f).resolve())
    try:
        spec.meshdir = ""; spec.texturedir = ""
    except Exception:
        pass


ASSET_KEYS = {"name", "xml", "x", "y", "z", "yaw_deg", "pitch_deg", "roll_deg", "quat", "joints"}


def asset_rotation(a):
    """The asset's orientation in the rig frame: 'quat' (w, x, y, z), else Rz(yaw) Ry(pitch) Rx(roll) from the degree keys."""
    unknown = sorted(set(a) - ASSET_KEYS)
    if unknown:
        raise ValueError(f"asset {a.get('name', '?')!r}: unknown key(s) {unknown}; allowed {sorted(ASSET_KEYS)} "
                         f"(orientation: yaw_deg / pitch_deg / roll_deg, or quat [w, x, y, z])")
    for k in ("name", "xml", "x", "y"):
        if k not in a:
            raise ValueError(f"asset {a.get('name', '?')!r}: missing {k!r}")
    if a.get("quat") is not None:
        if any(a.get(k) not in (None, 0, 0.0) for k in ("yaw_deg", "pitch_deg", "roll_deg")):
            raise ValueError(f"asset {a['name']!r}: give quat or yaw_deg/pitch_deg/roll_deg, not both")
        q = np.asarray(a["quat"], float).ravel()
        if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
            raise ValueError(f"asset {a['name']!r}: quat must be four finite numbers [w, x, y, z]")
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    yaw, pitch, roll = (np.radians(float(a.get(k) or 0.0)) for k in ("yaw_deg", "pitch_deg", "roll_deg"))
    cz, sz, cy, sy, cx, sx = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


def stage_on_bench(world_xml, assets, out_xml, arm_roots=("robot_left_arm_root", "robot_right_arm_root"),
                   table_top_z=0.0, rig=None, cameras=None):
    import mujoco
    for a in assets:                                    # refuse a misspelled or unsupported key before building anything
        asset_rotation(a)
    out = Path(out_xml)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rig is None:
        m0 = mujoco.MjModel.from_xml_path(str(world_xml))
        d0 = mujoco.MjData(m0)
        mujoco.mj_forward(m0, d0)
        origin, R = rig_frame(m0, d0, arm_roots, table_top_z)
        nq_before = int(m0.nq)
    else:
        origin, R = np.asarray(rig["origin"], float), np.asarray(rig["R"], float)
        nq_before = int(mujoco.MjModel.from_xml_path(str(world_xml)).nq)
    spec = mujoco.MjSpec.from_file(str(world_xml))
    _absolute_assets(spec, Path(world_xml).parent)
    for cname, cam in (cameras or {}).items():          # the video camera, so verifiers can render through it by name
        if isinstance(cam, (str, Path)):
            cam = json.loads(Path(cam).read_text())
        existing = [c for c in spec.cameras if c.name == cname]
        c = existing[0] if existing else spec.worldbody.add_camera()
        c.name = cname; c.pos = [float(v) for v in cam["pos"]]; c.quat = [float(v) for v in cam["quat_wxyz"]]; c.fovy = float(cam["fovy_deg"])
        if cam.get("width") and cam.get("height") and hasattr(c, "resolution"):
            # the image size the camera was calibrated at: renderers keep its aspect (render_trajectory_video renders an
            # EgoDex 1920 x 1080 camera 16:9, not squashed into 640 x 480); fovy still sets the projection
            c.resolution = [int(cam["width"]), int(cam["height"])]
    # the scene's timestep must serve its most demanding asset (a thread mechanism needs 0.25 ms)
    min_dt = float(spec.option.timestep)
    placed = []
    rigid_contact_names = []
    for a in assets:
        child = mujoco.MjSpec.from_file(str(_strip_lights(a["xml"], out.parent)))
        from tool_library.common.contact_defaults import asset_default_names
        rigid_contact_names.extend(asset_default_names(child, a["xml"], prefix=f"{a['name']}_"))
        _absolute_assets(child, Path(a["xml"]).parent)
        min_dt = min(min_dt, float(child.option.timestep))
        Rl = asset_rotation(a)
        p = origin + R @ np.array([float(a["x"]), float(a["y"]), float(a.get("z", 0.0))])
        q = mat_to_quat(R @ Rl)
        fr = spec.worldbody.add_frame()
        fr.pos, fr.quat = [float(v) for v in p], [float(v) for v in q]
        before = {b.name for b in spec.bodies}
        spec.attach(child, prefix=f"{a['name']}_", frame=fr)
        placed.append({"name": a["name"], "xml": str(a["xml"]), "x": float(a["x"]), "y": float(a["y"]),
                       "z": float(a.get("z", 0.0)), "yaw_deg": float(a.get("yaw_deg") or 0.0),
                       "pitch_deg": float(a.get("pitch_deg") or 0.0), "roll_deg": float(a.get("roll_deg") or 0.0),
                       "quat_rig": [round(float(v), 6) for v in mat_to_quat(Rl)],
                       "world_pos": [round(float(v), 4) for v in p], "world_quat": [round(float(v), 6) for v in q],
                       "bodies": sorted({b.name for b in spec.bodies} - before)})
    spec.option.timestep = min_dt
    from tool_library.common.contact_defaults import add_rigid_pairs
    add_rigid_pairs(spec, rigid_contact_names)
    model = spec.compile()
    # keyframes of the world cover only its own dofs: extend them with the attached assets' initial state
    # (a padded free joint would get a zero quaternion and the asset would vanish)
    if len(spec.keys):
        for key in spec.keys:
            old_q = np.asarray(key.qpos, float)        # the spec pads a short key with zeros: keep only the world's own dofs
            full = np.array(model.qpos0, float)
            n_keep = min(int(nq_before), len(old_q), model.nq)
            full[:n_keep] = old_q[:n_keep]
            for a in assets:                               # the objects' initial joint states as the video shows them
                for jn, val in (a.get("joints") or {}).items():
                    jname = f"{a['name']}_{jn}"
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    if jid < 0:
                        raise ValueError(f"asset {a['name']!r} has no joint {jn!r} (looked for {jname!r}); joints: "
                                         f"{[model.joint(j).name for j in range(model.njnt) if model.joint(j).name.startswith(a['name'] + '_')][:20]}")
                    if model.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                        full[int(model.jnt_qposadr[jid])] = float(val)
                    else:
                        raise ValueError(f"joints: {jn!r} is not a hinge or slide joint")
            key.qpos = full.tolist()
            old_c = np.asarray(key.ctrl, float)
            if len(old_c) and len(old_c) < model.nu:
                key.ctrl = np.concatenate([old_c, np.zeros(model.nu - len(old_c))]).tolist()
        model = spec.compile()
    out.write_text(spec.to_xml())
    return {"out_xml": str(out), "origin": [round(float(v), 4) for v in origin],
            "R": [[round(float(v), 6) for v in row] for row in R], "assets": placed,
            "nq_before": nq_before, "nq_after": int(model.nq)}


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(stage_on_bench(a[0], json.loads(Path(a[1]).read_text()), a[2]), indent=1))
