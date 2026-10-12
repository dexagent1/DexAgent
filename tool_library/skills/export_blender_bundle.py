"""Turn a scene + qpos trajectory into Blender render bundles (geometry, per-frame transforms, cameras, plates)."""
from __future__ import annotations

TOOL = {
    "name": "export_blender_bundle",
    "title": "Export Blender bundle",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Exports every visible geom once (meshes as OBJ with textures, primitives as parameters, PBR materials by rule), the world pose of every geom at every frame, a static ego camera and body-tracked extra cameras, lighting read off the clean plate, and re-projected background plates for the extra views.",
    "description": (
        "Cycles needs the scene as geometry plus animation, not as MJCF. This walks the "
        "model's visual geoms, writes shared OBJ meshes with UVs and texture PNGs, assigns a "
        "principled material from name/mesh rules (robot links, table, glass from alpha), then "
        "replays the qpos trajectory with forward kinematics to record a (N, G, 4, 4) transform "
        "array and the tracks of cameras parented to bodies (wrist cameras). World light colour "
        "comes from the plate's table pixels so the render sits in the footage. For non-ego views "
        "the ego plate is traced onto the table plane and sampled back, giving a per-frame "
        "background. Output per view: scene.json + transforms.npz (+ plates/), the input of "
        "render_cycles."
    ),
    "inputs": [
        {"name": "scene", "type": "path", "description": "Scene MJCF."},
        {"name": "trajectory", "type": "path | ndarray", "description": "npz with qpos (N,nq) [+ fps], or the array."},
        {"name": "out_dir", "type": "path", "description": "Receives meshes/, textures/, transforms.npz and one folder per view."},
        {"name": "ego", "type": "dict", "description": "{'pos','quat_wxyz', and 'fx','fy','cx','cy' (pixels of an image of 'image_size', default width x height) or 'fovy' alone, optional 'clip_start','near_fade'} the video camera."},
        {"name": "table", "type": "dict", "description": "{'top_z', 'center': [x,y], 'half_size': [hx,hy]}."},
        {"name": "plate_png", "type": "path", "description": "Clean background plate seen by the ego camera (erase_with_void)."},
        {"name": "views", "type": "dict", "description": "{name: {'parent_body','pos','quat_wxyz','fovy'}} cameras riding bodies (wrist views)."},
        {"name": "width, height, fps, stride", "type": "numbers", "description": "Render size and output frame rate / subsampling."},
        {"name": "render", "type": "dict", "description": "Overrides of samples (32), exclude_geoms, visual_groups, materials rules, object_roughness, lighting."},
        {"name": "robot_prefix", "type": "str", "description": "Body-name prefix that marks robot geoms ('robot_')."},
    ],
    "outputs": [
        {"name": "bundle", "type": "dict", "description": "out_dir, n_geoms, n_robot_geoms, n_frames, views -> folder with scene.json."},
    ],
    "status": "tested",
    "entrypoint": "export_blender_bundle",
    "requirements": ["mujoco", "opencv-python", "the stage4_robot_data_generation package (ships with DexAgent)"],
    "tags": ["Blender", "export", "PBR", "photoreal"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
}

import copy
import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common.stage4 import Cfg, defaults, stage4_module


def export_blender_bundle(scene, trajectory, out_dir, ego, table, plate_png, views=None, width=640, height=480,
                          fps=None, stride=1, render=None, robot_prefix="robot_", log=print):
    import cv2
    import mujoco
    ex, K = stage4_module("export"), stage4_module("kinematics")
    rc = copy.deepcopy(defaults()["render"])
    for k, v in (render or {}).items():
        rc[k] = {**rc[k], **v} if isinstance(v, dict) and isinstance(rc.get(k), dict) else v
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Cfg({"render": rc, "robot": {"body_prefix": robot_prefix}}, out, out)
    if isinstance(trajectory, (str, Path)):
        z = np.load(trajectory)
        Q = np.asarray(z["qpos"], float)
        fps = float(fps or (float(z["fps"]) if "fps" in z.files else 30.0))
    else:
        Q, fps = np.asarray(trajectory, float), float(fps or 30.0)
    model = K.load_model(str(scene))
    assert Q.shape[1] == model.nq, f"trajectory has {Q.shape[1]} dofs, model nq={model.nq}"
    geoms = ex.export_geoms(model, cfg, out)
    gids = [e["gid"] for e in geoms]
    plate = cv2.imread(str(plate_png))
    assert plate is not None, f"cannot read plate {plate_png}"
    ego = {"quat_wxyz": ego.get("quat_wxyz", ego.get("quat")), **{k: v for k, v in ego.items() if k != "quat"}}
    # fx, fy, cx, cy (or fovy alone) written out for the output size, as Stage 4's export does
    ego = stage4_module("config").resolved_camera(ego, int(width), int(height))
    light = dict(rc["lighting"])
    if light.get("world_color") is None:
        on, _ = ex.table_mask(ego, table, plate.shape[1], plate.shape[0])
        px = plate[on] if on.sum() > 100 else plate.reshape(-1, 3)
        light["world_color"] = [float(v) for v in ex.linear(np.median(px, axis=0)[::-1])]
    room = ex.room_tone(plate, ego, table)
    near = float(model.vis.map.znear * model.stat.extent)
    views = views or {}
    Q = Q[:: int(stride)]
    fps = fps / int(stride)
    n = len(Q)
    data = mujoco.MjData(model)
    parents = {v: K.body_id(model, c["parent_body"]) for v, c in views.items()}
    T = np.zeros((n, len(gids), 4, 4))
    T[:, :, 3, 3] = 1.0
    tracks = {v: [] for v in parents}
    for t in range(n):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        T[t, :, :3, :3] = data.geom_xmat[gids].reshape(-1, 3, 3)
        T[t, :, :3, 3] = data.geom_xpos[gids]
        for v, b in parents.items():
            c = views[v]
            R = data.xmat[b].reshape(3, 3)
            tracks[v].append({"pos": (data.xpos[b] + R @ np.asarray(c["pos"], float)).tolist(),
                              "quat_wxyz": K.mat2quat(R @ K.quat2mat(c.get("quat_wxyz", c.get("quat")))).tolist()})
    np.savez_compressed(out / "transforms.npz", T=T)
    common = {"geoms": geoms, "table": table, "width": int(width), "height": int(height), "fps": fps, "n_frames": n,
              "samples": int(rc["samples"]), "lighting": light, "fill_pos": ego["pos"], "transforms": str(out / "transforms.npz")}
    folders = {}
    for view in ["ego"] + list(parents):
        vdir = out / view
        vdir.mkdir(exist_ok=True)
        meta = dict(common)
        if view == "ego":
            meta["camera"] = {"pos": ego["pos"], "quat_wxyz": ego["quat_wxyz"], **{k: ego[k] for k in ("fx", "fy", "cx", "cy", "fovy")},
                              "clip_start": ego.get("clip_start", near)}
            meta["near_fade"] = ego.get("near_fade", [0.05, 0.15])
            meta["plate"] = str(Path(plate_png).resolve())
        else:
            c = views[view]
            meta["camera"] = {**tracks[view][0], "fovy": c["fovy"], "clip_start": c.get("clip_start", near)}
            meta["camera_track"] = tracks[view]
            meta["near_fade"] = c.get("near_fade", [0.05, 0.15])
            pdir = vdir / "plates"
            pdir.mkdir(exist_ok=True)
            for t in range(n):
                cv2.imwrite(str(pdir / f"frame{t:05d}.png"),
                            ex.reproject_plate(plate, ego, {**tracks[view][t], "fovy": c["fovy"]}, table, room, int(width), int(height)))
            meta["plates_dir"] = str(pdir)
        (vdir / "cache_key.txt").write_text(ex.render_key(meta, T))
        (vdir / "scene.json").write_text(json.dumps(meta, indent=1))
        folders[view] = str(vdir)
    n_robot = sum(e["robot"] for e in geoms)
    log(f"[export] {len(geoms)} geoms ({n_robot} robot), {n} frames @ {fps:g} fps, views {list(folders)} -> {out}")
    return {"out_dir": str(out), "n_geoms": len(geoms), "n_robot_geoms": int(n_robot), "n_frames": n, "fps": fps, "views": folders}


if __name__ == "__main__":
    a = sys.argv[1:]
    spec = json.loads(Path(a[3]).read_text())
    print(json.dumps(export_blender_bundle(a[0], a[1], a[2], **spec), indent=1))
