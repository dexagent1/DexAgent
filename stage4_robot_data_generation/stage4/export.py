"""Step 3: turn each accepted trajectory into Blender render bundles.

For every sample and every camera view this writes
    scene.json       geometry list with materials, camera (static or tracked),
                     lighting, table plane
    transforms.npz   world pose of every rendered geom at every output frame
and, for views other than the ego view, a background plate per frame: the ego
plate re-photographed from that camera by projecting it onto the table plane.
Meshes are written once and shared by all samples.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2
import mujoco
import numpy as np

from . import kinematics as K
from .config import ASPECT_TOL, aspect_mismatch, intrinsics, resolved_camera

PRESETS = json.loads((Path(__file__).parent / "materials.json").read_text())
GEOM = mujoco.mjtGeom
MESH = int(GEOM.mjGEOM_MESH)
TYPES = {int(GEOM.mjGEOM_SPHERE): "sphere", int(GEOM.mjGEOM_CAPSULE): "capsule",
         int(GEOM.mjGEOM_ELLIPSOID): "ellipsoid", int(GEOM.mjGEOM_CYLINDER): "cylinder",
         int(GEOM.mjGEOM_BOX): "box", MESH: "mesh"}


def validate_sample(directory, info, scene=None):
    """Refuse stale, rejected, or kinematic-only episodes before producing render bundles."""
    checks = info.get("checks") or {}
    trajectory = Path(directory) / "trajectory.npz"
    if (info.get("trajectory_source") != "physics_replay" or not checks.get("ok")
            or not (checks.get("physics") or {}).get("ok")):
        raise ValueError(f"{directory}: episode has no passing physics replay; rerun randomize")
    if hashlib.sha256(trajectory.read_bytes()).hexdigest() != info.get("trajectory_sha256"):
        raise ValueError(f"{directory}: trajectory changed after verification; rerun randomize")
    digest = hashlib.sha256(json.dumps(info.get("body_overrides", {}), sort_keys=True, default=float).encode()).hexdigest()
    if digest != info.get("body_overrides_sha256"):
        raise ValueError(f"{directory}: scene body overrides changed after verification; rerun randomize")
    if scene is not None and hashlib.sha256(Path(scene).read_bytes()).hexdigest() != info.get("source_scene_sha256"):
        raise ValueError(f"{directory}: runtime scene changed after verification; rerun randomize")
    runtime = Path(directory) / "scene_physics.xml"
    if not runtime.is_file() or hashlib.sha256(runtime.read_bytes()).hexdigest() != info.get("runtime_scene_sha256"):
        raise ValueError(f"{directory}: actual runtime scene is missing or changed after verification")
    with np.load(trajectory, allow_pickle=False) as z:
        if str(z["scene_sha256"].item()) != info.get("runtime_scene_sha256"):
            raise ValueError(f"{directory}: trajectory identifies a different runtime scene")
    if info.get("scene_recon_sha256"):
        recon = Path(directory) / "mechanism_scene_recon.json"
        if not recon.is_file() or hashlib.sha256(recon.read_bytes()).hexdigest() != info["scene_recon_sha256"]:
            raise ValueError(f"{directory}: mechanism evidence changed after verification")
    if info.get("deformables"):
        from .deformables import verify_sample
        bad = verify_sample(directory, info)
        if bad:
            raise ValueError(f"{directory}: {'; '.join(bad)}; rerun randomize")


def rendered_episode(directory, info):
    """(scene, trajectory) a sample renders from: its rigid runtime recording, or, with deformables, its Isaac mirror
    (the runtime scene plus each cord as capsule links posed from the Isaac recording)."""
    d = Path(directory)
    meta = info.get("deformables")
    if meta:
        return d / meta["scene_mirror"], d / meta["trajectory_mirror"]
    return d / "scene_physics.xml", d / "trajectory.npz"


def geom_key(model, g):
    mesh = ""
    if int(model.geom_type[g]) == MESH:
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[g])) or ""
    return f"{model.body(model.geom_bodyid[g]).name}/{model.geom(g).name}/{mesh}", mesh


def geom_rgba(model, g):
    """A geom's color: its own rgba if set, otherwise its material's."""
    rgba = model.geom_rgba[g]
    mid = int(model.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        rgba = model.mat_rgba[mid]
    return [float(v) for v in rgba]


def texture_id(model, g):
    """The color texture on a geom's material (any type), or -1."""
    mid = int(model.geom_matid[g])
    if mid < 0:
        return -1
    tid = model.mat_texid[mid]
    return int(np.atleast_1d(tid)[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] if np.ndim(tid) else tid)


def texture_png(model, g, tex_dir):
    """The 2D texture on a geom's material, dumped from the compiled model."""
    tid = texture_id(model, g)
    if tid < 0 or int(model.tex_type[tid]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
        return None
    h, w = int(model.tex_height[tid]), int(model.tex_width[tid])
    c = int(model.tex_nchannel[tid])
    a = int(model.tex_adr[tid])
    img = np.asarray(model.tex_data[a:a + h * w * c]).reshape(h, w, c)
    tex_dir.mkdir(parents=True, exist_ok=True)
    f = tex_dir / f"tex{tid}.png"
    ok, png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR if c == 3 else cv2.COLOR_RGBA2BGRA))
    assert ok, f"cannot encode texture {tid}"
    write_if_changed(f, png.tobytes())                    # a texture left from another scene is replaced
    return str(f)


def write_if_changed(path, data):
    """Write bytes (or text) unless the file already holds exactly them; returns the SHA-1 of the content.
    Shared meshes and textures are reused across samples, but never a stale one from an earlier scene."""
    data = data.encode() if isinstance(data, str) else bytes(data)
    path = Path(path)
    if not path.exists() or path.stat().st_size != len(data) or path.read_bytes() != data:
        path.write_bytes(data)
    return hashlib.sha1(data).hexdigest()


# how a 2D texture is laid onto a primitive that has no texture coordinates of its own (render_cycles builds the
# matching UVs): flat across the geom's local xy for a box (a board's top face carries the whole picture, as in
# MuJoCo's viewer), around the axis for a cylinder or capsule, by latitude and longitude for a sphere
PRIMITIVE_UV = {"box": "planar_xy", "cylinder": "cylindrical", "capsule": "cylindrical", "sphere": "spherical",
                "ellipsoid": "spherical"}


def box_projection_uv(verts, faces):
    """Texture coordinates for a mesh that has none: every face is projected along the axis its normal points
    most along, onto the mesh's bounding box side (a box / triplanar projection, one UV per face corner).
    Returns (uv [3F, 2], face-corner indices [F, 3])."""
    v = np.asarray(verts, float)
    f = np.asarray(faces, int)
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    axis = np.argmax(np.abs(n), axis=1)
    keep = np.array([[1, 2], [0, 2], [0, 1]])[axis]             # drop the axis the face looks along
    rel = (v[f] - lo) / span                                     # [F, 3 corners, 3]
    uv = np.take_along_axis(rel, keep[:, None, :].repeat(3, axis=1), axis=2)
    return uv.reshape(-1, 2), np.arange(3 * len(f)).reshape(-1, 3)


def resolve_material(rules, key, mesh, name, rgba, rough):
    """First matching config rule -> concrete Principled parameters."""
    albedo = float(np.mean(rgba[:3]))
    for r in rules:
        if not re.search(r["match"], key):
            continue
        lo, hi = r.get("albedo", [-1.0, 2.0])
        if not lo <= albedo < hi:
            continue
        if "preset" in r:
            assert r["preset"] in PRESETS, f"unknown material preset '{r['preset']}'"
            for pat, params in PRESETS[r["preset"]]:
                if re.search(pat, mesh or name, re.I):
                    return dict(params)
        return {k: r[k] for k in ("base", "metallic", "rough", "spec") if k in r}
    return {"base": rgba[:3], "metallic": 0.0, "rough": rough}


def export_geoms(model, cfg, shared, report=None):
    """Every visible geom, once: meshes as OBJ, primitives as parameters.

    report (a dict) collects how each textured geom is drawn: `mapped` (its own UVs), `projected` (a primitive
    with the projection render_cycles applies), `generated_uv` (a mesh without UVs, box-projected here) and
    `dropped` (a texture that cannot be drawn, such as a cube map: the geom renders in its flat color)."""
    rc = cfg["render"]
    prefix = cfg["robot"]["body_prefix"]
    mesh_dir = shared / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    report = {} if report is None else report
    for k in ("mapped", "projected", "generated_uv", "dropped"):
        report.setdefault(k, [])
    entries = []
    for g in range(model.ngeom):
        key, mesh = geom_key(model, g)
        rgba = geom_rgba(model, g)
        gtype = int(model.geom_type[g])
        if (int(model.geom_group[g]) not in rc["visual_groups"] or rgba[3] <= 0.0
                or gtype not in TYPES or any(re.search(p, key) for p in rc["exclude_geoms"])):
            continue
        name = model.geom(g).name or f"geom{g}"
        body = model.body(model.geom_bodyid[g]).name
        root = model.body(model.body_rootid[model.geom_bodyid[g]]).name
        e = {"gid": g, "name": name, "type": TYPES[gtype], "rgba": rgba,
             "size": [float(v) for v in model.geom_size[g]],
             "robot": bool(body.startswith(prefix) or root.startswith(prefix))}
        tid = texture_id(model, g)
        tex = texture_png(model, g, shared / "textures") if tid >= 0 else None
        label = key.rstrip("/")
        if tid >= 0 and tex is None:
            kind = {int(mujoco.mjtTexture.mjTEXTURE_CUBE): "cube", int(mujoco.mjtTexture.mjTEXTURE_SKYBOX): "skybox"}.get(
                int(model.tex_type[tid]), "unknown")
            report["dropped"].append(f"{label} ({kind} texture {tid}: only 2D textures are drawn; it renders in its flat color)")
        if gtype == MESH:
            mid = int(model.geom_dataid[g])
            va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            fa, fn = int(model.mesh_faceadr[mid]), int(model.mesh_facenum[mid])
            verts = model.mesh_vert[va:va + vn]
            lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
            faces = model.mesh_face[fa:fa + fn]
            ta = int(model.mesh_texcoordadr[mid])
            if ta >= 0:
                uv = model.mesh_texcoord[ta:ta + int(model.mesh_texcoordnum[mid])]
                ft = model.mesh_facetexcoord[fa:fa + fn]
                lines += [f"vt {u:.6f} {1.0 - v:.6f}" for u, v in uv]
                lines += [f"f {a + 1}/{ua + 1} {b + 1}/{ub + 1} {c + 1}/{uc + 1}"
                          for (a, b, c), (ua, ub, uc) in zip(faces, ft)]
                f = mesh_dir / f"mesh{mid}.obj"
                if tex:
                    report["mapped"].append(label)
            elif tex:
                # a textured mesh without texture coordinates (it rendered white): box-project them
                uv, ft = box_projection_uv(verts, faces)
                lines += [f"vt {u:.6f} {v:.6f}" for u, v in uv]
                lines += [f"f {a + 1}/{ua + 1} {b + 1}/{ub + 1} {c + 1}/{uc + 1}"
                          for (a, b, c), (ua, ub, uc) in zip(faces, ft)]
                f = mesh_dir / f"mesh{mid}_boxuv.obj"
                report["generated_uv"].append(label)
                e["uv"] = "box_projection"
            else:
                lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
                f = mesh_dir / f"mesh{mid}.obj"
            if tex:
                e["texture"] = tex
            e["obj"] = str(f)
            e["obj_sha1"] = write_if_changed(f, "\n".join(lines) + "\n")
        elif tex:
            e["texture"], e["uv"] = tex, PRIMITIVE_UV[TYPES[gtype]]
            mat = int(model.geom_matid[g])
            e["texrepeat"] = [float(x) for x in model.mat_texrepeat[mat]]
            report["projected"].append(f"{label} ({e['uv']})")
        if e.get("texture"):
            e["texture_sha1"] = hashlib.sha1(Path(e["texture"]).read_bytes()).hexdigest()
        e["material"] = resolve_material(rc["materials"], key, mesh, name, rgba,
                                         rc["object_roughness"])
        if rgba[3] < 1.0:
            e["transmission"] = round(1.0 - rgba[3], 3)       # MuJoCo alpha = see-through
        entries.append(e)
    assert entries, "no visible geoms to render: check render.visual_groups / exclude_geoms"
    return entries


def quat_R(q):
    return K.quat2mat(q)


def table_mask(cam, table, W, H):
    """Which pixels of a W x H camera image look at the table top, and where on it. The camera is a dict with
    pos, quat_wxyz and its intrinsics (fx, fy, cx, cy [+ image_size], or fovy; see config.intrinsics)."""
    fx, fy, cx, cy = intrinsics(cam, W, H)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)      # pixel centres
    # MuJoCo / Blender camera axes: x right, y up, looking along -z
    d = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u)], -1) @ quat_R(cam["quat_wxyz"]).T
    p = np.asarray(cam["pos"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (table["top_z"] - p[2]) / d[..., 2]
    pts = p + s[..., None] * d
    on = (np.isfinite(s) & (s > 0)
          & (np.abs(pts[..., 0] - table["center"][0]) < table["half_size"][0])
          & (np.abs(pts[..., 1] - table["center"][1]) < table["half_size"][1]))
    return on, pts


def project(cam, pts, W, H):
    """Pixel coordinates (u, v; see config.intrinsics) and depth of world points in a W x H camera image."""
    fx, fy, cx, cy = intrinsics(cam, W, H)
    q = (np.asarray(pts, float) - np.asarray(cam["pos"], float)) @ quat_R(cam["quat_wxyz"])
    z = -q[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return cx + fx * q[..., 0] / z, cy - fy * q[..., 1] / z, z


def reproject_plate(plate, ego, cam, table, room, W, H):
    """The ego plate as seen from another camera: every pixel that looks at the
    table plane is traced to it and sampled back through the ego lens. Pixels
    off the table, or outside what the ego camera saw, take the room tone."""
    on, pts = table_mask(cam, table, W, H)
    EH, EW = plate.shape[:2]
    eu, ev, z = project(ego, pts, EW, EH)
    eu, ev = (eu - 0.5).astype(np.float32), (ev - 0.5).astype(np.float32)    # remap samples at pixel centres
    ok = on & (z > 1e-6) & (eu >= 0) & (eu <= EW - 1) & (ev >= 0) & (ev <= EH - 1)
    eu[~ok], ev[~ok] = 0, 0
    img = cv2.remap(plate, eu, ev, cv2.INTER_LINEAR)
    img[~ok] = room
    return img


def room_tone(plate, ego, table):
    """Median color of what the ego camera sees that is not the table."""
    on, _ = table_mask(ego, table, plate.shape[1], plate.shape[0])
    px = plate[~on] if (~on).sum() > 100 else plate.reshape(-1, 3)
    return np.median(px, axis=0).astype(np.uint8)


def linear(srgb):
    c = np.asarray(srgb, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def render_key(meta, T):
    """Hash of everything a rendered frame depends on. The background plate
    and file locations are left out: changing them does not change a frame."""
    m = {k: v for k, v in meta.items() if k not in ("plate", "plates_dir", "transforms")}
    m["geoms"] = [{k: v for k, v in e.items() if k not in ("obj", "texture")}
                  for e in meta["geoms"]]
    return hashlib.sha1(json.dumps(m, sort_keys=True).encode()
                        + np.ascontiguousarray(T).tobytes()).hexdigest()


def stage_views(cfg):
    """The camera views every sample is rendered from: ego first, then cameras.views in config order."""
    return ["ego"] + list(cfg["cameras"]["views"])


def current_samples(cfg):
    """The samples the randomize step published (folders of outputs/samples/ with a trajectory)."""
    return sorted(p.parent for p in (cfg.out / "samples").glob("*/trajectory.npz"))


def prune_render(root, keep, views, log=print):
    """Remove render bundles of samples that no longer exist (a re-run kept fewer) and of views no longer
    configured, so the render and composite steps only ever see the current samples. The shared meshes/ and
    textures/ folders stay."""
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in ("meshes", "textures"):
            continue
        if d.name not in keep:
            shutil.rmtree(d)
            log(f"[export] removed the stale render folder {d.name}/ (no such sample any more)")
            continue
        for v in sorted(d.iterdir()):
            if v.is_dir() and v.name not in views:
                shutil.rmtree(v)
                log(f"[export] removed the render folder {d.name}/{v.name}/ (view no longer configured)")


def load_plate(plate_f, W, H):
    """The clean plate at the output size. A plate written for another size of the same aspect ratio is resized;
    one of another aspect ratio cannot line up with the renders."""
    plate = cv2.imread(str(plate_f))
    assert plate is not None, f"no clean plate at {plate_f}: run the plate step first"
    ph, pw = plate.shape[:2]
    if (pw, ph) != (W, H):
        assert aspect_mismatch((pw, ph), (W, H)) <= ASPECT_TOL, (
            f"the clean plate is {pw} x {ph}, the output {W} x {H}: different aspect ratios; re-run the plate step")
        plate = cv2.resize(plate, (W, H), interpolation=cv2.INTER_AREA if pw > W else cv2.INTER_CUBIC)
    return plate


def run(cfg, log=print):
    rc, cams = cfg["render"], cfg["cameras"]
    W, H, stride = int(cams["width"]), int(cams["height"]), int(rc["stride"])
    from .deformables import load as load_deformables
    dfm = load_deformables(cfg)
    # with deformables the geoms are the mirror's: the scene's own, then each cord's capsule links (appended last)
    base = K.load_model(dfm["scene_mirror"] if dfm is not None else cfg.path("scene"))
    root = cfg.out / "render"
    samples = current_samples(cfg)
    assert samples, "no samples: run the randomize step first"
    views = stage_views(cfg)
    prune_render(root, {s.name for s in samples}, set(views), log)
    textures = {}
    geoms = export_geoms(base, cfg, root, textures)
    gids = [e["gid"] for e in geoms]
    n_robot = sum(e["robot"] for e in geoms)
    log(f"[export] {len(geoms)} geoms ({n_robot} robot, {len(geoms) - n_robot} object); textures: "
        f"{len(textures['mapped'])} with their own UVs, {len(textures['projected'])} primitives projected, "
        f"{len(textures['generated_uv'])} meshes given box-projected UVs, {len(textures['dropped'])} dropped")
    for d in textures["dropped"]:
        log(f"[export] warning: texture dropped on {d}")

    plate_f = cfg.out / "plate" / "plate.png"
    plate = load_plate(plate_f, W, H)
    table = cfg["table"]
    # every camera with its intrinsics written out at the output size (fx, fy, cx, cy: config.intrinsics)
    ego = resolved_camera(cams["ego"], W, H)
    view_cams = {v: resolved_camera(c, W, H) for v, c in cams["views"].items()}
    light = dict(rc["lighting"])
    if light.get("world_color") is None:
        on, _ = table_mask(ego, table, W, H)
        px = plate[on] if on.sum() > 100 else plate.reshape(-1, 3)
        light["world_color"] = [float(v) for v in linear(np.median(px, axis=0)[::-1])]
    room = room_tone(plate, ego, table)
    # the simulator's own near plane, so a render shows what the sim camera shows
    near = float(base.vis.map.znear * base.stat.extent)
    lens = ("fx", "fy", "cx", "cy", "fovy")

    for sdir in samples:
        info = json.loads((sdir / "sample.json").read_text())
        validate_sample(sdir, info, cfg.path("scene"))
        if info.get("textures") != textures:              # how each textured geom is drawn, per episode
            info["textures"] = textures
            (sdir / "sample.json").write_text(json.dumps(info, indent=1))
        scene_f, traj_f = rendered_episode(sdir, info)
        if dfm is not None and not info.get("deformables"):
            raise ValueError(f"{sdir}: the task has deformables but the sample has no Isaac episode; rerun randomize")
        model = mujoco.MjModel.from_xml_path(str(scene_f))
        if dfm is not None and model.ngeom != base.ngeom:
            raise ValueError(f"{sdir}: {scene_f.name} has {model.ngeom} geoms, the exported scene {base.ngeom}")
        data = mujoco.MjData(model)
        z = np.load(traj_f)
        Q, fps = z["qpos"][::stride], float(z["fps"]) / stride
        n = len(Q)
        parents = {v: K.body_id(model, c["parent_body"]) for v, c in view_cams.items()}
        T = np.zeros((n, len(gids), 4, 4))
        T[:, :, 3, 3] = 1.0
        tracks = {v: [] for v in parents}
        for t in range(n):
            data.qpos[:] = Q[t]
            mujoco.mj_kinematics(model, data)
            T[t, :, :3, :3] = data.geom_xmat[gids].reshape(-1, 3, 3)
            T[t, :, :3, 3] = data.geom_xpos[gids]
            for v, b in parents.items():
                c = view_cams[v]
                R = data.xmat[b].reshape(3, 3)
                tracks[v].append({
                    "pos": (data.xpos[b] + R @ np.asarray(c["pos"], float)).tolist(),
                    "quat_wxyz": K.mat2quat(R @ quat_R(c["quat_wxyz"])).tolist()})
        assert np.abs(T[:, :, :3, 3]).max() < 50.0, "geom transforms are out of range"
        out = root / sdir.name
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "transforms.npz", T=T)
        common = {"geoms": geoms, "table": table, "width": W, "height": H, "fps": fps,
                  "n_frames": n, "samples": int(rc["samples"]), "lighting": light,
                  "fill_pos": ego["pos"], "transforms": str(out / "transforms.npz")}
        for view in views:
            vdir = out / view
            vdir.mkdir(exist_ok=True)
            meta = dict(common)
            if view == "ego":
                meta["camera"] = {"pos": ego["pos"], "quat_wxyz": ego["quat_wxyz"],
                                  **{k: ego[k] for k in lens},
                                  "clip_start": ego.get("clip_start", near)}
                meta["near_fade"] = ego.get("near_fade", [0.05, 0.15])
                meta["plate"] = str(plate_f)
            else:
                c = view_cams[view]
                meta["camera"] = {**tracks[view][0], **{k: c[k] for k in lens},
                                  "clip_start": c.get("clip_start", near)}
                meta["camera_track"] = tracks[view]
                meta["near_fade"] = c.get("near_fade", [0.05, 0.15])
                pdir = vdir / "plates"
                shutil.rmtree(pdir, ignore_errors=True)       # never leave plates of a longer earlier export
                pdir.mkdir()
                for t in range(n):
                    cam_t = {**c, **tracks[view][t]}
                    cv2.imwrite(str(pdir / f"frame{t:05d}.png"),
                                reproject_plate(plate, ego, cam_t, table, room, W, H))
                meta["plates_dir"] = str(pdir)
            blob = json.dumps(meta, indent=1)
            key = render_key(meta, T)
            kf = vdir / "cache_key.txt"
            if not kf.exists() or kf.read_text() != key:      # renders of anything else: drop them
                for f in (vdir / "frames").glob("*.png"):
                    f.unlink()
            kf.write_text(key)
            (vdir / "scene.json").write_text(blob)
        log(f"[export] {sdir.name}: {n} frames @ {fps:g} fps, views {views} -> {out}")
