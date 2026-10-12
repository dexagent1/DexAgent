"""Step 4: path-trace one view of one sample with Blender Cycles.

    python -m stage4.render_cycles <view_dir> [i/K]

Reads <view_dir>/scene.json, writes RGBA frames to <view_dir>/frames/. The
film is transparent: only the robot, the objects and the shadows they cast on
the table are drawn, so the clean plate shows through everywhere else.
With i/K only the i-th of K equal frame slices is rendered, so several
processes can share one GPU. Frames already on disk are skipped.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def parse_obj(path):
    verts, uvs, faces, fuvs = [], [], [], []
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append(tuple(float(x) for x in p[1:4]))
        elif p[0] == "vt":
            uvs.append(tuple(float(x) for x in p[1:3]))
        elif p[0] == "f":
            idx = [q.split("/") for q in p[1:4]]
            faces.append(tuple(int(q[0]) - 1 for q in idx))
            if len(idx[0]) > 1 and idx[0][1]:
                fuvs.append(tuple(int(q[1]) - 1 for q in idx))
    return verts, uvs, faces, fuvs


def principled(bpy, name, mat_params, texture=None, transmission=0.0, near_fade=None):
    """A Principled BSDF material.

    transmission  a see-through surface, as a mix with a transparent shader
                  (no refraction: a stack of solid geoms would turn into noise).
    near_fade     robot surfaces only: [fully_faded_m, fully_solid_m]. Back
                  faces glow a dim gray so an open link interior does not read
                  as a black void, and surfaces closer to the lens than
                  fully_solid_m dissolve. [0, 0] keeps the back-face part only.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    out = nt.nodes["Material Output"]
    bsdf.inputs["Metallic"].default_value = float(mat_params.get("metallic", 0.0))
    bsdf.inputs["Roughness"].default_value = float(mat_params.get("rough", 0.65))

    def set_spec(v):
        for nm in ("Specular IOR Level", "Specular"):
            if nm in bsdf.inputs:
                bsdf.inputs[nm].default_value = float(v)
                return

    if mat_params.get("spec") is not None:
        set_spec(mat_params["spec"])
    if texture:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*mat_params["base"][:3], 1.0)
    surface = bsdf.outputs["BSDF"]
    if transmission > 0.0:
        clear = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = float(transmission)
        nt.links.new(surface, mix.inputs[1])
        nt.links.new(clear.outputs["BSDF"], mix.inputs[2])
        surface = mix.outputs["Shader"]
        set_spec(1.0)
        bsdf.inputs["Roughness"].default_value = min(float(mat_params.get("rough", 0.65)), 0.12)
        mat.use_backface_culling = False
    if near_fade is not None:
        dim = nt.nodes.new("ShaderNodeEmission")
        dim.inputs["Color"].default_value = (0.05, 0.05, 0.055, 1.0)
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        mix_bf = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(geo.outputs["Backfacing"], mix_bf.inputs["Fac"])
        nt.links.new(surface, mix_bf.inputs[1])
        nt.links.new(dim.outputs["Emission"], mix_bf.inputs[2])
        surface = mix_bf.outputs["Shader"]
        if float(near_fade[1]) > 0.0:
            cam = nt.nodes.new("ShaderNodeCameraData")
            rng = nt.nodes.new("ShaderNodeMapRange")
            rng.inputs["From Min"].default_value = float(near_fade[0])
            rng.inputs["From Max"].default_value = float(near_fade[1])
            rng.inputs["To Min"].default_value = 1.0
            rng.inputs["To Max"].default_value = 0.0
            rng.clamp = True
            nt.links.new(cam.outputs["View Distance"], rng.inputs["Value"])
            trans = nt.nodes.new("ShaderNodeBsdfTransparent")
            mix_nf = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(rng.outputs["Result"], mix_nf.inputs["Fac"])
            nt.links.new(surface, mix_nf.inputs[1])
            nt.links.new(trans.outputs["BSDF"], mix_nf.inputs[2])
            surface = mix_nf.outputs["Shader"]
    nt.links.new(surface, out.inputs["Surface"])
    return mat


def primitive_uv(mode, gtype, co, size, repeat=(1.0, 1.0)):
    """Texture coordinate of a point co (geom frame) on a textured primitive, by export.PRIMITIVE_UV's mode."""
    x, y, z = co
    if mode == "planar_xy":                                  # box: flat across x and y
        u, v = (x / max(size[0], 1e-9) + 1) / 2, (y / max(size[1], 1e-9) + 1) / 2
    elif mode == "cylindrical":                              # around the axis, along its height
        u = (math.atan2(y, x) / (2 * math.pi)) % 1.0
        v = (z / max(size[1] + (size[0] if gtype == "capsule" else 0.0), 1e-9) + 1) / 2
    else:                                                    # spherical: longitude, latitude
        r = math.sqrt(x * x + y * y + z * z) or 1.0
        u = (math.atan2(y, x) / (2 * math.pi)) % 1.0
        v = 0.5 + math.asin(max(-1.0, min(1.0, z / r))) / math.pi
    return u * repeat[0], v * repeat[1]


def add_primitive_uvs(m, e):
    """A UV layer for a textured primitive (box, cylinder, capsule, sphere), from its vertex positions."""
    uvl = m.uv_layers.new()
    rep = e.get("texrepeat", [1.0, 1.0])
    for poly in m.polygons:
        for li in poly.loop_indices:
            co = m.vertices[m.loops[li].vertex_index].co
            uvl.data[li].uv = primitive_uv(e["uv"], e["type"], (co.x, co.y, co.z), e["size"], rep)


def build_mesh(bpy, e):
    import bmesh
    from mathutils import Matrix
    name, kind = e["name"], e["type"]
    m = bpy.data.meshes.new(name)
    if kind == "mesh":
        verts, uvs, faces, fuvs = parse_obj(e["obj"])
        m.from_pydata(verts, [], faces)
        if uvs and fuvs:
            uvl = m.uv_layers.new()
            for poly, fuv in zip(m.polygons, fuvs):
                for li, vt in zip(poly.loop_indices, fuv):
                    uvl.data[li].uv = uvs[vt]
        for p in m.polygons:
            p.use_smooth = True
        return m
    if kind == "box":
        x, y, z = e["size"][:3]
        vs = [(sx * x, sy * y, sz * z) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        m.from_pydata(vs, [], [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                               (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
        if e.get("uv") and e.get("texture"):
            add_primitive_uvs(m, e)
        return m
    bm = bmesh.new()
    r = e["size"][0]
    if kind == "sphere":
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r)
    elif kind == "ellipsoid":
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0,
                                  matrix=Matrix.Diagonal((*e["size"][:3], 1.0)))
    else:                                              # cylinder / capsule
        hl = e["size"][1]
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=r, radius2=r,
                              depth=2 * hl)
        if kind == "capsule":
            for zc in (-hl, hl):
                bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r,
                                          matrix=Matrix.Translation((0, 0, zc)))
    bm.to_mesh(m)
    bm.free()
    if e.get("uv") and e.get("texture"):
        add_primitive_uvs(m, e)
    if kind != "cylinder":
        for p in m.polygons:
            p.use_smooth = True
    return m


def lens_settings(cam, W, H, sensor_height=24.0):
    """Blender camera settings that reproduce a pinhole camera on a W x H render.

    cam holds fx, fy, cx, cy in pixels of the W x H image (pixel (i, j) covers [i, i+1) x [j, j+1), so a centred
    principal point is (W/2, H/2)), or only fovy (degrees: square pixels, centred). With the sensor fitted
    vertically, H pixels span sensor_height / lens, so lens = fy * sensor_height / H; fx != fy is a non-square
    pixel aspect (pixel_aspect_x / pixel_aspect_y = fy / fx, each at least 1, as Blender requires); and the
    principal point is a lens shift, in units of the fitted (vertical) frame size: shift_y = (cy - H/2) / H and,
    horizontally, where one unit is H * fx / fy pixels, shift_x = (W/2 - cx) * fy / (fx * H)."""
    if cam.get("fx") is not None and cam.get("fy") is not None:
        fx, fy = float(cam["fx"]), float(cam["fy"])
        cx = float(cam["cx"]) if cam.get("cx") is not None else W / 2.0
        cy = float(cam["cy"]) if cam.get("cy") is not None else H / 2.0
    else:
        fx = fy = (H / 2.0) / math.tan(math.radians(float(cam["fovy"])) / 2.0)
        cx, cy = W / 2.0, H / 2.0
    return {"sensor_fit": "VERTICAL", "sensor_height": float(sensor_height),
            "lens": fy * float(sensor_height) / H,
            "pixel_aspect_x": max(1.0, fy / fx), "pixel_aspect_y": max(1.0, fx / fy),
            "shift_x": (W / 2.0 - cx) * fy / (fx * H), "shift_y": (cy - H / 2.0) / H}


def set_intrinsics(sc, cd, cam, W, H):
    """Apply lens_settings to the camera data cd and the scene's render settings sc."""
    s = lens_settings(cam, W, H)
    cd.sensor_fit = s["sensor_fit"]
    cd.sensor_height = s["sensor_height"]
    cd.lens = s["lens"]
    cd.shift_x, cd.shift_y = s["shift_x"], s["shift_y"]
    sc.render.pixel_aspect_x, sc.render.pixel_aspect_y = s["pixel_aspect_x"], s["pixel_aspect_y"]
    return s


def build_scene(bpy, meta):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = int(meta["samples"])
    sc.cycles.use_denoising = True
    sc.render.use_persistent_data = True               # keep BVH and textures across frames
    sc.render.film_transparent = True
    sc.render.resolution_x, sc.render.resolution_y = meta["width"], meta["height"]
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "Standard"       # no filmic curve on a composite
    prefs = bpy.context.preferences.addons["cycles"].preferences
    device = "CPU"
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            if any(d.type == kind for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == kind              # GPU only
                sc.cycles.device, device = "GPU", kind
                break
        except Exception:
            continue
    if device == "OPTIX":
        try:
            sc.cycles.denoiser = "OPTIX"               # the default denoises on the CPU
        except Exception:
            pass
    print(f"[render] cycles device: {device}", flush=True)

    # camera: MuJoCo and Blender share the convention (-Z forward, +Y up)
    cam = meta["camera"]
    cd = bpy.data.cameras.new("cam")
    set_intrinsics(sc, cd, cam, int(meta["width"]), int(meta["height"]))
    cd.clip_start = float(cam.get("clip_start", 0.02))
    cd.clip_end = float(cam.get("clip_end", 100.0))
    co = bpy.data.objects.new("cam", cd)
    co.rotation_mode = "QUATERNION"
    co.location, co.rotation_quaternion = cam["pos"], cam["quat_wxyz"]
    sc.collection.objects.link(co)
    sc.camera = co
    prev = None
    for t, c in enumerate(meta.get("camera_track") or []):
        q = list(c["quat_wxyz"])
        if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0:
            q = [-v for v in q]
        prev = q
        co.location, co.rotation_quaternion = c["pos"], q
        co.keyframe_insert("location", frame=t)
        co.keyframe_insert("rotation_quaternion", frame=t)

    # lights: a soft sun, a fill pinned in the world so every view is lit the
    # same, and an ambient dome tinted like the table
    L = meta["lighting"]
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy, sun.data.angle = L["sun_energy"], L["sun_angle"]
    sun.rotation_euler = L["sun_euler"]
    sc.collection.objects.link(sun)
    fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "POINT"))
    fill.data.energy = L["fill_energy"]
    fill.data.shadow_soft_size = 0.5
    fill.location = meta["fill_pos"]
    sc.collection.objects.link(fill)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*[min(1.0, c * 1.1) for c in L["world_color"][:3]], 1.0)
    bg.inputs[1].default_value = L["world_strength"]
    sc.world = world

    # the table top catches shadows and is otherwise invisible
    tab = meta["table"]
    m = bpy.data.meshes.new("table")
    sx, sy = tab["half_size"][0] * 1.4, tab["half_size"][1] * 1.4
    m.from_pydata([(-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("table", m)
    ob.location = (tab["center"][0], tab["center"][1], tab["top_z"])
    ob.is_shadow_catcher = True
    sc.collection.objects.link(ob)

    objs, cache = [], {}
    for e in meta["geoms"]:
        key = json.dumps([e["material"], e.get("texture"), e.get("transmission", 0.0),
                          e["robot"]], sort_keys=True)
        if key not in cache:
            cache[key] = principled(
                bpy, f"mat{len(cache)}", e["material"], texture=e.get("texture"),
                transmission=float(e.get("transmission", 0.0)),
                near_fade=meta.get("near_fade", [0.05, 0.15]) if e["robot"] else None)
        ob = bpy.data.objects.new(e["name"], build_mesh(bpy, e))
        ob.data.materials.append(cache[key])
        ob.rotation_mode = "QUATERNION"
        sc.collection.objects.link(ob)
        objs.append(ob)
    return sc, objs


def main():
    view = Path(sys.argv[1])
    meta = json.loads((view / "scene.json").read_text())
    T = np.load(meta["transforms"])["T"]
    n = int(meta["n_frames"])
    lo, hi = 0, n
    if len(sys.argv) > 2:
        i, k = (int(x) for x in sys.argv[2].split("/"))
        lo, hi = (n * i) // k, (n * (i + 1)) // k
    frames = view / "frames"
    frames.mkdir(exist_ok=True)
    if all((frames / f"f{t:05d}.png").exists() for t in range(lo, hi)):
        print(f"[render] {view}: frames {lo}..{hi - 1} already rendered", flush=True)
        return

    import bpy
    from mathutils import Matrix
    t0 = time.time()
    sc, objs = build_scene(bpy, meta)
    # keyframe every geom once and let Blender's animation loop render: calling
    # render per frame would re-sync the whole scene each time
    for k, ob in enumerate(objs):
        if np.allclose(T[:, k], T[0, k], atol=1e-9):    # static geom: one key
            ob.location = T[0, k, :3, 3]
            ob.rotation_quaternion = Matrix(T[0, k, :3, :3].tolist()).to_quaternion()
            continue
        prev = None
        for t in range(n):
            q = np.array(Matrix(T[t, k, :3, :3].tolist()).to_quaternion())
            if prev is not None and np.dot(q, prev) < 0:
                q = -q                                  # hemisphere continuity
            prev = q
            ob.location, ob.rotation_quaternion = T[t, k, :3, 3], q
            ob.keyframe_insert("location", frame=t)
            ob.keyframe_insert("rotation_quaternion", frame=t)
    sc.frame_start, sc.frame_end = lo, hi - 1
    sc.render.filepath = str(frames / "f#####")
    sc.render.use_overwrite = False                     # skip frames already on disk
    sc.render.use_placeholder = False
    print(f"[render] {view}: {len(objs)} geoms keyframed in {time.time() - t0:.0f}s; "
          f"rendering frames {lo}..{hi - 1}", flush=True)
    bpy.ops.render.render(animation=True)
    print(f"[render] {view}: frames {lo}..{hi - 1} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
