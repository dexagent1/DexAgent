"""Construct asset from a scanned mesh: Wraps a reconstructed textured mesh (for example from SAM 3D) as an attachable MJCF body with metric scale, corrected up axis, base-centred origin and honest collision geometry.

TEXTURED-MESH ASSET — a reconstructed (SAM3D) mesh as an attachable MJCF body.

WHY THIS EXISTS. The parametric builders in this package author geometry from
numbers; a real object with a texture worth keeping (a mug with a logo, a
patterned box) comes out of SAM3D as an .obj + material png instead. This
builder is the bridge: it BAKES the reconstruction into the package's asset
contract — metric scale applied, up-axis corrected, origin at the base centre,
`<name>_params.json` beside the asset — so a scene builder (stage_on_bench,
build_tabletop_world) attaches it exactly like a bottle or a carton.

COLLISION IS NOT THE VISUAL. MuJoCo collides a mesh by its convex hull, and a
convex hull of a mug is a PLUG: nothing can ever go inside it, and a fingertip
"grasping the rim" closes on solid matter that is not there. So collision is a
choice, stated per asset:

    "vessel"  a hollow body of revolution measured off the mesh itself —
              a ring of boxes for the wall (same trick as the bottle cap's
              skirt), a disc for the floor, and one box for the handle if the
              mesh has one (detected as the azimuth window whose radius
              sticks out past the barrel). A mug, a cup, a bowl.
    "hull"    the baked mesh itself as one convex collision geom — fine for
              anything genuinely convex-ish that is only ever pushed/carried.
    "box"     the axis-aligned bounding box — cheapest, for scenery.

Every measured number (radius, height, handle azimuth/extent) is returned AND
written into the params json, so a wrong auto-measurement is fixed by editing
numbers and calling `rebuild()`, never by editing code.

    python -m tool_library.skills.construct_mesh_asset <out_dir> '{"mesh_obj": "path/to/mesh.obj"}'  # demo build
"""
from __future__ import annotations

TOOL = {'name': 'construct_mesh_asset',
 'title': 'Construct asset from a scanned mesh',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'Wraps a reconstructed textured mesh (for example from SAM 3D) as an attachable '
            'MJCF body with metric scale, corrected up axis, base-centred origin and honest '
            'collision geometry.',
 'description': 'Takes an .obj with its texture, applies the metric scale and the up-axis '
                'correction, moves the origin to the base centre and writes an MJCF body with '
                'the textured visual mesh and collision geometry chosen by shape: a convex '
                'hull for solid objects, or a measured open vessel (a ring of wall segments '
                'plus a floor) for cups, mugs and bowls, because a convex hull of a mug is a '
                'plug nothing can enter. Mass, friction, colour tint and placement are '
                'parameters; the measured dimensions (height, radius, wall) come back in the '
                'build dictionary.',
 'inputs': [{'name': 'out_dir', 'type': 'path', 'description': 'Folder the asset is written to.'},
            {'name': 'name', 'type': 'str', 'description': 'Asset and body name prefix. Default meshobj'},
            {'name': 'mesh_obj', 'type': 'path', 'description': 'Textured .obj to bake; its .mtl and texture png must sit in the same folder.'},
            {'name': 'scale', 'type': 'float', 'description': "Metric scale factor applied to the mesh (fit_object_poses' own scale for a SAM 3D mesh). Default 1.0"},
            {'name': 'up', 'type': 'str', 'description': "The source mesh's up axis before baking: 'y' for SAM 3D objs, 'z' for an already baked mesh. Default z"},
            {'name': 'z_rot_deg', 'type': 'float', 'description': 'Spin about the up axis baked into the mesh (texture and handle orientation), before yaw. Default 0.0'},
            {'name': 'rot_matrix', 'type': 'list[list[float]] | None', 'description': "A full 3x3 mesh-to-world rotation (row-major nested list), e.g. fit_object_poses' fitted frame-0 rotation; replaces up / z_rot_deg and is levelled: the residual tilt that would lift the object off the table is removed by the smallest rotation taking its fitted up to +Z, so the video's yaw survives. Default None"},
            {'name': 'align', 'type': 'str | None', 'description': "'revolution': ignore the axes above and stand the object on its own measured axis of revolution, mouth up (the fix for a fitted pose that matched the silhouette with the vessel tipped over). Default None"},
            {'name': 'handle_azimuth_deg', 'type': 'float | None', 'description': 'With align: spin about up so the detected handle points at this azimuth (degrees in the asset frame, 0 = +x, 90 = +y), read off the video frame. Default None'},
            {'name': 'pos', 'type': 'list[3]', 'description': 'Position, metres. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg', 'type': 'float', 'description': 'Yaw about +Z, degrees. Default 0.0'},
            {'name': 'free', 'type': 'bool', 'description': 'A free joint (an object the hands move) or bolted (scenery). Default True'},
            {'name': 'mass', 'type': 'float', 'description': 'Mass in kg. Default 0.25'},
            {'name': 'friction', 'type': 'list[3]', 'description': 'Sliding, torsional, rolling friction. Default [1.0, 0.02, 0.001]'},
            {'name': 'contact_solref', 'type': 'list[2]', 'description': 'Contact time constant and damping ratio. Default [0.004, 1.0]'},
            {'name': 'collision', 'type': 'str', 'description': "'vessel' (a wall ring of boxes, a floor disc and the handle: cups, mugs, bowls), 'hull' (the convex hull) or 'box'. Default vessel"},
            {'name': 'wall_segments', 'type': 'int', 'description': 'Vessel: boxes in the wall ring. Default 14'},
            {'name': 'wall_thickness', 'type': 'float', 'description': 'Vessel: wall thickness in metres. Default 0.004'},
            {'name': 'floor_thickness', 'type': 'float', 'description': 'Vessel: floor disc thickness in metres. Default 0.006'},
            {'name': 'body_radius', 'type': 'float | None', 'description': 'Vessel: body radius in metres; None measures it from the mesh. Default None'},
            {'name': 'body_height', 'type': 'float | None', 'description': 'Vessel: body height in metres; None measures it from the mesh. Default None'},
            {'name': 'handle', 'type': 'bool', 'description': 'Vessel: detect a protruding handle and give it collision. Default True'},
            {'name': 'handle_min_frac', 'type': 'float', 'description': 'Vessel: radius / body_radius above which a vertex counts as handle, not wall bumpiness. Default 1.12'},
            {'name': 'rgba', 'type': 'list[4] | None', 'description': "None keeps the mesh's own texture; else a flat colour. Default None"},
            {'name': 'include_floor', 'type': 'bool', 'description': 'Write a floor into the preview scene. Default True'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['mesh', 'scan', 'texture', 'cup', 'bowl', 'collision', 'reconstruct', 'parametric', 'object reconstruction'],
 'used_in': ['Place Cup in Bowl',
             'Multi-Cup Grasping',
             'Place Toy in Drawer',
             'Open the Bottle'],
 'status': 'tested',
 'entrypoint': 'construct_mesh_asset',
 'cli': 'python -m tool_library.skills.construct_mesh_asset out_dir \'{"mesh_obj": "object.obj", '
        "...}'"}


import shutil
from pathlib import Path

import numpy as np


def rotation_aligning(a, b):
    """Minimal rotation R with R @ a = b (unit vectors)."""
    a = np.asarray(a, float) / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, float) / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)

from tool_library.common.mjcf import (FLOOR_XML, LIGHTS_XML, merge_params, mjcf_header,
                                      rgba, save_params, vec)

DEFAULTS = {
    "name": "meshobj",

    # ---- source ----
    "mesh_obj": "",             # textured .obj; its .mtl + texture png must
                                #   sit in the same directory
    "scale": 1.0,               # metric scale factor (fit_object_poses' own
                                #   `scale` for a SAM3D mesh)
    "up": "z",                  # source up-axis BEFORE baking: SAM3D objs are
                                #   y-up ("y"); an already-baked mesh is "z"
    "z_rot_deg": 0.0,           # spin about the up axis baked INTO the mesh
                                #   (texture/handle orientation), before yaw
    "rot_matrix": None,         # a FULL 3x3 rotation (row-major nested list)
                                #   mesh->world, e.g. fit_object_poses' fitted
                                #   frame-0 rotation. When given it REPLACES
                                #   up/z_rot_deg, and is then LEVELLED: the
                                #   residual tilt that would lift the object
                                #   off the table is removed by the smallest
                                #   rotation taking its fitted up to +Z, so
                                #   the video's own yaw (a mug's handle
                                #   direction) survives and the object still
                                #   stands flat.
    "align": None,              # "revolution": IGNORE the axes above and
                                #   stand the object up by its own measured
                                #   axis of revolution (the eigenvector the
                                #   surface normals are all perpendicular
                                #   to), mouth up (the wider / open end).
                                #   THE fix for a fitted pose that matched
                                #   the silhouette with the vessel tipped
                                #   over — a tipped mug's outline still
                                #   scores an IoU of 0.8, its physics is
                                #   still wrong.
    "handle_azimuth_deg": None, # with align: spin about up so the detected
                                #   handle points at this azimuth (deg, in
                                #   the asset frame: 0 = +x, 90 = +y).
                                #   Measure it off the video frame.

    # ---- placement (the standard builder keys) ----
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "free": True,               # freejoint (an object) vs bolted (scenery)

    # ---- physics ----
    "mass": 0.25,
    "friction": [1.0, 0.02, 0.001],
    "contact_solref": [0.004, 1.0],

    # ---- collision ----
    "collision": "vessel",      # "vessel" | "hull" | "box"
    "wall_segments": 14,        # vessel: N boxes in the wall ring
    "wall_thickness": 0.004,    # vessel: wall thickness
    "floor_thickness": 0.006,   # vessel: floor disc thickness
    "body_radius": None,        # vessel: None -> measured from the mesh
    "body_height": None,        # vessel: None -> measured from the mesh
    "handle": True,             # vessel: detect + collide a protruding handle
    "handle_min_frac": 1.12,    # vessel: radius/body_radius above which a
                                #   vertex counts as handle, not wall bumpiness

    # ---- appearance ----
    "rgba": None,               # None -> the mesh's own texture; else flat
    "include_floor": True,
}


# ---------------------------------------------------------------------------
def _bake(p: dict, out_dir: Path):
    """Scale + reorient + recentre the mesh; write it (with its texture)
    next to the asset. Returns (trimesh mesh, relative obj path)."""
    import trimesh
    src = Path(p["mesh_obj"])
    if not src.exists():
        raise FileNotFoundError(f"mesh_obj not found: {src}")
    m = trimesh.load(src, force="mesh", process=False)
    T = np.eye(4)
    if p["align"] == "revolution":
        # the axis every surface normal is perpendicular to: smallest
        # eigenvector of the area-weighted normal covariance
        fn = np.asarray(m.face_normals, float)
        fa = np.asarray(m.area_faces, float)[:, None]
        C = (fn * fa).T @ fn
        w, V = np.linalg.eigh(C)
        axis = V[:, 0] / (np.linalg.norm(V[:, 0]) + 1e-12)
        R0 = rotation_aligning(axis, np.array([0.0, 0.0, 1.0]))
        v = np.asarray(m.vertices, float) @ R0.T
        z = v[:, 2]
        z0, z1 = np.percentile(z, 2), np.percentile(z, 98)
        span = z1 - z0

        def _end(z_lo, z_hi):
            sel = (z >= z_lo) & (z <= z_hi)
            vv = v[sel]
            c = vv[:, :2].mean(0)
            r = np.hypot(vv[:, 0] - c[0], vv[:, 1] - c[1])
            R95 = np.percentile(r, 95)
            interior = float((r < 0.55 * R95).mean())
            return R95, interior

        r_top, int_top = _end(z1 - 0.12 * span, z1)
        r_bot, int_bot = _end(z0, z0 + 0.12 * span)
        # mouth up: the open end is WIDER (vessels taper to the base) and has
        # no interior disc of vertices; both cues vote, radius breaks the tie
        up_score = (r_top - r_bot) / max(r_top, r_bot) + (int_bot - int_top)
        if up_score < 0:
            R0 = trimesh.transformations.rotation_matrix(
                np.pi, [1, 0, 0])[:3, :3] @ R0
        T[:3, :3] = R0
    elif p["rot_matrix"] is not None:
        T[:3, :3] = np.asarray(p["rot_matrix"], float).reshape(3, 3)
    elif str(p["up"]).lower() == "y":          # y-up -> z-up
        T[:3, :3] = trimesh.transformations.rotation_matrix(
            np.pi / 2, [1, 0, 0])[:3, :3]
    m.apply_transform(T)
    if p["rot_matrix"] is not None:
        # LEVEL the fitted pose: the object's own axis, measured as the vector
        # from the bottom-slab centroid to the top-slab centroid of the
        # transformed mesh, is brought to +Z by the smallest rotation. The
        # video's yaw (handle direction) survives; the few-degree fit tilt
        # that would rock the object on its base does not.
        v = np.asarray(m.vertices, float)
        z = v[:, 2]
        lo = v[z <= np.percentile(z, 12)].mean(axis=0)
        hi = v[z >= np.percentile(z, 88)].mean(axis=0)
        axis = hi - lo
        axis /= np.linalg.norm(axis) + 1e-12
        m.apply_transform(np.block(
            [[rotation_aligning(axis, np.array([0.0, 0.0, 1.0])),
              np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
    if float(p["z_rot_deg"]):
        m.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(float(p["z_rot_deg"])), [0, 0, 1]))
    m.apply_scale(float(p["scale"]))
    lo, hi = m.bounds
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    m.apply_translation(-centre)               # origin = base centre
    if p["align"] == "revolution" and p["handle_azimuth_deg"] is not None:
        hd = _vessel_measure(m, {**p, "handle": True}).get("handle")
        if hd is not None:
            m.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(float(p["handle_azimuth_deg"])
                           - hd["azimuth_deg"]), [0, 0, 1]))
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    obj_path = mesh_dir / f"{p['name']}_visual.obj"
    m.export(obj_path)                         # writes .mtl + texture too
    # trimesh names the material file after the obj; make sure the texture
    # actually landed (older trimesh leaves the png behind in some paths)
    for aux in src.parent.glob("*.png"):
        tgt = mesh_dir / aux.name
        if not tgt.exists():
            shutil.copy2(aux, tgt)
    # MuJoCo does NOT read the .mtl: the texture must be declared explicitly
    # in the MJCF. Find the image the export references (map_Kd), else any
    # png that came along.
    tex = None
    for mtl in mesh_dir.glob("*.mtl"):
        for line in mtl.read_text().splitlines():
            if line.strip().lower().startswith("map_kd"):
                cand = mesh_dir / line.split()[-1]
                if cand.exists():
                    tex = cand
    if tex is None:
        pngs = sorted(mesh_dir.glob("*.png"))
        tex = pngs[0] if pngs else None
    return m, obj_path, tex


def _vessel_measure(m, p: dict) -> dict:
    """Radius/height/handle window, measured from the baked vertices."""
    v = np.asarray(m.vertices, float)
    r = np.hypot(v[:, 0], v[:, 1])
    az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))          # -180..180
    height = float(v[:, 2].max())
    nb = 72
    bins = ((az + 180.0) / 360.0 * nb).astype(int).clip(0, nb - 1)
    rmax = np.zeros(nb)
    for b in range(nb):
        sel = bins == b
        if sel.any():
            rmax[b] = np.percentile(r[sel], 98)
    body_r = float(np.median(rmax[rmax > 0]))
    handle = None
    if p["handle"]:
        out = rmax > body_r * float(p["handle_min_frac"])
        if out.any():
            # the widest contiguous run of protruding azimuth bins (wrap-aware)
            idx = np.where(out)[0]
            runs, cur = [], [idx[0]]
            for a, b_ in zip(idx, idx[1:]):
                if b_ - a <= 2:
                    cur.append(b_)
                else:
                    runs.append(cur)
                    cur = [b_]
            runs.append(cur)
            if len(runs) > 1 and runs[0][0] <= 1 and runs[-1][-1] >= nb - 2:
                runs[0] = runs.pop() + runs[0]              # wraps -180/180
            run = max(runs, key=len)
            mid = np.deg2rad((np.mean([(b_ + 0.5) / nb * 360 - 180
                                       for b_ in run])))
            sel = (r > body_r * 1.02) & np.isin(bins, run)
            if sel.sum() >= 8:
                handle = {
                    "azimuth_deg": float(np.degrees(mid)),
                    "r_out": float(np.percentile(r[sel], 98)),
                    "z_lo": float(np.percentile(v[sel, 2], 2)),
                    "z_hi": float(np.percentile(v[sel, 2], 98)),
                    "half_width": float(np.percentile(
                        np.abs(v[sel, 0] * -np.sin(mid)
                               + v[sel, 1] * np.cos(mid)), 90)),
                }
    return {"body_radius": body_r, "body_height": height, "handle": handle}


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m, obj_path, tex = _bake(p, out_dir)
    lo, hi = m.bounds
    ext = hi - lo

    meas = _vessel_measure(m, p) if p["collision"] == "vessel" else {}
    R = float(p["body_radius"] or meas.get("body_radius") or max(ext[:2]) / 2)
    H = float(p["body_height"] or meas.get("body_height") or ext[2])
    fr, sref = vec(p["friction"]), vec(p["contact_solref"])

    coll = []
    if p["collision"] == "vessel":
        wall = float(p["wall_thickness"])
        nseg = int(p["wall_segments"])
        floor_t = float(p["floor_thickness"])
        chord = np.pi * R / nseg
        wall_h = H - floor_t
        for i in range(nseg):
            a = 2 * np.pi * i / nseg
            rr = R - wall / 2
            coll.append(
                f'      <geom name="{name}_w{i}" type="box" '
                f'size="{wall/2:.5f} {chord:.5f} {wall_h/2:.5f}" '
                f'pos="{rr*np.cos(a):.5f} {rr*np.sin(a):.5f} '
                f'{floor_t + wall_h/2:.5f}" euler="0 0 {np.degrees(a):.2f}" '
                f'rgba="0 0 0 0" friction="{fr}" solref="{sref}"/>')
        coll.append(f'      <geom name="{name}_floor" type="cylinder" '
                    f'size="{R - wall*0.6:.5f} {floor_t/2:.5f}" '
                    f'pos="0 0 {floor_t/2:.5f}" rgba="0 0 0 0" '
                    f'friction="{fr}" solref="{sref}"/>')
        hd = meas.get("handle")
        if hd:
            a = np.radians(hd["azimuth_deg"])
            r_mid = (R + hd["r_out"]) / 2
            coll.append(
                f'      <geom name="{name}_handle" type="box" '
                f'size="{max(0.004, (hd["r_out"]-R)/2):.5f} '
                f'{max(0.004, hd["half_width"]):.5f} '
                f'{max(0.006, (hd["z_hi"]-hd["z_lo"])/2):.5f}" '
                f'pos="{r_mid*np.cos(a):.5f} {r_mid*np.sin(a):.5f} '
                f'{(hd["z_lo"]+hd["z_hi"])/2:.5f}" '
                f'euler="0 0 {hd["azimuth_deg"]:.2f}" rgba="0 0 0 0" '
                f'friction="{fr}" solref="{sref}"/>')
    elif p["collision"] == "hull":
        coll.append(f'      <geom name="{name}_hull" type="mesh" '
                    f'mesh="{name}_vis_m" rgba="0 0 0 0" friction="{fr}" '
                    f'solref="{sref}"/>')
    else:                                                    # "box"
        coll.append(f'      <geom name="{name}_bbox" type="box" '
                    f'size="{ext[0]/2:.5f} {ext[1]/2:.5f} {ext[2]/2:.5f}" '
                    f'pos="0 0 {ext[2]/2:.5f}" rgba="0 0 0 0" '
                    f'friction="{fr}" solref="{sref}"/>')

    mass = float(p["mass"])
    izz = 0.5 * mass * R * R
    ixx = mass * (R * R / 4 + H * H / 3)
    free = f'      <freejoint name="{name}_free"/>\n' if p["free"] else ""
    # lights only in the standalone demo build: an ATTACHED asset must not
    # bring its own illumination into a world that owns its look
    floor = (LIGHTS_XML + FLOOR_XML) if p["include_floor"] else ""
    px, py, pz = p["pos"]
    # MuJoCo ignores the OBJ's own .mtl — the texture is an explicit MJCF
    # asset. A flat rgba override wins when given.
    tex_asset, vis_attr = "", 'rgba="0.7 0.7 0.7 1"'
    if p["rgba"]:
        vis_attr = f'rgba="{rgba(p["rgba"])}"'
    elif tex is not None:
        tex_asset = (f'\n    <texture name="{name}_tex" type="2d" '
                     f'file="{tex.name}"/>'
                     f'\n    <material name="{name}_mat" '
                     f'texture="{name}_tex"/>')
        vis_attr = f'material="{name}_mat"'

    header = mjcf_header(f"{name}_meshobj", timestep=0.002, meshdir="meshes")
    header = header.replace('meshdir="meshes"',
                            'meshdir="meshes" texturedir="meshes"')
    xml = header + f"""
  <asset>
    <mesh name="{name}_vis_m" file="{obj_path.name}"/>{tex_asset}
  </asset>

  <worldbody>{floor}
    <body name="{name}" pos="{vec([px, py, pz])}" euler="0 0 {p['yaw_deg']}">
{free}      <inertial pos="0 0 {0.45*H:.4f}" mass="{mass}"
                diaginertia="{ixx:.6f} {ixx:.6f} {izz:.6f}"/>
      <geom name="{name}_v" type="mesh" mesh="{name}_vis_m" {vis_attr}
            contype="0" conaffinity="0" group="2"/>
{chr(10).join(coll)}
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    save_params(out_dir / f"{name}_params.json", p)
    return {
        "xml": str(xml_path),
        "params_json": str(out_dir / f"{name}_params.json"),
        "base_body": name,
        "bodies": {"base": name},
        "extents_m": [round(float(x), 4) for x in ext],
        "measured": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in meas.items()},
        "collision": p["collision"],
        "note": ("origin = base centre, z-up, metric. 'vessel' collision is "
                 "hollow: things can go inside and fingers can take the rim."),
    }


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    from tool_library.common.mjcf import load_params
    p = load_params(params_json)
    return build(p, out_dir or Path(params_json).parent)


# ---------------------------------------------------------------------------


def construct_mesh_asset(out_dir, **params):
    """Bake the mesh named in params into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_mesh_asset(_sys.argv[1] if len(_sys.argv) > 1 else ".", **_json.loads(_sys.argv[2] if len(_sys.argv) > 2 else "{}")), indent=1, default=str))
