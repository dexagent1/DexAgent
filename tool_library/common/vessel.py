"""Parametric hollow vessels (cups, bowls, mugs) as MuJoCo assets.

A vessel is a solid of revolution: a (radius, height) wall profile from the
base to the rim, a wall thickness, a lip and a bottom. The visual is a closed
shell plus an interior lining mesh; the collision is `n_wedges` convex wall
wedges around the rim plus a base disk, so the inside is really empty: a ball
dropped in stays in, a cup lowered into a bowl goes in, and two cups nest.
Convex-hull collision of the shell would seal the mouth. Extracted from the
multi-cup reconstruction and made parametric.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mjcf import merge_params, save_params, rgba as _rgba, vec, mjcf_header, FLOOR_XML, LIGHTS_XML, load

DEFAULTS = {
    "name": "cup",
    # ---- shape: a frustum by default; `profile` overrides it ----
    "base_radius": 0.025,        # outer radius at the table
    "rim_radius":  0.042,        # outer radius at the mouth
    "height":      0.102,
    "profile":     None,         # [(r_outer, z), ...] from base to rim; None -> straight frustum
    "wall":        0.0012,       # wall thickness
    "lip":         0.0015,       # rolled lip proud of the wall at the rim
    "bottom":      0.003,        # base thickness
    "lining":      0.0004,       # inner lining shell thickness (visual)
    "handle":      None,         # {"radius": 0.018, "tube": 0.005, "z_frac": 0.55} for a mug
    "sections":    96,           # revolve resolution
    "n_wedges":    16,           # convex wall wedges for collision
    # ---- appearance ----
    "body_rgba":   [1.0, 0.57, 0.22, 1.0],
    "inner_rgba":  [1.0, 0.78, 0.75, 1.0],
    "specular":    0.3,
    "shininess":   0.3,
    "collision_rgba": [0.8, 0.2, 0.2, 0.0],    # wedges invisible by default
    # ---- physics ----
    "mass":        None,         # kg; None -> density
    "density":     250.0,        # kg/m3 over the wedge volume (thin plastic)
    "friction":    [1.0, 0.005, 0.0001],
    "condim":      4,
    # ---- placement ----
    "pos":         [0.0, 0.0, 0.0],
    "yaw_deg":     0.0,
    "fixed":       False,
    "include_floor": True,
    "timestep":    0.002,
}


def _profile(p: dict) -> np.ndarray:
    if p["profile"] is not None:
        prof = np.asarray(p["profile"], float)
        if prof.ndim != 2 or prof.shape[1] != 2 or len(prof) < 2:
            raise ValueError("profile must be [(r, z), ...] with at least two points")
        if prof[0, 1] != 0.0:
            raise ValueError("profile must start at z = 0 (the base)")
        return prof
    return np.array([[float(p["base_radius"]), 0.0],
                     [float(p["rim_radius"]), float(p["height"])]])


def _revolve(profile_rz, sections: int):
    import trimesh
    m = trimesh.creation.revolve(np.asarray(profile_rz, float), sections=int(sections))
    if m.volume < 0:
        m.invert()
    m.fix_normals()
    return m


def _inner(prof: np.ndarray, wall: float) -> np.ndarray:
    """The profile offset inward by the wall thickness (horizontally)."""
    q = prof.copy()
    q[:, 0] = np.maximum(q[:, 0] - wall, 0.002)
    return q


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = _profile(p)
    H = float(prof[-1, 1])
    wall, lip, bot, lin = (float(p["wall"]), float(p["lip"]), float(p["bottom"]), float(p["lining"]))
    if wall >= prof[:, 0].min():
        raise ValueError("wall thicker than the narrowest radius")
    if bot >= H:
        raise ValueError("bottom thicker than the vessel is tall")
    inner = _inner(prof, wall)
    rt = float(prof[-1, 0])
    # closed shell: outside up the wall, over the lip, down the inside, across the bottom
    outside = [(0.0, 0.0)] + [tuple(q) for q in prof[:-1]] + [(rt, H - 3 * lip), (rt + lip, H - 1.5 * lip), (rt + lip, H)]
    inside = [(rt - wall, H)] + [tuple(q) for q in inner[::-1][1:]]
    inside = [q for q in inside if q[1] > bot] + [(float(inner[0, 0]), bot), (0.0, bot)]
    shell = _revolve(outside + inside, p["sections"])
    # the lining: a thin skin on the inside, so the interior reads as a surface
    lin_prof = ([(0.0, bot)] + [(float(r), max(float(z), bot)) for r, z in inner[:-1]]
                + [(rt - wall, H - 0.0005), (rt - wall - lin, H - 0.0005)]
                + [(float(r) - lin, max(float(z), bot) + lin) for r, z in inner[::-1][1:]]
                + [(0.0, bot + lin)])
    lining = _revolve(lin_prof, p["sections"])
    shell_obj = out_dir / f"{name}_shell.obj"
    lin_obj = out_dir / f"{name}_lining.obj"
    shell.export(shell_obj)
    lining.export(lin_obj)
    # collision: convex wall wedges from the base ring to the lip, following
    # the profile (each wedge's hull is the wall between two rays)
    from scipy.spatial import ConvexHull
    n = int(p["n_wedges"])
    wedge_files = []
    rings = [(float(r), float(z)) for r, z in prof] + [(rt + lip, H)]
    rings_in = [(float(r), float(z)) for r, z in inner]
    for k in range(n):
        a0, a1 = 2 * np.pi * k / n, 2 * np.pi * (k + 1) / n
        am = 0.5 * (a0 + a1)
        pts = []
        for r, z in rings + rings_in:
            for a in (a0, am, a1):
                pts.append([r * np.cos(a), r * np.sin(a), z])
        pts = np.asarray(pts)
        hull = ConvexHull(pts)
        f = out_dir / f"{name}_wedge{k}.obj"
        with open(f, "w") as fh:
            for v in pts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for s in hull.simplices:
                fh.write(f"f {s[0]+1} {s[1]+1} {s[2]+1}\n")
        wedge_files.append(f)
    handle_xml = ""
    hnd = p["handle"]
    if hnd:
        hr, ht = float(hnd.get("radius", 0.018)), float(hnd.get("tube", 0.005))
        zc = float(hnd.get("z_frac", 0.55)) * H
        r_at = float(np.interp(zc, prof[:, 1], prof[:, 0]))
        cx = r_at + hr - ht            # a ring tangent to the wall
        segs = int(hnd.get("segments", 10))
        for i in range(segs):
            t0 = np.pi * (-0.5 + i / segs)
            t1 = np.pi * (-0.5 + (i + 1) / segs)
            a = [cx + hr * np.cos(t0), 0.0, zc + hr * np.sin(t0)]
            b = [cx + hr * np.cos(t1), 0.0, zc + hr * np.sin(t1)]
            handle_xml += (f'\n      <geom name="{name}_handle{i}" type="capsule" size="{ht:.5f}" '
                           f'fromto="{vec(a)} {vec(b)}" rgba="{_rgba(p["body_rgba"])}" '
                           f'density="{float(p["density"]):.1f}" friction="{vec(p["friction"])}" '
                           f'condim="{int(p["condim"])}"/>')
    gattr = (f'density="{float(p["density"]):.1f}" rgba="{_rgba(p["collision_rgba"])}" group="3" '
             f'friction="{vec(p["friction"])}" condim="{int(p["condim"])}"')
    wedges_xml = "".join(
        f'\n      <geom name="{name}_wedge{k}" type="mesh" mesh="{name}_wedge{k}" {gattr}/>'
        for k in range(n))
    free = "" if p["fixed"] else f'\n      <freejoint name="{name}_free"/>'
    mass_fix = ""
    xml = mjcf_header(f"{name}_vessel", timestep=p["timestep"]) + f"""
  <asset>
    <material name="{name}_mat" rgba="{_rgba(p['body_rgba'])}" specular="{float(p['specular']):.3f}" shininess="{float(p['shininess']):.3f}"/>
    <material name="{name}_inmat" rgba="{_rgba(p['inner_rgba'])}" specular="{float(p['specular'])*0.7:.3f}" shininess="{float(p['shininess'])*0.7:.3f}"/>
    <mesh name="{name}_shell" file="{shell_obj.name}"/>
    <mesh name="{name}_lining" file="{lin_obj.name}"/>{"".join(f'{chr(10)}    <mesh name="{name}_wedge{k}" file="{f.name}"/>' for k, f in enumerate(wedge_files))}
  </asset>
  <worldbody>{LIGHTS_XML}{FLOOR_XML if p["include_floor"] else ""}
    <body name="{name}" pos="{vec(p['pos'])}" euler="0 0 {float(p['yaw_deg'])}">{free}
      <geom name="{name}_shell" type="mesh" mesh="{name}_shell" material="{name}_mat" contype="0" conaffinity="0" mass="0" group="1"/>
      <geom name="{name}_lining" type="mesh" mesh="{name}_lining" material="{name}_inmat" contype="0" conaffinity="0" mass="0" group="1"/>
      <geom name="{name}_base" type="cylinder" pos="0 0 {bot/2:.5f}" size="{float(inner[0,0]):.5f} {bot/2:.5f}" {gattr}/>{wedges_xml}{handle_xml}
      <site name="{name}_rim" pos="0 0 {H:.5f}" size="0.002" rgba="0 1 0 0"/>
      <site name="{name}_floor" pos="0 0 {bot:.5f}" size="0.002" rgba="0 0 1 0"/>
    </body>
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    m = load(xml_path)
    if p["mass"] is not None:
        # scale every collision geom's density so the subtree mass is `mass`
        bid = m.body(name).id
        cur = float(m.body_subtreemass[bid])
        k = float(p["mass"]) / max(cur, 1e-9)
        xml = xml.replace(f'density="{float(p["density"]):.1f}"', f'density="{float(p["density"]) * k:.3f}"')
        xml_path.write_text(xml)
        m = load(xml_path)
    save_params(out_dir / f"{name}_params.json", p)
    bid = m.body(name).id
    return {
        "xml": str(xml_path), "params_json": str(out_dir / f"{name}_params.json"),
        "name": name, "body": name, "free_joint": None if p["fixed"] else f"{name}_free",
        "height_m": round(H, 5), "rim_radius_m": round(rt, 5),
        "base_radius_m": round(float(prof[0, 0]), 5), "inner_rim_radius_m": round(rt - wall, 5),
        "bottom_inner_z_m": round(bot, 5),
        "mass_kg": round(float(m.body_subtreemass[bid]), 4),
        "sites": {"rim": f"{name}_rim", "floor": f"{name}_floor"},
        "n_wedges": n, "handle": bool(hnd),
        "note": ("collision = base disk + convex wall wedges: the inside is empty. "
                 "Visual shell/lining are contype 0."),
    }


def rebuild(params_json: str | Path, out_dir: str | Path | None = None) -> dict:
    params_json = Path(params_json)
    return build(json.loads(params_json.read_text()), out_dir or params_json.parent)


def preview(built: dict, out_dir: str | Path) -> list[str]:
    import mujoco
    from .mjcf import free_camera, save_stills
    m = load(built["xml"])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    H = built["height_m"]
    frames = []
    with mujoco.Renderer(m, height=480, width=640) as r:
        for az, el in ((120, -20), (60, -55)):
            cam = free_camera([0, 0, 0.5 * H], 4.0 * H, az, el)
            r.update_scene(d, camera=cam)
            frames.append(r.render().copy())
    return save_stills(frames, out_dir, f"{built['name']}_preview", n=2)
