"""Construct a bead pile: a heap of many small beads as ONE body, cheap to collide with.

A loose heap of small parts (beads, buttons, pellets, sequins) that a task draws single items from, but never carries as
a whole. Dozens of free bodies would each need their own contact pairs with every hand geom (and an untied free body
inside one object lies loose), so the heap is one body: every bead is a visual-only cylinder (with an optional dark
hole on top) laid flat in the measured elliptical footprint, and the collision model is a handful of boxes, one per
occupied grid cell, each covering exactly the beads of its cell. The single beads the task picks up are built as their
own free objects and sit in `keep_out` spots carved out of the heap.
"""
from __future__ import annotations

TOOL = {
    "name": "construct_bead_pile",
    "title": "Construct a bead pile",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A heap of many small beads (or similar small parts) as one body: visual bead cylinders laid in a measured elliptical footprint with a measured colour palette, colliding through a few cell boxes, fixed by default, with keep-out spots for separately built free beads.",
    "description": (
        "Lays bead centres on a hexagonal grid (spacing 2 * bead_radius * spacing_frac) inside an ellipse of the given "
        "length (body x) and width (body y), shrunk by one bead radius so every bead lies inside the footprint, drops "
        "the grid points that fall within a keep_out circle (plus one bead radius), and keeps n_beads of them chosen at "
        "random with the seed (all of them when fewer fit; the count built is reported). With layers=2 a second, "
        "offset layer of about a third as many beads lies on top of the inner 60 percent of the footprint. Every bead "
        "is a visual-only cylinder (contype 0, conaffinity 0, density 0) of bead_radius and bead_height, lying flat, "
        "coloured by a seeded draw from the palette (optionally weighted), with an optional dark disk of hole_radius "
        "on top that reads as the bead's hole. Collision: with cell > 0 the footprint is divided into square cells of "
        "that size and every cell that holds a bead centre becomes one box covering the bounding box of its beads' "
        "disks (so the visual beads always lie inside the collision boxes) up to the top of its highest bead; a box "
        "that would reach into a keep_out circle is split along its longer side until none does (a lone bead beside "
        "the circle collides as its own cylinder), so the free beads in the keep_out spots never start inside the "
        "heap's collision; with "
        "cell = 0 every bead is its own collision cylinder. The collision geoms carry the whole mass, which is the "
        "beads' plastic volume (outer cylinder minus hole) times bead_density, so the heap weighs what its beads do. "
        "The body is fixed to the world by default (a heap is not carried; a fixed heap does not slide off as one "
        "lump when a finger brushes it); fixed=False gives it a free joint. The body origin is the footprint centre "
        "on the table (z = 0 at the bead bottoms). Assumes beads lie flat in one or two layers; the loose-bead "
        "dynamics of a real heap (spreading when stirred) are not modelled."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives <name>.xml, <name>_params.json and <name>_beads.json."},
        {"name": "name", "type": "str", "description": "Asset and body name. Default bead_pile"},
        {"name": "length", "type": "float", "description": "Footprint ellipse extent along body x, metres (>= 2 bead diameters). Default 0.12"},
        {"name": "width", "type": "float", "description": "Footprint ellipse extent along body y, metres (>= 2 bead diameters). Default 0.08"},
        {"name": "bead_radius", "type": "float", "description": "Outer radius of one bead, metres. Default 0.004"},
        {"name": "bead_height", "type": "float", "description": "Height of one bead lying flat, metres. Default 0.006"},
        {"name": "hole_radius", "type": "float", "description": "Radius of the bead's hole (visual disk on top and the plastic volume), 0 for solid beads. Default 0.002"},
        {"name": "n_beads", "type": "int", "description": "Beads in the bottom layer (fewer when they do not fit). Default 60"},
        {"name": "layers", "type": "int", "description": "1 or 2 layers of beads. Default 1"},
        {"name": "spacing_frac", "type": "float", "description": "Grid spacing as a multiple of the bead diameter (>= 1.0, no overlap). Default 1.05"},
        {"name": "palette", "type": "list[list[3 or 4]]", "description": "Bead colours, rgb or rgba in 0..1 (e.g. measure_object_color's palette). Default six bright colours"},
        {"name": "palette_weights", "type": "list[float] | None", "description": "Relative frequency of each palette colour; None for equal. Default None"},
        {"name": "hole_rgba", "type": "list[4]", "description": "Colour of the hole disk. Default [0.08, 0.06, 0.05, 1]"},
        {"name": "cell", "type": "float", "description": "Collision cell size, metres; 0 gives one collision cylinder per bead. Default 0.025"},
        {"name": "keep_out", "type": "list[[x, y, r]]", "description": "Circles in the body frame (metres) left free of beads and collision, for separately built beads. Default []"},
        {"name": "bead_density", "type": "float", "description": "Density of the bead material, kg/m3. Default 1050"},
        {"name": "friction", "type": "list[3]", "description": "Collision friction triple. Default [1.0, 0.01, 0.001]"},
        {"name": "solref", "type": "list[2]", "description": "Collision solref. Default [0.004, 1.0]"},
        {"name": "fixed", "type": "bool", "description": "True: no joint, fixed where staged; False: a free joint. Default True"},
        {"name": "seed", "type": "int", "description": "Random seed for the bead subset and colours. Default 0"},
        {"name": "pos", "type": "list[3]", "description": "Body position in the asset frame. Default [0, 0, 0]"},
        {"name": "yaw_deg", "type": "float", "description": "Body yaw about z, degrees. Default 0"},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane for stand-alone tests. Default False"},
        {"name": "timestep", "type": "float | None", "description": "Simulation timestep written into the asset, or None. Default None"},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "xml, params_json, beads_json (every bead's x, y, z, rgba), body, joint (None when fixed), collision_geoms, visual_geoms, n_beads (built, all layers), n_collision_geoms, mass_kg, top_z_m, footprint {length_m, width_m}, keep_out."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["beads", "pile", "heap", "granular", "small parts", "low contact cost", "fixed", "parametric", "object reconstruction"],
    "used_in": ["EgoDex thread_unthread_bead_necklace"],
    "status": "tested",
    "entrypoint": "construct_bead_pile",
    "cli": "python -m tool_library.skills.construct_bead_pile out_dir",
}

import json
import math
from pathlib import Path

import numpy as np

DEFAULTS = {
    "name": "bead_pile",
    "length": 0.12,
    "width": 0.08,
    "bead_radius": 0.004,
    "bead_height": 0.006,
    "hole_radius": 0.002,
    "n_beads": 60,
    "layers": 1,
    "spacing_frac": 1.05,
    "palette": [[0.85, 0.15, 0.15], [0.95, 0.45, 0.65], [0.55, 0.25, 0.75], [0.2, 0.45, 0.85], [0.25, 0.7, 0.3], [0.95, 0.8, 0.15]],
    "palette_weights": None,
    "hole_rgba": [0.08, 0.06, 0.05, 1.0],
    "cell": 0.025,
    "keep_out": [],
    "bead_density": 1050.0,
    "friction": [1.0, 0.01, 0.001],
    "solref": [0.004, 1.0],
    "fixed": True,
    "seed": 0,
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "include_floor": False,
    "timestep": None,
}


def _fmt(v):
    return " ".join(f"{float(x):.6g}" for x in v)


def _validate(p):
    unknown = sorted(set(p) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"construct_bead_pile: unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    r, h, hr = float(p["bead_radius"]), float(p["bead_height"]), float(p["hole_radius"])
    if r <= 0 or h <= 0:
        raise ValueError("construct_bead_pile: bead_radius and bead_height must be > 0")
    if not (0 <= hr < r):
        raise ValueError("construct_bead_pile: hole_radius must satisfy 0 <= hole_radius < bead_radius")
    if float(p["length"]) < 4 * r or float(p["width"]) < 4 * r:
        raise ValueError("construct_bead_pile: length and width must each be at least two bead diameters")
    if int(p["n_beads"]) < 1:
        raise ValueError("construct_bead_pile: n_beads must be >= 1")
    if int(p["layers"]) not in (1, 2):
        raise ValueError("construct_bead_pile: layers must be 1 or 2")
    if float(p["spacing_frac"]) < 1.0:
        raise ValueError("construct_bead_pile: spacing_frac must be >= 1.0 (beads would overlap)")
    if float(p["cell"]) < 0 or (0 < float(p["cell"]) < 2 * r):
        raise ValueError("construct_bead_pile: cell must be 0 (per-bead collision) or at least one bead diameter")
    pal = p["palette"]
    if not pal or any(len(c) not in (3, 4) for c in pal):
        raise ValueError("construct_bead_pile: palette must be a non-empty list of rgb or rgba colours")
    w = p["palette_weights"]
    if w is not None and (len(w) != len(pal) or any(float(x) < 0 for x in w) or sum(float(x) for x in w) <= 0):
        raise ValueError("construct_bead_pile: palette_weights must give one non-negative weight per palette colour")
    for k in p["keep_out"]:
        if len(k) != 3 or float(k[2]) <= 0:
            raise ValueError("construct_bead_pile: keep_out entries are [x, y, r] with r > 0")
    if float(p["bead_density"]) <= 0:
        raise ValueError("construct_bead_pile: bead_density must be > 0")


def _hex_points(a, b, s, shrink, off=(0.0, 0.0)):
    """Hexagonal grid points (spacing s) inside the ellipse with semi-axes a - shrink, b - shrink."""
    ea, eb = a - shrink, b - shrink
    if ea <= 0 or eb <= 0:
        return np.zeros((0, 2))
    pts = []
    dy = s * math.sqrt(3) / 2
    ny = int(eb / dy) + 2
    nx = int(ea / s) + 2
    for j in range(-ny, ny + 1):
        y = j * dy + off[1]
        xo = (s / 2 if j % 2 else 0.0) + off[0]
        for i in range(-nx, nx + 1):
            x = i * s + xo
            if (x / ea) ** 2 + (y / eb) ** 2 <= 1.0:
                pts.append((x, y))
    return np.asarray(pts, float).reshape(-1, 2)


def _drop_keep_out(pts, keep_out, r):
    if not len(pts) or not keep_out:
        return pts
    ok = np.ones(len(pts), bool)
    for kx, ky, kr in keep_out:
        ok &= np.hypot(pts[:, 0] - float(kx), pts[:, 1] - float(ky)) >= float(kr) + r
    return pts[ok]


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = dict(DEFAULTS)
    p.update(params or {})
    _validate(p)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(p["seed"]))
    r, h, hr = float(p["bead_radius"]), float(p["bead_height"]), float(p["hole_radius"])
    a, b = float(p["length"]) / 2, float(p["width"]) / 2
    s = 2 * r * float(p["spacing_frac"])

    base = _drop_keep_out(_hex_points(a, b, s, r), p["keep_out"], r)
    n = min(int(p["n_beads"]), len(base))
    if n < 1:
        raise ValueError("construct_bead_pile: no bead fits the footprint outside keep_out; enlarge length/width or shrink keep_out")
    sel = np.sort(rng.choice(len(base), size=n, replace=False))
    beads = [(float(x), float(y), 0.0) for x, y in base[sel]]
    if int(p["layers"]) == 2:
        top = _drop_keep_out(_hex_points(0.6 * a + r, 0.6 * b + r, s, r, off=(s / 2, s * math.sqrt(3) / 6)), p["keep_out"], r)
        m = min(max(1, n // 3), len(top))
        if m:
            sel2 = np.sort(rng.choice(len(top), size=m, replace=False))
            beads += [(float(x), float(y), h) for x, y in top[sel2]]

    pal = [list(map(float, c)) + ([1.0] if len(c) == 3 else []) for c in p["palette"]]
    w = None if p["palette_weights"] is None else np.asarray(p["palette_weights"], float) / float(sum(p["palette_weights"]))
    colours = [pal[int(i)] for i in rng.choice(len(pal), size=len(beads), p=w)]

    bead_vol = math.pi * (r * r - hr * hr) * h
    mass = bead_vol * float(p["bead_density"]) * len(beads)
    fr, sr = _fmt(p["friction"]), _fmt(p["solref"])

    geoms, col_names, vis_names = [], [], []
    # collision
    cell = float(p["cell"])
    if cell > 0:
        groups = {}
        for (x, y, z) in beads:
            groups.setdefault((math.floor(x / cell), math.floor(y / cell)), []).append((x, y, z))
        keep = [(float(kx), float(ky), float(kr)) for kx, ky, kr in p["keep_out"]]

        def hits_keep_out(x0, x1, y0, y1):
            return any(math.hypot(min(max(kx, x0), x1) - kx, min(max(ky, y0), y1) - ky) < kr for kx, ky, kr in keep)

        def cover(g):
            """Boxes over the beads g that stay out of every keep_out circle: a cell's bounding box that reaches into a
            circle is split along its longer side until it does not; a lone bead next to a circle gets its own
            cylinder (its disk, unlike its bounding square, keeps the bead-to-circle clearance)."""
            x0, x1 = g[:, 0].min() - r, g[:, 0].max() + r
            y0, y1 = g[:, 1].min() - r, g[:, 1].max() + r
            if not hits_keep_out(x0, x1, y0, y1):
                return [("box", x0, x1, y0, y1, g[:, 2].max() + h)]
            if len(g) == 1:
                return [("cyl", float(g[0, 0]), float(g[0, 1]), float(g[0, 2]))]
            ax = 0 if (x1 - x0) >= (y1 - y0) else 1
            order = np.argsort(g[:, ax], kind="stable")
            half = len(g) // 2
            return cover(g[order[:half]]) + cover(g[order[half:]])

        parts = []
        for key in sorted(groups):
            parts += cover(np.asarray(groups[key]))
        vol = sum((q[2] - q[1]) * (q[4] - q[3]) * q[5] if q[0] == "box" else math.pi * r * r * h for q in parts)
        dens = mass / vol
        for i, q in enumerate(parts):
            gn = f"{name}_cell{i:02d}"
            if q[0] == "box":
                _, x0, x1, y0, y1, z1 = q
                geoms.append(f'      <geom name="{gn}" type="box" size="{_fmt([(x1 - x0) / 2, (y1 - y0) / 2, z1 / 2])}" '
                             f'pos="{_fmt([(x0 + x1) / 2, (y0 + y1) / 2, z1 / 2])}" density="{dens:.6g}" friction="{fr}" '
                             f'solref="{sr}" condim="4" group="3" rgba="0.5 0.5 0.5 0"/>')
            else:
                _, x, y, z = q
                geoms.append(f'      <geom name="{gn}" type="cylinder" size="{_fmt([r, h / 2])}" pos="{_fmt([x, y, z + h / 2])}" '
                             f'density="{dens:.6g}" friction="{fr}" solref="{sr}" condim="4" group="3" rgba="0.5 0.5 0.5 0"/>')
            col_names.append(gn)
    else:
        dens = mass / (math.pi * r * r * h * len(beads))
        for i, (x, y, z) in enumerate(beads):
            gn = f"{name}_beadcol{i:03d}"
            geoms.append(f'      <geom name="{gn}" type="cylinder" size="{_fmt([r, h / 2])}" pos="{_fmt([x, y, z + h / 2])}" '
                         f'density="{dens:.6g}" friction="{fr}" solref="{sr}" condim="4" group="3" rgba="0.5 0.5 0.5 0"/>')
            col_names.append(gn)
    # visuals
    eps = min(0.0002, h * 0.05)
    for i, ((x, y, z), c) in enumerate(zip(beads, colours)):
        gn = f"{name}_bead{i:03d}"
        geoms.append(f'      <geom name="{gn}" type="cylinder" size="{_fmt([r - eps, h / 2 - eps])}" pos="{_fmt([x, y, z + h / 2])}" '
                     f'contype="0" conaffinity="0" density="0" group="1" rgba="{_fmt(c)}"/>')
        vis_names.append(gn)
        if hr > 0:
            hn = f"{name}_hole{i:03d}"
            geoms.append(f'      <geom name="{hn}" type="cylinder" size="{_fmt([hr, eps])}" pos="{_fmt([x, y, z + h - 2 * eps])}" '
                         f'contype="0" conaffinity="0" density="0" group="1" rgba="{_fmt(p["hole_rgba"])}"/>')
            vis_names.append(hn)

    yaw = math.radians(float(p["yaw_deg"]))
    quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    joint = None if p["fixed"] else f"{name}_free"
    jxml = "" if p["fixed"] else f'      <freejoint name="{joint}"/>\n'
    opt = f'  <option timestep="{float(p["timestep"]):.6g}"/>\n' if p["timestep"] else ""
    floor = ('    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.8 0.8 0.8 1"/>\n' if p["include_floor"] else "")
    xml = (f'<mujoco model="{name}">\n  <compiler angle="radian"/>\n{opt}  <worldbody>\n{floor}'
           f'    <body name="{name}" pos="{_fmt(p["pos"])}" quat="{_fmt(quat)}">\n{jxml}' + "\n".join(geoms) +
           '\n    </body>\n  </worldbody>\n</mujoco>\n')
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    pj = out_dir / f"{name}_params.json"
    pj.write_text(json.dumps(p, indent=1))
    bj = out_dir / f"{name}_beads.json"
    bj.write_text(json.dumps([{"x": x, "y": y, "z": z, "rgba": c} for (x, y, z), c in zip(beads, colours)]))
    return {"xml": str(xml_path), "params_json": str(pj), "beads_json": str(bj), "body": name, "joint": joint,
            "collision_geoms": col_names, "visual_geoms": vis_names, "n_beads": len(beads),
            "n_collision_geoms": len(col_names), "mass_kg": mass, "top_z_m": max(z for _, _, z in beads) + h,
            "footprint": {"length_m": 2 * a, "width_m": 2 * b}, "keep_out": [list(map(float, k)) for k in p["keep_out"]]}


def construct_bead_pile(out_dir, **params):
    """Build the bead pile into out_dir (any key of DEFAULTS; unknown keys raise). Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import sys as _sys
    print(json.dumps(construct_bead_pile(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
