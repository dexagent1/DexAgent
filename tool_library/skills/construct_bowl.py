"""Construct a parametric bowl: a wide hollow vessel with a curved wall whose
inside is really empty in simulation.

A bowl is measured as its rim radius, base radius and height; the wall is a
circular-arc profile (a straight frustum with curvature=0) between them. The
same hollow collision model as construct_cup (base disk + convex wall
wedges) means a cup lowered into the bowl goes in and rests on the bowl's
floor instead of floating on a sealed hull.

    from tool_library.skills.construct_bowl import construct_bowl
    built = construct_bowl(out_dir, rim_radius=0.08, base_radius=0.035, height=0.055)
"""
from __future__ import annotations

TOOL = {
    "name": "construct_bowl",
    "title": "Construct bowl",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "Parametric bowl from rim radius, base radius, height and a wall curvature, "
               "with the hollow wedge collision model so objects can be placed inside.",
    "description": (
        "Generates a curved (radius, height) wall profile, a circular arc bulging outward by "
        "`curvature` x the wall span (0 = straight frustum), and hands it to the vessel "
        "builder behind construct_cup: closed shell with lip, interior lining, base disk plus "
        "n_wedges convex wall wedges for contact, freejoint, sites at the rim centre and the "
        "inner floor. Defaults are a 16 cm rice bowl. Use it with construct_cup to rebuild a "
        "cup-into-bowl scene whose placement is a real containment."
    ),
    "inputs": [
        {"name": "out_dir", "type": "str", "description": "Directory for the XML, meshes and <name>_params.json."},
        {"name": "preview", "type": "bool", "description": "Also render two stills. Default False."},
        {"name": "name", "type": "str", "description": "Asset/body name. Default 'bowl'."},
        {"name": "rim_radius", "type": "float, metres", "description": "Outer radius at the mouth. Default 0.08."},
        {"name": "base_radius", "type": "float, metres", "description": "Outer radius of the foot. Default 0.035."},
        {"name": "height", "type": "float, metres", "description": "Height. Default 0.055."},
        {"name": "curvature", "type": "float", "description": "Outward bulge of the wall as a fraction of its span, 0..0.5. Default 0.25."},
        {"name": "wall", "type": "float, metres", "description": "Wall thickness. Default 0.003."},
        {"name": "bottom", "type": "float, metres", "description": "Base thickness. Default 0.004."},
        {"name": "lip", "type": "float, metres", "description": "Rolled lip proud of the wall at the rim. Default 0.001."},
        {"name": "density", "type": "float, kg/m3", "description": "Used when mass is None. Default 600."},
        {"name": "n_wedges", "type": "int", "description": "Convex wall wedges. Default 20."},
        {"name": "body_rgba / inner_rgba", "type": "list[float]", "description": "Measured outer / inner colours."},
        {"name": "mass", "type": "float | None", "description": "Mass in kg; None uses density."},
        {"name": "pos / yaw_deg / fixed / include_floor", "type": "...", "description": "As construct_cup."},
    ],
    "outputs": [
        {"name": "xml / params_json / body / sites / mass_kg", "type": "...", "description": "As construct_cup."},
        {"name": "height_m / rim_radius_m / base_radius_m / inner_rim_radius_m / bottom_inner_z_m", "type": "float", "description": "Geometry of what was built."},
    ],
    "status": "tested",
    "entrypoint": "construct_bowl",
    "requirements": ["mujoco", "numpy", "trimesh", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ['bowl', 'vessel', 'hollow', 'parametric', 'pick-and-place', 'reconstruct', 'object reconstruction'],
    "used_in": ["Place Cup in Bowl"],
    "cli": "python -m tool_library.skills.construct_bowl <out_dir>",
}

import numpy as np

from tool_library.common import vessel as _core

BOWL_DEFAULTS = {"name": "bowl", "rim_radius": 0.08, "base_radius": 0.035, "height": 0.055,
                 "curvature": 0.25, "wall": 0.003, "bottom": 0.004, "lip": 0.001,
                 "n_wedges": 20, "body_rgba": [0.92, 0.92, 0.9, 1.0],
                 "inner_rgba": [0.96, 0.96, 0.95, 1.0], "density": 600.0}
DEFAULTS = {**_core.DEFAULTS, **BOWL_DEFAULTS}   # every key is a parameter of construct_bowl(**params)


def bowl_profile(rim_radius: float, base_radius: float, height: float,
                 curvature: float = 0.25, n: int = 14) -> list:
    """Outer wall from the foot to the rim as a bulged arc."""
    t = np.linspace(0.0, 1.0, int(n))
    r = base_radius + (rim_radius - base_radius) * t
    z = height * t
    # bulge outward along the chord normal
    chord = np.array([rim_radius - base_radius, height])
    nrm = np.array([chord[1], -chord[0]]) / (np.linalg.norm(chord) + 1e-12)
    bulge = float(curvature) * np.linalg.norm(chord) * np.sin(np.pi * t)
    r = r + bulge * nrm[0]
    z = z + bulge * nrm[1]
    z[0], z[-1] = 0.0, height
    z = np.maximum.accumulate(np.maximum(z, 0.0))
    return [[float(a), float(b)] for a, b in zip(r, z)]


def construct_bowl(out_dir, preview: bool = False, **params) -> dict:
    p = dict(BOWL_DEFAULTS)
    p.update(params)
    curvature = float(p.pop("curvature"))
    if not (0.0 <= curvature <= 0.5):
        raise ValueError("curvature must be within 0..0.5")
    rim, base, h = float(p.pop("rim_radius")), float(p.pop("base_radius")), float(p.pop("height"))
    p["profile"] = bowl_profile(rim, base, h, curvature)
    p["base_radius"], p["rim_radius"], p["height"] = base, rim, h
    built = _core.build(p, out_dir)
    built["curvature"] = curvature
    if preview:
        built["stills"] = _core.preview(built, out_dir)
    return built


def rebuild(params_json, out_dir=None) -> dict:
    return _core.rebuild(params_json, out_dir)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(construct_bowl(sys.argv[1], preview=True), indent=1))
