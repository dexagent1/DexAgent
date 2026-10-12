"""Construct a parametric cup, tumbler or mug: a hollow solid of revolution
whose inside is really empty in simulation.

Measured off a video as base radius, rim radius and height (the frustum the
silhouette fit returns), or given as an arbitrary (radius, height) wall
profile, the cup is built as a closed shell with a rolled lip and an inner
lining for looks, and as a base disk plus convex wall wedges for contact. A
convex hull of the shell would seal the mouth; the wedges leave it open, so
a cup can be filled, nested, or placed into a bowl. An optional handle makes
it a mug. Colours are the measured body and interior tones.

    from tool_library.skills.construct_cup import construct_cup
    built = construct_cup(out_dir, base_radius=0.025, rim_radius=0.042, height=0.102,
                          body_rgba=[1, 0.57, 0.22, 1], inner_rgba=[1, 0.78, 0.75, 1])
"""
from __future__ import annotations

TOOL = {
    "name": "construct_cup",
    "title": "Construct cup",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "Parametric cup/tumbler/mug from base radius, rim radius, height (or a wall "
               "profile) with a hollow collision model: base disk plus convex wall wedges.",
    "description": (
        "Builds a solid of revolution from a (radius, height) wall profile, a straight frustum "
        "between base_radius and rim_radius by default. The visual is a watertight shell with "
        "a rolled lip and a separate interior lining mesh (so the inside has its own colour); "
        "the collision is a base disk plus n_wedges convex hulls of the wall between adjacent "
        "rays, which keeps the mouth open and the interior empty for stacking, filling and "
        "placing in bowls. Mass follows from density over the wedges or is set directly. An "
        "optional handle (ring of capsules tangent to the wall) turns it into a mug. The "
        "body gets a freejoint unless fixed. Sites mark the rim centre and the inner floor. "
        "The same builder underlies construct_bowl with a curved profile."
    ),
    "inputs": [
        {"name": "out_dir", "type": "str", "description": "Directory for the XML, meshes and <name>_params.json."},
        {"name": "preview", "type": "bool", "description": "Also render two stills. Default False."},
        {"name": "name", "type": "str", "description": "Asset/body name. Default 'cup'."},
        {"name": "base_radius", "type": "float, metres", "description": "Outer radius at the table. Default 0.025."},
        {"name": "rim_radius", "type": "float, metres", "description": "Outer radius at the mouth. Default 0.042."},
        {"name": "height", "type": "float, metres", "description": "Height. Default 0.102."},
        {"name": "profile", "type": "list[[r, z]] | None", "description": "Explicit outer wall profile from z=0 to the rim; overrides the frustum."},
        {"name": "wall", "type": "float, metres", "description": "Wall thickness. Default 0.0012."},
        {"name": "lip", "type": "float, metres", "description": "Rolled lip proud of the wall. Default 0.0015."},
        {"name": "bottom", "type": "float, metres", "description": "Base thickness. Default 0.003."},
        {"name": "handle", "type": "dict | None", "description": "{'radius', 'tube', 'z_frac', 'segments'} for a mug handle. Default None."},
        {"name": "n_wedges", "type": "int", "description": "Convex wall wedges. Default 16."},
        {"name": "body_rgba", "type": "list[float]", "description": "Outer colour (measured). Default orange."},
        {"name": "inner_rgba", "type": "list[float]", "description": "Interior colour (measured). Default pale."},
        {"name": "lining", "type": "float, metres", "description": "Inner lining shell thickness (visual). Default 0.0004."},
        {"name": "sections", "type": "int", "description": "Revolve resolution of the visual meshes. Default 96."},
        {"name": "specular", "type": "float", "description": "Material specular. Default 0.3."},
        {"name": "shininess", "type": "float", "description": "Material shininess. Default 0.3."},
        {"name": "collision_rgba", "type": "list[float]", "description": "Colour of the collision wedges; alpha 0 hides them. Default invisible."},
        {"name": "density", "type": "float, kg/m3", "description": "Used when mass is None. Default 250."},
        {"name": "friction", "type": "list[float]", "description": "MuJoCo friction triple. Default [1.0, 0.005, 0.0001]."},
        {"name": "condim", "type": "int", "description": "Contact dimensionality. Default 4."},
        {"name": "timestep", "type": "float, s", "description": "Simulation timestep written into the asset. Default 0.002."},
        {"name": "mass", "type": "float | None", "description": "Mass in kg; None uses density 250 over the wedges."},
        {"name": "pos", "type": "list[float]", "description": "Base centre on the table. Default [0, 0, 0]."},
        {"name": "yaw_deg", "type": "float", "description": "Yaw (matters for a mug). Default 0."},
        {"name": "fixed", "type": "bool", "description": "No freejoint. Default False."},
        {"name": "include_floor", "type": "bool", "description": "Floor and lights in the asset. Default True."},
    ],
    "outputs": [
        {"name": "xml", "type": "str", "description": "The MJCF asset."},
        {"name": "params_json", "type": "str", "description": "Parameters for rebuild()."},
        {"name": "body", "type": "str", "description": "Body name; free_joint is '<name>_free' unless fixed."},
        {"name": "height_m / rim_radius_m / base_radius_m / inner_rim_radius_m / bottom_inner_z_m", "type": "float", "description": "Measured geometry of what was built."},
        {"name": "mass_kg", "type": "float", "description": "Subtree mass of the compiled asset."},
        {"name": "sites", "type": "dict", "description": "rim and floor site names."},
        {"name": "stills", "type": "list[str]", "description": "Preview PNGs when preview=True."},
    ],
    "status": "tested",
    "entrypoint": "construct_cup",
    "requirements": ["mujoco", "numpy", "trimesh", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ['cup', 'mug', 'vessel', 'hollow', 'parametric', 'pick-and-place', 'reconstruct', 'object reconstruction'],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping"],
    "cli": "python -m tool_library.skills.construct_cup <out_dir>",
}

from tool_library.common import vessel as _core


def construct_cup(out_dir, preview: bool = False, **params) -> dict:
    built = _core.build(dict(params), out_dir)
    if preview:
        built["stills"] = _core.preview(built, out_dir)
    return built


rebuild = _core.rebuild
DEFAULTS = _core.DEFAULTS


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(construct_cup(sys.argv[1], preview=True), indent=1))
