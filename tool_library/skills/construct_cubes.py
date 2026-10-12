"""Construct loose cubes: N free-floating cubes, each its own body, sized, coloured and placed from the footage.

One MJCF holding N free-floating cubes, each its own body with a free joint so
the hands can pick it up and it can come to rest wherever it is dropped. Sizes,
colours and placements come from the footage they are reconstructed from, in
metres relative to the world frame the asset is placed in.

The module contract is the one every other constructor here uses (`build`
returning a dict with an "xml" key plus named handles), so a scene attaches it
like any other asset.
"""
from __future__ import annotations

TOOL = {'name': 'construct_cubes',
 'title': 'Construct loose cubes',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'N free-floating cubes, each its own body, sized, coloured and placed from the '
            'footage.',
 'description': 'Builds one MJCF holding N cubes as free bodies with the sizes, colours, '
                'masses and placements given, in metres relative to the frame the stage places '
                'them in, so hands can pick them up and they come to rest wherever they are '
                'dropped.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default cubes'},
            {'name': 'cubes', 'type': 'dict', 'description': '{cube name: {size (edge, m), rgba, offset [forward, lateral] (m)}}; given, it replaces the default four cubes (orange, yellow, red, blue: 32 mm). Unknown keys raise ValueError.'},
            {'name': 'density', 'type': 'float', 'description': 'Density. Default 700.0'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [1.1, 0.02, 0.001]'},
            {'name': 'solref',
             'type': 'list[2]',
             'description': 'Solref, MuJoCo solref. Default [0.015, 1.0]'},
            {'name': 'solimp',
             'type': 'list[3]',
             'description': 'Solimp, MuJoCo solimp. Default [0.9, 0.95, 0.002]'},
            {'name': 'lift_m',
             'type': 'float',
             'description': 'Height of the cubes above the asset frame (a settle drops them onto the table). Default 0.0'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default False'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses).'}],
 'requirements': ['mujoco>=3.2'],
 'simulators': ['MuJoCo'],
 'tags': ['cube', 'block', 'rigid', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Giftbox Packing'],
 'status': 'tested',
 'entrypoint': 'construct_cubes',
 'cli': 'python -m tool_library.skills.construct_cubes out_dir'}


import json
from pathlib import Path

DEFAULTS = {
    "name": "cubes",
    # each entry: name -> {size (edge, m), rgba, offset [forward, lateral] (m)}; the whole asset is placed by the
    # scene, each cube sits at its own offset in the asset's frame. 32 mm blocks: proportionate to a dexterous hand
    # whose pinch spans 60-88 mm with ~20 mm finger segments (the pads straddle a 32 mm block with 14 mm a side).
    "cubes": {
        "orange": dict(size=0.032, rgba=[0.92, 0.45, 0.13, 1.0],
                       offset=[-0.02, -0.29]),
        "yellow": dict(size=0.032, rgba=[0.95, 0.80, 0.10, 1.0],
                       offset=[-0.14, -0.29]),
        "red":    dict(size=0.032, rgba=[0.80, 0.20, 0.16, 1.0],
                       offset=[-0.08, 0.25]),
        "blue":   dict(size=0.032, rgba=[0.13, 0.35, 0.80, 1.0],
                       offset=[-0.14, 0.25]),
    },
    # light wooden blocks on cloth: high friction, a realistic 700 kg/m3 (a 32 mm block weighs ~23 g) and a 15 ms
    # contact time constant, so a nudge does not send one skating off the table
    "density": 700.0,
    "friction": [1.1, 0.02, 0.001],
    "solref": [0.015, 1.0],
    "solimp": [0.9, 0.95, 0.002],
    "lift_m": 0.0,            # height above the asset frame (a settle drops them onto the table)
    "include_floor": False,
}


CUBE_KEYS = ("size", "rgba", "offset")


def merge_params(defaults: dict, params: dict | None) -> dict:
    """DEFAULTS updated by params; a given `cubes` replaces the default cubes (it is the full set). Unknown keys raise."""
    params = dict(params or {})
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; allowed: {sorted(defaults)}")
    p = {**defaults, **params}
    if not isinstance(p["cubes"], dict) or not p["cubes"]:
        raise ValueError("cubes must be a non-empty {name: {size, rgba, offset}} dict")
    for cname, c in p["cubes"].items():
        if not isinstance(c, dict) or set(c) != set(CUBE_KEYS):
            raise ValueError(f"cube {cname!r} must give exactly {list(CUBE_KEYS)}, got {sorted(c) if isinstance(c, dict) else c!r}")
    return p


def build(params: dict | None = None, out_dir: str | Path = ".") -> dict:
    p = merge_params(DEFAULTS, params)
    name = str(p["name"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fr = " ".join(str(x) for x in p["friction"])
    sr = " ".join(str(x) for x in p.get("solref", [0.015, 1.0]))
    si = " ".join(str(x) for x in p.get("solimp", [0.9, 0.95, 0.002]))
    bodies, handles = [], {}
    for cname, c in p["cubes"].items():
        h = float(c["size"]) / 2.0
        fx, lx = (float(v) for v in c["offset"])
        rgba = " ".join(str(v) for v in c["rgba"])
        b, g, j = f"{name}_{cname}", f"{name}_{cname}_geom", f"{name}_{cname}_free"
        # the offset is (forward, lateral) in the stage's own frame; the stage
        # rotates the whole asset into the camera-anchored world, so the cubes
        # keep their measured layout relative to the box
        bodies.append(
            f'    <body name="{b}" pos="{lx} {fx} {h + float(p["lift_m"])}">\n'
            f'      <freejoint name="{j}"/>\n'
            f'      <geom name="{g}" type="box" size="{h} {h} {h}" '
            f'rgba="{rgba}" density="{p["density"]}" friction="{fr}" '
            f'condim="4" solimp="{si}" solref="{sr}"/>\n'
            f'    </body>')
        handles[cname] = {"body": b, "geom": g, "joint": j,
                          "size_m": float(c["size"])}

    xml = ('<mujoco model="' + name + '">\n'
           '  <compiler angle="degree"/>\n'
           '  <worldbody>\n' + "\n".join(bodies) + '\n'
           '  </worldbody>\n'
           '</mujoco>\n')
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    pj = out_dir / f"{name}_params.json"
    pj.write_text(json.dumps(p, indent=1))
    return {"xml": str(xml_path), "params_json": str(pj),
            "style": "cubes", "cubes": handles,
            "names": list(handles), "wall_joints": [], "flap_joints": []}


def construct_cubes(out_dir, **params):
    """Build the cubes into out_dir. Returns the build dictionary."""
    return build(params, out_dir)


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_cubes(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
