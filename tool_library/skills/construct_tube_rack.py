"""Construct microtube rack: A translucent microtube rack (an 80-well 5x16 block by default) whose wells accept the built microtubes, flip caps included.

Thin entry point over tool_library/common/labtube.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_tube_rack',
 'title': 'Construct microtube rack',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A translucent microtube rack (an 80-well 5x16 block by default) whose wells '
            'accept the built microtubes, flip caps included.',
 'description': 'Builds the block with a grid of wells sized to the microtube, from well '
                'count, pitch, block size and colour; rack_well_xy() gives a well position by '
                'row and column and tube_seated() reports whether a tube stands in its well on '
                'a compiled model.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default tuberack'},
            {'name': 'rows',
             'type': 'int',
             'description': 'The classic 80-well 2:1 blue rack. Default 5'},
            {'name': 'cols', 'type': 'int', 'description': 'Cols. Default 16'},
            {'name': 'pitch',
             'type': 'float',
             'description': 'Pitch, metres per turn. Default 0.0138'},
            {'name': 'length', 'type': 'float', 'description': 'Length, metres. Default 0.245'},
            {'name': 'width', 'type': 'float', 'description': 'Width, metres. Default 0.105'},
            {'name': 'height', 'type': 'float', 'description': 'Height, metres. Default 0.052'},
            {'name': 'plate_t', 'type': 'float', 'description': 'Plate t. Default 0.007'},
            {'name': 'wall_t',
             'type': 'float',
             'description': 'Wall t, metres. Default 0.0035'},
            {'name': 'hole_r',
             'type': 'float',
             'description': 'Moulded hole look (visual). Default 0.0056'},
            {'name': 'socket_half',
             'type': 'float',
             'description': 'Square opening half-width at sockets:. Default 0.0056'},
            {'name': 'shaft_depth',
             'type': 'float',
             'description': 'Shaft depth, metres. Default 0.018'},
            {'name': 'rack_rgba',
             'type': 'list[4]',
             'description': 'Rack rgba, colour as rgba in 0..1. Default [0.13, 0.3, 0.86, '
                            '0.62]'},
            {'name': 'hole_rgba',
             'type': 'list[4]',
             'description': 'Hole rgba, colour as rgba in 0..1. Default [0.03, 0.07, 0.3, '
                            '0.95]'},
            {'name': 'density',
             'type': 'float',
             'description': 'Ribbed polypropylene, mostly air:. Default 380.0'},
            {'name': 'friction',
             'type': 'str',
             'description': 'Friction, MuJoCo friction triple. Default 0.85 0.015 0.0004'},
            {'name': 'rack_free', 'type': 'bool', 'description': 'Rack free. Default True'},
            {'name': 'tubes',
             'type': 'list[0]',
             'description': '[{row, col, cap_deg, yaw_deg}] seated. Default []'},
            {'name': 'open_sockets',
             'type': 'list[0]',
             'description': '[[row, col]] wells given a REAL opening. Default []'},
            {'name': 'tube_params',
             'type': 'dict',
             'description': "Overrides for the seated tubes' build"},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Pos, metres, xyz. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.001'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['rack', 'lab', 'receptacle', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_tube_rack',
 'cli': 'python -m tool_library.skills.construct_tube_rack out_dir'}




from tool_library.common import labtube as _core

# every key below is a parameter of construct_tube_rack(**params); unknown keys raise
DEFAULTS = _core.RACK_DEFAULTS


def construct_tube_rack(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build_tube_rack(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_tube_rack(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
