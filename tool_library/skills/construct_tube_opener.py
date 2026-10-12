"""Construct tube decapper: The cylindrical decapper for flip-cap tubes: a hollow skirt with a tab window, an internal ledge that slides under the cap tab, and a rim that bears on the tube flange.

Thin entry point over tool_library/common/labtube.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_tube_opener',
 'title': 'Construct tube decapper',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'The cylindrical decapper for flip-cap tubes: a hollow skirt with a tab window, an '
            'internal ledge that slides under the cap tab, and a rim that bears on the tube '
            'flange.',
 'description': 'Builds the opener as a ring-of-boxes skirt (so it has a real bore), a tab '
                'window, an internal ledge and a handle, from its radii, heights, window angle '
                'and colour, sized to the microtube it opens. The demo shows the drop, twist '
                'and pry that pops a cap.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default tubeopener'},
            {'name': 'skirt_ir',
             'type': 'float',
             'description': "MUST be between the tube's cap_r and rim_r. Default 0.0061"},
            {'name': 'skirt_t', 'type': 'float', 'description': 'Skirt t. Default 0.0024'},
            {'name': 'skirt_h', 'type': 'float', 'description': 'Skirt h. Default 0.015'},
            {'name': 'window_deg',
             'type': 'list[2]',
             'description': 'The tab slot, centred on +x. Default [322.0, 38.0]'},
            {'name': 'ledge_deg',
             'type': 'list[2]',
             'description': 'The low step beside it (twist CCW to. Default [38.0, 76.0]'},
            {'name': 'ledge_h', 'type': 'float', 'description': 'Ledge h. Default 0.0012'},
            {'name': 'ledge_slot_h',
             'type': 'float',
             'description': 'Clearance above the ledge the tab rises. Default 0.0075'},
            {'name': 'flare_out',
             'type': 'float',
             'description': 'The funnel legs under the rim: how far. Default 0.0024'},
            {'name': 'flare_drop',
             'type': 'float',
             'description': '...and how far below the rim their. Default 0.0008'},
            {'name': 'collar_h',
             'type': 'float',
             'description': 'Full ring above the windows. Default 0.009'},
            {'name': 'grip_r', 'type': 'float', 'description': 'Grip r. Default 0.0075'},
            {'name': 'grip_h', 'type': 'float', 'description': 'Grip h. Default 0.017'},
            {'name': 'opener_rgba',
             'type': 'list[4]',
             'description': 'Opener rgba, colour as rgba in 0..1. Default [0.13, 0.38, 0.85, '
                            '1.0]'},
            {'name': 'density',
             'type': 'float',
             'description': 'Ribbed polypropylene, mostly air:. Default 1150.0'},
            {'name': 'friction',
             'type': 'str',
             'description': 'Friction, MuJoCo friction triple. Default 0.8 0.015 0.0004'},
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
 'tags': ['decapper', 'tool', 'lab', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_tube_opener',
 'cli': 'python -m tool_library.skills.construct_tube_opener out_dir'}




from tool_library.common import labtube as _core

# every key below is a parameter of construct_tube_opener(**params); unknown keys raise
DEFAULTS = _core.OPENER_DEFAULTS


def construct_tube_opener(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build_tube_opener(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_tube_opener(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
