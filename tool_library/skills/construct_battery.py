"""Construct cylindrical battery cell: A cylindrical cell (AA by default) as a free rigid body with real terminals: a proud positive button and a flat negative face, in a two-tone jacket that makes polarity readable.

Thin entry point over tool_library/common/battery_cell.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_battery',
 'title': 'Construct cylindrical battery cell',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A cylindrical cell (AA by default) as a free rigid body with real terminals: a '
            'proud positive button and a flat negative face, in a two-tone jacket that makes '
            'polarity readable.',
 'description': 'Builds the cell from its diameter, length, button size, mass and jacket '
                'colours. The negative face is at z=0 and the positive button at the top, as a '
                'real colliding cylinder proud of the face, so a holder spring pushes on one '
                'end and the contact bump meets the other, and a render shows which way the '
                'cell is in.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default battery'},
            {'name': 'diameter',
             'type': 'float',
             'description': 'Diameter, metres. Default 0.0145'},
            {'name': 'length',
             'type': 'float',
             'description': 'Cell length including the button, metres. Default 0.0505'},
            {'name': 'button_radius',
             'type': 'float',
             'description': 'Button radius, metres. Default 0.0027'},
            {'name': 'button_height',
             'type': 'float',
             'description': 'Button height, metres. Default 0.001'},
            {'name': 'crown_frac',
             'type': 'float',
             'description': 'Top fraction of the sleeve that is copper. Default 0.3'},
            {'name': 'base_ring_h',
             'type': 'float',
             'description': 'Thin copper ring at the negative rim. Default 0.0012'},
            {'name': 'body_rgba',
             'type': 'list[4]',
             'description': 'Body rgba, colour as rgba in 0..1. Default [0.055, 0.055, 0.06, '
                            '1.0]'},
            {'name': 'crown_rgba',
             'type': 'list[4]',
             'description': 'Crown rgba, colour as rgba in 0..1. Default [0.72, 0.45, 0.2, '
                            '1.0]'},
            {'name': 'terminal_rgba',
             'type': 'list[4]',
             'description': 'Terminal rgba, colour as rgba in 0..1. Default [0.75, 0.76, 0.78, '
                            '1.0]'},
            {'name': 'body_specular',
             'type': 'float',
             'description': 'Glossy shrink-wrap. Default 0.3'},
            {'name': 'body_shininess',
             'type': 'float',
             'description': 'Body shininess. Default 0.55'},
            {'name': 'metal_specular',
             'type': 'float',
             'description': 'Metal specular. Default 0.75'},
            {'name': 'metal_shininess',
             'type': 'float',
             'description': 'Metal shininess. Default 0.9'},
            {'name': 'mass', 'type': 'float', 'description': 'Mass, kilograms. Default 0.023'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': ">= 0.85 with the holder's own. Default [0.85, 0.008, 0.0002]"},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Of the NEGATIVE face centre. Default [0.0, 0.0, 0.0]'},
            {'name': 'upright',
             'type': 'bool',
             'description': 'False -> built lying along +x. Default True'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.002'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['battery', 'cell', 'rigid', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Battery Insertion'],
 'status': 'tested',
 'entrypoint': 'construct_battery',
 'cli': 'python -m tool_library.skills.construct_battery out_dir'}




from tool_library.common import battery_cell as _core

# every key below is a parameter of construct_battery(**params); unknown keys raise
DEFAULTS = _core.DEFAULTS


def construct_battery(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_battery(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
