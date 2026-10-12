"""Construct battery holder with springs: An open battery holder whose every slot embeds a real construct_spring coil, with the interior length derived from the cell and the spring preload.

Thin entry point over tool_library/common/battery_case.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_battery_case',
 'title': 'Construct battery holder with springs',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'An open battery holder whose every slot embeds a real construct_spring coil, with '
            'the interior length derived from the cell and the spring preload.',
 'description': 'Builds the holder as a case with N slots, each embedding a spring chain at '
                'alternating ends and a contact bump at the other, from the cell dimensions, '
                'slot count, wall thickness, spring parameters, seated preload and colours. '
                'The interior length is derived from cell length plus spring at preload, so a '
                'holder built for a different cell or spring is a different holder. probe() '
                'seats a built cell in a slot with force grips and measures whether the spring '
                'holds it. The battery_seat_test verifier wraps that probe.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default battery_case'},
            {'name': 'battery_diameter',
             'type': 'float',
             'description': 'Battery diameter, metres. Default 0.0145'},
            {'name': 'battery_length',
             'type': 'float',
             'description': 'Battery length, metres. Default 0.0505'},
            {'name': 'n_slots', 'type': 'int', 'description': 'Number of slots. Default 2'},
            {'name': 'springs_at',
             'type': 'list[2]',
             'description': "Per slot, slot 0 = smallest y.. Default ['-x', '+x']"},
            {'name': 'slot_clearance',
             'type': 'float',
             'description': 'Slot width - battery diameter. Default 0.0016'},
            {'name': 'seated_preload',
             'type': 'float',
             'description': 'Fraction of spring travel used up. Default 0.6'},
            {'name': 'spring_stiffness',
             'type': 'float',
             'description': 'Spring stiffness, N/m. Default 250.0'},
            {'name': 'spring_free_length',
             'type': 'float',
             'description': 'Spring free length, metres. Default 0.0115'},
            {'name': 'spring_coil_radius',
             'type': 'float',
             'description': 'Spring coil radius, metres. Default 0.0042'},
            {'name': 'spring_wire_radius',
             'type': 'float',
             'description': 'Spring wire radius, metres. Default 0.00045'},
            {'name': 'spring_n_coils',
             'type': 'int',
             'description': 'Spring n coils, count. Default 5'},
            {'name': 'bump_radius',
             'type': 'float',
             'description': 'Bump radius, metres. Default 0.0035'},
            {'name': 'bump_proud',
             'type': 'float',
             'description': 'Visual dome height above the wall. Default 0.0012'},
            {'name': 'plate_t',
             'type': 'float',
             'description': 'The colliding contact face. Default 0.0005'},
            {'name': 'wall',
             'type': 'float',
             'description': 'Wall thickness, metres. Default 0.0018'},
            {'name': 'end_wall',
             'type': 'float',
             'description': 'End wall, metres. Default 0.002'},
            {'name': 'divider', 'type': 'float', 'description': 'Divider. Default 0.002'},
            {'name': 'floor_t', 'type': 'float', 'description': 'Floor t. Default 0.0016'},
            {'name': 'height',
             'type': 'float',
             'description': 'Height, metres. Default 0.0145'},
            {'name': 'body_rgba',
             'type': 'list[4]',
             'description': 'Body rgba, colour as rgba in 0..1. Default [0.06, 0.06, 0.065, '
                            '1.0]'},
            {'name': 'body_specular',
             'type': 'float',
             'description': 'Glossy ABS. Default 0.35'},
            {'name': 'body_shininess',
             'type': 'float',
             'description': 'Body shininess. Default 0.65'},
            {'name': 'metal_rgba',
             'type': 'list[4]',
             'description': 'Metal rgba, colour as rgba in 0..1. Default [0.75, 0.76, 0.78, '
                            '1.0]'},
            {'name': 'density',
             'type': 'float',
             'description': "ABS -> the shell's mass. Default 1250.0"},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [0.7, 0.008, 0.0002]'},
            {'name': 'fixed_base',
             'type': 'bool',
             'description': 'True bolts it to the world. Default False'},
            {'name': 'with_wires', 'type': 'bool', 'description': 'With wires. Default True'},
            {'name': 'wire_radius',
             'type': 'float',
             'description': 'Wire radius, metres. Default 0.0008'},
            {'name': 'wire_length',
             'type': 'float',
             'description': 'How far the loop wanders on the table. Default 0.075'},
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
 'tags': ['battery', 'holder', 'spring', 'receptacle', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Battery Insertion'],
 'status': 'tested',
 'entrypoint': 'construct_battery_case',
 'cli': 'python -m tool_library.skills.construct_battery_case out_dir'}




from tool_library.common import battery_case as _core

# every key below is a parameter of construct_battery_case(**params); unknown keys raise
DEFAULTS = _core.DEFAULTS


def construct_battery_case(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_battery_case(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
