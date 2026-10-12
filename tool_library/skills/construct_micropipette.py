"""Construct micropipette: An Eppendorf-style single-channel micropipette with a two-stage plunger (soft aspirate stroke, then a preloaded blow-out spring) and a nose cone that tips snap onto.

Thin entry point over tool_library/common/pipettekit.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_micropipette',
 'title': 'Construct micropipette',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'An Eppendorf-style single-channel micropipette with a two-stage plunger (soft '
            'aspirate stroke, then a preloaded blow-out spring) and a nose cone that tips snap '
            'onto.',
 'description': 'Builds the body, the plunger on a slide against a main spring to the first '
                'stop and a separate preloaded blow-out pin the stem meets at that depth, the '
                'ejector, and the nose cone, from its dimensions, spring forces and strokes, '
                'colours and mass. The force step a thumb feels is a second body preload, not '
                'a lookup table. The plunger_two_stage_test verifier presses a built plunger '
                'and checks both stops.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default pipette'},
            {'name': 'nose_seat_z',
             'type': 'float',
             'description': "Where the mounted collar's TOP lands. The. Default 0.0075"},
            {'name': 'jaw_out',
             'type': 'float',
             'description': "Jaws' relaxed outer reach — sets the snap. Default 0.0038"},
            {'name': 'jaw_r', 'type': 'float', 'description': 'Jaw r. Default 0.0006'},
            {'name': 'jaw_travel',
             'type': 'float',
             'description': 'Jaw travel. Default 0.0009'},
            {'name': 'jaw_stiffness',
             'type': 'float',
             'description': 'Jaw stiffness, N/m. Default 9000.0'},
            {'name': 'jaw_damping',
             'type': 'float',
             'description': 'Jaw damping, joint damping. Default 8.0'},
            {'name': 'jaw_armature',
             'type': 'float',
             'description': 'Jaw armature. Default 0.01'},
            {'name': 'eject_travel',
             'type': 'float',
             'description': 'Eject travel. Default 0.01'},
            {'name': 'eject_stiffness',
             'type': 'float',
             'description': 'Eject stiffness, N/m. Default 800.0'},
            {'name': 'eject_damping',
             'type': 'float',
             'description': 'Eject damping, joint damping. Default 3.0'},
            {'name': 'stroke1',
             'type': 'float',
             'description': 'To the first stop. Default 0.012'},
            {'name': 'stroke2',
             'type': 'float',
             'description': 'First stop -> blow-out bottom. Default 0.0055'},
            {'name': 'spring1',
             'type': 'float',
             'description': 'N/m   (~2.6 N at the first stop). Default 220.0'},
            {'name': 'blowout_preload_N',
             'type': 'float',
             'description': 'Blowout preload N. Default 5.0'},
            {'name': 'blowout_stiffness',
             'type': 'float',
             'description': 'Blowout stiffness, N/m. Default 2600.0'},
            {'name': 'handle_r', 'type': 'float', 'description': 'Handle r. Default 0.013'},
            {'name': 'handle_len', 'type': 'float', 'description': 'Handle len. Default 0.105'},
            {'name': 'total_len', 'type': 'float', 'description': 'Total len. Default 0.29'},
            {'name': 'body_rgba',
             'type': 'list[4]',
             'description': 'Body rgba, colour as rgba in 0..1. Default [0.93, 0.93, 0.92, '
                            '1.0]'},
            {'name': 'grey_rgba',
             'type': 'list[4]',
             'description': 'Grey rgba, colour as rgba in 0..1. Default [0.52, 0.53, 0.55, '
                            '1.0]'},
            {'name': 'blue_rgba',
             'type': 'list[4]',
             'description': 'Blue rgba, colour as rgba in 0..1. Default [0.16, 0.32, 0.75, '
                            '1.0]'},
            {'name': 'accent_rgba',
             'type': 'list[4]',
             'description': 'Accent rgba, colour as rgba in 0..1. Default [0.8, 0.12, 0.12, '
                            '1.0]'},
            {'name': 'density',
             'type': 'float',
             'description': 'Plastic shell, mostly hollow. Default 820.0'},
            {'name': 'friction',
             'type': 'str',
             'description': 'Friction, MuJoCo friction triple. Default 0.7 0.01 0.0002'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Pos, metres, xyz. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'spin_deg',
             'type': 'float',
             'description': 'Lying: which side faces up. Default 0.0'},
            {'name': 'layout',
             'type': 'str',
             'description': 'Lying | upright (upright needs a holder). Default lying'},
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
 'tags': ['pipette', 'plunger', 'spring', 'tool', 'lab', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_micropipette',
 'cli': 'python -m tool_library.skills.construct_micropipette out_dir'}




from tool_library.common import pipettekit as _core

# every key below is a parameter of construct_micropipette(**params); unknown keys raise
DEFAULTS = _core.PIPETTE_DEFAULTS


def construct_micropipette(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build_micropipette(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_micropipette(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
