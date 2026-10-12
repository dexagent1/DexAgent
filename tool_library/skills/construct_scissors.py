"""Construct scissors: A pair of scissors as a real mechanism: two tapered blades with finger rings on one pivot, with a measured cutting throat so a cord between the blades can be severed.

Thin entry point over tool_library/common/scissors.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_scissors',
 'title': 'Construct scissors',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A pair of scissors as a real mechanism: two tapered blades with finger rings on '
            'one pivot, with a measured cutting throat so a cord between the blades can be '
            'severed.',
 'description': 'Builds two halves, each a tapered blade, a shank and a finger ring, stacked '
                'in the blade-thickness direction and joined by one hinge at the pivot, from '
                'blade length, handle and shank length, ring size, blade thickness, opening '
                'range, colours and mass. The build records the cutting throat: gap_at() '
                'measures the blade gap at a point and try_sever() parts a cuttable rope '
                '(attach_cuttable_thread) only when the measured gap closes below cut_gap. The '
                'scissors_cut_test verifier opens and shuts a built pair with forces at the '
                'rings and checks it cuts.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default scissors'},
            {'name': 'blade_length',
             'type': 'float',
             'description': 'Pivot to blade tip, metres. Default 0.051'},
            {'name': 'blade_width',
             'type': 'float',
             'description': 'At the pivot .... Default 0.01'},
            {'name': 'blade_tip_width',
             'type': 'float',
             'description': '... tapering to this at the point. Default 0.0016'},
            {'name': 'blade_thickness',
             'type': 'float',
             'description': 'ONE blade; the pair is twice this. Default 0.0016'},
            {'name': 'handle_length',
             'type': 'float',
             'description': 'Pivot to handle end, metres. Default 0.073'},
            {'name': 'shank_length',
             'type': 'float',
             'description': 'Pivot -> where the ring starts. Default 0.028'},
            {'name': 'ring_width',
             'type': 'float',
             'description': 'Ring outer size across the handle axis. Default 0.028'},
            {'name': 'ring_offset',
             'type': 'float',
             'description': "Each ring's lateral offset from the axis. Default 0.0126"},
            {'name': 'ring_lift', 'type': 'float', 'description': 'Ring lift. Default 0.0'},
            {'name': 'ring_tube',
             'type': 'float',
             'description': "The moulded grip's tube radius. Default 0.0034"},
            {'name': 'ring_strap', 'type': 'float | None', 'description': 'Ring strap'},
            {'name': 'ring_scale_b',
             'type': 'float',
             'description': 'The second ring is the bigger one on. Default 1.15'},
            {'name': 'ring_segments',
             'type': 'int',
             'description': 'Capsules per ring (a real hole to poke. Default 14'},
            {'name': 'pivot_radius',
             'type': 'float',
             'description': 'Pivot radius, metres. Default 0.0055'},
            {'name': 'open_deg',
             'type': 'float',
             'description': 'Blades fully open. Default 42.0'},
            {'name': 'close_deg',
             'type': 'float',
             'description': 'Blades shut (joint zero). Default 0.0'},
            {'name': 'joint_damping',
             'type': 'float',
             'description': 'Joint damping, joint damping. Default 0.015'},
            {'name': 'joint_frictionloss',
             'type': 'float',
             'description': 'Joint frictionloss, MuJoCo friction triple. Default 0.003'},
            {'name': 'joint_armature',
             'type': 'float',
             'description': 'Joint armature. Default 5e-05'},
            {'name': 'joint_solreflimit',
             'type': 'list[2]',
             'description': 'Joint solreflimit, MuJoCo solref. Default [0.01, 1.0]'},
            {'name': 'blade_rgba',
             'type': 'list[4]',
             'description': 'Stainless. Default [0.78, 0.79, 0.82, 1.0]'},
            {'name': 'handle_rgba',
             'type': 'list[4]',
             'description': 'Measured purple grip. Default [0.42, 0.28, 0.6, 1.0]'},
            {'name': 'pivot_rgba',
             'type': 'list[4]',
             'description': 'Pivot rgba, colour as rgba in 0..1. Default [0.55, 0.55, 0.58, '
                            '1.0]'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Pos, metres, xyz. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'mass', 'type': 'float', 'description': 'Whole tool. Default 0.045'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [0.9, 0.02, 0.004]'},
            {'name': 'condim', 'type': 'int', 'description': 'Condim. Default 6'},
            {'name': 'solref',
             'type': 'list[2]',
             'description': 'Solref, MuJoCo solref. Default [0.005, 1.0]'},
            {'name': 'solimp',
             'type': 'list[3]',
             'description': 'Solimp, MuJoCo solimp. Default [0.99, 0.999, 0.001]'},
            {'name': 'fixed_base', 'type': 'bool', 'description': 'Fixed base. Default False'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.002'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'},
            {'name': 'cut_gap_m',
             'type': 'float',
             'description': 'Edges this close = through the material. Default 0.0035'},
            {'name': 'cut_catch_m',
             'type': 'float',
             'description': 'How far off the blade line still counts. Default 0.011'},
            {'name': 'cut_pinch_mm',
             'type': 'float',
             'description': 'How deep EACH blade must bite in. Default 0.3'},
            {'name': 'cut_lat_m',
             'type': 'float',
             'description': "How far off the blades' own PLANE. A cord. Default 0.004"}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); cutter: the subset scissors.try_sever needs (bodies, '
                             'joint, blade sizes, cut gates) in the asset\'s own names -- record it '
                             'in the Stage 2 manifest entry as `cutter` so the Stage 3 executor '
                             'can cut a rope (the scene prefix is read off root_body); stills '
                             'when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['scissors', 'hinge', 'cut', 'tool', 'articulated', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Scissor Cutting'],
 'status': 'tested',
 'entrypoint': 'construct_scissors',
 'cli': 'python -m tool_library.skills.construct_scissors out_dir'}




from tool_library.common import scissors as _core

# every key below is a parameter of construct_scissors(**params); unknown keys raise
DEFAULTS = _core.DEFAULTS


def construct_scissors(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_scissors(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
