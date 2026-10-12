"""Construct flip-cap microcentrifuge tube: A 1.5 ml microcentrifuge tube whose flip cap really snaps: a moulded hinge and a plug bead that pops past a lip inside the bore.

Thin entry point over tool_library/common/labtube.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_microtube',
 'title': 'Construct flip-cap microcentrifuge tube',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A 1.5 ml microcentrifuge tube whose flip cap really snaps: a moulded hinge and a '
            'plug bead that pops past a lip inside the bore.',
 'description': 'Builds the conical tube, its rim flange and a hinged flip cap whose plug bead '
                'snaps past a lip in the bore, from tube and cap dimensions, hinge stiffness, '
                'snap force, colours and mass. Opening the cap is a force event of a few '
                'newtons at the tab. cap_state() reads whether a built cap is shut, open or in '
                'between. The flip_cap_snap_test verifier opens and closes a built tube with '
                'forces and checks the snap.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default microtube'},
            {'name': 'total_len',
             'type': 'float',
             'description': 'Tip to rim top. Default 0.039'},
            {'name': 'body_r',
             'type': 'float',
             'description': 'Cylindrical section outer radius. Default 0.0054'},
            {'name': 'cone_len',
             'type': 'float',
             'description': 'Conical lower section. Default 0.0175'},
            {'name': 'tip_r', 'type': 'float', 'description': 'Tip r. Default 0.0018'},
            {'name': 'rim_r',
             'type': 'float',
             'description': 'The rim FLANGE the rack (and decapper) bear on. Default 0.0066'},
            {'name': 'rim_t', 'type': 'float', 'description': 'Rim t. Default 0.0018'},
            {'name': 'bore_r',
             'type': 'float',
             'description': 'Inner mouth the plug seals into. Default 0.0045'},
            {'name': 'bore_depth',
             'type': 'float',
             'description': 'How deep the mouth is modelled hollow. Default 0.008'},
            {'name': 'cap_r',
             'type': 'float',
             'description': 'Cap disc — SMALLER than rim_r so a decapper. Default 0.0057'},
            {'name': 'cap_t', 'type': 'float', 'description': 'Cap t. Default 0.003'},
            {'name': 'plug_r', 'type': 'float', 'description': 'Plug r. Default 0.0038'},
            {'name': 'plug_depth',
             'type': 'float',
             'description': 'Plug depth, metres. Default 0.003'},
            {'name': 'tab_len',
             'type': 'float',
             'description': "The pry tab, mounted at the disc's TOP edge. Default 0.0042"},
            {'name': 'tab_w', 'type': 'float', 'description': 'Tab w. Default 0.005'},
            {'name': 'tab_t', 'type': 'float', 'description': 'Tab t. Default 0.0013'},
            {'name': 'hinge_gap',
             'type': 'float',
             'description': 'Rim edge -> hinge pivot (the moulded strap). Default 0.0022'},
            {'name': 'bead_out',
             'type': 'float',
             'description': 'Bead proud of the plug, radially (capsule r). Default 0.0005'},
            {'name': 'nub_in',
             'type': 'float',
             'description': 'Bore lip proud of the bore wall (capsule r).. Default 0.0007'},
            {'name': 'flex_travel',
             'type': 'float',
             'description': 'Flex travel. Default 0.0012'},
            {'name': 'flex_stiffness',
             'type': 'float',
             'description': 'N/m: pop force = stiffness x interference. Default 9000.0'},
            {'name': 'flex_damping',
             'type': 'float',
             'description': 'Flex damping, joint damping. Default 6.0'},
            {'name': 'flex_armature',
             'type': 'float',
             'description': 'What makes a sub-gram plug on a kN/m spring. Default 0.01'},
            {'name': 'fold_range_deg',
             'type': 'float',
             'description': 'Fold range deg, degrees. Default 185.0'},
            {'name': 'fold_damping',
             'type': 'float',
             'description': 'Moulded hinges are LOSSY. At 0.0004 an. Default 0.002'},
            {'name': 'fold_frictionloss',
             'type': 'float',
             'description': 'An opened cap must stay where it fell. Default 0.002'},
            {'name': 'fold_armature',
             'type': 'float',
             'description': 'Fold armature. Default 0.0005'},
            {'name': 'cap_start_deg',
             'type': 'float',
             'description': '0 = closed; ~165 = lying fully open. Default 0.0'},
            {'name': 'tube_rgba',
             'type': 'list[4]',
             'description': 'Tube rgba, colour as rgba in 0..1. Default [0.93, 0.94, 0.96, '
                            '0.55]'},
            {'name': 'cap_rgba',
             'type': 'list[4]',
             'description': 'Cap rgba, colour as rgba in 0..1. Default [0.93, 0.94, 0.96, '
                            '0.8]'},
            {'name': 'density',
             'type': 'float',
             'description': 'Ribbed polypropylene, mostly air:. Default 1050.0'},
            {'name': 'friction',
             'type': 'str',
             'description': 'Friction, MuJoCo friction triple. Default 0.7 0.01 0.0002'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Pos, metres, xyz. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'layout',
             'type': 'str',
             'description': 'A lone tube cannot stand on its cone:. Default lying'},
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
 'tags': ['tube', 'flip cap', 'snap', 'lab', 'articulated', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_microtube',
 'cli': 'python -m tool_library.skills.construct_microtube out_dir'}




from tool_library.common import labtube as _core

# every key below is a parameter of construct_microtube(**params); unknown keys raise
DEFAULTS = _core.TUBE_DEFAULTS


def construct_microtube(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build_microtube(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_microtube(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
