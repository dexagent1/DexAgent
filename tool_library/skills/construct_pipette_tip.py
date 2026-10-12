"""Construct pipette tip: A disposable pipette tip whose collar stretches over the micropipette nose cone and holds by friction.

Thin entry point over tool_library/common/pipettekit.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_pipette_tip',
 'title': 'Construct pipette tip',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A disposable pipette tip whose collar stretches over the micropipette nose cone '
            'and holds by friction.',
 'description': 'Builds the tapered tip with a hollow collar (a ring of boxes, so the nose '
                'cone can enter) sized to the micropipette, from its lengths, radii, colour '
                'and mass.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default ptip'},
            {'name': 'length', 'type': 'float', 'description': 'Length, metres. Default 0.076'},
            {'name': 'collar_or',
             'type': 'float',
             'description': 'Collar outer radius (rests on the tray hole). Default 0.0045'},
            {'name': 'collar_ir',
             'type': 'float',
             'description': 'Entrance bore. Default 0.0041'},
            {'name': 'collar_len', 'type': 'float', 'description': 'Collar len. Default 0.009'},
            {'name': 'lip_in',
             'type': 'float',
             'description': 'The inner lip the nose jaws snap past —. Default 0.0008'},
            {'name': 'lip_depth',
             'type': 'float',
             'description': "Below the collar's top edge. Default 0.002"},
            {'name': 'tip_rgba',
             'type': 'list[4]',
             'description': 'Tip rgba, colour as rgba in 0..1. Default [0.62, 0.72, 0.93, '
                            '0.55]'},
            {'name': 'density',
             'type': 'float',
             'description': 'Plastic shell, mostly hollow. Default 1010.0'},
            {'name': 'friction',
             'type': 'str',
             'description': 'Friction, MuJoCo friction triple. Default 0.6 0.01 0.0002'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.001'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Pos, metres, xyz. Default [0.0, 0.0, 0.0]'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['pipette tip', 'lab', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_pipette_tip',
 'cli': 'python -m tool_library.skills.construct_pipette_tip out_dir'}




from tool_library.common import pipettekit as _core

# every key below is a parameter of construct_pipette_tip(**params); unknown keys raise
DEFAULTS = _core.TIP_DEFAULTS


def construct_pipette_tip(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build_pipette_tip(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_pipette_tip(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
