"""Construct pipette tip box: A hinged-lid tip box: navy base, a tray of 96 tip collars and a translucent lid on a hinge, with the tips standing in it.

Thin entry point over tool_library/common/pipettekit.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_tip_box',
 'title': 'Construct pipette tip box',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A hinged-lid tip box: navy base, a tray of 96 tip collars and a translucent lid '
            'on a hinge, with the tips standing in it.',
 'description': 'Builds the base, the tray of tip collars on a grid, the hinged lid and the '
                'tips in their wells, from box dimensions, grid and pitch, lid hinge stiffness '
                'and colours; tipbox_well_xy() gives a well position by row and column. The '
                'moulded tips of the other wells are drawn as collar stubs and collide as slightly '
                'thinner cylinders of the same height (clear of a free tip in the next well), so '
                'nothing drawn stands outside the physics; the lid takes lid_mass spread over its '
                'plates and MuJoCo derives its inertia from them. Every well is one collision box '
                'and every free tip about 37 geoms, and the free tips are separate free bodies: when '
                'a robot hand works in or near the box (contact budget), or the free tips must be '
                'named objects of their own, use construct_tip_box_lowpoly (merged tray slabs with '
                'caged sockets, lighter tips, caller-named tips) instead.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default tipbox'},
            {'name': 'length', 'type': 'float', 'description': 'Length, metres. Default 0.127'},
            {'name': 'width', 'type': 'float', 'description': 'Width, metres. Default 0.096'},
            {'name': 'base_h', 'type': 'float', 'description': 'Base h. Default 0.072'},
            {'name': 'wall_t', 'type': 'float', 'description': 'Wall t, metres. Default 0.003'},
            {'name': 'tray_t', 'type': 'float', 'description': 'Tray t. Default 0.004'},
            {'name': 'rows', 'type': 'int', 'description': 'Rows. Default 8'},
            {'name': 'cols', 'type': 'int', 'description': 'Cols. Default 12'},
            {'name': 'pitch',
             'type': 'float',
             'description': 'Pitch, metres per turn. Default 0.009'},
            {'name': 'hole_r',
             'type': 'float',
             'description': 'Bears 0.75 mm of collar per side AND guides. Default 0.00375'},
            {'name': 'lid_h',
             'type': 'float',
             'description': 'Deep enough that the CLOSED lid clears the. Default 0.059'},
            {'name': 'lid_t', 'type': 'float', 'description': 'Lid t. Default 0.0022'},
            {'name': 'lid_deg',
             'type': 'float',
             'description': 'Where the lid starts (0 = closed). Default 100.0'},
            {'name': 'lid_open_max',
             'type': 'float',
             'description': 'Lid open max, metres. Default 118.0'},
            {'name': 'lid_friction',
             'type': 'float',
             'description': 'What makes it STAY where it is pushed. Default 0.055'},
            {'name': 'lid_damping',
             'type': 'float',
             'description': 'Lid damping, joint damping. Default 0.02'},
            {'name': 'lid_mass',
             'type': 'float',
             'description': "Lid mass in kg, spread over its plates by volume (MuJoCo derives the inertia from "
                            "them; a real 96-tip lid weighs 20-30 g). Default 0.001"},
            {'name': 'free_wells',
             'type': 'list[0]',
             'description': '[[row, col]] wells holding REAL tips. Default []'},
            {'name': 'tip_params', 'type': 'dict', 'description': 'Tip params'},
            {'name': 'base_rgba',
             'type': 'list[4]',
             'description': 'Base rgba, colour as rgba in 0..1. Default [0.093, 0.11, 0.3, '
                            '1.0]'},
            {'name': 'tray_rgba',
             'type': 'list[4]',
             'description': 'Tray rgba, colour as rgba in 0..1. Default [0.6, 0.72, 0.92, '
                            '1.0]'},
            {'name': 'lid_rgba',
             'type': 'list[4]',
             'description': 'Lid rgba, colour as rgba in 0..1. Default [0.86, 0.89, 0.93, '
                            '0.4]'},
            {'name': 'stub_rgba',
             'type': 'list[4]',
             'description': 'Stub rgba, colour as rgba in 0..1. Default [0.62, 0.72, 0.93, '
                            '0.75]'},
            {'name': 'label_rgba',
             'type': 'list[4]',
             'description': 'Label rgba, colour as rgba in 0..1. Default [0.75, 0.78, 0.85, '
                            '1.0]'},
            {'name': 'density',
             'type': 'float',
             'description': 'Plastic shell, mostly hollow. Default 480.0'},
            {'name': 'friction',
             'type': 'str',
             'description': 'Friction, MuJoCo friction triple. Default 0.85 0.015 0.0004'},
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
 'tags': ['tip box', 'lid', 'hinge', 'lab', 'receptacle', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Biology Experiment'],
 'status': 'tested',
 'entrypoint': 'construct_tip_box',
 'cli': 'python -m tool_library.skills.construct_tip_box out_dir'}




from tool_library.common import pipettekit as _core

# every key below is a parameter of construct_tip_box(**params); unknown keys raise
DEFAULTS = _core.TIPBOX_DEFAULTS


def construct_tip_box(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build_tip_box(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_tip_box(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
