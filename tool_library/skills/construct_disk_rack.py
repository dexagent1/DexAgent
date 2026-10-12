"""Construct drive-bay stack: An open skeletal drive cage of N identical bays, static, whose slot openings are sized from the disk that must enter them plus clearance, so every disk fits by construction.

Thin entry point over tool_library/common/diskbay.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_disk_rack',
 'title': 'Construct drive-bay stack',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'An open skeletal drive cage of N identical bays, static, whose slot openings are '
            'sized from the disk that must enter them plus clearance, so every disk fits by '
            'construction.',
 'description': 'Builds the column of bays (corner posts, per-level shelf rails, thin side and '
                'back walls, top and bottom plates) from bay count and pitch, slot width and '
                'depth, clearances and colours. slot_for() sizes a slot from a built disk, '
                'fits() checks a disk against a built rack, slot_world() returns a slot pose '
                'on a compiled model, and the same rack is embedded by construct_pc_case.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default diskbay'},
            {'name': 'n_slots', 'type': 'int', 'description': 'N slots. Default 8'},
            {'name': 'slot_w',
             'type': 'float',
             'description': 'Inner opening, across the disk width (x). Default 0.118'},
            {'name': 'slot_h',
             'type': 'float',
             'description': 'Inner opening height (z), floor to ceiling. Default 0.04'},
            {'name': 'bay_depth',
             'type': 'float',
             'description': 'How deep a disk can go (y). Default 0.155'},
            {'name': 'rail_reach',
             'type': 'float',
             'description': 'How far each shelf rail sticks into the. Default 0.014'},
            {'name': 'rail_t',
             'type': 'float',
             'description': 'Shelf rail thickness (z). Default 0.004'},
            {'name': 'wall_t',
             'type': 'float',
             'description': 'Side/back wall thickness. Default 0.004'},
            {'name': 'post_t',
             'type': 'float',
             'description': 'Corner post square section. Default 0.01'},
            {'name': 'plate_t',
             'type': 'float',
             'description': 'Top + bottom plates. Default 0.006'},
            {'name': 'frame_rgba',
             'type': 'list[4]',
             'description': 'Matte near-black steel. Default [0.1, 0.1, 0.11, 1.0]'},
            {'name': 'rail_rgba',
             'type': 'list[4]',
             'description': 'Rail rgba, colour as rgba in 0..1. Default [0.16, 0.16, 0.17, '
                            '1.0]'},
            {'name': 'open_back',
             'type': 'bool',
             'description': 'True drops the back wall (pure ledge rack). Default False'},
            {'name': 'disk_l',
             'type': 'float',
             'description': 'Drive length (y, insertion direction). Default 0.147'},
            {'name': 'disk_w',
             'type': 'float',
             'description': 'Drive width (x). Default 0.102'},
            {'name': 'disk_h',
             'type': 'float',
             'description': 'Drive height (z). Default 0.026'},
            {'name': 'tray_t',
             'type': 'float',
             'description': 'Tray side-rail thickness (adds to width). Default 0.006'},
            {'name': 'tray_h_frac',
             'type': 'float',
             'description': 'Tray rail height as a fraction of disk_h. Default 0.85'},
            {'name': 'handle_l',
             'type': 'float',
             'description': 'Handle bar sticking out past the +y end. Default 0.012'},
            {'name': 'top_rgba',
             'type': 'list[4]',
             'description': 'Bare-drive silver top. Default [0.71, 0.71, 0.73, 1.0]'},
            {'name': 'body_rgba',
             'type': 'list[4]',
             'description': 'Body rgba, colour as rgba in 0..1. Default [0.22, 0.22, 0.23, '
                            '1.0]'},
            {'name': 'tray_rgba',
             'type': 'list[4]',
             'description': 'Tray rgba, colour as rgba in 0..1. Default [0.07, 0.07, 0.08, '
                            '1.0]'},
            {'name': 'label_rgba',
             'type': 'float | None',
             'description': 'E.g. [0.92,0.92,0.90,1] paints a white'},
            {'name': 'mass', 'type': 'float', 'description': 'Mass, kilograms. Default 0.45'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Footprint centre on the support:. Default [0.0, 0.0, 0.0]'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Yaw deg, degrees. Default 0.0'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [0.75, 0.02, 0.002]'},
            {'name': 'condim',
             'type': 'int',
             'description': 'A slid tray must not glide forever. Default 6'},
            {'name': 'fixed_base',
             'type': 'float | None',
             'description': 'None -> rack fixed, disk free'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Timestep, seconds. Default 0.002'},
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
 'tags': ['rack', 'cage', 'receptacle', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Computer Installation'],
 'status': 'tested',
 'entrypoint': 'construct_disk_rack',
 'cli': 'python -m tool_library.skills.construct_disk_rack out_dir'}




from tool_library.common import diskbay as _core

# every key below is a parameter of construct_disk_rack(**params); unknown keys raise
DEFAULTS = {k: v for k, v in _core.DEFAULTS.items() if k != 'style'}   # style is fixed to 'rack'


def construct_disk_rack(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build(dict(params, style='rack'), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_disk_rack(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
