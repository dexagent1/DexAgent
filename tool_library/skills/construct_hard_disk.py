"""Construct hard disk in tray: A 3.5-inch hard disk in a tray sled as a free rigid body: a silver lid, a black caddy with a handle, sized from measured outside dimensions.

Thin entry point over tool_library/common/diskbay.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_hard_disk',
 'title': 'Construct hard disk in tray',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A 3.5-inch hard disk in a tray sled as a free rigid body: a silver lid, a black '
            'caddy with a handle, sized from measured outside dimensions.',
 'description': 'Builds the drive and its tray as one free body from length, width, height, '
                'tray wall and handle dimensions, colours and mass, with the handle end '
                'declared so a push knows where to press.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default hard_disk'},
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
             'type': 'list[4] | None',
             'description': 'Label patch on the top face, e.g. [0.92, 0.92, 0.90, 1] for a white label. Default None (no label)'},
            {'name': 'mass', 'type': 'float', 'description': 'Mass, kilograms. Default 0.45'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Footprint centre on the support (local z = 0 is the support surface). Default [0.0, 0.0, 0.0]'},
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
             'type': 'bool | None',
             'description': 'No free joint when True. Default None (a disk is free)'},
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
 'tags': ['hard disk', 'tray', 'rigid', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Computer Installation'],
 'status': 'tested',
 'entrypoint': 'construct_hard_disk',
 'cli': 'python -m tool_library.skills.construct_hard_disk out_dir'}




from tool_library.common import diskbay as _core

# every key below is a parameter of construct_hard_disk(**params); unknown keys raise
RACK_KEYS = {"n_slots", "slot_w", "slot_h", "bay_depth", "rail_reach", "rail_t", "wall_t", "post_t", "plate_t",
             "frame_rgba", "rail_rgba", "open_back"}
# style is fixed to 'disk' and the rack-only keys are left out; the default name is the drive's own
DEFAULTS = {k: v for k, v in _core.DEFAULTS.items() if k != 'style' and k not in RACK_KEYS}
DEFAULTS["name"] = "hard_disk"


def construct_hard_disk(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; allowed: {sorted(DEFAULTS)}")
    built = _core.build({"name": DEFAULTS["name"], **params, "style": "disk"}, out_dir)
    if preview:
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_hard_disk(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
