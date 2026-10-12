"""Construct helical compression spring: A compression spring built from stiffness, free length, coil and wire radius and coil count: a chain of slide-joint coils with the measured stiffness and a coil-bound hard stop.

Thin entry point over tool_library/common/spring.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_spring',
 'title': 'Construct helical compression spring',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A compression spring built from stiffness, free length, coil and wire radius and '
            'coil count: a chain of slide-joint coils with the measured stiffness and a '
            'coil-bound hard stop.',
 'description': 'Emits the spring as a chain of coil bodies on slide joints with the stiffness '
                'divided among them, a visual helix mesh, and a hard stop where the coils '
                'touch, from stiffness (N/m), free length, coil radius, wire radius and number '
                'of active coils. compression() reads how far a built spring is squeezed. The '
                'spring_stiffness_test verifier presses a built spring with known forces and '
                'checks the measured stiffness against the commanded one.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default spring'},
            {'name': 'stiffness',
             'type': 'float',
             'description': 'Spring rate, N/m. Default 250.0'},
            {'name': 'free_length',
             'type': 'float',
             'description': 'Relaxed length, metres. Default 0.0115'},
            {'name': 'coil_radius',
             'type': 'float',
             'description': 'Coil radius, metres. Default 0.0042'},
            {'name': 'wire_radius',
             'type': 'float',
             'description': 'Wire radius, metres. Default 0.00045'},
            {'name': 'n_coils', 'type': 'int', 'description': 'Active coils. Default 5'},
            {'name': 'plate_radius',
             'type': 'float | None',
             'description': 'None -> coil_radius + wire_radius: the disc'},
            {'name': 'plate_thickness',
             'type': 'float',
             'description': 'Plate thickness, metres. Default 0.0008'},
            {'name': 'seat_turns',
             'type': 'float',
             'description': 'A flat "ground" turn drawn at the base so the. Default 0.9'},
            {'name': 'rgba',
             'type': 'list[4]',
             'description': 'Rgba. Default [0.8, 0.82, 0.84, 1.0]'},
            {'name': 'specular', 'type': 'float', 'description': 'Specular. Default 0.7'},
            {'name': 'shininess', 'type': 'float', 'description': 'Shininess. Default 0.9'},
            {'name': 'moving_mass',
             'type': 'float',
             'description': 'Total translating mass (a real spring ~0.5 g). Default 0.0008'},
            {'name': 'zeta',
             'type': 'float',
             'description': 'Damping ratio per joint: 0.1 boings, 1.0 dead. Default 0.25'},
            {'name': 'armature',
             'type': 'float | None',
             'description': 'None -> derived for Euler @ 2 ms (see module'},
            {'name': 'friction_tip',
             'type': 'float',
             'description': 'Plate contact friction (slide). High on. Default 0.85'},
            {'name': 'include_floor',
             'type': 'bool',
             'description': 'Include floor. Default True'},
            {'name': 'timestep',
             'type': 'float',
             'description': 'Match the bench world: plain Euler @ 2 ms is. Default 0.002'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml: the MJCF path; params_json: every parameter as built; the '
                             'names of the bodies, joints and equality constraints the '
                             'mechanism exposes, and the derived measurements (sizes, travel, '
                             'masses); stills when preview is set.'}],
 'requirements': ['mujoco>=3.2', 'numpy', 'trimesh', 'imageio', 'Pillow'],
 'simulators': ['MuJoCo'],
 'tags': ['spring', 'mechanism', 'elastic', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Battery Insertion'],
 'status': 'tested',
 'entrypoint': 'construct_spring',
 'cli': 'python -m tool_library.skills.construct_spring out_dir'}




from tool_library.common import spring as _core

# every key below is a parameter of construct_spring(**params); unknown keys raise
DEFAULTS = _core.DEFAULTS


def construct_spring(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build(dict(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_spring(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
