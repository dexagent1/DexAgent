"""Construct rope or cable: A deformable rope, cord or cable: an elastic MuJoCo cable of N capsule segments, inextensible, bending with measured stiffness and colliding with itself so it can be knotted.

Thin entry point over tool_library/common/cable.py, which holds the builder and its measurement helpers.
"""
from __future__ import annotations

TOOL = {'name': 'construct_rope',
 'title': 'Construct rope or cable',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A deformable rope, cord or cable: an elastic MuJoCo cable of N capsule segments, '
            'inextensible, bending with measured stiffness and colliding with itself so it can '
            'be knotted.',
 'description': 'Builds the rope as a MuJoCo elastic cable from its length, radius, segment '
                'count, bending and twisting stiffness, mass per metre and colour, with end '
                'caps (ferrules) and a mocap grip per end that can be welded to a hand. '
                'Given `centreline` (the measured polyline: measure_rope\'s centreline_rig / points, '
                'its whole result dict or that JSON file, in the scene\'s rig/world frame) the cable is '
                'BUILT along it: resampled by arc length to `segments` vertices (default one per '
                '15 mm, 20 to 160), so its compiled pose lies where the video has it and length, '
                'pos and yaw follow from the points; 2-D points get z = radius (lying on the table). '
                '`rest_vertices` does the same with the vertices given exactly. A length, pos or yaw_deg '
                'that contradicts the points, or both centreline and rest_vertices, is refused rather '
                'than silently ignored. Without either the rope is straight along +x from pos. '
                'Helpers reset the centreline to a measured polyline, grab and release an end, '
                'and read the centreline back. Self-collision is on, which is the property '
                'without which a knot is not a knot. The cable_drag_test verifier drags a '
                'built rope and checks stretch, span and self-contact.',
 'inputs': [{'name': 'out_dir',
             'type': 'path',
             'description': 'Folder that receives the MJCF, its meshes and '
                            '<name>_params.json.'},
            {'name': 'preview',
             'type': 'bool',
             'description': 'Also render preview stills of the mechanism at its extremes. '
                            'Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Name. Default thread'},
            {'name': 'length',
             'type': 'float',
             'description': 'Rope length, metres, of a straight rope. Default 0.6. With a centreline it is the '
                            "centreline's arc length (a value more than 5% off is refused)."},
            {'name': 'segments',
             'type': 'int',
             'description': 'Cable vertices (the rope has segments - 1 capsules). Knots need ~40+; drag needs ~20. '
                            'Default 40 straight; with a centreline one per 15 mm of its length (20 to 160).'},
            {'name': 'radius',
             'type': 'float',
             'description': 'Rope radius, metres. Default 0.0035'},
            {'name': 'rgba',
             'type': 'list[4]',
             'description': 'Rope colour as rgba. Default [0.88, 0.24, 0.2, 1.0]'},
            {'name': 'cap_length',
             'type': 'float',
             'description': 'Cap length, metres. Default 0.0'},
            {'name': 'cap_length_a',
             'type': 'float | None',
             'description': 'None -> cap_length'},
            {'name': 'cap_length_b',
             'type': 'float | None',
             'description': 'Cap length b, metres'},
            {'name': 'cap_rgba',
             'type': 'list[4]',
             'description': 'Cap rgba, colour as rgba in 0..1. Default [0.83, 0.68, 0.24, '
                            '1.0]'},
            {'name': 'cap_radius_scale',
             'type': 'float',
             'description': 'A ferrule is usually a touch fatter. Default 1.0'},
            {'name': 'specular',
             'type': 'float',
             'description': 'Velvet: almost none. Default 0.12'},
            {'name': 'shininess', 'type': 'float', 'description': 'Shininess. Default 0.1'},
            {'name': 'cap_specular',
             'type': 'float',
             'description': 'Polished metal. Default 0.35'},
            {'name': 'cap_shininess',
             'type': 'float',
             'description': 'Cap shininess. Default 0.85'},
            {'name': 'grip_rgba',
             'type': 'list[4]',
             'description': 'Grip rgba, colour as rgba in 0..1. Default [0.15, 0.75, 0.35, '
                            '0.0]'},
            {'name': 'pos',
             'type': 'list[3]',
             'description': 'Start of a straight rope (it runs along +x from here), metres. Default [-0.3, 0.0, 0.32]. '
                            'With a centreline it is the first point (a different value is refused).'},
            {'name': 'yaw_deg',
             'type': 'float',
             'description': 'Heading of a straight rope, applied at run time by reset_straight, degrees. Default 0.0. '
                            'Refused with a centreline (the points carry the heading).'},
            {'name': 'centreline',
             'type': 'list[[x, y, z]] | dict | path | None',
             'description': "The measured polyline the rope is built along (>= 2 points, metres, rig/world frame): "
                            "measure_rope's centreline_rig (or points), its result dict, or a JSON file holding either; "
                            "[x, y] points lie on the table (z = radius). Default None (straight)."},
            {'name': 'rest_vertices',
             'type': 'list[[x, y, z]] | None',
             'description': 'The exact cable vertices (>= 3, metres): one capsule per consecutive pair; segments, length '
                            'and pos follow from them. Default None. Give this or centreline, not both.'},
            {'name': 'bend_stiffness',
             'type': 'float',
             'description': 'Bend stiffness, N/m. Default 100000.0'},
            {'name': 'twist_stiffness',
             'type': 'float',
             'description': 'Twist stiffness, N/m. Default 50000.0'},
            {'name': 'joint_damping',
             'type': 'float',
             'description': 'Joint damping, joint damping. Default 0.012'},
            {'name': 'joint_frictionloss',
             'type': 'float',
             'description': 'Joint frictionloss, MuJoCo friction triple. Default 0.02'},
            {'name': 'joint_armature',
             'type': 'float',
             'description': 'Joint armature. Default 1e-05'},
            {'name': 'joint_fric_solref',
             'type': 'list[2]',
             'description': 'Joint fric solref, MuJoCo solref. Default [0.001, 1.0]'},
            {'name': 'joint_fric_solimp',
             'type': 'list[5]',
             'description': 'Joint fric solimp, MuJoCo solimp. Default [0.999, 0.9999, 0.0001, '
                            '0.5, 2]'},
            {'name': 'density',
             'type': 'float',
             'description': 'Kg/m3 -> mass follows from radius+length. Default 1100.0'},
            {'name': 'friction',
             'type': 'list[3]',
             'description': 'Friction, MuJoCo friction triple. Default [0.7, 0.02, 0.001]'},
            {'name': 'condim', 'type': 'int', 'description': 'Condim. Default 4'},
            {'name': 'contact_solref',
             'type': 'list[2]',
             'description': 'Contact solref, MuJoCo solref. Default [0.004, 1.0]'},
            {'name': 'grips',
             'type': 'bool',
             'description': 'Mocap grip at each end (weld on/off = grasp). Default True'},
            {'name': 'grip_solref',
             'type': 'list[2]',
             'description': 'Grip solref, MuJoCo solref. Default [0.01, 1.0]'},
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
 'tags': ['rope', 'cable', 'deformable', 'knot', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Knot the Rope', 'Scissor Cutting'],
 'status': 'tested',
 'entrypoint': 'construct_rope',
 'cli': 'python -m tool_library.skills.construct_rope out_dir'}




import json
from pathlib import Path

from tool_library.common import cable as _core

# every key below is a parameter of construct_rope(**params); unknown keys raise
DEFAULTS = _core.DEFAULTS


SEG_M = 0.015          # a measured centreline gets one cable vertex per 15 mm (the default rope's spacing)
MIN_VERTS, MAX_VERTS = 20, 160


def _laid_params(params):
    """A centreline becomes the cable's rest vertices (resampled by arc length); parameters the points decide
    (length, pos, yaw_deg, and segments for rest_vertices) are refused when they contradict them, never ignored."""
    import numpy as np
    p = dict(params)
    has_c, has_v = p.get("centreline") is not None, p.get("rest_vertices") is not None
    if has_c and has_v:
        raise ValueError("construct_rope: give centreline (resampled to `segments` vertices) or rest_vertices (used "
                         "exactly), not both")
    if not (has_c or has_v):
        return p
    radius = float(p.get("radius", DEFAULTS["radius"]))
    if has_c:
        from tool_library.common.cutrope import load_centreline
        raw = p["centreline"]
        if isinstance(raw, (str, Path)):           # measure_rope's JSON (or a bare point list in JSON)
            raw = json.loads(Path(raw).read_text())
        if isinstance(raw, dict):                   # measure_rope's result: its rig-frame points
            raw = next((raw[k] for k in ("points", "centreline", "centreline_rig", "centreline_m", "centreline_world",
                                         "polyline", "path", "centreline_table") if raw.get(k) is not None), raw)
        flat = np.asarray(raw, dtype=object).ndim == 2 and np.asarray(raw, float).shape[-1] == 2
        P = load_centreline(raw)
        if flat:
            P[:, 2] = radius                       # a polyline on the table: the rope's axis is one radius up
        S = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        L = float(S[-1])
        if L < 0.02:
            raise ValueError(f"construct_rope: the centreline is {L * 1000:.1f} mm long; a rope needs at least 2 cm")
        n = int(p["segments"]) if "segments" in params else int(np.clip(np.ceil(L / SEG_M) + 1, MIN_VERTS, MAX_VERTS))
        if n < 3:
            raise ValueError("construct_rope: segments must be >= 3 with a centreline")
        t = np.linspace(0.0, L, n)
        verts = np.stack([np.interp(t, S, P[:, i]) for i in range(3)], axis=1)
        p.pop("segments", None)
        p["rest_vertices"] = verts.tolist()
        p["centreline"] = P.tolist()                # kept in the params sheet: reset_measured lays it back down
        src = "centreline"
    else:
        verts = np.asarray(p["rest_vertices"], float)
        if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
            raise ValueError("construct_rope: rest_vertices must be >= 3 [x, y, z] points (metres)")
        if "segments" in params and int(params["segments"]) != len(verts):
            raise ValueError(f"construct_rope: segments {params['segments']} contradicts the {len(verts)} rest_vertices; "
                             f"leave segments out (it follows from the vertices)")
        p.pop("segments", None)
        L = float(np.linalg.norm(np.diff(verts, axis=0), axis=1).sum())
        src = "rest_vertices"
    if "length" in params and abs(float(params["length"]) - L) > 0.05 * L:
        raise ValueError(f"construct_rope: length {float(params['length']):.3f} m contradicts the {src}'s {L:.3f} m; "
                         f"leave length out (it follows from the points)")
    if "pos" in params and float(np.linalg.norm(np.asarray(params["pos"], float) - verts[0])) > 0.005:
        raise ValueError(f"construct_rope: pos {list(params['pos'])} is not the {src}'s first point "
                         f"{np.round(verts[0], 4).tolist()}; leave pos out (it follows from the points)")
    if abs(float(params.get("yaw_deg", 0.0) or 0.0)) > 1e-9:
        raise ValueError(f"construct_rope: yaw_deg turns a straight rope; the {src} already carries the heading, leave it out")
    for k in ("length", "pos", "yaw_deg"):
        p.pop(k, None)
    return p


def construct_rope(out_dir, preview=False, **params):
    """Build the asset into out_dir from the parameters above (any key of the
    DEFAULTS above; unknown keys raise). Returns the build dictionary."""
    built = _core.build(_laid_params(params), out_dir)
    if preview and hasattr(_core, "preview"):
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    print(_json.dumps(construct_rope(_sys.argv[1] if len(_sys.argv) > 1 else "."), indent=1, default=str))
