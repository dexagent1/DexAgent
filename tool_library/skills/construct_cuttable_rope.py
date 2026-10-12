"""Construct a cuttable rope: two MuJoCo cables laid on a measured centreline and joined at the cut point by one CONNECT equality that the Stage 3 executor switches off when scissor blades really close on it.

Thin entry point over tool_library/common/cutrope.py (the builder) and scissors.attach_cuttable_thread (the join).
"""
from __future__ import annotations

TOOL = {'name': 'construct_cuttable_rope',
 'title': 'Construct cuttable rope',
 'type': 'skill',
 'category': 'Object Reconstruction',
 'summary': 'A rope that can be cut in MuJoCo: two elastic cables built along the measured centreline, split at the '
            'cut location and tied there by one CONNECT equality, as one MJCF that stage_on_bench stages like '
            'construct_rope; returns the Stage 2 manifest fields (cut_joins, halves) Stage 3 needs to cut it.',
 'description': 'MuJoCo cannot break a body chain at run time, so a rope the task cuts is authored as two cables '
                'and one CONNECT equality at the cut point; switching the equality off is the cut, and the Stage 3 '
                'executor (execute_subgoal_plan) does it through scissors.try_sever only when the blades have closed '
                'on the join (in their span, wedge and plane, closing, biting). The measured centreline (measure_rope '
                'output, a hand trace back-projected to the table, any polyline in rig metres) is densified, crossings '
                'of the rope with itself are lifted (the later strand over the earlier one by one diameter), and the '
                'polyline is split at the cut location: an arc-length fraction (cut_frac) or the rope point nearest a '
                'world point (cut_point). Each half is BUILT along its own piece (cable vertices), so qpos0 is the '
                'measured shape, both halves end exactly on the cut point (continuous at the join in the reference '
                'pose the CONNECT is baked from) and every keyframe stage_on_bench writes already has the rope laid '
                'where the video has it, with no keyframe editing. The two capsules at the join do not collide with '
                'each other (an exclude); everything else self-collides as in construct_rope. Stage it with '
                'stage_on_bench at x = y = z = 0, yaw 0 under the name `scene_name`; `scene_fields` in the result '
                'holds root_body, bodies, halves {a, b: root_body, first, last, bodies} and cut_joins [{eq, body, '
                'anchor_local, diameter_m, s_m, point, other_body}] in the staged scene names: copy them into the '
                'object\'s scene_recon.json entry (Stage 3 reads `cut_joins`). For an Isaac render of the same rope, '
                'build_isaac_scene takes the staged scene as usual.',
 'inputs': [{'name': 'out_dir', 'type': 'path', 'description': 'Folder that receives <name>.xml, <name>_params.json and the two halves.'},
            {'name': 'centreline', 'type': 'list[[x, y, z]] | dict | path',
             'description': 'The measured centreline in rig metres: a point list, a dict holding one (points, '
                            'centreline, centreline_rig, centreline_m, polyline) or a JSON file with either '
                            '(measure_rope output, a rope trace).'},
            {'name': 'cut_frac', 'type': 'float', 'description': 'Cut location as a fraction of the arc length from the first point. Default 0.5.'},
            {'name': 'cut_point', 'type': 'list[3] | None', 'description': 'Cut at the rope point nearest this world point instead (wins over cut_frac).'},
            {'name': 'scene_name', 'type': 'str | None', 'description': 'The asset name stage_on_bench will stage it under (prefix of every body, joint and equality); default `name`.'},
            {'name': 'preview', 'type': 'bool', 'description': 'Also render a still from above. Default False.'},
            {'name': 'name', 'type': 'str', 'description': 'Rope name. Default rope.'},
            {'name': 'radius', 'type': 'float', 'description': 'Rope radius, metres (measure_rope radius_m). Default 0.0035.'},
            {'name': 'segments', 'type': 'int', 'description': 'Capsules for the whole rope, shared by the halves by length. Default 48.'},
            {'name': 'min_half_segments', 'type': 'int', 'description': 'Fewest capsules a half may have. Default 3.'},
            {'name': 'z', 'type': 'float | None', 'description': 'Height of every centreline point, metres (one radius for a rope lying on the table); None keeps the points\' z.'},
            {'name': 'lift_crossings', 'type': 'bool', 'description': 'Lift the later strand over the earlier one where the centreline crosses itself. Default True.'},
            {'name': 'join_solref', 'type': 'list[2]', 'description': 'Stiffness of the join equality. Default [0.004, 1.0].'},
            {'name': 'rgba, cap_length, cap_length_a, cap_length_b, cap_rgba, cap_radius_scale, specular, shininess, cap_specular, cap_shininess',
             'type': 'see construct_rope', 'description': 'Appearance, as construct_rope. The end caps go on the rope\'s two real ends only (cap_length_a at the first point, cap_length_b at the last; cap_length sets both), never at the cut.'},
            {'name': 'bend_stiffness, twist_stiffness, joint_damping, joint_frictionloss, joint_armature, joint_fric_solref, joint_fric_solimp, density, friction, condim, contact_solref',
             'type': 'see construct_rope', 'description': 'Material and contact, as construct_rope.'},
            {'name': 'grips, grip_rgba, grip_solref', 'type': 'see construct_rope', 'description': 'Mocap grips at the ends of each half; default False (hands grasp the rope in a bench scene).'},
            {'name': 'timestep, include_floor', 'type': 'float, bool', 'description': 'Defaults 0.002, False.'}],
 'outputs': [{'name': 'built',
              'type': 'dict',
              'description': 'xml, params_json, root_body, bodies, halves {a, b}, join {eq, body, anchor_local, '
                             'diameter_m, s_m, point}, cut {s_m, frac, point, off_centreline_m}, length_m, radius_m, '
                             'crossings [{s_under, s_over, point}], join_gap_mm_at_build (0 = continuous), and '
                             'scene_fields {root_body, bodies, halves, cut_joins, stage_on_bench}: the manifest fields '
                             'in the staged scene names and the stage_on_bench asset entry.'}],
 'requirements': ['mujoco>=3.2', 'numpy'],
 'simulators': ['MuJoCo'],
 'tags': ['rope', 'cable', 'deformable', 'cut', 'scissors', 'severable', 'parametric', 'reconstruct', 'object reconstruction'],
 'used_in': ['Scissor Cutting'],
 'status': 'tested',
 'entrypoint': 'construct_cuttable_rope',
 'cli': 'python -m tool_library.skills.construct_cuttable_rope out_dir centreline.json [cut_frac]'}


from tool_library.common import cutrope as _core

# every key below is a parameter of construct_cuttable_rope(**params); unknown keys raise
DEFAULTS = _core.DEFAULTS


def construct_cuttable_rope(out_dir, centreline=None, cut_frac=None, cut_point=None, scene_name=None, preview=False, **params):
    """Build the cuttable rope into out_dir along `centreline`, cut at `cut_frac` (or nearest `cut_point`)."""
    p = dict(params)
    if centreline is not None:
        p["centreline"] = centreline
    if cut_frac is not None:
        p["cut_frac"] = float(cut_frac)
    if cut_point is not None:
        p["cut_point"] = [float(v) for v in cut_point]
    built = _core.build(p, out_dir)
    built["scene_fields"] = _core.scene_fields(built, scene_name)
    if preview:
        built["stills"] = _core.preview(built, out_dir)
    return built


if __name__ == "__main__":
    import json as _json
    import sys as _sys
    a = _sys.argv[1:]
    print(_json.dumps(construct_cuttable_rope(a[0], centreline=a[1], cut_frac=float(a[2]) if len(a) > 2 else None),
                      indent=1, default=str))
