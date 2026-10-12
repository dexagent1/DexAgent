"""Generic final-language structures and independently measured counterexamples.

These synthetic state arrays test goal evaluation only, not dynamics or robot
execution. No sampled task geometry or generated demonstration is reused.
"""
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import task_gates as G
from tool_library.common.final_goal_language import compile_final, measure_custom


CAMERA = '<camera name="video_cam" pos="0 -.7 1" quat=".9238795 .3826834 0 0"/>'
ENV = {'support_surface': 'white tabletop', 'camera': 'Fixed egocentric camera looking down toward the work surface.'}


class FinalStructureTests(unittest.TestCase):
    def context(self, model, sem, q):
        recon = {'objects': [{'id': o['id'], 'root_body': o['id']} for o in sem['objects']]}
        return G.GoalContext(G.Episode(model, qpos=q, fps=30.), sem, recon)

    def report(self, model, sem, q):
        return G.task_success_report(self.context(model, sem, q), sem)

    def stack(self):
        items = [('pedestal', 'umber display pedestal', [0, 0, .02], [.07, .045, .02]),
                 ('tile', 'cobalt mosaic tile', [0, 0, .055], [.055, .04, .015]),
                 ('west', 'silver left mosaic tile', [-.0275, 0, .08], [.0275, .04, .01]),
                 ('east', 'ochre right mosaic tile', [.0275, 0, .08], [.0275, .04, .01]),
                 ('cap', 'indigo cap', [0, 0, .10], [.035, .035, .01])]
        bodies = ''.join(f'<body name="{oid}" pos="{" ".join(map(str, p))}"><freejoint/>'
                         f'<geom type="box" size="{" ".join(map(str, sz))}" mass=".05"/></body>'
                         for oid, _, p, sz in items)
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/>' + CAMERA + bodies + '</worldbody></mujoco>')
        sem = {'objects': [{'id': oid, 'name': name, 'type': 'rigid'} for oid, name, _, _ in items],
               'environment': ENV,
               'task_success': 'A stack stands on the tabletop: from the bottom, the umber pedestal on the table, '
                               'the cobalt tile on it, the silver tile and ochre tile side by side on the cobalt, '
                               'and the indigo cap bridging them on top; nothing is held.'}
        return model, sem, np.repeat(model.qpos0[None], 40, axis=0)

    def test_abbreviated_names_coordinated_supports_and_stack_pronouns(self):
        model, sem, q = self.stack()
        result = self.report(model, sem, q)
        self.assertTrue(result['ok'], result)
        self.assertEqual(len([p for p in result['predicates'] if p['class'] == 'stacked_on']), 4)
        # A bridge resting on only one support cannot satisfy 'bridging them'.
        broken = q.copy(); broken[-20:, 28] = -.055
        result = self.report(model, sem, broken)
        self.assertFalse(result['ok'])
        self.assertTrue(any(p['class'] == 'stacked_on' and not p['ok'] for p in result['predicates']))

    def test_stack_cycle_empty_structure_and_uncovered_claim_fail_closed(self):
        model, sem, q = self.stack()
        for text in ['A tower stands on the table.',
                     'A stack stands on the table: the pedestal on the tile, the tile on the pedestal.',
                     sem['task_success'] + ' The cap radiates warmth.']:
            with self.subTest(text=text):
                sem1 = dict(sem, task_success=text)
                self.assertFalse(self.report(model, sem1, q)['ok'])

    def test_unknown_colour_ambiguous_abbreviation_and_bad_pronoun_fail(self):
        model, sem, q = self.stack()
        for text in ['The magenta cobalt tile rests on the pedestal.',
                     'The mosaic tile rests on the pedestal.',
                     'The silver tile and ochre tile rest on the cobalt tile, and the cap bridging it on top.']:
            _, _, issues = compile_final(self.context(model, sem, q), text)
            self.assertTrue(issues, text)

    def test_upright_on_support_is_not_inside_and_full_support_is_not_assumed(self):
        model, sem, q = self.stack()
        sem['task_success'] = 'The cobalt tile stands upright on the pedestal.'
        result = self.report(model, sem, q)
        self.assertTrue(result['ok'], result)
        self.assertIn('stacked_on', [p['class'] for p in result['predicates']])
        self.assertNotIn('inside_container', [p['class'] for p in result['predicates']])
        sem['task_success'] = 'The cobalt tile is wholly on the pedestal.'
        self.assertFalse(self.report(model, sem, q)['ok'])

    def cabinet(self, missing_secondary=False):
        cavity = ('<geom type="box" size=".09 .09 .005" pos="0 0 .035"/>'
                  '<geom type="box" size=".005 .09 .025" pos=".085 0 .065"/>'
                  '<geom type="box" size=".005 .09 .025" pos="-.085 0 .065"/>'
                  '<geom type="box" size=".09 .005 .025" pos="0 .085 .065"/>'
                  '<geom type="box" size=".09 .005 .025" pos="0 -.085 .065"/>')
        secondary = ''.join(f'<body name="{part}" pos="0 0 {z}"><joint name="{part}_slide" type="slide" axis="0 -1 0" limited="true" range="0 .1"/>'
                            '<geom type="box" size=".09 .09 .025" mass=".1"/></body>'
                            for part, z in [('upper_drawer', .15), ('top_drawer', .23)] if not (missing_secondary and part == 'top_drawer'))
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/>' + CAMERA +
            '<body name="cubby"><freejoint/><geom type="box" size=".11 .11 .015" pos="0 0 .015"/>'
            '<body name="lower_drawer"><joint name="lower_slide" type="slide" axis="0 -1 0" limited="true" range="0 .1"/>' + cavity + '</body>' + secondary +
            '</body><body name="token" pos="0 0 .054"><freejoint/><geom type="sphere" size=".012" mass=".03"/></body></worldbody></mujoco>')
        sem = {'objects': [{'id': 'cubby', 'name': 'brown cubby', 'type': 'articulated',
                            'parts': ['lower_drawer', 'upper_drawer', 'top_drawer'],
                            'articulation': {'moving_part': 'lower_drawer', 'fixed_part': 'housing'}},
                           {'id': 'token', 'name': 'orange token', 'type': 'rigid'}],
               'environment': ENV,
               'task_success': "The orange token has been removed from the lower drawer and placed on the tabletop to the cubby's right, "
                               'and the lower drawer is closed with the cubby upright and the other drawers unchanged.'}
        q = np.repeat(model.qpos0[None], 40, axis=0)
        adr = int(model.jnt_qposadr[model.body('token').jntadr[0]])
        q[5:, adr:adr+3] = [.2, 0, .012]
        return model, sem, q, adr

    def test_removal_placement_closure_and_all_secondary_parts_are_measured(self):
        model, sem, q, adr = self.cabinet()
        result = self.report(model, sem, q)
        self.assertTrue(result['ok'], result)
        classes = [p['class'] for p in result['predicates']]
        self.assertEqual(classes.count('final_part_unchanged'), 2)
        for key, changed in [('returned', q.copy()), ('never_inside', q.copy()), ('other_opened_then_closed', q.copy())]:
            if key == 'returned': changed[-20:, adr:adr+3] = [0, 0, .054]
            elif key == 'never_inside': changed[:5, adr] = -.3
            else: changed[10, model.joint('upper_drawer_slide').qposadr[0]] = .04
            with self.subTest(key=key):
                self.assertFalse(self.report(model, sem, changed)['ok'])

    def test_unchanged_part_is_relative_to_housing_not_world_and_no_root_fallback(self):
        model, sem, q, _ = self.cabinet()
        q[:, 0] = np.linspace(0., .08, len(q))
        gc = self.context(model, sem, q)
        result = measure_custom(gc, {'class': 'final_part_unchanged', 'A': 'cubby:top_drawer'}, len(q)-1)
        self.assertTrue(result['ok'], result)
        model, sem, q, _ = self.cabinet(missing_secondary=True)
        result = self.report(model, sem, q)
        self.assertFalse(result['ok'])
        self.assertTrue(any(p.get('unresolved') and p['class'] == 'final_part_unchanged' for p in result['predicates']))

    def test_direction_uses_rotated_source_camera_and_missing_camera_fails(self):
        model, sem, q, adr = self.cabinet()
        rot = Rotation.from_euler('z', 90, degrees=True) * Rotation.from_euler('x', 45, degrees=True)
        model.cam_quat[model.camera('video_cam').id] = np.roll(rot.as_quat(), 1)
        q[5:, adr:adr+3] = [0, .3, .012]
        self.assertTrue(self.report(model, sem, q)['ok'])
        q[5:, adr:adr+3] = [.3, 0, .012]
        self.assertFalse(self.report(model, sem, q)['ok'])
        sem['environment'] = dict(ENV, camera='Unspecified camera frame.')
        self.assertFalse(self.report(model, sem, q)['ok'])

    def test_immediate_and_side_by_side_have_distance_and_alignment_constraints(self):
        model, sem, q, adr = self.cabinet()
        sem['task_success'] = "The token is placed on the table immediately to the cubby's right."
        q[5:, adr:adr+3] = [.124, 0, .012]
        self.assertTrue(self.report(model, sem, q)['ok'])
        q[5:, adr] = .6
        self.assertFalse(self.report(model, sem, q)['ok'])
        model, sem, q = self.stack()
        q[-20:, 21 + 1] += .15  # same height, but one of the paired tiles is far behind the other
        result = self.report(model, sem, q)
        self.assertFalse(result['ok'])
        self.assertTrue(any(p['class'] == 'final_spatial' and not p['ok'] for p in result['predicates']))

    def test_behind_requires_forward_order_and_direct_lateral_overlap(self):
        model, sem, q, adr = self.cabinet()
        sem['task_success'] = 'The token is directly behind the cubby.'
        for pos, expected in [([0., .3, .012], True), ([0., -.3, .012], False), ([.4, .3, .012], False)]:
            q[5:, adr:adr+3] = pos
            self.assertEqual(self.report(model, sem, q)['ok'], expected, pos)
        model.cam_quat[model.camera('video_cam').id] = [1, 0, 0, 0]  # vertically downward: behind is undefined
        self.assertFalse(self.report(model, sem, q)['ok'])
        no_camera = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'
            '<body name="cubby"><freejoint/><geom type="box" size=".1 .1 .1"/></body>'
            '<body name="token" pos="0 .3 0"><freejoint/><geom type="sphere" size=".01"/></body>'
            '</worldbody></mujoco>')
        self.assertFalse(self.report(no_camera, sem, np.repeat(no_camera.qpos0[None], 40, axis=0))['ok'])


if __name__ == '__main__':
    unittest.main()
