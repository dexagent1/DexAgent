"""Opening and closing are opposite motions along the same articulation."""
import sys
import unittest
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool_library.verifiers.articulation_axis_semantics_check import articulation_axis_semantics_check, parse_cues


class ReversibleSemanticsTest(unittest.TestCase):
    def check(self, axis_words, world_axis='0 -1 0', joint_range='0 .2'):
        model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body name="cabinet">
         <geom type="box" size=".08 .15 .05"/>
         <body name="tray"><joint name="slide" type="slide" axis="{world_axis}" range="{joint_range}"/>
         <geom type="box" size=".07 .14 .02"/></body></body></worldbody></mujoco>''')
        sem = {'objects': [{'id': 'cabinet', 'type': 'articulated', 'articulation': {
            'joint_type': 'prismatic', 'axis': axis_words, 'moving_part': 'tray', 'fixed_part': 'cabinet'}}]}
        recon = {'objects': [{'id': 'cabinet', 'root_body': 'cabinet'}]}
        return articulation_axis_semantics_check(model, sem, recon)

    def test_reversible_motion_has_one_opening_direction(self):
        axis = 'horizontal, toward the operator while opening and away from the operator while closing'
        self.assertEqual(parse_cues(axis), ['horizontal', 'open_toward_operator'])
        self.assertTrue(self.check(axis)['ok'])
        self.assertFalse(self.check(axis, '0 1 0')['ok'])

    def test_closing_only_inverts_opening_sign(self):
        axis = 'horizontal, away from the operator to close'
        self.assertTrue(self.check(axis)['ok'])
        self.assertFalse(self.check(axis, '0 1 0')['ok'])
        self.assertTrue(self.check('horizontal, toward the operator to close', '0 1 0')['ok'])

    def test_condition_before_comma_is_preserved(self):
        self.assertEqual(parse_cues('horizontal; when closing, away from the operator'),
                         ['horizontal', 'open_toward_operator'])

    def test_state_transition_conditions_direction(self):
        self.assertTrue(self.check('horizontal; open -> closed, away from the operator')['ok'])
        self.assertEqual(parse_cues('horizontal; closed -> open toward the operator'),
                         ['horizontal', 'open_toward_operator'])

    def test_negative_joint_range_preserves_physical_opening_direction(self):
        axis = 'horizontal, toward the operator to open and away from the operator to close'
        self.assertTrue(self.check(axis, '0 1 0', '-.2 0')['ok'])
        self.assertFalse(self.check(axis, '0 -1 0', '-.2 0')['ok'])

    def test_unsigned_bidirectional_description_still_checks_axis(self):
        for words in ('horizontal, toward and away from the operator',
                      'horizontal, toward the operator and away from the operator'):
            with self.subTest(words=words):
                self.assertTrue(self.check(words)['ok'])
                self.assertTrue(self.check(words, '0 1 0')['ok'])
                self.assertFalse(self.check(words, '1 0 0')['ok'])
                self.assertIn('operator_axis', parse_cues(words))

    def test_contradictory_opening_sign_remains_unverified(self):
        report = self.check('horizontal, opens toward the operator and opens away from the operator')
        self.assertFalse(report['ok'])
        self.assertIn('contradictory', report['issues'][0])
        self.assertNotIn('orient the joint', report['issues'][0])

    def test_closing_image_direction_is_inverted(self):
        self.assertEqual(parse_cues('horizontal, closes away from the operator (up and right in the image)'),
                         ['horizontal', 'image_down', 'image_left', 'open_toward_operator'])
        self.assertEqual(parse_cues('horizontal, toward the operator (down and left in the image while opening)'),
                         ['horizontal', 'image_down', 'image_left', 'open_toward_operator'])

    def test_conditioned_pose_changes_are_not_opening_directions(self):
        self.assertEqual(parse_cues('long axis (vertical when standing; left-right while held)'), ['long_axis'])


if __name__ == '__main__':
    unittest.main()
