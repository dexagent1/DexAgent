"""Actual executor record shapes yield concrete per-hand repair calls."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.skills.diagnose_stage3_failure import diagnose_stage3_failure


class BimanualFailureDiagnosisTest(unittest.TestCase):
    def setUp(self):
        self.plan = {'subgoals': [{'index': 3, 'step': 2, 'motion': 'both', 'hands': {
            'left': {'motion': 'grasp_lift', 'object': 'part_left', 'pose': 'left_pose.json', 'hand_scene': 'left_hs.json'},
            'right': {'motion': 'open_joint', 'object': 'part_right', 'pose': 'right_pose.json', 'hand_scene': 'right_hs.json',
                      'joint': 'hinge_right', 'target': 1.2, 'hand_follows': 'level'}}}]}
        self.record = {'index': 3, 'step': 2, 'motion': 'both', 'side': 'both', 'ok': False,
                       'object': {'left': 'part_left', 'right': 'part_right'}, 'measured': {'hands': {
                           'left': {'rise_m': 0.}, 'right': {'joint_change': .1}}}}

    def diagnose(self, checks, measured=None):
        record = dict(self.record, failed=checks)
        if measured is not None:
            record['measured'] = measured
        return diagnose_stage3_failure(outcomes=[record], plan=self.plan, scene='scene.xml',
                                       include_workspace_history=False, check_hand_scenes=False)

    def test_failed_left_pickup_is_grasp_failure_with_left_pose(self):
        result = self.diagnose(['left:rise', 'left:link_in_hand:part_left'])
        self.assertEqual(result['n_failed_outcomes'], 1)
        self.assertEqual([c['class'] for c in result['classes']], ['no_stress_pass'])
        self.assertEqual(result['classes'][0]['objects'], ['part_left'])
        repair = next(r for r in result['repairs'] if r['tool'] == 'optimize_grasp_pose')
        self.assertEqual(repair['side'], 'left')
        self.assertEqual(repair['args']['starts'], 'left_pose.json')
        self.assertEqual(repair['args']['hand_scene'], 'left_hs.json')

    def test_distinct_hand_failures_keep_each_mechanism_and_context(self):
        result = self.diagnose(['left:arm_unreachable', 'right:arm_unreachable'])
        reach = {r['args']['side']: r['args'] for r in result['repairs'] if r['tool'] == 'optimize_pose_for_reach'}
        self.assertEqual(set(reach), {'left', 'right'})
        self.assertEqual(reach['left']['pose'], 'left_pose.json')
        self.assertNotIn('sweep', reach['left'])
        self.assertEqual(reach['right']['hand_scene'], 'right_hs.json')
        self.assertEqual(reach['right']['sweep'], {'joint': 'hinge_right', 'to': 1.2, 'hand_follows': 'level'})
        self.assertFalse(any(r['args'].get('side') == 'both' for r in result['repairs']))

    def test_unprefixed_arm_failure_uses_measured_arm(self):
        result = self.diagnose(['arm_unreachable'], {'infeasible': [{'side': 'right', 'residual_mm': 22.}]})
        self.assertTrue(result['repairs'])
        self.assertEqual({r['side'] for r in result['repairs']}, {'right'})
        self.assertEqual(result['classes'][0]['objects'], ['part_right'])

    def test_shared_lift_and_carry_use_their_correct_grasp_or_slip_class(self):
        self.plan['subgoals'][0]['motion'] = 'lift_together'
        self.record['motion'] = 'lift_together'
        for entry in self.plan['subgoals'][0]['hands'].values():
            entry.pop('motion')
        lift = self.diagnose(['left:rise'])
        self.assertEqual(lift['classes'][0]['class'], 'no_stress_pass')
        self.plan['subgoals'][0]['motion'] = self.record['motion'] = 'carry_together'
        carry = self.diagnose(['right:link_in_hand:part_right'])
        self.assertEqual(carry['classes'][0]['class'], 'slip_in_carry')
        self.assertEqual(carry['repairs'][0]['args']['hand_scene'], 'right_hs.json')

    def test_other_class_does_not_reuse_first_failed_hands_pose(self):
        result = self.diagnose(['left:rise', 'right:arm_unreachable'])
        reach = next(r for r in result['repairs'] if r['tool'] == 'optimize_pose_for_reach')
        self.assertEqual(reach['args']['side'], 'right')
        self.assertEqual(reach['args']['pose'], 'right_pose.json')
        self.assertEqual(reach['args']['sweep']['joint'], 'hinge_right')

    def test_single_hand_and_caller_inputs_remain_unchanged(self):
        before = copy.deepcopy((self.plan, self.record))
        self.diagnose(['left:rise'])
        self.assertEqual(before, (self.plan, self.record))
        single = dict(self.record, motion='grasp_lift', side='left', object='part_left', failed=['rise'])
        plan = {'subgoals': [{'index': 3, 'step': 2, **self.plan['subgoals'][0]['hands']['left'], 'side': 'left'}]}
        result = diagnose_stage3_failure(outcomes=[single], plan=plan, include_workspace_history=False, check_hand_scenes=False)
        self.assertEqual(result['classes'][0]['class'], 'no_stress_pass')
        self.assertEqual(result['repairs'][0]['args']['starts'], 'left_pose.json')


if __name__ == '__main__':
    unittest.main()
