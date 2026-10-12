"""Indirect contact checks need measured, time-aligned physical support motion."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tool_library.verifiers.hand_object_contact_consistency_check import hand_object_contact_consistency_check


class IndirectContactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='contact_support_')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.sem = {
            'objects': [{'id': 'platform'}, {'id': 'token'}],
            'subgoals': [
                {'index': 1, 'motion': 'pull', 'acting_hand': 'right',
                 'object_refs': ['platform'], 'time_window_s': [0.3, 1.8]},
                {'index': 2, 'motion': 'grasp_lift', 'acting_hand': 'right',
                 'object_refs': ['token'], 'time_window_s': [2.5, 3.0]}]}
        self.recon = {'objects': [{'id': 'platform', 'root_body': 'platform'},
                                  {'id': 'token', 'root_body': 'token'}]}
        self.xyz = np.empty((120, 1, 3))
        for frame in range(120):
            t = frame/30
            u = np.clip((t-0.3)/1.5, 0, 1)
            travel = .12*(3*u*u-2*u*u*u)
            self.xyz[frame, 0] = [-.181+travel, 0, .071]
            if t >= 1.9:
                self.xyz[frame, 0] = [.12, 0, .124]

    def run_check(self, token_y=0, damping=1, sem=None, xyz=None, hinge=False, attachment=None, stiffness=0):
        scene = self.path/'scene.xml'
        extra = ('<actuator><motor joint="token_free" gear="1 0 0 0 0 0"/></actuator>' if attachment == 'motor'
                 else '<equality><weld body1="token" body2="platform"/></equality>' if attachment == 'weld' else '')
        joint_type, axis, upper, token_x = ('hinge', '0 0 1', 1.0, .10) if hinge else ('slide', '1 0 0', .25, 0)
        scene.write_text(f'''<mujoco><compiler angle="radian"/><option timestep="0.002"/><default>
          <geom friction="1.5 0.005 0.0001" solref="0.01 1"/></default><worldbody>
          <geom name="floor" type="plane" size="1 1 .01"/>
          <body name="platform" pos="0 0 .055"><joint name="runner" type="{joint_type}" axis="{axis}"
           range="0 {upper}" damping="{damping}" stiffness="{stiffness}"/><geom type="box" size=".18 .13 .015" mass=".8"/></body>
          <body name="token" pos="{token_x} {token_y} .095"><freejoint name="token_free"/>
           <geom type="box" size=".025 .025 .025" mass=".05"/></body>
          </worldbody>{extra}</mujoco>''')
        hand_xyz = self.xyz if xyz is None else xyz
        names = ['rightIndexFingerTip'] if hand_xyz.shape[1] == 1 else ['leftIndexFingerTip', 'rightIndexFingerTip']
        np.savez(self.path/'hands.npz', names=np.array(names), rig_xyz=hand_xyz)
        (self.path/'sem.json').write_text(json.dumps(self.sem if sem is None else sem))
        (self.path/'recon.json').write_text(json.dumps(self.recon))
        before = scene.read_bytes()
        report = hand_object_contact_consistency_check(scene, self.path/'hands.npz', self.path/'sem.json',
                                                       self.path/'recon.json', fps=30)
        self.assertEqual(scene.read_bytes(), before)
        return report

    def test_real_support_contact_transports_object(self):
        report = self.run_check()
        self.assertTrue(report['ok'], report)
        witness = report['objects']['token']['support_motion_witness']
        self.assertEqual(witness['object_state_writes'], 0)
        self.assertLessEqual(witness['max_applied_joint_force'], 30)
        self.assertGreater(witness['initial_surface_gap_m'], .07)
        self.assertLessEqual(report['objects']['token']['gap_m'], .02)
        self.assertLessEqual(witness['max_physical_support_contact_gap_m'], .02)

    def test_revolute_support_contact_transports_object(self):
        xyz = self.xyz.copy()
        for frame in range(len(xyz)):
            t = frame/30
            u = np.clip((t-.3)/1.5, 0, 1)
            angle = .7*(3*u*u-2*u*u*u)
            xyz[frame, 0] = [-.181*np.cos(angle), -.181*np.sin(angle), .071]
            if t >= 1.9:
                xyz[frame, 0] = [.1*np.cos(.7), .1*np.sin(.7), .124]
        report = self.run_check(xyz=xyz, hinge=True, damping=.05)
        self.assertTrue(report['ok'], report)
        witness = report['objects']['token']['support_motion_witness']
        self.assertEqual(witness['joint_force_cap'], 5)
        self.assertEqual(witness['object_state_writes'], 0)

    def test_misplaced_object_still_fails(self):
        report = self.run_check(token_y=.09)
        self.assertFalse(report['ok'], report)
        self.assertNotIn('support_motion_witness', report['objects']['token'])

    def test_missing_prior_semantic_motion_does_not_authorize_sweep(self):
        sem = copy.deepcopy(self.sem)
        sem['subgoals'][0]['motion'] = 'hold'
        report = self.run_check(sem=sem)
        self.assertFalse(report['ok'], report)

    def test_later_hand_motion_cannot_explain_earlier_contact(self):
        sem = copy.deepcopy(self.sem)
        sem['subgoals'][1]['time_window_s'] = [1.0, 1.2]
        xyz = self.xyz.copy(); xyz[30:37, 0] = [.12, 0, .124]
        report = self.run_check(sem=sem, xyz=xyz)
        self.assertFalse(report['ok'], report)

    def test_other_possible_joint_position_cannot_replace_measured_motion(self):
        xyz = self.xyz.copy(); xyz[60:, 0] = [.23, 0, .124]
        report = self.run_check(xyz=xyz)
        self.assertFalse(report['ok'], report)
        rejected = report['objects']['token']['rejected_support_witnesses']
        self.assertTrue(any('source time window' in r['reason'] for r in rejected), rejected)

    def test_late_penetration_is_checked_across_entire_window(self):
        xyz = self.xyz.copy(); xyz[95, 0] = [.13, 0, .095]
        report = self.run_check(xyz=xyz)
        self.assertFalse(report['ok'], report)
        self.assertNotIn('support_motion_witness', report['objects']['token'])

    def test_overlapping_windows_keep_causal_measured_support_motion(self):
        sem = copy.deepcopy(self.sem)
        sem['subgoals'][0]['acting_hand'] = 'left'
        sem['subgoals'][0]['time_window_s'] = [.3, 2.8]
        xyz = np.zeros((len(self.xyz), 2, 3))
        for frame in range(len(xyz)):
            u = np.clip((frame/30-.3)/1.5, 0, 1)
            xyz[frame, 0] = [-.181+.12*(3*u*u-2*u*u*u), 0, .071]
            xyz[frame, 1] = [.12, 0, .124] if frame >= 75 else [.4, .4, .5]
        report = self.run_check(sem=sem, xyz=xyz)
        self.assertTrue(report['ok'], report)
        witness = report['objects']['token']['support_motion_witness']
        self.assertLess(witness['first_measured_motion_frame'], report['objects']['token']['closest_frame'])
        self.assertEqual(witness['external_force_outside_contact'], 0)
        sem['subgoals'][0]['time_window_s'] = [2.6, 3.2]
        reversed_report = self.run_check(sem=sem, xyz=xyz)
        self.assertFalse(reversed_report['ok'], reversed_report)
        self.assertNotIn('support_motion_witness', reversed_report['objects']['token'])

    def test_unrelated_object_is_not_carried(self):
        report = self.run_check(token_y=.3)
        self.assertFalse(report['ok'], report)
        self.assertNotIn('support_motion_witness', report['objects']['token'])

    def test_non_articulated_measured_travel_fails(self):
        xyz = self.xyz.copy()
        xyz[20:54, 0, 1] += .06
        report = self.run_check(xyz=xyz)
        self.assertFalse(report['ok'], report)

    def test_force_limited_support_cannot_teleport(self):
        report = self.run_check(damping=100000)
        self.assertFalse(report['ok'], report)
        rejected = report['objects']['token']['rejected_support_witnesses']
        self.assertTrue(any('did not follow' in r['reason'] for r in rejected), rejected)

    def test_motor_or_weld_cannot_substitute_for_support_contact(self):
        for attachment in ('motor', 'weld'):
            with self.subTest(attachment=attachment):
                report = self.run_check(attachment=attachment)
                self.assertFalse(report['ok'], report)
                self.assertNotIn('support_motion_witness', report['objects']['token'])

    def test_spring_return_after_contact_cannot_get_unobserved_holding_force(self):
        report = self.run_check(stiffness=60)
        self.assertFalse(report['ok'], report)
        rejected = report['objects']['token']['rejected_support_witnesses']
        self.assertTrue(any('source time window' in r['reason'] for r in rejected), rejected)
        self.assertEqual(rejected[0]['external_force_outside_contact'], 0)

    def test_missing_measured_support_tracking_does_not_pass(self):
        xyz = self.xyz.copy(); xyz[20, 0] = np.nan
        report = self.run_check(xyz=xyz)
        self.assertFalse(report['ok'], report)


if __name__ == '__main__':
    unittest.main()
