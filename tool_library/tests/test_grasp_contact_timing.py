"""Contact-speed feedback must change the executed reference, not be ignored."""
import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.subgoal import SubgoalPlanner
from tool_library.common import plan as P
from tool_library.skills.stage_on_bench import stage_on_bench


class ContactTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parents[2]
        ws = Path(cls.temp.name)
        asset = ws / 'box.xml'
        asset.write_text('<mujoco><worldbody><body name="box"><freejoint/>'
                         '<geom type="box" size=".024 .019 .028" pos="0 0 .028" mass=".1"/>'
                         '</body></worldbody></mujoco>')
        stage_on_bench(str(root / 'setup/scene.xml'),
                       assets=[dict(name='box', xml=str(asset), x=.12, y=.38, yaw_deg=20.)],
                       out_xml=str(ws / 'scene.xml'), rig={'origin': [0, 0, 0], 'R': np.eye(3).tolist()})
        cls.model = mujoco.MjModel.from_xml_path(str(ws / 'scene.xml'))
        cls.pose = json.loads((root / 'tool_library/tests/fixtures/grasp_robust/box_bench_right_generated.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def plan(self, motion, **timing):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        planner = SubgoalPlanner(self.model, data.qpos.copy(), fps=30.)
        return planner.run(dict(index=1, side='right', object='box_box', pose=self.pose,
                                motion=motion, mode='grip', **timing))['keyframes']

    def test_slow_contact_preserves_positions_and_increases_transition_time(self):
        for motion in ('grasp_lift', 'hold', 'pin'):
            with self.subTest(motion=motion):
                baseline = self.plan(motion)
                slower = self.plan(motion, t_reach=4., t_descend=3., t_close=2., t_grip=1.5)
                self.assertEqual(len(baseline), len(slower))
                for before, after in zip(baseline, slower):
                    np.testing.assert_array_equal(before['qpos'], after['qpos'])
                for needle, seconds in [('descend', 3.), ('close', 2.)]:
                    frame = next(k for k in slower if needle in k['label'])
                    old = next(k for k in baseline if needle in k['label'])
                    self.assertGreaterEqual(frame['transit'], 30. * seconds)
                    self.assertGreater(frame['transit'], old['transit'])

    def test_invalid_contact_durations_reject_before_planning(self):
        for duration in (0., -1., float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, 'finite positive'):
                self.plan('grasp_lift', t_close=duration)

    def test_new_grasp_lift_honors_requested_lift_duration(self):
        baseline = self.plan('grasp_lift')
        slower = self.plan('grasp_lift', duration_s=2.4)
        self.assertEqual(len(baseline), len(slower))
        for a, b in zip(baseline, slower):
            np.testing.assert_array_equal(a['qpos'], b['qpos'])
        old = next(k for k in baseline if ': lift ' in k['label'])
        new = next(k for k in slower if ': lift ' in k['label'])
        self.assertGreater(new['transit'], old['transit'])
        self.assertGreaterEqual(new['transit'], 72)
        for duration in (0., -1., float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError, 'finite positive'):
                self.plan('grasp_lift', duration_s=duration)

    def test_release_records_requested_settle_before_its_outcome(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        planner = SubgoalPlanner(self.model, data.qpos.copy(), fps=30.)
        planner.run(dict(index=1, motion='grasp_lift', side='right', object='box_box', pose=self.pose))
        release = planner.run(dict(index=2, motion='release', side='right', object='box_box',
                                   t_open=1., t_retreat=2., settle_s=1.2))
        frames = release['keyframes']
        self.assertIn('settle after release', frames[-1]['label'])
        self.assertEqual(frames[-1]['hold'], 36)
        np.testing.assert_array_equal(frames[-1]['qpos'], frames[-2]['qpos'])
        self.assertGreaterEqual(max(k['transit'] for k in frames if 'let go' in k['label']), 30)
        self.assertTrue(release['expect']['at_rest'])
        self.assertTrue(release['expect']['released'])

    def test_release_duration_scales_opening_and_retreat(self):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        planner = SubgoalPlanner(self.model, data.qpos.copy(), fps=30.)
        planner.run(dict(index=1, motion='grasp_lift', side='right', object='box_box', pose=self.pose))
        frames = planner.run(dict(index=2, motion='release', side='right', object='box_box', duration_s=2.6))['keyframes']
        self.assertGreaterEqual(max(k['transit'] for k in frames if 'let go' in k['label']), 30)
        self.assertGreaterEqual(max(k['transit'] for k in frames if 'retreat' in k['label']), 48)

    def together(self, *, left=None, right=None, **timing):
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        planner = SubgoalPlanner(self.model, data.qpos.copy(), fps=30.)
        poses = {side: copy.deepcopy(self.pose) for side in ('left', 'right')}
        poses['left']['pos'][0] *= -1
        hands = {side: {'object': 'box_box', 'pose': poses[side], **(entry or {})}
                 for side, entry in [('left', left), ('right', right)]}
        return planner, planner.run(dict(index=1, motion='lift_together', hands=hands, **timing))

    def test_together_contact_timing_changes_the_reference_and_reports_used_values(self):
        _, baseline = self.together()
        _, slower = self.together(t_reach=3., t_descend=2., t_close=1., t_grip=.6,
                                  left={'t_descend': 4.}, right={'t_descend': 1.})
        for before, after in zip(baseline['keyframes'], slower['keyframes']):
            np.testing.assert_array_equal(before['qpos'], after['qpos'])
        self.assertEqual(slower['approach_controls']['transit_s'],
                         dict(t_reach=3., t_descend=4., t_close=1., t_grip=.6))
        for label, seconds in [('reach for', 3.), ('descend on', 4.), ('close on', 1.), ('grip', .6)]:
            self.assertEqual(next(k for k in slower['keyframes'][1:] if label in k['label'])['transit'], 30 * seconds)

    def test_together_unequal_waypoints_keep_peer_open_and_stationary(self):
        left_pose = copy.deepcopy(self.pose)
        left_pose['pos'][0] *= -1
        left_pose['approach_waypoints'] = [[-.03, .02, .07], [-.02, .01, .04]]
        planner, result = self.together(left={'pose': left_pose})
        frames = result['keyframes']
        way = [k for k in frames if ': approach (' in k['label']]
        self.assertEqual(len(way), 2)
        reach = next(k for k in frames[1:] if ': reach for' in k['label'])
        peer_rig = planner.pl.rigs['right']
        peer_qadr = list(peer_rig.wrist_qpos.values()) + list(peer_rig.finger_qpos.values())
        for k, off in zip(way, left_pose['approach_waypoints']):
            np.testing.assert_array_equal(np.asarray(k['qpos'])[peer_qadr], np.asarray(reach['qpos'])[peer_qadr])
            data = mujoco.MjData(self.model); data.qpos[:] = k['qpos']; mujoco.mj_forward(self.model, data)
            wrist, _ = P.wrist_world(self.model, data, planner.pl.rigs['left'])
            np.testing.assert_allclose(wrist, np.asarray(left_pose['pos']) + off, atol=1e-9)
        descend = next(k for k in frames if ': descend on' in k['label'])
        data.qpos[:] = descend['qpos']; mujoco.mj_forward(self.model, data)
        wrist, _ = P.wrist_world(self.model, data, planner.pl.rigs['left'])
        np.testing.assert_allclose(wrist, left_pose['pos'], atol=1e-9)
        self.assertEqual(result['approach_controls']['hands']['left']['approach_waypoints'], left_pose['approach_waypoints'])

    def test_together_rejects_invalid_contact_times_and_waypoints(self):
        for value in (0., -1., float('nan'), float('inf')):
            for arguments in ({'t_close': value}, {'left': {'t_descend': value}}, {'duration_s': value}):
                with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, 'finite positive'):
                    self.together(**arguments)
        for waypoint in ([0., 1.], [0., float('nan'), 0.]):
            pose = copy.deepcopy(self.pose); pose['approach_waypoints'] = [waypoint]
            with self.subTest(waypoint=waypoint), self.assertRaisesRegex(ValueError, 'finite xyz'):
                self.together(left={'pose': pose})


if __name__ == '__main__':
    unittest.main()
