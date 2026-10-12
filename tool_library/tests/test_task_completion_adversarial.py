"""Task acceptance must measure complete source goals, identities and roles."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import task_gates as G
from tool_library.tests import test_task_success_gate as fixture_module


class TaskCompletionAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.TaskSuccessTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_unknown_subgoal_clause_cannot_hide_behind_motion_or_trusted_predicate(self):
        f = self.fixture
        self.assertTrue(f.report()['ok'])
        for goal in ('The panel is open and emits a melody.',
                     'The panel is open. The token is glowing.',
                     'The panel is open and ready for use.'):
            for declared in (False, True):
                with self.subTest(goal=goal, declared=declared):
                    sem = copy.deepcopy(f.sem)
                    sem['subgoals'][0]['goal'] = goal
                    if declared:
                        sem['subgoals'][0]['goal_predicates'] = [{
                            'class': 'joint_target', 'bodies': {'A': 'panel'},
                            'params': {'joint': 'panel_hinge', 'state': 'open'}}]
                    report = f.report(sem)
                    self.assertFalse(report['ok'], report)
                    self.assertTrue(report['task_success']['ok'])
                    self.assertTrue(any('subgoal atom' in e for e in report['subgoals'][0]['errors']))

    def test_fully_open_and_contradictory_requirements_are_not_merged_away(self):
        f = self.fixture
        sem = copy.deepcopy(f.sem)
        sem['subgoals'][0]['goal'] = 'The panel is fully open.'
        self.assertTrue(f.report(sem)['ok'])
        ep = f.episode(); ep.Q[:20] = .6
        bad = G.stage1_goal_report(ep, sem, f.man, f.recon)
        self.assertFalse(bad['ok'])
        self.assertTrue(bad['task_success']['ok'])
        sem['subgoals'][0]['goal'] = 'The panel is closed and open.'
        ep.Q[:20] = 0.
        self.assertFalse(G.stage1_goal_report(ep, sem, f.man, f.recon)['ok'])

    def test_subset_of_subgoals_is_never_full_task_acceptance(self):
        f = self.fixture
        ep = f.episode(); ep.Q[:] = 0.
        report = G.stage1_goal_report(ep, f.sem, f.man, f.recon, only=[2])
        self.assertTrue(report['subgoals'][0]['ok'])
        self.assertTrue(report['task_success']['ok'])
        self.assertFalse(report['ok'])
        self.assertTrue(any('diagnostic only' in i for i in report['issues']))

    def test_manifest_cannot_select_an_ambiguous_named_part_or_joint(self):
        for named in (False, True):
            left, right = ('cab_left_drawer', 'cab_right_drawer') if named else ('cab_drawer1', 'cab_drawer2')
            model = mujoco.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/><worldbody>
              <geom type="plane" size="1 1 .1"/>
              <body name="cab"><geom type="box" size=".1 .1 .1" pos="0 0 .1"/>
                <body name="{left}" pos="0 0 .25"><joint name="left_slide" type="slide" range="0 .2" limited="true"/>
                  <geom type="box" size=".04 .04 .02"/></body>
                <body name="{right}" pos=".3 0 .25"><joint name="right_slide" type="slide" range="0 .2" limited="true"/>
                  <geom type="box" size=".04 .04 .02"/></body>
              </body><body name="token" pos=".5 0 .02"><freejoint/><geom type="box" size=".02 .02 .02"/></body>
              </worldbody></mujoco>''')
            sem = {'objects': [{'id': 'cab', 'name': 'cabinet', 'type': 'articulated', 'parts': ['left drawer', 'right drawer']},
                               {'id': 'token', 'name': 'token'}],
                   'subgoals': [{'index': 1, 'motion': 'open_joint', 'acting_hand': 'left',
                                 'object_refs': ['cab:left drawer'], 'goal': 'The left drawer is open.'}],
                   'task_success': 'The token rests on the table.'}
            recon = {'objects': [{'id': 'cab', 'root_body': 'cab', 'bodies': ['cab', left, right]},
                                 {'id': 'token', 'root_body': 'token'}]}
            q = np.repeat(model.qpos0[None], 40, axis=0)
            q[:, model.jnt_qposadr[model.joint('left_slide' if named else 'right_slide').id]] = .2
            man = {'subgoals': [{'index': 1, 'side': 'left', 'motion': 'open_joint',
                                'object': left if named else right, 'frames': [0, 39],
                                'expect': {'joint': 'left_slide' if named else 'right_slide'}}]}
            ep = G.Episode(model, qpos=q, fps=30.)
            report = G.stage1_goal_report(ep, sem, man, recon)
            self.assertEqual(report['ok'], named, report)
            if not named:
                self.assertTrue(any('no body' in issue for issue in report['issues']), report)
            # A root naming multiple mechanisms cannot be disambiguated by a
            # generated weak expect, either.
            sem['subgoals'][0].update(object_refs=['cab'], goal='The cabinet is open.')
            self.assertFalse(G.stage1_goal_report(ep, sem, man, recon)['ok'])

    def test_generated_hand_role_cannot_override_source_hand(self):
        from tool_library.tests.test_bimanual_sync_gate import XML, build, RECON, SEM
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder / 'scene.xml').write_text(XML)
            model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
            path = build(folder, (0, 59), None, 'held', T=60)
            with np.load(path) as z:
                original = z['qpos'].copy()
            sem = copy.deepcopy(SEM)
            sem['subgoals'] = [{'index': 1, 'acting_hand': 'left', 'motion': 'hold', 'object_refs': ['pink_brick'],
                                'goal': 'The pink brick is held in the left hand.'}]
            sem['task_success'] = 'The pink brick is held.'
            for wrong in (False, True):
                q = original.copy()
                if wrong:
                    for pad in ('a', 'b'):
                        la = model.jnt_qposadr[model.body(f'robot_left_pad_{pad}').jntadr[0]]
                        ra = model.jnt_qposadr[model.body(f'robot_right_pad_{pad}').jntadr[0]]
                        q[:, ra:ra+7] = original[:, la:la+7]
                        q[:, la:la+3] = [.7, .7, .3]
                side = 'right' if wrong else 'left'
                man = {'subgoals': [{'index': 1, 'frames': [0, 59], 'motion': 'hold', 'side': side,
                                     'object': 'pink_brick_pink_brick'}]}
                report = G.stage1_goal_report(G.Episode(model, qpos=q, fps=30.), sem, man, RECON)
                self.assertTrue(report['task_success']['ok'], report)
                self.assertEqual(report['ok'], not wrong, report)

    def test_missing_final_goal_and_final_goal_undone_still_fail(self):
        f = self.fixture
        self.assertFalse(f.report(undone=True)['ok'])
        sem = copy.deepcopy(f.sem); sem.pop('task_success')
        self.assertFalse(f.report(sem)['ok'])

    def test_secure_hold_and_no_longer_supported_require_real_clearance(self):
        from tool_library.tests.test_bimanual_sync_gate import XML, build, RECON, SEM
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); (folder / 'scene.xml').write_text(XML)
            model = mujoco.MjModel.from_xml_path(str(folder / 'scene.xml'))
            path = build(folder, (5, 59), None, 'raised', T=60)
            with np.load(path) as z: original = z['qpos'].copy()
            sem = copy.deepcopy(SEM)
            sem['subgoals'] = [{'index': 1, 'acting_hand': 'left', 'motion': 'grasp_lift', 'object_refs': ['pink_brick'],
                                'goal': 'The pink brick is securely held by the left hand and no longer supported by the tabletop.'}]
            sem['task_success'] = 'The pink brick is held.'
            man = {'subgoals': [{'index': 1, 'frames': [0, 59], 'motion': 'grasp_lift', 'side': 'left',
                                 'object': 'pink_brick_pink_brick'}]}
            for supported in (False, True):
                q = original.copy()
                if supported:
                    for body in ('pink_brick_pink_brick', 'robot_left_pad_a', 'robot_left_pad_b'):
                        adr = model.jnt_qposadr[model.body(body).jntadr[0]]
                        q[:, adr+2] = .02
                report = G.stage1_goal_report(G.Episode(model, qpos=q), sem, man, RECON)
                self.assertEqual(report['ok'], not supported, report)
                if supported:
                    self.assertTrue(any(p['class'] == 'lift_clearance' and not p['ok'] for p in report['subgoals'][0]['predicates']))

    def test_opening_requires_whole_footprint_clearance_and_an_actual_hole(self):
        from tool_library.tests.test_final_goal_composition import FinalCompositionTests
        fixture = FinalCompositionTests(); fixture.setUp()
        self.addCleanup(fixture.tearDown)
        model, sem, recon, q = fixture.containment()
        sem['subgoals'] = [{'index': 1, 'motion': 'other', 'object_refs': ['vessel', 'tray'],
                            'goal': 'The vessel is above and horizontally within the open top of the tray.'}]
        sem['task_success'] = 'The tray is stationary.'
        man = {'subgoals': [{'index': 1, 'frames': [0, 39], 'motion': 'other', 'object': 'vessel'}]}
        adr = model.jnt_qposadr[model.body('vessel').jntadr[0]]
        for x, z, expected in [(0., .16, True), (.07, .16, False), (0., .10, False)]:
            q[:, adr:adr+3] = [x, 0, z]
            report = G.stage1_goal_report(G.Episode(model, qpos=q, fps=30.), sem, man, recon)
            self.assertEqual(report['ok'], expected, report)
        # Move the tray floor to its rim: a closed solid is not an opening.
        model.geom_pos[1, 2] = .12
        q[:, adr:adr+3] = [0, 0, .18]
        self.assertFalse(G.stage1_goal_report(G.Episode(model, qpos=q), sem, man, recon)['ok'])

    def test_release_and_demonstrated_orientation_remain_measured(self):
        from tool_library.tests.test_final_goal_composition import FinalCompositionTests
        fixture = FinalCompositionTests(); fixture.setUp(); self.addCleanup(fixture.tearDown)
        model, sem, recon, q = fixture.containment()
        sem['subgoals'] = [{'index': 1, 'motion': 'place', 'acting_hand': 'left', 'object_refs': ['vessel', 'tray'],
                            'goal': 'The vessel rests unsupported by either hand, fully inside the tray in the demonstrated inverted orientation.'}]
        man = {'subgoals': [{'index': 1, 'frames': [0, 39], 'motion': 'place', 'side': 'left', 'object': 'vessel'}]}
        for flipped, expected in [(False, True), (True, False)]:
            if flipped: q[5:, 10:14] = [0, 1, 0, 0]
            report = G.stage1_goal_report(G.Episode(model, qpos=q), sem, man, recon)
            self.assertEqual(report['ok'], expected, report)

    def test_completed_rotation_needs_local_motion_and_verified_settled_endpoint(self):
        from tool_library.tests.test_final_goal_composition import FinalCompositionTests
        fixture = FinalCompositionTests(); fixture.setUp(); self.addCleanup(fixture.tearDown)
        model, sem, recon, q, built = fixture.box()
        sem['subgoals'] = [{'index': 1, 'motion': 'close_joint', 'acting_hand': 'left', 'object_refs': ['parcel:lid'],
                            'goal': 'The lid has completed its rotation and lies flat over the base with the interior no longer visible.'}]
        man = {'subgoals': [{'index': 1, 'frames': [10, 39], 'motion': 'close_joint', 'side': 'left', 'object': built['lid_body']}]}
        adr = model.jnt_qposadr[model.joint(built['lid_joint']).id]
        for case, expected in [('moving', True), ('static', False), ('earlier_only', False), ('open', False), ('unsettled', False)]:
            states = q.copy()
            if case == 'moving': states[:21, adr] = .6; states[21:26, adr] = np.linspace(.6, 0., 5)
            if case == 'earlier_only': states[:5, adr] = .6
            if case == 'open': states[:, adr] = .6
            if case == 'unsettled': states[:-1, adr] = .6
            report = G.stage1_goal_report(G.Episode(model, qpos=states), sem, man, recon)
            self.assertEqual(report['ok'], expected, (case, report))
        sem['subgoals'][0]['goal'] = 'The lid has completed its rotation.'
        self.assertFalse(G.stage1_goal_report(G.Episode(model, qpos=q), sem, man, recon)['ok'])


if __name__ == '__main__':
    unittest.main()
