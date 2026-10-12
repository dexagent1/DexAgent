"""A workspace-relative executor report must reach automatic failure diagnosis."""
import json
import tempfile
import unittest
from pathlib import Path

from agent.config import load_config
from agent.library import library_root
from stage3_trajectory_generation.run_stage3 import make_diagnosis
from tool_library.skills.diagnose_stage3_failure import diagnose_stage3_failure


class ExecutorDiagnosisPathsTests(unittest.TestCase):
    def test_inline_mechanism_plan_retains_failed_step_and_sweep(self):
        plan = {"subgoals": [
            dict(index=7, step=1, motion='hold', side='left', pose='old_pose.json'),
            dict(index=7, step=2, motion='close_joint', side='left', object='panel',
                 pose='current_pose.json', hand_scene='current_hand_scene.json',
                 joint='panel_hinge', target=.4, hand_follows='level')]}
        failed = [dict(index=7, step=2, motion='close_joint', side='left', object='panel',
                       ok=False, failed=['arm_unreachable'])]
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'plan_stale.json').write_text(json.dumps({'subgoals': [
                dict(index=7, step=2, joint='wrong_hinge', target=2.)]}))
            result = diagnose_stage3_failure(workspace=ws, outcomes=failed, plan=plan,
                                             check_hand_scenes=False, include_workspace_history=False)
        repairs = {r['tool']: r for r in result['repairs']}
        self.assertNotIn('optimize_grasp_pose', repairs)
        for name in ('optimize_pose_for_reach', 'reachable_contact_search'):
            self.assertEqual(repairs[name]['args']['sweep'],
                             {'joint': 'panel_hinge', 'to': .4, 'hand_follows': 'level'})
        self.assertEqual(repairs['optimize_pose_for_reach']['args']['pose'], 'current_pose.json')
        self.assertEqual(repairs['optimize_pose_for_reach']['args']['hand_scene'], 'current_hand_scene.json')

    def test_real_hook_forwards_inline_plan(self):
        root = Path(__file__).resolve().parents[2]
        cfg = load_config(root / 'dexagent.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            hook = make_diagnosis(cfg, ws / 'absent_sem.json', ws / 'absent_recon.json', library_root(cfg))
            failed = dict(index=4, motion='close_joint', side='right', object='panel',
                          ok=False, failed=['arm_unreachable'])
            verdict = dict(ok=False, issues=[], reports={}, executor_result=dict(ok=False, subgoals=[failed]),
                           executor_args=dict(scene='scene.xml', plan={'subgoals': [
                               dict(index=4, side='right', motion='close_joint', joint='actual_hinge', target=.123,
                                    pose='actual_pose.json', hand_scene='actual_hand.json')]}))
            summary = hook(ws, verdict)
            self.assertIn('actual_hinge', summary)
            self.assertIn('actual_pose.json', summary)
            self.assertNotIn('could not run', summary)

    def test_relative_outcomes_are_resolved_in_candidate_workspace(self):
        root = Path(__file__).resolve().parents[2]
        cfg = load_config(root / 'dexagent.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            (ws / 'execution').mkdir()
            (ws / 'execution/outcomes.json').write_text(json.dumps([
                dict(index=7, motion='carry', side='right', object='part', ok=False,
                     failed=['arm_unreachable'], measured={})]))
            hook = make_diagnosis(cfg, ws / 'missing_sem.json', ws / 'missing_recon.json', library_root(cfg))
            verdict = {'ok': False, 'issues': [], 'reports': {}, 'executor_result': {
                'ok': False, 'files': {'outcomes': 'execution/outcomes.json'}, 'subgoals': []}}
            summary = hook(ws, verdict)
            self.assertIn('unreachable', summary)
            self.assertIn('subgoals [7]', summary)
            self.assertIn('Next steps:', summary)


if __name__ == '__main__':
    unittest.main()
