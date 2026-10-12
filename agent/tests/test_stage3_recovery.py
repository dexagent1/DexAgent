"""Recovery preserves search work, never old acceptance or executor authority."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('DEXAGENT_QUIET', '1')

from agent.llm import MockLLM
from agent.runner import StageRunner
from tool_library.common.evidence import file_digest
from stage3_trajectory_generation import run_stage3 as S3


def turn(name, **kwargs):
    return [{'type': 'tool_use', 'name': name, 'input': kwargs}]


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.public = self.root / 'runs' / 'trial' / 'stage3'
        self.ws = self.public / '.candidates' / 'interrupted'
        self.ws.mkdir(parents=True)
        self.lib = self.root / 'framework' / 'tool_library'
        (self.lib / 'verifiers').mkdir(parents=True)
        (self.lib / 'index.json').write_text('{"tools": []}')
        self.code = self.lib / 'verifiers' / 'current.py'
        self.code.write_text('FRAMEWORK = "current"\n')
        self.cfg = {'inputs': {'library': str(self.lib)}, 'agent': {'max_rounds': 2},
                    'run': {'name': 'trial', 'output_dir': str(self.root / 'runs')}}
        self.old = {
            'trace.jsonl': '{"event":"tool","name":"execute_subgoal_plan","result":{"ok":true},"old_private_summary":"OLD_CONTEXT_SENTINEL"}\n',
            'input_contract.json': '{"sha256":{"old_framework.py":"old-digest"}}',
            'run.json': '{"finished":false,"verdict":{"ok":true},"failure":"model backend failed"}',
            'failure_report.json': '{"reason":"model backend failed"}',
            'search_progress.json': '{"authority":"diagnostic_only","best_proposal":"prior_prefix"}',
            'backend_checkpoint.json': '{"reason":"backend_no_tool_progress"}',
        }
        for name, contents in self.old.items():
            (self.ws / name).write_text(contents)
        (self.ws / 'plan.json').write_text('{"model_generated_plan":true}')
        (self.ws / 'execution_report.json').write_text('{"ok":true,"old_report":true}')

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        ws, recovery = S3.prepare_attempt(self.public, self.ws)
        self.cfg['_verification_inputs'] = [recovery['archive']]
        return ws, recovery

    def test_preserves_metadata_bytes_and_exact_search_paths(self):
        plan = (self.ws / 'plan.json').read_bytes()
        ws, recovery = self.prepare()
        self.assertEqual(ws, self.ws)
        archive = Path(recovery['archive'])
        for name, content in self.old.items():
            self.assertEqual((archive / name).read_text(), content)
            self.assertEqual(recovery['previous_files'][name]['sha256'], file_digest(archive / name))
            self.assertFalse((ws / name).exists())
        self.assertEqual((ws / 'plan.json').read_bytes(), plan)
        self.assertTrue((ws / 'execution_report.json').exists())
        self.assertFalse(recovery['history_restored'])
        self.assertFalse(recovery['executor_receipts_restored'])
        # A second interruption archives the previous recovery record as well.
        (ws / 'trace.jsonl').write_text('second interruption\n')
        _, again = S3.prepare_attempt(self.public, ws)
        self.assertNotEqual(again['archive'], recovery['archive'])
        self.assertEqual(json.loads((Path(again['archive']) / 'recovery.json').read_text())['recovery_id'], recovery['recovery_id'])
        self.assertEqual((archive / 'trace.jsonl').read_text(), self.old['trace.jsonl'])

    def test_rejects_public_accepted_outside_nested_missing_and_symlink_paths(self):
        verified = self.public / 'verified' / 'accepted'; verified.mkdir(parents=True)
        nested = self.ws / 'nested'; nested.mkdir()
        outside = self.root / 'outside'; outside.mkdir()
        aliases = [self.public / '.candidates' / 'alias', self.root / 'alias']
        for alias in aliases: alias.symlink_to(self.ws, target_is_directory=True)
        wrong = [self.public, verified, self.public / '.candidates', outside, nested,
                 self.ws / 'plan.json', self.public / '.candidates' / 'missing', *aliases]
        for path in wrong:
            with self.subTest(path=path), self.assertRaises(ValueError):
                S3.prepare_attempt(self.public, path)
        self.assertEqual((self.ws / 'trace.jsonl').read_text(), self.old['trace.jsonl'])
        self.assertFalse((self.public / '.recoveries').exists())

    def test_rejects_finished_candidate_or_delivery_marker(self):
        for where in (self.ws, self.public):
            path = where / 'run.json'
            previous = path.read_bytes() if path.exists() else None
            path.write_text(json.dumps({'finished': True, 'workspace': str(self.ws)}))
            with self.assertRaisesRegex(ValueError, 'finished'):
                S3.prepare_attempt(self.public, self.ws)
            if previous is None: path.unlink()
            else: path.write_bytes(previous)
        (self.ws / 'delivery.json').write_text('{"passed":true}')
        with self.assertRaisesRegex(ValueError, 'accepted/delivery'):
            S3.prepare_attempt(self.public, self.ws)

    def test_metadata_symlink_cannot_overwrite_external_input(self):
        target = self.root / 'protected.json'; target.write_text('unchanged')
        (self.ws / 'input_contract.json').unlink()
        (self.ws / 'input_contract.json').symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'regular file'):
            S3.prepare_attempt(self.public, self.ws)
        self.assertEqual(target.read_text(), 'unchanged')

    def test_new_contract_and_empty_receipts_do_not_restore_old_success(self):
        ws, recovery = self.prepare()
        self.cfg['_executor_receipts'] = {'forged-old-receipt': str(ws / 'execution_report.json')}
        gate = Mock(return_value={'ok': True}); gate.__name__ = 'current_gate'
        finalizer = Mock(return_value={'ok': True})
        runner = StageRunner(self.cfg, MockLLM([]), ws, 'test', [gate], stage='stage3', recovery=recovery, finalize=finalizer)
        self.assertEqual(self.cfg['_executor_receipts'], {})
        contract = json.loads((ws / 'input_contract.json').read_text())
        self.assertEqual(contract['sha256'][str(self.code)], file_digest(self.code))
        self.assertNotIn('old_framework.py', contract['sha256'])
        self.assertEqual(runner.recovery['current_input_contract_sha256'], file_digest(ws / 'input_contract.json'))
        self.assertFalse(runner._verify()['ok'])
        gate.assert_not_called(); finalizer.assert_not_called()
        # Receipt-shaped disk data still has no in-memory authority.
        (ws / '_executor_receipts.json').write_text(json.dumps({'fake': 'execution_report.json'}))
        self.assertFalse(runner._verify()['ok'])
        self.assertEqual(self.cfg['_executor_receipts'], {})
        # Archived history is frozen as evidence, not imported as current code.
        (Path(recovery['archive']) / 'trace.jsonl').write_text('tampered history')
        self.assertFalse(runner.contract.check()['ok'])

    def test_only_fresh_executor_call_permits_current_gates_and_all_must_pass(self):
        ws, recovery = self.prepare()
        first = Mock(return_value={'ok': True}); first.__name__ = 'first_gate'
        second = Mock(return_value={'ok': False, 'issues': ['new check fails']}); second.__name__ = 'second_gate'
        finalizer = Mock(return_value={'ok': True})
        runner = StageRunner(self.cfg, MockLLM([]), ws, 'test', [first, second], stage='stage3', recovery=recovery, finalize=finalizer)
        response = {'ok': True, 'result': {'ok': True, 'trajectory_sha256': '0' * 64,
                                         'files': {'execution_report': str(ws / 'execution_report.json')}}}
        with patch('agent.runner.call_tool', return_value=response) as dispatch:
            runner.run_tool('unrelated_skill', {})
            self.assertEqual(self.cfg['_executor_receipts'], {})
            runner.run_tool('execute_subgoal_plan', {'dry_run': True})
            self.assertEqual(self.cfg['_executor_receipts'], {})
            for invalid in ({'ok': False}, {'dry_run': True}, {'trajectory_sha256': None}):
                dispatch.return_value = {'ok': True, 'result': {**response['result'], **invalid}}
                runner.run_tool('execute_subgoal_plan', {})
                self.assertEqual(self.cfg['_executor_receipts'], {})
            dispatch.return_value = response
            runner.run_tool('execute_subgoal_plan', {'resume': True})
            self.assertTrue(dispatch.call_args.args[3]['resume'])
        self.assertEqual(list(self.cfg['_executor_receipts']), [file_digest(ws / 'execution_report.json')])
        self.assertFalse(runner._verify()['ok'])
        self.assertEqual(first.call_count, 1); self.assertEqual(second.call_count, 1)
        finalizer.assert_not_called()
        second.return_value = {'ok': True}
        self.assertTrue(runner._verify()['ok'])
        self.assertEqual(first.call_count, 2); self.assertEqual(second.call_count, 2)
        finalizer.assert_called_once()
        # Another fresh runner cannot inherit this live run's authority either.
        next_runner = StageRunner(self.cfg, MockLLM([]), ws, 'test', [first], stage='stage3', recovery=recovery)
        self.assertEqual(self.cfg['_executor_receipts'], {})
        self.assertFalse(next_runner._verify()['ok'])

    def test_cli_resume_starts_fresh_and_skips_reuse_and_old_acceptance(self):
        semantic = self.root / 'semantic.json'; semantic.write_text('{"subgoals":[]}')
        recon = self.root / 'reconstruction' / 'scene_recon.json'; recon.parent.mkdir()
        recon.write_text('{"scene_xml":"scene.xml","objects":[]}')
        (recon.parent / 'scene.xml').write_text('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
        video = self.root / 'source.mp4'; video.write_bytes(b'identity stub fixture')
        self.cfg['inputs']['video'] = str(video)
        llm = MockLLM([turn('finish', summary='old files say passed')])
        initial_context = []
        complete = llm.complete
        def recording_complete(system, messages, tools):
            initial_context.append(json.loads(json.dumps(messages)))
            return complete(system, messages, tools)
        llm.complete = recording_complete
        def identity(_): return {'ok': True}
        with patch.object(S3, 'load_config', return_value=self.cfg), \
             patch('agent.input_identity.video_verifier', return_value=identity), \
             patch('agent.reference_evidence.revalidate_source_recipes', return_value={'ok': True}), \
             patch('agent.llm.make_llm', return_value=llm), \
             patch.object(S3, 'make_verifiers', return_value=[identity]), \
             patch.object(S3, 'make_diagnosis', return_value=None), \
             patch('agent.reuse.try_reuse') as reuse, \
             patch.object(S3, 'run_method_modules', return_value=None) as methods, \
             patch('agent.artifacts.publish_stage3') as publish:
            rc = S3.main(['ignored.yaml', '--semantic', str(semantic), '--recon', str(recon),
                          '--resume-attempt', str(self.ws), '--max-rounds', '1'])
        self.assertEqual(rc, 1)
        reuse.assert_not_called(); publish.assert_not_called()
        # the recovered session's search ended unfinished: the method modules are offered it like any other
        methods.assert_called_once()
        self.assertEqual(len(initial_context[0]), 1)
        self.assertIn(S3.recovery_message(), llm.seen[0][0]['content'][0]['text'])
        self.assertNotIn('OLD_CONTEXT_SENTINEL', json.dumps(llm.seen[0]))
        result = json.loads((self.public / 'run.json').read_text())
        self.assertFalse(result['finished'])
        self.assertEqual(result['workspace'], str(self.ws))
        self.assertFalse(result['recovery']['executor_receipts_restored'])
        self.assertFalse((self.public / 'accepted.json').exists())
        self.assertEqual(json.loads((self.public / 'candidate.json').read_text())['status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
