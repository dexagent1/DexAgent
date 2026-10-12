"""Only a completed harness run may expose a delivery, even after tool writes."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.artifacts import accepted_delivery, verify_delivery
from agent.external import atomic_json
from agent.llm import MockLLM
from tool_library.common.evidence import file_digest
from stage3_trajectory_generation import run_stage3 as S3

os.environ.setdefault('DEXAGENT_QUIET', '1')


class DeliveryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.public = self.root / 'runs/trial/stage3'
        self.bundle = self.public / 'verified/previous'
        self.bundle.mkdir(parents=True)
        (self.bundle / 'trajectory.npz').write_bytes(b'integrity-only fixture')
        atomic_json(self.bundle / 'delivery.json', {
            'schema': 'dexagent.delivery.v1', 'passed': True,
            'sha256': {'trajectory.npz': file_digest(self.bundle / 'trajectory.npz')}})
        self.pointer = {'directory': str(self.bundle),
                        'certificate_sha256': file_digest(self.bundle / 'delivery.json')}
        atomic_json(self.public / 'accepted.json', self.pointer)
        self.success = {'finished': True, 'verdict': {'ok': True, 'reports': {
            'verified_delivery': {'ok': True, **self.pointer}}}}
        atomic_json(self.public / 'run.json', self.success)

    def tearDown(self):
        self.temp.cleanup()

    def test_integrity_does_not_override_incomplete_failed_or_unbound_run(self):
        self.assertEqual(accepted_delivery(self.public), self.bundle)
        for result in ({'finished': False}, {'finished': False, 'status': 'running'},
                       {'finished': True, 'verdict': {'ok': False}},
                       {'finished': True, 'verdict': {'ok': True, 'reports': {}}}):
            with self.subTest(result=result):
                atomic_json(self.public / 'run.json', result)
                self.assertTrue(verify_delivery(self.bundle)['ok'])
                with self.assertRaises(ValueError):
                    accepted_delivery(self.public)

    def test_other_run_or_certificate_cannot_supply_acceptance(self):
        for key, value in (('directory', str(self.public / 'verified/other')),
                           ('certificate_sha256', 'different')):
            with self.subTest(key=key):
                run = json.loads(json.dumps(self.success))
                run['verdict']['reports']['verified_delivery'][key] = value
                atomic_json(self.public / 'run.json', run)
                with self.assertRaises(ValueError):
                    accepted_delivery(self.public)

    def test_failed_real_runner_quarantines_tool_written_success_marker(self):
        lib = self.root / 'framework/tool_library'; lib.mkdir(parents=True)
        (lib / 'index.json').write_text('{"tools":[]}')
        semantic = self.root / 'semantic.json'; semantic.write_text('{"subgoals":[]}')
        recon = self.root / 'stage2/scene_recon.json'; recon.parent.mkdir()
        recon.write_text('{"objects":[],"scene_xml":"scene.xml"}')
        (recon.parent / 'scene.xml').write_text('<mujoco><worldbody/></mujoco>')
        video = self.root / 'video.mp4'; video.write_bytes(b'identity checked separately')
        cfg = {'inputs': {'library': str(lib), 'video': str(video)},
               'run': {'name': 'trial', 'output_dir': str(self.root / 'runs')},
               'examples': {'reuse': False}, 'stage3': {'max_rounds': 2}}
        code = ('import json\nfrom pathlib import Path\n'
                f'p=Path({str(self.public)!r})\n'
                f'(p/"accepted.json").write_text({json.dumps(self.pointer)!r})\n'
                f'(p/"run.json").write_text({json.dumps(self.success)!r})\n')
        llm = MockLLM([
            [{'type': 'tool_use', 'name': 'run_python', 'input': {'code': code}}],
            [{'type': 'tool_use', 'name': 'finish', 'input': {'summary': 'saved PASS claim'}}]])
        original_complete = llm.complete
        observed = []
        def complete(*args, **kwargs):
            if not observed:
                current = json.loads((self.public / 'run.json').read_text())
                observed.append(current)
                self.assertFalse(current['finished'])
                self.assertFalse((self.public / 'accepted.json').exists())
            return original_complete(*args, **kwargs)
        llm.complete = complete
        with patch.object(S3, 'load_config', return_value=cfg), \
             patch('agent.input_identity.video_verifier', return_value=lambda _: {'ok': True}), \
             patch('agent.reference_evidence.revalidate_source_recipes', return_value={'ok': True}), \
             patch('agent.llm.make_llm', return_value=llm), \
             patch.object(S3, 'make_verifiers', return_value=[lambda _: {'ok': False, 'issues': ['physical rejection']}]), \
             patch.object(S3, 'make_diagnosis', return_value=None), \
             patch('agent.reuse.try_reuse', return_value=(None, None)):
            rc = S3.main(['unused', '--semantic', str(semantic), '--recon', str(recon)])
        self.assertEqual(rc, 1)
        self.assertFalse((self.public / 'accepted.json').exists())
        run = json.loads((self.public / 'run.json').read_text())
        self.assertFalse(run['finished'])
        rejected = Path(run['rejected_acceptance_marker'])
        self.assertEqual(json.loads(rejected.read_text()), self.pointer)
        self.assertTrue(verify_delivery(self.bundle)['ok'])  # historical data retained
        with self.assertRaises((ValueError, OSError)):
            accepted_delivery(self.public)

        # An administrative checkpoint must not launch the method modules.
        def stopped(runner, _):
            return {'finished': False, 'stopped': True,
                    'stop_reason': 'cooperative_search_checkpoint',
                    'workspace': str(runner.ws)}
        with patch.object(S3, 'load_config', return_value=cfg), \
             patch('agent.input_identity.video_verifier', return_value=lambda _: {'ok': True}), \
             patch('agent.reference_evidence.revalidate_source_recipes', return_value={'ok': True}), \
             patch('agent.llm.make_llm', return_value=MockLLM([])), \
             patch.object(S3, 'make_verifiers', return_value=[]), \
             patch.object(S3, 'make_diagnosis', return_value=None), \
             patch('agent.reuse.try_reuse', return_value=(None, None)), \
             patch('agent.runner.StageRunner.run', stopped), \
             patch.object(S3, 'run_method_modules') as methods:
            rc = S3.main(['unused', '--semantic', str(semantic), '--recon', str(recon)])
        self.assertEqual(rc, 1)
        methods.assert_not_called()
        self.assertTrue(json.loads((self.public / 'run.json').read_text())['stopped'])


if __name__ == '__main__':
    unittest.main()
