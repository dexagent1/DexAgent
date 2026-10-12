"""A backend can recover argument mistakes without guessing a new API."""
import tempfile
import unittest
from pathlib import Path

from agent.config import load_config
from agent.llm import MockLLM
from agent.runner import StageRunner


class ToolErrorFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parents[2]
        self.runner = StageRunner(load_config(root / 'dexagent.yaml'), MockLLM([]),
                                  Path(self.tmp.name), 'test', [lambda ws: {'ok': True}])

    def tearDown(self):
        self.tmp.cleanup()

    def test_bad_builtin_arguments_include_current_schema(self):
        _, result = self.runner.run_tool('list_dir', {'path': '.', 'unknown_argument': True})
        self.assertFalse(result['ok'])
        self.assertIn('path', result['input_schema']['properties'])
        self.assertNotIn('unknown_argument', result['input_schema']['properties'])

    def test_bad_library_arguments_include_current_schema(self):
        _, result = self.runner.run_tool('write_stage4_config', {'unknown_argument': True})
        self.assertFalse(result['ok'])
        self.assertIn('scene_recon_json', result['input_schema']['properties'])
        self.assertIn('episode_dir', result['input_schema']['properties'])

    def test_python_exit_status_is_not_reported_as_success(self):
        _, good = self.runner.run_tool('run_python', {'code': 'print("ok")'})
        _, bad = self.runner.run_tool('run_python', {'code': 'raise SystemExit(2)'})
        self.assertTrue(good['ok'])
        self.assertFalse(bad['ok'])
        self.assertEqual(bad['returncode'], 2)


if __name__ == '__main__':
    unittest.main()
