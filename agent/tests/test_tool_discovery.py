"""The entire library stays callable without sending hundreds of schemas per turn."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agent.config import load_config
from agent.llm import MockLLM
from agent.runner import StageRunner


def test_discovery_and_refresh():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(ROOT / 'dexagent.yaml')
        runner = StageRunner(cfg, MockLLM([]), Path(tmp), 'test', [lambda ws: {'ok': True}])
        initial = runner.tools()
        assert len(initial) < 20
        assert 'execute_subgoal_plan' not in {t['name'] for t in initial}
        _, result = runner.run_tool('library_search', {'query': 'execute_subgoal_plan', 'type': 'skill'})
        assert result['tools'] and 'execute_subgoal_plan' in {t['name'] for t in runner.tools()}
        for t in runner.index['tools']:
            runner._activate_tools([t['name']])
        assert len(runner.active_tools) == 64
        _, result = runner.run_tool('library_read', {'name': 'execute_subgoal_plan'})
        assert not result.get('error') and 'execute_subgoal_plan' in {t['name'] for t in runner.tools()}
        runner.tool_discovery = 'all'
        assert len(json.dumps(initial)) < len(json.dumps(runner.tools())) / 10


if __name__ == '__main__':
    test_discovery_and_refresh()
    print('[PASS] full library discoverable through bounded callable schema window')
