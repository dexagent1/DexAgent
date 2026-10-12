"""The no-API model transport must use the real tools and finish boundaries."""
import json
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.config import load_config
from agent.external import ExternalLLM, atomic_json, validate_response
from agent.llm import MockLLM
from agent.runner import StageRunner


def test_transport():
    with tempfile.TemporaryDirectory() as tmp:
        llm = ExternalLLM(tmp, timeout_s=5)
        result, errors = [], []
        def complete():
            try:
                result.append(llm.complete("unchanged system", [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
                                           [{"name": "finish"}]))
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=complete)
        thread.start()
        import time
        deadline = time.monotonic() + 4
        status = {}
        while time.monotonic() < deadline:
            status = json.loads((Path(tmp) / "status.json").read_text())
            if status.get("state") == "waiting":
                break
            time.sleep(0.01)
        assert status.get("state") == "waiting"
        request = json.loads(Path(status["request"]).read_text())
        assert request["system"] == "unchanged system"
        blocks = [{"type": "tool_use", "id": "finish_1", "name": "finish", "input": {"summary": "done"}}]
        atomic_json(Path(tmp) / "responses" / (request["id"] + ".json"),
                    {"id": request["id"], "session": request["session"], "content": blocks})
        thread.join(timeout=5)
        assert not errors and result == [blocks]
        llm.close({"finished": True})
        assert json.loads((Path(tmp) / "status.json").read_text())["state"] == "finished"


def test_finish_is_terminal():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(ROOT / "dexagent.yaml")
        turns = [[{"type": "tool_use", "id": "a", "name": "finish", "input": {"summary": "done"}},
                  {"type": "tool_use", "id": "b", "name": "write_text", "input": {"path": "bad.txt", "text": "modified after pass"}}]]
        out = StageRunner(cfg, MockLLM(turns), Path(tmp), "test", [lambda ws: {"ok": True}], max_rounds=1).run("task")
        assert out["finished"] and not (Path(tmp) / "bad.txt").exists()


def test_unverified_or_failed_delivery_never_passes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(ROOT / "dexagent.yaml")
        stage = StageRunner(cfg, MockLLM([]), Path(tmp), "test", [], max_rounds=1)
        assert stage._verify()["ok"] is False
        stage.verifiers = [lambda ws: {"ok": True}]
        stage.finalize = lambda ws, verdict: {"ok": False, "issues": ["render failed"]}
        assert stage._verify()["ok"] is False
        stage.verifiers = [lambda ws: {"ok": "false"}]
        assert stage._verify()["ok"] is False


def test_incremental_view_is_lossless_and_resets_on_history_change():
    from agent.external import inspect_request
    import copy
    first = dict(schema='dexagent.external.v1', session='a', id='a-1', turn=1,
                 system='rules', messages=[{'role': 'user', 'content': 'input'}],
                 tools=[{'name': 'one', 'input_schema': {}}, {'name': 'two', 'input_schema': {}}])
    current = copy.deepcopy(first)
    current.update(id='a-2', turn=2)
    current['messages'].append({'role': 'user', 'content': [{'type': 'image', 'path': '/tmp/image.png'}]})
    current['tools'] = [{'name': 'one', 'input_schema': {'x': 'str'}}, {'name': 'three', 'input_schema': {}}]
    delta = inspect_request(current, first)
    assert delta['context_mode'] == 'incremental'
    assert first['messages'] + delta['messages'] == current['messages']
    schemas = {t['name']: t for t in first['tools']}
    schemas.update({t['name']: t for t in delta['tools']})
    assert [schemas[n] for n in delta['allowed_tool_names']] == current['tools']
    assert delta['removed_tool_names'] == ['two']
    assert delta['tools_mode'] == 'complete_current_schemas'
    assert delta['tools'] == current['tools']
    same_tools = copy.deepcopy(current)
    same_tools.update(id='a-3', turn=3)
    same_tools['messages'].append({'role': 'user', 'content': 'a verifier failed; repair'})
    unchanged = inspect_request(same_tools, current)
    assert unchanged['changed_tool_names'] == []
    assert unchanged['tools'] == current['tools']  # never misleading tools=[]
    assert unchanged['allowed_tool_names'] == ['one', 'three']
    current['messages'][0]['content'] = 'revised'
    assert inspect_request(current, first)['context_mode'] == 'full'
    current['session'] = 'b'
    assert inspect_request(current, first)['context_mode'] == 'full'


def test_wait_for_next_stage_and_administrative_checkpoint():
    import contextlib
    import io
    import time
    from agent.external import main
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        atomic_json(p / 'status.json', {'state': 'finished', 'session': 'old'})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(['wait', tmp, '--after-session', 'old', '--timeout', '0.05'])
        assert json.loads(output.getvalue()) == {'state': 'pending'}
        def next_stage():
            time.sleep(.05)
            atomic_json(p / 'status.json', {'state': 'waiting', 'session': 'new', 'id': 'new-0001'})
        atomic_json(p / 'operator_checkpoint.json', {'kind': 'administrative_checkpoint'})
        thread = threading.Thread(target=next_stage)
        thread.start()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(['wait', tmp, '--after-session', 'old', '--timeout', '2'])
        thread.join()
        result = json.loads(output.getvalue())
        assert result['session'] == 'new' and result['state'] == 'waiting'
        assert result['operator_checkpoint']['kind'] == 'administrative_checkpoint'


if __name__ == "__main__":
    for test in (test_transport, test_finish_is_terminal, test_unverified_or_failed_delivery_never_passes,
                 test_incremental_view_is_lossless_and_resets_on_history_change,
                 test_wait_for_next_stage_and_administrative_checkpoint):
        test()
        print(f"[PASS] {test.__name__}")
