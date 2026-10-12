"""File transport for an external model using the real StageRunner (no API key).

Only the model response is supplied externally. Tool dispatch, feedback, budgets,
verification, and delivery remain in the ordinary harness. Requests contain the
same system, messages, images and tool schemas as the API clients.
"""
from __future__ import annotations

import base64
import copy
import json
import time
import uuid
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=1, allow_nan=False))
    temp.replace(path)


def validate_response(value, tool_names):
    if not isinstance(value, list) or not value:
        raise ValueError("response must be a nonempty list of content blocks")
    ids = set()
    for block in value:
        if not isinstance(block, dict):
            raise ValueError("invalid content block")
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            continue
        if block.get("type") != "tool_use" or block.get("name") not in tool_names:
            raise ValueError("response contains a tool absent from this request")
        if not isinstance(block.get("input"), dict):
            raise ValueError("tool input must be a JSON object")
        ident = block.get("id")
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("tool calls must have distinct nonempty ids")
        ids.add(ident)
    return value


def inspect_request(request, previous=None):
    """Lossless incremental view for a backend that retains its prior context.

    Full requests remain the audit record. A changed session/history forces a
    complete context refresh; otherwise only new messages are repeated. Tool
    schemas always list every currently callable tool. An empty schema delta
    must not look like removal of all tools to an external model.
    """
    old_messages = (previous or {}).get('messages', [])
    incremental = (previous is not None and previous.get('session') == request['session']
                   and request['messages'][:len(old_messages)] == old_messages)
    if not incremental:
        return dict(request, context_mode='full', tools_mode='complete_current_schemas')
    old_tools = {t['name']: t for t in previous['tools']}
    names = [t['name'] for t in request['tools']]
    result = {k: request[k] for k in ('schema', 'session', 'id', 'turn')}
    result.update(context_mode='incremental', previous_id=previous['id'],
                  messages=request['messages'][len(old_messages):],
                  tools=request['tools'], tools_mode='complete_current_schemas',
                  changed_tool_names=[t['name'] for t in request['tools'] if old_tools.get(t['name']) != t],
                  allowed_tool_names=names,
                  removed_tool_names=[n for n in old_tools if n not in names])
    if request['system'] != previous.get('system'):
        result['system'] = request['system']
    return result


class ExternalLLM:
    def __init__(self, directory, timeout_s=3600):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = uuid.uuid4().hex
        self.turn = 0
        self.timeout_s = float(timeout_s)
        atomic_json(self.directory / "status.json", {"state": "starting", "session": self.session})

    def _images(self, obj):
        """Expose identical image bytes as files instead of huge base64 requests."""
        if isinstance(obj, list):
            return [self._images(v) for v in obj]
        if not isinstance(obj, dict):
            return obj
        if obj.get("type") == "image" and "data" in obj:
            from tool_library.common.evidence import value_digest
            digest = value_digest(obj["data"])
            suffix = ".png" if obj.get("mime") == "image/png" else ".jpg"
            path = self.directory / "images" / (digest + suffix)
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                path.write_bytes(base64.b64decode(obj["data"], validate=True))
            return {"type": "image", "mime": obj.get("mime"), "path": str(path)}
        return {k: self._images(v) for k, v in obj.items()}

    def complete(self, system, messages, tools):
        self.turn += 1
        request_id = f"{self.session}-{self.turn:04d}"
        request = {"schema": "dexagent.external.v1", "session": self.session,
                   "id": request_id, "turn": self.turn, "system": system,
                   "messages": self._images(copy.deepcopy(messages)), "tools": tools}
        path = self.directory / "requests" / (request_id + ".json")
        atomic_json(path, request)
        atomic_json(self.directory / "status.json", {"state": "waiting", "session": self.session,
                    "turn": self.turn, "id": request_id, "request": str(path)})
        response = self.directory / "responses" / (request_id + ".json")
        deadline = time.monotonic() + self.timeout_s
        while not response.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"external model did not answer request {request_id}")
            time.sleep(0.1)
        answer = json.loads(response.read_text())
        if answer.get("id") != request_id or answer.get("session") != self.session:
            raise ValueError("external response belongs to another request/session")
        content = validate_response(answer.get("content"), {t["name"] for t in tools})
        atomic_json(self.directory / "status.json", {"state": "executing", "session": self.session,
                    "turn": self.turn, "id": request_id})
        return content

    def close(self, result):
        atomic_json(self.directory / "status.json", {"state": "finished", "session": self.session,
                    "turn": self.turn, "result": result})


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    wait = sub.add_parser("wait")
    wait.add_argument("directory")
    wait.add_argument("--after", default="")
    wait.add_argument("--after-session", default="", help="wait for a different session after a stage closes")
    wait.add_argument("--timeout", type=float, default=50)
    respond = sub.add_parser("respond")
    respond.add_argument("request")
    respond.add_argument("response", help="JSON file containing model content blocks")
    inspect = sub.add_parser("inspect", help="Read a full or lossless incremental request view")
    inspect.add_argument("request")
    inspect.add_argument("--previous", help="Previous request already read in this model context")
    a = ap.parse_args(argv)
    if a.command == "wait":
        deadline = time.monotonic() + min(a.timeout, 60)
        p = Path(a.directory) / "status.json"
        while time.monotonic() < deadline:
            status = json.loads(p.read_text()) if p.exists() else {}
            ready = status.get("state") == "finished" or (status.get("state") == "waiting" and status.get("id") != a.after)
            if ready and (not a.after_session or status.get("session") != a.after_session):
                checkpoint = p.parent / "operator_checkpoint.json"
                if checkpoint.is_file():
                    status["operator_checkpoint"] = json.loads(checkpoint.read_text())
                print(json.dumps(status))
                return 0
            time.sleep(0.2)
        print(json.dumps({"state": "pending"}))
        return 0
    request_path = Path(a.request).resolve()
    request = json.loads(request_path.read_text())
    if a.command == 'inspect':
        previous = json.loads(Path(a.previous).read_text()) if a.previous else None
        print(json.dumps(inspect_request(request, previous), indent=1))
        return 0
    content = validate_response(json.loads(Path(a.response).read_text()), {t["name"] for t in request["tools"]})
    destination = request_path.parent.parent / "responses" / (request["id"] + ".json")
    if destination.exists():
        raise FileExistsError("a response was already submitted for this request")
    atomic_json(destination, {"session": request["session"], "id": request["id"], "content": content})
    print(json.dumps({"submitted": request["id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
