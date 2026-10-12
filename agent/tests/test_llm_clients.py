"""The model clients end to end without an API key: a local stub server speaks the Anthropic Messages API and the
OpenAI Chat Completions API, and a real Stage 1 run drives it through the official SDKs. The stub checks every request
the harness sends (system prompt, tool schemas, tool results with text and images) and scripts the model's turns:
probe the video, make a contact sheet, look at it, call finish.

    python agent/tests/test_llm_clients.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
VIDEO = REPO / "stage1_semantic_understanding" / "examples" / "giftbox_packing" / "video.mp4"

TURNS = [  # (tool name, input) per model turn; the last turn is plain text (a finish request)
    ("probe_video", {"video": str(VIDEO)}),
    ("make_contact_sheet", {"video": str(VIDEO), "out_path": "sheet.jpg", "count": 4}),
    ("look", {"paths": ["sheet.jpg"]}),
    ("finish", {"summary": "stub run"}),
]
SEEN = {"requests": 0, "tool_results": 0, "images": 0, "errors": []}


def _check_request(kind: str, body: dict, turn: int):
    """What every request must carry; failures are collected and reported by the test."""
    tools = body.get("tools") or []
    if not tools:
        SEEN["errors"].append(f"{kind} request {turn}: no tools")
    if kind == "anthropic":
        if not body.get("system") or "DexAgent" not in body["system"]:
            SEEN["errors"].append("anthropic: system prompt missing")
        if any("input_schema" not in t for t in tools):
            SEEN["errors"].append("anthropic: a tool without input_schema")
        for m in body.get("messages", []):
            for c in m.get("content", []) if isinstance(m.get("content"), list) else []:
                if c.get("type") == "tool_result":
                    SEEN["tool_results"] += 1
                    SEEN["images"] += sum(1 for x in c.get("content", []) if x.get("type") == "image" and x["source"]["data"])
    else:
        if body["messages"][0]["role"] != "system":
            SEEN["errors"].append("openai: first message is not the system prompt")
        if any(t.get("type") != "function" or "parameters" not in t["function"] for t in tools):
            SEEN["errors"].append("openai: a tool is not a function with parameters")
        for m in body["messages"]:
            if m["role"] == "tool":
                SEEN["tool_results"] += 1
            if m["role"] == "user" and isinstance(m.get("content"), list):
                SEEN["images"] += sum(1 for x in m["content"] if x.get("type") == "image_url" and x["image_url"]["url"].startswith("data:image"))


def _turn_index(kind: str, body: dict) -> int:
    return sum(1 for m in body["messages"] if m["role"] == "assistant")


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        kind = "anthropic" if self.path.endswith("/messages") else "openai"
        turn = _turn_index(kind, body)
        SEEN["requests"] += 1
        _check_request(kind, body, turn)
        name, args = TURNS[min(turn, len(TURNS) - 1)]
        if kind == "anthropic":
            out = {"id": f"msg_{turn}", "type": "message", "role": "assistant", "model": body["model"], "stop_reason": "tool_use",
                   "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
                   "content": [{"type": "text", "text": f"turn {turn}"}, {"type": "tool_use", "id": f"toolu_{turn}", "name": name, "input": args}]}
        else:
            out = {"id": f"c{turn}", "object": "chat.completion", "created": 0, "model": body["model"],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                   "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {"role": "assistant", "content": f"turn {turn}",
                                "tool_calls": [{"id": f"call_{turn}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_case(provider: str, port: int, tmp: Path) -> list[str]:
    import yaml
    for k in ("requests", "tool_results", "images"):
        SEEN[k] = 0
    SEEN["errors"] = []
    base = f"http://127.0.0.1:{port}" + ("/v1" if provider == "openai" else "")
    # an API model run needs a spending cap and a price for its model; $1 at $1 / Mtok is far above the stub's tokens
    cfg = {"agent": {"provider": provider, "model": "stub-model", "api_key": "stub-key", "base_url": base,
                     "budget_usd": 1, "price_usd_per_mtok": {"input": 1, "output": 1}},
           "run": {"name": f"stub_{provider}", "output_dir": str(tmp / "runs")},
           "examples": {"reuse": False}, "inputs": {"video": str(VIDEO), "task": "Close the red gift box"}}
    cfg_path = tmp / f"{provider}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    sys.path.insert(0, str(REPO / "stage1_semantic_understanding"))
    import run_stage1
    rc = run_stage1.main([str(cfg_path), "--max-rounds", "4"])
    trace = [json.loads(l) for l in (tmp / "runs" / f"stub_{provider}" / "stage1" / "trace.jsonl").read_text().splitlines()]
    tools = [e["name"] for e in trace if e.get("event") == "tool"]
    fails = list(SEEN["errors"])
    if tools != ["probe_video", "make_contact_sheet", "look"]:
        fails.append(f"{provider}: tool calls executed {tools}")
    if SEEN["requests"] != 4:
        fails.append(f"{provider}: {SEEN['requests']} requests, expected 4")
    if SEEN["images"] < 1:
        fails.append(f"{provider}: the contact sheet image never reached the model")
    if not any(e.get("event") == "verify" for e in trace):
        fails.append(f"{provider}: finish did not run the verifiers")
    if rc != 1:   # the stub writes no deliverable, so the verifiers must fail and the stage must end unfinished
        fails.append(f"{provider}: return code {rc}, expected 1 (unfinished)")
    return fails


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["DEXAGENT_PLAIN"] = "1"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="dexagent_llm_", dir=os.environ.get("TMPDIR")))
    failed = False
    try:
        for provider in ("anthropic", "openai"):
            fails = run_case(provider, srv.server_address[1], tmp)
            if fails:
                failed = True
                print(f"[FAIL] {provider} client: " + "; ".join(fails))
            else:
                print(f"[PASS] {provider} client: 4 model turns, 3 tool calls, {SEEN['tool_results']} tool results, image delivered, verifiers ran")
    finally:
        srv.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
