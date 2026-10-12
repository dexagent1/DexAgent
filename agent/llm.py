"""Language-model clients with tool use and images: Anthropic, OpenAI-compatible, and a scripted mock.

A client returns a list of content blocks: {"type": "text", "text"} and
{"type": "tool_use", "id", "name", "input"}. The runner sends back tool results, each a
dict {"tool_use_id", "content": [text and image blocks]}. An API client also leaves the call's token counts in
`last_usage` ({input_tokens, output_tokens}), which the runner charges to the run's budget (agent/budget.py).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path


def image_block(path: str) -> dict:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return {"type": "image", "mime": mime, "data": base64.b64encode(p.read_bytes()).decode()}


class AnthropicLLM:
    priced = True                                   # its calls are charged to the run's budget

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, max_tokens: int = 8000):
        import anthropic
        kw = {"api_key": api_key or os.environ.get("ANTHROPIC_API_KEY")}
        if base_url:
            kw["base_url"] = base_url
        self.client = anthropic.Anthropic(**kw)
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        resp = self.client.messages.create(model=self.model, system=system, max_tokens=self.max_tokens,
                                           tools=tools, messages=[self._msg(m) for m in messages])
        u = getattr(resp, "usage", None)            # what this call cost, for the run's budget (agent/budget.py)
        self.last_usage = {"input_tokens": int(getattr(u, "input_tokens", 0) or 0) + int(getattr(u, "cache_creation_input_tokens", 0) or 0)
                           + int(getattr(u, "cache_read_input_tokens", 0) or 0),
                           "output_tokens": int(getattr(u, "output_tokens", 0) or 0)} if u is not None else None
        out = []
        for b in resp.content:
            if b.type == "text":
                out.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        return out

    @staticmethod
    def _msg(m: dict) -> dict:
        if m["role"] == "assistant":
            return {"role": "assistant", "content": m["content"]}
        content = []
        for c in m["content"]:
            if c["type"] == "tool_result":
                blocks = []
                for x in c["content"]:
                    if x["type"] == "text":
                        blocks.append({"type": "text", "text": x["text"]})
                    else:
                        blocks.append({"type": "image", "source": {"type": "base64", "media_type": x["mime"], "data": x["data"]}})
                content.append({"type": "tool_result", "tool_use_id": c["tool_use_id"], "content": blocks})
            elif c["type"] == "image":
                content.append({"type": "image", "source": {"type": "base64", "media_type": c["mime"], "data": c["data"]}})
            else:
                content.append({"type": "text", "text": c["text"]})
        return {"role": "user", "content": content}


class OpenAILLM:
    """Chat Completions with function calling. Images in tool results travel in a following
    user message, because the tool role carries text only."""
    priced = True                                   # its calls are charged to the run's budget

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, max_tokens: int = 8000):
        import openai
        self.client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url)
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> list[dict]:
        oa_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
        msgs = [{"role": "system", "content": system}]
        for m in messages:
            msgs.extend(self._msgs(m))
        resp = self.client.chat.completions.create(model=self.model, messages=msgs, tools=oa_tools, max_completion_tokens=self.max_tokens)
        u = getattr(resp, "usage", None)            # what this call cost, for the run's budget (agent/budget.py)
        self.last_usage = {"input_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                           "output_tokens": int(getattr(u, "completion_tokens", 0) or 0)} if u is not None else None
        ch = resp.choices[0].message
        out = []
        if ch.content:
            out.append({"type": "text", "text": ch.content})
        for tc in ch.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            out.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
        return out

    @staticmethod
    def _msgs(m: dict) -> list[dict]:
        if m["role"] == "assistant":
            text = "".join(c["text"] for c in m["content"] if c["type"] == "text")
            calls = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["input"])}}
                     for c in m["content"] if c["type"] == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            return [msg]
        out, images = [], []
        for c in m["content"]:
            if c["type"] == "tool_result":
                text = "\n".join(x["text"] for x in c["content"] if x["type"] == "text")
                imgs = [x for x in c["content"] if x["type"] == "image"]
                if imgs:
                    text += f"\n[{len(imgs)} image(s) follow in the next message]"
                    images.extend(imgs)
                out.append({"role": "tool", "tool_call_id": c["tool_use_id"], "content": text or "(no output)"})
            elif c["type"] == "image":
                images.append(c)
            else:
                out.append({"role": "user", "content": c["text"]})
        if images:
            out.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:{x['mime']};base64,{x['data']}"}} for x in images]})
        return out


class MockLLM:
    """Replays scripted turns; for testing the loop without an API."""

    def __init__(self, turns: list[list[dict]]):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []

    def complete(self, system, messages, tools):
        self.seen.append(messages)
        if not self.turns:
            return [{"type": "text", "text": "(mock has no more turns)"}]
        turn = self.turns.pop(0)
        for i, b in enumerate(turn):
            if b["type"] == "tool_use" and "id" not in b:
                b["id"] = f"mock_{len(self.seen)}_{i}"
        return turn


_MOCK_RUNS: dict = {}


def load_mock(path, stage: str | None = None) -> MockLLM:
    """A scripted mock from a JSON file: a list of turns (every stage replays it), or {stage: turns} per stage, or
    {stage: [turns of the first run, turns of the second run, ...]} when a stage runs more than once in one process (a
    stage re-opened by a cross-stage repair gets the next script)."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        return MockLLM(data)
    runs = data.get(stage) or []
    if runs and isinstance(runs[0], list) and runs[0] and isinstance(runs[0][0], list):
        k = _MOCK_RUNS.get((str(path), stage), 0)
        _MOCK_RUNS[(str(path), stage)] = k + 1
        return MockLLM(runs[k] if k < len(runs) else [])
    return MockLLM(runs)


def make_llm(cfg: dict):
    a = cfg.get("agent", {})
    provider = (a.get("provider") or "anthropic").lower()
    if provider == "external":
        from agent.external import ExternalLLM
        directory = a.get("transport_dir")
        if not directory:
            raise ValueError("agent.transport_dir is required for the external model transport")
        return ExternalLLM(directory, timeout_s=a.get("response_timeout_s", 3600))
    kw = dict(model=a.get("model") or "claude-opus-5-5", api_key=a.get("api_key") or None, base_url=a.get("base_url") or None)
    if provider in ("anthropic", "openai"):
        from agent.budget import require_cap
        require_cap(cfg)                            # a model API run needs agent.budget_usd (and a price for its model)
    if provider == "anthropic":
        return AnthropicLLM(**kw)
    if provider == "openai":
        return OpenAILLM(**kw)
    raise ValueError(f"unknown model provider: {provider}")
