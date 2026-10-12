# Agent harness

The loop that lets a language model run one DexAgent stage with the tool library: it sends the stage prompt and the tools, executes the model's tool calls, feeds back the results (including images), runs the stage's verifiers when the model declares itself done, and returns the verifier findings to the model until they pass or the budget is spent. After a failed `finish` the model may call `report_blocked` with the reason the stage cannot be solved (a Stage 1 misreading, a part no hand can hold); the stage then ends as failed at once. A failed stage, blocked or out of budget, leaves `failure_report.json` with the reason and the last verdict.

```
agent/
├── config.py          reads dexagent.yaml, resolves paths
├── library.py         the tool library as function-calling tools: schemas from each tool's TOOL metadata, dispatch, per-tool Python environments
├── devtools.py        built-in tools: look, read_text, write_text, list_dir, run_python, library_search, library_read, library_create, finish, report_blocked
├── llm.py             Anthropic and OpenAI-compatible clients (tool use + images), and a scripted mock
├── runner.py          the generation-verification loop; writes trace.jsonl, progress.json and, when a stage fails, failure_report.json
├── console.py         the live terminal view of a running stage
├── reuse.py           verified examples: reuse an identical video's results after re-verifying them, or start from the same task's
├── toolcall.py        CLI: run one library tool by name with JSON arguments
├── devcall.py         CLI: run one built-in tool in a workspace
├── render_prompt.py   writes the exact prompt pack a stage sends (system prompt, tools, first message)
├── prompts/
│   ├── system.md          how the agent works: tools only, evidence from frames, finish and verify
│   └── building_tools.md  the rules for adding skills and verifiers
└── tests/             the LLM clients against a local stub server; ending a stage as blocked
```

## The model

`dexagent.yaml` names the provider, model and API key under `agent:`. Any Anthropic model works through the Messages API; any OpenAI-compatible endpoint through `provider: openai` and `base_url`. The model must accept images, because the agent sees the video only through contact sheets and frames returned by `look`.

## Tools

Every skill and verifier of the library is offered to the model as a tool, with its name, summary and input schema derived from the `TOOL` metadata in its file. A stage may restrict the categories it exposes (Stage 1 exposes Perception & Grounding, Measurements and Scene Assertions); everything else is still reachable from `run_python`.

Tools that need another Python environment (a model's packages) are routed through `agent/toolcall.py` in that interpreter:

```yaml
environments:
  sam3: /path/to/envs/sam3/bin/python
tool_environments:
  detect_objects: sam3
  segment_with_text_prompts: sam3
```

## New tools

`library_create` is the only way the agent adds to the library, and it enforces `prompts/building_tools.md`: existing tools are never modified (fork under a new name); the file carries a complete `TOOL` literal with `status: tested`; the test the agent supplies must pass on at least three distinct variations; no machine paths, no dates, no imports from outside the library. A tool that fails is removed again. Created tools are listed in the workspace's `new_tools.json` and in the stage's result.

## Running a stage by hand

`python -m agent.render_prompt stage1 dexagent.yaml out/` writes `system.md`, `tools.json` and `user.txt`. With `agent/toolcall.py` and `agent/devcall.py` a person, or another model, can execute the same tool calls the harness would. `trace.jsonl` in a run's workspace records every model turn, tool call and verifier verdict.
