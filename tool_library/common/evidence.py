"""Content identity for tool inputs and execution evidence.

Hashes bind a report to its inputs; they are not a substitute for independently
running a verifier. In particular, a report supplied by a generator is untrusted.
"""
from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def value_digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_record(value, cwd=None):
    path = None
    if isinstance(value, (str, Path)):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = Path(cwd or Path.cwd()) / candidate
        if candidate.is_file():
            path = candidate.resolve()
            if path.suffix.lower() == ".json":
                value = json.loads(path.read_text())
            else:
                return {"path": str(path), "sha256": file_digest(path), "value": str(path), "encoding": "file"}
    # Keep the value that was actually supplied even if a tool later mutates an
    # in-memory dictionary or an element of an input list.
    value = copy.deepcopy(value)
    result = {"value": value, "sha256": value_digest(value)}
    if path:
        result["path"] = str(path)
    return result


def tool_inputs(args, cwd=None):
    """Snapshot all state-bearing grasp inputs before invoking the verifier.

    Plural poses contain *resolved pose values*, not merely mutable filenames.
    Their order matters: it determines which fingers retain each object.
    """
    inputs = {key: input_record(args[key], cwd)
              for key in ("pose", "scene", "objects", "trajectory", "frame", "side") if key in args}
    if "poses" in args:
        if not isinstance(args["poses"], (list, tuple)):
            raise ValueError("poses must be an ordered list")
        items = [input_record(pose, cwd) for pose in args["poses"]]
        inputs["poses"] = {**input_record([item["value"] for item in items]), "items": items}
    return inputs


def record_tool_evidence(lib, tool, args, result, workspace, inputs=None):
    if not isinstance(result, dict):
        raise ValueError("a stress test must return a report dictionary")
    lib, workspace = Path(lib).resolve(), Path(workspace).resolve()
    folder = "skills" if tool["type"] == "skill" else "verifiers"
    source = lib / folder / (tool["name"] + ".py")
    inputs = inputs if inputs is not None else tool_inputs(args, workspace)
    if tool_inputs(args, workspace) != inputs:
        raise ValueError("stress test mutated a bound input (pose, scene, object set, or recording)")
    raw = {k: v for k, v in result.items() if k != "evidence"}
    report = {"schema": "dexagent.tool_evidence.v1", "tool": {
        "name": tool["name"], "type": tool["type"], "category": tool.get("category"),
        "source_sha256": file_digest(source)}, "inputs": inputs,
        "arguments": args, "cwd": str(workspace), "result": raw}
    data = canonical_bytes(report)
    digest = hashlib.sha256(data).hexdigest()
    target = workspace / ".evidence" / (digest + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with target.open("xb") as stream:
            stream.write(data)
    elif file_digest(target) != digest:
        raise ValueError("existing evidence file has changed")
    return {**raw, "evidence": {"path": str(target), "sha256": digest}}
