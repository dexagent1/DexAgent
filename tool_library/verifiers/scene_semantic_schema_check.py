"""Does a scene_semantics.json have the form the later stages read?"""
from __future__ import annotations

TOOL = {
    "name": "scene_semantic_schema_check",
    "title": "Scene semantic schema check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Validates a Stage 1 scene_semantics.json against the schema: required keys, enumerated object types, roles, hands and motions, id format, and the per-entry fields of objects and subgoals.",
    "description": (
        "Loads the JSON and checks it against stage1_semantic_understanding/schema.json (JSON Schema draft 2020-12, "
        "via jsonschema when installed, otherwise an equivalent built-in walk). Reports every violation with its JSON "
        "path so the agent can fix the file. Form only: whether the content matches the video is the job of "
        "scene_semantic_consistency_check and of the agent."
    ),
    "inputs": [
        {"name": "path", "type": "path", "description": "The scene_semantics.json to check."},
        {"name": "schema_path", "type": "path | None", "description": "Schema file. Default: the repository's stage1_semantic_understanding/schema.json."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues: list of '<json path>: <problem>'; counts of objects and subgoals."},
    ],
    "status": "tested",
    "entrypoint": "scene_semantic_schema_check",
    "requirements": ["jsonschema (optional)"],
    "tags": ["stage 1", "semantic", "schema", "json"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.verifiers.scene_semantic_schema_check runs/my_task/stage1/scene_semantics.json",
}

import json
import re
import sys
from pathlib import Path

DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "stage1_semantic_understanding" / "schema.json"


def _walk(instance, schema, path, issues):
    """A small JSON Schema walker covering what schema.json uses: type, required, properties, enum,
    items, minItems, maxItems, pattern, minimum, additionalProperties (as a schema)."""
    t = schema.get("type")
    tmap = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}
    if t and not isinstance(instance, tmap[t]) or (t == "integer" and isinstance(instance, bool)):
        issues.append(f"{path}: expected {t}, got {type(instance).__name__}")
        return
    if "enum" in schema and instance not in schema["enum"]:
        issues.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if "pattern" in schema and isinstance(instance, str) and not re.fullmatch(schema["pattern"], instance):
        issues.append(f"{path}: {instance!r} does not match {schema['pattern']}")
    if "minimum" in schema and isinstance(instance, (int, float)) and instance < schema["minimum"]:
        issues.append(f"{path}: {instance} < {schema['minimum']}")
    if isinstance(instance, dict):
        for k in schema.get("required", []):
            if k not in instance:
                issues.append(f"{path}: missing required key {k!r}")
        props = schema.get("properties", {})
        for k, v in instance.items():
            if k in props:
                _walk(v, props[k], f"{path}.{k}", issues)
            elif isinstance(schema.get("additionalProperties"), dict):
                _walk(v, schema["additionalProperties"], f"{path}.{k}", issues)
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            issues.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            issues.append(f"{path}: at most {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(instance):
                _walk(v, schema["items"], f"{path}[{i}]", issues)


def scene_semantic_schema_check(path, schema_path=None):
    p = Path(path)
    if not p.exists():
        return {"ok": False, "issues": [f"{p} does not exist"], "n_objects": 0, "n_subgoals": 0}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        return {"ok": False, "issues": [f"{p}: not valid JSON: {e}"], "n_objects": 0, "n_subgoals": 0}
    schema = json.loads(Path(schema_path or DEFAULT_SCHEMA).read_text())
    issues = []
    try:
        import jsonschema
        v = jsonschema.Draft202012Validator(schema)
        for err in sorted(v.iter_errors(data), key=lambda e: list(e.path)):
            where = "$" + "".join(f"[{x}]" if isinstance(x, int) else f".{x}" for x in err.path)
            issues.append(f"{where}: {err.message}")
    except ImportError:
        _walk(data, schema, "$", issues)
    objs = data.get("objects", []) if isinstance(data, dict) else []
    subs = data.get("subgoals", []) if isinstance(data, dict) else []
    return {"ok": not issues, "issues": issues, "n_objects": len(objs) if isinstance(objs, list) else 0,
            "n_subgoals": len(subs) if isinstance(subs, list) else 0}


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if not a:
        print(__doc__); return 2
    r = scene_semantic_schema_check(a[0], a[1] if len(a) > 1 else None)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
