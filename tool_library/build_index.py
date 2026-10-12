"""Build tool_library/index.json from the TOOL metadata of every tool file.

    python tool_library/build_index.py

Each tool is one Python file under skills/ or verifiers/ that starts with a
module-level `TOOL = {...}` literal. This script reads those literals without
importing the files, so building the index needs no simulator installed.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/dexagent1/DexAgent/blob/main/tool_library"
REQUIRED = ("name", "title", "type", "category", "summary", "description", "inputs", "outputs",
            "status", "entrypoint")
STATUS = ("tested", "ported")
# the paper's eleven tasks (TASKS.md lists them first, in this order), then the extra meal-box sample; `used_in` may also name any other task a tool
# was made for (a user's own video), and those are listed under their own heading
TASKS = ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing",
         "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation",
         "Open the Bottle", "Scissor Cutting", "Biology Experiment", "Meal Box (extra sample)"]
PAPER_TASKS = TASKS[:11]
CATEGORIES = {
    "skill": ["Trajectory & Execution", "Object Reconstruction", "Perception & Grounding",
              "Scene Runtime & Rendering", "Grasp & Poses", "Measurements"],
    "verifier": ["Task & Rollout Gates", "Grasp Stress Tests", "Reconstruction Mechanism Tests",
                 "Episode Audits", "Scene Assertions", "Scene Compile Checks"],
}


def read_tool(path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "TOOL" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path}: no module-level TOOL = {{...}} literal")


# entry-point parameters that are code hooks or internals, never arguments a model passes
INTERNAL_PARAMS = {"log", "processor", "predictor", "model", "data", "progress", "record", "callback", "return_diag"}


def _entry_function(tree, name):
    """The FunctionDef the entrypoint names (following `name = other_function` aliases)."""
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for _ in range(3):
        if name in defs:
            return defs[name]
        alias = next((n.value.id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name)
                      and any(isinstance(t_, ast.Name) and t_.id == name for t_ in n.targets)), None)
        if alias is None:
            return None
        name = alias
    return None


def _type_of(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "any | None"


def mark_optional(path, tool):
    """inputs rows whose every name has a default in the entry point get optional: true (the schema then does not
    require them, whatever their description says)."""
    import re
    fn = _entry_function(ast.parse(path.read_text()), tool.get("entrypoint") or tool["name"])
    if fn is None:
        return
    args = fn.args.args + fn.args.kwonlyargs
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults) + list(fn.args.kw_defaults)
    has_default = {a_.arg for a_, d_ in zip(args, defaults) if d_ is not None}
    kw = fn.args.kwarg is not None                     # **params: the rows are its keys, all optional
    for e in tool["inputs"]:
        names = [n.strip() for n in re.split(r"[,/]", str(e.get("name", ""))) if n.strip()]
        if names and all(n in has_default or (kw and n not in {a_.arg for a_ in args}) for n in names):
            e["optional"] = True


def signature_inputs(path, tool):
    """Input rows for the entry point's parameters that TOOL['inputs'] does not list (grouped rows such as 'a, b' or
    'a / b' count as listing each name): typed from the default, so every argument the function takes reaches the
    model's schema. Code hooks and private parameters are left out."""
    tree = ast.parse(path.read_text())
    fn = _entry_function(tree, tool.get("entrypoint") or tool["name"])
    if fn is None:
        return []
    import re
    listed = {n.strip() for e in tool.get("inputs", []) for n in re.split(r"[,/]", str(e.get("name", ""))) if n.strip()}
    args = fn.args.args + fn.args.kwonlyargs
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + list(fn.args.defaults) + list(fn.args.kw_defaults)
    rows = []
    for a_, d_ in zip(args, defaults):
        if a_.arg in listed or a_.arg.startswith("_") or a_.arg in INTERNAL_PARAMS or a_.arg in ("self", "cls"):
            continue
        if d_ is None:
            rows.append({"name": a_.arg, "type": "any", "description": "Required."})
            continue
        try:
            v = ast.literal_eval(d_)
        except (ValueError, SyntaxError):
            continue                                  # a computed default: not something to pass as JSON
        if callable(v):
            continue
        rows.append({"name": a_.arg, "type": _type_of(v), "description": f"Default {v!r}."})
    return rows


def main(root=None):
    """Read every tool under `root` (default: this library) and write its index.json and TASKS.md."""
    root = Path(root) if root else ROOT
    tools = []
    for kind, folder in (("skill", "skills"), ("verifier", "verifiers")):
        for path in sorted((root / folder).glob("*.py")):
            if path.name.startswith("_"):
                continue
            tool = read_tool(path)
            missing = [k for k in REQUIRED if k not in tool]
            assert not missing, f"{path}: TOOL is missing {missing}"
            assert tool["type"] == kind, f"{path}: type must be '{kind}'"
            assert tool["category"] in CATEGORIES[kind], (
                f"{path}: category must be one of {CATEGORIES[kind]}")
            assert tool["status"] in STATUS, f"{path}: status must be one of {STATUS}"
            assert tool["name"] == path.stem, f"{path}: TOOL name must equal the file name"
            used = tool.get("used_in", [])
            assert isinstance(used, list), f"{path}: used_in must be a list of task names"
            for u in used:
                assert isinstance(u, str) and u.strip() and u == u.strip(), (
                    f"{path}: used_in entry {u!r} must be a non-empty task name without surrounding spaces")
            for io in ("inputs", "outputs"):
                for e in tool[io]:
                    assert {"name", "type", "description"} <= set(e), f"{path}: {io} entries need name, type, description"
            assert "added" not in tool and "date" not in tool, f"{path}: the library carries no time information"
            extra = signature_inputs(path, tool)
            if extra:
                tool["inputs"] = list(tool["inputs"]) + extra
            mark_optional(path, tool)
            tool["file"] = f"{folder}/{path.name}"
            tool["url"] = f"{REPO_URL}/{folder}/{path.name}"
            tools.append(tool)
    index = {
        "counts": {"skills": sum(t["type"] == "skill" for t in tools),
                   "verifiers": sum(t["type"] == "verifier" for t in tools)},
        "categories": CATEGORIES,
        "tasks": PAPER_TASKS,
        "other_tasks": other_tasks(tools),
        "tools": tools,
    }
    (root / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    write_tasks_table(tools, root / "TASKS.md")
    print(f"{index['counts']['skills']} skills, {index['counts']['verifiers']} verifiers -> {root / 'index.json'}")
    return index


def other_tasks(tools):
    """Task names in `used_in` that are not the paper's eleven, in first-seen order."""
    seen = []
    for t in tools:
        for u in t.get("used_in", []):
            if u not in PAPER_TASKS and u not in seen:
                seen.append(u)
    return seen


def _task_section(tools, task, level="##"):
    sk = sorted(t["name"] for t in tools if t["type"] == "skill" and task in t.get("used_in", []))
    ve = sorted(t["name"] for t in tools if t["type"] == "verifier" and task in t.get("used_in", []))
    return [f"{level} {task}", "", f"**Skills ({len(sk)})**: " + ", ".join(f"`{n}`" for n in sk), "",
            f"**Verifiers ({len(ve)})**: " + ", ".join(f"`{n}`" for n in ve), ""]


def write_tasks_table(tools, path=None):
    """TASKS.md: which tools each of the paper's tasks used, from the metadata, then the other tasks tools name."""
    lines = ["# Tools per task", "",
             "Generated by `build_index.py` from the `used_in` field of every tool. "
             "Each row is one of the paper's tasks; the columns are the skills and verifiers "
             "that were used to reconstruct it, generate its robot trajectory and verify it.", ""]
    for task in PAPER_TASKS:
        lines += _task_section(tools, task)
    others = other_tasks(tools)
    if others:
        lines += ["# Other tasks", "",
                  "Tasks outside the paper that a tool's `used_in` names (an extra sample, a video a user ran DexAgent on).", ""]
        for task in others:
            lines += _task_section(tools, task)
    (Path(path) if path else ROOT / "TASKS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
