"""build_index.py on a scratch library: `used_in` may name any task, TASKS.md lists the paper's tasks and then the others.

A tool made for a user's own task could not say so: build_index refused every used_in entry outside the paper's
eleven tasks. These cases build the index of a two-tool scratch library (never the real one) and check the
real library still builds.

    python tool_library/tests/test_build_index.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("build_index", ROOT / "tool_library" / "build_index.py")
BI = importlib.util.module_from_spec(spec); spec.loader.exec_module(BI)

TMP = Path(tempfile.mkdtemp(prefix="dexagent_index_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def tool(name, kind, used_in):
    cat = "Measurements" if kind == "skill" else "Scene Assertions"
    lit = {"name": name, "title": name, "type": kind, "category": cat, "summary": "s.", "description": "d.",
           "inputs": [{"name": "x", "type": "float", "description": "x"}], "outputs": [{"name": "y", "type": "dict", "description": "y"}],
           "status": "tested", "entrypoint": name}
    if used_in is not None:
        lit["used_in"] = used_in
    return f"TOOL = {lit!r}\n\ndef {name}(x=0):\n    return {{'ok': True}}\n"


def library(tools):
    lib = TMP / f"lib{len(list(TMP.iterdir()))}"
    for d in ("skills", "verifiers"):
        (lib / d).mkdir(parents=True)
    for name, kind, used in tools:
        (lib / ("skills" if kind == "skill" else "verifiers") / f"{name}.py").write_text(tool(name, kind, used))
    return lib


# 1. a paper task and the user's own task side by side
lib = library([("measure_thing", "skill", ["Place Cup in Bowl", "Fidget Spinner Rings"]),
               ("thing_check", "verifier", ["Fidget Spinner Rings"]), ("plain_skill", "skill", None)])
idx = BI.main(lib)
md = (lib / "TASKS.md").read_text()
paper_at, other_at = md.find("## Place Cup in Bowl"), md.find("# Other tasks")
check("used_in accepts a task outside the paper", idx["counts"] == {"skills": 2, "verifiers": 1} and idx["other_tasks"] == ["Fidget Spinner Rings"]
      and idx["tasks"] == BI.PAPER_TASKS, str(idx["other_tasks"]))
check("TASKS.md keeps the paper's tasks first and lists the other task under its own heading",
      0 <= paper_at < other_at and "## Fidget Spinner Rings" in md[other_at:] and "`measure_thing`" in md[other_at:] and "`thing_check`" in md[other_at:]
      and all(f"## {t}" in md[:other_at] for t in BI.PAPER_TASKS))

# 2. only paper tasks: no 'Other tasks' heading
lib2 = library([("measure_cup", "skill", ["Place Cup in Bowl"])])
idx2 = BI.main(lib2)
check("no other tasks -> no extra heading", idx2["other_tasks"] == [] and "# Other tasks" not in (lib2 / "TASKS.md").read_text())

# 3. malformed used_in entries are still refused
for bad, why in (("", "empty"), ("  Padded  ", "padded"), (7, "not a string")):
    lib3 = library([("bad_tool", "skill", [bad])])
    try:
        BI.main(lib3); refused = False
    except AssertionError as e:
        refused = "used_in" in str(e)
    check(f"used_in entry refused when {why}", refused)
lib4 = library([("bad_tool", "skill", "Place Cup in Bowl")])
try:
    BI.main(lib4); refused = False
except AssertionError as e:
    refused = "list" in str(e)
check("used_in that is not a list is refused", refused)

# 4. the real library still builds (into a copy, so the committed index is untouched)
copy = TMP / "real"
shutil.copytree(ROOT / "tool_library" / "skills", copy / "skills")
shutil.copytree(ROOT / "tool_library" / "verifiers", copy / "verifiers")
idx5 = BI.main(copy)
check("the library's own tools build", idx5["counts"]["skills"] > 50 and idx5["counts"]["verifiers"] > 50, json.dumps(idx5["counts"]))

# 6. every argument of the entry point reaches the model's schema: grouped rows expand to one property per name, a
# parameter the TOOL rows forget is added from the signature, and only parameters without a default are required
lib6 = library([("grouped_skill", "skill", None)])
src = (lib6 / "skills" / "grouped_skill.py").read_text()
src = src.replace("[{'name': 'x', 'type': 'float', 'description': 'x'}]",
                  "[{'name': 'scene', 'type': 'path', 'description': 'the scene'}, {'name': 'a / b', 'type': 'float', 'description': 'two knobs'}]")
src = src.replace("def grouped_skill(x=0):", "def grouped_skill(scene, a=1.0, b=2.0, forgotten_mm=3.5, log=print, _private=None):")
(lib6 / "skills" / "grouped_skill.py").write_text(src)
idx6 = BI.main(lib6)
sys.path.insert(0, str(ROOT))
from agent.library import tool_schema  # noqa: E402
s6 = tool_schema(idx6["tools"][0])["input_schema"]
check("grouped rows expand, a forgotten parameter is added, hooks and private ones are not",
      set(s6["properties"]) == {"scene", "a", "b", "forgotten_mm"}, str(sorted(s6["properties"])))
check("only parameters without a default are required", s6["required"] == ["scene"], str(s6["required"]))

shutil.rmtree(TMP, ignore_errors=True)
print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
