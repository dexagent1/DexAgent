"""agent.toolcall / agent.devcall resolve a relative '@file.json' argument against the call's workspace first.

The agent writes its argument files into its workspace and calls `python -m agent.toolcall <tool> @args.json
--cwd <workspace>` from the repository; the file was looked for in the repository (the process's cwd) and not found,
or, worse, a same-named file there was used. These cases check read_args directly and both command lines end to end.

    python tool_library/tests/test_toolcall_args.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.toolcall import read_args  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_toolcall_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


ws, other = TMP / "ws", TMP / "elsewhere"
ws.mkdir(); other.mkdir()
(ws / "args.json").write_text('{"where": "workspace"}')
(other / "args.json").write_text('{"where": "cwd"}')
(other / "only_cwd.json").write_text('{"where": "cwd only"}')
old = os.getcwd()
os.chdir(other)
try:
    check("relative @file: the workspace's copy wins over the cwd's", json.loads(read_args("@args.json", ws))["where"] == "workspace")
    check("relative @file missing in the workspace: falls back to the cwd", json.loads(read_args("@only_cwd.json", ws))["where"] == "cwd only")
    check("no workspace given: the cwd (as before)", json.loads(read_args("@args.json", None))["where"] == "cwd")
    check("absolute @file and a literal pass through", json.loads(read_args(f"@{ws / 'args.json'}", other))["where"] == "workspace"
          and read_args('{"a": 1}', ws) == '{"a": 1}')
    try:
        read_args("@nowhere.json", ws); msg = ""
    except FileNotFoundError as e:
        msg = str(e)
    check("a missing @file names every place looked in", str(ws) in msg and str(other) in msg, msg[:120])
finally:
    os.chdir(old)

# end to end, from the repository as the harness runs it
sem = ws / "scene_semantics.json"
sem.write_text(json.dumps({"task_prompt": "move the cup", "objects": [], "subgoals": []}))
(ws / "call.json").write_text(json.dumps({"path": str(sem)}))
env = dict(os.environ, PYTHONPATH=str(ROOT), DEXAGENT_QUIET="1")
r = subprocess.run([sys.executable, "-m", "agent.toolcall", "scene_semantic_consistency_check", "@call.json", "--cwd", str(ws), "--config", "/dev/null"],
                   cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
out = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
check("toolcall: @call.json read from --cwd", "result" in out or out.get("ok") is not None and "not found" not in r.stderr, (r.stderr or r.stdout)[-200:])
# with the console line on (not quiet) and stderr read into the same stream, the JSON result is still the last line
env_loud = dict(env); env_loud.pop("DEXAGENT_QUIET", None)
r = subprocess.run([sys.executable, "-m", "agent.toolcall", "scene_semantic_consistency_check", "@call.json", "--cwd", str(ws), "--config", "/dev/null"],
                   cwd=str(ROOT), env=env_loud, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
try:
    merged_ok = isinstance(json.loads(last), dict)
except ValueError:
    merged_ok = False
check("toolcall: with stderr merged into stdout the JSON result is the last line", merged_ok and "▶ scene_semantic_consistency_check" in r.stdout, last[:120])
(ws / "ls.json").write_text(json.dumps({"path": "."}))
r = subprocess.run([sys.executable, "-m", "agent.devcall", str(ws), "list_dir", "@ls.json"], cwd=str(ROOT), env=env,
                   capture_output=True, text=True, timeout=120)
out = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
check("devcall: @ls.json read from the workspace", "ls.json" in out.get("entries", []), (r.stderr or r.stdout)[-200:])

# the models block of dexagent.yaml fills the checkout arguments a call leaves out, and the schema makes them optional
from agent.library import configured_args, find_tool, load_index, tool_schema  # noqa: E402
index = load_index(ROOT / "tool_library")
cfg = {"models": {"sam3d_repo": "/opt/sam-3d-objects", "video_depth_anything_ckpt": "/w/vda.pth", "sam2_checkpoint": "/w/sam2.pt"},
       "environments": {"bodex": "/envs/bodex/bin/python"}}
rec, depth, track, bodex = (find_tool(index, n) for n in ("reconstruct_object_meshes", "estimate_depth", "track_objects", "grasp_prior_bodex"))
check("models.sam3d_repo fills reconstruct_object_meshes", configured_args(cfg, rec) == {"sam3d_repo": "/opt/sam-3d-objects"})
check("a per-tool checkpoint goes to its own tool only", configured_args(cfg, depth) == {"checkpoint": "/w/vda.pth"}
      and configured_args(cfg, track) == {"checkpoint": "/w/sam2.pt"})
check("environments.bodex fills bodex_python", configured_args(cfg, bodex).get("bodex_python") == "/envs/bodex/bin/python")
s = tool_schema(rec, cfg)["input_schema"]
check("a configured argument is optional in the schema", "sam3d_repo" not in s["required"] and "sam3d_repo" in s["properties"],
      str(s["required"]))
check("without a configuration it stays required", "sam3d_repo" in tool_schema(rec)["input_schema"]["required"])
check("nothing is filled without a configuration", configured_args(None, rec) == {})

# a tool routed to another interpreter with a relative --cwd: the subprocess starts in cwd and must not resolve it again
from agent.library import call_tool  # noqa: E402
shim = TMP / "other_python"
shim.write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n"); shim.chmod(0o755)
(TMP / "rel").mkdir(); (TMP / "rel" / "scene_semantics.json").write_text(sem.read_text())
os.chdir(TMP)
try:
    r = call_tool(ROOT / "tool_library", index, "scene_semantic_schema_check", {"path": "scene_semantics.json"},
                  cfg={"environments": {"other": str(shim)}, "tool_environments": {"scene_semantic_schema_check": "other"}}, cwd=Path("rel"))
finally:
    os.chdir(old)
check("a routed tool with a relative cwd runs in that folder", r.get("ok") and (r.get("result") or {}).get("n_subgoals") is not None, str(r)[:200])

# the result is the LAST stdout line even when the tool prints during the call and something writes an empty line to
# stdout at exit (scene_qc and stage2_gate_suite ended stdout with a blank line, so parsing the last line failed)
fake = TMP / "noisylib"
(fake / "skills").mkdir(parents=True)
(fake / "skills" / "noisy_tool.py").write_text(
    "import atexit, os, sys\n"
    "atexit.register(lambda: (sys.stdout.write('\\n'), sys.stdout.flush()))\n"
    "def noisy_tool(n=2):\n"
    "    print('progress line from the tool')\n"
    "    os.write(1, b'raw C-level write\\n')\n"
    "    return {'ok': True, 'n': n}\n")
(fake / "index.json").write_text(json.dumps({"tools": [{"name": "noisy_tool", "type": "skill", "category": "Measurements",
                                                       "entrypoint": "noisy_tool", "inputs": [], "outputs": []}]}))
r = subprocess.run([sys.executable, "-m", "agent.toolcall", "noisy_tool", '{"n": 3}', "--library", str(fake), "--config", "/dev/null"],
                   cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
lines = r.stdout.splitlines()
try:
    last = json.loads(lines[-1]) if lines else {}
except json.JSONDecodeError:
    last = {}
check("toolcall: the result JSON is the last stdout line despite tool prints and an empty line written at exit",
      last.get("ok") is True and (last.get("result") or {}).get("n") == 3 and "progress line" in r.stderr and "raw C-level" in r.stderr,
      repr(r.stdout[-200:]))
r = subprocess.run([sys.executable, "-m", "agent.toolcall", "no_such_tool", "{}", "--library", str(fake), "--config", "/dev/null"],
                   cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
lines = r.stdout.splitlines()
check("toolcall: an unknown tool is still one JSON line on stdout", len(lines) == 1 and json.loads(lines[0]).get("ok") is False, repr(r.stdout[-200:]))

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
