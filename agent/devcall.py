"""Call one built-in harness tool (library_search, library_read, library_create, run_python, list_dir, ...)
from the command line, inside a workspace, and print its JSON result.

    python -m agent.devcall runs/my_task/stage1 library_search '{"query": "hand pose"}'
    python -m agent.devcall runs/my_task/stage1 library_create @new_tool.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.devtools import BUILTIN_SCHEMAS, DevTools  # noqa: E402
from agent.toolcall import read_args  # noqa: E402


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("tool", choices=[s["name"] for s in BUILTIN_SCHEMAS if s["name"] not in ("finish", "look")])
    ap.add_argument("args", nargs="?", default="{}", help="JSON object, or @file.json")
    ap.add_argument("--library", default=str(REPO / "tool_library"))
    a = ap.parse_args(argv)
    raw = read_args(a.args, a.workspace)            # a relative @file is the workspace's before the process's cwd
    dev = DevTools(Path(a.library).resolve(), Path(a.workspace).resolve())
    try:
        out = getattr(dev, a.tool)(**json.loads(raw or "{}"))
    except Exception as e:
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out, default=str))
    return 0 if out.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
