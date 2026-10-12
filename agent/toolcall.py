"""Call one library tool from the command line and print its JSON result.

    python -m agent.toolcall make_contact_sheet '{"video": "clip.mp4", "out_path": "sheet.jpg"}'
    python -m agent.toolcall scene_semantic_schema_check @args.json

The harness uses this to run a tool in another Python environment; a person (or a
model acting as the agent) can use it to call tools exactly as the harness does.
The result is the only line on stdout (so always the last one): whatever the tool or
its libraries print, during the call or at exit, goes to stderr.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# a tool call is one process among many (an agent run, several runs on one node): one BLAS thread each keeps the
# node responsive; MuJoCo itself is single-threaded
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agent.library import call_tool, load_index  # noqa: E402


def read_args(spec, base=None):
    """The JSON arguments of a call: the literal itself, or with '@file.json' that file. A relative file is looked for in
    `base` first (the call's --cwd / workspace, where the agent writes its argument files), then in the process's
    current directory; a missing file is an error that names both places."""
    if not spec.startswith("@"):
        return spec
    p = Path(spec[1:]).expanduser()
    tried = [p] if p.is_absolute() else ([Path(base) / p] if base else []) + [Path.cwd() / p]
    for c in tried:
        if c.is_file():
            return c.read_text()
    raise FileNotFoundError(f"argument file {spec[1:]!r} not found (looked in: {', '.join(str(c) for c in tried)})")


def _divert_stdout():
    """Point file descriptor 1 (and with it sys.stdout, C libraries and child processes) at stderr and return a
    duplicate of the real stdout. A tool that prints progress, a library that writes a newline when it is torn down at
    exit, a worker process that inherited the descriptor: none of them can then put a line after the result, so
    'the result is the JSON on the last stdout line' holds by construction (callers parse exactly that line)."""
    sys.stdout.flush()
    real = os.dup(1)
    os.dup2(2, 1)
    return real


def _emit_result(real_fd, out):
    with os.fdopen(real_fd, "w") as f:          # closing it leaves fd 1 on stderr until the process exits
        f.write(json.dumps(out, default=str) + "\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="call a library tool by name with JSON arguments")
    ap.add_argument("tool")
    ap.add_argument("args", nargs="?", default="{}", help="JSON object, or @file.json")
    ap.add_argument("--library", default=str(REPO / "tool_library"))
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--config", default=os.environ.get("DEXAGENT_CONFIG", str(REPO / "dexagent.yaml")),
                    help="dexagent.yaml whose environments/tool_environments route tools to other interpreters")
    a = ap.parse_args(argv)
    cfg = None
    if a.config and a.config != "/dev/null" and Path(a.config).exists():
        from agent.config import load_config
        cfg = load_config(a.config)
        envs = cfg.get("environments") or {}
        if envs.get("isaac"):
            os.environ.setdefault("DEXAGENT_ISAAC_PYTHON", str(envs["isaac"]))
    raw = read_args(a.args, a.cwd)
    args = json.loads(raw) if raw.strip() else {}
    lib = Path(a.library).resolve()
    import time
    t0 = time.time()
    real_stdout = _divert_stdout()
    try:
        out = call_tool(lib, load_index(lib), a.tool, args, cfg=cfg, cwd=Path(a.cwd) if a.cwd else None)
    except Exception as e:                    # an unreadable index or library: still one JSON line, never a bare traceback
        import traceback
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
    finally:
        sys.stdout.flush()
    # the same one-line account the stage console prints, on stderr and in <cwd>/calls.log, so a run driven
    # call by call (a person, or a model acting as the agent) leaves the same visible trail as the harness; it goes
    # out BEFORE the result, so the JSON stays the last line even when stderr and stdout are read as one stream
    if a.cwd and os.environ.get("DEXAGENT_QUIET", "") != "1":
        from agent.console import _fmt_args, _result_line
        line, ok = _result_line(a.tool, out, time.time() - t0)
        mark = "✓" if ok else ("✗" if ok is False else "·")
        text = f"▶ {a.tool}({_fmt_args(args)})\n  {mark} {line}"
        print(text, file=sys.stderr, flush=True)
        try:
            with (Path(a.cwd) / "calls.log").open("a") as f:
                f.write(time.strftime("%H:%M:%S ") + text.replace("\n", "\n         ") + "\n")
        except OSError:
            pass
    _emit_result(real_stdout, out)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
