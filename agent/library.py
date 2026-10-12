"""The tool library as callable tools: schemas for function calling, dispatch, environment routing."""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def library_root(cfg: dict | None = None) -> Path:
    p = Path((cfg or {}).get("inputs", {}).get("library") or REPO / "tool_library")
    return p.resolve()


def load_index(lib: Path) -> dict:
    idx = lib / "index.json"
    if not idx.exists():
        rebuild_index(lib)
    return json.loads(idx.read_text())


def rebuild_index(lib: Path) -> str:
    r = subprocess.run([sys.executable, str(lib / "build_index.py")], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"build_index.py failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout.strip()


# ---------------------------------------------------------------- schemas
def _json_type(t: str) -> dict:
    """Describe outer argument types, preserving supported JSON alternatives.

    Nested value annotations must not change a container into a scalar (e.g.
    dict[str, bool]), and a pose/file union must not hide either representation.
    Loose prose annotations remain permissive instead of inventing a type.
    """
    t = (t or "").strip().lower()
    depth, start, alternatives = 0, 0, []
    for i, char in enumerate(t):
        if char in "[({":
            depth += 1
        elif char in "])}":
            depth = max(0, depth - 1)
        elif char == "|" and depth == 0:
            alternatives.append(t[start:i].strip())
            start = i + 1
    if alternatives:
        alternatives.append(t[start:].strip())
        schemas = []
        for alternative in alternatives:
            schema = _json_type(alternative)
            if not schema:
                return {}
            if schema not in schemas:
                schemas.append(schema)
        if {"type": "number"} in schemas and {"type": "integer"} in schemas:
            schemas.remove({"type": "integer"})
        return schemas[0] if len(schemas) == 1 else {"anyOf": schemas}
    kind = re.split(r"[^a-z_]+", t, maxsplit=1)[0]
    if kind in ("none", "null", "nonetype"):
        return {"type": "null"}
    if kind in ("str", "string", "path", "pathlib"):
        return {"type": "string"}
    if kind in ("bool", "boolean"):
        return {"type": "boolean"}
    if kind in ("dict", "object", "mapping") or t.startswith("{"):
        return {"type": "object"}
    if kind in ("list", "array", "tuple", "sequence", "ndarray", "np", "numpy") or t.startswith("["):
        return {"type": "array", "items": {}}
    if kind in ("int", "integer", "count", "index"):
        return {"type": "integer"}
    if kind in ("float", "number", "numbers", "metre", "meter", "seconds", "deg", "kg", "rad"):
        return {"type": "number"}
    return {}


def _optional(inp: dict) -> bool:
    if inp.get("optional"):                       # the entry point gives it a default (build_index)
        return True
    text = f"{inp.get('type', '')} {inp.get('description', '')}".lower()
    return "default" in text or "none" in inp.get("type", "").lower() or "optional" in text


# model checkouts and interpreters tools take as arguments; a call that leaves one out gets dexagent.yaml's value
CONFIG_ARGS = {
    "hamer_root": ("models", "hamer_root"), "sam3d_repo": ("models", "sam3d_repo"),
    "articraft_dir": ("models", "articraft_dir"), "void_dir": ("models", "void_dir"),
    "bodex_dir": ("models", "bodex_dir"), "bodex_python": ("environments", "bodex"),
    "dexgraspnet_dir": ("models", "dexgraspnet_dir"), "dexgraspnet_python": ("environments", "dexgraspnet"),
    "hand_urdf": ("models", "hand_urdf"), "vda_repo": ("models", "video_depth_anything_dir"),
    "isaac_python": ("environments", "isaac"),
    "spider_dir": ("models", "spider_dir"), "spider_python": ("environments", "spider"),
}
TOOL_CONFIG_ARGS = {"estimate_depth": {"checkpoint": ("models", "video_depth_anything_ckpt")},
                    "track_objects": {"checkpoint": ("models", "sam2_checkpoint")},
                    "optimize_grasp_pose": {"launcher": ("compute", "launcher"), "workers": ("compute", "workers")},
                    "sampling_mpc_refinement": {"workers": ("compute", "workers")},
                    "reinforcement_learning_tracking": {"workers": ("compute", "workers")}}


def configured_args(cfg: dict | None, tool: dict) -> dict:
    """The arguments of `tool` that the configuration supplies: {name: value} for every set key."""
    out = {}
    if not cfg:
        return out
    declared = {inp.get("name") for inp in tool.get("inputs") or []}
    for name, (block, key) in {**CONFIG_ARGS, **TOOL_CONFIG_ARGS.get(tool["name"], {})}.items():
        v = (cfg.get(block) or {}).get(key)
        if v and name in declared:
            out[name] = v if isinstance(v, (list, int, float)) else str(v)     # a command prefix stays a list, a count a number
    return out


def tool_schema(tool: dict, cfg: dict | None = None) -> dict:
    """Function-calling schema of one library tool, derived from its TOOL metadata. Arguments the configuration
    supplies (CONFIG_ARGS) are optional."""
    filled = configured_args(cfg, tool)
    props, required = {}, []
    rows = []
    for inp in tool.get("inputs", []):
        names = [n.strip() for n in re.split(r"[,/]", str(inp["name"]))]
        if len(names) > 1 and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in names):
            # a grouped row ('a, b' or 'a / b'): one property per name, the row's type and description shared
            rows += [dict(inp, name=n, description=f"(one of {inp['name']}) {inp.get('description', '')}") for n in names]
        else:
            rows.append(inp)
    for inp in rows:
        name = inp["name"]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue                      # '**params' or '...': documentation only
        schema = _json_type(inp.get("type", ""))
        schema["description"] = f"{inp.get('type', '')}: {inp.get('description', '')}".strip(": ")
        if name in filled:
            schema["description"] += f" (default from dexagent.yaml: {filled[name]})"
        props[name] = schema
        if not _optional(inp) and name not in filled:
            required.append(name)
    desc = tool.get("summary", "")
    outs = "; ".join(f"{o['name']}: {o['description']}" for o in tool.get("outputs", [])[:6])
    if outs:
        desc += f" Returns {outs}"
    return {"name": tool["name"], "description": desc[:1500],
            "input_schema": {"type": "object", "properties": props, "required": required}}


def tool_catalog(index: dict, categories=None, types=("skill", "verifier")) -> list[dict]:
    tools = [t for t in index["tools"] if t["type"] in types and (categories is None or t["category"] in categories)]
    return tools


# ---------------------------------------------------------------- dispatch
def _jsonable(v):
    import numpy as np
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist() if v.size <= 4096 else f"<array shape {v.shape}>"
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


def find_tool(index: dict, name: str) -> dict | None:
    for t in index["tools"]:
        if t["name"] == name:
            return t
    return None


def _load_json_dict_args(tool: dict, args: dict) -> dict:
    """An argument the tool declares as a dict (not 'dict | str') but that arrives as the path of a .json file is
    loaded: models naturally pass the file a previous tool wrote (hand_scene_from_recon's out_json)."""
    types = {}
    for spec in tool.get("inputs") or []:
        for n in str(spec.get("name", "")).replace("/", ",").split(","):
            types[n.strip()] = str(spec.get("type", ""))
    out = dict(args)
    for k, v in args.items():
        t = types.get(k, "")
        if isinstance(v, str) and v.endswith(".json") and "dict" in t and "str" not in t and Path(v).exists():
            try:
                out[k] = json.loads(Path(v).read_text())
            except Exception:
                pass
    return out


def _make_out_dirs(args: dict):
    """out_json / out_png / out_npz / out_xml / out_path ...: the folder of an output file exists before the tool writes it."""
    for k, v in args.items():
        if k.startswith("out") and k != "out_dir" and isinstance(v, str) and Path(v).suffix:
            try:
                Path(v).parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass


def call_tool_inprocess(lib: Path, tool: dict, args: dict):
    args = _load_json_dict_args(tool, args)
    _make_out_dirs(args)
    if str(lib.parent) not in sys.path:
        sys.path.insert(0, str(lib.parent))
    folder = "skills" if tool["type"] == "skill" else "verifiers"
    mod = importlib.import_module(f"{lib.name}.{folder}.{tool['name']}")
    fn = getattr(mod, tool.get("entrypoint") or tool["name"])
    return _jsonable(fn(**args))


def interpreter_for(cfg: dict | None, tool: dict) -> str | None:
    """A different python for tools whose packages live in another environment
    (dexagent.yaml: environments: {name: python_path}, tool_environments: {tool_name: name})."""
    if not cfg:
        return None
    envs = cfg.get("environments") or {}
    te = cfg.get("tool_environments") or {}
    env = te.get(tool["name"])
    if env and env in envs and envs[env]:
        return str(envs[env])
    return None


def call_tool(lib: Path, index: dict, name: str, args: dict, cfg: dict | None = None, cwd: Path | None = None,
              timeout: float = 3600.0, force_subprocess: bool = False) -> dict:
    """Run a library tool by name. Returns {ok, result | error}."""
    tool = find_tool(index, name)
    if tool is None:
        return {"ok": False, "error": f"no tool named {name!r} in the library"}
    cwd = Path(cwd).resolve() if cwd else None       # a routed tool's process starts in cwd: a relative one would resolve twice
    filled = configured_args(cfg, tool)
    args = {**filled, **{k: v for k, v in args.items() if not (v is None and k in filled)}}
    hf = ((cfg or {}).get("models") or {}).get("hf_home")
    if hf:
        os.environ.setdefault("HF_HOME", str(hf))
    python = interpreter_for(cfg, tool)
    try:
        stress = tool.get("type") == "verifier" and tool.get("category") == "Grasp Stress Tests"
        evidence_inputs = None
        if stress:
            from tool_library.common.evidence import tool_inputs
            evidence_inputs = tool_inputs(args, cwd)
        if force_subprocess or (python and Path(python).exists() and Path(python).resolve() != Path(sys.executable).resolve()):
            python = python or sys.executable
            cmd = [python, "-m", "agent.toolcall", name, json.dumps(args), "--library", str(lib), "--config", "/dev/null"]
            if cwd:
                cmd += ["--cwd", str(cwd)]
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{REPO}{os.pathsep}{lib.parent}{os.pathsep}" + env.get("PYTHONPATH", "")
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd or REPO), env=env, timeout=timeout)
            out = r.stdout.strip().splitlines()
            if out:
                try:
                    return json.loads(out[-1])        # the tool's own result, or its error with the traceback
                except json.JSONDecodeError:
                    pass
            if r.returncode != 0:
                return {"ok": False, "error": (r.stderr or r.stdout)[-4000:]}
            return {"ok": True, "result": None}
        old = os.getcwd()
        if cwd:
            os.chdir(cwd)
        try:
            result = call_tool_inprocess(lib, tool, args)
            if stress:
                from tool_library.common.evidence import record_tool_evidence
                result = record_tool_evidence(lib, tool, args, result, cwd or Path.cwd(), evidence_inputs)
            return {"ok": True, "result": result}
        finally:
            os.chdir(old)
    except Exception as e:                # the model needs the message, not a crash
        import traceback
        result = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        if isinstance(e, TypeError):
            result["input_schema"] = tool_schema(tool, cfg)["input_schema"]
        return result
