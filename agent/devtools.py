"""Built-in tools of the harness: files, python, looking at images, and the library's
search / read / create. `library_create` enforces the rules in prompts/building_tools.md."""
from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

from agent.library import REPO, load_index, rebuild_index

REQUIRED = ("name", "title", "type", "category", "summary", "description", "inputs", "outputs", "status", "entrypoint")
CATEGORIES = {
    "skill": ["Trajectory & Execution", "Object Reconstruction", "Perception & Grounding",
              "Scene Runtime & Rendering", "Grasp & Poses", "Measurements"],
    "verifier": ["Task & Rollout Gates", "Grasp Stress Tests", "Reconstruction Mechanism Tests",
                 "Episode Audits", "Scene Assertions", "Scene Compile Checks"],
}
MIN_VARIATIONS = 3
RUN_PYTHON_MAX_S = 900.0                      # agent.run_python_max_s overrides

BUILTIN_SCHEMAS = [
    {"name": "look", "description": "Look at one or more images (paths). Use it on contact sheets and extracted frames; it is the only way to see the video.",
     "input_schema": {"type": "object", "properties": {"paths": {"type": "array", "items": {"type": "string"}}}, "required": ["paths"]}},
    {"name": "read_text", "description": "Read a text file (first max_chars characters).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_text", "description": "Write a text file inside the workspace (creates folders).",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}, "required": ["path", "text"]}},
    {"name": "list_dir", "description": "List a folder.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "run_python", "description": "Run a Python snippet in the workspace with the tool library importable (from tool_library.skills.x import x). Returns stdout and stderr. Use it for quick analysis and to try a tool before creating it.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "timeout_s": {"type": "number"}}, "required": ["code"]}},
    {"name": "library_search", "description": "Find tools by keywords (name, summary, tags, category). Returns name, type, category, summary and exposes their callable schemas in the next round.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "type": {"type": "string", "enum": ["skill", "verifier", "any"]}}, "required": ["query"]}},
    {"name": "library_read", "description": "Read a tool's full source and metadata and expose its callable schema in the next round.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "library_create", "description": "Add a NEW skill or verifier to the library. Existing tools are never modified; fork by reading one and creating another under a new name. The source must start with a TOOL = {...} literal and define its entrypoint; test_source must exercise at least three distinct variations and print one '[PASS] <case>' line per passing case. The tool is kept only if the test passes.",
     "input_schema": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["skill", "verifier"]}, "name": {"type": "string"}, "source": {"type": "string"}, "test_source": {"type": "string"}}, "required": ["kind", "name", "source", "test_source"]}},
    {"name": "finish", "description": "Declare the stage done. The harness then runs the stage's verifiers on your deliverable and either ends or returns their failures to you.",
     "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}},
    {"name": "report_blocked", "description": "End the stage as FAILED, before the budget runs out, when a verifier failure cannot be fixed honestly in this stage. Only accepted after finish has returned failures; the reason goes into failure_report.json. When the cause is an earlier stage's result (Stage 1 read an occlusion as a pick; a Stage 2 asset lacks the part the hand must hold; an object misplaced by centimetres), name that stage in `upstream` and give the measured `evidence`: the run then re-opens that stage once with your reason in its first message, re-runs the stages after it, and continues. Without upstream the stage ends as failed at once. An operator administrative checkpoint can stop immediately.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"},
                                                       "upstream": {"type": "string", "enum": ["stage1", "stage2"], "description": "The earlier stage whose result is wrong (only an earlier stage than this one)."},
                                                       "evidence": {"type": "string", "description": "What was measured or seen that shows it (required with upstream): numbers, frames, files."}},
                      "required": ["reason"]}},
]


class DevTools:
    def __init__(self, lib: Path, workspace: Path, cfg: dict | None = None):
        self.lib = Path(lib)
        self.ws = Path(workspace)
        self.cfg = cfg or {}
        self.index = load_index(self.lib)
        self.created: list[dict] = []

    # ------------------------------------------------------------ files
    def _resolve(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (self.ws / q)

    def read_text(self, path: str, max_chars: int = 20000):
        t = self._resolve(path).read_text(errors="replace")
        return {"path": str(self._resolve(path)), "chars": len(t), "text": t[: int(max_chars)]}

    def write_text(self, path: str, text: str):
        p = self._resolve(path).resolve()
        if not p.is_relative_to(self.ws.resolve()):
            raise PermissionError("write_text may write only inside the candidate workspace; use library_create for a new tool")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return {"written": str(p), "chars": len(text)}

    def list_dir(self, path: str):
        p = self._resolve(path)
        return {"path": str(p), "entries": sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())[:500]}

    def look(self, paths: list[str]):
        out = []
        for p in paths:
            q = self._resolve(p)
            if not q.exists():
                out.append({"path": str(q), "error": "no such file"})
            else:
                out.append({"path": str(q)})
        return {"images": out}

    def run_python(self, code: str, timeout_s: float = 600.0):
        # one script may not hold the stage for long: an exploratory sweep that needs more than this belongs in a
        # library search tool (parallel, stoppable) or in several smaller scripts
        cap = float((self.cfg.get("agent") or {}).get("run_python_max_s", RUN_PYTHON_MAX_S))
        timeout_s = min(float(timeout_s or 600.0), cap)
        script = self.ws / f"_run_{int(time.time() * 1000)}.py"
        script.write_text(code)
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{self.lib.parent}{os.pathsep}{REPO}{os.pathsep}" + env.get("PYTHONPATH", "")
        env.setdefault("MUJOCO_GL", "egl")
        try:
            r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(self.ws), env=env, timeout=timeout_s)
            return {"ok": r.returncode == 0, "returncode": r.returncode,
                    "stdout": r.stdout[-12000:], "stderr": r.stderr[-6000:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"timed out after {timeout_s} s (run_python is capped at "
                    f"{cap:.0f} s: split the work, or use a library search tool)"}
        finally:
            script.unlink(missing_ok=True)

    # ------------------------------------------------------------ library
    def library_search(self, query: str, type: str = "any"):
        # Whole-token, length-normalized retrieval: repeated generic words in
        # long constructor summaries must not bury a relevant motion skill.
        # Description/argument terms make capabilities discoverable even when
        # the tool name does not name the manipulated mechanism.
        tokenize = lambda text: re.findall(r"[a-z0-9]+", str(text).lower())
        words = set(tokenize(query)) - {"a", "an", "the", "of", "to", "for", "in", "on", "at", "by", "with",
                                       "and", "or", "is", "are", "be", "it", "that", "this", "from", "as"}
        candidates = [t for t in self.index["tools"] if type == "any" or t["type"] == type]
        weights = (6., 4., 3., 2., 1., 1.)
        corpus = []
        for tool in candidates:
            fields = (tool["name"], tool.get("title", ""), " ".join(tool.get("tags", [])) + " " + tool.get("category", ""),
                      tool.get("summary", ""), tool.get("description", ""),
                      " ".join(str(i.get("name", "")) + " " + str(i.get("description", "")) for i in tool.get("inputs", [])))
            corpus.append([Counter(tokenize(field)) for field in fields])
        count = len(corpus)
        if not count or not words:
            return {"tools": []}
        averages = [max(1., sum(sum(doc[k].values()) for doc in corpus) / count) for k in range(len(weights))]
        frequencies = {w: sum(any(w in field for field in doc) for doc in corpus) for w in words}
        preferred = set(getattr(self, "preferred_categories", None) or [])
        hits = []
        for tool, doc in zip(candidates, corpus):
            score, matched = 0., 0
            for word in words:
                if not any(word in field for field in doc):
                    continue
                matched += 1
                idf = math.log(1. + (count - frequencies[word] + .5) / (frequencies[word] + .5))
                for field, weight, avg in zip(doc, weights, averages):
                    tf = field[word]
                    if tf:
                        score += weight * idf * (tf * 2.2) / (tf + 1.2 * (.25 + .75 * sum(field.values()) / avg))
            if score:
                score *= matched / len(words)
                if tool.get("category") in preferred:
                    score *= 1.3
                if query.strip().lower() == tool["name"].lower():
                    score += 1000.
                hits.append((score, tool))
        hits.sort(key=lambda x: (-x[0], x[1]["name"]))
        return {"tools": [{"name": t["name"], "type": t["type"], "category": t["category"], "summary": t["summary"]} for _, t in hits[:25]]}

    def library_read(self, name: str):
        for t in self.index["tools"]:
            if t["name"] == name:
                src = (self.lib / t["file"].split("/", 1)[1] if t["file"].startswith("tool_library/") else self.lib / t["file"]).read_text()
                return {"name": name, "file": t["file"], "metadata": {k: t[k] for k in ("type", "category", "summary", "inputs", "outputs", "requirements", "entrypoint") if k in t}, "source": src}
        return {"error": f"no tool named {name!r}"}

    def library_create(self, kind: str, name: str, source: str, test_source: str):
        folder = {"skill": "skills", "verifier": "verifiers"}.get(kind)
        if folder is None:
            return {"ok": False, "error": "kind must be skill or verifier"}
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,60}", name):
            return {"ok": False, "error": "name must be snake_case, 3-60 characters"}
        for f in ("skills", "verifiers"):
            if (self.lib / f / f"{name}.py").exists():
                return {"ok": False, "error": f"{name} already exists in {f}/; existing tools are never modified. Choose a new name (for example {name}_v2 or a more specific name) and fork."}
        meta, err = self._parse_tool_literal(source)
        if err:
            return {"ok": False, "error": err}
        if meta.get("name") != name:
            return {"ok": False, "error": f"TOOL['name'] is {meta.get('name')!r} but the file name is {name!r}; they must match"}
        if meta.get("type") != kind:
            return {"ok": False, "error": f"TOOL['type'] must be {kind!r}"}
        if meta.get("category") not in CATEGORIES[kind]:
            return {"ok": False, "error": f"TOOL['category'] must be one of {CATEGORIES[kind]}"}
        if meta.get("status") != "tested":
            return {"ok": False, "error": "TOOL['status'] must be 'tested': a tool enters the library only with a passing test"}
        if "added" in meta or "date" in meta:
            return {"ok": False, "error": "the library carries no time information: remove 'added'/'date'"}
        ep = meta.get("entrypoint") or name
        if not re.search(rf"^def {re.escape(ep)}\s*\(", source, re.M) and not re.search(rf"^{re.escape(ep)}\s*=", source, re.M):
            return {"ok": False, "error": f"entrypoint {ep!r} is not defined in the source"}
        for io in ("inputs", "outputs"):
            for e in meta.get(io, []):
                if not {"name", "type", "description"} <= set(e):
                    return {"ok": False, "error": f"every {io} entry needs name, type and description"}
        if source.count("import agents.") or re.search(r"^\s*(from|import)\s+(agents|main)\b", source, re.M):
            return {"ok": False, "error": "a library tool imports only tool_library.common, the standard library and third-party packages"}
        if re.search(r"/(home|Users)/\w+|/[a-z]+/u/\w+|/[a-z]+/data/\w+|[A-Z]:\\\\", source + test_source):
            return {"ok": False, "error": "no machine-specific paths in a library tool or its test"}
        if not re.search(rf"\b{re.escape(ep)}\b", test_source):
            return {"ok": False, "error": f"test_source must import and call the tool's entrypoint {ep!r}: its cases test the tool itself"}
        n_pass_lines = len(re.findall(r"\[PASS\]", test_source))
        if n_pass_lines < 1:
            return {"ok": False, "error": "test_source must print '[PASS] <case>' per passing variation (at least three distinct cases) and exit non-zero on failure"}
        tool_path = self.lib / folder / f"{name}.py"
        test_path = self.lib / "tests" / f"test_{name}.py"
        tool_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        tool_path.write_text(source)
        test_path.write_text(test_source)
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{self.lib.parent}{os.pathsep}{REPO}{os.pathsep}" + env.get("PYTHONPATH", "")
        env.setdefault("MUJOCO_GL", "egl")
        try:
            r = subprocess.run([sys.executable, str(test_path)], capture_output=True, text=True, cwd=str(self.ws), env=env, timeout=1800)
        except subprocess.TimeoutExpired:
            tool_path.unlink(missing_ok=True); test_path.unlink(missing_ok=True)
            return {"ok": False, "error": "the test timed out (30 min)"}
        passes = sorted(set(re.findall(r"\[PASS\]\s*(.+)", r.stdout)))
        fails = re.findall(r"\[FAIL\].*", r.stdout)
        if r.returncode != 0 or fails or len(passes) < MIN_VARIATIONS:
            tool_path.unlink(missing_ok=True); test_path.unlink(missing_ok=True)
            return {"ok": False, "error": f"test did not qualify the tool: returncode {r.returncode}, {len(passes)} distinct passing cases (need {MIN_VARIATIONS}), fails={fails[:5]}",
                    "stdout": r.stdout[-6000:], "stderr": r.stderr[-4000:]}
        try:
            # the index is built without importing; it validates the metadata the same way the repository does
            msg = rebuild_index(self.lib)
        except Exception as e:
            tool_path.unlink(missing_ok=True); test_path.unlink(missing_ok=True)
            return {"ok": False, "error": f"index rejected the tool: {e}"}
        self.index = load_index(self.lib)
        rec = {"kind": kind, "name": name, "file": str(tool_path.relative_to(self.lib.parent)), "test": str(test_path.relative_to(self.lib.parent)), "cases": passes}
        self.created.append(rec)
        (self.ws / "new_tools.json").write_text(json.dumps(self.created, indent=1))
        return {"ok": True, "created": rec, "index": msg}

    @staticmethod
    def _parse_tool_literal(source: str):
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return None, f"source does not parse: {e}"
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TOOL" for t in node.targets):
                try:
                    meta = ast.literal_eval(node.value)
                except Exception as e:
                    return None, f"TOOL must be a plain literal (no calls, no names): {e}"
                missing = [k for k in REQUIRED if k not in meta]
                if missing:
                    return None, f"TOOL is missing {missing}"
                return meta, None
        return None, "no module-level TOOL = {...} literal found"
