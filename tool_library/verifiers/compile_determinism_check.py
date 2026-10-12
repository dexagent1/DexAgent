"""Compile determinism check: does the scene compile to the same model every time, everywhere, and as recorded?"""
from __future__ import annotations

TOOL = {
    "name": "compile_determinism_check",
    "title": "Compile determinism check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Compiles the scene several times in this process and once in a fresh interpreter in another working directory and "
               "compares a stable hash of its sizes and key arrays; optionally compares with a recorded hash and rebuilds the scene "
               "twice with its builder to catch unseeded randomness.",
    "description": (
        "Stages hand scenes to each other by file: Stage 2 verifies a scene, Stage 3 executes it, Stage 4 randomizes it. If "
        "the file compiles to a different model later (a mesh rewritten in place, an include edited, a builder that draws "
        "random numbers without a seed), every verdict reached on the earlier model is void and nothing says so. The check "
        "fingerprints the compiled model: its sizes, its options and a digest of each key array (body masses and poses, joint "
        "ranges, geom sizes and contact parameters, mesh vertices, actuators, equalities, keyframes), rounded to 1e-9. It "
        "compiles `repeats` times in-process and, with `fresh_process`, once more in a new interpreter started in another "
        "directory, and fails on any difference, naming the arrays (MuJoCo's in-process asset cache can serve a mesh rewritten "
        "within the same second, which only the fresh interpreter sees). With `expected_hash` (the hash a manifest recorded when "
        "the scene was verified) it fails when the scene has changed since. With `rebuild` (a callable, or a command line "
        "list, that regenerates `scene`) it runs the builder twice and fails if the two rebuilt models differ."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF file path."},
        {"name": "repeats", "type": "int", "description": "In-process compiles to compare. Default 2."},
        {"name": "fresh_process", "type": "bool", "description": "Also compile in a fresh interpreter in another working directory. Default True."},
        {"name": "expected_hash", "type": "str | None", "description": "Hash recorded earlier for this scene (the report's 'hash')."},
        {"name": "rebuild", "type": "callable | list[str] | None", "description": "Builder that rewrites `scene`: a function of no arguments, or a command line run twice."},
        {"name": "timeout_s", "type": "float", "description": "Time limit per subprocess (fresh compile, builder run). Default 300."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; hash (of the file's current content: the fresh interpreter's when it ran); in_process_hash; n_compiles; fresh_process_hash; differences: [{between, arrays}]; rebuild: {hashes, differing_arrays}; "
                        "sizes."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "determinism", "hash", "provenance", "builder"],
    "status": "tested",
    "entrypoint": "compile_determinism_check",
    "cli": "python -m tool_library.verifiers.compile_determinism_check scene.xml [expected_hash]",
}

import json
import subprocess
import sys
import tempfile

from tool_library.common import compile_checks as C


def _diff_arrays(a, b):
    out = [k for k in sorted(set(a.get("sizes", {})) | set(b.get("sizes", {}))) if a.get("sizes", {}).get(k) != b.get("sizes", {}).get(k)]
    out += [k for k in sorted(set(a.get("arrays", {})) | set(b.get("arrays", {}))) if a.get("arrays", {}).get(k) != b.get("arrays", {}).get(k)]
    out += [f"opt.{k}" for k in sorted(set(a.get("opt", {})) | set(b.get("opt", {}))) if a.get("opt", {}).get(k) != b.get("opt", {}).get(k)]
    return out


def _clear_cache():
    """Drop MuJoCo's in-process asset cache (where the binding has it) so a rebuilt mesh is read from disk."""
    import mujoco
    if hasattr(mujoco, "mj_clearCache") and hasattr(mujoco, "mj_getCache"):
        try:
            mujoco.mj_clearCache(mujoco.mj_getCache())
        except Exception:
            pass


def compile_determinism_check(scene, repeats=2, fresh_process=True, expected_hash=None, rebuild=None, timeout_s=300.0):
    p = C.scene_path(scene)
    if p is None:
        raise ValueError("scene must be an MJCF file path (determinism is a property of the file)")
    if int(repeats) < 1:
        raise ValueError("repeats must be >= 1")
    if rebuild is not None and not (callable(rebuild) or isinstance(rebuild, (list, tuple))):
        raise ValueError("rebuild must be a callable or a command line list")
    if not p.exists():
        return {"ok": False, "issues": [f"scene file not found: {p}"], "hash": None}
    issues, diffs = [], []
    sums = [C.summarize(p, keep_small=False) for _ in range(int(repeats))]
    if not sums[0]["ok"]:
        return C.compile_failure("fix the error MuJoCo names", sums[0]["error"], hash=None)
    for i, s in enumerate(sums[1:], 1):
        d = _diff_arrays(sums[0], s)
        if d:
            diffs.append({"between": f"compile 0 and compile {i}", "arrays": d})
            issues.append(f"compiling the same file twice in one process gives different models ({d[:6]}): the scene reads something that "
                          "changes between compiles (a file being rewritten, an include generated on the fly)")
    fresh = None
    if fresh_process:
        with tempfile.TemporaryDirectory() as td:
            fresh = C.compile_summary_in(sys.executable, p, timeout_s=timeout_s, cwd=td)
        if not fresh.get("ok"):
            issues.append(f"the scene does not compile in a fresh interpreter started elsewhere ({fresh.get('error')}): it depends on "
                          "this process's working directory or state")
        else:
            d = _diff_arrays(sums[0], fresh)
            if d:
                diffs.append({"between": "this process and a fresh interpreter in another directory", "arrays": d})
                issues.append(f"a fresh interpreter in another directory compiles a different model ({d[:6]}): the scene depends on the "
                              "working directory or on process state")
    # MuJoCo keeps an in-process asset cache: a mesh rewritten within the same second can compile from the cache here, so the
    # file's current content is judged by the fresh interpreter when there is one
    current = fresh["hash"] if fresh and fresh.get("ok") else sums[0]["hash"]
    if expected_hash is not None and str(expected_hash) != current:
        issues.append(f"the scene's hash is {current}, not the recorded {expected_hash}: the file (or a mesh, texture or include "
                      "it reads) changed since it was verified; re-run the verifiers on this version")
    reb = None
    if rebuild is not None:
        hashes, outs = [], []
        for _ in range(2):
            if callable(rebuild):
                rebuild()
            else:
                r = subprocess.run([str(x) for x in rebuild], capture_output=True, text=True, timeout=float(timeout_s))
                if r.returncode != 0:
                    raise ValueError(f"rebuild command failed (exit {r.returncode}): {(r.stderr or r.stdout)[-400:]}")
            _clear_cache()
            s = C.summarize(p, keep_small=False)
            if not s["ok"]:
                issues.append(f"the rebuilt scene does not compile: {s['error']}")
                break
            outs.append(s)
            hashes.append(s["hash"])
        reb = {"hashes": hashes}
        if len(outs) == 2:
            d = _diff_arrays(outs[0], outs[1])
            reb["differing_arrays"] = d
            if d:
                issues.append(f"running the builder twice gives two different scenes ({d[:6]}): it uses randomness without a fixed seed (or "
                              "the clock); seed it so a rebuilt scene is the verified one")
    return C.jsonable({"ok": not issues, "compiled": True, "issues": issues, "hash": current, "in_process_hash": sums[0]["hash"],
                       "n_compiles": len(sums),
                       "fresh_process_hash": fresh.get("hash") if fresh else None, "differences": diffs, "rebuild": reb,
                       "sizes": sums[0]["sizes"], "mujoco": sums[0]["version"]})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = compile_determinism_check(argv[0], expected_hash=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
