"""Asset path portability check: does every file the scene references resolve relative to the scene's own folder?"""
from __future__ import annotations

TOOL = {
    "name": "asset_path_portability_check",
    "title": "Asset path portability check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Every mesh, texture, height field, skin and include path of an MJCF resolves relative to the scene's folder "
               "and exists; absolute or machine-specific paths, missing files and paths escaping the allowed root fail.",
    "description": (
        "A scene that names its meshes by absolute path opens only on the machine that wrote it: a reused scene copied "
        "into a repository or another run keeps pointing at the original checkout, and the copy silently depends on "
        "files that are not shipped with it. The check reads the MJCF as text (following <include> files), resolves "
        "every file attribute the way MuJoCo does (meshdir / texturedir / assetdir from <compiler>, relative to the main "
        "model's folder), and fails a reference that is absolute, starts with '~', uses backslashes or a drive letter, "
        "does not exist, or (when `root` is given) resolves outside `root`. A compiler meshdir/texturedir/assetdir that "
        "is absolute fails the same way. Each failing reference is reported with the relative path that would replace "
        "it, so the fix is a search-and-replace. Works on the file alone; nothing is compiled."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF file path (an XML string has no folder to be relative to)."},
        {"name": "root", "type": "str | None", "description": "Folder every referenced file must stay inside (e.g. the run or repository root); default: no containment rule."},
        {"name": "max_listed", "type": "int", "description": "Rows kept in the report per problem kind. Default 20."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; n_refs; counts {absolute, missing, outside_root, machine_specific}; bad: [{tag, name, file, "
                        "resolved, problem, suggest}]; worst: the first bad reference."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "assets", "paths", "portability", "reuse"],
    "status": "tested",
    "entrypoint": "asset_path_portability_check",
    "cli": "python -m tool_library.verifiers.asset_path_portability_check scene.xml",
}

import json
import os
import re
import sys
from pathlib import Path

from tool_library.common import compile_checks as C

_MACHINE = re.compile(r"^(~|[A-Za-z]:[\\/])|\\")


def asset_path_portability_check(scene, root=None, max_listed=20):
    p = C.scene_path(scene)
    if p is None:
        raise ValueError("scene must be an MJCF file path: an XML string or a compiled model has no folder to resolve against")
    if not p.exists():
        return {"ok": False, "issues": [f"scene file not found: {p}"], "n_refs": 0, "bad": [], "counts": {}}
    scene_dir = p.resolve().parent
    root_dir = Path(root).resolve() if root else None
    refs = C.mjcf_file_refs(p)
    bad, counts = [], {"absolute": 0, "missing": 0, "outside_root": 0, "machine_specific": 0, "unparsable": 0}
    for r in refs:
        if r["tag"] == "parse_error":
            counts["unparsable"] += 1
            bad.append({**r, "problem": f"cannot parse {r['file']}: {r.get('error')}", "suggest": None})
            continue
        f, res = r["file"], Path(r["resolved"])
        problems = []
        if _MACHINE.search(f):
            problems.append("machine-specific")
            counts["machine_specific"] += 1
        if os.path.isabs(f):
            problems.append("absolute")
            counts["absolute"] += 1
        if r["tag"] != "compiler" and not res.exists():
            problems.append("missing")
            counts["missing"] += 1
        if root_dir is not None:
            try:
                res.resolve().relative_to(root_dir)
            except ValueError:
                problems.append(f"outside root {root_dir}")
                counts["outside_root"] += 1
        if problems:
            suggest = None
            if res.exists() or r["tag"] == "compiler":
                try:
                    suggest = os.path.relpath(res.resolve(), scene_dir)
                except ValueError:
                    suggest = None
            bad.append({"tag": r["tag"], "name": r["name"], "attr": r["attr"], "file": f, "resolved": str(res),
                        "source": r["source"], "problem": ", ".join(problems), "suggest": suggest})
    issues = []
    if bad:
        kinds = {}
        for b in bad:
            kinds.setdefault(b["problem"], []).append(b)
        for kind, rows in kinds.items():
            ex = rows[0]
            fix = (f"rewrite it relative to the scene folder, e.g. file=\"{ex['suggest']}\"" if ex.get("suggest")
                   else "ship the file next to the scene and reference it relatively")
            if "missing" in kind:
                fix = "the file does not exist where MuJoCo will look: copy the asset next to the scene (or fix meshdir) and reference it relatively"
            issues.append(f"{len(rows)} {ex['tag']} reference(s) are {kind} (first: {ex['tag']} '{ex['name']}' file=\"{ex['file']}\" "
                          f"in {Path(ex['source']).name}): the scene only opens on this machine; {fix}")
    n = int(max_listed)
    return C.jsonable({"ok": not bad, "issues": issues, "n_refs": len([r for r in refs if r["tag"] != "compiler"]),
                       "counts": counts, "bad": bad[:n], "n_bad": len(bad), "worst": bad[0] if bad else None,
                       "scene_dir": str(scene_dir), "root": str(root_dir) if root_dir else None})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = asset_path_portability_check(argv[0], root=argv[1] if len(argv) > 1 else None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
