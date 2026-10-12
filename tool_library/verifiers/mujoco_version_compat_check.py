"""MuJoCo version compatibility check: does the scene compile, and to the same model, under every interpreter the pipeline uses?"""
from __future__ import annotations

TOOL = {
    "name": "mujoco_version_compat_check",
    "title": "MuJoCo version compatibility check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Compiles the scene in each configured Python interpreter (e.g. the Stage 3 env with MuJoCo 3.10 and the Stage 4 env with "
               "3.14) and fails when one cannot compile it or when the models disagree in sizes, options, masses, inertias, joint "
               "ranges, actuators, equalities or keyframes.",
    "description": (
        "Stage 3 executes a scene under one MuJoCo and Stage 4 replays and renders it under another, so a scene must mean the "
        "same thing in both. Newer MuJoCo versions accept attributes older ones reject (MuJoCo 3.14 knows geom 'adhesion', "
        "3.10 refuses the file), and a changed default (mesh inertia, mesh frame alignment) changes masses and poses without "
        "an error. The check runs the same standalone compile-and-fingerprint script in every interpreter in `pythons` (or "
        "DEXAGENT_PYTHONS, separated by commas or the path separator; just this interpreter when neither is set) and compares "
        "each with the first: a compile failure, any size or option difference, body masses or inertias differing by more "
        "than `mass_rtol`, joint ranges, armature, damping, actuator gains/biases/ranges, equality data or keyframe qpos "
        "differing by more than 1e-6, and body positions, primitive geom sizes or any geom's world bounds at qpos0 (from the mesh "
        "vertices themselves, so a version's choice of mesh frame does not count) differing by more than `geom_atol_m` fail. "
        "Smaller differences (vertex digests, a mesh geom's equivalent-inertia size) are notes. With a single interpreter it reports "
        "that nothing was compared."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF file path."},
        {"name": "pythons", "type": "list[str] | None", "description": "Python interpreters to compile in; default DEXAGENT_PYTHONS or this interpreter."},
        {"name": "mass_rtol", "type": "float", "description": "Relative tolerance on body masses and inertias. Default 1e-3."},
        {"name": "geom_atol_m", "type": "float", "description": "Absolute tolerance on body positions and geom sizes, metres. Default 1e-4."},
        {"name": "timeout_s", "type": "float", "description": "Time limit per interpreter. Default 300."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; notes; interpreters: [{python, version, ok, hash, error}]; differences: [{python, field, detail}]; compared (bool)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "version", "mujoco", "stage 3", "stage 4", "portability"],
    "status": "tested",
    "entrypoint": "mujoco_version_compat_check",
    "cli": "python -m tool_library.verifiers.mujoco_version_compat_check scene.xml [python ...]",
}

import json
import sys

import numpy as np

from tool_library.common import compile_checks as C

_EXACT = ("jnt_range", "qpos0", "dof_armature", "dof_damping", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange", "eq_data",
          "key_qpos", "geom_margin", "geom_gap")
_MASS = ("body_mass", "body_inertia")
_GEOM = ("body_pos", "geom_size", "geom_world_bounds")


def _compare(a, b, mass_rtol, geom_atol):
    probs, notes = [], []
    for k in sorted(set(a["sizes"]) | set(b["sizes"])):
        if a["sizes"].get(k) != b["sizes"].get(k):
            probs.append((k, f"{a['sizes'].get(k)} vs {b['sizes'].get(k)}"))
    for k in sorted(set(a["opt"]) & set(b["opt"])):
        if not np.isclose(a["opt"][k], b["opt"][k], rtol=1e-9, atol=0):
            probs.append((f"opt.{k}", f"{a['opt'][k]} vs {b['opt'][k]}"))
    if probs and any(k in ("nbody", "njnt", "ngeom", "nq") for k, _ in probs):
        return probs, notes                      # different structure: array comparisons are meaningless
    for k in _MASS + _EXACT + _GEOM:
        if k not in a["small"] or k not in b["small"]:
            continue
        x, y = np.asarray(a["small"][k], float), np.asarray(b["small"][k], float)
        if x.shape != y.shape:
            probs.append((k, f"length {x.size} vs {y.size}"))
            continue
        if not x.size:
            continue
        if k == "geom_size" and "geom_type" in a["small"] and len(a["small"]["geom_type"]) * 3 == x.size:
            # a mesh geom's size is its equivalent inertia box, which versions derive differently; its real extent is
            # compared through geom_world_bounds
            keep = np.repeat(np.asarray(a["small"]["geom_type"]) != 7, 3)
            x, y = x[keep], y[keep]
            if not x.size:
                continue
        if k in _MASS:
            bad = ~np.isclose(x, y, rtol=mass_rtol, atol=1e-9)
        elif k in _GEOM:
            bad = np.abs(x - y) > geom_atol
        else:
            bad = ~np.isclose(x, y, rtol=1e-6, atol=1e-9)
        if bad.any():
            i = int(np.argmax(np.abs(x - y) * bad))
            probs.append((k, f"{int(bad.sum())} entries differ, largest at {i}: {x[i]:.6g} vs {y[i]:.6g}"))
        elif not np.array_equal(x, y):
            notes.append(f"{k} differs below tolerance (max {np.max(np.abs(x - y)):.2g})")
    da = [k for k in a["arrays"] if a["arrays"][k] != b["arrays"].get(k)]
    if da and not probs:
        notes.append(f"array digests differ within tolerance: {da[:8]} (mesh processing differs slightly between versions)")
    return probs, notes


def mujoco_version_compat_check(scene, pythons=None, mass_rtol=1e-3, geom_atol_m=1e-4, timeout_s=300.0):
    p = C.scene_path(scene)
    if p is None:
        raise ValueError("scene must be an MJCF file path (it is compiled by other interpreters)")
    if pythons is not None and (isinstance(pythons, str) or not all(isinstance(x, str) for x in pythons)):
        raise ValueError("pythons must be a list of interpreter paths")
    pys = list(pythons) if pythons else C.default_pythons()
    if not p.exists():
        return {"ok": False, "issues": [f"scene file not found: {p}"], "interpreters": []}
    results = []
    for py in pys:
        s = C.summarize(p) if py == sys.executable else C.compile_summary_in(py, p, timeout_s=timeout_s)
        results.append((py, s))
    issues, notes, diffs = [], [], []
    rows = [{"python": py, "version": s.get("version"), "ok": bool(s.get("ok")), "hash": s.get("hash"), "error": s.get("error")}
            for py, s in results]
    for r in rows:
        if not r["ok"]:
            issues.append(f"the scene does not compile under {r['python']} (MuJoCo {r['version'] or '?'}): {r['error']}; write only "
                          "attributes every pipeline MuJoCo knows (save scenes with the oldest version, or strip newer attributes)")
    good = [(py, s) for py, s in results if s.get("ok")]
    compared = len(good) > 1
    if len(pys) == 1:
        notes.append("only one interpreter is configured (pass pythons=[...] or set DEXAGENT_PYTHONS): nothing was compared")
    for py, s in good[1:]:
        probs, nts = _compare(good[0][1], s, mass_rtol, geom_atol_m)
        for f, det in probs:
            diffs.append({"python": py, "field": f, "detail": det})
        if probs:
            issues.append(f"MuJoCo {s['version']} ({py}) compiles a different model than MuJoCo {good[0][1]['version']} ({good[0][0]}): "
                          + "; ".join(f"{f}: {d}" for f, d in probs[:5]) + "; the two stages simulate different scenes: pin the "
                          "attribute whose default changed (e.g. mesh inertia) explicitly in the scene")
        notes += [f"{s['version']}: {n}" for n in nts]
    return C.jsonable({"ok": not issues, "issues": issues, "notes": notes, "interpreters": rows, "differences": diffs,
                       "compared": compared})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = mujoco_version_compat_check(argv[0], pythons=argv[1:] or None)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
