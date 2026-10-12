"""Isaac export check: does an Isaac Sim stage exported from a MuJoCo scene load, and does it still place the objects where the scene does?"""
from __future__ import annotations

TOOL = {
    "name": "isaac_export_check",
    "title": "Isaac export check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Opens a USD stage written by build_isaac_scene (or its soft-body variant) in Isaac Sim's Python and compares every "
               "exported static geom's pose with the MuJoCo scene at the export state, plus the export.json record with the scene "
               "even without Isaac; skips the Isaac part with ok and a note when Isaac is not configured.",
    "description": (
        "Deformable objects are simulated in Isaac Sim on a stage exported from the MuJoCo scene; if the scene is re-placed "
        "after the export, or the export used another state or a wrong quaternion order, the cord lies on a table and "
        "objects that are no longer where MuJoCo has them, and nothing fails. The check (1) reads the export.json next to "
        "the USD (the geoms build_isaac_scene exported, by name, with their poses) and compares each with the same geom of "
        "`scene` at `qpos` (default its keyframe 0, the export's own default) - this needs no Isaac; (2) when an Isaac Sim "
        "Python is configured (`isaac_python` or DEXAGENT_ISAAC_PYTHON, HOME redirected to `isaac_home` or "
        "DEXAGENT_ISAAC_HOME), starts it headless, opens the stage through omni.usd, and reads the world transform of every "
        "/World/static/g<k> prim (and /World/soft/b<k>, /World/cords/<name>), comparing the statics with the scene again; "
        "optionally it resets the PhysX world and steps it `physics_steps` times and checks the dynamic prims stay finite. "
        "Non-robot geoms off by more than `pos_tol_m` or `rot_tol_deg` fail (robot geoms are reported, since the robot pose "
        "depends on the exported state); a stage that does not open, an Isaac run that exits non-zero or returns no result, "
        "a missing cord named in the export, or a missing static prim fail. Without Isaac the report says the load test was skipped."
    ),
    "inputs": [
        {"name": "usd", "type": "str", "description": "The exported stage (<tag>_scene.usda)."},
        {"name": "scene", "type": "str", "description": "The MuJoCo scene the stage must match."},
        {"name": "export_json", "type": "str | None", "description": "export.json written with the stage; default the one next to the USD."},
        {"name": "qpos", "type": "str | list | None", "description": "State the export was made at (npy or list); default the scene's keyframe 0."},
        {"name": "isaac_python", "type": "str | None", "description": "Isaac Sim's Python; default DEXAGENT_ISAAC_PYTHON (skipped when unset)."},
        {"name": "isaac_home", "type": "str | None", "description": "HOME for Isaac's caches; default DEXAGENT_ISAAC_HOME, else a temporary folder."},
        {"name": "physics_steps", "type": "int", "description": "PhysX steps after a world reset (0 = only open and read). Default 0."},
        {"name": "pos_tol_m", "type": "float", "description": "Allowed position difference per geom, metres. Default 0.002."},
        {"name": "rot_tol_deg", "type": "float", "description": "Allowed rotation difference per geom, degrees. Default 1.0."},
        {"name": "timeout_s", "type": "float", "description": "Time limit of the Isaac run. Default 900."},
        {"name": "robot_prefix", "type": "str", "description": "Name prefix of robot bodies (their geoms are reported, not judged). Default 'robot_'."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; issues; notes; skipped_isaac (bool); export_vs_scene: {n, n_bad, worst}; usd_vs_scene: {n, n_bad, worst, n_prims, "
                        "missing_statics, cords, soft}; bad: [{geom, body, source, pos_err_m, rot_err_deg}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "Isaac Sim 5.0 (pip isaacsim) in its own Python for the load test"],
    "simulators": ["Isaac Sim", "MuJoCo"],
    "tags": ["compile", "isaac sim", "usd", "export", "deformable"],
    "status": "tested",
    "entrypoint": "isaac_export_check",
    "cli": "python -m tool_library.verifiers.isaac_export_check stage.usda scene.xml",
}

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import compile_checks as C

# standalone: runs inside Isaac Sim's Python (no project imports)
_ISAAC_SCRIPT = r'''
import json, math, os
USD = os.environ["USD_PATH"]; STEPS = int(os.environ.get("PHYS_STEPS", "0"))
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.usd
from pxr import Usd, UsdGeom
out = {"opened": False}
try:
    ok = omni.usd.get_context().open_stage(USD)
    stage = omni.usd.get_context().get_stage()
    out["opened"] = bool(ok) and stage is not None and stage.GetPseudoRoot().IsValid()
    if out["opened"]:
        xc = UsdGeom.XformCache()
        statics, soft, cords, n = {}, {}, {}, 0
        for prim in stage.Traverse():
            n += 1
            parts = str(prim.GetPath()).split("/")
            if len(parts) == 4 and parts[1] == "World" and parts[2] in ("static", "soft") and prim.IsA(UsdGeom.Xformable):
                M = xc.GetLocalToWorldTransform(prim)
                t = M.ExtractTranslation(); q = M.ExtractRotationQuat()
                im = q.GetImaginary()
                (statics if parts[2] == "static" else soft)[parts[3]] = {"pos": [t[0], t[1], t[2]], "quat": [q.GetReal(), im[0], im[1], im[2]]}
            if len(parts) == 5 and parts[1] == "World" and parts[2] == "cords" and parts[4].startswith("link_"):
                cords[parts[3]] = cords.get(parts[3], 0) + 1
        out.update(n_prims=n, statics=statics, soft=soft, cords=cords)
        if STEPS > 0:
            from isaacsim.core.api import World
            world = World(physics_dt=1.0 / 480.0, stage_units_in_meters=1.0)
            world.reset()
            for _ in range(STEPS):
                world.step(render=False)
            xc = UsdGeom.XformCache()
            finite = True
            for prim in stage.Traverse():
                parts = str(prim.GetPath()).split("/")
                if len(parts) >= 4 and parts[1] == "World" and parts[2] in ("soft", "cords") and prim.IsA(UsdGeom.Xformable):
                    t = xc.GetLocalToWorldTransform(prim).ExtractTranslation()
                    if not all(math.isfinite(float(v)) for v in t):
                        finite = False
            out["physics_finite"] = finite
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("ISAAC_CHECK_JSON " + json.dumps(out), flush=True)
app.close()
'''


def _rot_err(q1, R2):
    R1 = np.zeros(9)
    mujoco.mju_quat2Mat(R1, np.asarray(q1, float) / max(np.linalg.norm(q1), 1e-12))
    R = R1.reshape(3, 3).T @ R2
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _scene_pose(model, data, rec):
    nm = rec.get("name", "")
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
    if g < 0 and nm.startswith("g") and nm[1:].isdigit() and int(nm[1:]) < model.ngeom:
        g = int(nm[1:])                            # build_isaac_scene names unnamed geoms g<index>
    if g < 0:
        return None
    return data.geom_xpos[g].copy(), data.geom_xmat[g].reshape(3, 3).copy(), C.body_name(model, int(model.geom_bodyid[g]))


def _compare(model, data, records, poses, source, pos_tol, rot_tol, robot_prefix):
    bad, n, worst, missing = [], 0, None, []
    for k, rec in enumerate(records):
        sp = _scene_pose(model, data, rec)
        if sp is None:
            missing.append(rec.get("name"))
            continue
        pose = poses(k, rec)
        if pose is None:
            continue
        n += 1
        pe = float(np.linalg.norm(np.asarray(pose["pos"], float) - sp[0]))
        re_ = _rot_err(pose["quat"], sp[1])
        row = {"geom": rec.get("name"), "body": sp[2], "source": source, "pos_err_m": pe, "rot_err_deg": re_,
               "robot": C.is_robot(sp[2], robot_prefix)}
        if worst is None or pe > worst["pos_err_m"]:
            worst = row
        if pe > pos_tol or re_ > rot_tol:
            bad.append(row)
    return bad, n, worst, missing


def isaac_export_check(usd, scene, export_json=None, qpos=None, isaac_python=None, isaac_home=None, physics_steps=0, pos_tol_m=0.002,
                       rot_tol_deg=1.0, timeout_s=900.0, robot_prefix=C.ROBOT_PREFIX):
    usd = Path(usd)
    if not usd.exists():
        raise ValueError(f"usd not found: {usd}")
    C.require_scene(scene)
    model, err = C.compile_scene(scene)
    if model is None:
        return C.compile_failure("the MuJoCo scene must compile to be compared", err)
    data = C.key_data(model)
    if qpos is not None:
        q = np.load(qpos) if isinstance(qpos, (str, Path)) else np.asarray(qpos, float)
        if len(q) > model.nq:
            raise ValueError(f"qpos has {len(q)} entries, the scene has nq={model.nq}")
        data.qpos[:len(q)] = q
        mujoco.mj_forward(model, data)
    ej = Path(export_json) if export_json else usd.parent / "export.json"
    issues, notes = [], []
    records, deformables, softs = [], [], []
    if ej.exists():
        exp = json.loads(ej.read_text())
        records = list(exp.get("geoms", []))
        deformables = [d.get("name") for d in exp.get("deformables") or []]
        softs = list(exp.get("soft") or [])
    else:
        notes.append(f"no export.json at {ej}: the stage cannot be matched geom by geom; only the load test runs")
    all_bad = []
    exp_rep = None
    if records:
        bad, n, worst, missing = _compare(model, data, records, lambda k, r: r, "export.json", pos_tol_m, rot_tol_deg, robot_prefix)
        exp_rep = {"n": n, "n_bad": len(bad), "worst": worst, "missing_in_scene": missing[:20]}
        all_bad += bad
        obj_bad = [b for b in bad if not b["robot"]]
        if obj_bad:
            w = max(obj_bad, key=lambda b: b["pos_err_m"])
            issues.append(f"the export no longer matches the scene: {len(obj_bad)} object geoms differ (worst '{w['geom']}' of '{w['body']}' "
                          f"{1000 * w['pos_err_m']:.1f} mm / {w['rot_err_deg']:.1f} deg): the scene was re-placed after the Isaac export or "
                          "the export used another state; re-run build_isaac_scene on this scene")
        if missing:
            issues.append(f"{len(missing)} exported geoms do not exist in the scene ({missing[:4]}): the stage was exported from another scene")
        if [b for b in bad if b["robot"]]:
            notes.append(f"{len([b for b in bad if b['robot']])} robot geoms differ (the robot pose depends on the exported state)")
    py = isaac_python or os.environ.get("DEXAGENT_ISAAC_PYTHON")
    usd_rep, skipped = None, False
    if not py or not Path(py).exists():
        skipped = True
        notes.append("Isaac Sim is not configured (isaac_python / DEXAGENT_ISAAC_PYTHON): the stage load test was skipped")
    else:
        with tempfile.TemporaryDirectory() as td:
            # Isaac writes gigabytes of cache under HOME: a configured home is reused, else a throw-away one
            home = Path(isaac_home or os.environ.get("DEXAGENT_ISAAC_HOME") or (Path(td) / "isaac_home"))
            home.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, HOME=str(home), OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PYTHONUNBUFFERED="1",
                       USD_PATH=str(usd.resolve()), PHYS_STEPS=str(int(physics_steps)))
            script = Path(td) / "isaac_export_probe.py"
            script.write_text(_ISAAC_SCRIPT)
            env.setdefault("TMPDIR", td)
            rc = None
            try:
                r = subprocess.run([str(py), str(script)], env=env, capture_output=True, text=True, timeout=float(timeout_s))
                text = (r.stdout or "") + (r.stderr or "")
                rc = r.returncode
            except subprocess.TimeoutExpired:
                text = ""
                issues.append(f"Isaac Sim did not finish opening the stage within {timeout_s} s")
        tail = " | ".join(text.strip().splitlines()[-3:])[:300] or "no output"
        line = next((l for l in text.splitlines()[::-1] if l.startswith("ISAAC_CHECK_JSON ")), None)
        if rc not in (None, 0):
            issues.append(f"Isaac Sim exited with code {rc} (log tail: {tail}): the load test did not complete; check the Isaac install")
        if line is None:
            if rc == 0:                              # a timeout or a non-zero exit is already an issue
                issues.append(f"Isaac Sim produced no result (log tail: {tail})")
        else:
            res = json.loads(line[len("ISAAC_CHECK_JSON "):])
            usd_rep = {"opened": res.get("opened"), "n_prims": res.get("n_prims"), "cords": res.get("cords"),
                       "n_statics": len(res.get("statics") or {}), "n_soft": len(res.get("soft") or {}),
                       "physics_finite": res.get("physics_finite"), "error": res.get("error")}
            if not res.get("opened"):
                issues.append(f"Isaac Sim could not open the stage {usd.name} ({res.get('error') or 'open_stage returned False'}): re-export it")
            else:
                statics = res.get("statics") or {}
                index = {}
                for key, pose in statics.items():
                    if key.startswith("g") and key[1:].isdigit():
                        index[int(key[1:])] = pose
                if records:
                    bad, n, worst, _ = _compare(model, data, records, lambda k, r: index.get(k), "usd", pos_tol_m, rot_tol_deg, robot_prefix)
                    missing_st = [records[k].get("name") for k in range(len(records)) if k not in index]
                    usd_rep.update({"n": n, "n_bad": len(bad), "worst": worst, "missing_statics": missing_st[:20]})
                    all_bad += bad
                    obj_bad = [b for b in bad if not b["robot"]]
                    if obj_bad:
                        w = max(obj_bad, key=lambda b: b["pos_err_m"])
                        issues.append(f"in Isaac Sim {len(obj_bad)} object geoms are not where the MuJoCo scene has them (worst '{w['geom']}' "
                                      f"{1000 * w['pos_err_m']:.1f} mm / {w['rot_err_deg']:.1f} deg): the cord will be simulated against "
                                      "the wrong scene; re-export")
                    if missing_st:
                        issues.append(f"{len(missing_st)} exported geoms have no /World/static prim in the stage ({missing_st[:4]})")
                for name in deformables:
                    if (res.get("cords") or {}).get(name, 0) < 2:
                        issues.append(f"deformable '{name}' from the export has no capsule chain under /World/cords in the stage")
                if softs and len(res.get("soft") or {}) < len(softs):
                    issues.append(f"{len(softs)} soft bodies were exported but the stage has {len(res.get('soft') or {})} under /World/soft")
                if physics_steps and res.get("physics_finite") is False:
                    issues.append(f"after {physics_steps} PhysX steps a dynamic prim (cord link or soft body) is not finite: the stage "
                                  "explodes in Isaac Sim")
    all_bad.sort(key=lambda b: -b["pos_err_m"])
    return C.jsonable({"ok": not issues, "issues": issues, "notes": notes, "skipped_isaac": skipped, "export_vs_scene": exp_rep,
                       "usd_vs_scene": usd_rep, "bad": all_bad[:30]})


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rep = isaac_export_check(argv[0], argv[1])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
