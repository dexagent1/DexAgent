"""The Isaac Sim recording of a Stage 3 episode's deformables, re-checked from its files (the Isaac counterpart of
physics_recording_check, episode_physics_check and end_state_rest_audit for a cord).

    from tool_library.verifiers.isaac_recording_check import isaac_recording_check
    r = isaac_recording_check("runs/x/stage3")          # reads trajectory_manifest.json's deformables block
"""
from __future__ import annotations

TOOL = {
    "name": "isaac_recording_check",
    "title": "Isaac recording check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Re-checks an episode's Isaac Sim cord recording from its files: bound to the executed scene, trajectory, cord "
               "specifications and scripts by sha256; no PhysX errors; finite, untorn, above the table, under the speed limit; "
               "every grasp held; at rest at the end; its MuJoCo mirror reproduces it; optionally re-simulated in Isaac Sim.",
    "description": (
        "Deformables run in Isaac Sim while the robot and rigid objects run in MuJoCo (execute_isaac_deformables). The "
        "MuJoCo gates read the cord through its mirror (a construct_rope-named cable whose qpos is the recording); this "
        "check is what the MuJoCo physics checks are for a rigid object. It reads the workspace's trajectory_manifest.json "
        "`deformables` block (or the given report) and verifies: (1) BINDING: isaac_episode_report.json's sha256 of the "
        "executed scene and trajectory equal the manifest's files, the recording, mirror scene and mirror trajectory "
        "equal theirs, the cord specifications, grasps and events equal deformables.json's, and the Isaac scripts are the "
        "library's current ones; (2) PHYSX: the log shows no PhysX error and no invalid transform; (3) the RECORDING itself, "
        "re-measured: every link pose finite; neighbouring link centres no further apart than (1 + max_stretch) link "
        "lengths (a torn chain); no link more than table_pen_m below the table top; the peak link speed at most "
        "max_link_speed (a cord that flies or is snapped); every planned grasp held a link (none missed, none never "
        "closed); over the last rest_s seconds, after the last release, the cord moves at most rest_drift_m (it lies "
        "still where it was left); (4) MIRROR: the mirror trajectory's robot and rigid columns equal the executed "
        "trajectory's, and its cord rows put every link centre within mirror_tol_m of the recording (forward kinematics "
        "of scene_mirror.xml); (5) REPLAY (replay=True): the episode is simulated again in Isaac Sim from deformables.json "
        "on the same scene and trajectory, and the Stage 1 goal gate must pass on the replay's mirror as on the "
        "recording's (PhysX CPU contacts are not bit-reproducible, so the outcome is compared, not the bits; the final "
        "centreline deviation is reported). Returns ok, issues and the measurements."
    ),
    "inputs": [
        {"name": "workspace", "type": "str", "description": "Stage 3 workspace with trajectory_manifest.json (its deformables block)."},
        {"name": "scene_semantic_json", "type": "str | None", "description": "Stage 1 scene_semantics.json (for the replay's goal gate)."},
        {"name": "scene_recon_json", "type": "str | None", "description": "Stage 2 scene_recon.json (for the replay's goal gate)."},
        {"name": "max_link_speed", "type": "float", "description": "Default: deformables.json's max_link_speed (1.5 m/s)."},
        {"name": "max_stretch", "type": "float", "description": "Allowed neighbour gap beyond one link length, as a share of it. Default 0.25."},
        {"name": "table_pen_m", "type": "float", "description": "Allowed link depth below the table top. Default 0.003."},
        {"name": "rest_s", "type": "float", "description": "Seconds at the end over which the cord must lie still. Default 1.0."},
        {"name": "rest_drift_m", "type": "float", "description": "Largest link displacement allowed over rest_s. Default 0.01."},
        {"name": "mirror_tol_m", "type": "float", "description": "Mirror link centre error allowed. Default 0.003."},
        {"name": "replay", "type": "bool", "description": "Simulate the episode again in Isaac Sim and compare the goal outcome. Default False."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, binding {name: ok}, physx, cords {name: finite, max_stretch, lowest_m, peak_speed, end_drift_m}, grasps, mirror_error_m, replay {ok, goal_ok, final_deviation_mean_m}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "Isaac Sim 5.0 for replay"],
    "simulators": ["Isaac Sim", "MuJoCo"],
    "tags": ["stage3", "isaac sim", "deformable", "rope", "recording", "binding", "replay"],
    "status": "tested",
    "entrypoint": "isaac_recording_check",
}

import json
from pathlib import Path

import numpy as np


def _rel(ws, p):
    p = Path(p)
    return p if p.is_absolute() else ws / p


def isaac_recording_check(workspace, scene_semantic_json=None, scene_recon_json=None, max_link_speed=None, max_stretch=0.25,
                          table_pen_m=0.003, rest_s=1.0, rest_drift_m=0.01, mirror_tol_m=0.003, replay=False):
    import mujoco
    from tool_library.common import isaac_cord as IC
    from tool_library.common.evidence import file_digest, value_digest
    ws = Path(workspace)
    man = json.loads((ws / "trajectory_manifest.json").read_text())
    blk = man.get("deformables")
    if not isinstance(blk, dict):
        return {"ok": False, "issues": ["the manifest has no deformables block: run execute_isaac_deformables and package its output"]}
    issues, out = [], {"binding": {}, "cords": {}}
    rep_p = _rel(ws, blk.get("report", "isaac_episode_report.json"))
    spec_p = _rel(ws, blk.get("spec", "deformables.json"))
    for p in (rep_p, spec_p):
        if not p.is_file():
            return {"ok": False, "issues": [f"{p.name} is missing"]}
    rep, dspec = json.loads(rep_p.read_text()), json.loads(spec_p.read_text())
    files = {"recording": _rel(ws, blk.get("recording", dspec.get("recording", "isaac/isaac_recording.npz"))),
             "scene_mirror": _rel(ws, blk.get("scene_mirror", "scene_mirror.xml")),
             "trajectory_mirror": _rel(ws, blk.get("trajectory_mirror", "trajectory_mirror.npz"))}
    scene = _rel(ws, man.get("scene_executed") or "scene_executed.xml")
    traj = _rel(ws, (man.get("episode") or {}).get("trajectory") or "trajectory.npz")
    for k, p in [*files.items(), ("scene_executed", scene), ("trajectory", traj)]:
        if not p.is_file():
            issues.append(f"{k} file missing: {p}")
    if issues:
        return {"ok": False, "issues": issues}
    b = rep.get("binding") or {}
    lib = Path(__file__).resolve().parents[1] / "common"
    want = {"scene_sha256": file_digest(scene), "trajectory_sha256": file_digest(traj), "recording_sha256": file_digest(files["recording"]),
            "mirror_scene_sha256": file_digest(files["scene_mirror"]), "mirror_trajectory_sha256": file_digest(files["trajectory_mirror"]),
            "cords_sha256": value_digest(dspec.get("cords")), "events_sha256": value_digest(dspec.get("events") or [])}
    for k, v in want.items():
        out["binding"][k] = b.get(k) == v
        if b.get(k) != v:
            issues.append(f"binding: {k} of the Isaac report does not match the packaged file (the recording is not of this episode)")
    for name, digest in (b.get("scripts_sha256") or {}).items():
        ok = (lib / name).is_file() and file_digest(lib / name) == digest
        out["binding"][f"script:{name}"] = ok
        if not ok:
            issues.append(f"binding: the recording was made with another version of {name}")
    if not b.get("scripts_sha256"):
        issues.append("binding: the Isaac report names no script digests")
    out["physx"] = {"invalid_transforms": rep.get("physx_invalid_transforms"), "errors": rep.get("physx_errors")}
    if rep.get("physx_invalid_transforms") or rep.get("physx_errors"):
        issues.append(f"PhysX: {rep.get('physx_invalid_transforms')} invalid transforms, {rep.get('physx_errors')} errors in the Isaac log")
    rec = np.load(files["recording"], allow_pickle=True)
    fps = float(rec["fps"])
    vmax = float(max_link_speed if max_link_speed is not None else dspec.get("max_link_speed", 1.5))
    m0 = mujoco.MjModel.from_xml_path(str(scene))
    from tool_library.common import episode_audit as EA
    top = (EA.table_info(m0) or {}).get("top_z", 0.0)
    releases = [int(g.get("released_at") or g.get("release_frame") or 0) for g in rep.get("grasps") or []]
    for spec in dspec["cords"]:
        n = spec["name"]
        C = np.asarray(rec[f"cord_{n}_centres"], float)
        r = {"finite": bool(np.isfinite(C).all())}
        if not r["finite"]:
            issues.append(f"{n}: a link pose is not finite"); out["cords"][n] = r; continue
        gaps = np.linalg.norm(np.diff(C, axis=1), axis=2) / float(spec["segment_m"]) - 1.0
        r["max_stretch"] = round(float(gaps.max()), 4)
        if gaps.max() > float(max_stretch):
            t = int(np.unravel_index(np.argmax(gaps), gaps.shape)[0])
            issues.append(f"{n}: neighbouring links {100 * gaps.max():.0f}% further apart than a link length at frame {t} (a torn chain)")
        low = (C[..., 2] - float(spec["radius"])) - top
        r["lowest_m"] = round(float(low.min()), 4)
        if low.min() < -float(table_pen_m):
            issues.append(f"{n}: a link {-low.min() * 1000:.1f} mm below the table top at frame {int(np.unravel_index(np.argmin(low), low.shape)[0])}")
        sp = np.linalg.norm(np.diff(C, axis=0), axis=2).max(axis=1) * fps if len(C) > 1 else np.zeros(0)
        r["peak_speed"] = round(float(sp.max()) if sp.size else 0.0, 3)
        if sp.size and sp.max() > vmax:
            issues.append(f"{n}: a link moved at {sp.max():.2f} m/s (frame {int(np.argmax(sp)) + 1}; limit {vmax} m/s): the cord flew or snapped")
        k = max(1, int(round(float(rest_s) * fps)))
        if len(C) > k and (not releases or max(releases) < len(C) - k):
            drift = float(np.linalg.norm(C[-1] - C[-1 - k], axis=1).max())
            r["end_drift_m"] = round(drift, 4)
            if drift > float(rest_drift_m):
                issues.append(f"{n}: still moving {drift * 1000:.0f} mm over the last {rest_s} s (not at rest)")
        elif len(C) > k:
            issues.append(f"{n}: a hand lets go of it within the last {rest_s} s, so the episode does not show it at rest")
        out["cords"][n] = r
    out["grasps"] = [{k: g.get(k) for k in ("side", "cord", "weld_frame", "state", "link")} for g in rep.get("grasps") or []]
    for g in rep.get("grasps") or []:
        if g.get("state") not in ("held", "released", "slipped"):
            issues.append(f"grasp {g.get('side')} at frame {g.get('weld_frame')}: {g.get('state')} (the hand held nothing)")
    # the mirror reproduces the recording
    mm = mujoco.MjModel.from_xml_path(str(files["scene_mirror"])); dm = mujoco.MjData(mm)
    zm, zt = np.load(files["trajectory_mirror"], allow_pickle=True), np.load(traj, allow_pickle=True)
    Qm, Qt = np.asarray(zm["qpos"], float), np.asarray(zt["qpos"], float)
    nq0 = (rep.get("mirror") or {}).get("nq_scene", Qt.shape[1])
    if Qm.shape[1] != mm.nq or nq0 != Qt.shape[1]:
        issues.append(f"mirror: trajectory has {Qm.shape[1]} columns for a mirror scene of nq={mm.nq}")
    else:
        T = min(len(Qm), len(Qt))
        if not np.allclose(Qm[:T, :nq0], Qt[:T], atol=1e-9):
            issues.append("mirror: its robot and rigid columns differ from the executed trajectory")
        err = 0.0
        for t in np.unique(np.linspace(0, T - 1, min(T, 60)).astype(int)):
            dm.qpos[:] = Qm[t]; mujoco.mj_kinematics(mm, dm)
            for spec in dspec["cords"]:
                Cr = np.asarray(rec[f"cord_{spec['name']}_centres"][t], float)
                err = max(err, float(np.linalg.norm(IC.mirror_centres(mm, dm, spec) - Cr, axis=1).max()))
        out["mirror_error_m"] = round(err, 5)
        if err > float(mirror_tol_m):
            issues.append(f"mirror: its links are up to {err * 1000:.1f} mm from the recording's")
    if replay:
        out["replay"] = _replay(ws, scene, traj, dspec, rec, scene_semantic_json or man.get("scene_semantic"),
                                scene_recon_json or man.get("scene_recon"), issues)
    out.update(ok=not issues, issues=issues)
    return out


def _replay(ws, scene, traj, dspec, rec, sem, recon, issues):
    """The episode simulated again from deformables.json; its goal outcome against the recording's."""
    import shutil
    import tempfile
    from tool_library.skills.execute_isaac_deformables import execute_isaac_deformables
    from tool_library.verifiers.stage1_goal_gate import stage1_goal_gate
    tmp = Path(tempfile.mkdtemp(prefix="isaac_replay_", dir=str(ws)))
    try:
        r = execute_isaac_deformables(str(scene), str(traj), out_dir=str(tmp), cords=dspec["cords"], grasps=dspec.get("grasps"),
                                      events=dspec.get("events"), hands=dspec.get("hands"), robot_collider_bodies=dspec.get("robot_collider_bodies"),
                                      max_link_speed=float(dspec.get("max_link_speed", 1.5)), substeps=int(dspec.get("substeps", 16)), render=False)
        res = {"ok": bool(r.get("ok")), "issues": r.get("issues", [])[:4]}
        z = np.load(r["recording"], allow_pickle=True)
        dev = [float(np.linalg.norm(np.asarray(z[f"cord_{s['name']}_centres"][-1], float) - np.asarray(rec[f"cord_{s['name']}_centres"][-1], float), axis=1).mean())
               for s in dspec["cords"]]
        res["final_deviation_mean_m"] = round(max(dev), 4)
        if sem and recon:
            man = json.loads((ws / "trajectory_manifest.json").read_text())
            mp = tmp / "trajectory_manifest.json"
            mp.write_text(json.dumps(man))
            g = stage1_goal_gate(str(sem), str(mp), r["scene_mirror"], r["trajectory_mirror"], scene_recon_json=str(recon))
            res["goal_ok"] = bool(g.get("ok"))
            res["goal_issues"] = [str(i)[:200] for i in (g.get("issues") or [])][:4]
            if not g.get("ok"):
                issues.append(f"replay: simulated again, the episode no longer meets its Stage 1 goals ({res['goal_issues'][:2]})")
        if not r.get("ok"):
            issues.append(f"replay: the Isaac re-simulation failed its own checks ({res['issues'][:2]})")
        return res
    except Exception as exc:
        issues.append(f"replay: {type(exc).__name__}: {exc}")
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
