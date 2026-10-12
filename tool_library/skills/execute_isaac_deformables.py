"""Deformable objects of a Stage 3 episode simulated in Isaac Sim: the robot and rigid objects as MuJoCo executed them,
the cords in PhysX; writes the Isaac recording and its MuJoCo mirror for the gates."""
from __future__ import annotations

TOOL = {
    "name": "execute_isaac_deformables",
    "title": "Execute deformables in Isaac Sim",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Runs the cords (ropes, cables, strings) of an executed Stage 3 episode in Isaac Sim / PhysX, driven by the "
               "robot motion MuJoCo executed: hands hold cords with compliant welds, rigid objects are kinematic colliders; "
               "writes the Isaac recording, its MuJoCo mirror (scene + trajectory) for the gates, and the episode video.",
    "description": (
        "DexAgent simulates deformable objects in Isaac Sim and everything else in MuJoCo. After execute_subgoal_plan "
        "has executed the robot (and the rigid objects) in MuJoCo, this tool replays that executed trajectory in Isaac "
        "Sim with every deformable of the Stage 2 scene simulated as the cord Stage 2 verified (scene_recon.json "
        "objects with simulator 'isaac' and their isaac.cord_json: geometry, PhysX coefficients and the settled link "
        "poses). Per video frame the MuJoCo row is put through forward kinematics: the robot and objects are posed, "
        "rigid objects are kinematic colliders, each hand drives a kinematic anchor, all interpolated over the physics "
        "substeps. GRASPS: `grasps` [{side, cord, weld_frame, release_frame, link (optional), stiff, slip_m, "
        "capture_m}], else read from the trajectory's labels: a run of frames in which a hand's label names the cord "
        "with a holding verb (grip, lift, carry, hold, lay, drag, pull, tuck, thread, cross, fold, place, move) is one "
        "grasp. At weld_frame the anchor is welded to the link nearest the hand's thumb-index pinch point (or `link` "
        "when it is within reach) if it is within capture_m (default 3 cm); else nothing is held and the grasp is "
        "reported missed. A weld lets go at release_frame, or earlier when the link is pulled further than slip_m from "
        "where the hand holds it, fading out over the cord's fade_frames. `events` [{frame, cord, bend_stiffness}] "
        "retune a cord during the episode. Outputs: isaac/isaac_recording.npz (every link's pose per frame, welds, "
        "strains, peak link speeds), isaac/isaac_events.json, deformables.json (the cords, grasps and events: what a replay needs), isaac_episode_report.json (grasps held / missed / "
        "slipped, peak and over-limit link speeds, the lowest link against the table, PhysX errors and invalid "
        "transforms, and the sha256 binding of scene, trajectory, cord specifications and scripts that "
        "isaac_recording_check re-checks); scene_mirror.xml (the executed scene plus each cord as a construct_rope-"
        "named MuJoCo cable that MuJoCo never steps) and trajectory_mirror.npz (the executed qpos plus the cords' "
        "recorded link poses, labels and segments), on which the gates that read object states run unchanged; "
        "sim_video.mp4 when render is on (the registered camera, Isaac RTX). Fails (ok false, issues) on PhysX "
        "errors, a non-finite pose, a missed grasp, a link below the table by more than 3 mm or a peak link speed "
        "above max_link_speed."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "path", "description": "The MuJoCo scene the episode was executed in (scene_executed.xml)."},
        {"name": "trajectory", "type": "path", "description": "The executed trajectory.npz (qpos [T, nq], fps, labels, subgoal)."},
        {"name": "scene_recon_json", "type": "path | None", "description": "Stage 2 scene_recon.json: the deformables with simulator 'isaac' and their cord_json."},
        {"name": "cords", "type": "list[dict] | path | None", "description": "Cord specifications instead of the Stage 2 ones (a deformables.json, or its cords): a replay of a moved layout."},
        {"name": "out_dir", "type": "path", "description": "Receives isaac/, isaac_episode_report.json, scene_mirror.xml, trajectory_mirror.npz, sim_video.mp4."},
        {"name": "grasps", "type": "list[dict] | path | None", "description": "Explicit grasp schedule; default read from the labels."},
        {"name": "events", "type": "list[dict] | None", "description": "[{frame, cord, bend_stiffness}] cord retunes during the episode."},
        {"name": "render", "type": "bool", "description": "Render the episode through the camera into sim_video.mp4. Default True."},
        {"name": "camera", "type": "path | dict | None", "description": "camera.json (OpenCV axes, rig frame); default the one next to scene_recon.json."},
        {"name": "hands", "type": "dict | None", "description": "{side: {anchor_body, pinch_sites}}; default the setup's hands (robot_<s>_<s>_hand_C_MC, thumb and index tip sites)."},
        {"name": "robot_collider_bodies", "type": "list[str] | None", "description": "Robot bodies the cords collide with (kinematic convex hulls). Default none: fingers would fight their own welds."},
        {"name": "max_link_speed", "type": "float", "description": "Peak link speed (m/s) above which the run fails. Default 1.5."},
        {"name": "substeps", "type": "int", "description": "Physics substeps per video frame. Default 16 (1/480 s at 30 fps)."},
        {"name": "frames", "type": "int | None", "description": "Simulate only the first n frames (a quick check). Default all."},
        {"name": "isaac_python", "type": "path | None", "description": "Isaac Sim's Python; default DEXAGENT_ISAAC_PYTHON (dexagent.yaml environments.isaac)."},
        {"name": "isaac_home", "type": "path | None", "description": "HOME for Isaac's caches; default DEXAGENT_ISAAC_HOME or out_dir/isaac/home."},
        {"name": "timeout_s", "type": "float", "description": "Default 7200."},
    ],
    "outputs": [
        {"name": "ok / issues", "type": "bool / list", "description": "Whether the Isaac episode is trustworthy."},
        {"name": "report", "type": "path", "description": "isaac_episode_report.json."},
        {"name": "recording / scene_mirror / trajectory_mirror / sim_video", "type": "path", "description": "Written files."},
        {"name": "grasps", "type": "list[dict]", "description": "Each grasp's outcome: held, missed, released, slipped; the link held."},
    ],
    "status": "tested",
    "entrypoint": "execute_isaac_deformables",
    "requirements": ["mujoco>=3.2", "numpy", "Isaac Sim 5.0 (pip isaacsim) with mujoco in its own Python"],
    "simulators": ["Isaac Sim", "MuJoCo"],
    "tags": ["isaac sim", "deformable", "rope", "cable", "physx", "stage 3", "execution"],
    "used_in": ["Knot the Rope"],
}

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from tool_library.common import isaac_cord as IC

COMMON = Path(__file__).resolve().parents[1] / "common"
SCRIPT = COMMON / "isaac_episode_script.py"
HOLD_VERBS = ("grip", "lift", "carry", "hold", "lay", "drag", "pull", "tuck", "thread", "cross", "fold", "place", "move", "tighten", "cinch")
TABLE_PEN_M = 0.003
DEFORMABLES_SCHEMA = "dexagent.isaac_deformables.v1"


def isaac_deformables(scene_recon_json):
    """[(object entry, cord spec)] of the Stage 2 objects simulated in Isaac Sim."""
    rp = Path(scene_recon_json)
    recon = json.loads(rp.read_text())
    out = []
    for o in recon.get("objects", []):
        if o.get("type") != "deformable" or str(o.get("simulator", "")).lower() != "isaac":
            continue
        cj = (o.get("isaac") or {}).get("cord_json")
        if not cj:
            raise ValueError(f"deformable {o.get('id')!r} is simulated in Isaac Sim but its entry has no isaac.cord_json")
        out.append((o, IC.load_spec(rp.parent / cj)))
    return out


def default_hands(scene_xml):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    hands = {}
    for s in ("left", "right"):
        body, tips = f"robot_{s}_{s}_hand_C_MC", [f"robot_{s}_{s}_thumb_tip", f"robot_{s}_{s}_index_tip"]
        try:
            m.body(body)
            [m.site(t) for t in tips]
        except KeyError:
            continue
        hands[s] = {"anchor_body": body, "pinch_sites": tips}
    if not hands:
        raise ValueError("no hands found: pass hands={side: {anchor_body, pinch_sites}}")
    return hands


def grasps_from_labels(labels, cords, sides=("left", "right")):
    """One grasp per run of frames in which a hand's label names a cord with a holding verb."""
    from tool_library.common import episode_audit as EA
    out = []
    for s in sides:
        cur = None
        for t, lab in enumerate(list(labels) + [""]):
            txt = EA.phase_text(lab, s) if lab else ""
            hit = next((n for n, aliases in cords.items() if any(a.lower() in txt for a in aliases)), None) \
                if any(v in txt for v in HOLD_VERBS) else None
            if cur is not None and hit != cur[0]:
                out.append({"side": s, "cord": cur[0], "weld_frame": cur[1], "release_frame": t})
                cur = None
            if cur is None and hit is not None:
                cur = (hit, t)
    return sorted(out, key=lambda g: g["weld_frame"])


def build_mirror(scene_xml, trajectory, specs, recording, out_dir):
    """scene_mirror.xml + trajectory_mirror.npz: the executed scene and qpos with each cord's recorded link poses. A cord
    whose planning copy is in the executed scene (build_cord_planning_scene) has that copy's columns overwritten with the
    Isaac recording; any other cord is appended as a mirror cable."""
    import mujoco
    out = Path(out_dir)
    m0 = mujoco.MjModel.from_xml_path(str(scene_xml))
    inside = {s["name"]: IC.cable_qpos_slice(m0, s) for s in specs}
    add = [s for s in specs if inside[s["name"]] is None]
    initial = {s["name"]: {"centres": recording[f"cord_{s['name']}_centres"][0], "quats": recording[f"cord_{s['name']}_quats"][0]} for s in add}
    info = IC.add_mirrors(scene_xml, add, out / "scene_mirror.xml", initial=initial)
    info["slices"].update({n: list(sl) for n, sl in inside.items() if sl is not None})
    z = np.load(trajectory, allow_pickle=True)
    Q = np.asarray(z["qpos"], float)
    T = min(len(Q), len(recording[f"cord_{specs[0]['name']}_centres"]))
    Qm = np.concatenate([Q[:T], np.zeros((T, sum(IC.mirror_nq(s["n_links"]) for s in add)))], axis=1)
    for s in specs:
        C, Qc = recording[f"cord_{s['name']}_centres"][:T].astype(float), recording[f"cord_{s['name']}_quats"][:T].astype(float)
        a_, b_ = info["slices"][s["name"]]
        Qm[:, a_:b_] = np.stack([IC.mirror_qpos(C[t], Qc[t], s["segment_m"]) for t in range(T)])
    keep = {k: z[k][:T] for k in ("labels", "subgoal", "time_s") if k in z.files}
    np.savez_compressed(out / "trajectory_mirror.npz", qpos=Qm, fps=np.float64(float(z["fps"]) if "fps" in z.files else 30.0),
                        mirror_of=np.array(str(Path(trajectory).name)), **keep)
    info["planning_copies"] = sorted(n for n, sl in inside.items() if sl is not None)
    return info, str(out / "trajectory_mirror.npz"), T


def execute_isaac_deformables(scene_xml, trajectory, scene_recon_json=None, out_dir=".", grasps=None, events=None, render=True, camera=None,
                              hands=None, robot_collider_bodies=None, max_link_speed=1.5, substeps=16, frames=None, cords=None,
                              isaac_python=None, isaac_home=None, timeout_s=7200.0):
    from tool_library.common.evidence import file_digest, value_digest
    from tool_library.skills.build_isaac_scene import physx_log_issues
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    iso = out / "isaac"; iso.mkdir(exist_ok=True)
    if cords is not None:                         # given specifications (Stage 4 replays a moved layout of the same cords)
        cords = json.loads(Path(cords).read_text())["cords"] if isinstance(cords, (str, Path)) else cords
        pairs = [({"id": c["name"]}, IC.cord_spec(c) if c.get("schema") != IC.SCHEMA else c) for c in cords]
    else:
        if scene_recon_json is None:
            raise ValueError("give scene_recon_json (the Stage 2 deformables) or cords")
        pairs = isaac_deformables(scene_recon_json)
    if not pairs:
        raise ValueError("the Stage 2 scene has no deformable simulated in Isaac Sim (objects with simulator 'isaac')")
    specs = [s for _, s in pairs]
    z = np.load(trajectory, allow_pickle=True)
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    labels = [str(x) for x in z["labels"]] if "labels" in z.files else []
    aliases = {s["name"]: {s["name"], str(o.get("id") or s["name"])} for o, s in pairs}
    if isinstance(grasps, (str, Path)):
        grasps = json.loads(Path(grasps).read_text())
    grasps = [dict(g) for g in (grasps if grasps is not None else grasps_from_labels(labels, aliases))]
    for g in grasps:
        if g.get("cord") not in aliases:
            hit = [n for n, a in aliases.items() if g.get("cord") in a]
            if not hit:
                raise ValueError(f"grasp {g}: cord {g.get('cord')!r} is not an Isaac deformable ({sorted(aliases)})")
            g["cord"] = hit[0]
    hands = hands or default_hands(scene_xml)
    cam = None
    if render:
        cp = camera or (Path(scene_recon_json).parent / "camera.json")
        cam = json.loads(Path(cp).read_text()) if isinstance(cp, (str, Path)) else dict(cp)
    job = {"scene_xml": str(Path(scene_xml).resolve()), "trajectory": str(Path(trajectory).resolve()), "fps": fps, "substeps": int(substeps),
           "frames": frames, "hands": hands, "cords": specs, "grasps": grasps, "events": events or [], "out_dir": str(iso.resolve()),
           "robot_collider_bodies": list(robot_collider_bodies or []),
           "render": ({"camera": cam, "width": int(cam["width"]), "height": int(cam["height"])} if cam else None)}
    jp = iso / "job.json"; jp.write_text(json.dumps(job, indent=1, default=float))
    py = isaac_python or os.environ.get("DEXAGENT_ISAAC_PYTHON")
    if not py or not Path(py).exists():
        raise FileNotFoundError("Isaac Sim's Python not found: pass isaac_python or set DEXAGENT_ISAAC_PYTHON (dexagent.yaml environments.isaac)")
    home = Path(isaac_home or os.environ.get("DEXAGENT_ISAAC_HOME") or (iso / "home")); home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home), OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PYTHONUNBUFFERED="1", JOB=str(jp.resolve()))
    env.setdefault("TMPDIR", str(iso / "tmp")); Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    log = iso / "isaac_episode.log"
    with log.open("w") as fh:
        r = subprocess.run([str(py), str(SCRIPT)], env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=float(timeout_s))
    text = log.read_text()
    if r.returncode != 0 or "ISAAC_EPISODE_DONE" not in text:
        raise RuntimeError(f"Isaac Sim episode failed (returncode {r.returncode}); log tail:\n{text[-3000:]}")
    rec = dict(np.load(iso / "isaac_recording.npz", allow_pickle=True))
    ev = json.loads((iso / "isaac_events.json").read_text())
    issues = []
    inv, err = physx_log_issues(text)
    if inv or err:
        issues.append(f"PhysX logged {inv} invalid transforms and {err} errors: the cord articulation blew up")
    speeds = np.asarray(rec["link_speed_max"], float)
    over = int((speeds > float(max_link_speed)).sum())
    if over:
        issues.append(f"links moved faster than {max_link_speed} m/s on {over} frames (peak {speeds.max():.2f} m/s, first at frame "
                      f"{int(np.argmax(speeds > float(max_link_speed)))}): a cord flew or was snapped")
    import mujoco
    mm = mujoco.MjModel.from_xml_path(str(scene_xml))
    from tool_library.common import episode_audit as EA
    table_top = (EA.table_info(mm) or {}).get("top_z", 0.0)
    lowest = {}
    for s in specs:
        C = np.asarray(rec[f"cord_{s['name']}_centres"], float)
        if not np.isfinite(C).all():
            issues.append(f"{s['name']}: a link pose is not finite (the simulation diverged)")
            continue
        low = float((C[..., 2] - s["radius"]).min() - table_top)
        lowest[s["name"]] = round(low, 4)
        if low < -TABLE_PEN_M:
            issues.append(f"{s['name']}: a link sank {-low * 1000:.1f} mm into the table")
    for g in ev["grasps"]:
        if g["state"] in ("missed", "planned"):
            issues.append(f"{g['side']} hand {'closed on nothing' if g['state'] == 'missed' else 'never closed'} at frame {g['weld_frame']} "
                          f"(grasp of {g['cord']}): move the pinch onto the cord")
    info, mirror_traj, T = build_mirror(scene_xml, trajectory, specs, rec, out)
    video = None
    if render and (iso / "frames").is_dir():
        import imageio.v2 as imageio
        frames_ = sorted((iso / "frames").glob("f*.png"))
        if frames_:
            video = out / "sim_video.mp4"
            with imageio.get_writer(str(video), fps=fps, codec="libx264", quality=8, macro_block_size=8) as wr:
                for f in frames_:
                    wr.append_data(imageio.imread(f))
    scripts = {p.name: file_digest(p) for p in (SCRIPT, COMMON / "isaac_cord.py")}
    report = {"ok": not issues, "issues": issues, "frames": T, "fps": fps,
              "peak_link_speed": round(float(speeds.max()), 4) if len(speeds) else 0.0, "frames_over_speed": over,
              "lowest_link_above_table_m": lowest, "physx_invalid_transforms": inv, "physx_errors": err,
              "grasps": ev["grasps"], "events": ev["events"],
              "binding": {"scene_sha256": file_digest(scene_xml), "trajectory_sha256": file_digest(trajectory),
                          "cords_sha256": value_digest(specs), "grasps_sha256": value_digest(grasps), "events_sha256": value_digest(events or []),
                          "scripts_sha256": scripts, "recording_sha256": file_digest(iso / "isaac_recording.npz"),
                          "mirror_scene_sha256": file_digest(info["path"]), "mirror_trajectory_sha256": file_digest(mirror_traj)},
              "files": {"recording": "isaac/isaac_recording.npz", "events": "isaac/isaac_events.json", "job": "isaac/job.json",
                        "log": "isaac/isaac_episode.log", "scene_mirror": "scene_mirror.xml", "trajectory_mirror": "trajectory_mirror.npz",
                        "sim_video": "sim_video.mp4" if video else None},
              "mirror": {"nq_scene": info["nq_scene"], "slices": info["slices"], "assets": info.get("assets", []),
                         "planning_copies": info.get("planning_copies", [])},
              "cords": {s["name"]: {"n_links": s["n_links"], "segment_m": s["segment_m"], "radius": s["radius"], "preset": s["preset"]} for s in specs}}
    rp = out / "isaac_episode_report.json"
    rp.write_text(json.dumps(report, indent=1, default=float))
    # everything a replay of this episode needs besides the robot trajectory (Stage 4 moves the cords and runs it again)
    (out / "deformables.json").write_text(json.dumps({
        "schema": DEFORMABLES_SCHEMA, "cords": specs, "grasps": [{k: g[k] for k in g if k not in ("state", "released_at", "weld", "off")} for g in grasps],
        "events": events or [], "hands": hands, "robot_collider_bodies": list(robot_collider_bodies or []), "substeps": int(substeps),
        "max_link_speed": float(max_link_speed), "recording": "isaac/isaac_recording.npz", "report": "isaac_episode_report.json",
        "scene_mirror": "scene_mirror.xml", "trajectory_mirror": "trajectory_mirror.npz"}, indent=1, default=float))
    return {"ok": report["ok"], "issues": issues, "report": str(rp), "recording": str(iso / "isaac_recording.npz"),
            "scene_mirror": info["path"], "trajectory_mirror": mirror_traj, "sim_video": str(video) if video else None,
            "grasps": [{k: g.get(k) for k in ("side", "cord", "weld_frame", "release_frame", "state", "link", "released_at")} for g in ev["grasps"]],
            "peak_link_speed": report["peak_link_speed"]}
