"""Deformables simulated in Isaac Sim, Stage 3 side: the cord specification and its MuJoCo mirror (isaac_cord), the
grasp schedule read from labels and the mirror of a recording (execute_isaac_deformables), the planning copy
(build_cord_planning_scene), and isaac_recording_check on a packaged workspace (a sound recording passes; a tampered
recording, a flying link and a missed grasp fail). With Isaac Sim's Python (DEXAGENT_ISAAC_PYTHON) a short episode is
simulated for real: a rope on the bench, a hand that closes on it and one that closes on air.

    python tool_library/tests/test_isaac_deformables.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import isaac_cord as IC  # noqa: E402
from tool_library.common.evidence import file_digest, value_digest  # noqa: E402

SETUP = ROOT / "setup" / "scene.xml"
TMP = Path(tempfile.mkdtemp(prefix="test_isaac_deformables_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def rope_points():
    xs = np.linspace(-0.2, 0.2, 40)
    return np.stack([xs, 0.45 + 0.08 * np.sin(xs * 8), np.full_like(xs, 0.012)], 1)


def test_spec_and_mirror():
    s = IC.cord_spec({"name": "rope", "points": rope_points().tolist(), "radius": 0.012, "preset": "segmented_rope"})
    t = IC.cord_spec({"name": "thread", "points": rope_points().tolist(), "radius": 0.001})
    check("cord_spec: presets resolve every coefficient (thin_cord keeps build_isaac_scene's model, segmented_rope the fitted rope)",
          s["physics"]["self_collision"] is True and t["physics"]["linear_damping"] == 3.0 and s["n_links"] == round(s["length_m"] / 0.027353)
          and abs(s["segment_m"] * s["n_links"] - s["length_m"]) < 1e-9)
    try:
        IC.cord_spec({"name": "x", "points": rope_points().tolist(), "radius": 0.01, "physics": {"stifness": 1}})
        check("cord_spec refuses an unknown coefficient", False)
    except ValueError:
        check("cord_spec refuses an unknown coefficient", True)
    info = IC.add_mirrors(SETUP, [s], TMP / "mirror" / "scene_mirror.xml")
    m = mujoco.MjModel.from_xml_path(info["path"]); d = mujoco.MjData(m)
    a, b = info["slices"]["rope"]
    C, Q = IC.lay_poses(s["points"], s["n_links"], s["radius"])
    d.qpos[a:b] = IC.mirror_qpos(C, Q, s["segment_m"]); mujoco.mj_kinematics(m, d)
    err = float(np.linalg.norm(IC.mirror_centres(m, d, s) - C, axis=1).max())
    check("mirror: construct_rope-named links whose qpos reproduce the Isaac link poses, its meshes next to it",
          IC.cable_qpos_slice(m, s) == (a, b) and err < 0.002 and all((TMP / "mirror" / r).is_file() for r in info["assets"]),
          f"(max centre error {err * 1000:.2f} mm, {len(info['assets'])} assets)")
    return s


def fake_recording(spec, T, out, speed=0.0, hands=("left", "right")):
    """An Isaac recording of a rope sliding 1 cm along x over T frames (speed adds a one-frame jump at T // 2)."""
    C0, Q0 = IC.lay_poses(spec["points"], spec["n_links"], spec["radius"])
    C = np.stack([C0 + [0.01 * t / max(T - 1, 1), 0, 0] for t in range(T)]).astype(np.float32)
    if speed:
        C[T // 2:] += np.array([speed / 30.0, 0, 0], np.float32)
    Qr = np.repeat(Q0[None], T, 0).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "isaac_recording.npz", fps=np.float64(30.0), substeps=np.int64(16), anchor=np.zeros((T, 2, 3), np.float32),
                        held=-np.ones((T, 2), np.int16), strain=np.zeros((T, 2), np.float32),
                        link_speed_max=np.zeros(T, np.float32), sides=np.array(hands), cords=np.array([spec["name"]]),
                        **{f"cord_{spec['name']}_centres": C, f"cord_{spec['name']}_quats": Qr})
    return dict(np.load(out / "isaac_recording.npz", allow_pickle=True))


def packaged_workspace(spec, tag, speed=0.0, grasp_state="released"):
    """A Stage 3 workspace as package_stage3_episode leaves it for an Isaac cord (built without Isaac Sim)."""
    from tool_library.skills.execute_isaac_deformables import build_mirror
    ws = TMP / tag
    ws.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(str(SETUP)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    T = 90
    IC.add_mirrors(SETUP, [], ws / "scene_executed.xml")          # the bench scene with its meshes beside it
    np.savez(ws / "trajectory.npz", qpos=np.tile(d.qpos, (T, 1)), fps=30.0, labels=np.array(["left: rest #1"] * T), subgoal=np.ones(T, int))
    rec = fake_recording(spec, T, ws / "isaac", speed=speed)
    info, mt, _ = build_mirror(ws / "scene_executed.xml", ws / "trajectory.npz", [spec], rec, ws)
    grasps = [{"side": "right", "cord": "rope", "weld_frame": 5, "release_frame": 20}]
    dspec = {"cords": [spec], "grasps": grasps, "events": [], "max_link_speed": 1.5, "substeps": 16}
    (ws / "deformables.json").write_text(json.dumps(dspec))
    lib = ROOT / "tool_library" / "common"
    rep = {"ok": True, "issues": [], "physx_invalid_transforms": 0, "physx_errors": 0,
           "grasps": [dict(grasps[0], state=grasp_state, link=3, released_at=20)],
           "mirror": {"nq_scene": int(m.nq), "slices": info["slices"]},
           "binding": {"scene_sha256": file_digest(ws / "scene_executed.xml"), "trajectory_sha256": file_digest(ws / "trajectory.npz"),
                       "recording_sha256": file_digest(ws / "isaac" / "isaac_recording.npz"), "mirror_scene_sha256": file_digest(info["path"]),
                       "mirror_trajectory_sha256": file_digest(mt), "cords_sha256": value_digest([spec]), "events_sha256": value_digest([]),
                       "scripts_sha256": {n: file_digest(lib / n) for n in ("isaac_episode_script.py", "isaac_cord.py")}}}
    (ws / "isaac_episode_report.json").write_text(json.dumps(rep))
    (ws / "trajectory_manifest.json").write_text(json.dumps({
        "scene_executed": "scene_executed.xml", "episode": {"trajectory": "trajectory.npz"},
        "deformables": {"report": "isaac_episode_report.json", "spec": "deformables.json", "recording": "isaac/isaac_recording.npz",
                        "scene_mirror": "scene_mirror.xml", "trajectory_mirror": "trajectory_mirror.npz"}}))
    return ws


def test_labels_and_mirror(spec):
    from tool_library.skills.execute_isaac_deformables import grasps_from_labels
    labels = ["left: rest #1"] * 5 + ["right: grip rope #1 | left: rest #1"] * 4 + ["right: lay rope #1"] * 6 + ["right: retreat #1"] * 3
    g = grasps_from_labels(labels, {"rope": {"rope"}})
    check("grasps from labels: one run of a hand naming the cord with a holding verb is one grasp",
          g == [{"side": "right", "cord": "rope", "weld_frame": 5, "release_frame": 15}], json.dumps(g))
    ws = packaged_workspace(spec, "mirror_append")
    zt, zm = np.load(ws / "trajectory.npz"), np.load(ws / "trajectory_mirror.npz")
    m = mujoco.MjModel.from_xml_path(str(ws / "scene_mirror.xml"))
    check("mirror trajectory: the executed columns unchanged, the cord's appended, as wide as the mirror scene",
          zm["qpos"].shape[1] == m.nq and np.allclose(zm["qpos"][:, :zt["qpos"].shape[1]], zt["qpos"]))


def test_planning_copy(spec):
    from tool_library.skills.build_cord_planning_scene import build_cord_planning_scene
    from tool_library.skills.execute_isaac_deformables import build_mirror
    s2 = TMP / "stage2"
    (s2 / "isaac").mkdir(parents=True, exist_ok=True)
    IC.add_mirrors(SETUP, [], s2 / "scene.xml")
    C, Q = IC.lay_poses(spec["points"], spec["n_links"], spec["radius"])
    IC.save_spec(dict(spec, link_poses={"centres": C.tolist(), "quats": Q.tolist()}), s2 / "isaac" / "isaac_rope_cord.json")
    (s2 / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
        {"id": "rope", "type": "deformable", "simulator": "isaac", "isaac": {"cord_json": "isaac/isaac_rope_cord.json"}}]}))
    r = build_cord_planning_scene(s2 / "scene_recon.json", TMP / "plan" / "scene_cords.xml")
    m = mujoco.MjModel.from_xml_path(r["scene"]); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    for _ in range(200):
        mujoco.mj_step(m, d)
    stable = bool(np.isfinite(d.qpos).all())
    rec = fake_recording(spec, 10, TMP / "plan" / "isaac")
    info, mt, _ = build_mirror(r["scene"], _traj(m, TMP / "plan"), [spec], rec, TMP / "plan" / "out")
    zm = np.load(mt)
    m2 = mujoco.MjModel.from_xml_path(info["path"]); d2 = mujoco.MjData(m2); d2.qpos[:] = zm["qpos"][-1]; mujoco.mj_kinematics(m2, d2)
    err = float(np.linalg.norm(IC.mirror_centres(m2, d2, spec) - rec["cord_rope_centres"][-1], axis=1).max())
    check("planning copy: a steppable chain of the cord's links in the scene; its columns are overwritten by the Isaac recording "
          "in the mirror (no second cable)", stable and m2.nq == m.nq and info["planning_copies"] == ["rope"] and err < 0.002,
          f"(mirror error {err * 1000:.2f} mm)")


def _traj(m, out):
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    np.savez(out / "traj.npz", qpos=np.tile(d.qpos, (10, 1)), fps=30.0, labels=np.array(["left: rest #1"] * 10))
    return out / "traj.npz"


def test_recording_check(spec):
    from tool_library.verifiers.isaac_recording_check import isaac_recording_check
    ok = isaac_recording_check(packaged_workspace(spec, "rc_ok"))
    check("isaac_recording_check passes a sound packaged recording", ok["ok"], json.dumps(ok.get("issues"))[:300])
    ws = packaged_workspace(spec, "rc_tampered")
    z = dict(np.load(ws / "isaac" / "isaac_recording.npz", allow_pickle=True)); z["cord_rope_centres"] = z["cord_rope_centres"] + 0.001
    np.savez_compressed(ws / "isaac" / "isaac_recording.npz", **z)
    bad = isaac_recording_check(ws)
    check("isaac_recording_check fails a recording edited after the run (binding)", not bad["ok"] and not bad["binding"]["recording_sha256"])
    fly = isaac_recording_check(packaged_workspace(spec, "rc_fly", speed=3.0))
    check("isaac_recording_check fails a cord that jumps (link speed)", not fly["ok"] and any("m/s" in i for i in fly["issues"]))
    miss = isaac_recording_check(packaged_workspace(spec, "rc_miss", grasp_state="missed"))
    check("isaac_recording_check fails a grasp that held nothing", not miss["ok"] and any("held nothing" in i for i in miss["issues"]))


def test_isaac_episode(spec):
    """A real Isaac Sim episode: the right hand's pinch is put on the rope, the left hand's is not."""
    py = os.environ.get("DEXAGENT_ISAAC_PYTHON")
    if not py or not Path(py).exists():
        print("[SKIP] the Isaac Sim episode (set DEXAGENT_ISAAC_PYTHON to Isaac Sim's Python)")
        return
    from tool_library.skills.execute_isaac_deformables import execute_isaac_deformables
    m = mujoco.MjModel.from_xml_path(str(SETUP)); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    pin = 0.5 * (d.site_xpos[m.site("robot_right_right_thumb_tip").id] + d.site_xpos[m.site("robot_right_right_index_tip").id])
    pts = np.stack([np.linspace(pin[0] - 0.3, pin[0], 30), np.full(30, pin[1]), np.full(30, 0.012)], 1)
    rope = IC.cord_spec({"name": "rope", "points": pts.tolist(), "radius": 0.012, "preset": "segmented_rope"})
    T = 45
    np.savez(TMP / "episode_traj.npz", qpos=np.tile(d.qpos, (T, 1)), fps=30.0, labels=np.array(["left: rest #1"] * T))
    r = execute_isaac_deformables(str(SETUP), str(TMP / "episode_traj.npz"), out_dir=str(TMP / "episode"), cords=[rope], render=False,
                                  grasps=[{"side": "right", "cord": "rope", "weld_frame": 5, "release_frame": 30, "capture_m": 0.15},
                                          {"side": "left", "cord": "rope", "weld_frame": 8, "release_frame": 20}])
    states = {g["side"]: g["state"] for g in r["grasps"]}
    zm = np.load(r["trajectory_mirror"])
    check("Isaac Sim episode: the hand on the rope holds a link, the hand on air holds nothing (reported, run not ok), the "
          "mirror and deformables.json are written", states == {"right": "released", "left": "missed"} and not r["ok"]
          and zm["qpos"].shape[0] == T and (TMP / "episode" / "deformables.json").is_file(), json.dumps(r["grasps"])[:300])


if __name__ == "__main__":
    spec = None
    for fn in (test_spec_and_mirror, test_labels_and_mirror, test_planning_copy, test_recording_check, test_isaac_episode):
        try:
            got = fn() if fn is test_spec_and_mirror else fn(spec)
            spec = got if fn is test_spec_and_mirror else spec
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(fn.__name__, False, f"crashed: {type(e).__name__}: {e}")
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)
