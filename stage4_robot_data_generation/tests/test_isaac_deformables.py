"""Stage 4 on a task with a deformable object simulated in Isaac Sim (stage4/deformables.py).

Unit checks (no Isaac): a layout's planar rigid transform of a cord's initial link poses, the retiming of the grasp
schedule, and the Episode of a stack task with a cord: the cord is drawn by its name, and a hand anchored to it moves
with it.

End to end (needs Isaac Sim's Python in DEXAGENT_ISAAC_PYTHON; skipped without it): the
rigid bench task of synthetic.py (the robot holds its home pose, a cube on a box) with a short cord on the table that
the right hand holds by a weld; Stage 3's Isaac episode is written with execute_isaac_deformables, then the randomize
step draws one layout that moves the box and the cord: the robot replays in MuJoCo, the cord in Isaac Sim, the
Stage 1 goals (the cube on the box, the cord on the table) are judged on the mirror, both samples publish their cord
files, the export renders from the mirror (the cord's capsules among the geoms), and the hand-off check and the
physics replay gate accept the folder.

    cd stage4_robot_data_generation && python tests/test_isaac_deformables.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

import synthetic  # noqa: E402
from stage4 import deformables as DF  # noqa: E402
from stage4 import kinematics as K  # noqa: E402
from tool_library.common import isaac_cord as IC  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="stage4_cords_", dir=os.environ.get("TMPDIR")))
FAILS = 0
ISAAC = os.environ.get("DEXAGENT_ISAAC_PYTHON") or ""


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def cord(points, n_links=10, name="rope"):
    s = IC.cord_spec({"name": name, "points": points, "radius": 0.01, "preset": "segmented_rope", "n_links": n_links,
                      "rgb": [0.7, 0.1, 0.1]})
    C, Q = IC.lay_poses(s["points"], s["n_links"], s["radius"])
    s["link_poses"] = {"centres": C.tolist(), "quats": Q.tolist()}
    return s


def test_transform():
    s = cord([[0.0, 0.4, 0.01], [0.25, 0.4, 0.01]])
    c0 = DF.cord_center(s)
    R = K.rot_z(np.radians(20.0))
    t = c0 + np.array([0.03, -0.02, 0.0]) - R @ c0
    m = DF.moved_spec(s, (R, t))
    C0, C1 = np.asarray(s["link_poses"]["centres"]), np.asarray(m["link_poses"]["centres"])
    x0 = np.array([IC.quat_to_mat(q)[:, 0] for q in s["link_poses"]["quats"]])
    x1 = np.array([IC.quat_to_mat(q)[:, 0] for q in m["link_poses"]["quats"]])
    check("a cord's layout transform moves its link centres rigidly and turns its links with them",
          np.allclose(C1, C0 @ R.T + t) and np.allclose(np.linalg.norm(np.diff(C1, axis=0), axis=1), np.linalg.norm(np.diff(C0, axis=0), axis=1))
          and np.allclose(x1, x0 @ R.T, atol=1e-9) and np.allclose(DF.cord_center(m), c0 + [0.03, -0.02, 0.0]),
          f"(centre moved {np.round(DF.cord_center(m) - c0, 4)})")
    src = np.concatenate([np.arange(0, 10, 0.5), np.arange(10, 20)])          # frames 0-9 slowed down twice
    g = DF.retimed([{"weld_frame": 4, "release_frame": 12}], src, ("weld_frame", "release_frame"))[0]
    check("a retimed episode moves the grasp schedule with its frames", g["weld_frame"] == 8 and g["release_frame"] == 22, str(g))


def fake_handoff(task, spec, T):
    """deformables.json + a still recording + its mirror, without Isaac (for the Episode unit check)."""
    (task / "isaac").mkdir(parents=True, exist_ok=True)
    C, Q = np.asarray(spec["link_poses"]["centres"]), np.asarray(spec["link_poses"]["quats"])
    np.savez_compressed(task / "isaac" / "isaac_recording.npz", **{"cord_rope_centres": np.repeat(C[None], T, 0),
                                                                   "cord_rope_quats": np.repeat(Q[None], T, 0)})
    (task / "isaac_episode_report.json").write_text(json.dumps({"ok": True, "binding": {}}))
    (task / "scene_mirror.xml").write_text((task / "scene.xml").read_text())
    np.savez_compressed(task / "trajectory_mirror.npz", qpos=np.load(task / "trajectory.npz")["qpos"])
    (task / "deformables.json").write_text(json.dumps({"schema": DF.SCHEMA, "cords": [spec], "grasps": [], "events": [],
                                                       "recording": "isaac/isaac_recording.npz", "report": "isaac_episode_report.json",
                                                       "scene_mirror": "scene_mirror.xml", "trajectory_mirror": "trajectory_mirror.npz"}))


def test_episode_with_cord():
    from stage4 import randomize
    from stage4.config import load_config
    task = TMP / "stack_cord"
    cfg_p, labels, _ = synthetic.stack_task(task)
    cfg = json.loads(cfg_p.read_text())
    s = cord([[0.30, 0.20, 0.01], [0.55, 0.20, 0.01]])
    fake_handoff(task, s, len(labels))
    cfg["deformables"] = "deformables.json"
    cfg["randomize"]["objects"]["rope"] = {"dx": [0.02, 0.03], "dy": [0.0, 0.0]}
    # the hand follows the cord while it reaches for A (a hand anchored to a cord moves with the cord's draw)
    cfg["segments"]["right"][0] = {"phases": ["right: reach for A #1"], "anchor": "rope"}
    cfg["segments"]["right"][1] = {"phases": ["right: descend on A #1", "right: grip A #1", "right: lift A #1"], "blend": ["rope", "A"]}
    cfg_p.write_text(json.dumps(cfg))
    c = load_config(cfg_p)
    ep = randomize.Episode(c)
    o = ep.objects["rope"]
    lay = ep.sample_layout(np.random.default_rng(0))
    pl = ep.plan(lay)
    R, t = pl["D"]["rope"]
    dx = lay["objects"]["rope"][0]
    t0 = 0
    P = ep.ref[("ee", "right")][0][t0]
    moved = R @ P + t - P
    check("a cord is a randomized object of the Episode (no body in the rigid scene) and a hand anchored to it moves "
          "with its draw", o.get("cord") and o["bid"] is None and np.allclose(pl["D"]["rope"][0], np.eye(3))
          and abs(moved[0] - dx) < 1e-9 and np.allclose(np.asarray(pl["side"]["right"][t0][1]) + pl["side"]["right"][t0][0] @ P - P, moved),
          f"(draw dx {dx:.4f}, hand moved {np.round(moved, 4)})")
    Q, ov, rep = ep.adapt(lay, pl)
    check("the warp of a layout that moves a cord passes the kinematic checks and leaves the rigid scene's bodies alone",
          rep["ok"] and ov == {} and Q.shape == ep.Q.shape, str(rep.get("failed")))


def isaac_task():
    """The rigid bench task with a cord on the table held by the right hand (Stage 3's Isaac episode included)."""
    from tool_library.skills.execute_isaac_deformables import execute_isaac_deformables
    task = TMP / "bench_cord"
    cfg_p = synthetic.rigid_bench_task(task, frames=45)
    s = cord([[0.05, 0.30, 0.0], [0.30, 0.30, 0.0]], n_links=9)
    grasps = [{"side": "right", "cord": "rope", "weld_frame": 10, "release_frame": 30, "capture_m": 0.5}]
    r = execute_isaac_deformables(str(task / "scene.xml"), str(task / "trajectory.npz"), out_dir=str(task), cords=[s], grasps=grasps,
                                  render=False, isaac_python=ISAAC, isaac_home=DF.isaac_home())
    cfg = json.loads(cfg_p.read_text())
    cfg["deformables"] = "deformables.json"
    cfg["num_samples"], cfg["max_attempts"] = 1, 4
    cfg["randomize"]["objects"]["rope"] = {"dx": [0.02, 0.03], "dy": [0.0, 0.0], "dyaw_deg": [-5.0, 5.0]}
    cfg["isaac"] = {"python": ISAAC}
    sem = json.loads((task / "scene_semantics.json").read_text())
    sem["objects"].append({"id": "rope", "name": "red rope", "type": "deformable"})
    sem["subgoals"].append({"index": 2, "acting_hand": "right", "motion": "other", "object_refs": ["rope"],
                            "goal": "The rope rests on the table."})
    (task / "scene_semantics.json").write_text(json.dumps(sem))
    go = json.loads((task / "goal_objects.json").read_text())
    go["objects"].append({"id": "rope", "root_body": "ropeB_first", "bodies": IC.body_names("rope", s["n_links"])})
    (task / "goal_objects.json").write_text(json.dumps(go))
    man = json.loads((task / "trajectory_manifest.json").read_text())
    man["subgoals"][0]["stage1_indices"] = [1, 2]
    (task / "trajectory_manifest.json").write_text(json.dumps(man))
    cfg_p.write_text(json.dumps(cfg, indent=1))
    return task, cfg_p, r


def test_end_to_end():
    if not ISAAC or not Path(ISAAC).exists():
        print("[SKIP] Isaac Sim's Python not found (DEXAGENT_ISAAC_PYTHON): the end-to-end cord replay was not run")
        return
    from stage4 import randomize, export
    from stage4.config import load_config
    task, cfg_p, r0 = isaac_task()
    check("Stage 3's Isaac episode of the cord passes and writes the hand-off files", r0["ok"] and (task / "deformables.json").is_file(),
          json.dumps({k: r0.get(k) for k in ("ok", "issues", "grasps", "peak_link_speed")}, default=str)[:300])
    from tool_library.verifiers.stage4_handoff_check import stage4_handoff_check
    h = stage4_handoff_check(str(task))
    check("the hand-off check accepts the folder with its deformables", h["ok"] and "deformables" in h["summary"], str(h["issues"])[:300])
    cfg = load_config(cfg_p)
    summary = randomize.run(cfg, log=lambda *_: None)
    samples = cfg.out / "samples"
    infos = {d.name: json.loads((d / "sample.json").read_text()) for d in sorted(samples.iterdir())}
    row = next(r for r in summary["rows"] if r["ok"])
    check("randomize keeps a layout whose cord replays in Isaac Sim after the robot's MuJoCo replay",
          summary["accepted"] == 1 and set(infos) == {"original", "0000"}
          and all(i.get("deformables") and not DF.verify_sample(samples / n, i) for n, i in infos.items()),
          json.dumps({"rows": [{k: r.get(k) for k in ("ok", "failed", "physics_failed", "layout")} for r in summary["rows"]]}, default=str)[:600])
    checks = infos["0000"]["checks"]["physics"]
    check("the Isaac episode and the Stage 1 goals (judged on the mirror, the cord's included) hold on the layout",
          checks["ok"] and "deformables: Isaac Sim episode" in json.dumps(checks), json.dumps(checks)[:400])
    m0, Q0, _ = DF.mirror({"scene_mirror": samples / "original" / "scene_mirror.xml",
                           "trajectory_mirror": samples / "original" / "trajectory_mirror.npz"})
    m1, Q1, _ = DF.mirror({"scene_mirror": samples / "0000" / "scene_mirror.xml",
                           "trajectory_mirror": samples / "0000" / "trajectory_mirror.npz"})
    import mujoco
    spec = json.loads((task / "deformables.json").read_text())["cords"][0]
    cen = []
    for m, Q in ((m0, Q0), (m1, Q1)):
        d = mujoco.MjData(m)
        d.qpos[:] = Q[0]
        mujoco.mj_kinematics(m, d)
        cen.append(IC.mirror_centres(m, d, spec).mean(axis=0))
    shift = cen[1] - cen[0]
    dx = row["layout"]["rope"][0]
    check("the sample's cord starts where its layout moved it (mirror at frame 0)", abs(shift[0] - dx) < 0.01 and abs(shift[1]) < 0.01,
          f"(drawn dx {dx:.4f}, mirror centre moved {np.round(shift, 4)}; original from the {infos['original']['deformables']['source']})")
    import cv2
    (cfg.out / "plate").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cfg.out / "plate" / "plate.png"), np.full((480, 640, 3), 120, np.uint8))
    export.run(cfg, log=lambda *_: None)
    meta = json.loads((cfg.out / "render" / "0000" / "ego" / "scene.json").read_text())
    names = [g.get("name", "") for g in meta["geoms"]]
    T = np.load(meta["transforms"])["T"]
    check("the export renders a deformable sample from its mirror: the cord's capsules are among the geoms",
          sum(n.startswith("ropeG") for n in names) == spec["n_links"] and T.shape[1] == len(names), f"({len(names)} geoms)")
    from tool_library.verifiers.stage4_physics_replay_gate import stage4_physics_replay_gate
    g = stage4_physics_replay_gate(str(task), draws=3, layouts=1)
    check("the physics replay gate replays the original and a layout with the cord in Isaac Sim", g["ok"],
          json.dumps({k: g.get(k) for k in ("issues", "layouts_held")}, default=str)[:400])


if __name__ == "__main__":
    for fn in (test_transform, test_episode_with_cord, test_end_to_end):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(fn.__name__, False, f"crashed: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)
