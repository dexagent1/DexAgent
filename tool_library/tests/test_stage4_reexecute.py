"""Stage 4's re-execution of the Stage 3 plan, as this release ships it.

Stage 4 accepts only rigid arm-hand runtime recordings (the hands fixed to the arm flanges, with the complete servo
recording), and re-executing the Stage 3 plan on a drawn layout (stage4/reexecute.py) is unavailable for them: a layout
the warp rejects stays rejected. This test pins that contract and the part of the re-execution code that remains a
library function:
  1. a pose moved with its object (reexecute.move_pose) re-anchors in the executor exactly as the original pose
     re-anchored, moved (every re-anchor mode): the bench (setup/scene.xml) with a box, an index-fingertip push pose;
  2. a rigid runtime task folder that hands off plan.json: Stage 4 turns re-execution off and says why, and the
     randomize step publishes warped layouts only (no 'reexecuted' sample);
  3. a task folder without a plan: warp only, with the way to regenerate the hand-off.
CPU only, about a minute.

    MUJOCO_GL=egl python tool_library/tests/test_stage4_reexecute.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage4_robot_data_generation" / "tests"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import synthetic  # noqa: E402
from tool_library.common import hand as H  # noqa: E402
from tool_library.common import stage4 as S4  # noqa: E402
from tool_library.common.subgoal import reanchor_pose  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.push_pose_from_point import push_pose_from_point  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="wp6_reexec_", dir=os.environ.get("TMPDIR")))
FAILS = 0
BOX_XY, HALF = (0.125, 0.40), (0.03, 0.03, 0.025)


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    FAILS += 0 if ok else 1


def make_stage2():
    """A Stage 2 workspace: the bench with a 6 x 6 x 5 cm box at (0.125, 0.40)."""
    ws = TMP / "stage2"
    (ws / "assets" / "box").mkdir(parents=True, exist_ok=True)
    (ws / "assets" / "box" / "box.xml").write_text(
        f'<mujoco model="box"><worldbody><body name="box" pos="0 0 {HALF[2]}"><freejoint name="box"/>'
        f'<geom name="box_geom" type="box" size="{HALF[0]} {HALF[1]} {HALF[2]}" mass="0.15" rgba="0.3 0.5 0.8 1"/>'
        '</body></worldbody></mujoco>')
    stage_on_bench(str(ROOT / "setup" / "scene.xml"),
                   assets=[{"name": "box", "xml": str(ws / "assets/box/box.xml"), "x": BOX_XY[0], "y": BOX_XY[1]}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    model = mujoco.MjModel.from_xml_path(str(ws / "scene.xml"))
    box = next(model.body(b).name for b in range(model.nbody) if model.body(b).name.startswith("box_") and model.body_parentid[b] == 0)
    cam = {"width": 640, "height": 480, "fovy_deg": 38.0, "pos": [0.0188, -0.1829, 0.5578], "quat_wxyz": [0.881567, 0.470897, 0.0249467, 0.021763]}
    (ws / "camera.json").write_text(json.dumps(cam))
    (ws / "scene_recon.json").write_text(json.dumps({
        "scene_xml": "scene.xml", "camera_json": "camera.json",
        "objects": [{"id": "box", "type": "rigid", "asset_xml": "assets/box/box.xml", "root_body": box, "bodies": [box]}]}))
    return ws, box


def body_pose(model, data, name):
    b = model.body(name).id
    return data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()


def rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def main():
    RX = S4.stage4_module("reexecute")
    config, randomize = S4.stage4_module("config"), S4.stage4_module("randomize")

    # 1. the moved pose re-anchors in the executor as the original did, moved (every mode)
    ws, box = make_stage2()
    scene = str(ws / "scene.xml")
    hs = hand_scene_from_recon(str(ws / "scene_recon.json"), "right", "box", out_json=str(TMP / "hs_right_box.json"))
    pose = push_pose_from_point(hs, [BOX_XY[0], BOX_XY[1] - HALF[1], 0.03], [0, -1, 0], finger="index")
    model = mujoco.MjModel.from_xml_path(scene)
    d0, d1 = mujoco.MjData(model), mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d0, 0)
    mujoco.mj_resetDataKeyframe(model, d1, 0)
    adr = model.jnt_qposadr[model.body(box).jntadr[0]]
    mujoco.mj_forward(model, d0)
    p0, R0 = body_pose(model, d0, box)
    anchored = dict(pose, anchor={"body": box, "pos": p0.tolist(), "quat_wxyz": RX.K.mat2quat(R0).tolist()})
    p_now, R_now = p0 + np.array([0.012, -0.004, 0.0]), rz(6.0) @ R0           # where the box is when the pose is used
    D = (rz(17.0), np.array([0.03, -0.02, 0.0]))
    errs = []
    for mode in ("full", "yaw", "translate"):
        d0.qpos[adr:adr + 3], d0.qpos[adr + 3:adr + 7] = p_now, RX.K.mat2quat(R_now)
        d1.qpos[adr:adr + 3], d1.qpos[adr + 3:adr + 7] = D[0] @ p_now + D[1], RX.K.mat2quat(D[0] @ R_now)
        a = reanchor_pose(model, d0, anchored, anchored["anchor"], mode)
        mv = RX.move_pose(D, anchored)
        b = reanchor_pose(model, d1, mv, mv["anchor"], mode)
        errs.append(max(np.linalg.norm(np.asarray(b["pos"]) - (D[0] @ np.asarray(a["pos"]) + D[1])),
                        np.linalg.norm(H.euler_to_mat(b["euler"]) - D[0] @ H.euler_to_mat(a["euler"]))))
    check("a pose moved with its object re-anchors like the original, moved (full / yaw / translate)", max(errs) < 1e-9, str(errs))

    # 2. a rigid runtime task folder that hands off a plan: re-execution is off, with the reason; randomize keeps warps only
    task = TMP / "rigid_task"
    synthetic.rigid_bench_task(task, attached=True)
    (task / "plan.json").write_text(json.dumps({"subgoals": [{"index": 1, "motion": "hold", "side": "left", "object": "cube",
                                                              "stage3_frames": [0, 59]}]}))
    cfg_j = json.loads((task / "stage4.json").read_text())
    cfg_j.update(num_samples=1, max_attempts=2, reexecute={"plan": "plan.json"})
    (task / "stage4.json").write_text(json.dumps(cfg_j, indent=1))
    cfg = config.load_config(task / "stage4.json")
    ep = randomize.Episode(cfg)
    rx, why = RX.make(ep, cfg, log=lambda *_: None)
    check("a rigid runtime scene: re-execution is unavailable even with plan.json, and says so",
          rx is None and "rigid runtime" in (why or "") and "unavailable" in why, str(why)[:120])
    logs = []
    summary = randomize.run(cfg, log=logs.append)
    rows = summary["rows"]
    check("randomize reports re-execution off and keeps warped layouts only",
          summary["accepted"] == 1 and summary["reexecution"] is False and set(summary["methods"]) <= {"warp", "warp+retime"}
          and any("re-execution of the Stage 3 plan: off (plan re-execution on rigid runtime" in m for m in logs)
          and all(r.get("method") != "reexecuted" for r in rows) and (cfg.out / "samples" / "0000" / "physics_replay.npz").exists(),
          f"{summary['accepted']}/{summary['attempts']} methods {summary['methods']}")

    # 3. without a plan (a proposal-scene diagnostic task), Stage 4 keeps the warp only and says how to get one
    cfg2_p, _, _ = synthetic.stack_task(TMP / "stack")
    cfg2 = config.load_config(cfg2_p)
    rz_ = randomize.Randomizer(randomize.Episode(cfg2), cfg2, log=lambda *_: None, physics_replay=False)
    check("no plan handed off: warp only, with the way to regenerate the hand-off", rz_.reexec is None and "write_stage4_config" in (rz_.reexec_note or ""),
          str(rz_.reexec_note)[:100])
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
