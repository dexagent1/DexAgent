"""write_stage4_config on a group object (tool_library/common/piece_groups.py): the synthetic stack task (A carried onto
B, C carried onto A) with A, B and C declared as the pieces of ONE Stage 1 object. The group's layout is randomized as
one (every other piece attached_to its root), each carried piece keeps its own carrier and release, and a sampled
layout adapts cleanly. The same scene with three single-body objects is unchanged.

    python tool_library/tests/test_group_objects_stage4.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage4_robot_data_generation" / "tests"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import synthetic  # noqa: E402
from tool_library.common import stage4 as S4  # noqa: E402
from tool_library.skills.write_stage4_config import write_stage4_config as _write_stage4_config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_group_s4_", dir=os.environ.get("TMPDIR")))
config = S4.stage4_module("config")
randomize = S4.stage4_module("randomize")
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  {detail}"))
    fails += 0 if ok else 1


def write_stage4_config(scene_recon_json, episode_dir, *args, **kwargs):
    # the runtime boundary is mocked as in test_stage4_relations.py: this tests the randomization specs only
    scene = Path(scene_recon_json).parent / "scene.xml"
    trajectory = Path(episode_dir) / "trajectory.npz"
    with np.load(trajectory) as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["reference"] = arrays["qpos"].copy()
    fixture = (scene, trajectory, mujoco.MjModel.from_xml_path(str(scene)), arrays)
    with patch.object(S4, "load_runtime_episode", return_value=fixture), patch.object(S4, "load_runtime_pair", return_value=fixture):
        return _write_stage4_config(scene_recon_json, episode_dir, *args, **kwargs)


def build(name, objects):
    _, labels, phases = synthetic.stack_task(TMP / f"{name}_fixture")
    Q = np.load(TMP / f"{name}_fixture" / "trajectory.npz")["qpos"]
    ws = TMP / name / "stage2"
    ws.mkdir(parents=True)
    (ws / "scene.xml").write_text((TMP / f"{name}_fixture" / "scene.xml").read_text())
    (ws / "camera.json").write_text(json.dumps({"width": 64, "height": 48, "fovy_deg": 45.0,
                                                **{k: synthetic.CAMERA[k] for k in ("pos", "quat_wxyz")}}))
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "camera_json": "camera.json", "objects": objects}))
    epd = TMP / name / "stage3" / "episode"
    epd.mkdir(parents=True)
    (epd / "phases.json").write_text(json.dumps(phases))
    np.savez_compressed(epd / "trajectory.npz", qpos=Q, fps=30.0, labels=np.array(labels))
    np.savez_compressed(epd / "trajectory_hands.npz", qpos=Q, fps=30.0, labels=np.array(labels), reference=Q)
    (TMP / name / "stage3" / "trajectory_manifest.json").write_text(json.dumps({"subgoals": [
        {"index": 1, "frames": [0, 120], "motion": "place", "side": "right", "object": "A",
         "plan": {"motion": "place", "object": "A", "on": "B"}, "expect": {"object": "A", "released": True}},
        {"index": 2, "frames": [121, len(labels) - 1], "motion": "place", "side": "right", "object": "C",
         "plan": {"motion": "place", "object": "C", "on": "A"}, "expect": {"object": "C"}}]}))
    (TMP / "human.mp4").write_bytes(b"x")
    out = write_stage4_config(str(ws / "scene_recon.json"), str(epd), str(TMP / name / "task"), human_video=str(TMP / "human.mp4"))
    ep4 = randomize.Episode(config.load_config(out["config"]))
    reps = []
    for s in range(3):
        _, _, rep = ep4.adapt(ep4.sample_layout(np.random.default_rng([0, s])))
        reps.append(rep["ok"])
    return out["objects"], reps


# three single-body objects: as before (each randomized on its own or grouped by contact)
single, reps_single = build("single", [{"id": n, "type": "rigid", "root_body": n, "asset_xml": f"assets/{n}/{n}.xml"} for n in ("A", "B", "C")])
check("single-body objects: carriers and releases as before, layouts adapt",
      single["A"].get("release") == "right: let go of A #2" and single["C"].get("release") == "right: let go of C #4" and all(reps_single),
      f"{single} {reps_single}")

# one group object with A, B, C as its pieces (root B, the piece that never moves)
group, reps_group = build("group", [{"id": "blocks", "type": "rigid", "root_body": "B", "parts": ["B", "A", "C"],
                                     "bodies": ["A", "B", "C"], "asset_xml": "assets/blocks/blocks.xml"}])
check("group object: every piece is randomized, the others attached to the root",
      set(group) >= {"A", "B", "C"} and group["A"].get("attached_to") == "B" and group["C"].get("attached_to") == "B"
      and "attached_to" not in group["B"], str(group))
check("group object: each carried piece keeps its own carrier and release",
      group["A"].get("carrier") == "right" and group["A"].get("release") == "right: let go of A #2"
      and group["C"].get("release") == "right: let go of C #4" and "carrier" not in group["B"], str(group))
check("group object: sampled layouts adapt", all(reps_group), str(reps_group))

print(f"{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}: group objects in write_stage4_config ({TMP})")
sys.exit(1 if fails else 0)
