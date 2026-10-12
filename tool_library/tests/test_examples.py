"""Verified-example reuse: find the example a video matches, copy its stages, adapt its trajectory to a new layout,
save a run as an example and find it again.

    MUJOCO_GL=egl python tool_library/tests/test_examples.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.skills.find_example import find_example  # noqa: E402
from tool_library.skills.reuse_example import reuse_example  # noqa: E402
from tool_library.skills.adapt_example_trajectory import adapt_example_trajectory  # noqa: E402
from tool_library.common import examples as EX  # noqa: E402

FAILS = 0
EXAMPLE = ROOT / "examples/default/battery_insertion"
VIDEO = ROOT / "stage1_semantic_understanding/examples/battery_insertion/video.mp4"


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    if not EXAMPLE.exists():
        print("SKIP: examples/default/battery_insertion is not present"); return 0
    tmp = Path(tempfile.mkdtemp(prefix="dexagent_examples_", dir=os.environ.get("DEXAGENT_TMP")))
    r = find_example(str(VIDEO))
    check("the example's own video is identical", r["match"] == "identical" and r["task"] == "battery_insertion", str(r["best"]["video"]))
    other = ROOT / "stage1_semantic_understanding/examples/open_the_bottle/video.mp4"
    r2 = find_example(str(other), scene_semantic=str(EXAMPLE / "stage1/scene_semantics.json"))
    check("another task's video is not identical to battery", not (r2["match"] == "identical" and r2["task"] == "battery_insertion"), r2["match"])
    sem = json.loads((EXAMPLE / "stage1/scene_semantics.json").read_text())
    r3 = EX.find_example(scene_semantic=sem)
    check("the same Stage 1 result is the same task", r3["match"] == "same_task" and r3["task"] == "battery_insertion", r3["match"])
    run = tmp / "run"
    for st in (1, 2, 3):
        out = reuse_example(str(EXAMPLE), st, str(run / f"stage{st}"), video=str(VIDEO))
        check(f"reuse stage {st}", len(out["files"]) >= 1, str(len(out["files"])))
    m = mujoco.MjModel.from_xml_path(str(run / "stage2/scene.xml"))
    a = np.load(run / "stage3/trajectory.npz"); b = np.load(EXAMPLE / "stage3/trajectory.npz")
    check("reused scene compiles and the trajectory is the example's", m.nq == a["qpos"].shape[1] and np.array_equal(a["qpos"], b["qpos"]))
    ad = adapt_example_trajectory(str(EXAMPLE), str(tmp / "adapt"), layout={"battery_battery": [0.012, -0.008, 0.0], "holder_battery_holder": [-0.008, 0.01, 3.0]})
    check("adapt to a new layout", ad["ok"], str(ad["failed"]))
    Q = np.load(tmp / "adapt/trajectory.npz")["qpos"]
    d = mujoco.MjData(m); d.qpos[:] = Q[0]; mujoco.mj_kinematics(m, d)
    p_cell = d.xpos[m.body("battery_battery").id].copy()
    d.qpos[:] = b["qpos"][0]; mujoco.mj_kinematics(m, d)
    moved = p_cell - d.xpos[m.body("battery_battery").id]
    check("the adapted trajectory starts with the cell moved by the layout", abs(moved[0] - 0.012) < 1e-6 and abs(moved[1] + 0.008) < 1e-6, str(moved.round(4)))
    # an example that ships a recorded Isaac Sim twin episode (scissor_cutting): Stage 3 reuse copies its video and files
    scis = EXAMPLE.parent / "scissor_cutting"
    twin = EX.twin_episode(scis)
    tw = reuse_example(str(scis), 3, str(tmp / "twin")) if twin else {"files": []}
    same = twin and (tmp / "twin/sim_video.mp4").read_bytes() == (scis / "stage3" / twin["files"]["sim_video"]).read_bytes()
    check("the scissor_cutting example's Stage 3 is its Isaac Sim twin episode: the video and files are reused as they are",
          bool(same) and all((tmp / "twin" / r).is_file() for r in twin["files"].values()), str(len(tw["files"])))
    # a reused run, saved, is found again
    check("the reused Stage 3 carries its Stage 4 task folder", (run / "stage3/stage4_task/stage4.json").exists())
    ex2 = tmp / "examples"
    sv = EX.save_example(run, name="saved_battery", video=str(VIDEO), examples_dir=ex2)
    r4 = EX.find_example(video=str(VIDEO), examples_dir=ex2)
    check("a saved run is found again", r4["match"] == "identical" and Path(r4["example"]).name == "saved_battery", r4["match"])
    # objects that share one asset (a box and its lid, a tip inside its box) share one file in every shipped example:
    # split copies hide the shared asset from the checks that compare such objects together
    import hashlib
    dup = {}
    for exd in sorted((ROOT / "examples/default").iterdir()):
        seen = {}
        for f in sorted((exd / "stage2/objects").glob("*/*.xml")):
            h = hashlib.md5(f.read_bytes()).hexdigest()
            if h in seen:
                dup.setdefault(exd.name, []).append(f"{seen[h]} = {f.parent.name}")
            seen.setdefault(h, f.parent.name)
    check("no shipped example splits a shared asset into identical copies", not dup, str(dup))
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL PASS" if not FAILS else f"\n{FAILS} FAILED")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
