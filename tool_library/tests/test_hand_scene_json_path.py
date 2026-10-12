"""The grasp verifiers take a hand-scene JSON path, as hand_scene_from_recon writes it.

open_scene accepted a path, but hand.grasp_body read scene.get(...) and crashed ("'str' object has no attribute 'get'")
inside grasp_lift_test, grasp_stress_test and every other tool that finds the grasped body, so the file
hand_scene_from_recon writes could not be passed to them. These cases write such a file (with an xml path relative to
it) and check grasp_body and two driven verifiers on it against the same scene given as a dict.

    python tool_library/tests/test_hand_scene_json_path.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.pose_from_points import pose_from_points  # noqa: E402
from tool_library.verifiers.grasp_hold_test import grasp_hold_test  # noqa: E402
from tool_library.verifiers.grasp_lift_test import grasp_lift_test  # noqa: E402

HAND = os.environ.get("DEXAGENT_HAND_XML") or str(ROOT / "setup" / "robot" / "sharpa_right.xml")
TMP = Path(tempfile.mkdtemp(prefix="dexagent_hsjson_", dir=os.environ.get("TMPDIR")))
FAILS = 0
T0 = time.time()


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


(TMP / "assets").mkdir()
(TMP / "assets" / "box.xml").write_text('<mujoco model="box"><worldbody><body name="box" pos="0 0 0.025"><freejoint/>'
                                        '<geom name="box_body" type="box" size="0.025 0.02 0.025" mass="0.12"/></body></worldbody></mujoco>')
(TMP / "assets" / "block.xml").write_text('<mujoco model="block"><worldbody><body name="block" pos="0 0 0.02">'
                                          '<geom name="block_body" type="box" size="0.02 0.02 0.02"/></body></worldbody></mujoco>')
desc = {"hand_xml": HAND, "side": "right", "floor": True,
        "objects": [{"name": "blk", "xml": "assets/block.xml", "pos": [0.15, 0.0, 0.0]},      # relative to the json, as written
                    {"name": "box", "xml": "assets/box.xml", "pos": [0.0, 0.0, 0.0]}]}
path = TMP / "hs_right_box.json"
path.write_text(json.dumps(desc))
as_dict = H.load_scene_spec(path)                        # the same description with the paths resolved

# 1. grasp_body: a path gives the same body as the dict, with and without an explicit grasp_body
model, data0, rig, ids = H.open_scene(str(path), settle_s=0.1)
try:
    b_path, err = H.grasp_body(model, str(path), ids), None
except Exception as e:                                    # the measured failure: AttributeError on str.get
    b_path, err = None, f"{type(e).__name__}: {e}"
b_dict = H.grasp_body(model, as_dict, ids)
check("grasp_body takes the json path (last object's free body)", err is None and b_path == b_dict and model.body(b_dict).name == "box_box",
      err or model.body(b_dict).name)
path2 = TMP / "hs_right_block.json"
path2.write_text(json.dumps(dict(desc, grasp_body="blk_block")))
check("grasp_body honours the file's grasp_body", model.body(H.grasp_body(model, Path(path2), ids)).name == "blk_block")

# 2. the driven verifiers on the json path: same verdict as on the dict
pose = pose_from_points(as_dict, {"thumb": [0.0, -0.02, 0.03], "index": [0.0, 0.02, 0.03]},
                        {"thumb": [0.0, -1.0, 0.0], "index": [0.0, 1.0, 0.0]}, yaw_step_deg=90, pitches_deg=(-30, 0, 30), iters=160)
try:
    lp, ld = grasp_lift_test(str(path), pose, lift_heights_m=(0.05,)), grasp_lift_test(as_dict, pose, lift_heights_m=(0.05,))
    err = None
except Exception as e:
    lp, ld, err = None, None, f"{type(e).__name__}: {e}"
check("grasp_lift_test on the json path runs and agrees with the dict", err is None and lp["ok"] == ld["ok"] and lp["lifts"] == ld["lifts"],
      err or f"ok={lp['ok']} lifts={[l.get('achieved_mm') for l in lp['lifts']]}")
try:
    hp, err = grasp_hold_test(str(path), pose, hold_s=0.3), None
except Exception as e:
    hp, err = None, f"{type(e).__name__}: {e}"
check("grasp_hold_test on the json path runs", err is None and "ok" in hp, err or f"ok={hp['ok']}")

print(f"{'ALL PASS' if not FAILS else f'{FAILS} FAILED'} ({time.time() - T0:.0f}s)")
sys.exit(1 if FAILS else 0)
