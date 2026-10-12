"""A one-sided fingertip press on a thin flat strip (a ribbon on the table, one index pad on its top, no thumb opposing).

Measured on an EgoDex unbraid run (Sharpa right hand, fingers-down index on a 1.2 mm ribbon strip):
- the executor's clamp flexed the pressing index (MCP 0.255 -> 0.335 rad), which on a fingers-down hand LIFTS the pad
  6.7 mm off the strip, so the hold failed ['in_hand', 'pads_touching'] with the pad 15.6 mm off. The switch to a wrist
  press existed but compared a hand-frame pad displacement with the world contact normal and never triggered.
- pose_from_points recorded "approach": [0.899, -0.438, 0.0] for that press (the pads-to-wrist direction projected
  into the strip's plane), so the planner's retreat axis slid the hand in along the table.
- the default 55% open approach shape extended that index (the reverse of the lifting flexion) into the strip on the
  way down: 6.9 mm of plan collision, the strip shoved 9.4 mm before the hand arrived.

    python tool_library/tests/test_one_sided_press.py
"""
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import grasp_robust as GR  # noqa: E402
from tool_library.common import hand as H  # noqa: E402

HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
TMP = Path(tempfile.mkdtemp(prefix="dexagent_press_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


# the strip: one 30 x 16 x 1.2 mm free box lying on the table under the index point (the end of a ribbon strand)
PT = [0.2535, 0.1705, 0.0012]
(TMP / "strip.xml").write_text('<mujoco model="strip"><worldbody><body name="strip" pos="0 0 0.0006"><freejoint/>'
                               '<geom name="strip_g" type="box" size="0.015 0.008 0.0006" mass="0.001" friction="0.6 0.005 0.0001"/>'
                               '</body></worldbody></mujoco>')
SCENE = {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": "strip_strip",
         "objects": [{"name": "strip", "xml": str(TMP / "strip.xml"), "pos": [PT[0], PT[1], 0.0]}]}
# the run's pose (pose_from_points, index pad on the strip's top, fingers down; approach set to the normal)
POSE = {"side": "right", "pos": [0.3438852806245273, 0.12643389096771016, 0.1626360485890471],
        "euler": [-3.148130783881836, -0.2819052998044531, 3.1448205223785615],
        "finger_targets": {"thumb_CMC_FE": 1.60574, "thumb_CMC_AA": -0.17455, "thumb_MCP_FE": 1.108315, "thumb_MCP_AA": -0.17455,
                           "thumb_IP": 1.483505, "index_MCP_FE": 0.25472945005285025, "index_MCP_AA": -0.17181681178199454,
                           "index_PIP": 0.4340759773562338, "index_DIP": 0.3487794290907929, "middle_MCP_FE": 1.3090000605,
                           "middle_MCP_AA": 0.0, "middle_PIP": 1.483505, "middle_DIP": 1.186855, "ring_MCP_FE": 1.3090000605,
                           "ring_MCP_AA": 0.0, "ring_PIP": 1.483505, "ring_DIP": 1.186855, "pinky_CMC": 0.22253,
                           "pinky_MCP_FE": 1.3090000605, "pinky_MCP_AA": 0.0, "pinky_PIP": 1.483505, "pinky_DIP": 1.186855},
        "grip_fingers": ["index"], "press_depth_rad": 0.12, "approach": [0.0, 0.0, 1.0],
        "normals": {"index": [0.0, 0.0, 1.0]}, "points": {"index": PT}, "standoff_m": 0.008}
POSE["finger_targets_squeezed"] = dict(POSE["finger_targets"], index_MCP_FE=0.3345229929488512, index_MCP_AA=-0.16314513149337,
                                       index_PIP=0.2527312115011753, index_DIP=0.2052754905196946)

model, data0, rig, ids = H.open_scene(SCENE)

# 1. the clamp of a fingers-down fingertip press comes from the wrist, along the contact normal
c = H.executor_clamp(rig, POSE)
idx = [k for k in c["closed"] if k.startswith("index")]
check("flexing a fingers-down index lifts its pad off the strip (world frame)", c["pad_into_m"] is not None and c["pad_into_m"] < -0.003,
      f"pad travel into the strip {1000 * (c['pad_into_m'] or 0):.1f} mm")
check("so the fingers stay as posed and the wrist presses 4 mm down the normal",
      c["grip_posed"] and all(abs(c["press"][k] - c["closed"][k]) < 1e-9 for k in idx)
      and c["wrist_press"] is not None and np.allclose(c["wrist_press"], [0, 0, -0.004], atol=1e-6), str(c["wrist_press"]))
# the same hand on the underside of a part (normal down): the same flexion now presses the pad in, the finger press stays
c_up = H.executor_clamp(rig, dict(POSE, normals={"index": [0.0, 0.0, -1.0]}))
check("a flexion that moves the pad into its surface keeps the finger press", not c_up["grip_posed"] and c_up["wrist_press"] is None
      and c_up["press"]["index_MCP_FE"] > c_up["closed"]["index_MCP_FE"] + 0.05, f"{1000 * c_up['pad_into_m']:.1f} mm")

# 2. the stress tests' executor grip and the planner agree with it
case = GR.Case(GR.load_scene(SCENE))
g = GR.executor_grip(case, POSE)
check("grasp_robust.executor_grip presses with the wrist the same way", g["wrist_press"] is not None
      and np.allclose(g["wrist_press"], c["wrist_press"]) and g["press"] == c["press"])
from tool_library.common.subgoal import SubgoalPlanner  # noqa: E402
bench = mujoco.MjModel.from_xml_path(str(ROOT / "setup" / "scene.xml"))
db = mujoco.MjData(bench)
mujoco.mj_resetDataKeyframe(bench, db, 0)
sp = SubgoalPlanner(bench, db.qpos.copy())
gp = sp._grip_from_pose("right", POSE)
check("SubgoalPlanner._grip_from_pose (the bench hand) presses with the wrist the same way", gp["wrist_press"] is not None
      and np.allclose(gp["wrist_press"], c["wrist_press"]) and all(abs(gp["press"][k] - c["press"][k]) < 1e-9 for k in idx), str(gp["wrist_press"]))

# 3. in physics: the hand set down with its posed fingers (open_frac 0: only the clamp is judged here), closed and
# clamped; the pad stays on the strip as the outcome's pad check reads it (its fingertip link touches the strip, or the
# pad centre is within 2 mm of the surface) and the strip stays where it lies
bid = H.grasp_body(model, SCENE, ids)
sid = rig.fingertip_sites["index"]
tip_body = int(model.site_bodyid[sid])
pad_r = H.pad_radius(model, rig, "index")


def pad_on_strip(pose, **kw):
    r = H.stress_rollout(model, data0, rig, ids, pose, bid, servo_reserve=False, **kw)
    d = r["data"]
    mujoco.mj_forward(model, d)
    touch = any({int(model.geom_bodyid[d.contact[i].geom1]), int(model.geom_bodyid[d.contact[i].geom2])} == {tip_body, bid} for i in range(d.ncon))
    geo = float(d.site_xpos[sid][2] - d.xpos[bid][2] - 0.0006)
    moved = float(np.linalg.norm(d.xpos[bid] - data0.xpos[bid]))
    return r, touch, geo, moved


r, touch, geo, moved = pad_on_strip(POSE, phases=("close", "hold"), open_frac=0.0)
check("the clamped index pad stays on the strip (stress_rollout, executor close)", (touch or geo - pad_r < 0.002) and moved < 0.003,
      f"fingertip link touching {touch}, pad centre {1000 * geo:.1f} mm above the strip top (pad radius {1000 * pad_r:.1f} mm), strip moved {1000 * moved:.1f} mm")
check("the rollout's wrist ends pressed 4 mm below the pose", abs(r["final_wrist_pos"][2] - (POSE["pos"][2] - 0.004)) < 1e-9,
      f"{r['final_wrist_pos'][2]:.4f}")

# 4. the approach shape keeps the pressing finger as posed (opening it is the reverse of the lifting flexion: it reaches
# into the strip on the way down); the idle fingers still curl, as every opening does
op = H.approach_open_targets(rig, c["closed"], 0.55, c["grip_fingers"], c["grip_posed"])
check("a wrist press approaches with its pressing index as posed", all(abs(op[k] - c["closed"][k]) < 1e-9 for k in idx))
op_up = H.approach_open_targets(rig, c_up["closed"], 0.55, c_up["grip_fingers"], c_up["grip_posed"])
check("a finger press still opens its grip fingers", op_up["index_PIP"] < c_up["closed"]["index_PIP"] - 0.1,
      f"PIP {c_up['closed']['index_PIP']:.3f} -> {op_up['index_PIP']:.3f}")
gp_open = sp._floor_clear_open("right", POSE, gp)
check("the executor's grip and the planner's floor-clear opening agree (index as posed)",
      all(abs(g["opened"][k] - c["closed"][k]) < 1e-9 and abs(gp_open[k] - c["closed"][k]) < 1e-9 for k in idx))
r, touch, geo, moved = pad_on_strip(POSE, phases=("approach", "close", "hold"), approach_m=0.05)
appr = next(p_ for p_ in r["phases"] if p_["name"] == "approach_open")
check("with the default opening the hand comes down onto the strip without shoving it, and the pad stays on",
      appr["ok"] and appr["object_nudged_mm"] < 2.0 and moved < 0.003 and (touch or geo - pad_r < 0.002),
      f"nudged {appr['object_nudged_mm']} mm on the approach, strip moved {1000 * moved:.1f} mm, fingertip link touching {touch}")

# 5. pose_from_points records the approach a one-sided contact really has: down the free normal for a press from above,
# along the surface where the normal is blocked (a fingertip pressing up under a lid 2 cm above the table)
from tool_library.skills.pose_from_points import pose_from_points  # noqa: E402
pp = pose_from_points(SCENE, {"index": PT}, {"index": [0, 0, 1]}, curl_free=0.85, approach_dirs=[[0, 0, 1]], yaw_step_deg=90,
                      pitches_deg=(-30, 0, 30))
check("a fingertip press from above approaches down its normal", pp["approach_kind"] == "normal" and np.allclose(pp["approach"], [0, 0, 1], atol=1e-6)
      and pp["worst_residual_mm"] < 2.0, f"approach {np.round(pp['approach'], 3).tolist()} ({pp['approach_kind']}), residual {pp['worst_residual_mm']} mm")
(TMP / "lid.xml").write_text('<mujoco model="lid"><worldbody><body name="lid" pos="0 0 0.02">'
                             '<geom name="lid_g" type="box" size="0.05 0.05 0.0015" mass="0.05"/></body>'
                             '<body name="peg" pos="0.2 0 0.01"><freejoint/><geom name="peg_g" type="box" size="0.01 0.01 0.01" mass="0.01"/></body>'
                             '</worldbody></mujoco>')
LID = {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": "lid_lid",
       "objects": [{"name": "lid", "xml": str(TMP / "lid.xml"), "pos": [0.25, 0.17, 0.0]}]}
pu = pose_from_points(LID, {"index": [0.21, 0.17, 0.0185]}, {"index": [0, 0, -1]}, iters=240, yaw_step_deg=45, pitches_deg=(-30, 0, 30))
check("a fingertip pressing up under a lid near the table approaches along the lid (its normal points into the table)",
      pu["approach_kind"] == "along_surface" and abs(pu["approach"][2]) < 1e-6, f"approach {np.round(pu['approach'], 3).tolist()} ({pu['approach_kind']})")

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
