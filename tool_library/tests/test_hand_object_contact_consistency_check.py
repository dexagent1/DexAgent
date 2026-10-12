"""hand_object_contact_consistency_check: fingertips that touch the reconstructed box pass; a box placed 8 cm off,
a fingertip deep inside, no hand joints, and an object no subgoal touches are told apart.

    python tool_library/tests/test_hand_object_contact_consistency_check.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from tool_library.verifiers.hand_object_contact_consistency_check import hand_object_contact_consistency_check as check  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="hand_contact_", dir=os.environ.get("TMPDIR")))
FAILS = []


def ok(name, cond, extra=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  ({extra})" if extra else ""))
    if not cond:
        FAILS.append(name)


def scene(box_xyz):
    p = TMP / f"scene_{box_xyz[0]:.2f}_{box_xyz[1]:.2f}.xml"
    p.write_text(f"""<mujoco><worldbody><geom name="table" type="plane" size="1 1 0.01"/>
      <body name="box_box" pos="{box_xyz[0]} {box_xyz[1]} {box_xyz[2]}"><freejoint/>
        <geom name="box_geom" type="box" size="0.03 0.02 0.025" mass="0.1"/></body>
      <body name="cup_cup" pos="-0.2 0.4 0.04"><freejoint/><geom type="cylinder" size="0.03 0.04" mass="0.1"/></body>
      </worldbody><keyframe><key qpos="{box_xyz[0]} {box_xyz[1]} {box_xyz[2]} 1 0 0 0 -0.2 0.4 0.04 1 0 0 0"/></keyframe></mujoco>""")
    return p


def hands(tip_at, contact_frames=range(20, 31), T=60):
    """Right thumb/index/middle tips 15 cm above the box, coming down to `tip_at` (index tip) over contact_frames."""
    names = ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip"]
    X = np.zeros((T, 3, 3))
    tip = np.asarray(tip_at, float)
    for f in range(T):
        lift = 0.0 if f in contact_frames else 0.15
        X[f, 1] = tip + [0, 0, lift]
        X[f, 0] = tip + [-0.04, 0, lift + 0.01]
        X[f, 2] = tip + [0.02, 0, lift + 0.01]
    p = TMP / f"hands_{len(list(TMP.iterdir()))}.npz"
    np.savez_compressed(p, names=np.array(names), rig_xyz=X.astype(np.float32))
    return p


sem = TMP / "scene_semantics.json"
sem.write_text(json.dumps({"objects": [{"id": "box"}, {"id": "cup"}],
                           "subgoals": [{"index": 1, "acting_hand": "right", "motion": "grasp_lift", "object_refs": ["box"], "time_window_s": [0.6, 1.1]}]}))
recon = TMP / "scene_recon.json"
recon.write_text(json.dumps({"objects": [{"id": "box", "root_body": "box_box"}, {"id": "cup", "root_body": "cup_cup"}]}))

top = [0.10, 0.40, 0.05 + 0.004]          # 4 mm above the box top
r = check(str(scene([0.10, 0.40, 0.025])), str(hands(top)), str(sem), str(recon), fps=30)
ok("fingertips touch the box at the grasp", r["ok"] and r["objects"]["box"]["gap_m"] <= 0.02, f"gap {r['objects']['box']['gap_m']}")
ok("an object no subgoal touches is skipped", "cup" in r["skipped"], str(r["skipped"].get("cup")))

r = check(str(scene([0.10, 0.48, 0.025])), str(hands(top)), str(sem), str(recon), fps=30)
ok("a box placed 8 cm too far is caught", not r["ok"] and r["objects"]["box"]["gap_m"] > 0.05,
   f"gap {r['objects']['box']['gap_m']}, offset {r['objects']['box']['offset_m']}")
ok("the issue reports the measured offset without prescribing a scene edit", r["issues"] and "surface-to-fingertip offset" in r["issues"][0])

r = check(str(scene([0.10, 0.68, 0.025])), str(hands(top)), str(sem), str(recon), fps=30)
ok("a large placement error does not falsely require a Stage1 rewrite", not r["ok"] and
   "Stage 1 must be revised" not in " ".join(r["issues"]) and
   r["objects"]["box"]["diagnostic_scope"].endswith("cause unresolved"))

r = check(str(scene([0.10, 0.40, 0.025])), str(hands([0.10, 0.40, 0.025])), str(sem), str(recon), fps=30, max_penetration_m=0.015)
ok("a fingertip at the box's centre (2 cm deep) is caught", not r["ok"] and r["objects"]["box"]["inside_depth_m"] > 0.015,
   f"depth {r['objects']['box']['inside_depth_m']}")

r = check(str(scene([0.10, 0.40, 0.025])), None, str(sem), str(recon))
ok("no hand joints: not applicable, ok", r["ok"] and r["applicable"] is False)

r = check(str(scene([0.10, 0.40, 0.025])), str(hands(top, contact_frames=range(50, 60))), str(sem), str(recon), fps=30)
ok("contact outside the subgoal window does not count", not r["ok"], f"gap {r['objects']['box']['gap_m']}")

# an object a hand already holds at the first frame is placed at rest by Stage 2: skipped with Stage 1's own words
held_sem = TMP / "scene_semantics_held.json"
held_sem.write_text(json.dumps({"objects": [{"id": "box", "initial_state": "already pinched in the right hand above the table"}, {"id": "cup"}],
                                "subgoals": [{"index": 1, "acting_hand": "right", "motion": "grasp_lift", "object_refs": ["box"], "time_window_s": [0.0, 1.1]}]}))
r = check(str(scene([0.10, 0.48, 0.025])), str(hands(top)), str(held_sem), str(recon), fps=30)
ok("a box held from the first frame is skipped with the reason", r["ok"] and "held from the first frame" in r["skipped"].get("box", ""))
held_sem.write_text(json.dumps({"objects": [{"id": "box", "initial_state": "already pinched in the right hand above the table"}, {"id": "cup"}],
                                "subgoals": [{"index": 1, "acting_hand": "right", "motion": "grasp_lift", "object_refs": ["box"], "time_window_s": [0.6, 1.1]}]}))
r = check(str(scene([0.10, 0.48, 0.025])), str(hands(top)), str(held_sem), str(recon), fps=30)
ok("the same words with a first contact later in the video are still judged", not r["ok"])
held_sem.write_text(json.dumps({"objects": [{"id": "box", "initial_state": "on the cloth to the right of the hands"}, {"id": "cup"}],
                                "subgoals": [{"index": 1, "acting_hand": "right", "motion": "grasp_lift", "object_refs": ["box"], "time_window_s": [0.0, 1.1]}]}))
r = check(str(scene([0.10, 0.48, 0.025])), str(hands(top)), str(held_sem), str(recon), fps=30)
ok("an object merely near the hands at the start is judged", not r["ok"])

sys.exit(1 if FAILS else 0)
