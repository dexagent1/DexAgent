"""A part held in the air at the moment a hand scene is built (mid-handover) keeps its pose through the scene's settle,
so poses are solved and re-anchored against where it really is; a part resting on the floor is unaffected, and gravity
still acts on the held part afterwards (a grasp is never proven on a floating part).

    python tool_library/tests/test_hand_scene_hold.py
"""
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import hand as H  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_hold_", dir=os.environ.get("TMPDIR")))
HAND = ROOT / "setup" / "robot" / "sharpa_right.xml"
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


pen = TMP / "pen.xml"
pen.write_text('<mujoco model="pen"><worldbody><body name="pen"><freejoint/>'
               '<geom name="pen_body" type="capsule" size="0.0065 0.06" quat="0.7071068 0 0.7071068 0" mass="0.012"/>'
               '</body></worldbody></mujoco>')


def pen_z(hold, z=0.25):
    model, data0, rig, ids = H.build_hand_scene([{"name": "pen", "xml": str(pen), "pos": [0.0, 0.3, z], "hold": hold}], str(HAND))
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen_pen")
    return model, data0, b, float(data0.xpos[b][2])


if not HAND.exists():
    print(f"SKIP: no hand MJCF at {HAND}")
    sys.exit(0)
_, _, _, z_free = pen_z(False)
check("a free pen in the air falls to the floor while the scene settles", z_free < 0.02, f"z {z_free:.3f} m")
model, data0, b, z_held = pen_z(True)
check("a pen marked hold keeps its pose through the settle", abs(z_held - 0.25) < 1e-6, f"z {z_held:.4f} m")
d = mujoco.MjData(model)
d.qpos[:] = data0.qpos
for _ in range(200):
    mujoco.mj_step(model, d)
check("gravity still acts on the held pen after the settle", float(d.xpos[b][2]) < 0.25 - 0.05, f"z after 0.4 s {d.xpos[b][2]:.3f} m")
_, _, _, z_rest = pen_z(True, z=0.0066)
check("a pen resting on the floor stays on it either way", abs(z_rest - 0.0065) < 0.003, f"z {z_rest:.4f} m")

# hand_scene_from_recon marks a part held only when a hand is on it: a block staged in the right hand's palm is held; a
# block staged 25 mm above the table (it will drop and settle) touches nothing either but no hand holds it
import json  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

blk = TMP / "blk.xml"
blk.write_text('<mujoco model="blk"><worldbody><body name="blk"><freejoint/><geom name="g" type="box" size="0.015 0.015 0.015" mass="0.03"/>'
               '</body></worldbody></mujoco>')
_m = mujoco.MjModel.from_xml_path(str(ROOT / "setup" / "scene.xml")); _d = mujoco.MjData(_m)
mujoco.mj_resetDataKeyframe(_m, _d, 0); mujoco.mj_forward(_m, _d)
palm = _d.geom_xpos[mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM, "robot_right_collision_hand_right_palm_0")].copy()
ws = TMP / "bench"; ws.mkdir()
stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "held", "xml": str(blk), "x": float(palm[0]), "y": float(palm[1]), "z": float(palm[2]) - 0.015},
                                                          {"name": "loose", "xml": str(blk), "x": 0.0, "y": 0.45, "z": 0.015 + 0.025}],
               out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
(ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
    {"id": n, "type": "rigid", "asset_xml": str(blk), "root_body": f"{n}_blk", "bodies": [f"{n}_blk"]} for n in ("held", "loose")]}))
hs = hand_scene_from_recon(str(ws / "scene_recon.json"), "right", "loose")
check("a block in the hand's palm is held in the air, a block staged above the table is not", hs["held_in_air"] == ["held"],
      str(hs["held_in_air"]))

# a handover in the stress tests: a held part stays where the other hand holds it while this hand approaches and closes,
# and is let go after the seat (here the hand closes far from it, so it then falls)
model, data0, rig, ids = H.build_hand_scene([{"name": "pen", "xml": str(pen), "pos": [0.0, 0.3, 0.25], "hold": True}], str(HAND))
b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pen_pen")
zs = {}
def rec(m, d, label):
    zs.setdefault(label, []).append(float(d.xpos[b][2]))
far = {"pos": [0.3, 0.0, 0.5], "euler": [3.14159, 0.0, 0.0], "finger_targets": {k: float(lo) for k, (lo, hi) in rig.finger_range.items()}}
H.stress_rollout(model, data0, rig, ids, far, b, phases=("approach", "close", "lift"), record=rec)
before = [z for lab, v in zs.items() if lab not in ("handover",) and not lab.startswith("lift") for z in v]
after = zs.get("handover", []) + [z for lab, v in zs.items() if lab.startswith("lift") for z in v]
check("the held pen stays put while the hand approaches and closes", before and max(abs(z - 0.25) for z in before) < 1e-6,
      f"{len(before)} steps, worst {max(abs(z - 0.25) for z in before) if before else None}")
check("after the handover gravity takes it (no hand holds it here)", after and min(after) < 0.24, f"lowest {min(after) if after else None:.3f} m")
check("the pin is cleared after the rollout", not H._PINNED)

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
