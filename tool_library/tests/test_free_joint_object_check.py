"""free_joint_object_check: moving objects with one free root and tied parts pass; an object meant to move but fixed, a loose
untied part, a static object with a free joint and a nested free joint fail.

    python tool_library/tests/test_free_joint_object_check.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.free_joint_object_check import free_joint_object_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="free_joints_"))
(tmp / "stage1").mkdir()
(tmp / "stage1" / "scene_semantics.json").write_text(json.dumps({"objects": [
    {"id": "cup", "role": "manipulated", "type": "rigid"}, {"id": "bottle", "role": "manipulated", "type": "rigid"},
    {"id": "rack", "role": "container", "type": "rigid"}]}))
(tmp / "stage2").mkdir()
recon = tmp / "stage2" / "scene_recon.json"
recon.write_text(json.dumps({"scene_semantic": "../stage1/scene_semantics.json", "objects": [
    {"id": "cup", "type": "rigid", "root_body": "cup_cup", "bodies": ["cup_cup"]},
    {"id": "bottle", "type": "rigid", "root_body": "bottle_bottle", "bodies": ["bottle_bottle", "bottle_bottle_cap"]},
    {"id": "rack", "type": "rigid", "root_body": "rack_rack", "bodies": ["rack_rack"]}]}))
CUP = '<body name="cup_cup" pos="0 0 .05">{j}<geom type="cylinder" size=".03 .05" mass=".1"/></body>'
BOTTLE = ('<body name="bottle_bottle" pos=".2 0 .06"><freejoint/><geom type="cylinder" size=".03 .06" mass=".2"/></body>'
          '<body name="bottle_bottle_cap" pos=".2 0 .13"><freejoint/><geom type="cylinder" size=".02 .01" mass=".01"/></body>')
RACK = '<body name="rack_rack" pos="-.2 0 .05">{j}<geom type="box" size=".05 .05 .05"/></body>'
WELD = '<equality><weld body1="bottle_bottle" body2="bottle_bottle_cap"/></equality>'


def scene(cup_free=True, weld=True, rack_free=False):
    return ('<mujoco><worldbody><body name="table"><geom type="box" size=".5 .5 .02" pos="0 0 -.02"/></body>'
            + CUP.format(j="<freejoint/>" if cup_free else "") + BOTTLE + RACK.format(j="<freejoint/>" if rack_free else "")
            + "</worldbody>" + (WELD if weld else "") + "</mujoco>")


# 1. the setup has no free joints; the object scene with a free cup and a welded cap passes, moving from the recon + Stage 1
r0 = free_joint_object_check(str(ROOT / "setup" / "scene.xml"))
r1 = free_joint_object_check(scene(), objects=str(recon))
check("setup and a scene with free cup / bottle and a welded cap pass (moving read from Stage 1 roles)",
      r0["ok"] and r1["ok"] and sorted(r1["moving"]) == ["bottle", "cup"], f"{r1['issues']} {r1['moving']}")

# 2. the cup, which Stage 1 says is manipulated, fixed to the world; the cap free but tied to nothing
r2 = free_joint_object_check(scene(cup_free=False), objects=str(recon))
r3 = free_joint_object_check(scene(weld=False), objects=str(recon))
check("a manipulated cup without a free root fails; an untied free cap fails",
      not r2["ok"] and "cup" in r2["issues"][0] and not r3["ok"] and r3["objects"]["bottle"]["untied_free_bodies"] == ["bottle_bottle_cap"],
      f"{r2['issues']} {r3['issues']}")

# 3. a rack declared static with a free joint
r4 = free_joint_object_check(scene(rack_free=True), objects=str(recon), static=["rack"])
r5 = free_joint_object_check(scene(rack_free=False), objects=str(recon), static=["rack"])
check("a static rack with a free joint fails; without it passes", not r4["ok"] and r5["ok"], f"{r4['issues']} {r5['issues']}")

# 4. a free joint nested below another body (an attach into the wrong parent) is listed even though MuJoCo refuses it
nested = ('<mujoco><worldbody><body name="tray"><geom type="box" size=".1 .1 .01"/>'
          '<body name="cup_cup" pos="0 0 .05"><freejoint name="cup_free"/><geom type="sphere" size=".02"/></body>'
          '<body name="cup_saucer" pos=".05 0 .05"><freejoint name="saucer_free"/><geom type="sphere" size=".02"/></body>'
          '</body></worldbody></mujoco>')
r6 = free_joint_object_check(nested)
check("nested free joints are all listed and fail", not r6["ok"] and len(r6["nested_free"]) == 2 and not r6["compiled"], str(r6["issues"]))

sys.exit(1 if fails else 0)
