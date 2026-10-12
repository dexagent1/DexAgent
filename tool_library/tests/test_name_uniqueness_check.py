"""name_uniqueness_check: claimed, named, non-capturing object parts pass; loose unlisted parts, unnamed protruding collision
geoms, cross-asset prefix capture, unprefixed robot parts and repeated names fail.

    python tool_library/tests/test_name_uniqueness_check.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.name_uniqueness_check import name_uniqueness_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="names_"))
HEAD = '<mujoco><worldbody><body name="table"><geom name="g_table" type="box" size=".5 .5 .02" pos="0 0 -.02"/></body>'
BOX = ('<body name="tipbox_tipbox" pos="0 0 .02"><freejoint/><geom name="tipbox_tipbox_base" type="box" size=".04 .03 .02"/>'
       '{inner}</body>'
       '<body name="tipbox_tipbox_lid" pos="0 0 .06"><freejoint/><geom name="tipbox_tipbox_lid_g" type="box" size=".04 .03 .003"/></body>')
TAIL = '</worldbody></mujoco>'


def recon(objs):
    p = tmp / f"recon_{len(list(tmp.iterdir()))}.json"
    p.write_text(json.dumps({"objects": objs}))
    return str(p)


xml = HEAD + BOX.format(inner="") + TAIL
full = recon([{"id": "tipbox", "root_body": "tipbox_tipbox", "bodies": ["tipbox_tipbox", "tipbox_tipbox_lid"], "asset_xml": "a.xml"}])
partial = recon([{"id": "tipbox", "root_body": "tipbox_tipbox", "bodies": ["tipbox_tipbox"], "asset_xml": "a.xml"}])

# 1. a tip box whose loose lid is listed under its object passes; the setup passes; a recon that forgot the lid fails
r0 = name_uniqueness_check(str(ROOT / "setup" / "scene.xml"))
r1 = name_uniqueness_check(xml, objects=full)
r2 = name_uniqueness_check(xml, objects=partial)
check("listed lid passes (and the setup); a separate lid body missing from the recon fails as unclaimed",
      r0["ok"] and r1["ok"] and not r2["ok"] and r2["unclaimed_bodies"] == ["tipbox_tipbox_lid"], f"{r1['issues']} {r2['issues']}")

# 2. an unnamed collidable geom that stands out of the named geometry fails; one buried inside it is a note
out = HEAD + BOX.format(inner='<geom type="sphere" size=".01" pos=".06 0 0"/>') + TAIL
buried = HEAD + BOX.format(inner='<geom type="sphere" size=".005" pos="0 0 0"/>') + TAIL
r3 = name_uniqueness_check(out, objects=full)
r4 = name_uniqueness_check(buried, objects=full)
check("an unnamed protruding collision geom fails; a buried one passes with a note",
      not r3["ok"] and "no name" in " ".join(r3["issues"]) and r4["ok"] and any("harmless" in n for n in r4["notes"]),
      f"{r3['issues']} {r4['notes']}")

# 3. prefix capture across assets: 'battery' (its own asset) is found by 'battery_', which also takes the holder's bodies
bat = (HEAD + '<body name="battery_battery" pos=".1 0 .02"><freejoint/><geom name="battery_battery_can" type="cylinder" size=".005 .02"/></body>'
       '<body name="battery_holder_battery_holder" pos="-.1 0 .01"><freejoint/><geom name="battery_holder_battery_holder_floor" type="box" size=".03 .02 .002"/></body>' + TAIL)
two_assets = recon([{"id": "battery", "root_body": "battery_battery", "bodies": ["battery_battery"], "asset_xml": "battery.xml"},
                    {"id": "battery_holder", "root_body": "battery_holder_battery_holder", "bodies": ["battery_holder_battery_holder"],
                     "asset_xml": "battery_holder.xml"}])
one_asset = recon([{"id": "battery", "root_body": "battery_battery", "bodies": ["battery_battery"], "asset_xml": "kit.xml"},
                   {"id": "battery_holder", "root_body": "battery_holder_battery_holder", "bodies": ["battery_holder_battery_holder"],
                    "asset_xml": "kit.xml"}])
r5 = name_uniqueness_check(bat, objects=two_assets)
r6 = name_uniqueness_check(bat, objects=one_asset)
check("'battery_' capturing another asset's holder fails; parts of one asset sharing a prefix pass",
      not r5["ok"] and r5["prefix_collisions"] and r6["ok"], f"{r5['issues']} {r6['issues']}")

# 4. an arm welded to the robot without the robot prefix fails; with the prefix it passes
arm = (HEAD + '<body name="{a}" pos="0 0 .5"><geom name="{a}_g" type="capsule" size=".02 .1"/></body>'
       '<body name="robot_hand" pos="0 0 .3"><freejoint/><geom name="robot_hand_g" type="box" size=".04 .04 .01" contype="0" conaffinity="0"/></body>'
       '</worldbody><equality><weld body1="{a}" body2="robot_hand" torquescale="0"/></equality></mujoco>')
r7 = name_uniqueness_check(arm.format(a="arm_left_gripper"))
r8 = name_uniqueness_check(arm.format(a="robot_arm_left_gripper"))
check("an arm welded to the robot without 'robot_' fails; prefixed it passes",
      not r7["ok"] and r7["robot"]["unprefixed_welded"] == ["arm_left_gripper"] and r8["ok"], f"{r7['issues']} {r8['issues']}")

# 5. a repeated body name is MuJoCo's compile error, reported with the fix
dup = HEAD + '<body name="cup"><geom type="sphere" size=".02"/></body><body name="cup"><geom type="sphere" size=".02"/></body>' + TAIL
r9 = name_uniqueness_check(dup)
check("a repeated name is reported as a compile failure", not r9["ok"] and not r9["compiled"] and "repeated" in r9["compile_error"],
      str(r9)[:300])

sys.exit(1 if fails else 0)
