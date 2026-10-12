"""Smoke test for actuator_limit_integrity_check: setup servos pass; stronger, extra, retargeted and uncapped servos fail."""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SETUP = ROOT / "setup"
TMP = Path(tempfile.mkdtemp())
fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {str(msg)[:600]}"))
    fails += 0 if cond else 1


def bench(name, objects="", key_extra="", edit=None):
    """The setup's bench (setup/scene.xml) with objects added to the worldbody and their free-joint qpos appended to
    keyframe 'home', written to a temp file (meshes resolved from setup/)."""
    xml = (SETUP / "scene.xml").read_text()
    xml = xml.replace('<compiler angle="radian" />', f'<compiler angle="radian" meshdir="{SETUP}" texturedir="{SETUP}" />', 1)
    xml = xml.replace("</worldbody>", objects + "\n  </worldbody>", 1)
    if key_extra:
        xml = re.sub(r'(<key name="home" qpos=")([^"]*)', lambda m: m.group(1) + m.group(2) + " " + key_extra, xml, count=1)
    if edit:
        xml = edit(xml)
    p = TMP / f"{name}.xml"
    p.write_text(xml)
    return str(p)


CUBE = ('<body name="cube" pos="0 0.5 0.025"><freejoint name="cube_free"/>'
        '<geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" rgba="0.8 0.2 0.2 1"/></body>')
CUBE_Q = "0 0.5 0.025 1 0 0 0"


from tool_library.verifiers.actuator_limit_integrity_check import actuator_limit_integrity_check as check_act

r = check_act(str(SETUP / "scene.xml"))
check("the setup's own actuators pass", r["ok"] and r["n_actuators"] == 56, r["issues"])
r = check_act(bench("cube", CUBE, CUBE_Q))
check("a bench with an object passes", r["ok"], r["issues"])

ACT = '<general name="robot_left_left_index_PIP_position" joint="robot_left_left_index_PIP" ctrlrange="0 1.7453" biastype="affine" gainprm="300" biasprm="0 -300 1" />'
p = bench("strong", CUBE, CUBE_Q, lambda x: x.replace(ACT, ACT.replace('gainprm="300" biasprm="0 -300 1"', 'gainprm="3000" biasprm="0 -3000 1"')))
r = check_act(p)
check("a finger servo ten times stronger is caught", (not r["ok"]) and any(c["field"] == "gain" for c in r["changed"]) and float(r["worst"]) >= 9.9, (r["changed"][:2], r["worst"]))

p = bench("extra", CUBE, CUBE_Q, lambda x: x.replace("</actuator>", '<motor name="boost" joint="robot_left_left_index_PIP" gear="1"/>\n</actuator>'))
r = check_act(p)
check("a second actuator on a finger joint is caught", (not r["ok"]) and any("boost" in e for e in r["extra"]), r["extra"])

p = bench("ctrl", CUBE, CUBE_Q, lambda x: x.replace(ACT, ACT.replace('ctrlrange="0 1.7453"', 'ctrlrange="-1 3"')))
r = check_act(p)
check("a widened control range is caught", (not r["ok"]) and any(c["field"] == "ctrlrange" for c in r["changed"]), r["changed"])

p = bench("cap", CUBE, CUBE_Q, lambda x: x.replace('name="robot_right_arm_dof_joint2" pos="0 0 0" axis="0 1 0" range="-0.15 3.14159" actuatorfrcrange="-1 1"',
                                                   'name="robot_right_arm_dof_joint2" pos="0 0 0" axis="0 1 0" range="-0.15 3.14159" actuatorfrcrange="-50 50"'))
r = check_act(p)
check("a widened joint force cap is caught", (not r["ok"]) and any(c["joint"] == "robot_right_arm_dof_joint2" for c in r["joint_caps_changed"]), r["joint_caps_changed"])

p = bench("retarget", CUBE, CUBE_Q, lambda x: x.replace(ACT, ACT.replace('joint="robot_left_left_index_PIP"', 'joint="robot_left_left_index_DIP"')))
r = check_act(p)
check("an actuator moved to another joint is caught", (not r["ok"]) and any(c["field"] == "target" for c in r["changed"]), r["changed"])

print(f"{'OK' if fails == 0 else 'FAILED'}: actuator_limit_integrity_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
