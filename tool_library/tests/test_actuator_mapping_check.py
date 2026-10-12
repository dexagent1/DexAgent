"""actuator_mapping_check: the setup's robot-first ctrl passes and object actuators are listed; actuators inserted before the
robot's, a robot list that differs from the setup, servo ranges beyond their joint and dead actuators fail.

    python tool_library/tests/test_actuator_mapping_check.py
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.actuator_mapping_check import actuator_mapping_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="actuators_"))
SETUP = ROOT / "setup" / "scene.xml"
BASE = re.sub(r"<keyframe>.*?</keyframe>", "", SETUP.read_text(), flags=re.S).replace('<size nkey="1" />', "")
BASE = BASE.replace('file="assets/', f'file="{ROOT}/setup/assets/')
RING = ('<body name="ring" pos="0 0.6 0.02"><freejoint/><geom name="ring_wall" type="cylinder" size=".015 .006" mass=".012"/>'
        '<body name="ring_magnet"><geom name="ring_magnet_shell" type="sphere" size=".016" mass="1e-5" margin=".0003" gap=".0003"/></body></body>')
MAG = '<general name="ring_magnet_pull" body="ring_magnet" gainprm="0" biastype="affine" biasprm="2 0 0" ctrllimited="true" ctrlrange="0 1"/>'


def variant(name, before=None, after=None, edit=None):
    s = BASE.replace("<worldbody>", "<worldbody>" + RING, 1) if (before or after) else BASE
    if before:
        s = s.replace("<actuator>", "<actuator>" + before, 1)
    if after:
        s = s.replace("</actuator>", after + "</actuator>", 1)
    if edit:
        s = edit(s)
    p = tmp / f"{name}.xml"
    p.write_text(s)
    return str(p)


# 1. the setup against itself; a ring's magnet actuator after the robot's 56 passes and is listed
r0 = actuator_mapping_check(str(SETUP), reference=str(SETUP), expected_robot_nu=56)
r1 = actuator_mapping_check(variant("ring_after", after=MAG), reference=str(SETUP), expected_robot_nu=56)
check("setup passes; an object actuator after the robot's passes and is listed with its ctrl slot",
      r0["ok"] and r1["ok"] and r1["non_robot"] and r1["non_robot"][0]["index"] == 56 and r1["robot_ctrl_slice"] == [0, 56],
      f"{r0['issues']} {r1['issues']} {r1['non_robot']}")

# 2. a consumer that needs a robot-only ctrl, and an actuator inserted before the robot's (shifts every command)
r2 = actuator_mapping_check(variant("ring_after2", after=MAG), allow_non_robot=False)
r3 = actuator_mapping_check(variant("ring_before", before=MAG), reference=str(SETUP))
check("allow_non_robot=False fails on the magnet; a magnet inserted before the robot fails as a shift",
      not r2["ok"] and not r3["ok"] and "first block" in " ".join(r3["issues"]), f"{r2['issues']} {r3['issues']}")

# 3. robot actuators reordered (left hand first) differ from the setup's list
def swap(s):
    acts = re.findall(r"<general name=\"robot_[^>]*/>", s)
    right, left = [a for a in acts if "robot_right" in a], [a for a in acts if "robot_left" in a]
    for a in acts:
        s = s.replace(a, "", 1)
    return s.replace("<actuator>", "<actuator>" + "".join(left + right), 1)
r4 = actuator_mapping_check(variant("swapped", edit=swap), reference=str(SETUP))
check("robot actuators in another order than the setup's fail", not r4["ok"] and r4["reference"]["first_difference"] == 0, str(r4["issues"]))

# 4. a finger servo whose ctrlrange reaches past its joint range, and an actuator with no gain and no bias
def widen(s):
    return s.replace('joint="robot_right_right_index_PIP" ctrlrange="0 1.7453"', 'joint="robot_right_right_index_PIP" ctrlrange="-0.5 2.5"')
r5 = actuator_mapping_check(variant("wide", edit=widen))
dead = '<general name="ring_dead" joint="ring_hinge" gainprm="0" biasprm="0 0 0"/>'
r6 = actuator_mapping_check(variant("dead", after=dead, edit=lambda s: s.replace('<body name="ring_magnet">',
                                                                                 '<body name="ring_magnet"><joint name="ring_hinge" type="hinge"/>')))
check("a servo ctrlrange beyond its joint range and a dead actuator fail",
      not r5["ok"] and "beyond its joint" in " ".join(r5["issues"]) and not r6["ok"] and "no force" in " ".join(r6["issues"]),
      f"{r5['issues']} {r6['issues']}")

sys.exit(1 if fails else 0)
