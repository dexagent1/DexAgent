"""Smoke test for robot_joint_range_check: setup ranges pass; widened, unlimited and narrowed robot joints fail."""
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


from tool_library.verifiers.robot_joint_range_check import robot_joint_range_check as check_ranges

r = check_ranges(str(SETUP / "scene.xml"))
check("the setup's own ranges pass", r["ok"] and r["n_joints"] >= 68, r["issues"])
r = check_ranges(bench("cube", CUBE, CUBE_Q))
check("a bench with an object passes", r["ok"], r["issues"])

# the hand-built giftbox twin widened its arm ranges to fit its trajectory
p = bench("widened", CUBE, CUBE_Q, lambda x: x.replace('name="robot_right_arm_dof_joint2" pos="0 0 0" axis="0 1 0" range="-0.15 3.14159"',
                                                       'name="robot_right_arm_dof_joint2" pos="0 0 0" axis="0 1 0" range="-0.6 3.3"'))
r = check_ranges(p)
row = next((c for c in r["changed"] if c["joint"] == "robot_right_arm_dof_joint2"), None)
check("a widened arm joint range is caught with its widening", (not r["ok"]) and row is not None and abs(row["widened_by"] - (0.45 + 0.15841)) < 1e-3, r["changed"])

p = bench("unlimited", CUBE, CUBE_Q, lambda x: re.sub(r'(name="robot_left_left_index_PIP"[^>]*?) range="[^"]*"', r'\1 limited="false"', x, count=1))
r = check_ranges(p)
check("a finger joint made unlimited is caught", (not r["ok"]) and any(c["joint"] == "robot_left_left_index_PIP" and c["widened_by"] == "unlimited" for c in r["changed"]), r["changed"])

p = bench("narrowed", CUBE, CUBE_Q, lambda x: x.replace('name="robot_right_arm_dof_joint2" pos="0 0 0" axis="0 1 0" range="-0.15 3.14159"',
                                                        'name="robot_right_arm_dof_joint2" pos="0 0 0" axis="0 1 0" range="0 3.14159"'))
r = check_ranges(p)
check("a narrowed range (the vendor 0 instead of the setup's -0.15) is reported too", not r["ok"] and r["changed"][0]["scene"][0] == 0.0, r["changed"])

p = bench("renamed", CUBE, CUBE_Q, lambda x: x.replace("robot_right_arm_", "arm_right_").replace("robot_left_arm_", "arm_left_"))
r = check_ranges(p)
check("renamed arm joints (arm_right_dof_joint2) are matched and pass", r["ok"] and not r["missing"], r["issues"][:2])

print(f"{'OK' if fails == 0 else 'FAILED'}: robot_joint_range_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
