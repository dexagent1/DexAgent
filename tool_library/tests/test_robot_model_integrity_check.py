"""Smoke test for robot_model_integrity_check: the setup's robot passes, deleted / shrunk / lighter / masked robot parts fail."""
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


from tool_library.verifiers.robot_model_integrity_check import robot_model_integrity_check as check_robot

# 1. the setup itself and a bench with an object are the setup's robot
r = check_robot(str(SETUP / "scene.xml"))
check("the setup's own scene is intact", r["ok"] and r["n_bodies"] > 50 and r["n_geoms"] > 50, r["issues"])
r = check_robot(bench("cube", CUBE, CUBE_Q))
check("a bench with an object added keeps the robot intact", r["ok"], r["issues"])

# 2. a shrunk finger collision capsule is named
p = bench("shrunk", CUBE, CUBE_Q, lambda x: x.replace(
    'name="robot_right_collision_hand_right_index_2" size="0.01 0.0235"', 'name="robot_right_collision_hand_right_index_2" size="0.006 0.0235"'))
r = check_robot(p)
check("a shrunk finger collision capsule is caught", (not r["ok"]) and any(c["name"] == "robot_right_collision_hand_right_index_2" and c["field"] == "size"
                                                                             for c in r["changed"]), r["changed"][:3])

# 3. a lighter hand
p = bench("light", CUBE, CUBE_Q, lambda x: x.replace('mass="0.72045"', 'mass="0.30000"', 1))
r = check_robot(p)
check("a lighter palm is caught as a mass change", (not r["ok"]) and any(c["field"] == "mass" for c in r["changed"]) and r["worst"] > 0.5, r["changed"][:3])

# 4. a deleted collision geom and a collision switched off
p = bench("deleted", CUBE, CUBE_Q, lambda x: re.sub(r'\s*<pair geom1="g_table" geom2="robot_left_collision_hand_left_palm_0"[^>]*/>', "",
                                                    re.sub(r'\s*<geom name="robot_left_collision_hand_left_palm_0"[^>]*/>', "", x, count=1), count=1))
r = check_robot(p)
check("a deleted palm collision box is reported missing", (not r["ok"]) and any("robot_left_collision_hand_left_palm_0" in m for m in r.get("missing", [])), r["issues"][:3])
p = bench("novisual", CUBE, CUBE_Q, lambda x: re.sub(r'\s*<geom name="robot_right_right_hand_C_MC_visual"[^>]*/>', "", x, count=1))
r = check_robot(p)
check("a deleted visual palm mesh is reported missing", (not r["ok"]) and any("robot_right_right_hand_C_MC_visual" in m for m in r.get("missing", [])), r["issues"][:3])
p = bench("contype", CUBE, CUBE_Q, lambda x: x.replace('name="robot_right_collision_hand_right_index_2" size="0.01 0.0235" pos="0.0235 0 0" quat="0.707107 0 0.707107 0" type="capsule" contype="0"',
                                                       'name="robot_right_collision_hand_right_index_2" size="0.01 0.0235" pos="0.0235 0 0" quat="0.707107 0 0.707107 0" type="capsule" contype="4"'))
r = check_robot(p)
check("a changed collision bit is caught", (not r["ok"]) and any(c["field"] == "contype" for c in r["changed"]), r["changed"][:3])

# 5. the hand-built twins' naming (arm_right_link1) matches by name tokens; scenes written before the hand hulls pass
p = bench("renamed", CUBE, CUBE_Q, lambda x: x.replace("robot_right_arm_", "arm_right_").replace("robot_left_arm_", "arm_left_"))
r = check_robot(p)
check("arm bodies renamed arm_<side>_* still match the setup's robot", r["ok"], r["issues"][:2])
p = bench("nohulls", CUBE, CUBE_Q, lambda x: re.sub(r'\s*<geom name="robot_(left|right)_collision_hand_[a-z]+_[A-Za-z_]+_hull"[^>]*/>', "", x))
r = check_robot(p)
check("a scene without the hand hulls (the executor adds them) passes", r["ok"] and len(r["hulls_absent"]) == 6, (r["issues"][:2], r["hulls_absent"]))

# 6. an extra robot collision geom and a scaled hand
p = bench("extra", CUBE, CUBE_Q, lambda x: x.replace('<site name="robot_right_right_palm"',
                                                     '<geom name="robot_right_extra_pad" type="box" size="0.05 0.05 0.01" contype="1" conaffinity="1"/>\n<site name="robot_right_right_palm"', 1))
r = check_robot(p)
check("an added robot collision geom is reported", (not r["ok"]) and any("extra_pad" in e for e in r["extra"]), r["extra"])

print(f"{'OK' if fails == 0 else 'FAILED'}: robot_model_integrity_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
