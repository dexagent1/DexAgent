"""Smoke test for contact_exclusion_audit: the setup's contact passes; excludes, missing or weakened pairs, masked bits and glued objects fail."""
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


from tool_library.verifiers.contact_exclusion_audit import contact_exclusion_audit as audit

r = audit(bench("cube", CUBE, CUBE_Q))
check("a bench with an object and the setup's hand-table pairs passes", r["ok"] and r["table_pairs"]["present"] == r["table_pairs"]["expected"] == 32, r["issues"])

p = bench("excl", CUBE, CUBE_Q, lambda x: x.replace("<contact>", '<contact>\n    <exclude body1="robot_right_right_index_DP" body2="cube"/>', 1))
r = audit(p)
check("an exclude between a finger and the object is caught", (not r["ok"]) and any(e["verdict"] == "robot-object" for e in r["excludes"]), r["excludes"])

p = bench("nopair", CUBE, CUBE_Q, lambda x: re.sub(r'\s*<pair geom1="g_table" geom2="robot_left_collision_hand_left_palm_0"[^>]*/>', "", x, count=1))
r = audit(p)
check("a missing hand-table pair is caught", (not r["ok"]) and r["table_pairs"]["missing"], r["table_pairs"])

p = bench("gap", CUBE, CUBE_Q, lambda x: x.replace("</contact>", '<pair geom1="robot_right_collision_hand_right_index_2" geom2="cube_geom" margin="0" gap="0.03"/>\n'
                                                   '<pair geom1="robot_right_collision_hand_right_middle_2" geom2="cube_geom" solref="0.5 1"/>\n</contact>', 1))
r = audit(p)
check("robot-object pairs with a gap or a spongy solref are caught", (not r["ok"]) and len(r["bad_pairs"]) == 2, r["bad_pairs"])
ok_pair = bench("okpair", CUBE, CUBE_Q, lambda x: x.replace("</contact>", '<pair geom1="robot_right_collision_hand_right_index_2" geom2="cube_geom" solref="0.006 1" solimp="0.99 0.999 0.001 0.5 2" condim="6"/>\n</contact>', 1))
r = audit(ok_pair)
check("a robot-object pair with the executor's parameters passes", r["ok"] and r["robot_pairs_checked"] == 1, r["issues"])

GLUED = ('<body name="glued" pos="0.05 0 0"><geom name="glued_geom" type="box" size="0.01 0.01 0.01" mass="0.01" rgba="0 0 1 1"/></body>')
p = bench("parented", CUBE, CUBE_Q, lambda x: x.replace('<site name="robot_right_right_palm"', GLUED + '\n<site name="robot_right_right_palm"', 1))
r = audit(p)
check("an object nested under a hand body is caught", (not r["ok"]) and any("glued" in s for s in r["parented_objects"]), r["parented_objects"])
p = bench("weld", CUBE, CUBE_Q, lambda x: x.replace("</equality>", '<weld name="grab" body1="robot_right_right_hand_C_MC" body2="cube" active="false"/>\n</equality>', 1))
r = audit(p)
check("a weld between the hand and the object (even inactive) is caught", (not r["ok"]) and r["robot_object_equalities"], r["robot_object_equalities"])

# another setup folder whose arm has collision geoms: object bits the arm cannot see, a table with other bits
MINI = TMP / "mini_setup"
MINI.mkdir(exist_ok=True)
SCENE = """<mujoco model="mini">
  <worldbody>
    <body name="table" pos="0 0.5 -0.02"><geom name="g_table" type="box" size="0.5 0.5 0.02" contype="1" conaffinity="1"/></body>
    <body name="robot_left_arm_root" pos="0 0 0">
      <geom name="robot_left_arm_link" type="capsule" fromto="0 0 0 0 0.3 0.2" size="0.03" contype="1" conaffinity="1"/>
      <body name="robot_left_left_hand" pos="0 0.3 0.2"><joint name="robot_left_left_pos_x" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="robot_left_collision_hand_left_palm_0" type="box" size="0.02 0.04 0.01" contype="0" conaffinity="0" mass="0.3"/></body>
    </body>
    OBJ
  </worldbody>
  <contact><pair geom1="g_table" geom2="robot_left_collision_hand_left_palm_0"/></contact>
  <keyframe><key name="home" qpos="0 KEY"/></keyframe>
</mujoco>"""
(MINI / "scene.xml").write_text(SCENE.replace("OBJ", "").replace(" KEY", ""))
(MINI / "setup.yaml").write_text("name: mini\nscene: scene.xml\ntable:\n  body: table\n  geom: g_table\n")
p = TMP / "mini_ok.xml"; p.write_text(SCENE.replace("OBJ", CUBE).replace("KEY", CUBE_Q))
r = audit(str(p), setup=str(MINI))
check("another setup folder: an ordinary object passes", r["ok"], r["issues"])
bad = SCENE.replace('contype="1" conaffinity="1"/></body>\n    <body name="robot_left_arm_root"', 'contype="3" conaffinity="3"/></body>\n    <body name="robot_left_arm_root"', 1)
p = TMP / "mini_bad.xml"; p.write_text(bad.replace("OBJ", CUBE.replace('mass="0.1"', 'mass="0.1" contype="2" conaffinity="2"')).replace("KEY", CUBE_Q))
r = audit(str(p), setup=str(MINI))
check("another setup folder: an object the arm cannot collide with and changed table bits are caught",
      (not r["ok"]) and r["masked_geoms"] == ["cube_geom"] and any("table geom" in i for i in r["issues"]), r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: contact_exclusion_audit ({fails} failure(s))")
sys.exit(1 if fails else 0)
