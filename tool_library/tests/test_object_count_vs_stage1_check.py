"""Smoke test for object_count_vs_stage1_check: matching slot, latch and cup counts pass; a 2-slot holder for a 3-slot one, a missing latch or cup fail; unknowable counts stay unverified."""
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


import json

from tool_library.verifiers.object_count_vs_stage1_check import object_count_vs_stage1_check as check_count


def scene(name, body):
    p = TMP / f"{name}.xml"
    p.write_text(f'<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>{body}</worldbody></mujoco>')
    return str(p)


def write(name, obj):
    p = TMP / name
    p.write_text(json.dumps(obj))
    return str(p)


HOLDER = '<body name="holder_holder" pos="0 0.5 0.01"><freejoint/><geom name="holder_g" type="box" size="0.03 0.03 0.01" mass="0.02"/></body>'
sem_h = write("sem_h.json", {"task_prompt": "insert the battery", "objects": [{"id": "holder", "name": "black battery holder", "role": "container",
            "appearance": {"housing": "black plastic tray with three side-by-side cylindrical slots"}, "properties": ["Each cell lies in one slot."]}]})
rec3 = write("rec3.json", {"objects": [{"id": "holder", "root_body": "holder_holder", "params": {"n_slots": 3}}]})
rec2 = write("rec2.json", {"objects": [{"id": "holder", "root_body": "holder_holder", "params": {"n_slots": 2}}]})
r = check_count(scene("holder", HOLDER), sem_h, rec3)
check("a 3-slot holder built with n_slots=3 passes", r["ok"] and r["counts"][0]["scene"] == 3, r["counts"])
r = check_count(scene("holder", HOLDER), sem_h, rec2)
check("a 3-slot holder that came out 2-slot is caught", (not r["ok"]) and r["counts"][0]["verdict"] == "mismatch", r["issues"])


def lid(n):
    parts = "".join(f'<body name="mb_latch_{s}" pos="0 0 0"><joint type="hinge" axis="1 0 0"/><geom name="mb_latch_{s}_g" type="box" size="0.01 0.01 0.001" mass="0.001"/></body>'
                    for s in ["xp", "xn", "yp", "yn"][:n])
    return f'<body name="mb_lid" pos="0 0.5 0.05"><freejoint/><geom name="mb_lid_g" type="box" size="0.1 0.1 0.005" mass="0.1"/>{parts}</body>'


sem_l = write("sem_l.json", {"objects": [{"id": "lid", "name": "clear lid", "role": "manipulated", "parts": ["lid_panel", "latch_left", "latch_right", "latch_near", "latch_far"],
                                          "properties": ["Four latch flaps, one on each edge, fold down over the rim."]}]})
rec_l = write("rec_l.json", {"objects": [{"id": "lid", "root_body": "mb_lid"}]})
r = check_count(scene("lid4", lid(4)), sem_l, rec_l)
check("a lid with its four latch bodies passes (flap ~ latch)", r["ok"] and {c["noun"]: c["scene"] for c in r["counts"]}.get("latch") == 4, r["counts"])
r = check_count(scene("lid3", lid(3)), sem_l, rec_l)
check("a lid with three latches where Stage 1 says four is caught", not r["ok"], r["issues"])
CUPS = "".join(f'<body name="cup{i}_cup" pos="{0.1 * i} 0.5 0.03"><freejoint/><geom name="cup{i}_g" type="cylinder" size="0.03 0.03" mass="0.05"/></body>' for i in range(3))
sem_c = write("sem_c.json", {"task_prompt": "Stack the three cups.", "task_summary": "The right hand stacks the three cups into one tower.",
                             "objects": [{"id": f"cup{i}", "name": f"blue cup {i}", "role": "manipulated", "properties": ["Stacks into the two other cups."]} for i in range(3)]})
r = check_count(scene("cups3", CUPS), sem_c)
check("three cups in the task and three cups in the scene pass", r["ok"] and any(c["object"] == "task" and c["scene"] == 3 for c in r["counts"]), r["counts"])
r = check_count(scene("cups2", CUPS.replace('<body name="cup2_cup" pos="0.2 0.5 0.03"><freejoint/><geom name="cup2_g" type="cylinder" size="0.03 0.03" mass="0.05"/></body>', "")), sem_c)
check("a cup missing from the scene is caught against the task's count", (not r["ok"]) and any(c["object"] == "task" and c["scene"] == 2 for c in r["counts"]), r["issues"])
sem_u = write("sem_u.json", {"objects": [{"id": "rack", "name": "tube rack", "role": "container", "properties": ["Holds twelve wells in two rows; a 3.5-inch drive bay."]}]})
r = check_count(scene("rack", HOLDER.replace("holder_holder", "rack_rack")), sem_u)
check("a count the scene cannot show is unverified, not failed (and '3.5-inch drive' is not 5 drives)",
      r["ok"] and r["unverified"] and not any(c["noun"] == "drive" for c in r["counts"]), r["counts"])

print(f"{'OK' if fails == 0 else 'FAILED'}: object_count_vs_stage1_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
