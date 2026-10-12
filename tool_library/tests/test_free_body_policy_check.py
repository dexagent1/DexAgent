"""Smoke test for free_body_policy_check: free manipulated and fixed static objects pass; fixed, welded or mocap manipulated objects and free static ones fail."""
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

from tool_library.verifiers.free_body_policy_check import free_body_policy_check as check_free


def sem(role, oid="cube"):
    p = TMP / f"sem_{oid}_{role}.json"
    p.write_text(json.dumps({"objects": [{"id": oid, "name": f"red {oid}", "type": "rigid", "role": role, "properties": ["x"], "evidence": "x"}]}))
    return str(p)


FIXED = CUBE.replace('<freejoint name="cube_free"/>', '')
r = check_free(bench("free", CUBE, CUBE_Q), sem("manipulated"))
check("a manipulated object with a free joint passes", r["ok"] and r["objects"][0]["free"], r["objects"])
r = check_free(bench("fixed", FIXED), sem("manipulated"))
check("a manipulated object without a free joint is caught", (not r["ok"]) and r["objects"][0]["verdict"] == "fixed", r["issues"])
r = check_free(bench("welded", CUBE, CUBE_Q, lambda x: x.replace("</equality>", '<weld body1="cube" body2="table"/>\n</equality>', 1)), sem("tool"))
check("a tool welded to the table at the start is caught", (not r["ok"]) and r["objects"][0]["verdict"] == "held", r["issues"])
r = check_free(bench("welded_off", CUBE, CUBE_Q, lambda x: x.replace("</equality>", '<weld body1="cube" body2="table" active="false"/>\n</equality>', 1)), sem("tool"))
check("an inactive weld (switched on later) does not count", r["ok"], r["issues"])
r = check_free(bench("static_free", CUBE, CUBE_Q), sem("static"))
check("a static object left free is caught", (not r["ok"]) and r["objects"][0]["verdict"] == "free but static", r["issues"])
r = check_free(bench("static_free2", CUBE, CUBE_Q), sem("static"), static_may_be_free=True)
check("... unless static_may_be_free", r["ok"], r["issues"])
r = check_free(bench("static_fixed", FIXED), sem("static"))
check("a static fixed object passes", r["ok"], r["issues"])
MOCAP = CUBE.replace('<body name="cube" pos="0 0.5 0.025"><freejoint name="cube_free"/>', '<body name="cube" pos="0 0.5 0.025" mocap="true">')
r = check_free(bench("mocap", MOCAP), sem("manipulated"))
check("a manipulated mocap body is caught", (not r["ok"]) and r["objects"][0]["verdict"] in ("mocap", "fixed"), r["issues"])
CAP = ('<body name="bottle" pos="0.1 0.5 0.05"><freejoint/><geom name="bottle_g" type="cylinder" size="0.03 0.05" mass="0.3"/></body>'
       '<body name="cap" pos="0.1 0.5 0.11"><freejoint/><geom name="cap_g" type="cylinder" size="0.015 0.01" mass="0.01"/></body>')
p = bench("cap", CAP, "0.1 0.5 0.05 1 0 0 0 0.1 0.5 0.11 1 0 0 0", lambda x: x.replace("</equality>", '<weld body1="bottle" body2="cap"/>\n</equality>', 1))
r = check_free(p, sem("manipulated", "cap"))
check("a cap welded to its free bottle (a thread) is free with the bottle", r["ok"], r["issues"])

sem_rope = TMP / "sem_rope.json"
sem_rope.write_text(json.dumps({"objects": [{"id": "cube", "role": "manipulated"}, {"id": "rope", "role": "manipulated"}]}))
rec_rope = TMP / "rec_rope.json"
rec_rope.write_text(json.dumps({"objects": [{"id": "cube", "root_body": "cube"}, {"id": "rope", "type": "deformable", "skill": "build_isaac_scene", "root_body": "rope"}]}))
r = check_free(bench("isaac", CUBE, CUBE_Q), str(sem_rope), recon=str(rec_rope))
check("a rope simulated in Isaac Sim only (no MuJoCo body) is not reported missing", r["ok"] and r["objects"][1]["verdict"] == "simulated in Isaac Sim", r["objects"])

# a cord whose far end lies under a fixed tape dispenser: its weld is allowed only when scene_recon declares the anchor
CORD = ('<body name="disp" pos="-0.1 0.5 0.03"><geom name="disp_g" type="box" size="0.03 0.02 0.03"/></body>'
        '<body name="cord" pos="0.0 0.5 0.01"><freejoint/><geom name="cord_g" type="capsule" size="0.004 0.03" mass="0.01"/></body>')
weld = lambda x: x.replace("</equality>", '<weld body1="cord" body2="disp"/>\n</equality>', 1)
sem_c = TMP / "sem_cord.json"
sem_c.write_text(json.dumps({"objects": [{"id": "cord", "role": "manipulated"}, {"id": "disp", "role": "static"}]}))
rec_c = {"objects": [{"id": "cord", "type": "deformable", "root_body": "cord"}, {"id": "disp", "type": "rigid", "root_body": "disp"}]}
pc = bench("cord", CORD, "0.0 0.5 0.01 1 0 0 0", weld)
r = check_free(pc, str(sem_c), recon=rec_c)
check("a cord welded to a fixed dispenser without a declaration is held", (not r["ok"]) and r["objects"][0]["verdict"] == "held", r["issues"])
rec_c["objects"][0]["anchored_to"] = {"body": "disp", "reason": "the far end lies under the tape dispenser"}
r = check_free(pc, str(sem_c), recon=rec_c)
check("with anchored_to declared the weld is allowed and reported", r["ok"] and r["objects"][0]["anchored_to"] == {"disp": "the far end lies under the tape dispenser"}, r["issues"])
rec_c["objects"][0]["anchored_to"] = {"body": "table"}
r = check_free(bench("cord_table", CORD, "0.0 0.5 0.01 1 0 0 0", lambda x: x.replace("</equality>", '<weld body1="cord" body2="table"/>\n</equality>', 1)),
               str(sem_c), recon=rec_c)
check("an anchor declared on the table is refused", (not r["ok"]) and r["objects"][0]["verdict"] == "held", r["issues"])

print(f"{'OK' if fails == 0 else 'FAILED'}: free_body_policy_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
