"""Smoke test for object_inertia_plausibility_check: geometry-consistent inertia passes; point masses, flywheels and outside centres of mass fail."""
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


from tool_library.verifiers.object_inertia_plausibility_check import object_inertia_plausibility_check as check_inertia

CUBE_INF = CUBE
r = check_inertia(bench("inferred", CUBE_INF, CUBE_Q))
check("a cube whose inertia MuJoCo infers from its box passes", r["ok"] and abs(r["bodies"][0]["ratios"][0] - 1) < 0.01, r["bodies"])
point = ('<body name="cube" pos="0 0.5 0.025"><freejoint/><inertial pos="0 0 0" mass="0.1" diaginertia="1e-9 1e-9 1e-9"/>'
         '<geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0" rgba="0.8 0.2 0.2 1"/></body>')
r = check_inertia(bench("point", point, CUBE_Q))
check("a point-mass inertia on a 5 cm cube is caught", (not r["ok"]) and r["bodies"][0]["verdict"].startswith("point-like"), r["issues"])
huge = point.replace('diaginertia="1e-9 1e-9 1e-9"', 'diaginertia="0.01 0.01 0.01"')
r = check_inertia(bench("huge", huge, CUBE_Q))
check("a flywheel inertia on a 100 g cube is caught", (not r["ok"]) and r["bodies"][0]["verdict"].startswith("too large"), r["issues"])
off = point.replace('<inertial pos="0 0 0"', '<inertial pos="0.2 0 0"').replace('diaginertia="1e-9 1e-9 1e-9"', 'diaginertia="4.2e-5 4.2e-5 4.2e-5"')
r = check_inertia(bench("off", off, CUBE_Q))
check("a centre of mass 20 cm outside the cube is caught", (not r["ok"]) and "com outside" in r["bodies"][0]["verdict"], r["issues"])
cap = ('<body name="rod" pos="0 0.5 0.02"><freejoint/><geom name="rod_g" type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.02" mass="0.2"/>'
       '<body name="rod_tip" pos="0.12 0 0"><joint type="hinge" axis="0 0 1"/><geom name="tip_g" type="sphere" size="0.02" mass="0.05"/></body></body>')
r = check_inertia(bench("rod", cap, "0 0.5 0.02 1 0 0 0 0"))
check("a capsule with a jointed sphere part (two bodies) passes", r["ok"] and len(r["bodies"]) == 2, r["bodies"])
spring = ('<body name="sp" pos="0 0.5 0.02"><freejoint/><geom name="sp_g" type="box" size="0.02 0.02 0.02" mass="0.1"/>'
          '<body name="sp_coil" pos="0 0 0.03"><inertial pos="0 0 0" mass="0.0002" diaginertia="1e-12 1e-12 1e-12"/>'
          '<geom type="cylinder" size="0.005 0.01" contype="0" conaffinity="0" mass="0"/></body></body>')
r = check_inertia(bench("spring", spring, "0 0.5 0.02 1 0 0 0"))
check("a 0.2 g helper body (a spring coil) is skipped, not judged", r["ok"] and any(s["body"] == "sp_coil" for s in r["skipped"]), r)

print(f"{'OK' if fails == 0 else 'FAILED'}: object_inertia_plausibility_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
