"""Smoke test for object_mass_plausibility_check: plausible densities pass; weightless, lead-heavy, wrong-material and unliftable objects fail."""
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

from tool_library.verifiers.object_mass_plausibility_check import object_mass_plausibility_check as check_mass


def box(name, half, mass, extra=""):
    return (f'<body name="{name}" pos="0 0.5 {half}"><freejoint/><geom name="{name}_g" type="box" size="{half} {half} {half}" '
            f'mass="{mass}" rgba="0.5 0.6 0.7 1" {extra}/></body>'), f"0 0.5 {half} 1 0 0 0"


o, q = box("cube", 0.025, 0.1)
r = check_mass(bench("ok", o, q))
check("a 100 g, 5 cm cube (800 kg/m^3) is plausible", r["ok"] and abs(r["objects"][0]["bulk_density"] - 800) < 1, r["objects"])
o, q = box("cube", 0.025, 1e-5)
r = check_mass(bench("light", o, q))
check("a 0.01 g cube is flagged too light", (not r["ok"]) and r["objects"][0]["verdict"] == "too light", r["issues"])
o, q = box("cube", 0.025, 20.0)
r = check_mass(bench("heavy", o, q))
check("a 20 kg, 5 cm cube (lead-heavy) is flagged too heavy", (not r["ok"]) and r["objects"][0]["verdict"] == "too heavy", r["issues"])
o, q = box("cube", 0.025, 0.1)
r = check_mass(bench("foam", o, q), materials={"cube": "foam"})
check("the same cube as foam (max 300 kg/m^3) is flagged", (not r["ok"]) and r["objects"][0]["material"] == ["foam"], r["issues"])
r = check_mass(bench("metal", o, q), materials={"cube": "steel"})
check("the same cube as steel passes", r["ok"], r["issues"])

# Stage 1 roles and appearance: a manipulated object heavier than a hand can lift; a hollow cardboard box modelled thin
o2 = ('<body name="crate" pos="0 0.6 0.1"><freejoint/>'
      '<geom name="crate_floor" type="box" size="0.15 0.15 0.002" pos="0 0 -0.098" mass="4"/>'
      '<geom name="crate_wall" type="box" size="0.15 0.002 0.1" pos="0 0.148 0" mass="1"/></body>')
sem = TMP / "sem.json"
sem.write_text(json.dumps({"objects": [{"id": "crate", "name": "red cardboard crate", "role": "manipulated", "appearance": {"body": "red cardboard"}}]}))
rec = TMP / "rec.json"
rec.write_text(json.dumps({"objects": [{"id": "crate", "root_body": "crate", "bodies": ["crate"]}]}))
r = check_mass(bench("crate", o2, "0 0.6 0.1 1 0 0 0"), recon=str(rec), semantic=str(sem))
row = r["objects"][0]
check("a 5 kg manipulated cardboard crate is flagged (above the lift limit and the cardboard band)",
      (not r["ok"]) and row["material"] == ["cardboard"] and row["role"] == "manipulated" and row["verdict"] == "too heavy to manipulate", r["issues"])
o3 = o2.replace('mass="4"', 'mass="0.2"').replace('mass="1"', 'mass="0.05"')
r = check_mass(bench("crate_ok", o3, "0 0.6 0.1 1 0 0 0"), recon=str(rec), semantic=str(sem))
check("a 250 g cardboard crate passes", r["ok"], r["issues"])
try:
    check_mass(bench("ok2", *box("cube", 0.025, 0.1)), materials={"cube": "unobtainium"})
    check("an unknown material raises ValueError", False)
except ValueError as e:
    check("an unknown material raises ValueError naming the parameter", "materials" in str(e), e)

print(f"{'OK' if fails == 0 else 'FAILED'}: object_mass_plausibility_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
