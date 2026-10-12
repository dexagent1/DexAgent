"""Smoke test for physics_options_check: setup options pass; weaker gravity, coarse steps, disabled contacts, overrides and media fail."""
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


from tool_library.verifiers.physics_options_check import physics_options_check as check_opts

r = check_opts(str(SETUP / "scene.xml"))
check("the setup's own options pass", r["ok"] and r["scene"]["integrator"] == "implicitfast", r["issues"])
opt = '<option impratio="5" integrator="implicitfast" />'
r = check_opts(bench("fine", CUBE, CUBE_Q, lambda x: x.replace(opt, '<option timestep="0.00025" impratio="5" integrator="implicitfast" cone="elliptic" noslip_iterations="5"/>')))
check("a finer timestep and elliptic cones (stronger physics) pass", r["ok"], r["issues"])
r = check_opts(bench("grav", CUBE, CUBE_Q, lambda x: x.replace(opt, '<option impratio="5" integrator="implicitfast" gravity="0 0 -2"/>')))
check("lowered gravity is caught", (not r["ok"]) and any(v["option"] == "gravity" for v in r["violations"]), r["violations"])
r = check_opts(bench("coarse", CUBE, CUBE_Q, lambda x: x.replace(opt, '<option timestep="0.01" impratio="5" integrator="implicitfast"/>')))
check("a coarser timestep is caught", (not r["ok"]) and any(v["option"] == "timestep" for v in r["violations"]), r["violations"])
r = check_opts(bench("nocontact", CUBE, CUBE_Q, lambda x: x.replace(opt, '<option impratio="5" integrator="implicitfast"><flag contact="disable"/></option>')))
check("contacts switched off globally are caught", (not r["ok"]) and any(v["option"] == "disableflags" for v in r["violations"]), r["violations"])
r = check_opts(bench("override", CUBE, CUBE_Q, lambda x: x.replace(opt, '<option impratio="5" integrator="implicitfast" o_friction="3 3 1 0.1 0.1"><flag override="enable"/></option>')))
check("the global contact override is caught", (not r["ok"]) and any(v["option"] == "enableflags" for v in r["violations"]), r["violations"])
r = check_opts(bench("syrup", CUBE, CUBE_Q, lambda x: x.replace(opt, '<option impratio="5" integrator="implicitfast" viscosity="5"/>')))
check("a viscous medium is caught", (not r["ok"]) and any(v["option"] == "viscosity" for v in r["violations"]), r["violations"])
r = check_opts(bench("cube", CUBE, CUBE_Q), require_execution_contact=True)
check("a pyramidal-cone scene fails the executor's contact model", (not r["ok"]) and any(v["option"] == "cone" for v in r["violations"]), r["violations"])
r = check_opts(bench("euler", CUBE, CUBE_Q, lambda x: x.replace(opt, '<option impratio="10" integrator="Euler"/>')))
check("another integrator and impratio are caught", (not r["ok"]) and {v["option"] for v in r["violations"]} >= {"integrator", "impratio"}, r["violations"])

print(f"{'OK' if fails == 0 else 'FAILED'}: physics_options_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
