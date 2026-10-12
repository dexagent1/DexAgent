"""mujoco_version_compat_check: a scene that compiles to the same model everywhere passes; an interpreter that cannot compile
it or that computes different masses fails. Stand-in interpreters exercise the comparison anywhere; when DEXAGENT_PYTHONS
names two real interpreters (e.g. MuJoCo 3.10 and 3.14) an attribute only the newer one knows is tested on them too.

    python tool_library/tests/test_mujoco_version_compat_check.py
"""
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common.compile_checks import default_pythons  # noqa: E402
from tool_library.verifiers.mujoco_version_compat_check import mujoco_version_compat_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="version_compat_"))
SETUP = str(ROOT / "setup" / "scene.xml")


def stand_in(name, body):
    """An 'interpreter' that runs the check's own compile script in this Python and then alters the result."""
    py = tmp / f"{name}.py"
    py.write_text("import json, sys\nns = {'__name__': 'stand_in'}\nexec(sys.argv[2], ns)\ns = ns['summarize'](sys.argv[3])\n"
                  + body + "\nprint('SUMMARY_JSON ' + json.dumps(s))\n")
    sh = tmp / name
    sh.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{py}" "$@"\n')
    sh.chmod(sh.stat().st_mode | stat.S_IEXEC)
    return str(sh)


same = stand_in("same", "s['version'] = s['version'] + '-copy'")
heavier = stand_in("heavier", "s['version'] = '9.9-other'\ns['small']['body_mass'] = [m * 1.5 for m in s['small']['body_mass']]")
older = stand_in("older", "s = {'ok': False, 'version': '0.0-old', 'error': \"XML Error: Schema violation: unrecognized attribute: 'adhesion'\"}")

# 1. one interpreter: nothing to compare, noted; two that agree: compared and passing
r0 = mujoco_version_compat_check(SETUP, pythons=[sys.executable])
r1 = mujoco_version_compat_check(SETUP, pythons=[sys.executable, same])
check("a single interpreter is noted; two agreeing interpreters pass", r0["ok"] and not r0["compared"] and r1["ok"] and r1["compared"],
      f"{r0} {r1['issues']}")

# 2. an interpreter whose MuJoCo computes 1.5x the masses (a changed mesh-inertia default) fails, naming body_mass
r2 = mujoco_version_compat_check(SETUP, pythons=[sys.executable, heavier])
check("different body masses under another interpreter fail", not r2["ok"] and any(d["field"] == "body_mass" for d in r2["differences"]),
      str(r2["issues"])[:300])

# 3. an interpreter that cannot compile the scene fails with its error
r3 = mujoco_version_compat_check(SETUP, pythons=[sys.executable, older])
check("an interpreter that refuses the scene fails", not r3["ok"] and "adhesion" in r3["issues"][0], str(r3["issues"]))

# 4. real interpreters when configured: geom 'adhesion' exists in newer MuJoCo only
pys = default_pythons()
if len(pys) >= 2 and all(Path(p).exists() for p in pys):
    (tmp / "adhesion.xml").write_text('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .1"><freejoint/>'
                                      '<geom type="box" size=".05 .05 .05" adhesion="0.5"/></body></worldbody></mujoco>')
    r4 = mujoco_version_compat_check(SETUP)
    r5 = mujoco_version_compat_check(str(tmp / "adhesion.xml"))
    versions = sorted({i["version"] for i in r5["interpreters"] if i["version"]})
    check(f"real interpreters {versions}: the setup agrees; an attribute only some versions know is caught",
          r4["ok"] and (not r5["ok"]) == (len({i["ok"] for i in r5["interpreters"]}) > 1),
          f"{r4['issues']} {r5['issues']}")
else:
    print("[SKIP] real-interpreter case (DEXAGENT_PYTHONS names fewer than two interpreters)")

# 5. misuse
try:
    mujoco_version_compat_check(SETUP, pythons=sys.executable)
    check("pythons given as a string raises ValueError", False)
except ValueError:
    check("pythons given as a string raises ValueError", True)

sys.exit(1 if fails else 0)
