"""compile_determinism_check: a file compiles to the same model in this process and in a fresh one, and matches its recorded
hash; an edited mesh changes the hash, and an unseeded builder rebuilds a different scene.

    python tool_library/tests/test_compile_determinism_check.py
"""
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.compile_determinism_check import compile_determinism_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="determinism_"))

# 1. the setup compiles identically twice in-process and in a fresh interpreter elsewhere; its recorded hash matches
r0 = compile_determinism_check(str(ROOT / "setup" / "scene.xml"))
r1 = compile_determinism_check(str(ROOT / "setup" / "scene.xml"), fresh_process=False, expected_hash=r0["hash"])
check("setup is deterministic in- and out-of-process and matches its recorded hash",
      r0["ok"] and r0["fresh_process_hash"] == r0["hash"] and r1["ok"], f"{r0['issues']} {r1['issues']}")

# 2. a scene whose mesh file is rewritten after verification: same XML, different model, hash mismatch named
(tmp / "part.obj").write_text("v 0 0 0\nv .03 0 0\nv 0 .03 0\nv 0 0 .03\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
(tmp / "scene.xml").write_text('<mujoco><asset><mesh name="m" file="part.obj"/></asset><worldbody><body><freejoint/>'
                               '<geom type="mesh" mesh="m"/></body></worldbody></mujoco>')
h = compile_determinism_check(str(tmp / "scene.xml"), fresh_process=False)["hash"]
(tmp / "part.obj").write_text("v 0 0 0\nv .04 0 0\nv 0 .04 0\nv 0 0 .04\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
r2 = compile_determinism_check(str(tmp / "scene.xml"), expected_hash=h)
check("a mesh rewritten since the recorded hash fails (seen by the fresh interpreter, past MuJoCo's in-process asset cache)",
      not r2["ok"] and any("changed since" in i for i in r2["issues"]), str(r2["issues"]))

# 3. a builder that places a box with an unseeded random offset rebuilds a different scene; a seeded one does not
built = tmp / "built.xml"


def builder(seed=None):
    rng = random.Random(seed)
    def run():
        x = rng.uniform(-0.1, 0.1) if seed is None else random.Random(seed).uniform(-0.1, 0.1)
        built.write_text(f'<mujoco><worldbody><body pos="{x:.6f} 0 .05"><freejoint/><geom type="box" size=".02 .02 .02"/></body>'
                         '</worldbody></mujoco>')
    return run


builder(seed=1)()
r3 = compile_determinism_check(str(built), fresh_process=False, rebuild=builder(seed=None))
r4 = compile_determinism_check(str(built), fresh_process=False, rebuild=builder(seed=7))
check("an unseeded builder fails (differing arrays named); a seeded one passes",
      not r3["ok"] and "body_pos" in r3["rebuild"]["differing_arrays"] and r4["ok"], f"{r3['issues']} {r4['issues']}")

# 4. a builder given as a command line, run twice
script = tmp / "build.py"
script.write_text(f"import random\nopen({str(built)!r}, 'w').write('<mujoco><worldbody><body pos=\"' + str(random.random()) + "
                  "' 0 .05\"><freejoint/><geom type=\"box\" size=\".02 .02 .02\"/></body></worldbody></mujoco>')\n")
r5 = compile_determinism_check(str(built), fresh_process=False, rebuild=[sys.executable, str(script)])
check("an unseeded command-line builder fails too", not r5["ok"] and r5["rebuild"]["differing_arrays"], str(r5["issues"]))

# 5. misuse
try:
    compile_determinism_check("<mujoco/>")
    check("an XML string raises ValueError", False)
except ValueError:
    check("an XML string raises ValueError", True)

sys.exit(1 if fails else 0)
