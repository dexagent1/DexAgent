"""solver_parameter_sanity_check: consistent solref/solimp pass; time constants under 2 x timestep, mixed-sign solref, zero
damping ratio and a gram-scale part resting deeper than its own size fail.

    python tool_library/tests/test_solver_parameter_sanity_check.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.solver_parameter_sanity_check import solver_parameter_sanity_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


def scene(solref="0.02 1", disk_solref=None, dt="0.002", mass="0.2", size=".03 .03 .03"):
    ds = f' solref="{disk_solref}"' if disk_solref else ""
    return (f'<mujoco><option timestep="{dt}"/><worldbody><geom name="floor" type="plane" size="1 1 .1" solref="{solref}"/>'
            f'<body name="part" pos="0 0 .05"><freejoint/><geom name="part_g" type="box" size="{size}" mass="{mass}"{ds}/></body>'
            '</worldbody></mujoco>')


# 1. the setup and a 200 g box on default contacts pass
r0 = solver_parameter_sanity_check(str(ROOT / "setup" / "scene.xml"))
r1 = solver_parameter_sanity_check(scene())
check("setup and a 200 g box on default contacts pass", r0["ok"] and r1["ok"], f"{r0['issues']} {r1['issues']}")

# 2. a 1 ms time constant at a 2 ms timestep is silently raised to 4 ms; at a 0.25 ms timestep it is honoured
r2 = solver_parameter_sanity_check(scene(solref="0.001 1"))
r3 = solver_parameter_sanity_check(scene(solref="0.001 1", dt="0.00025"))
check("solref 1 ms at dt 2 ms fails (clamped); at dt 0.25 ms it passes",
      not r2["ok"] and "silently" in r2["issues"][0] and r3["ok"], f"{r2['issues']} {r3['issues']}")

# 3. a time constant negated by accident (-0.02, 1) and a zero damping ratio
r4 = solver_parameter_sanity_check(scene(solref="-0.02 1"))
r5 = solver_parameter_sanity_check(scene(solref="0.02 0"))
check("mixed-sign solref and a zero damping ratio fail",
      not r4["ok"] and "mixed-sign" in r4["issues"][0] and not r5["ok"] and "damping ratio" in r5["issues"][0], f"{r4['issues']} {r5['issues']}")

# 4. a 2 g plate 3 mm thick on soft 80 ms contacts rests about 1.7 mm deep, more than its half-thickness (the depth does
#    not shrink with the mass); on 4 ms contacts it stays on its face
thin = dict(mass="0.002", size=".01 .01 .0015")
r6 = solver_parameter_sanity_check(scene(solref="0.08 1", disk_solref="0.08 1", **thin))
r7 = solver_parameter_sanity_check(scene(solref="0.004 1", disk_solref="0.004 1", **thin))
r8 = solver_parameter_sanity_check(scene(solref="0.08 1", disk_solref="0.08 1", mass="0.5", size=".03 .03 .03"))
check("a gram-scale plate sinking past its half-thickness fails; short contacts pass; a 500 g box on the same soft contacts "
      "is not judged (above light_mass_kg)",
      not r6["ok"] and "rests" in r6["issues"][-1] and r7["ok"] and r8["ok"],
      f"{r6['issues']} | {r6['light_objects']} | {r7['issues']} {r7['light_objects']} | {r8['issues']}")

sys.exit(1 if fails else 0)
