"""inertia_validity_check: real inertias pass; placeholder, impossible and ill-conditioned ones fail.

    python tool_library/tests/test_inertia_validity_check.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from tool_library.verifiers.inertia_validity_check import inertia_validity_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


def box(inertial=""):
    return ('<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
            f'<body name="box" pos="0 0 .05"><freejoint/>{inertial}<geom type="box" size=".05 .05 .05" mass="0.3"/></body>'
            '</worldbody></mujoco>')


# 1. the setup (hand virtual links included) and a 10 cm box with computed inertia pass
r0 = inertia_validity_check(str(ROOT / "setup" / "scene.xml"))
r1 = inertia_validity_check(box())
check("setup scene and a box with geometry-derived inertia pass", r0["ok"] and r1["ok"], f"{r0['issues']} {r1['issues']}")

# 2. URDF-style placeholders on a free 300 g, 10 cm box: far too small and far too large
small = inertia_validity_check(box('<inertial pos="0 0 0" mass="0.3" diaginertia="1e-9 1e-9 1e-9"/>'))
large = inertia_validity_check(box('<inertial pos="0 0 0" mass="0.3" diaginertia="2 2 2"/>'))
check("placeholder inertias (1e-9 and 2 kg m^2 on a 300 g box) fail with the radius of gyration named",
      not small["ok"] and "far too small" in small["issues"][0] and not large["ok"] and "far too large" in large["issues"][0],
      f"{small['issues']} {large['issues']}")

# 3. an inertia edited at run time that breaks the triangle inequality, and a zero-mass free body
m = mujoco.MjModel.from_xml_string(box())
m.body_inertia[1] = [1e-4, 1e-4, 1e-3]
tri = inertia_validity_check(m)
m2 = mujoco.MjModel.from_xml_string(box())
m2.body_mass[1] = 0.0
zero = inertia_validity_check(m2)
check("a triangle-inequality violation and a massless moving body fail",
      not tri["ok"] and "triangle" in " ".join(tri["issues"]) and not zero["ok"] and "mass" in " ".join(zero["issues"]),
      f"{tri['issues']} {zero['issues']}")

# 4. a hinge joint between a 10 kg base and a 1e-6 kg flap is ill-conditioned; a 50 g flap is fine
lever = ('<mujoco><worldbody><body name="base" pos="0 0 .1"><freejoint/><geom type="box" size=".1 .1 .02" mass="10"/>'
         '<body name="flap" pos=".1 0 0"><joint name="h" type="hinge" axis="0 1 0"/>'
         '<geom type="box" size=".02 .02 .002" mass="{m}"/></body></body></worldbody></mujoco>')
bad = inertia_validity_check(lever.format(m="1e-6"))
ok = inertia_validity_check(lever.format(m="0.05"))
check("a 1e7 mass ratio across a joint fails; 200:1 passes", not bad["ok"] and bad["mass_ratio_violations"] and ok["ok"],
      f"{bad['issues']} {ok['issues']}")

sys.exit(1 if fails else 0)
