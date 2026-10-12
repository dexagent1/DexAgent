"""joint_armature_damping_check: the setup's servos (armature 1, dampratio 1) pass; servos without armature, anti-damped servos,
joints that drop the setup's values and damping on a light free joint fail.

    python tool_library/tests/test_joint_armature_damping_check.py
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from tool_library.verifiers.joint_armature_damping_check import joint_armature_damping_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="armature_"))
SETUP = ROOT / "setup" / "scene.xml"
TXT = SETUP.read_text().replace('file="assets/', f'file="{ROOT}/setup/assets/')

# 1. the setup alone and against itself; the setup's fingers at the executor's armature 0.05 (expected dict) also pass
r0 = joint_armature_damping_check(str(SETUP))
r1 = joint_armature_damping_check(str(SETUP), reference=str(SETUP), expected={"robot_*_arm_dof_joint*": {"damping": 5.0}})
m = mujoco.MjModel.from_xml_path(str(SETUP))
for j in range(m.njnt):
    if "_index_" in m.joint(j).name:
        m.dof_armature[m.jnt_dofadr[j]] = 0.05
r1b = joint_armature_damping_check(m)
check("setup passes alone, against itself and with armature 0.05 on the index finger",
      r0["ok"] and r1["ok"] and r1b["ok"], f"{r0['issues']} {r1['issues']} {r1b['issues']}")

# 2. armature removed from every hand joint: w*dt up to ~24 on the fingertips, and the setup's armature is not kept
no_arm = re.sub(r'(<joint name="robot_(?:left|right)_(?:left|right)_[A-Za-z_]+"[^>]*?) armature="1"', r'\1 armature="0"', TXT)
no_arm = no_arm.replace('<joint armature="1" />', '<joint armature="0" />')
(tmp / "no_armature.xml").write_text(no_arm)
r2 = joint_armature_damping_check(str(tmp / "no_armature.xml"), reference=str(SETUP))
check("hand servos without armature fail on stability and on the reference",
      not r2["ok"] and r2["worst"]["omega_dt"] > 5 and any("armature" in i and "different" in i for i in r2["issues"]),
      str(r2["issues"])[:300])

# 3. a servo whose bias velocity term pumps energy (kv < 0)
m = mujoco.MjModel.from_xml_path(str(SETUP))
m.actuator_biasprm[10][2] = 5.0
r3 = joint_armature_damping_check(m)
check("an anti-damped servo fails", not r3["ok"] and "negative damping" in " ".join(r3["issues"]), str(r3["issues"]))

# 4. a value the expected dict asks for, and rotational damping on a 2 g free body
r4 = joint_armature_damping_check(str(SETUP), expected={"robot_*_arm_dof_joint*": {"damping": 2.0}})
light = ('<mujoco><worldbody><body name="tip" pos="0 0 .1"><joint type="free" damping="0.0002"/>'
         '<geom type="cylinder" size=".002 .01" mass=".002"/></body></worldbody></mujoco>')
r5 = joint_armature_damping_check(light)
check("a joint value differing from `expected` fails; free-joint damping on a light body fails",
      not r4["ok"] and not r5["ok"] and r5["free_joint_damping"][0]["ratio"] > 1, f"{r4['issues']} {r5['issues']}")

sys.exit(1 if fails else 0)
