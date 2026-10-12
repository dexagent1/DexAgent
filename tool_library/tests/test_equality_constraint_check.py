"""equality_constraint_check: satisfied, well-formed equalities pass; position-only mechanism welds, zero quaternions, welds
violated at the keyframe, a wrong start state and an episode that releases a weld the scene re-activates fail.

    python tool_library/tests/test_equality_constraint_check.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.equality_constraint_check import equality_constraint_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="equality_"))
# a bottle and its cap, the cap welded on (as the screw-cap assets do), and a fixture welded to the world
XML = ('<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
       '<body name="bottle" pos="0 0 .06"><freejoint/><geom type="cylinder" size=".03 .06" mass=".2"/></body>'
       '<body name="cap" pos="0 0 .13"><freejoint/><geom type="cylinder" size=".02 .01" mass=".01"/></body>'
       '<body name="rack" pos=".3 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass=".5"/></body>'
       '</worldbody><equality>'
       '<weld name="cap_on" body1="bottle" body2="cap" relpose="0 0 .07 1 0 0 0" torquescale="1" {active}/>'
       '<weld name="rack_fixed" body1="world" body2="rack" relpose=".3 0 .05 1 0 0 0" torquescale="{ts}"/>'
       '</equality>{key}</mujoco>')
good = XML.format(active="", ts="1", key="")

# 1. the setup's arm-hand welds hold at home; a well-formed object scene passes
r0 = equality_constraint_check(str(ROOT / "setup" / "scene.xml"))
r1 = equality_constraint_check(good)
check("setup arm-hand welds and a satisfied cap / fixture weld pass", r0["ok"] and r1["ok"], f"{r0['issues']} {r1['issues']}")

# 2. a fixture welded to the world with torquescale 0 (MjSpec's default data) holds position only; a zero relpose quaternion
r2 = equality_constraint_check(XML.format(active="", ts="0", key=""))
m = mujoco.MjModel.from_xml_string(good)
m.eq_data[1][6:10] = 0.0
r3 = equality_constraint_check(m)
check("a torquescale-0 world weld and a zero relative quaternion fail",
      not r2["ok"] and "torquescale 0" in r2["issues"][0] and not r3["ok"] and "quaternion" in r3["issues"][0], f"{r2['issues']} {r3['issues']}")

# 3. a keyframe that puts the cap 12 mm and 160 degrees off its weld, active; the same with the weld inactive passes
q = mujoco.MjData(mujoco.MjModel.from_xml_string(good)).qpos.copy()
q[7:10] = [0.0036, -0.0111, 0.13]
q[10:14] = [np.cos(np.radians(80)), 0, 0, np.sin(np.radians(80))]
key = '<keyframe><key qpos="' + " ".join(f"{v:.6g}" for v in q) + '"/></keyframe>'
r4 = equality_constraint_check(XML.format(active="", ts="1", key=key))
r5 = equality_constraint_check(XML.format(active='active="false"', ts="1", key=key))
check("a weld violated by 12 mm / 160 deg at the keyframe fails; inactive it passes",
      not r4["ok"] and any("12" in i or "11." in i for i in r4["issues"]) and r5["ok"], f"{r4['issues']} {r5['issues']}")

# 4. the intended start state: the cap weld is expected inactive but the scene starts it active
r6 = equality_constraint_check(good, expect_active={"cap_on": False})
r7 = equality_constraint_check(good, expect_active={"bottle|cap": True})
check("expect_active mismatch fails, a matching expectation (by body pair) passes", not r6["ok"] and r7["ok"], f"{r6['issues']} {r7['issues']}")

# 5. an episode that unscrews the cap: the recorded qpos takes it 20 cm away; without eq_active the scene re-engages the
#    weld on replay; with eq_active recorded (released from frame 10) it passes
Q = np.repeat(mujoco.MjData(mujoco.MjModel.from_xml_string(good)).qpos[None], 30, axis=0)
for t in range(10, 30):
    Q[t, 7:10] = [0.2 * (t - 9) / 20.0, 0.0, 0.13]
np.savez(tmp / "unscrew.npz", qpos=Q, fps=30.0)
ea = np.ones((30, 2), np.int8)
ea[10:, 0] = 0
np.savez(tmp / "unscrew_rec.npz", qpos=Q, fps=30.0, eq_active=ea, eq_names=np.array(["cap_on", "rack_fixed"]),
         eq_active_start=np.array([1, 1], np.int8))
r8 = equality_constraint_check(good, trajectory=str(tmp / "unscrew.npz"), stride=1)
r9 = equality_constraint_check(good, trajectory=str(tmp / "unscrew_rec.npz"), stride=1)
check("a released weld without recorded eq_active fails (first frame named); with eq_active recorded it passes",
      not r8["ok"] and r8["trajectory"]["released"][0]["first_frame"] in (10, 11) and r9["ok"], f"{r8['issues']} {r9['issues']}")

sys.exit(1 if fails else 0)
