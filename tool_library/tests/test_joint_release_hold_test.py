import os
import sys
import tempfile

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.joint_release_hold_test import joint_release_hold_test

LID = """<mujoco><option timestep="0.002"/>
<worldbody>
 <body name="base"><freejoint/>
  <geom type="box" size="0.10 0.07 0.002" pos="0 0 0.002" mass="0.1" friction="1 0.02 0.001"/>
  <body name="lid" pos="0 0.07 0.06">
   <joint name="lid" type="hinge" axis="-1 0 0" range="-5 100" frictionloss="{fl}" armature="{arm}" damping="0.02"/>
   <geom type="box" size="0.10 0.07 0.0015" pos="0 -0.07 0" mass="0.02" contype="0" conaffinity="0"/>
  </body>
 </body>
</worldbody></mujoco>"""

SLIDE = """<mujoco><option timestep="0.002"/>
<worldbody>
 <geom type="plane" size="1 1 0.1"/>
 <body name="frame" pos="0 0 0.3">
  <body name="drawer"><joint name="drawer" type="slide" axis="0 0.5 -0.866" range="0 0.2" frictionloss="{fl}" armature="0.5"/>
   <geom type="box" size="0.05 0.05 0.02" mass="0.2" contype="0" conaffinity="0"/>
  </body>
 </body>
</worldbody></mujoco>"""


def write(txt):
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    f.write(txt)
    f.close()
    return f.name


fails = 0


def case(name, cond):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name)
    fails += 0 if cond else 1


# 1: a stiff crease holds at mid travel and upright (floor added automatically)
r = joint_release_hold_test(write(LID.format(fl=0.03, arm=0.005)), "lid", [45, 90], hold_s=2.0)
case("stiff hinge lid holds at 45 and 90 deg", r["ok"] and r["joint_type"] == "hinge")

# 2: a frictionless lid falls shut from 45 deg -> reported, with an actionable issue
r = joint_release_hold_test(write(LID.format(fl=0.0, arm=0.0)), "lid", [45], hold_s=2.0)
case("frictionless lid detected falling", (not r["ok"]) and "frictionloss" in " ".join(r["issues"]))

# 3: an inclined slide with enough friction holds; without, it runs to the stop
r1 = joint_release_hold_test(write(SLIDE.format(fl=40.0)), "drawer", [0.1], hold_s=1.0)
r2 = joint_release_hold_test(write(SLIDE.format(fl=0.0)), "drawer", [0.1], hold_s=1.0)
case("slide joint hold vs slip", r1["ok"] and r1["joint_type"] == "slide" and not r2["ok"])

# 4: out-of-range position and unknown joint are reported, not raised
r = joint_release_hold_test(write(LID.format(fl=0.03, arm=0.005)), "lid", [150], hold_s=0.5)
r2 = joint_release_hold_test(write(LID.format(fl=0.03, arm=0.005)), "nope", [10], hold_s=0.5)
case("out of range and missing joint reported", (not r["ok"]) and "outside" in r["issues"][0] and not r2["ok"])

# 5: misuse raises
try:
    joint_release_hold_test(write(LID.format(fl=0.03, arm=0.005)), "lid", [], hold_s=0.5)
    case("empty positions raises", False)
except ValueError:
    case("empty positions raises", True)

sys.exit(1 if fails else 0)
