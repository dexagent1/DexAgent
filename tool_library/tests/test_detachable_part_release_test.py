"""Variations for detachable_part_release_test: three detachable parts that must pass (pen cap on a free pen,
press-fit lid on a fixed jar, plug on a negative-axis slide) and four faults that must be reported or refused (a press fit tighter than the pull, a soft press fit that creeps under gravity, a missing weld, a zero release travel)."""
import os
import sys
import tempfile

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.detachable_part_release_test import detachable_part_release_test as T


STIFF = ' solreffriction="0.004 1" solimpfriction="0.999 0.9999 0.001"'


def pen(hold=1.5, eq="cap_engaged", fr=STIFF):
    return f"""<mujoco><option timestep="0.002" integrator="implicitfast" cone="elliptic"/><worldbody>
 <geom name="floor" type="plane" size="1 1 0.1"/>
 <body name="pen" pos="0 0 0.008"><freejoint/>
  <geom type="cylinder" fromto="-0.06 0 0 0.04 0 0" size="0.007" mass="0.012"/>
  <geom type="cylinder" fromto="0.04 0 0 0.058 0 0" size="0.0025" mass="0.0001"/>
  <site name="tip" pos="0.058 0 0"/>
  <body name="carrier"><joint name="cap_slide" type="slide" axis="1 0 0" range="0 0.05" frictionloss="{hold}" damping="0.5" armature="0.05"{fr}/>
   <inertial pos="0 0 0" mass="0.0005" diaginertia="1e-8 1e-8 1e-8"/></body></body>
 <body name="cap" pos="0 0 0.008"><freejoint/><geom type="cylinder" fromto="0.028 0 0 0.061 0 0" size="0.008" mass="0.003"/></body>
</worldbody><equality><weld name="{eq}" body1="carrier" body2="cap" solref="0.004 1"/></equality>
<contact><exclude body1="pen" body2="cap"/></contact></mujoco>"""


JAR_T = """<mujoco><option timestep="0.002"/><worldbody>
 <body name="jar" pos="0 0 0">
  <geom type="box" pos="0 0 0.002" size="0.04 0.04 0.002"/>
  <geom type="box" pos="0.038 0 0.03" size="0.002 0.04 0.03"/><geom type="box" pos="-0.038 0 0.03" size="0.002 0.04 0.03"/>
  <geom type="box" pos="0 0.038 0.03" size="0.04 0.002 0.03"/><geom type="box" pos="0 -0.038 0.03" size="0.04 0.002 0.03"/>
  <site name="mouth" pos="0 0 0.065"/>
  <body name="lid_carrier"><joint name="lid_slide" type="slide" axis="0 0 1" range="0 0.02" frictionloss="2.0" damping="0.5" armature="0.05"FR/>
   <inertial pos="0 0 0" mass="0.0005" diaginertia="1e-8 1e-8 1e-8"/></body></body>
 <body name="lid" pos="0 0 0.065"><freejoint/><geom type="box" size="0.042 0.042 0.006" mass="0.02"/></body>
</worldbody><equality><weld name="lid_engaged" body1="lid_carrier" body2="lid" solref="0.004 1"/></equality>
<contact><exclude body1="jar" body2="lid"/></contact></mujoco>"""
JAR = JAR_T.replace("FR", STIFF)
JAR_SOFT = JAR_T.replace("FR", "")

PLUG = """<mujoco><option timestep="0.002"/><worldbody>
 <geom name="floor" type="plane" size="1 1 0.1"/>
 <body name="socket" pos="0 0 0.02"><freejoint/><geom type="box" size="0.02 0.03 0.02" mass="0.2"/>
  <body name="plug_carrier"><joint name="plug_slide" type="slide" axis="0 -1 0" range="0 0.03" frictionloss="1.0" damping="0.3" armature="0.05" solreffriction="0.004 1" solimpfriction="0.999 0.9999 0.001"/>
   <inertial pos="0 0 0" mass="0.0005" diaginertia="1e-8 1e-8 1e-8"/></body></body>
 <body name="plug" pos="0 -0.035 0.02"><freejoint/><geom type="capsule" fromto="0 0.01 0 0 -0.02 0" size="0.006" mass="0.01"/></body>
</worldbody><equality><weld name="plug_in" body1="plug_carrier" body2="plug" solref="0.004 1"/></equality>
<contact><exclude body1="socket" body2="plug"/></contact></mujoco>"""


def write(tmp, name, xml):
    p = os.path.join(tmp, name + ".xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


fails = 0
with tempfile.TemporaryDirectory() as tmp:
    r = T(write(tmp, "pen", pen()), "cap_engaged", "cap_slide", "cap", 0.021, expose_site="tip")
    if r["ok"] and r["measured"]["pull_off"]["covered_before"] and r["measured"]["pull_off"]["exposed_after"]:
        print("[PASS] pen cap on a free pen comes off and uncovers the tip:", r["verdict"])
    else:
        fails += 1; print("[FAIL] pen", r["issues"])

    r = T(write(tmp, "jar", JAR), "lid_engaged", "lid_slide", "lid", 0.008, expose_site="mouth", pull_force_N=5.0)
    if r["ok"] and abs(r["measured"]["axis_world"][2] - 1.0) < 1e-6:
        print("[PASS] press-fit lid on a fixed jar lifts off along +z:", r["verdict"])
    else:
        fails += 1; print("[FAIL] jar", r["issues"])

    r = T(write(tmp, "plug", PLUG), "plug_in", "plug_slide", "plug", 0.012, pull_force_N=3.0)
    if r["ok"] and abs(r["measured"]["axis_world"][1] + 1.0) < 1e-6:
        print("[PASS] plug on a negative-axis slide of a free socket pulls out:", r["verdict"])
    else:
        fails += 1; print("[FAIL] plug", r["issues"])

    r = T(write(tmp, "tight", pen(hold=6.0)), "cap_engaged", "cap_slide", "cap", 0.021, expose_site="tip")
    if (not r["ok"]) and (not r["measured"]["pull_off"]["released"]) and any("did not slide" in i for i in r["issues"]):
        print("[PASS] a press fit tighter than the pull is reported (never released)")
    else:
        fails += 1; print("[FAIL] tight", r)

    r = T(write(tmp, "jarsoft", JAR_SOFT), "lid_engaged", "lid_slide", "lid", 0.008, expose_site="mouth", pull_force_N=5.0)
    if (not r["ok"]) and any("under gravity" in i for i in r["issues"]):
        print("[PASS] a soft (default solreffriction) press fit that creeps off under gravity is reported:", r["measured"]["seated_hold"])
    else:
        fails += 1; print("[FAIL] jarsoft", r)

    r = T(write(tmp, "noweld", pen(eq="other_name")), "cap_engaged", "cap_slide", "cap", 0.021)
    if (not r["ok"]) and any("not found" in i for i in r["issues"]):
        print("[PASS] a missing weld is reported, not raised")
    else:
        fails += 1; print("[FAIL] noweld", r)

    try:
        T(write(tmp, "pen2", pen()), "cap_engaged", "cap_slide", "cap", 0.0)
        fails += 1; print("[FAIL] release_travel_m=0 was accepted")
    except ValueError as e:
        print("[PASS] release_travel_m=0 refused:", e)

sys.exit(1 if fails else 0)
