"""simulation_stability_check: resting scenes simulate quietly; a spring too stiff for its inertia, a cap snapping onto a
violated weld, and finger servos without armature blow up and fail.

    python tool_library/tests/test_simulation_stability_check.py
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from tool_library.verifiers.simulation_stability_check import simulation_stability_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="sim_stability_"))

# 1. the setup at home (hands servoed, arms held) and a box on a table are quiet
r0 = simulation_stability_check(str(ROOT / "setup" / "scene.xml"), seconds=1.0)
r1 = simulation_stability_check('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .03"><freejoint/>'
                                '<geom type="box" size=".03 .03 .03" mass=".2"/></body></worldbody></mujoco>', seconds=1.0)
check("setup at home and a resting box are stable", r0["ok"] and r1["ok"] and r0["finite"], f"{r0['issues']} {r1['issues']}")

# 2. a 1 g finger link on a hinge driven by a kp-300 position servo with no armature (inertia ~1e-9 kg m^2): the servo's
#    stiffness is integrated explicitly and the link blows up; with armature 0.05 it holds
finger = ('<mujoco><compiler angle="radian"/><worldbody><body name="base" pos="0 0 .1"><geom type="box" size=".05 .05 .01"/>'
          '<body name="link" pos=".05 0 0"><joint name="h" type="hinge" axis="0 1 0" range="-1 1" armature="{a}"/>'
          '<geom type="capsule" size=".002 .005" mass=".001"/></body></body></worldbody>'
          '<actuator><position joint="h" kp="300" dampratio="1"/></actuator>'
          '<keyframe><key qpos="0.3" ctrl="0"/></keyframe></mujoco>')
r2 = simulation_stability_check(finger.format(a="0"), seconds=0.5, ctrl="key")
r2b = simulation_stability_check(finger.format(a="0.05"), seconds=0.5, ctrl="key")
check("an explicitly unstable servo (no armature) diverges and fails; with armature it passes",
      not r2["ok"] and (r2["warnings"] or not r2["finite"]) and r2b["ok"], f"{r2['issues']} | {r2b['issues']}")

# 3. a 5 g cap welded to a bottle by a stiff weld but placed 2 cm off and turned 160 degrees at the keyframe
cap = ('<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/>'
       '<body name="bottle" pos="0 0 .06"><freejoint/><geom type="cylinder" size=".03 .06" mass=".2"/></body>'
       '<body name="cap" pos="0.02 0 .13" euler="0 0 2.8"><freejoint/><geom type="cylinder" size=".02 .01" mass=".005"/></body>'
       '</worldbody><equality><weld body1="bottle" body2="cap" relpose="0 0 .07 1 0 0 0" solref="0.004 1" torquescale="1"/></equality>'
       '</mujoco>')
r3 = simulation_stability_check(cap, seconds=0.5)
check("a light cap snapping 2 cm / 160 deg onto its weld is thrown (speed or energy gain) and fails",
      not r3["ok"] and (r3["max_speed_mps"]["value"] > 5 or r3["energy"]["max_gain_j"] > 0.02 or r3["warnings"]),
      f"{r3['issues']} {r3['max_speed_mps']} {r3['energy']}")

# 4. the setup with the hand joints' armature removed: kp-300 finger servos on 1e-6 kg m^2 links blow up
txt = (ROOT / "setup" / "scene.xml").read_text().replace('file="assets/', f'file="{ROOT}/setup/assets/')
txt = re.sub(r'(<joint name="robot_(?:left|right)_(?:left|right)_[A-Za-z_]+"[^>]*?) armature="1"', r'\1 armature="0"', txt)
txt = txt.replace('<joint armature="1" />', '<joint armature="0" />')
(tmp / "no_armature.xml").write_text(txt)
r4 = simulation_stability_check(str(tmp / "no_armature.xml"), seconds=0.5)
check("finger servos without armature trigger MuJoCo's instability warning and fail",
      not r4["ok"] and "BADQACC" in r4["warnings"], str(r4["issues"])[:300])

# 5. misuse
try:
    simulation_stability_check(finger.format(a="0"), ctrl="bogus")
    check("an unknown ctrl mode raises ValueError", False)
except ValueError:
    check("an unknown ctrl mode raises ValueError", True)

sys.exit(1 if fails else 0)
