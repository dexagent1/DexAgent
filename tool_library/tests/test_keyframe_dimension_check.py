"""keyframe_dimension_check: the setup's home passes; a short keyframe (silently padded), a keyframe split at the wrong index
after attaching an object, a servo ctrl off its joint and a robot scene without keyframe fail.

    python tool_library/tests/test_keyframe_dimension_check.py
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.keyframe_dimension_check import keyframe_dimension_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="keyframes_"))
SETUP = ROOT / "setup" / "scene.xml"
BASE = SETUP.read_text().replace('file="assets/', f'file="{ROOT}/setup/assets/')
home_q = re.search(r'<key name="home" qpos="([^"]+)"', BASE).group(1).split()
home_c = re.search(r'ctrl="([^"]+)"', BASE[BASE.index('<key name="home"'):]).group(1).split()


def variant(name, qpos=None, ctrl=None, obj=False, drop_key=False):
    s = BASE
    if obj:          # a cup attached first in the worldbody: its 7 qpos come before the robot's
        s = s.replace("<worldbody>", '<worldbody><body name="cup_cup" pos="0 0.6 0.05"><freejoint name="cup_free"/>'
                                      '<geom type="cylinder" size=".04 .05" mass=".1"/></body>', 1)
    if drop_key:
        s = re.sub(r"<keyframe>.*?</keyframe>", "", s, flags=re.S).replace('<size nkey="1" />', "")
    elif qpos is not None or ctrl is not None:
        key = f'<key name="home" qpos="{" ".join(qpos if qpos is not None else home_q)}" ctrl="{" ".join(ctrl if ctrl is not None else home_c)}" />'
        s = re.sub(r'<key name="home"[^>]*/>', key, s)
    p = tmp / f"{name}.xml"
    p.write_text(s)
    return str(p)


def _write(name, text):
    p = tmp / f"{name}.xml"
    p.write_text(text)
    return str(p)


# 1. the setup against itself, and a correctly extended keyframe for an attached cup
cup7 = ["0", "0.6", "0.05", "1", "0", "0", "0"]
r0 = keyframe_dimension_check(str(SETUP), reference=str(SETUP))
r1 = keyframe_dimension_check(variant("cup_good", qpos=cup7 + home_q, obj=True), reference=str(SETUP))
check("setup home and a keyframe extended at the cup's qpos address pass", r0["ok"] and r1["ok"], f"{r0['issues']} {r1['issues']}")

# 2. one value short: MuJoCo compiles it by padding from qpos0; the authored length gives it away
r2 = keyframe_dimension_check(variant("short", qpos=home_q[:-1]))
check("a keyframe one qpos short fails (silent padding named)", not r2["ok"] and "pads" in " ".join(r2["issues"]), str(r2["issues"])[:300])

# 3. the cup's values appended at the end instead of at its address: non-unit quaternion, robot off its home
r3 = keyframe_dimension_check(variant("wrong_split", qpos=home_q + cup7, obj=True), reference=str(SETUP))
txt = " ".join(r3["issues"])
check("a keyframe split at the wrong index fails on the quaternion and on the reference home",
      not r3["ok"] and "not unit" in txt and r3["reference"]["mismatches"], txt[:300])

# 4. a wrist servo commanded 30 cm away from where the keyframe puts its joint
bad_c = list(home_c)
bad_c[1] = str(float(home_c[1]) - 0.3)
r4 = keyframe_dimension_check(variant("ctrl_off", ctrl=bad_c))
check("a servo ctrl 0.3 m off its joint's keyframe position fails", not r4["ok"] and "yank" in " ".join(r4["issues"]), str(r4["issues"])[:300])

# 5. a robot scene without keyframe fails, and so does one whose only keyframe is the qpos0 one <size nkey> reserves;
#    an object-only scene without keyframe passes
r5 = keyframe_dimension_check(variant("nokey", drop_key=True))
r5b = keyframe_dimension_check(_write("nkey_only", re.sub(r"<keyframe>.*?</keyframe>", "", BASE, flags=re.S)))
r6 = keyframe_dimension_check('<mujoco><worldbody><body><freejoint/><geom size=".05"/></body></worldbody></mujoco>')
check("a robot scene without keyframe, or with only a reserved qpos0 one, fails; an object-only scene passes",
      not r5["ok"] and "no keyframe" in r5["issues"][0] and not r5b["ok"] and "qpos0" in " ".join(r5b["issues"]) and r6["ok"],
      f"{r5['issues']} {r5b['issues']} {r6['issues']}")

sys.exit(1 if fails else 0)
