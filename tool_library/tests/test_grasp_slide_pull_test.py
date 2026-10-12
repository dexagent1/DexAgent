import os, sys, tempfile
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tool_library
from tool_library.verifiers.grasp_slide_pull_test import grasp_slide_pull_test

REPO = Path(list(tool_library.__path__)[0]).resolve().parent
HAND = REPO / "setup" / "robot" / "sharpa_right.xml"
XML = """<mujoco model="slidebox"><worldbody><body name="housing" pos="0 0 0">
<geom type="box" size="0.07 0.06 0.004" pos="0 0 0.004"/>
<body name="tray" pos="0 0 0.04"><joint name="tray_slide" type="slide" axis="0 -1 0" range="0 {rng}" damping="2"/>
<geom type="box" size="0.06 0.003 0.03" pos="0 -0.06 0"/>
<geom type="box" size="0.055 0.05 0.003" pos="0 0 -0.025"/>
<geom name="bar" type="capsule" size="0.004" fromto="-0.03 -0.095 0 0.03 -0.095 0"/>
<geom type="capsule" size="0.003" fromto="-0.03 -0.095 0 -0.03 -0.063 0"/>
<geom type="capsule" size="0.003" fromto="0.03 -0.095 0 0.03 -0.063 0"/></body>
<body name="lid" pos="0 0.05 0.08"><joint name="lid_hinge" type="hinge" axis="1 0 0" range="0 1.5"/><geom type="box" size="0.05 0.01 0.003"/></body>
</body></worldbody></mujoco>"""

fails = 0
def case(name, fn):
    global fails
    try:
        fn(); print(f"[PASS] {name}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {e}")

tmp = Path(tempfile.mkdtemp())
scenes = {}
for rng in (0.08, 0.12):
    p = tmp / f"box_{rng}.xml"; p.write_text(XML.format(rng=rng))
    scenes[rng] = {"hand_xml": str(HAND), "side": "right", "objects": [{"name": "box", "xml": str(p), "pos": [0, 0.5, 0], "quat": [1, 0, 0, 0]}], "floor": True}
far = {"pos": [0.0, 0.2, 0.45], "euler": [3.14159, 0.0, 0.0], "finger_targets": {}}

def stalled(rng, travel):
    def f():
        r = grasp_slide_pull_test(scenes[rng], far, "tray_slide", travel_m=travel, seconds=0.6)
        assert r["ok"] is False, r
        assert r["verdict"].startswith("STALLED"), r["verdict"]
        assert r["issues"] and abs(r["measured"]["joint_moved_m"]) < 0.01, r["measured"]
        assert any(k.endswith("lid_hinge") for k in r["measured"]["siblings_moved"]), r["measured"]
    return f

def raises(fn, word):
    def f():
        try:
            fn()
        except ValueError as e:
            assert word in str(e), str(e); return
        raise AssertionError("no ValueError")
    return f

case("hand far from the handle stalls (range 0.08, travel 0.05)", stalled(0.08, 0.05))
case("hand far from the handle stalls (range 0.12, travel 0.10)", stalled(0.12, 0.10))
case("negative travel is refused", raises(lambda: grasp_slide_pull_test(scenes[0.08], far, "tray_slide", travel_m=-0.02), "travel_m"))
case("hinge joint is refused", raises(lambda: grasp_slide_pull_test(scenes[0.08], far, "lid_hinge"), "not a slide joint"))
case("unknown joint is refused", raises(lambda: grasp_slide_pull_test(scenes[0.08], far, "no_such_joint_xyz"), "no joint"))
case("pose without finger targets is refused", raises(lambda: grasp_slide_pull_test(scenes[0.08], {"pos": [0, 0, 0.3], "euler": [0, 0, 0]}, "tray_slide"), "finger_targets"))
sys.exit(1 if fails else 0)
