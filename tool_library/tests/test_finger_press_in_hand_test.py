import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tool_library
from tool_library.verifiers.finger_press_in_hand_test import finger_press_in_hand_test
from tool_library.skills.finger_press_targets import finger_press_targets
from tool_library.skills.construct_pump_sprayer_down_axis import construct_pump_sprayer_down_axis
from tool_library.skills.pose_from_points import pose_from_points
from tool_library.skills.optimize_fingertip_contacts import optimize_fingertip_contacts

REPO = Path(list(tool_library.__path__)[0]).resolve().parent
HAND = REPO / "setup" / "robot" / "sharpa_right.xml"
SPRAYER = {"name": "spray_bottle", "body_radius": 0.0205, "body_height": 0.075, "shoulder_height": 0.005, "neck_radius": 0.011, "neck_height": 0.004,
           "collar_radius": 0.016, "collar_height": 0.012, "stem_radius": 0.006, "stem_height": 0.012, "head_radius": 0.011, "head_height": 0.022,
           "nozzle_length": 0.005, "nozzle_radius": 0.0025, "nozzle_azimuth_deg": 180, "stroke": 0.01, "spring_stiffness": 400, "preload_N": 2.5,
           "head_damping": 1.0, "head_armature": 0.002, "head_mass": 0.005, "mass": 0.08, "friction": [1.0, 0.01, 0.001], "free": True, "include_floor": False}
BOTTLE_POS = [0.234053, 0.35793, 0.0]
tmp = Path(tempfile.mkdtemp())
def scene_for(**over):
    d = tmp / ("spray_" + "_".join(f"{k}{v}" for k, v in over.items()))
    built = construct_pump_sprayer_down_axis(str(d), **{**SPRAYER, **over})
    xml = built["xml"] if Path(built["xml"]).is_absolute() else str(d / Path(built["xml"]).name)
    if not Path(xml).exists():
        xml = str(next(d.rglob("*.xml")))
    return {"hand_xml": str(HAND), "side": "right", "objects": [{"name": "spray_bottle", "xml": xml, "pos": BOTTLE_POS, "quat": [1, 0, 0, 0]}], "floor": True}
soft, stiff = scene_for(), scene_for(preload_N=60.0)
# a side grasp of the sprayer: thumb and middle pads on opposite sides of the body near its top, the index pad resting on
# the pump head (pose_from_points, the pads cinched to touch by optimize_fingertip_contacts). Before, a hard-coded
# thumb-middle-ring grasp was used: its thumb's proximal link sat 5.3 mm inside the body, it had no approach, and two
# torque-capped fingers against one thumb shoved the free-standing sprayer 8-9 deg over, so its head left the index pad
# (it only pumped while the shove happened to bring the head under the pad). One finger against the thumb balances.
_R = SPRAYER["body_radius"]
_TOP = sum(SPRAYER[k] for k in ("body_height", "shoulder_height", "neck_height", "collar_height", "stem_height", "head_height"))   # the head's top
_g = pose_from_points(soft, {"thumb": [BOTTLE_POS[0], BOTTLE_POS[1] - _R, 0.066], "middle": [BOTTLE_POS[0], BOTTLE_POS[1] + _R, 0.066],
                             "index": [BOTTLE_POS[0], BOTTLE_POS[1], _TOP]},
                      {"thumb": [0, -1, 0], "middle": [0, 1, 0], "index": [0, 0, 1]}, approach_dirs=[[1.0, 0.0, 0.5], [0.6, 0.0, 0.8]], iters=300)
GRASP = optimize_fingertip_contacts(soft, _g.get("pose", _g), fingers=["thumb", "middle", "index"])
GRASP["grip_fingers"] = ["index", "middle", "thumb"]
P = finger_press_targets(soft, GRASP, "pump_slide", overdrive_m=0.004)

fails = 0
def case(name, fn):
    global fails
    try:
        fn(); print(f"[PASS] {name}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {type(e).__name__}: {e}")

def pumped_on_support():
    r = finger_press_in_hand_test(soft, P["hold"], P["pressed"], "pump_slide", lift_m=0.0, cycles=2)
    assert r["ok"] and r["verdict"] == "PUMPED", (r["verdict"], r["issues"], r["measured"]["presses"])
    assert all(p["frac"] >= 0.5 for p in r["measured"]["presses"]) and all(x["frac"] <= 0.25 for x in r["measured"]["releases"])

def stiff_spring_stalls():
    r = finger_press_in_hand_test(stiff, P["hold"], P["pressed"], "pump_slide", lift_m=0.0, cycles=2)
    assert not r["ok"] and r["verdict"] == "STALLED" and r["issues"], (r["verdict"], r["measured"]["presses"])

def no_press_motion():
    r = finger_press_in_hand_test(soft, P["hold"], P["released"], "pump_slide", lift_m=0.0, cycles=1)
    assert not r["ok"] and any("moves no joint" in i for i in r["issues"]), r["issues"]

def far_hand_drops():
    far = dict(P["hold"], pos=[P["hold"]["pos"][0] + 0.3, P["hold"]["pos"][1], P["hold"]["pos"][2] + 0.2])
    r = finger_press_in_hand_test(soft, far, P["pressed"], "pump_slide", lift_m=0.05, cycles=1)
    assert not r["ok"] and r["verdict"] == "DROPPED", r["verdict"]

def raises(fn, word):
    def f():
        try:
            fn()
        except ValueError as e:
            assert word in str(e), str(e); return
        raise AssertionError("no ValueError")
    return f

case("pump on a supported sprayer strokes and springs back (PUMPED)", pumped_on_support)
case("a 60 N preload does not stroke under a finger (STALLED)", stiff_spring_stalls)
case("a pressed pose that moves no finger joint is reported", no_press_motion)
case("a hand away from the object drops it (DROPPED)", far_hand_drops)
case("malformed tilt is refused", raises(lambda: finger_press_in_hand_test(soft, P["hold"], P["pressed"], "pump_slide", tilt=[0, 1, 0]), "tilt"))
case("free joint is refused", raises(lambda: finger_press_in_hand_test(soft, P["hold"], P["pressed"], "spray_bottle_free"), "slide or hinge"))
sys.exit(1 if fails else 0)
