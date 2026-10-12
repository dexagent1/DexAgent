import sys, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import tool_library
from tool_library.skills.finger_press_targets import finger_press_targets
from tool_library.skills.construct_pump_sprayer_down_axis import construct_pump_sprayer_down_axis

REPO = Path(list(tool_library.__path__)[0]).resolve().parent
HAND = REPO / "setup" / "robot" / "sharpa_right.xml"
SPRAYER = {"name": "spray_bottle", "body_radius": 0.0205, "body_height": 0.075, "shoulder_height": 0.005, "neck_radius": 0.011, "neck_height": 0.004,
           "collar_radius": 0.016, "collar_height": 0.012, "stem_radius": 0.006, "stem_height": 0.012, "head_radius": 0.011, "head_height": 0.022,
           "nozzle_length": 0.005, "nozzle_radius": 0.0025, "nozzle_azimuth_deg": 180, "stroke": 0.01, "spring_stiffness": 400, "preload_N": 2.5,
           "head_damping": 1.0, "head_armature": 0.002, "head_mass": 0.005, "mass": 0.08, "friction": [1.0, 0.01, 0.001], "free": True, "include_floor": False}
BOTTLE_POS = [0.234053, 0.35793, 0.0]
# a thumb-middle-ring side grasp of that sprayer with the index pad resting on the pump head (pose_from_points)
GRASP = {"side": "right", "pos": [0.361339694376823, 0.3833070004471951, 0.08485010845251652], "euler": [3.2024856507704884, -2.188573295634254, 6.281724039296297],
         "grip_fingers": ["index", "middle", "ring", "thumb"],
         "finger_targets": {"thumb_CMC_FE": 0.9716885910935352, "thumb_CMC_AA": 0.0986819565979652, "thumb_MCP_FE": 0.042333866662685224, "thumb_MCP_AA": -0.33249205048017183, "thumb_IP": 0.15922642929813094, "index_MCP_FE": -0.17453293, "index_MCP_AA": 0.23435939388787905, "index_PIP": 1.5144358111201004, "index_DIP": 1.3963, "middle_MCP_FE": 1.2995996308240412, "middle_MCP_AA": 0.13506234308057646, "middle_PIP": 0.7564500641417379, "middle_DIP": 0.3509442490958395, "ring_MCP_FE": 1.5706300645198086, "ring_MCP_AA": -0.22698064802124643, "ring_PIP": 0.509275123195973, "ring_DIP": 0.1344239940634799, "pinky_CMC": 0.17017, "pinky_MCP_FE": 0.9599334745000001, "pinky_MCP_AA": 0.0, "pinky_PIP": 1.1344450000000001, "pinky_DIP": 0.907595}}
tmp = Path(tempfile.mkdtemp())
built = construct_pump_sprayer_down_axis(str(tmp / "spray"), **{k: v for k, v in SPRAYER.items()})
xml = built["xml"] if Path(built["xml"]).is_absolute() else str(tmp / "spray" / Path(built["xml"]).name)
if not Path(xml).exists():
    xml = str(next((tmp / "spray").rglob("*.xml")))
scene = {"hand_xml": str(HAND), "side": "right", "objects": [{"name": "spray_bottle", "xml": xml, "pos": BOTTLE_POS, "quat": [1, 0, 0, 0]}], "floor": True}
TRIGGER = """<mujoco model="trigger"><worldbody><body name="box" pos="0 0 0"><freejoint name="box_free"/>
<geom type="box" size="0.03 0.02 0.04" pos="0 0 0.04" mass="0.1"/>
<body name="lever" pos="0.03 0 0.06"><joint name="lever_hinge" type="hinge" axis="0 1 0" range="0 0.4"/>
<geom type="box" size="0.004 0.015 0.02" pos="0.004 0 -0.02" mass="0.01"/></body></body></worldbody></mujoco>"""
(tmp / "trigger.xml").write_text(TRIGGER)
scene_hinge = {"hand_xml": str(HAND), "side": "right", "objects": [{"name": "trig", "xml": str(tmp / "trigger.xml"), "pos": BOTTLE_POS, "quat": [1, 0, 0, 0]}], "floor": True}

fails = 0
def case(name, fn):
    global fails
    try:
        fn(); print(f"[PASS] {name}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {type(e).__name__}: {e}")

def slide_default():
    r = finger_press_targets(scene, GRASP, "pump_slide", out_dir=str(tmp / "out"), prefix="t")
    assert np.allclose(r["direction"], [0, 0, -1], atol=0.05), r["direction"]
    assert r["pad_travel_m"] >= 0.5 * r["stroke_m"], r
    assert "index" not in r["grip_fingers"] and set(r["grip_fingers"]) == {"thumb", "middle", "ring"}, r["grip_fingers"]
    for k in ("hold", "pressed", "released"):
        assert Path(r["paths"][k]).exists(), k
    pr, rl = r["pressed"]["finger_targets"], r["released"]["finger_targets"]
    moved = [k for k in pr if abs(pr[k] - rl[k]) > 1e-6]
    assert moved and all(k.startswith("index") for k in moved), moved
    assert r["pressed"]["finger_targets_open"] == rl and r["released"]["finger_targets_open"] == pr
    assert r["pressed"]["press_depth_rad"] == 0.0 and not r["issues"], r["issues"]

def slide_short_stroke():
    a = finger_press_targets(scene, GRASP, "pump_slide", overdrive_m=0.0)
    b = finger_press_targets(scene, GRASP, "pump_slide", stroke_m=0.004, overdrive_m=0.0)
    assert b["stroke_m"] == 0.004 and b["pad_travel_m"] < a["pad_travel_m"] - 0.003, (a["pad_travel_m"], b["pad_travel_m"])

def rest_offset():
    r = finger_press_targets(scene, GRASP, "pump_slide", rest_offset_m=[0.004, 0.0, 0.0])
    h = r["hold"]["finger_targets"]
    assert any(abs(h[k] - GRASP["finger_targets"][k]) > 1e-4 for k in h if k.startswith("index")), "index rest unchanged"
    assert all(abs(h[k] - GRASP["finger_targets"][k]) < 1e-9 for k in h if not k.startswith("index")), "other fingers moved"
    assert r["rest_residual_mm"] is not None and r["rest_residual_mm"] < 5.0, r["rest_residual_mm"]

def hinge_lever():
    pose = dict(GRASP, pos=[BOTTLE_POS[0] + 0.12, BOTTLE_POS[1], 0.07])
    r = finger_press_targets(scene_hinge, pose, "lever_hinge", finger="index")
    d = np.array(r["direction"])
    assert abs(d[1]) < 0.05 and abs(np.linalg.norm(d) - 1) < 1e-3, d          # perpendicular to the hinge axis (y)
    assert 0 < r["stroke_m"] < 0.4 * 0.2, r["stroke_m"]                        # range x lever, lever well under 20 cm

def raises(fn, word):
    def f():
        try:
            fn()
        except ValueError as e:
            assert word in str(e), str(e); return
        raise AssertionError("no ValueError")
    return f

case("pump slide: press straight down the bottle axis, index out of the grip, three poses written", slide_default)
case("stroke_m override shortens the pad travel", slide_short_stroke)
case("rest_offset_m moves only the pressing finger's resting targets", rest_offset)
case("hinge trigger: press direction is the hinge tangent", hinge_lever)
case("free joint is refused", raises(lambda: finger_press_targets(scene, GRASP, "spray_bottle_free"), "slide or hinge"))
case("unknown finger is refused", raises(lambda: finger_press_targets(scene, GRASP, "pump_slide", finger="toe"), "finger"))
case("a grip of only the pressing finger is refused", raises(lambda: finger_press_targets(scene, dict(GRASP, grip_fingers=["index"]), "pump_slide"), "gripping finger"))
case("unknown joint is refused", raises(lambda: finger_press_targets(scene, GRASP, "no_such_joint_xyz"), "joint"))
sys.exit(1 if fails else 0)
