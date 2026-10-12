import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tool_library
from tool_library.verifiers.flat_press_drag_test import flat_press_drag_test
from tool_library.skills.flat_hand_pose import flat_hand_pose

REPO = Path(list(tool_library.__path__)[0]).resolve().parent
XML = """<mujoco model="pad"><worldbody><body name="pad" pos="0 0 {hz}"><freejoint name="pad_free"/>
<geom name="pad_geom" type="box" size="{hx} {hy} {hz}" mass="{m}" friction="1 0.01 0.001"/></body></worldbody></mujoco>"""

fails = 0
def case(name, fn):
    global fails
    try:
        fn(); print(f"[PASS] {name}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {e}")

tmp = Path(tempfile.mkdtemp())
def scene(side, hx, hy, hz, m, xy):
    p = tmp / f"pad_{side}_{hx}_{hy}.xml"; p.write_text(XML.format(hx=hx, hy=hy, hz=hz, m=m))
    return {"hand_xml": str(REPO / "setup" / "robot" / f"sharpa_{side}.xml"), "side": side,
            "objects": [{"name": "pad", "xml": str(p), "pos": [xy[0], xy[1], 0.0], "quat": [1, 0, 0, 0]}], "floor": True}

def drags(side, hx, hy, hz, m, fdir, ddir):
    def f():
        sc = scene(side, hx, hy, hz, m, (0.0, 0.4))
        pose = flat_hand_pose(sc, [0.0, 0.4, 2 * hz], [0, 0, 1], fdir, contact="pads")
        r = flat_press_drag_test(sc, pose, ddir, travel_m=0.04, press_m=0.004)
        assert r["ok"], r
        assert r["verdict"].startswith("DRAGGED") and r["measured"]["fraction"] >= 0.6, r
    return f

def json_paths():
    # the hand scene and the pose as the files hand_scene_from_recon and the pose skills write (an in-process call from
    # run_python refused the path: "scene must be a hand-scene dict"); the verdict is the dict call's
    import json
    sc = scene("right", 0.05, 0.03, 0.006, 0.03, (0.0, 0.4))
    pose = flat_hand_pose(sc, [0.0, 0.4, 0.012], [0, 0, 1], [0.3, 0.95, 0], contact="pads")
    (tmp / "hs.json").write_text(json.dumps(sc)); (tmp / "pose.json").write_text(json.dumps(pose))
    a = flat_press_drag_test(sc, pose, [-1, 0, 0], travel_m=0.04, press_m=0.004)
    b = flat_press_drag_test(str(tmp / "hs.json"), str(tmp / "pose.json"), [-1, 0, 0], travel_m=0.04, press_m=0.004)
    assert b["ok"] and b["verdict"] == a["verdict"] and b["measured"]["fraction"] == a["measured"]["fraction"], (a, b)

def stalls():
    sc = scene("right", 0.05, 0.04, 0.01, 0.05, (0.0, 0.4))
    pose = flat_hand_pose(sc, [0.0, 0.4, 0.02], [0, 0, 1], [0, 1, 0], contact="pads")
    high = dict(pose); high["pos"] = [pose["pos"][0], pose["pos"][1], pose["pos"][2] + 0.15]
    r = flat_press_drag_test(sc, high, [1, 0, 0], travel_m=0.04, press_m=0.004)
    assert r["ok"] is False and r["verdict"].startswith("STALLED"), r

def raises(fn, word):
    def f():
        try:
            fn()
        except ValueError as e:
            assert word in str(e), str(e); return
        raise AssertionError("no ValueError")
    return f

case("left hand drags a 12x8 cm pad along +x", drags("left", 0.06, 0.04, 0.01, 0.05, [0, 1, 0], [1, 0, 0]))
case("right hand drags a 10x6 cm thin pad along -x", drags("right", 0.05, 0.03, 0.006, 0.03, [0.3, 0.95, 0], [-1, 0, 0]))
case("right hand drags a pad diagonally (toward +x+y)", drags("right", 0.06, 0.05, 0.012, 0.08, [0, 1, 0], [0.7, 0.7, 0]))
case("a 2D direction [dx, dy] drags the same way", drags("left", 0.06, 0.04, 0.01, 0.05, [0, 1, 0], [1, 0]))
case("the scene and the pose given as JSON files drag the same", json_paths)
case("hand 15 cm above the pad stalls", stalls)
case("a missing scene file is refused", raises(lambda: flat_press_drag_test(str(tmp / "nope.json"), {"pos": [0, 0.2, 0.1], "euler": [0, 0, 0], "finger_targets": {}}, [1, 0, 0]), "does not exist"))
case("vertical direction is refused", raises(lambda: flat_press_drag_test(scene("right", 0.05, 0.04, 0.01, 0.05, (0, 0.4)), {"pos": [0, 0.2, 0.1], "euler": [0, 0, 0], "finger_targets": {}}, [0, 0, 1]), "direction"))
case("pose without finger targets is refused", raises(lambda: flat_press_drag_test(scene("right", 0.05, 0.04, 0.01, 0.05, (0, 0.4)), {"pos": [0, 0.2, 0.1], "euler": [0, 0, 0]}, [1, 0, 0]), "finger_targets"))
sys.exit(1 if fails else 0)
