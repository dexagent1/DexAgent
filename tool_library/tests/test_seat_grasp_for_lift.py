"""seat_grasp_for_lift: the Drawing and Wiping marker tripod (lifts at no seat with realistic contact: closing rolls the
pen out, every configuration reported), a pinch on a light box (lifts already at 0 mm; seated 2 mm deeper the pose moves
by its seat and still lifts), a pose 5 cm above a part (nothing lifts, every configuration reported), and a seat that
would sink the hand into the table or the pen is refused."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.skills.pose_from_points import pose_from_points  # noqa: E402
from tool_library.skills.seat_grasp_for_lift import seat_grasp_for_lift  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="seat_", dir=os.environ.get("TMPDIR")))
HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def scene(name, body):
    p = TMP / f"{name}.xml"
    p.write_text(f"<mujoco model=\"{name}\"><worldbody>{body}</worldbody></mujoco>")
    return {"hand_xml": HAND, "side": "right", "floor": True, "objects": [{"name": name, "xml": str(p), "pos": [0.0, 0.45, 0.0], "quat": [1, 0, 0, 0]}]}


def main():
    # 1. the Drawing and Wiping marker (13 mm) lying on the table and the tripod an agent solved for it with
    # pose_from_points (fixtures/marker_tripod_pose.json): its pads sit 40 deg above the pen's equator (lower, the
    # fingertips' shells go 6-15 mm into the table) and its thumb arrives first. It lifted seated 2 mm deeper only while
    # the pad contacts carried 0.1 m of rolling friction (the pen could not roll against a pad); with realistic contact
    # the closing thumb rolls the pen 20 deg out of the tripod at every seat, and the tool reports every configuration
    import json
    from tool_library.skills.construct_marker_capped import construct_marker_capped
    prm = {"length": 0.1213, "radius": 0.0066, "cap_length": 0.0316, "cap_radius": 0.0075, "nib_length": 0.0088, "nib_radius": 0.0023,
           "collar_length": 0.0088, "back_length": 0.0105, "cap_clearance": 0.0026, "off_margin": 0.03, "hold_force_N": 1.5, "cap_damping": 0.5,
           "mass": 0.012, "cap_mass": 0.003, "detachable_cap": True}
    built = construct_marker_capped(str(TMP / "marker"), **prm)
    pen = {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": "marker_marker",
           "objects": [{"name": "marker", "xml": built["xml"], "pos": [0.25042, 0.33884, 0.0], "quat": [-0.6530241057425694, 0.0, 0.0, 0.757337122633717]}],
           "joint_init": {"marker_marker_cap_slide": 0.0, "marker_marker_cap_free": [0.25042, 0.33884, 0.0075, 0.6530241057425694, 0.0, 0.0, -0.7573371226337169]}}
    pp = json.loads((ROOT / "tool_library" / "tests" / "fixtures" / "marker_tripod_pose.json").read_text())
    r0 = seat_grasp_for_lift(pen, pp, seats_mm=(0.0,))
    rp = seat_grasp_for_lift(pen, pp)
    check("the marker tripod does not lift at contact", not r0["ok"], str([t["first_fail"] for t in r0["tried"]]))
    check("seated, the marker tripod still lifts nothing and every seat is reported", (not rp["ok"]) and rp["pose"] is None
          and sorted({t["seat_mm"] for t in rp["tried"]}) == [0.0, 1.0, 2.0, 3.0, 4.0] and all(t["first_fail"] for t in rp["tried"]),
          str([(t["seat_mm"], t["press_depth_rad"], t["first_fail"]) for t in rp["tried"]][:6]))
    # 2. a light box pinched across its short side: lifts with no seat
    box = scene("blk", """<body name="blk" pos="0 0 0.0125"><freejoint/><geom name="g" type="box" size="0.035 0.05 0.0125" mass="0.12" friction="1 0.01 0.001"/></body>""")
    bp = pose_from_points(box, {"thumb": [0.035, 0.45, 0.016], "index": [-0.035, 0.45, 0.016]}, {"thumb": [1, 0, 0], "index": [-1, 0, 0]},
                          approach_dirs=[[0, 0, 1]], iters=320)
    bp = bp.get("pose", bp)
    legacy = seat_grasp_for_lift(box, bp, seats_mm=(0.0, 2.0))
    check("the old low two-pad box contact is rejected when it cannot lift", not legacy["ok"]
          and all(t["first_fail"] for t in legacy["tried"]), str(legacy["tried"]))
    # Positive: a collision-free three-pad grasp, independently stress tested.
    # Seating cannot rescue a contact that has no stable grasp geometry.
    from tool_library.tests import grasp_robust_fixtures as F
    box, bp = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    rb = seat_grasp_for_lift(box, bp, seats_mm=(0.0, 2.0))
    check("a box pinch that already lifts needs no seat", rb["ok"] and rb["seat_m"] == 0.0, f"lift {rb.get('lift_mm')}")
    # 2b. the same pinch with only a 2 mm seat on the ladder: the returned pose is the seated one
    r2 = seat_grasp_for_lift(box, bp, seats_mm=(2.0,))
    moved = np.linalg.norm(np.asarray(r2["pose"]["pos"]) - np.asarray(bp["pos"])) if r2["ok"] else None
    check("the seated pose moves the wrist by its seat", r2["ok"] and r2["seat_m"] == 0.002 and abs(moved - 0.002) < 1e-6 and min(r2["lift_mm"]) > 20,
          f"seat {r2.get('seat_m')} moved {None if moved is None else round(moved * 1000, 2)} mm lift {r2.get('lift_mm')}")
    # 3. the box pinch away from the box: nothing lifts, every configuration reported
    far = dict(bp); far["pos"] = (np.asarray(bp["pos"]) + [0, 0, 0.10]).tolist()
    rf = seat_grasp_for_lift(box, far, seats_mm=(0.0, 2.0), press_depths=[0.12, 0.25], open_fracs=[0.2])
    check("a pose off the part finds no seat", (not rf["ok"]) and rf["pose"] is None and len(rf["tried"]) == 4, str([t["first_fail"] for t in rf["tried"]]))
    # 4. a seat whose visible overlap with the part or the table passes the cap is refused before any rollout
    rd = seat_grasp_for_lift(pen, pp, seats_mm=(20.0,), max_visible_mm=1.0)
    check("a seat past the visible-overlap cap is refused", (not rd["ok"]) and rd["tried"] and all(t["first_fail"] == "visible_overlap" and t["visible_mm"] > 1.0 for t in rd["tried"]),
          str(rd["tried"][:1]))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
