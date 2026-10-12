"""A valid fixed/free-base cap grip turns and preserves joint initialization;
a weak dial grip is rejected when it loses the part. A folded flap keeps its
initial state, a drawer follows with level wrist orientation, and bad inputs fail."""
import copy
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.skills.pose_from_points import pose_from_points  # noqa: E402
from tool_library.verifiers.grasp_turn_test_keep_state import grasp_turn_test_keep_state as T  # noqa: E402
from tool_library.verifiers.grasp_turn_test import grasp_turn_test as T_reference  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="turnks_", dir=os.environ.get("TMPDIR")))
HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    if ok:
        print(f"[PASS] {name} {detail}")
    else:
        print(f"[FAIL] {name} {detail}")
    FAILS += 0 if ok else 1


def scene(name, body_xml, grasp_body, joint_init=None):
    p = TMP / f"{name}.xml"
    p.write_text(f"<mujoco model=\"{name}\"><worldbody>{body_xml}</worldbody></mujoco>")
    sc = {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": grasp_body,
          "objects": [{"name": name, "xml": str(p), "pos": [0.0, 0.45, 0.0], "quat": [1, 0, 0, 0]}]}
    if joint_init:
        sc["joint_init"] = joint_init
    return sc


def pinch(sc, c, half, axis, z):
    a = np.asarray(axis, float)
    pts = {"thumb": (np.asarray(c) + a * half + [0, 0, z]).tolist(), "index": (np.asarray(c) - a * half + [0, 0, z]).tolist()}
    nrm = {"thumb": a.tolist(), "index": (-a).tolist()}
    r = pose_from_points(sc, pts, nrm, approach_dirs=[[0, 0, 1], [0, -0.5, 0.85]], iters=240)
    return r.get("pose", r)


DIAL = """<body name="base" pos="0 0 0.02"><freejoint/><geom name="foot" type="box" size="0.05 0.03 0.02" mass="0.1"/>
<geom name="post" type="box" size="0.006 0.006 0.03" pos="-0.03 0 0.05"/>
<body name="wheel" pos="0 0 0.06"><joint name="axle" type="hinge" axis="1 0 0" damping="0.05" frictionloss="0.02"/>
<geom name="wheelg" type="cylinder" size="0.018 0.012" euler="0 90 0" mass="0.01" friction="1 0.01 0.001"/></body></body>"""


def main():
    # 1. a cap on a narrow neck turning about z: rigid follow (the original arc)
    cap = scene("capb", """<body name="bottle" pos="0 0 0"><geom name="body" type="cylinder" size="0.03 0.03" pos="0 0 0.03" mass="0.3"/>
<geom name="neck" type="cylinder" size="0.011 0.02" pos="0 0 0.08"/>
<body name="cap" pos="0 0 0.1"><joint name="yaw" type="hinge" axis="0 0 1" damping="0.01" frictionloss="0.002"/>
<geom name="capg" type="cylinder" size="0.018 0.012" pos="0 0 0.012" mass="0.01" friction="1 0.01 0.001"/></body></body>""", "capb_cap")
    pc = pinch(cap, [0.0, 0.45, 0.112], 0.018, [1, 0, 0], 0.0)
    r1 = T(cap, pc, "capb_yaw", sweep_deg=60.0, seconds=2.0)
    check("a pinched cap turns with the hand (rigid)", r1["ok"] and r1["measured"]["follow"] == "rigid", r1.get("verdict", "")[:100])
    free_cap = copy.deepcopy(cap)
    src = Path(cap["objects"][0]["xml"])
    free_xml = src.with_name("free_cap.xml")
    free_xml.write_text(src.read_text().replace('<body name="bottle" pos="0 0 0">',
                                               '<body name="bottle" pos="0 0 0"><freejoint/>'))
    free_cap["objects"][0]["xml"] = str(free_xml)
    free_cap["joint_init"] = {"capb_yaw": .6}
    reference = T_reference(free_cap, pc, "capb_yaw", sweep_deg=40., seconds=2.)
    rf = T(free_cap, pc, "capb_yaw", sweep_deg=40., seconds=2.)
    check("held free cap keeps initialization and matches the independently reported closing state", rf["ok"] and reference["ok"]
          and abs(reference["measured"]["joint_before_close"] - .6) < .02
          and abs(rf["measured"]["state_start"]["capb_yaw"] - reference["measured"]["joint_start"]) < 1e-6, rf["measured"])
    # 2. a dial on a light FREE base, started at 0.6 rad by joint_init: the held root keeps that start, and it turns
    dial0 = scene("dial", DIAL, "dial_wheel")
    pl = pinch(dial0, [0.0, 0.45, 0.08], 0.018, [0, 1, 0], 0.0)
    dial = scene("dial", DIAL, "dial_wheel", joint_init={"dial_axle": 0.6})
    r2 = T(dial, pl, "dial_axle", sweep_deg=40.0, seconds=2.0)
    st = r2["measured"]["state_start"].get("dial_axle")
    check("the former dial positive is rejected when its grasp loses the part", not r2["ok"] and st is not None
          and len(r2["measured"]["pads_end"]) < 2,
          f"start {st} {r2.get('verdict', '')[:70]}")
    # 3. a lid with a flap folded down by joint_init (1.5708) on a free tray: the flap starts folded, not flat
    lid = scene("lidbox", """<body name="tray" pos="0 0 0"><freejoint/><geom name="floor" type="box" size="0.08 0.05 0.002" pos="0 0 0.002" mass="0.1"/>
<geom name="front" type="box" size="0.08 0.002 0.02" pos="0 -0.045 0.022"/><geom name="back" type="box" size="0.08 0.002 0.02" pos="0 0.045 0.022"/>
<body name="lid" pos="0 0.05 0.043"><joint name="hinge" type="hinge" axis="-1 0 0" range="0 170" damping="0.02"/>
<geom name="plate" type="box" size="0.08 0.05 0.0015" pos="0 -0.05 0" mass="0.03"/>
<body name="flap" pos="0 -0.1 0"><joint name="crease" type="hinge" axis="1 0 0" range="-5 90"/>
<geom name="flapg" type="box" size="0.08 0.02 0.0015" pos="0 -0.02 0" mass="0.01"/></body></body></body>""",
                "lidbox_lid", joint_init={"lidbox_crease": 1.5708})
    r3 = T(lid, pl, "lidbox_hinge", sweep_deg=20.0, seconds=0.6)
    st3 = r3["measured"]["state_start"].get("lidbox_crease")
    check("a flap folded by joint_init starts folded while the root is held", st3 is not None and abs(st3 - 1.5708) < 0.15, f"crease at start {st3}")
    # 4. a drawer front on a slide follows a 5 cm pull with a level wrist
    drw = scene("drw", """<body name="cab" pos="0 0 0"><geom type="box" size="0.08 0.02 0.005" pos="0 0.06 0.005"/>
<body name="front" pos="0 0 0.03"><joint name="slide" type="slide" axis="0 -1 0" range="0 0.1" damping="0.5"/>
<geom type="box" size="0.014 0.015 0.02" mass="0.05" friction="1 0.01 0.001"/></body></body>""", "drw_front")
    pd = pinch(drw, [0.0, 0.45, 0.03], 0.014, [1, 0, 0], 0.0)
    r4 = T(drw, pd, "drw_slide", sweep_m=0.05, direction=1, seconds=2.0, follow="level")
    check("a pinched drawer front follows a 5 cm pull (level)", r4["ok"] and r4["measured"]["follow"] == "level", r4.get("verdict", "")[:100])
    # 5. the cap pinch moved 5 cm up: nothing to turn
    far = dict(pc); far["pos"] = (np.asarray(pc["pos"], float) + [0, 0, 0.05]).tolist()
    r5 = T(cap, far, "capb_yaw", sweep_deg=60.0, seconds=1.0, follow="level")
    check("a pose off the part stalls", not r5["ok"], r5.get("verdict", "")[:80])
    # 6. misuse
    try:
        T(cap, pc, "capb_yaw", follow="sideways")
        check("an unknown follow mode raises", False)
    except ValueError as e:
        check("an unknown follow mode raises", "follow" in str(e))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
