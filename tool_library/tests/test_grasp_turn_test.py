"""Physical mechanism controls: fixed/free-base caps and a drawer follow valid grasps;
the former dial positive loses its grasp with visible self contacts enabled and
must be rejected. Joint initialization survives root holding and close motion is reported."""
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
from tool_library.verifiers.grasp_turn_test import grasp_turn_test  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="turn_", dir=os.environ.get("TMPDIR")))
HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def scene(name, body_xml, grasp_body):
    p = TMP / f"{name}.xml"
    p.write_text(f"<mujoco model=\"{name}\"><worldbody>{body_xml}</worldbody></mujoco>")
    return {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": grasp_body,
            "objects": [{"name": name, "xml": str(p), "pos": [0.0, 0.45, 0.0], "quat": [1, 0, 0, 0]}]}


def pinch(sc, c, half, axis, z, middle=False):
    a = np.asarray(axis, float)
    pts = {"thumb": (np.asarray(c) + a * half + [0, 0, z]).tolist(), "index": (np.asarray(c) - a * half + [0, 0, z]).tolist()}
    nrm = {"thumb": a.tolist(), "index": (-a).tolist()}
    if middle:                                   # index and middle side by side on the far face (a firmer lever grip)
        side = np.cross(a, [0, 0, 1.0]); side /= np.linalg.norm(side)
        pts["index"] = (np.asarray(pts["index"]) + side * 0.009).tolist()
        pts["middle"] = (np.asarray(c) - a * half + [0, 0, z] - side * 0.009).tolist(); nrm["middle"] = (-a).tolist()
    r = pose_from_points(sc, pts, nrm, approach_dirs=[[0, 0, 1], [0, -0.5, 0.85]], iters=240)
    return r.get("pose", r)


def main():
    # 1. a cap (r 18 mm) on a bottle's narrow neck (the thumb's lower links clear the shoulder), turning about z
    cap = scene("capb", """<body name="bottle" pos="0 0 0"><geom name="body" type="cylinder" size="0.03 0.03" pos="0 0 0.03" mass="0.3"/>
<geom name="neck" type="cylinder" size="0.011 0.02" pos="0 0 0.08"/>
<body name="cap" pos="0 0 0.1"><joint name="yaw" type="hinge" axis="0 0 1" damping="0.01" frictionloss="0.002"/>
<geom name="capg" type="cylinder" size="0.018 0.012" pos="0 0 0.012" mass="0.01" friction="1 0.01 0.001"/></body></body>""", "capb_cap")
    pc = pinch(cap, [0.0, 0.45, 0.112], 0.018, [1, 0, 0], 0.0)
    r1 = grasp_turn_test(cap, pc, "capb_yaw", sweep_deg=60.0, seconds=2.0)
    check("a pinched cap turns with the hand", r1["ok"], r1.get("verdict", "")[:120])
    free_cap = copy.deepcopy(cap)
    src = Path(cap["objects"][0]["xml"])
    free_xml = src.with_name("free_cap.xml")
    free_xml.write_text(src.read_text().replace('<body name="bottle" pos="0 0 0">',
                                               '<body name="bottle" pos="0 0 0"><freejoint/>'))
    free_cap["objects"][0]["xml"] = str(free_xml)
    free_cap["joint_init"] = {"capb_yaw": .6}
    rf = grasp_turn_test(free_cap, pc, "capb_yaw", sweep_deg=40., seconds=2.)
    check("a cap on a held free base keeps its initial angle and turns", rf["ok"]
          and abs(rf["measured"]["joint_before_close"] - .6) < .02 and rf["measured"]["close_moved"] < 10., rf["measured"])
    # 2. Former positive fixture: a dial on a held free base turns almost the
    # commanded angle, but loses its pads. Motion alone cannot certify a grasp.
    dial = scene("dial", """<body name="base" pos="0 0 0.02"><freejoint/><geom name="foot" type="box" size="0.05 0.03 0.02" mass="0.1"/>
<geom name="post" type="box" size="0.006 0.006 0.03" pos="-0.03 0 0.05"/>
<body name="wheel" pos="0 0 0.06"><joint name="axle" type="hinge" axis="1 0 0" damping="0.05" frictionloss="0.02"/>
<geom name="wheelg" type="cylinder" size="0.018 0.012" euler="0 90 0" mass="0.01" friction="1 0.01 0.001"/></body></body>""", "dial_wheel")
    pl = pinch(dial, [0.0, 0.45, 0.08], 0.018, [0, 1, 0], 0.0)
    r2 = grasp_turn_test(dial, pl, "dial_axle", sweep_deg=40.0, seconds=2.0)
    check("a dial that turns but loses its pads is rejected", not r2["ok"] and len(r2["measured"]["pads_end"]) < 2,
          r2.get("verdict", "")[:120])
    # 2b. On a nearly frictionless axle, that weak grasp flings or releases the
    # dial. The existing overtravel and retained-contact checks must reject it.
    spin = scene("spin", """<body name="base" pos="0 0 0.02"><geom name="foot" type="box" size="0.05 0.03 0.02" mass="0.1"/>
<body name="wheel" pos="0 0 0.06"><joint name="axle" type="hinge" axis="1 0 0" damping="0.0005"/>
<geom name="wheelg" type="cylinder" size="0.018 0.012" euler="0 90 0" mass="0.01" friction="1 0.01 0.001"/></body></body>""", "spin_wheel")
    rs = grasp_turn_test(spin, pl, "spin_axle", sweep_deg=40.0, seconds=0.6)
    check("a dial flung or released on its free axle is rejected", not rs["ok"]
          and (rs["measured"]["flung"] or len(rs["measured"]["pads_end"]) < 2),
          f"{rs.get('verdict', '')[:80]} jump {rs['measured']['max_step_jump']}")
    # 2c. the dial's start angle from joint_init is kept on a free (held) base: the test turns from 30 deg, not from 0.
    # Read as the open hand arrives: the closing pinch then turns this free-axle dial a few degrees by itself (its pads
    # curl along the rim; measured 4.5 deg from 0 and from 30 alike), which close_moved reports
    dial30 = dict(dial, joint_init={"dial_axle": 0.5236})
    r30 = grasp_turn_test(dial30, pl, "dial_axle", sweep_deg=10.0, seconds=0.5)
    m30 = r30["measured"]
    check("a joint_init start state survives the held root", abs(m30["joint_before_close"] - 0.5236) < 0.02
          and abs(abs(m30["joint_start"] - m30["joint_before_close"]) - np.radians(m30["close_moved"])) < 1e-3,
          f"{m30['joint_before_close']} -> {m30['joint_start']} (close moved {m30['close_moved']} deg)")
    # 3. a drawer block on a slide along y, pulled 5 cm
    drw = scene("drw", """<body name="cab" pos="0 0 0"><geom type="box" size="0.08 0.02 0.005" pos="0 0.06 0.005"/>
<body name="front" pos="0 0 0.03"><joint name="slide" type="slide" axis="0 -1 0" range="0 0.1" damping="0.5"/>
<geom type="box" size="0.014 0.015 0.02" mass="0.05" friction="1 0.01 0.001"/></body></body>""", "drw_front")
    pd = pinch(drw, [0.0, 0.45, 0.03], 0.014, [1, 0, 0], 0.0)
    r3 = grasp_turn_test(drw, pd, "drw_slide", sweep_m=0.05, direction=1, seconds=2.0)
    check("a pinched drawer front follows a 5 cm pull", r3["ok"], r3.get("verdict", "")[:120])
    # 4. the cap pinch moved 5 cm up: nothing to turn
    far = dict(pc); far["pos"] = (np.asarray(pc["pos"], float) + [0, 0, 0.05]).tolist()
    r4 = grasp_turn_test(cap, far, "capb_yaw", sweep_deg=60.0, seconds=1.0)
    check("a pose off the part stalls", not r4["ok"], r4.get("verdict", "")[:80])
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
