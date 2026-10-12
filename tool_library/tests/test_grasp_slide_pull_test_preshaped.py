"""Tests for grasp_slide_pull_test_preshaped on synthetic drawers with a proud bar handle."""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.common import hand as H
from tool_library.skills.pose_from_points import pose_from_points
from tool_library.verifiers.grasp_slide_pull_test_preshaped import grasp_slide_pull_test_preshaped as T

HAND = str(Path(H.__file__).resolve().parents[2] / "setup" / "robot" / "sharpa_left.xml")


def drawer_xml(w, depth, h, proud, bar_half):
    """A housing (free) with one tray on a slide; the tray front carries a bar handle `proud` in front of it."""
    hw, hd, hh = w / 2, depth / 2, h / 2
    ty = 0.0
    return f"""<mujoco model="drawer">
  <default><geom condim="4" friction="0.8 0.02 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <body name="carcass">
      <freejoint name="free"/>
      <inertial pos="0 0 {hh}" mass="0.6" diaginertia="0.004 0.0012 0.004"/>
      <geom name="left" type="box" size="0.003 {hd} {hh}" pos="{-hw+0.003} 0 {hh}"/>
      <geom name="right" type="box" size="0.003 {hd} {hh}" pos="{hw-0.003} 0 {hh}"/>
      <geom name="top" type="box" size="{hw-0.003} {hd} 0.003" pos="0 0 {h-0.003}"/>
      <geom name="bottom" type="box" size="{hw-0.003} {hd} 0.003" pos="0 0 0.003"/>
      <geom name="rear" type="box" size="{hw-0.003} 0.003 {hh-0.006}" pos="0 {-hd+0.003} {hh}"/>
      <body name="tray" pos="0 {ty} {hh}">
        <joint name="tray_slide" type="slide" axis="0 1 0" range="0 {0.8*depth}" damping="3" frictionloss="0.4" armature="0.01"/>
        <inertial pos="0 0 0" mass="0.09" diaginertia="0.0003 0.0001 0.0003"/>
        <geom name="front" type="box" size="{hw-0.008} 0.003 {hh-0.011}" pos="0 {hd-0.003} 0"/>
        <geom name="tside_l" type="box" size="0.003 {hd-0.03} {hh-0.014}" pos="{-hw+0.013} 0 0"/>
        <geom name="tside_r" type="box" size="0.003 {hd-0.03} {hh-0.014}" pos="{hw-0.013} 0 0"/>
        <geom name="tbottom" type="box" size="{hw-0.016} {hd-0.03} 0.003" pos="0 0 {-hh+0.017}"/>
        <geom name="post0" type="cylinder" size="0.0036 {proud/2}" pos="{-bar_half+0.005} {hd+proud/2} 0" euler="90 0 0"/>
        <geom name="post1" type="cylinder" size="0.0036 {proud/2}" pos="{bar_half-0.005} {hd+proud/2} 0" euler="90 0 0"/>
        <geom name="bar" type="cylinder" size="0.0051 {bar_half}" pos="0 {hd+proud} 0" euler="0 90 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""


def make_case(tmp, name, pos, yaw_deg, w, depth, h, proud, bar_half):
    path = os.path.join(tmp, f"{name}.xml")
    Path(path).write_text(drawer_xml(w, depth, h, proud, bar_half))
    q = R.from_euler("z", yaw_deg, degrees=True).as_quat()
    scene = {"hand_xml": HAND, "side": "left", "floor": True, "joint_init": {}, "grasp_body": f"{name}_tray",
             "objects": [{"name": name, "xml": path, "pos": list(pos), "quat": [q[3], q[0], q[1], q[2]]}]}
    rz = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
    out = rz @ np.array([0.0, 1.0, 0.0])                   # pull-out direction (toward the bar)
    bar_c = np.asarray(pos) + rz @ np.array([0.0, depth / 2 + proud, h / 2])
    back = bar_c - out * 0.0051                            # the bar's face toward the drawer
    return scene, out, bar_c, back


def hook_pose(scene, out, bar_c, back):
    fwd = -out
    seeds = []
    for pitch in (20, 30):
        for roll in (-10, 0):
            X = np.array([0.0, 0.0, -1.0]); Z = fwd.copy(); lat = np.cross(Z, X)
            Rp = R.from_rotvec(np.deg2rad(pitch) * lat / np.linalg.norm(lat)).as_matrix()
            Rr = R.from_rotvec(np.deg2rad(roll) * fwd).as_matrix()
            X2, Z2 = Rr @ Rp @ X, Rr @ Rp @ Z
            M = np.stack([X2, np.cross(Z2, X2), Z2], 1)
            for b, u in ((0.12, 0.07), (0.14, 0.05)):
                seeds.append([(bar_c - fwd * b + np.array([0, 0, u])).tolist(), R.from_matrix(M).as_euler("XYZ").tolist()])
    return pose_from_points(scene, {"index": back.tolist()}, {"index": fwd.tolist()}, seeds=seeds, fix_orientation=True,
                            palm_face_at=bar_c.tolist(), curl_free=0.75)


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        # case 1: a deep unit with a bar 35 mm proud, pulled 12 cm with the hook dropped in vertically, closed shape
        sc, out, bc, bk = make_case(tmp, "unitA", (0.0, 0.0, 0.0), 0.0, 0.113, 0.254, 0.103, 0.035, 0.022)
        pose = hook_pose(sc, out, bc, bk)
        r = T(sc, pose, "tray_slide", travel_m=0.12, open_frac=0.0, approach_dir=[0, 0, 1])
        print(r["verdict"], r["issues"])
        if r["ok"] and r["measured"]["fraction"] >= 0.8:
            print("[PASS] hook dropped in vertically pulls a 35 mm-proud bar 12 cm")
        else:
            print("[FAIL] case 1", r["measured"]); fails += 1
        # case 2: a wider, taller unit turned 90 degrees with a 40 mm-proud bar, slight pre-opening, via waypoint
        sc2, out2, bc2, bk2 = make_case(tmp, "unitB", (0.05, 0.02, 0.0), 90.0, 0.14, 0.22, 0.12, 0.04, 0.025)
        pose2 = hook_pose(sc2, out2, bc2, bk2)
        r2 = T(sc2, pose2, "tray_slide", travel_m=0.08, open_frac=0.1, approach_dir=[0, 0, 1],
               approach_waypoints=[[0.0, 0.0, 0.02]])
        print(r2["verdict"], r2["issues"])
        if r2["ok"] and r2["measured"]["n_waypoints"] == 1:
            print("[PASS] rotated unit, 40 mm-proud bar, pre-opened 10% through a waypoint pulls 8 cm")
        else:
            print("[FAIL] case 2", r2["measured"]); fails += 1
        # case 3: the same hook arriving wide open (55%) along its own approach does not get behind the bar
        r3 = T(sc, pose, "tray_slide", travel_m=0.12, open_frac=0.55)
        print(r3["verdict"], r3["issues"])
        if not r3["ok"] and r3["issues"]:
            print("[PASS] a wide-open approach is reported as a failure with an actionable issue")
        else:
            print("[FAIL] case 3 should fail", r3["measured"]); fails += 1
        # case 4: misuse raises
        bad = 0
        for kw in ({"open_frac": 1.5}, {"direction": 0}, {"travel_m": -0.1}, {"approach_dir": [0, 0, 0]}):
            try:
                T(sc, pose, "tray_slide", **kw)
            except ValueError:
                bad += 1
        try:
            T(sc, pose, "no_such_joint_anywhere")
        except ValueError:
            bad += 1
        if bad == 5:
            print("[PASS] misuse (open_frac, direction, travel, zero approach, unknown joint) raises ValueError")
        else:
            print(f"[FAIL] misuse raised {bad}/5"); fails += 1
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
