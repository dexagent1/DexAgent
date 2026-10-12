import sys, tempfile
from pathlib import Path
import mujoco
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.common import hand as H
from tool_library.verifiers.pinch_support_clearance_check import pinch_support_clearance_check

ROOT = Path(H.__file__).resolve().parents[2]
HAND = ROOT / "setup" / "robot" / "sharpa_right.xml"
tmp = Path(tempfile.mkdtemp())
ROD = tmp / "rod.xml"
ROD.write_text('<mujoco model="rod"><worldbody><body name="rod" pos="0 0 0.0066"><freejoint name="rod_free"/>'
               '<geom name="rod_geom" type="cylinder" size="0.0066 0.06" euler="0 90 0" mass="0.012"/></body></worldbody></mujoco>')
BOX = tmp / "box.xml"
BOX.write_text('<mujoco model="box"><worldbody><body name="box" pos="0 0 0.06"><freejoint name="box_free"/>'
               '<geom name="box_geom" type="box" size="0.02 0.03 0.06" mass="0.1"/></body></worldbody></mujoco>')
fails = 0


def scene(xml):
    return {"hand_xml": str(HAND), "side": "right", "objects": [{"name": "obj", "xml": str(xml), "pos": [0, 0, 0], "yaw_deg": 0}], "floor": True}


def fingers_down_pose(sc, xy, lowest_z):
    """Among axis-aligned orientations, the one whose thumb/index pads sit lowest in the hand, placed so the hand's
    lowest collision point is at lowest_z (a measured test pose, not a grasp)."""
    model, data0, rig, ids = H.open_scene(sc)
    best = None
    for e in ([0, 0, 0], [np.pi, 0, 0], [np.pi / 2, 0, 0], [-np.pi / 2, 0, 0], [0, np.pi / 2, 0], [0, -np.pi / 2, 0]):
        d = mujoco.MjData(model); d.qpos[:] = data0.qpos
        p = {"pos": [xy[0], xy[1], 0.5], "euler": e, "finger_targets": {}}
        H.apply_pose(model, d, rig, p); mujoco.mj_forward(model, d)
        lo = float(H.object_extent(model, d, sorted(rig.collision_gids))["lo"][2])
        pads = np.mean([d.site_xpos[rig.fingertip_sites[f]][2] for f in ("thumb", "index")])
        if best is None or pads - lo < best[0]:
            best = (pads - lo, e, lo)
    _, e, lo = best
    return {"pos": [xy[0], xy[1], 0.5 + (lowest_z - lo)], "euler": list(map(float, e)), "finger_targets": {}, "grip_fingers": ["thumb", "index"]}


def check(name, cond):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name)
    fails += 0 if cond else 1


# 1. hand far above a rod: no support contact, pads above the rod -> fails on 'above the top'
sc = scene(ROD)
p = fingers_down_pose(sc, (0.0, 0.0), 0.10)
r = pinch_support_clearance_check(sc, p, round_section=True)
check("hand above rod: not ok, pads above the top, no support penetration",
      (not r["ok"]) and r["support_mm"] == 0.0 and all(v["above_top"] for v in r["pads"].values()))

# 2. the same hand sunk 20 mm into the floor beside the rod -> support_mm ~ 20, lift ~ 0.02
p2 = fingers_down_pose(sc, (0.0, 0.25), -0.020)
r2 = pinch_support_clearance_check(sc, p2, round_section=True)
# the placement uses geom bounding boxes, which enclose the hulls: the true depth is positive and at most 20 mm
print("   sunk case support_mm =", r2["support_mm"])
check("hand sunk up to 20 mm: support penetration measured, pads lifted by it, reported first",
      (not r2["ok"]) and 3.0 < r2["support_mm"] <= 20.5 and abs(r2["support_lift_m"] - r2["support_mm"] / 1000.0) < 1e-4
      and "inside the support" in r2["issues"][0])

# 3. a tall box: hand lowest point 5 mm above the floor beside it, pads below the box top -> ok
sc3 = scene(BOX)
p3 = fingers_down_pose(sc3, (0.0, 0.25), 0.005)
r3 = pinch_support_clearance_check(sc3, p3)
check("tall box, hand clear of the floor: ok", r3["ok"] and r3["support_mm"] <= 3.0 and r3["object"]["hi_z"] > 0.11)

# 4. round-section rule: the same clear pose next to the rod is judged against the hold line, not the top
r4 = pinch_support_clearance_check(sc, p3, round_section=True, mu=0.85)
hl = r4["object"]["centre_z"] + r4["object"]["radius_m"] * np.sin(np.arctan(0.85))
check("round-section hold line computed from axis and radius", abs(r4["hold_line_z_m"] - hl) < 1e-3 and r4["object"]["radius_m"] < 0.01)

# 5. misuse raises
try:
    pinch_support_clearance_check(sc, p, tol_mm=-1)
    check("negative tol_mm raises", False)
except ValueError:
    check("negative tol_mm raises", True)
sys.exit(1 if fails else 0)
