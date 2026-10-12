"""hand.deepest_overlap and pose_from_points against the floor: the pads of keep_tips are excused against the objects
but not against the table (a scissors pinch reported 0-6 mm with the thumb's distal hull 7.5-9.9 mm in the floor), and
with visual_floor the hand's visible meshes are measured against the floor plane as the Stage 3 penetration audits do.
Cases: a hand lowered until its middle pad is 5 mm into the floor reports >= 4 mm with that pad kept; the same hand 3
mm above the floor reports nothing either way; a height where only a visible mesh dips 3 mm into the floor is seen by
visual_floor alone; pose_from_points asked to put a pad 5 mm under the floor either keeps it out (a residual) or reports
the overlap, never a clean pad in the table; a pinch on a block's sides reports no floor overlap."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.pose_from_points import pose_from_points  # noqa: E402

HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
TMP = Path(tempfile.mkdtemp(prefix="overlap_floor_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    p = TMP / "blk.xml"
    p.write_text('<mujoco model="blk"><worldbody><body name="blk" pos="0 0 0.02"><freejoint/>'
                 '<geom name="g" type="box" size="0.03 0.03 0.02" mass="0.1"/></body></worldbody></mujoco>')
    scene = {"hand_xml": HAND, "side": "right", "floor": True, "objects": [{"name": "blk", "xml": str(p), "pos": [0.0, 0.45, 0.0]}]}
    model, data0, rig, ids = H.open_scene(scene)
    fl = ids["floor_gid"]
    d = mujoco.MjData(model)
    straight = {k: (lo if H.joint_role(k) in ("flex", "pinky_arch") else 0.0) for k, (lo, hi) in rig.finger_range.items()}
    xy = np.array([0.15, 0.30])                     # away from the block

    def at(z, eul):
        d.qpos[:] = data0.qpos
        H.apply_pose(model, d, rig, {"pos": [xy[0], xy[1], z], "euler": list(eul)}, straight)
        return d

    def pad_gap(dd, f):
        return min(float(mujoco.mj_geomDistance(model, dd, g, fl, 0.5, None)) for g in rig.tip_geoms(f))

    # 1. fingers pointing 50 deg down, the middle pad (the lowest geometry) lowered 5 mm into the floor
    down = (0.0, np.pi / 2 + 0.9, 0.0)
    z = 0.25
    for _ in range(4):
        z -= pad_gap(at(z, down), "middle")
    dd = at(z - 0.005, down)
    pad = -pad_gap(dd, "middle")
    kept = H.deepest_overlap(model, dd, rig, ids, keep_tips=("middle",))
    check("a middle pad 5 mm into the floor reports >= 4 mm though the pad is kept", pad > 0.0045 and 0.004 <= kept <= 0.0055,
          f"pad {pad * 1000:.1f} mm, deepest_overlap {kept * 1000:.1f} mm")
    # 2. the same hand 3 mm above: nothing either way
    dc = at(z + 0.003, down)
    clean = (H.deepest_overlap(model, dc, rig, ids, keep_tips=H.FINGERS), H.deepest_overlap(model, dc, rig, ids, keep_tips=(), visual_floor=True))
    check("the same hand 3 mm above the floor reports nothing either way", max(clean) == 0.0, f"{[round(v * 1000, 2) for v in clean]} mm")
    # 3. on its side (pinky edge down) the pinky metacarpal's visible shell hangs 4.6 mm below every collision geom: a
    # height where only the visible mesh is in the floor
    side = (np.pi / 2, 0.0, 0.0)

    def lowest_visible(dd):                         # height of the lowest visible-mesh vertex above the floor (z = 0)
        return min(float(np.min((V @ dd.geom_xmat[g].reshape(3, 3).T + dd.geom_xpos[g])[:, 2])) for g, V in H.visual_floor_hulls(model, rig))

    zs = 0.25
    zs -= lowest_visible(at(zs, side))
    dv = at(zs - 0.003, side)
    vis = H.visual_floor_depth(model, dv, rig, fl)
    col = H.deepest_overlap(model, dv, rig, ids, keep_tips=())
    both = H.deepest_overlap(model, dv, rig, ids, keep_tips=(), visual_floor=True)
    check("a visible mesh 3 mm in the floor is seen with visual_floor only", 0.0025 < vis < 0.0035 and col == 0.0 and abs(both - vis) < 1e-9,
          f"visual {vis * 1000:.1f} mm, collision {col * 1000:.1f} mm, with visual_floor {both * 1000:.1f} mm")
    # 4. pose_from_points asked to put the index pad 5 mm under the floor: the floor push-out keeps the pad out (a residual)
    # or the overlap says so; never a clean report of a pad in the table
    r = pose_from_points(scene, {"index": [0.12, 0.30, -0.005]}, {"index": [0.0, 0.0, 1.0]}, approach_dirs=[[0, 0, 1.0]], iters=200)
    pz = r.get("pose", r)
    d4 = mujoco.MjData(model); d4.qpos[:] = data0.qpos; H.apply_pose(model, d4, rig, pz)
    true_floor = max(-min(float(mujoco.mj_geomDistance(model, d4, g, fl, 0.5, None)) for g in rig.collision_gids), H.visual_floor_depth(model, d4, rig, fl), 0.0)
    check("pose_from_points never reports a pad in the floor as clean", pz["overlap_mm"] >= 1000 * true_floor - 0.5 and (pz["overlap_mm"] >= 4.0 or pz["worst_residual_mm"] >= 4.0),
          f"overlap {pz['overlap_mm']} mm, measured in the floor {1000 * true_floor:.1f} mm, residual {pz['worst_residual_mm']} mm")
    # 5. a pinch on the block's sides 25 mm up keeps the hand out of the floor (no false alarm)
    r2 = pose_from_points(scene, {"thumb": [0.03, 0.45, 0.025], "index": [-0.03, 0.45, 0.025]}, {"thumb": [1, 0, 0], "index": [-1, 0, 0]},
                          approach_dirs=[[0, 0, 1.0]], iters=240)
    p2 = r2.get("pose", r2)
    d2 = mujoco.MjData(model); d2.qpos[:] = data0.qpos; H.apply_pose(model, d2, rig, p2)
    check("a pinch on a block's sides keeps the hand's meshes out of the floor", H.visual_floor_depth(model, d2, rig, fl) < 0.002 and p2["overlap_mm"] < 4.0,
          f"visual floor {H.visual_floor_depth(model, d2, rig, fl) * 1000:.1f} mm, overlap {p2['overlap_mm']} mm")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
