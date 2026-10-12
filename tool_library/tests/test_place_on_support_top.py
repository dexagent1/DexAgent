"""Placement heights for `on`, `over` and `into` from the bodies' real collision geometry (subgoal._target_from with
episode.support_top / bodies_extent). The planner used bounding spheres: on 2x2 stud bricks it put the support's top
3.4 cm too high and the carried brick's bottom 2.1 cm too low, so a brick placed `on` another hovered 5.5 cm above it
(the lego run compensated with offset -0.06). Cases: a brick on a two-brick stack lands on the stud tops; a small cube
on a brick rests on the stud caps (not above their apex); a lid on a cup lands on the rim; a cube `into` a cup lands on
its floor; a cup `into` a bowl lands inside it (between floor and rim); `over` clears the whole support; `on` the table;
exact geom extents of rotated primitives; the planner's place target and its recorded `placement`. Pure kinematics,
a few seconds.

    MUJOCO_GL=egl python tool_library/tests/test_place_on_support_top.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import episode as E  # noqa: E402
from tool_library.common.subgoal import SubgoalPlanner  # noqa: E402
from tool_library.skills.construct_bowl import construct_bowl  # noqa: E402
from tool_library.skills.construct_cup import construct_cup  # noqa: E402
from tool_library.skills.construct_stud_brick import construct_stud_brick  # noqa: E402
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_ontop_", dir=os.environ.get("TMPDIR")))
FAILS = 0
# the lego run's brick (EgoDex build/unstack): 3 cm pitch, 26 mm body, 6.5 mm studs with a clutch
BRICK = dict(pitch=0.030, body_height=0.026, stud_radius=0.0082, stud_height=0.0065, clutch_margin=0.0008, clearance=0.0006,
             lead_in_frac=0.6, cap_friction=0.05, mass=0.02)


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def box_xml(name, half, mass=0.05):
    d = TMP / "assets" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.xml").write_text(f'<mujoco model="{name}"><worldbody><body name="{name}" pos="0 0 {half[2]}"><freejoint/>'
                                   f'<geom name="{name}_g" type="box" size="{half[0]} {half[1]} {half[2]}" mass="{mass}"/></body></worldbody></mujoco>')
    return str(d / f"{name}.xml")


def old_top(model, data, bodies):
    """The planner's former estimate: bounding spheres."""
    return max(float(data.geom_xpos[g][2] + model.geom_rbound[g]) for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies)


def main():
    H, HS = BRICK["body_height"], BRICK["stud_height"]
    a = construct_stud_brick(str(TMP / "assets" / "bricka"), name="bricka", **BRICK)
    b = construct_stud_brick(str(TMP / "assets" / "brickb"), name="brickb", **BRICK)
    c = construct_stud_brick(str(TMP / "assets" / "brickc"), name="brickc", **BRICK)
    cup = construct_cup(str(TMP / "assets" / "cup"), name="cup", base_radius=0.034, rim_radius=0.038, height=0.09, include_floor=False)
    bowl = construct_bowl(str(TMP / "assets" / "bowl"), name="bowl", include_floor=False)
    cupb = construct_cup(str(TMP / "assets" / "cupb"), name="cupb", base_radius=0.03, rim_radius=0.034, height=0.08, include_floor=False)
    scene = TMP / "scene.xml"
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), out_xml=str(scene), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()}, assets=[
        {"name": "bricka", "xml": a["xml"], "x": 0.12, "y": 0.40}, {"name": "brickb", "xml": b["xml"], "x": 0.12, "y": 0.40, "z": H},
        {"name": "brickc", "xml": c["xml"], "x": 0.22, "y": 0.30}, {"name": "cup", "xml": cup["xml"], "x": -0.10, "y": 0.40},
        {"name": "bowl", "xml": bowl["xml"], "x": -0.12, "y": 0.62}, {"name": "cupb", "xml": cupb["xml"], "x": 0.05, "y": 0.62},
        {"name": "cube", "xml": box_xml("cube", (0.01, 0.01, 0.01)), "x": -0.25, "y": 0.30},
        {"name": "lid", "xml": box_xml("lid", (0.06, 0.06, 0.005)), "x": 0.30, "y": 0.55}])
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
    sp = SubgoalPlanner(m, d.qpos.copy())
    sub = lambda n: E.subtree_bodies(m, m.body(n).id)
    root = lambda n: f"{n}_{n}"

    # 1. a brick on a two-brick stack: its bottom on the stud tops (2 H + stud height), not 5 cm above
    tgt, ref = sp._target_from({"on": root("brickb")}, root("brickc"))
    apex = 2 * H + HS
    check("on a stacked brick: the brick's bottom lands 2 mm above the stud tops", abs(tgt[2] - (apex + 0.002)) < 2e-4 and ref == root("brickb"),
          f"z {tgt[2]:.4f} (stud tops {apex:.4f}) via {sp._target_info['method']}")
    old_low = min(float(d.geom_xpos[g][2] - m.geom_rbound[g]) for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub(root("brickc")))
    old_z = old_top(m, d, sub(root("brickb"))) - (old_low - d.xpos[m.body(root("brickc")).id][2]) + 0.002
    check("the bounding-sphere estimate it replaces was 5 cm too high", old_z - tgt[2] > 0.04, f"old {old_z:.4f} vs new {tgt[2]:.4f}")
    # 2. a part smaller than the stud grid rests on the stud caps: between the cap bases and the apex
    tc, _ = sp._target_from({"on": root("brickc")}, root("cube"))
    bottom_c = tc[2] - 0.01 - 0.002
    side_top = H + (1 - BRICK["lead_in_frac"]) * HS
    check("a cube smaller than the stud grid rests on the stud caps, not above their apex", side_top - 1e-4 <= bottom_c <= H + HS + 1e-4,
          f"cube bottom {bottom_c:.4f} in [{side_top:.4f}, {H + HS:.4f}]")
    # 3. a lid wider than a cup lands on its rim
    tl, _ = sp._target_from({"on": root("cup")}, root("lid"))
    check("a lid on a cup lands on the rim", abs((tl[2] - 0.005) - (cup["height_m"] + 0.002)) < 1.5e-3,
          f"lid bottom {tl[2] - 0.005:.4f}, rim {cup['height_m']:.4f}")
    # 4. a small cube into the cup lands on its floor (3 mm clearance), not on its rim
    ti, _ = sp._target_from({"into": root("cup")}, root("cube"))
    check("a cube into a cup lands on its inner floor", abs((ti[2] - 0.01) - (cup["bottom_inner_z_m"] + 0.003)) < 1.5e-3,
          f"cube bottom {ti[2] - 0.01:.4f}, floor {cup['bottom_inner_z_m']:.4f}")
    # 5. a cup into a bowl (the place-cup-in-bowl task): inside, between the bowl's floor and its rim
    tb, _ = sp._target_from({"into": root("bowl")}, root("cupb"))
    check("a cup into a bowl lands inside it, between floor and rim", bowl["bottom_inner_z_m"] < tb[2] < bowl["height_m"],
          f"cup bottom {tb[2]:.4f}, bowl floor {bowl['bottom_inner_z_m']:.4f} rim {bowl['height_m']:.4f}")
    # 6. over: the travel height clears the whole support (its rim), whatever lies under the part
    to, _ = sp._target_from({"over": root("bowl"), "height_m": 0.05}, root("cupb"))
    check("over a bowl: the part's bottom height_m above the rim", abs(to[2] - (bowl["height_m"] + 0.05)) < 1.5e-3, f"z {to[2]:.4f}")
    # 7. on the table: the bottom at z = 0 (+2 mm); the old table estimate was 2 m up (the table slab's sphere)
    tt, _ = sp._target_from({"on": "table", "offset": [0.0, 0.05, 0.0]}, root("cube"))
    check("on the table: the cube's bottom 2 mm above the tabletop", abs(tt[2] - 0.012) < 1e-4, f"z {tt[2]:.4f}")
    # 8. exact extents of rotated primitives (a box and a cylinder tilted 30 deg about x)
    sx = mujoco.MjSpec(); bb = sx.worldbody.add_body(); bb.quat = [np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0]
    gb = bb.add_geom(); gb.type = mujoco.mjtGeom.mjGEOM_BOX; gb.size = [0.05, 0.02, 0.01]
    gc = bb.add_geom(); gc.type = mujoco.mjtGeom.mjGEOM_CYLINDER; gc.size = [0.02, 0.03, 0]
    mx = sx.compile(); dx = mujoco.MjData(mx); mujoco.mj_forward(mx, dx)
    s30, c30 = np.sin(np.pi / 6), np.cos(np.pi / 6)
    box_h, cyl_h = 0.02 * s30 + 0.01 * c30, 0.03 * c30 + 0.02 * s30
    check("geom_extent is exact for a tilted box and cylinder", abs(E.geom_extent(mx, dx, 0, (0, 0, 1))[1] - box_h) < 1e-9
          and abs(E.geom_extent(mx, dx, 1, (0, 0, 1))[1] - cyl_h) < 1e-9, f"{box_h:.4f} {cyl_h:.4f}")
    # 9. the planner: a place `on` the stack targets the stud tops and records how it measured
    sp2 = SubgoalPlanner(m, d.qpos.copy(), state={"held": {"right": root("brickc")}})
    tgt2, _ = sp2._target_from({"on": root("brickb")}, root("brickc"))
    pl = plan_subgoal_keyframes(m, {"index": 1, "motion": "carry", "side": "right", "object": root("brickc"), "on": root("brickb")},
                                qpos0=d.qpos.copy(), state={"held": {"right": root("brickc")}, "grip": {}}, auto_via=False)
    ex = pl["expect"]
    check("a planned carry `on` a brick targets the stud tops and records the measurement",
          abs(ex["target_pos"][2] - tgt2[2]) < 1e-6 and ex["placement"]["support_top_z"] == round(apex, 5) and ex["placement"]["method"] == "footprint",
          json.dumps(ex["placement"]))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
