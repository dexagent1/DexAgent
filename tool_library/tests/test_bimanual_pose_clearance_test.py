"""bimanual_pose_clearance_test: two grasps solved in separate hand scenes must keep the hands apart.

    MUJOCO_GL=egl python tool_library/tests/test_bimanual_pose_clearance_test.py
"""
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402


def moved(pose, dx, approach=None):
    q = copy.deepcopy(pose)
    q["pos"] = [pose["pos"][0] + dx, pose["pos"][1], pose["pos"][2]]
    if approach is not None:
        q["approach"] = approach
    return q


def main():
    from tool_library.common import grasp_robust as GR
    from tool_library.verifiers.bimanual_pose_clearance_test import bimanual_pose_clearance_test as T
    rep = F.Report()
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="dexagent_bimanual_"))
    right = F.pose(F.BOX_PINCH)          # right hand pinching a box at the origin
    left = F.pose(F.CYL_TOP)             # left hand on a cylinder at the origin

    r = T(moved(left, -0.35), right)
    rep.check("hands on objects 35 cm apart are clear", r["ok"] and r["min_clearance_mm"] > 100 and len(r["phases"]) == 4, r["min_clearance_mm"])
    r = T(moved(left, -0.14), right, check_approach=False)
    rep.check("hands 2.5 cm apart at their poses pass a 10 mm clearance", r["ok"] and 10 < r["min_clearance_mm"] < 40, r["min_clearance_mm"])
    r = T(moved(left, -0.14), right, min_clearance_mm=30.0, check_approach=False)
    rep.check("the same pair fails a 30 mm clearance", not r["ok"], r["min_clearance_mm"])

    # both grasps solved on the same spot: palms inside each other
    r = T(left, right)
    rep.check("overlapping grasps are rejected at their poses", (not r["ok"]) and r["phases"][0]["min_mm"] < 0 and r["phases"][0]["pair"]
              and "interpenetrate" in r["issues"][0], r["phases"][0])
    # clear at the poses (8 cm apart) but the left hand comes in from the right hand's side
    r = T(moved(left, -0.20, approach=[1.0, 0.0, 0.0]), right)
    final = r["phases"][0]
    rep.check("an approach through the other hand is rejected although the poses are clear", (not r["ok"]) and final["ok"]
              and any(not p["ok"] and "approach" in p["name"] for p in r["phases"]), [(p["name"], p["min_mm"]) for p in r["phases"]])
    r = T(moved(left, -0.20, approach=[1.0, 0.0, 0.0]), right, approach_sides=["right"])
    rep.check("only the moving hand's approach is checked when the other already holds", r["ok"] and len(r["phases"]) == 2,
              [(p["name"], p["min_mm"]) for p in r["phases"]])

    # inputs as files with hand scenes; a scene of the wrong hand is refused
    (tmp / "left.json").write_text(json.dumps(moved(left, -0.35)))
    (tmp / "hs_left.json").write_text(json.dumps(F.scene("cyl", "left")))
    (tmp / "hs_right.json").write_text(json.dumps(F.scene("box", "right")))
    r = T(str(tmp / "left.json"), str(F.FX / f"{F.BOX_PINCH}.json"), left_scene=str(tmp / "hs_left.json"), right_scene=str(tmp / "hs_right.json"))
    rep.check("JSON poses and hand scenes", r["ok"], r.get("issues"))
    rep.raises("a right-hand scene given as the left scene is refused", lambda: T(left, right, left_scene=str(tmp / "hs_right.json")))
    rep.raises("too few approach samples are refused", lambda: T(left, right, n_samples=1))
    rep.raises("an unknown approach side is refused", lambda: T(left, right, approach_sides=["middle"]))

    # a configuration where MuJoCo 3.10's mj_geomDistance read an exact 0.0 for two palms 123 mm apart: re-measured
    import mujoco
    model, rigs = GR.two_hand_model(F.hand_xml("left"), F.hand_xml("right"))
    d = mujoco.MjData(model)
    rigs["right"].teleport(d, right["pos"], right["euler"])
    rigs["left"].teleport(d, [-0.23643, left["pos"][1], left["pos"][2]], left["euler"])
    mujoco.mj_kinematics(model, d)
    pl, pr = model.geom("handl_collision_hand_left_palm_0").id, model.geom("handr_collision_hand_right_palm_0").id
    rep.check("palm-palm distance at a known false-zero pose is the real 12 cm", GR.geom_distance(model, d, pl, pr) > 0.1,
              GR.geom_distance(model, d, pl, pr))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
