"""grasp_opposition_test: the contacts must squeeze the object between opposing groups.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_opposition_test.py
"""
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402


def thumb_open(rig, pose, also_middle=False):
    """The pinch with the thumb straightened away from the box: only the index (and with also_middle the middle, on the
    same face) is left on it."""
    from tool_library.common import hand as H
    q = copy.deepcopy(pose)
    q.pop("finger_targets_squeezed", None)
    for k in list(q["finger_targets"]):
        if k.startswith("thumb") and H.joint_role(k) == "flex":
            q["finger_targets"][k] = rig.finger_range[k][0]
        if also_middle and k.startswith("middle"):
            q["finger_targets"][k] = pose["finger_targets"]["index" + k[len("middle"):]]
    q["grip_fingers"] = ["index", "middle"] if also_middle else ["index"]
    return q


def main():
    from tool_library.common import grasp_robust as GR
    from tool_library.verifiers.grasp_opposition_test import grasp_opposition_test as T
    rep = F.Report()
    t0 = time.time()
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)
    rig = GR.Case(GR.load_scene(box)).rig

    r = T(box, pinch)
    rep.check("three-finger pinch: thumb opposed to the fingers across the box", r["ok"] and r["judged"] == "lifted" and r["opposition_deg"] > 150
              and "thumb" in r["best_pair"]["pair"] and min(r["com_inside_mm"]) > 10, r["best_pair"])
    r = T(cyl, top)
    rep.check("left-hand top grasp: rim pads squeeze across their common push", r["ok"] and r["opposition_deg"] >= 150
              and r["best_pair"]["across_common_push_deg"] is not None, r["best_pair"])
    r = T(box, pinch, lift_m=0.0)
    rep.check("seated reading also passes", r["ok"] and r["judged"] == "seated" and r["lifted"] is None, r.get("issues"))

    # the index alone on one face: a hook that holds nothing. Its clamp presses the pad 4 mm into the face (the executor's
    # one-sided press, judged in the world frame since hand.executor_clamp), so the box leans on the idle middle finger
    # resting on its top: rejected as a single contact or as forces that all push one way
    r = T(box, thumb_open(rig, pinch))
    rep.check("single finger on one face is rejected", (not r["ok"]) and any("single contact" in i or "only" in i or "push one way" in i
                                                                             for i in r["issues"]), r["issues"])
    # index and middle side by side on the same face: two groups pushing the same way
    r = T(box, thumb_open(rig, pinch, also_middle=True))
    rep.check("two fingers on the same face are rejected", not r["ok"], r["issues"])
    rep.raises("an opposition angle above 180 is refused", lambda: T(box, pinch, min_opposition_deg=200))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
