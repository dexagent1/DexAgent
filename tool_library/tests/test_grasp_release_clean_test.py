"""grasp_release_clean_test: letting go must leave the object where it was set down.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_release_clean_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402


def main():
    from tool_library.common import grasp_robust as GR
    from tool_library.verifiers.grasp_release_clean_test import grasp_release_clean_test as T
    rep = F.Report()
    t0 = time.time()
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)
    grip = GR.executor_grip(GR.Case(GR.load_scene(box)), pinch)

    r = T(box, pinch)
    rep.check("three-finger pinch lets go cleanly", r["ok"] and r["shift_mm"] < 20 and r["rise_mm"] < 2 and not r["touching_after"], r)
    r = T(cyl, top)
    rep.check("left-hand top grasp lets go cleanly", r["ok"] and r["tip_deg"] < 5, r.get("issues"))
    r = T(box, pinch, lift_m=0.0, retreat_dir=(0.0, -0.3, 1.0))
    rep.check("release in place with a slanted retreat is clean", r["ok"], r.get("issues"))

    # the fingers keep the clamp and the hand backs off upward: the box is carried away with it
    r = T(box, pinch, open_targets=grip["press"])
    rep.check("a hand that does not open drags the box up", (not r["ok"]) and r["rise_mm"] > 20 and r["touching_after"], (r["rise_mm"], r["touching_after"]))
    # the fingers only relax to touch and the hand backs off sideways across the pads: the box is dragged along the table
    r = T(box, pinch, open_targets=grip["closed"], retreat_dir=(1.0, 0.0, 0.0))
    rep.check("a sideways retreat with the pads still on drags the box", (not r["ok"]) and r["shift_mm"] > 25 and "retreat" in r["issues"][0],
              (r["shift_mm"], r["issues"]))
    rep.raises("zero retreat direction is refused", lambda: T(box, pinch, retreat_dir=(0, 0, 0)))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
