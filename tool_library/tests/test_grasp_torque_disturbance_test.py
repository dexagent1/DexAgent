"""grasp_torque_disturbance_test: the lifted object must not turn in the hand under a torque up to the limit.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_torque_disturbance_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402


def main():
    from tool_library.common import grasp_robust as GR
    from tool_library.verifiers.grasp_torque_disturbance_test import grasp_torque_disturbance_test as T
    rep = F.Report()
    t0 = time.time()
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)

    r = T(box, pinch)
    lim = 0.12 * 9.81 * r["lever_m"]
    rep.check("three-finger pinch resists the box's own weight moment about all six axis directions", r["ok"] and len(r["axes"]) == 6
              and abs(r["torque_limit_nm"] - lim) < 1e-3 and 0.025 < r["lever_m"] < 0.04, (r["torque_limit_nm"], r.get("issues")))
    r = T(cyl, top)
    rep.check("left-hand top grasp resists the cylinder's weight moment", r["ok"], r.get("issues"))
    r = T(box, pinch, torque_nm=0.1)
    rep.check("three-finger pinch resists a 0.1 N m tool load", r["ok"] and r["torque_limit_nm"] == 0.1, [(a["axis"], a["turn_deg"]) for a in r["axes"]])

    # a 0.2 N m load (a tool meeting resistance) twists the box out of the pinch: pads hold torque only through their
    # 2 cm contact patches and the finger spread
    r = T(box, pinch, torque_nm=0.2)
    broke = [a for a in r["axes"] if a["breakaway_nm"] is not None]
    rep.check("0.2 N m twists the box out of the pinch", (not r["ok"]) and broke and r["issues"] and "broke away" in r["issues"][0],
              [(a["axis"], a["turn_deg"], a["breakaway_nm"]) for a in r["axes"]])
    # a pose above the object never lifts it: reported, no twist is applied
    r = T(box, GR.perturb_pose(pinch, [0.0, 0.0, 0.04], [0.0, 0.0, 0.0]))
    rep.check("a grasp that does not lift is rejected before twisting", (not r["ok"]) and not r["axes"] and "does not lift" in r["issues"][0], r["issues"])

    rep.raises("negative torque is refused", lambda: T(box, pinch, torque_nm=-1.0))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
