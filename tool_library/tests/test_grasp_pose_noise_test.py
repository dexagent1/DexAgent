"""grasp_pose_noise_test: the grasp must still lift and hold with the hand 2 mm / 2 deg off.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_pose_noise_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402


def main():
    from tool_library.common import grasp_robust as GR
    from tool_library.verifiers.grasp_pose_noise_test import grasp_pose_noise_test as T
    rep = F.Report()
    t0 = time.time()
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    weak = F.pose(F.BOX_PINCH)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)

    r = T(box, pinch)
    rep.check("three-finger box pinch survives 2 mm / 2 deg errors", r["ok"] and r["pass_frac"] == 1.0 and len(r["trials"]) == 6, r["worst"])
    rep.check("trials come in antithetic pairs", all(abs(a + b) < 1e-6 for a, b in zip(r["trials"][0]["offset_mm"], r["trials"][1]["offset_mm"])),
              [t["offset_mm"] for t in r["trials"][:2]])
    r = T(cyl, top, n_trials=4, pivot="pads")
    rep.check("left-hand top grasp survives errors about its pads", r["ok"] and r["noise"]["pivot"] == "pads", r.get("issues"))

    # the two-pad pinch lifts at the exact pose but loses the box under half the calibration errors
    r = T(box, weak)
    bad = [t for t in r["trials"] if not t["ok"]]
    rep.check("two-pad pinch fails under calibration error though the exact pose lifts", (not r["ok"]) and r["nominal"]["ok"] and bad
              and bad[0]["why"] and r["issues"], (r["pass_frac"], [t["why"] for t in bad]))
    # The regenerated, non-crossing pinch is near the top edge at +20 mm:
    # it holds nominally but loses two of the six calibration perturbations.
    edge = GR.perturb_pose(pinch, [0.0, 0.0, 0.020], [0.0, 0.0, 0.0])
    r = T(box, edge)
    rep.check("edge pinch fails under calibration error though the exact pose lifts", (not r["ok"]) and r["nominal"]["ok"], r["pass_frac"])
    r = T(box, edge, min_pass_frac=0.5)
    rep.check("min_pass_frac relaxes the verdict only for a valid nominal grasp",
              r["ok"] == (r["nominal"]["ok"] and r["pass_frac"] >= 0.5), r["pass_frac"])

    # a grasp 4 cm above the box does not even lift at the exact pose
    r = T(box, GR.perturb_pose(pinch, [0.0, 0.0, 0.04], [0.0, 0.0, 0.0]), n_trials=2)
    rep.check("pose above the object is rejected at the nominal trial", not r["ok"] and not r["nominal"]["ok"]
              and "unperturbed" in r["issues"][0], r["issues"])

    rep.raises("unknown pivot is refused", lambda: T(box, pinch, pivot="elbow"))
    rep.raises("zero trials are refused", lambda: T(box, pinch, n_trials=0))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
