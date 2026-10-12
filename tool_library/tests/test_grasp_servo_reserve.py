"""The hand-scene grasp stress tests judge the finger servos' reserve the way finger_servo_saturation_audit judges an
episode (hand.ServoReserve: the smallest share of the finger torque caps with which the hand's contacts carry the part,
on the executor's 30 fps frames while the hand alone holds it), so a grasp they pass is not failed later for holding only
at the torque cap. On a 600 g hard drive (construct_hard_disk, the computer-installation drive): a pinch through its
centre lifts with reserve (about 70 % of the cap); a pinch about 23 mm off its centre with the thumb on the far side lifts
it too but only with the thumb's abduction servo at about 104 % of its cap, and grasp_lift_test / grasp_stress_test_preopen
fail it at servo_reserve (with the idle fingers curled clear of each other, the earlier 29 mm pinch no longer lifts at all); servo_reserve=False gives the earlier verdict; the thresholds and the linear program are the
audit's; a rollout without a carry phase judges nothing.

    python tool_library/tests/test_grasp_servo_reserve.py
"""
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.construct_hard_disk import construct_hard_disk  # noqa: E402
from tool_library.verifiers import finger_servo_saturation_audit as FSA  # noqa: E402
from tool_library.verifiers.grasp_lift_test import grasp_lift_test  # noqa: E402
from tool_library.verifiers.grasp_stress_test_preopen import grasp_stress_test_preopen  # noqa: E402

HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
FIX = ROOT / "tool_library" / "tests" / "fixtures" / "grasp_servo_reserve"
TMP = Path(tempfile.mkdtemp(prefix="servo_reserve_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def reserve(rep):
    return next((p for p in rep.get("phases", []) if p.get("name") == "servo_reserve"), None)


def main():
    b = construct_hard_disk(str(TMP / "drive"), name="hdd_right", disk_l=0.133, disk_w=0.092, disk_h=0.024, tray_t=0.004, handle_l=0.012, mass=0.6)
    xml = b["xml"] if Path(b["xml"]).is_absolute() else str(TMP / "drive" / Path(b["xml"]).name)
    scene = {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": "hdd_right_hdd_right",
             "objects": [{"name": "hdd_right", "xml": xml, "pos": [0.1, 0.3706, 0.0], "quat": [-0.06087432674132177, 0.0, 0.0, 0.998145438472666]}]}
    centre = json.loads((FIX / "drive_pinch_centre.json").read_text())
    off = json.loads((FIX / "drive_pinch_off_centre.json").read_text())
    # 1. through the centre: lifts with reserve
    a = grasp_lift_test(scene, centre)
    ra = reserve(a)
    check("a pinch through the drive's centre lifts it with servo reserve", a["ok"] and ra and ra["judged"] > 10 and ra["median_s"] < H.SAT_FRAC,
          f"lifts {[x['achieved_mm'] for x in a['lifts']]}, reserve {ra}")
    p = grasp_stress_test_preopen(scene, centre)
    check("grasp_stress_test_preopen passes it too", p["ok"], str(reserve(p)))
    # 2. about 23 mm off the centre, thumb on the far side: lifts, but only with a servo at its cap
    b2 = grasp_lift_test(scene, off)
    rb = reserve(b2)
    rose = all(x["achieved_mm"] > 0.5 * x["commanded_mm"] for x in b2["lifts"])
    check("an off-centre pinch that lifts only with a finger servo at its cap fails at servo_reserve", rose and not b2["ok"]
          and b2["first_fail"] == "servo_reserve" and rb["saturated_frac"] >= H.SAT_HOLD_FRAC and rb["limiting_joint"] == "thumb_CMC_AA",
          f"lifts {[x['achieved_mm'] for x in b2['lifts']]}, reserve {rb}")
    q = grasp_stress_test_preopen(scene, off)
    check("grasp_stress_test_preopen fails it at servo_reserve", not q["ok"] and q["first_fail"] == "servo_reserve", str(reserve(q)))
    # 3. without the reserve check the earlier verdict (it lifts)
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    r0 = H.stress_rollout(model, data0, rig, ids, off, bid, phases=("approach", "close", "lift"), servo_reserve=False)
    check("servo_reserve=False gives the lift verdict alone", r0["ok"] and not any(p_["name"] == "servo_reserve" for p_ in r0["phases"]), str(r0["first_fail"]))
    # 4. the audit's thresholds and linear program
    sig = inspect.signature(FSA.finger_servo_saturation_audit).parameters
    cons = [(np.array([0.0, 0.01, 0.0]), np.array([0.0, -1.0, 0.0]), 1.2, 0.02, 0.001, np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.0]])),
            (np.array([0.0, -0.01, 0.0]), np.array([0.0, 1.0, 0.0]), 1.2, 0.02, 0.001, np.array([[0.0, 0.0], [0.0, -0.05], [0.0, 0.0]]))]
    capv = np.array([0.8, 0.8])
    s_h = H.finger_torque_need(cons, np.zeros(3), np.array([0.0, 0.0, 2.0]), [0, 1], capv)
    s_a = FSA._solve(cons, np.zeros(3), np.array([0.0, 0.0, 2.0]), [0, 1], capv, 1.5)
    check("the stress tests use the audit's thresholds and its linear program", sig["sat_frac"].default == H.SAT_FRAC and sig["hold_frac"].default == H.SAT_HOLD_FRAC
          and sig["mu_cap"].default == H.SAT_MU_CAP and s_h is not None and s_a is not None and abs(s_h[0] - s_a[0]) < 1e-9,
          f"s {None if s_h is None else round(s_h[0], 4)}")
    # 5. a rollout that never carries judges nothing
    rc = H.stress_rollout(model, data0, rig, ids, centre, bid, phases=("approach", "close"))
    check("a rollout without a carry phase has no servo_reserve phase", not any(p_["name"] == "servo_reserve" for p_ in rc["phases"]))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
