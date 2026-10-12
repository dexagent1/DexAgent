"""grasp_table_support_dependence_test: the hand alone must keep the object once its support is taken away.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_table_support_dependence_test.py
"""
import copy
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402

FIXED = """<mujoco model="fixedbox"><worldbody>
  <body name="fixedbox" pos="0 0 0.025">
    <geom name="fixedbox_body" type="box" size="0.025 0.02 0.025" mass="0.12" rgba="0.5 0.5 0.5 1"/>
  </body></worldbody></mujoco>"""
BASE = """<mujoco model="base"><worldbody>
  <body name="base" pos="0 0 0.02"><freejoint/>
    <geom name="base_body" type="box" size="0.04 0.04 0.02" mass="0.3" rgba="0.4 0.4 0.7 1"/>
  </body></worldbody></mujoco>"""


def main():
    from tool_library.common import grasp_robust as GR
    from tool_library.common import hand as H
    from tool_library.verifiers.grasp_table_support_dependence_test import grasp_table_support_dependence_test as T, _support_by_body
    rep = F.Report()
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="dexagent_support_"))
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    weak = F.pose(F.BOX_PINCH)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)
    rig = GR.Case(GR.load_scene(box)).rig

    r = T(box, pinch)
    # Closing a valid pinch can already lift the box a fraction of a mm;
    # its weight need not still be on the table when support is removed.
    rep.check("three-finger pinch holds the box without the table", r["ok"] and r["applicable"] and r["support_before"]["weight_N"] > 1.
              and r["drift_mm"] < 5, r)
    r = T(cyl, top)
    rep.check("left-hand top grasp holds the cylinder without the table", r["ok"], r.get("issues"))
    # the box standing on a second box (a stack): the pinch holds it when the lower box and the table go
    (tmp / "base.xml").write_text(BASE)
    stack = F.scene(objects=[{"name": "base", "xml": str(tmp / "base.xml"), "pos": [0, 0, 0]},
                             {"name": "box", "xml": str(F.FX / "box.xml"), "pos": [0, 0, 0.04]}])
    stacked = GR.perturb_pose(pinch, [0.0, 0.0, 0.04], [0.0, 0.0, 0.0])
    baseline = GR.Case(GR.load_scene(stack))
    bd = baseline.fresh()
    parked, rotation, opened = [0., 0., .5], H.euler_to_mat(stacked["euler"]), baseline.rig.finger_targets(0.)
    baseline.place(bd, parked, rotation, opened)
    GR.hold(baseline, bd, parked, rotation, opened, .3)
    initial_support = _support_by_body(baseline, bd)
    r = T(stack, stacked)
    rep.check("pinch on a stacked box holds it without the box below", r["ok"] and initial_support.get("base_base", 0.) > 1.,
              {"before_grasp": initial_support, "after_close": r["support_before"]})

    # the thumb straightened away: the index only leans on the box, which the table and the index keep upright
    q = copy.deepcopy(pinch)
    q.pop("finger_targets_squeezed", None)
    for k in list(q["finger_targets"]):
        if k.startswith("thumb") and H.joint_role(k) == "flex":
            q["finger_targets"][k] = rig.finger_range[k][0]
    q["grip_fingers"] = ["index"]
    r = T(box, q)
    rep.check("an index leaning on the box loses it with the table gone", (not r["ok"]) and r["drift_mm"] > 20 and "rests on" in r["issues"][0],
              (r["drift_mm"], r["issues"]))
    r = T(stack, GR.perturb_pose(q, [0.0, 0.0, 0.04], [0.0, 0.0, 0.0]))
    rep.check("the same lean on a stacked box loses it with the box below gone", not r["ok"], r["drift_mm"])
    # the two-pad pinch: the table carried the box, the pads alone let it twist out and fall
    r = T(box, weak)
    rep.check("the weak two-pad pinch drops the box once the table is gone", (not r["ok"]) and r["drift_mm"] > 20, r["drift_mm"])

    (tmp / "fixedbox.xml").write_text(FIXED)
    r = T(F.scene(objects=[{"name": "fixedbox", "xml": str(tmp / "fixedbox.xml"), "pos": [0, 0, 0]}]), pinch)
    rep.check("a part on a fixed asset is not applicable", r["ok"] and r["applicable"] is False, r)
    rep.raises("non-positive hold is refused", lambda: T(box, pinch, hold_s=0))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
