"""grasp_mass_friction_robustness_test: the grasp must hold a 1.3x heavier, 0.7x slipperier object.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_mass_friction_robustness_test.py
"""
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402

FIXED = """<mujoco model="fixedbox"><worldbody>
  <body name="fixedbox" pos="0 0 0.025">
    <geom name="fixedbox_body" type="box" size="0.025 0.02 0.025" mass="0.12" rgba="0.5 0.5 0.5 1"/>
  </body></worldbody></mujoco>"""


def main():
    from tool_library.verifiers.grasp_mass_friction_robustness_test import grasp_mass_friction_robustness_test as T
    rep = F.Report()
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="dexagent_massfric_")
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    weak = F.pose(F.BOX_PINCH)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)

    r = T(box, pinch)
    rep.check("three-finger pinch keeps a 0.156 kg box at friction x0.7", r["ok"] and r["nominal"]["ok"] and r["scaling"]["mass_kg"][1] > 0.155
              and r["scaling"]["pairs_scaled"] > 0, r.get("issues"))
    r = T(box, pinch, compare_nominal=False)
    rep.check("scaling also passes when the nominal run is omitted", r["ok"] and r["nominal"] is None, r.get("issues"))
    heavy = F.box_variant(tmp, name="hbox", mass=0.5)
    r = T(heavy, pinch, compare_nominal=False)
    rep.check("the self-clear pinch rejects a 0.65 kg overload", not r["ok"] and r["nominal"] is None
              and r["scaled"]["first_fail"] == "lift" and "not tested" in r["issues"][0], r.get("issues"))

    # A scaled run may succeed because its changed inertia/friction seats a
    # different contact. That cannot certify an explicitly failed baseline.
    with patch("tool_library.verifiers.grasp_mass_friction_robustness_test._ladder",
               side_effect=[{"ok": False, "first_fail": "lift"}, {"ok": True, "first_fail": None}]):
        mixed = T(box, pinch)
    rep.check("a scaled success cannot hide a failed nominal grasp", not mixed["ok"] and mixed["issues"], mixed)

    # the two-pad pinch holds the box at its estimate but not 1.3x heavier on slipperier pads
    r = T(box, weak)
    rep.check("two-pad pinch holds nominally but loses the heavier, slipperier box", (not r["ok"]) and r["nominal"]["ok"]
              and r["scaled"]["first_fail"] and "no margin" in r["issues"][0], (r["scaled"]["first_fail"], r["issues"]))
    # A rim grasp holds the nominal cylinder, but not a much heavier,
    # slipperier reconstruction. The physical load threshold changed when
    # previously disabled hand self contacts were restored.
    r = T(cyl, top)
    rep.check("left-hand rim grasp keeps a 0.325 kg cylinder at friction x0.7", r["ok"], r.get("issues"))
    r = T(cyl, top, mass_scale=3.12, friction_scale=.5)
    rep.check("rim grasp drops the 0.78 kg cylinder at half friction", (not r["ok"]) and r["nominal"]["ok"]
              and str(r["scaled"]["first_fail"]).startswith(("move", "hold", "lift")), r["scaled"]["first_fail"])
    # unity scaling is the nominal object
    r = T(cyl, top, mass_scale=1.0, friction_scale=1.0, compare_nominal=False)
    rep.check("unity scaling passes like the nominal run", r["ok"], r.get("issues"))

    fp = Path(tmp) / "fixedbox.xml"
    fp.write_text(FIXED)
    r = T(F.scene(objects=[{"name": "fixedbox", "xml": str(fp), "pos": [0, 0, 0]}]), pinch)
    rep.check("a part on a fixed asset is not applicable", r["ok"] and r["applicable"] is False, r)
    rep.raises("non-positive scale is refused", lambda: T(box, pinch, mass_scale=0.0))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
