"""grasp_closing_rotation_test: the object must keep its pose while the executor's close runs.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_closing_rotation_test.py
"""
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402

TALL = """<mujoco model="tall"><worldbody>
  <body name="tall" pos="0 0 0.05"><freejoint/>
    <geom name="tall_body" type="box" size="0.025 0.02 0.05" mass="0.08" rgba="0.3 0.6 0.3 1"/>
  </body></worldbody></mujoco>"""


def main():
    from tool_library.common import grasp_robust as GR
    from tool_library.verifiers.grasp_closing_rotation_test import grasp_closing_rotation_test as T
    rep = F.Report()
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="dexagent_close_")
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    weak = F.pose(F.BOX_PINCH)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)

    # a three-finger pinch through the centre and a five-finger top grasp keep their objects close to where they were
    r = T(box, pinch)
    rep.check("three-finger box pinch closes within the limits", r["ok"] and r["max_rot_deg"] < 18 and r["max_shift_mm"] < 25
              and set(r["seated"]["fingers"]) >= {"thumb", "index"}, r)
    r = T(cyl, top)
    rep.check("left-hand top grasp closes without moving the cylinder", r["ok"] and r["max_shift_mm"] < 5, r)

    # the inputs as files: hand-scene JSON + pose JSON, and an objects MJCF with the side
    sp = Path(tmp) / "hs.json"
    sp.write_text(json.dumps(box))
    r2 = T(str(sp), str(F.FX / f"{F.BOX_PINCH3}.json"))
    r3 = T(str(F.FX / "box.xml"), pinch, side="right")
    rep.check("JSON and MJCF inputs give the same verdict", r2["ok"] and r3["ok"] and abs(r2["max_rot_deg"] - r3["max_rot_deg"]) < 0.5,
              (r2["max_rot_deg"], r3["max_rot_deg"]))

    # A light box with the wrist 35 mm toward one side: the first pad turns
    # it before the opposing pads meet. Contact timing must explain rejection.
    light = F.box_variant(tmp, name="lbox", mass=0.03)
    r = T(light, GR.perturb_pose(pinch, [0.0, -0.035, 0.0], [0.0, 0.0, 0.0]))
    rep.check("off-centre pinch that turns a light box is rejected", not r["ok"] and r["max_rot_deg"] > 30 and r["issues"]
              and "first touch" in r["issues"][0], (r["max_shift_mm"], r["first_touch_s"]))

    # a tall box pinched near its top by the weak two-pad pinch, wrist 15 mm off: one pad tips and pushes the box alone
    tp = Path(tmp) / "tall.xml"
    tp.write_text(TALL)
    tall = F.scene(objects=[{"name": "tall", "xml": str(tp), "pos": [0, 0, 0]}])
    r = T(tall, GR.perturb_pose(weak, [0.0, -0.015, 0.06], [0.0, 0.0, 0.0]))
    rep.check("one pad pushing a tall box alone is rejected", not r["ok"] and r["max_rot_deg"] > 20, (r["max_rot_deg"], r["max_shift_mm"]))

    # a box standing 30 deg to the pinch line: the closing pads turn it into line (the bench's turned-brick failure)
    turned = copy.deepcopy(light)
    turned["objects"][0]["yaw_deg"] = 30
    r = T(turned, pinch)
    rep.check("box turned 30 deg to the pinch is turned by the close", not r["ok"] and r["max_rot_deg"] > 20 and "turned" in r["issues"][0],
              r["max_rot_deg"])
    r_loose = T(turned, pinch, max_rot_deg=45.0, max_shift_mm=45.0)
    rep.check("the same close passes with looser limits", r_loose["ok"], (r_loose["max_rot_deg"], r_loose["max_shift_mm"]))

    rep.raises("pose without finger targets is refused", lambda: T(box, {"pos": [0, 0, 0.2], "euler": [0, 0, 0]}))
    rep.raises("non-positive limit is refused", lambda: T(box, pinch, max_rot_deg=0))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
