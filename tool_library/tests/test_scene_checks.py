"""Smoke tests for the scene compile checks and scene assertions.

    MUJOCO_GL=egl python tool_library/tests/test_scene_checks.py

Every check is exercised on a synthetic scene in a case that must pass and one
that must fail. No external assets are needed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.common import scene as S  # noqa: E402
from tool_library.skills.compile_scene import compile_scene  # noqa: E402
from tool_library.skills.probe_scene import probe_scene  # noqa: E402
from tool_library.verifiers.above_check import check_above  # noqa: E402
from tool_library.verifiers.articulation_check import check_articulation  # noqa: E402
from tool_library.verifiers.color_match_check import check_color_match  # noqa: E402
from tool_library.verifiers.contact_check import check_contact  # noqa: E402
from tool_library.verifiers.height_check import check_height  # noqa: E402
from tool_library.verifiers.joint_range_check import check_joint_range  # noqa: E402
from tool_library.verifiers.model_finite_check import check_model_finite  # noqa: E402
from tool_library.verifiers.moving_joint_check import check_moving_joint  # noqa: E402
from tool_library.verifiers.no_contact_check import check_no_contact  # noqa: E402
from tool_library.verifiers.no_float_check import check_no_float  # noqa: E402
from tool_library.verifiers.required_bodies_check import check_required_bodies  # noqa: E402
from tool_library.verifiers.scene_qc import run_scene_qc  # noqa: E402
from tool_library.verifiers.settle_stability_check import check_settle_stability  # noqa: E402
from tool_library.verifiers.silhouette_iou_check import check_silhouette_iou  # noqa: E402
from tool_library.verifiers.support_check import check_support  # noqa: E402
from tool_library.verifiers.units_check import check_units  # noqa: E402
from tool_library.verifiers.urdf_articulation_check import check_urdf_articulation  # noqa: E402

HEAD = """<mujoco model="t"><compiler angle="radian"/><option timestep="0.002" gravity="0 0 -9.81"/>
<visual><global offwidth="320" offheight="240"/></visual>
<worldbody>
  <light pos="0 0 3" dir="0 0 -1"/>
  <camera name="cam" pos="0 -1.6 1.3" xyaxes="1 0 0 0 0.6 0.8" fovy="45"/>
  <body name="table"><geom name="top" type="box" size="0.8 0.8 0.02" pos="0 0 -0.02" rgba=".7 .7 .7 1"/></body>
"""
TAIL = "</worldbody></mujoco>"


def scene(body_xml):
    return HEAD + body_xml + TAIL


CUP = '<body name="red_cup" pos="0 0 0.05"><freejoint/><geom type="box" size=".04 .04 .05" rgba=".8 .1 .1 1" mass="0.2"/></body>'
BOWL = '<body name="bowl" pos="0.3 0 0.03"><freejoint/><geom type="box" size=".08 .08 .03" rgba=".1 .1 .8 1" mass="0.3"/></body>'
CABINET = ('<body name="cabinet" pos="-0.4 0 0.15"><geom name="carcass" type="box" size=".1 .12 .15" rgba=".6 .5 .4 1"/>'
           '<body name="cabinet_door" pos="0.1 -0.12 0"><joint name="door_hinge" type="hinge" axis="0 0 1" range="0 1.2"/>'
           '<geom type="box" size=".005 .12 .14" pos="0 0.12 0" rgba=".3 .3 .3 1"/></body></body>')
GOOD = scene(CUP + BOWL + CABINET)


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    assert cond, name


def main():
    tmp = Path(tempfile.mkdtemp(prefix="scene_checks_"))
    good = tmp / "good.xml"
    good.write_text(GOOD)

    print("model_finite_check")
    check("good scene is finite", check_model_finite(good)["ok"])
    m = S.load_model(good)
    m.body_mass[1] = -1.0
    check("negative mass is caught", not check_model_finite(m)["ok"])

    print("units_check")
    check("metre scene passes", check_units(good)["ok"])
    far = scene(CUP.replace('pos="0 0 0.05"', 'pos="640 480 0.05"'))
    check("pixel-scale position is caught", not check_units(far)["ok"])

    print("settle_stability_check")
    r = check_settle_stability(good)
    check("resting objects stay put", r["ok"] and r["drift"]["red_cup"]["verdict"] == "at rest")
    tipped = scene(CUP.replace('pos="0 0 0.05"', 'pos="0 0 0.09" quat="0.94 0.34 0 0"') + BOWL)
    r = check_settle_stability(tipped)
    check("a tilted box tips over", not r["ok"] and "red_cup" in r["failures"])
    hover = scene(CUP.replace('pos="0 0 0.05"', 'pos="0 0 0.30"'))
    check("a hovering box falls and fails", not check_settle_stability(hover)["ok"])

    print("support_check")
    check("everything supported", check_support(good)["ok"])
    void = scene(CUP.replace('pos="0 0 0.05"', 'pos="3 3 0.05"'))      # beyond the table edge
    r = check_support(void)
    check("box with nothing under it floats", not r["ok"] and r["floating"] == ["red_cup"])
    edge = scene(CABINET.replace('pos="-0.4 0 0.15"', 'pos="-0.85 0 0.15"'))
    r = check_support(edge)
    check("fixed cabinet over the edge", not r["ok"] and r["static"]["cabinet"]["verdict"] == "over the edge")
    hov = scene(CABINET.replace('pos="-0.4 0 0.15"', 'pos="-0.4 0 0.20"'))
    r = check_support(hov)
    check("fixed cabinet hovering 5 cm", r["static"]["cabinet"]["verdict"] == "hovering")

    print("required_bodies_check")
    check("present objects found", check_required_bodies(good, ["red cup", "bowl", "cabinet door"])["ok"])
    r = check_required_bodies(good, ["spoon"])
    check("missing object reported", not r["ok"] and r["missing"] == ["spoon"])

    print("articulation_check")
    r = check_articulation(good, ["cabinet"])
    check("hinged door is articulated", r["ok"] and r["objects"]["cabinet"]["verdict"] == "articulated")
    r = check_articulation(good, ["red_cup"])
    check("free box is not", not r["ok"] and r["objects"]["red_cup"]["verdict"] == "rigid free body")
    degenerate = scene(CABINET.replace('range="0 1.2"', 'range="0 0.01"'))
    check("degenerate range caught", check_articulation(degenerate, ["cabinet"])["objects"]["cabinet"]["verdict"] == "degenerate range")

    print("relation checks")
    stack = scene(CUP + CUP.replace("red_cup", "lid").replace('pos="0 0 0.05"', 'pos="0 0 0.16"').replace(".8 .1 .1", ".2 .8 .2"))
    check("stacked boxes touch", check_contact(stack, "lid", "red cup")["ok"])
    check("cup and bowl do not touch", check_no_contact(good, "red_cup", "bowl")["ok"])
    check("contact fails when apart", not check_contact(good, "red_cup", "bowl")["ok"])
    check("static pair reported", check_contact(good, "table", "cabinet")["both_static"])
    check("lid above cup", check_above(stack, "lid", "red_cup", min_gap=0.05)["ok"])
    check("above fails the other way", not check_above(stack, "red_cup", "lid")["ok"])
    check("cup height in band", check_height(good, "red_cup", 0.04, 0.06)["ok"])
    check("cup height relative to table", check_height(good, "red_cup", 0.04, 0.06, relative_to="table")["ok"])
    check("height outside band fails", not check_height(good, "red_cup", 0.2, 0.3)["ok"])
    check("joint range as built", check_joint_range(good, "door_hinge", 0.0, 1.2)["ok"])
    check("wrong joint range fails", not check_joint_range(good, "door_hinge", 0.0, 1.0)["ok"])
    try:
        check_joint_range(good, "nope", 0, 1)
        check("unknown joint raises", False)
    except ValueError as exc:
        check("unknown joint raises with the joint list", "door_hinge" in str(exc))
    check("cup does not float", check_no_float(good, "red_cup")["ok"])
    check("welded cabinet trivially passes", check_no_float(good, "cabinet")["ok"] and not check_no_float(good, "cabinet")["free"])
    check("box over nothing floats", not check_no_float(void, "red_cup")["ok"])

    print("silhouette_iou_check")
    model = S.load_model(good)
    data = S.forward(model)
    masks = S.render_body_masks(model, data, 320, 240, "cam")
    r = check_silhouette_iou(good, {"red cup": masks["red_cup"], "bowl": masks["bowl"]}, "cam", out_dir=tmp / "sil")
    check("own render has IoU 1", r["ok"] and r["iou"]["red cup"] == 1.0)
    shifted = {k: np.roll(masks[k], 60, axis=1) for k in ("red_cup", "bowl")}
    r = check_silhouette_iou(good, shifted, "cam")
    check("shifted masks fail and blame the camera", not r["ok"] and r["camera_suspect"])
    r = check_silhouette_iou(good, {"red_cup": shifted["red_cup"], "bowl": masks["bowl"]}, "cam")
    check("one bad object does not blame the camera", not r["ok"] and not r["camera_suspect"])
    check("constructed object matched by prefix", check_silhouette_iou(good, {"cabinet": masks["cabinet"] | masks["cabinet_door"]}, "cam")["ok"])
    # a thin cord: its segmentation mask is three times wider than the cord; scored by centreline it still passes, and a
    # cord laid 4 cm off the mask's centreline still fails
    import cv2
    cord = scene('<body name="cord" pos="0 0.1 0.004"><geom type="capsule" size="0.003 0.25" quat="0.7071068 0 0.7071068 0" rgba=".9 .9 .9 1"/></body>')
    cm = S.render_body_masks(S.load_model(cord), S.forward(S.load_model(cord)), 320, 240, "cam")["cord"]
    wide = cv2.dilate(cm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    r = check_silhouette_iou(cord, {"cord": wide}, "cam", min_iou=0.4)
    print(f"    cord score {r['iou']['cord']}, plain {r['plain_iou']['cord']}")
    check("a thin cord against a 3x wider mask passes by centreline", r["ok"] and r["plain_iou"]["cord"] < 0.5 and "cord" in r["thin"])
    off = scene('<body name="cord" pos="0 0.14 0.004"><geom type="capsule" size="0.003 0.25" quat="0.7071068 0 0.7071068 0" rgba=".9 .9 .9 1"/></body>')
    r = check_silhouette_iou(off, {"cord": wide}, "cam", min_iou=0.4)
    print(f"    offset cord score {r['iou']['cord']}")
    check("the same cord laid 4 cm off fails", not r["ok"])
    r = check_silhouette_iou(cord, {"cord": wide}, "cam", min_iou=0.4, thin=[])
    check("thin=[] keeps the plain IoU", r["iou"]["cord"] == r["plain_iou"]["cord"] and not r["thin"])

    print("color_match_check")
    check("measured red matches", check_color_match(good, "red_cup", [{"rgb": [0.8, 0.1, 0.1], "weight": 1.0}])["ok"])
    check("blue palette rejected", not check_color_match(good, "red_cup", [[0.1, 0.1, 0.8]])["ok"])

    print("urdf_articulation_check")
    (tmp / "part.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    urdf = tmp / "model.urdf"
    urdf.write_text("""<robot name="cab"><link name="base"><visual><geometry><mesh filename="part.obj"/></geometry></visual></link>
<link name="drawer"><visual><geometry><mesh filename="part.obj"/></geometry></visual></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="drawer"/><limit lower="0" upper="0.3"/></joint></robot>""")
    check("constructed URDF passes", check_urdf_articulation(urdf)["ok"])
    hand = tmp / "hand.urdf"
    hand.write_text("""<robot name="h"><link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
<link name="b"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
<joint name="j" type="sliding"><parent link="a"/><child link="b"/></joint></robot>""")
    r = check_urdf_articulation(hand)
    check("hand-typed boxes with an invalid joint fail", not r["ok"] and len(r["reasons"]) >= 2)

    print("moving_joint_check")
    drawers = scene(
        '<body name="cab" pos="-0.2 0.2 0.15"><geom type="box" size=".15 .12 .15" rgba=".6 .5 .4 1"/>'
        '<body name="cab_drawer_top" pos="0 -0.12 0.07"><joint name="cab_slide_top" type="slide" axis="0 -1 0" range="0 0.2"/>'
        '<geom type="box" size=".13 .01 .05" rgba=".2 .2 .2 1"/></body>'
        '<body name="cab_drawer_bottom" pos="0 -0.12 -0.07"><joint name="cab_slide_bottom" type="slide" axis="0 -1 0" range="0 0.2"/>'
        '<geom type="box" size=".13 .01 .05" rgba=".2 .2 .2 1"/></body></body>')
    md = S.load_model(drawers)
    d = S.forward(md)
    d.qpos[int(md.jnt_qposadr[S.joint_id(md, "cab_slide_bottom")])] = 0.2
    S.mujoco.mj_forward(md, d)
    fake_video = S.render_rgb(md, d, 320, 240, "cam")       # the bottom drawer is the one that moved
    r = check_moving_joint(drawers, fake_video, "cam", joints="cab_slide", expected="cab_slide_bottom", out_dir=tmp / "mj")
    check("bottom drawer identified", r["ok"] and r["best"] == "cab_slide_bottom")
    check("wrong expectation fails", not check_moving_joint(drawers, fake_video, "cam", expected="cab_slide_top")["ok"])

    print("scene_qc / compile_scene / probe_scene")
    r = run_scene_qc(good, required=["red cup", "bowl"], articulated=["cabinet"],
                     masks={"red cup": masks["red_cup"]}, camera="cam")
    check("good scene passes the battery", r["ok"] and not r["failed"])
    r = run_scene_qc(tipped, required=["spoon"])
    check("bad scene names its failures", not r["ok"] and {"settle_stability", "required_bodies"} <= set(r["failed"]))
    c = compile_scene(str(good), qc=False, save_mjb=str(tmp / "good.mjb"))
    check("compile reports contents", c["compiled"] and c["counts"]["joints"] == 3 and (tmp / "good.mjb").exists())
    check("compile error is returned, not raised", not compile_scene("<mujoco><worldbody><geom type='nope'/></worldbody></mujoco>")["compiled"])
    p = probe_scene(good)
    check("probe lists bodies and joints", p["bodies"]["red_cup"]["kind"] == "free" and p["joints"][-1]["type"] == "hinge")
    print("all scene checks passed")


if __name__ == "__main__":
    main()
