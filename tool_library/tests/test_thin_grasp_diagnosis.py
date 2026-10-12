"""thin_grasp_diagnosis on parametric parts lying on the setup bench, and the finite-table floor the thin-part poses
are solved on. Flat boxes 5, 9 and 12 mm thick (a phone is 9 mm: too low for side contacts, which need a pad radius +
3 mm = 10 mm; 12 mm is pinchable as it lies), at several yaws and distances from the near table edge: the diagnosis
says slide + edge_pinch with the overhang the moment balance needs, the slide's mode (an upright fingertip pushes even
a 5 mm part: its contact meets the top edge 25 deg above horizontal; a 1.5 mm strip it cannot push), and an
already-overhanging box is edge-pinchable as it lies; a 25 mm block is pinchable; a 10 mm ring with a 24 mm opening is
hookable with the thumb outside, a 16 mm opening is not; a flat ribbon lying flat has no gap and nothing that makes it
graspable (pads closing along it slide over it on this table); a 30 mm chip cannot
overhang enough to be pinched and nothing makes it graspable. The hand scene's box floor: a hand's visible meshes below
the tabletop beyond the table's edge are not in the table, over it they are. Seconds.

    python tool_library/tests/test_thin_grasp_diagnosis.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.common import nonprehensile as NP  # noqa: E402
from tool_library.skills.construct_multi_strand_cable import construct_multi_strand_cable  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.skills.thin_grasp_diagnosis import thin_grasp_diagnosis  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="thin_diag_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def slab_xml(path, size, mass=0.17, friction=0.9):
    """A flat box lying on its broad face (origin on the face it rests on)."""
    hx, hy, hz = (0.5 * float(v) for v in size)
    Path(path).write_text(f'<mujoco model="slab"><worldbody><body name="b"><freejoint/><geom name="g" type="box" size="{hx} {hy} {hz}" '
                          f'pos="0 0 {hz}" mass="{mass}" friction="{friction} 0.01 0.001"/></body></worldbody></mujoco>')
    return str(path)


def ring_xml(path, r_in, wall=0.004, height=0.010, n=16, tab=0.06):
    """A flat ring (a scissors finger strap) of box segments, with a flat tab like a blade."""
    segs = []
    for k in range(n):
        th = 2 * np.pi * k / n; rc = r_in + 0.5 * wall; L = 2 * np.pi * rc / n * 0.55
        segs.append(f'<geom name="w{k}" type="box" size="{0.5 * wall} {L} {0.5 * height}" pos="{rc * np.cos(th):.5f} {rc * np.sin(th):.5f} {0.5 * height}" '
                    f'quat="{np.cos(th / 2):.6f} 0 0 {np.sin(th / 2):.6f}" mass="0.0008" friction="0.8 0.01 0.001"/>')
    segs.append(f'<geom name="tab" type="box" size="{0.5 * tab} 0.006 {0.3 * height}" pos="{r_in + wall + 0.5 * tab - 0.002:.4f} 0 {0.3 * height}" '
                'mass="0.012" friction="0.8 0.01 0.001"/>')
    Path(path).write_text(f'<mujoco model="ring"><worldbody><body name="b"><freejoint/>{"".join(segs)}</body></worldbody></mujoco>')
    return str(path)


def main():
    a = TMP / "assets"; a.mkdir()
    # near table edge: y = 0.05. Parts far apart so none is another's support
    parts = [("t5", slab_xml(a / "t5.xml", (0.072, 0.148, 0.005)), 0.30, 0.20, 0.0),
             ("t9", slab_xml(a / "t9.xml", (0.072, 0.148, 0.009)), 0.10, 0.17, 15.0),
             ("t12", slab_xml(a / "t12.xml", (0.065, 0.14, 0.012)), -0.10, 0.20, -10.0),
             ("ov9", slab_xml(a / "ov9.xml", (0.072, 0.148, 0.009)), -0.30, 0.074, 0.0),       # its near end overhangs the edge 50 mm
             ("blk", slab_xml(a / "blk.xml", (0.05, 0.07, 0.025), mass=0.1), 0.30, 0.45, 30.0),
             ("chip", slab_xml(a / "chip.xml", (0.02, 0.03, 0.004), mass=0.005), 0.10, 0.45, 0.0),
             ("ring", ring_xml(a / "ring.xml", 0.012), -0.10, 0.45, 0.0),
             ("narrow", ring_xml(a / "narrow.xml", 0.008), -0.30, 0.45, 0.0)]
    rib = construct_multi_strand_cable(str(a / "rib"), name="rib", strands=[{"width": 0.008, "thickness": 0.0015, "label": "a",
                                       "centreline": [[x, 0.62, 0.00075] for x in np.linspace(-0.1, 0.1, 9)]}], thickness=0.0015)
    assets = [{"name": n, "xml": x, "x": px, "y": py, "yaw_deg": yaw} for n, x, px, py, yaw in parts]
    assets.append({"name": "rib", "xml": rib["xml"], "x": 0.0, "y": 0.0})
    scene = str(TMP / "scene.xml")
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=assets, out_xml=scene, rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    D = {n: thin_grasp_diagnosis(scene, f"{n}_b") for n, *_ in parts}
    D["rib"] = thin_grasp_diagnosis(scene, "rib_" + rib["strands"]["a"]["first"])          # the strand (its links), not the knot
    v = {n: r["verdicts"] for n, r in D.items()}
    hand = D["t9"]["hand"]
    check("the hand's sizes are read from its capsules (7 mm pads, 14 mm fingertips)", abs(hand["pad_radius_m"] - 0.007) < 1e-4
          and abs(hand["tip_thickness_m"] - 0.014) < 1e-4 and hand["span_m"] > 0.08, json.dumps(hand))
    for n in ("t5", "t9"):
        g = D[n]["geometry"]
        check(f"a {1000 * g['height_m']:.0f} mm flat box is not pinchable where it lies and a slide + edge_pinch is recommended",
              not D[n]["graspable_as_lies"] and not v[n]["side_pinch"]["ok"] and D[n]["recommend"][0]["plan"] == ["slide", "edge_pinch"]
              and v[n]["slide"]["edge"] == "-y" and v[n]["slide"]["distance_m"] > 0.02,
              f"{v[n]['side_pinch']['why']}; {D[n]['recommend'][0]['why']}")
    e9 = v["t9"]["edge_pinch"]
    check("the edge pinch's overhang comes from the moment balance and stays under what keeps the box on the table",
          0.025 < e9["need_overhang_m"] < e9["max_overhang_m"] and e9["finger_inset_m"] > e9["thumb_inset_m"],
          json.dumps({k: e9[k] for k in ("need_overhang_m", "max_overhang_m", "finger_inset_m", "thumb_inset_m")}))
    check("an upright fingertip pushes a 9 mm and a 5 mm box (its contact meets the 5 mm box's top edge 25 deg up)",
          v["t9"]["slide"]["mode"] == "push" and v["t5"]["slide"]["mode"] == "push" and v["t9"]["slide"]["push_contact_deg"] == 0.0
          and 20 < v["t5"]["slide"]["push_contact_deg"] < 30, f"{v['t5']['slide']['why']}")
    check("a 12 mm box is pinchable from its sides as it lies", v["t12"]["side_pinch"]["ok"] and D["t12"]["graspable_as_lies"],
          v["t12"]["side_pinch"]["why"])
    check("a box already overhanging the edge 50 mm is edge-pinchable as it lies",
          v["ov9"]["edge_pinch"]["ok"] and abs(v["ov9"]["edge_pinch"]["overhang_m"] - 0.05) < 0.003 and D["ov9"]["recommend"][0]["plan"] == ["edge_pinch"],
          v["ov9"]["edge_pinch"]["why"])
    check("a 25 mm block is pinchable where it lies", v["blk"]["side_pinch"]["ok"] and D["blk"]["recommend"][0]["plan"] == ["grasp_lift"], v["blk"]["side_pinch"]["why"])
    check("a 30 mm chip cannot overhang far enough to be pinched and nothing makes it graspable",
          not v["chip"]["edge_pinch"]["ok"] and not v["chip"]["slide"]["ok"] and D["chip"]["recommend"][0]["plan"] == [],
          v["chip"]["edge_pinch"]["why"])
    h = v["ring"]["hook"]
    check("a 10 mm ring with a 24 mm opening is hookable, the thumb outside its wall", h["ok"] and h["thumb_outside"]
          and abs(h["opening"]["inscribed_radius_m"] - 0.012) < 0.0015 and ["hook"] in [r["plan"] for r in D["ring"]["recommend"]], h["why"])
    check("a 16 mm opening is narrower than a fingertip + 3 mm: no hook", not v["narrow"]["hook"]["ok"], v["narrow"]["hook"]["why"])
    check("a flat 1.5 mm ribbon lying flat: no side pinch, no gap, too low to push, and nothing recommended (it says why)",
          not v["rib"]["strip_pinch"]["ok"] and not v["rib"]["side_pinch"]["ok"] and v["rib"]["pinch_under"]["gap_m"] < 0.002
          and v["rib"]["slide"]["mode"] != "push" and v["rib"]["slide"]["push_contact_deg"] > 60 and D["rib"]["recommend"][0]["plan"] == [],
          f"{v['rib']['strip_pinch']['why']}; {v['rib']['slide']['why']}")
    # the finite table: visible meshes below the tabletop count only over the table
    hs = {"hand_xml": str(ROOT / "setup/robot/sharpa_right.xml"), "side": "right", "objects": [{"name": "t", "xml": a / "t5.xml" and str(a / "t5.xml"), "pos": [0.0, 0.3, 0.0]}],
          "floor": NP.table_floor(scene)}
    model, d0, rig, ids = H.open_scene(hs)
    d = mujoco.MjData(model); d.qpos[:] = d0.qpos
    rig.teleport(d, [0.0, -0.15, -0.01], [0.0, np.pi / 2, 0.0]); mujoco.mj_forward(model, d)      # beyond the near edge, below the top
    beyond = H.visual_floor_depth(model, d, rig, ids["floor_gid"])
    rig.teleport(d, [0.0, 0.40, -0.01], [0.0, np.pi / 2, 0.0]); mujoco.mj_forward(model, d)       # the same over the table
    over = H.visual_floor_depth(model, d, rig, ids["floor_gid"])
    check("a box floor: the hand below the tabletop beyond the edge is clear, over the table it is in the table", beyond == 0.0 and over > 0.01,
          f"beyond {1000 * beyond:.1f} mm, over {1000 * over:.1f} mm")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
