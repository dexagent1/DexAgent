"""generate_flat_pinches on three flat parts: a light box (lifts), a heavy drive-sized slab, and a part wider than the
hand's span along one axis (that axis is skipped); a rounded part (contacts on its real surface, not its bounding box);
and a block between two tray walls (pinches whose opening hand goes through a wall are rejected)."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.skills.generate_flat_pinches import generate_flat_pinches  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="flat_pinch_", dir=os.environ.get("TMPDIR")))
HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def box(name, hx, hy, hz, mass):
    p = TMP / f"{name}.xml"
    p.write_text(f"""<mujoco model="{name}"><worldbody><body name="{name}" pos="0 0 {hz}"><freejoint name="{name}_free"/>
<geom name="{name}_g" type="box" size="{hx} {hy} {hz}" mass="{mass}" friction="1.0 0.01 0.001"/></body></worldbody></mujoco>""")
    return {"hand_xml": HAND, "side": "right", "floor": True, "objects": [{"name": name, "xml": str(p), "pos": [0.1, 0.45, 0.0], "yaw_deg": 15}]}


def capsule_lying(name, r, half, mass):
    p = TMP / f"{name}.xml"
    p.write_text(f"""<mujoco model="{name}"><worldbody><body name="{name}" pos="0 0 {r}"><freejoint name="{name}_free"/>
<geom name="{name}_g" type="capsule" size="{r} {half}" euler="0 90 0" mass="{mass}" friction="1.0 0.01 0.001"/></body></worldbody></mujoco>""")
    return {"hand_xml": HAND, "side": "right", "floor": True, "objects": [{"name": name, "xml": str(p), "pos": [0.1, 0.45, 0.0], "yaw_deg": 0}]}


def tray_block(gap):
    hx, hy, hz, w, wall_h = 0.03, 0.045, 0.0125, 0.004, 0.05
    t = TMP / "tray.xml"
    t.write_text(f"""<mujoco model="tray"><worldbody><body name="tray" pos="0 0 0">
<geom name="tray_wl" type="box" size="{w} {hy + 0.03} {wall_h / 2}" pos="{-(hx + gap + w)} 0 {wall_h / 2}"/>
<geom name="tray_wr" type="box" size="{w} {hy + 0.03} {wall_h / 2}" pos="{hx + gap + w} 0 {wall_h / 2}"/>
</body></worldbody></mujoco>""")
    b = TMP / "blk.xml"
    b.write_text(f"""<mujoco model="blk"><worldbody><body name="blk" pos="0 0 {hz}"><freejoint name="blk_free"/>
<geom name="blk_g" type="box" size="{hx} {hy} {hz}" mass="0.1" friction="1.0 0.01 0.001"/></body></worldbody></mujoco>""")
    return {"hand_xml": HAND, "side": "right", "floor": True, "objects": [
        {"name": "tray", "xml": str(t), "pos": [0.1, 0.45, 0.0], "yaw_deg": 15},
        {"name": "blk", "xml": str(b), "pos": [0.1, 0.45, 0.0], "yaw_deg": 15}]}


def main():
    r = generate_flat_pinches(box("light", 0.035, 0.05, 0.0125, 0.12), verify_top=2, max_candidates=4, along_fracs=(0.0,))
    check("light flat box: feasible pinches, one lifts", r["n_feasible"] >= 1 and any((g.get("lift") or {}).get("ok") for g in r["grasps"]),
          str([(g["pinch"]["fingers"], (g.get("lift") or {}).get("achieved_mm")) for g in r["grasps"][:3]]))
    check("poses carry their pre-opening and clamp", all("open_frac" in g and "press_depth_rad" in g for g in r["grasps"]))
    r2 = generate_flat_pinches(box("slab", 0.045, 0.07, 0.012, 0.6), verify_top=0, max_candidates=10, along_fracs=(0.0,))
    check("heavy slab: pinches found with no finger in the table", r2["n_feasible"] >= 1 and r2["rejected"]["table"] < 4, str(r2["rejected"]))
    r3 = generate_flat_pinches(box("wide", 0.04, 0.09, 0.012, 0.2), verify_top=0, max_candidates=4, along_fracs=(0.0,))
    check("an axis wider than the hand's span is skipped", any("long" in s_ for s_ in r3["skipped_axes"]), str(r3["skipped_axes"]))
    # a rounded part: a capsule lying along x (r 22 mm) pinched across y below its axis; the bounding box face is up to
    # several mm away from the real surface there, the ray-cast contacts sit on it
    r4 = generate_flat_pinches(capsule_lying("roll", 0.022, 0.03, 0.08), verify_top=0, max_candidates=6, along_fracs=(0.0,), grip="down")
    worst = 0.0
    for g in r4["grasps"]:
        for p_ in g["points"].values():
            p_ = np.asarray(p_, float) - np.array([0.1, 0.45, 0.022])
            seg = np.array([float(np.clip(p_[0], -0.03, 0.03)), 0.0, 0.0])
            worst = max(worst, abs(float(np.linalg.norm(p_ - seg)) - 0.022))
    check("rounded part: contact points on the real surface", r4["n_feasible"] >= 1 and worst < 0.0015,
          f"{r4['n_feasible']} poses, worst off-surface {worst * 1000:.2f} mm")
    # a block in a tray with 30 mm between its sides and the walls: some pinches close there but open through a wall
    from tool_library.skills.generate_flat_pinches import _sweep_overlap
    from tool_library.common import hand as H
    sc = tray_block(0.03)
    r5 = generate_flat_pinches(sc, verify_top=0, max_candidates=8, along_fracs=(0.0,))
    model, data0, rig, ids = H.open_scene(H.load_scene_spec(sc))
    gp = sorted(ids["grasp_gids"])
    clean = all(_sweep_overlap(model, data0, rig, ids, g, g["open_frac"], gp) <= 0.0025 for g in r5["grasps"])
    # A new feasibility check may reject an unsafe candidate before the
    # approach stage. Verify the actual geometric behavior, not which counter
    # gets incremented first: clear-scene grasps collide in this tray, while
    # every returned tray grasp clears its complete opening/arrival sweep.
    blocked = max((_sweep_overlap(model, data0, rig, ids, g, g["open_frac"], gp) for g in r["grasps"]), default=0.)
    check("tray: pinches that open through a wall are rejected", blocked > 0.0025
          and r5["n_feasible"] < r5["progress"]["candidates_solved"] and clean,
          f"known blocked sweep {blocked * 1000:.2f} mm; {r5['rejected']}")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
