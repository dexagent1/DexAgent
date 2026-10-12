"""Smoke tests for the Grasp & Poses skills and the Grasp Stress Tests
verifiers, on a dexterous hand holding a cylinder and a box.

    MUJOCO_GL=egl python tool_library/tests/test_grasp.py

The hand MJCF comes from DEXAGENT_HAND_XML, by default the setup's own
right hand (setup/robot/sharpa_right.xml); the test skips cleanly when it
does not exist. Objects are written here as
plain MJCF primitives, so nothing else is needed.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

ASSETS = os.environ.get("DEXAGENT_ASSETS_ROOT", str(ROOT.parent))     # where optional checkouts (BODex) live
HAND = os.environ.get("DEXAGENT_HAND_XML", str(ROOT / "setup" / "robot" / "sharpa_right.xml"))

CYL = """<mujoco model="cylinder"><worldbody>
  <body name="cyl" pos="0 0 0.06"><freejoint/>
    <geom name="cyl_body" type="cylinder" size="0.03 0.06" mass="0.25" rgba="0.2 0.5 0.8 1"/>
  </body></worldbody></mujoco>"""
BOX = """<mujoco model="box"><worldbody>
  <body name="box" pos="0 0 0.025"><freejoint/>
    <geom name="box_body" type="box" size="0.025 0.02 0.025" mass="0.12" rgba="0.8 0.45 0.2 1"/>
  </body></worldbody></mujoco>"""
ROD = """<mujoco model="rod"><worldbody>
  <body name="rod" pos="0 0 0.03"><freejoint/>
    <geom name="rod_body" type="capsule" fromto="-0.06 0 0 0.06 0 0" size="0.006" mass="0.03" rgba="0.3 0.3 0.3 1"/>
  </body></worldbody></mujoco>"""
ROPE = """<mujoco model="rope"><worldbody>
  <body name="seg0" pos="-0.06 0 0.006">
    <geom name="rope_0" type="capsule" fromto="0 0 0 0.04 0 0" size="0.004" mass="0.01" rgba="0.9 0.8 0.2 1"/>
    <body name="seg1" pos="0.04 0 0">
      <geom name="rope_1" type="capsule" fromto="0 0 0 0.04 0 0" size="0.004" mass="0.01" rgba="0.9 0.8 0.2 1"/>
      <body name="seg2" pos="0.04 0 0">
        <geom name="rope_2" type="capsule" fromto="0 0 0 0.04 0 0" size="0.004" mass="0.01" rgba="0.9 0.8 0.2 1"/>
      </body></body></body></worldbody></mujoco>"""


def scene_for(xml_text, name, tmp):
    p = Path(tmp) / f"{name}.xml"
    p.write_text(xml_text)
    return {"hand_xml": HAND, "side": "right", "objects": [{"name": name, "xml": str(p), "pos": [0, 0, 0]}]}


def main():
    if not Path(HAND).exists():
        print(f"SKIP: hand MJCF not found at {HAND} (set DEXAGENT_HAND_XML)")
        return 0
    from tool_library.common import hand as H
    from tool_library.skills.pose_from_points import pose_from_points
    from tool_library.skills.push_pose_from_point import push_pose_from_point
    from tool_library.skills.optimize_fingertip_contacts import optimize_fingertip_contacts
    from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps
    from tool_library.skills.handle_pinch_pose import handle_pinch_pose
    from tool_library.skills.rope_pinch_pose import rope_pinch_pose
    from tool_library.skills.rank_grasps_q1 import rank_grasps_q1
    from tool_library.verifiers.grasp_approach_open_test import grasp_approach_open_test
    from tool_library.verifiers.grasp_lift_test import grasp_lift_test
    from tool_library.verifiers.grasp_shake_test import grasp_shake_test
    from tool_library.verifiers.grasp_hold_test import grasp_hold_test
    from tool_library.verifiers.grasp_stress_test import grasp_stress_test
    from tool_library.verifiers.grasp_straddle_test import grasp_straddle_test
    from tool_library.verifiers.grasp_through_hole_test import grasp_through_hole_test
    from tool_library.verifiers.pregrasp_clearance_test import pregrasp_clearance_test
    from tool_library.verifiers.wrist_servo_offset_check import wrist_servo_offset_check
    from tool_library.verifiers.force_closure_check import force_closure_check

    tmp = tempfile.mkdtemp(prefix="dexagent_grasp_", dir=os.environ.get("TMPDIR"))
    cyl, box, rod, rope = (scene_for(x, n, tmp) for x, n in
                           ((CYL, "cyl"), (BOX, "box"), (ROD, "rod"), (ROPE, "rope")))
    t0 = time.time()

    # -- the rig itself -------------------------------------------------------
    model, data0, rig, ids = H.open_scene(cyl)
    assert len(rig.fingertip_sites) >= 4 and len(rig.finger_act) >= 10, "hand rig discovery"
    assert ids["n_pairs"] > 0 and rig.palm_body >= 0
    assert abs(float(data0.xpos[H.grasp_body(model, cyl, ids)][2]) - 0.06) < 0.002, "object settled in place"
    assert H.clearance(model, data0, rig, ids)["object_mm"] == 0.0, "parked hand is clear"
    print(f"[rig] {len(rig.finger_act)} finger joints, {ids['n_pairs']} contact pairs  ({time.time()-t0:.0f}s)")

    # -- pose_from_points: three pads on the cylinder --------------------------
    r = 0.03
    pts = {"thumb": [-r, 0.0, 0.07], "index": [r * np.cos(0.4), r * np.sin(0.4), 0.08], "middle": [r, 0.0, 0.06]}
    nrm = {"thumb": [-1, 0, 0], "index": [np.cos(0.4), np.sin(0.4), 0], "middle": [1, 0, 0]}
    pose = pose_from_points(cyl, pts, nrm, yaw_step_deg=60, pitches_deg=(-60, 0, 60), iters=200)
    assert pose["worst_residual_mm"] < 3.0, pose["residual_mm"]
    assert pose["overlap_mm"] < 6.0
    print(f"[pose_from_points] residual {pose['worst_residual_mm']} mm, overlap {pose['overlap_mm']} mm, "
          f"palm {pose['palm_facing']}  ({time.time()-t0:.0f}s)")

    # -- optimize_fingertip_contacts: pads end touching ------------------------
    pose2 = optimize_fingertip_contacts(cyl, pose)
    gaps = {f: v["gap_mm"] for f, v in pose2["cinch"].items()}
    assert all(-2.0 <= g <= 0.5 for g in gaps.values()), gaps
    assert "finger_targets_squeezed" in pose2
    print(f"[optimize_fingertip_contacts] gaps {gaps}  ({time.time()-t0:.0f}s)")

    # -- push_pose_from_point: index points into the box's face ----------------
    push = push_pose_from_point(box, [-0.025, 0.0, 0.03], [-1.0, 0.0, 0.0], finger="index")
    assert push["worst_residual_mm"] < 1.0 and push["finger_alignment"] > 0.99, push
    assert push["overlap_mm"] < 4.0
    print(f"[push_pose_from_point] residual {push['worst_residual_mm']} mm, alignment {push['finger_alignment']}, "
          f"overlap {push['overlap_mm']} mm  ({time.time()-t0:.0f}s)")

    # -- generate_rigid_grasps + the driven verifiers on the box ---------------
    gen = generate_rigid_grasps(box, yaw_step_deg=90, verify=True, verify_top=20)
    assert gen["n_feasible"] >= 1, gen
    passing = [g for g in gen["grasps"] if g.get("lift", {}).get("ok")]
    assert passing, "no generated grasp lifted the box"
    best = passing[0]
    print(f"[generate_rigid_grasps] {gen['n_candidates']} candidates, {gen['n_feasible']} feasible, "
          f"{len(passing)} lift; best family {best['family']} {best['grip_fingers']}  ({time.time()-t0:.0f}s)")

    a = grasp_approach_open_test(box, best)
    assert a["ok"], a
    pc = pregrasp_clearance_test(box, best)
    assert pc["ok"] and pc["retract_m"] is not None, pc
    lift = grasp_lift_test(box, best)
    assert lift["ok"], lift["phases"]
    hold = grasp_hold_test(box, best)
    assert hold["ok"], hold
    shake = grasp_shake_test(box, best)
    assert shake["ok"], shake["phases"]
    full = grasp_stress_test(box, best)
    assert full["ok"], (full["first_fail"], full["phases"])
    fc = force_closure_check(box, best)
    assert fc["ok"] and fc["q1"] > 0, fc
    so = wrist_servo_offset_check(box, best)
    assert so["residual_mm"] < 5.0, so
    print(f"[verifiers/pass] approach nudged {a['object_nudged_mm']} mm; pregrasp retract {pc['retract_m']} m; "
          f"lift {lift['lifts'][-1]['achieved_mm']} mm; hold slip {hold['slip_mm_s']} mm/s; "
          f"shake drift {shake['max_drift_mm']} mm; Q1 {fc['q1']}; servo offset {so['offset_m']}  ({time.time()-t0:.0f}s)")

    # -- the same verifiers must FAIL a bad grasp: the hand 4 cm above the box -
    bad = dict(best, pos=[best["pos"][0], best["pos"][1], best["pos"][2] + 0.04])
    assert not grasp_lift_test(box, bad)["ok"]
    assert not grasp_hold_test(box, bad)["ok"]
    assert not grasp_stress_test(box, bad)["ok"]
    assert not force_closure_check(box, bad)["ok"]
    # a hand buried in the floor has no clear pre-grasp, and coming down with
    # the fingers already closed shoves the box (here held to a 0.5 mm budget: the finger
    # servos are torque-capped and the floor has the bench table's stiff contact, so the
    # shove is a fraction of a millimetre to a few millimetres, not centimetres)
    buried = dict(best, pos=[best["pos"][0], best["pos"][1], best["pos"][2] - 0.03])
    assert not pregrasp_clearance_test(box, buried, tries=0)["ok"]
    closed_entry = dict(best, finger_targets=best["finger_targets_squeezed"])
    a_bad = grasp_approach_open_test(box, closed_entry, open_frac=0.0, nudge_max_mm=0.5)
    assert not a_bad["ok"] and a_bad["object_nudged_mm"] > 0.5, a_bad
    print(f"[verifiers/fail] lifted-off pose, buried pose and closed-finger entry rejected "
          f"(closed entry nudged {a_bad['object_nudged_mm']} mm)  ({time.time()-t0:.0f}s)")

    # -- rank_grasps_q1 orders the real grasp above the bad one -----------------
    rk = rank_grasps_q1(box, [bad, best])
    assert rk["best"] == 1 and rk["ranked"][0]["index"] == 1, rk
    print(f"[rank_grasps_q1] {[(x['index'], x['q1']) for x in rk['ranked']]}  ({time.time()-t0:.0f}s)")

    # -- handle_pinch_pose + grasp_straddle_test on a rod -----------------------
    hp = handle_pinch_pose(rod, "rod_rod_body", s=0.5, fingers=("thumb", "index"),
                           yaw_step_deg=60, pitches_deg=(-45, 0, 45), iters=240)
    assert hp["worst_residual_mm"] < 4.0, hp["residual_mm"]
    st = grasp_straddle_test(rod, hp)
    assert st["ok"] and st["opposite_sides"] and st["along_axis_cos"] < 0.5, st
    # pads lifted 2 cm above the rod are no longer a straddle
    st_bad = grasp_straddle_test(rod, dict(hp, pos=[hp["pos"][0], hp["pos"][1], hp["pos"][2] + 0.02]))
    assert not st_bad["ok"], st_bad
    print(f"[handle_pinch_pose] residual {hp['worst_residual_mm']} mm; straddle ok={st['ok']} "
          f"rise {st['pad_rise_mm']} mm  ({time.time()-t0:.0f}s)")

    # -- rope_pinch_pose finds the segment by arc length -------------------------
    rp = rope_pinch_pose(rope, arc_frac=0.5, fingers=("thumb", "index"), yaw_step_deg=60,
                         pitches_deg=(-45, 0, 45), iters=240)
    assert rp["rope"]["n_segments"] == 3 and rp["segment"]["index"] == 1, rp["segment"]
    assert rp["worst_residual_mm"] < 4.0, rp["residual_mm"]
    print(f"[rope_pinch_pose] segment {rp['segment']['geom']}, residual {rp['worst_residual_mm']} mm  ({time.time()-t0:.0f}s)")

    # -- grasp_through_hole_test: a synthetic hole around the index finger -------
    model, data0, rig, ids = H.open_scene(box)
    import mujoco
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, best)
    tip = np.asarray(d.site_xpos[rig.fingertip_sites["index"]], float)
    kn = rig.knuckle_point(d, "index")
    mid, axis = 0.5 * (tip + kn), (tip - kn) / np.linalg.norm(tip - kn)
    # the hole's normal points back toward the knuckle (the near side)
    th = grasp_through_hole_test(box, best, "index", mid, -axis, [0.02, 0.02])
    assert th["ok"], th
    side = np.cross(axis, [0, 0, 1.0])
    if np.linalg.norm(side) < 0.2:                    # a finger pointing straight down: any horizontal offset is beside it
        side = np.cross(axis, [1.0, 0, 0])
    side = side / np.linalg.norm(side)
    th_bad = grasp_through_hole_test(box, best, "index", mid + 0.05 * side, -axis, [0.02, 0.02])
    assert not th_bad["ok"], th_bad
    print(f"[grasp_through_hole_test] through={th['ok']} beside={th_bad['ok']}  ({time.time()-t0:.0f}s)")

    # -- grasp_prior_bodex: only when a BODex environment is available ----------
    bodex_dir = os.environ.get("DEXAGENT_BODEX_DIR", f"{ASSETS}/BODex")
    bodex_py = os.environ.get("DEXAGENT_BODEX_PYTHON", "")
    if bodex_py and Path(bodex_py).exists() and Path(bodex_dir, "example_grasp", "plan_batch_env.py").exists():
        import trimesh
        from tool_library.skills.grasp_prior_bodex import grasp_prior_bodex
        mesh = Path(tmp) / "cyl.obj"
        trimesh.creation.cylinder(radius=0.03, height=0.12, sections=48).export(mesh)
        pr = grasp_prior_bodex(cyl, str(mesh), bodex_dir, bodex_py, hand="sharpa", n_seeds=8,
                               out_dir=str(Path(tmp) / "bodex"), timeout_s=1500)
        assert pr["candidates"], pr
        lifted = 0
        for c in pr["candidates"][:4]:
            c2 = optimize_fingertip_contacts(cyl, c, fingers=list(rig.fingertip_sites))
            lifted += int(grasp_lift_test(cyl, c2, approach=False)["ok"])
        assert lifted >= 1, "no BODex candidate lifted the cylinder after cinching"
        print(f"[grasp_prior_bodex] {pr['n_seeds']} seeds, {pr['n_force_closure']} force-closure, "
              f"joint sign {pr['joint_sign']}, {lifted}/4 lift after cinch  ({time.time()-t0:.0f}s)")
    else:
        print("[SKIP] grasp_prior_bodex (no BODex environment; set DEXAGENT_BODEX_DIR / DEXAGENT_BODEX_PYTHON)")

    print(f"OK test_grasp ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
