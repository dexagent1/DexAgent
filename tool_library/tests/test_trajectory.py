"""Smoke tests for the trajectory skills, episode audits and task gates.

    MUJOCO_GL=egl python tool_library/tests/test_trajectory.py

A synthetic bench is built here: a table, a 6-joint arm with visual-only
links, a floating two-finger hand welded (position-only) to the arm's tip,
a cup and a bowl. A pick-and-place episode is generated kinematically, the
arm solved by IK, and every tool is run on a case that must pass and one
that must fail. Nothing external is needed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import trajectory as TJ  # noqa: E402

SCENE = """<mujoco model="bench"><compiler angle="radian"/>
<option timestep="0.002" gravity="0 0 -9.81"/>
<default><geom friction="1 0.005 0.0001" solref="0.01 1"/></default>
<worldbody>
  <light pos="0 0 3" dir="0 0 -1"/>
  <body name="table"><geom name="g_table" type="box" size="0.8 0.8 0.02" pos="0 0 -0.02" rgba=".7 .7 .7 1"/></body>
  <body name="robot_right_arm_root" pos="-0.45 -0.15 0">
   <geom type="cylinder" size="0.04 0.02" pos="0 0 0.02" group="1" contype="0" conaffinity="0" rgba=".3 .3 .3 1"/>
   <body name="robot_right_arm_link1" pos="0 0 0.04">
    <joint name="robot_right_arm_j1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
    <geom type="capsule" fromto="0 0 0 0 0 0.08" size="0.03" group="1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
    <body name="robot_right_arm_link2" pos="0 0 0.08">
     <joint name="robot_right_arm_j2" type="hinge" axis="0 1 0" range="-1.9 1.9"/>
     <geom type="capsule" fromto="0 0 0 0 0 0.30" size="0.028" group="1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
     <body name="robot_right_arm_link3" pos="0 0 0.30">
      <joint name="robot_right_arm_j3" type="hinge" axis="0 1 0" range="-2.6 2.6"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.26" size="0.025" group="1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
      <body name="robot_right_arm_link4" pos="0 0 0.26">
       <joint name="robot_right_arm_j4" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
       <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.022" group="1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
       <body name="robot_right_arm_link5" pos="0 0 0.06">
        <joint name="robot_right_arm_j5" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.02" group="1" contype="0" conaffinity="0" rgba=".8 .8 .8 1"/>
        <body name="robot_right_arm_tip" pos="0 0 0.05">
         <joint name="robot_right_arm_j6_roll" type="hinge" axis="0 0 1" range="-2.09 2.09"/>
         <geom type="cylinder" size="0.02 0.01" pos="0 0 0.01" group="1" contype="0" conaffinity="0" rgba=".5 .5 .9 1"/>
        </body></body></body></body></body></body></body>
  <body name="robot_right_hand" pos="0 0 0.3">
   <joint name="robot_right_pos_x" type="slide" axis="1 0 0" range="-2 2"/>
   <joint name="robot_right_pos_y" type="slide" axis="0 1 0" range="-2 2"/>
   <joint name="robot_right_pos_z" type="slide" axis="0 0 1" range="-2 2"/>
   <joint name="robot_right_rot_x" type="hinge" axis="1 0 0" range="-6.28 6.28"/>
   <joint name="robot_right_rot_y" type="hinge" axis="0 1 0" range="-6.28 6.28"/>
   <joint name="robot_right_rot_z" type="hinge" axis="0 0 1" range="-6.28 6.28"/>
   <geom name="palm" type="box" size="0.03 0.05 0.012" pos="0 0 -0.012" rgba=".9 .7 .5 1" mass="0.3"/>
   <body name="robot_right_finger_a" pos="0 0.045 -0.024">
    <joint name="robot_right_finger_a_j" type="hinge" axis="-1 0 0" range="0 1.5"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.008" rgba=".9 .7 .5 1" mass="0.03"/>
    <body name="robot_right_finger_a_tip" pos="0 0 -0.05"><geom name="tip_a" type="sphere" size="0.009" rgba=".9 .5 .5 1" mass="0.01"/></body>
   </body>
   <body name="robot_right_finger_b" pos="0 -0.045 -0.024">
    <joint name="robot_right_finger_b_j" type="hinge" axis="1 0 0" range="0 1.5"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.008" rgba=".9 .7 .5 1" mass="0.03"/>
    <body name="robot_right_finger_b_tip" pos="0 0 -0.05"><geom name="tip_b" type="sphere" size="0.009" rgba=".9 .5 .5 1" mass="0.01"/></body>
   </body>
  </body>
  <body name="cup" pos="0.05 0.10 0.05"><freejoint name="cup_free"/>
   <geom name="cup_body" type="cylinder" size="0.028 0.05" rgba=".8 .2 .2 1" mass="0.05"/></body>
  <body name="bowl" pos="0.15 -0.22 0.015"><freejoint name="bowl_free"/>
   <geom name="bowl_body" type="box" size="0.07 0.07 0.015" rgba=".2 .2 .8 1" mass="0.3"/></body>
</worldbody>
<equality>
  <weld name="arm_hand" body1="robot_right_arm_tip" body2="robot_right_hand" torquescale="0"/>
</equality>
</mujoco>"""

EE = {"right": "robot_right_hand"}
FPS = 30.0


def build_bench(tmp):
    path = Path(tmp) / "bench.xml"
    path.write_text(SCENE)
    return str(path)


def hand_pose_path():
    """Hand (palm centre) path of a pick-and-place: home -> above cup -> grasp -> lift -> carry -> place -> retreat."""
    cup0 = np.array([0.05, 0.10, 0.05])
    bowl = np.array([0.15, -0.22, 0.015])
    grasp = cup0 + np.array([0.0, 0.0, 0.085])           # palm bottom 1 cm above the cup, tips around its upper half
    place = np.array([bowl[0], bowl[1], 0.03 + 0.05 + 0.085])
    keys = [("home", np.array([-0.12, -0.05, 0.28]), 20),
            ("reach", grasp + [0, 0, 0.10], 40),
            ("descend", grasp, 25),
            ("close", grasp, 15),
            ("lift", grasp + [0, 0, 0.12], 30),
            ("carry", place + [0, 0, 0.12], 50),
            ("place", place, 25),
            ("open", place, 15),
            ("retreat", place + [0, 0, 0.12], 30),
            ("home", np.array([-0.12, -0.05, 0.28]), 40)]
    P, labels = [], []
    for i, (lab, p, n) in enumerate(keys):
        if i == 0:
            P += [p] * n; labels += [lab] * n
        else:
            u = TJ.smoothstep(np.arange(1, n + 1) / n)[:, None]
            P += list(keys[i - 1][1] + u * (p - keys[i - 1][1])); labels += [lab] * n
    return np.array(P), labels


def make_episode(scene_path):
    """Kinematic pick-and-place: hand path, fingers, cup riding the hand, arm by position-only IK."""
    model = TJ.load_model(scene_path)
    P, labels = hand_pose_path()
    T = len(P)
    Q = np.repeat(model.qpos0[None], T, axis=0)
    hid = TJ.body_id(model, "robot_right_hand")
    hand = TJ.Chain(model, hid)
    cup_adr = TJ.free_joint_qadr(model, "cup")
    f_a = TJ.joint_qadr(model, ["robot_right_finger_a_j"])[0]
    f_b = TJ.joint_qadr(model, ["robot_right_finger_b_j"])[0]
    close_ang = 0.20                                        # tips 8 mm clear when open, ~2 mm into the cup closed
    data = mujoco.MjData(model)
    holding = False
    cup_rel = None
    R_hand = np.eye(3)
    for t in range(T):
        data.qpos[:] = Q[t]
        hand.solve(data, P[t], R_hand)
        Q[t, hand.qadr] = data.qpos[hand.qadr]
        lab = labels[t]
        if lab == "close":
            k = labels[:t + 1].count("close") / 15.0
            Q[t, [f_a, f_b]] = close_ang * TJ.smoothstep(k)
            if k >= 1.0:
                holding = True
                cup_rel = Q[t, cup_adr:cup_adr + 3] - P[t]
        elif lab in ("lift", "carry", "place"):
            Q[t, [f_a, f_b]] = close_ang
        elif lab == "open":
            k = labels[:t + 1].count("open") / 15.0
            Q[t, [f_a, f_b]] = close_ang * (1 - TJ.smoothstep(k))
            holding = False
        if holding:
            Q[t, cup_adr:cup_adr + 3] = P[t] + cup_rel
        elif t > 0 and lab in ("open", "retreat", "home") and labels[t - 1] in ("place", "open", "retreat", "home"):
            Q[t, cup_adr:cup_adr + 7] = Q[t - 1, cup_adr:cup_adr + 7]
    # the arm follows the hand (position-only weld: its attachment point meets the hand origin)
    sides = TJ.discover_sides(model, EE)["right"]
    arm, point = sides["follower"]["chain"], sides["follower"]["point"]
    rest = np.array([0.0, 0.6, 0.9, 0.0, 0.3, 0.0])
    for t in range(T):
        data.qpos[:] = Q[t]
        if t:
            data.qpos[arm.qadr] = Q[t - 1, arm.qadr]
        e, _ = arm.solve(data, P[t], None, point=point, rest=rest, iters=120)
        assert e < 0.003, f"arm cannot reach frame {t}: {e * 1000:.1f} mm"
        Q[t, arm.qadr] = data.qpos[arm.qadr]
    return Q, [f"right: {l}" for l in labels]


def save_traj(path, Q, labels):
    np.savez_compressed(path, qpos=Q, labels=np.array(labels), fps=FPS)
    return str(path)


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    assert cond, f"{name}: {detail}"


def main():
    tmp = tempfile.mkdtemp(prefix="dexagent_traj_", dir=os.environ.get("TMPDIR"))
    scene = build_bench(tmp)
    Q, labels = make_episode(scene)
    traj = save_traj(Path(tmp) / "traj.npz", Q, labels)
    model = TJ.load_model(scene)
    T = len(Q)
    print(f"bench episode: {T} frames, nq={model.nq}")

    # ---------------- episode audits ----------------
    from tool_library.verifiers.joint_speed_check import check_joint_speed
    r = check_joint_speed(scene, traj, end_effectors=EE)
    ok("joint_speed_check passes", r["ok"], {k: v["max"] for k, v in r.items() if isinstance(v, dict)})
    Qj = Q.copy(); Qj[100:, TJ.joint_qadr(model, ["robot_right_arm_j1"])[0]] += 0.3
    r = check_joint_speed(scene, Qj, end_effectors=EE)
    ok("joint_speed_check fails on a joint jump", not r["ok"], r["right_arm"]["max"])
    r = check_joint_speed(scene, Qj, end_effectors=EE, reference=Q)
    ok("joint_speed_check relative to reference fails too", not r["ok"])

    from tool_library.verifiers.arm_jerk_check import check_arm_jerk
    r = check_arm_jerk(scene, traj, end_effectors=EE)
    ok("arm_jerk_check passes", r["ok"], r["right_arm"]["max"])
    Qk = Q.copy(); a = TJ.joint_qadr(model, ["robot_right_arm_j2"])[0]; Qk[150, a] += 0.08
    r = check_arm_jerk(scene, Qk, end_effectors=EE)
    ok("arm_jerk_check fails on a spike", not r["ok"], r["right_arm"]["max"])

    from tool_library.verifiers.weld_gap_check import check_weld_gap
    r = check_weld_gap(scene, traj)
    ok("weld_gap_check passes", r["ok"], r["eq0"]["gap_mm"]["max"])
    Qw = Q.copy(); Qw[:, TJ.joint_qadr(model, ["robot_right_pos_z"])[0]] += 0.02
    r = check_weld_gap(scene, Qw)
    ok("weld_gap_check fails when the hand leaves the flange", not r["ok"], r["eq0"]["gap_mm"]["max"])

    from tool_library.verifiers.wrist_stop_check import check_wrist_stop
    r = check_wrist_stop(scene, traj)
    ok("wrist_stop_check passes on the model limits", r["ok"], list(r)[0])
    r = check_wrist_stop(scene, traj, joints={"robot_right_arm_j6_roll": (-5.0, 5.0)}, margin_deg=0.0)
    print("     wrist roll range:", r["robot_right_arm_j6_roll"]["min"], r["robot_right_arm_j6_roll"]["max"])
    Qs = Q.copy(); Qs[:, TJ.joint_qadr(model, ["robot_right_arm_j6_roll"])[0]] = np.radians(100.0)
    r = check_wrist_stop(scene, Qs, joints={"robot_right_arm_j6_roll": (-20.0, 90.0)})
    ok("wrist_stop_check fails outside the real stops", not r["ok"], r["robot_right_arm_j6_roll"]["frames_outside"])

    from tool_library.verifiers.wrist_spin_check import check_wrist_spin
    r = check_wrist_spin(scene, traj)
    ok("wrist_spin_check passes", r["ok"], r["robot_right_arm_j6_roll"]["largest_turn_deg"])
    Qsp = Q.copy(); a = TJ.joint_qadr(model, ["robot_right_arm_j6_roll"])[0]
    Qsp[:, a] = np.radians(np.linspace(-100, 100, T))
    r = check_wrist_spin(scene, Qsp)
    ok("wrist_spin_check fails on a 200 degree turn", not r["ok"], r["robot_right_arm_j6_roll"]["largest_turn_deg"])
    Qrd = Q.copy(); Qrd[25:55, a] = np.radians(40 * np.sin(np.linspace(0, np.pi, 30)))   # out and back inside "reach"
    r = check_wrist_spin(scene, (Qrd, labels, FPS))
    ok("wrist_spin_check fails on redundant turning inside one phase", not r["ok"], r["robot_right_arm_j6_roll"]["redundant_reversals"])

    from tool_library.verifiers.object_disturbance_check import check_object_disturbance
    r = check_object_disturbance(scene, traj, manipulated=["cup"])
    ok("object_disturbance_check passes (bowl untouched)", r["ok"], r["untouched"]["bowl"]["max_drift_mm"])
    Qd = Q.copy(); ba = TJ.free_joint_qadr(model, "bowl"); Qd[200:, ba] += 0.03
    r = check_object_disturbance(scene, Qd, manipulated=["cup"])
    ok("object_disturbance_check fails when the bowl is pushed 3 cm", not r["ok"], r["disturbed"])
    Qt = Q.copy(); ca = TJ.free_joint_qadr(model, "cup"); Qt[150, ca] += 0.05
    r = check_object_disturbance(scene, Qt, manipulated=["cup"])
    ok("object_disturbance_check fails on a cup teleport", not r["ok"], r["jumped"])

    from tool_library.verifiers.contact_force_check import check_contact_force
    r = check_contact_force(scene, traj, holder="robot_right_hand", held="cup", frames=["right: lift", "right: carry"], min_n=0.2, max_n=80.0)
    ok("contact_force_check passes while carrying", r["ok"], r["force_n"])
    r2 = check_contact_force(scene, traj, holder="robot_right_hand", held="cup", frames=["right: reach"], min_n=0.2)
    ok("contact_force_check fails before the grasp", not r2["ok"], r2["force_n"])

    from tool_library.verifiers.uncovered_frames_check import check_uncovered_frames
    r = check_uncovered_frames(traj, video_frames=T, video_fps=FPS, expected_phases=["*lift*", "*place*"])
    ok("uncovered_frames_check passes", r["ok"], r["covered_frac"])
    bad = list(labels); bad[40:50] = [""] * 10; bad[70] = "right: blip"
    r = check_uncovered_frames((Q, bad, FPS), video_frames=T + 200)
    ok("uncovered_frames_check fails (gap, sliver, short)", not r["ok"], (r["unlabeled_frames"], len(r["short_phases"]), r["covered_frac"]))

    from tool_library.verifiers.trajectory_audit import audit_trajectory
    r = audit_trajectory(scene, traj, EE, manipulated=["cup"], video_frames=T, video_fps=FPS)
    print(r["table"])
    ok("trajectory_audit passes on the bench episode", r["ok"], r["failed"])
    r = audit_trajectory(scene, Qj, EE, manipulated=["cup"], reference=Q)
    ok("trajectory_audit fails on the jump episode", not r["ok"], r["failed"])

    # ---------------- task gates ----------------
    from tool_library.verifiers.final_pose_check import check_final_pose
    r = check_final_pose(scene, traj, "cup", reference=traj)
    ok("final_pose_check passes vs itself", r["ok"], r["pos_err_m"])
    r = check_final_pose(scene, traj, "cup", target_pos=[0.5, 0.5, 0.05])
    ok("final_pose_check fails vs a far target", not r["ok"], r["pos_err_m"])

    from tool_library.verifiers.lift_check import check_lift
    r = check_lift(scene, traj, "cup", reference=traj, holder="robot_right_hand")
    ok("lift_check passes", r["ok"], (r["rise_m"], r.get("contact_frac_while_up")))
    Qn = Q.copy(); Qn[:, ca:ca + 7] = Q[0, ca:ca + 7]
    r = check_lift(scene, Qn, "cup", reference=traj)
    ok("lift_check fails when the cup never rises", not r["ok"], r["rise_m"])

    from tool_library.verifiers.table_clearance_check import check_table_clearance
    r = check_table_clearance(scene, traj, "cup", frames=["right: carry"])
    ok("table_clearance_check passes", r["ok"], r["min_clearance_m"])
    Ql = Q.copy(); Ql[:, ca + 2] -= 0.20
    r = check_table_clearance(scene, (Ql, labels, FPS), "cup", frames=["right: carry"])
    ok("table_clearance_check fails when dragged through the table", not r["ok"], r["min_clearance_m"])

    from tool_library.verifiers.holds_at_video_end_check import check_holds_at_video_end
    t_carry = labels.index("right: carry") + 25
    r = check_holds_at_video_end(scene, traj, "cup", holder="robot_right_hand", frame=t_carry, settle_s=0.5)
    ok("holds_at_video_end_check: the pinch holds the cup mid-carry", r["ok"], (r["drift_m"], r["robot_bodies_touching_start"]))
    Qo = Q.copy(); Qo[:, [f for f in TJ.joint_qadr(model, ["robot_right_finger_a_j", "robot_right_finger_b_j"])]] = 0.0
    r = check_holds_at_video_end(scene, Qo, "cup", holder="robot_right_hand", frame=t_carry, settle_s=0.5)
    ok("holds_at_video_end_check: open fingers drop it", not r["ok"], r["drift_m"])
    r = check_holds_at_video_end(scene, traj, "cup", holder="robot_right_hand")
    ok("holds_at_video_end_check: placed cup stays put at the end", r["ok"], (r["on_table_at_start"], r["drift_m"]))

    # ---------------- measurement skills ----------------
    from tool_library.skills.measure_object_trajectory import measure_object_trajectory
    m = measure_object_trajectory(scene, traj, "cup")
    ok("measure_object_trajectory: window starts at the lift, motion stops at the place", m["moved"] and labels[m["motion_window"][0]] in ("right: close", "right: lift")
       and m["last_move"]["label"] in ("right: place", "right: open") and abs(m["max_lift_m"] - 0.12) < 0.04,
       (m["motion_window"], m["last_move"], m["max_lift_m"], m["start_to_end_m"]))
    m = measure_object_trajectory(scene, traj, "bowl")
    ok("measure_object_trajectory: bowl did not move", not m["moved"])

    from tool_library.skills.measure_joint_trajectory import measure_joint_trajectory
    m = measure_joint_trajectory(scene, traj, end_effectors=EE)
    ok("measure_joint_trajectory: groups found", set(m["groups"]) == {"right_ee", "right_arm", "right_fingers"}, {k: v["max_step"] for k, v in m["groups"].items()})

    from tool_library.skills.measure_carry_clearance import measure_carry_clearance
    m = measure_carry_clearance(scene, traj, "cup", frames=["right: carry"])
    ok("measure_carry_clearance: clearance positive, nearest bodies listed", m["min_table_clearance_m"] > 0.05 and "robot_right_finger_a_tip" in m["nearest"],
       (m["min_table_clearance_m"], list(m["nearest"])[:3]))

    from tool_library.skills.analyze_manipulation import analyze_manipulation
    H, W = 120, 160
    Tm = 90
    masks = np.zeros((Tm, H, W), bool)
    yy, xx = np.mgrid[:H, :W]
    hand_pts = {"right": {}, "left": {}}
    cx = np.concatenate([np.full(30, 40.0), np.linspace(40, 120, 30), np.full(30, 120.0)])
    for t in range(Tm):
        masks[t] = (xx - cx[t]) ** 2 + (yy - 60) ** 2 < 10 ** 2
        hx = 40.0 if t < 10 else (cx[t] if t < 60 else 120.0 + (t - 60) * 1.5)
        hand_pts["right"][t] = np.array([[hx - 12, 55.0], [hx - 11, 60.0], [hx - 12, 65.0]])
        hand_pts["left"][t] = np.array([[130.0, 20.0], [131.0, 22.0]])            # resting hand
    pos = np.stack([cx / 400.0, np.zeros(Tm), np.zeros(Tm)], axis=1)
    rep = analyze_manipulation({"disc": masks}, hand_pts, {"disc": pos}, stride=1)
    ok("analyze_manipulation: right hand manipulates the disc", rep["objects"]["disc"]["manipulated_by"] == "right",
       (rep["objects"]["disc"].get("hand_evidence"), rep["hands"]["right"]["touches"].get("disc", {}).get("first_contact_frame")))

    # ---------------- trajectory skills ----------------
    from tool_library.skills.solve_arm_ik import solve_arm_ik
    P, _ = TJ.pose_track(model, Q, TJ.body_id(model, "robot_right_arm_tip"))
    res = solve_arm_ik(scene, "robot_right_arm_tip", P[::10], rest=Q[0, TJ.joint_qadr(model, [f"robot_right_arm_j{i}" for i in range(1, 6)] + ["robot_right_arm_j6_roll"])])
    ok("solve_arm_ik: tip path reached (position-only)", res["ok"], res["worst"])
    Rt = np.stack([TJ.quat2mat([1, 0, 0, 0])] * 3)
    res = solve_arm_ik(scene, "robot_right_arm_tip", P[[0, 100, 200]], rotations=Rt)
    print("     full-pose IK:", res["worst"], "ok" if res["ok"] else "(unreachable orientation, reported honestly)")

    from tool_library.skills.refit_to_real_rig import refit_to_real_rig
    Ptip, Rtip = TJ.pose_track(model, Q, TJ.body_id(model, "robot_right_arm_tip"))
    res = refit_to_real_rig(scene, "robot_right_arm_tip", Ptip, Rtip, fps=FPS, table_top_z=0.0, floor_margin_m=0.0,
                            out_path=str(Path(tmp) / "right_arm.npz"))
    ok("refit_to_real_rig: tracks the flange path within 10 mm, rate under 180 deg/s", res["residual"]["pos_mm_max"] < 10 and res["ok"],
       (res["residual"], res["max_rate_deg_s"], res["resets"]))

    from tool_library.skills.adapt_trajectory_to_layout import Episode, adapt_trajectory_to_layout
    segs = {"right": [{"phases": ["right: reach"], "blend": ["world", "cup"]},
                      {"phases": ["right: descend", "right: close", "right: lift"], "anchor": "cup"},
                      {"phases": ["right: carry"], "blend": ["cup", "bowl"]},
                      {"phases": ["right: place", "right: open"], "anchor": "bowl"},
                      {"phases": ["right: retreat"], "blend": ["bowl", "world"]}]}
    objs = {"cup": {"carrier": "right", "release": "right: open"}, "bowl": {}}
    ep = Episode(scene, traj, EE, objs, segs)
    Q0, ov, rep = ep.adapt({"objects": {"cup": [0, 0, 0], "bowl": [0, 0, 0]}, "robot": {}})
    ok("adapt_trajectory_to_layout: zero layout reproduces the episode", np.array_equal(Q0, Q) and rep["ok"], rep["failed"])
    Q1, ov, rep = ep.adapt({"objects": {"cup": [0.04, -0.03, 25.0], "bowl": [-0.03, 0.02, 0.0]}, "robot": {}})
    P1, _ = TJ.pose_track(model, Q1, TJ.body_id(model, "cup"))
    Pb, _ = TJ.pose_track(model, Q1, TJ.body_id(model, "bowl"))
    ok("adapt_trajectory_to_layout: displaced layout passes all checks and the cup ends on the moved bowl",
       rep["ok"] and np.linalg.norm(P1[-1, :2] - Pb[-1, :2]) < 0.02 and abs(P1[0, 0] - 0.09) < 1e-6, (rep["failed"], rep["ik"]))
    samples, summary = adapt_trajectory_to_layout(scene, traj, EE, objs, segs,
                                                  ranges={"cup": {"dx": [-0.03, 0.03], "dy": [-0.03, 0.03], "dyaw_deg": [-20, 20]}},
                                                  num_samples=2, max_attempts=6, out_dir=str(Path(tmp) / "samples"), log=lambda *a: None)
    ok("adapt_trajectory_to_layout: sampled layouts written", summary["ok"] and (Path(tmp) / "samples" / "0000" / "trajectory.npz").exists(), summary["accepted"])

    from tool_library.skills.export_policy_trajectory import export_policy_trajectory
    res = export_policy_trajectory(scene, traj, str(Path(tmp) / "episode.npz"),
                                   groups={"right_arm": "chain:robot_right_arm_tip", "right_hand": "subtree:robot_right_hand"},
                                   end_effectors=EE, objects=["cup"], target_fps=15, csv=True)
    z = np.load(res["paths"]["npz"])
    ok("export_policy_trajectory: 15 fps, 6+8 dims, poses and csv", res["action_dim"] == 14 and abs(res["n_frames"] - T / 2) <= 1
       and z["right_arm"].shape[1] == 6 and "ee_right_quat_wxyz" in z and Path(res["paths"]["csv"]).exists(), (res["n_frames"], res["groups"]))

    from tool_library.skills.optimize_reward_cem import optimize_reward_cem
    best = optimize_reward_cem(lambda p: {"gates": {"x_pos": p["x"] > 0}, "terms": {"q": -(p["x"] - 0.7) ** 2 - (p["y"] + 0.2) ** 2}},
                               {"x": (-1, 1), "y": (-1, 1)}, iterations=10, population=20, elite=5)
    ok("optimize_reward_cem: finds the gated optimum", not best["gates_failed"] and abs(best["params"]["x"] - 0.7) < 0.05 and abs(best["params"]["y"] + 0.2) < 0.05,
       (best["params"], best["n_evals"]))

    from tool_library.skills.chain_subgoals import chain_subgoals
    q_a = Q[0].copy(); q_b = Q[100].copy(); q_c = Q[200].copy()
    res = chain_subgoals(scene, [{"qpos": q_a, "hold": 5, "label": "a"}, {"qpos": q_b, "transit": 5, "hold": 3, "label": "b"},
                                 {"joints": {"robot_right_finger_a_j": 1.0}, "transit": 10, "hold": 4, "label": "close"},
                                 {"qpos": q_c, "transit": 30, "label": "c"}], fps=FPS, max_rate_deg_s=90.0)
    ok("chain_subgoals: continuous, rate-limited, transits stretched where needed", res["max_rate_deg_s"] <= 90.0 + 1e-6 and len(res["stretched"]) >= 1
       and res["labels"][res["keyframe_frames"][2]] == "close" and len(res["qpos"]) == len(res["labels"]), (res["max_rate_deg_s"], res["stretched"][:2]))

    from tool_library.skills.optimize_frame_locked import optimize_frame_locked
    grasp_frames = [t for t, l in enumerate(labels) if l in ("right: descend", "right: close")]

    t_mid = grasp_frames[len(grasp_frames) // 2]              # inside the window's plateau

    def evaluate(Qc):
        Pz, _ = TJ.pose_track(model, Qc[t_mid:t_mid + 1], TJ.body_id(model, "robot_right_hand"))
        z = float(Pz[0, 2])
        return {"gates": {"low_enough": z < 0.12}, "terms": {"height": -abs(z - 0.115)}}

    best = optimize_frame_locked(scene, traj, "robot_right_hand", [{"frames": ["right: descend", "right: close"], "params": {"dz": (-0.03, 0.01)}, "ramp": 6}],
                                 evaluate, iterations=4, population=8, elite=3)
    Qb = best["qpos"]
    untouched = [t for t, l in enumerate(labels) if l not in ("right: descend", "right: close")]
    ok("optimize_frame_locked: gate met by lowering the grasp, other frames untouched",
       not best["gates_failed"] and np.array_equal(Qb[untouched], Q[untouched]) and best["params"]["0:dz"] < -0.005,
       (best["params"], best["ik_worst"]))

    from tool_library.skills.attach_arms import attach_arms
    res = attach_arms(scene, traj, arm_root="robot_right_arm_root", arm_tip="robot_right_arm_tip", hand_body="robot_right_hand",
                      candidates={"radius_frac": [0.5, 0.7], "n_yaw": 8}, table_top_z=0.0, out_scene=str(Path(tmp) / "mounted.xml"))
    m2 = mujoco.MjModel.from_xml_path(str(Path(tmp) / "mounted.xml"))
    ok("attach_arms: a base is found that tracks the hand path, scene written with the weld",
       res["ok"] and res["residual"]["pos_mm_max"] < 15 and m2.neq == 2, (res["base_pos"], res["base_yaw_deg"], res["residual"]))

    from tool_library.skills.retarget_hand_motion import retarget_hand_motion
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    hb = TJ.body_id(model, "robot_right_hand")
    Rh, ph = data.xmat[hb].reshape(3, 3), data.xpos[hb]
    tgt = {}
    for f, body in (("a", "robot_right_finger_a_tip"), ("b", "robot_right_finger_b_tip")):
        bid = TJ.body_id(model, body)
        pts = []
        for ang in np.linspace(0.0, 1.2, 40):
            data.qpos[TJ.joint_qadr(model, [f"robot_right_finger_{f}_j"])[0]] = ang
            mujoco.mj_kinematics(model, data)
            pts.append(Rh.T @ (data.xpos[bid] - ph))
        tgt[f] = np.array(pts)
    res = retarget_hand_motion(scene, "robot_right_hand", {"a": "robot_right_finger_a_tip", "b": "robot_right_finger_b_tip"}, tgt, fps=FPS)
    ok("retarget_hand_motion: fingertip targets reproduced", res["ok"] and max(v["max_mm"] for v in res["residual_mm"].values()) < 3.0,
       (res["residual_mm"], res["joint_names"]))

    # ---------------- image measurements ----------------
    from tool_library.common import tablecam as TC
    from tool_library.skills.measure_box_on_table import hull_mask, measure_box_on_table, prism_points
    cam = TC.camera(focal_px=870, width=640, height=480, h_above_table_m=0.684, pitch_deg=36.0)
    truth = dict(L=0.20, W=0.14, H=0.06, rad=0.015)
    views = []
    for fwd, lat, yaw in ((0.80, 0.05, 10.0), (0.85, -0.08, 70.0)):
        views.append({"mask": hull_mask(TC.project(prism_points(truth["L"], truth["W"], truth["H"], truth["rad"], fwd, lat, yaw), cam), (480, 640))})
    cam_wrong = TC.camera(870, 640, 480, 0.684, pitch_deg=33.0)
    fit = measure_box_on_table(views, cam_wrong, fit_pitch=True, restarts=2)
    ok("measure_box_on_table: footprint within 1 cm, yaws recovered, IoU > 0.97 from a wrong pitch and no init",
       abs(fit["length_m"] - truth["L"]) < 0.01 and abs(fit["width_m"] - truth["W"]) < 0.01 and fit["mean_iou"] > 0.97
       and abs(fit["views"][0]["yaw_deg"] - 10.0) < 3.0 and abs(fit["views"][1]["yaw_deg"] - 70.0) < 3.0,
       (fit["length_m"], fit["width_m"], fit["pitch_deg"], fit["mean_iou"], [v["yaw_deg"] for v in fit["views"]]))

    from tool_library.skills.measure_rope import measure_rope
    import cv2
    s = np.linspace(0, 1, 200)
    poly = np.stack([0.70 + 0.25 * s, -0.10 + 0.15 * np.sin(2 * np.pi * s), np.full_like(s, 0.006)], axis=1)
    true_len = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
    uv = TC.project(poly, cam)
    mask = np.zeros((480, 640), np.uint8)
    for (u0, v0), (u1, v1) in zip(uv[:-1], uv[1:]):
        r_px = int(round(0.006 * cam["focal_px"] / float(TC.optical_depth(poly[:1], cam)[0])))
        cv2.line(mask, (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1))), 1, max(2, 2 * r_px))
    mr = measure_rope(mask > 0, cam)
    ok("measure_rope: length within 4% and radius within 2 mm", abs(mr["length_m"] - true_len) / true_len < 0.04 and abs(mr["radius_m"] - 0.006) < 0.002,
       (mr["length_m"], round(true_len, 4), mr["radius_m"], mr["self_crossings"]))

    print("\nall trajectory tests passed")


if __name__ == "__main__":
    main()
