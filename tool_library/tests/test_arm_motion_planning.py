"""Arm motion planning (common/armplan.py) through the executor's arm fill (armhand.couple, _couple_two_hands), on the
setup bench with parametric boxes, each case varied over several placements:

- a box between two arm configurations that the straight joint leg runs into: the planned leg clears it by a margin and
  passes swept_collision_audit and robot_self_collision_check; a carried part rides the planned carry and clears the box
  too; a clean leg is left exactly as it was;
- two arms whose simultaneous straight legs cross in a two-hand entry: no arm-arm, hand-hand or hand-arm contact;
- an idle arm resting where the active hand must go, or standing at its home pose there once the stations are moved
  toward the objects: parked out of the way (not when its hand holds something), and its own next reach or its return
  home starts from the parked configuration and arrives without contact; a planned hand that runs into a hand holding
  a part fails before physics (hand_clearance);
- a too-fast segment (a one-frame step and a fast swing): retimed to pass joint_acceleration_audit and
  robot_joint_jump_audit, close to the minimum time, with the contact phase's frames kept 1:1.

Measured failures behind it: the steadying left forearm in the right hand's way at the bottle cap (arm_collision on
link 3/4 in 30 of the bottle run's attempts), the parked right thumb hit by the left hand at the braid, two hands 3 cm
apart on the magnetic rings, and Stage 4 layouts rejected for joint speed.

    OMP_NUM_THREADS=1 python tool_library/tests/test_arm_motion_planning.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import armhand as AH  # noqa: E402
from tool_library.common import armplan as AP  # noqa: E402
from tool_library.common import episode as E  # noqa: E402
from tool_library.common import rigid_robot as RR  # noqa: E402
from tool_library.skills.execute_subgoal_plan import ONE_HAND_OVERLAP_M, _couple_two_hands, _hand_gap_failure  # noqa: E402
from tool_library.verifiers.arm_workspace_map import named_orientation  # noqa: E402
from tool_library.verifiers.joint_acceleration_audit import joint_acceleration_audit  # noqa: E402
from tool_library.verifiers.robot_joint_jump_audit import robot_joint_jump_audit  # noqa: E402
from tool_library.verifiers.robot_self_collision_check import robot_self_collision_check  # noqa: E402
from tool_library.verifiers.swept_collision_audit import swept_collision_audit  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_armplan_", dir=os.environ.get("TMPDIR")))
SCENE = ROOT / "setup" / "scene.xml"
FPS = 30.0
FAILS = 0
T0 = time.time()


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def absolute_assets(spec, xml_path):
    base = Path(xml_path).resolve().parent
    for m in spec.meshes:
        if m.file and not Path(m.file).is_absolute():
            m.file = str(base / (spec.meshdir or "") / m.file)
    for t in spec.textures:
        if t.file and not Path(t.file).is_absolute():
            t.file = str(base / (spec.texturedir or "") / t.file)
    spec.meshdir = ""; spec.texturedir = ""
    return spec


def scene_with(name, boxes, parts=()):
    """The setup bench with static boxes [(centre, half sizes)] and free boxes `parts` [(centre, half sizes)]."""
    sp = absolute_assets(mujoco.MjSpec.from_file(str(SCENE)), SCENE)
    for i, (pos, half) in enumerate(boxes):
        b = sp.worldbody.add_body(); b.name = f"obstacle{i}"; b.pos = [float(v) for v in pos]
        g = b.add_geom(); g.name = f"obstacle{i}_box"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [float(h) for h in half]
        g.contype = 1; g.conaffinity = 1; g.rgba = [0.8, 0.2, 0.2, 1]
    for i, (pos, half) in enumerate(parts):
        b = sp.worldbody.add_body(); b.name = f"part{i}"; b.pos = [float(v) for v in pos]
        b.add_freejoint()
        g = b.add_geom(); g.name = f"part{i}_box"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [float(h) for h in half]
        g.contype = 1; g.conaffinity = 1; g.rgba = [0.2, 0.6, 0.2, 1]
    p = TMP / f"{name}.xml"
    p.write_text(sp.to_xml())
    return str(p)


class Bench:
    def __init__(self, scene):
        self.scene = str(scene)
        self.m = mujoco.MjModel.from_xml_path(self.scene)
        d = mujoco.MjData(self.m); mujoco.mj_resetDataKeyframe(self.m, d, 0)
        self.q0 = d.qpos.copy()
        self.S = {s: AH.ArmSide(self.m, s) for s in ("left", "right")}
        self.pl = AP.ArmPlanner(self.scene)
        self.wa = {s: [self.S[s].pt["rig"].wrist_qpos[k] for k in ("x", "y", "z", "rx", "ry", "rz")] for s in self.S}

    def ik(self, side, p, orient="palm_down"):
        S, R = self.S[side], named_orientation(side, orient)
        ep, er, qa = S.solve(self.q0, np.asarray(p, float) + R @ S.p_m, R @ S.R_m, S.home, multistart="force")
        return qa, 1000 * ep

    def reference(self, q_start, moves, n_hold=10, n_leg=50, n_after=10):
        """Hand reference: each moving side's wrist from its start to its goal (smoothstep), the rest still."""
        T = n_hold + n_leg + n_after
        Q = np.repeat(np.asarray(q_start, float)[None], T, 0)
        for t in range(T):
            u = float(np.clip((t - n_hold) / (n_leg - 1.0), 0.0, 1.0)); u = 3 * u ** 2 - 2 * u ** 3
            for s, (qa_s, qe_s) in moves.items():
                Q[t, self.wa[s]] = (1 - u) * qa_s[self.wa[s]] + u * qe_s[self.wa[s]]
        return Q, [n_hold, n_hold + n_leg - 1]

    def audit(self, cp, name):
        # Coupling produces a planning reference with redundant virtual wrist
        # Euler coordinates. Audit the named physical joints on the fixed robot
        # tree: an Euler branch change is not a physical hand rotation and must
        # not consume the swept check's finite subdivision budget.
        if not hasattr(self, "runtime"):
            self.runtime = RR.compile_rigid(self.scene)
            self.runtime_scene = TMP / f"{Path(self.scene).stem}_runtime.xml"
            self.runtime_scene.write_text(self.runtime.xml)
        f = TMP / f"{name}.npz"
        Q = self.runtime.to_runtime_qpos(cp["qpos"])
        E.save_episode(f, Q, cp["labels"], FPS, artifact_role="planning_reference")
        # A planning reference has only 30 Hz frames; production recordings
        # contain every 500 Hz integrator step. Spend enough interpolation
        # samples here to retain the audit's unchanged 3 mm spatial coverage.
        return robot_self_collision_check(str(self.runtime_scene), str(f), stride=1), swept_collision_audit(str(self.runtime_scene), str(f), max_sub=64), str(f)


# ------------------------------------------------------------------------------------------- 1. around an obstacle
bench0 = Bench(SCENE)
LEGS = [((-0.15, 0.40, 0.14), (0.15, 0.60, 0.14)), ((-0.05, 0.50, 0.14), (0.25, 0.60, 0.14)), ((-0.15, 0.40, 0.20), (0.25, 0.60, 0.20)),
        ((-0.05, 0.50, 0.20), (0.25, 0.50, 0.20))]
n_obst = 0
for k, (pa, pb) in enumerate(LEGS):
    qa, ea = bench0.ik("right", pa); qb, eb = bench0.ik("right", pb)
    if ea > 20.0 or eb > 20.0:                        # the solved configurations are the case; only a gross miss is skipped
        continue
    # the box where the straight joint leg puts the hand halfway: the straight swing runs through it
    mid = bench0.pl.hand_pose(bench0.pl.place(bench0.q0, "right", 0.5 * (qa + qb)), "right")[0]
    scene = scene_with(f"obstacle_{k}", [(mid - [0.0, 0.0, 0.02], [0.025, 0.025, 0.025])])
    B = Bench(scene)
    qA, qB = B.pl.place(B.q0, "right", qa), B.pl.place(B.q0, "right", qb)
    B.pl.setup("right", margin=0.05)
    ends = min(B.pl.clearance(qA)[0], B.pl.clearance(qB)[0])
    if ends < 0.015:
        continue                                      # the box touches an end: not this case
    Qref, kf = B.reference(qA, {"right": (qA, qB)})
    labels = ["right: hold"] * kf[0] + ["right: reach for obstacle0"] * (kf[1] - kf[0] + 1) + ["right: descend on obstacle0"] * (len(Qref) - kf[1] - 1)
    start = {"right": qa, "left": B.S["left"].home}
    straight = AH.couple(B.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0] + kf)
    planned = AH.couple(B.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0] + kf, planner=B.pl)
    s0, w0, _ = B.audit(straight, f"obstacle_{k}_straight")
    s1, w1, _ = B.audit(planned, f"obstacle_{k}_planned")
    legs = planned["report"]["plan"]["planned_legs"]
    clr = legs[0]["clearance_mm"] if legs else None
    n_obst += 1
    check(f"obstacle {k}: the straight leg runs the arm into the box", "arm-object" in s0["worst"] or "hand-object" in s0["worst"], str(s0["worst"])[:160])
    check(f"obstacle {k}: the planned leg clears it by a margin (>= 10 mm) and passes robot_self_collision_check and swept_collision_audit",
          bool(legs) and clr >= 10.0 and s1["ok"] and w1["ok"], f"{legs[0]['how'] if legs else planned['report']['plan']} clearance {clr} mm; self {s1['worst']}")
    check(f"obstacle {k}: the hand still reaches the leg's target", np.allclose(planned["arm"]["right"][kf[1]], straight["arm"]["right"][kf[1]], atol=1e-6))
check("obstacle placements tried", n_obst >= 3, f"{n_obst} of {len(LEGS)}")

# a carried part: a joint-space carry ('(free)') whose straight swing takes the part through a box; the part rides
# with the hand on the planned leg and clears the box too
n_carry = 0
for k, (pa, pb) in enumerate(LEGS[:3]):
    qa, ea = bench0.ik("right", pa); qb, eb = bench0.ik("right", pb)
    if ea > 20.0 or eb > 20.0:
        continue
    qm = bench0.pl.place(bench0.q0, "right", 0.5 * (qa + qb))
    mid = bench0.pl.hand_pose(qm, "right")[0]
    ha = bench0.pl.hand_pose(bench0.pl.place(bench0.q0, "right", qa), "right")[0]
    part0 = ha - [0.0, 0.0, 0.075]                     # a 3 cm cube under the palm-down hand
    scene = scene_with(f"carry_{k}", [(mid - [0.0, 0.0, 0.09], [0.03, 0.03, 0.03])], parts=[(part0, [0.015, 0.015, 0.015])])
    B = Bench(scene)
    qA = B.pl.place(B.q0, "right", qa)
    adr = int(B.m.jnt_qposadr[B.m.body("part0").jntadr[0]])
    qA[adr:adr + 3], qA[adr + 3:adr + 7] = part0, [1, 0, 0, 0]
    pyA, RyA = B.pl.hand_pose(qA, "right")
    rel = (RyA.T @ (part0 - pyA), RyA.T)
    qB = B.pl.place(qA, "right", qb, held=[(adr, rel[0], rel[1])])
    Qref, kf = B.reference(qA, {"right": (qA, qB)})
    for t in range(len(Qref)):                           # the part follows the hand reference (the kinematic carry)
        py, Ry = B.pl.hand_pose(Qref[t], "right")
        Qref[t, adr:adr + 3], Qref[t, adr + 3:adr + 7] = py + Ry @ rel[0], AP.TJ.mat2quat(Ry @ rel[1])
    labels = ["right: lift part0"] * kf[0] + ["right: carry part0 (free)"] * (kf[1] - kf[0] + 1) + ["right: carry part0 (down)"] * (len(Qref) - kf[1] - 1)
    start = {"right": qa, "left": B.S["left"].home}
    straight = AH.couple(B.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0] + kf)
    B.pl.setup("right", held=B.pl.free_body[adr], margin=0.05)
    if min(B.pl.clearance(q_)[0] for q_ in straight["qpos"][::2]) >= 0:
        continue                                      # the straight carry misses the box here
    n_carry += 1
    planned = AH.couple(B.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0] + kf, planner=B.pl)
    s1, w1, _ = B.audit(planned, f"carry_{k}_planned")
    legs = planned["report"]["plan"]["planned_legs"]
    rode = max(np.linalg.norm(planned["qpos"][t, adr:adr + 3] - (lambda h: h[0] + h[1] @ rel[0])(B.pl.hand_pose(planned["qpos"][t], "right")))
               for t in range(len(planned["qpos"])))
    B.pl.setup("right", held=B.pl.free_body[adr], obstacles="world", margin=0.05)
    part_clear = min(B.pl.clearance(q_)[0] for q_ in planned["qpos"])
    check(f"carry {k}: the planned carry takes the part around the box (part rides the hand within 1 mm, clear by {1000 * part_clear:.1f} mm)",
          bool(legs) and rode < 1e-3 and part_clear > 0.005 and w1["ok"] and s1["ok"], f"{legs[0]['how'] if legs else planned['report']['plan']} {s1['worst']}")
check("carry placements tried", n_carry >= 2, f"{n_carry}")

# a clean leg is left exactly as the straight fill had it
qa, _ = bench0.ik("right", LEGS[0][0]); qb, _ = bench0.ik("right", LEGS[0][1])
qA, qB = bench0.pl.place(bench0.q0, "right", qa), bench0.pl.place(bench0.q0, "right", qb)
Qref, kf = bench0.reference(qA, {"right": (qA, qB)})
labels = ["right: hold"] * kf[0] + ["right: reach for x"] * (kf[1] - kf[0] + 1) + ["right: descend on x"] * (len(Qref) - kf[1] - 1)
c0 = AH.couple(bench0.m, Qref, FPS, arm_start={"right": qa}, labels=labels, keyframe_frames=[0] + kf)
c1 = AH.couple(bench0.m, Qref, FPS, arm_start={"right": qa}, labels=labels, keyframe_frames=[0] + kf, planner=bench0.pl)
check("a clean leg is not replanned and the fill is unchanged", not c1["report"]["plan"]["planned_legs"] and len(c0["qpos"]) == len(c1["qpos"])
      and np.abs(c0["qpos"] - c1["qpos"]).max() < 1e-12, f"{c1['report']['plan']['checked_legs']} legs checked")

# ------------------------------------------------------------------------------------------- 2. two arms crossing
# found by sampling reachable palm-down poses (start and end pairs 4 cm apart or more): the two straight legs, run at
# the same time, bring the hands into each other half way
CROSS = [((-0.2, 0.4, 0.2), (0.05, 0.55, 0.12), (0.0, 0.55, 0.2), (0.2, 0.55, 0.2)),
         ((-0.25, 0.6, 0.2), (-0.15, 0.45, 0.2), (-0.05, 0.45, 0.2), (0.15, 0.55, 0.2)),
         ((-0.25, 0.6, 0.2), (-0.2, 0.4, 0.2), (-0.05, 0.45, 0.2), (0.05, 0.55, 0.2)),
         ((-0.1, 0.45, 0.2), (-0.25, 0.6, 0.2), (0.15, 0.55, 0.2), (-0.05, 0.45, 0.2)),
         ((-0.2, 0.6, 0.12), (-0.1, 0.4, 0.2), (0.0, 0.45, 0.2), (0.1, 0.55, 0.2)),
         ((-0.25, 0.55, 0.2), (-0.05, 0.45, 0.2), (0.0, 0.45, 0.2), (0.25, 0.6, 0.2)),
         ((-0.2, 0.4, 0.18), (0.05, 0.55, 0.10), (0.0, 0.55, 0.18), (0.2, 0.55, 0.18))]
n_cross = 0
for k, pts in enumerate(CROSS):
    q, bad = {}, False
    for key, side, p_ in zip(("l0", "l1", "r0", "r1"), ("left", "left", "right", "right"), pts):
        q[key], err = bench0.ik(side, p_)
        bad |= err > 20.0
    if bad:
        continue
    pl = bench0.pl
    qs = pl.place(pl.place(bench0.q0, "left", q["l0"]), "right", q["r0"])
    qe = pl.place(pl.place(bench0.q0, "left", q["l1"]), "right", q["r1"])
    Qref, kf = bench0.reference(qs, {"left": (qs, qe), "right": (qs, qe)})
    labels = (["left: hold a | right: hold b"] * kf[0] + ["left: reach for a | right: reach for b"] * (kf[1] - kf[0] + 1)
              + ["left: descend on a | right: descend on b"] * (len(Qref) - kf[1] - 1))
    start = {"left": q["l0"], "right": q["r0"]}
    straight = _couple_two_hands(bench0.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0] + kf)
    s0, _, _ = bench0.audit(straight, f"cross_{k}_straight")
    if not ({"hand-hand", "arm-hand", "arm-arm"} & set(s0["worst"])):
        continue                                      # this placement's straight legs do not meet
    n_cross += 1
    planned = _couple_two_hands(bench0.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0] + kf, planner=bench0.pl)
    s1, w1, _ = bench0.audit(planned, f"cross_{k}_planned")
    rep = planned["report"]["plan"]
    check(f"crossing {k}: no arm-arm, hand-hand or arm-hand contact after planning (straight: {sorted(s0['worst'])})",
          not ({"hand-hand", "arm-hand", "arm-arm"} & set(s1["worst"])) and s1["ok"] and not rep["failed_legs"],
          f"{[(p_['side'], p_['how'], p_['clearance_mm']) for p_ in rep['planned_legs']]} {s1['worst']}")
    ends_ok = all(np.allclose(planned["arm"][s][-1], straight["arm"][s][-1], atol=1e-6) for s in ("left", "right"))
    check(f"crossing {k}: both hands still end where the plan puts them", ends_ok)
check("crossing placements tried", n_cross >= 3, f"{n_cross}")

# ------------------------------------------------------------------------------------------- 3. an idle arm parked
n_park = 0
PARK = [((-0.08, 0.50, 0.15), (0.0, 0.50, 0.14)), ((-0.10, 0.50, 0.20), (-0.02, 0.50, 0.20)), ((-0.05, 0.50, 0.15), (0.05, 0.50, 0.14)),
        ((-0.12, 0.50, 0.15), (-0.05, 0.50, 0.14)), ((-0.08, 0.55, 0.15), (0.0, 0.50, 0.14)), ((-0.15, 0.45, 0.15), (-0.05, 0.50, 0.14))]
for k, (lp, rp) in enumerate(PARK):
    qL, e1 = bench0.ik("left", lp); qR, e2 = bench0.ik("right", rp)
    if e1 > 20.0 or e2 > 20.0:
        continue
    pl = bench0.pl
    qs = pl.place(bench0.q0, "left", qL)               # the left arm rests forward, where the right hand must go
    qe = pl.place(qs, "right", qR)
    Qref, kf = bench0.reference(qs, {"right": (qs, qe)}, n_hold=5, n_leg=60, n_after=15)
    labels = ["right: reach for x"] * (kf[1] + 1) + ["right: descend on x"] * (len(Qref) - kf[1] - 1)
    start = {"left": qL, "right": bench0.S["right"].home}
    straight = AH.couple(bench0.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0, kf[1]])
    s0, _, _ = bench0.audit(straight, f"park_{k}_straight")
    if "hand-hand" not in s0["worst"] and "arm-hand" not in s0["worst"]:
        continue
    n_park += 1
    planned = AH.couple(bench0.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0, kf[1]], planner=bench0.pl)
    s1, w1, _ = bench0.audit(planned, f"park_{k}_planned")
    pk = planned["report"]["plan"]["parked"]
    check(f"park {k}: the idle left arm is parked and nothing touches (straight: {sorted(s0['worst'])})",
          pk and pk[0]["parked"] and s1["ok"] and w1["ok"], f"{pk} {s1['worst']}")
    # it comes back: its own next reach (to where it rested) starts from the parked configuration
    if pk and pk[0]["parked"]:
        q_end = planned["qpos"][-1].copy()
        qpark = planned["arm"]["left"][-1]
        q_goal = pl.place(q_end, "left", qL)
        Q2, kf2 = bench0.reference(q_end, {"left": (q_end, q_goal)}, n_hold=5, n_leg=50, n_after=5)
        lab2 = ["left: reach for y"] * (kf2[1] + 1) + ["left: descend on y"] * (len(Q2) - kf2[1] - 1)
        back = AH.couple(bench0.m, Q2, FPS, arm_start={"left": qpark, "right": planned["arm"]["right"][-1]}, labels=lab2,
                         keyframe_frames=[0, kf2[1]], planner=bench0.pl)
        s2, w2, _ = bench0.audit(back, f"park_{k}_back")
        miss = 1000 * np.linalg.norm(pl.hand_pose(back["qpos"][-1], "left")[0] - pl.hand_pose(q_goal, "left")[0])
        check(f"park {k}: the parked arm's next reach starts there and arrives without contact",
              np.abs(back["arm"]["left"][0] - qpark).max() < 0.05 and miss < 2.0 and s2["ok"] and w2["ok"],
              f"arrives {miss:.2f} mm off; {s2['worst']} {back['report']['plan']['planned_legs']}"[:220])
check("parking placements tried", n_park >= 3, f"{n_park}")

# the stations moved toward the objects as place_arm_stations moves them (+0.15, +0.10 m, the braid run's first shift):
# the right arm at its HOME pose stands where the left hand must go (the braid run's parked right thumb hit the left hand
# and the right station was moved a second time). Its base link now sits 18 mm over the bench, under the 20 mm margin:
# links moved by at most the first joint count only when they penetrate the table
from tool_library.skills.place_arm_stations import place_arm_stations  # noqa: E402
far = scene_with("far_part", [], parts=[((0.0, 0.8, 0.02), (0.02, 0.02, 0.02))])
shifted = place_arm_stations(far, objects=["part0"], dx_range=[0.15, 0.15], dy_range=[0.10, 0.10], step_m=0.05, out_xml=str(TMP / "shifted.xml"))
Bs = Bench(shifted["scene"])
n_home = 0
for k, p_ in enumerate([(0.34, 0.34, 0.18), (0.37, 0.22, 0.18), (0.37, 0.26, 0.18), (0.37, 0.30, 0.18), (0.37, 0.34, 0.18)]):
    qL, err = Bs.ik("left", p_, "fingers_in")
    if err > 20.0:
        continue
    q0s = Bs.q0.copy()
    qe = Bs.pl.place(q0s, "left", qL)
    Qref, kf = Bs.reference(q0s, {"left": (q0s, qe)}, n_hold=5, n_leg=60, n_after=15)
    labels = ["left: reach for x"] * (kf[1] + 1) + ["left: descend on x"] * (len(Qref) - kf[1] - 1)
    start = {"left": Bs.S["left"].home, "right": Bs.S["right"].home}
    straight = AH.couple(Bs.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0, kf[1]])
    s0, _, _ = Bs.audit(straight, f"home_{k}_straight")
    if not ({"hand-hand", "arm-hand", "arm-arm"} & set(s0["worst"])):
        continue
    n_home += 1
    planned = AH.couple(Bs.m, Qref, FPS, arm_start=start, labels=labels, keyframe_frames=[0, kf[1]], planner=Bs.pl)
    s1, w1, _ = Bs.audit(planned, f"home_{k}_planned")
    pk = planned["report"]["plan"]["parked"]
    ok_park = bool(pk) and pk[0]["parked"] and s1["ok"] and w1["ok"]
    # and back home once the left hand is where it works: the right's own 'return home' leg from the parked configuration
    q_end = planned["qpos"][-1].copy()
    q_home = Bs.pl.place(q_end, "right", Bs.S["right"].home)
    Q2, kf2 = Bs.reference(q_end, {"right": (q_end, q_home)}, n_hold=5, n_leg=50, n_after=5)
    lab2 = ["right: return home"] * (kf2[1] + 1) + ["right: wait"] * (len(Q2) - kf2[1] - 1)
    back = AH.couple(Bs.m, Q2, FPS, arm_start={"left": planned["arm"]["left"][-1], "right": planned["arm"]["right"][-1]}, labels=lab2,
                     keyframe_frames=[0, kf2[1]], planner=Bs.pl)
    s2, w2, _ = Bs.audit(back, f"home_{k}_back")
    home_err = float(np.abs(back["arm"]["right"][-1] - Bs.S["right"].home).max())
    check(f"shifted bench {k}: the right arm at home blocks the left target ({sorted(s0['worst'])}); parked, no contact, and back home after",
          ok_park and s2["ok"] and w2["ok"] and home_err < 1e-3, f"{pk[0] if pk else None}; back {s2['worst']} home err {home_err:.4f}"[:260])
# a station shifted over the table stands on its top (place_arm_stations' table lift), a little higher than the setup's:
# two of the five placements still put the home arm in the left hand's way, enough to exercise parking and return
check("shifted-bench placements tried", n_home >= 2, f"{n_home}")

# an idle hand that holds something is not parked: a box under the resting left hand, the pads on it
qL, _ = bench0.ik("left", (-0.08, 0.50, 0.15)); qR, _ = bench0.ik("right", (0.0, 0.50, 0.14))
qs = bench0.pl.place(bench0.q0, "left", qL)
bench0.pl.data.qpos[:] = qs
mujoco.mj_kinematics(bench0.pl.model, bench0.pl.data)
lowest = min(bench0.pl.data.geom_xpos[g][2] - bench0.pl.model.geom_rbound[g] for g in range(bench0.pl.model.ngeom)
             if (bench0.pl.model.geom(g).name or "").startswith("robot_left_collision_hand_left"))
hp = bench0.pl.hand_pose(qs, "left")[0]
top = lowest - 0.004
for _ in range(3):                                    # raise the box until its top is 2-5 mm under the hand
    held_scene = scene_with("held", [((hp[0], hp[1], top / 2.0), (0.04, 0.04, top / 2.0))])
    Bh = Bench(held_scene)
    Bh.pl.setup("left", obstacles="objects", margin=0.2, check=("hand",))
    gap = Bh.pl.clearance(Bh.pl.place(Bh.q0, "left", qL))[0]
    if 0.002 <= gap <= 0.006:
        break
    top += gap - 0.004
qs = Bh.pl.place(Bh.q0, "left", qL); qe = Bh.pl.place(qs, "right", qR)
Qref, kf = Bh.reference(qs, {"right": (qs, qe)}, n_hold=5, n_leg=60, n_after=15)
labels = ["right: reach for x"] * (kf[1] + 1) + ["right: descend on x"] * (len(Qref) - kf[1] - 1)
held = AH.couple(Bh.m, Qref, FPS, arm_start={"left": qL, "right": Bh.S["right"].home}, labels=labels, keyframe_frames=[0, kf[1]], planner=Bh.pl)
pk = held["report"]["plan"]["parked"]
check("an idle hand on a part (holding or steadying it) is not parked", pk and not pk[0]["parked"] and "holds or steadies" in pk[0]["why"], str(pk))
gap = _hand_gap_failure(Bh.m, held["qpos"], held["labels"], {}, "as planned", need=-ONE_HAND_OVERLAP_M)
check("then the acting hand's pose runs into the holding hand: hand_clearance before physics, with the reason",
      gap is not None and gap["failed"] == ["hand_clearance"] and "holds or steadies" in gap["error"], (gap or {}).get("error", "")[:160])

# ------------------------------------------------------------------------------------------- 4. retiming
m, q0 = bench0.m, bench0.q0
S = bench0.S["right"]
T = 150
A = np.repeat(S.home[None], T, 0)
A[40:] += np.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])               # a one-frame step of 0.25 rad (14 deg)
u = np.clip((np.arange(T) - 80) / 11.0, 0, 1); u = 3 * u ** 2 - 2 * u ** 3
A += u[:, None] * np.array([0.0, 0.0, 0.9, 0.0, 0.4, 0.0])         # a 0.9 rad swing in 12 frames (about 4 rad/s)
Q = np.repeat(q0[None], T, 0)
for t in range(T):
    Q[t] = S.put_hand(Q[t], A[t])
labels = ["right: carry x"] * 60 + ["right: grip x"] * 15 + ["right: carry x"] * (T - 75)
keep = np.array(["grip" in l for l in labels])
E.save_episode(TMP / "fast.npz", Q, labels, FPS)
acc0, jmp0 = joint_acceleration_audit(str(SCENE), str(TMP / "fast.npz")), robot_joint_jump_audit(str(SCENE), str(TMP / "fast.npz"))
r = AP.retime_arms(m, Q, FPS, keep=keep)
E.save_episode(TMP / "fast_retimed.npz", r["qpos"], [labels[i] for i in np.floor(r["src"]).astype(int)], FPS)
acc1, jmp1 = joint_acceleration_audit(str(SCENE), str(TMP / "fast_retimed.npz")), robot_joint_jump_audit(str(SCENE), str(TMP / "fast_retimed.npz"))
check("the fast segment fails the acceleration and jump audits as it is", not acc0["ok"] and not jmp0["ok"])
check("retimed, it passes joint_acceleration_audit and robot_joint_jump_audit", acc1["ok"] and jmp1["ok"] and r["ok"],
      f"{acc1.get('issues', [])[:1]} {jmp1.get('issues', [])[:1]}")


def t_min(D, vmax, amax):
    """Shortest rest-to-rest time for a distance D under a speed and an acceleration limit (bang-bang / trapezoid)."""
    return 2 * np.sqrt(D / amax) if D < vmax ** 2 / amax else D / vmax + vmax / amax


vmax, amax, _ = AP.arm_limits()
least = max(0.0, t_min(0.25, vmax, amax) - 1 / FPS) + max(0.0, t_min(0.9, vmax, amax) - 11 / FPS)   # over the motions' own durations
uniform = (T - 1) / FPS * (np.abs(np.diff(A, axis=0)).max() * FPS / vmax - 1.0)
check("with little added time: under 2x the least the two motions need, far under a uniform slow-down",
      r["added_s"] <= 2.0 * least and r["added_s"] < 0.1 * uniform, f"added {r['added_s']} s, least {least:.2f} s, uniform {uniform:.1f} s")
ks = np.where(keep[np.floor(r["src"]).astype(int)])[0]
steps = np.diff(r["src"][ks])
check("the contact phase keeps its pace (one source frame per frame)", len(ks) >= 14 and np.allclose(steps[steps > 0], 1.0, atol=1e-6), f"{np.unique(np.round(steps, 3))}")
# Stage 4's joint-speed bound on a warped layout (each chain at most 1.5 x the original's peak + 0.5 rad/s, the check that
# rejected meal-box layouts): the retimed demonstration warped to run twice as fast through its swing passes it
Q0r = r["qpos"]
arm_cols = E.robot_parts(m)["right"]["arm_qadr"]
peak = lambda X: float(np.abs(np.diff(X[:, arm_cols], axis=0)).max() * FPS)
bound = 1.5 * peak(Q0r) + 0.5
src_w = np.concatenate([np.arange(0, 100, 1.0), np.arange(100, 160, 2.6), np.arange(160, len(Q0r), 1.0)])
Qw = AP.resample(Q0r, src_w[src_w <= len(Q0r) - 1], AP._quat_cols(m))
rw = AP.retime_arms(m, Qw, FPS, vmax_rad_s={"right": bound, "left": bound}, flange_amax_m_s2=0)
check("a warped layout over Stage 4's joint-speed bound is retimed under it", peak(Qw) > bound and peak(rw["qpos"]) <= bound + 1e-6 and rw["ok"],
      f"peak {peak(Qw):.2f} -> {peak(rw['qpos']):.2f} rad/s, bound {bound:.2f}, added {rw['added_s']} s")
# a trajectory already within the limits is returned unchanged
r0 = AP.retime_arms(m, np.repeat(q0[None], 40, 0), FPS)
check("a trajectory within the limits is not retimed", r0["added_frames"] == 0 and len(r0["qpos"]) == 40)

print(f"\n{time.time() - T0:.0f} s")
print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
