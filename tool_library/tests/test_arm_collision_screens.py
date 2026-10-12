"""The arm's links in the reach tools: optimize_pose_for_reach and arm_reach_check(poses=...).

Measured failures: optimize_pose_for_reach called a cream-brick grasp reachable and the arm's link 3 then hit the tower
in the executor; arm_reach_check(poses=[...]) crashed on pose json paths ("string indices must be integers") and passed
a pose whose arm ran a link 8-10 mm into its own palm (caught only by the executor's dry run). These cases put a box
where the arm's links are (at the grasp, or only on the way in at the pre-grasp) and grow the hand's collision shape
into its own arm, on the setup bench, and check both tools report it, and accept the same poses without it.

    python tool_library/tests/test_arm_collision_screens.py
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.common import plan as P  # noqa: E402
from tool_library.skills.optimize_pose_for_reach import optimize_pose_for_reach  # noqa: E402
from tool_library.verifiers.arm_reach_check import arm_reach_check  # noqa: E402
from tool_library.verifiers.arm_workspace_map import named_orientation  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_armscreen_", dir=os.environ.get("TMPDIR")))
SCENE = ROOT / "setup" / "scene.xml"
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


def scene_with(name, edit):
    sp = absolute_assets(mujoco.MjSpec.from_file(str(SCENE)), SCENE)
    edit(sp)
    p = TMP / f"{name}.xml"; p.write_text(sp.to_xml())
    return str(p)


def add_box(pos, half=0.025, name="obstacle"):
    def edit(sp):
        b = sp.worldbody.add_body(); b.name = name; b.pos = [float(v) for v in pos]
        g = b.add_geom(); g.name = f"{name}_box"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [half] * 3
        g.contype = 1; g.conaffinity = 1; g.rgba = [0.8, 0.2, 0.2, 1]
    return edit


model = mujoco.MjModel.from_xml_path(str(SCENE))
rig = P.scene_rigs(model)["right"]
eul = H.mat_to_euler(named_orientation("right", "palm_down")).tolist()
POSE = {"side": "right", "pos": [0.2, 0.55, 0.15], "euler": eul, "finger_targets": rig.finger_targets(0.3),
        "grip_fingers": ["thumb", "index"], "approach": [0.0, 0.0, 1.0]}
KW = dict(axis="vertical", range_deg=0.0, step_deg=10.0, approach_m=0.08)


def link_pos(qa, link, pose=POSE):
    """World position of an arm body for the right arm at qa with the hand at `pose`."""
    from tool_library.common import episode as E
    pt = E.robot_parts(model)["right"]
    d = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, d, 0)
    rig.teleport(d, pose["pos"], pose["euler"]); d.qpos[pt["arm_qadr"]] = qa
    mujoco.mj_kinematics(model, d)
    return d.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_right_arm_{link}")].copy()


# --- optimize_pose_for_reach -----------------------------------------------------------------------------------
clean = optimize_pose_for_reach(str(SCENE), "right", POSE, **KW)
check("optimize_pose_for_reach: the clean pose is reachable and collision free", clean["original_reach"]["ok"] and clean["original_reach"]["arm_hit"] is None
      and "note" in clean, str(clean["original_reach"]))

# a pose that reaches unturned stays unturned: swept over +-90 deg, 0 and +30 deg both reach with a 0.0 mm residual
# (measured: the turn chosen was +30 deg, the lowest score by micrometres)
sw = optimize_pose_for_reach(str(SCENE), "right", POSE, axis="vertical", range_deg=90.0, step_deg=30.0, approach_m=0.08)
zero_ok = [c for c in sw["candidates"] if c["angles"][0] == 0.0 and c["ok"]]
check("optimize_pose_for_reach keeps the smallest turn among equally good candidates", zero_ok
      and all(abs(v) < 1e-9 for v in sw["rotation_deg"].values()) and sw["reach"]["ok"],
      f"rotation {sw['rotation_deg']}, ok candidates {[c['angles'] for c in sw['candidates'] if c['ok']]}")
check("a zero-turn exact witness avoids redundant orientation searches", sw["n_evaluated"] == 1 and not sw["budget_reached"])
sw2 = optimize_pose_for_reach(str(SCENE), "right", POSE, axis="vertical", range_deg=50.0, step_deg=20.0, approach_m=0.08)
check("the unturned pose is a candidate even when the grid steps over it", any(c["angles"][0] == 0.0 for c in sw2["candidates"])
      and all(abs(v) < 1e-9 for v in sw2["rotation_deg"].values()), f"rotation {sw2['rotation_deg']}")

from tool_library.skills.optimize_pose_for_reach import _reach_solver  # noqa: E402
solve = _reach_solver(model, "right")
R0 = H.euler_to_mat(eul)
qa_grasp = solve(POSE["pos"], R0)[2]
qa_pre = solve(np.asarray(POSE["pos"]) + [0, 0, 0.08], R0, qa_grasp)[2]
elbow = link_pos(qa_grasp, "link3")
blocked = scene_with("elbow_box", add_box(elbow))
r = optimize_pose_for_reach(blocked, "right", POSE, **KW)
limited = optimize_pose_for_reach(blocked, "right", POSE, axis="vertical", range_deg=90., time_budget_s=.000001)
check("a timed-out collision search cannot certify a partial candidate", limited["budget_reached"] and not limited["reach"]["ok"]
      and limited["search_status"] == "inconclusive" and limited["n_evaluated"] == 1)
hit = r["original_reach"]["arm_hit"] or {}
check("a box at the elbow (30 cm from the hand): the pose is infeasible and the arm hit is named",
      not r["original_reach"]["ok"] and "obstacle_box" in " ".join(hit.get("geoms", [])) and r["original_reach"]["overlap_mm"] <= 3.0,
      f"{r['original_reach']['overlap_with']}")

pre_link = link_pos(qa_pre, "link5", dict(POSE, pos=(np.asarray(POSE["pos"]) + [0, 0, 0.08]).tolist()))
pre_box = scene_with("pregrasp_box", add_box(pre_link + [0.0, 0.0, 0.035], half=0.02))
r_pre = optimize_pose_for_reach(pre_box, "right", POSE, **KW)
r_pre_off = optimize_pose_for_reach(pre_box, "right", POSE, **dict(KW, approach_m=0.0))
hit = r_pre["original_reach"]["arm_hit"] or {}
check("a box only on the way in: caught at the pre-grasp, and only by the approach screen",
      not r_pre["original_reach"]["ok"] and hit.get("where") in ("pre-grasp", "approach") and r_pre_off["original_reach"]["ok"],
      f"{r_pre['original_reach']['overlap_with']} | without approach: ok={r_pre_off['original_reach']['ok']}")

# --- arm_reach_check(poses=...) --------------------------------------------------------------------------------
pf = TMP / "pose.json"; pf.write_text(json.dumps(POSE))
lf = TMP / "grasps.json"; lf.write_text(json.dumps([POSE, dict(POSE, pos=[0.2, 1.3, 0.15])]))
try:
    rr = arm_reach_check(str(SCENE), poses=[str(pf), POSE, str(lf)], side="right")
    crash = None
except Exception as e:                      # the measured failure: TypeError: string indices must be integers
    rr, crash = None, f"{type(e).__name__}: {e}"
check("arm_reach_check takes pose json paths, dicts and a list file", crash is None and len(rr["poses"]) == 4
      and [p["ok"] for p in rr["poses"]] == [True, True, True, False] and rr["poses"][0]["source"] == str(pf), crash or str(rr["issues"][:1]))
check("arm_reach_check: the clean pose's arm hits nothing", rr is not None and rr["poses"][0]["arm_hits"] == [] and rr["poses"][0]["arm_collision_checked"])
try:
    arm_reach_check(str(SCENE), poses=[{"pos": [0, 0, 0]}], side="right"); misuse = None
except ValueError as e:
    misuse = str(e)
check("a pose without euler is a ValueError naming the parameter", misuse is not None and "poses" in misuse, str(misuse))

# the hand's collision shape grown into its own arm: a sphere on the palm body around link 4 of the solved arm
qa = np.asarray(rr["poses"][0]["arm_qpos"], float)
l4 = link_pos(qa, "link4")


def grow_palm(sp):
    m = sp.compile(); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
    rg = P.scene_rigs(m)["right"]; rg.teleport(d, POSE["pos"], POSE["euler"]); mujoco.mj_kinematics(m, d)
    palm = m.body(rg.palm_body)
    Rp = d.xmat[rg.palm_body].reshape(3, 3); pp = d.xpos[rg.palm_body]
    g = sp.body(palm.name).add_geom(); g.name = "robot_right_collision_hand_right_probe_0"; g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size = [0.03, 0, 0]; g.pos = (Rp.T @ (l4 - pp)).tolist(); g.contype = 0; g.conaffinity = 0; g.mass = 0.0; g.group = 3


own = scene_with("own_hand", grow_palm)
ro = arm_reach_check(own, poses=[POSE], side="right")
row = ro["poses"][0]
check("arm_reach_check: an arm link inside its own hand fails the pose and says 'own hand'",
      not ro["ok"] and row["reach_ok"] and row["own_hand_mm"] > 3.0 and any(h["what"] == "own hand" for h in row["arm_hits"])
      and any("own hand" in i for i in ro["issues"]), f"{row['arm_hits'][:1]} | {ro['issues'][:1]}")
# the scene state: a box moved onto the arm in the "episode so far" is seen only when state is given
def add_mover(sp):
    b = sp.worldbody.add_body(); b.name = "mover"; b.pos = [-0.4, 0.8, 0.025]       # parked far away in the keyframe
    b.add_freejoint()
    g = b.add_geom(); g.name = "mover_box"; g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.025] * 3; g.mass = 0.05
    for k in sp.keys:                                                                 # the new free joint is last in qpos
        k.qpos = list(k.qpos) + [-0.4, 0.8, 0.025, 1.0, 0.0, 0.0, 0.0]


free = scene_with("free_box", add_mover)
m_box = mujoco.MjModel.from_xml_path(free)
qs = np.array(m_box.key_qpos[0], float)
adr = int(m_box.jnt_qposadr[int(m_box.body("mover").jntadr[0])])
qs[adr:adr + 3] = l4
r_key = arm_reach_check(free, poses=[POSE], side="right")
r_state = arm_reach_check(free, poses=[POSE], side="right", state=qs.tolist())
check("state: an object moved onto the arm since the keyframe is caught only with the episode's state",
      not any("mover" in " ".join(h["geoms"]) for h in r_key["poses"][0]["arm_hits"])
      and any("mover" in " ".join(h["geoms"]) for h in r_state["poses"][0]["arm_hits"]) and "given state" in " ".join(r_state["issues"]),
      f"{r_state['poses'][0]['arm_hits'][:1]}")
ro_off = arm_reach_check(own, poses=[POSE], side="right", arm_collision=False)
check("arm_collision=False keeps the former reach-only verdict", ro_off["ok"] and not ro_off["poses"][0]["arm_collision_checked"])

print(f"{'ALL PASS' if not FAILS else f'{FAILS} FAILED'} ({time.time() - T0:.0f}s)")
sys.exit(1 if FAILS else 0)
