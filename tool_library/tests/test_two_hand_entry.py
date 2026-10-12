"""Two hands acting on two different parts at once (plan entry {motion: 'both', hands: {left, right}, sync}). Before,
one hand moved per subgoal, so a human's simultaneous two-hand action could not be expressed. Cases on the setup bench
with two bars in a row, 6 cm apart: both hands lift their own bar in ONE rollout (sync 'start'; the bars rise in the same frames;
each hand's expectations, both arms planned with rigidly attached hands and bounded motor torques); then they pull the
two bars apart (sync 'end'); the Stage 4 task folder written from the episode passes stage4_handoff_check (each hand
anchored to its own bar); the planner refuses both hands on one part and unequal keyframe counts under sync
'keyframes'; the plan gate refuses hands planned through each other (hand_clearance, before physics); and the outcome
rejects a trajectory where the hands touch (hands_apart) or one hand's bar misses its target. About a minute.

    MUJOCO_GL=egl python tool_library/tests/test_two_hand_entry.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import episode as E  # noqa: E402
from tool_library.common import outcome as O  # noqa: E402
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan  # noqa: E402
from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose  # noqa: E402
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.skills.write_stage4_config import write_stage4_config  # noqa: E402
from tool_library.verifiers.stage4_handoff_check import stage4_handoff_check  # noqa: E402
from tool_library.verifiers.subgoal_outcome_check import subgoal_outcome_check  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_twohand_", dir=os.environ.get("TMPDIR")))
FAILS = 0
HX, HY, HZ = 0.05, 0.015, 0.04          # each bar 10 x 3 x 8 cm, in a row along x with a 6 cm gap centred on x = 0.01
CX, CY = 0.01, 0.46
GAP = 0.06
XA, XB = CX - HX - GAP / 2, CX + HX + GAP / 2   # the bars' centres


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def bar(name, rgba):
    d = TMP / "assets" / name
    d.mkdir(parents=True)
    (d / f"{name}.xml").write_text(f'<mujoco model="{name}"><worldbody><body name="{name}" pos="0 0 {HZ}"><freejoint/>'
                                   f'<geom name="{name}_g" type="box" size="{HX} {HY} {HZ}" rgba="{rgba}" mass="0.08" friction="1.0 0.01 0.001"/>'
                                   f'</body></worldbody></mujoco>')
    return str(d / f"{name}.xml")


def make_scene():
    ws = TMP / "stage2"; ws.mkdir()
    xa, xb = bar("bara", "0.8 0.2 0.2 1"), bar("barb", "0.2 0.3 0.8 1")
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "bara", "xml": xa, "x": XA, "y": CY}, {"name": "barb", "xml": xb, "x": XB, "y": CY}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    cam = {"width": 640, "height": 480, "fx": 700.0, "fy": 700.0, "cx": 320.0, "cy": 240.0, "fovy_deg": 38.0,
           "pos": [0.0188, -0.1829, 0.5578], "quat_wxyz": [0.881567, 0.470897, 0.0249467, 0.021763]}
    (ws / "camera.json").write_text(json.dumps(cam))
    man = {"scene_xml": "scene.xml", "camera_json": "camera.json",
           "objects": [{"id": n, "type": "rigid", "asset_xml": x, "root_body": f"{n}_{n}", "bodies": [f"{n}_{n}"]} for n, x in (("bara", xa), ("barb", xb))]}
    (ws / "scene_recon.json").write_text(json.dumps(man))
    import imageio.v2 as iio
    w = iio.get_writer(str(TMP / "human.mp4"), fps=30)
    for _ in range(10):
        w.append_data(np.full((480, 640, 3), 120, np.uint8))
    w.close()
    return ws


def outer_grasp(ws, side, obj):
    """Use the same general grasp search offered to a task agent, including
    the actual station arm and both lift and independent hold tests."""
    hs = hand_scene_from_recon(str(ws / "scene_recon.json"), side, obj, out_json=str(TMP / f"hs_{side}.json"))
    result = optimize_grasp_pose(hs, family="rigid", bench_scene=str(ws / "scene.xml"),
                                 arm_waypoints=[[0, 0, .06]], workers=2, max_evals=96, iterations=6,
                                 out_json=str(TMP / f"pose_{side}.json"))
    assert result["ok"], f"no physically verified {side} grasp: {result}"
    return str(TMP / f"pose_{side}.json"), str(TMP / f"hs_{side}.json")


def main():
    ws = make_scene()
    scene = str(ws / "scene.xml")
    A, B = "bara_bara", "barb_barb"
    (pl_, hl), (pr_, hr) = outer_grasp(ws, "left", "bara"), outer_grasp(ws, "right", "barb")
    lift = {"index": 1, "motion": "both", "sync": "start", "hands": {
        "left": {"motion": "grasp_lift", "object": A, "pose": pl_, "hand_scene": hl, "lift_m": 0.06},
        "right": {"motion": "grasp_lift", "object": B, "pose": pr_, "hand_scene": hr, "lift_m": 0.06}}}
    apart = {"index": 2, "motion": "both", "sync": "end", "hands": {
        "left": {"motion": "carry", "object": A, "target_pos": [XA - 0.08, CY, HZ + 0.06]},
        "right": {"motion": "carry", "object": B, "target_pos": [XB + 0.08, CY, HZ + 0.06], "duration_s": 2.0}}}
    # 1. the planner: one merged time line, two-hand labels, a phase per hand, each hand's expectations
    pl = plan_subgoal_keyframes(scene, lift, auto_via=False)
    labs = [k["label"] for k in pl["keyframes"]]
    check("a two-hand plan joins both hands' phases in one label", all(" | " in l for l in labs) and pl["expect"]["hands"]["left"]["object"] == A
          and pl["expect"]["hands"]["right"]["object"] == B and {p_["hand"] for p_ in pl["phases"]} == {"left", "right"}
          and A not in pl["expect"]["untouched"] and B not in pl["expect"]["untouched"], labs[1])
    # 2. executed: both bars lifted in ONE rollout, then pulled apart
    s = execute_subgoal_plan(scene, {"scene_recon_json": str(ws / "scene_recon.json"), "subgoals": [lift, apart]}, str(TMP / "episode"))
    check("both hands lift their own bar at once and pull the bars apart", s["ok"],
          str([(r["index"], r["ok"], r.get("failed"), r.get("error")) for r in s["subgoals"]]))
    if not s["ok"]:
        print(json.dumps(s.get("first_failure"), indent=1)[:3500])
        return 1
    ep = E.load_episode(TMP / "episode" / "trajectory.npz")
    runtime_scene = str(TMP / "episode" / "scene_physics.xml")
    m = mujoco.MjModel.from_xml_path(runtime_scene)
    za, zb = E.body_track(m, ep["qpos"], A)[0], E.body_track(m, ep["qpos"], B)[0]
    seg1 = np.nonzero(np.asarray(ep["subgoal"]) == 1)[0]
    ra = seg1[np.nonzero(za[seg1, 2] - za[0, 2] > 0.01)[0]]; rb = seg1[np.nonzero(zb[seg1, 2] - zb[0, 2] > 0.01)[0]]
    overlap = len(set(ra.tolist()) & set(rb.tolist()))
    check("the two bars rise during the same frames (one rollout, not one hand after the other)", len(ra) > 10 and len(rb) > 10 and overlap >= 0.8 * min(len(ra), len(rb)),
          f"rising frames {len(ra)} / {len(rb)}, common {overlap}")
    gap0, gap1 = abs(za[0, 0] - zb[0, 0]), abs(za[-1, 0] - zb[-1, 0])
    check("the bars end pulled apart", gap1 - gap0 > 0.12, f"centre gap {gap0:.3f} -> {gap1:.3f} m")
    outc = json.loads((TMP / "episode" / "outcomes.json").read_text())
    arm = outc[0]["measured"]["arm"]["runtime"]
    check("both arms carry their rigidly attached hands throughout physical execution", set(arm) == {"left", "right"}
          and all(v["residual_mm_max"] <= 2.0 for v in arm.values()), str({k: v["residual_mm_max"] for k, v in arm.items()}))
    check("each hand's checks are named per hand and the hands never touched", any(k.startswith("left:") for k in outc[0]["checks"])
          and any(k.startswith("right:") for k in outc[0]["checks"]) and outc[0]["checks"].get("hands_apart") is True, str(sorted(outc[0]["checks"])[:6]))
    # the delivered trajectory re-checked by the verifier the finish harness runs
    oc = subgoal_outcome_check(runtime_scene, str(TMP / "episode" / "trajectory.npz"), outc[1]["expect"], subgoal=2)
    check("subgoal_outcome_check re-checks a two-hand entry on the delivered trajectory", oc["ok"], str(oc["failed"]))
    # 3. Stage 4: each hand anchored to its own bar, no jump between anchors
    cfg = write_stage4_config(str(ws / "scene_recon.json"), str(TMP / "episode"), str(TMP / "stage4_task"), human_video=str(TMP / "human.mp4"), name="twohand")
    ho = stage4_handoff_check(str(TMP / "stage4_task"), camera_json=str(ws / "camera.json"))
    blends = {s_: [seg["blend"] for seg in cfg["segments"][s_] if "blend" in seg] for s_ in ("left", "right")}
    check("the Stage 4 task folder passes the handoff check, each hand blending to its own bar", ho["ok"] and ["world", A] in blends["left"]
          and ["world", B] in blends["right"] and cfg["objects"][A].get("carrier") == "left" and cfg["objects"][B].get("carrier") == "right",
          str(ho["issues"][:2]) + " " + str(blends))
    # 4. what the planner refuses
    try:
        plan_subgoal_keyframes(scene, {"index": 3, "motion": "both", "hands": {"left": dict(lift["hands"]["left"]), "right": dict(lift["hands"]["left"], pose=pr_)}}, auto_via=False)
        same = None
    except ValueError as e:
        same = str(e)
    check("both hands on one part is refused (a hold + the acting hand couples them)", same is not None and "same part" in same, (same or "")[:90])
    try:
        plan_subgoal_keyframes(scene, dict(lift, sync="keyframes", hands={"left": lift["hands"]["left"], "right": {"motion": "wait", "duration_s": 0.5}}), auto_via=False)
        kf = None
    except ValueError as e:
        kf = str(e)
    check("sync 'keyframes' with unequal keyframe counts is refused", kf is not None and "same number of keyframes" in kf, (kf or "")[:90])
    # 5. the plan gate: after the lift, the right hand carries its bar to where the left hand holds its own -> hand_clearance,
    # before any physics (a dry run: planning and the arm checks only)
    clash = {"index": 2, "motion": "both", "hands": {
        "left": {"motion": "carry", "object": A, "target_pos": [XA, CY, HZ + 0.06]},
        "right": {"motion": "carry", "object": B, "target_pos": [XA - 0.08, CY, HZ + 0.06]}}}
    sd = execute_subgoal_plan(scene, {"subgoals": [lift, clash]}, str(TMP / "clash"), dry_run=True)
    r1 = sd["subgoals"][-1]
    check("hands planned through each other fail as hand_clearance before any physics", not sd["ok"] and sd["subgoals"][0]["ok"]
          and r1["failed"] == ["hand_clearance"] and not (TMP / "clash" / "trajectory.npz").exists(), str(r1.get("error"))[:140])
    # 6. the outcome rejects hands that touch and a hand whose part misses its target
    idx1 = list(np.nonzero(np.asarray(ep["subgoal"]) == 1)[0])
    proposal = E.build_execution_model(scene)[0]
    recording = TMP / "episode" / "trajectory.npz"
    Q = np.array([E.state_qpos(proposal, recording, frame=i) for i in (idx1[0], idx1[-1])])
    idx = [0, 1]
    rig_l, rig_r = E.robot_parts(proposal)["left"]["rig"], E.robot_parts(proposal)["right"]["rig"]
    t = 1
    for k in ("x", "y", "z", "rx", "ry", "rz"):                # the left hand put where the right hand is
        Q[t, rig_l.wrist_qpos[k]] = Q[t, rig_r.wrist_qpos[k]] + (0.0 if k != "x" else -0.02)
    exp = dict(outc[0]["expect"])
    ev = O.evaluate(proposal, Q, 30.0, idx, json.loads(json.dumps(exp)))
    check("the outcome rejects hands that touch each other (hands_apart)", ev["checks"].get("hands_apart") is False, str(ev["measured"].get("hands_touching_first")))
    exp2 = json.loads(json.dumps(outc[1]["expect"]))
    exp2["hands"]["left"]["target_pos"] = [XA - 0.20, CY, HZ + 0.06]
    idx2 = list(np.nonzero(np.asarray(ep["subgoal"]) == 2)[0])
    ev2 = O.evaluate(m, ep["qpos"], 30.0, idx2, exp2)
    check("the outcome names the hand whose part missed its target", "left:at_target" in ev2["failed"] and ev2["checks"].get("right:at_target") is True, str(ev2["failed"]))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
