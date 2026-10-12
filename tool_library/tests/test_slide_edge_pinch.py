"""slide + edge_pinch on the setup bench: a flat box too low to pinch where it lies (a phone) is slid toward the operator
until it overhangs the table's near edge, then pinched from below (fingers under the overhang, deeper than the thumb on
its top) and lifted, all through execute_subgoal_plan in contact physics. Cases vary the thickness (6, 9, 12 mm), the
footprint, the yaw and the start distance from the edge: thin_grasp_diagnosis names the plan and the overhang, the rake
pose (slide_contact_pose: upright fingertips behind the box) pushes it, the overhang pinch (overhang_pinch_pose,
solved for where the slide will leave it, stress-screened in the hand scene with the finite table) re-anchors to where
it really ended. Checks: the slide moved it to the overhang without tipping it or a finger pressing into the table;
the pinch lifts it at least 5 cm, in hand, without slip, its hand never more than 2 mm into the table; the chosen pose
passes grasp_lift_test and grasp_stress_test in its hand scene; the planner refuses an edge_pinch on a box that does not
overhang enough (naming the slide), and a slide that would tip the box off. The 9 mm episode also passes the
contact audits (episode_physics_check, table_workspace_bounds_audit, robot_self_collision_check). About 10 minutes
(the cases run in parallel).

    MUJOCO_GL=egl python tool_library/tests/test_slide_edge_pinch.py
"""
import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="slide_edge_", dir=os.environ.get("TMPDIR"))) if __name__ == "__main__" else None
FAILS = 0
# (tag, thickness, length, width, yaw_deg, x, start gap of the near end inside the edge, the pinching hand)
# x >= 0.07: the right arm reaching nearer the middle of the bench sweeps its links through the idle left arm
CASES = [("t9", 0.009, 0.148, 0.072, 0.0, 0.09, 0.006, "right"),
         ("t6", 0.006, 0.140, 0.068, 8.0, 0.07, 0.012, "right"),
         ("t12", 0.012, 0.150, 0.070, -6.0, 0.10, 0.002, "right")]


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def build(out, tag, T, L, Wd, yaw, x, gap):
    from tool_library.skills.stage_on_bench import stage_on_bench
    out.mkdir(parents=True, exist_ok=True)
    xml = out / f"{tag}.xml"
    xml.write_text(f'<mujoco model="{tag}"><worldbody><body name="b"><freejoint/><geom name="g" type="box" size="{Wd / 2} {L / 2} {T / 2}" '
                   f'pos="0 0 {T / 2}" mass="0.17" friction="0.9 0.01 0.001" rgba="0.25 0.25 0.3 1"/></body></worldbody></mujoco>')
    ws = out / "stage2"; ws.mkdir(parents=True, exist_ok=True)
    # the near end `gap` inside the table's near edge (y = 0.05), measured on the yawed footprint
    c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    reach_y = 0.5 * (abs(s_) * Wd + abs(c) * L)
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": tag, "xml": str(xml), "x": x, "y": 0.05 + gap + reach_y, "yaw_deg": yaw}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [{"id": tag, "type": "rigid", "asset_xml": str(xml),
                                                                                            "root_body": f"{tag}_b", "bodies": [f"{tag}_b"]}]}))
    return ws


def run_case(args):
    tmp, (tag, T, L, Wd, yaw, x, gap, side) = args
    sys.path.insert(0, str(ROOT))
    from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
    from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
    from tool_library.skills.overhang_pinch_pose import overhang_pinch_pose
    from tool_library.skills.slide_contact_pose import slide_contact_pose
    from tool_library.skills.thin_grasp_diagnosis import thin_grasp_diagnosis
    from tool_library.verifiers.grasp_lift_test import grasp_lift_test
    from tool_library.verifiers.grasp_stress_test import grasp_stress_test
    out = Path(tmp) / tag
    ws = build(out, tag, T, L, Wd, yaw, x, gap)
    scene, body = str(ws / "scene.xml"), f"{tag}_b"
    res = {"tag": tag, "T": T, "yaw": yaw}
    diag = thin_grasp_diagnosis(scene, body)
    res["diag"] = {"as_lies": diag["graspable_as_lies"], "plan": [r["plan"] for r in diag["recommend"]], "slide": diag["verdicts"]["slide"],
                   "edge": diag["verdicts"]["edge_pinch"]}
    target = diag["verdicts"]["edge_pinch"]["need_overhang_m"] + 0.012
    hsR = hand_scene_from_recon(str(ws / "scene_recon.json"), "right", tag, out_json=str(out / "hs_right.json"))
    sp = slide_contact_pose(hsR, [0, -1])
    hsP = hand_scene_from_recon(str(ws / "scene_recon.json"), side, tag, out_json=str(out / f"hs_{side}.json"))
    op = overhang_pinch_pose(hsP, [0, -1], overhang_m=target, out_json=str(out / "overhang.json"))
    res["pinch"] = {"ok": op["ok"], "why": op.get("why"), "n": len(op["poses"]), "dropped": op.get("dropped")}
    if not op["ok"]:
        return res
    plan = {"subgoals": [{"index": 1, "motion": "slide", "side": "right", "object": body, "pose_candidates": sp["poses"], "direction": [0, -1],
                          "overhang_m": target},
                         {"index": 2, "motion": "edge_pinch", "side": side, "object": body, "pose_candidates": op["poses"], "lift_m": 0.08}]}
    (out / "plan.json").write_text(json.dumps(plan, default=float))
    summ = execute_subgoal_plan(scene, str(out / "plan.json"), str(out / "ep"))
    outs = json.loads((out / "ep" / "outcomes.json").read_text())
    res["ok"] = summ["ok"]
    res["outcomes"] = [{"motion": o["motion"], "ok": o["ok"], "failed": o.get("failed"), "variant": o.get("variant"),
                        "m": {k: (o.get("measured") or {}).get(k) for k in ("displacement_along_m", "overhang_m", "com_inside_table_m", "rise_m",
                                                                             "max_table_pen_mm", "slip_in_hand_m", "hand_links_touching", "object_tilt_end_deg")},
                        "error": (o.get("error") or "")[:200]} for o in outs]
    used = next((o for o in outs if o["motion"] == "edge_pinch"), {})
    if used.get("ok"):
        k = int(str(used.get("variant", "pose candidate 1")).split()[-1]) - 1
        pose = op["poses"][k]
        lt = grasp_lift_test(op["hand_scene"], pose, lift_heights_m=[0.04, 0.10, 0.15])
        st = grasp_stress_test(op["hand_scene"], pose)
        res["stress"] = {"lift": lt["ok"], "lift_fail": lt["first_fail"], "stress": st["ok"], "stress_fail": st["first_fail"]}
        res["episode"] = str(out / "ep")
        res["scene_executed"] = str(out / "ep" / "scene_physics.xml")
    return res


def main():
    from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(2, len(CASES))) as pool:     # two physics processes at a time (a shared node)
        results = pool.map(run_case, [(str(TMP), c) for c in CASES])
    for r in results:
        tag = r["tag"]
        d = r["diag"]
        thin = r["T"] < 0.010
        check(f"[{tag}] the diagnosis names slide + edge_pinch (a {1000 * r['T']:.0f} mm box{' is pinchable as it lies too' if not thin else ''})",
              (["slide", "edge_pinch"] in d["plan"]) and (d["as_lies"] != thin), f"{d['plan']} {d['slide']['why']}")
        check(f"[{tag}] the overhang pinch has stress-screened candidates", r["pinch"]["ok"], json.dumps(r["pinch"])[:300])
        if not r["pinch"]["ok"]:
            continue
        sl = next((o for o in r["outcomes"] if o["motion"] == "slide"), None)
        ep = next((o for o in r["outcomes"] if o["motion"] == "edge_pinch"), None)
        check(f"[{tag}] the slide pushed it over the edge (yaw {r['yaw']:.0f} deg), upright, no finger in the table",
              sl is not None and sl["ok"] and sl["m"]["overhang_m"] >= d["edge"]["need_overhang_m"] and sl["m"]["max_table_pen_mm"] <= 2.0,
              json.dumps(sl)[:400])
        check(f"[{tag}] the edge pinch lifts it >= 5 cm in hand without slip, no finger in the table",
              ep is not None and ep["ok"] and ep["m"]["rise_m"] >= 0.05 and ep["m"]["max_table_pen_mm"] <= 2.0, json.dumps(ep)[:400])
        if r.get("stress"):
            check(f"[{tag}] the executed pinch passes grasp_lift_test and grasp_stress_test in its hand scene",
                  r["stress"]["lift"] and r["stress"]["stress"], json.dumps(r["stress"]))
    # audits on the 9 mm episode
    r9 = next((r for r in results if r["tag"] == "t9" and r.get("episode")), None)
    if r9:
        from tool_library.verifiers.episode_physics_check import episode_physics_check
        from tool_library.verifiers.robot_self_collision_check import robot_self_collision_check
        from tool_library.verifiers.table_workspace_bounds_audit import table_workspace_bounds_audit
        sc, tj = r9["scene_executed"], str(Path(r9["episode"]) / "trajectory.npz")
        a = {"episode_physics_check": episode_physics_check(sc, tj), "robot_self_collision_check": robot_self_collision_check(sc, tj),
             "table_workspace_bounds_audit": table_workspace_bounds_audit(sc, tj)}
        check("the 9 mm episode passes the contact audits", all(v["ok"] for v in a.values()),
              json.dumps({k: v.get("issues", [])[:2] for k, v in a.items()})[:500])
    # what the planner refuses: an edge pinch on a box that does not overhang enough, a slide that tips it off
    out = TMP / "refuse"
    ws = build(out, "rf", 0.009, 0.148, 0.072, 0.0, 0.05, 0.02)
    scene = str(ws / "scene.xml")
    candidates = json.loads((TMP / "t9" / "overhang.json").read_text()).get("poses", []) if (TMP / "t9" / "overhang.json").exists() else []
    pose = candidates[0] if candidates else None
    errs = {}
    if pose is not None:
        try:
            plan_subgoal_keyframes(scene, {"index": 1, "motion": "edge_pinch", "side": "right", "object": "rf_b", "pose": pose}, auto_via=False)
            errs["edge"] = None
        except ValueError as e:
            errs["edge"] = str(e)
    try:
        plan_subgoal_keyframes(scene, {"index": 1, "motion": "slide", "side": "right", "object": "rf_b", "pose": pose or {}, "direction": [0, -1],
                                       "overhang_m": 0.12}, auto_via=False)
        errs["tip"] = None
    except ValueError as e:
        errs["tip"] = str(e)
    check("an edge_pinch on a box that does not overhang is refused with the slide that fixes it",
          bool(errs.get("edge")) and "slide it first" in errs["edge"], str(errs.get("edge"))[:200])
    check("a slide that would leave the centre of mass past the edge is refused", bool(errs.get("tip")) and "tip off" in errs["tip"], str(errs.get("tip"))[:200])
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
