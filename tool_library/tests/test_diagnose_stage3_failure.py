"""diagnose_stage3_failure: failure classes and ranked repairs from stored failed Stage 3 attempts.

Fixtures (tests/fixtures/diagnose_stage3_failure/, from two released failed runs, trimmed): the unbraid task's
gate_suite_report.json, its blocked failure_report.json and last finish verdict, two of its executor outcomes (a pinch
that did not lift and slipped; a set-down that tracked badly and raced the arm), the charge task's blocked
failure_report.json (a 9 mm phone and a Stage 2 cable asset without its plug), and a dry run of the magnetic-rings task
(an arm into the other hand). Synthetic bench scenes check the two MEASURED classes: a 9 mm plate next to a 6 cm box
(thin object), and a cable whose bench body carries a plug the hand scene's asset lacks (upstream: Stage 2).

    MUJOCO_GL=egl python tool_library/tests/test_diagnose_stage3_failure.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.skills.diagnose_stage3_failure import diagnose_stage3_failure  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "diagnose_stage3_failure"
TMP = Path(tempfile.mkdtemp(prefix="dexagent_diag_", dir=os.environ.get("TMPDIR")))
FAILS = []
INDEX = {t["name"] for t in json.loads((ROOT / "tool_library" / "index.json").read_text())["tools"]}


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def classes(r):
    return [c["class"] for c in r["classes"]]


def tools_known(r):
    return all(rp["tool"] in INDEX or rp["tool"] in ("report_blocked", "optimize_grasp_pose", "human_grasp_seeds") for rp in r["repairs"])


def bench(name, assets):
    """A bench scene from setup/ with parametric assets and a scene_recon.json for it."""
    ws = TMP / name
    (ws / "assets").mkdir(parents=True)
    staged, objs = [], []
    for a in assets:
        (ws / "assets" / a["name"]).mkdir()
        p = ws / "assets" / a["name"] / f"{a['name']}.xml"
        p.write_text(a["xml"])
        staged.append({"name": a["name"], "xml": str(p), "x": a["x"], "y": a["y"], "yaw_deg": 0.0})
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=staged, out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ws / "scene.xml"))
    for a in assets:
        root = next(m.body(b).name for b in range(m.nbody) if m.body(b).name.startswith(a["name"] + "_") and m.body_parentid[b] == 0)
        objs.append({"id": a["name"], "type": "rigid", "asset_xml": f"assets/{a['name']}/{a['name']}.xml", "root_body": root, "bodies": [root],
                     "pose": {"x": a["x"], "y": a["y"], "z": 0.0, "yaw_deg": 0.0}})
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": objs}))
    return ws, m


def main():
    t0 = time.time()
    # -- the unbraid run: gate suite + blocked reason --------------------------------------------------------------
    ws = TMP / "braid"
    ws.mkdir()
    shutil.copy(FIX / "braid_gate_suite_report.json", ws / "gate_suite_report.json")
    shutil.copy(FIX / "braid_failure_report.json", ws / "failure_report.json")
    r = diagnose_stage3_failure(str(ws), check_hand_scenes=False)
    c = classes(r)
    want = {"goal_not_met", "stage4_randomize", "physics_audit", "unreachable", "upstream_suspect", "needs_steadying_hand", "thin_object"}
    check("unbraid: the gate findings (crossings left, a subgoal not executed, replay lost the grasp, the ribbon through itself, a "
          "two-step station shift) and the blocked reason (a free junction nothing steadies, 1.2 mm strips, Stage 2 must change) are "
          "all classified", want <= set(c) and r["upstream"] and r["upstream"]["stage"] == "stage2" and tools_known(r)
          and not r["upstream"]["verified"] and "report_blocked" not in [x["tool"] for x in r["repairs"]]
          and "place_arm_stations" not in [x["tool"] for x in r["repairs"]] and r["summary"].startswith("Diagnosis"),
          f"classes {[(x['class'], x['score']) for x in r['classes']]}; upstream {r['upstream'] and r['upstream']['stage']}")
    # the same run's last finish verdict as the harness hands it (runner format)
    v = json.loads((FIX / "braid_finish_verdict.json").read_text())
    r = diagnose_stage3_failure(verdict=v, check_hand_scenes=False)
    check("unbraid finish verdict: 'no passing stress test on its hand pose' is a grasp that passed no stress test; the crossings a "
          "goal not met", {"no_stress_pass", "goal_not_met"} <= set(classes(r)), str([(x["class"], x["score"]) for x in r["classes"]]))
    # -- the executor outcomes ---------------------------------------------------------------------------------------
    r = diagnose_stage3_failure(outcomes=str(FIX / "braid_outcomes_grasp.json"), check_hand_scenes=False)
    check("a pinch that did not rise and slipped (grasp_lift: rise, in_hand, no_slip, plan_collision) -> no stress-test pass, slip, "
          "collision", {"no_stress_pass", "slip_in_carry", "collision"} <= set(classes(r))
          and any(x["tool"] == "optimize_grasp_pose" for x in r["repairs"]), str(classes(r)))
    r = diagnose_stage3_failure(outcomes=str(FIX / "braid_outcomes_place.json"), check_hand_scenes=False)
    check("a set-down that tracked 17 mm off and raced the arm (place: upright, servo_tracking, plan_collision, arm_speed) -> goal not "
          "met, collision, joint speed; repairs resume from the subgoal", {"goal_not_met", "collision", "joint_speed"} <= set(classes(r))
          and any(x["tool"] == "execute_subgoal_plan" and x["args"].get("resume") and x["args"].get("from_subgoal") == 1 for x in r["repairs"]),
          str(classes(r)))
    r = diagnose_stage3_failure(outcomes=str(FIX / "rings_outcomes_dry.json"), check_hand_scenes=False)
    check("a dry run where the arm runs into the other hand's thumb -> collision first", classes(r)[:1] == ["collision"], str(classes(r)))
    # -- the charge run: blocked before anything executed --------------------------------------------------------------
    ws = TMP / "charge"
    ws.mkdir()
    shutil.copy(FIX / "charge_failure_report.json", ws / "failure_report.json")
    r = diagnose_stage3_failure(str(ws), check_hand_scenes=False)
    check("charge: reported asset mismatch remains an unverified upstream hypothesis until measured; thin/grasp/goal failures retained",
          {"upstream_suspect", "thin_object", "no_stress_pass", "goal_not_met"} <= set(classes(r)) and r["upstream"]["stage"] == "stage2"
          and not r["upstream"]["verified"] and "report_blocked" not in [x["tool"] for x in r["repairs"]],
          f"{[(x['class'], x['score']) for x in r['classes']]}; first repair {r['repairs'][0]['tool']}")
    # -- measured: a part lower than the pads ---------------------------------------------------------------------------
    plate = '<mujoco model="phone"><worldbody><body name="phone"><freejoint/><geom name="phone_g" type="box" size="0.036 0.074 0.0045" pos="0 0 0.0045" mass="0.15"/></body></worldbody></mujoco>'
    box = '<mujoco model="box"><worldbody><body name="box"><freejoint/><geom name="box_g" type="box" size="0.03 0.03 0.03" pos="0 0 0.03" mass="0.1"/></body></worldbody></mujoco>'
    wsb, m = bench("thin", [{"name": "phone", "xml": plate, "x": -0.12, "y": 0.40}, {"name": "box", "xml": box, "x": 0.12, "y": 0.40}])
    recon = json.loads((wsb / "scene_recon.json").read_text())
    phone_root = recon["objects"][0]["root_body"]
    box_root = recon["objects"][1]["root_body"]
    out = [{"index": 1, "motion": "grasp_lift", "side": "left", "object": phone_root, "ok": False, "failed": ["rise", "in_hand"], "measured": {}},
           {"index": 2, "motion": "grasp_lift", "side": "right", "object": box_root, "ok": False, "failed": ["in_hand"], "measured": {}}]
    (wsb / "ep").mkdir()
    (wsb / "ep" / "outcomes.json").write_text(json.dumps(out))
    r = diagnose_stage3_failure(str(wsb), verdict={"ok": False, "issues": ["subgoal 1: outcome not ok (['rise', 'in_hand'])"]},
                                scene_recon_json=str(wsb / "scene_recon.json"), check_hand_scenes=False)
    thin = next((x for x in r["classes"] if x["class"] == "thin_object"), None)
    check("measured: the 9 mm phone is lower than two pad radii (thin object, with flat-pinch and push repairs); the 6 cm box is not",
          thin is not None and phone_root in thin["objects"] and box_root not in thin["objects"]
          and {"generate_flat_pinches", "push_pose_from_point"} <= {x["tool"] for x in r["repairs"]}, f"{thin and thin['evidence'][:1]}")
    r = diagnose_stage3_failure(str(wsb), verdict={"ok": False, "issues": ["Stage 3 cannot synthesize the opening grasps: no pose (0/168)"]},
                                outcomes=[], scene_recon_json=str(wsb / "scene_recon.json"), check_hand_scenes=False)
    thin = next((x for x in r["classes"] if x["class"] == "thin_object"), None)
    check("measured, nothing executed: every manipulated object is measured; only the phone is thin",
          thin is not None and thin["objects"] == [phone_root], str(thin and thin["objects"]))
    # -- measured: the Stage 2 asset the hand scenes are built from lacks the plug the bench scene has -------------------
    cable_bench = ('<mujoco model="cable"><worldbody><body name="cable_a"><freejoint/><geom name="cable_ga" type="capsule" fromto="0 0 0.004 0.05 0 0.004" size="0.003" mass="0.003"/>'
                   '<body name="cable_last" pos="0.05 0 0"><joint type="ball"/><geom name="cable_glast" type="capsule" fromto="0 0 0.004 0.04 0 0.004" size="0.003" mass="0.003"/>'
                   '<geom name="plug_housing" type="box" size="0.012 0.006 0.004" pos="0.052 0 0.005" mass="0.004"/></body></body></worldbody></mujoco>')
    wsc, m = bench("plug", [{"name": "cable", "xml": cable_bench, "x": 0.10, "y": 0.40}])
    bare = (wsc / "assets" / "cable" / "cable.xml").read_text().replace('<geom name="plug_housing" type="box" size="0.012 0.006 0.004" pos="0.052 0 0.005" mass="0.004"/>', "")
    (wsc / "assets" / "cable" / "cable_bare.xml").write_text(bare)
    (wsc / "poses").mkdir()
    hand = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
    (wsc / "poses" / "hs_right_plug.json").write_text(json.dumps({"hand_xml": hand, "side": "right", "floor": True,
                                                                  "objects": [{"name": "cable", "xml": str(wsc / "assets" / "cable" / "cable_bare.xml"), "pos": [0.10, 0.40, 0.0]}]}))
    (wsc / "failure_report.json").write_text(json.dumps({"stage": "stage3", "reason": "blocked: no pose of the plug: every pinch overlaps 8 mm",
                                                         "last_verdict": {"ok": False, "issues": ["subgoal 2: no passing stress test on its hand pose"]}}))
    r = diagnose_stage3_failure(str(wsc), scene_recon_json=str(wsc / "scene_recon.json"))
    up = next((x for x in r["classes"] if x["class"] == "upstream_suspect"), None)
    check("measured: the hand scene's cable_last has 1 collision geom where the bench's has 2 (the plug) -> upstream Stage 2 first, "
          "report_blocked(upstream='stage2') the first repair", up is not None and r["classes"][0]["class"] == "upstream_suspect"
          and any("cable_cable_last" in e for e in up["evidence"]) and r["upstream"]["stage"] == "stage2" and r["repairs"][0]["tool"] == "report_blocked"
          and r["repairs"][0]["args"]["upstream"] == "stage2" and r["upstream"]["verified"], f"{up and up['evidence'][:1]}")
    (wsc / "poses" / "hs_right_plug.json").write_text(json.dumps({"hand_xml": hand, "side": "right", "floor": True,
                                                                  "objects": [{"name": "cable", "xml": str(wsc / "assets" / "cable" / "cable.xml"), "pos": [0.10, 0.40, 0.0]}]}))
    r = diagnose_stage3_failure(str(wsc), scene_recon_json=str(wsc / "scene_recon.json"))
    check("a hand scene built from the same asset as the bench raises no Stage 2 suspicion", "upstream_suspect" not in classes(r), str(classes(r)))
    r = diagnose_stage3_failure(verdict={"ok": True, "issues": []}, check_hand_scenes=False)
    check("nothing to diagnose: no classes, a plain summary", not r["classes"] and "no failure class" in r["summary"])
    print(f"{len(FAILS)} failed, {time.time() - t0:.0f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
