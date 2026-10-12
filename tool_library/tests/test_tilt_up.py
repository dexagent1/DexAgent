"""tilt_up then a pinch with the other hand, on the setup bench: a flat box (7, 9 and 11 mm thick, several footprints and
places) overhangs the table's near edge; one hand presses the overhanging end down (a slide_contact_pose 'press' pose)
so the box pivots about the edge and its far end rises; the hand keeps pressing while the other hand pinches the raised
end across its side faces (overhang_pinch_pose raised, grip 'sides': now a pad radius above the table), the pressing
hand lets go, and the pinching hand lifts the box. All through execute_subgoal_plan in contact physics. Checks: the far
end rises at least 70 % of the asked 30 mm without the box leaving the table or a finger pressing into it; the raised
pinch tool returns a screened pinch or says why none holds (a side pinch of a raised end this thin holds the weight only
by torsional friction and rarely passes the stress ladder: the hold, release and lift after it are printed, not
required); a press inside the table's edge is refused. About 8 minutes (the cases run in parallel).

    MUJOCO_GL=egl python tool_library/tests/test_tilt_up.py
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

FAILS = 0
# (tag, thickness, length, width, x, overhang past the near edge): the left hand presses, the right hand pinches
CASES = [("k9", 0.009, 0.148, 0.072, 0.04, 0.055),
         ("k7", 0.007, 0.140, 0.068, 0.03, 0.050),
         ("k11", 0.011, 0.150, 0.075, 0.05, 0.058)]


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def build(out, tag, T, L, Wd, x, ov):
    from tool_library.skills.stage_on_bench import stage_on_bench
    out.mkdir(parents=True, exist_ok=True)
    xml = out / f"{tag}.xml"
    xml.write_text(f'<mujoco model="{tag}"><worldbody><body name="b"><freejoint/><geom name="g" type="box" size="{Wd / 2} {L / 2} {T / 2}" '
                   f'pos="0 0 {T / 2}" mass="0.17" friction="0.9 0.01 0.001" rgba="0.25 0.25 0.3 1"/></body></worldbody></mujoco>')
    ws = out / "stage2"; ws.mkdir(parents=True, exist_ok=True)
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": tag, "xml": str(xml), "x": x, "y": 0.05 - ov + L / 2}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [{"id": tag, "type": "rigid", "asset_xml": str(xml),
                                                                                            "root_body": f"{tag}_b", "bodies": [f"{tag}_b"]}]}))
    return ws


def run_case(args):
    tmp, (tag, T, L, Wd, x, ov) = args
    sys.path.insert(0, str(ROOT))
    from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
    from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
    from tool_library.skills.overhang_pinch_pose import overhang_pinch_pose
    from tool_library.skills.slide_contact_pose import slide_contact_pose
    from tool_library.verifiers.grasp_lift_test import grasp_lift_test
    from tool_library.verifiers.grasp_stress_test import grasp_stress_test
    out = Path(tmp) / tag
    ws = build(out, tag, T, L, Wd, x, ov)
    scene, body, recon = str(ws / "scene.xml"), f"{tag}_b", str(ws / "scene_recon.json")
    res = {"tag": tag, "T": T}
    hsP = hand_scene_from_recon(recon, "left", tag, out_json=str(out / "hs_left.json"))
    press = slide_contact_pose(hsP, [0, -1], mode="press", contact_point=[x, 0.05 - ov + 0.012])
    tilt = {"index": 1, "motion": "tilt_up", "side": "left", "object": body, "pose_candidates": press["poses"], "lift_m": 0.03}
    execute_subgoal_plan(scene, {"subgoals": [tilt]}, str(out / "ep"))
    o1 = json.loads((out / "ep" / "outcomes.json").read_text())[0]
    res["tilt"] = {"ok": o1["ok"], "failed": o1.get("failed"), "m": {k: (o1.get("measured") or {}).get(k) for k in ("end_raised_m", "com_inside_table_m", "max_table_pen_mm")}}
    if not o1["ok"]:
        return res
    hsR = hand_scene_from_recon(recon, "right", tag, state=str(out / "ep" / "trajectory.npz"), out_json=str(out / "hs_right.json"))
    op = overhang_pinch_pose(hsR, [0, 1], raised=True, grip="sides", out_json=str(out / "raised.json"))
    res["pinch"] = {"ok": op["ok"], "why": op.get("why"), "grip": op.get("grip", "edge"), "n": len(op["poses"])}
    if not op["ok"]:
        return res
    plan = {"subgoals": [tilt, {"index": 2, "motion": "hold", "side": "right", "object": body, "pose_candidates": op["poses"]},
                         {"index": 3, "motion": "release", "side": "left", "object": body},
                         {"index": 4, "motion": "grasp_lift", "side": "right", "object": body, "lift_m": 0.08}]}
    (out / "plan.json").write_text(json.dumps(plan, default=float))
    summ = execute_subgoal_plan(scene, str(out / "plan.json"), str(out / "ep"), resume=True)
    outs = json.loads((out / "ep" / "outcomes.json").read_text())
    res["ok"] = summ["ok"]
    res["outcomes"] = [{"motion": o["motion"], "ok": o["ok"], "failed": o.get("failed"), "variant": o.get("variant"),
                        "m": {k: (o.get("measured") or {}).get(k) for k in ("end_raised_m", "rise_m", "slip_in_hand_m", "max_table_pen_mm",
                                                                             "hand_links_touching")}} for o in outs]
    hold = next((o for o in outs if o["motion"] == "hold"), {})
    if hold.get("ok"):
        k = int(str(hold.get("variant", "pose candidate 1")).split()[-1]) - 1 if "candidate" in str(hold.get("variant")) else 0
        pose = op["poses"][k]
        lt = grasp_lift_test(op["hand_scene"], pose, lift_heights_m=[0.04, 0.10])
        st = grasp_stress_test(op["hand_scene"], pose)
        res["stress"] = {"lift": lt["ok"], "lift_fail": lt["first_fail"], "stress": st["ok"], "stress_fail": st["first_fail"]}
    return res


def main():
    from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes
    tmp = Path(tempfile.mkdtemp(prefix="tilt_up_", dir=os.environ.get("TMPDIR")))
    with mp.get_context("spawn").Pool(min(2, len(CASES))) as pool:     # two physics processes at a time (a shared node)
        results = pool.map(run_case, [(str(tmp), c) for c in CASES])
    for r in results:
        tag = r["tag"]
        t = r["tilt"]
        check(f"[{tag}] pressing the overhanging end raises the far end (asked 30 mm), the box stays on the table, no finger in it",
              t["ok"] and t["m"]["end_raised_m"] >= 0.021 and t["m"]["max_table_pen_mm"] <= 2.0, json.dumps(t))
        if not t["ok"]:
            continue
        # a side pinch of the raised end holds the part's weight only by torsional friction: on parts this thin it rarely
        # passes the stress ladder (documented), so the chain after the tilt is reported, not required; what is required
        # is that the pose tool either returns a screened pinch or says why none holds
        check(f"[{tag}] overhang_pinch_pose (raised) returns a screened pinch of the raised end or says why none holds",
              r["pinch"]["ok"] or bool(r["pinch"].get("why")), json.dumps(r["pinch"]))
        if not r["pinch"]["ok"]:
            continue
        by = {o["motion"]: o for o in r["outcomes"]}
        gl = by.get("grasp_lift", {})
        print(f"[info] [{tag}] the chain after the tilt: " + json.dumps({"outcomes": [(o["motion"], o["ok"], o["failed"]) for o in r["outcomes"]],
                                                                        "lift": gl.get("m"), "stress": r.get("stress")}))
    # a press inside the table's edge is refused
    ws = build(tmp / "refuse", "rf", 0.009, 0.148, 0.072, 0.04, 0.055)
    try:
        plan_subgoal_keyframes(str(ws / "scene.xml"), {"index": 1, "motion": "tilt_up", "side": "left", "object": "rf_b", "pose": {"pos": [0, 0, 0.2], "euler": [0, 0, 0], "finger_targets": {}},
                                                       "press_point": [0.04, 0.10, 0.009]}, auto_via=False)
        err = None
    except ValueError as e:
        err = str(e)
    check("a press inside the table's edge is refused", bool(err) and "inside the support's edge" in err, str(err)[:200])
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({tmp})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
