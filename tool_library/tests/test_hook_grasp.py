"""hook on the setup bench: a flat ring lying on the table (a scissor strap: a 10-12 mm wall around a 26-30 mm opening,
with a flat tab like a blade) is too low for side pinches and has no gap under it; thin_grasp_diagnosis finds its
opening and names the hook; hook_grasp_pose puts an upright index through the opening against the wall's inner face
and the thumb on its outer face (screened for the station's arm and through the grasp_stress_test ladder), and the
`hook` motion lifts it through execute_subgoal_plan in contact physics. Cases vary the opening, the wall height, the
yaw and the place. Checks: the ring rises at least 5 cm in hand without slip and no finger presses into the table; the
executed pose passes grasp_lift_test and grasp_stress_test; an opening narrower than a fingertip + 3 mm is refused.
About 10 minutes (two cases at a time).

    MUJOCO_GL=egl python tool_library/tests/test_hook_grasp.py
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
sys.path.insert(0, str(ROOT / "tool_library" / "tests"))

import numpy as np  # noqa: E402

FAILS = 0
# (tag, inner radius, wall height, x, y, yaw_deg)
CASES = [("r15", 0.015, 0.010, 0.06, 0.24, 0.0),
         ("r14", 0.014, 0.011, 0.02, 0.22, 25.0),
         ("r13", 0.013, 0.012, 0.10, 0.20, -20.0)]


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def build(out, tag, r_in, h, x, y, yaw):
    from test_thin_grasp_diagnosis import ring_xml
    from tool_library.skills.stage_on_bench import stage_on_bench
    out.mkdir(parents=True, exist_ok=True)
    xml = ring_xml(out / f"{tag}.xml", r_in, height=h)
    ws = out / "stage2"; ws.mkdir(parents=True, exist_ok=True)
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": tag, "xml": xml, "x": x, "y": y, "yaw_deg": yaw}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [{"id": tag, "type": "rigid", "asset_xml": xml,
                                                                                            "root_body": f"{tag}_b", "bodies": [f"{tag}_b"]}]}))
    return ws


def run_case(args):
    tmp, (tag, r_in, h, x, y, yaw) = args
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tool_library" / "tests"))
    from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan
    from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
    from tool_library.skills.hook_grasp_pose import hook_grasp_pose
    from tool_library.skills.thin_grasp_diagnosis import thin_grasp_diagnosis
    from tool_library.verifiers.grasp_lift_test import grasp_lift_test
    from tool_library.verifiers.grasp_stress_test import grasp_stress_test
    out = Path(tmp) / tag
    ws = build(out, tag, r_in, h, x, y, yaw)
    scene, body = str(ws / "scene.xml"), f"{tag}_b"
    res = {"tag": tag, "wall": h}
    dg = thin_grasp_diagnosis(scene, body)
    res["diag"] = {"as_lies": dg["graspable_as_lies"], "hook": dg["verdicts"]["hook"].get("ok"), "plans": [r["plan"] for r in dg["recommend"]]}
    hs = hand_scene_from_recon(str(ws / "scene_recon.json"), "right", tag, out_json=str(out / "hs.json"))
    hp = hook_grasp_pose(hs, out_json=str(out / "hook.json"))
    res["pose"] = {"ok": hp["ok"], "why": hp.get("why"), "n": len(hp["poses"])}
    if not hp["ok"]:
        return res
    plan = {"subgoals": [{"index": 1, "motion": "hook", "side": "right", "object": body, "pose_candidates": hp["poses"], "lift_m": 0.08}]}
    execute_subgoal_plan(scene, plan, str(out / "ep"))
    o = json.loads((out / "ep" / "outcomes.json").read_text())[0]
    res["outcome"] = {"ok": o["ok"], "failed": o.get("failed"), "variant": o.get("variant"),
                      "m": {k: (o.get("measured") or {}).get(k) for k in ("rise_m", "slip_in_hand_m", "max_table_pen_mm", "hand_links_touching")},
                      "error": (o.get("error") or "")[:200]}
    if o["ok"]:
        k = int(str(o.get("variant", "pose candidate 1")).split()[-1]) - 1 if "candidate" in str(o.get("variant")) else 0
        pose = hp["poses"][k]
        lt = grasp_lift_test(hs, pose, lift_heights_m=[0.04, 0.10, 0.15])
        st = grasp_stress_test(hs, pose)
        res["stress"] = {"lift": lt["ok"], "lift_fail": lt["first_fail"], "stress": st["ok"], "stress_fail": st["first_fail"]}
    return res


def main():
    tmp = Path(tempfile.mkdtemp(prefix="hook_grasp_", dir=os.environ.get("TMPDIR")))
    with mp.get_context("spawn").Pool(min(2, len(CASES))) as pool:     # two physics processes at a time (a shared node)
        results = pool.map(run_case, [(str(tmp), c) for c in CASES])
    for r in results:
        tag = r["tag"]
        tall = r["wall"] >= 0.0105               # a wall over pad radius + 3 mm is pinchable from its sides as well
        check(f"[{tag}] the diagnosis finds the opening and names the hook ({'also' if tall else 'not'} a side pinch as it lies)",
              r["diag"]["hook"] and r["diag"]["as_lies"] == tall and ["hook"] in r["diag"]["plans"], json.dumps(r["diag"]))
        check(f"[{tag}] hook_grasp_pose has arm- and stress-screened candidates", r["pose"]["ok"], json.dumps(r["pose"]))
        if not r["pose"]["ok"]:
            continue
        o = r["outcome"]
        check(f"[{tag}] the hook lifts the ring >= 5 cm in hand without slip, no finger in the table",
              o["ok"] and o["m"]["rise_m"] >= 0.05 and o["m"]["max_table_pen_mm"] <= 2.0, json.dumps(o)[:400])
        if r.get("stress"):
            check(f"[{tag}] the executed hook passes grasp_lift_test and grasp_stress_test", r["stress"]["lift"] and r["stress"]["stress"], json.dumps(r["stress"]))
    # an opening narrower than a fingertip + 3 mm is refused, with the reason
    from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
    from tool_library.skills.hook_grasp_pose import hook_grasp_pose
    ws = build(tmp / "narrow", "nr", 0.008, 0.010, 0.06, 0.24, 0.0)
    hs = hand_scene_from_recon(str(ws / "scene_recon.json"), "right", "nr")
    hp = hook_grasp_pose(hs, screen_lift=False, arm_screen=False)
    check("a 16 mm opening is refused: a fingertip needs 17 mm", not hp["ok"] and "a fingertip needs" in (hp.get("why") or ""), str(hp.get("why")))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({tmp})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
