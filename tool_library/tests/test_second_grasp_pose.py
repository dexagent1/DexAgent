"""grasp_more, second_grasp_pose and multi_object_hold_test: one hand takes a second object with its free fingers while it
keeps the first (finger allocation). The multi-cup run failed because the second cup had to be taken while the first stayed
in the same hand, and the planner could not express it. Cases on the setup bench with a parametric mug and a cup, two sizes
and placements, each run by the real executor in contact physics:

1. second_grasp_pose designs the pair: the mug between thumb and index (a pinch across its handle), the hand turned so
   middle, ring and pinky point into the cup, which they enter together and spread against its inner wall; the first
   grasp is designed in the second's wrist family and kept only when its clamp presses both pads onto the handle; the
   mug rides the hand clear of the cup; both grasps reachable by the arm;
2. the first grasp lifts the mug on its own (grasp_lift_test) and the pair holds both through lifts, shakes and turns
   (multi_object_hold_test);
3. executed: grasp_lift the mug, grasp_more the cup (only the free fingers close), then carry both 10 cm: both rise at
   least 4 cm and travel with the hand, the mug is checked in hand on the second grasp and the carry (`also_in_hand`);
4. the planner refuses a second grasp on fingers that already hold, the stress test refuses an overlapping allocation,
   and a release naming the cup opens only its fingers (the mug stays held).
About fifteen minutes on an idle machine.

    MUJOCO_GL=egl python tool_library/tests/test_second_grasp_pose.py
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
from tool_library.common import hand as H  # noqa: E402
from tool_library.skills.construct_cup import construct_cup  # noqa: E402
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes  # noqa: E402
from tool_library.skills.second_grasp_pose import second_grasp_pose  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.verifiers.grasp_lift_test import grasp_lift_test  # noqa: E402
from tool_library.verifiers.multi_object_hold_test import multi_object_hold_test  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_twocups_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def cups_scene(wd, rim, base, hgt, A, B):
    # the first: a mug (a handle ring of capsules on its wall), the second: a plain cup
    ca = construct_cup(str(wd / "assets/cupa"), name="cupa", base_radius=base, rim_radius=rim, height=hgt, mass=0.05,
                       handle={"radius": 0.022, "tube": 0.005, "z_frac": 0.55, "segments": 8})
    cb = construct_cup(str(wd / "assets/cupb"), name="cupb", base_radius=base, rim_radius=rim, height=hgt, mass=0.03, body_rgba=[0.2, 0.5, 0.8, 1])
    ws = wd / "stage2"; ws.mkdir(parents=True, exist_ok=True)
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "cupa", "xml": ca["xml"], "x": A[0], "y": A[1]},
                                                              {"name": "cupb", "xml": cb["xml"], "x": B[0], "y": B[1]}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
        {"id": "cupa", "type": "rigid", "asset_xml": ca["xml"], "root_body": "cupa_cupa", "bodies": ["cupa_cupa"]},
        {"id": "cupb", "type": "rigid", "asset_xml": cb["xml"], "root_body": "cupb_cupb", "bodies": ["cupb_cupb"]}]}))
    return str(ws / "scene.xml"), str(ws / "scene_recon.json")


def case(k, rim, base, hgt, A, B, side="right"):
    wd = TMP / f"cups{k}"
    scene, recon = cups_scene(wd, rim, base, hgt, A, B)
    hand_scene_from_recon(recon, side, "cupa", out_json=str(wd / "hs_a.json"))
    hand_scene_from_recon(recon, side, "cupb", out_json=str(wd / "hs_b.json"))
    m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    ca = d.xpos[m.body("cupa_cupa").id].copy()
    # the pinch across the handle's outer bar: thumb and index on its two sides, the pinch axis along the cup's tangent
    hg = sorted((g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("cupa_cupa_handle")),
                key=lambda g: -np.linalg.norm(d.geom_xpos[g][:2] - ca[:2]))[:2]
    c = np.mean([d.geom_xpos[g] for g in hg], axis=0)
    rr = c - ca; rr[2] = 0.0; rr /= np.linalg.norm(rr); tt = np.cross([0.0, 0.0, 1.0], rr)
    tube = float(m.geom_size[hg[0]][0])
    res = None
    for sgn in (1.0, -1.0):                         # the thumb on either side of the handle
        pts = {"thumb": (c + sgn * tt * (tube + 0.0005)).tolist(), "index": (c - sgn * tt * (tube + 0.0005)).tolist()}
        res = second_grasp_pose(str(wd / "hs_b.json"), None, "cupa_cupa", "cupb_cupb", first_points=pts,
                                first_normals={"thumb": (sgn * tt).tolist(), "index": (-sgn * tt).tolist()},
                                bench_scene=scene, side=side, out_json=str(wd / "pose2.json"))
        if res["ok"]:
            break
    sg = (res.get("pose") or {}).get("second_grasp") or {}
    check(f"cups {k}: second_grasp_pose designs the pair, the mug riding the hand (thumb + index on its handle | middle, ring, pinky inside the cup)",
          res["ok"] and sorted(res["pose"]["grip_fingers"]) == ["middle", "pinky", "ring"] and sorted(res["pose"]["first_pose"]["grip_fingers"]) == ["index", "thumb"]
          and len(sg.get("wall_fingers", [])) >= 2 and sg.get("clear_mm", 0) >= 4.0 and (sg.get("reach") or {}).get("ok"),
          f"{sg} rejected {res['rejected']}")
    if not res["ok"]:
        return None
    p2 = res["pose"]; p1 = p2["first_pose"]
    (wd / "pose1.json").write_text(json.dumps(p1, default=float)); (wd / "pose2.json").write_text(json.dumps(p2, default=float))
    gl = grasp_lift_test(str(wd / "hs_a.json"), p1, lift_heights_m=(0.04, 0.08))
    check(f"cups {k}: the first grasp lifts the mug on its own (grasp_lift_test)", gl["ok"], f"{gl.get('first_fail')} {gl['lifts']}")
    mh = multi_object_hold_test(str(wd / "hs_b.json"), [p1, p2], ["cupa_cupa", "cupb_cupb"])
    check(f"cups {k}: the hand holds the mug and the cup through lifts, shakes and turns (multi_object_hold_test)", mh["ok"],
          f"first fail {mh['first_fail']}: {[ph for ph in mh['phases'] if not ph['ok']][:1]}")
    plan = {"subgoals": [
        {"index": 1, "motion": "grasp_lift", "side": side, "object": "cupa_cupa", "pose": str(wd / "pose1.json"), "hand_scene": str(wd / "hs_a.json"), "lift_m": 0.04},
        {"index": 2, "motion": "grasp_more", "side": side, "object": "cupb_cupb", "pose": str(wd / "pose2.json"), "hand_scene": str(wd / "hs_b.json"), "lift_m": 0.05}]}
    s = execute_subgoal_plan(scene, plan, str(wd / "ep"))
    if s["ok"]:
        # Measure the carry from its actual initial state. Grasp seating can
        # shift a cup on the table; the reconstruction's initial body origin
        # therefore does not define a 10 cm carry after both grasps.
        prefix = E.load_episode(wd / "ep" / "trajectory.npz")
        runtime = mujoco.MjModel.from_xml_path(str(wd / "ep" / "scene_physics.xml"))
        current = E.body_track(runtime, prefix["qpos"], "cupb_cupb")[0][-1]
        plan["subgoals"].append({"index": 3, "motion": "carry", "side": side, "object": "cupb_cupb",
            "target_pos": (current + np.array([0., -.10, 0.])).tolist(), "duration_s": 2.0})
        s = execute_subgoal_plan(scene, plan, str(wd / "ep"), resume=True)
    o = json.loads((wd / "ep" / "outcomes.json").read_text())
    if not s["ok"]:
        check(f"cups {k}: physical multi-object plan executes", False, json.dumps(s.get("first_failure"))[:1800])
        return None
    ep = E.load_episode(wd / "ep" / "trajectory.npz")
    m = mujoco.MjModel.from_xml_path(str(wd / "ep" / "scene_physics.xml"))
    PA, PB = E.body_track(m, ep["qpos"], "cupa_cupa")[0], E.body_track(m, ep["qpos"], "cupb_cupb")[0]
    riseA, riseB = PA[-1, 2] - PA[0, 2], PB[-1, 2] - PB[0, 2]
    seg3 = np.nonzero(np.asarray(ep["subgoal"]) == 3)[0]
    movedA, movedB = (float(np.linalg.norm((P[seg3[-1]] - P[seg3[0]])[:2])) for P in (PA, PB))
    check(f"cups {k}: one hand lifts and carries the mug and the cup (grasp_lift, grasp_more with the free fingers, carry 10 cm)",
          s["ok"] and riseA >= 0.04 and riseB >= 0.04 and movedA >= 0.09 and movedB >= 0.09,
          f"rise A {1000 * riseA:.0f} mm, B {1000 * riseB:.0f} mm; carried A {1000 * movedA:.0f} mm, B {1000 * movedB:.0f} mm; "
          f"{[(x['index'], x['ok'], x.get('failed'), x.get('error')) for x in o]}")
    check(f"cups {k}: the mug is checked in the hand on the second grasp and on the carry (also_in_hand)",
          all(x.get("checks", {}).get("also_in_hand:cupa_cupa") is True for x in o[1:]), str([x.get("measured", {}).get("also_in_hand") for x in o[1:]]))
    return scene, wd, side, p1, p2, o


def refusals(scene, wd, side, p1, p2, o):
    st = json.loads(json.dumps(json.loads((wd / "ep" / "subgoal_02" / "state.json").read_text())))
    check("the planner state lists the mug and the cup in the hand with their fingers", [e["body"] for e in st["in_hand"][side]] == ["cupa_cupa", "cupb_cupb"]
          and sorted(st["in_hand"][side][0]["grip_fingers"]) == ["index", "thumb"], json.dumps(st.get("in_hand")))
    st1 = json.loads((wd / "ep" / "subgoal_01" / "state.json").read_text())
    bad = dict(p2, grip_fingers=["index", "middle"])
    try:
        plan_subgoal_keyframes(scene, {"index": 2, "motion": "grasp_more", "side": side, "object": "cupb_cupb", "pose": bad}, state=st1, auto_via=False)
        e1 = None
    except ValueError as e:
        e1 = str(e)
    check("grasp_more on a finger that already holds the mug is refused", e1 is not None and "already hold" in e1, (e1 or "")[:120])
    try:
        multi_object_hold_test(str(wd / "hs_b.json"), [p1, bad], ["cupa_cupa", "cupb_cupb"])
        e2 = None
    except ValueError as e:
        e2 = str(e)
    check("multi_object_hold_test refuses an overlapping allocation", e2 is not None and "already hold" in e2, (e2 or "")[:120])
    ep = E.load_episode(wd / "ep" / "trajectory.npz")
    rel = plan_subgoal_keyframes(scene, {"index": 4, "motion": "release", "side": side, "object": "cupb_cupb"}, qpos0=E.state_qpos(E.build_execution_model(scene)[0], wd / "ep" / "trajectory.npz").tolist(), state=json.loads((wd / "ep" / "subgoal_03" / "state.json").read_text()),
                                 auto_via=False)
    m = mujoco.MjModel.from_xml_path(scene)
    rig = H.HandRig.build(m, f"robot_{side}_", side)
    k0, k1 = np.asarray(rel["keyframes"][0]["qpos"]), np.asarray(rel["keyframes"][-1]["qpos"])
    moved = {f: max(abs(k1[a] - k0[a]) for kk, a in rig.finger_qpos.items() if H.finger_of_joint(kk) == f) for f in H.FINGERS}
    check("a release naming the cup opens only its fingers; the mug stays held", moved["thumb"] < 1e-6 and moved["index"] < 1e-6 and moved["middle"] > 0.1
          and rel["state"]["held"][side] == "cupa_cupa" and rel["expect"].get("also_in_hand"), str({f: round(v, 3) for f, v in moved.items()}))


def main():
    first = case(1, 0.036, 0.027, 0.085, [0.10, 0.26], [0.12, 0.38])
    case(2, 0.038, 0.028, 0.09, [0.16, 0.28], [0.06, 0.34])
    if first:
        refusals(*first)
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
