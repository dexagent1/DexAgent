"""lift_together / carry_together: both hands on two links of ONE heavy cable, lifted and carried together.
Cases on the setup bench with a thick heavy cable (construct_rope, 3.2 cm thick: 1.2 m / 1.1 kg, and 1.5 m / 1.4 kg like the
knot run's), at two places, each run by the real executor in contact physics:

1. one hand alone pinching near mid-span is run for comparison and reported (with a three-finger pinch it lifts a
   1.1 kg cable here: a single pinch is not always the problem; lifting a SPAN, so the end can pass under it, is);
2. both hands pinch two links 30 cm apart (handle_pinch_pose, thumb against index and middle), close in the same
   keyframes and lift together: both links rise at least 5 cm, then carry_together moves them 10 cm; the cable never
   drops (each hand still on its link or its neighbours at the end), the spacing between the grips stays within 2 cm,
   each hand's checks are named per hand and the hands never touch;
3. the planner refuses two links of DIFFERENT objects and a carry_together whose hands hold nothing.
Seed screening checks a short 4 cm single-hand pickup, then the actual two-hand
execution must lift 7 cm and carry 10 cm. Requiring each hand to lift the full
load to the cooperative height would exclude the load-sharing behavior under
test. This seed screen is not final Stage 3 grasp certification: the production
manifest still independently runs stress verifiers at their required defaults.

    MUJOCO_GL=egl python tool_library/tests/test_lift_together.py
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
from tool_library.skills.construct_rope import construct_rope  # noqa: E402
from tool_library.skills.execute_subgoal_plan import execute_subgoal_plan  # noqa: E402
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon  # noqa: E402
from tool_library.skills.handle_pinch_pose import handle_pinch_pose  # noqa: E402
from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose  # noqa: E402
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_lift2_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def cable_scene(wd, L, R_, y0, nseg=40):
    b = construct_rope(str(wd / "assets/rope"), name="rope", length=L, radius=R_, segments=nseg, pos=[0, 0, R_], grips=False, include_floor=False,
                       friction=[1.0, 0.02, 0.001], density=700)
    m0 = mujoco.MjModel.from_xml_path(b["xml"])
    bodies = ["rope_" + m0.body(i).name for i in range(1, m0.nbody)]
    ws = wd / "stage2"; ws.mkdir(parents=True, exist_ok=True)
    stage_on_bench(str(ROOT / "setup" / "scene.xml"), assets=[{"name": "rope", "xml": b["xml"], "x": -L / 2 + 0.02, "y": y0}],
                   out_xml=str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
        {"id": "rope", "type": "deformable", "skill": "construct_rope", "asset_xml": b["xml"], "root_body": bodies[0], "bodies": bodies}]}))
    return str(ws / "scene.xml"), str(ws / "scene_recon.json"), bodies, float(sum(m0.body_mass))


def pinch(wd, recon, scene, side, link):
    hs = hand_scene_from_recon(recon, side, "rope", grasp_body=link, out_json=str(wd / f"hs_{side}.json"))
    m = mujoco.MjModel.from_xml_path(scene)
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body(link).id and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE)
    pr = handle_pinch_pose(str(wd / f"hs_{side}.json"), bar=m.geom(g).name, s=0.5, fingers=["thumb", "index", "middle"], tilt_deg=0.0)
    candidate = wd / f"pinch_{side}.json"
    # Screen contact and the complete arm path before testing load sharing in
    # the real two-hand execution. A short single-hand pickup only screens seeds.
    result = optimize_grasp_pose(str(wd / f"hs_{side}.json"), starts=[pr], family="bar", bar=m.geom(g).name,
                                 bench_scene=scene, arm_waypoints=[[0, 0, .07], [0, .10, .07]],
                                 tests=["grasp_lift_test"], lift_heights_m=[.04],
                                 max_evals=48, iterations=4, workers=2,
                                 out_json=str(candidate))
    (wd / f"seed_screen_{side}.json").write_text(json.dumps(result, indent=1))
    assert result["ok"], f"no cable pickup seed for {side}; full report: {wd / f'seed_screen_{side}.json'}"
    return str(wd / f"pinch_{side}.json"), str(wd / f"hs_{side}.json")


def case(k, L, R_, y0, nseg=40, single=False):
    wd = TMP / f"cable{k}"
    scene, recon, bodies, mass = cable_scene(wd, L, R_, y0, nseg=nseg)
    m = mujoco.MjModel.from_xml_path(scene)
    links = [n for n in bodies if "B_" in n and not n.endswith(("first", "last"))]
    # two links 30 cm apart around the middle of the cable
    seg = L / (len(links) + 1)
    n_off = int(round(0.15 / seg))
    mid = len(links) // 2
    la, lb = links[mid - n_off], links[mid + n_off]
    (pa, ha), (pb, hb) = pinch(wd, recon, scene, "left", la), pinch(wd, recon, scene, "right", lb)
    lift, carry = 0.07, [0.0, 0.10, 0.0]
    if single:
        # the same right-hand pinch, with no left hand: the hand alone at 15 cm from the middle
        s1 = execute_subgoal_plan(scene, {"subgoals": [{"index": 1, "motion": "grasp_lift", "side": "right", "object": lb, "pose": pb, "hand_scene": hb,
                                                         "lift_m": lift}]}, str(wd / "single"))
        o1 = json.loads((wd / "single" / "outcomes.json").read_text())[0]
        print(f"[INFO] cable {k} ({mass:.2f} kg): one hand pinching near mid-span alone -> ok={s1['ok']} failed={o1.get('failed')} "
              f"rise {(o1.get('measured') or {}).get('rise_m')} m")
    plan = {"subgoals": [
        {"index": 1, "motion": "lift_together", "lift_m": lift, "hands": {"left": {"object": la, "pose": pa, "hand_scene": ha},
                                                                         "right": {"object": lb, "pose": pb, "hand_scene": hb}}},
        {"index": 2, "motion": "carry_together", "offset": carry, "duration_s": 2.0}]}
    s = execute_subgoal_plan(scene, plan, str(wd / "together"))
    o = json.loads((wd / "together" / "outcomes.json").read_text())
    if not s["ok"]:
        check(f"cable {k}: two-hand physical plan executes", False, json.dumps(s.get("first_failure"))[:1800])
        return scene, la, lb, pa, pb, ha, hb
    ep = E.load_episode(wd / "together" / "trajectory.npz")
    m = mujoco.MjModel.from_xml_path(str(wd / "together" / "scene_physics.xml"))
    A, Bt = E.body_track(m, ep["qpos"], la)[0], E.body_track(m, ep["qpos"], lb)[0]
    seg1 = np.nonzero(np.asarray(ep["subgoal"]) == 1)[0]
    rise = min(A[seg1[-1], 2] - A[0, 2], Bt[seg1[-1], 2] - Bt[0, 2])
    mid0, mid1 = 0.5 * (A[seg1[-1]] + Bt[seg1[-1]]), 0.5 * (A[-1] + Bt[-1])
    moved = float(np.linalg.norm((mid1 - mid0)[:2]))
    still_up = min(A[-1, 2] - A[0, 2], Bt[-1, 2] - Bt[0, 2])
    check(f"cable {k} ({L} m, {mass:.2f} kg): two hands lift it mid-span by at least 5 cm and carry it 10 cm without dropping it",
          s["ok"] and rise >= 0.05 and moved >= 0.09 and still_up >= 0.05,
          f"rise {1000 * rise:.0f} mm, carried {1000 * moved:.0f} mm, still up {1000 * still_up:.0f} mm, {[(x['index'], x['ok'], x.get('failed')) for x in o]}")
    sp = [x.get("measured", {}).get("spacing") for x in o]
    check(f"cable {k}: the spacing between the grips stays within 2 cm through the lift and the carry", all(x and abs(x["max_mm"] - x["d0_mm"]) <= 20
                                                                                                          and abs(x["min_mm"] - x["d0_mm"]) <= 20 for x in sp), str(sp))
    ck = o[0].get("checks", {})
    check(f"cable {k}: each hand's checks are named per hand (rise, link in hand, no slip) and the hands never touch",
          all(ck.get(f"{s_}:rise") and ck.get(f"{s_}:link_in_hand:{l_}") and ck.get(f"{s_}:no_slip") for s_, l_ in (("left", la), ("right", lb)))
          and ck.get("hands_apart") is True, str(sorted(ck)))
    return scene, la, lb, pa, pb, ha, hb


def refusals(scene, la, lb, pa, pb, ha, hb):
    try:
        plan_subgoal_keyframes(scene, {"index": 1, "motion": "carry_together", "offset": [0, 0.1, 0]}, auto_via=False)
        e1 = None
    except ValueError as e:
        e1 = str(e)
    check("carry_together with empty hands is refused", e1 is not None and "lift_together first" in e1, (e1 or "")[:100])
    try:
        plan_subgoal_keyframes(scene, {"index": 1, "motion": "lift_together", "hands": {"left": {"object": la, "pose": pa, "hand_scene": ha},
                                                                                       "right": {"object": "table", "pose": pb, "hand_scene": hb}}}, auto_via=False)
        e2 = None
    except ValueError as e:
        e2 = str(e)
    check("lift_together on two different objects is refused (that is a 'both' entry)", e2 is not None and "one free object" in e2, (e2 or "")[:100])
    pl = plan_subgoal_keyframes(scene, {"index": 1, "motion": "lift_together", "lift_m": 0.07, "hands": {"left": {"object": la, "pose": pa, "hand_scene": ha},
                                                                                                        "right": {"object": lb, "pose": pb, "hand_scene": hb}}}, auto_via=False)
    labs = [k["label"] for k in pl["keyframes"]]
    check("one plan, both hands in every keyframe: two-hand labels, a phase per hand, the cable root among the handled parts",
          all(" | " in l for l in labs[1:]) and {p_["hand"] for p_ in pl["phases"]} == {"left", "right"} and "rope_ropeB_first" in pl["expect"]["involved"]
          and pl["state"]["held"] == {"left": la, "right": lb}, str(labs[-1]))


def main():
    first = case(1, 1.2, 0.016, 0.40, single=True)
    case(2, 1.5, 0.016, 0.44, nseg=50)          # the knot run's cable: 1.5 m, 3.2 cm thick, 1.4 kg
    refusals(*first)
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
