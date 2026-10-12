"""Smoke tests for the Stage 3 cut route: the executor's cut rule (episode.cut_rules_live / cut_step around
scissors.try_sever) on three scissors / cord / cut-position variations with blades closed by ring forces, and not
cutting when the blades only open; the cut rule read from a Stage 2 manifest (cut_rules_from_recon, the scissors'
`cutter` or one derived from params); servo_rollout recording eq_active only when a scene has cut rules; the `severed`
expectation (outcome.evaluate, subgoal_outcome_check); trajectory_manifest_check rejecting synthetic cut claims without
trusted execution and grasp evidence; the planner's `cut` motion; and scene_recon_manifest_check's
gate on a manipulated deformable that is only in Isaac. CPU only, about two minutes.

    MUJOCO_GL=egl python tool_library/tests/test_scissor_cut_route.py
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
from tool_library.common import scissors as SC  # noqa: E402
from tool_library.common.mjcf import clear_forces  # noqa: E402
from tool_library.skills.construct_cuttable_rope import construct_cuttable_rope  # noqa: E402
from tool_library.skills.construct_scissors import construct_scissors  # noqa: E402
from tool_library.skills.execute_subgoal_plan import cut_rules_from_recon  # noqa: E402
from tool_library.skills.plan_subgoal_keyframes import plan_subgoal_keyframes  # noqa: E402
from tool_library.skills.stage_on_bench import stage_on_bench  # noqa: E402
from tool_library.verifiers import scissors_cut_test as SCT  # noqa: E402
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check  # noqa: E402
from tool_library.verifiers.subgoal_outcome_check import subgoal_outcome_check  # noqa: E402
from tool_library.verifiers.trajectory_manifest_check import trajectory_manifest_check  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_cutroute_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def rule_trial(built, cord_r, cut_frac, close=True):
    """The cut rule in the scissors_cut_test trial scene (pair held, cord hanging through the open wedge): ring forces
    close the blades (or only open them), and every step runs episode.cut_step exactly as servo_rollout does."""
    out = TMP / f"rule_{built['blade_length_m']}_{cord_r}_{int(close)}"
    for F in ([0.7, 1.4, 2.8] if close else [1.4]):
        m, d, joins, _ = SCT._assemble(built, True, {"radius": cord_r}, cut_frac, out)
        rules = [{"kind": "cut", "cutter": dict(built["cutter"], prefix=""), "joins": joins}]
        live = E.cut_rules_live(m, rules)
        drv = SC.ScissorsDriver(m, d, built, joins=[], hold_root=False)     # forces only: the rule does the cutting
        drv.MAX_N = F
        drv.T["open"] = drv.T["settle"]
        if not close:                                                       # push the rings the OPENING way all along
            drv.T.update(open=drv.T["end"], hold=drv.T["end"], close=drv.T["end"])
        inactive, released = set(), []
        # the trial holds the cord at both ends; once it is cut the lower piece is let go, as the hanging strand of the
        # task is (otherwise the two cut ends stay pinned in the blades)
        lower_holds = [e for e in range(m.neq) if (m.equality(e).name or "").startswith("cord_b_") and "_hold_" in (m.equality(e).name or "")]
        n = int(drv.T["end"] / m.opt.timestep)
        for i in range(n):
            clear_forces(d)
            drv.step(i * m.opt.timestep)
            mujoco.mj_step(m, d)
            if E.cut_step(m, d, live, inactive, released, i):
                for e in lower_holds:
                    d.eq_active[e] = 0
        gap = O.join_end_gap(m, d, joins[0]["eq"])
        if released or not close:
            return {"force_N": F, "released": released, "gap_mm": 1000 * gap, "live": len(live), "open_deg": SC.opening_deg(m, d, built)}
    return {"force_N": F, "released": released, "gap_mm": 1000 * gap, "live": len(live), "open_deg": SC.opening_deg(m, d, built)}


def bench(ws):
    """scissors + a cuttable rope staged on the setup bench, with a Stage 2-style manifest and Stage 1 semantics."""
    scb = construct_scissors(str(ws / "assets" / "scissors"), blade_length=0.07, include_floor=False)
    u = np.linspace(0, 1, 15)
    line = np.stack([-0.25 + 0.30 * u, 0.42 + 0.03 * np.sin(6 * u), np.full_like(u, 0.004)], 1)
    rb = construct_cuttable_rope(str(ws / "assets" / "rope"), centreline=line.tolist(), cut_frac=0.5, radius=0.004, segments=24,
                                 scene_name="cord", name="cord")
    sf = rb["scene_fields"]
    stage_on_bench(str(ROOT / "setup" / "scene.xml"),
                   [{"name": "scissors", "xml": scb["xml"], "x": 0.15, "y": 0.42, "yaw_deg": 90.0}, sf["stage_on_bench"]],
                   str(ws / "scene.xml"), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
    sem = {"task_prompt": "cut the cord", "objects": [{"id": "cord", "type": "deformable", "role": "manipulated", "properties": []},
                                                     {"id": "scissors", "type": "articulated", "role": "tool", "properties": []}],
           "subgoals": [{"index": 1, "acting_hand": "left", "motion": "grasp_lift", "object_refs": ["cord"]},
                        {"index": 2, "acting_hand": "right", "motion": "cut", "object_refs": ["scissors", "cord"]}]}
    (ws.parent / "stage1").mkdir(exist_ok=True)
    (ws.parent / "stage1" / "scene_semantics.json").write_text(json.dumps(sem))
    man = {"scene_semantic": "../stage1/scene_semantics.json", "scene_xml": "scene.xml", "simulator": "mujoco+isaac",
           "objects": [{"id": "cord", "type": "deformable", "skill": "construct_cuttable_rope", "root_body": sf["root_body"],
                        "bodies": sf["bodies"], "cut_joins": sf["cut_joins"], "halves": sf["halves"]},
                       {"id": "scissors", "type": "articulated", "skill": "construct_scissors", "root_body": "scissors_scissors",
                        "cutter": scb["cutter"]}]}
    (ws / "scene_recon.json").write_text(json.dumps(man))
    return scb, rb, sf, man


def main():
    # ---------------- 1. the cut rule on three variations
    variants = [("51 mm blades, 5 mm cord at 55 %", construct_scissors(str(TMP / "sc1"), include_floor=False), 0.0025, 0.55),
                ("83 mm blades (the clip's pair), 8.2 mm cord at 60 %",
                 construct_scissors(str(TMP / "sc2"), include_floor=False, blade_length=0.083, blade_width=0.014, handle_length=0.102,
                                    shank_length=0.036, ring_width=0.032, ring_tube=0.0058, mass=0.06), 0.0041, 0.6),
                ("40 mm blades, 3 mm cord at 45 %",
                 construct_scissors(str(TMP / "sc3"), include_floor=False, blade_length=0.04, blade_width=0.008, handle_length=0.06,
                                    shank_length=0.024, mass=0.03), 0.0015, 0.45)]
    for tag, built, r, frac in variants:
        res = rule_trial(built, r, frac, close=True)
        rec = res["released"][0] if res["released"] else {}
        check(f"cut rule [{tag}]: the closing blades sever the join", bool(res["released"]) and res["live"] == 1 and res["gap_mm"] > 3.0,
              f"ring force {res['force_N']} N, frame {rec.get('frame')}, cut {rec.get('cut', {}).get('s_m')} m from the pivot, "
              f"ends {res['gap_mm']:.1f} mm apart")
    neg = rule_trial(variants[0][1], 0.0025, 0.55, close=False)
    check("cut rule: blades that only open never cut", not neg["released"] and neg["gap_mm"] < 1.0, f"open {neg['open_deg']:.1f} deg, gap {neg['gap_mm']:.2f} mm")

    # ---------------- 2. the rule from a Stage 2 manifest
    ws = TMP / "s" / "stage2"; ws.mkdir(parents=True)
    scb, rb, sf, man = bench(ws)
    rules = cut_rules_from_recon(ws / "scene_recon.json")
    check("cut_rules_from_recon: cutter + joins with the scene prefix", len(rules) == 1 and rules[0]["cutter"]["prefix"] == "scissors_"
          and rules[0]["joins"][0]["eq"] == sf["cut_joins"][0]["eq"], json.dumps({k: rules[0][k] for k in ("kind", "gates")}) if rules else "none")
    old = json.loads(json.dumps(man)); sco = next(o for o in old["objects"] if o["id"] == "scissors")
    sco.pop("cutter"); sco["params"] = {"blade_length": 0.07}
    (ws / "recon_old.json").write_text(json.dumps(old))
    r2 = cut_rules_from_recon(ws / "recon_old.json")
    check("cut_rules_from_recon derives the cutter from params for older manifests",
          len(r2) == 1 and abs(r2[0]["cutter"]["blade_length_m"] - 0.07) < 1e-9 and r2[0]["cutter"]["prefix"] == "scissors_")
    (ws / "recon_none.json").write_text(json.dumps({"objects": [o for o in man["objects"] if o["id"] == "scissors"]}))
    check("no cut joins -> no cut rules", cut_rules_from_recon(ws / "recon_none.json") == [])

    # ---------------- 3. servo_rollout records the equality state only with cut rules
    model, _ = E.build_execution_model(str(ws / "scene.xml"))
    d = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, d, 0)
    Qref = np.tile(d.qpos.copy(), (12, 1))
    r_cut = E.servo_rollout(model, Qref, 30.0, rules=rules)
    r_plain = E.servo_rollout(model, Qref, 30.0)
    names = r_cut.get("eq_names") or []
    j = sf["cut_joins"][0]["eq"]
    check("servo_rollout with a cut rule records eq_active [T, neq] + names", "eq_active" in r_cut and r_cut["eq_active"].shape == (12, model.neq)
          and j in names and bool(r_cut["eq_active"][:, names.index(j)].all()) and r_cut["cuts"] == [], f"neq {model.neq}")
    check("servo_rollout without cut rules records the equality state too, with nothing cut",
          "eq_active" in r_plain and r_plain["eq_active"].shape == (12, model.neq) and not r_plain.get("cuts")
          and bool((r_plain["eq_active"] == r_plain["eq_active"][0]).all()))

    # ---------------- 4. the `severed` expectation
    Q = r_cut["qpos"].copy()
    ea = r_cut["eq_active"].copy(); ea[5:, names.index(j)] = 0
    fj = model.joint(next(jj for jj in range(model.njnt) if model.body(int(model.jnt_bodyid[jj])).name == sf["halves"]["b"]["first"])).qposadr[0]
    Q[6:, fj + 1] += 0.01                                                   # the second half moves 1 cm away after the cut
    diag = {"eq_active": ea, "eq_names": names, "eq_active_start": r_cut["eq_active_start"]}
    ev = O.evaluate(model, Q, 30.0, list(range(12)), {"severed": [j]}, diagnostics=diag)
    check("severed: cut during the frames and the ends apart", ev["ok"] and ev["measured"]["severed"][j]["cut_frame"] == 5,
          json.dumps(ev["measured"]["severed"]))
    ev2 = O.evaluate(model, Q, 30.0, list(range(12)), {"severed": True}, diagnostics=diag)
    check("severed: true finds the rope join that was cut", ev2["ok"] and j in ev2["measured"]["severed"])
    ev3 = O.evaluate(model, Q, 30.0, list(range(6, 12)), {"severed": [j]}, diagnostics=diag)
    check("severed: a join already cut before the subgoal does not count", not ev3["ok"] and "severed" in ev3["failed"])
    ev4 = O.evaluate(model, r_cut["qpos"], 30.0, list(range(12)), {"severed": [j]}, diagnostics={"eq_active": ea, "eq_names": names})
    check("cut_ends_apart: a cut whose ends did not move fails", not ev4["checks"]["cut_ends_apart"] and ev4["checks"]["severed"])
    ev5 = O.evaluate(model, Q, 30.0, list(range(12)), {"severed": [j]}, diagnostics=None)
    check("severed without a recorded eq_active fails with a hint", not ev5["ok"] and "severed_hint" in ev5["measured"])
    traj = TMP / "s" / "stage3" / "trajectory.npz"; traj.parent.mkdir(parents=True)
    labels = ["right: close the blades on cord"] * 12
    E.save_episode(traj, Q, labels, 30.0, subgoal=np.full(12, 2), eq_active=ea, eq_names=np.array(names), eq_active_start=r_cut["eq_active_start"])
    so = subgoal_outcome_check(str(ws / "scene.xml"), str(traj), {"severed": [j]}, subgoal=2)
    check("subgoal_outcome_check reads eq_active from the npz", so["ok"] and so["checks"]["severed"], str(so["failed"]))

    # ---------------- 5. synthetic outcome data is not an accepted robot execution
    # Q above was manually displaced to unit-test the severed predicate. Neither
    # that data, a placeholder video, nor hand-written ok:true stress claims can
    # establish physical task completion. Real force-driven cut positives are
    # covered in section 1; this section checks rejection at the manifest boundary.
    s3 = traj.parent
    (s3 / "sim_video.mp4").write_bytes(b"0")
    (s3 / "stage4_task").mkdir(); (s3 / "stage4_task" / "stage4.json").write_text("{}")
    sub = [{"index": 1, "motion": "grasp_lift", "side": "left", "object": sf["halves"]["b"]["bodies"][3],
            "pose": {"source": "rope_pinch_pose", "stress_tests": {"grasp_lift_test": {"ok": True}}}, "outcome": {"ok": True}},
           {"index": 2, "motion": "cut", "side": "right", "object": "scissors_scissors",
            "outcome": {"ok": True, "checks": {"severed": True, "cut_ends_apart": True}, "measured": {"severed": {j: {"cut_frame": 5}}}}}]
    mf = {"scene_semantic": "../stage1/scene_semantics.json", "scene_recon": "../stage2/scene_recon.json", "subgoals": sub,
          "episode": {"trajectory": "trajectory.npz", "sim_video": "sim_video.mp4"}, "stage4": {"task_dir": "stage4_task"},
          "checks": {n: {"ok": True} for n in ("episode_physics_check", "robot_self_collision_check", "arm_reach_check", "stage4_handoff_check")}}
    (s3 / "trajectory_manifest.json").write_text(json.dumps(mf))
    tm = trajectory_manifest_check(str(s3 / "trajectory_manifest.json"))
    check("trajectory_manifest_check rejects a synthetic cut without executor and grasp evidence",
          not tm["ok"] and any("episode.execution_report" in i for i in tm["issues"])
          and any("no independently passing stress evidence" in i for i in tm["issues"]), str(tm["issues"]))
    bad = json.loads(json.dumps(mf)); bad["subgoals"][1]["outcome"] = {"ok": True, "checks": {}}
    (s3 / "bad1.json").write_text(json.dumps(bad))
    tb = trajectory_manifest_check(str(s3 / "bad1.json"))
    check("... rejects a cut without a severed check", not tb["ok"] and any("severed" in i for i in tb["issues"]), str(tb["issues"])[:160])
    E.save_episode(s3 / "noeq.npz", Q, labels, 30.0)
    bad2 = json.loads(json.dumps(mf)); bad2["episode"]["trajectory"] = "noeq.npz"
    (s3 / "bad2.json").write_text(json.dumps(bad2))
    tb2 = trajectory_manifest_check(str(s3 / "bad2.json"))
    check("... rejects a cut whose trajectory carries no eq_active", not tb2["ok"] and any("eq_active" in i for i in tb2["issues"]), str(tb2["issues"])[:160])

    # ---------------- 6. the planner's cut motion
    st = {"held": {"right": "scissors_scissors"}, "grip": {}}
    pl = plan_subgoal_keyframes(str(ws / "scene.xml"), {"index": 2, "motion": "cut", "side": "right", "object": "scissors_scissors",
                                                       "join": j, "cut_s_frac": 0.5}, state=st, auto_via=False)
    labs = [k["label"] for k in pl["keyframes"]]
    check("plan cut: approach, slide in, close; expect severed + in hand",
          any("bring the open blades" in l for l in labs) and any("slide" in l for l in labs) and any("close the blades" in l for l in labs)
          and pl["expect"].get("severed") == [j] and pl["expect"].get("in_hand") and sf["halves"]["a"]["first"] in pl["expect"]["involved"],
          str(labs[-3:]))
    pl2 = plan_subgoal_keyframes(str(ws / "scene.xml"), {"index": 2, "motion": "cut", "side": "right", "object": "scissors_scissors"},
                                 state=st, auto_via=False)
    check("plan cut without a join: just shut them (joint_target closed)", "severed" not in pl2["expect"]
          and pl2["expect"].get("joint") == "scissors_scissors_open" and pl2["expect"].get("joint_target") == 0.0)
    try:
        plan_subgoal_keyframes(str(ws / "scene.xml"), {"index": 2, "motion": "cut", "side": "right", "object": "scissors_scissors", "join": j},
                               state={"held": {}, "grip": {}}, auto_via=False)
        refused = False
    except ValueError as e:
        refused = "holds nothing" in str(e)
    check("plan cut refuses a hand that holds nothing", refused)

    # ---------------- 7. scene_recon_manifest_check: a manipulated deformable must be in scene.xml; a cut one needs cut_joins
    def new_issues(m2, name):
        (ws / name).write_text(json.dumps(m2))
        a = scene_recon_manifest_check(str(ws / name))["issues"]
        b = scene_recon_manifest_check(str(ws / name), require_mujoco_deformables=False)["issues"]
        return [i for i in a if i not in b]

    check("recon gate: the cord in scene.xml with its cut joins adds no issue", new_issues(man, "m_ok.json") == [])
    isaac_only = json.loads(json.dumps(man)); co = isaac_only["objects"][0]
    co.update({"root_body": "cord", "bodies": ["cord"], "skill": "build_isaac_scene"}); co.pop("cut_joins")
    ni = new_issues(isaac_only, "m_isaac.json")
    check("recon gate: an Isaac-only manipulated cord is reported", len(ni) == 1 and "scene.xml" in ni[0] and "construct_cuttable_rope" in ni[0], ni[0][:140] if ni else "")
    nocut = json.loads(json.dumps(man)); nocut["objects"][0].pop("cut_joins")
    nc = new_issues(nocut, "m_nocut.json")
    check("recon gate: a cut cord without cut_joins is reported", len(nc) == 1 and "cut_joins" in nc[0], nc[0][:140] if nc else "")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
