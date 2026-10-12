"""stage1_goal_gate on a synthetic bench episode: a push that reaches the brick vs one that stops 13 mm short under a weak
self-written expect; a pick, a stack and an unresolvable goal (fixed with goal_predicates); a deviation that changes the
support (accepted) and one that swaps 'touches' for a displacement (rejected); capped explicit tolerances; misuse."""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.stage1_goal_gate import stage1_goal_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {str(detail)[:1500]}"))
    FAILS += 0 if ok else 1


BRICK = '<freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/>'
PAD = '<freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/>'
XML = f"""<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.5 0.5 0.02"/></body>
<body name="yellow_brick_yellow_brick" pos="0 0 0.02">{BRICK}</body>
<body name="lavender_brick_lavender_brick" pos="0.1 0 0.02">{BRICK}</body>
<body name="pink_brick_pink_brick" pos="0.2 0.1 0.02">{BRICK}</body>
<body name="sage_brick_sage_brick" pos="-0.15 0.1 0.02">{BRICK}</body>
<body name="robot_left_pad_a" pos="0.3 0.3 0.2">{PAD}</body><body name="robot_left_pad_b" pos="0.3 0.35 0.2">{PAD}</body>
<body name="robot_right_pad_a" pos="-0.3 0.3 0.2">{PAD}</body><body name="robot_right_pad_b" pos="-0.3 0.35 0.2">{PAD}</body>
</worldbody></mujoco>"""

SEM = {"task_success": "The pink brick rests on top of the yellow brick.",
       "objects": [{"id": i, "name": f"{i.split('_')[0]} lego brick", "type": "rigid", "initial_state": "on the cloth"}
                   for i in ("yellow_brick", "lavender_brick", "pink_brick", "sage_brick")],
       "subgoals": [
           {"index": 1, "acting_hand": "right", "motion": "push", "object_refs": ["lavender_brick", "yellow_brick"],
            "goal": "The lavender brick sits on the cloth directly to the right of the yellow brick, side faces touching."},
           {"index": 2, "acting_hand": "left", "motion": "grasp_lift", "object_refs": ["pink_brick"],
            "goal": "The pink brick is held in the left hand, clear of the cloth."},
           {"index": 3, "acting_hand": "left", "motion": "place", "object_refs": ["pink_brick", "yellow_brick"],
            "goal": "The pink brick rests on top of the yellow brick without hand support."},
           {"index": 4, "acting_hand": "right", "motion": "other", "object_refs": ["sage_brick"],
            "goal": "The sage brick shows its best side to the camera."}]}
RECON = {"objects": [{"id": i, "root_body": f"{i}_{i}", "bodies": [f"{i}_{i}"]} for i in ("yellow_brick", "lavender_brick", "pink_brick", "sage_brick")]}


def build(tmp, push_to=0.04, stack_on="yellow"):
    m = mujoco.MjModel.from_xml_path(str(tmp / "scene.xml"))
    q0 = m.qpos0.copy()
    A = {b: int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in
         ("lavender_brick_lavender_brick", "pink_brick_pink_brick", "robot_left_pad_a", "robot_left_pad_b", "robot_right_pad_a", "robot_right_pad_b")}
    T = 130
    Q = np.repeat(q0[None], T, 0)
    labels = []
    lav = np.r_[np.full(10, 0.1), np.linspace(0.1, push_to, 15), np.full(T - 25, push_to)]
    target = {"yellow": [0.0, 0.0], "lavender": [push_to, 0.0]}[stack_on]
    for t in range(T):
        Q[t, A["lavender_brick_lavender_brick"]] = lav[t]
        # the right pad pushes the lavender brick from its +x side, then backs away
        rx = lav[t] + 0.0248 if t < 26 else 0.3
        Q[t, A["robot_right_pad_a"]:A["robot_right_pad_a"] + 3] = [rx, 0, 0.02]
        # the left pads pinch the pink brick (frames 30-99), lift it, carry it over its support and lower it there
        if t < 40:
            c = [0.2, 0.1, 0.02]
        elif t < 55:
            c = [0.2, 0.1, 0.02 + 0.04 * (t - 40) / 15]
        elif t < 85:
            u = (t - 55) / 30
            c = [0.2 + (target[0] - 0.2) * u, 0.1 + (target[1] - 0.1) * u, 0.06 + 0.01 * np.sin(np.pi * u)]
        else:
            c = [target[0], target[1], 0.06]
        Q[t, A["pink_brick_pink_brick"]:A["pink_brick_pink_brick"] + 3] = c
        gap = 0.0248 if 32 <= t < 100 else 0.05
        Q[t, A["robot_left_pad_a"]:A["robot_left_pad_a"] + 3] = [c[0] - gap, c[1], c[2]]
        Q[t, A["robot_left_pad_b"]:A["robot_left_pad_b"] + 3] = [c[0] + gap, c[1], c[2]]
        labels.append("right: push lavender" if t < 30 else ("left: grip pink" if t < 40 else ("left: lift pink" if t < 55 else
                      ("left: carry pink" if t < 85 else ("left: let go" if t < 120 else "right: home")))))
    f = tmp / f"traj_{push_to}_{stack_on}.npz"
    np.savez(f, qpos=Q, fps=30.0, labels=np.array(labels), subgoal=np.r_[np.ones(30), 2 * np.ones(25), 3 * np.ones(65), 4 * np.ones(10)].astype(int))
    return str(f)


def manifest(tmp, name, edits=None):
    man = {"scene_semantic": "sem.json", "scene_recon": "recon.json", "subgoals": [
        {"index": 1, "stage1_indices": [1], "frames": [0, 29], "motion": "push", "side": "right", "object": "lavender_brick_lavender_brick",
         "expect": {"displacement": {"dir": [-1, 0, 0], "min_m": 0.011}}, "outcome": {"ok": True}},
        {"index": 2, "stage1_indices": [2], "frames": [30, 54], "motion": "grasp_lift", "side": "left", "object": "pink_brick_pink_brick", "outcome": {"ok": True}},
        {"index": 3, "stage1_indices": [3], "frames": [55, 119], "motion": "place", "side": "left", "object": "pink_brick_pink_brick", "outcome": {"ok": True}},
        {"index": 4, "stage1_indices": [4], "frames": [120, 129], "motion": "other", "side": "right", "object": "sage_brick_sage_brick", "outcome": {"ok": True}}]}
    for k, v in (edits or {}).items():
        man["subgoals"][k - 1].update(v)
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(man))
    return str(p)


def run(tmp, man, traj, only=None):
    return stage1_goal_gate(str(tmp / "sem.json"), man, str(tmp / "scene.xml"), traj, subgoals=only)


def row(rep, k):
    return next(r for r in rep["subgoals"] if r["index"] == k)


def seated_inference(tmp, traj):
    """'X is seated flush' names no target: the subgoal's other object is the slot it must be home in."""
    from tool_library.common import task_gates as G
    ep = G.Episode(str(tmp / "scene.xml"), str(traj))
    gc = G.GoalContext(ep, SEM, RECON)
    s1 = {"index": 9, "acting_hand": "right", "motion": "push", "object_refs": ["pink_brick", "yellow_brick"], "goal": "The pink brick is seated flush."}
    preds, _ = G.infer_goal_predicates(gc, s1, [{"object": "pink_brick_pink_brick", "side": "right"}])
    ins = [p for p in preds if p["class"] == "insertion_depth"]
    check("'seated flush' with no target named is an insertion into the subgoal's other object",
          len(ins) == 1 and ins[0].get("B") == "yellow_brick" and ins[0].get("flush"), str(preds))
    s1 = dict(s1, object_refs=["pink_brick"])
    preds, _ = G.infer_goal_predicates(gc, s1, [{"object": "pink_brick_pink_brick", "side": "right"}])
    check("... and nothing is guessed when the subgoal names no other object", not [p for p in preds if p["class"] == "insertion_depth"], str(preds))


def strands_case(tmp):
    """'No strand crosses another' is a strand_crossings predicate on the subgoal's ribbon; judged on its link tree."""
    import mujoco
    import numpy as np
    from tool_library.common import task_gates as G
    cap = '<geom type="capsule" size="0.003 0.008" quat="0.7071 0 0.7071 0" mass="0.001"/>'
    capy = '<geom type="capsule" size="0.003 0.008" quat="0.7071 0.7071 0 0" mass="0.001"/>'

    def chain(prefix, first, step, n, g):
        xml, close = "", ""
        for k in range(n):
            pos = first if k == 0 else step
            xml += f'<body name="{prefix}{k}" pos="{pos}"><joint type="ball"/>{g}'
            close += "</body>"
        return xml + close

    def scene(b_first, b_step, b_geom):
        return (f'<mujoco><worldbody><body name="table" pos="0 0 -0.02"><geom type="box" size="0.5 0.5 0.02"/></body>'
                f'<body name="rib_j" pos="0 0.4 0.004"><freejoint/><geom type="sphere" size="0.004" mass="0.002"/>'
                f'{chain("rib_a", "0.02 0 0", "0.02 0 0", 6, cap)}{chain("rib_b", b_first, b_step, 5, b_geom)}'
                f'{chain("rib_c", "-0.02 0 0", "-0.02 0 0", 6, cap)}</body></worldbody></mujoco>')
    sem = {"objects": [{"id": "ribbon", "name": "ribbon", "type": "deformable"}],
           "subgoals": [{"index": 1, "acting_hand": "right", "motion": "other", "object_refs": ["ribbon"],
                         "goal": "No strand crosses another: the three strands lie fanned out."}]}
    rec = {"objects": [{"id": "ribbon", "root_body": "rib_j"}]}
    res = {}
    for name, (bf, bs, bg) in {"crossed": ("0.06 0.04 0.008", "0 -0.02 0", capy), "fanned": ("0 0.02 0", "0 0.02 0", capy)}.items():
        (tmp / f"rib_{name}.xml").write_text(scene(bf, bs, bg))
        m = mujoco.MjModel.from_xml_path(str(tmp / f"rib_{name}.xml")); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        ep = G.Episode(m, qpos=np.tile(d.qpos, (3, 1)), fps=30.0, labels=["right: wait"] * 3)
        gc = G.GoalContext(ep, sem, rec)
        preds, _ = G.infer_goal_predicates(gc, sem["subgoals"][0], [{"object": "rib_j", "side": "right", "frames": [0, 2]}])
        sc = [q for q in preds if q["class"] == "strand_crossings"]
        rows = [G.evaluate_predicate(gc, sem["subgoals"][0], [{"object": "rib_j", "side": "right", "frames": [0, 2]}], q) for q in sc]
        res[name] = (sc, rows)
    sc, rows = res["crossed"]
    check("'no strand crosses another' infers strand_crossings (max 0) on the subgoal's ribbon",
          len(sc) == 1 and sc[0].get("max_crossings") == 0, str(sc))
    check("a ribbon whose strand lies across another fails it", rows and not rows[0]["ok"] and rows[0]["measured"]["n_crossings"] >= 1
          and rows[0]["measured"]["n_strands"] == 3, str(rows[0].get("measured") if rows else None))
    _, rows = res["fanned"]
    check("the same ribbon fanned out passes", rows and rows[0]["ok"] and rows[0]["measured"]["n_crossings"] == 0, str(rows[0] if rows else None))


def main():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scene.xml").write_text(XML)
    (tmp / "sem.json").write_text(json.dumps(SEM))
    (tmp / "recon.json").write_text(json.dumps(RECON))
    good, short = build(tmp, 0.04), build(tmp, 0.053)
    region = {"class": "target_region", "bodies": {"A": "sage_brick_sage_brick"}, "params": {"region": {"box": [[-0.2, 0.05], [-0.1, 0.15]]}}}

    seated_inference(tmp, good)
    strands_case(tmp)
    r = run(tmp, manifest(tmp, "m1"), good)
    s1, s2, s3, s4 = (row(r, k) for k in (1, 2, 3, 4))
    check("a pick and stack pass; the push's contact cannot waive its unresolved camera direction/side-face clause",
          not s1["ok"] and s2["ok"] and s3["ok"] and set(s1["required_classes"]) >= {"contact"} and set(s2["required_classes"]) >= {"held", "lift_clearance"}
          and set(s3["required_classes"]) >= {"stacked_on", "release_settle"}, json.dumps([s1, s2, s3], default=str))
    check("a goal with no inferable predicate is an issue that asks for goal_predicates",
          (not r["ok"]) and (not s4["ok"]) and any("add goal_predicates" in e for e in s4["errors"]), s4)
    r2 = run(tmp, manifest(tmp, "m2", {4: {"goal_predicates": [region]}}), good)
    check("a generated predicate cannot define success for an ambiguous independent goal", not r2["ok"] and not row(r2, 4)["ok"], r2["issues"])
    trusted_sem = json.loads(json.dumps(SEM))
    trusted_sem["subgoals"][3]["goal_predicates"] = [region]
    (tmp / "sem.json").write_text(json.dumps(trusted_sem))
    r2_trusted = run(tmp, manifest(tmp, "m2_trusted"), good)
    check("a reviewed region predicate cannot certify an unknown visual 'best side' claim", not r2_trusted["ok"] and not row(r2_trusted, 4)["ok"], r2_trusted["issues"])
    r3 = run(tmp, manifest(tmp, "m3", {4: {"goal_predicates": [region]}}), short)
    p1 = row(r3, 1)["predicates"]
    check("the lego miss: 4.7 cm of travel under a 1.1 cm expect, 13 mm short of the brick -> contact fails",
          (not r3["ok"]) and any(p["class"] == "contact" and not p["ok"] and abs(p["measured"]["end_gap_m"] - 0.013) < 0.002 for p in p1), p1)
    disp = {"class": "press_displacement", "bodies": {"A": "lavender_brick_lavender_brick"}, "params": {"body": "lavender_brick_lavender_brick",
            "direction": [-1, 0, 0], "min_m": 0.011, "peak": False}}
    r4 = run(tmp, manifest(tmp, "m4", {1: {"deviation": {"reason": "the brick cannot reach", "predicates": [disp]}}, 4: {"goal_predicates": [region]}}), short, only=[1])
    check("a deviation that replaces 'touches' with a displacement is rejected and the subgoal still fails",
          (not row(r4, 1)["ok"]) and row(r4, 1)["deviation"]["accepted"] is False and any("cannot replace" in e for e in row(r4, 1)["errors"]), row(r4, 1))
    on_lav = build(tmp, 0.04, stack_on="lavender")
    r5 = run(tmp, manifest(tmp, "m5"), on_lav, only=[3])
    dev = {"reason": "the yellow brick is out of the left arm's reach", "predicates": [
        {"class": "stacked_on", "bodies": {"A": "pink_brick_pink_brick", "B": "lavender_brick_lavender_brick"}}]}
    r6 = run(tmp, manifest(tmp, "m6", {3: {"deviation": dev}}), on_lav, only=[3])
    r7 = run(tmp, manifest(tmp, "m7", {3: {"deviation": dict(dev, reason="")}}), on_lav, only=[3])
    check("changing the support remains a failure even with a same-class deviation and a reason",
          (not row(r5, 3)["ok"]) and not row(r6, 3)["ok"] and not row(r6, 3)["deviation"]["accepted"] and not row(r7, 3)["ok"],
          [row(r5, 3), row(r6, 3), row(r7, 3)])
    weak = {"class": "contact", "bodies": {"A": "lavender_brick_lavender_brick", "B": "yellow_brick_yellow_brick"}, "params": {"tol": 0.05}}
    r8 = run(tmp, manifest(tmp, "m8", {1: {"goal_predicates": [weak]}}), short, only=[1])
    pc = next(p for p in row(r8, 1)["predicates"] if p["class"] == "contact" and p.get("source") == "goal_predicates")
    check("an explicit contact predicate with a 5 cm tolerance is capped and still fails", (not pc["ok"]) and pc.get("capped_params") == ["tol"], pc)
    raised = []
    for args in ((str(tmp / "nope.json"), manifest(tmp, "m9"), str(tmp / "scene.xml"), good),
                 (str(tmp / "sem.json"), manifest(tmp, "m9"), str(tmp / "scene.xml"), str(tmp / "none.npz"))):
        try:
            stage1_goal_gate(*args)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a missing Stage 1 file or trajectory raises naming it", raised == ["scene_semantic_json", "trajectory"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
