"""bimanual_sync_gate: two bricks picked by both hands at once vs one after the other, a declared sequential deviation
(with and without a reason), a two-hand step done by one hand, a hand that keeps holding while the other acts, one-hand
subgoals ignored, a simultaneous two-hand manifest entry {motion: 'both', hands: {left, right}} (also read by
stage1_goal_gate through its per-hand sub-entries), misuse."""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.verifiers.bimanual_sync_gate import bimanual_sync_gate  # noqa: E402
from tool_library.verifiers.stage1_goal_gate import stage1_goal_gate  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" {str(detail)[:1200]}"))
    FAILS += 0 if ok else 1


BRICK = '<freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.02"/>'
PAD = '<freejoint/><geom type="box" size="0.005 0.01 0.01" mass="0.01"/>'
XML = f"""<mujoco><worldbody>
<body name="table" pos="0 0 -0.02"><geom type="box" size="0.5 0.5 0.02"/></body>
<body name="pink_brick_pink_brick" pos="-0.15 0.1 0.02">{BRICK}</body>
<body name="sage_brick_sage_brick" pos="0.15 0.1 0.02">{BRICK}</body>
<body name="robot_left_pad_a" pos="-0.3 0.3 0.2">{PAD}</body><body name="robot_left_pad_b" pos="-0.3 0.35 0.2">{PAD}</body>
<body name="robot_right_pad_a" pos="0.3 0.3 0.2">{PAD}</body><body name="robot_right_pad_b" pos="0.3 0.35 0.2">{PAD}</body>
</worldbody></mujoco>"""
SEM = {"task_success": "The pink brick rests on the cloth.",
       "objects": [{"id": "pink_brick", "name": "pink brick"}, {"id": "sage_brick", "name": "sage brick"}],
       "subgoals": [{"index": 1, "acting_hand": "both", "motion": "grasp_lift", "object_refs": ["pink_brick", "sage_brick"],
                     # stage1_goal_gate fails closed on prose outside its supported vocabulary, so the goal names
                     # each object's state; the hand that holds each brick comes from the manifest, not the text
                     "goal": "The pink brick is held and the sage brick is held."},
                    {"index": 2, "acting_hand": "left", "motion": "place", "object_refs": ["pink_brick"], "goal": "The pink brick rests on the cloth."}]}
RECON = {"objects": [{"id": i, "root_body": f"{i}_{i}", "bodies": [f"{i}_{i}"]} for i in ("pink_brick", "sage_brick")]}


def build(tmp, left_span, right_span, name, T=120):
    """Each hand pinches its brick on the frames of its span (pads closed, brick lifted 3 cm)."""
    m = mujoco.MjModel.from_xml_path(str(tmp / "scene.xml"))
    q0 = m.qpos0.copy()
    A = {b: int(m.jnt_qposadr[m.body(b).jntadr[0]]) for b in ("pink_brick_pink_brick", "sage_brick_sage_brick", "robot_left_pad_a",
                                                            "robot_left_pad_b", "robot_right_pad_a", "robot_right_pad_b")}
    Q = np.repeat(q0[None], T, 0)
    for side, brick, x, span in (("left", "pink_brick_pink_brick", -0.15, left_span), ("right", "sage_brick_sage_brick", 0.15, right_span)):
        for t in range(T):
            held = span is not None and span[0] <= t <= span[1]
            c = [x, 0.1, 0.05 if held else 0.02]
            Q[t, A[brick]:A[brick] + 3] = c
            g = 0.0248 if held else 0.06
            Q[t, A[f"robot_{side}_pad_a"]:A[f"robot_{side}_pad_a"] + 3] = [x - g, 0.1, c[2]]
            Q[t, A[f"robot_{side}_pad_b"]:A[f"robot_{side}_pad_b"] + 3] = [x + g, 0.1, c[2]]
    f = tmp / f"{name}.npz"
    np.savez(f, qpos=Q, fps=30.0)
    return str(f)


def manifest(tmp, name, left_frames, right_frames, deviation=None, right_entry=True):
    subs = [{"index": 1, "stage1_indices": [1], "frames": left_frames, "motion": "grasp_lift", "side": "left", "object": "pink_brick_pink_brick"}]
    if right_entry:
        subs.append({"index": 2, "stage1_indices": [1], "frames": right_frames, "motion": "grasp_lift", "side": "right", "object": "sage_brick_sage_brick"})
    if deviation is not None:
        subs[0]["deviation"] = deviation
    subs.append({"index": 3, "stage1_indices": [2], "frames": [110, 119], "motion": "place", "side": "left", "object": "pink_brick_pink_brick"})
    p = tmp / f"{name}.json"
    p.write_text(json.dumps({"scene_recon": "recon.json", "subgoals": subs}))
    return str(p)


def main():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "scene.xml").write_text(XML)
    (tmp / "sem.json").write_text(json.dumps(SEM))
    (tmp / "recon.json").write_text(json.dumps(RECON))
    sem, scene = str(tmp / "sem.json"), str(tmp / "scene.xml")
    together = build(tmp, (10, 60), (20, 70), "together")
    r1 = bimanual_sync_gate(sem, manifest(tmp, "m1", [0, 60], [10, 70]), scene, together)
    check("both hands holding their bricks at the same time pass (the one-hand subgoal is not judged)",
          r1["ok"] and len(r1["subgoals"]) == 1 and r1["subgoals"][0]["overlap_s"] > 1.0, r1)
    seq = build(tmp, (5, 40), (60, 100), "seq")
    r2 = bimanual_sync_gate(sem, manifest(tmp, "m2", [0, 45], [55, 105]), scene, seq)
    check("one hand after the other fails, naming both activity spans",
          (not r2["ok"]) and "one after the other" in r2["issues"][0] and r2["subgoals"][0]["overlap_s"] == 0.0, r2)
    dev = {"reason": "the two arms cannot both reach over the tower", "sequential": True}
    r3 = bimanual_sync_gate(sem, manifest(tmp, "m3", [0, 45], [55, 105], deviation=dev), scene, seq)
    r4 = bimanual_sync_gate(sem, manifest(tmp, "m4", [0, 45], [55, 105], deviation={"sequential": True}), scene, seq)
    check("a sequential deviation cannot waive the simultaneous two-hand requirement", (not r3["ok"]) and (not r4["ok"]), [r3, r4])
    alone = build(tmp, (5, 60), None, "alone")
    r5 = bimanual_sync_gate(sem, manifest(tmp, "m5", [0, 60], None, right_entry=False), scene, alone)
    check("a two-hand step done by the left hand alone fails naming the idle hand", (not r5["ok"]) and "right" in r5["issues"][0], r5)
    # the right hand's own entry comes later, but it already holds its brick while the left hand acts: contact counts
    holds = build(tmp, (5, 50), (0, 100), "holds")
    r6 = bimanual_sync_gate(sem, manifest(tmp, "m6", [0, 50], [60, 100]), scene, holds)
    check("a hand that keeps holding its object while the other acts counts as active", r6["ok"], r6)
    # one simultaneous two-hand entry: per-hand sub-entries carry the objects, the frames are labelled 'left: X | right: Y'
    both = build(tmp, (10, 80), (15, 80), "both")
    z = dict(np.load(both))
    z["labels"] = np.array([f"left: {'lift pink' if 10 <= t <= 80 else 'reach'} | right: {'lift sage' if 15 <= t <= 80 else 'reach'}"
                            for t in range(len(z["qpos"]))])
    np.savez(both, **z)
    bm = tmp / "m_both.json"
    bm.write_text(json.dumps({"scene_recon": "recon.json", "subgoals": [
        {"index": 1, "stage1_indices": [1], "frames": [0, 80], "motion": "both", "sync": "start", "hands": {
            "left": {"motion": "grasp_lift", "object": "pink_brick_pink_brick", "expect": {"left:rise_m": 0.02}},
            "right": {"motion": "grasp_lift", "object": "sage_brick_sage_brick", "expect": {"right:rise_m": 0.02}}}},
        {"index": 2, "stage1_indices": [2], "frames": [110, 119], "motion": "place", "side": "left", "object": "pink_brick_pink_brick"}]}))
    r8 = bimanual_sync_gate(sem, str(bm), scene, both)
    r9 = stage1_goal_gate(sem, str(bm), scene, both)
    held = [p for p in r9["subgoals"][0]["predicates"] if p["class"] == "held"]
    check("a simultaneous 'both' entry is synchronous, and stage1_goal_gate judges each hand's object with that hand",
          # the goal text, the grasp_lift motion and the compositional reading each add a held predicate: every one
          # of them must judge its brick with that brick's hand
          r8["ok"] and r9["ok"] and held and all(p["ok"] for p in held)
          and {(p["A"], p["hand"]) for p in held} == {("pink_brick_pink_brick", "left"), ("sage_brick_sage_brick", "right")},
          [r8, r9["issues"], held])
    raised = []
    for args, kw in (((str(tmp / "none.json"), manifest(tmp, "m7", [0, 60], [10, 70]), scene, together), {}),
                     ((sem, manifest(tmp, "m7", [0, 60], [10, 70]), scene, together), {"min_overlap_s": -1})):
        try:
            bimanual_sync_gate(*args, **kw)
        except ValueError as e:
            raised.append(str(e).split(":")[0])
    check("a missing Stage 1 file and a negative overlap raise", raised == ["scene_semantic_json", "min_overlap_s"], str(raised))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
