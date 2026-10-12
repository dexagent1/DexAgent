"""stage4_randomization_gate (and stage4_handoff_check with draws) on synthetic Stage 4 task folders.

The stack task of stage4_robot_data_generation/tests/synthetic.py (a free hand stacks A on B and C on A) is
randomizable as written; three broken variants must be refused with the reason named: a wall that every draw
pushes A into (penetration), a config whose A rests on B while its hand sets it down relative to the world
(object_steps), and a cycle of resting relations (refused before drawing).

    python tool_library/tests/test_stage4_randomization_gate.py
"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage4_robot_data_generation" / "tests"))

import synthetic  # noqa: E402
from tool_library.verifiers.stage4_handoff_check import stage4_handoff_check  # noqa: E402
from tool_library.verifiers.stage4_randomization_gate import stage4_randomization_gate  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_s4gate_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    # 1. the stack task as written: layouts are kept
    synthetic.stack_task(TMP / "good")
    g3 = stage4_randomization_gate(str(TMP / "good"), draws=3)
    g10 = stage4_randomization_gate(str(TMP / "good"), draws=10, min_fraction=0.5)
    check("by default two thirds of the draws must pass (a margin, not one lucky layout)", g3["min_accepted"] == 2 and g3["ok"]
          and g10["min_accepted"] == 5 and g3["methods"].get("warp", 0) == g3["accepted"], f"{g3['accepted']}/3 need {g3['min_accepted']}")
    g = stage4_randomization_gate(str(TMP / "good"), draws=5, min_accepted=3)
    check("feasible task passes", g["ok"] and g["accepted"] >= 3 and not g["issues"] and g["relations"].get("C") == "rests_on A",
          f"{g['accepted']}/{g['draws']} in {g['seconds_per_draw']} s/draw")

    # 2. a fixed wall 1 cm to the right of A while A's range moves it 2-3 cm right: every draw collides
    objs = json.loads((TMP / "good" / "stage4.json").read_text())["randomize"]["objects"]
    wall = "<body name='wall' pos='0.135 0.40 0.03'><geom name='wall' type='box' size='0.005 0.05 0.03' rgba='0.5 0.5 0.5 1'/></body>"
    pushed = copy.deepcopy(objs)
    pushed["A"].update(dx=[0.02, 0.03], dy=[0.0, 0.0])
    synthetic.stack_task(TMP / "wall", objects=pushed, extra_xml=wall)
    w = stage4_randomization_gate(str(TMP / "wall"), draws=3)
    check("every draw into a wall is refused, pair named", not w["ok"] and w["accepted"] == 0 and w["rejections"].get("penetration") == 3
          and any("wall" in i and "penetration" in i for i in w["issues"]), w["issues"][1][:140] if len(w["issues"]) > 1 else str(w["issues"]))

    # 3. A rests on B but its set-down is anchored to the world: A jumps whenever B moves
    segs = synthetic.stack_segments()
    segs["right"][2] = {"phases": ["right: carry A #2"], "blend": ["A", "world"]}
    segs["right"][3] = {"phases": ["right: set down A #2", "right: let go of A #2", "right: retreat #2"], "anchor": "world"}
    segs["right"][4] = {"phases": ["right: reach for C #3"], "blend": ["world", "C"]}
    synthetic.stack_task(TMP / "jump", segments=segs)
    j = stage4_randomization_gate(str(TMP / "jump"), draws=3)
    check("an object that would jump at its release is refused", not j["ok"] and j["rejections"].get("object_steps", 0) == 3
          and any("object_steps" in i and "rests_on" in i for i in j["issues"]) and j["warnings"], str(j["rejections"]))

    # 4. a cycle of resting relations: refused before drawing, reported (not raised)
    cyc = copy.deepcopy(objs)
    cyc["A"]["rests_on"], cyc["C"]["rests_on"] = "C", "A"
    synthetic.stack_task(TMP / "cycle", objects=cyc)
    c = stage4_randomization_gate(str(TMP / "cycle"), draws=2)
    check("a resting cycle is reported", not c["ok"] and "cycle" in c["issues"][0], c["issues"][0][:100])

    # 5. misuse raises, a missing config is a report
    bad = []
    for kw in ({"draws": 0}, {"draws": 2, "min_accepted": 3}):
        try:
            stage4_randomization_gate(str(TMP / "good"), **kw)
        except ValueError as e:
            bad.append(str(e))
    miss = stage4_randomization_gate(str(TMP / "nowhere"))
    check("misuse raises ValueError, missing folder reports", len(bad) == 2 and not miss["ok"] and "stage4.json" in miss["issues"][0], str(bad))

    # 6. the handoff check runs the gate when asked for draws
    h_good = stage4_handoff_check(str(TMP / "good"), draws=3)
    h_wall = stage4_handoff_check(str(TMP / "wall"), draws=3)
    h_cfg = stage4_handoff_check(str(TMP / "wall"))
    check("stage4_handoff_check(draws) catches a task Stage 4 cannot randomize",
          h_good["ok"] and h_good["summary"]["randomization"]["accepted"] >= 1 and not h_wall["ok"]
          and any(i.startswith("stage4 randomization") for i in h_wall["issues"]) and h_cfg["ok"],
          f"good {h_good['ok']} wall {h_wall['ok']} wall-without-draws {h_cfg['ok']}")
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
