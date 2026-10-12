"""stage4_physics_replay_gate on the real bench (setup/scene.xml compiled to the rigid arm-hand runtime, with a complete
servo recording as Stage 3 hands it off) with a box and a small cube resting on it.

The robot holds its home pose; the manifest asks the cube to stay put and at rest, and Stage 1's goal is that the cube
rests on top of the box. Stage 4 moves the box 4.5-5 cm:
  - cube attached_to the box: it rides along, the replay holds every check -> pass;
  - cube left where it was: the kinematic checks see nothing (no new penetration, the cube just hovers), the physics
    replay sees it fall -> fail, naming untouched:cube;
  - the same task with an explicit 1 mm layout: the cube is still supported -> pass;
  - a cube that floats 2 cm above the box in the ORIGINAL recording: the original itself fails in physics -> fail, named;
  - a task folder without a manifest: reported.

    python tool_library/tests/test_stage4_physics_replay_gate.py
"""
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
from tool_library.verifiers.stage4_physics_replay_gate import stage4_physics_replay_gate  # noqa: E402
from tool_library.verifiers.stage4_randomization_gate import stage4_randomization_gate  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_s4phys_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    synthetic.rigid_bench_task(TMP / "attached", attached=True)
    a = stage4_physics_replay_gate(str(TMP / "attached"))
    check("cube riding its box holds in physics", a["ok"] and not a["issues"] and a["physics"]["ok"] and a["original"]["not_held"] == [],
          f"{a.get('seconds')} s, layout {a.get('layout')} {a['issues']}")

    synthetic.rigid_bench_task(TMP / "left", attached=False)
    k = stage4_randomization_gate(str(TMP / "left"), draws=2, min_accepted=2)
    p = stage4_physics_replay_gate(str(TMP / "left"))
    check("cube left over thin air: kinematics pass, physics fails", k["ok"] and not p["ok"] and "subgoal 1 hold: untouched:cube" in p["physics"]["failed"]
          and any("untouched:cube" in i for i in p["issues"]), f"kinematic {k['accepted']}/2 {k['issues']}, physics failed {(p.get('physics') or {}).get('failed')}")

    # Stage 1's goals are judged on every replayed layout, whatever the entry's own expectation asks: with a weak
    # expectation (nothing but the object) a layout that leaves the cube off its box still fails "rests on top of the box"
    g_task = TMP / "goal"
    synthetic.rigid_bench_task(g_task, attached=False, expect={"object": "cube"})
    g = stage4_physics_replay_gate(str(g_task))
    fails = (g.get("physics") or {}).get("failed") or []
    check("a layout that leaves the cube off its box fails Stage 1's goal even under a weak expectation",
          not g["ok"] and any(f.startswith("Stage 1 goal 1") for f in fails) and g["original"]["not_held"] == [], f"failed {fails}")
    gn = stage4_physics_replay_gate(str(g_task), layout={"objects": {"box": [0.001, 0.0, 0.0]}})
    check("... and a layout that keeps it on the box holds the goal", gn["ok"], str(gn.get("issues"))[:160])

    s = stage4_physics_replay_gate(str(TMP / "left"), layout={"objects": {"box": [0.001, 0.0, 0.0]}})
    check("a 1 mm layout keeps the cube supported", s["ok"] and s["layout"]["box"] == [0.001, 0.0, 0.0], str(s["issues"])[:120])

    synthetic.rigid_bench_task(TMP / "mixed", attached=False, base_dx=(0.0, 0.06))
    mx = stage4_physics_replay_gate(str(TMP / "mixed"), layouts=4)
    check("several layouts: passes when one holds, counts them", mx["ok"] and mx["layouts_replayed"] == 4 and 1 <= mx["layouts_held"] < 4,
          f"{mx.get('layouts_held')}/{mx.get('layouts_replayed')} held")

    # the recording itself drops the cube 2 cm onto the box: the original fails its own expectation in the replay
    synthetic.rigid_bench_task(TMP / "floating", attached=True, cube_z=0.0599)
    f = stage4_physics_replay_gate(str(TMP / "floating"))
    check("an original that does not hold up in physics is named", not f["ok"] and f["original"]["not_held"]
          and "original layout fails" in f["issues"][0], f["issues"][0][:120])

    synthetic.rigid_bench_task(TMP / "nomanifest", attached=True)
    cfg = json.loads((TMP / "nomanifest" / "stage4.json").read_text())
    cfg.pop("manifest")
    (TMP / "nomanifest" / "stage4.json").write_text(json.dumps(cfg))
    n = stage4_physics_replay_gate(str(TMP / "nomanifest"))
    check("a task folder without a manifest is reported", not n["ok"] and any("manifest" in i for i in n["issues"]), str(n["issues"])[:120])

    bad = []
    for kw in ({"draws": 0}, {"layout": [0.0, 0.0]}, {"layouts": 0}):
        try:
            stage4_physics_replay_gate(str(TMP / "attached"), **kw)
        except ValueError as e:
            bad.append(str(e))
    check("misuse raises ValueError", len(bad) == 3, str(bad))
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
