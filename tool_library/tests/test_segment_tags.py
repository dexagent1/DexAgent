"""Every plan entry gets its own segment folder in execute_subgoal_plan. Entries sharing a Stage 1 index without a
`step` used to write the same subgoal_NN folder, each overwriting the last, so `resume` read another entry's rollout
(the lego run had to add `step` by hand). Cases: the tag rule (_assign_tags: unique index, explicit steps, missing steps,
duplicate steps, screw bites); a run whose entries share an index writes distinct folders and phase labels; resume keeps
both finished segments (same episode length); an old run's folder (subgoal_NN written by the last entry) is reused for
the entry it really holds and refused for the one it does not; a two-hand entry's folder. Cheap motions on the setup
bench, under a minute.

    MUJOCO_GL=egl python tool_library/tests/test_segment_tags.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common.subgoal import expand_subgoals  # noqa: E402
from tool_library.skills.execute_subgoal_plan import _assign_tags, _label_suffix, _resume_signature, execute_subgoal_plan  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_segtags_", dir=os.environ.get("TMPDIR")))
FAILS = 0
SCENE = str(ROOT / "setup" / "scene.xml")


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    # 1. the rule
    t = _assign_tags([{"index": 3}, {"index": 3}, {"index": 4}, {"index": 5, "step": 1}, {"index": 5, "step": 2},
                      {"index": 6, "step": 1}, {"index": 6}, {"index": 6, "step": 1}, {"index": 7, "step": 2}])
    tags = [x["_tag"] for x in t]
    check("tags: shared index without steps gets step numbers, unique index and explicit steps keep their old names",
          tags == ["subgoal_03_01", "subgoal_03_02", "subgoal_04", "subgoal_05_01", "subgoal_05_02", "subgoal_06_01", "subgoal_06_02",
                   "subgoal_06_03", "subgoal_07_02"] and len(set(tags)) == len(tags), str(tags))
    bites = _assign_tags(expand_subgoals([{"index": 2, "motion": "unscrew", "turns": 0.5, "bite_deg": 90}, {"index": 2, "motion": "unscrew", "turns": 0.5, "bite_deg": 90}]))
    bt = [x["_tag"] for x in bites]
    check("tags: two unscrew entries with one index do not collide bite by bite", len(set(bt)) == 4 and bt[:2] == ["subgoal_02_01", "subgoal_02_02"], str(bt))
    check("label suffixes stay distinct per tag", len({_label_suffix(x) for x in tags}) == len(tags))
    # 2. a run whose entries share an index: distinct folders, distinct phase labels
    plan = {"subgoals": [{"index": 1, "motion": "wait", "side": "right", "duration_s": 0.3},
                         {"index": 1, "motion": "home", "side": "right", "duration_s": 0.4},
                         {"index": 2, "motion": "wait", "side": "left", "duration_s": 0.3}]}
    out = TMP / "run"
    s = execute_subgoal_plan(SCENE, plan, str(out), settle_s=0.0)
    folders = sorted(p.name for p in out.iterdir() if p.is_dir())
    ph = json.loads((out / "phases.json").read_text())
    check("shared index: every entry its own folder", s["ok"] and folders == ["subgoal_01_01", "subgoal_01_02", "subgoal_02"]
          and [r["tag"] for r in s["subgoals"]] == folders, str(folders))
    check("shared index: phases carry the step suffix", any(p_["label"].endswith("#1.1") for p_ in ph) and any(p_["label"].endswith("#1.2") for p_ in ph),
          str([p_["label"] for p_ in ph]))
    # 3. resume keeps both finished segments of the shared index
    s2 = execute_subgoal_plan(SCENE, plan, str(out), resume=True, from_subgoal=2, settle_s=0.0)
    check("resume reuses both segments of the shared index", s2["ok"] and s2["n_frames"] == s["n_frames"], f"{s2.get('n_frames')} vs {s['n_frames']}")
    # 4. an old run: entries 1.1 and 1.2 both wrote subgoal_01; the folder holds the LAST one written (home)
    old = TMP / "old_both"
    shutil.copytree(out, old)
    shutil.rmtree(old / "subgoal_01_01"); (old / "subgoal_01_02").rename(old / "subgoal_01")
    recs = json.loads((old / "outcomes.json").read_text())
    for r in recs:
        r["tag"] = f"subgoal_{int(r['index']):02d}"
    (old / "outcomes.json").write_text(json.dumps(recs))
    rec_home = json.loads((old / "subgoal_01" / "outcome.json").read_text())
    rec_home["tag"] = "subgoal_01"; (old / "subgoal_01" / "outcome.json").write_text(json.dumps(rec_home))
    s3 = execute_subgoal_plan(SCENE, plan, str(old), resume=True, settle_s=0.0)
    check("old layout: a folder holding another entry (home) is not taken for the wait before it", s3["ok"] and (old / "subgoal_01_01").exists()
          and (old / "subgoal_01_02").exists() and s3["n_frames"] == s["n_frames"], str(sorted(p.name for p in old.iterdir() if p.is_dir())))
    # 5. an old run that stopped after entry 1.1: subgoal_01 holds the wait, which resume reuses
    old2 = TMP / "old_first"
    old2.mkdir()
    shutil.copytree(out / "subgoal_01_01", old2 / "subgoal_01")
    r1 = json.loads((old2 / "subgoal_01" / "outcome.json").read_text()); r1["tag"] = "subgoal_01"
    (old2 / "subgoal_01" / "outcome.json").write_text(json.dumps(r1)); (old2 / "outcomes.json").write_text(json.dumps([r1]))
    (old2 / "scene_physics.xml").write_text((out / "scene_physics.xml").read_text())
    s4 = execute_subgoal_plan(SCENE, plan, str(old2), resume=True, settle_s=0.0)
    reused = json.loads((old2 / "outcomes.json").read_text())[0]
    check("old layout: the folder holding the first entry is reused for it", s4["ok"] and reused.get("legacy_folder") == "subgoal_01"
          and not (old2 / "subgoal_01_01").exists() and s4["n_frames"] == s["n_frames"], str(reused.get("legacy_folder")))
    with np.load(out / "trajectory.npz", allow_pickle=False) as first, np.load(old2 / "trajectory.npz", allow_pickle=False) as resumed:
        check("legacy resume preserves physical states and controls", all(np.array_equal(first[k], resumed[k])
              for k in ("qpos", "qvel", "ctrl", "substep_qpos", "substep_ctrl", "initial_state", "final_state")))
    # Changed plans invalidate the affected segment and every later segment,
    # while retaining the identical physical prefix.
    changed = json.loads(json.dumps(plan))
    changed["subgoals"][1]["duration_s"] = .8
    first_path = out / "subgoal_01_01/rollout_runtime.npz"
    first_mtime = first_path.stat().st_mtime_ns
    previous_count = s2["n_frames"]
    changed_result = execute_subgoal_plan(SCENE, changed, str(out), resume=True)
    check("changed duration automatically recomputes the affected suffix", changed_result["ok"]
          and changed_result["n_frames"] > previous_count and first_path.stat().st_mtime_ns == first_mtime,
          f"{previous_count} -> {changed_result.get('n_frames')}")
    pose_path = TMP / "signature_pose.json"
    pose_path.write_text('{"pos":[0,0,0]}')
    entry = {"motion": "grasp_lift", "pose": str(pose_path), "hands": {"left": {"pose": str(pose_path)}}}
    before_sig = _resume_signature(entry, [], {"fps": 30}, "scene")
    pose_path.write_text('{"pos":[0,0,1]}')
    check("changing a pose file invalidates its cached segment", before_sig != _resume_signature(entry, [], {"fps": 30}, "scene"))
    check("control and scene changes invalidate cached segments",
          _resume_signature(entry, [], {"fps": 30}, "scene") != _resume_signature(entry, [], {"fps": 60}, "scene")
          and _resume_signature(entry, [], {}, "scene") != _resume_signature(entry, [], {}, "other_scene"))
    # A controller/state-adapter fix must not reuse an earlier implementation's
    # cached outcome, even when its scene and task plan stayed the same.
    stamp = first_path.stat().st_mtime_ns
    with patch("tool_library.skills.execute_subgoal_plan._execution_implementation_digest", return_value="new-code-version"):
        rerun = execute_subgoal_plan(SCENE, changed, str(out), resume=True)
    check("implementation changes force a fresh physical rollout", rerun["ok"] and first_path.stat().st_mtime_ns != stamp)
    # Older unsigned outcomes and planning-only passes cannot stand in for a
    # physical execution, even if stale rollout files happen to remain nearby.
    for mutation in ("unsigned", "dry_run"):
        recs = json.loads((out / "outcomes.json").read_text())
        if mutation == "unsigned": recs[0].pop("resume_signature")
        else: recs[0]["dry_run"] = True
        (out / "outcomes.json").write_text(json.dumps(recs))
        stamp = first_path.stat().st_mtime_ns
        rerun = execute_subgoal_plan(SCENE, changed, str(out), resume=True)
        check(f"{mutation} outcome forces a fresh physical rollout", rerun["ok"] and first_path.stat().st_mtime_ns != stamp)
    # 6. a two-hand entry sharing an index with a one-hand entry: separate folders
    plan2 = {"subgoals": [{"index": 1, "motion": "both", "hands": {"left": {"motion": "wait", "duration_s": 0.2}, "right": {"motion": "wait", "duration_s": 0.4}}},
                          {"index": 1, "motion": "home", "side": "left", "duration_s": 0.3}]}
    s5 = execute_subgoal_plan(SCENE, plan2, str(TMP / "run_both"), settle_s=0.0)
    check("a two-hand entry and a one-hand entry with one index: separate folders", s5["ok"] and [r["tag"] for r in s5["subgoals"]] == ["subgoal_01_01", "subgoal_01_02"]
          and s5["subgoals"][0]["side"] == "both", str([(r["tag"], r["side"], r["ok"], r.get("failed")) for r in s5["subgoals"]]))
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
