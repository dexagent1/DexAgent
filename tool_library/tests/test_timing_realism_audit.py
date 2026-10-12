"""timing_realism_audit against Stage 1 time windows: a robot two to four times slower than the person passes; a
skipped subgoal, a subgoal out of proportion, a rushed subgoal and an absurd episode length fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import json
import numpy as np
from tool_library.verifiers.timing_realism_audit import timing_realism_audit

tmp = S.tmpdir()
sem = {"video": {"duration_s": 6.0, "fps": 30, "frames": 180},
       "subgoals": [{"index": 1, "time_window_s": [0.5, 2.0]}, {"index": 2, "time_window_s": [2.0, 3.0]},
                    {"index": 3, "time_window_s": [3.0, 5.0]}]}
semp = tmp / "scene_semantics.json"; semp.write_text(json.dumps(sem))
man = {"subgoals": [{"index": 1, "stage1_indices": [1]}, {"index": 2, "stage1_indices": [2]}, {"index": 3, "stage1_indices": [3]}]}
manp = tmp / "trajectory_manifest.json"; manp.write_text(json.dumps(man))


def episode(durs_s, name):
    sg = np.concatenate([np.full(int(round(d * 30)), i + 1) for i, d in enumerate(durs_s) if d > 0])
    Q = np.zeros((len(sg), 3))
    return S.save(tmp / f"{name}.npz", Q, [f"right: phase #{k}" for k in sg], subgoal=sg)


C = S.Cases()


def paced_passes():
    r = timing_realism_audit(None, episode([4.5, 3.0, 6.0], "ok"), scene_semantic=str(semp), manifest=str(manp))
    assert r["ok"], r["issues"]
    assert abs(r["median_ratio"] - 3.0) < 0.01 and r["episode"]["ratio"] == 2.25, r


def skipped_subgoal_fails():
    r = timing_realism_audit(None, episode([4.5, 0, 6.0], "skip"), scene_semantic=str(semp), manifest=man)
    assert not r["ok"] and any("no robot frames" in s for s in r["issues"]), r["issues"]


def out_of_proportion_and_rushed_fail():
    r = timing_realism_audit(None, episode([4.5, 60.0, 6.0], "stall"), scene_semantic=sem, manifest=man)
    assert not r["ok"] and any("median" in s for s in r["issues"]), r["issues"]
    r = timing_realism_audit(None, episode([4.5, 0.1, 6.0], "rush"), scene_semantic=sem, manifest=man)
    assert not r["ok"] and any("fewer than" in s for s in r["issues"]) and any("faster" in s for s in r["issues"]), r["issues"]


def absurd_episode_fails_and_inputs_are_optional():
    r = timing_realism_audit(None, episode([150.0, 100.0, 150.0], "long"), scene_semantic=sem)
    assert not r["ok"] and any("x66" in s for s in r["issues"]), r["issues"]
    assert any("manifest" in s for s in r["skipped"]), r["skipped"]
    r = timing_realism_audit(None, episode([4.5, 3.0, 6.0], "bare"))
    assert r["ok"] and len(r["skipped"]) == 2, r


def misuse():
    S.expect_value_error(lambda: timing_realism_audit(None, episode([1, 1, 1], "m"), ratio_band=(2, 1)), "ratio_band")
    S.expect_value_error(lambda: timing_realism_audit(None, episode([1, 1, 1], "m2"), scene_semantic=str(tmp / "nope.json")), "scene_semantic")


C.run("a robot three times slower than the person passes", paced_passes)
C.run("a Stage 1 subgoal without robot frames fails", skipped_subgoal_fails)
C.run("a 60 s subgoal among 4-6 s ones and a 0.1 s subgoal fail", out_of_proportion_and_rushed_fail)
C.run("a 400 s episode for a 6 s video fails; missing inputs are skipped and named", absurd_episode_fails_and_inputs_are_optional)
C.run("an inverted band and a missing semantic file are refused", misuse)
C.exit()
