"""Cross-stage repair: a stage that finds an earlier stage wrong ends with report_blocked(upstream=..., evidence=...);
run_dexagent.py re-opens that stage with the reason in its first message, re-runs the stages after it and continues, at
most repair.max_loops_per_stage times per stage, and records every loop. Scripted with MockLLM (one script per stage
and per run); the stage verifiers are replaced by a marker file the scripts write, so no model, scene or GPU is needed.

    python agent/tests/test_cross_stage_repair.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEXAGENT_QUIET", "1")

import run_dexagent  # noqa: E402
import agent.artifacts as delivery  # noqa: E402
import stage1_semantic_understanding.run_stage1 as S1  # noqa: E402
import stage2_scene_reconstruction.run_stage2 as S2  # noqa: E402
import stage3_trajectory_generation.run_stage3 as S3  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.llm import MockLLM  # noqa: E402
from agent.runner import StageRunner  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def marker_verifier(ws):
    """Passes once the stage's script wrote verified.txt into its workspace."""
    return {"ok": (Path(ws) / "verified.txt").exists(), "issues": [] if (Path(ws) / "verified.txt").exists() else ["deliverable not verified"]}


marker_verifier.__name__ = "marker_check"
for mod in (S1, S2, S3):
    mod.make_verifiers = lambda *a, **k: [marker_verifier]
S3.make_diagnosis = lambda *a, **k: (lambda ws, verdict: "Diagnosis (test): upstream_suspect -> report_blocked(upstream='stage2')")
# This fixture tests repair routing only; actual sealed delivery has separate
# physical integration tests. Keep its publication boundary explicit.
delivery.publish_stage3 = lambda *a, **k: {"ok": True, "issues": [], "test_stub": "repair routing"}


def T(name, **inp):
    return [{"type": "tool_use", "name": name, "input": inp}]


def write(ws, rel, text):
    return T("write_text", path=str(Path(ws) / rel), text=text)


def start_messages(ws):
    """The first user message of every run of a stage, from its trace."""
    out, seen = [], set()
    paths = [Path(ws) / "trace.jsonl"] + list((Path(ws) / ".candidates").glob("*/trace.jsonl"))
    for p in sorted((p for p in paths if p.exists()), key=lambda p: p.stat().st_mtime_ns):
        for ln in p.read_text().splitlines():
            r = json.loads(ln)
            # Stage 3 copies its last attempt's trace beside its outputs: the same start record, counted once
            if r.get("event") == "start" and ln not in seen:
                seen.add(ln)
                out.append(r.get("user", ""))
    return out


def setup(tmp):
    tiny_video(tmp / "video.mp4")
    y = tmp / "run.yaml"
    y.write_text(f"run:\n  name: repair\n  output_dir: {tmp / 'runs'}\nexamples:\n  reuse: false\ninputs:\n  video: {tmp / 'video.mp4'}\n"
                 "  task: move the bead\nstage3:\n  max_rounds: 12\n  rounds_per_subgoal: 2\n  max_rounds_cap: 30\n")
    cfg = load_config(y)
    ws = {s: Path(cfg["run"]["output_dir"]) / "repair" / s for s in ("stage1", "stage2", "stage3")}
    ws["stage2"].mkdir(parents=True)
    (ws["stage2"] / "scene.xml").write_text('<mujoco><worldbody><geom type="plane" size="1 1 .1"/></worldbody></mujoco>')
    return y, ws


SEM = json.dumps({"task_prompt": "move the bead", "subgoals": [{"index": 1, "acting_hand": "right", "motion": "grasp_lift", "object_refs": ["bead"]},
                                                              {"index": 2, "acting_hand": "right", "motion": "place", "object_refs": ["bead"]}]})


def tiny_video(path):
    """A real (3-frame) source video with one bright square: the runner pins inputs.video by its digest when it is
    constructed, and Stage 2 segments the square through the harness for its source-reference receipts."""
    import numpy as np
    from tool_library.common.video import write_video
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((64, 64, 3), 40, dtype=np.uint8)
    frame[20:44, 20:44] = 230
    write_video([frame] * 3, path, fps=30)
    return path


def source_reference(ws, video):
    """Stage 2 turns that mint trusted source-reference receipts the real way (an approved tool called directly through
    the harness) and declare them in scene_recon.json; the runner checks these before any stage verifier."""
    ref = Path(ws) / "ref"
    recon = {"scene_xml": "scene.xml", "frame0": "ref/source.png", "objects": [{"id": "bead", "mask": "ref/mask.png"}]}
    return [T("segment_source_frame", video=str(video), frame_index=0, box=[14, 14, 50, 50], out_dir=str(ref)),
            write(ws, "scene_recon.json", json.dumps(recon))]


def runner_checks(tmp):
    cfg = load_config(ROOT / "dexagent.yaml")
    cfg["run"]["output_dir"] = str(tmp / "runner")
    cfg["inputs"]["video"] = str(tiny_video(tmp / "runner" / "video.mp4"))

    def failing(ws):
        return {"ok": False, "issues": ["hand_object_contact_consistency_check: the fingertips stop 13 cm short of bead_pile"]}

    turns = [T("finish", summary="done"),
             T("report_blocked", reason="x", upstream="stage2", evidence="e"),
             T("report_blocked", reason="Stage 1 subgoal 5 reads an occlusion as a pick", upstream="stage1"),
             T("report_blocked", reason="Stage 1 subgoal 5 reads an occlusion as a pick", upstream="stage1",
               evidence="the right fingertips stay 8-13 cm above the table in frames 69-117")]
    llm = MockLLM(turns)
    ws = tmp / "runner" / "stage2"
    out = StageRunner(cfg, llm, ws, "test stage", [failing], max_rounds=10, stage="stage2").run("go")
    conv = json.dumps(llm.seen[-1])
    check("an upstream that is not an earlier stage is refused", "not an earlier stage" in conv)
    check("an upstream without evidence is refused", "upstream needs evidence" in conv)
    rep = json.loads((ws / "failure_report.json").read_text())
    check("report_blocked with upstream and evidence ends the stage; the run result and failure_report.json carry both",
          not out["finished"] and out.get("upstream", {}).get("stage") == "stage1" and rep.get("upstream") == "stage1"
          and "8-13 cm" in rep.get("upstream_evidence", "") and out["rounds"] == 4,
          json.dumps({k: out.get(k) for k in ("finished", "rounds", "upstream")}))


def loop_checks(tmp):
    y, ws = setup(tmp / "loop")
    script = {
        "stage1": [[*[write(ws["stage1"], "scene_semantics.json", SEM)], write(ws["stage1"], "verified.txt", "1"), T("finish", summary="s1")],
                   [T("finish", summary="s1 revised: subgoal 5 removed")]],
        "stage2": [[*source_reference(ws["stage2"], tmp / "loop" / "video.mp4"), T("finish", summary="s2"),
                    T("report_blocked", reason="subgoal 5 names bead_pile but the hand never reaches it", upstream="stage1",
                      evidence="fingertips 13.3 cm short of bead_pile at frames 69-117")],
                   [write(ws["stage2"], "verified.txt", "1"), T("finish", summary="s2 again")],
                   [T("finish", summary="s2 re-opened by stage 3: plug added")]],
        "stage3": [[T("finish", summary="s3"),
                    T("report_blocked", reason="the cable asset has no plug", upstream="stage2",
                      evidence="hand scene body cable_last has 1 collision geom, the bench scene 2")],
                   [T("write_text", path="verified.txt", text="1"), T("finish", summary="s3 done")]],
    }
    mock = tmp / "loop" / "mock.json"
    mock.write_text(json.dumps(script))
    rc = run_dexagent.main([str(y), "--to", "stage3", "--llm", f"mock:{mock}", "--yes"])
    summ = json.loads((Path(ws["stage1"]).parent / "run_summary.json").read_text())
    loops = summ.get("repair_loops") or []
    s1 = start_messages(ws["stage1"])
    s2 = start_messages(ws["stage2"])
    s3 = start_messages(ws["stage3"])
    check("the run continues after two repair loops (stage 2 re-opens stage 1, stage 3 re-opens stage 2) and ends ok",
          rc == 0 and summ["ok"] and [(lp["requested_by"], lp["stage"], lp["resolved"]) for lp in loops] == [("stage2", "stage1", True), ("stage3", "stage2", True)],
          json.dumps([(lp["requested_by"], lp["stage"], lp.get("resolved")) for lp in loops]))
    check("the re-opened stage's first message carries the reason and the evidence",
          len(s1) == 2 and "REPAIR REQUEST" in s1[1] and "bead_pile" in s1[1] and "13.3 cm" in s1[1] and "REPAIR" not in s1[0]
          and len(s2) == 3 and "REPAIR REQUEST" in s2[2] and "no plug" in s2[2], f"stage1 runs {len(s1)}, stage2 runs {len(s2)}")
    check("the stages after a re-opened one are told it was revised", "UPSTREAM REVISED" in s2[1] and len(s3) == 2 and "UPSTREAM REVISED" in s3[1],
          f"stage3 runs {len(s3)}")
    check("stage 3's budget scales with the Stage 1 subgoals (12 + 2 x 2) and is in its first message", "Round budget: 16 rounds" in s3[0], s3[0][-300:])
    check("the blocked stage's failure report is archived per loop; no run-level failure report on success",
          (ws["stage2"] / "failure_report_loop1.json").exists() and (ws["stage3"] / "failure_report_loop2.json").exists()
          and not (Path(ws["stage1"]).parent / "failure_report.json").exists())
    check("notes are used once (renamed), so a later run of the stage does not see them again",
          not any((ws[s] / n).exists() for s in ws for n in ("repair_request.json", "upstream_revised.json"))
          and (ws["stage1"] / "repair_request_loop1.used.json").exists())


def bounded_checks(tmp):
    y, ws = setup(tmp / "bounded")
    script = {
        "stage1": [[write(ws["stage1"], "scene_semantics.json", SEM), write(ws["stage1"], "verified.txt", "1"), T("finish", summary="s1")]],
        "stage2": [[*source_reference(ws["stage2"], tmp / "bounded" / "video.mp4"), write(ws["stage2"], "verified.txt", "1"),
                    T("finish", summary="s2")],
                   [T("finish", summary="s2 re-opened")]],
        "stage3": [[T("finish", summary="s3"), T("report_blocked", reason="phone asset too thin", upstream="stage2", evidence="9 mm")],
                   [T("finish", summary="s3 again"), T("report_blocked", reason="still no plug", upstream="stage2", evidence="still 1 geom")]],
    }
    mock = tmp / "bounded" / "mock.json"
    mock.write_text(json.dumps(script))
    rc = run_dexagent.main([str(y), "--to", "stage3", "--llm", f"mock:{mock}", "--yes"])
    fr = Path(ws["stage1"]).parent / "failure_report.json"
    rep = json.loads(fr.read_text()) if fr.exists() else {}
    loops = rep.get("repair_loops") or []
    check("a stage is re-opened at most once: the second request is refused and the run fails",
          rc == 1 and len(loops) == 2 and loops[0]["resolved"] is False and "already re-opened" in (loops[1].get("refused") or ""),
          json.dumps([(lp["stage"], lp.get("resolved"), lp.get("refused")) for lp in loops]))
    check("the run's failure_report.json names the failed stage, its reason and upstream, and records every loop",
          rep.get("failed_stage") == "stage3" and "still no plug" in str(rep.get("reason")) and (rep.get("upstream") or {}).get("stage") == "stage2"
          and Path(rep.get("stage_failure_report") or "").exists())


def main():
    tmp = Path(tempfile.mkdtemp(prefix="dexagent_repair_", dir=os.environ.get("TMPDIR")))
    runner_checks(tmp)
    loop_checks(tmp)
    bounded_checks(tmp)
    print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
