"""The agent can end a stage as blocked (report_blocked) only after finish returned verifier failures; the stage then
fails with the reason in failure_report.json instead of spending the rest of its budget.

    python agent/tests/test_report_blocked.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DEXAGENT_QUIET", "1")

import numpy as np  # noqa: E402

from agent.config import load_config  # noqa: E402
from agent.llm import MockLLM  # noqa: E402
from agent.runner import StageRunner  # noqa: E402
from tool_library.common.video import write_video  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def failing_verifier(ws):
    return {"ok": False, "issues": ["bead_pile: ... Stage 1 must be revised (re-run Stage 1)"]}


failing_verifier.__name__ = "hand_object_contact_consistency_check"
tmp = Path(tempfile.mkdtemp(prefix="dexagent_blocked_", dir=os.environ.get("TMPDIR")))
cfg = load_config(ROOT / "dexagent.yaml")
cfg["run"]["output_dir"] = str(tmp)
# The runner pins inputs.video by its digest when it is constructed, so the fixture needs a real (tiny) source video
# instead of the placeholder path in dexagent.yaml.
write_video([np.zeros((32, 32, 3), dtype=np.uint8)] * 3, tmp / "video.mp4", fps=30)
cfg["inputs"]["video"] = str(tmp / "video.mp4")

turns = [
    [{"type": "tool_use", "name": "report_blocked", "input": {"reason": "too early"}}],
    [{"type": "tool_use", "name": "finish", "input": {"summary": "done"}}],
    [{"type": "tool_use", "name": "report_blocked", "input": {"reason": "Stage 1 subgoal 5 reads an occlusion as a pick"}}],
    [{"type": "tool_use", "name": "finish", "input": {"summary": "should never be reached"}}],
]
llm = MockLLM(turns)
ws = tmp / "stage2"
out = StageRunner(cfg, llm, ws, "test stage", [failing_verifier], max_rounds=10).run("go")
first = json.dumps(llm.seen[-1]) if llm.seen else ""          # the whole conversation (the list grows in place)
check("report_blocked before any finish is refused", "call finish first" in first, first[:120])
check("after a failed finish it ends the stage as failed, early", not out["finished"] and out["rounds"] == 3 and out["failure"].startswith("blocked:"),
      json.dumps({k: out[k] for k in ("finished", "rounds", "failure")}))
rep = json.loads((ws / "failure_report.json").read_text()) if (ws / "failure_report.json").exists() else {}
check("failure_report.json carries the reason and the last verdict", "occlusion" in rep.get("reason", "") and rep.get("last_verdict", {}).get("ok") is False,
      str(rep.get("reason")))

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
