"""Run every test script of the repository, each in its own process, and print one line per script.

    python tool_library/tests/run_all.py            # all of them (CPU; about an hour on 4 cores)
    python tool_library/tests/run_all.py grasp      # only the scripts whose name contains "grasp"

Logs go to $DEXAGENT_TEST_LOGS (default: <temp dir>/dexagent_tests). Missing
dependencies and skipped checks are reported separately and never counted as
passing. Exit 1 means failures, exit 2 means incomplete coverage without failures.
"""
import os
import subprocess
import sys
import tempfile
import time
import re
import json
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = (sorted(REPO.glob("tool_library/tests/test_*.py")) + sorted(REPO.glob("agent/tests/test_*.py"))
           + sorted(REPO.glob("stage4_robot_data_generation/tests/test_*.py")))
TIMEOUT_S = 2400


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filter", nargs="?", default="")
    parser.add_argument("--start-at", help="Start at this script stem; earlier scripts remain untested in this run.")
    args = parser.parse_args()
    selected = [p for p in SCRIPTS if args.filter in p.stem]
    if args.start_at:
        starts = [i for i, p in enumerate(selected) if p.stem == args.start_at]
        if len(starts) != 1:
            parser.error("--start-at must identify exactly one selected script")
        selected = selected[starts[0]:]
    logs = Path(os.environ.get("DEXAGENT_TEST_LOGS") or Path(tempfile.gettempdir()) / "dexagent_tests")
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
    failed, skipped, passed, records = [], [], [], []
    def checkpoint(running=None):
        # Preserve completed evidence if a job is interrupted during a long script.
        report = {"passed": passed, "failed": failed, "incomplete": skipped,
                  "records": records, "selected": [str(p.relative_to(REPO)) for p in selected],
                  "running": running, "complete": len(records) == len(selected) and running is None}
        temp = logs / "summary.json.tmp"
        temp.write_text(json.dumps(report, indent=1))
        temp.replace(logs / "summary.json")
    checkpoint()
    for s in selected:
        t0 = time.time()
        log = logs / f"{s.stem}.log"
        checkpoint(str(s.relative_to(REPO)))
        try:
            with log.open("w") as stream:
                rc = subprocess.run([sys.executable, str(s)], cwd=REPO, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        text = log.read_text(errors="replace")
        bad = rc != 0 or "Traceback" in text or "[FAIL]" in text
        # script-style tests print [SKIP] / SKIP; unittest reports "OK (skipped=N)"
        skip = bool(re.search(r"(?im)^\s*(?:\[SKIP\]|SKIP\b|skipping\b)|^OK \(.*\bskipped=\d+", text))
        status = "FAIL" if bad else "INCOMPLETE" if skip else "PASS"
        elapsed = round(time.time() - t0, 2)
        print(f"{status}  {s.relative_to(REPO)}  ({elapsed:.0f} s)", flush=True)
        records.append({"script": str(s.relative_to(REPO)), "status": status, "returncode": rc,
                        "seconds": elapsed, "log": str(log)})
        if bad:
            failed.append(s.stem)
        elif skip:
            skipped.append(s.stem)
        else:
            passed.append(s.stem)
        checkpoint()
    print(f"\n{len(passed)} passed, {len(failed)} failed, {len(skipped)} incomplete; logs in {logs}")
    return 1 if failed else 2 if skipped or not records else 0


if __name__ == "__main__":
    sys.exit(main())
