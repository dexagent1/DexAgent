"""Run Stage 1 on the annotated examples through the configured model and score each result.

    python stage1_semantic_understanding/tests/run_examples.py dexagent.yaml [--only battery_insertion,...]

Needs the API key of the model in dexagent.yaml. Writes runs/<example>/stage1/ and prints a table.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compare_semantics import compare  # noqa: E402


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default=str(REPO / "dexagent.yaml"))
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    only = {x for x in a.only.split(",") if x}
    rows = []
    for d in sorted((HERE.parent / "examples").iterdir()):
        ref = d / "scene_semantics.json"
        if not d.is_dir() or (only and d.name not in only) or not ref.exists():
            continue
        prompt = (d / "prompt.txt").read_text().strip()
        r = subprocess.run([sys.executable, str(HERE.parent / "run_stage1.py"), a.config, "--video", str(d / "video.mp4"),
                            "--prompt", prompt, "--name", f"example_{d.name}"], capture_output=True, text=True)
        out = REPO / "runs" / f"example_{d.name}" / "stage1" / "scene_semantics.json"
        if out.exists():
            c = compare(json.loads(out.read_text()), json.loads(ref.read_text()))
            rows.append((d.name, r.returncode == 0, c["objects"]["matched"], c["objects"]["reference"], c["objects"]["type_correct"],
                         c["subgoals"]["recall"], c["subgoals"]["precision"], c["hands_used_match"]))
        else:
            rows.append((d.name, False, 0, 0, 0, 0.0, 0.0, False))
    print(f"{'example':24s} done  objects(matched/ref, type ok)  subgoal recall/precision  hands")
    for n, ok, m, ref, ty, rc, pr, h in rows:
        print(f"{n:24s} {str(ok):5s} {m}/{ref}, {ty}               {rc:.2f}/{pr:.2f}                 {h}")


if __name__ == "__main__":
    main()
