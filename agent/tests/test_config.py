"""Run configuration: stage budgets, an EgoDex episode's video, the `stage4` block reaching Stage 4's task file, and
the Stage 1 prompt test of example reuse.

    python agent/tests/test_config.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.config import egodex_video, load_config  # noqa: E402
from agent.reuse import _same_prompt  # noqa: E402
from run_dexagent import apply_stage4_settings  # noqa: E402

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


tmp = Path(tempfile.mkdtemp(prefix="dexagent_config_", dir=os.environ.get("TMPDIR")))
cfg = load_config(ROOT / "dexagent.yaml")
check("every agent stage has its own round budget", all(int(cfg[s]["max_rounds"]) > 0 for s in ("stage1", "stage2", "stage3")),
      str([cfg[s]["max_rounds"] for s in ("stage1", "stage2", "stage3")]))

(tmp / "ep").mkdir()
(tmp / "ep/7.hdf5").write_bytes(b""); (tmp / "ep/7.mp4").write_bytes(b"")
egodex_video(cfg, tmp / "ep/7.hdf5")
check("an EgoDex episode's .mp4 is the video when none is given", cfg["inputs"]["video"] == str((tmp / "ep/7.mp4").resolve()), cfg["inputs"]["video"])
y = tmp / "run.yaml"
y.write_text(f"inputs:\n  video: {tmp / 'own.mp4'}\n  egodex_hdf5: {tmp / 'ep/7.hdf5'}\n")
(tmp / "own.mp4").write_bytes(b"")
check("a video that exists is kept", load_config(y)["inputs"]["video"] == str((tmp / "own.mp4").resolve()))
y.write_text(f"inputs:\n  egodex_hdf5: {tmp / 'ep/7.hdf5'}\n")
check("egodex_hdf5 in the YAML alone gives the video", load_config(y)["inputs"]["video"] == str((tmp / "ep/7.mp4").resolve()))

task = tmp / "stage4.json"
task.write_text(json.dumps({"num_samples": 10, "seed": 0, "cameras": {"views": {"wrist_left": {}, "wrist_right": {}, "top": {}}},
                            "render": {"stride": 2, "samples": 64}}))
c = apply_stage4_settings(task, {"num_samples": 3, "seed": 7, "views": ["wrist_right"], "render": {"samples": 16, "chunks": 4}})
c2 = json.loads(task.read_text())
check("the stage4 block reaches the task file", c2["num_samples"] == 3 and c2["seed"] == 7 and list(c2["cameras"]["views"]) == ["wrist_right"]
      and c2["render"] == {"stride": 2, "samples": 16, "chunks": 4}, json.dumps(c2))

ex = ROOT / "examples/default/place_cup_in_bowl"
if (ex / "manifest.json").exists():
    check("the example's own prompt matches", _same_prompt("Put the blue cup into the green bowl.", ex))
    check("another task's prompt does not", not _same_prompt("Stack the red blocks on the tray.", ex))
print("\nALL PASS" if not FAILS else f"\n{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
