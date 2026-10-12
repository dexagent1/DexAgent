"""detect_objects' frame selection, with a stand-in for the SAM 3 processor (no model needed).

    python tool_library/tests/test_detect_objects.py

Called without frame_indices on a folder of a few sampled frames, it scanned one frame (20% of 5 rounds to 1, and
that one was the last). The default now reads at least 8 frames (all of them when there are fewer), evenly spaced
from the first to the last, and the scan says which.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tool_library.skills.detect_objects import detect_objects, sample_indices  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_detect_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


class _T:
    """The slice of a torch tensor the tool touches."""
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeProcessor:
    """Finds 'drawer' only in frames whose top-left pixel is bright (the drawer is open there), 'cabinet' everywhere."""
    def __init__(self):
        self.seen = []

    def set_image(self, img):
        self.seen.append(np.asarray(img)[0, 0, 0])
        return {"bright": np.asarray(img)[0, 0, 0] > 128}

    def set_text_prompt(self, prompt, state):
        if prompt == "drawer" and not state["bright"]:
            return {"scores": _T([]), "boxes": _T(np.zeros((0, 4)))}
        return {"scores": _T([0.9]), "boxes": _T([[10, 10, 50, 50]])}


def frames(n, open_at=()):
    d = TMP / f"frames_{n}"
    d.mkdir()
    for i in range(n):
        v = 200 if i in open_at else 40
        Image.fromarray(np.full((32, 32, 3), v, np.uint8)).save(d / f"frame{i:05d}.jpg")
    return d


# 1. a folder of 5 frames, the drawer open only in frame 2: every frame is scanned and the drawer is found
proc = FakeProcessor()
scan = detect_objects(frames(5, open_at={2}), ["cabinet", "drawer"], TMP / "s5.json", processor=proc)
check("5 frames, no frame_indices: all 5 scanned (was 1, the last) and the drawer seen open in frame 2",
      scan["frame_indices"] == [0, 1, 2, 3, 4] and len(proc.seen) == 5 and scan["summary"]["frames_with_hit"]["drawer"] == 1
      and scan["frame_selection"].startswith("default"), f"{scan['frame_indices']} {scan['summary']}")

# 2. a long folder: 20% evenly spaced from the first frame to the last
idx = sample_indices(100)
check("100 frames: 20 scanned, first and last included, evenly spaced", len(idx) == 20 and idx[0] == 0 and idx[-1] == 99
      and max(np.diff(idx)) - min(np.diff(idx)) <= 1, str(idx))
idx = sample_indices(20)
check("20 frames: the floor of 8 applies (20% would be 4)", len(idx) == 8 and idx[0] == 0 and idx[-1] == 19, str(idx))

# 3. explicit frame indices are honoured; ones outside the folder are dropped; all outside is an error
proc = FakeProcessor()
scan = detect_objects(TMP / "frames_5", ["cabinet"], TMP / "s5b.json", frame_indices=[1, 3, 99], processor=proc)
check("given frame_indices: only those inside the folder are scanned", scan["frame_indices"] == [1, 3] and scan["frame_selection"] == "given"
      and len(proc.seen) == 2, str(scan["frame_indices"]))
try:
    detect_objects(TMP / "frames_5", ["cabinet"], TMP / "s5c.json", frame_indices=[50, 60], processor=FakeProcessor())
    check("frame_indices all outside the folder raise ValueError", False)
except ValueError as e:
    check("frame_indices all outside the folder raise ValueError", "outside" in str(e))

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
