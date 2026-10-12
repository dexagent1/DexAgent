"""Smoke tests for the Stage 1 tools: make_contact_sheet and the two scene_semantic verifiers.

    python tool_library/tests/test_semantic.py
"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.common.video import write_frames, write_video  # noqa: E402
from tool_library.skills.make_contact_sheet import make_contact_sheet  # noqa: E402
from tool_library.skills.measure_motion_timeline import measure_motion_timeline  # noqa: E402
from tool_library.verifiers.scene_semantic_consistency_check import scene_semantic_consistency_check  # noqa: E402
from tool_library.verifiers.scene_semantic_schema_check import scene_semantic_schema_check  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_semantic_", dir=os.environ.get("TMPDIR")))
EXAMPLES = ROOT / "stage1_semantic_understanding" / "examples"
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


# ---- a synthetic clip: a moving square, 60 frames at 20 fps
frames = []
for t in range(60):
    f = np.full((120, 160, 3), 200, np.uint8)
    f[40:80, 10 + t * 2:50 + t * 2] = (30, 60, 200)
    frames.append(f)
clip = write_video(frames, TMP / "clip.mp4", fps=20.0)
fdir = write_frames(frames, TMP / "frames")

r = make_contact_sheet(clip, TMP / "a.jpg", count=12, columns=4, tile_width=160)
check("contact sheet: even spacing over the clip", Path(r["sheet"]).exists() and r["grid"] == [3, 4] and r["frames"][0] == 0 and r["frames"][-1] == 59, str(r["frames"]))
r = make_contact_sheet(clip, TMP / "b.jpg", count=6, start=1.0, end=2.0, columns=3, tile_width=100)
check("contact sheet: time window", all(20 <= i <= 40 for i in r["frames"]) and r["grid"] == [2, 3], str(r["frames"]))
r = make_contact_sheet(clip, TMP / "c.jpg", frames=[0, 30, 59], columns=3, crop=[0.0, 0.25, 0.5, 0.75])
check("contact sheet: explicit frames and crop", r["frames"] == [0, 30, 59] and r["grid"] == [1, 3])
r = make_contact_sheet(TMP / "frames", TMP / "d.jpg", count=4, fps=20.0, columns=2)
check("contact sheet: frame folder input", r["n_frames"] == 60 and abs(r["fps"] - 20.0) < 1e-6 and r["grid"] == [2, 2])
try:
    make_contact_sheet(clip, TMP / "e.jpg", frames=[0, 999])
    check("contact sheet refuses frames outside the clip", False)
except ValueError:
    check("contact sheet refuses frames outside the clip", True)
# a pixel crop in another video's frame (the human video's 1920 x 1080 on a smaller sim video) is refused, not clipped
# to a sliver of the wrong region; the same box as fractions works on any size
r = make_contact_sheet(clip, TMP / "f.jpg", frames=[0, 30], columns=2, crop=[40, 30, 120, 90])
check("contact sheet: a pixel crop inside the frame", r["grid"] == [1, 2])
try:
    make_contact_sheet(clip, TMP / "g.jpg", frames=[0, 30], crop=[100, 60, 1500, 1000])
    check("contact sheet refuses a pixel crop running past the frame", False)
except ValueError as e:
    check("contact sheet refuses a pixel crop running past the frame", "fractions" in str(e), str(e)[:90])

# ---- motion timeline on synthetic clips
static = write_video([np.full((120, 160, 3), 200, np.uint8)] * 40, TMP / "static.mp4", fps=20.0)
r = measure_motion_timeline(static, stride=1)
check("motion timeline: a static clip has no active intervals", r["whole_frame"]["intervals_s"] == [] and r["regions"]["left"]["first_active_s"] is None)
r = measure_motion_timeline(clip, stride=1)       # the square moves from the left half into the right half
L, R = r["regions"]["left"], r["regions"]["right"]
check("motion timeline: left region active first, right later", L["first_active_s"] is not None and R["first_active_s"] is not None and L["first_active_s"] < R["first_active_s"], f"{L['first_active_s']} vs {R['first_active_s']}")
two = []
for t_ in range(80):
    f = np.full((120, 160, 3), 200, np.uint8)
    x = 10 + (t_ if t_ < 20 else 20 if t_ < 60 else 20 + (t_ - 60)) * 3
    f[40:80, x:x + 30] = (30, 60, 200)
    two.append(f)
clip2 = write_video(two, TMP / "two.mp4", fps=20.0)
r = measure_motion_timeline(clip2, stride=1, regions={"all": [0, 0, 160, 120]})
iv = r["regions"]["all"]["intervals_s"]
check("motion timeline: two motion phases give two intervals (pixel region)", len(iv) == 2 and iv[0][1] < 1.5 < iv[1][0], str(iv))
spike = [np.full((120, 160, 3), 200, np.uint8) for _ in range(40)]
spike[20][:, :] = 50                                   # one sudden global change
clip3 = write_video(spike, TMP / "spike.mp4", fps=20.0)
r = measure_motion_timeline(clip3, stride=1)
check("motion timeline: a sudden change shows up as the top event", bool(r["events"]) and abs(r["events"][0]["t"] - 1.0) < 0.11, str(r["events"][:2]))
try:
    measure_motion_timeline(clip, regions={"bad": [0.9, 0.9, 0.1, 0.1]}, stride=1)
    check("motion timeline accepts reversed region corners", True)
except ValueError:
    check("motion timeline accepts reversed region corners", False)

# ---- verifiers on the annotated examples (the ones present) and on corrupted copies
refs = sorted(EXAMPLES.glob("*/scene_semantics.json"))
check("found annotated examples", len(refs) >= 3, f"({len(refs)})")
for p in refs:
    a = scene_semantic_schema_check(p); b = scene_semantic_consistency_check(p)
    check(f"schema + consistency pass on {p.parent.name}", a["ok"] and b["ok"], str((a["issues"] + b["issues"])[:2]))

base = json.loads(refs[0].read_text())


def corrupted(mutate, name):
    d = copy.deepcopy(base); mutate(d)
    q = TMP / f"{name}.json"; q.write_text(json.dumps(d)); return q


def m_missing(d): del d["task_summary"]
def m_badtype(d): d["objects"][0]["type"] = "squishy"
def m_badref(d): d["subgoals"][0]["object_refs"] = ["ghost"]
def m_order(d): d["subgoals"][0]["index"] = 7
def m_hand(d): d["subgoals"][0]["acting_hand"] = "left" if d["subgoals"][0]["acting_hand"] == "right" else "right"; d["hands"]["used"] = [d["subgoals"][1]["acting_hand"]] if len(d["subgoals"]) > 1 else []
def m_art(d):
    d["objects"][0]["type"] = "articulated"; d["objects"][0].pop("articulation", None)
def m_prompt(d): d["task_prompt"] = "Juggle the pineapple"
def m_dupid(d): d["objects"].append(copy.deepcopy(d["objects"][0]))

check("schema check flags a missing required key", not scene_semantic_schema_check(corrupted(m_missing, "c1"))["ok"])
check("schema check flags an unknown object type", not scene_semantic_schema_check(corrupted(m_badtype, "c2"))["ok"])
check("consistency flags an unresolved object_ref", any("ghost" in i for i in scene_semantic_consistency_check(corrupted(m_badref, "c3"))["issues"]))
check("consistency flags subgoal indices out of order", any("index" in i for i in scene_semantic_consistency_check(corrupted(m_order, "c4"))["issues"]))
check("consistency flags an acting hand not in hands.used", any("acting_hand" in i or "hands.used" in i for i in scene_semantic_consistency_check(corrupted(m_hand, "c5"))["issues"]))
check("consistency flags an articulated object without articulation", any("articulation" in i for i in scene_semantic_consistency_check(corrupted(m_art, "c6"))["issues"]))
check("consistency flags prompt words no object covers", "pineapple" in scene_semantic_consistency_check(corrupted(m_prompt, "c7"))["uncovered_prompt_words"])
check("consistency flags duplicate ids", any("more than once" in i for i in scene_semantic_consistency_check(corrupted(m_dupid, "c8"))["issues"]))
check("consistency flags windows outside the clip", not scene_semantic_consistency_check(refs[0], video_duration_s=1.0)["ok"])

print(f"\n{'all passed' if not FAILS else str(FAILS) + ' FAILED'}; outputs in {TMP}")
sys.exit(1 if FAILS else 0)
