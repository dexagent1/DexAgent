"""scene_semantic_consistency_check's prompt coverage looks at the prompt's NOUNS only.

Both EgoDex Stage 1 agents had to reword their summaries because verbs and adverbs of the prompt ('build', 'freely',
'keeping', 'connected') were reported as uncovered object words. These cases use the two real prompts with summaries
that never say those words, and check that a missing object noun is still caught.

    python tool_library/tests/test_scene_semantic_prompt_nouns.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.scene_semantic_consistency_check import prompt_nouns, scene_semantic_consistency_check  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_sem_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def doc(prompt, summary, objects):
    objs = [{"id": o, "name": n, "type": "rigid", "role": "manipulated", "evidence": "visible 0 to 3 s", "properties": ["rigid"],
             "appearance": a} for o, n, a in objects]
    d = {"task_prompt": prompt, "task_summary": summary, "environment": {"support_surface": "table"},
         "hands": {"used": ["right"]}, "objects": objs,
         "subgoals": [{"index": 1, "acting_hand": "right", "action": "move them", "object_refs": [o for o, _, _ in objects],
                       "hand_roles": {"right": "moves the objects"}, "time_window_s": [0.0, 3.0]}]}
    p = TMP / f"s{len(list(TMP.iterdir()))}.json"
    p.write_text(json.dumps(d))
    return p


# 1. the lego prompt: 'build' and 'freely' are not object words
r = scene_semantic_consistency_check(doc("Stack lego bricks to build freely in any form on a blue tablecloth.",
                                         "The hands stack eight lego bricks into a tower on the tablecloth.",
                                         [("brick", "lego brick", "blue, yellow and pink 2x2 bricks")]))
check("lego prompt: 'build' / 'freely' / 'form' need no rewording", r["ok"] and not r["uncovered_prompt_words"],
      f"checked {r['checked_prompt_words']}, uncovered {r['uncovered_prompt_words']}")

# 2. the rings prompt: 'keeping' and 'connected' are not object words; 'ring' (an -ing noun) is still checked
r = scene_semantic_consistency_check(doc("Use both hands to pull apart two magnetic rings while keeping them connected to the third ring.",
                                         "Both hands pull two magnetic rings apart; the third hangs between them.",
                                         [("ring", "magnetic ring", "black and yellow")]))
check("rings prompt: 'keeping' / 'connected' / 'apart' need no rewording", r["ok"] and "ring" in r["checked_prompt_words"],
      f"checked {r['checked_prompt_words']}, uncovered {r['uncovered_prompt_words']}")

# 3. a missing object noun is still caught (the scene has a cup, the prompt a pineapple and a bowl)
r = scene_semantic_consistency_check(doc("Carefully put the pineapple into the bowl", "The hand places the cup.", [("cup", "cup", "white")]))
check("a missing object noun is flagged", not r["ok"] and set(r["uncovered_prompt_words"]) == {"pineapple", "bowl"}, str(r["uncovered_prompt_words"]))

# 4. the noun heuristic on its own: verbs at the head of a clause, participles, adverbs out; determiner-led -ing/-ed nouns in
cases = {
    "Juggle the pineapple": ["pineapple"],
    "Open the drawer, place the plush toy inside and close it": ["drawer", "plush"],
    "Place the drawing on the shelf": ["drawing", "shelf"],
    "Tie a knot in the string while holding the spring gently": ["string", "spring"],
    "Using the pipette, transfer the liquid to the dish": ["pipette", "liquid", "dish"],
    "Slowly slide the sealed envelope under the folded towel": ["envelope", "towel"],
}
bad = {p: (prompt_nouns(p), want) for p, want in cases.items() if prompt_nouns(p) != want}
check("noun heuristic on six prompts", not bad, str(bad))

# A missing user prompt is not a task containing the words 'prompt' and 'video'.
r = scene_semantic_consistency_check(doc("", "A hand places a cup.", [("cup", "cup", "blue")]), prompt="")
check("absent user prompt needs no boilerplate objects or summary", r["ok"] and not r["checked_prompt_words"], str(r["issues"]))

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
