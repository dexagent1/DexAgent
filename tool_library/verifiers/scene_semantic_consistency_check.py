"""Is a scene_semantics.json internally consistent, grounded in the clip, and faithful to the prompt?"""
from __future__ import annotations

TOOL = {
    "name": "scene_semantic_consistency_check",
    "title": "Scene semantic consistency check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Cross-field checks on a Stage 1 scene_semantics.json: unique ids, subgoal references that resolve, consecutive ordered subgoals with time windows inside the clip, acting hands that match the hand roles and the hands list, type-specific blocks present, evidence that cites times, and the prompt's nouns covered by the objects.",
    "description": (
        "Reads the JSON and checks what a schema cannot: every object id is unique and every subgoal object_ref "
        "(id or id:part) resolves to an object (and a declared part when parts are listed); subgoal indices run "
        "1..N in order; time windows are ordered, start no later than the next subgoal's start, and lie inside the "
        "clip when its duration is known; the acting hand of each subgoal is in hands.used and its hand_roles text "
        "is not a resting phrase; every hand in hands.used acts in at least one subgoal; articulated objects carry an "
        "articulation block whose moving and fixed parts are named parts, deformable objects a deformation block; "
        "a group object's `pieces` has an integer count of at least 2 and a piece kind, and is not typed articulated; "
        "every object is referenced by some subgoal unless its role is static; every evidence string cites a time or "
        "frame; and each content NOUN of the prompt (4+ letters) shares a stem with some word in the objects' names, parts, "
        "appearance or properties, the task summary, or the environment description. Only nouns are checked: function words, "
        "verbs, adverbs and common adjectives are on a stop list, and a word is taken for a verb or modifier by its form (-ly "
        "adverbs, -ing and -ed participles, unless a determiner precedes them or they are known nouns such as 'ring' or "
        "'drawing') or its place (a clause-initial or post-'to' word followed by a determiner, as in 'build a tower'), so "
        "'build freely' or 'keeping them connected' need no rewording while a missing object ('the pineapple') is still "
        "caught. Each finding names the JSON path and what to change."
    ),
    "inputs": [
        {"name": "path", "type": "path", "description": "The scene_semantics.json to check."},
        {"name": "prompt", "type": "str | None", "description": "The task prompt; default: the file's task_prompt field."},
        {"name": "video_duration_s", "type": "float | None", "description": "Clip length for the time-window check; default: the file's video.duration_s when present."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; uncovered_prompt_words; checked_prompt_words (the nouns the coverage check looked for); unreferenced_objects; hands_without_subgoals."},
    ],
    "status": "tested",
    "entrypoint": "scene_semantic_consistency_check",
    "requirements": [],
    "tags": ["stage 1", "semantic", "consistency", "subgoals", "prompt coverage"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.verifiers.scene_semantic_consistency_check runs/my_task/stage1/scene_semantics.json",
}

import json
import re
import sys
from pathlib import Path

STOP = {"the", "and", "then", "into", "onto", "from", "with", "that", "this", "them", "your", "please", "after", "before",
        "while", "there", "here", "over", "under", "down", "back", "side", "right", "left", "both", "each", "every", "some",
        "about", "using", "take", "make", "have", "will", "where", "when", "which", "what", "them", "their", "than", "very",
        "also", "just", "only", "again", "away", "around", "through", "between", "inside", "outside", "front", "behind",
        "table", "hand", "hands", "first", "second", "third", "three", "four", "five",
        # imperative verbs: the summary paraphrases them, the objects need not
        "pick", "place", "move", "open", "close", "remove", "insert", "write", "erase", "wipe", "pour", "mount", "take",
        "grab", "grasp", "hold", "lift", "push", "pull", "press", "turn", "unscrew", "screw", "fold", "draw", "stack",
        "nest", "carry", "drop", "release", "rotate", "slide", "put", "bring", "keep", "return", "finish", "start",
        # scene and posture words that describe the setting, not an object
        "placed", "sitting", "standing", "lying", "while", "background", "against", "wall", "floor", "room"}
# more words that are never an object: function words, verbs (base and inflected forms the suffix rules miss), adverbs,
# common adjectives of size, shape, state and manner, and abstract nouns a prompt uses to say HOW, not WHAT
STOP |= {
    "about", "above", "across", "along", "among", "another", "anything", "apart", "below", "beneath", "beside", "besides",
    "beyond", "could", "does", "done", "each", "either", "else", "enough", "even", "ever", "everything", "following",
    "former", "from", "further", "hers", "herself", "himself", "into", "itself", "least", "less", "many", "more", "most",
    "much", "must", "near", "neither", "next", "none", "nothing", "once", "other", "others", "ours", "ourselves", "past",
    "perhaps", "rather", "same", "several", "should", "since", "something", "such", "theirs", "themselves", "these",
    "those", "though", "thus", "till", "toward", "towards", "until", "upon", "unless", "various", "within", "without",
    "would", "yours", "yourself", "whose", "whom", "whether", "onto", "into", "also", "any", "all", "together", "upward",
    "upwards", "downward", "downwards", "forward", "forwards", "backward", "backwards", "sideways", "afterwards", "later",
    "soon", "still", "already", "almost", "quite", "really", "instead", "otherwise", "anyway", "somewhere", "everywhere",
    "anywhere", "nowhere", "twice", "half", "fully", "freely", "first", "last", "final", "fourth", "fifth", "sixth",
    "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "single", "double", "pair", "couple", "both",
    # verbs and their forms
    "build", "builds", "built", "assemble", "assembles", "attach", "attaches", "detach", "detaches", "connect", "connects",
    "disconnect", "separate", "separates", "join", "joins", "align", "aligns", "fill", "fills", "empty", "empties",
    "clean", "cleans", "spray", "sprays", "cut", "cuts", "tie", "ties", "untie", "unstack", "unstacks", "stacks",
    "pack", "packs", "unpack", "unpacks", "load", "loads", "unload", "plug", "plugs", "unplug", "charge", "charges",
    "spin", "spins", "twist", "twists", "flip", "flips", "shake", "shakes", "squeeze", "squeezes", "tap", "taps",
    "use", "uses", "used", "using", "make", "makes", "made", "keep", "keeps", "kept", "leave", "leaves", "left", "let",
    "lets", "get", "gets", "got", "give", "gives", "gave", "set", "sets", "go", "goes", "went", "come", "comes", "came",
    "see", "sees", "look", "looks", "show", "shows", "try", "tries", "want", "wants", "need", "needs", "help", "helps",
    "begin", "begins", "continue", "continues", "stop", "stops", "end", "ends", "reach", "reaches", "touch", "touches",
    "hang", "hangs", "rest", "rests", "sit", "sits", "stand", "stands", "lie", "lies", "lay", "lays", "fit", "fits",
    "pulls", "pushes", "presses", "turns", "lifts", "holds", "grabs", "grasps", "picks", "places", "puts", "moves",
    "opens", "closes", "removes", "inserts", "writes", "erases", "wipes", "pours", "mounts", "takes", "rotates",
    "slides", "drops", "releases", "brings", "returns", "finishes", "starts", "carries", "folds", "draws", "nests",
    "screws", "unscrews", "fidget", "fidgets", "play", "plays", "arrange", "arranges", "sort", "sorts", "match",
    "matches", "swap", "swaps", "replace", "replaces", "transfer", "transfers", "stir", "stirs", "mix", "mixes",
    "knot", "knots", "wrap", "wraps", "unwrap", "roll", "rolls", "drag", "drags", "toss", "tosses", "throw", "throws",
    "catch", "catches", "balance", "balances", "seat", "seats", "snap", "snaps", "lock", "locks", "unlock", "unlocks",
    "clip", "clips", "hook", "hooks", "hang", "attach", "store", "stores", "organize", "organizes", "tidy", "gather",
    "gathers", "collect", "collects", "split", "splits", "divide", "divides", "combine", "combines", "lower", "lowers",
    "raise", "raises", "tilt", "tilts", "shift", "shifts", "adjust", "adjusts", "position", "positions", "orient",
    "orients", "cover", "covers", "uncover", "uncovers", "seal", "seals", "unseal", "remain", "remains", "stay", "stays",
    # adverbs and adjectives (manner, size, shape, state)
    "gently", "slowly", "quickly", "carefully", "firmly", "tightly", "loosely", "neatly", "fast", "slow", "gentle",
    "careful", "firm", "tight", "loose", "neat", "small", "large", "little", "big", "tall", "short", "long", "wide",
    "narrow", "thin", "thick", "heavy", "light", "full", "round", "square", "flat", "straight", "upright", "vertical",
    "horizontal", "upside", "inside", "outside", "middle", "center", "centre", "near", "nearby", "closer", "farther",
    "different", "similar", "whole", "entire", "remaining", "free", "loose", "open", "closed", "empty", "clean", "dirty",
    "new", "old", "same", "correct", "proper", "desired", "specific", "certain", "possible", "available", "visible",
    # abstract nouns that say how, not what
    "form", "forms", "shape", "shapes", "way", "ways", "manner", "order", "kind", "kinds", "type", "types", "style",
    "pattern", "arrangement", "configuration", "direction", "position", "motion", "movement", "action", "task", "time",
    "times", "step", "steps", "process", "fashion", "state", "goal", "end",
}
_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "its", "their", "his", "her", "my", "your", "our",
                "each", "every", "some", "any", "all", "both", "two", "three", "four", "five", "another", "other", "one",
                "several", "many", "few", "no"}
# -ing / -ed / -ly words that are nouns (objects) all the same
_ING_NOUNS = {"ring", "rings", "string", "strings", "spring", "springs", "thing", "things", "wing", "wings", "king", "sling",
              "swing", "ceiling", "building", "drawing", "drawings", "painting", "paintings", "clothing", "bedding",
              "stuffing", "filling", "frosting", "icing", "topping", "toppings", "wiring", "tubing", "railing", "siding",
              "earring", "earrings", "coupling", "bearing", "bearings", "fitting", "fittings", "housing", "casing",
              "lining", "padding", "packing", "pudding", "dumpling", "dumplings", "seedling", "sapling", "awning",
              "opening", "handling", "binding", "molding", "moulding", "sibling", "stocking", "stockings", "napkin"}
_ED_NOUNS = {"bed", "beds", "shed", "sled", "seed", "seeds", "reed", "weed", "speed", "steed", "feed", "embed", "bobsled",
             "flatbed", "seabed", "hundred", "sacred", "breed", "creed", "tweed"}
_LY_NOUNS = {"assembly", "jelly", "belly", "supply", "supplies", "family", "lily", "doily", "butterfly", "dragonfly",
             "anomaly", "dolly", "trolly", "holly", "rally", "ally", "gully", "tally", "monopoly", "firefly", "housefly",
             "reply", "fly", "jellyfish"}
_CLAUSE_BREAK = {"and", "then", "or", "but", "to", "so", "while", "after", "before", "once", "until", "when", "please"}


def prompt_nouns(prompt: str) -> list:
    """The prompt's words that name things (content nouns, 4+ letters): stop words, verbs, adverbs and modifiers out.
    Rules: the stop list; -ly words (adverbs) unless known nouns; -ing / -ed words (participles: 'keeping', 'connected',
    'the folded towel') unless they head a noun phrase after a determiner or are known nouns ('ring', 'drawing'); and a
    clause-initial word, or one
    after 'to', that a determiner follows ('build a tower', 'juggle the ball') is the clause's verb."""
    toks = re.findall(r"[a-z]+|[.,;:!?]", str(prompt).lower())
    out = []
    for i, w in enumerate(toks):
        if not w.isalpha() or len(w) < 4 or w in STOP:
            continue
        prev = toks[i - 1] if i > 0 else "."
        nxt = toks[i + 1] if i + 1 < len(toks) else "."
        # a participle is a noun only as the head of its phrase: after a determiner and not before another content word
        # ('the drawing on the shelf' yes, 'the folded towel' no)
        head_noun = prev in _DETERMINERS and not (nxt.isalpha() and nxt not in STOP and nxt not in _DETERMINERS)
        if w.endswith("ly") and w not in _LY_NOUNS:
            continue
        if (w.endswith("ing") or w.endswith("ings")) and w not in _ING_NOUNS and not head_noun:
            continue
        if w.endswith("ed") and w not in _ED_NOUNS and not head_noun:
            continue
        if (not prev.isalpha() or prev in _CLAUSE_BREAK) and nxt in _DETERMINERS:
            continue                                   # 'build a ...', 'to keep the ...': the clause's verb
        out.append(w)
    return out


# a role that BEGINS by saying the hand is inactive; mid-sentence "rests" may describe an object
REST = re.compile(r"^\s*(rests?|resting|idle|does nothing|not involved|not used|stays (still|flat|idle|put)|no role|inactive|remains (still|idle|flat))\b", re.I)
TIME = re.compile(r"\d+(\.\d+)?\s*(s\b|sec|second|frame|#)|\bframes?\s*\d+|\d+\s*(to|-|–)\s*\d+(\.\d+)?\s*s", re.I)


def _stems(text: str) -> set:
    return {w[:4] for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4}


def scene_semantic_consistency_check(path, prompt=None, video_duration_s=None):
    p = Path(path)
    issues, info = [], {}
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        return {"ok": False, "issues": [f"{p}: cannot read JSON: {e}"]}
    objs = d.get("objects") or []
    subs = d.get("subgoals") or []
    hands_used = set((d.get("hands") or {}).get("used") or [])
    prompt = prompt if prompt is not None else d.get("task_prompt", "")
    dur = video_duration_s if video_duration_s is not None else (d.get("video") or {}).get("duration_s")

    # objects
    ids = [o.get("id") for o in objs]
    for i, o in enumerate(objs):
        if ids.count(o.get("id")) > 1:
            issues.append(f"$.objects[{i}].id: {o.get('id')!r} is used more than once")
        parts = o.get("parts") or []
        if o.get("type") == "articulated":
            a = o.get("articulation")
            if not a:
                issues.append(f"$.objects[{i}]: articulated object {o.get('id')!r} needs an articulation block (moving_part, fixed_part, joint_type, axis, state_over_video)")
            elif parts:
                for k in ("moving_part", "fixed_part"):
                    v = str(a.get(k, ""))
                    if not any(pt in v for pt in parts):
                        issues.append(f"$.objects[{i}].articulation.{k}: {v!r} does not name one of parts {parts}")
            if not parts:
                issues.append(f"$.objects[{i}].parts: an articulated object lists its parts (at least the moving and the fixed part)")
        if o.get("type") == "deformable" and not o.get("deformation"):
            issues.append(f"$.objects[{i}]: deformable object {o.get('id')!r} needs a deformation block (kind, how_the_task_deforms_it)")
        if not TIME.search(str(o.get("evidence", ""))):
            issues.append(f"$.objects[{i}].evidence: cite the times or frames that show this object (for example '2.6 to 5 s')")
        if o.get("role") in ("manipulated", "tool") and not o.get("properties"):
            issues.append(f"$.objects[{i}].properties: a manipulated object needs the properties the reconstruction must show")
        pc = o.get("pieces")
        if isinstance(pc, dict):
            # a group object: several separate pieces of one kind (Stage 2 builds each as its own body)
            if o.get("type") == "articulated":
                issues.append(f"$.objects[{i}].pieces: a group of separate pieces is not one articulated mechanism; list the "
                              "mechanism as its own object, or describe the pieces (type of one piece) without articulation")
            cnt = pc.get("count")
            if not isinstance(cnt, int) or isinstance(cnt, bool) or cnt < 2:
                issues.append(f"$.objects[{i}].pieces.count: the number of pieces at the first frame, an integer of at least 2 "
                              "(set at_least: true when only a lower bound can be counted); one piece is not a group")
            if not str(pc.get("kind") or "").strip():
                issues.append(f"$.objects[{i}].pieces.kind: the singular noun for one piece (block, card, brick)")

    # subgoals
    referenced = set()
    acting = set()
    prev_start = None
    for i, s in enumerate(subs):
        if s.get("index") != i + 1:
            issues.append(f"$.subgoals[{i}].index: expected {i + 1}, got {s.get('index')}")
        for r in s.get("object_refs") or []:
            oid, _, part = str(r).partition(":")
            referenced.add(oid)
            o = next((x for x in objs if x.get("id") == oid), None)
            if o is None:
                issues.append(f"$.subgoals[{i}].object_refs: {r!r} is not an object id")
            elif part and o.get("parts") and part not in o["parts"]:
                issues.append(f"$.subgoals[{i}].object_refs: part {part!r} is not in {oid}.parts {o['parts']}")
        h = s.get("acting_hand")
        roles = s.get("hand_roles") or {}
        for hh in (("left", "right") if h == "both" else (h,)):
            if hh not in hands_used:
                issues.append(f"$.subgoals[{i}].acting_hand: {h!r} but hands.used is {sorted(hands_used)}")
            acting.add(hh)
            if hh in roles and REST.search(str(roles.get(hh, ""))):
                issues.append(f"$.subgoals[{i}].hand_roles.{hh}: the acting hand is described as resting: {roles.get(hh)!r}")
        tw = s.get("time_window_s")
        if tw:
            a, b = float(tw[0]), float(tw[1])
            if b < a:
                issues.append(f"$.subgoals[{i}].time_window_s: end {b} before start {a}")
            if dur and (a < -0.01 or b > float(dur) + 0.5):
                issues.append(f"$.subgoals[{i}].time_window_s: [{a}, {b}] is outside the clip (0 to {dur} s)")
            if prev_start is not None and a + 1e-6 < prev_start:
                issues.append(f"$.subgoals[{i}].time_window_s: starts at {a} s, before subgoal {i} ({prev_start} s); subgoals are in order of occurrence")
            prev_start = a
        if len(str(s.get("action", "")).split(". ")) > 2:
            issues.append(f"$.subgoals[{i}].action: one motion per subgoal; split '{s.get('action')}'")
    for h in hands_used - acting:
        issues.append(f"$.hands.used: {h!r} is listed but acts in no subgoal")
    unref = [o.get("id") for o in objs if o.get("id") not in referenced and o.get("role") != "static"]
    for oid in unref:
        issues.append(f"$.objects: {oid!r} is not referenced by any subgoal; either a subgoal involves it or its role is 'static'")

    # prompt coverage
    hay = " ".join([d.get("task_summary", ""), json.dumps(d.get("environment", {}))] + [json.dumps(o.get(k, "")) for o in objs for k in ("name", "parts", "appearance", "properties", "pieces")])
    hay_stems = _stems(hay)
    words = prompt_nouns(prompt)
    uncovered = sorted({w for w in words if w[:4] not in hay_stems})
    for w in uncovered:
        issues.append(f"$.objects: prompt word {w!r} is not reflected in any object name, part, appearance, property or the summary; add the object it refers to or explain it in the summary")

    info.update({"uncovered_prompt_words": uncovered, "checked_prompt_words": sorted(set(words)), "unreferenced_objects": unref, "hands_without_subgoals": sorted(hands_used - acting),
                 "n_objects": len(objs), "n_subgoals": len(subs)})
    return {"ok": not issues, "issues": issues, **info}


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if not a:
        print(__doc__); return 2
    r = scene_semantic_consistency_check(a[0], a[1] if len(a) > 1 else None)
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
