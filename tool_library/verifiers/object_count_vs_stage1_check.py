"""Object count vs Stage 1 check: the scene has as many slots, drawers, latches, studs, cups or bricks as Stage 1 says."""
from __future__ import annotations

TOOL = {
    "name": "object_count_vs_stage1_check",
    "title": "Object count vs Stage 1 check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Reads the counts Stage 1 states for each object ('two slots', 'four flaps', 'a 3-slot holder', 'studs in a 2x2 grid', 'three drives', and parts lists such as near_slot/far_slot) and counts the same things in the scene (constructor parameters in scene_recon.json, Stage 1 objects of that kind present in the scene, or named parts inside the object), failing on a mismatch.",
    "description": (
        "A 3-slot battery holder once came out with 2 slots; a meal box lid needs its four latch flaps, a stacking "
        "task all of its bricks. For each Stage 1 object, counts are read from its name, appearance, size, properties "
        "and initial state: a number word or digit (one..twelve, single, pair of) followed within three words by a "
        "countable noun (slot, drawer, compartment, latch, flap, stud, hole, cell, tube, well, bay, drive, ring, cup, "
        "brick, block, door, shelf, hook, peg, pin, button, battery, lid, tray, ...), 'N-noun' compounds, 'AxB grid' "
        "next to a noun, and the parts list (parts whose name contains the noun). Several counts of one noun take the "
        "largest (the text also mentions subsets: 'fills one slot'). The scene count is, in order: an integer "
        "constructor parameter of the object (n_<noun>s, num_<noun>s, <noun>s, <noun>_count, or <noun>s_x * "
        "<noun>s_y), the number of Stage 1 objects named with that noun that exist in the scene, or the number of "
        "distinct named parts inside the object carrying the noun or a synonym (latch/flap/strap, drawer/tray, "
        "slot/bay, drive/hdd/disk). A count that cannot be found in the scene is reported as unverified, not failed. "
        "A group object (Stage 1 `pieces`: N separate pieces of one kind) is counted by the pieces its scene_recon.json "
        "entry declares under `parts` (at least N when Stage 1 says at_least); in the task-level counts a group counts "
        "its pieces toward its piece kind (Stage 1 `pieces.kind`, else the noun most of its part bodies are named with) "
        "and, since a task sentence then names some of a group's pieces, the scene must have at least as many as stated."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "semantic", "type": "str | dict", "description": "scene_semantics.json (Stage 1)."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json (objects, bodies and constructor params)."},
        {"name": "nouns", "type": "list[str] | None", "description": "Extra countable nouns (singular)."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; counts [{object, noun, stage1, phrases, scene, source, verdict}]; unverified."},
    ],
    "requirements": ["mujoco>=3.2"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "count", "stage1", "consistency", "parts", "stage2"],
    "status": "tested",
    "entrypoint": "object_count_vs_stage1_check",
}

import json
import re

from tool_library.common import piece_groups as PG
from tool_library.common import scene_assert as SA

PART_NOUNS = ["slot", "drawer", "compartment", "latch", "flap", "stud", "hole", "cell", "well", "bay", "door", "shelf", "hook", "peg",
              "pin", "button", "strap", "clip", "leg", "wheel", "knob", "socket", "port", "cavity", "tab", "hinge"]
OBJECT_NOUNS = ["cup", "brick", "block", "ring", "drive", "battery", "bowl", "marker", "cube", "tube", "bottle", "plate", "box", "ball",
                "can", "jar", "pen", "coin", "card", "disk", "lid", "mug", "glass", "toy", "sponge"]
SYN = {"flap": ["flap", "latch", "strap"], "latch": ["latch", "flap", "clip", "strap"], "drawer": ["drawer", "tray"],
       "slot": ["slot", "bay"], "bay": ["bay", "slot"], "drive": ["drive", "hdd", "disk"], "brick": ["brick", "block", "lego"],
       "block": ["block", "brick", "cube"], "cup": ["cup", "mug"], "cube": ["cube", "block"]}
WORDS = {"one": 1, "single": 1, "two": 2, "pair of": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12}
UNITS = {"inch", "inches", "mm", "cm", "m", "metre", "metres", "meter", "meters", "degree", "degrees", "percent", "times", "s", "seconds",
         "second", "kg", "g", "handed", "fingered", "way"}
FOLLOWERS = {"holder", "case", "rack", "stand", "cap", "pack", "cover", "opener", "sleeve", "label", "end", "contact", "terminal",
             "cage", "mount", "shell", "body", "handle", "tip", "unit", "stack", "tower", "set", "row", "column", "grid", "assembly"}
POSITIONAL = {"left", "right", "near", "far", "front", "back", "top", "bottom", "upper", "lower", "inner", "outer", "first", "second",
              "third", "fourth", "middle", "centre", "center", "a", "b", "c", "d", "x", "y", "xp", "xn", "yp", "yn"}


def _plural(n):
    return rf"{n}(?:s|es)?"


def text_counts(txt, nouns, all_nouns):
    """{noun: [(count, phrase)]} for '<number> [<= 3 words] <noun>' phrases and 'AxB grid' next to a plural noun."""
    txt = re.sub(r"[\"{}]", " ", str(txt).lower())
    num = r"(?<![\d.x])(\d{1,2})(?![\d.x])|\b(" + "|".join(sorted(WORDS, key=len, reverse=True)) + r")\b"
    nxt = re.compile(r"\s*-?\s*([a-z]+)")
    found = {}
    for n in nouns:
        pat = re.compile(rf"(?:{num})((?:[- ][a-z][a-z-]*){{0,3}}?)[- ]({_plural(n)})\b")
        for m in pat.finditer(txt):
            mid = [w for w in re.split(r"[- ]", m.group(3)) if w]
            if any(w in UNITS or w in WORDS or w.isdigit() for w in mid):
                continue
            after = nxt.match(txt, m.end())
            if after and (after.group(1) in FOLLOWERS or any(re.fullmatch(_plural(x), after.group(1)) for x in all_nouns)):
                continue                                     # "drive slots": the count belongs to the next noun
            k = int(m.group(1)) if m.group(1) else WORDS[m.group(2)]
            if 0 < k <= 64:
                found.setdefault(n, []).append((k, m.group(0).strip()))
        for m in re.finditer(r"(\d+)\s*x\s*(\d+)\s+grid", txt):
            before = txt[max(0, m.start() - 40): m.start()]
            if re.search(rf"\b{n}(?:s|es)\b[^.;,]*$", before):          # "studs in a 2x2 grid"
                found.setdefault(n, []).append((int(m.group(1)) * int(m.group(2)), m.group(0)))
    return found


def part_counts(o, nouns):
    """{noun: (count, phrases)} from a Stage 1 object's own description and its parts list (>= 2 parts that are the noun
    plus positional words: near_slot, far_slot)."""
    txt = " ".join([str(o.get("name", "")), json.dumps(o.get("appearance", "")), str(o.get("approximate_size", "")),
                    " ".join(o.get("properties", []) or []), str(o.get("initial_state", ""))])
    found = text_counts(txt, nouns, PART_NOUNS + OBJECT_NOUNS)
    for n in nouns:
        hits = []
        for p in o.get("parts", []) or []:
            toks = [t for t in re.split(r"[^a-z0-9]+", str(p).lower()) if t]
            if any(re.fullmatch(_plural(n), t) for t in toks) and all(re.fullmatch(_plural(n), t) or t in POSITIONAL or t.isdigit() for t in toks):
                hits.append(str(p))
        if len(hits) >= 2:
            found.setdefault(n, []).append((len(hits), "parts " + ", ".join(hits)))
    return {n: (max(k for k, _ in items), [ph for _, ph in items]) for n, items in found.items()}


def _body_id(model, name):
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))


def _param_count(params, noun):
    if not isinstance(params, dict):
        return None
    for k in (f"n_{noun}s", f"num_{noun}s", f"{noun}s", f"{noun}_count", f"n_{noun}", f"{noun}es", f"n_{noun}es"):
        v = params.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            return v, f"param {k}"
    a, b = params.get(f"{noun}s_x"), params.get(f"{noun}s_y")
    if isinstance(a, int) and isinstance(b, int):
        return a * b, f"param {noun}s_x * {noun}s_y"
    return None


def _named_parts(model, bodies, strip, noun):
    """Distinct '<noun><suffix>' tokens in the names of the object's bodies, geoms and joints, with the object's own
    names (its ids) removed first so 'drawer_unit' does not count as a drawer."""
    names = [model.body(b).name for b in bodies] + [model.geom(g).name for g in SA.body_geoms(model, bodies)]
    names += [model.joint(j).name for j in range(model.njnt) if int(model.jnt_bodyid[j]) in bodies]
    toks = set()
    for nm in names:
        if not nm:
            continue
        rest = nm.lower()
        for sx in sorted(strip, key=len, reverse=True):
            rest = rest.replace(sx.lower(), " ")
        for m in re.finditer(rf"(?<![a-z]){noun}(?:s|es)?_?([a-z]{{0,2}}\d+[a-z0-9]*|xp|xn|yp|yn|left|right|near|far|front|back|top|bottom|[a-z]{{1,2}})?(?![a-z])", rest):
            toks.add(m.group(1) or "")
    toks.discard("vis")
    return len(toks), sorted(toks)


def object_count_vs_stage1_check(scene, semantic, recon=None, nouns=None):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    sem_objs = SA.semantic_objects(semantic)
    if not sem_objs:
        raise ValueError("semantic: no Stage 1 objects found")
    extra = [str(n).lower() for n in (nouns or [])]
    part_nouns = list(dict.fromkeys(PART_NOUNS + extra))
    rec, _ = SA.load_json(recon) if recon is not None else (None, None)
    sem, _ = SA.load_json(semantic)
    rec_objs = {str(o.get("id")): o for o in (rec or {}).get("objects", [])}
    mapping = SA.stage1_groups(model, semantic, recon, ref=st["model"], cfg=st["cfg"])
    rows, unverified, issues = [], [], []

    def judge(row, c1, phrases, sc, src, what, at_least=False):
        row.update({"stage1": c1, "phrases": phrases[:4], "scene": sc, "source": src})
        if at_least:
            row["at_least"] = True
        if sc is None:
            row["verdict"] = "unverified"
            unverified.append(f"{what}: {c1} {row['noun']}(s)")
        elif (sc < c1) if at_least else (sc != c1):
            row["verdict"] = "mismatch"
            issues.append(f"{what}: Stage 1 says {'at least ' if at_least else ''}{c1} {row['noun']}(s) ({'; '.join(phrases[:2])}) but the "
                          f"scene has {sc} ({src}); rebuild {'the object' if what != 'task' else 'the scene'} with "
                          f"{'at least ' if at_least else ''}{c1}")
        else:
            row["verdict"] = "ok"
        rows.append(row)

    # group objects (Stage 1 `pieces`): the scene has as many separate pieces as Stage 1 says
    for o in sem_objs:
        oid = str(o.get("id"))
        pieces = PG.stage1_pieces(o)
        if not pieces or pieces["count"] is None:
            continue
        parts = PG.recon_parts(rec_objs.get(oid))
        g = mapping.get(oid)
        sc, src = None, None
        if parts and g is not None:
            have = [n for n in parts if _body_id(model, n) >= 0]
            sc, src = len(have), f"declared parts of {oid} in the scene"
        else:
            pc = _param_count((rec_objs.get(oid) or {}).get("params"), pieces["kind"] or "piece")
            if pc is not None:
                sc, src = pc
        judge({"object": oid, "noun": pieces["kind"] or "piece"}, pieces["count"], ["pieces"], sc, src, oid, at_least=pieces["at_least"])

    # parts of each object, from its own description
    for o in sem_objs:
        oid = str(o.get("id"))
        g = mapping.get(oid)
        for noun, (c1, phrases) in part_counts(o, part_nouns).items():
            sc, src = None, None
            pc = _param_count((rec_objs.get(oid) or {}).get("params"), noun)
            if pc is not None:
                sc, src = pc
            if sc is None and g is not None and g["root"] >= 0:
                strip = {oid, str(o.get("name", "")).replace(" ", "_")}
                rid = (rec_objs.get(oid) or {}).get("params", {}).get("name") if isinstance((rec_objs.get(oid) or {}).get("params"), dict) else None
                if rid:
                    strip.add(str(rid))
                for syn in SYN.get(noun, [noun]):
                    k, toks = _named_parts(model, g["bodies"], strip, syn)
                    if k:
                        sc, src = k, f"named {syn} parts {toks[:6]}"
                        break
            judge({"object": oid, "noun": noun}, c1, phrases, sc, src, oid)
    # objects of a kind, from the task-level statements
    task_txt = " ".join(str((sem or {}).get(k, "")) for k in ("task_prompt", "task_summary", "task_success"))
    obj_nouns = list(dict.fromkeys(OBJECT_NOUNS + extra))
    for noun, items in text_counts(task_txt, obj_nouns, PART_NOUNS + OBJECT_NOUNS).items():
        c1 = max(k for k, _ in items)
        def head(x):
            """The head noun of a Stage 1 object's name ('silver hard disk drive in black tray' -> drive) and of its id."""
            nm = re.split(r"\b(?:in|with|on|of|for|and)\b", str(x.get("name", "")).lower())[0]
            words = [re.sub(r"\d+$", "", w) for w in re.split(r"[^a-z0-9]+", nm)]
            words = [w for w in words if w and w not in POSITIONAL]
            toks = [re.sub(r"\d+$", "", t) for t in re.split(r"[^a-z0-9]+", str(x.get("id", "")).lower())]
            toks = [t for t in toks if t and t not in POSITIONAL]
            return (words[-1] if words else ""), (toks[-1] if toks else "")
        syns = SYN.get(noun, [noun])
        kin = [x for x in sem_objs if any(re.fullmatch(_plural(syn), h) for syn in syns for h in head(x))]
        # group objects whose pieces are of this kind (a pyramid of cards counts its cards)
        grouped = {}
        for x in sem_objs:
            xid = str(x.get("id"))
            parts = PG.recon_parts(rec_objs.get(xid))
            if len(parts) < 2:
                continue
            kind = (PG.stage1_pieces(x) or {}).get("kind")
            named = sum(1 for n in parts if any(re.fullmatch(_plural(syn), t) for syn in syns for t in re.split(r"[^a-z]+", n.lower()) if t))
            if (kind and any(re.fullmatch(_plural(syn), kind) for syn in syns)) or (not kind and named * 2 > len(parts)):
                grouped[xid] = parts
                if x not in kin:
                    kin.append(x)
        present = [x for x in kin if mapping.get(str(x.get("id"))) is not None and mapping[str(x.get("id"))]["root"] >= 0]
        n_of = {str(x.get("id")): (len([b for b in grouped[str(x.get("id"))] if _body_id(model, b) >= 0]) if str(x.get("id")) in grouped else 1)
                for x in present}
        listed = ", ".join(k + (f" ({n} pieces)" if k in grouped else "") for k, n in n_of.items())
        sc, src = (sum(n_of.values()), f"Stage 1 {noun} objects in the scene ({listed})") if kin else (None, None)
        judge({"object": "task", "noun": noun}, c1, [ph for _, ph in items], sc, src, "task", at_least=any(str(x.get("id")) in grouped for x in present))
    return SA.report(issues, counts=rows, unverified=unverified)
