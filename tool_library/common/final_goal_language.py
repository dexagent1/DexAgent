"""Compositional, fail-closed final-state language.

This is a deliberately finite grammar, not a general natural-language judge.
Every accepted atom emits a measurement. Identity words come only from frozen
Stage 1; no generated trajectory bindings or proposed success claims are used.
Unsupported atoms remain errors even if explicit predicates are also supplied.

Ordered stacks require explicit acyclic support chains; bridging requires top
contact with every named support. Observer directions use the reconstructed
initial video camera, never an assumed world axis. 'Immediately' and 'side by
side' permit at most a quarter of the smaller object's width between envelopes;
the latter also requires overlapping depth and similar base height. Unchanged
secondary parts must resolve by name and retain their housing-relative poses at
every recorded frame. These finite operational definitions do not certify
arbitrary visual, functional, time-dependent-camera or material claims.
"""
from __future__ import annotations

import re
from itertools import combinations
import numpy as np
import mujoco

# These predicates are private final-state measurements; arbitrary manifest
# declarations cannot introduce them or choose their thresholds.
CUSTOM_CLASSES = {"final_orientation", "final_flat", "final_cover", "final_cable_integrity", "final_knot_layout", "final_full_containment",
                  "final_spatial", "final_removed", "final_part_unchanged", "final_over_opening", "rotation_completed"}


def compile_final(gc, goal):
    from tool_library.common import task_gates as G
    refs, aliases = {}, {}

    def alias(text, ref):
        text = re.sub(r"[_-]+", " ", str(text).lower()).strip()
        if text:
            aliases.setdefault(text, set()).add(ref)

    def noun_aliases(text, ref):
        # A shortened name may omit known words, but may not add a new colour,
        # location or property. Shared abbreviations remain ambiguous. Direction
        # words alone are reserved for relations, never object identities.
        phrase = re.split(r"\b(?:with|of|in|on|for|and)\b", str(text).lower())[0]
        words = [w for w in re.findall(r"[a-z0-9]+", phrase) if w not in G._STOP]
        if not words:
            return
        reserved = {"left", "right", "front", "back", "top", "bottom", "middle", "center", "centre",
                    "open", "closed", "upright", "horizontal", "flat", "inverted", "stationary"}
        if len(words) > 8:
            alias(" ".join(words), ref)
            return
        for n in range(1, min(len(words), 8) + 1):
            for subset in combinations(words, n):
                if len(subset) == 1 and subset[0] in reserved:
                    continue
                phrase = " ".join(subset)
                alias(phrase, ref)
                if subset[-1] == words[-1] and not words[-1].endswith("s"):
                    alias(phrase + "s", ref)

    for oid, obj in gc.objs.items():
        alias(oid, oid)
        noun_aliases(obj.get("name", oid), oid)
        for part in obj.get("parts", []):
            if isinstance(part, str):
                alias(part, oid + ":" + part)
                alias(part.split()[-1], oid + ":" + part)
    # Colours/materials attached to the support noun must occur in its frozen
    # Stage 1 description. Unknown modifiers are not discarded.
    env = str(gc.sem.get("environment", {}).get("support_surface", "")).lower()
    support_modifiers = {w for w in G._toks(env) if w not in G.TABLE_WORDS | {"covered", "support"}}
    for word in G.TABLE_WORDS | {"support surface"}:
        alias(word, "table")
        for modifier in support_modifiers:
            alias(modifier + " " + word, "table")
    keys = sorted((key for key, values in aliases.items() if len(values) == 1), key=len, reverse=True)
    text = re.sub(r"[_-]+", " ", goal.lower())
    if keys:
        regex = r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![a-z0-9])"
        def replace(match):
            ref = next(iter(aliases[match.group()]))
            token = next((key for key, value in refs.items() if value == ref), None)
            if token is None:
                token = "@" + str(len(refs)); refs[token] = ref
            return token
        text = re.sub(regex, replace, text)
    predicates, coverage, issues = [], [], []
    subject = None
    ends_subject = False
    mentioned_parts = set()
    assembly = None
    base_ref = None

    def members(ref):
        return list(ref) if isinstance(ref, tuple) else [ref]

    def emit(cls, A, **kw):
        if isinstance(A, tuple):
            return [emit(cls, a, **kw) for a in A]
        p = {"class": cls, "A": A, "source": "task_success: compositional language", **kw}
        if cls == "final_knot_layout":
            existing = next((q for q in predicates if q["class"] == cls and q["A"] == A), None)
            if existing is not None:
                existing.update(kw)
                return existing
        if p not in predicates:
            predicates.append(p)
        return p

    def release_object(ref, **kw):
        # An articulated part is supported through its physical joint. Judge
        # release/contact/support on that connected object, never demand a
        # nonexistent collision contact across a structural hinge.
        if isinstance(ref, tuple):
            return [release_object(a, **kw) for a in ref]
        target = ref
        if ":" in ref:
            oid = G._oid(ref); root = gc.object_root(oid); b = gc.body_of(ref)
            rigidly_connected = b is not None and root is not None
            while rigidly_connected and b != root:
                if b <= 0 or any(int(gc.ep.model.jnt_type[j]) == G.FREE for j in range(int(gc.ep.model.body_jntadr[b]), int(gc.ep.model.body_jntadr[b] + gc.ep.model.body_jntnum[b]))):
                    rigidly_connected = False; break
                b = int(gc.ep.model.body_parentid[b])
            if rigidly_connected: target = oid
        return emit("release_settle", target, **kw)

    def invariant(ref, throughout=True):
        # "has not moved" at the scale of the object: 5 % of its size, 5 to 10 mm (a cup set into a 15 cm bowl nudges
        # the bowl 6 mm in the maintainers' own verified episode; a fixed 5 mm rejects that)
        emit("relative_pose", ref, B="table", params={"reference_frame": 0, "pos_tol_m": invariant_tol_m(gc, ref),
             "rot_tol_deg": 5. if throughout else 180.}, throughout=throughout)

    def support(ref, target, settle=False):
        if target == "table":
            release_object(ref, on=target)
        else:
            emit("stacked_on", ref, B=list(target) if isinstance(target, tuple) else target)
            if settle: release_object(ref)
        if assembly is not None:
            for a in members(ref):
                assembly["edges"].setdefault(a, set()).update(members(target))

    def finish_assembly():
        nonlocal assembly
        if assembly is None:
            return
        edges = assembly["edges"]
        def grounded(a, seen):
            if a == assembly["support"]: return True
            if a in seen or a not in edges: return False
            return all(grounded(b, seen | {a}) for b in edges[a])
        valid = len(edges) >= 2 and all(grounded(a, set()) for a in edges)
        if not valid:
            issues.append("a described stack/tower needs an explicit acyclic support chain to its stated surface")
        else:
            for a in edges: release_object(a)
        assembly = None

    def adjective(match):
        word, token = match.groups(); ref = refs[token]
        if word == "held":
            emit("held", ref, mode="in_hand")
        elif word in ("stationary", "unmoved", "unchanged"):
            invariant(ref)
        else:
            emit("final_orientation", ref, orientation=word.replace(" ", "_"))
        return token
    text = re.sub(r"\b(held|stationary|unmoved|unchanged|inverted|upside down|mouth down|upright|horizontal)\s+(@\d+)", adjective, text)
    text = re.sub(r"\babove and horizontally within (?:the )?open top of (?:the )?(@\d+)",
                  r"over the opening of \1", text)
    text = re.sub(r"\bin (?:the )?(?:demonstrated )?(inverted|upside down|mouth down|upright|horizontal) orientation\b",
                  r"and \1", text)
    # Shared complements in coordinated verb phrases. This is syntactic
    # distribution, e.g. 'contained by and resting inside X' yields two atoms.
    text = re.sub(r"\b(contained by|inside|on)\s+and\s+((?:resting|sitting|lying)(?: stably)? (?:inside|in|on))\s+(?:the )?(@\d+)",
                  lambda m: f"{m[1]} {m[3]} and {m[2]} {m[3]}", text)
    text = re.sub(r"\bwith\s+(?=(?:neither|no|both|one|a|an)\b)", ", ", text)
    text = re.sub(r"\bwith\s+(?=(?:the )?@\d+\b)", ", ", text)
    text = re.sub(r"\bwith\s+(?=(?:the |its )?interior\b)", ", ", text)
    # Preserve noun coordination before splitting predicate conjunctions.
    def coordinated(match):
        left, right = refs[match[1]], refs[match[2]]
        token = "@" + str(len(refs)); refs[token] = tuple(dict.fromkeys(members(left) + members(right)))
        return token
    pattern = r"(@\d+)\s+and\s+(?:the )?(@\d+)"
    while re.search(pattern, text):
        text = re.sub(pattern, coordinated, text)
    # A semicolon terminates an ordered assembly description. Retain that
    # boundary so neighbouring loose objects cannot become implicit supports.
    text = text.replace(";", ", __assembly_end__, ")
    chunks = [c.strip(" .") for c in re.split(r"[:,]|\.\s+|\band\b", text) if c.strip(" .")]
    for chunk in chunks:
        begin = len(predicates)
        raw = chunk
        if chunk == "__assembly_end__":
            finish_assembly(); continue
        chunk = re.sub(r"\b(?:the|its|a|an)\s+(?=@\d+)", "", chunk)
        header = re.fullmatch(r"(?:a |the )?(?:free form )?(?:tower|stack) stands on (@\d+)", chunk)
        if header and refs[header[1]] == "table":
            finish_assembly(); assembly = {"support": "table", "edges": {}}
            base_ref = None
            coverage.append({"clause": raw, "inferred_classes": ["release_settle", "stacked_on"],
                             "unknown_terms": [], "unsupported_classes": [], "unresolved_conjunction": False,
                             "inferred_coverage": True})
            continue
        if chunk == "from the bottom" and assembly is not None and not assembly["edges"]:
            continue
        if base_ref:
            noun = re.findall(r"[a-z0-9]+", gc.objs[base_ref].get("name", base_ref).lower())[-1]
            base_token = next(k for k, v in refs.items() if v == base_ref)
            chunk = re.sub(r"\b(?:the )?bottom " + re.escape(noun) + r"\b", base_token, chunk)
        # Temporal final withdrawal clauses demand independent release/settling.
        withdraw = re.search(r"\s+(?:(?:when|after) (?:the |both )?hands? (?:withdraw|leave|move away|let go)|without (?:any )?hand support|with no hand contact)$", chunk)
        release = bool(withdraw)
        if withdraw:
            chunk = chunk[:withdraw.start()]
        # A noun-explicit atom updates its subject; predicate-only conjuncts
        # inherit the preceding subject. Hand clauses keep their object explicit.
        previous = subject
        match = re.match(r"^(@\d+)\s+(.+)$", chunk)
        if match:
            token, body = match.groups(); active = refs[token]; subject = active; ends_subject = False
        else:
            active, body = subject, chunk
        body = re.sub(r"^(?:is|are|was|were|remains?|stays?|has been|have been)\s+", "", body)
        if active:
            mentioned_parts.update(a for a in members(active) if ":" in a)
        # Pronouns refer to the preceding explicit subject, not the new noun
        # being placed on it. Ambiguous singular/plural agreement stays unknown.
        if previous:
            token = next((k for k, v in refs.items() if v == previous), None)
            pronoun = "them" if isinstance(previous, tuple) else "it"
            if token: body = re.sub(r"\b" + pronoun + r"\b", token, body)
        accepted = False
        if re.fullmatch(r"nothing is held|no objects? (?:is|are) held", chunk):
            for oid in gc.objs:
                release_object(oid)
            accepted = bool(gc.objs)
        others = re.fullmatch(r"(?:the )?other ([a-z]+) (?:are |remain )?unchanged", chunk)
        if others and subject and isinstance(subject, str):
            oid = G._oid(subject); noun = G._norm(others[1])
            parts = [oid + ":" + p for p in gc.objs.get(oid, {}).get("parts", []) if isinstance(p, str)
                     and G._toks(p)[-1:] == [noun] and oid + ":" + p not in mentioned_parts]
            if parts:
                for part in parts: emit("final_part_unchanged", part)
                accepted = True
        hand = re.fullmatch(r"(?:neither|no|both|the) hands? (?:is |are )?(?:touch(?:es|ing)?|supports?|supporting|away from|off) (?:the )?(@\d+|it|them)", chunk)
        if hand and (hand[1] in refs or subject):
            target = refs.get(hand[1], subject)
            if chunk.startswith(("neither", "no")) or re.search(r"away from|\boff\b", chunk):
                release_object(target); accepted = True
        # 'an (un)tightened overhand knot exists', 'the knot is tightened': the cord's knot itself, measured
        knot2 = re.fullmatch(r"(?:an? |the )?(?:untightened |loose |tight |tightened |closed )?(overhand )?knot (?:exists|is (?:tightened|tight|closed|formed))", chunk)
        if knot2 and not accepted:
            target = active or next((o for o in gc.objs if gc.objs[o].get("type") == "deformable"), None)
            if target:
                emit("knot", target, params={"expect": "trefoil" if knot2[1] else "knot"}, persistent=True); accepted = True
        if not accepted and active:
            spatial = re.search(r"(?:^|\s+)(?:(directly|immediately) )?(?:to (@\d+)'s (right|left)|to the (right|left) of (@\d+)|(?:to the )?(right|left) of (@\d+)|(behind|in front of) (@\d+))$", body)
            spatial_pred = None
            if spatial:
                target = spatial[2] or spatial[5] or spatial[7] or spatial[9]
                direction = spatial[3] or spatial[4] or spatial[6] or spatial[8]
                spatial_pred = {"B": refs[target], "relation": direction, "direct": bool(spatial[1]),
                                "near": spatial[1] == "immediately"}
                body = body[:spatial.start()]
            paired = "side by side" in body and isinstance(active, tuple) and len(active) == 2
            if paired:
                body = re.sub(r"\s*side by side\s*", " ", body).strip()
            if not body and (spatial_pred or paired):
                accepted = True
            bridge = re.fullmatch(r"bridging (@\d+)(?: on top)?", body)
            if bridge and isinstance(refs[bridge[1]], tuple) and len(refs[bridge[1]]) >= 2:
                support(active, refs[bridge[1]], settle=True); accepted = True
            removed = re.fullmatch(r"removed from (@\d+)", body)
            if removed:
                emit("final_removed", active, B=refs[removed[1]]); accepted = True
            contact = re.fullmatch(r"(?:touch(?:es|ing)?|in contact with|against) (@\d+)", body)
            if contact:
                emit("contact", active, B=refs[contact[1]]); accepted = True
            clear = re.fullmatch(r"clear of (@\d+)", body)
            if clear:
                emit("lift_clearance", active, support=refs[clear[1]]); accepted = True
            # held, optionally in a named hand and upright, and lifted off a support: "held clear of the table",
            # "held level above the table", "held upright in the left hand", "held in the right hand above the table"
            held_lift = re.fullmatch(r"(?:securely )?held(?: (upright|level))?(?: (?:in|by) (?:the )?(left|right) hand)?"
                                     r"(?: (?:clear of|above|off) (@\d+))?", body)
            if held_lift and (held_lift[1] or held_lift[3]):
                emit("held", active, side=held_lift[2], mode="in_hand")
                if held_lift[3]:
                    emit("lift_clearance", active, support=refs[held_lift[3]])
                if held_lift[1] == "upright":
                    emit("final_orientation", active, orientation="upright")
                accepted = True
            # still held: "still in the hand", "in the left hand", "upright in the pinch" (the grip keeps it upright)
            in_hand = re.fullmatch(r"(?:(upright|level) )?(?:still |securely )?(?:held )?in (?:the |its )?(?:(left|right) )?"
                                   r"(?:hand|pinch|grip|grasp|fingers)", body)
            if in_hand:
                emit("held", active, side=in_hand[2], mode="in_hand")
                if in_hand[1] == "upright":
                    emit("final_orientation", active, orientation="upright")
                accepted = True
            # over a container's opening: "above the bowl's interior", "above the tray interior", "over the mouth of ..."
            over = (re.fullmatch(r"(?:held )?(?:directly )?(?:above|over) (@\d+)(?:'s)? (?:interior|opening|mouth|rim)", body)
                    or re.fullmatch(r"(?:held )?(?:directly )?(?:above|over) (?:the )?(?:interior|opening|mouth|rim) of (@\d+)", body))
            if over and refs[over[1]] != "table":
                emit("final_over_opening", active, B=refs[over[1]]); accepted = True
            unsupported = re.fullmatch(r"no longer supported by (@\d+)", body)
            if unsupported:
                emit("lift_clearance", active, support=refs[unsupported[1]]); accepted = True
            opening = re.fullmatch(r"over the opening of (@\d+)", body)
            if opening:
                emit("final_over_opening", active, B=refs[opening[1]]); accepted = True
            if re.fullmatch(r"(?:has|have) not moved|(?:stationary|unchanged|unmoved)|(?:in|at) (?:its |the )?(?:same|original) (?:place|position)", body):
                invariant(active, "same" not in body and "original" not in body); accepted = True
            orientation = re.fullmatch(r"(?:rests? |resting |stands? |standing )?(inverted|upside down|mouth down|upright|horizontal)(?: (on|inside|in) (@\d+))?", body)
            if orientation:
                emit("final_orientation", active, orientation=orientation[1].replace(" ", "_"))
                if orientation[3]:
                    target = refs[orientation[3]]
                    if orientation[2] == "on": support(active, target, settle=True)
                    else:
                        emit("inside_container", active, B=target)
                        release_object(active, on=target)
                accepted = True
            relation = re.fullmatch(r"(?:(fully|entirely|wholly|completely) )?(?:(contained by|contained in|inside|in|into|on|on top of)|(?:(rests?|resting|sits?|sitting|lies|lying|stands?|standing|placed)(?: (stably|stable))? (inside|in|on|on top of)))\s+(@\d+)", body)
            if relation:
                target = refs[relation[6]]; location = relation[2] or relation[5]
                if location in ("contained by", "contained in", "inside", "in", "into"):
                    emit("inside_container", active, B=target, params={"entirely": bool(relation[1])})
                    if relation[1]: emit("final_full_containment", active, B=target)
                else:
                    support(active, target, settle=bool(relation[3]))
                    if assembly is not None and len(assembly["edges"]) == 1 and target == "table" and isinstance(active, str):
                        base_ref = active
                if relation[3] and location not in ("on", "on top of"): release_object(active, on=target)
                accepted = not relation[1] or location in ("contained by", "contained in", "inside", "in", "into")
            if re.fullmatch(r"(?:fully |completely )?(?:closed|shut|open|opened)", body):
                closed = body.endswith(("closed", "shut"))
                obj = gc.objs.get(G._oid(active), {})
                target = active
                if ":" not in active:
                    moving = obj.get("articulation", {}).get("moving_part")
                    if isinstance(moving, str) and moving in obj.get("parts", []):
                        target = active + ":" + moving
                params = {}
                if body.startswith(("fully ", "completely ")):
                    bid = gc.body_of(target)
                    joints = gc.joints_of(bid, G._oid(target)) if bid is not None else []
                    if len(joints) == 1:
                        info = G.joint_info(gc.ep, joints[0])
                        if info["range"] is not None:
                            closed_value = G.closed_value(info["range"])
                            target_value = closed_value if closed else max(info["range"], key=lambda value: abs(value - closed_value))
                            params = {"joint": joints[0], "target": target_value,
                                      "tol": float(np.radians(2.)) if info["type"] == "hinge" else .002}
                emit("joint_target", target, state="closed" if closed else "open", params=params,
                     persistent=release or bool(re.search(r"\b(?:remains?|stays?)\b", chunk)))
                accepted = True
            if body in ("flat", "level"):
                emit("final_flat", active); accepted = True
            seat = re.fullmatch(r"(?:seated|sitting|lying|lies|flat)(?: (?:flat|evenly))? (?:over|on) (@\d+)", body)
            if seat:
                other = refs[seat[1]]
                if G._oid(active) == G._oid(other) and ":" in active:
                    emit("final_cover", G._oid(active), part=active)
                    emit("final_flat", active)
                    accepted = True
            if re.fullmatch(r"(?:(?:has|have) )?completed (?:its |their |the )?rotation", body):
                emit("rotation_completed", active); accepted = True
            held = re.fullmatch(r"(?:securely )?(?:held|grasped|gripped)(?: (fixed|steady|still))?(?: (?:in|by) (?:the )?(left|right) hand)?", body)
            if held:
                emit("held", active, side=held[2], mode="fixed" if held[1] else "in_hand"); accepted = True
            if re.fullmatch(r"(?:released|stable|stably supported|at rest)(?: (?:on) (@\d+))?", body):
                target = re.search(r"@\d+", body)
                release_object(active, on=refs[target.group()] if target else None); accepted = True
            if re.fullmatch(r"(?:rests?|resting) unsupported by (?:either|any) hand", body):
                release_object(active); accepted = True
            # The knot type is actually measured, not inferred from crossings.
            knot = re.fullmatch(r"(?:has |with )?(?:(?:one|a|an) )?(persistent |maintained )?(overhand )?knot(?: (?:in it|in the rope|in @\d+))?( near (?:its |the )?(?:center|centre|middle))?", body)
            if knot and (not re.search(r"\bone\b", body) or knot[2]):
                emit("knot", active, params={"expect": "trefoil" if knot[2] else "knot"}, persistent=True)
                if knot[3]: emit("final_knot_layout", active, central=True)
                accepted = True
            # a cut cord: 'is cut through', 'is severed into two pieces' (the join the blades switch off, rope_severed)
            if re.fullmatch(r"(?:is |has been )?(?:cut through|cut in two|severed)(?: (?:in|into) two (?:pieces|halves|parts))?", body):
                emit("rope_severed", active); accepted = True
            # a cord that crosses itself / forms a loop: at least one plan-view self-crossing (strand_crossings)
            if re.fullmatch(r"(?:crosses|crossing|lies crossed) (?:over |under )?(?:itself|its own strand)(?: once)?|"
                            r"(?:forms?|forming|makes?|making) an? (?:closed |open )?loop(?: (?:on|over) @\d+)?", body):
                emit("strand_crossings", active, params={"min_crossings": 1}); accepted = True
            if re.fullmatch(r"both (?:metal fitted |fitted )?ends (?:remain|are|stay) attached", body):
                emit("final_cable_integrity", active); accepted = True; ends_subject = True
            opposite = re.fullmatch(r"(?:(?:both )?ends (?:lie|are|remain)|lie) on opposite sides of (?:the |its )?knot", body)
            if opposite and (ends_subject or body.startswith(("both ends", "ends"))):
                emit("final_knot_layout", active, opposite=True); accepted = True
            if accepted and spatial_pred:
                emit("final_spatial", active, **spatial_pred)
            if accepted and paired:
                emit("final_spatial", active[0], B=active[1], relation="side by side")
        # 'the interior is hidden' refers only to the current articulated
        # container, and has its own dense geometric cover check.
        if re.fullmatch(r"(?:the |its )?interior (?:is |remains? )?(?:hidden|covered|no longer visible)", chunk) and subject:
            emit("final_cover", G._oid(subject)); accepted = True
        if release and active and accepted:
            release_object(active)
        # Meaningful unknown clauses never disappear behind a recognised atom.
        coverage.append({"clause": raw, "inferred_classes": sorted({p["class"] for p in predicates[begin:]}),
                         "unknown_terms": [] if accepted else G._toks(raw), "unsupported_classes": [],
                         "unresolved_conjunction": False, "inferred_coverage": accepted})
        if not accepted:
            issues.append(f"cannot independently cover final-state atom '{raw}'; express observable relations separately in Stage 1 or implement its measurement")
    finish_assembly()
    for event in [p for p in predicates if p["class"] == "rotation_completed"]:
        ref = event["A"]
        endpoint = any(p["class"] == "joint_target" and p["A"] == ref for p in predicates)
        endpoint |= (any(p["class"] == "final_flat" and p["A"] == ref for p in predicates) and
                     any((p["class"] == "final_cover" and p.get("part") == ref) or
                         (p["class"] == "stacked_on" and p["A"] == ref) for p in predicates))
        if not endpoint:
            issues.append("completed rotation requires a separately measured joint endpoint or a flat supported/covered endpoint; an unspecified rotation is not completion")
    return predicates, coverage, issues


def invariant_tol_m(gc, ref, frac=.05, lo=.005, hi=.01):
    """Position tolerance of "has not moved": a fraction of the object's size (its bodies' geoms), within [lo, hi]."""
    try:
        import mujoco
        m = gc.ep.model
        bid = gc.body_of(ref)
        if bid is None:
            return lo
        def inside(b):                                   # the body and its descendants
            while b > 0:
                if b == bid:
                    return True
                b = int(m.body_parentid[b])
            return False
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        gids = [g for g in range(m.ngeom) if inside(int(m.geom_bodyid[g]))]
        r = max((float(np.linalg.norm(d.geom_xpos[g] - d.xpos[bid])) + float(m.geom_rbound[g]) for g in gids), default=0.)
        return float(min(hi, max(lo, frac * 2. * r)))
    except Exception:
        return lo


def measure_custom(gc, p, frame):
    from tool_library.common import task_gates as G
    ep, model = gc.ep, gc.ep.model
    ref = p["A"]
    row = {"class": p["class"], "source": p.get("source"), "frames": [frame, frame], "A": ref}
    try:
        bodies, name, bid = G._resolve_set(gc, ref, ())
        row["A"] = name
        if p["class"] == "rotation_completed":
            joints = gc.joints_of(bid, G._oid(ref))
            if len(joints) != 1 or G.joint_info(ep, joints[0])["type"] != "hinge":
                raise ValueError("completed rotation requires exactly one independently resolved hinge")
            info = G.joint_info(ep, joints[0]); start = int(p.get("start_frame", 0))
            first = max(start, frame - int(np.ceil(.3 * ep.fps)))
            states = ep.Q[first:frame + 1, info["adr"]]
            displacement = abs(float(ep.Q[frame, info["adr"]] - ep.Q[start, info["adr"]]))
            drift = float(np.ptp(states)); duration = (frame - first) / ep.fps
            ok = displacement >= np.radians(2.) and drift <= np.radians(2.) and duration >= .3 - 1e-9
            report = G._report(ok, [] if ok else ["rotation must change its uniquely resolved hinge by at least 2 degrees in this execution window and remain within 2 degrees over the terminal 0.30 s"],
                               joint=joints[0], start_frame=start, end_frame=frame,
                               displacement_deg=float(np.degrees(displacement)), terminal_drift_deg=float(np.degrees(drift)), terminal_s=duration)
        elif p["class"] == "final_orientation":
            _, initial = ep.pose(0, bid); _, actual = ep.pose(frame, bid)
            up = G.up_axis(initial)
            angle = G.tilt_deg(actual, up)
            initial_text = str(gc.objs[G._oid(ref)].get("initial_state", "")).lower()
            flipped = bool(re.search(r"\bupside[ -]down\b|\binverted\b|\bmouth[ -]down\b", initial_text))
            requested = p["orientation"]
            want = 90. if requested == "horizontal" else (180. if (requested in ("inverted", "upside_down", "mouth_down")) != flipped else 0.)
            error = abs(angle - want)
            report = G._report(error <= 15., [] if error <= 15. else [f"orientation is {angle:.1f} degrees from initial up, expected {want:.1f} +/- 15 degrees"],
                               angle_deg=angle, expected_deg=want, tolerance_deg=15.)
        elif p["class"] == "final_over_opening":
            other, other_name, _ = G._resolve_set(gc, p["B"], (), exclude=bodies)
            origin, axes, rim, radius = opening_disk(ep, frame, other)
            alo, ahi = ep.extent(frame, ep.geoms(bodies), axes)
            off = axes.T @ origin
            alo, ahi = alo - off, ahi - off
            corners = np.array([[x, y] for x in (alo[0], ahi[0]) for y in (alo[1], ahi[1])])
            reach = float(np.linalg.norm(corners, axis=1).max())
            above = bool(alo[2] >= rim - G.TOUCH_TOL_M)
            within = bool(reach <= radius + G.TOUCH_TOL_M)
            report = G._report(above and within, ([] if above else ["the object's lowest point is below the opening rim"]) +
                               ([] if within else ["the complete horizontal object envelope is not inside the measured opening"]),
                               container=other_name, opening_radius_m=radius, envelope_radius_m=reach,
                               lowest_height_m=float(alo[2]), rim_height_m=rim,
                               scope="collision-free inscribed disk at the mouth plane; complete horizontal envelope")
        elif p["class"] == "final_full_containment":
            other, other_name, _ = G._resolve_set(gc, p["B"], (), exclude=bodies)
            centre, axes = G._container_frame(ep, frame, other)
            alo, ahi = ep.extent(frame, ep.geoms(bodies), axes)
            blo, bhi = ep.extent(frame, ep.geoms(other), axes)
            fits = bool(np.all(alo >= blo - G.TOUCH_TOL_M) and np.all(ahi <= bhi + G.TOUCH_TOL_M))
            report = G._report(fits, [] if fits else [f"the complete object does not fit within {other_name}'s extent in all three dimensions"],
                               object_bounds=[alo.tolist(), ahi.tolist()], container_bounds=[blo.tolist(), bhi.tolist()], tolerance_m=G.TOUCH_TOL_M)
        elif p["class"] == "final_spatial":
            other, other_name, _ = G._resolve_set(gc, p["B"], (), exclude=bodies)
            axes = _source_axes(gc, p["relation"])
            alo, ahi = ep.extent(frame, ep.geoms(bodies), axes)
            blo, bhi = ep.extent(frame, ep.geoms(other), axes)
            size = np.minimum(ahi - alo, bhi - blo)
            tol = G.TOUCH_TOL_M
            cross_overlap = np.minimum(ahi, bhi) - np.maximum(alo, blo)
            relation = p["relation"]
            if relation == "side by side":
                gap = max(alo[0] - bhi[0], blo[0] - ahi[0])
                ok = (-tol <= gap <= .25 * size[0] + tol and cross_overlap[1] >= .5 * size[1] - tol
                      and abs(alo[2] - blo[2]) <= .25 * size[2] + tol)
            else:
                axis = 0 if relation in ("right", "left") else 1
                positive = relation in ("right", "behind")
                gap = alo[axis] - bhi[axis] if positive else blo[axis] - ahi[axis]
                centres = (alo[axis] + ahi[axis] - blo[axis] - bhi[axis]) * (.5 if positive else -.5)
                ok = gap >= -tol and centres > tol
                if p.get("direct"): ok &= cross_overlap[1-axis] >= -tol
                if p.get("near"): ok &= gap <= .25 * size[axis] + tol
            report = G._report(bool(ok), [] if ok else [f"{name} is not {relation} {other_name} in the source camera's horizontal frame"],
                               relation=relation, frame_source="video_cam at initial frame", axes=axes.tolist(),
                               A_bounds=[alo.tolist(), ahi.tolist()], B_bounds=[blo.tolist(), bhi.tolist()],
                               separation_m=float(gap), tolerance_m=tol)
        elif p["class"] == "final_removed":
            target = _strict_part(gc, p["B"]) if ":" in p["B"] else gc.body_of(p["B"])
            if target is None: raise ValueError("removal requires the independently resolved original container")
            other = ep.subtree(target) - bodies
            if not other: raise ValueError("removal requires a distinct container")
            _, R0 = ep.pose(0, target); _, R1 = ep.pose(frame, target)
            a0, a1 = ep.extent(0, ep.geoms(bodies), R0)
            b0, b1 = ep.extent(0, ep.geoms(other), R0)
            initial_inside = bool(np.all(a0 >= b0 - G.TOUCH_TOL_M) and np.all(a1 <= b1 + G.TOUCH_TOL_M))
            a0, a1 = ep.extent(frame, ep.geoms(bodies), R1)
            b0, b1 = ep.extent(frame, ep.geoms(other), R1)
            final_outside = bool(np.any(a0 >= b1 + G.TOUCH_TOL_M) or np.any(b0 >= a1 + G.TOUCH_TOL_M))
            ok = initial_inside and final_outside
            report = G._report(ok, [] if ok else ["removal requires initial full containment and a final object wholly outside the container envelope"],
                               initially_contained=initial_inside, finally_outside=final_outside,
                               final_object_bounds=[a0.tolist(), a1.tolist()], final_container_bounds=[b0.tolist(), b1.tolist()])
        elif p["class"] == "final_part_unchanged":
            part = _strict_part(gc, ref); root = gc.object_root(G._oid(ref))
            # Track the part relative to its own housing at every frame; a
            # moved cabinet does not itself count as an opened drawer. Include
            # every descendant to catch local motion hidden by a static root.
            failures, worst_p, worst_r = [], 0., 0.
            for child in sorted(ep.subtree(part)):
                for t in range(frame + 1):
                    check = G.gate_relative_pose(ep, {child}, {root}, t, reference_frame=0,
                                                pos_tol_m=.002, rot_tol_deg=2.)
                    worst_p = max(worst_p, check.get("pos_err_m", 0.))
                    worst_r = max(worst_r, check.get("rot_err_deg", 0.))
                    if not check["ok"]: failures.append(t)
            report = G._report(not failures, [] if not failures else ["secondary part changed relative to its housing during the episode"],
                               frames_evaluated=frame + 1, worst_position_error_m=worst_p, worst_rotation_error_deg=worst_r,
                               changed_frames=sorted(set(failures))[:10])
        elif p["class"] == "final_flat":
            data = ep.at(frame)
            geoms = [g for g in ep.geoms({bid}) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
            panels = [g for g in geoms if np.min(model.geom_size[g]) < .35 * np.partition(model.geom_size[g], 1)[1]]
            if not panels: raise ValueError("flatness requires a resolved panel geometry")
            g = max(panels, key=lambda g: np.prod(np.sort(model.geom_size[g])[1:]))
            normal = data.geom_xmat[g].reshape(3, 3)[:, np.argmin(model.geom_size[g])]
            angle = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
            report = G._report(angle <= 10., [] if angle <= 10. else [f"panel is tilted {angle:.1f} degrees from the support plane"], tilt_deg=float(angle), tolerance_deg=10.)
        elif p["class"] == "final_cover":
            report = _cover_report(ep, bid, frame, p.get("part"), gc)
        elif p["class"] == "final_knot_layout":
            pre, chain = G.cable_bodies(ep, name.split("B_")[0])
            data = ep.at(frame)
            points = [data.xpos[b].copy() for b in chain]
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pre + "S_last")
            if sid >= 0: points.append(data.site_xpos[sid].copy())
            report = knot_layout(np.asarray(points), central=bool(p.get("central")), opposite=bool(p.get("opposite")))
        elif p["class"] == "final_cable_integrity":
            pre, chain = G.cable_bodies(ep, name.split("B_")[0])
            # A serial MuJoCo chain cannot detach. Equality-joined chains need
            # recorded activation; do not infer intactness from proximity.
            links_ok = all(int(model.body_parentid[b]) == a for a, b in zip(chain, chain[1:]))
            free_interior = any(int(model.jnt_type[j]) == G.FREE for b in chain[1:]
                                for j in range(int(model.body_jntadr[b]), int(model.body_jntadr[b] + model.body_jntnum[b])))
            geoms = [ep.geoms({b}) for b in (chain[0], chain[-1])]
            ok = links_ok and not free_interior and all(geoms)
            report = G._report(ok, [] if ok else ["rope terminal attachment is not a continuous serial cable with physical endpoint geometry"],
                               serial_chain=links_ok, free_interior=free_interior, endpoint_geoms=geoms, links=len(chain))
        else:
            raise ValueError("unsupported final-state measurement")
        row.update(ok=bool(report["ok"]), issues=report["issues"], measured={k: v for k, v in report.items() if k not in ("ok", "issues")})
    except (ValueError, G._Unresolved) as error:
        row.update(ok=False, unresolved=True, issues=[str(error)], measured={})
    return row


def opening_disk(ep, frame, bodies):
    """Conservative empty disk at a container mouth, measured from collision geometry.

    Slice every convex collision solid just below the rim. The nearest slice
    boundary limits an empty disk about the container origin; convex envelopes
    for non-mesh primitives only shrink it. Closed tops and unbounded openings
    fail rather than treating the outer bounding box as an empty aperture.
    """
    from scipy.spatial import ConvexHull, QhullError
    from itertools import product
    from tool_library.common import task_gates as G
    origin, axes = G._container_frame(ep, frame, bodies)
    def convex(points):
        try:
            return ConvexHull(points)
        except QhullError as error:
            raise ValueError("cannot resolve a nondegenerate container mouth cross-section") from error
    geoms = ep.geoms(bodies); lo, hi = ep.extent(frame, geoms, axes)
    off = axes.T @ origin; rim = float(hi[2] - off[2])
    height = rim - min(.002, .02 * float(hi[2] - lo[2]))
    data, model = ep.at(frame), ep.model
    radius, sections = np.inf, []
    for geom in geoms:
        low, high = ep.extent(frame, [geom], axes)
        low, high = low - off, high - off
        if height < low[2] or height > high[2]:
            continue
        if int(model.geom_type[geom]) == G.MESH:
            verts = (ep._verts(geom) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom] - origin) @ axes
        elif int(model.geom_type[geom]) == G.BOX:
            local = np.array(list(product((-1, 1), repeat=3))) * model.geom_size[geom]
            verts = (local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom] - origin) @ axes
        else:
            verts = np.array(list(product(*zip(low, high))))
        hull = convex(verts)
        edges = {tuple(sorted((int(a), int(b)))) for face in hull.simplices for a, b in combinations(face, 2)}
        points = [v[:2] for v in verts if abs(v[2] - height) < 1e-9]
        for a, b in edges:
            va, vb = verts[a], verts[b]
            if (va[2] - height) * (vb[2] - height) < 0:
                points.append((va + (vb - va) * ((height - va[2]) / (vb[2] - va[2])))[:2])
        if len(points) < 3:
            continue
        points = np.unique(np.round(points, 10), axis=0)
        section = convex(points)
        if np.all(section.equations[:, -1] <= 1e-9):
            raise ValueError("container mouth is closed or its independently defined centre lies in a wall")
        polygon = points[section.vertices]; sections.append(polygon)
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = b - a
            u = np.clip(-float(a @ edge) / max(float(edge @ edge), 1e-20), 0., 1.)
            radius = min(radius, float(np.linalg.norm(a + u * edge)))
    if len(sections) < 3 or not np.isfinite(radius) or radius <= G.TOUCH_TOL_M:
        raise ValueError("cannot resolve a bounded, open container mouth from collision geometry")
    boundary = convex(np.concatenate(sections))
    if not np.all(boundary.equations[:, -1] < -G.TOUCH_TOL_M):
        raise ValueError("container mouth does not surround its independently defined centre")
    return origin, axes, rim, radius


def _strict_part(gc, ref):
    """Resolve a named secondary part without GoalContext's root fallback."""
    from tool_library.common import task_gates as G
    oid, _, part = ref.partition(":")
    root = gc.object_root(oid)
    if not part or root is None:
        raise ValueError("a named part and its reconstructed object root are required")
    tokens = G._toks(part)
    candidates = []
    for b in gc.ep.subtree(root) - {root}:
        name = G._toks(gc.ep.name(b))
        if len(name) >= len(tokens) and name[-len(tokens):] == tokens:
            candidates.append(b)
    if len(candidates) != 1:
        raise ValueError(f"cannot resolve exactly one physical body for named part {ref!r}; no root or nearest-name fallback")
    return candidates[0]


def _source_axes(gc, relation):
    """Observer relations need the reconstructed source camera, never world-x.

    Frame zero defines the fixed observer frame. Directional goals from a moving
    camera that require a different time-specific frame are unsupported.
    """
    description = str(gc.sem.get("environment", {}).get("camera", "")).lower()
    if not re.search(r"egocentric|head[ -]mounted|operator", description):
        raise ValueError("directional final relations require an independently declared egocentric/operator camera frame")
    ep = gc.ep; cam = mujoco.mj_name2id(ep.model, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam")
    if cam < 0:
        raise ValueError("directional final relations require the reconstructed source video_cam")
    data = ep.at(0)
    mujoco.mj_camlight(ep.model, data)
    R = data.cam_xmat[cam].reshape(3, 3)
    x = R[:, 0].copy(); x[2] = 0.
    if np.linalg.norm(x) < .2: raise ValueError("source camera has no reliable horizontal right direction")
    x /= np.linalg.norm(x); z = np.array([0., 0., 1.]); y = np.cross(z, x)
    if relation in ("behind", "in front of"):
        forward = -R[:, 2].copy(); forward[2] = 0.
        if np.linalg.norm(forward) < .2 or y @ forward <= .1:
            raise ValueError("source camera has no reliable forward direction on the support plane")
    return np.stack([x, y, z], axis=1)


def _cover_report(ep, root, frame, part, gc):
    """A seated opaque cover over a box-panel cavity, checked with 81 rays.

    This supports a reconstructed rectangular cavity and a solid box-panel lid.
    One opaque panel must continuously cover the entire opening in projection;
    rays additionally check seating within 3 mm of the rim. A patch, a floating
    lid, a perforated mesh, or robot occlusion cannot certify closure. Multi-panel
    and mesh lids require a separate surface-coverage implementation.
    """
    from tool_library.common import task_gates as G
    model, data = ep.model, ep.at(frame)
    origin, R = ep.pose(frame, root)
    base = {root}
    # Include fixed construction children, never articulated lid descendants.
    for b in sorted(ep.subtree(root) - {root}):
        if int(model.body_parentid[b]) in base and int(model.body_jntnum[b]) == 0:
            base.add(b)
    panels = []
    for g in ep.geoms(base):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX: continue
        relative = R.T @ data.geom_xmat[g].reshape(3, 3)
        extent = abs(relative) @ model.geom_size[g]
        pos = R.T @ (data.geom_xpos[g] - origin)
        panels.append((g, pos, extent))
    bottoms = [x for x in panels if x[2][2] < .35 * min(x[2][:2])]
    if not bottoms: raise ValueError("covered interior requires a reconstructed panel-box bottom and walls")
    bottom = min(bottoms, key=lambda x: x[1][2])
    _, pos, size = bottom
    bottom_top = pos[2] + size[2]
    walls = [x for x in panels if x[0] != bottom[0] and x[1][2] + x[2][2] > bottom_top + .005]
    if len(walls) < 4: raise ValueError("covered interior requires four physical base walls")
    rim = max(x[1][2] + x[2][2] for x in walls)
    # Infer the four inner wall faces. A fixed arbitrary inset could certify a
    # lid which leaves a visible strip of the actual opening uncovered.
    limits = []
    for axis in (0, 1):
        thin = [x for x in walls if x[2][axis] < .35 * min(x[2][1-axis], x[2][2])]
        negative = [x for x in thin if x[1][axis] < pos[axis]]
        positive = [x for x in thin if x[1][axis] > pos[axis]]
        if not negative or not positive:
            raise ValueError("cannot independently resolve rectangular cavity's four inner wall faces")
        low = min(negative, key=lambda x: x[1][axis])
        high = max(positive, key=lambda x: x[1][axis])
        limits.append((low[1][axis] + low[2][axis], high[1][axis] - high[2][axis]))
    xs = np.linspace(limits[0][0] + 1e-5, limits[0][1] - 1e-5, 9)
    ys = np.linspace(limits[1][0] + 1e-5, limits[1][1] - 1e-5, 9)
    lid_bodies = ep.subtree(root) - base
    if part:
        lid_bodies, _, _ = G._resolve_set(gc, part, ())
    if not lid_bodies: raise ValueError("no independently resolved cover part")
    from scipy.spatial import ConvexHull
    from itertools import product
    aperture = np.array(list(product(limits[0], limits[1])))
    continuous_cover = []
    for g in ep.geoms(lid_bodies, collision=False):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX or model.geom_rgba[g, 3] < .95:
            continue
        corners = np.array(list(product((-1., 1.), repeat=3))) * model.geom_size[g]
        world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        projection = ((world - origin) @ R)[:, :2]
        hull = ConvexHull(projection)
        if np.all(aperture @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-9):
            continuous_cover.append(int(g))
    if not continuous_cover:
        return G._report(False, ["no opaque solid panel continuously covers the entire reconstructed cavity opening"],
                         continuous_cover=False, scope="convex solid-panel projection over all four inner-wall corners")
    gid = np.zeros(1, dtype=np.int32)
    failed, gaps = [], []
    for x in xs:
        for y in ys:
            local = np.array([x, y, rim - .001])
            start = origin + R @ local
            distance = mujoco.mj_ray(model, data, start, R[:, 2].copy(), None, 1, -1, gid)
            g = int(gid[0])
            own = g >= 0 and int(model.geom_bodyid[g]) in lid_bodies and model.geom_rgba[g, 3] >= .95
            gap = float(distance - .001) if distance >= 0 else None
            if not own or gap is None or abs(gap) > .003:
                failed.append([float(x), float(y), gap])
            if gap is not None: gaps.append(gap)
    return G._report(not failed, [] if not failed else [f"{len(failed)}/81 interior rays have no opaque cover seated within 3 mm of the rim"],
                     rays=81, failed_rays=len(failed), failures=failed[:5], max_gap_m=max(gaps) if gaps else None,
                     continuous_cover=True, cover_geoms=continuous_cover,
                     scope="continuous solid-panel coverage plus 9 by 9 seating ray grid")


def knot_layout(points, central=False, opposite=False):
    """Locate a demonstrably knotted contiguous subarc, then test its placement.

    Greedy trimming must preserve the four-closure knot verdict at every step.
    Taking the union of both trimming orders is conservative about the core's
    size. The centre must lie in the middle half of the rope's arc length;
    opposing ends must project beyond opposite sides of the complete core.
    This is geometric localisation of a single recognised knot, not a proof
    for arbitrary tangled curves. Ambiguous closures fail closed.
    """
    from tool_library.common import task_gates as G
    from tool_library.common import knot as K
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 8 or not np.isfinite(P).all():
        return G._report(False, ["knot layout requires a finite cable centreline with at least eight points"])
    cache = {}
    def knotted(a, b):
        if (a, b) not in cache:
            cache[a, b] = K.check(P[a:b + 1])
        q = cache[a, b]
        return bool(q["agree"] and q["knot"] and q["type"] == whole["type"])
    whole = K.check(P)
    if not whole["agree"] or not whole["knot"]:
        return G._report(False, ["cannot locate a knot core: the full rope has no unambiguous knot"], type=whole["type"])
    if whole["type"] not in ("trefoil", "figure-eight"):
        return G._report(False, ["cannot locate a certified knot core: the named topology is unresolved; equal 'other' labels do not establish the same knot"], type=whole["type"])
    cores = []
    for order in (0, 1):
        a, b = 0, len(P) - 1
        for side in (order, 1 - order):
            while b - a >= 8:
                aa, bb = (a + 1, b) if side == 0 else (a, b - 1)
                if not knotted(aa, bb): break
                a, b = aa, bb
        cores.append((a, b))
    a, b = min(x[0] for x in cores), max(x[1] for x in cores)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    total = float(lengths[-1])
    if total <= 1e-8: return G._report(False, ["rope has zero measured length"])
    mid_fraction = float((lengths[a] + lengths[b]) / (2 * total))
    issues = []
    if central and not .25 <= mid_fraction <= .75:
        issues.append(f"knot core centre is at {mid_fraction:.3f} of rope length, outside its middle half")
    core = P[a:b + 1]
    centre = np.mean(core, axis=0)
    # Opposite means separated by the knot, not merely separated from each other.
    axis = P[-1] - P[0]
    axis /= max(np.linalg.norm(axis), 1e-12)
    projected = (core - centre) @ axis
    ends = (P[[0, -1]] - centre) @ axis
    opposite_ok = bool(ends[0] < np.min(projected) - .001 and ends[1] > np.max(projected) + .001)
    if opposite and not opposite_ok:
        issues.append("the two rope ends do not extend beyond opposite sides of the measured knot core")
    return G._report(not issues, issues, type=whole["type"], core_indices=[int(a), int(b)],
                     core_mid_arclength_fraction=mid_fraction, opposite_sides=opposite_ok,
                     endpoint_projections_m=ends.tolist(), core_projection_range_m=[float(np.min(projected)), float(np.max(projected))],
                     scope="four-closure-preserving subarc localisation; middle half of arc length and end-to-end projection")
