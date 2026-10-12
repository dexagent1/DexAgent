"""Score a scene_semantics.json against an annotated reference.

    python stage1_semantic_understanding/tests/compare_semantics.py predicted.json reference.json

Objects are matched greedily by name/part word overlap plus type and role agreement; subgoals are
aligned as an ordered sequence (longest common subsequence) on (acting hand, motion family, main
object). The numbers are a guide for prompt engineering, not a benchmark: read both files too.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MOTION_FAMILY = {"grasp_lift": "take", "regrasp": "take", "carry": "move", "place": "put", "release": "put", "hold": "hold",
                 "push": "actuate", "pull": "actuate", "press": "actuate", "open_joint": "actuate", "close_joint": "actuate",
                 "unscrew": "actuate", "screw": "actuate", "insert": "put", "pour": "pour", "tilt": "pour", "cut": "cut",
                 "write": "write", "wipe": "wipe", "tie": "tie", "fold": "actuate", "other": "other"}


def words(s) -> set:
    return {w for w in re.findall(r"[a-z]+", json.dumps(s).lower()) if len(w) > 2}


def obj_sig(o: dict) -> set:
    return words(o.get("name", "")) | words(o.get("id", "")) | words(o.get("parts", []))


def match_objects(pred: list, ref: list):
    pairs, used = [], set()
    for r in ref:
        best, bs = None, 0.0
        for i, p in enumerate(pred):
            if i in used:
                continue
            a, b = obj_sig(p), obj_sig(r)
            s = len(a & b) / max(1, len(a | b))
            s += 0.3 * (p.get("type") == r.get("type")) + 0.1 * (p.get("role") == r.get("role"))
            if s > bs:
                best, bs = i, s
        if best is not None and bs >= 0.35:
            pairs.append((best, r)); used.add(best)
    return pairs


def sub_key(s: dict, objmap: dict) -> tuple:
    refs = [objmap.get(r.split(":")[0], r.split(":")[0]) for r in s.get("object_refs", [])]
    return (s.get("acting_hand"), MOTION_FAMILY.get(s.get("motion"), "other"), refs[0] if refs else "")


def lcs(a: list, b: list, eq) -> int:
    n, m = len(a), len(b)
    T = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            T[i + 1][j + 1] = T[i][j] + 1 if eq(a[i], b[j]) else max(T[i][j + 1], T[i + 1][j])
    return T[n][m]


def compare(pred: dict, ref: dict) -> dict:
    po, ro = pred.get("objects", []), ref.get("objects", [])
    pairs = match_objects(po, ro)
    type_ok = sum(po[i].get("type") == r.get("type") for i, r in pairs)
    objmap = {po[i]["id"]: r["id"] for i, r in pairs}
    ps = [sub_key(s, objmap) for s in pred.get("subgoals", [])]
    rs = [sub_key(s, {r["id"]: r["id"] for r in ro}) for s in ref.get("subgoals", [])]
    full = lcs(ps, rs, lambda x, y: x == y)
    hand_motion = lcs(ps, rs, lambda x, y: x[0] == y[0] and x[1] == y[1])
    hands_pred = set((pred.get("hands") or {}).get("used", [])); hands_ref = set((ref.get("hands") or {}).get("used", []))
    return {
        "objects": {"reference": len(ro), "predicted": len(po), "matched": len(pairs), "type_correct": type_ok,
                    "missed": [r["id"] for r in ro if r not in [x[1] for x in pairs]],
                    "extra": [po[i]["id"] for i in range(len(po)) if i not in {x[0] for x in pairs}]},
        "subgoals": {"reference": len(rs), "predicted": len(ps), "aligned_hand_motion_object": full, "aligned_hand_motion": hand_motion,
                     "recall": round(full / max(1, len(rs)), 2), "precision": round(full / max(1, len(ps)), 2)},
        "hands_used_match": hands_pred == hands_ref,
    }


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    pred, ref = json.loads(Path(a[0]).read_text()), json.loads(Path(a[1]).read_text())
    print(json.dumps(compare(pred, ref), indent=1))


if __name__ == "__main__":
    main()
