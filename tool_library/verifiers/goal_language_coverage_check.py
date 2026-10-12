"""Can Stage 3's goal gate measure every clause of the Stage 1 goals?"""
from __future__ import annotations

TOOL = {
    "name": "goal_language_coverage_check",
    "title": "Goal language coverage check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "Compiles every Stage 1 subgoal goal and task_success with the goal language Stage 3's stage1_goal_gate uses, without a "
               "scene, and lists each clause it cannot turn into a measurement, so Stage 1 rephrases it while it still can.",
    "description": (
        "Stage 3 accepts an episode only when stage1_goal_gate measures every clause of the Stage 1 goals (held, clear of a "
        "support, above an opening, inside or on an object, upright, unmoved, closed/open, released, knotted ...). A clause it "
        "cannot measure ('upright in the pinch' before the vocabulary knew it, 'gently', 'nicely') makes Stage 3 unpassable "
        "whatever the robot does. This check runs the same compiler on the Stage 1 file alone: object names resolve against "
        "Stage 1's objects and parts, and only the language is judged (a clause that needs the scene to resolve, such as "
        "which drawer is meant, is left to Stage 3; any other error of the compiler fails the check). Each uncovered clause "
        "is reported with its sentence and the measurable phrasings to use instead."
    ),
    "inputs": [
        {"name": "path", "type": "path", "description": "The scene_semantics.json to check."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues (one per uncovered clause, or per goal the compiler failed on, naming the JSON path); clauses {path: [clause, covered]}."},
    ],
    "status": "tested",
    "entrypoint": "goal_language_coverage_check",
    "requirements": ["mujoco>=3.2", "numpy"],
    "tags": ["stage 1", "semantic", "goals", "stage 3 goal gate"],
    "used_in": ["Place Cup in Bowl"],
    "cli": "python -m tool_library.verifiers.goal_language_coverage_check runs/my_task/stage1/scene_semantics.json",
}

import json
from pathlib import Path

SUGGEST = ("measurable phrasings: 'X is held in the left hand', 'X is held clear of the table', 'X is above the Y's opening', "
           "'X rests inside Y', 'X rests on Y', 'X stands upright', 'Y has not moved', 'the lid is closed', 'X is released', "
           "'X has a knot in it'; one relation per clause, separated by commas or 'and'")


class _NoScene:
    """An episode without a scene: every body lookup fails, so only the language is compiled."""
    model = None

    def body(self, name):
        raise ValueError(f"no scene for {name!r}")


def _needs_scene(exc, G):
    """Whether compile_final stopped because a clause needs the scene: a body or part that cannot be resolved without
    one (ValueError, the goal gate's unresolved reference) or a read of the absent scene (an AttributeError on this
    scene-less episode, or a model field read off its missing model). Anything else is an error of the compiler."""
    if isinstance(exc, (ValueError, G._Unresolved)):
        return True
    if not isinstance(exc, AttributeError):
        return False
    import mujoco
    obj, name = getattr(exc, "obj", _NoScene), getattr(exc, "name", None)
    return isinstance(obj, _NoScene) or (obj is None and bool(name) and hasattr(mujoco.MjModel, name))


def goal_language_coverage_check(path):
    from tool_library.common import task_gates as G
    from tool_library.common.final_goal_language import compile_final
    sem = json.loads(Path(path).read_text())
    gc = G.GoalContext(_NoScene(), sem, None)
    texts = [(f"$.subgoals[{i}].goal", str(s.get("goal") or "")) for i, s in enumerate(sem.get("subgoals") or [])]
    if sem.get("task_success"):
        texts.append(("$.task_success", str(sem["task_success"])))
    issues, clauses = [], {}
    for where, text in texts:
        if not text.strip():
            issues.append(f"{where}: empty goal; state the observable end state")
            continue
        try:
            _, coverage, errs = compile_final(gc, text)
        except Exception as exc:
            if _needs_scene(exc, G):                  # a clause that needs the scene to resolve: Stage 3 judges it
                clauses[where] = [{"clause": text, "covered": None, "note": f"needs the scene ({type(exc).__name__})"}]
                continue
            clauses[where] = [{"clause": text, "covered": False, "note": f"compiler error ({type(exc).__name__})"}]
            issues.append(f"{where}: the goal compiler failed on \"{text[:160]}\" ({type(exc).__name__}: {str(exc)[:200]}), "
                          "so its coverage is unknown; rephrase the goal or report the error")
            continue
        clauses[where] = [{"clause": c.get("clause"), "covered": bool(c.get("inferred_coverage"))} for c in coverage]
        for c in coverage:
            if not c.get("inferred_coverage"):
                issues.append(f"{where}: clause '{c.get('clause')}' of \"{text[:160]}\" cannot be measured by Stage 3's goal gate; "
                              f"rephrase it ({SUGGEST})")
    return {"ok": not issues, "issues": issues, "clauses": clauses}


if __name__ == "__main__":
    import sys
    print(json.dumps(goal_language_coverage_check(sys.argv[1]), indent=1))
