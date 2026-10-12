"""Articulation axis semantics check: each articulated object's joints are of the kind and direction Stage 1 described."""
from __future__ import annotations

TOOL = {
    "name": "articulation_axis_semantics_check",
    "title": "Articulation axis semantics check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "For every Stage 1 articulation (prismatic, revolute, screw, free part), finds joints of the matching MuJoCo type in the object and checks their world axis, opening direction and hinge location at keyframe 0 against the direction words of the Stage 1 description (vertical, horizontal, left-right, toward the operator, along the long axis, the back wall, down and left in the image).",
    "description": (
        "A drawer whose slide opens away from the operator, a lid hinged on the front wall instead of the back, or a "
        "screw cap on a horizontal axis still compile and pass a joint-travel test, and Stage 3 then pulls in the "
        "wrong direction. Stage 1 states each articulation in words (articulation.joint_type and .axis). The check maps "
        "joint_type to MuJoCo joints in the object's bodies (prismatic -> slide, revolute -> hinge, screw -> a hinge "
        "and a slide, free_part -> a free joint or a weld on a part; other -> not judged) and reads cues from the axis "
        "text: intrinsic axis cues use the first clause; directional clauses and parentheticals are conditioned on "
        "opening or closing when specified. Conditions about changing object poses are not axis constraints. 'vertical', 'up', "
        "'down' -> axis within angle_tol_deg of world z; 'horizontal' -> within angle_tol_deg of the table plane; "
        "'left-right', 'side to side', 'image's horizontal' -> along the camera's horizontal (world x without a "
        "camera); 'toward the operator/camera/person' -> the opening direction (the joint's larger range side) "
        "points to -y within 60 deg, 'away from the operator', 'into the work' -> +y. Directions explicitly describing "
        "closing are reversed before checking opening; toward-and-away travel without an opening condition checks "
        "the unsigned operator axis. Contradictory or unresolved opening/closing cues fail with that reason. 'long axis' -> along the "
        "longest dimension of the object's fixed body (skipped when its two largest dimensions are within 15%); "
        "'back/rear/far' or 'front/near' with 'wall/edge/side' -> the hinge anchor lies on that side of the fixed "
        "body's centre; '<up|down|left|right> in the image' (with a camera) -> the projected opening direction lies within 90 deg of the "
        "combined image direction (Stage 1's image words are approximate). An "
        "articulation passes when some joint of the right type satisfies every cue; no cue means only the type is "
        "judged. Assumes the rig frame (operator at -y, z up)."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF."},
        {"name": "semantic", "type": "str | dict", "description": "scene_semantics.json (Stage 1)."},
        {"name": "recon", "type": "str | dict | None", "description": "scene_recon.json mapping Stage 1 ids to bodies."},
        {"name": "camera", "type": "str | dict | None", "description": "camera.json for image-direction cues and the image horizontal."},
        {"name": "angle_tol_deg", "type": "float", "description": "Allowed angle between a joint axis and a cued axis. Default 30."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; articulations [{id, joint_type, cues, candidates [{joint, type, axis, open_dir, anchor, failed}], verdict}]."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "articulation", "joint", "axis", "semantics", "stage1", "stage2"],
    "status": "tested",
    "entrypoint": "articulation_axis_semantics_check",
}

import math
import re

import mujoco
import numpy as np

from tool_library.common import rigcam as RC
from tool_library.common import scene_assert as SA

TYPES = {"prismatic": ("slide",), "revolute": ("hinge",), "screw": ("hinge", "slide"), "free_part": ("free",)}
MJT = {int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge", int(mujoco.mjtJoint.mjJNT_FREE): "free",
       int(mujoco.mjtJoint.mjJNT_BALL): "ball"}
CONDITIONAL = re.compile(r"\b(while|when|during|after|before|once|if)\b")


_OPEN = re.compile(r"\b(open|opens|opened|opening)\b")
_CLOSE = re.compile(r"\b(close|closes|closed|closing|shut|shuts|shutting)\b")
_TOWARD = re.compile(r"towards? (?:the )?(?:operator|camera|person|viewer|you|user)\b")
_AWAY = re.compile(r"away from (?:the )?(?:operator|camera|person|viewer|you|user)\b|into the work")


def _motion_condition(clause):
    if re.search(r"closed\s*(?:to|->|→)\s*open\b", clause):
        return "opening"
    if re.search(r"open\s*(?:to|->|→)\s*closed\b", clause):
        return "closing"
    opening, closing = bool(_OPEN.search(clause)), bool(_CLOSE.search(clause))
    if opening and closing:
        return "ambiguous"
    return "opening" if opening else "closing" if closing else None


def _operator_cues(text):
    """Translate motion-conditioned directions to the joint's opening sign.

    Explicitly reversible, unconditioned travel specifies an axis, not which
    end of its range is open. Contradictory opening cues stay contradictory.
    """
    reversible = re.search(r"towards?\s*(?:and|/)\s*away(?: from)? (?:the )?(?:operator|camera|person|viewer|you|user)", text)
    if reversible and not (_OPEN.search(text) or _CLOSE.search(text)):
        return {"operator_axis"}
    cues, inherited = set(), None
    for clause in re.split(r"[,;.]|\b(?:and|but)\b", text):
        condition = _motion_condition(clause)
        directions = []
        if _TOWARD.search(clause):
            directions.append("toward")
        if _AWAY.search(clause):
            directions.append("away")
        if not directions:
            # Only explicit clause-leading conditions propagate over a comma:
            # 'when closing, away from the operator'.
            inherited = condition if re.search(r"\b(when|while|during|on|for|to)\b|->|→", clause) else None
            continue
        condition = condition or inherited
        inherited = None
        if condition == "ambiguous":
            cues.add("ambiguous_operator_direction")
            continue
        for direction in directions:
            toward = (direction == "toward") != (condition == "closing")
            cues.add("open_toward_operator" if toward else "open_away")
    if (cues == {"open_toward_operator", "open_away"}
            and not (_OPEN.search(text) or _CLOSE.search(text))
            and re.search(r"\band\b|/|back and forth|bidirectional|both directions", text)):
        return {"operator_axis"}
    return cues


def parse_cues(text):
    """Direction cues from a Stage 1 axis description."""
    t = str(text or "").lower()
    parens = list(re.finditer(r"\(([^)]*)\)", t))
    main = re.sub(r"\([^)]*\)", " ", t).split(";")[0]
    cues = set()
    w = lambda pat, s: re.search(pat, s) is not None
    if w(r"\bvertical\b|\bupright\b", main) or (w(r"\b(up|down|downward|upward)\b", main) and not w(r"in the image", main)):
        cues.add("vertical")
    if w(r"\bhorizontal\b", main) and "vertical" not in cues:
        cues.add("horizontal")
    if w(r"left[- ]right|side to side|sideways|image'?s horizontal", main):
        cues.add("image_horizontal")
    cues.update(_operator_cues(re.sub(r"\([^)]*\)", " ", t)))
    if w(r"long(est)? axis|lengthwise", main):
        cues.add("long_axis")
    if w(r"\b(back|rear|far)\b[^,.;]*\b(wall|edge|side)\b", main):
        cues.add("anchor_back")
    elif w(r"\b(front|near)\b[^,.;]*\b(wall|edge|side)\b", main):
        cues.add("anchor_front")
    for match in parens:
        p = match.group(1)
        prefix = re.split(r"[,;.]|\b(?:and|but)\b", t[:match.start()])[-1]
        condition = _motion_condition(p) or _motion_condition(prefix)
        if not CONDITIONAL.search(p) or condition in {"opening", "closing"}:
            cues.update(_operator_cues(p if _motion_condition(p) else f"{condition or ''} {p}"))
        if CONDITIONAL.search(p) and condition not in {"opening", "closing"}:
            continue
        if w(r"image'?s horizontal|left[- ]right", p):
            cues.add("image_horizontal")
        if w(r"in the image", p):
            for word in ("up", "down", "left", "right"):
                if w(rf"\b{word}\b", p):
                    direction = {"up": "down", "down": "up", "left": "right", "right": "left"}[word] if condition == "closing" else word
                    cues.add(f"image_{direction}")
    return sorted(cues)


def articulation_axis_semantics_check(scene, semantic, recon=None, camera=None, angle_tol_deg=30.0):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    st = SA.load_setup(None)
    cam = SA.load_camera(camera) if camera is not None else None
    sem = SA.semantic_objects(semantic)
    mapping = SA.stage1_groups(model, semantic, recon, ref=st["model"], cfg=st["cfg"])
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    cos_t = math.cos(math.radians(angle_tol_deg))
    if cam is not None:
        h_dir = np.asarray(cam["R_cv"], float)[:, 0].copy(); h_dir[2] = 0
        h_dir /= max(np.linalg.norm(h_dir), 1e-9)
    else:
        h_dir = np.array([1.0, 0.0, 0.0])
    rows, issues = [], []
    for o in sem:
        art = o.get("articulation")
        if not isinstance(art, dict):
            continue
        oid = str(o.get("id"))
        jt = str(art.get("joint_type", "other"))
        if jt not in TYPES:
            rows.append({"id": oid, "joint_type": jt, "verdict": "not judged"})
            continue
        if oid not in mapping:
            rows.append({"id": oid, "joint_type": jt, "verdict": "simulated in Isaac Sim"})
            continue
        g = mapping.get(oid)
        if g is None or g["root"] < 0:
            issues.append(f"articulated object {oid} has no body in the scene")
            rows.append({"id": oid, "joint_type": jt, "verdict": "missing"})
            continue
        cues = parse_cues(art.get("axis", ""))
        root = g["root"]
        joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in g["bodies"]]
        cands = [j for j in joints if MJT.get(int(model.jnt_type[j])) in TYPES[jt] and not (jt != "free_part" and int(model.jnt_bodyid[j]) == root and MJT.get(int(model.jnt_type[j])) == "free")]
        if jt == "free_part":
            cands = [j for j in joints if MJT.get(int(model.jnt_type[j])) == "free" and int(model.jnt_bodyid[j]) != root]
            welded = [e for e in range(model.neq) if int(model.eq_type[e]) == int(mujoco.mjtEq.mjEQ_WELD)
                      and (int(model.eq_obj1id[e]) in g["bodies"] or int(model.eq_obj2id[e]) in g["bodies"])]
            if not cands and not welded:
                issues.append(f"{oid}: Stage 1 says a detachable part (free_part) but the object has no free part or weld; build the part as its own body")
                rows.append({"id": oid, "joint_type": jt, "cues": cues, "verdict": "no free part"})
            else:
                rows.append({"id": oid, "joint_type": jt, "cues": cues, "verdict": "ok"})
            continue
        present = {MJT.get(int(model.jnt_type[j])) for j in cands}
        missing_types = [t for t in TYPES[jt] if t not in present]
        if missing_types:
            have = sorted({MJT.get(int(model.jnt_type[j]), "?") for j in joints if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)})
            issues.append(f"{oid}: Stage 1 says {jt} but the object has no {'/'.join(missing_types)} joint (it has {have or 'none'}); "
                          f"build the {art.get('moving_part', 'moving part')} on a {'/'.join(TYPES[jt])} joint")
            rows.append({"id": oid, "joint_type": jt, "cues": cues, "candidates": [], "verdict": "wrong joint type"})
            continue
        # the fixed body's geometry: centre and long axis
        fixed = [b for b in g["bodies"] if b == root or not any(int(model.jnt_bodyid[j]) == b for j in joints if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE))]
        P = [SA.geom_world_points(model, data, x) for x in SA.body_geoms(model, fixed) if SA.geom_collidable(model, x) or SA.geom_visible(model, x)]
        P = np.vstack([p for p in P if len(p)]) if any(len(p) for p in P) else data.xpos[root][None]
        centre = P.mean(axis=0)
        Q = P - centre
        long_ax, long_ok = None, False
        if len(P) >= 3:
            _, sv, Vt = np.linalg.svd(Q, full_matrices=False)
            ext = np.ptp(Q @ Vt.T, axis=0)
            order = np.argsort(ext)[::-1]
            long_ax = Vt[order[0]]
            long_ok = ext[order[0]] > 1.15 * ext[order[1]]
        cand_rows = []
        passed = False
        for j in cands:
            ax = np.array(data.xaxis[j], float)
            ax /= max(np.linalg.norm(ax), 1e-12)
            lo, hi = (float(x) for x in model.jnt_range[j]) if model.jnt_limited[j] else (0.0, 1.0)
            open_dir = ax if abs(hi) >= abs(lo) else -ax
            anchor = np.array(data.xanchor[j], float)
            failed = []
            for c in cues:
                if c == "ambiguous_operator_direction":
                    failed.append("opening/closing direction is ambiguous in the semantic description")
                elif c == "operator_axis" and MJT[int(model.jnt_type[j])] == "slide" and abs(ax[1]) < 0.5:
                    failed.append("not along the toward/away operator axis")
                elif c == "vertical" and abs(ax[2]) < cos_t:
                    failed.append("not vertical")
                elif c == "horizontal" and abs(ax[2]) > math.sin(math.radians(angle_tol_deg)):
                    failed.append("not horizontal")
                elif c == "image_horizontal" and abs(ax @ h_dir) < cos_t:
                    failed.append("not left-right")
                elif c == "open_toward_operator" and MJT[int(model.jnt_type[j])] == "slide" and open_dir @ np.array([0, -1.0, 0]) < 0.5:
                    failed.append("does not open toward the operator")
                elif c == "open_away" and MJT[int(model.jnt_type[j])] == "slide" and open_dir @ np.array([0, 1.0, 0]) < 0.5:
                    failed.append("does not open away from the operator")
                elif c == "long_axis" and long_ok and abs(ax @ long_ax) < cos_t:
                    failed.append("not along the long axis")
                elif c in ("anchor_back", "anchor_front") and MJT[int(model.jnt_type[j])] == "hinge":
                    dy = float(anchor[1] - centre[1])
                    if (c == "anchor_back" and dy <= 0) or (c == "anchor_front" and dy >= 0):
                        failed.append(f"hinge on the {'front' if dy < 0 else 'back'} side")
                elif c.startswith("image_") and c != "image_horizontal":
                    continue                                 # judged together below
            img = [c.split("_", 1)[1] for c in cues if c in ("image_up", "image_down", "image_left", "image_right")]
            if img and cam is not None and MJT[int(model.jnt_type[j])] == "slide":
                # one combined image direction ("down and left"); fail only when the joint opens more than 90 deg away from
                # it, since Stage 1's image words are approximate (a drawer opening down and slightly right was called
                # "down and left")
                want = np.array([{"left": -1, "right": 1}.get(x, 0) for x in img]).sum(), np.array([{"up": -1, "down": 1}.get(x, 0) for x in img]).sum()
                want = np.array(want, float)
                uv, z = RC.project(cam, np.stack([anchor, anchor + 0.05 * open_dir]))
                d = uv[1] - uv[0]
                if np.linalg.norm(want) > 0 and np.linalg.norm(d) > 1e-6 and float(d @ want) / (np.linalg.norm(d) * np.linalg.norm(want)) < 0.0:
                    failed.append(f"opens toward image ({d[0]:+.0f}, {d[1]:+.0f}) px, not {' and '.join(img)} in the image")
            cand_rows.append({"joint": model.joint(j).name, "type": MJT[int(model.jnt_type[j])], "axis": SA.r6(ax, 3), "open_dir": SA.r6(open_dir, 3),
                              "anchor": SA.r6(anchor, 4), "failed": failed})
            if not failed:
                passed = True
        if jt == "screw" and cues:
            # a screw needs both its hinge and its slide on the cued axis
            passed = all(any(not r["failed"] for r in cand_rows if r["type"] == t) for t in ("hinge", "slide"))
        verdict = "ok" if passed or not cand_rows else "wrong direction"
        if verdict != "ok" and ("ambiguous_operator_direction" in cues or
                                 {"open_toward_operator", "open_away"}.issubset(cues)):
            issues.append(f"{oid}: opening/closing direction cues are ambiguous or contradictory in "
                          f"'{art.get('axis')}'; the sign cannot be verified from this description. "
                          "Review the source direction evidence; changing a joint axis cannot resolve contradictory cues.")
        elif verdict != "ok":
            best = min(cand_rows, key=lambda r: len(r["failed"]))
            issues.append(f"{oid}: Stage 1 describes the {jt} axis as '{art.get('axis')}' but no {'/'.join(TYPES[jt])} joint matches "
                          f"(closest {best['joint']} axis {best['axis']}: {', '.join(best['failed'])}); orient the joint as the video shows")
        rows.append({"id": oid, "joint_type": jt, "cues": cues, "candidates": cand_rows, "verdict": verdict})
    return SA.report(issues, articulations=rows)
