"""Does a Stage 2 scene_recon.json account for every Stage 1 object, with an asset, a pose, a mask fit and the verifiers its type needs?"""
from __future__ import annotations

TOOL = {
    "name": "scene_recon_manifest_check",
    "title": "Scene reconstruction manifest check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "Checks a Stage 2 scene_recon.json against the Stage 1 scene_semantics.json: every object has an entry with the same type, a reconstruction skill that exists in the library, an asset file, a root body present in the compiled scene, a fitted pose, a mask with its silhouette IoU above a floor, a verification plan and passing verifier reports covering every Stage 1 property (a mechanism test for articulated objects), and for deformable objects their Isaac Sim cord (specification with settled poses, clean settling report, no MuJoCo copy) or, for one a subgoal cuts, a MuJoCo cable with cut joins; the scene, camera and comparison image exist.",
    "description": (
        "Form and bookkeeping of the Stage 2 deliverable so that Stage 3 can rely on it. Reads both JSON files and the "
        "library index. For each Stage 1 object (static ones may be omitted when `allow_missing_static`): an entry keyed "
        "by the same id; `type` equal to Stage 1's; `skill` naming a library skill of category Object Reconstruction (or "
        "`skill_origin: created` with a tool that now exists); `asset_xml` existing; `root_body` found in the compiled "
        "scene (deformables: the Isaac report instead); `pose` with x, y, yaw_deg; `mask` existing and `iou` >= min_iou; "
        "an object hidden at source frame 0 declares `hidden_at_frame0` {reason, reveal_frame, reveal_mask?} and needs no IoU (it "
        "must not be static in Stage 1; scene_qc tests that it renders invisible); "
        "a group object (Stage 1 `pieces`) lists its separate pieces' top-level root bodies under `parts` (root_body one of "
        "them, the count Stage 1 states; without Stage 1 `pieces` only similar pieces are accepted); "
        "`verifiers`: a dict of report dicts each with ok true, including at least one Reconstruction Mechanism Test for "
        "articulated objects and articulation_check or joint evidence; deformable objects simulated in Isaac Sim (entry "
        "`simulator: isaac`, or skill build_isaac_scene; the scene's simulator mujoco+isaac): no asset_xml or root_body, "
        "`isaac` with cord_json (an isaac_cord specification with its settled link poses), report_json (a passing "
        "settling report of that cord, no PhysX errors), sim_png, scene_usda (or .gz) and a drift below max_drift_m, and "
        "no MuJoCo body for it in scene.xml (isaac_deformable_rest_check re-measures the physics from these files); "
        "Isaac Sim cannot cut, so a deformable a `cut` subgoal names must instead be a MuJoCo cable: "
        "`simulation_model.kind=mujoco_cable`, its root_body (or one of its bodies) in scene.xml and `cut_joins` whose "
        "equalities and bodies are in scene.xml (construct_cuttable_rope), and any manipulated deformable not simulated "
        "in Isaac Sim must be such a MuJoCo cable (`require_mujoco_deformables=False` skips the subgoal-driven rules). "
        "Top level: scene_xml compiles, camera_json loads, frame0_compare_png "
        "exists, and the simulator field says mujoco or mujoco+isaac consistently with the verified backends; MuJoCo-only never claims Isaac rendering verification. Every finding "
        "names the object and the key to fix. Explicit full-containment final goals are independently replayed with "
        "rest_inside_container_test on the trusted scene and body identities; recorded reports may propose a placement "
        "orientation, but cannot certify success or relax the final gate's fixed 2 mm tolerance. Ordinary partial-inside "
        "goals do not require full containment. Articulated mechanism reports are proposals, never certificates: "
        "supported articulated_joint_travel_test claims are independently replayed on the current scene component "
        "with fixed verifier defaults after local asset/scene physics binding. This proves isolated fixed-base "
        "component mobility only. Specialized/coupled mechanism claims without an independent adapter fail closed."
    ),
    "inputs": [
        {"name": "path", "type": "path", "description": "scene_recon.json."},
        {"name": "scene_semantic_json", "type": "path | None", "description": "Stage 1 output; default: the manifest's scene_semantic field."},
        {"name": "library_index", "type": "path | None", "description": "tool_library/index.json; default: the repository's."},
        {"name": "min_iou", "type": "float", "description": "Per-object silhouette IoU floor. Default 0.4."},
        {"name": "max_drift_m", "type": "float", "description": "Largest mean settling drift accepted for a deformable. Default 0.02."},
        {"name": "allow_missing_static", "type": "bool", "description": "Stage 1 objects with role static may be absent. Default True."},
        {"name": "require_mujoco_deformables", "type": "bool", "description": "A deformable that Stage 1 subgoals manipulate must be simulated somewhere (Isaac Sim with its cord, else a MuJoCo cable in scene.xml), and one that a cut subgoal names must be a MuJoCo cable with cut_joins (Isaac Sim has no cutting). Default True."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; per_object: {id: {found, iou, verifiers_ok, mechanism_evidence_ok, ...}}; counts; full_containment_trials and independent_mechanism_trials with scope, supported/unsupported claims, physical bindings and freshly replayed reports."},
    ],
    "status": "tested",
    "entrypoint": "scene_recon_manifest_check",
    "requirements": ["mujoco>=3.1"],
    "tags": ["stage 2", "manifest", "reconstruction", "bookkeeping"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.verifiers.scene_recon_manifest_check runs/my_task/stage2/scene_recon.json",
}

import json
import math
import os
import sys
from pathlib import Path

MECH_CATEGORY = "Reconstruction Mechanism Tests"
DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "index.json"


def _resolve(base: Path, p):
    if p is None:
        return None
    q = Path(p)
    if q.is_absolute():
        return q
    return (base / q) if (base / q).exists() or not q.exists() else q.resolve()


def _group_parts(oid, so, rec, model, claimed):
    """Issues and info for a group object's `parts` (its separate pieces' root bodies; piece_groups) against the
    scene and Stage 1's `pieces`."""
    from tool_library.common import piece_groups as PG
    issues, info = [], {}
    raw = rec.get("parts")
    pieces = PG.stage1_pieces(so)
    where = f"$.objects[{oid}].parts"
    if raw is not None and (not isinstance(raw, list) or not all(isinstance(x, str) and x for x in raw)):
        return [f"{where}: a list of the pieces' root body names"], info
    parts = PG.recon_parts(rec)
    if raw is not None and len(parts) != len(raw):
        issues.append(f"{where}: lists a body more than once")
    if not parts:
        if pieces and pieces["loose"] and so.get("type") != "deformable":
            issues.append(f"{where}: Stage 1 says {oid!r} is {PG.count_phrase(pieces)} separate loose {pieces['kind'] or 'piece'}(s); "
                          "build each piece as its own free body and list every piece's root body under `parts`")
        return issues, info
    info["parts"] = len(parts)
    if len(parts) < 2:
        issues.append(f"{where}: a group object has at least two pieces; a single-body object has no `parts`")
    rb = rec.get("root_body")
    if rb and rb not in parts:
        issues.append(f"{where}: root_body {rb!r} must be one of the pieces (the one the first contact acts on)")
    ids = []
    if model is not None:
        import mujoco
        for nm in parts:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm)
            if b < 0:
                issues.append(f"{where}: {nm!r} is not a body of scene_xml")
                continue
            if int(model.body_parentid[b]) != 0:
                issues.append(f"{where}: {nm!r} is not a top-level body: each piece is its own body staged under the world, "
                              f"not a child of {model.body(int(model.body_parentid[b])).name!r}")
            ids.append(b)
    for nm in parts:
        if claimed.get(nm, oid) != oid:
            issues.append(f"{where}: {nm!r} already belongs to {claimed[nm]!r}")
        claimed.setdefault(nm, oid)
    if pieces is not None:
        info["stage1_pieces"] = pieces
        if not PG.count_matches(len(parts), pieces):
            issues.append(f"{where}: {len(parts)} pieces, but Stage 1 says {PG.count_phrase(pieces)} {pieces['kind'] or 'piece'}(s)")
    elif model is not None and len(ids) >= 2:
        # Stage 1 does not declare pieces (an older description): accept parts only when they are several of the same
        # thing, never an object and the part that belongs on it (a bottle and its cap still need their joint)
        import mujoco
        d = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        mujoco.mj_forward(model, d)
        odd = PG.dissimilar_parts(model, d, ids)
        info["stage1_pieces"] = None
        if odd:
            issues.append(f"{where}: Stage 1 declares no `pieces` for {oid!r} and the parts are not alike ({odd[:3]} differ from the "
                          "median piece by more than 3x): only separate, similar pieces may be declared as parts; a part that belongs "
                          "on the object needs its joint or weld, and a Stage 1 group needs `pieces`")
    return issues, info


def _isaac_only(rec):
    from tool_library.common.scene_assert import is_isaac_only
    return is_isaac_only(rec)


def _isaac_cord_entry(oid, rec, base, sim, body_names, max_drift_m):
    """Bookkeeping of a deformable simulated in Isaac Sim only: its isaac block's files exist, the cord specification
    loads with settled link poses, the settling report is clean, and scene.xml has no MuJoCo copy of it."""
    import re
    import numpy as np
    from tool_library.common import isaac_cord as IC
    issues, info, where = [], {"simulation_scope": "isaac_cord"}, f"$.objects[{oid}]"
    if sim != "mujoco+isaac":
        issues.append(f"$.simulator: {oid!r} is simulated in Isaac Sim: the scene's simulator is 'mujoco+isaac'")
    isa = rec.get("isaac") or {}
    for k in ("cord_json", "report_json", "sim_png", "scene_usda"):
        q = _resolve(base, isa.get(k))
        if not q or not (Path(q).exists() or (k == "scene_usda" and Path(str(q) + ".gz").exists())):
            issues.append(f"{where}.isaac.{k}: missing or does not exist (build_isaac_scene writes it)")
    drift = isa.get("mean_drift_m")
    if drift is None:
        issues.append(f"{where}.isaac.mean_drift_m: report the settling drift from the Isaac report")
    elif not math.isfinite(float(drift)) or float(drift) > float(max_drift_m):
        issues.append(f"{where}.isaac.mean_drift_m: {drift} m is invalid or above {max_drift_m} m; the cord was floating or interpenetrating")
    spec = None
    cj = _resolve(base, isa.get("cord_json"))
    if cj and Path(cj).exists():
        try:
            spec = IC.load_spec(cj)
        except Exception as e:                               # noqa: BLE001
            issues.append(f"{where}.isaac.cord_json: not a cord specification ({e}); write it with build_isaac_scene")
    if spec is not None:
        lp = spec.get("link_poses") or {}
        C, Q = np.asarray(lp.get("centres") or [], float), np.asarray(lp.get("quats") or [], float)
        n = int(spec.get("n_links") or 0)
        info.update(cord=spec.get("name"), n_links=n, preset=spec.get("preset"))
        if C.shape != (n, 3) or Q.shape != (n, 4) or not (np.isfinite(C).all() and np.isfinite(Q).all()):
            issues.append(f"{where}.isaac.cord_json: needs the settled link_poses ({n} finite centres and quaternions); rebuild with build_isaac_scene")
        pat = re.compile(rf"^(?:{re.escape(str(spec.get('name')))}|{re.escape(str(oid))}).*B_(?:first|last|\d+)$")
        dup = sorted({b for b in body_names if pat.match(b)} | {b for b in [rec.get("root_body"), *(rec.get("bodies") or [])]
                                                                if isinstance(b, str) and b in body_names})
        if dup:
            issues.append(f"{where}: simulated in Isaac Sim, but scene.xml also has MuJoCo bodies for it ({dup[:3]}); a deformable lives in "
                          "Isaac Sim only: remove the MuJoCo cable (a cut cord is the exception: a MuJoCo cable without simulator 'isaac')")
        rj = _resolve(base, isa.get("report_json"))
        if rj and Path(rj).exists():
            try:
                rep = json.loads(Path(rj).read_text())
            except ValueError as e:
                rep = None
                issues.append(f"{where}.isaac.report_json: does not load ({e})")
            if rep is not None:
                meta = rep.get("_") or {}
                if rep.get("ok") is not True or not isinstance(rep.get(spec.get("name")), dict):
                    issues.append(f"{where}.isaac.report_json: not a passing settling report of cord {spec.get('name')!r} ({(rep.get('issues') or [])[:2]})")
                if int(meta.get("physx_invalid_transforms") or 0) or int(meta.get("physx_errors") or 0):
                    issues.append(f"{where}.isaac.report_json: PhysX logged errors while settling the cord")
    return issues, info


def _stage1_uses(sem):
    """{object id: [subgoal indices that name it]} and the same for subgoals whose motion is `cut`."""
    uses, cuts = {}, {}
    for sg in sem.get("subgoals", []) or []:
        if not isinstance(sg, dict):
            continue
        for ref in sg.get("object_refs") or []:
            oid = str(ref).split(":")[0]
            uses.setdefault(oid, []).append(sg.get("index"))
            if str(sg.get("motion", "")) == "cut":
                cuts.setdefault(oid, []).append(sg.get("index"))
    return uses, cuts


def _full_containment_trials(model, scene_xml, sem, man):
    """Independently test only full-containment relations required by Stage 1.

    Recorded verifier reports may propose placement parameters, never success,
    the scene, the body identities, or a looser containment tolerance.
    """
    import mujoco
    from tool_library.common import task_gates as G
    from tool_library.common.final_goal_language import compile_final
    from tool_library.verifiers.rest_inside_container_test import rest_inside_container_test
    d = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    gc = G.GoalContext(G.Episode(model, qpos=d.qpos.copy()), sem, man)
    predicates, _, _ = compile_final(gc, str(sem.get("task_success") or ""))
    pairs = list(dict.fromkeys((p["A"], p["B"]) for p in predicates if p["class"] == "final_full_containment"))
    issues, trials = [], []
    entries = {str(o.get("id")): o for o in man.get("objects", []) if isinstance(o, dict)}
    for subject, container in pairs:
        try:
            _, body_name, _ = G._resolve_set(gc, subject, ())
            _, container_name, _ = G._resolve_set(gc, container, ())
            proposals = []
            for entry in entries.values():
                for verifier, report in (entry.get("verifiers") or {}).items():
                    if verifier != "rest_inside_container_test" or not isinstance(report, dict):
                        continue
                    # Legacy reports under the subject may lack identities;
                    # they can only propose the default original orientation.
                    if report.get("body", body_name if entry.get("id") == str(subject).split(":")[0] else None) != body_name:
                        continue
                    if report.get("container", container_name) != container_name:
                        continue
                    declared = report.get("trial_args") or {}
                    proposals.append({k: declared[k] for k in ("floor_z", "drop_gap", "xy_offset", "placed_quat_wxyz") if k in declared})
            if not proposals:
                proposals = [{}]
            results = []
            for proposed in proposals[:8]:
                # Rest, contact, and orientation-change thresholds retain the
                # verifier defaults; only placement choices are proposals.
                result = rest_inside_container_test(str(scene_xml), body_name, container_name,
                                                     require_full_containment=True, **proposed)
                results.append(result)
                if result.get("ok") and result.get("fully_contained"):
                    break
            ok = any(r.get("ok") and r.get("fully_contained") for r in results)
            trials.append({"subject": subject, "container": container, "body": body_name, "container_body": container_name,
                           "ok": bool(ok), "independently_replayed": True, "reports": results})
            if not ok:
                detail = "; ".join(results[-1].get("issues", [])) if results else "no trial completed"
                issues.append(f"$.task_success: full containment of {subject} in {container} has no passing independent gravity trial: "
                              f"{detail}. A partial inside/rest report cannot prove this requirement; "
                              "rest_inside_container_test may propose placed_quat_wxyz for the intended placement.")
        except Exception as exc:
            trials.append({"subject": subject, "container": container, "ok": False, "error": str(exc)})
            issues.append(f"$.task_success: cannot verify required full containment of {subject} in {container}: {exc}")
    return {"ok": not issues, "issues": issues, "trials": trials}


def scene_recon_manifest_check(path, scene_semantic_json=None, library_index=None, min_iou=0.4, max_drift_m=0.02, allow_missing_static=True,
                               require_mujoco_deformables=True):
    p = Path(path)
    issues, per = [], {}
    try:
        man = json.loads(p.read_text())
    except Exception as e:
        return {"ok": False, "issues": [f"{p}: cannot read JSON: {e}"], "per_object": {}}
    base = p.parent
    sem_path = _resolve(base, scene_semantic_json or man.get("scene_semantic"))
    if not sem_path or not Path(sem_path).exists():
        return {"ok": False, "issues": ["$.scene_semantic: path to the Stage 1 scene_semantics.json is missing or does not exist"], "per_object": {}}
    sem = json.loads(Path(sem_path).read_text())
    idx = json.loads(Path(library_index or DEFAULT_INDEX).read_text())
    tools = {t["name"]: t for t in idx["tools"]}
    # top level
    scene_xml = _resolve(base, man.get("scene_xml"))
    model = None
    if not scene_xml or not Path(scene_xml).exists():
        issues.append("$.scene_xml: missing or does not exist")
    else:
        try:
            os.environ.setdefault("MUJOCO_GL", "egl")
            import mujoco
            model = mujoco.MjModel.from_xml_path(str(scene_xml))
        except Exception as e:
            issues.append(f"$.scene_xml: does not compile: {str(e)[:300]}")
    cam = _resolve(base, man.get("camera_json"))
    if not cam or not Path(cam).exists():
        issues.append("$.camera_json: missing or does not exist")
    else:
        try:
            c = json.loads(Path(cam).read_text())
            for k in ("fx", "fy", "cx", "cy", "pos", "quat_wxyz", "width", "height"):
                if k not in c:
                    issues.append(f"$.camera_json: {k} missing (write it with register_table_camera or read_egodex_episode)")
        except Exception as e:
            issues.append(f"$.camera_json: cannot read: {e}")
    cmp_png = _resolve(base, man.get("frame0_compare_png"))
    if not cmp_png or not Path(cmp_png).exists():
        issues.append("$.frame0_compare_png: missing or does not exist (render_frame_compare writes it)")
    objs = {o.get("id"): o for o in man.get("objects", [])}
    sem_objs = sem.get("objects", [])
    sim = man.get("simulator")
    if sim not in ("mujoco", "mujoco+isaac"):
        issues.append("$.simulator: must be 'mujoco' or 'mujoco+isaac'")
    body_names = {model.body(i).name for i in range(model.nbody)} if model is not None else set()
    # body -> the object it belongs to: every root body, then each group's declared pieces
    claimed = {o["root_body"]: k for k, o in objs.items() if isinstance(o, dict) and isinstance(o.get("root_body"), str)}
    eq_names = {model.equality(i).name for i in range(model.neq)} if model is not None else set()
    uses, cuts = _stage1_uses(sem)
    for so in sem_objs:
        oid = so.get("id"); rec = objs.get(oid)
        info = {"found": rec is not None}
        if rec is None:
            if allow_missing_static and so.get("role") == "static":
                info["skipped"] = "static"
            else:
                issues.append(f"$.objects: Stage 1 object {oid!r} ({so.get('type')}) has no entry")
            per[oid] = info; continue
        if rec.get("type") != so.get("type"):
            issues.append(f"$.objects[{oid}].type: {rec.get('type')!r} differs from Stage 1's {so.get('type')!r}")
        skill = rec.get("skill")
        if not skill:
            issues.append(f"$.objects[{oid}].skill: name the reconstruction skill used")
        elif skill not in tools:
            issues.append(f"$.objects[{oid}].skill: {skill!r} is not in the library index (a created skill must have passed library_create)")
        elif tools[skill]["category"] != "Object Reconstruction" and tools[skill]["type"] == "skill" and so.get("type") != "deformable":
            issues.append(f"$.objects[{oid}].skill: {skill!r} is a {tools[skill]['category']} skill, not an Object Reconstruction one")
        if rec.get("skill_origin") not in ("library", "created"):
            issues.append(f"$.objects[{oid}].skill_origin: 'library' or 'created'")
        if so.get("type") != "deformable" or sim == "mujoco":
            ax = _resolve(base, rec.get("asset_xml"))
            if not ax or not Path(ax).exists():
                issues.append(f"$.objects[{oid}].asset_xml: missing or does not exist")
            rb = rec.get("root_body")
            if not rb:
                issues.append(f"$.objects[{oid}].root_body: name the object's root body in the scene")
            elif body_names and rb not in body_names:
                issues.append(f"$.objects[{oid}].root_body: {rb!r} is not a body of scene_xml")
        if rec.get("parts") is not None or isinstance(so.get("pieces"), dict):
            gi, ginfo = _group_parts(oid, so, rec, model, claimed)
            issues.extend(gi)
            info.update(ginfo)
        pose = rec.get("pose") or {}
        if not all(k in pose for k in ("x", "y", "yaw_deg")):
            issues.append(f"$.objects[{oid}].pose: needs x, y, yaw_deg (rig frame)")
        mk = _resolve(base, rec.get("mask"))
        if not mk or not Path(mk).exists():
            issues.append(f"$.objects[{oid}].mask: missing or does not exist")
        iou = rec.get("iou")
        info["iou"] = iou
        if rec.get("hidden_at_frame0") is not None:
            # hidden at source frame 0 (tool_library.common.hidden_objects): its receipted frame-0 mask may be empty and
            # scene_qc tests that it renders invisible instead of an IoU; the declaration must be complete
            from tool_library.common.hidden_objects import declaration as hidden_declaration, declaration_issues
            info["hidden_at_frame0"] = True
            issues.extend(declaration_issues(rec))
            if so.get("role") == "static":
                issues.append(f"$.objects[{oid}].hidden_at_frame0: Stage 1 calls {oid!r} static; only an object the task uses may be "
                              "declared hidden at frame 0 (an untouched hidden object need not be reconstructed)")
            rm = hidden_declaration(rec)["reveal_mask"]
            if rm and not Path(_resolve(base, rm)).exists():
                issues.append(f"$.objects[{oid}].hidden_at_frame0.reveal_mask: {rm!r} does not exist")
        elif iou is None:
            issues.append(f"$.objects[{oid}].iou: report the silhouette IoU of the placed object")
        elif float(iou) < float(min_iou):
            issues.append(f"$.objects[{oid}].iou: {iou} is below {min_iou}; refit the placement, the size or the camera")
        if not rec.get("verification_plan"):
            issues.append(f"$.objects[{oid}].verification_plan: state which verifiers prove which properties (chosen before building)")
        if not rec.get("properties_covered"):
            issues.append(f"$.objects[{oid}].properties_covered: map every Stage 1 property to a parameter, a verifier, or a reason")
        else:
            missing = [pr for pr in (so.get("properties") or []) if pr not in rec["properties_covered"]]
            if missing:
                issues.append(f"$.objects[{oid}].properties_covered: {len(missing)} Stage 1 properties unmapped, e.g. {missing[0][:80]!r}")
        ver = rec.get("verifiers") or {}
        bad = [k for k, v in ver.items() if not (isinstance(v, dict) and v.get("ok") is True)]
        info["verifiers_ok"] = not bad and bool(ver)
        if not ver:
            issues.append(f"$.objects[{oid}].verifiers: no verifier reports; run the checks the properties call for and record their reports")
        if bad:
            issues.append(f"$.objects[{oid}].verifiers: {bad} did not pass")
        if so.get("type") == "articulated":
            mech = [k for k in ver if k in tools and tools[k]["category"] == MECH_CATEGORY]
            if not mech and "articulated_joint_travel_test" not in ver:
                issues.append(f"$.objects[{oid}].verifiers: an articulated object needs a Reconstruction Mechanism Test report (articulated_joint_travel_test or the mechanism's own test)")
            if "articulation_check" not in ver and not mech:
                issues.append(f"$.objects[{oid}].verifiers: add articulation_check (the object must have a usable joint in the scene)")
        # a box-like OBJECT (its id, name or type says so; a part merely named 'tray' on a solid drive does not) needs the enclosure check
        import re as _re
        head = " ".join([str(so.get("id", "")), str(so.get("name", "")), str(so.get("type", ""))]).lower()
        part_names = [str(x).split(":")[0].split(" ")[0].lower() for x in (so.get("parts") or []) if isinstance(x, str) and len(str(x).split()) <= 3]
        boxy = {"box", "tray", "drawer", "bin", "crate", "carton", "chest", "case", "container", "basket", "bowl_box"}
        is_box = any(_re.search(rf"\b{w}\b", head) for w in boxy) or any(pn in boxy for pn in part_names)
        if is_box and "bottle" not in head:
            if "box_enclosure_check" not in ver:
                issues.append(f"$.objects[{oid}].verifiers: a box-like object needs box_enclosure_check (a bottom, four walls and a hollow inside)")
        if so.get("type") == "deformable" and _isaac_only(rec):
            # simulated in Isaac Sim only: the cord specification Stage 3 and Stage 4 simulate, its settling report and
            # stage; no MuJoCo body (isaac_deformable_rest_check re-measures the physics from these files)
            ii, iinfo = _isaac_cord_entry(oid, rec, base, sim, body_names, max_drift_m)
            issues.extend(ii)
            info.update(iinfo)
            if require_mujoco_deformables and oid in cuts:
                issues.append(f"$.objects[{oid}].simulator: Stage 1 subgoals {cuts[oid]} cut this deformable, and Isaac Sim has no cutting "
                              f"yet: build it as a MuJoCo cable with construct_cuttable_rope at the video's cut location, stage it into "
                              f"scene.xml with its scene_fields.cut_joins, and drop simulator 'isaac' for it")
        elif so.get("type") == "deformable":
            isa = rec.get("isaac") or {}
            if sim == "mujoco":
                if (rec.get("simulation_model") or {}).get("kind") != "mujoco_cable":
                    issues.append(f"$.objects[{oid}].simulation_model: the MuJoCo-only deformable path requires kind=mujoco_cable, independently checked by deformable_rest_check and deformable_topology_check")
                info["simulation_scope"] = "mujoco_cable; Isaac rendering not verified"
            if sim == "mujoco+isaac" or isa:
                for k in ("scene_usda", "sim_png"):
                    q = _resolve(base, isa.get(k))
                    if not q or not Path(q).exists():
                        issues.append(f"$.objects[{oid}].isaac.{k}: missing or does not exist (build_isaac_scene writes it)")
                drift = isa.get("mean_drift_m")
                if drift is None:
                    issues.append(f"$.objects[{oid}].isaac.mean_drift_m: report the settling drift from the Isaac report")
                elif not math.isfinite(float(drift)) or float(drift) > float(max_drift_m):
                    issues.append(f"$.objects[{oid}].isaac.mean_drift_m: {drift} m is invalid or above {max_drift_m} m; the cord was floating or interpenetrating")
            # a deformable that is neither simulated in Isaac Sim (simulator 'isaac' with its cord) nor a MuJoCo cable of
            # scene.xml exists in no simulation: no grasp, lift or cut of it can be solved, tested or rolled out
            if require_mujoco_deformables and oid in uses and model is not None:
                names = [rec.get("root_body")] + list(rec.get("bodies") or [])
                info["in_scene_xml"] = any(n in body_names for n in names if isinstance(n, str))
                if not info["in_scene_xml"]:
                    issues.append(f"$.objects[{oid}].root_body: Stage 1 subgoals {uses[oid]} manipulate this deformable, but scene.xml "
                                  f"has no body for it ({rec.get('root_body')!r}) and its entry is not simulated in Isaac Sim. Build it in "
                                  f"Isaac Sim (build_isaac_scene; the entry gets simulator 'isaac' and isaac.cord_json), or, when a subgoal "
                                  f"cuts it, with construct_cuttable_rope staged into scene.xml (Isaac Sim has no cutting yet)")
            # a rope that a subgoal CUTS must carry its cut join(s) in the scene: the executor cuts by switching one off
            if require_mujoco_deformables and oid in cuts and model is not None and info.get("in_scene_xml", True):
                cj = [j for j in (rec.get("cut_joins") or []) if isinstance(j, dict)]
                if not cj:
                    issues.append(f"$.objects[{oid}].cut_joins: Stage 1 subgoals {cuts[oid]} cut this deformable, so it needs a "
                                  f"severable join in scene.xml: build it with construct_cuttable_rope at the video's cut location "
                                  f"and copy its scene_fields.cut_joins here")
                for j in cj:
                    eqn = j.get("eq")
                    if not eqn or eqn not in eq_names:
                        issues.append(f"$.objects[{oid}].cut_joins: equality {eqn!r} is not in scene.xml")
                    if not j.get("body") or j.get("body") not in body_names:
                        issues.append(f"$.objects[{oid}].cut_joins: join body {j.get('body')!r} is not in scene.xml")
        per[oid] = info
    extra = [k for k in objs if k not in {o.get("id") for o in sem_objs}]
    for k in extra:
        issues.append(f"$.objects[{k}]: not a Stage 1 object id")
    containment = _full_containment_trials(model, scene_xml, sem, man) if model is not None else None
    if containment is not None:
        issues.extend(containment["issues"])
    from tool_library.common.reconstruction_mechanism import independent_mechanism_trials
    mechanisms = independent_mechanism_trials(model, scene_xml, sem, man, base, tools,
                                             allow_missing_static=allow_missing_static) if model is not None else None
    if mechanisms is not None:
        issues.extend(mechanisms["issues"])
        for trial in mechanisms["trials"]:
            if trial["id"] in per:
                per[trial["id"]]["mechanism_evidence_ok"] = trial["ok"]
    return {"ok": not issues, "issues": issues, "per_object": per, "n_stage1_objects": len(sem_objs), "n_entries": len(objs),
            "full_containment_trials": containment, "independent_mechanism_trials": mechanisms}


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if not a:
        print(__doc__); return 2
    r = scene_recon_manifest_check(a[0], a[1] if len(a) > 1 else None)
    print(json.dumps(r, indent=1)); return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
