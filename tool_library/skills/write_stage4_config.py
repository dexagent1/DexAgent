"""Assemble the Stage 4 task folder from a Stage 3 episode."""
from __future__ import annotations

TOOL = {
    "name": "write_stage4_config",
    "title": "Write Stage 4 config",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Assemble the folder Stage 4 consumes: scene.xml with its assets, trajectory.npz (with the servo commands), the "
               "human and sim videos, the Stage 3 subgoal expectations, the executable Stage 3 plan (plan.json; Stage 4 does "
               "not re-execute it on the rigid runtime), and stage4.json with end effectors, per-phase anchors, the objects to "
               "randomize with their carriers, releases, resting relations and clearance-sized ranges, cameras, table, inpaint "
               "prompts and robot materials.",
    "description": (
        "Stage 4 re-anchors every hand frame to an object or the world and lets carried objects ride the hand; it "
        "needs to know which phase is anchored to what. This skill walks the Stage 3 phases per side in order and "
        "emits the segment list with the rule Stage 4 enforces (an anchor changes only through a blend over the phase "
        "in which it changes): reaching blends world -> object, grasp/lift/hold phases stay on the object, a carry "
        "blends from the carried object to its destination, an unanchored let-go or retreat stays with what the hand "
        "held, a return home blends back to world; phases are walked in time order and a label that recurs in "
        "several frame ranges gets a segment per range; a two-hand phase ('left: X | right: Y') gives each hand its "
        "own anchor. Per object, the "
        "carrier is the hand whose own lift/carry/set-down phase names that object and the release is that hand's "
        "first let-go phase naming it, never a later phase merely anchored to it (the hand stacking the next brick on "
        "it); an object that travels under 2 cm has no carrier. What a released object rests on or in (`rests_on` / "
        "`inside`, so a stacked goal follows the brick it is stacked on; `next_to` for a part set down at a world target within "
        "group_clearance_m of another randomized part, the part or its hand, so it keeps its place beside it) comes from the Stage 3 plan entry (`on`, "
        "`into`, the expectation's `on`), else the anchor of its release phase, else the contacts holding it up in the "
        "final state; the carrier's set-down phases that were anchored to the world are then anchored to that body. "
        "Objects whose geometry comes within group_clearance_m of each other while both still sit at their start (a "
        "pile of bricks, a brick lifted past its neighbour) move together (attached_to the group's first uncarried "
        "member). A 'deformable' that is a single free body (a plush toy) is randomized like a rigid one. Ranges default to half each object's start clearance to its nearest neighbour, 5 mm to 3 cm, with a "
        "yaw range scaled to match (3 to 15 degrees, and at most the shift range divided by how far the object reaches from its origin, so no point of a large object moves farther than the shift); `ranges` overrides per object. An object sharing its asset "
        "with another (a cap on its bottle) is attached_to it. For the physics replay of augmented episodes the "
        "folder gets trajectory_manifest.json (the manifest's executed subgoals with frames, plan and expect; rebuilt "
        "from the executor's outcomes.json when the manifest is not written yet), the servo commands as `reference` in "
        "trajectory.npz (from the same rigid runtime recording), the scene's screw-thread / cut / connector-latch rules, and "
        "checks.physics_replay. The executable plan goes to plan.json (stage4.json reexecute.plan): one entry per executed "
        "plan entry of the execution folder, in order, as it ran (its segment's keyframes.json, else the matching entry of "
        "`plan` or of the manifest), every hand pose inlined (the pose that ran first, then the other candidates the plan "
        "offered) with the object pose it was solved against (`anchor`), and its frame window in the trajectory; with "
        "the screw-thread / cut / latch rules. Without an execution folder that adds up to the trajectory no plan is "
        "written (a warning says why) and Stage 4 only warps. An object carried more than once (put down by one hand and "
        "picked up by the other, or handed over) gets every carrier and release in order (`carrier` and `release` as "
        "lists). A Stage 2 object whose root is fixed to the world (an articulated object's frame, a cabinet) can be "
        "randomized as a whole (fixed_bases=true); a cable or rope with a free root is randomized as one rigid piece "
        "(`deformable`: Stage 4 turns it about its middle, with a yaw range that moves its ends no farther than the shift "
        "range, and settles it before re-executing the plan on it). The video camera comes from Stage 2's camera.json with its full intrinsics (fx, fy, cx, cy at its image size), the wrist views from the scene's "
        "cameras under the hand bodies, the table from the scene's table geom, erase prompts from Stage 1's object "
        "names. Robot materials use the sharpa / yam presets. Copies scene.xml and the assets it references into the "
        "task folder with relative paths."
    ),
    "inputs": [
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 manifest (scene, camera, objects)."},
        {"name": "episode_dir", "type": "str", "description": "execute_subgoal_plan output folder (scene_physics.xml, complete rigid trajectory.npz, phases.json, outcomes.json)."},
        {"name": "out_dir", "type": "str", "description": "Task folder to write (stage4.json, scene.xml, assets/, trajectory.npz, trajectory_manifest.json, human_video.mp4, sim_video.mp4)."},
        {"name": "human_video", "type": "str", "description": "The human demonstration video."},
        {"name": "sim_video", "type": "str | None", "description": "Stage 3 simulation video (optional, for the comparison video)."},
        {"name": "scene", "type": "str | None", "description": "Not read: the scene is always the one the execution recorded (episode_dir/scene_physics.xml, which already holds place_arm_stations' stations); accepted for call compatibility."},
        {"name": "scene_semantic_json", "type": "str | None", "description": "Stage 1 manifest for the erase prompts (default: next to the Stage 2 manifest)."},
        {"name": "num_samples", "type": "int", "description": "Randomized episodes to keep. Default 3."},
        {"name": "ranges", "type": "dict | None", "description": "Per object {dx, dy, dyaw_deg} overrides (such an object is never grouped); objects set to null are not randomized."},
        {"name": "name", "type": "str | None", "description": "Task name (default the run name)."},
        {"name": "manifest", "type": "str | None", "description": "Stage 3 trajectory_manifest.json (default: next to episode_dir or in it; else rebuilt from outcomes.json)."},
        {"name": "physics_replay", "type": "bool", "description": "checks.physics_replay in stage4.json: replay every accepted layout in contact physics. Must stay true (the default): a Stage 4 handoff cannot disable it, and false raises ValueError."},
        {"name": "group_clearance_m", "type": "float", "description": "Objects closer than this while still at their start move together (default 0.01; 0 disables grouping)."},
        {"name": "plan", "type": "dict | str | None", "description": "The plan execute_subgoal_plan ran (dict or JSON path), when its segments' keyframes.json and the manifest's plan entries do not hold the hand poses (inline pose dicts)."},
        {"name": "deformables", "type": "bool", "description": "Randomize cables and ropes with a free root as one rigid piece (default true)."},
        {"name": "fixed_bases", "type": "bool", "description": "Also randomize Stage 2 objects whose root is fixed to the world, an articulated object's frame or a cabinet (default false: a base the arms work on near the edge of their reach leaves few layouts; turn it on when the base has room)."},
        {"name": "next_to", "type": "bool", "description": "A part set down within group_clearance_m of another randomized part (the part or its hand, while it sets it down and backs off) gets `next_to` that part (default true)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "config path, segments per side, the objects block, relations {object: {rests_on|inside: body, source}}, "
                                                         "groups, n_subgoals_with_expectations, physics_rules, plan_entries (0: no plan handed off), fixed_bases, "
                                                         "deformables, reference (bool), copied files, and warnings."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "stage4", "config", "handoff"],
    "used_in": ["Open the Bottle", "Giftbox Packing", "Place Toy in Drawer", "Battery Insertion", "Place Cup in Bowl", "Computer Installation"],
    "status": "tested",
    "entrypoint": "write_stage4_config",
}

import json
import hashlib
import shutil
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E
from tool_library.common import stage4 as S4
from tool_library.common.piece_groups import recon_parts


def _copy_scene(scene_xml, out_dir):
    """Flatten includes and bundle resolved assets; a missing dependency is an error."""
    src = Path(scene_xml).resolve()
    spec = mujoco.MjSpec.from_file(str(src))
    original = spec.compile()
    assets = Path(out_dir) / "assets"; assets.mkdir(parents=True, exist_ok=True)
    seen = {}
    for collection, directory in ((spec.meshes, spec.meshdir), (spec.textures, spec.texturedir), (spec.hfields, "")):
        for asset in collection:
            if not asset.file:
                continue
            p = Path(asset.file)
            if not p.is_absolute():
                p = (src.parent / (directory or "") / p).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"scene asset is missing: {p}")
            key = str(p)
            if key not in seen:
                suffix = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                dst = assets / f"{p.stem}_{suffix}{p.suffix}"
                if not dst.exists():
                    shutil.copy2(p, dst)
                seen[key] = f"assets/{dst.name}"
            # MjSpec recompiles when serializing; it still resolves files from
            # the source directory, so serialize with absolute bundled paths.
            asset.file = str((Path(out_dir) / seen[key]).resolve())
    spec.meshdir = ""
    spec.texturedir = ""
    output = Path(out_dir) / "scene.xml"
    import xml.etree.ElementTree as ET
    root = ET.fromstring(spec.to_xml())
    for element in root.findall("./asset/*"):
        if element.get("file"):
            element.set("file", "assets/" + Path(element.get("file")).name)
    output.write_text(ET.tostring(root, encoding="unicode"))
    bundled = mujoco.MjModel.from_xml_path(str(output))
    if (original.nq, original.nv, original.nu) != (bundled.nq, bundled.nv, bundled.nu):
        raise ValueError("bundling changed the runtime model dimensions")
    if [original.joint(j).name for j in range(original.njnt)] != [bundled.joint(j).name for j in range(bundled.njnt)]:
        raise ValueError("bundling changed runtime joint ordering")
    return len(seen)


def _sel(sel):
    """A fresh copy of a segment selector (the same phase item is walked once per side)."""
    return {k: list(v) for k, v in sel.items()}


LEAVE_WORDS = ("let go", "release", "relax", "retreat", "back off", "lift off", "withdraw")


def _segments(phases, sides, objects_free, root_of=None):
    """Per side: ordered segment list with anchors that change only through blends.

    The phases are walked in time order. A phase whose label recurs in several frame ranges (a retreat after a
    let-go and again after a press) can need a different anchor each time, so each of its ranges gets its own
    segment by frames; every other phase is named by its label."""
    segs = {s: [] for s in sides}
    warnings = []
    timed = all(isinstance(p.get("frames"), list) and p["frames"] and all(isinstance(r, (list, tuple)) and len(r) == 2 for r in p["frames"])
                for p in phases)
    items = []                                    # (start frame, list position, phase, selector)
    for k, p in enumerate(phases):
        if timed:
            multi = len(p["frames"]) > 1
            for a, b in p["frames"]:
                items.append((int(a), k, p, {"frames": [int(a), int(b)]} if multi else {"phases": [p["label"]]}))
        else:
            items.append((k, k, p, {"phases": [p["label"]]}))
    items.sort(key=lambda x: (x[0], x[1]))
    for s in sides:
        cur = "world"
        # a two-hand phase ('left: X | right: Y') is listed once per hand: this hand's own entry decides its anchor,
        # the other hand's entry for the same label must not override it (both hands may change anchor there)
        own = {p["label"] for p in phases if p.get("hand") == s}
        for _t, _k, p, sel in items:          # every phase, in order: an idle hand keeps its anchor through the other hand's phases
            lab = p["label"]
            if p.get("hand") != s:
                if lab not in own:
                    segs[s].append({**_sel(sel), "anchor": cur})
                continue
            anchor = p.get("anchor") or "world"
            low = S4.phase_verb(lab, s)
            if not p.get("anchor") and low.startswith(LEAVE_WORDS):
                # letting go of what the hand holds, and backing away from it, happen where that object is: a box
                # held still, a lid pressed shut, a part pressed home stay put, so the hand stays with them (a part
                # carried to a world goal reaches here with the hand already anchored to the world)
                anchor = cur
            if anchor not in objects_free and anchor != "world":
                # a part of an articulated object (a latch on a lid): anchor to the randomized object it hangs from in
                # the scene, else one whose name it extends, else the world
                root = (root_of or {}).get(anchor)
                if root not in objects_free:
                    root = next((o for o in objects_free if anchor.startswith(o)), None)
                anchor = root or "world"
            if "return home" in low:
                anchor = "world"
            if anchor != cur:
                segs[s].append({**_sel(sel), "blend": [cur, anchor]})
                cur = anchor
            else:
                segs[s].append({**_sel(sel), "anchor": anchor})
    # merge consecutive same-anchor entries named by label
    out = {}
    for s, lst in segs.items():
        merged = []
        for e in lst:
            if (merged and "anchor" in e and "anchor" in merged[-1] and merged[-1]["anchor"] == e["anchor"]
                    and "phases" in e and "phases" in merged[-1]):
                merged[-1]["phases"] = merged[-1]["phases"] + e["phases"]
            else:
                merged.append(e)
        out[s] = merged
    return out, warnings


def _outcome_spans(ep_dir):
    """Use the executor's joined windows, which omit repeated boundary states."""
    path = Path(ep_dir) / "outcomes.json"
    if not path.exists():
        return []
    out, next_frame = [], 0
    for row in json.loads(path.read_text()):
        n = int(row.get("n_frames") or 0)
        if not n:
            continue
        frames = row.get("frames", [next_frame, next_frame + n - 1])
        if (not isinstance(frames, list) or len(frames) != 2
                or any(not isinstance(v, int) or isinstance(v, bool) for v in frames)
                or frames[0] != next_frame or frames[1] < frames[0]):
            raise ValueError("outcome frames must be contiguous executor windows")
        omitted = n - (frames[1] - frames[0] + 1)
        if omitted not in (0, 1) or (next_frame == 0 and omitted):
            raise ValueError("outcome window does not match its recorded segment length")
        out.append((row, frames, omitted))
        next_frame = frames[1] + 1
    return out


def _subgoals(manifest, ep_dir):
    """[{index, step, motion, side, object, frames, expect, plan}] from a trajectory manifest, or rebuilt from the
    executor's outcomes.json, preserving its actual joined frame windows."""
    if manifest is not None:
        man = json.loads(Path(manifest).read_text())
        out = []
        for sg in man.get("subgoals", []):
            if sg.get("skipped"):
                continue
            out.append({k: sg[k] for k in ("index", "stage1_indices", "frames", "motion", "side", "object", "plan", "expect",
                                            "step", "hands", "sync", "goal_predicates", "deviation", "skip_kind") if k in sg})
        return out
    out = []
    for r, frames, _ in _outcome_spans(ep_dir):
        out.append({"index": r.get("index"), "stage1_indices": r.get("stage1_indices", [r.get("index")]),
                    "step": r.get("step"), "motion": r.get("motion"), "side": r.get("side"),
                    "object": r.get("object"), "frames": frames, "expect": r.get("expect"),
                    "plan": {"motion": r.get("motion"), "object": r.get("object"), "step": r.get("step")}})
    return out


def _single_rigid_body(model, body):
    """A body on a free joint with no other joint below it: it moves rigidly with its root."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body))
    if b < 0:
        return False
    sub = {b} | {c for c in range(model.nbody) if c != b and _ancestor(model, c, b)}
    joints = [j for j in range(model.njnt) if int(model.jnt_bodyid[j]) in sub]
    return len(joints) == 1 and int(model.jnt_type[joints[0]]) == int(mujoco.mjtJoint.mjJNT_FREE)


def _ancestor(model, c, b):
    while c > 0:
        c = int(model.body_parentid[c])
        if c == b:
            return True
    return False


def _recorded_releases(ep_dir):
    """[{equality, frame}] the executor switched off during the episode (outcomes.json `released`, frames made
    episode-global using the same boundary deduplication as the executor)."""
    out = []
    for r, frames, omitted in _outcome_spans(ep_dir):
        for e in r.get("released") or []:
            if isinstance(e, dict) and e.get("equality") is not None and e.get("frame") is not None:
                out.append({"equality": str(e["equality"]), "frame": frames[0] + int(e["frame"]) - omitted})
    return out


def _anchor_to_support(phases, objs):
    """A carried object that rests on another randomized body must be put down relative to that body. Walking back
    from its release over the carrier's phases that act on it (the let-go, the set-down, the carries, across
    subgoals) and forward over its retreat, every phase the planner anchored to the world (a world target_pos that
    happened to lie on the support) is anchored to the support instead, so the hand follows the support when Stage 4
    moves it. The walk stops at a phase anchored to anything else (the grasp and the lift stay on the object)."""
    out = [dict(p) for p in phases]
    move = ("carry", "set down", "lower", "insert", "let go", "release", "relax")
    for nm, o in objs.items():
        body, side, release = o.get("rests_on") or o.get("inside") or o.get("next_to"), o.get("carrier"), o.get("release")
        if isinstance(side, list):                    # carried more than once: the final carry puts it there
            side, release = side[-1], (release[-1] if isinstance(release, list) else release)
        if not body or not release or not side:
            continue
        r = next((i for i, p in enumerate(out) if p["label"] == release and p.get("hand") == side), None)
        if r is None:
            continue

        def walk(rng, ok_verb):
            for i in rng:
                p = out[i]
                if p.get("hand") != side:
                    continue                          # the other hand's phases interleave
                if not S4.names_object(p, nm) or (p.get("anchor") or "world") != "world" or not ok_verb(S4.phase_verb(p["label"], side)):
                    return
                p["anchor"] = body

        walk(range(r, -1, -1), lambda v: v.startswith(move))
        walk(range(r + 1, len(out)), lambda v: not v.startswith("return home"))
    return out


def write_stage4_config(scene_recon_json, episode_dir, out_dir, human_video, sim_video=None, scene_semantic_json=None,
                        num_samples=3, ranges=None, name=None, scene=None, manifest=None, physics_replay=True,
                        group_clearance_m=0.01, plan=None, deformables=True, fixed_bases=False, next_to=True):
    man_p = Path(scene_recon_json).resolve(); ws = man_p.parent; man = json.loads(man_p.read_text())
    ep_dir = Path(episode_dir).resolve(); out = Path(out_dir).resolve()
    if not physics_replay:
        raise ValueError("a Stage 4 handoff cannot disable physics replay")
    scene_xml, traj, model, runtime_arrays = S4.load_runtime_episode(ep_dir, scene_recon_json=str(man_p))
    phases = json.loads((ep_dir / "phases.json").read_text())
    ep = E.load_episode(traj)
    labels = ep["labels"]
    out.mkdir(parents=True, exist_ok=True)
    n_assets = _copy_scene(scene_xml, out)
    runtime_arrays["scene_sha256"] = np.asarray(hashlib.sha256((out / "scene.xml").read_bytes()).hexdigest())
    np.savez_compressed(out / "trajectory.npz", **runtime_arrays)
    # Full unmodified Stage 2 evidence, so the finalizer can compare these exact
    # bytes with its frozen input rather than trusting editable derived rules.
    mechanism_recon = out / "mechanism_scene_recon.json"
    mechanism_recon.write_bytes(man_p.read_bytes())
    S4.load_runtime_pair(out / "scene.xml", out / "trajectory.npz", str(mechanism_recon))
    shutil.copy2(human_video, out / "human_video.mp4")
    if sim_video and Path(sim_video).exists():
        shutil.copy2(sim_video, out / "sim_video.mp4")
    sides = [s for s in ("left", "right") if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{s}_{s}_base_yaw") >= 0]
    end_effectors = {s: f"robot_{s}_{s}_base_yaw" for s in sides}
    # objects: free roots from the manifest
    free = E.object_free_joints(model)
    objs = {}
    by_asset = {}
    cables = set()
    for o in man.get("objects", []):
        rb = o.get("root_body")
        if not rb:
            continue
        if o.get("type") == "deformable" and not _single_rigid_body(model, rb):
            # a rope or a cable (a free root with a chain of joints below it) is moved as one rigid piece and settled
            # in the cable twin when Stage 4 re-executes the plan on it; a cloth or anything without a free root is
            # left where it is. A "deformable" that is one free body (a plush toy) is moved like a rigid one
            if rb not in free or not deformables:
                continue
            cables.add(rb)
        by_asset.setdefault(str((ws / o["asset_xml"]).resolve()) if o.get("asset_xml") else rb, []).append(o)
    # bases fixed to the world (a cabinet, a holder bolted down, an articulated object's frame): Stage 4 moves the
    # base as a whole, its parts keep their joint values
    fixed = set(S4.fixed_bases(model, [o["root_body"] for g in by_asset.values() for o in g])) if fixed_bases else set()
    for ax, group in by_asset.items():
        main = group[0]
        for o in group:
            rb = o["root_body"]
            if rb in free or rb in fixed:
                spec = {"dx": [-0.03, 0.03], "dy": [-0.03, 0.03], "dyaw_deg": [-15.0, 15.0]}
                if o is not main and main["root_body"] in set(free) | fixed:
                    spec = {"attached_to": main["root_body"]}
                if rb in cables:
                    spec["deformable"] = True
                objs[rb] = spec
                # a group object (separate pieces, scene_recon `parts`): its layout moves as one, every other free
                # piece attached to its root, and each piece keeps its own carrier and release below
                for pn in recon_parts(o):
                    if pn != rb and pn in free and pn not in objs:
                        objs[pn] = {"attached_to": rb}
    if ranges:
        for k, v in ranges.items():
            if v is None:
                objs.pop(k, None)
            elif k in objs:
                objs[k].update(v)
            else:
                objs[k] = v
    # carriers and releases from the phases that act on THAT object (its own lift/carry and its own let-go), never
    # from a later phase that is merely anchored to it (the hand stacking the next brick on it); an object that does
    # not travel (a destination, a support) has no carrier
    moved = {}
    Qt = ep["qpos"]
    for nm, adr in free.items():
        if nm in objs:
            P = Qt[:, adr:adr + 3]
            moved[nm] = float(np.linalg.norm(P - P[0], axis=1).max())
    roles, role_warn = S4.object_roles(phases, list(objs), moved)
    rides = S4.object_rides(phases, [k for k in objs if k not in fixed], moved)
    for nm, r in roles.items():
        if nm in fixed:
            continue
        objs[nm]["carrier"] = r["carrier"]
        if r["release"]:
            objs[nm]["release"] = r["release"]
        rd = rides.get(nm) or []
        if len(rd) > 1:
            # carried more than once (handed over, or put down and picked up again by the other hand): every carrier
            # in order, with its own release; the resting relation applies after the last one
            objs[nm]["carrier"] = [x["side"] for x in rd]
            objs[nm]["release"] = [x["release"] for x in rd]
    role_warn = [w for w in role_warn if not any(w.startswith(f"{nm}: carried by") and len(rides.get(nm) or []) > 1 for nm in objs)]
    # the Stage 3 subgoals with their expectations: the manifest when it exists, else the executor's outcomes
    man_src = Path(manifest).resolve() if manifest else next(
        (q for q in (ep_dir / "trajectory_manifest.json", ep_dir.parent / "trajectory_manifest.json") if q.exists()), None)
    subgoals = _subgoals(man_src, ep_dir)
    T_ep = len(labels)
    if man_src is not None and not manifest and any(g.get("frames") and int(g["frames"][-1]) >= T_ep for g in subgoals):
        # a manifest left from an earlier execution of this workspace: its windows do not fit this episode
        role_warn_stale = f"{man_src} does not match this episode ({T_ep} frames); rebuilt the subgoals from outcomes.json"
        man_src, subgoals = None, _subgoals(None, ep_dir)
    else:
        role_warn_stale = None
    # companions a plan entry carries along with its object (`with`: a tip mounted on the pipette) ride with that object:
    # the same carrier and release, and never its support
    companions = {}
    for sg in subgoals:
        pe = S4.plan_entry(sg.get("plan"))
        main = pe.get("object") or sg.get("object")
        withs = pe.get("with") or []
        for w in ([withs] if isinstance(withs, str) else list(withs)):
            cr, mr = S4.root_body(model, w), S4.root_body(model, main) if main else None
            if cr and mr and cr != mr:
                companions[cr] = mr
    for comp, main in companions.items():
        if comp in objs and main in roles and not (roles.get(comp) or {}).get("carrier"):
            roles[comp] = dict(roles[main])
            objs[comp]["carrier"] = objs[main].get("carrier", roles[main]["carrier"])
            if objs[main].get("release"):
                objs[comp]["release"] = objs[main]["release"]
    # what each released object rests on or in: the plan's on / into, its release anchor, the final contacts
    # the resting relation is the final placement's: for an object carried more than once, its last carrier's release
    roles_final = {nm: (dict(r, carrier=rides[nm][-1]["side"], release=rides[nm][-1]["release"]) if len(rides.get(nm) or []) > 1 else r)
                   for nm, r in roles.items()}
    relations = S4.infer_relations(model, subgoals, phases, roles_final, list(objs), q_end=Qt[-1])
    relations = {nm: r for nm, r in relations.items()
                 if companions.get(r.get("rests_on") or r.get("inside")) != nm}      # a part riding the object is not its support
    for nm, r in relations.items():
        kind = "inside" if "inside" in r else "rests_on"
        objs[nm][kind] = r[kind]
    # a part set down right beside another randomized part (a brick slid next to a brick) keeps its place next to it
    beside = S4.set_down_neighbours(model, str(scene_xml), Qt, phases, labels, objs, list(objs),
                                    clearance_m=float(group_clearance_m or 0.01)) if next_to else {}
    for nm, r in beside.items():
        if companions.get(r["next_to"]) == nm or S4.root_body(model, r["next_to"]) == nm:
            continue
        objs[nm]["next_to"] = r["next_to"]
        relations[nm] = r
    # objects that come within group_clearance_m of each other while both are still at their start (bricks touching
    # in the pile, a brick lifted past its neighbour) move together: independently they collide in most layouts
    groups = []
    if group_clearance_m:
        leave = S4.leave_start_frames(phases, labels, {k: (o["carrier"][0] if isinstance(o.get("carrier"), list) else o.get("carrier"))
                                                       for k, o in objs.items()})
        free_choice = [k for k in objs if "attached_to" not in objs[k] and k not in (ranges or {}) and k not in cables]
        groups, _close = S4.interaction_groups(model, Qt, free_choice, release_frames=leave, clearance_m=float(group_clearance_m))
        for g in groups:
            lead = next((k for k in g if not objs[k].get("carrier")), g[0])
            for k in g:
                if k != lead:
                    for key in ("dx", "dy", "dyaw_deg"):
                        objs[k].pop(key, None)
                    objs[k]["attached_to"] = lead
    # default ranges from the room each object (or group) has at the start: half the distance to its nearest
    # neighbour, 5 mm to 3 cm (a brick in a tight pile moves a few millimetres, a lone cup 3 cm)
    clear = S4.start_clearances(model, Qt[0], list(objs), groups)
    for k, o in objs.items():
        if "attached_to" in o or k in (ranges or {}):
            continue
        members = [k] + [j for j, oj in objs.items() if oj.get("attached_to") == k]
        o.update(S4.default_ranges(min(clear.get(j, 0.06) for j in members)))
        # no point of a large object (a cabinet, a long cable) moves farther under its yaw than its shift range allows:
        # the yaw is capped at shift / radius (a cable turns about its middle, anything else about its origin)
        rad = 0.5 * S4.body_span(model, Qt[0], k) if k in cables else max(
            S4.plan_radius(model, Qt[0], j) + S4.origin_distance(model, Qt[0], k, j) for j in members)
        if rad > 0:
            yaw = min(float(o["dyaw_deg"][1]), float(np.degrees(o["dx"][1] / rad)))
            o["dyaw_deg"] = [-round(yaw, 1), round(yaw, 1)]
    phases = _anchor_to_support(phases, objs)
    root_of = {p["anchor"]: S4.root_body(model, p["anchor"]) for p in phases if p.get("anchor")}
    segments, warnings = _segments(phases, sides, list(objs), root_of)
    warnings += role_warn + ([role_warn_stale] if role_warn_stale else [])
    # cameras: the video camera with its full intrinsics (fx != fy and an off-centre principal point are kept, for
    # the image size they were measured at), the output at that size
    cam = json.loads((ws / man.get("camera_json", "camera.json")).read_text())
    cam_w, cam_h = int(cam.get("width", 640)), int(cam.get("height", 480))
    if cam.get("fovy_deg") is not None:
        fovy = float(cam["fovy_deg"])
    elif cam.get("fy") is not None:
        fovy = float(np.degrees(2 * np.arctan(0.5 * cam_h / float(cam["fy"]))))
    else:
        raise ValueError("camera.json needs fovy_deg or fy (with height)")
    ego_cam = {"pos": cam["pos"], "quat_wxyz": cam["quat_wxyz"], "fovy": fovy}
    if cam.get("fx") is not None and cam.get("fy") is not None:
        ego_cam.update(fx=float(cam["fx"]), fy=float(cam["fy"]), cx=float(cam.get("cx", cam_w / 2)),
                       cy=float(cam.get("cy", cam_h / 2)), image_size=[cam_w, cam_h])
    views = {}
    for c in range(model.ncam):
        nm = model.camera(c).name
        if nm.startswith("wrist_"):
            views[nm] = {"parent_body": model.body(int(model.cam_bodyid[c])).name, "pos": model.cam_pos[c].tolist(),
                         "quat_wxyz": model.cam_quat[c].tolist(), "fovy": float(model.cam_fovy[c])}
    # table
    gt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_table")
    d = mujoco.MjData(model); mujoco.mj_forward(model, d)
    if gt >= 0:
        top_z = float(d.geom_xpos[gt][2] + model.geom_size[gt][2]); center = d.geom_xpos[gt][:2].tolist(); half = model.geom_size[gt][:2].tolist()
    else:
        top_z, center, half = 0.0, [0.0, 0.5], [1.0, 1.0]
    # prompts
    sem_p = Path(scene_semantic_json) if scene_semantic_json else (ws / man.get("scene_semantic", "../stage1/scene_semantics.json"))
    if not scene_semantic_json and not sem_p.exists() and (sem_p.parent / "scene_semantic.json").exists():
        sem_p = sem_p.parent / "scene_semantic.json"           # a run made before the release
    prompts = ["hand", "arm"]
    if sem_p.exists():
        sem = json.loads(sem_p.read_text())
        prompts += [o.get("name") for o in sem.get("objects", []) if o.get("name")]
        bg = sem.get("environment", {}).get("support_surface")
        background = f"An empty {bg}, nothing on it, static background." if bg else None
    else:
        background = None
    cfg = {"name": name or ep_dir.parent.name, "scene": "scene.xml", "trajectory": "trajectory.npz", "human_video": "human_video.mp4",
           "sim_video": "sim_video.mp4" if (out / "sim_video.mp4").exists() else None, "output_dir": "outputs", "num_samples": int(num_samples), "seed": 0,
           "robot": {"body_prefix": "robot_", "end_effectors": end_effectors, "init_jitter_rad": 0.05},
           "randomize": {"objects": objs}, "segments": segments,
           "cameras": {"width": cam_w, "height": cam_h, "ego": ego_cam, "views": views},
           "table": {"top_z": round(top_z, 4), "center": [round(x, 4) for x in center], "half_size": [round(x, 3) for x in half]},
           "inpaint": {"erase_prompts": prompts, **({"background_prompt": background} if background else {})},
           "render": {"stride": 2, "samples": 32, "chunks": 4, "exclude_geoms": ["^table", "groundplane", "floor", "^g_table"],
                      "materials": [{"match": "^robot_\\w+_arm_", "preset": "yam_black"}, {"match": "^robot_", "preset": "sharpa"}]}}
    # what the physics replay of every augmented episode re-checks: the subgoal expectations, the servo commands
    # (the executed fingers stop on the object; their commands press), the run-time rules of the scene
    (out / "trajectory_manifest.json").write_text(json.dumps({"source": str(man_src) if man_src else "episode outcomes",
                                                             "subgoals": subgoals}, indent=1, default=float))
    cfg["manifest"] = "trajectory_manifest.json"
    # Stage 1's goals and the object -> body map: Stage 4's physics replay judges every augmented episode against the
    # same goal sentences Stage 3's finish did (stage1_goal_gate), not only the per-entry expectations
    if sem_p.exists():
        (out / "scene_semantics.json").write_text(sem_p.read_text())
        keep = ("id", "type", "root_body", "bodies", "parts", "part_bodies", "joints", "equalities")
        (out / "goal_objects.json").write_text(json.dumps({"objects": [{k: o[k] for k in keep if k in o} for o in man.get("objects", [])]}, indent=1))
        cfg["goal"] = {"semantics": "scene_semantics.json", "objects": "goal_objects.json"}
    # Runtime commands and states were copied together above. Never overwrite them with proposal
    # trajectory_hands.npz, even if two unrelated schemas happen to have the same column count.
    rules = S4.physics_rules(man_p)
    cfg["checks"] = {"physics_replay": bool(physics_replay)}
    # the replay must run the MuJoCo this episode was executed with (this process's, the Stage 3 environment's)
    cfg["physics"] = {"rules": rules, "mujoco_version": mujoco.__version__,
                      "scene_recon_json": "mechanism_scene_recon.json",
                      "scene_recon_sha256": hashlib.sha256(mechanism_recon.read_bytes()).hexdigest()}
    # constraints the execution switched off (a screw thread that let its cap go), at their frame in the episode: the
    # replay starts every later segment with them off, whether or not the scene manifest declares the rule
    releases = _recorded_releases(ep_dir)
    if releases:
        cfg["physics"]["releases"] = releases
    # the executable plan (its poses follow their objects); the rigid runtime does not re-execute it in this release
    entries, plan_notes = S4.executable_plan(ep_dir, manifest=man_src, plan=plan, n_frames=T_ep)
    if entries:
        (out / "plan.json").write_text(json.dumps({"source": str(ep_dir), "fps": float(ep.get("fps", 30.0)), "rules": rules,
                                                  "subgoals": entries, "notes": plan_notes}, indent=1, default=float))
        cfg["reexecute"] = {"plan": "plan.json"}
    warnings += plan_notes
    (out / "stage4.json").write_text(json.dumps(cfg, indent=1, default=float))
    return {"config": str(out / "stage4.json"), "segments": segments, "objects": objs, "relations": relations, "groups": groups,
            "n_subgoals_with_expectations": sum(1 for g in subgoals if g.get("expect")), "physics_rules": len(rules),
            "plan_entries": len(entries or []), "fixed_bases": sorted(fixed & set(objs)), "deformables": sorted(cables & set(objs)),
            "reference": "reference" in np.load(out / "trajectory.npz").files,
            "n_assets_copied": n_assets, "n_frames": int(len(labels)),
            "warnings": warnings, "files": sorted(p.name for p in out.iterdir())}
