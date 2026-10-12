"""Does the reconstructed object sit where the demonstrator's hand actually touched it?

    from tool_library.verifiers.hand_object_contact_consistency_check import hand_object_contact_consistency_check
    r = hand_object_contact_consistency_check("stage2/scene.xml", "stage2/hands.npz",
                                              "stage1/scene_semantics.json", "stage2/scene_recon.json")
    r["ok"], r["objects"]

A silhouette fit through one camera pins an object's image position but not its depth: a brick placed 8 cm too far
along the camera ray still overlaps its mask. When the demonstration carries metric hand joints (an EgoDex episode's
hands.npz from read_egodex_episode, in the rig frame), the fingertips at the first moment the hand grasps or pushes an
object must reach that object's surface in the reconstructed scene; if they stop centimetres short or pass through,
the initial-state comparison is inconsistent and its cause needs investigation.
"""
from __future__ import annotations

TOOL = {
    "name": "hand_object_contact_consistency_check",
    "title": "Hand-object contact consistency",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "With metric hand joints from the demonstration (EgoDex hands.npz), the fingertips must reach each object's "
               "reconstructed surface at first contact, including a physically replayed previous support motion when "
               "measured tracking supports one. Unexplained gaps remain failures.",
    "description": (
        "For every Stage 1 object, the first subgoal whose acting hand grasps, holds, pushes, pulls, presses or opens it "
        "(motion in a fixed set) gives a time window (time_window_s, converted to frames with the episode fps). The "
        "object's collision and visible material is measured per geom: analytic primitives, convex collision mesh "
        "hulls, and actual visual mesh triangles. Empty spaces between walls are not filled. Over the window the "
        "acting hand's fingertips (all five for the gap, thumb, index "
        "and middle for the depth inside; "
        "both hands for 'both') come closest to that surface at some frame; the smallest fingertip-to-surface distance "
        "must be at most `tol_m`. A failed initial-surface comparison can also be checked against an internally "
        "replayed previous articulated support motion, anchored to measured fingertip tracking and initial physical "
        "contact. Only bounded force on that support joint is applied; carried objects move through physics. "
        "Replay states are aligned to the source time window; no supplied object poses or arbitrary joint sweeps count. "
        "This narrow consistency witness does not prove robot execution feasibility. An object Stage 1 says a hand already holds at the first frame (initial_state) whose first contact starts "
        "there is skipped with that reason: the scene must start it at rest. The report gives, per object, the window, the closest frame and fingertip, the gap and the offset from "
        "the surface to the fingertip (which direction the object would have to move). Objects the hands never act on, "
        "or whose first window falls outside the joints' frames, are listed as skipped. Without hands.npz (a video "
        "without metric hand joints) the check does not apply and says so. A group object (its scene_recon.json entry "
        "lists its separate pieces under `parts`) is measured against the surface of all its pieces, so the gap is to "
        "the nearest piece, which the report names (`part`)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Stage 2 scene.xml (objects at their fitted frame-0 poses; keyframe 0 used when present)."},
        {"name": "hands_npz", "type": "str | None", "description": "hands.npz from read_egodex_episode (names, rig_xyz [T, J, 3]) in the scene's rig frame. None: not applicable."},
        {"name": "scene_semantic_json", "type": "str", "description": "Stage 1 scene_semantic(s).json (subgoals with acting_hand, motion, object_refs, time_window_s)."},
        {"name": "scene_recon_json", "type": "str", "description": "Stage 2 scene_recon.json (object id -> root_body)."},
        {"name": "fps", "type": "float | None", "description": "Frames per second of the joints; default: episode.json next to hands.npz, else 30."},
        {"name": "tol_m", "type": "float", "description": "Largest allowed fingertip-to-surface gap at the closest frame. Default 0.02."},
        {"name": "max_penetration_m", "type": "float", "description": "Largest allowed depth of a fingertip inside any material component over the contact window. Default 0.02."},
        {"name": "pad_s", "type": "float", "description": "Seconds added on both sides of the window. Default 0.3."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok, issues, applicable, objects {id: {subgoal, frames, closest_frame, fingertip, gap_m, offset_m, ok, part (group objects: the nearest piece)}}, skipped {id: reason}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "trimesh"],
    "simulators": ["MuJoCo"],
    "tags": ["reconstruction", "depth", "hand", "egodex", "placement", "scene"],
    "status": "tested",
    "entrypoint": "hand_object_contact_consistency_check",
}

import json
import re
from pathlib import Path

import numpy as np

CONTACT_MOTIONS = {"grasp_lift", "hold", "push", "pull", "press", "open_joint", "close_joint", "unscrew", "screw", "regrasp", "insert", "tilt", "pour"}
TIPS = ("ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip")   # any of them may touch first
# an object a hand already holds in the first frame (Stage 1's initial_state says so and its first contact starts there)
HELD_START_S = 0.25
HELD_WORDS = re.compile(r"\b(in|by|between) (the )?(left |right |both )?(hand|hands|fingers|thumb)\b|\bheld\b|\bholding\b|\bpinched\b|\bgrasped\b|\bin hand\b")
DEPTH_TIPS = ("ThumbTip", "IndexFingerTip", "MiddleFingerTip")    # the tracked tips precise enough to judge depth by


def _surface_points(model, data, root_body, n_prim=400):
    """World points on the collision and visual geoms of a body subtree."""
    import mujoco
    rid = model.body(root_body).id
    sub = [b for b in range(model.nbody) if _is_under(model, b, rid)]
    rng = np.random.default_rng(0)
    pts = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in sub:
            continue
        R = data.geom_xmat[g].reshape(3, 3); p = data.geom_xpos[g]; t = int(model.geom_type[g]); s = model.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g]); a, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            v = model.mesh_vert[a:a + n]
            pts.append(v @ R.T + p)
            continue
        u = rng.normal(size=(n_prim, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            loc = u * s[0]
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            loc = u * s[:3]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            loc = u / np.abs(u).max(axis=1, keepdims=True) * s[:3]
        elif t in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            ang = rng.uniform(0, 2 * np.pi, n_prim); z = rng.uniform(-s[1], s[1], n_prim)
            side = np.stack([s[0] * np.cos(ang), s[0] * np.sin(ang), z], 1)
            r = np.sqrt(rng.uniform(0, 1, n_prim // 2)) * s[0]; a2 = rng.uniform(0, 2 * np.pi, n_prim // 2)
            caps = np.stack([r * np.cos(a2), r * np.sin(a2), np.where(rng.uniform(size=n_prim // 2) < 0.5, -s[1], s[1])], 1)
            loc = np.vstack([side, caps])
        else:
            continue
        pts.append(loc @ R.T + p)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _is_under(model, b, root):
    while b > 0:
        if b == root:
            return True
        b = int(model.body_parentid[b])
    return b == root


def _inside(points, q):
    """Whether q lies inside the convex hull of the object's surface points (False when no hull can be built)."""
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(points)
    except Exception:
        return False
    return bool((h.equations[:, :3] @ q + h.equations[:, 3] <= 0).all())


def hand_object_contact_consistency_check(scene, hands_npz, scene_semantic_json, scene_recon_json, fps=None, tol_m=0.02,
                                          max_penetration_m=0.02, pad_s=0.3):
    import mujoco
    if hands_npz is None or not Path(hands_npz).exists():
        return {"ok": True, "applicable": False, "issues": [], "objects": {}, "skipped": {},
                "note": "no metric hand joints (hands.npz): not applicable"}
    if not Path(scene).exists():
        raise ValueError(f"scene does not exist: {scene}")
    sem = json.loads(Path(scene_semantic_json).read_text())
    recon = json.loads(Path(scene_recon_json).read_text())
    root_of = {o["id"]: o.get("root_body") for o in recon.get("objects", [])}
    from tool_library.common import piece_groups as PG
    pieces_of = {o["id"]: PG.group_bodies(o) for o in recon.get("objects", []) if PG.is_group(o)}
    with np.load(str(hands_npz), allow_pickle=False) as z:
        names = [str(n) for n in z["names"]]; X = np.asarray(z["rig_xyz"], float)
    col = {n: i for i, n in enumerate(names)}
    if fps is None:
        ep = Path(hands_npz).with_name("episode.json")
        fps = float(json.loads(ep.read_text()).get("fps", 30.0)) if ep.exists() else 30.0
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    first = {}
    for sg in sem.get("subgoals", []):
        if sg.get("motion") not in CONTACT_MOTIONS or not sg.get("time_window_s"):
            continue
        for ref in sg.get("object_refs", []) or []:
            oid = str(ref).split(":")[0]
            first.setdefault(oid, sg)
    objects, skipped, issues = {}, {}, []
    for o in sem.get("objects", []):
        oid = o["id"]
        sg = first.get(oid)
        if sg is None:
            skipped[oid] = "no subgoal grasps or pushes it"; continue
        root = root_of.get(oid)
        if not root or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root) < 0:
            skipped[oid] = f"no root body for {oid!r} in the scene"; continue
        t0, t1 = sg["time_window_s"]
        if float(t0) <= HELD_START_S and HELD_WORDS.search(str(o.get("initial_state") or "").lower()):
            skipped[oid] = (f"held from the first frame ({o.get('initial_state')!r}): the scene starts it at rest, so the first "
                            f"contact cannot be compared")
            continue
        f0, f1 = max(0, int((t0 - pad_s) * fps)), min(len(X) - 1, int(np.ceil((t1 + pad_s) * fps)))
        if f1 < f0:
            skipped[oid] = f"window {t0}-{t1} s outside the {len(X)} joint frames"; continue
        sides = ["left", "right"] if sg.get("acting_hand") == "both" else [sg.get("acting_hand")]
        tips = [(s, t) for s in sides for t in TIPS if f"{s}{t}" in col]
        if not tips:
            skipped[oid] = f"hands.npz has no fingertips for {sides}"; continue
        from tool_library.common.material_surface import MaterialSurface
        rid = model.body(root).id
        # a group object: every piece is its surface, so the first contact is measured to the nearest piece
        piece_ids = [b for b in (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in pieces_of.get(oid, [])) if b >= 0] or [rid]
        material = MaterialSurface(model, {b for b in range(model.nbody) if any(_is_under(model, b, r) for r in piece_ids)})
        if material.uncovered:
            issues.append(f"{oid}: material penetration coverage is unavailable for {material.uncovered}; "
                          "a closed material surface is required to verify fingertip depth")
        if not material.geoms:
            skipped[oid] = "no supported material surface geometry"; continue

        def measure(states=None, contact_after=-1):
            probe = mujoco.MjData(model) if states is not None else data
            closest = deepest = None
            for ff in range(f0, f1 + 1):
                if states is not None:
                    if ff not in states:
                        continue
                    probe.qpos[:] = states[ff]
                    mujoco.mj_forward(model, probe)
                finite = [(side, name, X[ff, col[f"{side}{name}"]]) for side, name in tips
                          if np.isfinite(X[ff, col[f"{side}{name}"]]).all()]
                if not finite:
                    continue
                distances, surfaces = material.distances(probe, np.array([entry[2] for entry in finite]))
                for (side, name, tip_point), signed_distance, surface in zip(finite, distances, surfaces):
                    gap_here = max(0.0, float(signed_distance))
                    depth_here = max(0.0, -float(signed_distance))
                    if ff > contact_after and (closest is None or gap_here < closest[0]):
                        closest = (gap_here, ff, f"{side}{name}", surface, tip_point)
                    if depth_here > 0 and name in DEPTH_TIPS and (deepest is None or depth_here > deepest[0]):
                        deepest = (depth_here, ff, f"{side}{name}")
            return closest, deepest

        closest, deepest = measure()
        witness = None
        rejected_witnesses = []
        if closest is not None and (closest[0] > tol_m or (deepest and deepest[0] > max_penetration_m)):
            from tool_library.common.contact_state_witness import prior_support_witnesses
            for trial in prior_support_witnesses(model, data, sem, root_of, oid, sg, names, X, fps, _surface_points, f1):
                states = trial.pop("_states", None)
                if trial["ok"] and states:
                    alt_closest, alt_deepest = measure(states, trial["first_measured_motion_frame"])
                    if (alt_closest is not None and alt_closest[0] <= tol_m
                            and (not alt_deepest or alt_deepest[0] <= max_penetration_m)):
                        trial["initial_surface_gap_m"] = float(closest[0])
                        closest, deepest, witness = alt_closest, alt_deepest, trial
                        break
                    trial["ok"] = False
                    trial["reason"] = "physical support states do not explain contact in its source time window"
                    trial["contact_gap_m"] = None if alt_closest is None else float(alt_closest[0])
                    trial["inside_depth_m"] = float(alt_deepest[0]) if alt_deepest else 0.0
                rejected_witnesses.append(trial)
        if closest is None:
            skipped[oid] = "no finite fingertip positions in the window"; continue
        gap, f, tip, sp, q = closest
        depth = deepest[0] if deepest else 0.0
        ok = gap <= tol_m and depth <= max_penetration_m and not material.uncovered
        part = None
        if len(piece_ids) > 1 and witness is None:
            # which piece the closest fingertip came to (the pieces at their initial state)
            dists = [(float(MaterialSurface(model, {b for b in range(model.nbody) if _is_under(model, b, r)}).distances(data, q[None])[0][0]), r)
                     for r in piece_ids]
            part = model.body(min(dists)[1]).name
        objects[oid] = {"subgoal": sg.get("index"), "frames": [f0, f1], "closest_frame": f, "fingertip": tip, "part": part,
                        "gap_m": round(gap, 4), "inside_depth_m": round(depth, 4),
                        "deepest": {"frame": deepest[1], "fingertip": deepest[2]} if deepest else None,
                        "offset_m": [round(float(x), 4) for x in (q - sp)], "ok": ok,
                        "distance_geometry": "individual material components", "uncovered_material_geoms": material.uncovered}
        if witness is not None:
            objects[oid]["support_motion_witness"] = witness
            objects[oid]["surface_state"] = "time-aligned bounded-force support replay"
        elif rejected_witnesses:
            objects[oid]["rejected_support_witnesses"] = rejected_witnesses
        # Absolute height alone cannot distinguish contact with a tall object
        # from an incorrect time window or camera registration.
        tip_z = [float(X[ff, col[f"{s}{tt}"]][2]) for ff in range(f0, f1 + 1) for s, tt in tips if np.isfinite(X[ff, col[f"{s}{tt}"]]).all()]
        low = min(tip_z) if tip_z else None
        objects[oid]["fingertip_min_height_m"] = round(low, 4) if low is not None else None
        if gap > tol_m:
            msg = (f"{oid}: at its first contact (subgoal {sg.get('index')}, frames {f0}-{f1}) the fingertips stop "
                   f"{gap * 100:.1f} cm short of its initial-state surface"
                   f"{f' (the nearest of its {len(piece_ids)} pieces, {part})' if part else ''} (closest at frame {f}, {tip}); "
                   f"surface-to-fingertip offset {[round(float(x), 3) for x in (q - sp)]} m. "
                   "This mismatch does not identify its cause. Check shared camera/hand coordinates, object "
                   "placement and scale, earlier motion of its support/container, and the contact time window "
                   "against the source frames. Do not move an object or rewrite Stage 1 solely to close this gap.")
            objects[oid]["diagnostic_scope"] = "initial-state surface consistency; cause unresolved"
            issues.append(msg)
        if depth > max_penetration_m:
            issues.append(f"{oid}: at its first contact (subgoal {sg.get('index')}) {deepest[2]} reaches {depth * 100:.1f} cm inside "
                          f"its material at frame {deepest[1]}. This overlap alone does not identify a placement or scale error; "
                          "check the shared coordinates, contact timing and earlier physical motion against source evidence.")
    return {"ok": not issues, "applicable": True, "issues": issues, "objects": objects, "skipped": skipped,
            "tol_m": tol_m, "fps": fps}
