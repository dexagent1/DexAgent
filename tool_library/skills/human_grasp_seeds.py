"""Ranked Sharpa grasp seeds from the demonstrator's own hand at every subgoal's contact window."""
from __future__ import annotations

TOOL = {
    "name": "human_grasp_seeds",
    "title": "Human grasp seeds",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Turns the human hand at each Stage 1 subgoal's contact window (EgoDex ARKit joints, an EgoDex .hdf5, or HaMeR "
               "meshes) into ranked Sharpa grasp-pose seeds per subgoal: the wrist relative to the object and the finger "
               "targets, retargeted through human_grasp_pose_prior / retarget_hand_motion, snapped onto the object, the grip "
               "fingers cinched onto its surface, and written as pose lists the executor (pose / pose_candidates), the "
               "stress tests and optimize_grasp_pose take.",
    "description": (
        "The demonstration already shows how the hand held each part; this skill reads it before any search. For every "
        "subgoal whose acting hand starts a contact (grasp_lift, hold, regrasp, pull, open/close_joint, unscrew, screw, "
        "tie, fold, or any motion on an object that hand did not touch in its previous subgoal; 'both' gives a seed list "
        "per hand) it takes the subgoal's time window (time_window_s x fps, or explicit frames), measures in every frame "
        "of the window (and `pre_s` before it) how far the human thumb and nearest finger are from the grasped body's "
        "bounding box in the hand scene, and picks up to `frames_per_window` frames: the arrival (the first frame within "
        "1.5 cm of the closest approach), frames after it and the closest one. Each frame is retargeted with "
        "human_grasp_pose_prior (knuckle frames matched for the wrist, per-finger IK on fingertip targets scaled by finger "
        "length, the hand lifted off the table, backed out of the object along the human's approach and snapped onto the "
        "object by at most snap_radius_m, because the demonstration and the reconstruction disagree by centimetres). The "
        "grip fingers are then cinched from the human's finger shape until their pads touch the grasped body (avoiding the "
        "table and bystanders), giving the closed and squeezed targets the executor clamps from; the human shape is kept "
        "as finger_targets_human. Seeds are ranked by a static score: hand overlap with the scene, the thumb's and the next "
        "pad's gap to the object, fingers the robot could not bend like the human, and how far the snap moved the hand "
        "(a long snap is a warning that Stage 2 misplaced the object). Near-duplicates (8 mm / 10 deg) are dropped. With "
        "verify the top seeds of each subgoal are driven through grasp_lift_test and the lifting ones rank first. The seed "
        "is a PRIOR: hand it to optimize_grasp_pose (starts=<the seed json>) or execute it after its stress tests."
    ),
    "inputs": [
        {"name": "scene_semantic_json", "type": "path | dict | None", "description": "Stage 1 scene_semantics.json: subgoals with index, acting_hand, motion, object_refs, time_window_s."},
        {"name": "subgoals", "type": "list | None", "description": "Instead of (or filtering) Stage 1: [{index, side, object, grasp_body?, frames: [f0, f1] | time_window_s, hand_scene?}]."},
        {"name": "hands_npz", "type": "path | None", "description": "EgoDex hand joints in the rig frame (read_egodex_episode's or place_egodex_camera_for_reach's hands.npz: names, rig_xyz)."},
        {"name": "egodex_hdf5", "type": "path | None", "description": "An EgoDex .hdf5 read directly (ARKit transforms); needs episode_json for the rig frame."},
        {"name": "episode_json", "type": "path | None", "description": "episode.json of the Stage 2 camera (rig: R_world_rig, origin_world; fps) for egodex_hdf5."},
        {"name": "hamer_npz", "type": "path | None", "description": "extract_hand_meshes' hands.npz (alternative source); needs camera and joint_regressor."},
        {"name": "camera", "type": "path | dict | None", "description": "camera.json of the video (HaMeR source)."},
        {"name": "joint_regressor", "type": "path | None", "description": "MANO J_regressor (.npy or MANO_RIGHT.pkl) for the HaMeR source."},
        {"name": "hand_scenes", "type": "dict | None", "description": "{subgoal index or '<index>_<side>' or object id: hand scene json/dict} (hand_scene_from_recon's out_json)."},
        {"name": "scene_recon_json", "type": "path | None", "description": "Stage 2 scene_recon.json: hand scenes not given are built with hand_scene_from_recon (target = the object id)."},
        {"name": "out_dir", "type": "path", "description": "Receives seeds_sg<NN>_<side>.json (a ranked JSON list of poses) per subgoal and human_grasp_seeds.json. Default 'poses'."},
        {"name": "fps", "type": "float | None", "description": "Frame rate of the hand track (default: episode_json's fps, else 30)."},
        {"name": "frames_per_window", "type": "int", "description": "Contact frames retargeted per subgoal. Default 5."},
        {"name": "pre_s", "type": "float", "description": "Seconds before the window also searched for the arrival (Stage 1 windows start late). Default 0.5."},
        {"name": "top_k", "type": "int", "description": "Seeds kept per subgoal. Default 6."},
        {"name": "motions", "type": "list[str] | None", "description": "Motions that start a contact (default the list above)."},
        {"name": "snap_radius_m", "type": "float", "description": "How far the retargeted hand may be translated onto the object. Default 0.07."},
        {"name": "verify", "type": "bool", "description": "Run grasp_lift_test on the top verify_top seeds of each subgoal. Default false."},
        {"name": "verify_top", "type": "int", "description": "Default 2."},
        {"name": "setup_dir", "type": "path | None", "description": "Setup folder (hand MJCFs); default the repository's setup/."},
    ],
    "outputs": [
        {"name": "subgoals", "type": "list", "description": "[{index, side, object, grasp_body, window_frames, arrival_frame, frames_tried, n_seeds, seeds_json, best {frame, score, overlap_mm, touching, snap_mm, lift?}, warnings}]."},
        {"name": "seeds", "type": "dict", "description": "{'<index>_<side>': seeds_json path}: give one to optimize_grasp_pose(starts=...) or as a plan entry's pose / pose_candidates."},
        {"name": "out_json", "type": "path", "description": "human_grasp_seeds.json (this summary)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "h5py (egodex_hdf5 only)"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "retargeting", "human prior", "egodex", "hamer", "seeds", "stage 3"],
    "used_in": [],
    "status": "tested",
    "entrypoint": "human_grasp_seeds",
}

import json
from pathlib import Path

import numpy as np

START_MOTIONS = ("grasp_lift", "hold", "regrasp", "pull", "open_joint", "close_joint", "unscrew", "screw", "tie", "fold")
NO_CONTACT_MOTIONS = ("release", "place", "carry", "insert", "pour", "tilt", "home")
DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "setup"


def _load(x):
    if x is None:
        return None
    if isinstance(x, (str, Path)):
        return json.loads(Path(x).read_text())
    return x


def hands_npz_from_hdf5(hdf5, episode_json, out_npz):
    """EgoDex joints straight from the .hdf5 in the Stage 2 rig frame (episode.json's rig: world <- rig rotation and the
    rig origin in ARKit world), written in read_egodex_episode's hands.npz format (names, rig_xyz)."""
    import h5py
    ep = _load(episode_json)
    rig = ep.get("rig") or {}
    if "R_world_rig" not in rig or "origin_world" not in rig:
        raise ValueError("episode_json has no rig {R_world_rig, origin_world}: give read_egodex_episode's episode.json")
    R = np.asarray(rig["R_world_rig"], float)
    o = np.asarray(rig["origin_world"], float)
    with h5py.File(str(hdf5), "r") as f:
        names = sorted(k for k in f["transforms"].keys() if k != "camera")
        P = np.stack([f[f"transforms/{k}"][()][:, :3, 3].astype(float) for k in names], axis=1)
    X = (P - o) @ R
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_npz), names=np.array(names), rig_xyz=X.astype(np.float32))
    return str(out_npz)


def _subgoal_list(sem, subgoals, motions, fps):
    """[{index, side, object, part, grasp_body, frames, hand_scene}] of the subgoals whose hand starts a contact."""
    out = []
    if subgoals:
        for s in subgoals:
            fr = s.get("frames")
            if fr is None and s.get("time_window_s"):
                fr = [int(round(float(s["time_window_s"][0]) * fps)), int(round(float(s["time_window_s"][1]) * fps))]
            if fr is None:
                raise ValueError(f"subgoal {s.get('index')}: give frames or time_window_s")
            obj = str(s.get("object") or "")
            out.append({"index": int(s["index"]), "side": s["side"], "object": obj.split(":")[0], "part": obj.split(":")[1] if ":" in obj else None,
                        "grasp_body": s.get("grasp_body"), "frames": [int(fr[0]), int(fr[1])], "hand_scene": s.get("hand_scene"),
                        "motion": s.get("motion")})
        return out
    if not sem:
        raise ValueError("give scene_semantic_json or subgoals")
    last = {"left": None, "right": None}
    for s in sem.get("subgoals") or []:
        sides = ["left", "right"] if s.get("acting_hand") == "both" else [s.get("acting_hand")]
        refs = s.get("object_refs") or []
        if not refs or not s.get("time_window_s"):
            for sd in sides:
                if sd in last:
                    last[sd] = refs[0] if refs else None
            continue
        motion = s.get("motion")
        for k, side in enumerate(sides):
            if side not in ("left", "right"):
                continue
            ref = refs[min(k, len(refs) - 1)] if s.get("acting_hand") == "both" else refs[0]
            new_contact = ref != last.get(side)
            last[side] = ref
            if motion in NO_CONTACT_MOTIONS and not new_contact:
                continue
            if motion not in motions and not new_contact:
                continue
            if motion in ("release", "home"):
                continue
            t0, t1 = (float(v) for v in s["time_window_s"])
            out.append({"index": int(s["index"]), "side": side, "object": ref.split(":")[0], "part": ref.split(":")[1] if ":" in ref else None,
                        "grasp_body": None, "frames": [int(round(t0 * fps)), int(round(t1 * fps))], "hand_scene": None, "motion": motion})
    return out


def _box_dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0.0), p - hi)))


def _arrival_frames(kp, frames, lo, hi, n_pick):
    """The frames the hand is on the object: thumb gap + nearest finger gap to the box, per frame; the arrival is the first
    frame within 1.5 cm of the closest approach. Returns (picked frames, cost per frame, arrival)."""
    fingers = ("index", "middle", "ring", "pinky")
    cost = []
    for i in range(len(frames)):
        th = _box_dist(np.asarray(kp["thumb"][i][-1]), lo, hi)
        ot = min(_box_dist(np.asarray(kp[f][i][-1]), lo, hi) for f in fingers)
        cost.append(th + ot)
    cost = np.asarray(cost)
    best = float(cost.min())
    arr_i = int(np.flatnonzero(cost <= best + 0.015)[0])
    picks = [arr_i]
    step = max(1, (len(frames) - arr_i) // max(n_pick, 1))
    for i in range(arr_i + step, len(frames), step):
        if len(picks) >= n_pick - 1:
            break
        picks.append(i)
    picks.append(int(np.argmin(cost)))
    seen, out = set(), []
    for i in picks:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return [int(frames[i]) for i in out[:n_pick]], cost, int(frames[arr_i])


def _cinch(scene, pose):
    """The grip fingers curled from the human's shape until their pads touch the grasped body (table and bystanders
    avoided): closed and squeezed targets, which pads touch and the hand's overlap with the scene."""
    import mujoco
    from tool_library.common import hand as H
    model, data0, rig, ids = H.open_scene(scene)
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    H.apply_pose(model, d, rig, pose)
    grip = [f for f in (pose.get("grip_fingers") or ["thumb", "index"]) if f in rig.fingertip_sites]
    obj = sorted(ids["grasp_gids"])
    avoid = [g for g in ids["obj_gids"] if g not in ids["grasp_gids"]] + ([ids["floor_gid"]] if ids["floor_gid"] >= 0 else [])
    rep = H.cinch_fingers(model, d, rig, obj, fingers=grip, avoid_gids=avoid)
    closed = {k: float(d.qpos[a]) for k, a in rig.finger_qpos.items()}
    touching = [f for f, v in rep.items() if v["gap_mm"] <= 0.5]
    overlap = H.deepest_overlap(model, d, rig, ids, keep_tips=tuple(grip))
    d2 = mujoco.MjData(model)
    d2.qpos[:] = data0.qpos
    H.apply_pose(model, d2, rig, pose, closed)
    H.cinch_fingers(model, d2, rig, obj, fingers=grip, touch_m=-0.0025, max_pen_m=0.0035, avoid_gids=avoid)
    squeezed = {k: float(d2.qpos[a]) for k, a in rig.finger_qpos.items()}
    return {"closed": closed, "squeezed": squeezed, "touching": touching, "gaps_mm": {f: v["gap_mm"] for f, v in rep.items()},
            "overlap_mm": round(1000 * overlap, 2)}


def _score(p):
    """Lower is better: overlap, the thumb's and the next pad's gaps, unreachable fingers, the snap distance."""
    gaps = p.get("cinch_gaps_mm") or {}
    th = abs(float(gaps.get("thumb", 30.0)))
    oth = sorted(abs(float(v)) for f, v in gaps.items() if f != "thumb") or [30.0]
    touching = p.get("touching") or []
    snap = float(np.linalg.norm((p.get("snap") or {}).get("shift_m") or [0.0, 0.0, 0.0])) * 1000.0
    s = 2.0 * float(p.get("overlap_mm", 0.0)) + th + oth[0] + 5.0 * len(p.get("unreachable_fingers") or []) + 0.1 * snap
    if "thumb" not in touching or len(touching) < 2:
        s += 40.0
    return round(s, 2)


def _rot_deg(e1, e2):
    from tool_library.common import hand as H
    R = H.euler_to_mat(e1).T @ H.euler_to_mat(e2)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def human_grasp_seeds(scene_semantic_json=None, subgoals=None, hands_npz=None, egodex_hdf5=None, episode_json=None, hamer_npz=None,
                      camera=None, joint_regressor=None, hand_scenes=None, scene_recon_json=None, out_dir="poses", fps=None,
                      frames_per_window=5, pre_s=0.5, top_k=6, motions=None, snap_radius_m=0.07, verify=False, verify_top=2,
                      setup_dir=None):
    from tool_library.common import hand as H
    from tool_library.common import human_prior as HP
    from tool_library.skills.human_grasp_pose_prior import human_grasp_pose_prior
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = [s for s in (hands_npz, egodex_hdf5, hamer_npz) if s is not None]
    if len(sources) != 1:
        raise ValueError("give exactly one hand source: hands_npz (EgoDex joints), egodex_hdf5 (+ episode_json) or hamer_npz (+ camera, joint_regressor)")
    ep = _load(episode_json) if episode_json is not None else None
    fps = float(fps or (ep or {}).get("fps") or 30.0)
    if egodex_hdf5 is not None:
        if ep is None:
            raise ValueError("egodex_hdf5 needs episode_json (the rig frame of the Stage 2 camera)")
        hands_npz = hands_npz_from_hdf5(egodex_hdf5, ep, out / "hands_rig.npz")
    if hamer_npz is not None and (camera is None or joint_regressor is None):
        raise ValueError("hamer_npz needs camera (camera.json) and joint_regressor (MANO J_regressor)")
    if hands_npz is not None:
        with np.load(str(hands_npz), allow_pickle=False) as z:
            n_frames = int(np.asarray(z["rig_xyz"]).shape[0])
    else:
        # HaMeR misses hands in some frames: per side, the frames where it found that hand
        with np.load(str(hamer_npz), allow_pickle=True) as z:
            hamer_frames = {"left": [], "right": []}
            for f, isr in zip(z["frame_idx"], z["is_right"]):
                for r in np.atleast_1d(isr):
                    hamer_frames["right" if int(r) == 1 else "left"].append(int(f))
        hamer_frames = {k: sorted(set(v)) for k, v in hamer_frames.items()}
        n_frames = max([max(v) + 1 for v in hamer_frames.values() if v] or [0])
    sem = _load(scene_semantic_json)
    todo = _subgoal_list(sem, subgoals, tuple(motions or START_MOTIONS), fps)
    hand_scenes = {str(k): v for k, v in (hand_scenes or {}).items()}
    results, seeds_index = [], {}
    for sg in todo:
        key = f"{sg['index']}_{sg['side']}"
        warnings = []
        hs = sg.get("hand_scene") or hand_scenes.get(key) or hand_scenes.get(str(sg["index"])) or hand_scenes.get(sg["object"])
        if hs is None:
            if not scene_recon_json:
                results.append({"index": sg["index"], "side": sg["side"], "object": sg["object"], "n_seeds": 0,
                                "warnings": ["no hand scene: give hand_scenes or scene_recon_json"]})
                continue
            from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon
            hs_path = out / f"hs_sg{sg['index']:02d}_{sg['side']}.json"
            hand_scene_from_recon(str(scene_recon_json), sg["side"], sg["object"], grasp_body=sg.get("grasp_body"),
                                  setup_dir=setup_dir, out_json=str(hs_path))
            hs = str(hs_path)
            if sg.get("part"):
                warnings.append(f"Stage 1 names the part '{sg['part']}': the hand scene targets the object's root; give hand_scenes "
                                "with grasp_body = the part's body to seed on the part")
        scene = H.load_scene_spec(hs)
        if scene.get("side", sg["side"]) != sg["side"]:
            results.append({"index": sg["index"], "side": sg["side"], "object": sg["object"], "n_seeds": 0,
                            "warnings": [f"the hand scene is for the {scene.get('side')} hand"]})
            continue
        model, data0, rig, ids = H.open_scene(scene)
        ext = H.object_extent(model, data0, sorted(ids["grasp_gids"]))
        lo, hi = np.asarray(ext["lo"], float), np.asarray(ext["hi"], float)
        f0 = max(0, int(sg["frames"][0] - round(pre_s * fps)))
        f1 = min(n_frames - 1, int(sg["frames"][1]))
        if hamer_npz is not None:
            window = [f for f in hamer_frames[sg["side"]] if f0 <= f <= f1]
        else:
            window = list(range(f0, f1 + 1))
        if not window:
            results.append({"index": sg["index"], "side": sg["side"], "object": sg["object"], "n_seeds": 0,
                            "warnings": [f"no hand frames in {f0}-{f1}"]})
            continue
        try:
            kp = (HP.keypoints_from_egodex(hands_npz, sg["side"], window) if hands_npz is not None
                  else HP.keypoints_from_hamer(hamer_npz, camera, sg["side"], window, joint_regressor))
        except ValueError as e:
            results.append({"index": sg["index"], "side": sg["side"], "object": sg["object"], "n_seeds": 0, "warnings": [str(e)]})
            continue
        picks, cost, arrival = _arrival_frames(kp, window, lo, hi, int(frames_per_window))
        if float(cost.min()) > 0.05:
            warnings.append(f"the human {sg['side']} hand comes no closer than {1000 * float(cost.min()):.0f} mm (thumb + nearest finger) to "
                            f"{sg['object']} in frames {f0}-{f1}: Stage 1's window or hand, or Stage 2's placement, disagrees with the demonstration")
        cands = []
        for fr in picks:
            kw = {"hands_npz": hands_npz} if hands_npz is not None else {"hamer_npz": hamer_npz, "camera": camera, "joint_regressor": joint_regressor}
            try:
                r = human_grasp_pose_prior(sg["side"], fr, hand_scene=scene, snap_radius_m=snap_radius_m, fps=fps,
                                           setup_dir=setup_dir, hand_xml=scene.get("hand_xml"), **kw)
            except ValueError as e:
                if hamer_npz is None:
                    warnings.append(f"frame {fr}: {e}")
                    continue
                try:                                       # a HaMeR gap in the frames before: the contact frame alone
                    r = human_grasp_pose_prior(sg["side"], fr, hand_scene=scene, snap_radius_m=snap_radius_m, fps=fps, window=0,
                                               approach_frames=0, setup_dir=setup_dir, hand_xml=scene.get("hand_xml"), **kw)
                except ValueError as e2:
                    warnings.append(f"frame {fr}: {e2}")
                    continue
            p = r["pose"]
            c = _cinch(scene, p)
            p["finger_targets_human"] = p["finger_targets"]
            p["finger_targets"] = c["closed"]
            p["finger_targets_squeezed"] = c["squeezed"]
            p["touching"] = c["touching"]
            p["cinch_gaps_mm"] = c["gaps_mm"]
            p["overlap_mm"] = max(float(p.get("overlap_mm", 0.0)), c["overlap_mm"])
            if c["touching"] and "thumb" in c["touching"] and len(c["touching"]) >= 2:
                p["grip_fingers"] = c["touching"]
            p.update({"source": "human_grasp_seeds", "family": "human", "subgoal": sg["index"], "object": sg["object"],
                      "grasp_body": p.get("grasp_body") or scene.get("grasp_body"), "press_depth_rad": float(p.get("press_depth_rad", 0.2))})
            p.pop("robot_tips", None)
            p["seed_score"] = _score(p)
            cands.append(p)
        cands.sort(key=lambda q: q["seed_score"])
        kept = []
        for p in cands:
            if any(np.linalg.norm(np.asarray(p["pos"]) - np.asarray(q["pos"])) < 0.008 and _rot_deg(p["euler"], q["euler"]) < 10.0 for q in kept):
                continue
            kept.append(p)
        kept = kept[:int(top_k)]
        if verify and kept:
            from tool_library.verifiers.grasp_lift_test import grasp_lift_test
            for p in kept[:int(verify_top)]:
                # Use the verifier's complete default lift ladder. A seed that
                # survives only the shorter lifts must not outrank one that
                # survives the full published grasp test.
                lt = grasp_lift_test(scene, p)
                p["lift"] = {"ok": bool(lt["ok"]), "first_fail": lt["first_fail"],
                             "tested_heights_m": [float(x['commanded_mm']) / 1000 for x in lt['lifts']],
                             "achieved_mm": lt["lifts"][-1]["achieved_mm"] if lt["lifts"] else 0.0}
            kept.sort(key=lambda q: (0 if q.get("lift", {}).get("ok") else 1, q["seed_score"]))
        for i, p in enumerate(kept):
            p["seed_rank"] = i
            warnings += [w for w in (p.get("warnings") or []) if w not in warnings][:2]
        path = out / f"seeds_sg{sg['index']:02d}_{sg['side']}.json"
        path.write_text(json.dumps(kept, indent=1, default=float))
        seeds_index[key] = str(path)
        best = kept[0] if kept else None
        results.append({"index": sg["index"], "side": sg["side"], "object": sg["object"], "grasp_body": scene.get("grasp_body"),
                        "hand_scene": hs if isinstance(hs, str) else None, "window_frames": [f0, f1], "arrival_frame": arrival,
                        "frames_tried": picks, "n_seeds": len(kept), "seeds_json": str(path),
                        "best": None if best is None else {"frame": best["frame"], "score": best["seed_score"], "overlap_mm": best.get("overlap_mm"),
                                                           "touching": best.get("touching"),
                                                           "snap_mm": round(1000 * float(np.linalg.norm((best.get("snap") or {}).get("shift_m") or [0, 0, 0])), 1),
                                                           "lift": best.get("lift")},
                        "warnings": warnings[:6]})
    summary = {"subgoals": results, "seeds": seeds_index, "fps": fps, "source": "hands_npz" if hamer_npz is None else "hamer_npz"}
    p = out / "human_grasp_seeds.json"
    p.write_text(json.dumps(summary, indent=1, default=float))
    summary["out_json"] = str(p)
    return summary
