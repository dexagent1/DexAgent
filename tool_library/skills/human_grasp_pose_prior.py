"""A robot hand pose retargeted from the demonstrator's hand at the contact frame."""
from __future__ import annotations

TOOL = {
    "name": "human_grasp_pose_prior",
    "title": "Human grasp pose prior",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Retargets the demonstrated human hand at the contact frame (EgoDex ARKit joints from read_egodex_episode, or HaMeR meshes) onto the Sharpa hand: the wrist pose by matching the hands' knuckle frames, the fingers by per-finger IK on fingertip targets scaled by finger length, written as a pose json the grasp stress tests and execute_subgoal_plan take.",
    "description": (
        "Both hands get the same anatomical frame from four keypoints (common/human_prior.py): origin at the four "
        "finger knuckles' centroid, z from the wrist to the middle knuckle, y across the knuckles toward the thumb side "
        "on a right hand (the little-finger side on a left hand), x = y cross z out of the palm, which is the Sharpa "
        "root's own convention. The fixed mapping from the human frame to the Sharpa root is the Sharpa's anatomical "
        "frame measured once from its own sites (a 6 degree tilt and a 9 cm offset from the root to the knuckles); "
        "for EgoDex it equals the ARKit wrist transform composed with the axis map Sharpa x = -y, y = +z, z = -x "
        "(right) / x = +y, y = +z, z = +x (left) up to a constant 6 degree offset. Each finger's target is the Sharpa finger base plus the human "
        "base-to-tip vector, expressed in the hand frame and scaled by the ratio of the two fingers' chain lengths "
        "(scaling='per_finger'; 'hand' uses one wrist-to-middle-tip ratio), so a curled human finger gives a curled "
        "robot finger of the robot's size; retarget_hand_motion solves the finger joints over a short window ending at "
        "the contact frame (warm starts, limits). The knuckles are matched, not the fingertips, so the pads land "
        "within about the size difference of the two hands of where the human's were. With a hand scene "
        "(hand_scene_from_recon's json) the pose is measured against the objects (overlap with them and with the table, "
        "each pad's distance to the grasped body); clear_overlap lifts the hand off the table and backs it out of the "
        "object along its approach; snap_to_object then translates the whole retargeted hand, orientation and finger "
        "shape kept (the human's grasp aperture), by at most snap_radius_m to where the thumb and its two nearest pads "
        "are closest to the object without the hand entering it, because the demonstration and the reconstruction "
        "disagree by centimetres (measured 1-10 cm on an EgoDex lego episode). grip_fingers are the thumb plus the "
        "pads within contact_tol_m of the object (the close phase must bring the thumb in), or without a scene the "
        "thumb plus the fingers the human curled toward it. approach is the direction the human wrist came from over the last approach_frames frames (pads to "
        "wrist, as the planner expects). The pose is a PRIOR: it says how the human held the object; the grasp tests "
        "decide whether the robot can."
    ),
    "inputs": [
        {"name": "side", "type": "str", "description": "'left' or 'right' (the robot station that takes the grasp)."},
        {"name": "frame", "type": "int", "description": "The contact frame (a Stage 1 subgoal's grasp moment, video frame index)."},
        {"name": "out_json", "type": "path | None", "description": "Where the pose json is written."},
        {"name": "hands_npz", "type": "path | None", "description": "read_egodex_episode's hands.npz (ARKit joints in the rig frame)."},
        {"name": "hamer_npz", "type": "path | None", "description": "extract_hand_meshes' hands.npz (alternative source); needs camera and joint_regressor."},
        {"name": "camera", "type": "path | dict | None", "description": "camera.json of the video (HaMeR source only)."},
        {"name": "joint_regressor", "type": "path | None", "description": "MANO J_regressor (.npy or MANO_RIGHT.pkl from HaMeR's _DATA) for the HaMeR source."},
        {"name": "hand_xml", "type": "path | None", "description": "The robot hand alone; default <setup>/robot/sharpa_<side>.xml."},
        {"name": "setup_dir", "type": "path | None", "description": "Setup folder; default the repository's setup/."},
        {"name": "hand_scene", "type": "path | dict | None", "description": "hand_scene_from_recon's scene (json path or dict) to measure the pose against the objects."},
        {"name": "grip_fingers", "type": "list[str] | None", "description": "Override the fingers that grip."},
        {"name": "scaling", "type": "str", "description": "'per_finger' (default) or 'hand'."},
        {"name": "window", "type": "int", "description": "Frames before the contact frame retargeted with it (warm start; the first is reported as the preshape). Default 10."},
        {"name": "approach_frames", "type": "int", "description": "Frames of wrist motion the approach direction is read from. Default 8."},
        {"name": "clear_overlap", "type": "bool", "description": "With a hand scene, back the hand out along the approach until the overlap is at most 2 mm (at most 4 cm). Default true."},
        {"name": "contact_tol_m", "type": "float", "description": "A pad this close to the grasped body grips. Default 0.015."},
        {"name": "snap_to_object", "type": "bool", "description": "With a hand scene: translate the retargeted hand (orientation and finger shape kept) by at most snap_radius_m to where the thumb and its two nearest pads are closest to the object without entering it; the unshifted position is kept as prior_pos. Default true."},
        {"name": "snap_radius_m", "type": "float", "description": "How far the snap may translate the retargeted hand (per axis; 0.5x downward) to bring the thumb and its two nearest pads onto the object. Default 0.07."},
        {"name": "fps", "type": "float", "description": "Frame rate of the source (retargeting rate limits). Default 30."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "{side, pos, euler (intrinsic XYZ, the hand root's world pose = the wrist joints), finger_targets {short joint name: rad}, grip_fingers, approach, finger_targets_preshape, source 'human_prior', frame, residual_mm per finger, finger_scale, human {wrist, tips, knuckle_origin}, robot_tips, and with a scene overlap_mm, cleared_m, tip_to_object_mm}; the input grasp_stress_test, grasp_lift_test and execute_subgoal_plan take."},
        {"name": "out_json", "type": "path", "description": "The written pose file."},
    ],
    "status": "tested",
    "entrypoint": "human_grasp_pose_prior",
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["grasp", "retargeting", "human prior", "egodex", "hamer", "hand pose", "stage 3"],
    "cli": "python -m tool_library.skills.human_grasp_pose_prior right 42 pose.json --hands-npz hands.npz",
}

import json
import math
import sys
from pathlib import Path

import numpy as np

from tool_library.common import human_prior as HP

DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "setup"


def _curl_deg(chain):
    """Bend of a finger: angle between its first segment and its base-to-tip line, summed over the joints."""
    P = np.asarray(chain, float)
    seg = np.diff(P, axis=0)
    ang = 0.0
    for a, b in zip(seg[:-1], seg[1:]):
        c = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)
        ang += math.degrees(math.acos(np.clip(c, -1, 1)))
    return ang


def human_grasp_pose_prior(side, frame, out_json=None, hands_npz=None, hamer_npz=None, camera=None, joint_regressor=None,
                           hand_xml=None, setup_dir=None, hand_scene=None, grip_fingers=None, scaling="per_finger", window=10,
                           approach_frames=8, clear_overlap=True, contact_tol_m=0.015, snap_to_object=True, snap_radius_m=0.07,
                           fps=30.0):
    from tool_library.skills.retarget_hand_motion import retarget_hand_motion
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if scaling not in ("per_finger", "hand"):
        raise ValueError("scaling must be 'per_finger' or 'hand'")
    if (hands_npz is None) == (hamer_npz is None):
        raise ValueError("give exactly one source: hands_npz (EgoDex, read_egodex_episode) or hamer_npz (extract_hand_meshes)")
    frame = int(frame)
    first = max(0, frame - max(int(window), int(approach_frames)))
    frames = np.arange(first, frame + 1)
    if hands_npz is not None:
        kp = HP.keypoints_from_egodex(hands_npz, side, frames)
    else:
        if camera is None:
            raise ValueError("hamer_npz needs camera (the video's camera.json) to place the hand in the rig frame")
        kp = HP.keypoints_from_hamer(hamer_npz, camera, side, frames, joint_regressor)
    hand_xml = str(hand_xml or (Path(setup_dir) if setup_dir else DEFAULT_SETUP) / "robot" / f"sharpa_{side}.xml")
    ref = HP.sharpa_reference(hand_xml, side)
    R_a, c_a = ref["R_anat"], ref["c_anat"]
    T = len(frames)
    # the retargeting window: the last `window` frames up to the contact
    w0 = max(0, T - 1 - int(window))
    roots, targets = [], {f: [] for f in HP.FINGERS}
    scale = {}
    kp_c = HP.frame_at(kp, T - 1)
    for f in HP.FINGERS:
        if scaling == "per_finger":
            scale[f] = ref["lengths"][f] / max(HP.chain_length(kp_c[f]), 1e-6)
        else:
            scale[f] = ref["hand_length"] / max(float(np.linalg.norm(kp_c["middle"][-1] - kp_c["wrist"])), 1e-6)
    for t in range(w0, T):
        k = HP.frame_at(kp, t)
        R_h, c_h = HP.anatomical_frame(k, side)
        R_root = R_h @ R_a.T
        p_root = c_h - R_root @ c_a
        roots.append((p_root, R_root))
        for f in HP.FINGERS:
            v = R_h.T @ (np.asarray(k[f][-1]) - np.asarray(k[f][0]))          # base -> tip in the hand frame
            targets[f].append(ref["kp"][f][0] + scale[f] * (R_a @ v))           # in the Sharpa root frame
    res = retarget_hand_motion(hand_xml, hand_root=ref["root_body"], fingertips={f: ref["tips"][f]["body"] for f in HP.FINGERS},
                               targets={f: np.array(v) for f, v in targets.items()},
                               tip_points={f: ref["tips"][f]["point"] for f in HP.FINGERS}, fps=fps)
    head = f"{side}_"
    short = [n[len(head):] if n.startswith(head) else n for n in res["joint_names"]]
    q_contact = {k: float(v) for k, v in zip(short, res["qpos"][-1])}
    q_pre = {k: float(v) for k, v in zip(short, res["qpos"][0])}
    p_root, R_root = roots[-1]
    from tool_library.common import hand as H
    euler = H.mat_to_euler(R_root)
    # approach: where the wrist came from (pads -> wrist), else wrist minus fingertip centroid
    wrist = np.asarray(kp["wrist"], float)
    a0 = max(0, T - 1 - int(approach_frames))
    mv = wrist[a0] - wrist[T - 1]
    tips_h = np.array([kp_c[f][-1] for f in HP.FINGERS])
    if np.linalg.norm(mv) >= 0.015:
        approach = mv / np.linalg.norm(mv)
        approach_src = f"wrist motion over frames {int(frames[a0])}-{frame}"
    else:
        d = np.asarray(kp_c["wrist"]) - tips_h.mean(axis=0)
        approach = d / max(np.linalg.norm(d), 1e-9)
        approach_src = "wrist minus fingertip centroid (the wrist barely moved before contact)"
    # robot fingertips at the pose (world), from the hand model at the solved joints
    import mujoco
    m = mujoco.MjModel.from_xml_path(hand_xml)
    d = mujoco.MjData(m)
    rig = H.HandRig.build(m, "", side)
    rig.teleport(d, p_root, euler)
    rig.set_fingers(d, q_contact)
    mujoco.mj_forward(m, d)
    robot_tips = {f: d.site_xpos[sid].astype(float).tolist() for f, sid in rig.fingertip_sites.items()}
    pose = {"side": side, "pos": [float(v) for v in p_root], "euler": [float(v) for v in euler], "finger_targets": q_contact,
            "finger_targets_preshape": q_pre, "approach": [float(v) for v in approach], "approach_source": approach_src,
            "source": "human_prior", "frame": frame, "preshape_frame": int(frames[w0]), "scaling": scaling,
            "finger_scale": {f: round(float(s), 4) for f, s in scale.items()},
            "residual_mm": {f: v["max_mm"] for f, v in res["residual_mm"].items()},
            "unreachable_fingers": sorted(res["unreachable"]),
            "human": {"wrist": [float(v) for v in kp_c["wrist"]], "tips": {f: [float(v) for v in kp_c[f][-1]] for f in HP.FINGERS},
                      "knuckle_origin": [float(v) for v in HP.anatomical_frame(kp_c, side)[1]]},
            "robot_tips": robot_tips, "hand_xml": hand_xml}
    warnings = []
    if res["unreachable"]:
        warnings.append(f"fingers {sorted(res['unreachable'])} could not reach their scaled targets (residual above 4 mm): the "
                        "human finger bent beyond the robot's joint range; the joints are clamped at their limits")
    if hand_scene is not None:
        meas = _measure_in_scene(hand_scene, pose, side, clear_overlap, contact_tol_m, snap_to_object, snap_radius_m)
        pose.update(meas["pose_update"])
        warnings += meas["warnings"]
        auto = meas["grip"]
    else:
        # without objects: the thumb and every finger the human curled (or brought near the thumb tip)
        thumb_tip = np.asarray(kp_c["thumb"][-1])
        auto = [f for f in ("index", "middle", "ring", "pinky")
                if _curl_deg(kp_c[f]) > 35.0 or np.linalg.norm(np.asarray(kp_c[f][-1]) - thumb_tip) < 0.06]
        auto = (["thumb"] + auto) if auto else ["thumb", "index"]
    pose["grip_fingers"] = list(grip_fingers) if grip_fingers else auto
    pose["warnings"] = warnings
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(pose, indent=1))
    return {"pose": pose, "out_json": str(out_json) if out_json else None}


def _overlaps(model, d, rig, ids):
    """(object overlap, floor overlap) in metres: the hand's deepest interpenetration with the objects and the table."""
    from tool_library.common import hand as H
    ids_obj = dict(ids); ids_obj["floor_gid"] = -1
    ids_floor = dict(ids); ids_floor["obj_gids"] = set()
    return H.deepest_overlap(model, d, rig, ids_obj, keep_tips=()), H.deepest_overlap(model, d, rig, ids_floor, keep_tips=())


def _tip_distances(model, d, rig, tgt):
    """Each finger's pad: signed distance (m) to the grasped body and the nearest pair of points (pad, body)."""
    import mujoco
    from tool_library.common import hand as H
    out = {}
    for f in H.FINGERS:
        gs = [g for g in rig.tip_geoms(f) if g in rig.collision_gids] or list(rig.tip_geoms(f))
        best = None
        for g in gs:
            for og in tgt:
                ft = np.zeros(6)
                dist = mujoco.mj_geomDistance(model, d, g, og, 0.15, ft)
                if best is None or dist < best[0]:
                    best = (float(dist), ft[:3].copy(), ft[3:].copy(), og)
        out[f] = best
    return out


def _measure_in_scene(hand_scene, pose, side, clear_overlap, contact_tol_m, snap, snap_radius_m):
    import mujoco
    from tool_library.common import episode as E
    from tool_library.common import hand as H
    scene = H.load_scene_spec(hand_scene)
    if scene.get("side", side) != side:
        raise ValueError(f"hand_scene is for the {scene.get('side')} hand, the pose is for the {side} hand")
    model, data0, rig, ids = H.open_scene(scene)
    bid = H.grasp_body(model, scene, ids)
    sub = E.subtree_bodies(model, bid)
    tgt = [g for g in ids["obj_gids"] if int(model.geom_bodyid[g]) in sub] or list(ids["obj_gids"])
    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    pos = np.asarray(pose["pos"], float).copy()
    eul = np.asarray(pose["euler"], float)
    rig.set_fingers(d, pose["finger_targets"])

    def place(p):
        rig.teleport(d, p, eul)
        mujoco.mj_forward(model, d)

    place(pos)
    ov_obj0, ov_floor0 = _overlaps(model, d, rig, ids)
    lifted, cleared = 0.0, 0.0
    a = np.asarray(pose["approach"], float)
    if clear_overlap:
        # the table first (straight up), then the object (back along the approach)
        if ov_floor0 > 0.002:
            lifted = ov_floor0 - 0.0015
            for _ in range(8):
                place(pos + [0.0, 0.0, lifted])
                f = _overlaps(model, d, rig, ids)[1]
                if f <= 0.002:
                    break
                lifted += f - 0.0015
            pos = pos + [0.0, 0.0, lifted]
        place(pos)
        if _overlaps(model, d, rig, ids)[0] > 0.002:
            lo, hi = 0.0, 0.04
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                place(pos + a * mid)
                if _overlaps(model, d, rig, ids)[0] > 0.002:
                    lo = mid
                else:
                    hi = mid
            cleared = hi
            pos = pos + a * cleared
    place(pos)
    ov_obj, ov_floor = _overlaps(model, d, rig, ids)
    td = _tip_distances(model, d, rig, tgt)
    tipd = {f: (round(1000 * v[0], 1) if v else None) for f, v in td.items()}
    grip = [f for f in H.FINGERS if td.get(f) and td[f][0] <= contact_tol_m]
    warnings = []
    upd = {"pos": [float(v) for v in pos], "overlap_mm": round(1000 * max(ov_obj, ov_floor), 2), "overlap_object_mm": round(1000 * ov_obj, 2),
           "overlap_table_mm": round(1000 * ov_floor, 2), "overlap_mm_before_clear": round(1000 * max(ov_obj0, ov_floor0), 2),
           "lifted_m": round(lifted, 4), "cleared_m": round(cleared, 4), "tip_to_object_mm": tipd, "grasp_body": model.body(bid).name}
    if snap:
        # The demonstration and the reconstruction disagree by centimetres, so the retargeted pads float beside the object
        # or sink into it: translate the whole hand (orientation and finger shape kept: the human's grasp aperture) to
        # where the thumb and its two nearest finger pads sit closest to the object's surface without the hand entering it.
        def pad_cost(delta):
            place(pos + delta)
            t = _tip_distances(model, d, rig, tgt)
            if not t.get("thumb"):
                return 9.0, None
            others = sorted((abs(v[0]), f) for f, v in t.items() if v is not None and f != "thumb")[:2]
            return abs(t["thumb"][0]) + sum(c for c, _ in others), ["thumb"] + [f for _, f in others]

        R = float(snap_radius_m)
        cands = []
        for dx in np.linspace(-R, R, 7):
            for dy in np.linspace(-R, R, 7):
                for dz in np.linspace(-R / 2, R, 6):
                    delta = np.array([dx, dy, dz])
                    c, fs = pad_cost(delta)
                    cands.append((c, tuple(delta), fs))
        cands.sort(key=lambda t: t[0])
        best = None
        for c, delta, fs in cands[:40]:                  # the hand-enters-nothing check only on the most promising shifts
            for k in range(3):                           # refine around it at a third of the step
                step = R / 3 / (k + 1)
                for e in np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) * step:
                    cand = np.clip(np.asarray(delta) + e, [-R, -R, -R / 2], [R, R, R])
                    c2, fs2 = pad_cost(cand)
                    if c2 < c:
                        c, delta, fs = c2, tuple(cand), fs2
            place(pos + np.asarray(delta))
            oo, of = _overlaps(model, d, rig, ids)
            if oo > 0.004 or of > 0.004:
                continue
            if best is None or c < best[0]:
                best = (c, np.asarray(delta), fs)
        if best is None:
            warnings.append(f"snap skipped: every shift within {1000 * R:.0f} mm that brings the pads to the object drives the hand "
                            "into the object or the table; the pose stays where the demonstration put it")
            snap = False
    if snap:
        delta, contact = best[1], best[2]
        place(pos + delta)
        oo, of = _overlaps(model, d, rig, ids)
        t = _tip_distances(model, d, rig, tgt)
        upd["snap"] = {"contact_fingers": contact, "shift_m": [round(float(v), 4) for v in delta],
                       "tip_to_object_mm": {f: (round(1000 * v[0], 1) if v else None) for f, v in t.items()},
                       "overlap_object_mm": round(1000 * oo, 2), "overlap_table_mm": round(1000 * of, 2)}
        upd["prior_pos"] = [float(v) for v in pos]
        pos = pos + delta
        upd["pos"] = [float(v) for v in pos]
        upd["overlap_mm"] = round(1000 * max(oo, of), 2)
        td = t
        grip = [f for f in H.FINGERS if td.get(f) and td[f][0] <= contact_tol_m]
        if float(np.linalg.norm(delta)) > 0.8 * R:
            warnings.append(f"snap: the hand had to move {1000 * float(np.linalg.norm(delta)):.0f} mm to reach the object, close to the "
                            "search limit: the demonstration and the reconstruction disagree by more than a few centimetres here "
                            "(check the object's size and placement against the hand skeleton)")
    if grip and "thumb" not in grip:
        grip = ["thumb"] + grip                     # the close phase must bring the thumb in against the fingers
    if upd["overlap_mm"] > 4.0:
        warnings.append(f"the hand overlaps the scene by {upd['overlap_mm']:.1f} mm (object {1000 * ov_obj:.1f}, table {1000 * ov_floor:.1f}) after "
                        "clearing: the human hand and the reconstructed object disagree here (registration or object size)")
    if len(grip) < 2:
        warnings.append(f"only {grip or 'no fingers'} within {1000 * contact_tol_m:.0f} mm of the grasped body: the prior does not touch "
                        "the object; use it as the wrist seed of pose_from_points or generate_rigid_grasps")
    return {"pose_update": upd, "grip": grip if len(grip) >= 2 else ["thumb", "index"], "warnings": warnings}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("side"); ap.add_argument("frame", type=int); ap.add_argument("out_json")
    ap.add_argument("--hands-npz"); ap.add_argument("--hand-scene")
    a = ap.parse_args(argv)
    r = human_grasp_pose_prior(a.side, a.frame, a.out_json, hands_npz=a.hands_npz, hand_scene=a.hand_scene)
    print(json.dumps({k: r["pose"][k] for k in ("pos", "euler", "grip_fingers", "residual_mm")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
