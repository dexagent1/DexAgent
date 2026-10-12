"""human_grasp_seeds: ranked Sharpa grasp seeds from a synthetic EgoDex-like hand track.

The 'human' is the Sharpa hand itself at a grasp generate_rigid_grasps found and grasp_lift_test passed, shrunk to 93 %
about the wrist (a smaller hand), registered 1-1.5 cm off the reconstructed object (the demonstration and the
reconstruction disagree by centimetres), with 1.5 mm of keypoint noise, an approach from 10 cm out along the grasp's
approach and a lift after the contact. The track is written in read_egodex_episode's hands.npz format (and, for one
case, as an EgoDex .hdf5 of ARKit transforms with an episode.json rig frame). The seeds must lift the object
(grasp_lift_test) and hold it (grasp_hold_test) in the hand scene. Four objects (three boxes, a cylinder), both hands; one
track as HaMeR meshes with the hand missed in every seventh frame.

    MUJOCO_GL=egl python tool_library/tests/test_human_grasp_seeds.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.common import hand as H  # noqa: E402
from tool_library.common import human_prior as HP  # noqa: E402
from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps  # noqa: E402
from tool_library.skills.human_grasp_seeds import human_grasp_seeds  # noqa: E402
from tool_library.verifiers.grasp_hold_test import grasp_hold_test  # noqa: E402
from tool_library.verifiers.grasp_lift_test import grasp_lift_test  # noqa: E402

FAILS = []
TMP = Path(tempfile.mkdtemp(prefix="dexagent_seeds_", dir=os.environ.get("TMPDIR")))
FPS = 30.0


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def object_scene(kind, dims, side, pos, yaw):
    if kind == "box":
        g = f'<geom name="{kind}_body" type="box" size="{dims[0]} {dims[1]} {dims[2]}" pos="0 0 {dims[2]}" mass="0.1" rgba="0.8 0.5 0.2 1"/>'
    else:
        g = f'<geom name="{kind}_body" type="cylinder" size="{dims[0]} {dims[2]}" pos="0 0 {dims[2]}" mass="0.1" rgba="0.2 0.5 0.8 1"/>'
    p = TMP / f"{kind}_{side}.xml"
    p.write_text(f'<mujoco model="{kind}"><worldbody><body name="{kind}"><freejoint/>{g}</body></worldbody></mujoco>')
    return {"hand_xml": str(ROOT / "setup" / "robot" / f"sharpa_{side}.xml"), "side": side, "floor": True,
            "objects": [{"name": kind, "xml": str(p), "pos": list(pos), "yaw_deg": yaw}]}


def reference_grasp(scene):
    """A grasp of the robot hand that lifts the object: the 'demonstrated' grasp."""
    g = generate_rigid_grasps(scene, yaw_step_deg=90, include_top=False, finger_dirs=(), thumb_oppose_fracs=(0.4,), pinch_ik_iters=160)
    for c in g["grasps"][:4]:
        if grasp_lift_test(scene, c)["ok"]:
            return c
    return None


def human_track(side, pose, scale=0.93, offset=(0.012, -0.009, 0.004), noise_m=0.0015, seed=0, n=66):
    """ARKit-named joints of the Sharpa hand at the grasp, shrunk about the wrist, shifted and noisy: frames 0-24 the
    open hand 10 cm out along the approach, 25-34 the approach, 35-44 closing, 45-54 holding, 55-65 lifting 6 cm."""
    xml = ROOT / "setup" / "robot" / f"sharpa_{side}.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    rig = H.HandRig.build(m, "", side)
    closed = {k: float(v) for k, v in pose["finger_targets"].items() if k in rig.finger_range}
    opened = H.open_targets(rig, closed, 0.55)
    pos, eul = np.asarray(pose["pos"], float), np.asarray(pose["euler"], float)
    a = np.asarray(pose.get("approach") or [0, 0, 1.0], float)
    a = a / np.linalg.norm(a)
    rng = np.random.default_rng(seed)
    frames, names = [], None
    for t in range(n):
        if t < 25:
            p, u = pos + 0.10 * a, 0.0
        elif t < 35:
            s = (t - 25) / 10.0
            p, u = pos + 0.10 * (1 - s) * a, 0.0
        elif t < 45:
            p, u = pos, (t - 35) / 9.0
        elif t < 55:
            p, u = pos, 1.0
        else:
            p, u = pos + np.array([0, 0, 0.06 * (t - 55) / 10.0]), 1.0
        fingers = {k: opened[k] + (closed[k] - opened[k]) * min(u, 1.0) for k in closed}
        rig.teleport(d, p, eul)
        rig.set_fingers(d, fingers)
        mujoco.mj_forward(m, d)
        wrist = d.xpos[m.body(f"{side}_base_yaw").id].copy()
        row = {side + HP.ARKIT_WRIST: wrist}
        for f, chain in HP.ARKIT_CHAINS.items():
            for jn, k in zip(chain, HP.SHARPA_SITES):
                row[side + jn] = d.site_xpos[m.site(f"{side}_{f}_{k}").id].copy()
        names = sorted(row)
        X = np.array([wrist + scale * (row[k] - wrist) for k in names]) + np.asarray(offset) + rng.normal(0.0, noise_m, (len(names), 3))
        frames.append(X)
    return np.array(names), np.array(frames, np.float32)


def write_hdf5(names, X, path):
    """The same joints as an EgoDex .hdf5 (4x4 ARKit transforms, y up) plus the episode.json rig frame that maps them back."""
    import h5py
    R_wr = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])      # world <- rig: rig z up = ARKit +y
    origin = np.array([0.3, -1.1, 0.7])
    with h5py.File(str(path), "w") as f:
        for j, nm in enumerate(names):
            T = np.tile(np.eye(4), (len(X), 1, 1))
            T[:, :3, 3] = origin + X[:, j, :].astype(float) @ R_wr.T
            f.create_dataset(f"transforms/{nm}", data=T)
        f.create_dataset("transforms/camera", data=np.tile(np.eye(4), (len(X), 1, 1)))
    ep = {"fps": FPS, "rig": {"R_world_rig": R_wr.tolist(), "origin_world": origin.tolist()}}
    Path(str(path).replace(".hdf5", "_episode.json")).write_text(json.dumps(ep))
    return str(path).replace(".hdf5", "_episode.json")


def write_hamer(names, X, side, d, drop_every=7):
    """The same joints in extract_hand_meshes' hands.npz format (MANO vertices in a camera at the rig origin, HaMeR missing
    the hand in every drop_every-th frame) with a J_regressor that reads joint j off vertex j."""
    col = {str(n): i for i, n in enumerate(names)}
    J = np.zeros((16, 778))
    J[np.arange(16), np.arange(16)] = 1.0
    joint_of = {0: HP.ARKIT_WRIST}
    for f in ("index", "middle", "pinky", "ring", "thumb"):
        for k, jid in enumerate(HP.MANO_JOINTS[f]):
            joint_of[jid] = HP.ARKIT_CHAINS[f][k]
    fidx, isr, verts, cam_t = [], [], [], []
    for t in range(len(X)):
        if t % drop_every == drop_every - 1:
            continue
        V = np.zeros((778, 3))
        for jid, jn in joint_of.items():
            V[jid] = X[t, col[side + jn]]
        for f, vid in HP.MANO_TIP_VERTS.items():
            V[vid] = X[t, col[side + HP.ARKIT_CHAINS[f][-1]]]
        fidx.append(t); isr.append(np.array([1 if side == "right" else 0])); verts.append(V[None]); cam_t.append(np.zeros((1, 3)))
    obj = lambda L: np.array(L + [None], dtype=object)[:-1]      # noqa: E731 (ragged per-frame arrays, as HaMeR writes them)
    np.savez(Path(d) / "hamer_hands.npz", frame_idx=np.array(fidx), is_right=obj(isr), verts=obj(verts), cam_t=obj(cam_t))
    np.save(Path(d) / "J_regressor.npy", J)
    cam = {"width": 640, "height": 480, "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "R_cv": np.eye(3).tolist(), "pos": [0.0, 0.0, 0.0]}
    return str(Path(d) / "hamer_hands.npz"), str(Path(d) / "J_regressor.npy"), cam


def run_case(name, kind, dims, side, pos, yaw, source="npz", semantic=False):
    t0 = time.time()
    scene = object_scene(kind, dims, side, pos, yaw)
    ref = reference_grasp(scene)
    if ref is None:
        check(f"{name}: a reference grasp to demonstrate", False, "generate_rigid_grasps found none that lifts")
        return
    names, X = human_track(side, ref, seed=len(name))
    hs_path = TMP / f"hs_{name}.json"
    hs_path.write_text(json.dumps(scene))
    kw = {}
    if source == "hdf5":
        ep = write_hdf5(names, X, TMP / f"{name}.hdf5")
        kw = {"egodex_hdf5": str(TMP / f"{name}.hdf5"), "episode_json": ep}
    elif source == "hamer":
        (TMP / name).mkdir(exist_ok=True)
        hn, jr, cam = write_hamer(names, X, side, TMP / name)
        kw = {"hamer_npz": hn, "joint_regressor": jr, "camera": cam, "fps": FPS}
    else:
        np.savez(TMP / f"{name}_hands.npz", names=names, rig_xyz=X)
        kw = {"hands_npz": str(TMP / f"{name}_hands.npz"), "fps": FPS}
    out_dir = TMP / f"seeds_{name}"
    if semantic:
        sem = {"subgoals": [{"index": 1, "acting_hand": "left" if side == "right" else "right", "motion": "hold", "object_refs": ["other"],
                             "time_window_s": [0.0, 0.5]},
                            {"index": 2, "acting_hand": side, "motion": "grasp_lift", "object_refs": [kind], "time_window_s": [1.2, 2.0]},
                            {"index": 3, "acting_hand": side, "motion": "carry", "object_refs": [kind], "time_window_s": [2.0, 2.2]}]}
        r = human_grasp_seeds(scene_semantic_json=sem, hand_scenes={f"2_{side}": str(hs_path)}, out_dir=str(out_dir), verify=True,
                              verify_top=3, **kw)
        mine = [s for s in r["subgoals"] if s["index"] == 2]
        others = [s for s in r["subgoals"] if s["index"] != 2]
        check(f"{name}: Stage 1 subgoals read; the carry continuing the grasp gets no seeds, the other hand's subgoal "
              "without a hand scene is reported", len(mine) == 1 and not any(s["index"] == 3 for s in r["subgoals"])
              and all(s["n_seeds"] == 0 and s["warnings"] for s in others), json.dumps([(s["index"], s["side"], s["n_seeds"]) for s in r["subgoals"]]))
        sg = mine[0] if mine else None
    else:
        r = human_grasp_seeds(subgoals=[{"index": 1, "side": side, "object": kind, "time_window_s": [1.2, 2.0]}],
                              hand_scenes={"1": str(hs_path)}, out_dir=str(out_dir), verify=True, verify_top=3, **kw)
        sg = r["subgoals"][0]
    if not sg or not sg.get("n_seeds"):
        check(f"{name}: seeds produced", False, json.dumps(sg))
        return
    seeds = json.loads(Path(sg["seeds_json"]).read_text())
    keys = {"pos", "euler", "finger_targets", "finger_targets_squeezed", "grip_fingers", "approach", "side"}
    passing = [s for s in seeds if s.get("lift", {}).get("ok")]
    best = passing[0] if passing else None
    hold = grasp_hold_test(scene, best) if best else {"ok": False}
    lift = grasp_lift_test(scene, best) if best else {"ok": False, "lifts": []}
    from tool_library.skills.execute_subgoal_plan import _pose_candidates
    cands = _pose_candidates({"pose": sg["seeds_json"]})
    check(f"{name}: the seed file is a plan entry's ranked pose list (the executor's pose candidates)",
          len(cands) == len(seeds) and cands[0]["pos"] == seeds[0]["pos"])
    check(f"{name}: a ranked seed from the human track passes grasp_lift_test (60/100/150 mm) and grasp_hold_test",
          bool(best) and lift["ok"] and hold["ok"] and keys <= set(best) and best["source"] == "human_grasp_seeds"
          and seeds[0] is best,
          f"{len(seeds)} seeds, arrival frame {sg['arrival_frame']}, frames {sg['frames_tried']}, best frame "
          f"{best and best['frame']} score {best and best['seed_score']} touching {best and best['touching']} snap "
          f"{sg['best'] and sg['best']['snap_mm']} mm, lift {[x['achieved_mm'] for x in lift.get('lifts', [])]} mm, "
          f"hold slip {hold.get('slip_mm_s')} mm/s ({time.time() - t0:.0f} s)")


def main():
    t0 = time.time()
    run_case("box_right", "box", (0.022, 0.02, 0.03), "right", (0.12, 0.40, 0.0), 15.0)
    run_case("cylinder_left_semantic", "cyl", (0.028, 0.0, 0.05), "left", (-0.12, 0.42, 0.0), 0.0, semantic=True)
    run_case("flat_box_left_hdf5", "box", (0.032, 0.017, 0.02), "left", (-0.08, 0.36, 0.0), -30.0, source="hdf5")
    run_case("tall_box_right_hamer", "box", (0.02, 0.024, 0.04), "right", (0.15, 0.44, 0.0), 40.0, source="hamer")
    # misuse
    bad = 0
    for kw in ({}, {"hands_npz": "a.npz", "hamer_npz": "b.npz"}, {"egodex_hdf5": "x.hdf5"}, {"hamer_npz": "b.npz"}):
        try:
            human_grasp_seeds(subgoals=[{"index": 1, "side": "right", "object": "box", "frames": [0, 5]}], out_dir=str(TMP / "bad"), **kw)
        except ValueError:
            bad += 1
    check("misuse raises ValueError (no source, two sources, hdf5 without episode_json, HaMeR without camera)", bad == 4, f"{bad}/4")
    print(f"{len(FAILS)} failed, {time.time() - t0:.0f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
