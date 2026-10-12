"""human_grasp_pose_prior: a Sharpa pose retargeted from the demonstrated hand at the contact frame.

Fixture (tests/fixtures/human_grasp_pose_prior/lego59_left_pick.npz): the 50 hand joints of EgoDex
build_unstack_lego/59 (read_egodex_episode, cam_xy [-0.001, 0.095]) for frames 0-40, where the left hand picks up the
yellow brick, in hands.npz's format; brick.json is that brick's Stage 2 pose and size.
Run: MUJOCO_GL=egl python tool_library/tests/test_human_grasp_pose_prior.py
"""
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import hand as H  # noqa: E402
from tool_library.common import human_prior as HP  # noqa: E402
from tool_library.skills.human_grasp_pose_prior import human_grasp_pose_prior  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "human_grasp_pose_prior"
FAILS = []


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def robot_as_human(side, pos, euler, fingers, n=12, drift=(0.0, 0.0, 0.0)):
    """A hands.npz whose 'human' is the Sharpa hand itself at a known pose: the wrist is the root origin and each
    ARKit chain is the finger's mcp/pip/dip/tip sites."""
    import mujoco
    xml = ROOT / "setup" / "robot" / f"sharpa_{side}.xml"
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    rig = H.HandRig.build(m, "", side)
    names, frames = [], []
    for t in range(n):
        rig.teleport(d, np.asarray(pos) + np.asarray(drift) * (n - 1 - t), euler)
        rig.set_fingers(d, fingers)
        mujoco.mj_forward(m, d)
        row = {side + HP.ARKIT_WRIST: d.xpos[m.body(f"{side}_base_yaw").id].copy()}
        for f, chain in HP.ARKIT_CHAINS.items():
            for jn, k in zip(chain, HP.SHARPA_SITES):
                row[side + jn] = d.site_xpos[m.site(f"{side}_{f}_{k}").id].copy()
        names = sorted(row)
        frames.append([row[k] for k in names])
    tips = {f: d.site_xpos[m.site(f"{side}_{f}_tip").id].copy() for f in HP.FINGERS}
    return np.array(names), np.array(frames, np.float32), tips


def identity_case(side, d):
    rng = np.random.default_rng(3 if side == "right" else 4)
    pos, euler = np.array([0.1, 0.45, 0.12]), np.array([2.6, 0.3, -0.4])
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / "setup" / "robot" / f"sharpa_{side}.xml"))
    rig = H.HandRig.build(m, "", side)
    fingers = {}
    for k, (lo, hi) in rig.finger_range.items():
        role = H.joint_role(k)
        fingers[k] = 0.0 if k.endswith("pinky_CMC") else (lo + rng.uniform(0.15, 0.5) * (hi - lo) if role == "flex" else np.clip(rng.uniform(-0.15, 0.15), lo, hi))
    names, X, tips = robot_as_human(side, pos, euler, fingers, drift=(0.0, 0.004, 0.003))
    npz = Path(d) / f"ident_{side}.npz"
    np.savez(npz, names=names, rig_xyz=X)
    r = human_grasp_pose_prior(side, len(X) - 1, Path(d) / f"ident_{side}.json", hands_npz=npz)
    p = r["pose"]
    pe = 1000 * float(np.linalg.norm(np.asarray(p["pos"]) - pos))
    Ra, Rb = H.euler_to_mat(p["euler"]), H.euler_to_mat(euler)
    re = math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))
    te = max(1000 * float(np.linalg.norm(np.asarray(p["robot_tips"][f]) - tips[f])) for f in HP.FINGERS)
    sc = max(abs(v - 1.0) for v in p["finger_scale"].values())
    check(f"{side} hand retargeted onto itself: wrist, orientation and fingertips recovered",
          pe < 1.0 and re < 0.5 and te < 4.0 and sc < 1e-3 and Path(r["out_json"]).exists(),
          f"wrist {pe:.2f} mm, rotation {re:.2f} deg, worst fingertip {te:.1f} mm, scale off by {sc:.4f}")
    return p


def brick_scene(d):
    b = json.loads((FIX / "brick.json").read_text())
    xml = Path(d) / "brick.xml"
    hx, hy, hz = (0.5 * v for v in b["size"])
    xml.write_text(f'<mujoco model="brick"><worldbody><body name="brick"><freejoint/>'
                   f'<geom name="brick_body" type="box" size="{hx} {hy} {hz}" pos="0 0 {hz}" mass="0.02" rgba="0.8 0.7 0.1 1"/>'
                   f'</body></worldbody></mujoco>')
    return {"hand_xml": str(ROOT / "setup" / "robot" / "sharpa_left.xml"), "side": "left", "floor": True,
            "objects": [{"name": "brick", "xml": str(xml), "pos": [b["x"], b["y"], 0.0], "yaw_deg": b["yaw_deg"]}]}


def main():
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        identity_case("right", d)
        identity_case("left", d)
        # real EgoDex: the left hand picking the yellow brick
        meta = json.loads((FIX / "brick.json").read_text())
        fr = int(meta["contact_frame"])
        r = human_grasp_pose_prior("left", fr, Path(d) / "lego.json", hands_npz=FIX / "lego59_left_pick.npz")
        p = r["pose"]
        kp = HP.keypoints_from_egodex(FIX / "lego59_left_pick.npz", "left", [fr])
        Rh, ch = HP.anatomical_frame(HP.frame_at(kp, 0), "left")
        ref = HP.sharpa_reference(ROOT / "setup" / "robot" / "sharpa_left.xml", "left")
        R = H.euler_to_mat(p["euler"])
        knuckles = np.asarray(p["pos"]) + R @ ref["c_anat"]
        fdir = R @ ref["R_anat"][:, 2]
        ang = math.degrees(math.acos(np.clip(fdir @ Rh[:, 2], -1, 1)))
        worst = max(v for f, v in p["residual_mm"].items() if f != "thumb")
        thumb_ok = p["residual_mm"]["thumb"] < 4.0 or ("thumb" in p["unreachable_fingers"] and any("thumb" in w for w in p["warnings"]))
        check("EgoDex left pick: knuckles on the human's, palm along the human's, fingers solved (an unreachable thumb reported)",
              np.linalg.norm(knuckles - ch) < 1e-3 and ang < 0.5 and worst < 5.0 and thumb_ok and "thumb" in p["grip_fingers"]
              and len(p["finger_targets"]) == 22 and np.linalg.norm(p["approach"]) > 0.99,
              f"residual {p['residual_mm']} mm, grip {p['grip_fingers']}, approach from {p['approach_source']}")
        # the same with the brick as a hand scene: measured against the object and backed out of it
        sc = brick_scene(d)
        r2 = human_grasp_pose_prior("left", fr, Path(d) / "lego_scene.json", hands_npz=FIX / "lego59_left_pick.npz", hand_scene=sc)
        p2 = r2["pose"]
        tips = p2["tip_to_object_mm"]
        snap = p2.get("snap") or {}
        check("with the brick: lifted off the table, snapped onto the brick without entering it",
              p2["overlap_mm"] <= 4.0 and p2["overlap_mm"] < p2["overlap_mm_before_clear"] and tips["thumb"] is not None
              and snap.get("tip_to_object_mm", {}).get("thumb", 99.0) <= 5.0 and "prior_pos" in p2,
              f"overlap {p2['overlap_mm_before_clear']} -> {p2['overlap_mm']} mm (lifted {1000 * p2['lifted_m']:.0f} mm, backed out "
              f"{1000 * p2['cleared_m']:.0f} mm), pads before the snap {tips} mm, snap shift {snap.get('shift_m')} -> pads "
              f"{snap.get('tip_to_object_mm')} mm, grip {p2['grip_fingers']}; warnings {p2['warnings']}")
        # misuse
        bad = 0
        for kw in ({"side": "up", "frame": 3, "hands_npz": FIX / "lego59_left_pick.npz"},
                   {"side": "left", "frame": 3},
                   {"side": "left", "frame": 3, "hands_npz": FIX / "lego59_left_pick.npz", "hamer_npz": "x.npz"},
                   {"side": "left", "frame": 999, "hands_npz": FIX / "lego59_left_pick.npz"},
                   {"side": "right", "frame": fr, "hands_npz": FIX / "lego59_left_pick.npz", "hand_scene": sc}):
            try:
                human_grasp_pose_prior(**kw)
            except ValueError:
                bad += 1
        check("misuse raises ValueError (side, no source, two sources, frame out of range, scene of the other hand)", bad == 5, f"{bad}/5")
    print(f"{len(FAILS)} failed, {time.time() - t0:.0f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
