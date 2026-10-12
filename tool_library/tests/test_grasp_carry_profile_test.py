"""grasp_carry_profile_test: the object must stay put in the hand through the plan's own carry motion.

    MUJOCO_GL=egl python tool_library/tests/test_grasp_carry_profile_test.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grasp_robust_fixtures as F  # noqa: E402


def wrist_path(pose, disp, rotvec, seconds, fps=30):
    """World wrist poses from the grasp pose: a quintic displacement and a turn about world axes."""
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from tool_library.common import hand as H
    n = int(round(seconds * fps)) + 1
    p0, R0 = np.asarray(pose["pos"], float), H.euler_to_mat(pose["euler"])
    rows = []
    for i in range(n):
        u = i / (n - 1)
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        R = Rot.from_rotvec(np.asarray(rotvec, float) * s).as_matrix() @ R0
        rows.append([float(v) for v in p0 + np.asarray(disp, float) * s] + [float(v) for v in H.mat_to_euler(R)])
    return {"wrist": rows, "fps": fps}


def bench_scene(tmp):
    """A bench-like scene: the right hand under the bench prefix robot_right_ (its qpos is what trajectories hold)."""
    import mujoco
    hx = Path(F.hand_xml("right"))
    h = mujoco.MjSpec.from_file(str(hx))
    h.meshdir = str((hx.parent / (h.meshdir or "")).resolve())
    sp = mujoco.MjSpec.from_string('<mujoco model="bench"><worldbody/></mujoco>')
    sp.meshdir = h.meshdir
    sp.attach(h, prefix="robot_right_", frame=sp.worldbody.add_frame())
    model = sp.compile()
    out = Path(tmp) / "bench.xml"
    out.write_text(sp.to_xml())
    return out, model


def main():
    import numpy as np
    from tool_library.common import episode as E
    from tool_library.common import grasp_robust as GR
    from tool_library.common import plan as P
    from tool_library.verifiers.grasp_carry_profile_test import grasp_carry_profile_test as T
    rep = F.Report()
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="dexagent_carry_")
    box, pinch = F.scene("box", "right"), F.pose(F.BOX_PINCH3)
    weak = F.pose(F.BOX_PINCH)
    cyl, top = F.scene("cyl", "left"), F.pose(F.CYL_TOP)

    gentle = wrist_path(pinch, [0.15, 0.10, 0.05], [0.0, 0.0, np.radians(20)], 1.5)
    r = T(box, pinch, gentle)
    rep.check("three-finger pinch through a gentle carry", r["ok"] and r["drift_mm"] < 3 and r["path"]["turned_deg"] > 19, (r["drift_mm"], r["path"]))
    r = T(cyl, top, wrist_path(top, [-0.15, 0.10, 0.05], [0.0, 0.0, np.radians(-20)], 1.5))
    rep.check("left-hand top grasp through a gentle carry (wrist chain re-based)", r["ok"] and r["wrist_rebased"], (r["drift_mm"], r["wrist_rebased"]))
    r = T(box, pinch, wrist_path(pinch, [0.0, 0.0, 0.0], [np.radians(180), 0.0, 0.0], 1.2))
    rep.check("three-finger pinch survives turning the box upside down in 1.2 s", r["ok"], (r["drift_mm"], r["turn_deg"]))

    # The same turn in half the time exceeds the wrist's physical torque
    # budget. A stable grasp cannot make an unreproduced command pass.
    r = T(box, pinch, wrist_path(pinch, [0.0, 0.0, 0.0], [np.radians(180), 0.0, 0.0], 0.6))
    rep.check("a fast upside-down carry is rejected when the wrist cannot follow", (not r["ok"])
              and any("did not follow" in issue for issue in r["issues"]), (r["drift_mm"], r["issues"][:1]))
    # the weak two-pad pinch does not even carry the box once the table is gone: reported, every key still there
    r = T(box, weak, gentle)
    rep.check("a grasp that cannot lift is reported with a full report", (not r["ok"]) and "does not lift" in r["issues"][0]
              and r["drift_mm"] is None and "tracking" in r, r["issues"])
    # a carry no wrist can follow (0.3 m in a quarter second)
    r = T(box, pinch, wrist_path(pinch, [0.3, 0.0, 0.0], [0.0, 0.0, 0.0], 0.25))
    rep.check("a carry faster than the wrist servos is reported as not reproduced", (not r["ok"])
              and any("did not follow" in i for i in r["issues"]), r["tracking"])

    # the same gentle carry read from a trajectory npz (by label, through the joint names) and from a keyframes JSON
    bxml, bm = bench_scene(tmp)
    rig = P.scene_rigs(bm)["right"]
    q0 = np.zeros(bm.nq)
    for k, v in pinch["finger_targets"].items():
        if k in rig.finger_qpos:
            q0[rig.finger_qpos[k]] = v
    W = np.asarray(gentle["wrist"])
    Q, labels = [], []
    for i in range(10):
        q = q0.copy(); q[[rig.wrist_qpos[k] for k in ("x", "y", "z", "rx", "ry", "rz")]] = W[0]
        Q.append(q); labels.append("right: lift box_box")
    for w in W[1:]:
        q = q0.copy(); q[[rig.wrist_qpos[k] for k in ("x", "y", "z", "rx", "ry", "rz")]] = w
        Q.append(q); labels.append("right: carry box_box")
    Q.append(Q[-1].copy()); labels.append("right: set down box_box")
    npz = Path(tmp) / "trajectory.npz"
    np.savez_compressed(npz, qpos=np.asarray(Q), labels=np.array(labels), fps=30.0, joint_names=np.array(E.joint_names(bm)))
    r = T(box, pinch, str(npz))
    rep.check("trajectory npz: the carry run is found by its label", r["ok"] and r["path"]["frames"] == [9, 9 + len(W) - 1]
              and r["path"]["source"] == "trajectory", r["path"])
    kf = {"keyframes": [{"qpos": Q[0].tolist(), "transit": 0, "hold": 3, "label": "right: lift box_box"},
                        {"qpos": Q[len(W) // 2 + 9].tolist(), "transit": 22, "hold": 1, "label": "right: carry box_box (up)"},
                        {"qpos": Q[-1].tolist(), "transit": 22, "hold": 3, "label": "right: carry box_box"}]}
    kp = Path(tmp) / "keyframes.json"
    kp.write_text(json.dumps(kf))
    r = T(box, pinch, str(kp), bench_scene=str(bxml))
    rep.check("keyframes JSON on the bench scene replays the same carry", r["ok"] and r["path"]["source"] == "keyframes"
              and abs(r["path"]["turned_deg"] - 20.0) < 0.5, r["path"])

    rep.raises("an unknown carry source is refused", lambda: T(box, pinch, 42))
    rep.raises("frames outside the source are refused", lambda: T(box, pinch, str(npz), frames=[0, 10000]))
    rep.raises("a missing carry label is refused", lambda: T(box, pinch, str(npz), phase="pour"))
    print(f"done in {time.time() - t0:.0f} s")
    return rep.exit()


if __name__ == "__main__":
    sys.exit(main())
