"""Start state audit: does the episode start from the scene's keyframe, and is that a state the scene holds?

    from tool_library.verifiers.start_state_audit import start_state_audit
    start_state_audit("scene.xml", "trajectory.npz")

Stage 4 randomizes and replays from the scene's keyframe; the episode was executed from its
own frame 0. If they differ, the replay starts somewhere the trajectory never was. And if that
state is not at rest (an object hovering a few millimetres up, tilted on a lid it does not
rest on, interpenetrating its neighbour), the first second of every replay is the scene
settling, not the task.
"""
from __future__ import annotations

TOOL = {
    "name": "start_state_audit",
    "title": "Start state audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Frame 0 equals the scene's keyframe (robot, objects and mechanisms), and simulating the scene from frame 0 "
               "with the robot held still leaves every object where it is.",
    "description": (
        "No other audit compares the trajectory with the scene's keyframe or simulates the start. (1) The scene must "
        "have a keyframe (index `key`, default 0). (2) Frame 0 must equal it: robot joints within robot_tol (default "
        "0.01 rad or m), free objects within pos_tol_m (default 3 mm) and rot_tol_deg (default 2 deg), mechanism joints "
        "within robot_tol. (3) Settled: the scene as the bench executor runs it (hand-object pairs) is simulated for "
        "settle_s (default 1 s) from frame 0 with every robot joint held where it is (the hands' servos hold their "
        "start pose), and again from the keyframe when it differs from frame 0 (from qpos0 when the scene has no "
        "keyframe: that is where a replay would start); each free object's peak displacement and rotation must stay "
        "under settle_m (default 3 mm) and settle_deg (default 2 deg). An object that drops, slides or tips here starts the episode from a state the "
        "scene does not hold: re-place it (rest it on its support) and write the keyframe from the settled state. "
        "Reports the largest frame-0 differences and each object's settle motion."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Window whose first frame is judged as the start (the keyframe comparison applies when it is frame 0). Default all."},
        {"name": "key", "type": "int | str", "description": "Keyframe index or name. Default 0."},
        {"name": "robot_tol", "type": "float", "description": "Allowed robot / mechanism joint difference (rad or m; an arm re-fitted to the executed hand differs from the keyframe by a few thousandths). Default 0.01."},
        {"name": "pos_tol_m, rot_tol_deg", "type": "float", "description": "Allowed object pose difference. Defaults 0.003, 2."},
        {"name": "settle_s", "type": "float", "description": "Simulated time. Default 1.0."},
        {"name": "settle_m, settle_deg", "type": "float", "description": "Allowed peak object motion while settling. Defaults 0.003, 2."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst; keyframe {name, robot_max_diff, objects {name: diff mm, deg}, mechanisms}; settle {name: peak moved mm, rotated deg, final}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "keyframe", "initial state", "settling", "stage4"],
    "status": "tested",
    "entrypoint": "start_state_audit",
}

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA


def start_state_audit(scene, trajectory, frames=None, key=0, robot_tol=0.01, pos_tol_m=0.003, rot_tol_deg=2.0, settle_s=1.0,
                      settle_m=0.003, settle_deg=2.0):
    EA.check_positive(robot_tol=robot_tol, pos_tol_m=pos_tol_m, rot_tol_deg=rot_tol_deg, settle_s=settle_s, settle_m=settle_m,
                      settle_deg=settle_deg)
    ep = EA.Episode(scene, trajectory, frames)
    m = ep.model
    t0 = ep.a                                      # the start judged: frame 0, or the window's first frame
    issues, worst = [], None
    kf = None
    ref, ref_name = None, None
    if t0 == 0:
        if m.nkey == 0:
            issues.append("the scene has no keyframe: Stage 4 cannot start a replay from the episode's state (it would start from "
                          "the scene's qpos0); write frame 0 (after it settles) into the scene as keyframe 'home'")
            ref, ref_name = np.asarray(m.qpos0, float), "qpos0 (no keyframe)"
        else:
            if isinstance(key, str):
                k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
                if k < 0:
                    raise ValueError(f"key: the scene has no keyframe named {key!r}")
            else:
                k = int(key)
                if not 0 <= k < m.nkey:
                    raise ValueError(f"key: index {k} outside the scene's {m.nkey} keyframes")
            ref, ref_name = np.asarray(m.key_qpos[k], float), f"keyframe {m.key(k).name or k!r}"
        kq = ref
        q0 = ep.Q[t0]
        rq, _ = EA.robot_dofs(ep)
        rd = np.abs(q0[rq] - kq[rq]) if len(rq) else np.zeros(1)
        jn = EA.joint_names_by_qadr(m)
        kf = {"name": ref_name, "robot_max_diff": round(float(rd.max()), 5),
              "robot_worst_joint": jn.get(int(rq[int(np.argmax(rd))])) if len(rq) else None, "objects": {}, "mechanisms": {}}
        if m.nkey and rd.max() > robot_tol:
            issues.append(f"frame 0 puts the robot {rd.max():.4f} away from {ref_name} at {kf['robot_worst_joint']}: start the "
                          f"episode from the keyframe (or rewrite the keyframe from frame 0)")
        for n, o in ep.objects.items():
            a = o["qadr"]
            dp = float(np.linalg.norm(q0[a:a + 3] - kq[a:a + 3]))
            da = float(np.degrees(EA.quat_angle(q0[a + 3:a + 7], kq[a + 3:a + 7])))
            kf["objects"][n] = {"diff_mm": round(1000 * dp, 2), "diff_deg": round(da, 2)}
            if m.nkey and (dp > pos_tol_m or da > rot_tol_deg):
                issues.append(f"{n} starts {1000 * dp:.1f} mm / {da:.1f} deg away from its pose in {ref_name}: the episode and the "
                              f"scene disagree on where it is (a constraint or the settling moved it); rewrite the keyframe from frame 0")
        for jn_, (j, qa, jt) in ep.mech_joints.items():
            dd = abs(float(q0[qa] - kq[qa]))
            kf["mechanisms"][jn_] = round(dd, 5)
            if m.nkey and dd > robot_tol:
                issues.append(f"mechanism joint {jn_} starts {dd:.4f} away from {ref_name}: the keyframe is not the state the "
                              f"scene settles to; rewrite the keyframe from frame 0")
    settle = {}
    if ep.objects:
        starts = [("frame 0" if t0 == 0 else f"frame {t0}", ep.Q[t0])]
        if ref is not None and np.max(np.abs(ref - ep.Q[t0])) > 1e-6:
            starts.append((ref_name, ref))             # the state a replay starts from must hold too
        for sname, qs in starts:
            res, _, bad = EA.settle(ep, qs, seconds=settle_s)
            for n, r in res.items():
                settle.setdefault(sname, {})[n] = {"peak_moved_mm": EA.r3(1000 * r["peak_moved_m"], 2),
                                                   "peak_rotated_deg": EA.r3(r["peak_rotated_deg"], 2), "moved_mm": EA.r3(1000 * r["moved_m"], 2)}
                if bad is not None or r["peak_moved_m"] > settle_m or r["peak_rotated_deg"] > settle_deg:
                    issues.append(f"{n} does not hold its pose in {sname}: simulated {settle_s:g} s from it with the robot still it moves "
                                  f"{1000 * r['peak_moved_m']:.1f} mm / {r['peak_rotated_deg']:.1f} deg (from {np.round(r['start'], 3).tolist()} to "
                                  f"{np.round(r['end'], 3).tolist()}): rest it on its support (drop it to contact, level it) and write the "
                                  f"keyframe from the settled state")
                    if worst is None or r["peak_moved_m"] > worst.get("peak_moved_m", 0):
                        worst = {"body": n, "start": sname, "peak_moved_m": round(r["peak_moved_m"], 4),
                                 "peak_rotated_deg": round(r["peak_rotated_deg"], 2)}
            if bad is not None:
                issues.append(f"the simulation from {sname} became unstable after {bad} steps")
    if worst is None and issues:
        worst = {"issue": issues[0]}
    return EA.finish(issues, worst, keyframe=kf, settle=settle, settle_s=settle_s, start_frame=t0)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(start_state_audit(sys.argv[1], sys.argv[2]), indent=1))
