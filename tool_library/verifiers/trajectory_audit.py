"""Trajectory audit: every episode check, one call, one table.

    from tool_library.verifiers.trajectory_audit import audit_trajectory
    rep = audit_trajectory("scene.xml", "trajectory.npz",
                           end_effectors={"right": "robot_right_right_base_yaw",
                                          "left": "robot_left_left_base_yaw"},
                           manipulated=["cup*"], reference="verified.npz")
    print(rep["table"])

    python -m tool_library.verifiers.trajectory_audit scene.xml trajectory.npz \\
        --ee right=robot_right_right_base_yaw left=robot_left_left_base_yaw

This is the audit every delivered episode of the paper went through before
it was rendered or deployed: smoothness (joint speed, arm jerk), assembly
(weld gaps), deployability (wrist stops and spins), scene integrity (object
disturbance, penetration on all visible geometry) and timeline coverage.
Each part is its own verifier in this library; the audit runs the ones that
apply to the scene and prints a pass/fail table with the deciding numbers.
"""
from __future__ import annotations

TOOL = {
    "name": "trajectory_audit",
    "title": "Trajectory audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Runs the smoothness, weld, wrist, disturbance, penetration and coverage checks on an episode and returns one pass/fail table.",
    "description": (
        "Bundles joint_speed_check, arm_jerk_check, weld_gap_check, wrist_stop_check, "
        "wrist_spin_check, object_disturbance_check, penetration_check and "
        "uncovered_frames_check. Robot structure is discovered from the end-effector bodies "
        "(the arm bolted to each is found through its equality constraint); checks whose "
        "preconditions the scene does not meet (no arm, no wrist hinge by name, no welds) "
        "are reported as skipped, not failed; a trajectory whose qpos width is not the scene's nq fails (a 'qpos_width' "
        "row) instead of letting every check skip. With a reference trajectory the speed, "
        "disturbance and penetration checks become relative to it, which is how an adapted "
        "or re-fitted episode is held to the verified one. The report keeps every "
        "sub-report, a list of the failing checks and a plain-text table."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF scene (.xml, needed for the all-geometry penetration check)."},
        {"name": "trajectory", "type": "str | ndarray [T, nq]", "description": "trajectory.npz (qpos, labels, fps)."},
        {"name": "end_effectors", "type": "dict side -> body name | None", "description": "Hand root per side. Default: the setup's stations (setup.yaml hand_root_body)."},
        {"name": "manipulated", "type": "list[str]", "description": "Objects the task may move (fnmatch)."},
        {"name": "reference", "type": "str | None", "description": "Verified trajectory the episode is derived from."},
        {"name": "wrist_joints", "type": "dict name -> (lo, hi) deg | None", "description": "Wrist joints with their real stops; default found by name."},
        {"name": "real_stops", "type": "dict side -> (lo, hi) deg | None", "description": "Real wrist range per side, applied to the default joints by name."},
        {"name": "video_frames, video_fps", "type": "int, float | None", "description": "For the coverage check."},
        {"name": "skip", "type": "list[str]", "description": "Check names to leave out."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; failed: names; skipped: names with reasons; checks: every sub-report; table: text."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["audit", "episode", "smoothness", "deployability", "bundle"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "trajectory_audit",
    "cli": "python -m tool_library.verifiers.trajectory_audit scene.xml trajectory.npz --ee right=<hand body> left=<hand body>",
}

import json
import sys
import traceback
from pathlib import Path

import mujoco

from tool_library.common import trajectory as TJ
from tool_library.verifiers.arm_jerk_check import check_arm_jerk
from tool_library.verifiers.joint_speed_check import check_joint_speed
from tool_library.verifiers.object_disturbance_check import check_object_disturbance
from tool_library.verifiers.penetration_check import check_penetration
from tool_library.verifiers.uncovered_frames_check import check_uncovered_frames
from tool_library.verifiers.weld_gap_check import check_weld_gap
from tool_library.verifiers.wrist_spin_check import check_wrist_spin
from tool_library.verifiers.wrist_stop_check import check_wrist_stop


def _headline(name, r):
    """One line of numbers per check for the table."""
    if name in ("joint_speed", "arm_jerk"):
        parts = [f"{k} {v['max']}/{v['limit']:.3g} {v.get('unit', '')}" for k, v in r.items()
                 if isinstance(v, dict) and "max" in v]
        return "; ".join(parts)
    if name == "weld_gap":
        return "; ".join(f"{'/'.join(v['bodies'])} {v['gap_mm']['max']} mm" for k, v in r.items() if k.startswith("eq"))
    if name == "wrist_stop":
        return "; ".join(f"{k} [{v['min']}, {v['max']}] in {v['range']} ({v['range_source']})" for k, v in r.items() if isinstance(v, dict))
    if name == "wrist_spin":
        return "; ".join(f"{k} turn {v['largest_turn_deg']} deg, {len(v['redundant_reversals'])} redundant" for k, v in r.items() if isinstance(v, dict))
    if name == "object_disturbance":
        return f"disturbed {r.get('disturbed', [])}, jumped {r.get('jumped', [])}"
    if name == "penetration":
        w = r.get("worst")
        return (f"worst {w['depth_m'] * 1000:.1f} mm {w['geoms']} @ {w['frame']}" if w else "no penetration") + \
               f"; {r.get('n_pairs_touching', 0)} pairs touching"
    if name == "coverage":
        return f"{r['frames']} frames, {r['unlabeled_frames']} unlabeled, {len(r['short_phases'])} short phases" + \
               (f", covers {r['covered_frac']:.3f} of video" if "covered_frac" in r else "")
    return ""


def _setup_end_effectors(model):
    """{side: hand_root_body} of the default setup's stations that exist in the model."""
    import yaml
    sy = yaml.safe_load((Path(__file__).resolve().parents[2] / "setup" / "setup.yaml").read_text())
    out = {}
    for side, st in (sy.get("stations") or {}).items():
        b = (st or {}).get("hand_root_body")
        if b and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) >= 0:
            out[side] = b
    return out


def audit_trajectory(scene, trajectory, end_effectors=None, manipulated=(), reference=None, wrist_joints=None,
                     real_stops=None, video_frames=None, video_fps=None, skip=(), robot_prefix="robot_"):
    model = TJ.load_model(scene)
    if not end_effectors:
        end_effectors = _setup_end_effectors(model)
        if not end_effectors:
            raise ValueError("end_effectors: give {side: hand root body}; the setup's stations are not in this scene")
    sides = TJ.discover_sides(model, end_effectors)
    limbs = [[model.body(b).name for b in grp] for grp in TJ.limb_bodies(model, sides)]
    Q, labels, fps = TJ.as_trajectory(trajectory)
    traj = (Q, labels, fps)
    ref = TJ.as_trajectory(reference)[0] if reference is not None else None
    checks, skipped = {}, {}
    if Q.ndim != 2 or Q.shape[1] != model.nq:      # every check needs it: a failure, not a precondition to skip
        checks["qpos_width"] = {"ok": False, "error": f"trajectory qpos has {Q.shape[-1]} columns, the scene has nq={model.nq}"}

    def run(name, fn, *a, **k):
        if name in skip:
            skipped[name] = "skipped by request"
            return
        try:
            checks[name] = fn(*a, **k)
        except ValueError as e:                    # a precondition the scene does not meet
            skipped[name] = str(e)
        except Exception as e:                     # a real failure of the check itself
            checks[name] = {"ok": False, "error": repr(e), "trace": traceback.format_exc(limit=3)}

    run("joint_speed", check_joint_speed, model, traj, end_effectors=end_effectors, reference=ref)
    run("arm_jerk", check_arm_jerk, model, traj, end_effectors=end_effectors)
    if model.neq:
        # the robot's own attachments (an arm bolted to its hand); a mechanism's weld (a cap held on its thread) is
        # switched off at run time by its rule, which a qpos trajectory does not record
        run("weld_gap", check_weld_gap, model, traj, only=[f"{robot_prefix}*"])
    else:
        skipped["weld_gap"] = "no equality constraints"
    run("wrist_stop", check_wrist_stop, model, traj, joints=wrist_joints, real_stops=real_stops)
    run("wrist_spin", check_wrist_spin, model, traj, joints=list(wrist_joints) if wrist_joints else None)
    run("object_disturbance", check_object_disturbance, model, traj, manipulated=manipulated, reference=ref,
        robot_prefix=robot_prefix)
    run("penetration", check_penetration, scene, Q, baseline_qpos=ref, limbs=limbs, stride=1)
    run("coverage", check_uncovered_frames, traj, video_frames=video_frames, video_fps=video_fps)

    failed = [n for n, r in checks.items() if not r.get("ok")]
    rows = [f"{'check':<20} {'result':<6} detail"]
    for n, r in checks.items():
        rows.append(f"{n:<20} {'PASS' if r.get('ok') else 'FAIL':<6} {r.get('error') or _headline(n, r)}")
    for n, why in skipped.items():
        rows.append(f"{n:<20} {'skip':<6} {why}")
    return {"ok": not failed, "failed": failed, "skipped": skipped, "checks": checks,
            "frames": len(Q), "fps": fps, "table": "\n".join(rows)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("scene"); ap.add_argument("trajectory")
    ap.add_argument("--ee", nargs="+", required=True, help="side=hand root body")
    ap.add_argument("--manipulated", nargs="*", default=[])
    ap.add_argument("--reference")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args()
    rep = audit_trajectory(a.scene, a.trajectory, dict(kv.split("=", 1) for kv in a.ee),
                           manipulated=a.manipulated, reference=a.reference)
    print(rep["table"])
    if a.json:
        open(a.json, "w").write(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if rep["ok"] else 1)


trajectory_audit = audit_trajectory   # the library calls every tool by its own name
