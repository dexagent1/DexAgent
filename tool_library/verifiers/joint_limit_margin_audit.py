"""Joint limit margin audit: every robot joint inside the REAL robot's limits, not the scene's.

    from tool_library.verifiers.joint_limit_margin_audit import joint_limit_margin_audit
    joint_limit_margin_audit("scene.xml", "trajectory.npz")                # reference: the repo's setup/scene.xml

A scene can widen a joint range so that IK or a physics replay never gets stuck; the episode
then passes every check that reads limits from the scene, and the real arm or hand cannot
execute it. This audit reads the limits of a reference robot (the setup the robot really is)
and matches joints by name.
"""
from __future__ import annotations

TOOL = {
    "name": "joint_limit_margin_audit",
    "title": "Joint limit margin audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Every robot joint of the episode stays inside the real limits of a reference robot (default: the "
               "repository's setup/scene.xml, matched by joint-name suffix and side), and the scene does not widen any range.",
    "description": (
        "Covers what wrist_stop_check (wrist roll joints only, scene limits unless real stops are passed) and "
        "arm_reach_check (margin to the SCENE's ranges, reported but not failed) miss. Every hinge and slide of the "
        "scene is matched to a joint of the reference robot: same name, else the same side word (left / right) and "
        "the longest common run of trailing name tokens (at least two, e.g. 'dof_joint2', 'index_MCP_FE'), so a twin "
        "whose arm is named 'arm_left_dof_joint2' still meets 'robot_left_arm_dof_joint2'. The hand's virtual wrist "
        "joints (pos_x..rot_z, the hand's world pose, no hardware) are skipped by default. Two failures: (1) the "
        "trajectory leaves the reference range by more than the tolerance on any frame (0.5 deg for arm joints, which "
        "are driven kinematically and should never pass a stop; 1.5 deg for finger joints, which MuJoCo's soft joint "
        "limits let a contact press about a degree past the stop; 0.5 mm for slides); (2) the scene's own range is wider than the reference's by more than the tolerance, or the scene "
        "joint is unlimited where the reference is limited: the scene was widened, and every limit-reading check is "
        "blind to it. The setup's arm joints 2 and 3 start at -0.15 rad on purpose (setup/README.md); they are judged "
        "against that. Reports per joint the reference and scene ranges, the extreme values and frames."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "reference", "type": "str | None", "description": "MJCF of the real robot whose limits rule. Default the repository's setup/scene.xml."},
        {"name": "tol_deg, tol_mm", "type": "float", "description": "Allowed excess over a reference limit for arm (and other) hinges and for slides. Defaults 0.5 deg, 0.5 mm."},
        {"name": "finger_tol_deg", "type": "float", "description": "Allowed excess for hand joints: a contact-loaded finger sits up to about 1 deg past its stop under MuJoCo's soft joint limits. Default 1.5."},
        {"name": "skip", "type": "list[str]", "description": "fnmatch patterns of joints left out. Default the virtual wrist joints ['*_pos_x', '*_pos_y', '*_pos_z', '*_rot_x', '*_rot_y', '*_rot_z']."},
        {"name": "check_scene_ranges", "type": "bool", "description": "Also fail a scene range wider than the reference. Default True."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {joint, frame, label, value, limit, excess}; joints {scene joint: reference joint, reference range, scene range, min, max, excess}; widened [joints]; unmatched [robot joints with no reference]; notes (the unmatched joints were not audited)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "joint limits", "deployability", "real robot"],
    "status": "tested",
    "entrypoint": "joint_limit_margin_audit",
}

import fnmatch

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA

VIRTUAL_WRIST = ["*_pos_x", "*_pos_y", "*_pos_z", "*_rot_x", "*_rot_y", "*_rot_z"]


def _tokens(name):
    return [t for t in name.lower().split("_") if t]


def _side(tokens):
    return "left" if "left" in tokens else ("right" if "right" in tokens else None)


def _match(name, ref_names):
    if name in ref_names:
        return name
    tk = _tokens(name)
    side = _side(tk)
    best, best_n = None, 1
    for r in ref_names:
        rt = _tokens(r)
        if _side(rt) != side:
            continue
        n = 0
        while n < min(len(tk), len(rt)) and tk[-1 - n] == rt[-1 - n]:
            n += 1
        if n > best_n:
            best, best_n = r, n
    return best


def joint_limit_margin_audit(scene, trajectory, frames=None, reference=None, tol_deg=0.5, finger_tol_deg=1.5, tol_mm=0.5,
                             skip=None, check_scene_ranges=True):
    EA.check_positive(tol_deg=tol_deg, finger_tol_deg=finger_tol_deg, tol_mm=tol_mm)
    ep = EA.Episode(scene, trajectory, frames)
    ref_path = reference or EA.setup_scene_path()
    if ref_path is None:
        raise ValueError("reference: no setup/scene.xml next to tool_library; pass the real robot's MJCF")
    try:
        rm = mujoco.MjModel.from_xml_path(str(ref_path))
    except Exception as e:
        raise ValueError(f"reference: cannot load {ref_path} ({e})")
    skip = VIRTUAL_WRIST if skip is None else list(skip)
    ref = {}
    for j in range(rm.njnt):
        if int(rm.jnt_type[j]) in (EA.HINGE, EA.SLIDE) and rm.jnt_limited[j]:
            ref[rm.joint(j).name] = (int(rm.jnt_type[j]), float(rm.jnt_range[j][0]), float(rm.jnt_range[j][1]))
    m = ep.model
    rows, viol, widened, unmatched = {}, [], [], []
    seg = ep.Q[ep.a:ep.b + 1]
    for j in range(m.njnt):
        jt = int(m.jnt_type[j])
        if jt not in (EA.HINGE, EA.SLIDE):
            continue
        name = m.joint(j).name or ""
        if any(fnmatch.fnmatchcase(name, p) for p in skip):
            continue
        robot = (m.body(int(m.jnt_bodyid[j])).name or "").startswith(ep.robot_prefix)
        r = _match(name, list(ref)) if name else None
        if r is None or ref[r][0] != jt:
            if robot:
                unmatched.append(name)
            continue
        _, lo, hi = ref[r]
        hinge = jt == EA.HINGE
        k = np.degrees(1.0) if hinge else 1000.0
        arm = "_arm_" in name or "_arm_" in r or not robot
        tol = ((tol_deg if arm else finger_tol_deg) if hinge else tol_mm) / k
        unit = "deg" if hinge else "mm"
        x = seg[:, int(m.jnt_qposadr[j])]
        i_lo, i_hi = int(np.argmin(x)), int(np.argmax(x))
        ex_lo, ex_hi = lo - x[i_lo], x[i_hi] - hi
        sl, sh = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) if m.jnt_limited[j] else (-np.inf, np.inf)
        row = {"reference_joint": r, "unit": unit, "reference_range": [round(lo * k, 3), round(hi * k, 3)],
               "scene_range": [EA.r3(sl * k), EA.r3(sh * k)] if m.jnt_limited[j] else None,
               "min": round(float(x[i_lo]) * k, 3), "max": round(float(x[i_hi]) * k, 3),
               "margin_low": round(float(-ex_lo) * k, 3), "margin_high": round(float(-ex_hi) * k, 3)}
        rows[name] = row
        for ex, i, lim, which in ((ex_lo, i_lo, lo, "lower"), (ex_hi, i_hi, hi, "upper")):
            if ex > tol:
                t = ep.a + i
                viol.append({"joint": name, "reference_joint": r, "frame": t, "label": ep.labels[t], "value": round(float(x[i]) * k, 3),
                             "limit": round(lim * k, 3), "which": which, "excess": round(float(ex) * k, 3), "unit": unit})
        if check_scene_ranges and (sl < lo - tol or sh > hi + tol):
            widened.append({"joint": name, "reference_joint": r, "scene_range": row["scene_range"],
                            "reference_range": row["reference_range"], "unit": unit})
    issues = []
    for v in sorted(viol, key=lambda v: -v["excess"])[:8]:
        issues.append(f"{v['joint']} reaches {v['value']} {v['unit']} at frame {v['frame']} ({v['label']}), past the real "
                      f"{v['which']} limit {v['limit']} {v['unit']} of {v['reference_joint']} by {v['excess']} {v['unit']}; "
                      f"re-plan that stretch with the real range (re-solve the arm with the reference limits or choose another grasp)")
    if len(viol) > 8:
        issues.append(f"... {len(viol) - 8} more joints past their real limits")
    for w in widened[:8]:
        issues.append(f"the scene widened {w['joint']} to {w['scene_range']} {w['unit']} where the real robot allows "
                      f"{w['reference_range']} {w['unit']}: restore the reference range in the scene, then re-execute so "
                      f"every check reads the real limit")
    if len(widened) > 8:
        issues.append(f"... {len(widened) - 8} more widened ranges")
    worst = max(viol, key=lambda v: v["excess"]) if viol else (widened[0] if widened else None)
    notes = []
    if unmatched:
        notes.append(f"{len(unmatched)} robot joint(s) have no counterpart in the reference and were not audited "
                     f"({unmatched[:6]}{' ...' if len(unmatched) > 6 else ''}): name them like the reference robot's joints "
                     f"or pass the reference that has them")
    return EA.finish(issues, worst, joints=rows, violations=viol[:30], widened=[w["joint"] for w in widened],
                     unmatched=unmatched, notes=notes, reference=str(ref_path), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(joint_limit_margin_audit(sys.argv[1], sys.argv[2]), indent=1))
