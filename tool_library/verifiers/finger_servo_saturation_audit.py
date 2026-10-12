"""Finger servo saturation audit: does the grasp hold only with the finger motors at their torque cap?

    from tool_library.verifiers.finger_servo_saturation_audit import finger_servo_saturation_audit
    finger_servo_saturation_audit("scene.xml", "trajectory.npz")

The executor drives finger servos with a torque cap (0.8 N m on the bench hand) and presses the
fingers past the surface, so in simulation every grip sits at its cap whether it needs to or
not. What matters is whether it NEEDS to: a grasp whose contacts can only carry the object when
the fingers push at full torque has no reserve for a real object's weight error, a faster carry
or a nudge, and a real hand at stall for seconds overheats. This audit computes the smallest
finger torque the grasp's contacts need, frame by frame.
"""
from __future__ import annotations

TOOL = {
    "name": "finger_servo_saturation_audit",
    "title": "Finger servo saturation audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "During carry phases, the smallest finger torques that let the grasp's actual contacts carry the object "
               "(gravity and the measured acceleration) stay below the finger servos' torque cap for most of the hold.",
    "description": (
        "contact_force_check sums the contact force in a window; it cannot tell a grip that needs its full torque from "
        "one with reserve, and the executor's position servos sit at the cap either way. For each carry window (lift, "
        "carry, transport phases of a hand, object named in the label or touching the hand) and each judged frame "
        "(every `stride`-th; frames where the object also touches a support are skipped), the contacts between the "
        "robot and the object within margin_m (the scene as the bench executor runs it) each get a friction pyramid "
        "(mu from the contact, capped at mu_cap) plus torsional and rolling friction bounded by the contact's own "
        "coefficients times its normal force: the physics the episode was executed in (spin_m / roll_m override them; "
        "spin_m=0.005, roll_m=0 judges against a realistic fingertip pad). A linear program finds contact forces that balance the object's "
        "weight and inertial force (mass x (acceleration - gravity)) and zero net torque about its centre of mass "
        "while minimising the largest finger-joint torque as a fraction s of each joint's cap (torque = J^T f through "
        "the finger's contact Jacobian; palm contacts load the arm, not a finger). Caps: the actuator's forcerange "
        "when the scene limits it, else finger_cap_nm (default: the bench hand's 0.8 N m that the executor applies at "
        "run time), times the actuator gear. A frame whose contacts cannot balance the load at all (residual above "
        "20% of the weight) is reported, not judged. It fails when s >= sat_frac (default 0.95) on at least hold_frac "
        "(default 0.6) of a window's judged frames. Reports per window the median and peak s and the joint that "
        "limits. The hand-scene grasp stress tests judge the same measure with the same thresholds (hand.ServoReserve), so a "
        "grasp they pass is not failed here for holding only at the torque cap."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "finger_cap_nm", "type": "float | None", "description": "Finger joint torque cap when the scene sets none. Default the bench hand's 0.8 N m."},
        {"name": "sat_frac", "type": "float", "description": "Share of the cap that counts as saturated. Default 0.95."},
        {"name": "hold_frac", "type": "float", "description": "Share of judged frames that must be saturated to fail. Default 0.6."},
        {"name": "margin_m", "type": "float", "description": "Contact distance that counts. Default 0.001."},
        {"name": "mu_cap", "type": "float", "description": "Largest friction coefficient credited. Default 1.5."},
        {"name": "spin_m, roll_m", "type": "float | None", "description": "Torsional / rolling friction length credited per contact (torque <= value x normal force). Default None: the contact's own coefficients."},
        {"name": "stride", "type": "int", "description": "Judge every n-th frame of a window. Default 2."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {side, object, frames, saturated_frac, median_s, peak_s, limiting_joint}; windows; unbalanced frames."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "grasp", "torque", "servo", "saturation", "reserve"],
    "status": "tested",
    "entrypoint": "finger_servo_saturation_audit",
}

from collections import Counter

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA
from tool_library.common.hand import FINGER_FORCE_NM


def _caps(ep, m, finger_cap_nm):
    """{dof address: cap N m} for every finger joint of every hand."""
    caps = {}
    for s, h in ep.hands.items():
        for j in h["finger_joints"]:
            a = h["finger_act"].get(j)
            gear = abs(float(m.actuator_gear[a][0])) if a is not None else 1.0
            if a is not None and m.actuator_forcelimited[a]:
                cap = max(abs(float(m.actuator_forcerange[a][0])), abs(float(m.actuator_forcerange[a][1])))
            elif finger_cap_nm is not None:
                cap = float(finger_cap_nm)
            elif h.get("bench"):
                cap = float(FINGER_FORCE_NM)
            else:
                continue
            caps[int(m.jnt_dofadr[j])] = (cap * gear, m.joint(j).name)
    return caps


def _solve(contacts, com, F_req, dofs, capv, mu_cap):
    """min s s.t. contact forces in friction pyramids (plus torsional and rolling friction bounded by the normal force)
    balance (F_req, 0 torque) up to a heavily penalised slack and |J^T f| <= s cap per finger dof: hand.finger_torque_need,
    which the hand-scene grasp stress tests use too, so a grasp they pass is judged the same way here.
    Returns (s, residual N, limiting dof index) or None."""
    from tool_library.common.hand import finger_torque_need
    return finger_torque_need(contacts, com, F_req, dofs, capv, mu_cap)


def finger_servo_saturation_audit(scene, trajectory, frames=None, finger_cap_nm=None, sat_frac=0.95, hold_frac=0.6,
                                  margin_m=0.001, mu_cap=1.5, spin_m=None, roll_m=None, stride=2):
    EA.check_positive(sat_frac=sat_frac, hold_frac=hold_frac, margin_m=margin_m, mu_cap=mu_cap, stride=stride)
    for nm, v in (("spin_m", spin_m), ("roll_m", roll_m)):
        if v is not None and float(v) < 0:
            raise ValueError(f"{nm} must be >= 0 or None")
    if finger_cap_nm is not None:
        EA.check_positive(finger_cap_nm=finger_cap_nm)
    ep = EA.Episode(scene, trajectory, frames)
    if not ep.hands or not ep.objects:
        return EA.finish([], None, windows=[], note="no robot hand or no free object")
    wins = EA.held_windows(ep, words=EA.CARRY_WORDS, merge=True)
    if not wins:
        return EA.finish([], None, windows=[], note="no carry phase naming (or touching) a free object")
    scan = EA.ContactScan(ep, margin=margin_m)
    m, d = scan.m, scan.d
    caps = _caps(ep, m, finger_cap_nm)
    if not caps:
        return EA.finish([], None, windows=[], note="no finger actuator with a torque cap (pass finger_cap_nm)")
    dofs = sorted(caps)
    capv = np.array([caps[k][0] for k in dofs])
    dof_index = {k: i for i, k in enumerate(dofs)}
    jp = np.zeros((3, m.nv))
    g = ep.gravity
    rows, unbalanced = [], []
    for side, ob, a, b in wins:
        o = ep.objects[ob]
        M = o["mass"]
        if M <= 0:
            continue
        P = ep.com_track(o["bodies"])
        svals, lim_j = [], []
        for t in range(max(a, 1), min(b, ep.T - 2) + 1, int(stride)):
            cs = scan.at(t, forces=False)
            mujoco.mj_comPos(m, d)
            contacts, support = [], False
            for b1, b2, dist, g1, g2, i in cs:
                in1, in2 = b1 in o["bodies"], b2 in o["bodies"]
                if in1 == in2:
                    continue
                other = b2 if in1 else b1
                if other not in ep.robot_bodies:
                    support = True
                    break
                c = d.contact[i]
                n_into = (1.0 if in2 else -1.0) * np.asarray(c.frame[:3], float)
                mujoco.mj_jac(m, d, jp, None, np.asarray(c.pos, float), int(other))
                J = np.zeros((3, len(dofs)))
                for k, col in dof_index.items():
                    J[:, col] = -jp[:, k]                 # the finger pushes the object: its joints feel the reaction
                fr = np.asarray(c.friction, float)
                spin = float(spin_m) if spin_m is not None else (float(fr[2]) if c.dim >= 4 else 0.0)
                roll = float(roll_m) if roll_m is not None else (float(fr[3]) if c.dim >= 6 else 0.0)
                contacts.append((np.asarray(c.pos, float), n_into, float(fr[0]) if c.dim > 1 else 0.0, spin, roll, J))
            if support or not contacts:
                continue
            acc = (P[t + 1] - 2 * P[t] + P[t - 1]) * ep.fps ** 2
            F_req = M * (acc - g)
            sol = _solve(contacts, P[t], F_req, dofs, capv, mu_cap)
            if sol is None:
                continue
            s, resid, jl = sol
            if resid > 0.2 * M * np.linalg.norm(g):
                unbalanced.append({"side": side, "object": ob, "frame": t, "label": ep.labels[t], "residual_n": round(resid, 3)})
                continue
            svals.append(s)
            lim_j.append(caps[dofs[jl]][1] if jl >= 0 else None)
        if not svals:
            continue
        sv = np.array(svals)
        sat = sv >= sat_frac
        k = int(np.argmax(sv))
        lj =Counter([j for j, x in zip(lim_j, sat) if x and j]).most_common(1)
        r = {"side": side, "object": ob, "frames": [a, b], "labels": [ep.labels[a], ep.labels[b]], "judged": len(sv),
             "saturated_frac": round(float(sat.mean()), 3), "median_s": round(float(np.median(sv)), 3), "peak_s": round(float(sv[k]), 3),
             "limiting_joint": lj[0][0] if lj else lim_j[k], "fail": bool(sat.mean() >= hold_frac)}
        rows.append(r)
    bad = [r for r in rows if r["fail"]]
    issues = [f"the {r['side']} hand carries {r['object']} (frames {r['frames'][0]}-{r['frames'][1]}, {r['labels'][0]}) only with its "
              f"fingers at {100 * r['median_s']:.0f}% of their torque cap (median; {100 * r['saturated_frac']:.0f}% of the hold at or "
              f"above {100 * sat_frac:.0f}%, limited by {r['limiting_joint']}): no reserve; choose a grasp that supports the "
              f"object from below or wraps it (more fingers, contacts closer to the joints, palm contact)" for r in bad]
    worst = max(bad, key=lambda r: r["median_s"]) if bad else None
    return EA.finish(issues, worst, windows=rows, unbalanced=unbalanced[:10], n_unbalanced=len(unbalanced),
                     caps_nm={caps[k][1]: round(caps[k][0], 3) for k in dofs[:4]}, frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(finger_servo_saturation_audit(sys.argv[1], sys.argv[2]), indent=1))
