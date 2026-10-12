"""Shared contact-frame semantics for in-hand slip measurements.

An articulated assembly is tracked at the link and material point the hand
actually contacts, rather than at a remote root. Ball-jointed links may bend
and twist at that point; a free rigid body has no such angular exemption.
"""
import copy
from dataclasses import dataclass

import mujoco
import numpy as np


RIGID_SLIP_DEG = 15.0
HELD_SEARCH_S = .5


def merge_motion_expectations(generated, requested):
    """Add plan expectations without waiving a motion's required stable grasp.

    The planner decides which motions require no slip. Intentional sliding
    motions keep their own semantics. Required windows come from phase labels.
    """
    result = copy.deepcopy(generated)
    requested = requested or {}
    if generated.get("no_slip"):
        if "no_slip" in requested and requested["no_slip"] is not True:
            raise ValueError("this motion requires no_slip; repair the grasp or motion instead of disabling its check")
        if any(key in requested for key in ("slip_from", "slip_from_rel")):
            raise ValueError("this motion's no_slip window is determined by its executed phases")
        for key in ("hand", "object"):
            if key in requested and key in generated and requested[key] != generated[key]:
                raise ValueError(f"cannot redirect required no_slip check to a different {key}")
    result.update(requested)
    if generated.get("also_in_hand"):
        # Acquiring a second object cannot waive retention of the first one.
        extra = requested.get("also_in_hand", generated["also_in_hand"])
        if not isinstance(extra, list) or any(item not in extra for item in generated["also_in_hand"]):
            raise ValueError("also_in_hand must retain every object already held by this motion")
        result["also_in_hand"] = copy.deepcopy(extra)
    if generated.get("hands"):
        if "hands" in requested and not isinstance(requested["hands"], dict):
            raise ValueError("hands expectations must preserve the motion's required checks")
        result["hands"] = copy.deepcopy(generated["hands"])
        for side, expectation in generated["hands"].items():
            result["hands"][side] = merge_motion_expectations(expectation, requested.get("hands", {}).get(side))
    if generated.get("no_slip"):
        result["_motion_no_slip"] = True
    return result


def ball_link(model, body):
    """A ball link, including geometry welded below its joint-bearing body."""
    body = int(body)
    while body > 0 and int(model.body_jntnum[body]) == 0:
        body = int(model.body_parentid[body])
    return (body > 0 and int(model.body_jntnum[body]) == 1
            and int(model.jnt_type[int(model.body_jntadr[body])]) == int(mujoco.mjtJoint.mjJNT_BALL))


def articulated(model, bodies, root):
    return ball_link(model, root) or any(int(model.body_jntnum[b]) > 0 for b in bodies if b != root)


@dataclass(frozen=True)
class HeldFrame:
    body: int
    point: np.ndarray | None = None

    def relative_pose(self, data, wrist):
        R = data.xmat[self.body].reshape(3, 3)
        p = data.xpos[self.body]
        if self.point is not None:
            p = p + R @ self.point
        Rw = data.xmat[wrist].reshape(3, 3)
        return Rw.T @ (p - data.xpos[wrist]), Rw.T @ R


def contacted_frame(model, data, hand_bodies, object_bodies, margin=.003):
    """Most-contacted body and contact centroid; deepest contact breaks ties.

    The caller has updated collision data. None means no contact was found,
    not evidence that the object's root is physically held.
    """
    counts, points = {}, {}
    for contact in data.contact:
        # Flex contacts use -1 for a geom id. They do not identify a rigid
        # material point; indexing -1 would silently select the last geom.
        if contact.geom1 < 0 or contact.geom2 < 0:
            continue
        if float(contact.dist) > margin:
            continue
        b1, b2 = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        for hand, body in ((b1, b2), (b2, b1)):
            if hand in hand_bodies and body in object_bodies:
                count, depth = counts.get(body, (0, 0.))
                counts[body] = (count + 1, max(depth, -float(contact.dist)))
                points.setdefault(body, []).append(np.array(contact.pos, float))
    if not counts:
        return None
    body = max(counts, key=lambda b: counts[b])
    R = data.xmat[body].reshape(3, 3)
    return HeldFrame(body, R.T @ (np.mean(points[body], axis=0) - data.xpos[body]))


def measure_slip(model, Q, fps, start, stop, root, wrist, hand_bodies, object_bodies, data=None):
    """Peak position/angle change on every recorded state of an inclusive window.

    Contact selection may look ahead 0.5 s, as the episode audit does, but the
    reference stays at `start`: discovering the held link never discards early
    motion. Callers retain their own phase windows and translation limits.
    """
    data = data if data is not None else mujoco.MjData(model)
    frame, found = HeldFrame(int(root)), None
    if articulated(model, object_bodies, root):
        end = min(stop, start + max(1, int(round(HELD_SEARCH_S * fps))))
        for t in range(start, end + 1):
            data.qpos[:] = Q[t]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_collision(model, data)
            contact = contacted_frame(model, data, hand_bodies, object_bodies)
            if contact is not None:
                frame, found = contact, t
                break
    p0 = R0 = None
    drift = turn = 0.
    peak_pos = peak_rot = start
    for t in range(start, stop + 1):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        p, R = frame.relative_pose(data, wrist)
        if p0 is None:
            p0, R0 = p.copy(), R.copy()
        distance = float(np.linalg.norm(p - p0))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1.) / 2., -1., 1.))))
        if distance > drift:
            drift, peak_pos = distance, t
        if angle > turn:
            turn, peak_rot = angle, t
    return {"body": model.body(frame.body).name, "body_id": frame.body,
            "contact_point_local": None if frame.point is None else frame.point.tolist(),
            "contact_frame": found, "turn_judged": not ball_link(model, frame.body),
            "drift_m": drift, "turn_deg": turn, "translation_peak_frame": peak_pos,
            "rotation_peak_frame": peak_rot, "frames": [int(start), int(stop)]}
