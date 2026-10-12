"""Rod seated in slot check: does a cylindrical part lie in its channel, along it, between its end stops?

    from tool_library.verifiers.rod_seated_in_slot_check import rod_seated_in_slot_check
    rod_seated_in_slot_check("scene.xml", "trajectory.npz", body="battery_battery", holder="battery_holder_battery_holder",
                             slot_center=[0.0, -0.00705, 0.00685], slot_axis=[1, 0, 0], rod_center=[0, 0, 0.02225],
                             rod_half_length=0.02225, stops=[-0.0263, 0.0263])
"""
from __future__ import annotations

TOOL = {
    "name": "rod_seated_in_slot_check",
    "title": "Rod seated in slot check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "A cylindrical part (cell, dowel, pen, fuse) ends lying in its channel: its axis parallel to the slot axis, its centre line on the slot's, and both ends between the slot's end stops, measured in the holder's own frame on an executed trajectory.",
    "description": (
        "The outcome gate of an insertion into an open channel, where the part's root barely moves but its far end must "
        "come down (a battery whose + end is pressed past the end contact): a position check on the root cannot tell a "
        "seated cell from one still resting on the end wall. The check reads the frame (default the last; the median "
        "over the last `settle_frames` frames), expresses the rod's axis and centre in the frame of the holder body (so a "
        "holder that slid or turned during the task is followed), and measures: the angle between the rod axis and the "
        "slot axis (sign-free), the radial distance of the rod centre from the slot's axis line, the axial position of "
        "the centre, and, with `rod_half_length` and `stops`, where both ends lie along the slot. It passes when the angle "
        "is within axis_tol_deg, the radial offset within radial_tol_m and both ends inside the stops widened by stop_tol_m. "
        "It also reports the first frame from which the rod stays seated to the end. Kinematic only: it judges the "
        "recorded states, not forces."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF scene whose qpos layout the trajectory uses."},
        {"name": "trajectory", "type": "str | ndarray [T, nq] | None", "description": "Executed episode (npz with qpos, or an array); None judges the scene's initial state."},
        {"name": "body", "type": "str", "description": "The rod's body."},
        {"name": "holder", "type": "str", "description": "The body that carries the slot (its frame defines slot_center and slot_axis)."},
        {"name": "slot_center", "type": "list[3], metres", "description": "A point on the slot's axis line in the holder frame (the channel centre)."},
        {"name": "slot_axis", "type": "list[3]", "description": "Slot axis direction in the holder frame. Default [1, 0, 0]."},
        {"name": "rod_axis", "type": "list[3]", "description": "The rod's axis in its body frame. Default [0, 0, 1]."},
        {"name": "rod_center", "type": "list[3], metres", "description": "The rod's mid-point on its axis in its body frame. Default [0, 0, 0]."},
        {"name": "rod_half_length", "type": "float | None", "description": "Half length of the rod along its axis; with stops, both ends are checked."},
        {"name": "stops", "type": "list[2] | None", "description": "Axial positions (holder frame, along slot_axis, relative to slot_center) of the two end stops the ends must lie between."},
        {"name": "frame", "type": "int", "description": "Frame judged. Default -1 (last)."},
        {"name": "settle_frames", "type": "int", "description": "Median of the measurements over this many frames ending at `frame`. Default 1."},
        {"name": "axis_tol_deg, radial_tol_m, stop_tol_m", "type": "float", "description": "Defaults 3 deg, 1.5 mm, 1 mm."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; angle_deg, radial_m, axial_m, ends_m [2] (when half length given), stops, seated_from_frame, n_frames, frame."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["insertion", "seat", "battery", "channel", "task gate", "rollout"],
    "used_in": ["Battery Insertion"],
    "status": "tested",
    "entrypoint": "rod_seated_in_slot_check",
}

from pathlib import Path

import mujoco
import numpy as np

_KNOWN = {"slot_axis", "rod_axis", "rod_center", "rod_half_length", "stops", "frame", "settle_frames", "axis_tol_deg",
          "radial_tol_m", "stop_tol_m"}


def _load_traj(model, trajectory):
    if trajectory is None:
        return np.asarray(model.qpos0, float)[None, :]
    if isinstance(trajectory, (str, Path)):
        z = np.load(str(trajectory), allow_pickle=True)
        q = np.asarray(z["qpos"] if hasattr(z, "files") else z, float)
    else:
        q = np.asarray(trajectory, float)
    if q.ndim == 1:
        q = q[None, :]
    if q.shape[1] != model.nq:
        raise ValueError(f"trajectory has {q.shape[1]} qpos columns but the scene has nq={model.nq}: give the scene the trajectory was recorded in")
    return q


def _unit(v, name):
    v = np.asarray(v, float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError(f"{name} must be a non-zero 3-vector")
    return v / n


def rod_seated_in_slot_check(scene, trajectory=None, body=None, holder=None, slot_center=None, **kw):
    unknown = set(kw) - _KNOWN
    if unknown:
        raise ValueError(f"unknown parameter(s) {sorted(unknown)}; known: {sorted(_KNOWN)}")
    if body is None or holder is None or slot_center is None:
        raise ValueError("body, holder and slot_center are required")
    model = scene if isinstance(scene, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, holder)
    if bid < 0 or hid < 0:
        raise ValueError(f"no body {body if bid < 0 else holder!r} in the scene")
    c0 = np.asarray(slot_center, float).reshape(3)
    sa = _unit(kw.get("slot_axis", (1.0, 0.0, 0.0)), "slot_axis")
    ra = _unit(kw.get("rod_axis", (0.0, 0.0, 1.0)), "rod_axis")
    rc = np.asarray(kw.get("rod_center", (0.0, 0.0, 0.0)), float).reshape(3)
    half = kw.get("rod_half_length")
    stops = kw.get("stops")
    if stops is not None and (half is None or len(stops) != 2):
        raise ValueError("stops needs two values and rod_half_length")
    a_tol = float(kw.get("axis_tol_deg", 3.0)); r_tol = float(kw.get("radial_tol_m", 0.0015)); s_tol = float(kw.get("stop_tol_m", 0.001))
    Q = _load_traj(model, trajectory)
    T = len(Q)
    frame = int(kw.get("frame", -1)); frame = frame + T if frame < 0 else frame
    if not 0 <= frame < T:
        raise ValueError(f"frame {kw.get('frame')} outside the trajectory's {T} frames")

    def measure(q):
        data.qpos[:] = q; mujoco.mj_kinematics(model, data)
        Rh = data.xmat[hid].reshape(3, 3); ph = data.xpos[hid]
        Rb = data.xmat[bid].reshape(3, 3); pb = data.xpos[bid]
        u = Rh.T @ (Rb @ ra)
        c = Rh.T @ (pb + Rb @ rc - ph) - c0
        ang = float(np.degrees(np.arccos(np.clip(abs(float(u @ sa)), 0.0, 1.0))))
        ax = float(c @ sa)
        rad = float(np.linalg.norm(c - ax * sa))
        ends = None
        if half is not None:
            e = sorted([float((c + s * float(half) * u) @ sa) for s in (-1.0, 1.0)])
            ends = e
        return ang, rad, ax, ends

    def seated(m_):
        ang, rad, ax, ends = m_
        ok = ang <= a_tol and rad <= r_tol
        if stops is not None and ends is not None:
            lo, hi = sorted(float(s) for s in stops)
            ok = ok and ends[0] >= lo - s_tol and ends[1] <= hi + s_tol
        return ok

    n_set = max(1, int(kw.get("settle_frames", 1)))
    ms = [measure(Q[t]) for t in range(max(0, frame - n_set + 1), frame + 1)]
    ang = float(np.median([m[0] for m in ms])); rad = float(np.median([m[1] for m in ms])); ax = float(np.median([m[2] for m in ms]))
    ends = None
    if half is not None:
        ends = [float(np.median([m[3][0] for m in ms])), float(np.median([m[3][1] for m in ms]))]
    issues = []
    if ang > a_tol:
        issues.append(f"the {body} axis is {ang:.1f} deg off the slot axis (tolerance {a_tol:g}): it still rests on an end or a wall; press its raised end down or push it along the slot first")
    if rad > r_tol:
        issues.append(f"the {body} centre is {1000 * rad:.1f} mm off the slot's axis line (tolerance {1000 * r_tol:.1f} mm): it is not down in the channel")
    if stops is not None and ends is not None:
        lo, hi = sorted(float(s) for s in stops)
        if ends[0] < lo - s_tol or ends[1] > hi + s_tol:
            issues.append(f"the {body} ends lie at [{1000 * ends[0]:.1f}, {1000 * ends[1]:.1f}] mm along the slot, outside the stops [{1000 * lo:.1f}, {1000 * hi:.1f}] mm")
    seated_from = None
    for t in range(frame, -1, -1):
        if seated(measure(Q[t])):
            seated_from = t
        else:
            break
    return {"ok": not issues, "issues": issues, "angle_deg": round(ang, 2), "radial_m": round(rad, 5), "axial_m": round(ax, 5),
            "ends_m": None if ends is None else [round(e, 5) for e in ends], "stops": stops, "seated_from_frame": seated_from,
            "frame": frame, "n_frames": T}


if __name__ == "__main__":
    import json, sys
    print(json.dumps(rod_seated_in_slot_check(sys.argv[1], sys.argv[2], body=sys.argv[3], holder=sys.argv[4], slot_center=json.loads(sys.argv[5])), indent=1))
