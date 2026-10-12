"""Release-settle gate: after the hand lets go, the object stays where it was put."""
from __future__ import annotations

TOOL = {
    "name": "release_settle_gate",
    "title": "Release and settle gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "After the release (the last frame a robot body touches the object in the window) the object seats itself "
               "by at most a small allowance and then stays still for a settle time: no fall, roll, slide or tip-over, "
               "supported at the end, optionally upright / upside down / on a named support.",
    "description": (
        "The goal predicate behind 'rests on the table without hand support' and 'stays in the slot when the hands let "
        "go'. The release is the last frame of the window on which any robot body touches the object (1 mm margin); with "
        "no touch in the window the object is judged from the window's start. In the first seat_s after the release it may "
        "seat itself (a brick let go at a slight tilt drops onto its studs, a battery pivots into its slot) by at most "
        "seat_tol_m of centre motion and seat_tol_deg of tipping; over the following settle_s its centre must stay within "
        "pos_tol_m and its up axis within rot_tol_deg, comparing the mean pose over the first third of that window with "
        "the last third, so an object rocking or spinning in place passes while one sliding, rolling or tipping away "
        "fails. When the window ends before the settle time, the following frames are watched until a robot body touches "
        "the object again; less than min_settle_s of observation fails (let it rest before the next move). At the end "
        "the object must touch something other than the robot (else it is falling), the `on` support when given, and "
        "with orientation 'upright' / 'upside_down' its frame-0 up axis (or up_local) must point up / down within "
        "orient_tol_deg ('horizontal': within orient_tol_deg + 15 deg of horizontal)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "The released object."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] window holding the release."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "release_frame", "type": "int | None", "description": "Release frame instead of the last robot contact."},
        {"name": "settle_s", "type": "float", "description": "Time the object must stay still after seating, s. Default 0.3."},
        {"name": "pos_tol_m", "type": "float", "description": "Allowed drift of its centre while settled, m. Default 0.004."},
        {"name": "rot_tol_deg", "type": "float", "description": "Allowed tipping of its up axis while settled, deg. Default 10."},
        {"name": "seat_s", "type": "float", "description": "Seating time right after the release, s. Default 0.15."},
        {"name": "seat_tol_m", "type": "float", "description": "Allowed centre motion from the release to the end, m. Default 0.015."},
        {"name": "seat_tol_deg", "type": "float", "description": "Allowed tipping from the release to the end, deg. Default 35."},
        {"name": "min_settle_s", "type": "float", "description": "Least observed settle time, s. Default 0.15."},
        {"name": "orientation", "type": "str | None", "description": "'upright', 'upside_down' or 'horizontal' relative to up_local."},
        {"name": "up_local", "type": "list[float] | None", "description": "Body-frame axis that is 'up' for the orientation test; default the axis that points up on frame 0."},
        {"name": "orient_tol_deg", "type": "float", "description": "Orientation tolerance, deg. Default 15."},
        {"name": "on", "type": "str | None", "description": "Body (or 'table') the object must rest on at the end."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; release_frame; settle_frames; settle_s; moved_m; rotated_deg; moved_since_release_m; rotated_since_release_deg; dz_since_release_m; supported; on_support; tilt_from_start_up_deg (or up_local_tilt_deg with up_local)."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "release", "placement", "settle", "task gate"],
    "status": "tested",
    "entrypoint": "release_settle_gate",
}

import numpy as np

from tool_library.common import task_gates as G


def release_settle_gate(scene, trajectory, body, frames=None, subgoal=None, release_frame=None, settle_s=0.3, pos_tol_m=0.004,
                        rot_tol_deg=10.0, seat_s=0.15, seat_tol_m=0.015, seat_tol_deg=35.0, min_settle_s=0.15,
                        orientation=None, up_local=None, orient_tol_deg=15.0, on=None, out_json=None):
    if orientation not in (None, "upright", "upside_down", "horizontal"):
        raise ValueError("orientation: 'upright', 'upside_down', 'horizontal' or None")
    if float(settle_s) <= 0:
        raise ValueError("settle_s: must be positive")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal)
    A = ep.bodies(body, "body")
    S = ep.bodies(on, "on") - A if on is not None else None
    if release_frame is not None and not 0 <= int(release_frame) < ep.T:
        raise ValueError(f"release_frame: {release_frame} is outside the trajectory")
    u = None
    if up_local is not None and orientation:
        u = np.asarray(up_local, float)
        if u.shape != (3,) or np.linalg.norm(u) < 1e-9:
            raise ValueError("up_local: a non-zero 3-vector in the body frame")
        u = u / np.linalg.norm(u)
    # with up_local the gate judges settling only; the orientation is judged below on that body axis itself (the gate's
    # own test reads the body axis that points up on frame 0, which is a different axis when up_local is not vertical then)
    rep = G.gate_release_settle(ep, A, t0, t1, settle_s=float(settle_s), pos_tol_m=float(pos_tol_m), rot_tol_deg=float(rot_tol_deg),
                                min_settle_s=float(min_settle_s), seat_s=float(seat_s), seat_tol_m=float(seat_tol_m),
                                seat_tol_deg=float(seat_tol_deg), orientation=None if u is not None else orientation,
                                orient_tol_deg=float(orient_tol_deg), on=S, release_frame=release_frame, names={"A": str(body), "on": on})
    if u is not None and "moved_m" in rep:                    # the gate judged the settle window (it did not stop early)
        end = int(rep["settle_frames"][1])
        _, Re = ep.pose(end, min(A))
        tl = G.tilt_deg(Re, u)
        rep["up_local_tilt_deg"] = round(tl, 2)
        tol = float(orient_tol_deg)
        lo, hi = {"upright": (0.0, tol), "upside_down": (180.0 - tol, 180.0), "horizontal": (90.0 - tol - 15.0, 90.0 + tol + 15.0)}[orientation]
        if not (lo - 1e-6 <= tl <= hi + 1e-6):
            rep["issues"].append(f"{body}'s up axis {np.round(u, 3).tolist()} ends {tl:.0f} deg from world up at frame {end}: it is not "
                                 f"{orientation.replace('_', ' ')}")
            rep["ok"] = False
    rep["frames"] = [t0, t1]
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(release_settle_gate(sys.argv[1], sys.argv[2], sys.argv[3]), indent=1))
