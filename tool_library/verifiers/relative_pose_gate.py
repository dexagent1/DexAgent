"""Relative pose gate: the final pose of A in B's frame against a target (assembly, insertion)."""
from __future__ import annotations

TOOL = {
    "name": "relative_pose_gate",
    "title": "Relative pose gate",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "The final pose of body A expressed in body B's frame is within position and orientation tolerances of a "
               "target relative pose (an assembled part, a seated plug), with optional axis masks and rotational symmetry.",
    "description": (
        "Assembly goals are relations between two parts, wherever the pair ends on the table. On the decision frame A's "
        "root pose is expressed in B's root frame (p_rel = R_B^T (p_A - p_B), R_rel = R_B^T R_A) and compared with the "
        "target: rel_pos [m] on the axes listed in pos_axes, and rel_quat (w, x, y, z) or rel_euler_deg (xyz) as the "
        "geodesic rotation angle. A symmetric part (a cap, a square peg) gives sym_axis (in A's frame) and sym_deg: the "
        "smallest error over the symmetric copies of the target counts. reference_frame takes the target from the pair's "
        "relative pose on another frame (a part that must return to where it sat). The issue gives the correction as a "
        "world displacement."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the trajectory was executed in."},
        {"name": "trajectory", "type": "str", "description": "Episode npz."},
        {"name": "body", "type": "str", "description": "Body A whose pose is judged."},
        {"name": "reference_body", "type": "str", "description": "Body B whose frame the pose is expressed in."},
        {"name": "rel_pos", "type": "list[float] | None", "description": "Target position of A in B's frame, m."},
        {"name": "rel_quat", "type": "list[float] | None", "description": "Target orientation of A in B's frame (w, x, y, z)."},
        {"name": "rel_euler_deg", "type": "list[float] | None", "description": "Target orientation as xyz Euler angles, deg."},
        {"name": "reference_frame", "type": "int | None", "description": "Take the target relative pose from this frame."},
        {"name": "pos_tol_m", "type": "float", "description": "Position tolerance, m. Default 0.01."},
        {"name": "rot_tol_deg", "type": "float", "description": "Orientation tolerance, deg. Default 10."},
        {"name": "pos_axes", "type": "str", "description": "Axes of B's frame the position is judged on, e.g. 'xy'. Default 'xyz'."},
        {"name": "sym_axis", "type": "list[float] | None", "description": "Symmetry axis in A's frame."},
        {"name": "sym_deg", "type": "float | None", "description": "Symmetry angle, deg (90 for a square peg)."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last]; decided on the last frame."},
        {"name": "subgoal", "type": "int | None", "description": "Use the frames of this npz subgoal index."},
        {"name": "frame", "type": "int | None", "description": "Decide on this frame instead."},
        {"name": "out_json", "type": "str | None", "description": "Optional report path."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; frame; rel_pos; rel_euler_deg; target_rel_pos; pos_err_m; rot_err_deg."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "goal", "assembly", "pose", "task gate"],
    "status": "tested",
    "entrypoint": "relative_pose_gate",
}

from tool_library.common import task_gates as G


def relative_pose_gate(scene, trajectory, body, reference_body, rel_pos=None, rel_quat=None, rel_euler_deg=None,
                       reference_frame=None, pos_tol_m=0.01, rot_tol_deg=10.0, pos_axes="xyz", sym_axis=None, sym_deg=None,
                       frames=None, subgoal=None, frame=None, out_json=None):
    if not pos_axes or set(pos_axes) - set("xyz"):
        raise ValueError("pos_axes: a non-empty subset of 'xyz'")
    if rel_quat is not None and len(rel_quat) != 4:
        raise ValueError("rel_quat: four numbers (w, x, y, z)")
    ep, t0, t1 = G.open_window(scene, trajectory, frames, subgoal, frame)
    A = ep.bodies(body, "body")
    B = ep.bodies(reference_body, "reference_body") - A
    if not B:
        raise ValueError("reference_body: it lies inside body's subtree")
    if reference_frame is not None and not -ep.T <= int(reference_frame) < ep.T:
        raise ValueError(f"reference_frame: {reference_frame} is outside the trajectory")
    rep = G.gate_relative_pose(ep, A, B, t1, rel_pos=rel_pos, rel_quat=rel_quat, rel_euler_deg=rel_euler_deg,
                               reference_frame=None if reference_frame is None else int(reference_frame) % ep.T,
                               pos_tol_m=float(pos_tol_m), rot_tol_deg=float(rot_tol_deg), pos_axes=pos_axes,
                               sym_axis=sym_axis, sym_deg=sym_deg, names=(str(body), str(reference_body)))
    return G.finish(rep, out_json)


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(relative_pose_gate(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], reference_frame=0), indent=1))
