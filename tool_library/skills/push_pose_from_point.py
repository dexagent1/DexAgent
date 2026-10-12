TOOL = {
    "name": "push_pose_from_point",
    "title": "Push pose from a surface point",
    "type": "skill",
    "category": "Grasp & Poses",
    "summary": "Construct a hand pose whose extended finger points straight into a surface at a given point, for pushing an object with the fingertip.",
    "description": "A push is not a grasp: the finger's direction is part of the intent. The posture is fixed (one finger extended, the others curled, thumb tucked), the wrist rotation is constructed so the finger axis (knuckle site to tip site) points along minus the surface normal, the wrist position follows analytically so the tip site lands one pad thickness outside the point, and the single remaining freedom, roll about the finger axis, is swept in 10 degree steps and ranked by non-pad interpenetration with the scene and then by knuckles-up (palm normal downward). Nothing is searched iteratively, so the result is deterministic and takes well under a second.",
    "inputs": [
        {"name": "scene", "type": "dict", "description": "{hand_xml, side, objects:[{name, xml, pos, yaw_deg}], floor}"},
        {"name": "point", "type": "xyz (m, world)", "description": "surface point to push at"},
        {"name": "normal", "type": "xyz", "description": "outward surface normal at the point; the finger will point along -normal"},
        {"name": "finger", "type": "str", "description": "pushing finger (default index); its axis runs from the knuckle site (or the tip link's parent joint) to the tip site"},
        {"name": "standoff_m", "type": "float", "description": "tip-site standoff from the surface (default 0.015)"},
        {"name": "posture", "type": "dict joint->fraction", "description": "override of the fixed posture as fractions of joint range (None = joint at 0)"},
        {"name": "roll_step_deg", "type": "float", "description": "roll sweep step about the finger axis (default 10)"},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "{side, pos, euler, finger_targets, grip_fingers:[finger], residual_mm, overlap_mm, palm_facing (down-facing cosine), finger_alignment (cosine of finger axis with -normal)}"},
    ],
    "status": "tested",
    "entrypoint": "push_pose_from_point",
    "requirements": ["mujoco>=3.1", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["push", "pose", "dexterous-hand"],
    "used_in": ["Place Toy in Drawer", "Battery Insertion", "Computer Installation"],
}

import numpy as np

# fractions of joint range: index extended, the other fingers curled out of
# the way, the thumb tucked across the palm. Keys follow the Sharpa naming;
# joints a hand does not have are ignored, joints not listed sit at 0.
PUSH_POSTURE = {
    "index_MCP_FE": 0.16, "index_MCP_AA": None, "index_PIP": 0.04, "index_DIP": 0.04,
    "middle_MCP_FE": 0.80, "middle_MCP_AA": None, "middle_PIP": 0.85, "middle_DIP": 0.70,
    "ring_MCP_FE": 0.80, "ring_MCP_AA": None, "ring_PIP": 0.85, "ring_DIP": 0.70,
    "pinky_CMC": 0.40, "pinky_MCP_FE": 0.80, "pinky_MCP_AA": None, "pinky_PIP": 0.85, "pinky_DIP": 0.70,
    "thumb_CMC_FE": 0.75, "thumb_CMC_AA": None, "thumb_MCP_FE": 0.55, "thumb_MCP_AA": None, "thumb_IP": 0.45,
    # Shadow hand equivalents
    "FFJ3": 0.16, "FFJ2": 0.04, "FFJ1": 0.04,
    "MFJ3": 0.80, "MFJ2": 0.85, "MFJ1": 0.70, "RFJ3": 0.80, "RFJ2": 0.85, "RFJ1": 0.70,
    "LFJ5": 0.40, "LFJ3": 0.80, "LFJ2": 0.85, "LFJ1": 0.70,
    "THJ4": 0.75, "THJ2": 0.55, "THJ1": 0.45,
}


def _rot_between(a, b):
    a = np.asarray(a, float) / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, float) / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return _axis_angle(axis, np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def _axis_angle(axis, th):
    axis = np.asarray(axis, float) / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def push_pose_from_point(scene, point, normal, finger="index", standoff_m=0.015,
                         posture=None, roll_step_deg=10.0):
    import mujoco
    from tool_library.common import hand as H

    model, data0, rig, ids = H.open_scene(scene)
    if finger not in rig.fingertip_sites:
        raise ValueError(f"finger '{finger}' has no tip site; fingers are {sorted(rig.fingertip_sites)}")
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    aim = -n
    target = np.asarray(point, float) + n * float(standoff_m)
    site = int(rig.fingertip_sites[finger])
    palm_bid, palm_n = H.palm_frame(model, data0, rig)
    post = dict(PUSH_POSTURE, **(posture or {}))

    d = mujoco.MjData(model)
    d.qpos[:] = data0.qpos
    rig.teleport(d, np.zeros(3), np.zeros(3))
    for k, adr in rig.finger_qpos.items():
        lo, hi = rig.finger_range[k]
        frac = post.get(k)
        d.qpos[adr] = float(np.clip(0.0, lo, hi)) if frac is None else lo + float(frac) * (hi - lo)
    mujoco.mj_forward(model, d)
    tip_l = np.asarray(d.site_xpos[site], float).copy()
    kn_l = rig.knuckle_point(d, finger)
    v_l = tip_l - kn_l
    v_l /= np.linalg.norm(v_l) + 1e-12
    R0 = _rot_between(v_l, aim)
    best = None
    for th in np.arange(0.0, 360.0, float(roll_step_deg)):
        R = _axis_angle(aim, np.radians(th)) @ R0
        p = target - R @ tip_l
        rig.teleport(d, p, H.mat_to_euler(R))
        mujoco.mj_forward(model, d)
        res = float(np.linalg.norm(np.asarray(d.site_xpos[site], float) - target))
        overlap = H.deepest_overlap(model, d, rig, ids, keep_tips=(finger,))
        down = 0.0
        if palm_n is not None:
            pn = np.asarray(d.xmat[palm_bid], float).reshape(3, 3) @ palm_n
            down = float(pn @ np.array([0.0, 0.0, -1.0]))
        key = (round(max(res, 0.002), 3), round(max(overlap, 0.002) * 500) / 500.0, -round(down, 1))
        if best is None or key < best[0]:
            best = (key, res, overlap, down, d.qpos.copy())
    _, res, overlap, down, qpos = best
    d.qpos[:] = qpos
    mujoco.mj_forward(model, d)
    vw = np.asarray(d.site_xpos[site], float) - rig.knuckle_point(d, finger)
    align = float((vw / (np.linalg.norm(vw) + 1e-12)) @ aim)
    pose = H.pose_from_data(model, d, rig, rig.side)
    pose.update({"grip_fingers": [finger], "residual_mm": {finger: round(res * 1000, 2)},
                 "worst_residual_mm": round(res * 1000, 2), "overlap_mm": round(overlap * 1000, 2),
                 "palm_facing": round(down, 3), "finger_alignment": round(align, 3),
                 "standoff_m": float(standoff_m), "point": [float(v) for v in point],
                 "normal": [float(v) for v in n]})
    return pose
