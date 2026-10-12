"""The demonstrator's hand as a prior for the robot: keypoints, hand frames, silhouettes and reach.

Three tools share this module: `register_camera_hand_overlay` (the camera from the robot hands' silhouette laid
over the human hands), `human_grasp_pose_prior` (a robot hand pose from the human hand at the contact frame) and
`place_egodex_camera_for_reach` (where EgoDex's head camera stands on the bench so the arms reach the objects).

Hand frame. Every hand, human or robot, gets the same anatomical frame built from four keypoints, so a pose moves
from one hand to the other by matching frames:

* origin: the centroid of the four finger knuckles (index, middle, ring, little MCP joints);
* z: from the wrist to the middle knuckle (along the palm, toward the fingers);
* y: across the knuckles, little -> index on a right hand and index -> little on a left hand, made orthogonal to z;
* x = y cross z, which points OUT OF THE PALM on both hands.

This is the Sharpa hand root's own convention (root z along the fingers, x the palm normal, y the thumb side on the
right hand and the little-finger side on the left hand; see arm_workspace_map), so on the Sharpa the anatomical
frame is the root frame up to a 6 degree tilt and a 9 cm offset to the knuckles; `sharpa_reference` measures that
fixed offset from the hand's own sites. For EgoDex (ARKit skeleton) the frame from the joints agrees with the ARKit
wrist transform composed with a fixed axis map: Sharpa x = -ARKit y, y = +ARKit z, z = -ARKit x on the right hand and
x = +ARKit y, y = +ARKit z, z = +ARKit x on the left (`ARKIT_TO_ANATOMICAL`) up to a constant 6 degree offset
(measured over EgoDex build_unstack_lego/59, both hands: 6.3 degrees median, and 0.6 degrees median once that constant
is removed, so the joints carry the same wrist rotation as ARKit's transform). The keypoint frame is used because
hands.npz and HaMeR both give joint positions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# ARKit (EgoDex) joint names per finger, base -> tip. For the fingers the base is the knuckle (MCP); the thumb's
# 'Knuckle' is its CMC joint near the wrist, the same place as the Sharpa thumb's first site.
ARKIT_CHAINS = {
    "thumb": ("ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"),
    "index": ("IndexFingerKnuckle", "IndexFingerIntermediateBase", "IndexFingerIntermediateTip", "IndexFingerTip"),
    "middle": ("MiddleFingerKnuckle", "MiddleFingerIntermediateBase", "MiddleFingerIntermediateTip", "MiddleFingerTip"),
    "ring": ("RingFingerKnuckle", "RingFingerIntermediateBase", "RingFingerIntermediateTip", "RingFingerTip"),
    "pinky": ("LittleFingerKnuckle", "LittleFingerIntermediateBase", "LittleFingerIntermediateTip", "LittleFingerTip"),
}
ARKIT_WRIST = "Hand"
# columns: the anatomical (= Sharpa root) axes x, y, z written in the ARKit wrist frame
ARKIT_TO_ANATOMICAL = {
    "right": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "left": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
}
# MANO (HaMeR): the 16 regressed joints are wrist, index 1-3, middle 4-6, little 7-9, ring 10-12, thumb 13-15; the
# fingertips are these mesh vertices (the convention HaMeR and most MANO code use for the OpenPose hand).
MANO_JOINTS = {"wrist": 0, "index": (1, 2, 3), "middle": (4, 5, 6), "pinky": (7, 8, 9), "ring": (10, 11, 12), "thumb": (13, 14, 15)}
MANO_TIP_VERTS = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "pinky": 673}
# Sharpa sites per finger, base -> tip ('{side}_{finger}_{k}')
SHARPA_SITES = ("mcp", "pip", "dip", "tip")


# ------------------------------------------------------------------ hand frames --
def anatomical_frame(kp: dict, side: str):
    """(R [3, 3] whose columns are the frame axes in the keypoints' frame, origin [3]) of a hand given as
    {'wrist': xyz, finger: [[base], ..., [tip]]}."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    wrist = np.asarray(kp["wrist"], float)
    base = {f: np.asarray(kp[f], float)[0] for f in ("index", "middle", "ring", "pinky")}
    c = np.mean([base[f] for f in base], axis=0)
    z = base["middle"] - wrist
    if np.linalg.norm(z) < 1e-6:
        raise ValueError("the wrist and the middle knuckle coincide; the hand keypoints are degenerate")
    z /= np.linalg.norm(z)
    across = base["index"] - base["pinky"]
    y = across if side == "right" else -across
    y = y - z * (z @ y)
    if np.linalg.norm(y) < 1e-6:
        raise ValueError("the knuckles are collinear with the wrist; the hand keypoints are degenerate")
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1), c


def chain_length(points) -> float:
    P = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def keypoints_from_egodex(hands_npz, side: str, frames):
    """{'wrist': [T, 3], finger: [T, 4, 3]} rig-frame joints of one hand at the given frames from a hands.npz
    written by read_egodex_episode (25 ARKit joints per hand)."""
    with np.load(str(hands_npz), allow_pickle=False) as z:
        names = [str(n) for n in z["names"]]
        X = np.asarray(z["rig_xyz"], float)
    frames = np.atleast_1d(np.asarray(frames, int))
    if frames.min() < 0 or frames.max() >= len(X):
        raise ValueError(f"frames must lie in 0..{len(X) - 1}")
    col = {n: i for i, n in enumerate(names)}

    def j(n):
        key = side + n
        if key not in col:
            raise ValueError(f"hands.npz has no joint {key!r}; is it an EgoDex hands.npz from read_egodex_episode?")
        return X[frames, col[key]]

    out = {"wrist": j(ARKIT_WRIST)}
    for f, chain in ARKIT_CHAINS.items():
        out[f] = np.stack([j(n) for n in chain], axis=1)
    return out


def keypoints_from_hamer(hamer_npz, camera, side: str, frames, joint_regressor, hamer_focal_px=None):
    """The same keypoints from extract_hand_meshes' hands.npz (MANO vertices in HaMeR's camera frame). HaMeR places
    each hand with a nominal focal length (5000 / 256 * the longer image side); the hand's depth is rescaled to the
    real camera's focal so its pixels stay where HaMeR put them, then the points go to the rig through camera.json.
    `joint_regressor` is MANO's J_regressor (16 x 778: an .npy, or the MANO_RIGHT.pkl HaMeR ships in its _DATA)."""
    from tool_library.common import rigcam as RC
    cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
    J = _mano_regressor(joint_regressor)
    with np.load(str(hamer_npz), allow_pickle=True) as zf:
        z = {k: zf[k] for k in ("frame_idx", "is_right", "verts", "cam_t")}
    fidx = [int(v) for v in z["frame_idx"]]
    f_nom = float(hamer_focal_px) if hamer_focal_px else 5000.0 / 256.0 * max(cam["width"], cam["height"])
    f_real = 0.5 * (cam["fx"] + cam["fy"])
    R, p = np.asarray(cam["R_cv"], float), np.asarray(cam["pos"], float)
    want = 1 if side == "right" else 0
    rows = {"wrist": []} | {f: [] for f in FINGERS}
    for fr in np.atleast_1d(np.asarray(frames, int)):
        if int(fr) not in fidx:
            raise ValueError(f"frame {int(fr)} is not in the HaMeR output")
        i = fidx.index(int(fr))
        isr = np.atleast_1d(z["is_right"][i])
        hit = [k for k, r in enumerate(isr) if int(r) == want]
        if not hit:
            raise ValueError(f"HaMeR found no {side} hand at frame {int(fr)}")
        k = hit[0]
        V = np.asarray(z["verts"][i][k], float)
        t = np.asarray(z["cam_t"][i][k], float).copy()
        t[2] *= f_real / f_nom
        Vc = V + t
        Jc = J @ Vc
        to_rig = lambda P: p + np.atleast_2d(P) @ R.T
        rows["wrist"].append(to_rig(Jc[MANO_JOINTS["wrist"]])[0])
        for f in FINGERS:
            if f == "thumb":
                chain = [Jc[a] for a in MANO_JOINTS["thumb"]] + [Vc[MANO_TIP_VERTS["thumb"]]]
            else:
                chain = [Jc[a] for a in MANO_JOINTS[f]] + [Vc[MANO_TIP_VERTS[f]]]
            rows[f].append(to_rig(np.array(chain)))
    return {k: np.array(v) for k, v in rows.items()}


def _mano_regressor(src):
    if src is None:
        raise ValueError("joint_regressor is required for HaMeR input: MANO's J_regressor as .npy, or MANO_RIGHT.pkl")
    if isinstance(src, np.ndarray):
        J = src
    elif str(src).endswith(".npy"):
        J = np.load(str(src))
    else:
        import pickle
        with open(str(src), "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        J = d["J_regressor"]
        J = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
    J = np.asarray(J, float)
    if J.shape[0] < 16:
        raise ValueError(f"joint_regressor has {J.shape[0]} rows; MANO's has 16")
    return J[:16]


def frame_at(kp: dict, i: int) -> dict:
    """One frame of a keypoint track."""
    return {k: np.asarray(v)[i] for k, v in kp.items()}


def sharpa_reference(hand_xml, side: str):
    """The Sharpa hand's own keypoints at its zero pose, in its root frame (the hand-alone model has the root at the
    origin): wrist = the root origin, each finger's mcp/pip/dip/tip sites; plus the fingertip body and tip point the
    retargeting drives, the anatomical frame in root coordinates and each finger's chain length."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(hand_xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_base_yaw")
    if root < 0:
        raise ValueError(f"{hand_xml} has no body {side}_base_yaw: not a Sharpa {side} hand")
    p0, R0 = d.xpos[root].copy(), d.xmat[root].reshape(3, 3).copy()
    to_root = lambda P: (np.asarray(P, float) - p0) @ R0
    kp = {"wrist": to_root(p0)}
    tips = {}
    for f in FINGERS:
        pts = []
        for k in SHARPA_SITES:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{f}_{k}")
            if sid < 0:
                raise ValueError(f"{hand_xml} has no site {side}_{f}_{k}")
            pts.append(to_root(d.site_xpos[sid]))
            if k == "tip":
                tips[f] = {"body": m.body(int(m.site_bodyid[sid])).name, "point": m.site_pos[sid].astype(float).tolist()}
        kp[f] = np.array(pts)
    R_a, c_a = anatomical_frame(kp, side)
    return {"kp": kp, "tips": tips, "R_anat": R_a, "c_anat": c_a, "root_body": f"{side}_base_yaw",
            "lengths": {f: chain_length(kp[f]) for f in FINGERS},
            "hand_length": float(np.linalg.norm(kp["middle"][-1] - kp["wrist"]))}


# ------------------------------------------------------------- camera helpers --
def camera_angles(cam) -> tuple:
    """(pitch_deg below horizontal, yaw_deg from +x, roll_deg) of a camera.json in the convention of
    rigcam.R_cv_from_angles, so camera_from_params(camera_angles(cam)...) rebuilds it."""
    from tool_library.common import rigcam as RC
    R = np.asarray(cam["R_cv"], float)
    fwd = R[:, 2]
    pitch = math.degrees(math.asin(float(np.clip(-fwd[2], -1, 1))))
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    R0 = RC.R_cv_from_angles(yaw, pitch, 0.0)
    roll = math.degrees(math.atan2(-float(R[:, 0] @ R0[:, 1]), float(R[:, 0] @ R0[:, 0])))
    return pitch, yaw, roll


def camera_from_params(width, height, fx, fy, cx, cy, pitch, yaw, roll, x, y, h, extra=None) -> dict:
    from tool_library.common import rigcam as RC
    return RC.camera_dict(width, height, fx, fy, cx, cy, RC.R_cv_from_angles(yaw, pitch, roll), [x, y, h], extra)


def angles_from_gravity(g_cam):
    """(pitch_deg below horizontal, roll_deg) in rigcam's convention from a gravity (down) vector in OpenCV camera
    coordinates, such as estimate_gravity's or fit_camera_to_table_plane's gravity_cam."""
    from tool_library.common import rigcam as RC
    from scipy.optimize import least_squares
    g = np.asarray(g_cam, float)
    g = g / np.linalg.norm(g)

    def res(v):
        R = RC.R_cv_from_angles(90.0, v[0], v[1])
        return R.T @ np.array([0.0, 0.0, -1.0]) - g

    pitch0 = math.degrees(math.asin(float(np.clip(g[2], -1, 1))))
    r = least_squares(res, [pitch0, 0.0])
    return float(r.x[0]), float(r.x[1])


# ------------------------------------------------------------- robot silhouette --
def robot_hand_hulls(scene_xml, keyframe="home", sides=("left", "right"), max_points=120, seed=0):
    """Convex-hull vertices (world) of every visible mesh of each station's HAND at a keyframe (the arm is left
    out: at home it is outside the ego view), with each hand root's pose. A mesh's projected silhouette is the 2D
    hull of its projected hull vertices, so `render_silhouette` needs no OpenGL."""
    import mujoco
    from scipy.spatial import ConvexHull
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    if keyframe is not None:
        k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, str(keyframe)) if isinstance(keyframe, str) else int(keyframe)
        if k < 0 or k >= m.nkey:
            raise ValueError(f"the scene has no keyframe {keyframe!r}")
        mujoco.mj_resetDataKeyframe(m, d, k)
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(seed)
    out = {}
    for side in sides:
        head = f"robot_{side}_{side}_"
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{head}base_yaw")
        if root < 0:
            raise ValueError(f"the scene has no {side} hand (body {head}base_yaw)")
        parts = []
        for g in range(m.ngeom):
            b = m.body(int(m.geom_bodyid[g])).name or ""
            if not b.startswith(head) or int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            if int(m.geom_group[g]) > 2 or float(m.geom_rgba[g][3]) == 0.0 or "collision_hand_" in (m.geom(g).name or ""):
                continue                                   # collision copies (group 3, hulls): the visual mesh is what the camera sees
            mid = int(m.geom_dataid[g])
            V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            try:
                V = V[ConvexHull(V).vertices]
            except Exception:
                pass
            if len(V) > max_points:
                V = V[rng.choice(len(V), int(max_points), replace=False)]
            parts.append(d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T)
        if not parts:
            raise ValueError(f"no visible meshes found on the {side} hand")
        out[side] = {"parts": parts, "root_pos": d.xpos[root].copy(), "root_R": d.xmat[root].reshape(3, 3).copy()}
    return out


def pose_hands(hulls: dict, drop_to_table=False, rest_height_m=0.0, spread_m=0.0, yaw_deg=0.0):
    """The hands moved as a demonstrator's resting hands may differ from the robot's home: lowered rigidly onto the
    table (lowest point at rest_height_m), shifted inward (spread_m > 0) or outward, mirrored about the stations'
    midline, and turned about vertical by yaw_deg (positive turns both fingertips inward)."""
    out = []
    for side, h in hulls.items():
        sgn = 1.0 if side == "left" else -1.0
        c = h["root_pos"]
        P = np.concatenate(h["parts"])
        dz = (float(rest_height_m) - float(P[:, 2].min())) if drop_to_table else 0.0
        a = math.radians(-sgn * float(yaw_deg))
        Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        off = np.array([sgn * float(spread_m), 0.0, dz])
        out.extend([(Q - c) @ Rz.T + c + off for Q in h["parts"]])
    return out


def render_silhouette(cam, parts, near_m=0.02) -> np.ndarray:
    """Binary silhouette (uint8 0/1, cam['height'] x cam['width']) of convex parts given as world point sets."""
    import cv2
    from tool_library.common import rigcam as RC
    img = np.zeros((int(cam["height"]), int(cam["width"])), np.uint8)
    if not parts:
        return img
    sizes = [len(P) for P in parts]
    uv, z = RC.project(cam, np.concatenate(parts))
    i = 0
    for n in sizes:
        u, zz = uv[i:i + n], z[i:i + n]
        i += n
        if (zz < near_m).any() or not np.all(np.isfinite(u)) or np.abs(u).max() > 1e5:
            continue
        hp = cv2.convexHull(np.round(u * 4).astype(np.int32))
        cv2.fillConvexPoly(img, hp, 1, shift=2)
    return img


# --------------------------------------------------------------------- reach --
def reach_field(scene_xml, sides=("left", "right"), z_levels=(0.10,), x_range=(-0.5, 0.5), y_range=(0.05, 0.9), step_m=0.05,
                orientation="palm_down", pos_tol_m=0.02, tilt_tol_deg=20.0):
    """Where each station's arm can hold the hand in `orientation` (arm_workspace_map's names) at each height:
    a grid of IK residuals {'x', 'y', 'z_levels', 'pos_mm' {side: [nz, ny, nx]}, 'rot_deg' {...}}. Each cell is
    solved from the neighbouring cells' solutions, the home pose and the five nearest configurations of the arm's
    seed library (the multistart object_reach_check uses, without its slow last-resort restarts), so a cell marked
    reachable is reachable and a few cells at the band's edge may be missed (the field is conservative); the final
    placement is re-checked with the exact solver."""
    import mujoco
    from tool_library.common import armhand as AH
    from tool_library.verifiers.arm_workspace_map import named_orientation
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    q0 = np.array(m.key_qpos[0] if m.nkey else m.qpos0, float)
    xs = np.round(np.arange(x_range[0], x_range[1] + 1e-9, step_m), 4)
    ys = np.round(np.arange(y_range[0], y_range[1] + 1e-9, step_m), 4)
    zs = [float(z) for z in z_levels]
    pos, rot = {}, {}
    for side in sides:
        S = AH.ArmSide(m, side)
        R = named_orientation(side, orientation)
        Rt = R @ S.R_m
        P = np.full((len(zs), len(ys), len(xs)), np.nan)
        A = np.full_like(P, np.nan)
        for iz, z in enumerate(zs):
            prev_row = None
            for iy, y in enumerate(ys):
                order = range(len(xs)) if iy % 2 == 0 else range(len(xs) - 1, -1, -1)
                prev = None
                row = {}
                for ix in order:
                    pt = np.array([xs[ix], y, z]) + R @ S.p_m
                    seeds = [S.home] + S.nearest_seeds(pt, Rt, k=5)
                    if prev is not None:
                        seeds.append(prev)
                    if prev_row is not None and ix in prev_row:
                        seeds.append(prev_row[ix])
                    best = None
                    for sd in seeds:
                        ep, er, qa = S._ik(q0, pt, Rt, np.asarray(sd, float), S.cfg["ik_iters"])
                        if best is None or ep + 0.05 * er < best[0] + 0.05 * best[1]:
                            best = (ep, er, qa)
                        if ep <= 0.002 and er <= math.radians(1.0):
                            break
                    P[iz, iy, ix], A[iz, iy, ix] = 1000 * best[0], math.degrees(best[1])
                    prev = best[2] if (best[0] <= pos_tol_m and best[1] <= math.radians(tilt_tol_deg)) else prev
                    row[ix] = best[2]
                prev_row = row
        pos[side], rot[side] = P.tolist(), A.tolist()
    return {"x": xs.tolist(), "y": ys.tolist(), "z_levels": zs, "orientation": orientation, "pos_tol_m": pos_tol_m,
            "tilt_tol_deg": tilt_tol_deg, "pos_mm": pos, "rot_deg": rot, "scene": str(scene_xml), "step_m": step_m}


def field_margin_maps(field: dict, res_m=0.01):
    """Signed distance (m) to the edge of each side's reachable region, per height, on a res_m grid: positive inside
    (how far the work may move before it leaves the band), negative outside. Returns (xs, ys, {side: [nz, ny, nx]})."""
    import cv2
    xs0, ys0 = np.asarray(field["x"]), np.asarray(field["y"])
    xs = np.arange(xs0[0], xs0[-1] + 1e-9, res_m)
    ys = np.arange(ys0[0], ys0[-1] + 1e-9, res_m)
    out = {}
    for side, P in field["pos_mm"].items():
        P = np.asarray(P, float)
        A = np.asarray(field["rot_deg"][side], float)
        ok0 = ((P <= 1000 * field["pos_tol_m"]) & (A <= field["tilt_tol_deg"])).astype(np.float32)
        maps = []
        for iz in range(P.shape[0]):
            # bilinear upsampling of the reachable indicator, then the 0.5 level as the band's edge
            up = cv2.resize(ok0[iz], (len(xs), len(ys)), interpolation=cv2.INTER_LINEAR) >= 0.5
            inside = cv2.distanceTransform(up.astype(np.uint8), cv2.DIST_L2, 5) * res_m
            outside = cv2.distanceTransform((~up).astype(np.uint8), cv2.DIST_L2, 5) * res_m
            maps.append(np.where(up, inside, -outside))
        out[side] = np.array(maps)
    return xs, ys, out


def sample_map(xs, ys, M, pts):
    """Bilinear lookup of a 2D map M[ny, nx] at points [N, 2]; points off the grid read the nearest edge value
    minus their distance past the edge (so a point far outside stays outside)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    fx = (pts[:, 0] - xs[0]) / dx
    fy = (pts[:, 1] - ys[0]) / dy
    over = np.hypot(np.clip(-fx, 0, None) + np.clip(fx - (len(xs) - 1), 0, None),
                    np.clip(-fy, 0, None) + np.clip(fy - (len(ys) - 1), 0, None)) * dx
    fx = np.clip(fx, 0, len(xs) - 1.000001)
    fy = np.clip(fy, 0, len(ys) - 1.000001)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - x0, fy - y0
    v = (M[y0, x0] * (1 - tx) * (1 - ty) + M[y0, x0 + 1] * tx * (1 - ty) + M[y0 + 1, x0] * (1 - tx) * ty
         + M[y0 + 1, x0 + 1] * tx * ty)
    return v - over


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    return str(path)
