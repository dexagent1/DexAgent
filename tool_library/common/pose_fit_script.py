"""
pose_fit_script.py — 6-DoF object pose optimizer (vendored for tool_library.skills.fit_object_poses).

Fits the SAM3D mesh (reconstructed ONCE at the init frame) into every video frame by
optimizing per-frame 6-DoF pose (rotation + translation) until two losses are minimized:

  L_mask   MASK CONTROL: the Stage-2 mask converted to pixels; symmetric 2D chamfer between
           the projected mesh surface points and the mask pixels — the projected mesh must
           occupy exactly the mask's appearance (both directions: mesh inside mask AND mask
           covered by mesh). The mask is AMODALLY COMPLETED first: enclosed holes are filled
           (an object resting ON this one — food on a tray — or held in front of it carves
           holes the solid mesh silhouette can never reproduce; on a tray holding food the
           un-filled holes were 11% of the mask). The RAW mask keeps gating the depth term, so hole
           pixels (the occluder's depth, not ours) never enter L_depth.
  L_depth  DEPTH CONTROL: metric Video-Depth-Anything depth turned into INVERSE depth
           (foreground differences become distinct); Huber between the camera-facing mesh
           points' inverse depth and the scene's inverse depth at their projected pixels
           (only where they land inside the object's own mask — occluders don't pollute).
  L_grav   GRAVITY CONTROL: a rounded object's silhouette barely changes under tilt, so
           mask+depth admit tilted / half-suspended poses. GeoCalib gives per-frame world-up
           in camera coords (gravity.json); the object's own up-axis u_obj is AUTO-DETECTED
           at the init frame (candidate rotations are built gravity-aligned — each stands the
           mesh on one canonical axis; mask+depth pick which one, e.g. cup=+z, saucer=+y);
           then L_grav = 1 - dot(R u_obj, up) holds the object standing straight everywhere.

Per-frame parameters are ONLY 6-DoF (R as a 6D continuous repr, t). A single GLOBAL scale is
solved once at the init frame (the SAM3D mesh is unit-normalized; without it the two losses
would contradict each other) and then frozen. Init frame: ALL 24 gravity-aligned rotation
candidates get a cheap t-only warm start and are then EACH refined with rotation+scale free —
no top-K truncation. (A cheap-prescore top-3 cut picks a perpendicular tray: for
a near-square object the pre-refine chamfer misranks the 90-deg yaw family, while AFTER
refinement the correct yaw wins the very same loss by ~2x. Ranking is only trustworthy after
refinement.) Every other frame warm-starts from its neighbor (translation shifted by the 3D
mask-centroid displacement) + short refine (+ a small prior-to-neighbor term against jitter).

Robustness under occlusion + anti-shake (three layers):
  - trust gating during the fit: trust = visible-mask fraction vs init. Below
    --trust-freeze-rot the ROTATION is frozen (a partial mask cannot tell how the object is
    rotated — this is what used to make it spin); below --trust-freeze-all the frame is not
    fit at all and coasts on the centroid-shifted carry.
  - STATIC-UNDER-OCCLUSION FREEZE (2 conditions, BOTH required): (1) the current mask still
    majorly fits inside the previous pose's silhouette (containment >= --freeze-contain) and
    (2) an occluder — the Stage-1 hand layers or another object's mask — is on/next to the
    object, with any sizeable missing silhouette region covered by it (--freeze-explain).
    Then the object is surely occluded but hasn't moved: the pose is carried VERBATIM (no
    centroid shift) and the frame becomes an anchor. Self-releasing: once the object really
    moves, condition 1 fails and normal fitting resumes.
  - anchor-and-interpolate: per-frame confidence = trust x IoU; frames below --anchor-conf
    (and not frozen) are REPLACED by interpolation between anchors (slerp/linear) — bad fits
    never leak into the output.
  - OSCILLATION RESILIENCY (--osc-win): the deviate-and-RETURN wobble (mesh drifts off and
    snaps back while an arm sweeps over the mask) re-enters through a small anchor ISLAND —
    a few frames that re-clear --anchor-conf between two occlusion gaps while the mask has
    only half-recovered. Islands (<= --osc-island anchors, window <= --osc-win frames) that
    sit > --osc-trig-mm / --osc-trig-deg off the bridge between the stable anchors on either
    side are demoted and the whole window interpolates from those shores: the pose before,
    during and after the occlusion stays consistent. Real motion keeps long confident anchor
    runs and is never flagged.
  - --smooth-win: 3-frame median + zero-lag SavGol on the final T_seq (flicker/shake).

Outputs <out-dir>/<obj>.npz {T_seq (N,4,4), scale, center, ious}:
  verts_cam(t) = scale * (V - center) @ T_seq[t,:3,:3].T + T_seq[t,:3,3]     (OpenCV cam frame)

ENV: torch (CUDA recommended) + opencv-python + trimesh + scipy + pillow.
RUN: through the fit_object_poses skill (tool_library.skills.fit_object_poses), which launches
     python pose_fit_script.py --frames-dir D --masks-dir D --mesh-dir D --depth-npz P
                               --objects cup,saucer --out-dir D --focal F
                               [--gravity G --hand-layers D --max-frames N --init-frame 0
                                --iters-init 400 --iters-step 150 --w-mask 6.0 --w-depth 3.0
                                --anchor-conf 0.25 --smooth-win 5 ...]
"""
import argparse, glob, json, os, re
import numpy as np
import cv2
import torch
import trimesh
from PIL import Image
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def slug(s):
    """filename-safe object name ("coffee cup" -> coffee_cup) — matches tool_library.common.video.slug."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

ap = argparse.ArgumentParser()
ap.add_argument("--frames-dir", required=True)
ap.add_argument("--masks-dir", required=True, help="palette label maps frame*.png; result.json beside or above it maps prompt->label")
ap.add_argument("--labels", default="", help="JSON {name: label}; default read from result.json next to masks-dir")
ap.add_argument("--mesh-dir", required=True, help="dir holding <obj>.glb|.obj|.ply or <obj>/<obj>.obj")
ap.add_argument("--depth-npz", required=True)
ap.add_argument("--objects", required=True)
ap.add_argument("--out-dir", required=True)
ap.add_argument("--focal", type=float, required=True)
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--init-frame", type=int, default=0)
ap.add_argument("--iters-init", type=int, default=400)
ap.add_argument("--iters-step", type=int, default=150)
ap.add_argument("--w-mask", type=float, default=6.0)
ap.add_argument("--w-depth", type=float, default=3.0)
ap.add_argument("--w-temp", type=float, default=1.0)
ap.add_argument("--trust-freeze-rot", type=float, default=0.5,
                help="mask visibility below which ROTATION is frozen (unobservable from a sliver)")
ap.add_argument("--trust-freeze-all", type=float, default=0.2,
                help="visibility below which the frame is not fit at all (coasts on centroid shift)")
ap.add_argument("--anchor-conf", type=float, default=0.25,
                help="min per-frame confidence (visibility x IoU) to anchor the interpolation pass")
ap.add_argument("--smooth-win", type=int, default=0,
                help="median+SavGol window on the final T_seq (odd frames; 0 = off)")
ap.add_argument("--osc-win", type=int, default=20,
                help="oscillation resiliency: max frames a deviate-and-return wobble may span "
                     "before it is trusted as real motion (0 = off)")
ap.add_argument("--osc-island", type=int, default=4,
                help="max consecutive anchors of an 'island' between two occlusion gaps that "
                     "may be demoted as oscillation")
ap.add_argument("--osc-trig-mm", type=float, default=40.0,
                help="min island deviation (mm) from the stable-shore bridge to call it an oscillation")
ap.add_argument("--osc-trig-deg", type=float, default=5.0,
                help="min island rotation deviation (deg) from the bridge to call it an oscillation")
ap.add_argument("--osc-return-mm", type=float, default=25.0,
                help="world-mode wobble pass: max WORLD endpoint distance (mm) to say the pose 'returned'")
ap.add_argument("--osc-return-deg", type=float, default=4.0,
                help="world-mode wobble pass: max WORLD endpoint rotation (deg) to say the pose 'returned'")
ap.add_argument("--camera-hdf5", default="",
                help="EgoDex-style hdf5 with GT camera extrinsics (transforms/camera, Nx4x4 cam->world). "
                     "When given, anchor interpolation + oscillation logic run in WORLD frame: a static "
                     "object stays pinned to the scene instead of drifting on the camera-frame bridge, "
                     "and deviate-and-return wobbles are detected even without occlusion gaps")
ap.add_argument("--hand-layers", default="",
                help="Stage-1 hand RGBA layer dir (frameNNNNN.png) — occluder cue for the freeze rule")
ap.add_argument("--freeze-contain", type=float, default=0.85,
                help="freeze rule cond 1: min fraction of the current mask inside the prev-pose silhouette")
ap.add_argument("--freeze-explain", type=float, default=0.5,
                help="freeze rule cond 2: min fraction of the MISSING silhouette covered by occluders")
ap.add_argument("--gravity", default="",
                help="gravity.json (GeoCalib per-frame roll/pitch) — enables the gravity loss")
ap.add_argument("--w-grav", type=float, default=2.0,
                help="weight of L_grav = 1 - dot(R u_obj, world-up): kills tilted/half-suspended poses")
ap.add_argument("--w-axis", type=float, default=1.0,
                help="init-candidate penalty x rotational-ASYMMETRY of the axis stood on: tableware "
                     "stands on its symmetry axis; breaks the standing-vs-lying tie the mask can't see")
ap.add_argument("--up-flip", default="",
                help="csv of objects whose up-axis SIGN gets negated (fit came out upside-down; "
                     "the mask cannot see the 180-deg flip, so it is a manual switch)")
ap.add_argument("--up-force", default="",
                help="csv of objects whose up-axis SIGN is FORCED +1 (canonical axis up): the fit "
                     "runs as usual but a -1 init pose is 180-deg flipped; overrides --up-flip")
args = ap.parse_args()

UP_FLIP = {s.strip() for s in args.up_flip.split(",") if s.strip()}
UP_FORCE = {s.strip() for s in args.up_force.split(",") if s.strip()}

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PTS, N_MASK = 1500, 1500
HUBER_DELTA = 0.2            # inverse-depth units (1/m)
MIN_MASK_PX = 30

# ---------------------------------- inputs ----------------------------------
frame_files = sorted(glob.glob(f"{args.frames_dir}/frame*.jpg") + glob.glob(f"{args.frames_dir}/frame*.png"))
mask_files = sorted(glob.glob(f"{args.masks_dir}/frame*.png"))
depths = np.load(args.depth_npz)["depths"].astype(np.float32)          # (N,H,W) meters
NF = min(len(frame_files), len(mask_files), len(depths))
if args.max_frames:
    NF = min(NF, args.max_frames)
H, W = cv2.imread(frame_files[0]).shape[:2]
FX = FY = args.focal
CX, CY = W / 2.0, H / 2.0
DIAG = float(np.hypot(H, W))
INIT = min(args.init_frame, NF - 1)

# GT camera extrinsics (cam->world, one 4x4 per frame) -> world-frame interpolation/oscillation
CAMS = CAMS_INV = None
if args.camera_hdf5:
    import h5py
    with h5py.File(args.camera_hdf5, "r") as _h5:
        _cam = _h5["transforms/camera"][:].astype(np.float64)
    assert len(_cam) >= NF, f"camera hdf5 has {len(_cam)} frames < {NF} video frames"
    CAMS = _cam[:NF]
    CAMS_INV = np.linalg.inv(CAMS)
    print(f"[camera] {args.camera_hdf5}: {NF} GT extrinsics -> WORLD-frame interpolation", flush=True)

if args.labels:
    label_of = {k: int(v) for k, v in json.loads(args.labels).items()}
else:
    _rj = next((p for p in (f"{args.masks_dir}/result.json", f"{os.path.dirname(args.masks_dir.rstrip('/'))}/result.json")
                if os.path.exists(p)), None)
    assert _rj, "pass --labels or keep result.json next to the masks"
    label_of = {o["prompt"]: o["label"] for o in json.load(open(_rj))["objects"]}
OBJECTS = [o for o in args.objects.split(",") if o]
print(f"[fit] {NF} frames {W}x{H}  focal={args.focal:.1f}  objects={OBJECTS} labels={label_of}", flush=True)

pal = np.stack([np.array(Image.open(mask_files[t])) for t in range(NF)])        # (N,H,W) label ids
inv_maps = torch.tensor(1.0 / np.clip(depths[:NF], 0.05, 50.0), device=DEV)   # inverse depth (bright=near)

# per-frame WORLD-UP in the OpenCV camera frame, from GeoCalib roll/pitch. For a camera
# pitched down by theta, up must be (0,-cos t,-sin t) — exactly
# GeoCalib's vec3d, i.e. [-sin(r)cos(p), -cos(r)cos(p), sin(p)] normalized.
UPS = None
if args.gravity and os.path.exists(args.gravity):
    gj = json.load(open(args.gravity))
    pf = gj.get("per_frame", [])
    if pf and "frame" in pf[0] and isinstance(pf[0]["frame"], int):   # entries carry their frame index
        _by = {int(e["frame"]): e for e in pf}
        pf = [_by.get(t, {}) for t in range(max(_by) + 1)]

    def _mov_avg(x, w=5):
        h = w // 2
        return np.array([x[max(0, i - h):i + h + 1].mean() for i in range(len(x))])

    rolls = _mov_avg(np.array([(pf[t]["roll_deg"] if t < len(pf) and "roll_deg" in pf[t] and not pf[t].get("outlier")
                                else gj["roll_deg"]) for t in range(NF)], float))
    pitches = _mov_avg(np.array([(pf[t]["pitch_deg"] if t < len(pf) and "pitch_deg" in pf[t] and not pf[t].get("outlier")
                                  else gj["pitch_deg"]) for t in range(NF)], float))

    def _up(roll, pitch):
        r, p = np.radians(roll), np.radians(pitch)
        v = np.array([-np.sin(r) * np.cos(p), -np.cos(r) * np.cos(p), np.sin(p)])
        return v / np.linalg.norm(v)

    UPS = torch.tensor(np.stack([_up(rolls[t], pitches[t]) for t in range(NF)]),
                       dtype=torch.float32, device=DEV)
    print(f"[fit] gravity: up@init={UPS[min(args.init_frame, NF - 1)].cpu().numpy().round(3)}  "
          f"w_grav={args.w_grav}", flush=True)


def cube24():
    """the 24 proper rotations of the cube (signed permutation matrices with det=1)."""
    from itertools import permutations, product
    out = []
    for perm in permutations(range(3)):
        for signs in product([1.0, -1.0], repeat=3):
            R = np.zeros((3, 3), np.float32)
            for i, (j, sg) in enumerate(zip(perm, signs)):
                R[i, j] = sg
            if np.isclose(np.linalg.det(R), 1.0):
                out.append(R)
    return out


CUBE24 = cube24()


def sixd_to_R(x):
    a, b = x[..., :3], x[..., 3:]
    b1 = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = b - (b1 * b).sum(-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], -1)


def R_to_sixd(R):
    return torch.cat([R[:, 0], R[:, 1]], -1)


def project(P):
    z = P[:, 2].clamp(min=1e-4)
    return torch.stack([FX * P[:, 0] / z + CX, FY * P[:, 1] / z + CY], 1), z


def grid_sample_map(m, uv, mode):
    u = (uv[:, 0].clamp(0, W - 1) / (W - 1)) * 2 - 1
    v = (uv[:, 1].clamp(0, H - 1) / (H - 1)) * 2 - 1
    g = torch.stack([u, v], 1)[None, None]
    return torch.nn.functional.grid_sample(m[None, None], g, align_corners=True, mode=mode)[0, 0, 0]


def fill_holes(mb):
    """Amodal mask completion: fill regions fully ENCLOSED by the mask. Another object
    resting on / held in front of this one carves holes into the visible mask that the
    projected SOLID mesh silhouette can never reproduce — without filling, those holes
    bias the chamfer (and IoU/trust/containment) toward poses that dodge them. Padding
    makes the background border-connected, so objects touching the frame edge are safe.
    (Genuine see-through holes — a mug-handle loop — get filled too; they are small and
    cost the chamfer far less than a resting-object hole costs it unfilled.)"""
    m = np.pad(mb.astype(np.uint8), 1)
    ff = m.copy()
    cv2.floodFill(ff, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    return mb | ((ff[1:-1, 1:-1] == 0) & ~mb)


def frame_obs(label, t):
    """(mask pixel samples (n,2), hole-FILLED bool mask, RAW bool mask) — or (None,)*3 if
    the object is absent this frame. The filled mask drives all silhouette geometry
    (chamfer targets, trust, containment, IoU); the raw mask keeps gating the depth term
    so occluder pixels inside holes never pollute it."""
    mb_raw = pal[t] == label
    if mb_raw.sum() < MIN_MASK_PX:
        return None, None, None
    mb = fill_holes(mb_raw)
    ys, xs = np.nonzero(mb)
    idx = np.random.choice(len(xs), min(N_MASK, len(xs)), replace=False)
    mpts = torch.tensor(np.stack([xs[idx], ys[idx]], 1), dtype=torch.float32, device=DEV)
    return mpts, mb, mb_raw


def losses(pts, nrm, rot6, t, log_s, mpts, invd, maskf, up_t, trust=1.0, u_obj=None):
    """(L_mask, L_depth, L_grav) for one frame at the given pose. maskf = float mask (H,W) on DEV.
    trust = visible-mask fraction vs the init frame: under occlusion the points->mask chamfer
    direction would drag the WHOLE mesh into the remaining sliver, so it is down-weighted;
    the mask->points direction (every mask pixel explained by the mesh) is always safe.
    L_grav = 1 - dot(R u_obj, world-up): the silhouette of a rounded object barely changes
    under tilt (mask CANNOT see it), so gravity supplies the missing constraint — tilted /
    half-suspended poses that still fit the mask are pushed back to the standing pose."""
    R = sixd_to_R(rot6[None])[0]
    P = torch.exp(log_s) * pts @ R.T + t
    uv, z = project(P)
    d = torch.cdist(uv, mpts)
    l_mask = (trust * d.min(1).values.mean() + d.min(0).values.mean()) / DIAG
    l_grav = (1.0 - (R @ u_obj) @ up_t) if (u_obj is not None and up_t is not None) \
        else torch.zeros((), device=DEV)
    # depth: camera-facing points whose projection lands INSIDE this object's mask
    n_cam = nrm @ R.T
    front = (n_cam * P).sum(1) < 0
    inside = grid_sample_map(maskf, uv.detach(), "nearest") > 0.5
    sel = front & inside
    if sel.sum() < 20:
        return l_mask, torch.zeros((), device=DEV), l_grav
    obs = grid_sample_map(invd, uv[sel], "bilinear")
    l_depth = torch.nn.functional.huber_loss(1.0 / z[sel], obs, delta=HUBER_DELTA)
    return l_mask, l_depth, l_grav


def refine(pts, nrm, rot6, t, log_s, obs, iters, lr, opt_scale, prev=None, trust=1.0,
           opt_rot=True, u_obj=None):
    """Adam-refine (t [, rot6] [, log_s]) against L_mask + L_depth + L_grav (+ prior to prev)."""
    mpts, invd, maskf, up_t = obs
    rot6 = rot6.clone().requires_grad_(opt_rot)
    t = t.clone().requires_grad_(True)
    log_s = log_s.clone().requires_grad_(opt_scale)
    params = [t] + ([rot6] if opt_rot else []) + ([log_s] if opt_scale else [])
    opt = torch.optim.Adam(params, lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.1)
    for _ in range(iters):
        opt.zero_grad()
        lm, ld, lg = losses(pts, nrm, rot6, t, log_s, mpts, invd, maskf, up_t, trust, u_obj)
        loss = args.w_mask * lm + args.w_depth * ld + args.w_grav * lg
        if prev is not None:
            r6p, tp = prev
            loss = loss + args.w_temp * (25.0 * ((t - tp) ** 2).sum() + 0.5 * ((rot6 - r6p) ** 2).mean())
        loss.backward()
        opt.step()
        sch.step()
    with torch.no_grad():
        lm, ld, lg = losses(pts, nrm, rot6, t, log_s, mpts, invd, maskf, up_t, trust, u_obj)
    return rot6.detach(), t.detach(), log_s.detach(), \
        float(args.w_mask * lm + args.w_depth * ld + args.w_grav * lg)


def smooth_Tseq(T_seq, win):
    """anti-flicker on the final pose sequence: 3-frame median (spike kill) + zero-lag
    SavGol(poly 2), translation directly, rotation via sign-aligned quaternions."""
    n = len(T_seq)
    w = min(win, n if n % 2 else n - 1)
    if w < 5:
        return T_seq
    tr = median_filter(T_seq[:, :3, 3], size=(3, 1), mode="nearest")
    tr = savgol_filter(tr, w, 2, axis=0, mode="interp")
    q = Rotation.from_matrix(T_seq[:, :3, :3]).as_quat()
    for i in range(1, n):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    q = median_filter(q, size=(3, 1), mode="nearest")
    q = savgol_filter(q, w, 2, axis=0, mode="interp")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = T_seq.copy()
    out[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    out[:, :3, 3] = tr
    return out


def sil_iou(pts, R, t, s, gtmask):
    P = s * pts @ R.T + t
    uv, _ = project(P)
    uv = uv.detach().cpu().numpy().astype(int)
    ren = np.zeros((H, W), np.uint8)
    ren[np.clip(uv[:, 1], 0, H - 1), np.clip(uv[:, 0], 0, W - 1)] = 1
    ren = cv2.dilate(ren, np.ones((7, 7), np.uint8)) > 0
    return (ren & gtmask).sum() / max((ren | gtmask).sum(), 1)


def splat_sil(pts, R, t, s, dil=9):
    """binary (amodal) silhouette of the mesh at a pose — point splat + dilation."""
    P = s * pts @ R.T + t
    uv, _ = project(P)
    uv = uv.detach().cpu().numpy().astype(int)
    ren = np.zeros((H, W), np.uint8)
    ren[np.clip(uv[:, 1], 0, H - 1), np.clip(uv[:, 0], 0, W - 1)] = 1
    return cv2.dilate(ren, np.ones((dil, dil), np.uint8)).astype(bool)


_hand_cache = {}


def hand_mask(t):
    """Stage-1 hand layer alpha for window frame t (None if unavailable)."""
    if not args.hand_layers:
        return None
    if t not in _hand_cache:
        fi = int(re.search(r"frame(\d+)", os.path.basename(frame_files[t])).group(1))
        p = f"{args.hand_layers}/frame{fi:05d}.png"
        m = None
        if os.path.exists(p):
            im = np.array(Image.open(p))
            if im.ndim == 3 and im.shape[-1] == 4:
                a = im[..., 3]
                if a.shape != (H, W):
                    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
                m = a > 127
        _hand_cache[t] = m
    return _hand_cache[t]


def occluder_map(t, label):
    """everything that can hide this object at frame t: other objects' masks + the hands."""
    occ = (pal[t] != 0) & (pal[t] != label)
    hm = hand_mask(t)
    return occ | hm if hm is not None else occ


FREEZE_PROX = max(5, int(0.015 * DIAG))            # "very close" radius (px)


def check_freeze(mb, occ, prev_sil):
    """Static-under-occlusion rule — BOTH conditions must hold to freeze the pose:
      1. the current mask still MAJORLY FITS the previous pose's silhouette (containment,
         not IoU — an occluded mask shrinks, so IoU would punish a perfectly static object)
      2. an occluder (hand / other object) is ON or NEXT TO the object, and any sizeable
         missing part of the silhouette is covered by it
    -> the object is surely occluded but has not moved: carry the pose verbatim."""
    inside = (mb & prev_sil).sum() / max(mb.sum(), 1)
    if inside < args.freeze_contain:                # cond 1: mask no longer fits -> it moved
        return False
    k = np.ones((2 * FREEZE_PROX + 1, 2 * FREEZE_PROX + 1), np.uint8)
    near = cv2.dilate(mb.astype(np.uint8), k).astype(bool)
    if (near & occ).sum() < 20:                     # cond 2a: no occluder in contact/close
        return False
    missing = prev_sil & ~mb
    if missing.sum() / max(prev_sil.sum(), 1) <= 0.15:
        return True                                 # barely occluded: fits + occluder adjacent
    return (missing & occ).sum() / max(missing.sum(), 1) >= args.freeze_explain


os.makedirs(args.out_dir, exist_ok=True)
done = 0
total = len(OBJECTS) * NF
for obj in OBJECTS:
    label = label_of.get(obj)
    sl = slug(obj)
    _cands = [f"{args.mesh_dir}/{sl}{e}" for e in (".glb", ".obj", ".ply")] + [f"{args.mesh_dir}/{sl}/{sl}.obj", f"{args.mesh_dir}/{obj}.glb"]
    _mp = next((c for c in _cands if os.path.exists(c)), None)
    assert _mp, f"{obj}: no mesh among {_cands}"
    mesh = trimesh.load(_mp, force="mesh")
    samp, fidx = trimesh.sample.sample_surface(mesh, N_PTS)
    center = np.asarray(mesh.vertices).mean(0)
    pts = torch.tensor(np.asarray(samp) - center, dtype=torch.float32, device=DEV)
    nrm = torch.tensor(np.asarray(mesh.face_normals)[fidx], dtype=torch.float32, device=DEV)

    # rotational-symmetry chamfer per canonical axis (rotate samples 90deg about the axis,
    # distance to the original set): tableware STANDS on its symmetry axis. Used as a soft
    # init-candidate prior — decisive for cups/lids (3-10x gap), ~flat (harmless) for
    # asymmetric or degenerate meshes. NOTE: bbox extents are useless for this (the EgoDex
    # cup's bbox is a near-perfect cube); actual symmetry is what identifies "up".
    d_sym = []
    for k in range(3):
        Rk = np.eye(3, dtype=np.float32)
        i, j = [a for a in range(3) if a != k]
        Rk[i, i] = 0; Rk[j, j] = 0; Rk[i, j] = -1.0; Rk[j, i] = 1.0
        q = pts @ torch.tensor(Rk, device=DEV).T
        d = torch.cdist(q, pts)
        d_sym.append(float(d.min(1).values.mean() + d.min(0).values.mean()) / 2)
    d_sym = np.array(d_sym) - min(d_sym)
    print(f"[{obj}] axis asymmetry xyz={d_sym.round(3)} (symmetry axis = {'xyz'[int(np.argmin(d_sym))]})", flush=True)

    # ---- init frame: translation/scale from mask+depth, rotation by 24-way search ----
    mpts0, mb0, mb0_raw = frame_obs(label, INIT)
    assert mpts0 is not None, f"{obj}: no mask at init frame {INIT}"
    if mb0.sum() > mb0_raw.sum():
        print(f"[{obj}] init mask holes filled: +{int(mb0.sum() - mb0_raw.sum())} px "
              f"({(mb0.sum() / mb0_raw.sum() - 1) * 100:.1f}%)", flush=True)
    maskf0 = torch.tensor(mb0_raw.astype(np.float32), device=DEV)   # depth gate: RAW mask
    up0 = UPS[INIT] if UPS is not None else None
    obs0 = (mpts0, inv_maps[INIT], maskf0, up0)
    ys, xs = np.nonzero(mb0)                                   # filled: true centroid/extent
    ys_r, xs_r = np.nonzero(mb0_raw)                           # raw: depth is only valid here
    z0 = float(np.median(depths[INIT][ys_r, xs_r]))
    t0 = torch.tensor([(xs.mean() - CX) * z0 / FX, (ys.mean() - CY) * z0 / FY, z0],
                      dtype=torch.float32, device=DEV)
    r_px = float(np.sqrt(mb0.sum() / np.pi)) * 1.35            # mask radius (pad: mask misses depth extent)
    r_m = float(pts.norm(dim=1).median())
    log_s0 = torch.tensor(np.log(max(r_px * z0 / FX, 1e-4) / max(r_m, 1e-6)), dtype=torch.float32, device=DEV)
    print(f"[{obj}] init f{INIT}: z={z0:.3f}m  scale~{float(torch.exp(log_s0)):.4f}  "
          f"searching 24 rotations", flush=True)

    # Candidate rotations: with gravity, build them GRAVITY-ALIGNED — an orthonormal basis
    # whose 3rd axis is the measured world-up, composed with the 24 cube rotations. Every
    # candidate then STANDS the mesh on one of its canonical axes (upright or lying flat,
    # any 90-deg yaw); the tilted-but-mask-perfect continuum is excluded from the search.
    # Mask+depth pick WHICH axis is up; that axis becomes u_obj for the gravity loss.
    if up0 is not None:
        u = up0.cpu().numpy()
        b1 = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
        b1 /= np.linalg.norm(b1)
        B = np.stack([b1, np.cross(u, b1), u], 1).astype(np.float32)   # columns: [horiz, horiz, up]
        cand_Rs = [B @ Rc for Rc in CUBE24]
    else:
        cand_Rs = CUBE24

    def snap_u_obj(Rc):
        """(axis vector, axis index) this candidate stands on: argmax |component| of R^T up."""
        if up0 is None:
            return None, None
        v = Rc.T @ up0
        k = int(torch.argmax(v.abs()))
        e = torch.zeros(3, device=DEV)
        e[k] = torch.sign(v[k])
        return e, k

    # EVERY candidate gets a cheap t-only warm start AND a full rotation+scale refine; the
    # winner is picked from REFINED losses only. Ranking by the cheap fit and refining just
    # the top-3 fails: pre-refinement the chamfer cannot rank silhouette look-alikes (a
    # near-square tray's correct yaw ranked 5th-8th behind its own 90-deg rotations, while
    # after refinement it won the same loss by ~2x).
    best = None
    for Rc in cand_Rs:
        r6 = R_to_sixd(torch.tensor(Rc, device=DEV))
        _, tc, _, _ = refine(pts, nrm, r6, t0, log_s0, obs0, 40, 0.02, False,
                             opt_rot=False)                    # t-only warm start
        uo, k = snap_u_obj(sixd_to_R(r6[None])[0])
        pen = args.w_axis * d_sym[k] if k is not None else 0.0  # standing on an asymmetric axis
        r6, tc, ls, l = refine(pts, nrm, r6, tc, log_s0, obs0, 120, 0.01, True, u_obj=uo)
        if best is None or l + pen < best[0]:
            best = (l + pen, r6, tc, ls, uo)
    _, r6, t, log_s, u_obj = best
    variants = [(r6, t, log_s, u_obj, "as-selected")]
    if u_obj is not None and any(k in obj or k in sl for k in UP_FLIP):
        # 180-deg flip about a horizontal canonical axis: same silhouette, opening
        # inverted. The mask alone cannot see the sign, so for --up-flip objects
        # BOTH signs are fully polished and the better mask+depth fit wins — the
        # same cup may be upright in one clip and upside-down in another.
        k = int(torch.argmax(u_obj.abs()))
        j = (k + 1) % 3
        R180 = -torch.eye(3, device=DEV)
        R180[j, j] = 1.0
        variants.append((R_to_sixd(sixd_to_R(r6[None])[0] @ R180), t, log_s,
                         -u_obj, "flipped"))
    if u_obj is not None and any(k in obj or k in sl for k in UP_FORCE):
        # sign is DECREED +1 (canonical axis up), not fitted — a bowl silhouettes the
        # same either way but is never upside-down on a table. Keep only +sign
        # variants; if the search came out -1, 180-deg flip it (silhouette unchanged).
        k = int(torch.argmax(u_obj.abs()))
        variants = [v for v in variants if float(v[3][k]) > 0]
        if not variants:
            j = (k + 1) % 3
            R180 = -torch.eye(3, device=DEV)
            R180[j, j] = 1.0
            variants = [(R_to_sixd(sixd_to_R(r6[None])[0] @ R180), t, log_s,
                         -u_obj, "forced+1")]
        print(f"[{obj}] up-axis sign FORCED +1 (--up-force)", flush=True)
    polished = []
    for r6v, tv, lsv, uov, tag in variants:
        r6v, tv, lsv, lv = refine(pts, nrm, r6v, tv, lsv, obs0, args.iters_init,
                                  0.005, True, u_obj=uov)
        polished.append((lv, r6v, tv, lsv, uov, tag))
        if len(variants) > 1:
            print(f"[{obj}] up-axis sign candidate '{tag}': loss={lv:.4f}", flush=True)
    l, r6, t, log_s, u_obj, tag = min(polished, key=lambda p: p[0])
    if len(polished) > 1:
        margin = abs(polished[0][0] - polished[1][0])
        print(f"[{obj}] up-axis sign AUTO-chosen: {tag} (loss margin {margin:.4f}"
              f"{' — LOW confidence' if margin < 0.005 else ''})", flush=True)
    tilt = ""
    if u_obj is not None:
        d0 = float((sixd_to_R(r6[None])[0] @ u_obj) @ up0)
        tilt = f"  up-axis={u_obj.cpu().numpy().round().astype(int)}  tilt={np.degrees(np.arccos(np.clip(d0, -1, 1))):.1f}deg"
    print(f"[{obj}] init pose loss={l:.4f}  scale={float(torch.exp(log_s)):.4f}{tilt} (frozen from here)", flush=True)

    # ---- sequential 6-DoF over the clip: init frame outward both directions ----
    # 3D mask centroid per frame (centroid pixel back-projected at the median mask depth):
    # its frame-to-frame displacement warm-starts the translation, so fast motion (at 5 fps a
    # pick-up moves the object ~100 px ~ 0.4 m) starts near the answer instead of asking the
    # optimizer to travel there against the prior.
    cent3d = [None] * NF
    for t_idx in range(NF):
        mbt = pal[t_idx] == label
        if mbt.sum() >= MIN_MASK_PX:
            ys_t, xs_t = np.nonzero(mbt)
            z_t = float(np.median(depths[t_idx][ys_t, xs_t]))
            cent3d[t_idx] = np.array([(xs_t.mean() - CX) * z_t / FX,
                                      (ys_t.mean() - CY) * z_t / FY, z_t], np.float32)
    poses = {INIT: (r6, t)}
    order = list(range(INIT + 1, NF)) + list(range(INIT - 1, -1, -1))
    m0_px = float(mb0.sum())
    ious, s = np.full(NF, np.nan), float(torch.exp(log_s))
    conf = np.zeros(NF)                                        # per-frame confidence = trust x IoU
    frozen = np.zeros(NF, bool)                                # static-under-occlusion freezes
    ious[INIT] = sil_iou(pts, sixd_to_R(r6[None])[0], t, s, mb0)
    conf[INIT] = ious[INIT]
    done += 1
    print(f"PROGRESS {done} {total}", flush=True)
    for t_idx in order:
        ref = t_idx - 1 if t_idx > INIT else t_idx + 1
        r6p, tp = poses[ref]
        mpts, mb, mb_raw = frame_obs(label, t_idx)
        t_init = tp
        if cent3d[t_idx] is not None and cent3d[ref] is not None:
            t_init = tp + torch.tensor(cent3d[t_idx] - cent3d[ref], device=DEV)
        if mpts is None:                                       # object absent -> carry neighbor pose
            poses[t_idx] = (r6p, tp)
            done += 1
            print(f"PROGRESS {done} {total}", flush=True)
            continue
        # static-under-occlusion freeze: mask still fits the prev pose AND an occluder is
        # on/next to it -> the object hasn't moved; carry the pose VERBATIM (no centroid
        # shift — that is exactly the biased signal that made occluded objects sag/spin)
        Rp = sixd_to_R(r6p[None])[0]
        if check_freeze(mb, occluder_map(t_idx, label), splat_sil(pts, Rp, tp, s)):
            poses[t_idx] = (r6p, tp)
            frozen[t_idx] = True
            ious[t_idx] = sil_iou(pts, Rp, tp, s, mb)
            done += 1
            print(f"PROGRESS {done} {total}", flush=True)
            continue
        trust = min(1.0, float(mb.sum()) / max(m0_px, 1.0))
        if trust < args.trust_freeze_all:                      # a sliver constrains nothing -> coast
            poses[t_idx] = (r6p, t_init)                       # (centroid-shifted carry, conf stays 0)
            done += 1
            print(f"PROGRESS {done} {total}", flush=True)
            continue
        obs = (mpts, inv_maps[t_idx], torch.tensor(mb_raw.astype(np.float32), device=DEV),
               UPS[t_idx] if UPS is not None else None)          # depth gate: RAW mask
        # rotation is unobservable from a partial mask -> only fit it when enough is visible
        opt_rot = trust >= args.trust_freeze_rot
        r6, tt, _, _ = refine(pts, nrm, r6p, t_init, log_s, obs, args.iters_step, 0.005, False,
                              prev=(r6p, t_init), trust=trust, opt_rot=opt_rot, u_obj=u_obj)
        poses[t_idx] = (r6, tt)
        ious[t_idx] = sil_iou(pts, sixd_to_R(r6[None])[0], tt, s, mb)
        conf[t_idx] = trust * ious[t_idx]
        done += 1
        print(f"PROGRESS {done} {total}", flush=True)

    T_seq = np.tile(np.eye(4, dtype=np.float32), (NF, 1, 1))
    for t_idx in range(NF):
        r6, tt = poses[t_idx]
        T_seq[t_idx, :3, :3] = sixd_to_R(r6[None])[0].cpu().numpy()
        T_seq[t_idx, :3, 3] = tt.cpu().numpy()
    ious_fit = ious.copy()

    # ---- anchor-and-interpolate: low-confidence fits never get a vote ----
    # Frames whose (visibility x IoU) clears --anchor-conf are trusted anchors; FROZEN frames
    # are anchors by construction (they equal an already-accepted pose and must not be
    # overwritten by a moving interpolation). Every other frame's pose is REPLACED by
    # interpolation between anchors (slerp rotation, linear translation, ends held) — an
    # awkward occlusion-frame fit cannot leak into the output.
    anchors = sorted(set([INIT] + [k for k in range(NF) if conf[k] >= args.anchor_conf or frozen[k]]))

    # With GT camera poses everything below runs in WORLD frame: a static object's camera-frame
    # trajectory is curved by ego-motion, so camera-frame bridges leave a drift-and-return arc
    # (~60mm here) over long occlusions; a world-frame bridge pins it to the scene.
    def to_world(T):
        return T if CAMS is None else np.einsum("nij,njk->nik", CAMS, T.astype(np.float64))

    def bridge_dev(Tw, k, a, b):
        """deviation (mm, deg) of frame k's pose from the a->b slerp/lerp bridge (same space as Tw)"""
        w = (k - a) / (b - a)
        tb = (1 - w) * Tw[a, :3, 3] + w * Tw[b, :3, 3]
        Rb = Slerp([a, b], Rotation.from_matrix(Tw[[a, b], :3, :3]))(k).as_matrix()
        mm = float(np.linalg.norm(Tw[k, :3, 3] - tb)) * 1000
        cosr = (np.trace(Tw[k, :3, :3].T @ Rb) - 1) / 2
        return mm, float(np.degrees(np.arccos(np.clip(cosr, -1, 1))))

    def pose_diff(Tw, a, b):
        mm = float(np.linalg.norm(Tw[a, :3, 3] - Tw[b, :3, 3])) * 1000
        cosr = (np.trace(Tw[a, :3, :3].T @ Tw[b, :3, :3]) - 1) / 2
        return mm, float(np.degrees(np.arccos(np.clip(cosr, -1, 1))))

    def interp_from_anchors(T_seq, anchors):
        ts = np.arange(NF)
        Tw = to_world(T_seq)
        if len(anchors) >= 2:
            ai = np.array(anchors)
            slerp = Slerp(ai, Rotation.from_matrix(Tw[ai, :3, :3]))
            R_i = np.tile(Tw[ai[0], :3, :3], (NF, 1, 1))
            inner = (ts >= ai[0]) & (ts <= ai[-1])
            R_i[inner] = slerp(ts[inner]).as_matrix()
            R_i[ts > ai[-1]] = Tw[ai[-1], :3, :3]
            t_i = np.stack([np.interp(ts, ai, Tw[ai, :3, 3][:, k]) for k in range(3)], 1)
        else:
            R_i = np.tile(Tw[anchors[0], :3, :3], (NF, 1, 1))
            t_i = np.tile(Tw[anchors[0], :3, 3], (NF, 1))
        weak = np.ones(NF, bool)
        weak[anchors] = False
        if CAMS is None:
            T_seq[weak, :3, :3] = R_i[weak].astype(T_seq.dtype)
            T_seq[weak, :3, 3] = t_i[weak].astype(T_seq.dtype)
        else:                                   # world bridge -> back to each frame's camera
            Ti = np.tile(np.eye(4), (NF, 1, 1))
            Ti[:, :3, :3] = R_i
            Ti[:, :3, 3] = t_i
            Tc = np.einsum("nij,njk->nik", CAMS_INV, Ti)
            T_seq[weak] = Tc[weak].astype(T_seq.dtype)
        return T_seq

    # ---- oscillation resiliency: demote anchor ISLANDS between occlusion gaps ----
    # The wobble signature: the object deviates and RETURNS within a short window — the mesh
    # seems to move and snap back. It happens when the mask is briefly corrupted (an arm
    # sweeping over) and a few frames in the middle re-clear --anchor-conf while the mask has
    # only half-recovered: a small "island" of anchors trapped between two low-confidence gaps,
    # sitting far off the bridge between the stable anchors on either side. Real motion never
    # looks like this — it keeps long confident anchor runs — so islands (<= --osc-island
    # anchors, window <= --osc-win frames, deviation > --osc-trig-mm/--osc-trig-deg) are
    # demoted and the window is interpolated from the stable shores: the pose before, during
    # and after the occlusion stays the SAME.
    osc = []
    if args.osc_win > 0 and len(anchors) >= 2:
        Tw = to_world(T_seq)
        aset = set(anchors)
        gaps, cur = [], None                               # maximal runs of non-anchor frames
        for f in range(anchors[0] + 1, anchors[-1]):
            if f not in aset:
                cur = [f, f] if cur is None else [cur[0], f]
            elif cur:
                gaps.append(cur); cur = None
        if cur:
            gaps.append(cur)
        chains, ch = [], []                                # gap[-island-gap...] chains
        for g in gaps:
            if ch and g[0] - ch[-1][1] - 1 > args.osc_island:
                chains.append(ch); ch = []
            ch.append(g)
        if ch:
            chains.append(ch)
        demote = set()
        for ch in chains:
            if len(ch) < 2:
                continue                                   # single gap: plain interp handles it
            a, b = ch[0][0] - 1, ch[-1][1] + 1             # stable shores (anchors by construction)
            island = [k for k in anchors if a < k < b]
            if not island or b - a > args.osc_win or INIT in island:
                continue
            devs = [bridge_dev(Tw, k, a, b) for k in island]
            pk_mm, pk_dg = max(d[0] for d in devs), max(d[1] for d in devs)
            if pk_mm >= args.osc_trig_mm or pk_dg >= args.osc_trig_deg:
                demote.update(island)
                osc.append((a, b))
                print(f"[{obj}] OSCILLATION frames {a}..{b}: anchors {island} deviate "
                      f"{pk_mm:.0f}mm/{pk_dg:.1f}deg from the stable bridge under occlusion "
                      f"-> demoted, window held consistent", flush=True)
        if demote:
            anchors = [k for k in anchors if k not in demote]

        # ---- world-mode wobble pass: deviate-and-RETURN among plain anchors ----
        # Needs GT camera (world frame), where "returned to the same pose" is meaningful: the
        # post-occlusion echo — anchors that veer off and come back with NO occlusion gap
        # around them — is invisible to the island rule but obvious in world coordinates.
        # Real motion does not return to the same world pose within --osc-win frames.
        if CAMS is not None:
            for _ in range(3):                             # a demote can expose the next wobble
                hit = None
                for ii, a in enumerate(anchors):
                    for jj in range(ii + 2, len(anchors)):
                        b = anchors[jj]
                        if b - a > args.osc_win:
                            break
                        e_mm, e_dg = pose_diff(Tw, a, b)
                        if e_mm > args.osc_return_mm or e_dg > args.osc_return_deg:
                            continue                       # endpoints differ: pose did not return
                        island = anchors[ii + 1:jj]
                        if INIT in island:
                            continue
                        devs = [bridge_dev(Tw, k, a, b) for k in island]
                        pk_mm, pk_dg = max(d[0] for d in devs), max(d[1] for d in devs)
                        if pk_mm >= args.osc_trig_mm or pk_dg >= args.osc_trig_deg:
                            hit = (a, b, island, pk_mm, pk_dg)
                            break
                    if hit:
                        break
                if not hit:
                    break
                a, b, island, pk_mm, pk_dg = hit
                anchors = [k for k in anchors if k not in island]
                osc.append((a, b))
                print(f"[{obj}] WOBBLE frames {a}..{b}: pose returns to start (world) but anchors "
                      f"{island} deviate {pk_mm:.0f}mm/{pk_dg:.1f}deg in between -> demoted, "
                      f"window held consistent", flush=True)
    T_seq = interp_from_anchors(T_seq, anchors)

    # ---- temporal smoothing (median + SavGol) on the whole sequence ----
    if args.smooth_win >= 5:
        T_seq = smooth_Tseq(T_seq, args.smooth_win)

    # final IoU after interpolation+smoothing (the numbers that describe what gets rendered)
    for t_idx in range(NF):
        mbt = pal[t_idx] == label
        if mbt.sum() >= MIN_MASK_PX:
            ious[t_idx] = sil_iou(pts, torch.tensor(T_seq[t_idx, :3, :3], device=DEV),
                                  torch.tensor(T_seq[t_idx, :3, 3], device=DEV), s,
                                  fill_holes(mbt))

    # per-frame mask presence -> `visible`: True from the object's FIRST nonempty
    # mask to its LAST (leading/trailing absence = the object is not in the scene
    # yet / anymore, so downstream renders must not draw it there; mid-clip gaps
    # stay visible — those are occlusions, not absence)
    present = np.array([c is not None for c in cent3d], bool)
    visible = np.zeros(NF, bool)
    if present.any():
        visible[int(np.argmax(present)):NF - int(np.argmax(present[::-1]))] = True
    np.savez(f"{args.out_dir}/{sl}.npz", T_seq=T_seq, scale=s, center=center,
             ious=ious, ious_fit=ious_fit, conf=conf, anchors=np.array(anchors), frozen=frozen,
             osc=np.array(osc, np.int64).reshape(-1, 2), visible=visible,
             u_obj=(u_obj.cpu().numpy() if u_obj is not None else np.zeros(3)))
    v = ious[~np.isnan(ious)]
    print(f"[{obj}] fitted {NF} frames  anchors {len(anchors)}/{NF}  frozen {int(frozen.sum())}  "
          f"point-IoU mean={v.mean():.3f} min={v.min():.3f} "
          f"(pre-smooth {np.nanmean(ious_fit):.3f}) -> {args.out_dir}/{sl}.npz", flush=True)
