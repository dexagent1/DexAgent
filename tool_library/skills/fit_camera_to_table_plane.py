"""Camera up, pitch, roll and height from the surface the objects sit on."""
from __future__ import annotations

TOOL = {
    "name": "fit_camera_to_table_plane",
    "title": "Fit camera to table plane",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "RANSAC plane through the depth pixels around the tracked objects gives camera up, pitch, roll and height; cross-checks a GeoCalib estimate and overrules it when it points at the wrong side of the world.",
    "description": (
        "GeoCalib reads world-up from horizon lines and upright structure and fails "
        "confidently on a featureless white table (measured: +40 deg pitch, looking up, on a "
        "clip that plainly looks down). The support surface is the strongest plane that "
        "matters, so this un-projects depth pixels in a ring around the object masks, fits a "
        "plane by RANSAC, and takes the normal facing the camera as up. If the ring is too "
        "small or noisy it falls back to all non-object pixels below the objects' top row on "
        "the first frame. With a GeoCalib result passed in it reports which to trust: GeoCalib "
        "unless the two disagree by more than 60 deg, or by more than 18 deg when the plane fit "
        "is confident (inlier fraction >= 0.55, normal spread <= 12 deg)."
    ),
    "inputs": [
        {"name": "depth_npz", "type": "path", "description": "estimate_depth output (depths in metres)."},
        {"name": "masks_dir", "type": "path", "description": "Palette label maps frame%05d.png; objects = every non-zero label unless labels is given."},
        {"name": "focal_px", "type": "float", "description": "Focal length in pixels at the mask resolution."},
        {"name": "labels", "type": "list[int] | None", "description": "Object labels that rest on the table."},
        {"name": "geocalib", "type": "dict | path | None", "description": "estimate_gravity output to cross-check."},
        {"name": "n_frames, stride, tol, grow, keep_out", "type": "numbers", "description": "Frames sampled (5), pixel stride (3), RANSAC tolerance in m (0.008), ring width and keep-out in pixels (30, 4)."},
    ],
    "outputs": [
        {"name": "plane", "type": "dict | None",
         "description": "gravity_cam, up (vec3d), pitch_deg, roll_deg, camera_height_m, inlier_frac, normal_spread_deg, mode (object_ring | full_surface); plus 'decision' when geocalib was given: source, disagreement_deg, note."},
    ],
    "status": "tested",
    "entrypoint": "fit_camera_to_table_plane",
    "requirements": ["numpy", "scipy", "pillow"],
    "tags": ["camera", "gravity", "plane", "RANSAC"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Computer Installation", "Open the Bottle"],
}

import json
import sys
from pathlib import Path

import numpy as np

from tool_library.common.video import frame_files, load_label_map, read_json

DISAGREE_DEG = 60.0
CONFIDENT_DISAGREE_DEG = 18.0
CONFIDENT_INLIER_FRAC = 0.55
CONFIDENT_SPREAD_DEG = 12.0


def plane_ls(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-12)
    return n, float(n @ c)


def plane_ransac(P, iters=800, tol=0.008, seed=0):
    """Best plane (unit normal n, offset d with n.p = d) and inlier fraction."""
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(int(iters)):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        L = float(np.linalg.norm(n))
        if L < 1e-9:
            continue
        n /= L
        d = float(n @ a)
        k = int((np.abs(P @ n - d) < tol).sum())
        if k > best[0]:
            best = (k, (n, d))
    if best[1] is None:
        return None, None, 0.0
    n, d = best[1]
    m = np.abs(P @ n - d) < tol
    n, d = plane_ls(P[m])
    return n, d, float(m.mean())


def _unproject(xs, ys, z, f, cx, cy):
    return np.stack([(xs - cx) * z / f, (ys - cy) * z / f, z], 1)


def _result(up, hts, inl, n_fr, spread, mode):
    up = np.asarray(up, float)
    up /= np.linalg.norm(up) + 1e-12
    g = -up
    return {"gravity_cam": [float(x) for x in g], "up": [float(x) for x in up],
            "pitch_deg": round(float(np.degrees(np.arcsin(np.clip(-g[2], -1, 1)))), 3),
            "roll_deg": round(float(np.degrees(np.arctan2(g[0], g[1]))), 3),
            "camera_height_m": round(float(np.median(hts)), 4), "inlier_frac": round(float(inl), 3),
            "n_frames": int(n_fr), "normal_spread_deg": round(float(spread), 2), "mode": mode}


def cross_check(geocalib, plane, threshold_deg=DISAGREE_DEG):
    """Keep GeoCalib unless the table says it points the wrong way."""
    if not plane:
        return {"source": "geocalib", "checked": False, "note": "no support plane available"}
    g1 = np.asarray(geocalib.get("up_cam") or geocalib.get("up") or geocalib.get("vec3d") or -np.asarray(geocalib["gravity_cam"]), float)
    g1 /= np.linalg.norm(g1) + 1e-12
    g2 = np.asarray(plane["up"], float)
    ang = float(np.degrees(np.arccos(np.clip(float(g1 @ g2), -1, 1))))
    bad = ang > threshold_deg
    if (not bad and ang > CONFIDENT_DISAGREE_DEG and plane["inlier_frac"] >= CONFIDENT_INLIER_FRAC
            and plane["normal_spread_deg"] <= CONFIDENT_SPREAD_DEG):
        bad = True
    return {"source": "support_plane" if bad else "geocalib", "checked": True, "disagreement_deg": round(ang, 2),
            "geocalib_pitch_deg": geocalib.get("pitch_deg"), "plane_pitch_deg": plane["pitch_deg"],
            "camera_height_m": plane["camera_height_m"],
            "note": (f"GeoCalib disagrees with the support plane by {ang:.0f} deg: world-up on the wrong side, plane used."
                     if bad else f"GeoCalib agrees with the support plane to {ang:.0f} deg: kept.")}


def fit_camera_to_table_plane(depth_npz, masks_dir, focal_px, labels=None, geocalib=None, n_frames=5, stride=3,
                              tol=0.008, grow=30, keep_out=4):
    from scipy.ndimage import binary_dilation
    D = np.load(depth_npz)["depths"]
    pred = frame_files(masks_dir)
    assert pred and len(D), "need depth frames and mask frames"
    f = float(focal_px)
    ups, hts, fr = [], [], []
    for t in np.linspace(0, min(len(pred), len(D)) - 1, int(n_frames)).astype(int):
        pal, dep = load_label_map(pred[t]), np.asarray(D[t], float)
        h, w = dep.shape
        cx, cy = w / 2.0, h / 2.0
        if pal.shape != dep.shape:
            from PIL import Image
            pal = np.asarray(Image.fromarray(pal).resize((w, h), Image.NEAREST))
        obj = np.isin(pal, labels) if labels else pal > 0
        if obj.sum() < 50:
            continue
        ring = binary_dilation(obj, iterations=int(grow)) & ~binary_dilation(obj, iterations=int(keep_out))
        ys, xs = np.nonzero(ring)
        ys, xs = ys[::stride], xs[::stride]
        z = dep[ys, xs]
        ok = np.isfinite(z) & (z > 0.05) & (z < 6.0)
        if int(ok.sum()) < 150:
            continue
        n, d, inl = plane_ransac(_unproject(xs[ok], ys[ok], z[ok], f, cx, cy), tol=tol)
        if n is None:
            continue
        if d > 0:                       # up is the normal that points back at the camera
            n, d = -n, -d
        ups.append(n); hts.append(abs(d)); fr.append(inl)
    plane = None
    if ups:
        U = np.asarray(ups)
        up = U.mean(0)
        up /= np.linalg.norm(up) + 1e-12
        plane = _result(up, hts, np.mean(fr), len(ups), np.degrees(np.arccos(np.clip((U @ up).min(), -1, 1))), "object_ring")
    if plane is None or plane["inlier_frac"] < 0.4 or plane["normal_spread_deg"] > 15.0:
        pal0, dep0 = load_label_map(pred[0]), np.asarray(D[0], float)
        h, w = dep0.shape
        cx, cy = w / 2.0, h / 2.0
        if pal0.shape != dep0.shape:
            from PIL import Image
            pal0 = np.asarray(Image.fromarray(pal0).resize((w, h), Image.NEAREST))
        obj = np.isin(pal0, labels) if labels else pal0 > 0
        rows = np.nonzero(obj.any(1))[0]
        surf = ~binary_dilation(obj, iterations=int(keep_out))
        surf[: max(0, (int(rows.min()) if len(rows) else 0) - 5)] = False
        ys, xs = np.nonzero(surf)
        ys, xs = ys[::3], xs[::3]
        z = dep0[ys, xs]
        ok = np.isfinite(z) & (z > 0.05) & (z < 6.0)
        if int(ok.sum()) >= 500:
            n, d, inl = plane_ransac(_unproject(xs[ok], ys[ok], z[ok], f, cx, cy), tol=tol)
            if n is not None and inl >= 0.45:
                if d > 0:
                    n, d = -n, -d
                plane = _result(n, [abs(d)], inl, 1, 0.0, "full_surface")
    if geocalib is not None:
        if isinstance(geocalib, (str, Path)):
            geocalib = read_json(geocalib)
        decision = cross_check(geocalib, plane)
        if plane is None:
            return {"plane": None, "decision": decision}
        plane["decision"] = decision
    return plane


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(fit_camera_to_table_plane(a[0], a[1], float(a[2])), indent=1))
