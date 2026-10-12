"""Put a reconstructed object where the video shows it: silhouette-IoU placement through the registered camera."""
from __future__ import annotations

TOOL = {
    "name": "fit_object_placement",
    "title": "Fit object placement",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Optimizes the pose of one object body in a scene (x, y, yaw by default; optionally z, pitch, roll, and a uniform size scale) so that its rendered silhouette through the registered video camera matches the object's video mask, by coarse search then Nelder-Mead on 1 - IoU with hand pixels ignored; writes the scene with the fitted pose.",
    "description": (
        "The paper's placement step. Loads the scene, adds the video camera (camera.json, or a camera already in the "
        "scene by name), and renders the segmentation of the object's body subtree at the mask resolution for each "
        "candidate pose. A fixed root body is moved by writing its pose; a free-jointed body by writing its free joint. "
        "Search: the yaw range on a grid combined with random x, y jitter inside the bounds around the start pose (the "
        "current pose, or the mask footprint's contact centre when `init_from_mask` is set), then scipy Nelder-Mead "
        "from the best candidate. IoU is computed on pixels outside `ignore_mask` only. With `scale` in the parameters the "
        "object's mesh and primitive sizes are scaled uniformly about its root (for an object whose size was guessed). "
        "Returns the fitted pose in the rig frame, the IoU before and after, the pixel offset of the silhouette centroids, "
        "and an overlay image (mask magenta, fitted silhouette green). With out_xml it rewrites the scene so the body "
        "starts at the fitted pose. A low final IoU with a good shape means the camera or the size is wrong, not the pose."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "path", "description": "The scene (bench with the asset attached, from stage_on_bench)."},
        {"name": "body", "type": "str", "description": "Root body of the object (for example 'btl_bottle')."},
        {"name": "mask", "type": "path | ndarray", "description": "The object's video mask at the camera resolution."},
        {"name": "camera", "type": "path | dict | str", "description": "camera.json, its dict, or the name of a camera in the scene."},
        {"name": "ignore_mask", "type": "path | ndarray | None", "description": "Pixels excluded from the IoU (hands, arms, occluders)."},
        {"name": "params", "type": "list[str]", "description": "Which of x, y, z, yaw, pitch, roll, scale to optimize. Default ['x', 'y', 'yaw']."},
        {"name": "bounds", "type": "dict | None", "description": "Half-ranges around the start pose that every search stage respects (coarse grid, yaw scan and Nelder-Mead): {x: 0.15, y: 0.15, z: 0.02, yaw: 180, pitch: 15, roll: 15, scale: 0.3} (metres, degrees, fraction). Give yaw: 20 to keep an object's known facing. Defaults as listed."},
        {"name": "init_from_mask", "type": "bool", "description": "Start x, y at the mask's contact centre on the table (with a camera.json; a scene camera name starts at the current pose). Default True."},
        {"name": "n_coarse", "type": "int", "description": "Random candidates in the coarse stage (times the yaw grid). Default 48."},
        {"name": "n_yaw", "type": "int", "description": "Yaw grid size when yaw is optimized. Default 12."},
        {"name": "n_refine", "type": "int", "description": "Nelder-Mead evaluations. Default 120."},
        {"name": "render_scale", "type": "float", "description": "Render at this fraction of the mask resolution for speed. Default 0.5."},
        {"name": "out_xml", "type": "path | None", "description": "Write the scene with the fitted pose here."},
        {"name": "out_png", "type": "path | None", "description": "Overlay image path (needs frame)."},
        {"name": "frame", "type": "path | None", "description": "Video frame for the overlay."},
    ],
    "outputs": [
        {"name": "pose", "type": "dict", "description": "x, y, z, yaw_deg, pitch_deg, roll_deg (rig frame) and scale."},
        {"name": "iou / iou_start", "type": "float", "description": "Silhouette IoU after and before the fit."},
        {"name": "centroid_offset_px", "type": "list[2]", "description": "Rendered minus mask silhouette centroid."},
        {"name": "out_xml / overlay_png", "type": "path", "description": "Written files."},
    ],
    "status": "tested",
    "entrypoint": "fit_object_placement",
    "requirements": ["mujoco>=3.1", "numpy", "scipy", "opencv-python"],
    "tags": ["placement", "silhouette", "IoU", "pose fit", "camera", "stage 2"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope", "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting", "Biology Experiment"],
    "cli": "python -m tool_library.skills.fit_object_placement scene.xml btl_bottle mask.png camera.json --out-xml scene_fit.xml --out-png fit.png --frame frame00000.jpg",
}

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import rigcam as RC

DEFAULT_BOUNDS = {"x": 0.15, "y": 0.15, "z": 0.02, "yaw": 180.0, "pitch": 15.0, "roll": 15.0, "scale": 0.3}


def _mask(m):
    if isinstance(m, (str, Path)):
        import cv2
        a = cv2.imread(str(m), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(m)
        if a.ndim == 3:
            a = a[..., 0]
        return a > 0
    return np.asarray(m) > 0


class _Scene:
    def __init__(self, scene_xml, body, camera, W, H, render_scale):
        self.spec = mujoco.MjSpec.from_file(str(scene_xml))
        self.cam_name = None
        if isinstance(camera, str) and not str(camera).endswith(".json") and not os.path.exists(str(camera)):
            self.cam_name = camera
        else:
            cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
            if abs(cam["width"] / cam["height"] - W / H) > 1e-3:
                raise ValueError(f"camera is {cam['width']}x{cam['height']} but the mask is {W}x{H}")
            c = self.spec.worldbody.add_camera()
            c.name = "fit_video_cam"; c.pos = cam["pos"]; c.quat = cam["quat_wxyz"]; c.fovy = cam["fovy_deg"]
            self.cam_name = "fit_video_cam"
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if self.bid < 0:
            raise ValueError(f"no body named {body!r}; bodies: {[self.model.body(i).name for i in range(self.model.nbody)][:40]}")
        self.subtree = {b for b in range(self.model.nbody) if self._in_subtree(b)}
        self.geom_in = np.array([int(self.model.geom_bodyid[g]) in self.subtree for g in range(self.model.ngeom)])
        # how the body is moved
        jadr = self.model.body_jntadr[self.bid]
        self.free = self.model.body_jntnum[self.bid] > 0 and self.model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE
        self.qadr = int(self.model.jnt_qposadr[jadr]) if self.free else None
        self.base_pos = self.model.body_pos[self.bid].copy(); self.base_quat = self.model.body_quat[self.bid].copy()
        self.geom_size0 = self.model.geom_size.copy(); self.geom_pos0 = self.model.geom_pos.copy()
        self.body_pos0 = self.model.body_pos.copy()
        self.scaling = False
        self.rw, self.rh = max(32, int(round(W * render_scale))), max(32, int(round(H * render_scale)))
        from tool_library.common.render import segmentation_renderer
        self.renderer = segmentation_renderer(self.model, self.rh, self.rw)
        self.cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if self.cid < 0:
            raise ValueError(f"camera {self.cam_name!r} not in the scene")

    def _in_subtree(self, b):
        while b > 0:
            if b == self.bid:
                return True
            b = int(self.model.body_parentid[b])
        return False

    def current_pose(self):
        if self.free:
            q = self.data.qpos[self.qadr:self.qadr + 7]
            p, (w, x, y, z) = q[:3], q[3:7]
        else:
            p, (w, x, y, z) = self.base_pos, self.base_quat
        e = Rot.from_quat([x, y, z, w]).as_euler("ZYX", degrees=True)   # yaw, pitch, roll
        return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "yaw": float(e[0]), "pitch": float(e[1]), "roll": float(e[2]), "scale": 1.0}

    def set_pose(self, P):
        q = Rot.from_euler("ZYX", [P["yaw"], P["pitch"], P["roll"]], degrees=True).as_quat()
        quat = np.array([q[3], q[0], q[1], q[2]])
        pos = np.array([P["x"], P["y"], P["z"]])
        s = float(P.get("scale", 1.0))
        if self.scaling:                                   # always from the unscaled sizes, so s = 1 restores them
            for g in range(self.model.ngeom):
                if self.geom_in[g]:
                    self.model.geom_size[g] = self.geom_size0[g] * s
                    self.model.geom_pos[g] = self.geom_pos0[g] * s
            for b in self.subtree:
                if b != self.bid:
                    self.model.body_pos[b] = self.body_pos0[b] * s
        if self.free:
            self.data.qpos[self.qadr:self.qadr + 3] = pos; self.data.qpos[self.qadr + 3:self.qadr + 7] = quat
        else:
            self.model.body_pos[self.bid] = pos; self.model.body_quat[self.bid] = quat
        mujoco.mj_forward(self.model, self.data)

    def silhouette(self):
        self.renderer.update_scene(self.data, camera=self.cid)
        seg = self.renderer.render()
        gid = seg[..., 0]
        ok = gid >= 0
        out = np.zeros(gid.shape, bool)
        out[ok] = self.geom_in[gid[ok]]
        return out


def _iou(a, b, keep):
    a = a & keep; b = b & keep
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def fit_object_placement(scene_xml, body, mask, camera, ignore_mask=None, params=("x", "y", "yaw"), bounds=None, init_from_mask=True,
                         n_coarse=48, n_yaw=12, n_refine=120, render_scale=0.5, out_xml=None, out_png=None, frame=None, seed=0):
    import cv2
    from scipy.optimize import minimize
    params = list(params)
    bad = [p for p in params if p not in DEFAULT_BOUNDS]
    if bad:
        raise ValueError(f"unknown parameters {bad}; choose from {list(DEFAULT_BOUNDS)}")
    B = dict(DEFAULT_BOUNDS, **(bounds or {}))
    M = _mask(mask); H, W = M.shape
    keep = ~_mask(ignore_mask) if ignore_mask is not None else np.ones_like(M)
    sc = _Scene(scene_xml, body, camera, W, H, render_scale)
    sc.scaling = "scale" in params
    Ms = cv2.resize(M.astype(np.uint8), (sc.rw, sc.rh), interpolation=cv2.INTER_NEAREST) > 0
    Ks = cv2.resize(keep.astype(np.uint8), (sc.rw, sc.rh), interpolation=cv2.INTER_NEAREST) > 0
    start = sc.current_pose()
    # the mask footprint needs the camera.json numbers: a camera named in the scene starts at the current pose
    if init_from_mask and ("x" in params or "y" in params) and sc.cam_name == "fit_video_cam":
        try:
            from tool_library.skills.measure_mask_footprint import measure_mask_footprint
            fp = measure_mask_footprint(M, camera, ignore_mask)
            cx, cy = fp["contact"]["center_xy"]
            if "x" in params: start["x"] = cx
            if "y" in params: start["y"] = cy
        except Exception:
            pass
    sc.set_pose(start)
    iou0 = _iou(sc.silhouette(), Ms, Ks)
    rng = np.random.default_rng(seed)

    rows_m = np.nonzero((Ms & Ks).any(axis=1))[0]
    mask_bottom = float(rows_m.max()) if len(rows_m) else None

    def excess(k, v):
        """How far v is outside the bound of parameter k around the start (0 inside); angles compare wrapped."""
        d = float(v) - float(start[k])
        if k in ("yaw", "pitch", "roll"):
            if float(B[k]) >= 180.0:
                return 0.0                               # the full circle: every angle is allowed
            d = (d + 180.0) % 360.0 - 180.0
        half = B[k] * start["scale"] if k == "scale" else B[k]
        return max(0.0, abs(d) - float(half))

    def evaluate(vec):
        P = dict(start)
        for k, v in zip(params, vec):
            P[k] = v
        # every stage, the Nelder-Mead refinements included, stays inside the bounds: a pose outside them costs more
        # than any pose inside (a yaw fitted to 170 deg with bounds of +-20 is a flipped object the caller ruled out)
        out_of = sum(excess(k, v) / max(float(B[k]), 1e-9) for k, v in zip(params, vec))
        if out_of > 0:
            return 1.0 + out_of
        if "scale" in P and P["scale"] <= 0.2:
            return 1.0
        sc.set_pose(P)
        sil = sc.silhouette()
        cost = 1.0 - _iou(sil, Ms, Ks)
        if sc.scaling and mask_bottom is not None:
            # size and distance trade off along the view ray; the contact row of the silhouette pins the distance
            rows = np.nonzero((sil & Ks).any(axis=1))[0]
            if len(rows):
                cost += 0.5 * abs(float(rows.max()) - mask_bottom) / sc.rh
        return cost

    # coarse stage 1: translation (and z / scale) jitter at the start orientation
    best_vec, best_cost = [start[k] for k in params], evaluate([start[k] for k in params])
    n1 = max(8, int(n_coarse) // 2)
    for i in range(n1):
        vec = []
        for k in params:
            if k in ("yaw", "pitch", "roll"):
                vec.append(start[k])
            elif k == "scale":
                vec.append(start["scale"] * (1 + rng.uniform(-B["scale"], B["scale"])))
            else:
                vec.append(start[k] + rng.uniform(-B[k], B[k]) * (0.5 if i < n1 // 2 else 1.0))
        c = evaluate(vec)
        if c < best_cost:
            best_vec, best_cost = vec, c
    # coarse stage 2: orientation grid at the best translation, with a little jitter
    if "yaw" in params:
        base = list(best_vec); iy = params.index("yaw")
        for yw in np.linspace(-B["yaw"], B["yaw"], int(n_yaw), endpoint=False):
            for j in range(max(1, (int(n_coarse) - n1) // max(1, int(n_yaw)))):
                vec = list(base)
                jit = rng.uniform(-0.5, 0.5) * 2 * B["yaw"] / int(n_yaw) if j else 0.0
                vec[iy] = start["yaw"] + float(np.clip(yw + jit, -B["yaw"], B["yaw"]))
                for k in ("x", "y"):
                    if k in params and j:
                        vec[params.index(k)] = base[params.index(k)] + rng.uniform(-0.3, 0.3) * B[k]
                c = evaluate(vec)
                if c < best_cost:
                    best_vec, best_cost = vec, c
    # refine
    steps = []
    for k in params:
        steps.append({"x": 0.02, "y": 0.02, "z": 0.005, "yaw": 15.0, "pitch": 5.0, "roll": 5.0, "scale": 0.05}[k])
    simplex = [np.array(best_vec, float)]
    for i in range(len(params)):
        v = np.array(best_vec, float); v[i] += steps[i]; simplex.append(v)
    res = minimize(evaluate, np.array(best_vec, float), method="Nelder-Mead",
                   options={"initial_simplex": np.array(simplex), "maxfev": int(n_refine), "xatol": 1e-3, "fatol": 1e-4})
    final_vec = np.array(res.x if res.fun <= best_cost else best_vec, float)
    n_eval = int(res.nfev)
    if "yaw" in params:                                  # yaw has local optima: scan it at the refined position, refine once more
        iy = params.index("yaw"); best_c = evaluate(final_vec); best_y = final_vec[iy]
        half = min(float(B["yaw"]), 180.0)               # the scan covers the bounds around the start, not the full circle
        for yw in start["yaw"] + np.arange(-half, half + 1e-9 if half < 180.0 else half, 10.0):
            alt = np.array(final_vec); alt[iy] = yw; c = evaluate(alt); n_eval += 1
            if c < best_c - 1e-6:
                best_c, best_y = c, yw
        if abs(best_y - final_vec[iy]) > 1e-6:
            final_vec[iy] = best_y
            res2 = minimize(evaluate, final_vec, method="Nelder-Mead", options={"maxfev": max(40, int(n_refine) // 2), "xatol": 1e-3, "fatol": 1e-4})
            n_eval += int(res2.nfev)
            if res2.fun <= best_c:
                final_vec = np.array(res2.x, float)
    P = dict(start)
    for k, v in zip(params, final_vec):
        P[k] = float(v)
    P["yaw"] = float((P["yaw"] + 180) % 360 - 180)
    sc.set_pose(P)
    sil = sc.silhouette()
    iou = _iou(sil, Ms, Ks)
    # centroid offset in mask pixels
    def cen(m):
        ys, xs = np.nonzero(m); return np.array([xs.mean(), ys.mean()]) if len(xs) else np.array([np.nan, np.nan])
    off = (cen(sil & Ks) - cen(Ms & Ks)) / render_scale
    result = {"pose": {"x": P["x"], "y": P["y"], "z": P["z"], "yaw_deg": P["yaw"], "pitch_deg": P["pitch"], "roll_deg": P["roll"], "scale": P.get("scale", 1.0)},
              "iou": round(iou, 4), "iou_start": round(iou0, 4), "centroid_offset_px": [round(float(off[0]), 1), round(float(off[1]), 1)],
              "body": body, "params": params, "evaluations": int(res.nfev) + int(n_coarse) + 1}
    if out_xml:
        txt = Path(scene_xml).read_text()
        q = Rot.from_euler("ZYX", [P["yaw"], P["pitch"], P["roll"]], degrees=True).as_quat()
        pos_s = f'{P["x"]:.6f} {P["y"]:.6f} {P["z"]:.6f}'; quat_s = f'{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}'
        pat = re.compile(rf'<body name="{re.escape(body)}"([^>]*)>')
        m = pat.search(txt)
        if not m:
            raise ValueError(f"cannot find <body name=\"{body}\"> in {scene_xml} to write the pose")
        attrs = m.group(1)
        attrs = re.sub(r'\spos="[^"]*"', "", attrs); attrs = re.sub(r'\squat="[^"]*"', "", attrs); attrs = re.sub(r'\seuler="[^"]*"', "", attrs)
        txt = txt[:m.start()] + f'<body name="{body}" pos="{pos_s}" quat="{quat_s}"{attrs}>' + txt[m.end():]
        if abs(P.get("scale", 1.0) - 1.0) > 1e-9:
            result["scale_note"] = "scale was fitted; the written XML keeps the original sizes: rebuild the asset with the scaled dimensions"
        if sc.free:
            # keyframes hold the free joint's pose explicitly: write the fitted pose into every key
            new7 = [P["x"], P["y"], P["z"], float(q[3]), float(q[0]), float(q[1]), float(q[2])]
            def _patch_key(mk):
                vals = mk.group(2).split()
                if len(vals) > sc.qadr + 6:
                    vals[sc.qadr:sc.qadr + 7] = [f"{v:.6g}" for v in new7]
                return f'{mk.group(1)}qpos="{" ".join(vals)}"'
            txt = re.sub(r'(<key\b[^>]*?)qpos="([^"]*)"', _patch_key, txt)
        Path(out_xml).write_text(txt)
        result["out_xml"] = str(out_xml)
    if out_png:
        if frame is not None:
            img = cv2.imread(str(frame)); img = cv2.resize(img, (W, H)) if img.shape[:2] != (H, W) else img
        else:
            img = np.full((H, W, 3), 60, np.uint8)
        full = cv2.resize(sil.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        ov = img.copy()
        ov[M] = (0.55 * ov[M] + 0.45 * np.array([255, 0, 255])).astype(np.uint8)
        ov[full] = (0.55 * ov[full] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.putText(ov, f"{body}: IoU {iou0:.2f} -> {iou:.2f}  pose x {P['x']:.3f} y {P['y']:.3f} yaw {P['yaw']:.0f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_png), ov)
        result["overlay_png"] = str(out_png)
    sc.renderer.close()
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("scene_xml"); ap.add_argument("body"); ap.add_argument("mask"); ap.add_argument("camera")
    ap.add_argument("--ignore-mask"); ap.add_argument("--params", nargs="*", default=["x", "y", "yaw"])
    ap.add_argument("--out-xml"); ap.add_argument("--out-png"); ap.add_argument("--frame")
    a = ap.parse_args(argv)
    r = fit_object_placement(a.scene_xml, a.body, a.mask, a.camera, a.ignore_mask, a.params, out_xml=a.out_xml, out_png=a.out_png, frame=a.frame)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.exit(main())
