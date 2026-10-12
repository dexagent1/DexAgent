"""Camera visibility audit: do the manipulated objects and the acting hands stay in the video camera's view?

    from tool_library.verifiers.camera_visibility_audit import camera_visibility_audit
    camera_visibility_audit("scene.xml", "trajectory.npz", camera="camera.json")

Stage 4 renders the episode through the camera the human video was filmed with, and the
policy trained on it learns from what that camera sees. A carry that swings the object out of
frame, or a grasp made at the edge of the image, produces frames where the task happens
off-screen.
"""
from __future__ import annotations

TOOL = {
    "name": "camera_visibility_audit",
    "title": "Camera visibility audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "The manipulated objects and the acting hands project inside the video camera's image (camera json or a "
               "scene camera) on the frames that matter: while an object moves or is touched, while a hand touches one.",
    "description": (
        "No other audit looks through the camera. The camera is a camera.json (width, height, fx, fy, cx, cy, pos, "
        "quat_wxyz in MuJoCo's convention: looking along its -z, y up; Stage 2 writes one) or a camera of the scene "
        "by name (default: 'video_cam' when the scene has it, else 'ego'; intrinsics from its fovy and "
        "width x height). On every frame each free object's centre of mass and each hand's palm are projected; the "
        "point counts as out of view when it is behind the camera or farther than outside_frac (default 0.2) of the "
        "image size beyond its border: a palm or object centre a little past the edge still shows most of the body, "
        "one a fifth of the image beyond it shows none (the paper's executed episodes reach 0.16 at the edges of "
        "egocentric frames). Frames that "
        "matter: for an object, those where it moves faster than 1 cm/s or a robot hand touches it; for a hand, those "
        "where it touches a free object. It fails when an object or hand is out of view on more than max_out_frac "
        "(default 0.25) of its frames that matter, or for a contiguous stretch longer than max_out_s (default 1 s). "
        "Reports per body the share of frames out of view, the longest stretch and its first frame and phase label."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "camera", "type": "str | dict | None", "description": "camera.json path or dict, or the name of a scene camera. Default 'video_cam' if present, else 'ego'."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "width, height", "type": "int", "description": "Image size for a scene camera without a resolution of its own. Default 640 x 480 (the human video's size matters: pass the Stage 2 camera.json when the video is not 4:3)."},
        {"name": "outside_frac", "type": "float", "description": "How far beyond the image border, as a share of the image size, a centre may lie and still count as in view. Default 0.2."},
        {"name": "max_out_frac", "type": "float", "description": "Largest share of frames that matter out of view. Default 0.25."},
        {"name": "max_out_s", "type": "float", "description": "Longest contiguous stretch out of view. Default 1.0."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {body, out_frac, longest_out_s, first_out_frame, label}; bodies {name: frames that matter, out of view, longest stretch}; camera."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "camera", "visibility", "framing", "stage4"],
    "status": "tested",
    "entrypoint": "camera_visibility_audit",
}

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA


def _camera(ep, camera, width, height):
    m = ep.model
    if isinstance(camera, dict) or (isinstance(camera, str) and camera.endswith(".json")):
        c = EA.read_json(camera, "camera")
        try:
            W, H = int(c["width"]), int(c["height"])
            fx = float(c.get("fx") or (H / 2) / np.tan(np.radians(float(c["fovy_deg"])) / 2))
            fy = float(c.get("fy") or fx)
            cx, cy = float(c.get("cx", W / 2)), float(c.get("cy", H / 2))
            pos = np.asarray(c["pos"], float); R = EA.quat_to_mat(np.asarray(c["quat_wxyz"], float))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"camera: the json needs width, height, fx/fy (or fovy_deg), pos, quat_wxyz ({e})")
        return {"source": str(camera) if not isinstance(camera, dict) else "dict", "W": W, "H": H, "fx": fx, "fy": fy,
                "cx": cx, "cy": cy, "pos": pos, "R": R, "fixed": True}
    name = camera
    if name is None:
        name = "video_cam" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "video_cam") >= 0 else "ego"
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, str(name))
    if cid < 0:
        raise ValueError(f"camera: the scene has no camera {name!r}; pass a camera.json")
    res = np.asarray(m.cam_resolution[cid], int) if hasattr(m, "cam_resolution") else np.array([1, 1])
    W, H = (int(res[0]), int(res[1])) if res.min() > 1 else (int(width), int(height))
    fy = (H / 2) / np.tan(np.radians(float(m.cam_fovy[cid])) / 2)
    return {"source": f"scene camera {name}", "W": W, "H": H, "fx": fy, "fy": fy, "cx": W / 2, "cy": H / 2, "cam_id": cid,
            "fixed": int(m.cam_bodyid[cid]) == 0 or int(m.cam_bodyid[cid]) in ep.static_bodies}


def camera_visibility_audit(scene, trajectory, camera=None, frames=None, width=640, height=480, outside_frac=0.2,
                            max_out_frac=0.25, max_out_s=1.0):
    EA.check_positive(width=width, height=height, max_out_frac=max_out_frac, max_out_s=max_out_s)
    if not 0 <= float(outside_frac) < 1:
        raise ValueError("outside_frac must lie in [0, 1)")
    ep = EA.Episode(scene, trajectory, frames)
    cam = _camera(ep, camera, width, height)
    m = ep.model
    d = mujoco.MjData(m)
    T = ep.T
    pts = {}
    for n, o in ep.objects.items():
        pts[n] = ep.com_track(o["bodies"])
    for s, h in ep.hands.items():
        pts[f"{s} hand"] = ep.body_pos(h["palm"])
    cpos = np.zeros((T, 3)); cR = np.zeros((T, 3, 3))
    if "cam_id" in cam:
        cid = cam["cam_id"]
        for t in range(T):
            if t == 0 or not cam["fixed"]:
                d.qpos[:] = ep.Q[t]; mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
            cpos[t] = d.cam_xpos[cid]; cR[t] = d.cam_xmat[cid].reshape(3, 3)
    else:
        cpos[:] = cam["pos"]; cR[:] = cam["R"]
    tl = EA.object_contact_timeline(ep, margin=0.003) if ep.objects else None
    matter = {}
    for n in ep.objects:
        P = pts[n]
        v = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1) * ep.fps]
        matter[n] = np.array([v[t] > 0.01 or bool(tl[t][n]["robot"]) for t in range(T)])
    for s in ep.hands:
        matter[f"{s} hand"] = np.array([any(s in tl[t][n]["robot"] for n in ep.objects) for t in range(T)]) if tl else np.zeros(T, bool)
    rows, bad = {}, []
    for key, P in pts.items():
        mk = matter[key].copy()
        mk[:ep.a] = False; mk[ep.b + 1:] = False
        if not mk.any():
            continue
        pc = np.einsum("tji,tj->ti", cR, P - cpos)          # camera frame: R^T (p - c)
        front = pc[:, 2] < -1e-6
        z = np.where(front, -pc[:, 2], 1.0)
        u = cam["cx"] + cam["fx"] * pc[:, 0] / z
        v = cam["cy"] - cam["fy"] * pc[:, 1] / z
        mu, mv = outside_frac * cam["W"], outside_frac * cam["H"]
        vis = front & (u >= -mu) & (u <= cam["W"] + mu) & (v >= -mv) & (v <= cam["H"] + mv)
        out = mk & ~vis
        runs = EA.runs_of(out)
        longest = max(runs, key=lambda r: r[1] - r[0]) if runs else None
        r = {"frames_that_matter": int(mk.sum()), "out_of_view": int(out.sum()), "out_frac": round(float(out.sum() / mk.sum()), 3),
             "longest_out_s": round((longest[1] - longest[0] + 1) / ep.fps, 2) if longest else 0.0,
             "first_out_frame": int(np.argmax(out)) if out.any() else None}
        if r["first_out_frame"] is not None:
            t = r["first_out_frame"]
            r["label"] = ep.labels[t]; r["pixel"] = [round(float(u[t]), 1), round(float(v[t]), 1)] if front[t] else "behind the camera"
        rows[key] = r
        if r["out_frac"] > max_out_frac or r["longest_out_s"] > max_out_s:
            bad.append(dict(r, body=key))
    issues = [f"{b['body']} is out of the camera's view on {b['out_of_view']} of its {b['frames_that_matter']} frames that matter "
              f"({100 * b['out_frac']:.0f}%; longest {b['longest_out_s']} s), first at frame {b['first_out_frame']} ({b['label']}, "
              f"pixel {b['pixel']}); keep the manipulation inside the frame the video shows (move the work toward the "
              f"image centre or lower the carry)" for b in bad]
    worst = max(bad, key=lambda b: b["out_frac"]) if bad else None
    return EA.finish(issues, worst, bodies=rows, camera={k: (v if not isinstance(v, np.ndarray) else v.tolist())
                                                          for k, v in cam.items() if k not in ("R",)}, frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(camera_visibility_audit(sys.argv[1], sys.argv[2], *sys.argv[3:4]), indent=1))
