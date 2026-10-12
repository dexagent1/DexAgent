"""Stage 4 configuration: one JSON file, paths relative to that file."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

ASPECT_TOL = 0.01                    # relative aspect-ratio difference allowed between images that must line up

DEFAULTS = {
    "num_samples": 10,
    "seed": 0,
    "max_attempts": None,            # default: 20 x num_samples
    "include_original": True,        # also render the un-randomized trajectory
    "sim_video": None,
    "robot": {
        "body_prefix": "robot_",
        "end_effectors": {},
        "init_jitter_rad": 0.0,      # random start offset of every arm joint
        "init_blend_frames": 90,     # frames over which that offset fades out
    },
    "manifest": None,                # Stage 3 trajectory_manifest.json: what the physics replay must re-check
    "goal": None,                    # {semantics, objects}: Stage 1's goals and the object -> body map, judged on every replay
    "randomize": {"objects": {}},
    "segments": {},
    "checks": {
        "ik_pos_tol_m": 0.002,       # reach: worst end-effector position error
        "ik_rot_tol_deg": 1.0,       # reach: worst end-effector rotation error
        "joint_speed_scale": 1.5,    # joint speed limit = scale x original max
        "joint_speed_floor": 0.5,    # ... plus this much, rad/s (or m/s)
        "penetration_tol_m": 0.003,  # new penetration allowed beyond the original
        "object_step_tol_m": 0.01,   # object motion continuity
        "physics_replay": True,      # re-run every accepted layout in contact physics (see stage4/physics.py)
        "physics_end_tol_m": 0.03,   # physics replay: an object ends this close to where the plan puts it
    },
    "physics": {"rules": []},        # run-time rules of the replay (screw threads that let go, latches), from Stage 3
    "sampling": {
        "adaptive": True,            # steer later draws away from the layouts earlier draws failed on
        "candidates": 6,             # layouts drawn per attempt once draws have failed; one is picked by weight
        "bandwidth": 0.5,            # width of a failure's influence, in units of each range's half width
        "strength": 0.85,            # how much a failure lowers the odds of a layout right on top of it
        "prescreen": True,           # check the reach of the anchored keyframes before the full adaptation
        "prescreen_tol_scale": 1.5,  # ... dropping layouts that miss by more than this times the reach tolerance
    },
    "retime": {"enabled": True},     # a layout rejected only on joint speed is slowed down where it is too fast
    "reexecute": {
        "enabled": True,             # re-run the Stage 3 plan on a layout the warp rejects (needs `plan`)
        "plan": None,                # plan.json: the executable Stage 3 plan, written by write_stage4_config
        "python": None,              # an interpreter that imports the tool library's executor (default: this one)
        "max_draws": None,           # draws that may fall back to re-execution (default 2 x num_samples)
        "max_attempts": 6,           # the executor's variants per subgoal, as in Stage 3
        "settle_s": None,            # physics before the first subgoal (default 0.3 s, 1 s with a moved cable)
        "keep_files": False,         # keep each re-execution's folder under outputs/reexecute/
    },
    "cameras": {"width": 640, "height": 480, "views": {}},
    "table": None,
    "inpaint": {
        "erase_prompts": ["hand", "arm"],
        "background_prompt": "An empty table surface with nothing on it, static background.",
        "max_frames": 85,            # frames sampled across the video (one VOID window)
        "min_score": 0.3,
        "mask_dilate_px": 11,
        "mask_scale": 1.15,
        "interaction_px": 61,
        "ring_diff_threshold": 18,
        "color_drift_tol": 4.0,      # frames whose color is further off are not used
        "gpu_memory_mode": "model_full_load",
        "masks_dir": None,           # bring your own erase masks (skips SAM 3)
        "clean_plate": None,         # bring your own clean plate (skips SAM 3 and VOID)
    },
    "render": {
        "stride": 1,
        "samples": 32,
        "chunks": 4,
        "exclude_geoms": ["^table", "groundplane", "floor"],
        "visual_groups": [0, 1, 2],
        "materials": [],
        "object_roughness": 0.65,
        "lighting": {
            "sun_energy": 3.0, "sun_angle": 0.30, "sun_euler": [0.55, 0.15, 0.9],
            "world_strength": 0.55, "fill_energy": 6.0, "world_color": None,
        },
        "crf": 15,
    },
}


def aspect_mismatch(a, b):
    """Relative difference between the aspect ratios of two (width, height) sizes."""
    ra, rb = float(a[0]) / float(a[1]), float(b[0]) / float(b[1])
    return abs(ra - rb) / rb


def intrinsics(cam, width, height):
    """(fx, fy, cx, cy) of a pinhole camera, in pixels of a width x height image.

    Pixel coordinates are measured from the image's top-left corner, so pixel (i, j) covers [i, i+1) x [j, j+1)
    and the centre of the image is (width / 2, height / 2): a point at camera coordinates (x right, y down, z
    forward) lands at u = cx + fx x / z, v = cy + fy y / z (Stage 2's camera.json, rigcam.project). A camera given
    by fx, fy (and optionally cx, cy) holds them for an image of `image_size` [w, h] (default width x height) and
    they are scaled to width x height; one given only by its vertical field of view (`fovy` or `fovy_deg`,
    degrees) has square pixels and its principal point at the centre. Raises ValueError on an invalid camera."""
    W, H = int(width), int(height)
    if W <= 0 or H <= 0:
        raise ValueError(f"image size must be positive, got {W} x {H}")
    if cam.get("fx") is not None or cam.get("fy") is not None:
        if cam.get("fx") is None or cam.get("fy") is None:
            raise ValueError("give both fx and fy (pixels), or neither")
        w0, h0 = cam.get("image_size") or (W, H)
        if float(w0) <= 0 or float(h0) <= 0:
            raise ValueError(f"image_size must be positive, got {cam.get('image_size')}")
        if aspect_mismatch((W, H), (w0, h0)) > ASPECT_TOL:
            raise ValueError(f"the intrinsics are given for a {w0} x {h0} image, whose aspect ratio differs from the "
                             f"{W} x {H} output: set cameras.width / height to the video's aspect ratio")
        sx, sy = W / float(w0), H / float(h0)
        fx, fy = float(cam["fx"]) * sx, float(cam["fy"]) * sy
        cx = float(cam["cx"]) * sx if cam.get("cx") is not None else W / 2.0
        cy = float(cam["cy"]) * sy if cam.get("cy") is not None else H / 2.0
    else:
        fovy = cam.get("fovy", cam.get("fovy_deg"))
        if fovy is None:
            raise ValueError("a camera needs fx and fy (pixels) or its vertical field of view fovy (degrees)")
        if not 0.0 < float(fovy) < 180.0:
            raise ValueError(f"fovy must be between 0 and 180 degrees, got {fovy}")
        fx = fy = (H / 2.0) / math.tan(math.radians(float(fovy)) / 2.0)
        cx, cy = W / 2.0, H / 2.0
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError(f"invalid intrinsics fx {fx}, fy {fy}, cx {cx}, cy {cy}")
    return fx, fy, cx, cy


def resolved_camera(cam, width, height):
    """The camera with its intrinsics written out for a width x height image (fx, fy, cx, cy, image_size, and the
    vertical field of view they imply), so every later step reads the same numbers."""
    fx, fy, cx, cy = intrinsics(cam, width, height)
    out = {k: v for k, v in cam.items() if k not in ("fovy_deg",)}
    out.update(fx=fx, fy=fy, cx=cx, cy=cy, image_size=[int(width), int(height)],
               fovy=math.degrees(2.0 * math.atan(0.5 * int(height) / fy)))
    return out


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """Config dict with the sample root and the resolved output directory."""

    root: Path
    out: Path

    def path(self, key):
        """A top-level path entry, resolved against the config's folder."""
        return self.path_of(self.get(key))

    def path_of(self, value):
        if value is None:
            return None
        p = Path(value)
        return p if p.is_absolute() else self.root / p


def load_config(path):
    path = Path(path).resolve()
    cfg = Config(_merge(DEFAULTS, json.loads(path.read_text())))
    cfg.root = path.parent
    for key in ("scene", "trajectory", "human_video"):
        assert cfg.get(key), f"config is missing '{key}'"
        assert cfg.path(key).exists(), f"{key}: {cfg.path(key)} does not exist"
    assert cfg["robot"]["end_effectors"], "config needs robot.end_effectors"
    assert cfg["cameras"].get("ego"), "config needs cameras.ego"
    cams = cfg["cameras"]
    for name, cam in [("ego", cams["ego"])] + [(f"views.{k}", v) for k, v in (cams.get("views") or {}).items()]:
        try:
            intrinsics(cam, cams["width"], cams["height"])
        except (ValueError, TypeError) as e:
            raise AssertionError(f"cameras.{name}: {e}") from None
    assert cfg.get("table"), "config needs table (top_z, center, half_size)"
    out = Path(cfg.get("output_dir", "outputs"))
    cfg.out = out if out.is_absolute() else cfg.root / out
    if cfg["max_attempts"] is None:
        cfg["max_attempts"] = 20 * int(cfg["num_samples"])
    if cfg["reexecute"].get("max_draws") is None:
        cfg["reexecute"]["max_draws"] = 2 * int(cfg["num_samples"])
    return cfg
