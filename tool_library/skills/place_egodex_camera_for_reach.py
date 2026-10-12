"""Where EgoDex's head camera stands on the bench so that both arms can reach every object."""
from __future__ import annotations

TOOL = {
    "name": "place_egodex_camera_for_reach",
    "title": "Place EgoDex camera for reach",
    "type": "skill",
    "category": "Perception & Grounding",
    "summary": "Searches the EgoDex head camera's foot position (x, y) and yaw on the bench so every Stage 1 object, back-projected from its frame-0 mask through the camera, lands inside a station arm's reachable band with margin and clear of the robot at home; returns the cam_xy (and yaw) to give read_egodex_episode, re-frames camera.json / hands.npz / episode.json, and re-checks the result with the exact arm IK.",
    "description": (
        "An EgoDex episode is filmed by a head camera with no fixed place on our bench: read_egodex_episode stands its "
        "foot at cam_xy (default (0.019, 0.05)) and points the rig's y along the view. Moving that foot moves every "
        "object rigidly, and whether the arms reach them is not monotonic in the shift (fixing one object can push two "
        "others out), so this skill searches it instead of the agent. Each object's footprint comes from its mask "
        "(measure_mask_footprint: the contact line's centre pushed back by half the object's width along the view, "
        "radius half the width, height the apparent height or a given one) or is given in metres. The arms' reach is "
        "precomputed once as a field: for each station, the arm IK residual of a palm-down hand hover_m above the "
        "object's top (object_reach_check's palm-down test, the same solver) on a grid over the table, turned into a "
        "signed distance to the band's edge; this is stricter than object_reach_check, which also accepts a level hand "
        "beside the object, because a pick from above needs the palm-down reach. A candidate placement (dx, dy, yaw about the camera foot) scores each "
        "object by its centre's distance inside the better station's band; candidates that put an object off the "
        "table's near edge or under the robot's home pose are rejected. The choice is the smallest move whose worst "
        "object is at least margin_m inside the band, else the placement with the largest worst-object margin. The "
        "result is then checked with the exact solver (palm-down hover and fingers-in, as object_reach_check does) "
        "before and after. With an episode folder it writes camera.json, hands.npz and episode.json re-framed to the "
        "new foot and yaw (read_egodex_episode takes cam_xy but not a yaw), a top view of the band with the objects "
        "before and after, and the reach field for reuse. Object positions already staged in a scene move by the "
        "returned `transform` (apply_transform). Masks must be of objects standing on the table: a part on top of "
        "another (a cap on a bottle) back-projects far behind it and drags the camera; leave it out with exclude. Objects "
        "lying on the table (cords, ribbons, cloth: Stage 1's deformables) go in `flat`. The camera stays where it is when "
        "that placement already gives every object its margin and clearance; otherwise `reason` says what it violated."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Receives placement.json, reach_field.json, placement.png and, with episode_dir, the re-framed camera.json, hands.npz, episode.json."},
        {"name": "episode_dir", "type": "path | None", "description": "read_egodex_episode's out_dir (camera.json, episode.json, hands.npz)."},
        {"name": "camera", "type": "path | dict | None", "description": "camera.json instead of episode_dir (its foot pos[:2] is the current cam_xy)."},
        {"name": "masks", "type": "path | dict | None", "description": "Frame-0 object masks: a folder of <id>.png (hands*.png is the ignore mask) or {id: png}."},
        {"name": "footprints", "type": "dict | None", "description": "{id: {center: [x, y], radius, height}} in metres in the camera's current rig frame (instead of masks)."},
        {"name": "heights", "type": "dict | None", "description": "{id: height m} overriding the apparent heights read off the masks."},
        {"name": "objects", "type": "list[str] | None", "description": "Only these mask ids are placed: the Stage 1 objects the hands manipulate. Other masks in the folder (helper masks, static fixtures such as a dispenser the robot never touches) are ignored. Default every mask."},
        {"name": "flat", "type": "list[str] | None", "description": "Mask ids of objects lying flat on the table (Stage 1's deformable cords, ribbons, cloth): their footprint is the whole mask projected onto the table, 2 cm high, split into zones along its length (one per 12 cm, ids '<id>@k') that must each be reachable, instead of a standing object's contact line and apparent height (a lying ribbon across the frame would read as a 19 cm tall column)."},
        {"name": "exclude", "type": "list[str] | None", "description": "Mask ids to leave out: parts that do not stand on the table (a cap on a bottle back-projects far behind it), extra masks of one object. A mask covering two or more others (a union) is skipped automatically, as is one hidden by the hands."},
        {"name": "setup_dir", "type": "path | None", "description": "Setup folder (scene.xml with the stations at home, setup.yaml); default the repository's setup/."},
        {"name": "margin_m", "type": "float", "description": "How far inside the band every object centre must be. Default 0.03."},
        {"name": "x_range / y_range", "type": "list[2]", "description": "Search range of the foot shift in metres. Defaults [-0.15, 0.15] and [-0.12, 0.25]."},
        {"name": "yaw_range_deg", "type": "list[2]", "description": "Search range of the yaw about the camera foot. Default [-15, 15]; [0, 0] keeps the view straight into the work."},
        {"name": "xy_step_m / yaw_step_deg", "type": "float", "description": "Search steps. Defaults 0.01 and 2.5."},
        {"name": "hover_m", "type": "float", "description": "Palm-down wrist height above an object's top (object_reach_check's). Default 0.06."},
        {"name": "field", "type": "path | dict | None", "description": "A reach_field.json from an earlier call on the same setup (skips the precomputation, about 30-60 s per height)."},
        {"name": "field_step_m", "type": "float", "description": "Grid step of the reach field. Default 0.05."},
        {"name": "verify", "type": "bool", "description": "Re-check every object before and after with the exact arm IK. Default true."},
    ],
    "outputs": [
        {"name": "cam_xy / yaw_deg", "type": "list[2] / float", "description": "The camera foot and yaw to use (read_egodex_episode(cam_xy=...) when yaw_deg is 0, else the re-framed files)."},
        {"name": "transform", "type": "dict", "description": "{dx, dy, dyaw_deg, about}: old rig -> new rig, p' = about + d + Rz(dyaw) (p - about)."},
        {"name": "objects", "type": "dict", "description": "Per object: center before/after, radius, height, margin_m before/after, best_side, reachable before/after (exact IK)."},
        {"name": "ok", "type": "bool", "description": "Every object reachable after the move (exact check) and at least margin_m inside the band (field)."},
        {"name": "notes", "type": "list[str]", "description": "Masks skipped and why (a union of others, hidden by the hands)."},
        {"name": "files", "type": "dict", "description": "placement_json, reach_field_json, png, and with episode_dir camera_json, hands_npz, episode_json."},
    ],
    "status": "tested",
    "entrypoint": "place_egodex_camera_for_reach",
    "requirements": ["mujoco>=3.2", "numpy", "scipy", "opencv-python", "pyyaml", "matplotlib"],
    "simulators": ["MuJoCo"],
    "tags": ["egodex", "camera", "reach", "placement", "workspace", "stage 2"],
    "cli": "python -m tool_library.skills.place_egodex_camera_for_reach out/ --episode-dir stage2/ --masks stage2/masks",
}

import json
import math
import sys
from pathlib import Path

import numpy as np

from tool_library.common import human_prior as HP
from tool_library.common import rigcam as RC

DEFAULT_SETUP = Path(__file__).resolve().parents[2] / "setup"


def _rz(deg):
    a = math.radians(float(deg))
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def apply_transform(transform, xy, yaw_deg=None):
    """Move rig-frame points [N, 2] (and yaws) by the returned transform."""
    a = np.asarray(transform["about"][:2], float)
    d = np.array([transform["dx"], transform["dy"]], float)
    R = _rz(transform["dyaw_deg"])[:2, :2]
    P = a + d + (np.atleast_2d(np.asarray(xy, float)) - a) @ R.T
    if yaw_deg is None:
        return P
    return P, np.asarray(yaw_deg, float) + float(transform["dyaw_deg"])


def _flat_footprint(path, cam, height):
    """A lying object's footprint: every mask pixel on the table plane; centre the median, radius half the minor extent."""
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    m = (a if a.ndim == 2 else a[..., 0]) > 0
    v, u = np.nonzero(m)
    if len(u) < 20:
        raise ValueError("too few mask pixels")
    step = max(1, len(u) // 4000)
    P = RC.pixel_on_plane(cam, np.stack([u[::step], v[::step]], axis=1).astype(float), 0.0)[:, :2]
    if len(P) < 10:
        raise ValueError("the mask does not meet the table plane")
    c = np.median(P, axis=0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    minor = (P - c) @ V[:, 0]
    r = float(np.clip(0.5 * (np.percentile(minor, 90) - np.percentile(minor, 10)), 0.005, 0.05))
    return {"center": c.tolist(), "radius": r, "height": float(height), "source": "mask (flat)"}


FLAT_ZONE_M = 0.12          # a lying object longer than this is placed as several zones along its length, each reachable


def _flat_zones(path, cam, height, zone_m=FLAT_ZONE_M):
    """A lying object's footprint as zones along its length ({id suffix: footprint}): a 30 cm ribbon is not a disk at its
    centroid, its ends must be reachable too. One zone per zone_m of length (1 to 5)."""
    import cv2
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    m = (a if a.ndim == 2 else a[..., 0]) > 0
    v, u = np.nonzero(m)
    if len(u) < 20:
        raise ValueError("too few mask pixels")
    step = max(1, len(u) // 4000)
    P = RC.pixel_on_plane(cam, np.stack([u[::step], v[::step]], axis=1).astype(float), 0.0)[:, :2]
    c = np.median(P, axis=0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    major, minor = (P - c) @ V[:, 1], (P - c) @ V[:, 0]
    lo, hi = np.percentile(major, 3), np.percentile(major, 97)
    n = int(np.clip(np.ceil((hi - lo) / float(zone_m)), 1, 5))
    edges = np.linspace(lo, hi, n + 1)
    out = {}
    for k in range(n):
        sel = (major >= edges[k]) & (major <= edges[k + 1])
        if sel.sum() < 5:
            continue
        mi = minor[sel]
        r = float(np.clip(0.5 * (np.percentile(mi, 90) - np.percentile(mi, 10)), 0.005, 0.05))
        out["" if n == 1 else f"@{k + 1}"] = {"center": np.median(P[sel], axis=0).tolist(), "radius": r, "height": float(height),
                                               "source": f"mask (flat, zone {k + 1}/{n})"}
    return out


def _footprints_from_masks(masks, cam, heights, exclude=(), flat=(), objects=None):
    import cv2
    from tool_library.skills.measure_mask_footprint import measure_mask_footprint
    if isinstance(masks, (str, Path)):
        md = Path(masks)
        if not md.is_dir():
            raise ValueError(f"masks must be a folder of <id>.png or a dict, not {masks}")
        files = {p.stem: str(p) for p in sorted(md.glob("*.png")) if not p.stem.startswith("hand")}
        ignore = next((str(p) for p in sorted(md.glob("hand*.png"))), None)
    else:
        files, ignore = {str(k): str(v) for k, v in dict(masks).items()}, None
    files = {k: v for k, v in files.items() if k not in set(exclude or ()) and (objects is None or k in set(objects))}
    if not files:
        raise ValueError("no object masks found")
    notes = []
    # a mask that covers two or more of the others is their union (rings_union.png), not one more object
    bits = {}
    for oid, f in files.items():
        a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if a is not None:
            bits[oid] = (a if a.ndim == 2 else a[..., 0]) > 0
    for oid, m in list(bits.items()):
        inside = [o for o, n in bits.items() if o != oid and n.sum() > 0 and (n & m).sum() >= 0.8 * n.sum()]
        if len(inside) >= 2 and oid not in set(objects or ()):   # an object listed explicitly (a pile holding beads) stays
            files.pop(oid, None)
            notes.append(f"{oid}: skipped, its mask is the union of {inside}")
    out = {}
    R = np.asarray(cam["R_cv"], float)
    fwd = R[:, 2].copy(); fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    for oid, f in files.items():
        if oid in set(flat or ()):
            try:
                for suffix, fp in _flat_zones(f, cam, (heights or {}).get(oid, 0.02)).items():
                    out[oid + suffix] = fp
            except ValueError as e:
                notes.append(f"{oid}: skipped, {e}")
            continue
        try:
            m = measure_mask_footprint(f, cam, ignore_mask=ignore)
        except ValueError as e:
            notes.append(f"{oid}: skipped, {e} (hidden at frame 0: give its footprint, or segment a frame where it shows)")
            continue
        c = m["contact"]
        width = float(c["width_m"])
        h = float((heights or {}).get(oid, np.clip(m["apparent_height_m"], 0.01, 0.30)))
        # the contact line is the object's near edge: its centre lies half a width further along the view
        ctr = np.asarray(c["center_xy"], float) + 0.5 * width * fwd[:2]
        out[oid] = {"center": ctr.tolist(), "radius": 0.5 * width, "height": h, "source": "mask"}
    if not out:
        raise ValueError("no object footprint could be read from the masks: " + "; ".join(notes))
    return out, notes


def _robot_keepout(scene_xml, z_max, xs, ys):
    """Distance (m) on the grid to the nearest home-pose robot point lower than z_max (a hand or an arm the object
    would stand under or in)."""
    import cv2
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    occ = np.zeros((len(ys), len(xs)), np.uint8)
    res = xs[1] - xs[0]
    for g in range(m.ngeom):
        if not (m.body(int(m.geom_bodyid[g])).name or "").startswith("robot_"):
            continue
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[g])
            V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]][::7].astype(float)
            P = d.geom_xpos[g] + V @ d.geom_xmat[g].reshape(3, 3).T
        else:
            P = d.geom_xpos[g][None, :]
        P = P[P[:, 2] < z_max]
        if not len(P):
            continue
        ix = np.round((P[:, 0] - xs[0]) / res).astype(int)
        iy = np.round((P[:, 1] - ys[0]) / res).astype(int)
        k = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        occ[iy[k], ix[k]] = 1
    if not occ.any():
        return np.full(occ.shape, 9.0)
    return cv2.distanceTransform((1 - occ).astype(np.uint8), cv2.DIST_L2, 5) * res


def _exact_reach(scene_xml, objs, hover_m, pos_tol_m=0.02, tilt_tol_deg=20.0):
    """object_reach_check's test on points: a palm-down hand hover_m above the top, or a fingers-in hand beside."""
    import mujoco
    from tool_library.skills.optimize_pose_for_reach import _reach_solver
    from tool_library.verifiers.arm_workspace_map import named_orientation
    m = mujoco.MjModel.from_xml_path(str(scene_xml))
    solvers = {s: _reach_solver(m, s) for s in ("left", "right")}
    out = {}
    tol = lambda e: e[0] <= pos_tol_m and e[1] <= math.radians(tilt_tol_deg)
    for oid, o in objs.items():
        c, top = np.asarray(o["center"], float), float(o["height"])
        best = None
        # palm-down over the object first; the fingers-in test only when no station holds a palm-down hand there
        # (the verdict equals object_reach_check's: reachable when any of the four is within tolerance)
        for oname in ("palm_down", "fingers_in"):
            for s, solve in solvers.items():
                if oname == "palm_down":
                    p, R = np.array([c[0], c[1], top + hover_m]), named_orientation(s, "palm_down")
                else:
                    p, R = np.array([c[0] + (0.17 if s == "right" else -0.17), c[1], max(top * 0.5, 0.04)]), named_orientation(s, "fingers_in")
                e = solve(p, R)
                if best is None or e[0] + 0.05 * e[1] < best[0] + 0.05 * best[1]:
                    best = (e[0], e[1], s, oname)
                if tol(e):
                    break
            if tol(best):
                break
        out[oid] = {"reachable": bool(tol(best)), "side": best[2],
                    "orientation": best[3], "residual_mm": round(1000 * best[0], 1), "residual_deg": round(math.degrees(best[1]), 1)}
    return out


def place_egodex_camera_for_reach(out_dir, episode_dir=None, camera=None, masks=None, footprints=None, heights=None, setup_dir=None,
                                  margin_m=0.03, x_range=(-0.15, 0.15), y_range=(-0.12, 0.25), yaw_range_deg=(-15.0, 15.0),
                                  xy_step_m=0.01, yaw_step_deg=2.5, hover_m=0.06, field=None, field_step_m=0.05, verify=True,
                                  exclude=None, flat=None, objects=None):
    import yaml
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    setup = Path(setup_dir) if setup_dir else DEFAULT_SETUP
    scene_xml = setup / "scene.xml"
    sy = yaml.safe_load((setup / "setup.yaml").read_text())
    tab = sy.get("table", {})
    near_y = float(tab["center"][1]) - float(tab["half_extents"][1]) if "center" in tab else 0.05
    if episode_dir is None and camera is None and footprints is None:
        raise ValueError("give episode_dir (read_egodex_episode's out_dir) or camera, plus masks or footprints")
    cam, episode = None, None
    if episode_dir is not None:
        ed = Path(episode_dir)
        cam = RC.load(ed / "camera.json")
        if (ed / "episode.json").exists():
            episode = json.loads((ed / "episode.json").read_text())
    elif camera is not None:
        cam = RC.load(camera) if isinstance(camera, (str, Path)) else dict(camera)
    if masks is None and footprints is None:
        raise ValueError("give masks (frame-0 object masks) or footprints")
    notes = []
    if footprints is not None:
        objs = {}
        for oid, f in footprints.items():
            if "center" not in f:
                raise ValueError(f"footprint {oid} needs center [x, y]")
            if oid in set(exclude or ()):
                continue
            objs[oid] = {"center": [float(v) for v in f["center"][:2]], "radius": float(f.get("radius", 0.03)),
                         "height": float((heights or {}).get(oid, f.get("height", 0.05))), "source": "given"}
    else:
        if cam is None:
            raise ValueError("masks need the camera (episode_dir or camera)")
        objs, notes = _footprints_from_masks(masks, cam, heights, exclude, flat, objects)
    if not objs:
        raise ValueError("every object was excluded; nothing to place")
    foot0 = np.array(cam["pos"][:2], float) if cam is not None else np.array([0.019, 0.05])
    if episode and "rig" in episode and "cam_xy" in episode["rig"]:
        foot0 = np.asarray(episode["rig"]["cam_xy"], float)
    # reach field at the heights the objects need (one per 3 cm of object top)
    levels = sorted({round((o["height"] + hover_m) / 0.03) * 0.03 for o in objs.values()})
    if field is not None:
        fld = json.loads(Path(field).read_text()) if isinstance(field, (str, Path)) else field
        have = np.asarray(fld["z_levels"], float)
        if any(np.min(np.abs(have - z)) > 0.031 for z in levels):
            raise ValueError(f"the given field covers heights {have.tolist()}, the objects need {levels}; omit field to recompute")
    else:
        fld = HP.reach_field(scene_xml, z_levels=levels, step_m=field_step_m)
    field_json = HP.save_json(fld, out / "reach_field.json")
    xs, ys, maps = HP.field_margin_maps(fld)
    zl = np.asarray(fld["z_levels"], float)
    zmax = max(o["height"] for o in objs.values()) + 0.01
    keep = _robot_keepout(scene_xml, zmax, xs, ys)
    ids = list(objs)
    C = np.array([objs[i]["center"] for i in ids])
    rad = np.array([objs[i]["radius"] for i in ids])
    lev = np.array([int(np.argmin(np.abs(zl - (objs[i]["height"] + hover_m)))) for i in ids])

    def margins(P):
        """Per object: the better station's signed distance (m) at its centre, and which station."""
        best = np.full(len(ids), -9.0); side = np.array(["left"] * len(ids), dtype=object)
        for s, M in maps.items():
            for k in range(len(ids)):
                v = float(HP.sample_map(xs, ys, M[lev[k]], P[k:k + 1])[0])
                if v > best[k]:
                    best[k], side[k] = v, s
        return best, side

    def transformed(dx, dy, dyaw):
        return foot0 + np.array([dx, dy]) + (C - foot0) @ _rz(dyaw)[:2, :2].T

    m0, s0 = margins(C)
    cands = []
    dxs = np.arange(x_range[0], x_range[1] + 1e-9, xy_step_m)
    dys = np.arange(y_range[0], y_range[1] + 1e-9, xy_step_m)
    yaws = np.arange(yaw_range_deg[0], yaw_range_deg[1] + 1e-9, yaw_step_deg) if yaw_range_deg[1] > yaw_range_deg[0] else np.array([float(yaw_range_deg[0])])
    # vectorised evaluation: every candidate's object centres at once, per object
    for dyaw in yaws:
        Rz = _rz(dyaw)[:2, :2]
        rel = (C - foot0) @ Rz.T
        GX, GY = np.meshgrid(dxs, dys, indexing="ij")
        n = GX.size
        best = np.full((n, len(ids)), -9.0)
        clear = np.ones(n, bool)
        for k in range(len(ids)):
            P = np.stack([foot0[0] + GX.ravel() + rel[k, 0], foot0[1] + GY.ravel() + rel[k, 1]], axis=1)
            for s, M in maps.items():
                best[:, k] = np.maximum(best[:, k], HP.sample_map(xs, ys, M[lev[k]], P))
            clear &= (P[:, 1] - rad[k]) >= near_y + 0.02
            clear &= HP.sample_map(xs, ys, keep, P) >= rad[k] + 0.01
        worst = best.min(axis=1)
        for i in np.flatnonzero(clear):
            cands.append((float(worst[i]), float(best[i].mean()), float(GX.ravel()[i]), float(GY.ravel()[i]), float(dyaw)))
    if not cands:
        raise ValueError("no placement in the search range keeps every object on the table and clear of the robot: widen x_range / y_range")
    cost = lambda c: math.hypot(c[2], c[3]) + 0.003 * abs(c[4])
    good = [c for c in cands if c[0] >= margin_m]
    if good:
        pick = min(good, key=lambda c: (round(cost(c), 3), -c[1]))
        reason = f"smallest move with every object at least {1000 * margin_m:.0f} mm inside a station's band"
    else:
        top = max(c[0] for c in cands)
        near = [c for c in cands if c[0] >= top - 0.005]           # within 5 mm of the best: the smallest move among them
        pick = min(near, key=lambda c: (round(cost(c), 3), -c[1]))
        reason = f"no placement reaches {1000 * margin_m:.0f} mm for every object; the one with the largest worst-object margin ({1000 * top:.0f} mm)"
    worst, mean, dx, dy, dyaw = pick
    if (abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dyaw) > 1e-9) and float(m0.min()) >= margin_m:
        # the camera moves although every object already has its margin: say what the current placement violates
        why = [f"{oid} {'off the table near edge' if (C[k, 1] - rad[k]) < near_y + 0.02 else 'under the robot at home'}"
               for k, oid in enumerate(ids)
               if (C[k, 1] - rad[k]) < near_y + 0.02 or HP.sample_map(xs, ys, keep, C[k:k + 1])[0] < rad[k] + 0.01]
        reason += f"; the current placement already has the margin but is rejected: {', '.join(why) or 'clearance'}"
    C1 = transformed(dx, dy, dyaw)
    m1, s1 = margins(C1)
    transform = {"dx": round(dx, 4), "dy": round(dy, 4), "dyaw_deg": round(dyaw, 3), "about": [float(foot0[0]), float(foot0[1])]}
    cam_xy = [round(float(foot0[0] + dx), 4), round(float(foot0[1] + dy), 4)]
    rep_obj = {}
    for k, oid in enumerate(ids):
        rep_obj[oid] = {"center_before": [round(float(v), 4) for v in C[k]], "center_after": [round(float(v), 4) for v in C1[k]],
                        "radius": round(float(rad[k]), 4), "height": round(float(objs[oid]["height"]), 4),
                        "margin_before_m": round(float(m0[k]), 4), "margin_after_m": round(float(m1[k]), 4), "best_side": str(s1[k])}
    ok = bool(worst >= margin_m)
    if verify:
        before = _exact_reach(scene_xml, {oid: {"center": C[k], "height": objs[oid]["height"]} for k, oid in enumerate(ids)}, hover_m)
        after = _exact_reach(scene_xml, {oid: {"center": C1[k], "height": objs[oid]["height"]} for k, oid in enumerate(ids)}, hover_m)
        for oid in ids:
            rep_obj[oid]["reachable_before"] = before[oid]["reachable"]
            rep_obj[oid]["reachable_after"] = after[oid]["reachable"]
            rep_obj[oid]["exact_after"] = after[oid]
        ok = ok and all(after[o]["reachable"] for o in ids)
    files = {"reach_field_json": field_json}
    if episode_dir is not None:
        files.update(_reframe(Path(episode_dir), out, transform, cam_xy))
    files["png"] = _plot(out / "placement.png", xs, ys, maps, lev, ids, C, C1, rad, foot0, cam_xy, dyaw)
    res = {"ok": ok, "cam_xy": cam_xy, "yaw_deg": round(dyaw, 3), "transform": transform, "reason": reason,
           "worst_margin_before_m": round(float(m0.min()), 4), "worst_margin_after_m": round(float(m1.min()), 4),
           "margin_m": margin_m, "objects": rep_obj, "n_candidates": len(cands), "notes": notes,
           "read_egodex_args": {"cam_xy": cam_xy} if abs(dyaw) < 1e-9 else None,
           "note": ("call read_egodex_episode(cam_xy=...) or use the re-framed files" if abs(dyaw) < 1e-9 else
                    "the yaw is not a read_egodex_episode parameter: use the re-framed camera.json / hands.npz / episode.json"),
           "files": files}
    res["files"]["placement_json"] = HP.save_json(res, out / "placement.json")
    return res


def _reframe(ed, out, tr, cam_xy):
    """camera.json, hands.npz and episode.json of a read_egodex_episode folder in the moved rig frame."""
    a = np.array([tr["about"][0], tr["about"][1], 0.0])
    d = np.array([tr["dx"], tr["dy"], 0.0])
    Rz = _rz(tr["dyaw_deg"])
    move = lambda P: a + d + (np.asarray(P, float) - a) @ Rz.T
    files = {}
    cam = RC.load(ed / "camera.json")
    extra = {k: v for k, v in cam.items() if k not in ("width", "height", "fx", "fy", "cx", "cy", "fovy_deg", "pos", "quat_wxyz", "R_cv",
                                                       "pitch_deg", "yaw_deg", "height_above_table_m")}
    extra["placed_for_reach"] = {"cam_xy": cam_xy, "yaw_deg": tr["dyaw_deg"]}
    new = RC.camera_dict(cam["width"], cam["height"], cam["fx"], cam["fy"], cam["cx"], cam["cy"], Rz @ np.asarray(cam["R_cv"]),
                         move(cam["pos"]), extra)
    files["camera_json"] = RC.save(new, out / "camera.json")
    if (ed / "hands.npz").exists():
        with np.load(ed / "hands.npz", allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}
        X = np.asarray(z["rig_xyz"], float)
        z["rig_xyz"] = move(X.reshape(-1, 3)).reshape(X.shape).astype(np.float32)
        Rwr = np.asarray(z["R_world_rig"], float)
        org = np.asarray(z["origin_world"], float)
        z["R_world_rig"] = Rwr @ Rz.T
        z["origin_world"] = org + Rwr @ (a - Rz.T @ (a + d))
        np.savez_compressed(out / "hands.npz", **z)
        files["hands_npz"] = str(out / "hands.npz")
    if (ed / "episode.json").exists():
        ep = json.loads((ed / "episode.json").read_text())
        Rwr = np.asarray(ep["rig"]["R_world_rig"], float)
        org = np.asarray(ep["rig"]["origin_world"], float)
        ep["rig"] = {"R_world_rig": (Rwr @ Rz.T).tolist(), "origin_world": (org + Rwr @ (a - Rz.T @ (a + d))).tolist(),
                     "cam_xy": cam_xy, "yaw_deg": tr["dyaw_deg"]}
        ep["camera_json"] = files["camera_json"]
        if "hands_npz" in files:
            ep["hands_npz"] = files["hands_npz"]
        (out / "episode.json").write_text(json.dumps(ep, indent=1))
        files["episode_json"] = str(out / "episode.json")
    return files


def _plot(path, xs, ys, maps, lev, ids, C, C1, rad, foot0, cam_xy, dyaw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    k0 = int(np.bincount(lev).argmax())
    U = np.maximum(maps["left"][k0], maps["right"][k0]) if "left" in maps and "right" in maps else next(iter(maps.values()))[k0]
    ax.contourf(xs, ys, U, levels=[0, 0.02, 0.04, 0.08, 1.0], colors=["#d8ecd3", "#a9d6a0", "#6fb965", "#3f8f3a"], alpha=0.8)
    ax.contour(xs, ys, U, levels=[0.0], colors=["#2a6a26"], linewidths=1.0)
    for k, oid in enumerate(ids):
        ax.add_patch(plt.Circle(C[k], rad[k], fill=False, ls="--", color="#999999"))
        ax.add_patch(plt.Circle(C1[k], rad[k], fill=True, color="#c0392b", alpha=0.6))
        ax.annotate(oid, C1[k], fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.plot(*foot0, "x", color="#999999", ms=8)
    ax.plot(*cam_xy, "x", color="#c0392b", ms=8)
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(min(ys[0], foot0[1] - 0.05), ys[-1]); ax.set_aspect("equal")
    ax.set_xlabel("rig x [m]"); ax.set_ylabel("rig y [m]")
    ax.set_title(f"reach band (green, darker = deeper) and objects: grey before, red after\ncamera foot {cam_xy}, yaw {dyaw:+.1f} deg", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return str(path)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("out_dir"); ap.add_argument("--episode-dir"); ap.add_argument("--masks"); ap.add_argument("--field")
    ap.add_argument("--margin-m", type=float, default=0.03)
    a = ap.parse_args(argv)
    r = place_egodex_camera_for_reach(a.out_dir, episode_dir=a.episode_dir, masks=a.masks, field=a.field, margin_m=a.margin_m)
    print(json.dumps({k: r[k] for k in ("ok", "cam_xy", "yaw_deg", "reason", "worst_margin_before_m", "worst_margin_after_m")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
