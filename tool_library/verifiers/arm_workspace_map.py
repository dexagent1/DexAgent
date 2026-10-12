"""Where can a station's arm put the hand, with a given hand orientation?"""
from __future__ import annotations

TOOL = {
    "name": "arm_workspace_map",
    "title": "Arm workspace map",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "Samples a grid of wrist positions over the table for one or several hand orientations and reports, per orientation, "
               "the region the station's arm reaches (residual below tolerance), its far edge in y, and whether given points of "
               "interest (an object's part, a handle) are inside; a map image shows the reachable cells.",
    "description": (
        "Reach is a property of the station, not the pose: the same hand orientation is reachable at 0.5 m and not at "
        "0.65 m, and at the same place a level hand is reachable where a fingers-down hand is not. Before choosing a grasp "
        "or a strategy, ask this verifier where the work can happen. For each named orientation (`orientations`: "
        "'palm_down' (fingers forward +y), 'palm_up', 'fingers_down' (palm toward the other station), 'fingers_in' "
        "(level, palm down, fingers pointing across the table toward the other station), 'fingers_in_palm_up', 'palm_in' "
        "(vertical, fingers forward, palm toward the other station), or any euler [rx, ry, rz] of the hand root, whose "
        "local z runs along the fingers and local x is the palm normal) it solves the arm's IK (position plus the home-pose mount rotation, multi-start) on a grid of "
        "(x, y) at heights `z_m` and marks the cells whose residual is within `pos_tol_m` / `tilt_tol_deg`. The report "
        "names, per orientation, the reachable fraction, the far edge (largest reachable y at each x band), and, for every "
        "point in `points`, whether it is reachable with that orientation and the residual there. The side's own "
        "orientation convention is handled (a right-hand 'fingers_in' points -x, a left-hand one +x). Solved cells are "
        "cached on disk per (robot signature, side, orientation, cell): the signature hashes only what reach depends on "
        "(the arm chain's placement and joints, its home posture, the hand mount and the solver settings), so a Stage 2 "
        "scene that adds objects to the default setup reuses the cache shipped for the default setup "
        "(tool_library/assets/arm_workspace_cache.json, the default grid for both stations), and a repeated call on any "
        "station answers from the user cache ($DEXAGENT_CACHE_DIR, else $XDG_CACHE_HOME/dexagent, else ~/.cache/dexagent) "
        "instead of re-solving for minutes."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF."},
        {"name": "side", "type": "str", "description": "'left' or 'right'."},
        {"name": "orientations", "type": "list | None", "description": "Names or eulers. Default ['palm_down', 'fingers_in', 'fingers_down', 'palm_in']."},
        {"name": "points", "type": "dict | None", "description": "{label: [x, y, z]} wrist positions to classify (an object's edge, a handle)."},
        {"name": "x_range", "type": "list", "description": "[min, max] in metres. Default [-0.4, 0.4]."},
        {"name": "y_range", "type": "list", "description": "Default [0.1, 0.9]."},
        {"name": "z_m", "type": "float | list", "description": "Height(s) of the wrist. Default 0.12."},
        {"name": "step_m", "type": "float", "description": "Grid step. Default 0.1 (a 9 x 9 grid per orientation: about a minute per orientation on a busy node when solved, instant from the cache; 0.05 is four times finer and slower)."},
        {"name": "pos_tol_m", "type": "float", "description": "Default 0.015."},
        {"name": "tilt_tol_deg", "type": "float", "description": "Default 15."},
        {"name": "out_png", "type": "str | None", "description": "Map image (one panel per orientation and height)."},
        {"name": "out_json", "type": "str | None", "description": "Report."},
        {"name": "cache", "type": "bool | str", "description": "True (default): read the shipped and the user cache, write new cells to the user cache; a path: read and write that json file (the shipped cache is still read); False: solve every cell."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok (every requested point reachable with at least one orientation), per orientation: euler, reachable_fraction, far_edge_y {x: y}, cells [[x, y, z, pos_err_mm, rot_err_deg]], points {label: {ok, pos_err_mm, rot_err_deg}}; best_orientation per point; verdict; cache {hits, solved, file, signature}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "matplotlib"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "reach", "arm", "workspace", "planning"],
    "used_in": ["Giftbox Packing", "Place Toy in Drawer", "Open the Bottle"],
    "status": "tested",
    "entrypoint": "arm_workspace_map",
}

import hashlib
import json
import os
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import hand as H

CACHE_VERSION = "1"          # bump when the reach solver changes what it returns for a cell
SHIPPED_CACHE = Path(__file__).resolve().parents[1] / "assets" / "arm_workspace_cache.json"


def user_cache_path():
    """Where solved cells are kept between calls: $DEXAGENT_CACHE_DIR, else $XDG_CACHE_HOME/dexagent, else
    ~/.cache/dexagent."""
    base = os.environ.get("DEXAGENT_CACHE_DIR") or (Path(os.environ["XDG_CACHE_HOME"]) / "dexagent" if os.environ.get("XDG_CACHE_HOME")
                                                    else Path.home() / ".cache" / "dexagent")
    return Path(base) / "arm_workspace_cache.json"


def robot_signature(model, side):
    """A hash of everything one station's reach depends on and nothing else: the bodies from the arm's flange down to
    the world (placement of the station included), the arm joints (type, axis, anchor, range), the home posture the IK
    starts from, the hand mount (flange in the hand-root frame) and the solver settings. Objects in the scene do not
    enter: a Stage 2 scene on the default setup has the default setup's signature."""
    from tool_library.common import armhand as AH
    from tool_library.common import episode as E
    parts = E.robot_parts(model)
    pt = parts[side]
    chunks = [CACHE_VERSION, side, json.dumps(AH.DEFAULTS, sort_keys=True)]
    b = int(pt["gripper"])
    while b > 0:
        chunks.append(np.round(np.concatenate([model.body_pos[b], model.body_quat[b]]), 6).tobytes())
        b = int(model.body_parentid[b])
    for j in pt["arm_joints"]:
        chunks.append(np.round(np.concatenate([[float(model.jnt_type[j])], model.jnt_axis[j], model.jnt_pos[j], model.jnt_range[j]]), 6).tobytes())
    q0 = np.asarray(model.key_qpos[0] if model.nkey else model.qpos0, float)
    chunks.append(np.round(q0[pt["arm_qadr"]], 6).tobytes())
    p_m, R_m, _src = AH.mount(model, side, parts=parts)
    chunks.append(np.round(np.concatenate([p_m, np.asarray(R_m).ravel()]), 5).tobytes())
    h = hashlib.sha1()
    for c in chunks:
        h.update(c if isinstance(c, bytes) else str(c).encode())
    return h.hexdigest()[:16]


def _orient_key(R):
    return ",".join(f"{v:.4f}" for v in np.asarray(R, float).ravel()).replace("-0.0000", "0.0000")


def _cell_key(p):
    return ",".join(f"{float(v):.4f}" for v in p).replace("-0.0000", "0.0000")


def _read_cache(path):
    try:
        d = json.loads(Path(path).read_text())
        return d.get("entries", {}) if d.get("version") == CACHE_VERSION else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _write_cache(path, new_entries):
    """Merge `new_entries` into the json at `path` (what another process wrote meanwhile is kept); atomic replace."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cur = _read_cache(path)
        for k, cells in new_entries.items():
            cur.setdefault(k, {}).update(cells)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"version": CACHE_VERSION, "entries": cur}, separators=(",", ":")))
        os.replace(tmp, path)
        return True
    except OSError:
        return False

def _R_from_dirs(fingers, palm):
    """Hand root rotation from world directions: local z runs along the fingers, local x is the palm normal (out of
    the palm), local y the thumb side (+y on the right hand, -y on the left)."""
    z = np.asarray(fingers, float); z /= np.linalg.norm(z)
    x = np.asarray(palm, float); x = x - z * (z @ x); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def named_orientation(side, name):
    """World direction pairs (fingers, palm normal) of the named orientations; 'in' means toward the other station."""
    inward = -1.0 if side == "right" else 1.0
    table = {"palm_down": ((0, 1, 0), (0, 0, -1)), "palm_up": ((0, 1, 0), (0, 0, 1)), "fingers_down": ((0, 0, -1), (inward, 0, 0)),
             "fingers_in": ((inward, 0, 0), (0, 0, -1)), "fingers_in_palm_up": ((inward, 0, 0), (0, 0, 1)), "palm_in": ((0, 1, 0), (inward, 0, 0))}
    if name not in table:
        raise ValueError(f"unknown orientation {name!r}; use one of {list(table)} or an euler")
    return _R_from_dirs(*table[name])


def _orientation(side, o):
    if isinstance(o, str):
        return o, named_orientation(side, o)
    e = np.asarray(o, float)
    return "euler[" + ",".join(f"{v:.2f}" for v in e) + "]", H.euler_to_mat(e)


def arm_workspace_map(scene, side, orientations=None, points=None, x_range=(-0.4, 0.4), y_range=(0.1, 0.9), z_m=0.12, step_m=0.1,
                      pos_tol_m=0.015, tilt_tol_deg=15.0, out_png=None, out_json=None, cache=True):
    from tool_library.skills.optimize_pose_for_reach import _reach_solver
    model = mujoco.MjModel.from_xml_path(str(scene))
    solve0 = _reach_solver(model, side)
    # the cache: residuals of solved cells by (signature, side, orientation) and cell; tolerances apply on reading
    sig = robot_signature(model, side) if cache else None
    write_path = (Path(cache) if isinstance(cache, (str, Path)) else user_cache_path()) if cache else None
    known = {}
    if cache:
        for src in (SHIPPED_CACHE, write_path):
            for k, cells in _read_cache(src).items():
                known.setdefault(k, {}).update(cells)
    fresh, stats = {}, {"hits": 0, "solved": 0}

    def solve(p, R, q_prev=None):
        """(pos_err_m, rot_err_rad, arm q or None): from the cache when this cell was solved before."""
        if not cache:
            stats["solved"] += 1
            return solve0(p, R, q_prev)
        ek, ck = f"{sig}|{side}|{_orient_key(R)}", _cell_key(p)
        hit = known.get(ek, {}).get(ck)
        if hit is not None:
            stats["hits"] += 1
            return float(hit[0]), float(hit[1]), None
        r = solve0(p, R, q_prev)
        stats["solved"] += 1
        rec = [float(r[0]), float(r[1])]               # full precision: a cached map reads exactly like the solved one
        known.setdefault(ek, {})[ck] = rec; fresh.setdefault(ek, {})[ck] = rec
        return r
    orientations = orientations or ["palm_down", "fingers_in", "fingers_down", "palm_in"]
    zs = [float(z_m)] if np.isscalar(z_m) else [float(z) for z in z_m]
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step_m); ys = np.arange(y_range[0], y_range[1] + 1e-9, step_m)
    per, panels = {}, []
    for o in orientations:
        name, R = _orientation(side, o); e = H.mat_to_euler(R)
        cells, grid = [], {}
        for z in zs:
            G = np.full((len(ys), len(xs)), np.nan)
            for ix, x in enumerate(xs):
                q = None
                for iy, y in enumerate(ys):
                    r = solve(np.array([x, y, z]), R, q); q = r[2]
                    ok = r[0] <= pos_tol_m and r[1] <= np.radians(tilt_tol_deg)
                    cells.append([round(float(x), 3), round(float(y), 3), z, round(1000 * r[0], 1), round(float(np.degrees(r[1])), 1), bool(ok)])
                    G[iy, ix] = 1000 * r[0] if ok else np.nan
            grid[z] = G; panels.append((name, z, G))
        okc = [c for c in cells if c[5]]
        far = {}
        for c in okc:
            far[c[0]] = max(far.get(c[0], -1), c[1])
        pts = {}
        for lab, p in (points or {}).items():
            r = solve(np.asarray(p, float), R)
            pts[lab] = {"ok": bool(r[0] <= pos_tol_m and r[1] <= np.radians(tilt_tol_deg)), "pos_err_mm": round(1000 * r[0], 1), "rot_err_deg": round(float(np.degrees(r[1])), 1)}
        per[name] = {"euler": [round(float(v), 4) for v in e], "reachable_fraction": round(len(okc) / max(len(cells), 1), 3),
                     "far_edge_y": {f"{x:.2f}": round(y, 3) for x, y in sorted(far.items())}, "cells": cells, "points": pts}
    best = {}
    for lab in (points or {}):
        cand = [(per[n]["points"][lab]["pos_err_mm"] + 0.5 * per[n]["points"][lab]["rot_err_deg"], n) for n in per]
        best[lab] = {"orientation": min(cand)[1], "ok": any(per[n]["points"][lab]["ok"] for n in per)}
    ok = all(b["ok"] for b in best.values()) if best else any(per[n]["reachable_fraction"] > 0 for n in per)
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(panels); fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
        for ax, (name, z, G) in zip(axs[0], panels):
            ax.imshow(G, origin="lower", extent=[xs[0] - step_m / 2, xs[-1] + step_m / 2, ys[0] - step_m / 2, ys[-1] + step_m / 2], cmap="Greens_r", vmin=0, vmax=1000 * pos_tol_m)
            for lab, p in (points or {}).items():
                okp = per[name]["points"][lab]["ok"]
                ax.plot(p[0], p[1], "o", color="tab:blue" if okp else "tab:red", ms=6); ax.annotate(lab, (p[0], p[1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_title(f"{side} {name} z={z:.2f}", fontsize=9); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
        fig.suptitle("reachable cells (green = residual mm); red points are out of reach", fontsize=9); fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    unreach = [lab for lab, b in best.items() if not b["ok"]]
    verdict = ("REACH OK: " +", ".join(f"{n} to y<={max(per[n]['far_edge_y'].values()) if per[n]['far_edge_y'] else 0:.2f} m" for n in per)
               if ok else "OUT OF REACH: " + ", ".join(f"{lab} (best {best[lab]['orientation']}: {per[best[lab]['orientation']]['points'][lab]['pos_err_mm']} mm)" for lab in unreach))
    rep = {"ok": bool(ok), "verdict": verdict, "side": side, "orientations": per, "best_orientation": best, "pos_tol_m": pos_tol_m, "tilt_tol_deg": tilt_tol_deg}
    if cache and fresh:
        stats["written"] = _write_cache(write_path, fresh)
    rep["cache"] = {"hits": stats["hits"], "solved": stats["solved"], "signature": sig,
                    "file": str(write_path) if write_path else None, **({"written": stats["written"]} if "written" in stats else {})}
    if out_png:
        rep["out_png"] = str(out_png)
    if out_json:
        Path(out_json).write_text(json.dumps(rep, indent=1, default=float)); rep["out_json"] = str(out_json)
    return rep
