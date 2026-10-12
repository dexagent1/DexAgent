"""Table workspace bounds audit: hands and objects stay above the tabletop and over the table.

    from tool_library.verifiers.table_workspace_bounds_audit import table_workspace_bounds_audit
    table_workspace_bounds_audit("scene.xml", "trajectory.npz")

Contact-based checks (robot_self_collision_check's hand-table category, episode_physics_check's
hand-table pairs) only see the table where its geometry is, on the frames they sample. They do
not see an object that slid off the edge and fell, a hand that dips below the tabletop plane
beyond the table's edge, an object lifted half a metre above the table, or an object left
outside the area the camera covers.
"""
from __future__ import annotations

TOOL = {
    "name": "table_workspace_bounds_audit",
    "title": "Table workspace bounds audit",
    "type": "verifier",
    "category": "Episode Audits",
    "summary": "On every frame, hands and objects stay above the tabletop plane (vertex-exact), over the table's "
               "footprint, under a height ceiling, and optionally inside a given work area.",
    "description": (
        "Plane-and-footprint bounds, where robot_self_collision_check and episode_physics_check judge contacts with the "
        "table geometry on sampled frames. The table is the body named 'table' (top = the highest point of its geoms, "
        "footprint = their x-y extent) or a ground plane. On every frame: (1) the lowest point of each hand's visible "
        "geoms (mesh vertices, primitive surfaces) and of each object's colliding geoms may not lie more than "
        "below_tol_m (default 5 mm) under the tabletop while over the footprint; (2) an object's centre of mass may "
        "not leave the footprint (pushed or dropped off the table) nor drop more than fall_m (default 2 cm) under the "
        "tabletop anywhere (fell through or off); (3) no hand or object centre more than max_height_m (default 0.6 m) "
        "above the tabletop; (4) with work_area {x: [lo, hi], y: [lo, hi]} (setup.yaml's table.work_area, the cloth the "
        "camera covers), objects' centres stay inside it. Hands are the bench hands (or the robot bodies grouped by "
        "side); arms are left to robot_self_collision_check. Reports per body the worst frame and margin."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF the episode was executed in."},
        {"name": "trajectory", "type": "str", "description": "trajectory.npz (qpos [T, nq], fps, labels)."},
        {"name": "frames", "type": "[first, last] | None", "description": "Inclusive frame window. Default all."},
        {"name": "below_tol_m", "type": "float", "description": "Allowed depth of a hand or object point under the tabletop over the table. Default 0.005."},
        {"name": "fall_m", "type": "float", "description": "Object centre this far under the tabletop anywhere = fell. Default 0.02."},
        {"name": "max_height_m", "type": "float", "description": "Ceiling above the tabletop for hand and object centres. Default 0.6."},
        {"name": "work_area", "type": "dict | None", "description": "{x: [lo, hi], y: [lo, hi]} objects must stay inside (metres). Default None."},
        {"name": "table_top_z", "type": "float | None", "description": "Override the tabletop height."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; worst {body, kind, frame, label, value}; table {top_z, xy}; bodies {name: lowest margin mm, frame, max height}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "audit", "table", "workspace", "bounds", "camera"],
    "status": "tested",
    "entrypoint": "table_workspace_bounds_audit",
}

import mujoco
import numpy as np

from tool_library.common import episode_audit as EA


def table_workspace_bounds_audit(scene, trajectory, frames=None, below_tol_m=0.005, fall_m=0.02, max_height_m=0.6,
                                 work_area=None, table_top_z=None):
    EA.check_positive(below_tol_m=below_tol_m, fall_m=fall_m, max_height_m=max_height_m)
    if work_area is not None:
        try:
            wx, wy = [float(v) for v in work_area["x"]], [float(v) for v in work_area["y"]]
            assert wx[0] < wx[1] and wy[0] < wy[1]
        except Exception:
            raise ValueError("work_area must be {'x': [lo, hi], 'y': [lo, hi]} in metres")
    ep = EA.Episode(scene, trajectory, frames)
    m = ep.model
    tb = ep.table
    if tb is None and table_top_z is None:
        raise ValueError("scene: no body named 'table' and no ground plane; pass table_top_z")
    top = float(table_top_z) if table_top_z is not None else tb["top_z"]
    xy = tb["xy"] if tb is not None else [-np.inf, np.inf, -np.inf, np.inf]
    groups = {}
    for s, h in ep.hands.items():
        gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in h["bodies"] and int(m.geom_group[g]) <= 2
              and int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)]
        if gs:
            groups[f"{s} hand"] = ("hand", gs, h["root"])
    for n, o in ep.objects.items():
        gs = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in o["bodies"] and EA.geom_collides(m, g)]
        gs = gs or [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in o["bodies"]]
        if gs:
            groups[n] = ("object", gs, o["body"])
    local = {}
    for kind, gs, _ in groups.values():
        for g in gs:
            if g not in local:
                P = EA.geom_local_points(m, g)
                if len(P) > 300:
                    P = P[np.linspace(0, len(P) - 1, 300).astype(int)]
                local[g] = P
    d = mujoco.MjData(m)
    rows = {k: {"lowest_mm": np.inf, "lowest_frame": None, "max_height_m": -np.inf} for k in groups}
    events = []
    com = {n: ep.com_track(o["bodies"]) for n, o in ep.objects.items()}
    for t in ep.window:
        d.qpos[:] = ep.Q[t]
        mujoco.mj_kinematics(m, d)
        for key, (kind, gs, root) in groups.items():
            low, lowxy = np.inf, None
            for g in gs:
                zc = float(d.geom_xpos[g][2]); rb = float(m.geom_rbound[g])
                if rb > 0 and zc - rb > top + 0.01:          # its bounding sphere clears the tabletop by a centimetre
                    if zc - rb - top < low:
                        low = zc - rb - top; lowxy = d.geom_xpos[g][:2].copy()
                    continue
                W = EA.geom_world_points(m, d, g, local[g])
                i = int(np.argmin(W[:, 2]))
                if W[i, 2] - top < low:
                    low = float(W[i, 2] - top); lowxy = W[i, :2].copy()
            r = rows[key]
            if low * 1000 < r["lowest_mm"]:
                r["lowest_mm"] = round(low * 1000, 2); r["lowest_frame"] = t
            over = lowxy is not None and xy[0] <= lowxy[0] <= xy[1] and xy[2] <= lowxy[1] <= xy[3]
            if low < -below_tol_m and over:
                events.append({"body": key, "kind": "under_tabletop", "frame": t, "label": ep.labels[t], "value": round(-low * 1000, 1), "unit": "mm"})
            c = com[key][t] if kind == "object" else d.xpos[root]
            h = float(c[2] - top)
            r["max_height_m"] = max(r["max_height_m"], round(h, 3))
            if h > max_height_m:
                events.append({"body": key, "kind": "too_high", "frame": t, "label": ep.labels[t], "value": round(h, 3), "unit": "m"})
            if kind == "object":
                if not (xy[0] <= c[0] <= xy[1] and xy[2] <= c[1] <= xy[3]):
                    events.append({"body": key, "kind": "off_table", "frame": t, "label": ep.labels[t],
                                   "value": [round(float(c[0]), 3), round(float(c[1]), 3)], "unit": "m"})
                if c[2] < top - fall_m:
                    events.append({"body": key, "kind": "fell", "frame": t, "label": ep.labels[t], "value": round(float(top - c[2]), 3), "unit": "m"})
                if work_area is not None and not (wx[0] <= c[0] <= wx[1] and wy[0] <= c[1] <= wy[1]):
                    events.append({"body": key, "kind": "outside_work_area", "frame": t, "label": ep.labels[t],
                                   "value": [round(float(c[0]), 3), round(float(c[1]), 3)], "unit": "m"})
    advice = {"under_tabletop": "raise that phase's targets so the part clears the tabletop (the table is solid for the robot too)",
              "too_high": "keep the motion within reach of the bench (lower the via or lift height)",
              "off_table": "the object left the table: find the push or drop that sent it there and re-execute",
              "fell": "the object fell below the tabletop: it left the table or passed through it",
              "outside_work_area": "place and move the object inside the camera's work area"}
    issues = []
    for key, kind in sorted({(e["body"], e["kind"]) for e in events}):
        evs = [e for e in events if e["body"] == key and e["kind"] == kind]
        w = max(evs, key=lambda e: e["value"] if not isinstance(e["value"], list) else 0)
        issues.append(f"{key}: {kind.replace('_', ' ')} on {len(evs)} frames, first at frame {evs[0]['frame']} ({evs[0]['label']}), "
                      f"worst {w['value']} {w['unit']} at frame {w['frame']}; {advice[kind]}")
    worst = None
    if events:
        worst = max(events, key=lambda e: (e["kind"] in ("fell", "off_table", "under_tabletop"),
                                           e["value"] if not isinstance(e["value"], list) else 1.0))
    return EA.finish(issues, worst, table={"top_z": round(top, 4), "xy": [EA.r3(v) for v in xy]}, bodies=rows,
                     n_events=len(events), frames=[ep.a, ep.b])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(table_workspace_bounds_audit(sys.argv[1], sys.argv[2]), indent=1))
