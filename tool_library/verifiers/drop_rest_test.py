"""Drop a free object onto a floor from several orientations and check it comes to rest where it lands.

    from tool_library.verifiers.drop_rest_test import drop_rest_test
    rep = drop_rest_test("assets/toy/toy.xml")
    rep["ok"], rep["issues"], rep["drops"]

end_state_rest_audit fails a Stage 3 episode whose objects are still moving or move again when the scene is
simulated on from the last frame. An object that cannot come to rest on a flat support by itself (a body of
ellipsoids rocking on its curved underside, a cylinder rolling) fails that audit whatever the robot does, so the
property belongs to the reconstruction: this test drops the asset alone and measures it.
"""
from __future__ import annotations

TOOL = {
    "name": "drop_rest_test",
    "title": "Drop-and-rest test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Drops a free object onto a floor from several tilted orientations and checks that each time it comes to rest "
               "within the settle time and then stays put (the rest criteria of end_state_rest_audit).",
    "description": (
        "Loads the asset MJCF, finds its free body (or `body`), adds a floor plane at z = 0 when the asset has none, and "
        "for each drop orientation (roll, pitch, yaw in degrees about the world axes) places the body so its lowest "
        "collision point is `drop_height` above the floor, at rest. It simulates `settle_s`; over the last `rest_window_s` "
        "of that the body must move slower than `rest_speed_m_s` and turn slower than `rest_deg_s`; then it simulates "
        "`check_s` more and the body's peak motion from where it was must stay under `settle_m` and `settle_deg`, the same "
        "numbers end_state_rest_audit applies to a Stage 3 episode's last frame. A body that rocks on a curved underside, "
        "rolls away or keeps spinning fails with the drop and the numbers. Every joint of the asset other than the free "
        "joint keeps its springs and limits. Lowest points come from each collision geom's world-axis extent (exact for "
        "spheres, ellipsoids, boxes, capsules and cylinders; the bounding sphere for meshes)."
    ),
    "inputs": [
        {"name": "asset_xml", "type": "str", "description": "MJCF of the object asset (one free body, optional floor)."},
        {"name": "body", "type": "str | None", "description": "The free body to drop; default the first body with a free joint."},
        {"name": "drops", "type": "list[[roll, pitch, yaw]] | None", "description": "Drop orientations in degrees. Default five: level and four tilts up to 40 deg."},
        {"name": "drop_height", "type": "float, metres", "description": "Gap between the lowest collision point and the floor at release. Default 0.03."},
        {"name": "settle_s", "type": "float, seconds", "description": "Time to come to rest after the release. Default 1.5."},
        {"name": "rest_window_s", "type": "float, seconds", "description": "Final stretch of settle_s over which speed is measured. Default 0.3."},
        {"name": "rest_speed_m_s", "type": "float", "description": "Largest speed at rest. Default 0.02."},
        {"name": "rest_deg_s", "type": "float", "description": "Largest turn rate at rest. Default 15."},
        {"name": "check_s", "type": "float, seconds", "description": "Time simulated on after settle_s. Default 1.0."},
        {"name": "settle_m", "type": "float, metres", "description": "Largest peak motion during check_s. Default 0.003."},
        {"name": "settle_deg", "type": "float, degrees", "description": "Largest peak rotation during check_s. Default 2."},
        {"name": "out_json", "type": "str | None", "description": "Write the report here."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; body; drops [{orientation_deg, final_speed_m_s, final_turn_deg_s, check_peak_mm, check_peak_deg, rest_z_m, ok}]; worst."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage2", "rest", "settle", "rocking", "rolling", "drop", "free body", "mechanism test"],
    "used_in": ["Place Toy in Drawer"],
    "status": "tested",
    "entrypoint": "drop_rest_test",
    "cli": "python -m tool_library.verifiers.drop_rest_test asset.xml",
}

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_DROPS = [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [35.0, 15.0, 30.0], [-25.0, 10.0, -60.0]]


def _quat_rpy(r, p, y):
    r, p, y = (math.radians(float(v)) for v in (r, p, y))
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _angle(q0, q1):
    return float(np.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))))


def _lowest(m, d, gids):
    import mujoco
    zmin = np.inf
    for g in gids:
        R = d.geom_xmat[g].reshape(3, 3); s = m.geom_size[g]; t = int(m.geom_type[g]); c = float(d.geom_xpos[g][2])
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            e = s[0]
        elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            e = math.sqrt(sum((R[2, j] * s[j]) ** 2 for j in range(3)))
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            e = sum(abs(R[2, j]) * s[j] for j in range(3))
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            e = abs(R[2, 2]) * s[1] + s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            e = abs(R[2, 2]) * s[1] + s[0] * math.sqrt(max(0.0, 1 - R[2, 2] ** 2))
        else:
            e = float(m.geom_rbound[g])
        zmin = min(zmin, c - e)
    return zmin


def drop_rest_test(asset_xml, body=None, drops=None, drop_height=0.03, settle_s=1.5, rest_window_s=0.3, rest_speed_m_s=0.02,
                   rest_deg_s=15.0, check_s=1.0, settle_m=0.003, settle_deg=2.0, out_json=None):
    import mujoco
    for k, v in dict(drop_height=drop_height, settle_s=settle_s, rest_window_s=rest_window_s, rest_speed_m_s=rest_speed_m_s,
                     rest_deg_s=rest_deg_s, check_s=check_s, settle_m=settle_m, settle_deg=settle_deg).items():
        if not (float(v) > 0):
            raise ValueError(f"{k} must be positive (got {v})")
    if rest_window_s >= settle_s:
        raise ValueError("rest_window_s must be shorter than settle_s")
    p = Path(str(asset_xml))
    if not p.exists():
        raise ValueError(f"asset_xml {asset_xml} does not exist")
    spec = mujoco.MjSpec.from_file(str(p))
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
        spec.worldbody.add_geom(name="drop_rest_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1])
    m = spec.compile()
    free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    if body is None:
        if not free:
            raise ValueError("the asset has no free body: drop_rest_test drops a free object (give a free asset)")
        jid = free[0]; bid = int(m.jnt_bodyid[jid])
    else:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, str(body))
        if bid < 0:
            raise ValueError(f"body {body!r} is not in the asset")
        js = [j for j in free if int(m.jnt_bodyid[j]) == bid]
        if not js:
            raise ValueError(f"body {body!r} has no free joint")
        jid = js[0]
    qa = int(m.jnt_qposadr[jid])
    sub = set()
    for b in range(m.nbody):
        a = b
        while a > 0 and a != bid:
            a = int(m.body_parentid[a])
        if a == bid:
            sub.add(b)
    gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if not gids:
        raise ValueError("the free body has no collision geometry")
    dt = float(m.opt.timestep)
    rows, issues, worst = [], [], None
    for rpy in (drops or DEFAULT_DROPS):
        if len(rpy) != 3:
            raise ValueError("each drop is [roll, pitch, yaw] in degrees")
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[qa:qa + 3] = [0.0, 0.0, 1.0]; d.qpos[qa + 3:qa + 7] = _quat_rpy(*rpy)
        mujoco.mj_forward(m, d)
        d.qpos[qa + 2] += float(drop_height) - _lowest(m, d, gids)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        n_settle, n_win, n_check = int(round(settle_s / dt)), int(round(rest_window_s / dt)), int(round(check_s / dt))
        track = []
        bad = False
        for k in range(n_settle + n_check):
            mujoco.mj_step(m, d)
            if not np.isfinite(d.qpos).all() or int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number):
                bad = True
                break
            track.append((d.xpos[bid].copy(), d.xquat[bid].copy()))
        if bad:
            row = {"orientation_deg": list(map(float, rpy)), "ok": False, "unstable": True}
            issues.append(f"drop {list(rpy)}: the simulation went unstable; check the asset's contact parameters and inertia")
            rows.append(row); worst = worst or row
            continue
        a, e = n_settle - n_win, n_settle - 1
        v = float(np.linalg.norm(track[e][0] - track[a][0])) / (n_win * dt)
        w = _angle(track[a][1], track[e][1]) / (n_win * dt)
        p0, q0 = track[e]
        pk_m = max(float(np.linalg.norm(t[0] - p0)) for t in track[e:])
        pk_d = max(_angle(q0, t[1]) for t in track[e:])
        ok = v <= rest_speed_m_s and w <= rest_deg_s and pk_m <= settle_m and pk_d <= settle_deg
        row = {"orientation_deg": list(map(float, rpy)), "final_speed_m_s": round(v, 4), "final_turn_deg_s": round(w, 2),
               "check_peak_mm": round(1000 * pk_m, 2), "check_peak_deg": round(pk_d, 2), "rest_z_m": round(float(p0[2]), 4), "ok": bool(ok)}
        rows.append(row)
        if not ok:
            issues.append(f"drop {list(rpy)}: not at rest {settle_s:g} s after landing ({v:.3f} m/s, {w:.1f} deg/s) or moves on "
                          f"({1000 * pk_m:.1f} mm / {pk_d:.1f} deg in {check_s:g} s): a curved underside rocks or rolls without rolling "
                          "resistance (condim 6 with a rolling friction for a soft body), or a flat base / damping is missing")
            if worst is None or (row.get("check_peak_deg", 0) > worst.get("check_peak_deg", 0)):
                worst = row
    rep = {"ok": not issues, "issues": issues, "body": m.body(bid).name, "drops": rows, "worst": worst,
           "thresholds": {"rest_speed_m_s": rest_speed_m_s, "rest_deg_s": rest_deg_s, "settle_m": settle_m, "settle_deg": settle_deg,
                          "settle_s": settle_s, "check_s": check_s}}
    if out_json:
        Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    import sys
    print(json.dumps(drop_rest_test(sys.argv[1]), indent=1))
