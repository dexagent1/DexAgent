TOOL = {
    "name": "rest_inside_container_test",
    "title": "Object set down inside a container comes to rest on its floor",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Sets a free object in an open container, settles under gravity, and checks rest/contact/orientation. Always measures full 3D containment; require_full_containment makes protrusion fail rather than permitting a partially inserted object.",
    "description": (
        "Gravity-only trial for 'X fits inside Y and rests there' properties (a cup set into a bowl, a part dropped into a bin, "
        "an upside-down cup placed in a bowl). The object's free joint is moved so its lowest collision point is drop_gap above the "
        "container's floor level estimate (container root z + floor_z) at the container's xy plus an optional xy_offset, with its "
        "original orientation unless placed_quat_wxyz is supplied; every other free body starts in its original state. After "
        "settle_s seconds the verdict needs: the object's xy within the container-frame collision extents shrunk by wall_margin, "
        "the object's bottom below the container's top, tilt from the "
        "placed orientation <= max_tilt_deg, linear speed <= max_speed, and, when avoid_rim_contact, no contact between the object and "
        "container geoms whose contact point is higher than rim_band_frac of the container height (the walls and rim). "
        "The container is identified by its root body (all of its descendant geoms count). Full containment uses exact "
        "collision-geometry support extents in the container's axes and the final task gate's fixed 2 mm tolerance. "
        "Default require_full_containment=False preserves tasks that permit protrusion; an ordinary inside/rest pass "
        "does not establish full containment. placed_quat_wxyz can propose a placement orientation without editing the scene."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Compiled-able MJCF path with both bodies."},
        {"name": "body", "type": "str", "description": "Root body of the free object to set down."},
        {"name": "container", "type": "str", "description": "Root body of the open container."},
        {"name": "floor_z", "type": "float", "description": "Container floor (inner bottom) height above its root origin. Default 0.004."},
        {"name": "drop_gap", "type": "float", "description": "Release height of the object's lowest point above the floor. Default 0.005 m."},
        {"name": "xy_offset", "type": "list", "description": "Release offset from the container centre [dx, dy] in metres. Default [0, 0]."},
        {"name": "settle_s", "type": "float", "description": "Simulated seconds. Default 2.0."},
        {"name": "max_tilt_deg", "type": "float", "description": "Allowed change of orientation. Default 10."},
        {"name": "max_speed", "type": "float", "description": "Final speed limit m/s. Default 0.01."},
        {"name": "avoid_rim_contact", "type": "bool", "description": "Fail on contacts with the walls/rim. Default True."},
        {"name": "rim_band_frac", "type": "float", "description": "Contacts above this fraction of the container height count as wall/rim. Default 0.35."},
        {"name": "wall_margin", "type": "float", "description": "Shrink of the container-frame extents for the partial inside test. Default 0.0."},
        {"name": "require_full_containment", "type": "bool", "description": "Require the entire object within all three container-frame extents, using fixed 0.002 m tolerance. Default False permits partial protrusion but still reports it."},
        {"name": "placed_quat_wxyz", "type": "list | None", "description": "Optional world quaternion for the trial placement; default the original object orientation. The normalized quaternion is recorded in trial_args for independent replay."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; body; container; trial_args; inside (partial criterion); fully_contained; full_containment {ok, object_bounds, container_bounds, protrusion_m, tolerance_m, frame}; rest/contact/orientation measurements."},
    ],
    "status": "tested",
    "entrypoint": "rest_inside_container_test",
    "tags": ["container", "bowl", "nest", "place inside", "rest", "gravity", "cup"],
}

import numpy as np


def _subtree_geoms(m, root_id):
    ids = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b != 0 and b != root_id:
            b = m.body_parentid[b]
        if b == root_id and (m.geom_contype[g] or m.geom_conaffinity[g]):
            ids.append(g)
    return ids


def _aabb(m, d, geoms):
    # Exact primitive support / mesh vertices, shared with episode geometry;
    # rotated geom-local AABBs overestimate cylinders and meshes.
    from tool_library.common import episode as E
    bounds = [E.bodies_extent(m, d, (), u=axis, geoms=geoms) for axis in np.eye(3)]
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def _full_containment(m, d, bid, cid):
    """The exact same extent/frame/tolerance semantics as final_full_containment."""
    from tool_library.common import task_gates as G
    ep = G.Episode(m, qpos=d.qpos.copy())
    objects, containers = ep.subtree(bid), ep.subtree(cid)
    origin, axes = G._container_frame(ep, 0, containers)
    alo, ahi = ep.extent(0, ep.geoms(objects), axes)
    blo, bhi = ep.extent(0, ep.geoms(containers), axes)
    offset = axes.T @ origin
    alo, ahi, blo, bhi = (v - offset for v in (alo, ahi, blo, bhi))
    protrusion = np.maximum(np.maximum(blo - alo, ahi - bhi), 0.)
    return {"ok": bool(np.all(protrusion <= G.TOUCH_TOL_M)), "object_bounds": [alo.tolist(), ahi.tolist()],
            "container_bounds": [blo.tolist(), bhi.tolist()], "protrusion_m": protrusion.tolist(),
            "tolerance_m": G.TOUCH_TOL_M, "frame": "container_up_axes", "origin_world": origin.tolist(),
            "axes_world": axes.tolist()}


def rest_inside_container_test(scene, body, container, floor_z=0.004, drop_gap=0.005, xy_offset=(0.0, 0.0), settle_s=2.0,
                               max_tilt_deg=10.0, max_speed=0.01, avoid_rim_contact=True, rim_band_frac=0.35, wall_margin=0.0,
                               require_full_containment=False, placed_quat_wxyz=None):
    import mujoco
    issues = []
    try:
        m = mujoco.MjModel.from_xml_path(str(scene))
    except Exception as e:
        return {"ok": False, "issues": [f"scene does not compile: {e}"]}
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    try:
        bid = m.body(body).id; cid = m.body(container).id
    except KeyError as e:
        return {"ok": False, "issues": [f"body not found: {e}; pass root body names"]}
    jadr = [j for j in range(m.njnt) if m.jnt_bodyid[j] == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not jadr:
        return {"ok": False, "issues": [f"{body} has no free joint; only a free object can be set down"]}
    qa = m.jnt_qposadr[jadr[0]]; va = m.jnt_dofadr[jadr[0]]
    if placed_quat_wxyz is not None:
        quat = np.asarray(placed_quat_wxyz, float)
        if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-9:
            return {"ok": False, "issues": ["placed_quat_wxyz must be a finite nonzero quaternion [w,x,y,z]"]}
        d.qpos[qa + 3:qa + 7] = quat / np.linalg.norm(quat)
    mujoco.mj_forward(m, d)
    og = _subtree_geoms(m, bid); cg = _subtree_geoms(m, cid)
    if not og or not cg:
        return {"ok": False, "issues": ["object or container has no collision geoms"]}
    clo, chi = _aabb(m, d, cg)
    olo, ohi = _aabb(m, d, og)
    croot = d.xpos[cid].copy()
    floor = croot[2] + floor_z
    target = np.array([croot[0] + xy_offset[0], croot[1] + xy_offset[1]])
    shift = np.array([target[0] - d.xpos[bid][0], target[1] - d.xpos[bid][1], floor + drop_gap - olo[2]])
    d.qpos[qa:qa + 3] += shift
    q_place = d.qpos[qa + 3:qa + 7].copy()
    d.qvel[va:va + 6] = 0
    mujoco.mj_forward(m, d)
    rim_contacts = []; floor_contacts = 0
    ch = chi[2] - clo[2]
    steps = int(settle_s / m.opt.timestep)
    for k in range(steps):
        mujoco.mj_step(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (g1 in og and g2 in cg) or (g2 in og and g1 in cg):
                hz = c.pos[2] - clo[2]
                if hz > rim_band_frac * ch:
                    if len(rim_contacts) < 20:
                        rim_contacts.append({"geoms": [m.geom(g1).name, m.geom(g2).name], "height_m": round(float(hz), 4)})
                else:
                    floor_contacts += 1
    mujoco.mj_forward(m, d)
    full = _full_containment(m, d, bid, cid)
    olo2, ohi2 = np.asarray(full["object_bounds"], float)
    clo2, chi2 = np.asarray(full["container_bounds"], float)
    ctr = 0.5 * (olo2 + ohi2)
    inside = bool(np.all(ctr[:2] >= clo2[:2] + wall_margin) and np.all(ctr[:2] <= chi2[:2] - wall_margin)
                  and olo2[2] < chi2[2])
    qerr = np.zeros(4); qinv = np.zeros(4)
    mujoco.mju_negQuat(qinv, q_place); mujoco.mju_mulQuat(qerr, qinv, d.qpos[qa + 3:qa + 7])
    tilt = float(np.degrees(2 * np.arccos(np.clip(abs(qerr[0]), 0, 1))))
    speed = float(np.linalg.norm(d.qvel[va:va + 3]))
    bottom = float(olo2[2] - floor_z)
    if not inside:
        issues.append("object did not end inside the container (too large for the mouth, or dropped off-centre): enlarge the container or shrink the object")
    if require_full_containment and not full["ok"]:
        issues.append("object is not fully contained: protrusion along the container axes is "
                      + str(np.round(1000 * np.asarray(full["protrusion_m"]), 2).tolist())
                      + " mm (fixed tolerance 2 mm)")
    if tilt > max_tilt_deg:
        issues.append(f"object turned {tilt:.1f} deg (> {max_tilt_deg}): it does not stand in the container in the placed orientation; check the floor size versus the object's footprint")
    if speed > max_speed:
        issues.append(f"object still moving at {speed:.3f} m/s after {settle_s} s")
    if avoid_rim_contact and rim_contacts:
        issues.append(f"object touches the container wall/rim ({len(rim_contacts)} contacts above {rim_band_frac:.2f} of its height): the floor is too small for the object's footprint")
    if bottom > 0.25 * ch:
        issues.append(f"object rests {bottom*1000:.1f} mm above the floor estimate: it is held by the walls, not the floor")
    trial_args = {"floor_z": float(floor_z), "drop_gap": float(drop_gap), "xy_offset": list(xy_offset), "settle_s": float(settle_s),
                  "max_tilt_deg": float(max_tilt_deg), "max_speed": float(max_speed), "avoid_rim_contact": bool(avoid_rim_contact),
                  "rim_band_frac": float(rim_band_frac), "wall_margin": float(wall_margin),
                  "require_full_containment": bool(require_full_containment), "placed_quat_wxyz": q_place.tolist()}
    return {"ok": not issues, "issues": issues, "body": str(body), "container": str(container), "trial_args": trial_args,
            "require_full_containment": bool(require_full_containment), "fully_contained": full["ok"], "full_containment": full,
            "final_offset_xy_m": np.round(ctr[:2], 4).tolist(),
            "bottom_above_floor_m": round(bottom, 4), "inside": inside, "tilt_deg": round(tilt, 2), "speed_m_s": round(speed, 4),
            "rim_contacts": rim_contacts, "floor_contacts": int(floor_contacts), "container_height_m": round(float(ch), 4)}
