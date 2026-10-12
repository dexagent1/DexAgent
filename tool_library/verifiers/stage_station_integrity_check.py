"""Stage and station integrity check: the robot stations, the table and the home pose are the setup's, or moved
exactly as a recorded place_arm_stations shift says."""
from __future__ import annotations

TOOL = {
    "name": "stage_station_integrity_check",
    "title": "Stage and station integrity check",
    "type": "verifier",
    "category": "Scene Assertions",
    "summary": "The robot station roots, the table (body, geom pose and size) and the home keyframe of a scene equal the setup's, except for a station shift recorded by place_arm_stations, which must explain every moved station and the moved hand home exactly.",
    "description": (
        "Moving a station closer to an object, raising the table under a hand or lowering it under a forearm makes a "
        "trajectory feasible in simulation only. The reference is the setup (default: the repository's setup/): "
        "setup.yaml's stations.<side>.base_body / base_pos / base_quat_wxyz and table body / geom, the setup scene's "
        "table geom and its home keyframe. Each station root must sit at the reference position with the reference "
        "orientation, or be displaced by exactly the horizontal shift [dx, dy] of the place_arm_stations report "
        "(optional input; the skill moves both stations by default or only some), within tol_m. The table body and "
        "its collision geom must keep their pose and size. In keyframe 0 the arm joints, finger joints and wrist "
        "rotations must equal the setup's home, and each hand's wrist position must equal the setup's home plus its "
        "own station's displacement (place_arm_stations moves the hand home with its flange). A scene without a "
        "keyframe is reported (initial_keyframe_state_check says more)."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "Scene MJCF (Stage 2 scene.xml or the executed scene_stations.xml)."},
        {"name": "setup", "type": "str | None", "description": "Setup folder (or its setup.yaml / scene.xml). Default: the repository's setup/."},
        {"name": "stations_report", "type": "str | dict | None", "description": "place_arm_stations.json (or its dict) recording the shift [dx, dy]. Default None."},
        {"name": "unrecorded_shift_ok", "type": "bool", "description": "Without a report, accept a pure in-plane translation of a station within shift_bounds_m (the move place_arm_stations makes) as a warning. Default True."},
        {"name": "shift_bounds_m", "type": "[[dx_min, dx_max], [dy_min, dy_max]]", "description": "Allowed unrecorded translation. Default ((-0.5, 0.5), (-0.2, 0.5))."},
        {"name": "tol_m", "type": "float", "description": "Allowed position difference, metres. Default 1e-4."},
        {"name": "tol_rad", "type": "float", "description": "Allowed joint difference in the home keyframe, rad (or m for slides). Default 1e-4."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; stations {side: {reference, scene, shift, explained}}; recorded_shift; table {body, geom, changed}; home {joints compared, changed [{joint, reference, scene}]}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "pyyaml"],
    "simulators": ["MuJoCo"],
    "tags": ["assertion", "anti-tampering", "station", "table", "home", "keyframe", "stage2", "stage3"],
    "status": "tested",
    "entrypoint": "stage_station_integrity_check",
}

import mujoco
import numpy as np

from tool_library.common import scene_assert as SA


def stage_station_integrity_check(scene, setup=None, stations_report=None, tol_m=1e-4, tol_rad=1e-4, unrecorded_shift_ok=True,
                                  shift_bounds_m=((-0.5, 0.5), (-0.2, 0.5))):
    model, err = SA.try_load_model(scene)
    if model is None:
        return SA.report([f"the scene does not compile: {err}"], compiled=False)
    from tool_library.verifiers._reference import load_reference
    st = load_reference(model, setup)
    ref, cfg = st["model"], st["cfg"]
    rep, _ = SA.load_json(stations_report) if stations_report is not None else (None, None)
    rec = None
    if rep is not None:
        if "shift" not in rep:
            raise ValueError("stations_report: not a place_arm_stations report (no 'shift')")
        rec = np.array([float(rep["shift"][0]), float(rep["shift"][1]), 0.0])
    # per-station totals (a report of chained calls, or of one station moved alone) take precedence over the common shift;
    # their third entry is the lift of a station moved over the tabletop onto it (place_arm_stations' table_lift_m)
    rec_side = {s: np.array([float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 0.0])
                for s, v in ((rep or {}).get("station_shifts") or {}).items()}
    issues, warnings = [], []
    bmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_BODY)
    stations_cfg = (cfg.get("stations") or {}) if cfg else {}
    roots = SA.station_root_names(cfg) or {s: f"robot_{s}_arm_root" for s in ("left", "right")}
    stations, shifts = {}, {}
    # a report recording a shift while every station of the scene stands exactly where the setup puts it describes another
    # scene (a probe call that wrote scene_stations.xml beside it, which the episode never ran in): it does not apply here
    if rec is not None or rec_side:
        unmoved = True
        for side, rname in roots.items():
            rb = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, rname)
            sb = bmap.get(rb, -1) if rb >= 0 else -1
            if rb >= 0 and sb >= 0:
                rpos = np.array(stations_cfg.get(side, {}).get("base_pos", ref.body_pos[rb]), float)
                unmoved = unmoved and bool(np.linalg.norm(np.array(model.body_pos[sb], float) - rpos) <= tol_m)
        if unmoved:
            warnings.append(f"the place_arm_stations report records a shift {SA.r6((rec if rec is not None else next(iter(rec_side.values())))[:2], 3)} m "
                            "but every station of this scene stands where the setup puts it: the report describes another scene and is ignored")
            rec, rec_side = None, {}
    for side, rname in roots.items():
        rb = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, rname)
        sb = bmap.get(rb, -1) if rb >= 0 else -1
        if rb < 0:
            continue
        if sb < 0:
            issues.append(f"station root {rname} ({side}) is missing from the scene; rebuild the scene from the setup")
            continue
        rpos = np.array(stations_cfg.get(side, {}).get("base_pos", ref.body_pos[rb]), float)
        rquat = np.array(stations_cfg.get(side, {}).get("base_quat_wxyz", ref.body_quat[rb]), float)
        spos = np.array(model.body_pos[sb], float)
        d = spos - rpos
        ang = SA.quat_angle_deg(model.body_quat[sb], rquat)
        moved = bool(np.linalg.norm(d) > tol_m)
        rs = rec_side.get(side, rec)
        explained = (not moved and (rs is None or np.linalg.norm(rs) <= tol_m or side not in rec_side)) or \
                    (rs is not None and np.linalg.norm(d - rs) <= tol_m)
        if moved and rec is None and unrecorded_shift_ok and abs(d[2]) <= tol_m and ang <= 0.01 \
                and shift_bounds_m[0][0] <= d[0] <= shift_bounds_m[0][1] and shift_bounds_m[1][0] <= d[1] <= shift_bounds_m[1][1]:
            # a pure translation of the station on the table, the move place_arm_stations makes: allowed without its report
            explained = True
            warnings.append(f"station {side} is translated by {SA.r6(d[:2], 3)} m on the table with no place_arm_stations report; "
                            f"accepted as a station shift (keep the report next to the scene)")
        stations[side] = {"reference": SA.r6(rpos), "scene": SA.r6(spos), "shift": SA.r6(d), "explained": bool(explained), "rotated_deg": round(ang, 4)}
        shifts[side] = d
        if ang > 0.01:
            issues.append(f"station {side} ({rname}) is rotated {ang:.3f} deg from the setup's orientation; restore base_quat_wxyz")
        if not explained:
            if rec is None:
                issues.append(f"station {side} ({rname}) is moved by {SA.r6(d, 4)} m from the setup with no place_arm_stations record; "
                              f"restore it, or move the stations with place_arm_stations and pass its report")
            else:
                issues.append(f"station {side} ({rname}) is moved by {SA.r6(d, 4)} m but the recorded shift is {SA.r6(rs, 4)} m; "
                              f"use the scene place_arm_stations wrote, unedited")
    # table
    table = {"changed": []}
    tname = ((cfg.get("table") or {}).get("body") if cfg else None) or "table"
    gname = ((cfg.get("table") or {}).get("geom") if cfg else None) or "g_table"
    rtb, stb = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tname) for m in (ref, model))
    rtg, stg = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname) for m in (ref, model))
    table["body"], table["geom"] = tname, gname
    if rtb >= 0:
        if stb < 0:
            issues.append(f"the setup's table body '{tname}' is missing from the scene")
        else:
            rd, sd = mujoco.MjData(ref), mujoco.MjData(model)
            mujoco.mj_kinematics(ref, rd); mujoco.mj_kinematics(model, sd)
            if SA.differs(sd.xpos[stb], rd.xpos[rtb], tol_m, 0.0) or SA.quat_angle_deg(sd.xquat[stb], rd.xquat[rtb]) > 0.01:
                table["changed"].append({"what": "body pose", "reference": SA.r6(rd.xpos[rtb]), "scene": SA.r6(sd.xpos[stb])})
            if rtg >= 0:
                if stg < 0:
                    table["changed"].append({"what": "geom", "reference": gname, "scene": None})
                else:
                    if SA.differs(sd.geom_xpos[stg], rd.geom_xpos[rtg], tol_m, 0.0):
                        table["changed"].append({"what": "geom position", "reference": SA.r6(rd.geom_xpos[rtg]), "scene": SA.r6(sd.geom_xpos[stg])})
                    if SA.differs(model.geom_size[stg], ref.geom_size[rtg], tol_m, 0.0) or int(model.geom_type[stg]) != int(ref.geom_type[rtg]):
                        table["changed"].append({"what": "geom size", "reference": SA.r6(ref.geom_size[rtg]), "scene": SA.r6(model.geom_size[stg])})
                    top_r = float(rd.geom_xpos[rtg][2] + ref.geom_size[rtg][2]) if int(ref.geom_type[rtg]) == int(mujoco.mjtGeom.mjGEOM_BOX) else None
                    top_s = float(sd.geom_xpos[stg][2] + model.geom_size[stg][2]) if int(model.geom_type[stg]) == int(mujoco.mjtGeom.mjGEOM_BOX) else None
                    table["top_z"] = {"reference": top_r, "scene": top_s}
    for c in table["changed"]:
        issues.append(f"table {c['what']} is {c['scene']} in the scene but {c['reference']} in the setup; restore the setup's table "
                      f"(a moved or resized table changes every contact with it)")
    # home keyframe
    home = {"changed": []}
    q_ref = SA.ref_home_qpos(ref, cfg)
    if model.nkey == 0:
        issues.append("the scene has no keyframe: the robot's home pose is undefined; start from the setup's 'home' keyframe")
    elif q_ref is not None:
        jmap = SA.robot_map(model, ref, mujoco.mjtObj.mjOBJ_JOINT)
        q_s = model.key_qpos[0]
        n = 0
        for rj, sj in jmap.items():
            if sj < 0 or int(ref.jnt_type[rj]) in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
                continue
            jn = ref.joint(rj).name
            want = float(q_ref[int(ref.jnt_qposadr[rj])])
            for side in ("left", "right"):
                if jn.endswith(f"{side}_{side}_pos_x") and side in shifts:
                    want += float(shifts[side][0])
                elif jn.endswith(f"{side}_{side}_pos_y") and side in shifts:
                    want += float(shifts[side][1])
                elif jn.endswith(f"{side}_{side}_pos_z") and side in shifts:
                    want += float(shifts[side][2])
            got = float(q_s[int(model.jnt_qposadr[sj])])
            n += 1
            if abs(got - want) > tol_rad:
                home["changed"].append({"joint": jn, "reference": round(want, 6), "scene": round(got, 6)})
        home["joints_compared"] = n
        if home["changed"]:
            c = home["changed"]
            issues.append(f"keyframe 0 puts {len(c)} robot joint(s) off the setup's home pose (shifted with its station): "
                          + ", ".join(f"{x['joint']} {x['scene']} vs {x['reference']}" for x in c[:6])
                          + ("..." if len(c) > 6 else "") + "; start from the setup's home keyframe")
    return SA.report(issues, warnings=warnings, stations=stations, recorded_shift=None if rec is None else SA.r6(rec[:2]), table=table,
                     home=home, setup=str(st["dir"]))
