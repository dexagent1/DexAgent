"""Move the two arm stations together so the task's objects are inside the arms' workspace."""
from __future__ import annotations

TOOL = {
    "name": "place_arm_stations",
    "title": "Place arm stations",
    "type": "skill",
    "category": "Trajectory & Execution",
    "summary": "Search a common shift of both arm stations (their roots and the hands' home pose) toward or away from the "
               "objects for the placement at which the arms reach the most task objects with the most margin, and write the "
               "bench scene with the stations moved.",
    "description": (
        "The bench's arms hold a palm-down hand only in a band about 0.4-0.6 m in front of their stations, and a clip "
        "recorded in another session can put the objects 20-30 cm outside it (fitting the arm after the hand hides this "
        "behind centimetres of gap at the wrist). When object_reach_check fails, this skill moves "
        "both stations by the same (dx, dy), keeping their separation and orientation and keeping each hand on its flange "
        "at home, and scores every candidate by exact arm IK of the flange (the executor's solver) at each object: palm "
        "down above it, palm down pitched 45 degrees, fingers in from the side, each at the object and 3 cm around it. The "
        "stations stay where they are when every object's centre is reached; otherwise the shift reaching the most probe points "
        "(the smallest among equals) wins. The scene is written as a new file next to the "
        "input with the shift recorded in `place_arm_stations.json`; use it for every Stage 3 tool and the Stage 4 task "
        "folder. Prefer no shift: use it when an object or a destination (`points`) is out of reach, and report the shift. "
        "Generic probes do not know the grasp: when the task's hand poses are known, pass them as `poses` (each with the "
        "arm that must hold it and the displacements it must also be held at, e.g. the carry to a slot); the shift is then "
        "ranked first by how many of those poses their own arm holds exactly at the grasp, the pre-grasp and every "
        "waypoint without hitting anything (the executor's screen), then by the probes. `sides` limits the probes to "
        "the arms that act."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Bench scene MJCF (Stage 2 scene.xml)."},
        {"name": "objects", "type": "list[str] | None", "description": "Bodies the arms must reach (default: every free object)."},
        {"name": "points", "type": "list[list[float]] | None", "description": "World points the hand must also reach: destinations of carries and places (the box a lid goes onto), measured like objects (hover above, pitched, from the side)."},
        {"name": "dy_range", "type": "list[float]", "description": "Shifts toward the objects (+y) tried, metres. Default [-0.15, 0.45]."},
        {"name": "dx_range", "type": "list[float]", "description": "Lateral shifts tried. Default [0, 0] (none)."},
        {"name": "step_m", "type": "float", "description": "Grid step. Default 0.025."},
        {"name": "poses", "type": "list[dict] | None", "description": "Hand poses the shift must serve: [{pose: <pose json path or dict>, side?: 'right'|'left' (default the pose's own side), waypoints?: [[dx,dy,dz], ...] displacements from the grasp the same arm must also hold (a lift, the carry's goal)}]."},
        {"name": "sides", "type": "list[str] | None", "description": "Arms whose reach counts for objects and points. Default both."},
        {"name": "stations", "type": "list[str] | None", "description": "Stations that move: both (default, keeping their separation) or one, e.g. ['right'] to bring only the right arm closer to a part the left arm's layout already serves."},
        {"name": "out_xml", "type": "str | None", "description": "Default <scene dir>/scene_stations.xml."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "shift [dx, dy] of this call, stations_moved, station_shifts {side: [dx, dy]} (each station's total from the setup, carried over from a previous call whose scene this one moved further), scene (the written file), reached {object: n points reached of 5}, poses_held (per pose at the chosen shift: ok, residual, reason), before {...}, candidates [{shift, reached, poses_held, worst_mm}], note."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "scipy"],
    "simulators": ["MuJoCo"],
    "tags": ["stage3", "setup", "reach", "arm", "workspace"],
    "used_in": ["Place Cup in Bowl", "Place Toy in Drawer", "Battery Insertion", "Meal Box (extra sample)"],
    "status": "tested",
    "entrypoint": "place_arm_stations",
}

import json
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import episode as E


def _hand_wrist_xy(model, side):
    from tool_library.common import plan as P
    rig = P.scene_rigs(model)[side]
    return rig.wrist_qpos["x"], rig.wrist_qpos["y"], rig


def _orientations(side):
    from scipy.spatial.transform import Rotation as Rot
    from tool_library.verifiers.arm_workspace_map import named_orientation
    pd = named_orientation(side, "palm_down")
    # palm down pitched 45 deg: fingers tipped toward the table about the station's x axis
    pitched = Rot.from_rotvec([np.radians(45.0), 0.0, 0.0]).as_matrix() @ pd
    return {"palm_down": pd, "palm_down_pitched": pitched, "fingers_in": named_orientation(side, "fingers_in")}


def _score(model, objects, sides, hover_m=0.06, tol_mm=4.0, tol_deg=3.0, points=None, probe=None):
    """Per object: how many of 5 probe points (the object and +-3 cm in x and y) some arm holds a hand at, and the
    worst best-residual."""
    from tool_library.common import armhand as AH
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    q0 = data.qpos.copy()
    arms = {s: AH.ArmSide(model, s) for s in sides}

    def solve_fast(s, p, R):
        """A probe, not a plan: the 4 nearest seeds of the station's library, 60 iterations each. A solution whose arm
        links run into the table or an object does not count (moving the stations can fold an elbow into the table)."""
        S = arms[s]
        pt_, Rt = np.asarray(p, float) + R @ S.p_m, R @ S.R_m
        best = None
        for sd in S.nearest_seeds(pt_, Rt, k=4):
            ep, er, qa = S._ik(q0, pt_, Rt, sd, 60)
            if S.ok(ep, er) and probe is not None:
                qq = q0.copy(); qq[S.arm.qadr] = qa
                if probe.check(qq[None], frames=[0]):
                    ep = max(ep, 0.05)                 # reached, but only through the table or an object: unreachable
            if best is None or ep + 0.05 * er < best[0] + 0.05 * best[1]:
                best = (ep, er)
            if S.ok(ep, er):
                break
        return best
    solvers = {s: (lambda p, R, s=s: solve_fast(s, p, R)) for s in sides}
    ors = {s: _orientations(s) for s in sides}
    out = {}
    probes = [(n, None) for n in objects] + [(f"point_{i}", np.asarray(pt, float)) for i, pt in enumerate(points or [])]
    for name, point in probes:
        if point is None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            sub = E.subtree_bodies(model, bid)
            gids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in sub and (model.geom_contype[g] | model.geom_conaffinity[g])]
            top = max((float(data.geom_xpos[g][2] + model.geom_rbound[g]) for g in gids), default=float(data.xpos[bid][2]))
            c = data.xpos[bid].copy()
        else:                                        # a destination the hand must reach (where a carried part goes)
            c, top = point.copy(), float(point[2])
        n_ok, worst = 0, 0.0
        for off in ((0, 0), (0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)):
            best = None
            for s in sides:
                for oname, R in ors[s].items():
                    if oname == "fingers_in":
                        p = np.array([c[0] + off[0] + (0.17 if s == "right" else -0.17), c[1] + off[1], max(top * 0.5, 0.04)])
                    else:
                        p = np.array([c[0] + off[0], c[1] + off[1], top + hover_m])
                    e = solvers[s](p, R)
                    sc = 1000 * e[0] + 3.0 * np.degrees(e[1])
                    if best is None or sc < best[0]:
                        best = (sc, 1000 * e[0], np.degrees(e[1]))
                    if best[1] <= tol_mm and best[2] <= tol_deg:
                        break
                if best[1] <= tol_mm and best[2] <= tol_deg:
                    break
            ok_pt = best[1] <= tol_mm and best[2] <= tol_deg
            if off == (0, 0):
                centre = bool(ok_pt)
            n_ok += int(ok_pt)
            worst = max(worst, best[1])
        out[name] = {"reached": n_ok, "centre": centre, "worst_mm": round(worst, 1)}
    return out


def _norm_poses(poses):
    out = []
    for i, it in enumerate(poses or []):
        if isinstance(it, (str, Path)) or (isinstance(it, dict) and "pose" not in it):
            it = {"pose": it}
        from tool_library.common.subgoal import load_pose
        pz = load_pose(it["pose"])
        side = it.get("side") or pz.get("side")
        if side not in ("left", "right"):
            raise ValueError(f"poses[{i}]: give its side ('left' or 'right'); the pose has none")
        out.append({"pose": pz, "side": side, "waypoints": [list(map(float, w)) for w in (it.get("waypoints") or [])],
                    "name": str(it["pose"]) if isinstance(it["pose"], (str, Path)) else f"pose_{i}"})
    return out


def _screen(model, plist, probe):
    """Per pose: does its own arm hold it (grasp, pre-grasp, waypoints) exactly and clear of the scene?"""
    from tool_library.common import armhand as AH
    if not plist:
        return []
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    q0 = data.qpos.copy()
    res = []
    for it in plist:
        r = AH.screen_poses(model, it["side"], [it["pose"]], q0, probe=probe, waypoints=it["waypoints"])[0]
        res.append({"pose": it["name"], "side": it["side"], "ok": bool(r.get("ok")), "residual_mm": r.get("residual_mm"),
                    "residual_deg": r.get("residual_deg"), "reason": r.get("reason")})
    return res


def _shift(model, base, sides, roots, wxy, dx, dy):
    for s in sides:
        model.body_pos[roots[s]] = base["root"][s] + np.array([dx, dy, 0.0])
        ax, ay, _ = wxy[s]
        if model.nkey:
            model.key_qpos[0][ax] = base["kx"][s] + dx
            model.key_qpos[0][ay] = base["ky"][s] + dy


def _table_lift(xml, sides, clearance_m=0.001):
    """Per moved station: how far it must rise so no collision geom of its arm sits inside the table (a station shifted
    over the tabletop stands on it instead of in it). Measured on the rigid runtime model the executor runs, at keyframe 0."""
    from tool_library.common import rigid_robot as RR
    from tool_library.common import hand as H
    m = RR.compile_rigid(str(xml)).model
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g_table")
    if table < 0:
        return {s: 0.0 for s in sides}
    out = {}
    for s in sides:
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot_{s}_arm_root")
        lift = 0.0
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if root < 0:                                 # every geom of the arm: the executor pairs them with the scene explicitly
                continue
            while b > 0 and b != root:
                b = int(m.body_parentid[b])
            if b != root:
                continue
            dist = H.geom_distance(m, d, g, table, 0.1)
            if dist < 0:
                lift = max(lift, -dist + clearance_m)
        out[s] = round(float(lift), 6)
    return out


def place_arm_stations(scene, objects=None, dy_range=(-0.15, 0.45), dx_range=(0.0, 0.0), step_m=0.025, out_xml=None, points=None,
                       poses=None, sides=None, stations=None):
    scene = Path(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    parts = E.robot_parts(model)
    sides_all = [s for s, pt in parts.items() if pt["gripper"] >= 0 and pt["arm_joints"]]
    if sides is not None and any(s_ not in sides_all for s_ in sides):
        raise ValueError(f"sides {sides}: the scene's arms are {sides_all}")
    probe_sides = list(sides) if sides else sides_all
    if stations is not None and (not stations or any(s_ not in sides_all for s_ in stations)):
        raise ValueError(f"stations {stations}: give one or both of {sides_all}")
    sides = list(stations) if stations else sides_all  # the stations that move (both by default, keeping their separation)
    plist = _norm_poses(poses)
    roots = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{s}_arm_root") for s in sides}
    if any(r < 0 for r in roots.values()):
        raise ValueError("place_arm_stations needs the bench's robot_<side>_arm_root bodies")
    if objects is None:
        objects = [n for n in E.object_free_joints(model)]
    wxy = {s: _hand_wrist_xy(model, s) for s in sides}
    base = {"root": {s: model.body_pos[roots[s]].copy() for s in sides},
            "kx": {s: float(model.key_qpos[0][wxy[s][0]]) if model.nkey else 0.0 for s in sides},
            "ky": {s: float(model.key_qpos[0][wxy[s][1]]) if model.nkey else 0.0 for s in sides}}
    from tool_library.common import armhand as AH
    probe = AH.ArmProbe(str(scene))
    proots = {s: mujoco.mj_name2id(probe.model, mujoco.mjtObj.mjOBJ_BODY, f"robot_{s}_arm_root") for s in sides}
    pbase = {s: probe.model.body_pos[proots[s]].copy() for s in sides}

    def shift_all(dx, dy):
        _shift(model, base, sides, roots, wxy, dx, dy)
        for s_ in sides:
            probe.model.body_pos[proots[s_]] = pbase[s_] + np.array([dx, dy, 0.0])

    before = _score(model, objects, probe_sides, points=points, probe=probe)
    cands, seen = [], set()

    def evaluate(dx, dy):
        key = (round(dx, 4), round(dy, 4))
        if key in seen:
            return
        seen.add(key)
        shift_all(float(dx), float(dy))
        sc = _score(model, objects, probe_sides, points=points, probe=probe)
        ph = _screen(model, plist, probe)
        cands.append({"shift": [float(key[0]), float(key[1])], "reached": int(sum(v["reached"] for v in sc.values())),
                      "poses_held": int(sum(r["ok"] for r in ph)),
                      "pose_err_mm": round(float(sum(min(r["residual_mm"] or 999.0, 999.0) for r in ph)), 1),
                      "all_centres": all(v["centre"] for v in sc.values()), "worst_mm": max((v["worst_mm"] for v in sc.values()), default=0.0),
                      "per_object": sc, "per_pose": ph})

    # coarse grid at twice the step, then the step around the best coarse candidates
    coarse = 2.0 * float(step_m)
    dys = np.round(np.arange(dy_range[0], dy_range[1] + 1e-9, coarse), 4)
    dxs = np.round(np.arange(dx_range[0], dx_range[1] + 1e-9, coarse), 4) if dx_range[1] > dx_range[0] else [float(dx_range[0])]
    if 0.0 not in dys and dy_range[0] <= 0.0 <= dy_range[1]:     # no shift is a candidate whenever the range allows it
        dys = np.sort(np.append(dys, 0.0))
    for dx in dxs:
        for dy in dys:
            evaluate(dx, dy)
    rank = lambda c: (c["poses_held"], -c["pose_err_mm"], c["reached"], -abs(c["shift"][1]) - abs(c["shift"][0]))
    for c in sorted(cands, key=rank, reverse=True)[:3]:
        for ddx, ddy in ((0.0, -float(step_m)), (0.0, float(step_m))) + (((-float(step_m), 0.0), (float(step_m), 0.0)) if dx_range[1] > dx_range[0] else ()):
            if dy_range[0] - 1e-9 <= c["shift"][1] + ddy <= dy_range[1] + 1e-9 and dx_range[0] - 1e-9 <= c["shift"][0] + ddx <= dx_range[1] + 1e-9:
                evaluate(c["shift"][0] + ddx, c["shift"][1] + ddy)
    shift_all(0.0, 0.0)
    zero = next((c for c in cands if abs(c["shift"][0]) < 1e-9 and abs(c["shift"][1]) < 1e-9), None)
    if zero is not None and all(v["centre"] for v in zero["per_object"].values()) and zero["poses_held"] == len(plist):
        best = zero                                   # every object and pose is reached where it is: the stations stay
    else:
        best = max(cands, key=rank)
    dx, dy = best["shift"]
    # write the scene with the stations moved (and each hand's home moved with its flange)
    spec = mujoco.MjSpec.from_file(str(scene))
    for s in sides:
        b = spec.body(f"robot_{s}_arm_root")
        b.pos = list(np.asarray(b.pos, float) + np.array([dx, dy, 0.0]))
    if spec.keys:
        k = spec.keys[0]
        q = np.array(k.qpos, float)
        for s in sides:
            ax, ay, rig = wxy[s]
            q[ax] += dx; q[ay] += dy
        k.qpos = q.tolist()
        if len(k.ctrl):
            c = np.array(k.ctrl, float)
            for s in sides:
                rig = wxy[s][2]
                for kk, d_ in (("x", dx), ("y", dy)):
                    a = rig.wrist_act.get(kk)
                    if a is not None and a < len(c):
                        c[a] += d_
            k.ctrl = c.tolist()
    out = Path(out_xml) if out_xml else scene.with_name("scene_stations.xml")
    out.write_text(spec.to_xml())
    # a station moved over the tabletop stands on it: lift it (and its hand's home) by what its arm would sink in
    lift = _table_lift(out, sides) if (dx, dy) != (0.0, 0.0) else {s: 0.0 for s in sides}
    if any(v > 0 for v in lift.values()):
        for s in sides:
            if lift[s] > 0:
                b = spec.body(f"robot_{s}_arm_root")
                b.pos = list(np.asarray(b.pos, float) + np.array([0.0, 0.0, lift[s]]))
        if spec.keys:
            k = spec.keys[0]
            q = np.array(k.qpos, float)
            c = np.array(k.ctrl, float) if len(k.ctrl) else None
            for s in sides:
                rig = wxy[s][2]
                if lift[s] > 0 and "z" in rig.wrist_qpos:
                    q[rig.wrist_qpos["z"]] += lift[s]
                    a = rig.wrist_act.get("z")
                    if c is not None and a is not None and a < len(c):
                        c[a] += lift[s]
            k.qpos = q.tolist()
            if c is not None:
                k.ctrl = c.tolist()
        out.write_text(spec.to_xml())
    m2 = mujoco.MjModel.from_xml_path(str(out))
    from tool_library.common import armhand as AH
    W = {s: AH.ArmSide(m2, s).residual(m2.key_qpos[0]) if m2.nkey else None for s in sides}
    # each station's total shift from the setup: a scene a previous call wrote carries its shifts forward
    prev = {}
    pj = scene.parent / "place_arm_stations.json"
    if pj.exists():
        try:
            pr = json.loads(pj.read_text())
            if Path(pr.get("scene", "")).resolve() == scene.resolve():
                prev = pr.get("station_shifts") or {s_: pr["shift"] for s_ in pr.get("stations_moved", sides_all)}
        except (ValueError, KeyError, OSError):
            prev = {}
    totals = {s_: [round(float(prev.get(s_, [0.0, 0.0])[0]) + (dx if s_ in sides else 0.0), 6),
                   round(float(prev.get(s_, [0.0, 0.0])[1]) + (dy if s_ in sides else 0.0), 6),
                   round(float((list(prev.get(s_, [0.0, 0.0])) + [0.0])[2]) + (lift.get(s_, 0.0) if s_ in sides else 0.0), 6)]
              for s_ in sides_all}
    res = {"shift": [dx, dy], "stations_moved": list(sides), "station_shifts": totals, "table_lift_m": lift,
           "scene": str(out), "reached": best["per_object"], "poses_held": best["per_pose"], "before": before,
           "home_weld_mm_deg": W, "candidates": [{k: v for k, v in c.items() if k not in ("per_object", "per_pose")} for c in cands],
           "note": ("no shift needed: every object is reached from the stations as they are" if (dx, dy) == (0.0, 0.0) else
                    f"stations moved {dy:+.3f} m toward the objects{f' and {dx:+.3f} m sideways' if dx else ''}; use {out.name} for Stage 3 and Stage 4")}
    (out.parent / "place_arm_stations.json").write_text(json.dumps(res, indent=1, default=float))
    return res
