"""Validate a deformable reconstructed in Isaac Sim: its cord specification, its settled state and its placement."""
from __future__ import annotations

TOOL = {
    "name": "isaac_deformable_rest_check", "title": "Isaac deformable rest check",
    "type": "verifier", "category": "Scene Assertions",
    "summary": "Re-measures every deformable a Stage 2 scene simulates in Isaac Sim from its files: the cord specification and "
               "its settled link poses, the settling report (no PhysX errors), drift from the laid centreline, chain "
               "connectivity and stretch, self-interpenetration, sinking into the table or a rigid object, no MuJoCo copy, and "
               "the settled cord's silhouette through the camera against the object's mask.",
    "description": (
        "The Isaac Sim counterpart of deformable_rest_check and deformable_topology_check: deformable objects are "
        "simulated in Isaac Sim only, so their rest state is the one build_isaac_scene settled, and nothing written in "
        "scene_recon.json about it is taken on trust. For every entry with type deformable and simulator 'isaac' (or "
        "skill build_isaac_scene): isaac.cord_json must load as an isaac_cord specification with settled link_poses "
        "(one centre and one unit quaternion per link, finite); n_links, segment_m, length_m and radius must agree with "
        "each other and with the measured centreline (points); isaac.report_json must say ok with no PhysX invalid "
        "transforms or errors. From the poses: (1) drift of every link from where the cord was laid (the points "
        "resampled into its links): mean <= max_mean_drift_m (0.02) and max <= max_link_drift_m (0.05), as the MuJoCo "
        "rest check; the report's and the entry's claimed mean drift must match the recomputed one within 5 mm; (2) "
        "connectivity: neighbouring links meet at their joint (each link's end against the next link's start within "
        "max(2 mm, 10% of a link)): one chain, two free ends, no torn or fused link; neighbouring centres at most "
        "10% further apart than one link; (3) self-interpenetration at rest: capsules (the PhysX geometry, trimmed "
        "cylinder plus radius) at least two links apart overlap by at most self_tol_m (2 mm), as deformable_topology_check; "
        "(4) the table: no link's lowest point more than pen_m (3 mm) below the table top of scene.xml; (5) rigid "
        "objects and the robot: no link's axis point deeper than pen_m inside a collidable geom of scene.xml at keyframe "
        "0; (6) no MuJoCo copy: scene.xml has no body named like the cord's or the entry's MuJoCo cable "
        "(<name>B_first...) and none of the entry's root_body / bodies; (7) placement: the settled centreline drawn as "
        "a tube of the cord's radius through camera.json is scored against the object's mask by centreline (the "
        "thin-object score of silhouette_iou_check: both skeletons widened to a common tolerance) with the "
        "demonstrator's hands (masks/hands.png) left out; below min_iou (0.25) fails. Scope: cords (capsule chains); "
        "cloth and volumetric soft bodies are not certified."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Stage 2 scene.xml (the MuJoCo bench and rigid objects)."},
        {"name": "recon", "type": "str | dict", "description": "scene_recon.json (paths relative to it)."},
        {"name": "camera", "type": "str | dict | None", "description": "camera.json; default the recon's camera_json."},
        {"name": "ignore_mask", "type": "str | None", "description": "Hand mask left out of the silhouettes; default masks/hands.png next to the recon."},
        {"name": "objects", "type": "list[str] | None", "description": "Only these object ids."},
        {"name": "max_mean_drift_m", "type": "float", "description": "Largest mean link drift while settling. Default 0.02."},
        {"name": "max_link_drift_m", "type": "float", "description": "Largest single link drift. Default 0.05."},
        {"name": "pen_m", "type": "float", "description": "Allowed sinking into the table or a geom. Default 0.003."},
        {"name": "self_tol_m", "type": "float", "description": "Allowed overlap of capsules two or more links apart. Default 0.002."},
        {"name": "min_iou", "type": "float", "description": "Centreline silhouette score floor. Default 0.25."},
    ],
    "outputs": [{"name": "report", "type": "dict", "description": "ok, issues, objects {id: {cord, n_links, segment_m, mean_drift_m, max_link_drift_m, claimed_drift_m, joint_gap_m, link_stretch, self_overlap_m, below_table_m, inside_geom {geom, depth_m}, iou, plain_iou}}; scope=isaac_cord."}],
    "entrypoint": "isaac_deformable_rest_check", "status": "tested",
    "requirements": ["mujoco>=3.2", "numpy", "opencv-python", "scikit-image"], "simulators": ["Isaac Sim", "MuJoCo"],
    "tags": ["deformable", "rope", "cable", "isaac sim", "rest", "topology", "stage2"],
    "used_in": ["Knot the Rope"],
}

import json
import re
from pathlib import Path

import mujoco
import numpy as np

from tool_library.common import isaac_cord as IC
from tool_library.common import scene_assert as SA


def _seg_dist(p0, p1, q0, q1):
    """Distance between segments p0p1 and q0q1."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0.0:
        t, s = 0.0, (np.clip(-c / a, 0.0, 1.0) if a > 1e-12 else 0.0)
    elif t > 1.0:
        t, s = 1.0, (np.clip((b - c) / a, 0.0, 1.0) if a > 1e-12 else 0.0)
    return float(np.linalg.norm((p0 + s * d1) - (q0 + t * d2)))


def _resolve(base, p):
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else base / q


def _table_top(model):
    from tool_library.common import episode_audit as EA
    t = EA.table_info(model)
    return float(t["top_z"]) if t else 0.0


def cord_of(entry, base):
    """(spec, report row, report, issues) of an Isaac deformable entry, from its isaac.cord_json and report_json."""
    isa = entry.get("isaac") or {}
    issues, spec, row, rep = [], None, None, None
    cj = _resolve(base, isa.get("cord_json"))
    if cj is None or not cj.exists():
        issues.append("isaac.cord_json is missing or does not exist (build_isaac_scene writes <tag>_<name>_cord.json)")
    else:
        try:
            spec = IC.load_spec(cj)
        except Exception as e:                               # noqa: BLE001
            issues.append(f"isaac.cord_json does not load as a cord specification: {e}")
    rj = _resolve(base, isa.get("report_json"))
    if rj is None or not rj.exists():
        issues.append("isaac.report_json is missing or does not exist")
    else:
        try:
            rep = json.loads(rj.read_text())
        except Exception as e:                               # noqa: BLE001
            issues.append(f"isaac.report_json does not load: {e}")
    if spec is not None and rep is not None:
        row = rep.get(spec["name"])
        if not isinstance(row, dict):
            issues.append(f"isaac.report_json has no row for cord {spec['name']!r}: it is the report of another run")
        if rep.get("ok") is not True:
            issues.append(f"the Isaac settling report is not ok: {rep.get('issues')}")
        meta = rep.get("_") or {}
        if int(meta.get("physx_invalid_transforms") or 0) or int(meta.get("physx_errors") or 0):
            issues.append(f"PhysX logged {meta.get('physx_invalid_transforms')} invalid transforms and {meta.get('physx_errors')} errors while "
                          "settling: the poses were never simulated")
    return spec, row, rep, issues


def isaac_deformable_rest_check(scene, recon, camera=None, ignore_mask=None, objects=None, max_mean_drift_m=0.02, max_link_drift_m=0.05,
                                pen_m=0.003, self_tol_m=0.002, min_iou=0.25):
    man, base = SA.load_json(recon)
    base = Path(base) if base is not None else Path(scene).parent
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    top = _table_top(model)
    table = SA.table_body_ids(model)
    names = {model.body(b).name for b in range(model.nbody)}
    geoms = [g for g in range(model.ngeom) if SA.geom_collidable(model, g) and int(model.geom_bodyid[g]) not in table
             and int(model.geom_bodyid[g]) != 0]
    cam = camera if camera is not None else _resolve(base, man.get("camera_json") or "camera.json")
    cam = SA.load_camera(cam) if cam is not None and (isinstance(cam, dict) or Path(cam).exists()) else None
    ign_p = Path(ignore_mask) if ignore_mask else base / "masks" / "hands.png"
    issues, rows = [], {}
    want = None if objects is None else {str(x) for x in objects}
    for o in man.get("objects", []):
        if o.get("type") != "deformable" or not SA.is_isaac_only(o) or (want is not None and str(o.get("id")) not in want):
            continue
        oid = str(o.get("id"))
        spec, rrow, rep, iss = cord_of(o, base)
        issues += [f"{oid}: {i}" for i in iss]
        if spec is None:
            rows[oid] = {"verdict": "no cord specification"}
            continue
        row = {"cord": spec["name"], "n_links": spec["n_links"], "segment_m": round(spec["segment_m"], 5), "preset": spec.get("preset")}
        rows[oid] = row
        lp = spec.get("link_poses") or {}
        C, Q = np.asarray(lp.get("centres") or [], float), np.asarray(lp.get("quats") or [], float)
        n, seg, R = int(spec["n_links"]), float(spec["segment_m"]), float(spec["radius"])
        if C.shape != (n, 3) or Q.shape != (n, 4):
            issues.append(f"{oid}: the cord specification needs settled link_poses with {n} centres and {n} quaternions "
                          f"(got {C.shape} / {Q.shape}): rebuild it with build_isaac_scene")
            continue
        if not (np.isfinite(C).all() and np.isfinite(Q).all()):
            issues.append(f"{oid}: a settled link pose is not finite (the simulation diverged)")
            continue
        if np.abs(np.linalg.norm(Q, axis=1) - 1.0).max() > 1e-3:
            issues.append(f"{oid}: settled link orientations are not unit quaternions")
            continue
        # the specification against itself and the measured centreline
        L = IC.polyline_length(spec["points"])
        if R <= 0 or abs(n * seg - float(spec["length_m"])) > 1e-4 or abs(L - float(spec["length_m"])) > 0.01 * L + 1e-4:
            issues.append(f"{oid}: the specification is inconsistent (n_links {n} x segment {seg:.4f} m vs length {spec['length_m']:.4f} m, "
                          f"centreline {L:.4f} m, radius {R})")
        if rrow and (int(rrow.get("n_links", n)) != n or abs(float(rrow.get("radius_m", R)) - R) > 1e-6):
            issues.append(f"{oid}: the settling report's cord ({rrow.get('n_links')} links, radius {rrow.get('radius_m')}) is not this specification's")
        # (1) drift from the lay
        laid, _ = IC.lay_poses(spec["points"], n, R)
        dr = np.linalg.norm(C - laid, axis=1)
        row.update(mean_drift_m=round(float(dr.mean()), 5), max_link_drift_m=round(float(dr.max()), 5))
        if dr.mean() > max_mean_drift_m or dr.max() > max_link_drift_m:
            issues.append(f"{oid}: the cord does not keep the shape it was laid in (mean drift {dr.mean() * 1000:.1f} mm, max "
                          f"{dr.max() * 1000:.1f} mm; limits {max_mean_drift_m * 1000:.0f} / {max_link_drift_m * 1000:.0f} mm): it was laid "
                          f"floating, inside geometry, or the chain is unstable")
        claims = [("the report", (rrow or {}).get("mean_drift_m")), ("the entry", (o.get("isaac") or {}).get("mean_drift_m"))]
        row["claimed_drift_m"] = {k: v for k, v in claims}
        for who, v in claims:
            if v is not None and (not np.isfinite(float(v)) or abs(float(v) - float(dr.mean())) > 0.005):
                issues.append(f"{oid}: {who} claims a mean drift of {float(v) * 1000:.1f} mm, the files give {dr.mean() * 1000:.1f} mm")
        # (2) connectivity: each link's end meets the next link's start
        X = np.stack([IC.quat_to_mat(q)[:, 0] for q in Q])
        gap = np.linalg.norm((C[:-1] + 0.5 * seg * X[:-1]) - (C[1:] - 0.5 * seg * X[1:]), axis=1) if n > 1 else np.zeros(0)
        stretch = (np.linalg.norm(np.diff(C, axis=0), axis=1).max() / seg - 1.0) if n > 1 else 0.0
        row.update(joint_gap_m=round(float(gap.max()) if len(gap) else 0.0, 5), link_stretch=round(float(stretch), 4))
        bad = np.where(gap > max(0.002, 0.1 * seg))[0]
        if len(bad):
            issues.append(f"{oid}: the chain is broken at {len(bad)} joint(s) (links {int(bad[0])}/{int(bad[0]) + 1} end "
                          f"{gap[bad[0]] * 1000:.1f} mm apart): not one connected strand")
        if stretch > 0.1:
            issues.append(f"{oid}: neighbouring links are {100 * stretch:.0f}% further apart than one link: the chain is stretched or torn")
        # (3) self-interpenetration at rest (the PhysX capsules: trimmed cylinder plus radius)
        half = 0.5 * max(seg - float(spec["physics"]["capsule_trim_m"]), 0.001)
        A0, A1 = C - half * X, C + half * X
        worst = (0.0, None)
        for i in range(n):
            for j in range(i + 2, n):
                if np.linalg.norm(C[i] - C[j]) > 2 * half + 2 * R + 0.01:
                    continue
                ov = 2 * R - _seg_dist(A0[i], A1[i], A0[j], A1[j])
                if ov > worst[0]:
                    worst = (ov, (i, j))
        row["self_overlap_m"] = round(worst[0], 5)
        if worst[0] > self_tol_m:
            issues.append(f"{oid}: links {worst[1][0]} and {worst[1][1]} overlap by {worst[0] * 1000:.1f} mm at rest: the cord passes through itself")
        # (4) the table, (5) collidable geoms
        below = float(top - (C[:, 2] - R).min())
        row["below_table_m"] = round(below, 5)
        if below > pen_m:
            issues.append(f"{oid}: link {int(np.argmin(C[:, 2]))} sinks {below * 1000:.1f} mm into the table (table top z = {top:.4f} m)")
        P = np.concatenate([A0, C, A1])
        deep = (0.0, None)
        for g in geoms:
            dmin = float(np.min(SA.signed_distance(model, data, g, P)) - R)
            if -dmin > deep[0]:
                deep = (-dmin, g)
        row["inside_geom"] = {"geom": SA.geom_label(model, deep[1]) if deep[1] is not None else None, "depth_m": round(deep[0], 5)}
        if deep[0] > pen_m:
            issues.append(f"{oid}: the cord lies {deep[0] * 1000:.1f} mm inside {SA.geom_label(model, deep[1])} of scene.xml at keyframe 0")
        # (6) no MuJoCo copy
        pat = re.compile(rf"^(?:{re.escape(spec['name'])}|{re.escape(oid)}).*B_(?:first|last|\d+)$")
        dup = sorted({b for b in names if pat.match(b)} | {b for b in [o.get("root_body"), *(o.get("bodies") or [])] if isinstance(b, str) and b in names})
        if dup:
            issues.append(f"{oid}: simulated in Isaac Sim, but scene.xml also has MuJoCo bodies for it ({dup[:3]}): a deformable lives in "
                          f"Isaac Sim only; remove the MuJoCo cable")
        # (7) placement through the camera
        mk = _resolve(base, o.get("mask"))
        if cam is None or mk is None or not mk.exists():
            issues.append(f"{oid}: the placement cannot be checked (camera.json or the object's mask is missing)")
            continue
        gt = SA.load_mask(mk)
        H, W = gt.shape
        sil = SA.cord_tube_mask(cam, C, R, seg, size=(W, H), quats=Q)
        if ign_p.exists():
            ign = SA.load_mask(ign_p, (W, H))
            sil, gt = sil & ~ign, gt & ~ign
        from tool_library.verifiers.silhouette_iou_check import _centreline_score
        score, info = _centreline_score(sil, gt)
        plain = float((sil & gt).sum() / max((sil | gt).sum(), 1))
        row.update(iou=round(float(score if score is not None else plain), 3), plain_iou=round(plain, 3), thin=info)
        if row["iou"] < min_iou:
            issues.append(f"{oid}: the settled cord's silhouette scores {row['iou']:.2f} against its mask (floor {min_iou}): the cord is "
                          f"not laid where the video shows it")
    return {"ok": not issues, "issues": issues, "objects": rows, "scope": "isaac_cord", "table_top_z": round(top, 5),
            "note": None if rows else "no deformable simulated in Isaac Sim"}
