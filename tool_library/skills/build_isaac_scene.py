"""Deformable objects live in Isaac Sim: the scene exported from MuJoCo plus cords as jointed capsule chains, settled and rendered."""
from __future__ import annotations

TOOL = {
    "name": "build_isaac_scene",
    "title": "Build Isaac Sim scene",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Exports a MuJoCo scene (robot, table, rigid objects at a given state) to Isaac Sim 5.0, adds each deformable cord (rope, cable, string) as a chain of capsule rigid bodies on D6 joints with bending stiffness and joint friction laid along its measured centreline, settles under PhysX gravity, renders through the registered camera, and saves the USD stage, the settled centrelines and a drift report.",
    "description": (
        "The deformable route of Stage 2: non-deformable objects are reconstructed and placed in MuJoCo; a scene with a "
        "rope-like object is then rebuilt in Isaac Sim with the same bench, camera and placement so the cord can be "
        "simulated with PhysX. The export reads every visible geom of the scene at `qpos` (boxes, capsules, cylinders, "
        "spheres, meshes with their vertices), the table gets collision, rigid objects get convex-hull collision, robot "
        "geoms are visual. Each deformable is {name, points (N, 3) in the rig frame, radius, rgb, density, fix_start, "
        "fix_end, bend_stiffness, bend_damping, joint_friction} plus, optionally, `preset` ('thin_cord', the default: "
        "8 mm heavily damped links for cords of a few millimetres; 'segmented_rope': the coefficient set fitted on a "
        "real 27.7 mm velvet rope, 27 mm links, self-colliding, sticky table and slippery rope-on-rope contact), "
        "`physics` (overrides of any coefficient of tool_library.common.isaac_cord.PRESETS), `n_links`, `cap_rgb` with "
        "`cap_length_a_m` / `cap_length_b_m` (rigid end caps: their colour) and `grasp` (how Stage 3's hands hold it); "
        "every cord is built by isaac_cord.build_cord, the builder Stage 3's Isaac episodes and Stage 4's replays use. "
        "`junction` {center, radius, rgb} ties several strands "
        "to one small body (a knot). An explicit n_links wins and a preset other than thin_cord takes its own link length; "
        "for thin cords the LINK LENGTH RULE: links are 8 mm long unless that makes more than `max_links` (default "
        "80) links in one articulation (a cord, or all strands tied to one junction); then the cord is split into exactly "
        "max_links equal links (a 1.9 m cord gets 24 mm links). Measured on a 1.25 mm cord: 110 links settle, 120 explode "
        "(drift 1e7 m), 128-206 make PhysX report invalid transforms and leave the cord where it was laid, 300 crash Isaac; "
        "a 6 mm cord holds 206 but not 300. An explicit `segment_m` that needs more than max_links links is refused. "
        "1/480 s TGS, settled for `settle_s`. A run is refused (RuntimeError naming the issue) when PhysX logs invalid "
        "transforms or errors, a settled centreline is non-finite, a chain is torn (neighbouring links further apart "
        "than one link length) or the cord moved more than `max_mean_drift_m` on average while settling; the report is "
        "still written with ok false and the issues. COLOUR: a cord's rgb goes to a UsdPreviewSurface diffuseColor (0..255 "
        "values are scaled to 0..1) and renders as that colour; a cord thinner than about 3 mm is a couple of pixels wide, "
        "and its hard shadow from the key light, as wide as the cord, made it read as nearly black, so thin cords cast no "
        "shadow by default (cord_cast_shadows). "
        "Runs the Isaac script in the Isaac Sim Python given by `isaac_python`, DEXAGENT_ISAAC_PYTHON or dexagent.yaml, "
        "with HOME redirected to `isaac_home` (Isaac writes gigabytes of cache) and the EULA accepted. Returns the render, "
        "the stage path, per-cord link counts and the mean drift during settling (a cord laid on the table should drift "
        "millimetres; centimetres mean it was floating or interpenetrating), and per cord `<tag>_<name>_cord.json`: the "
        "resolved specification with the SETTLED link poses (centres and orientations), which a Stage 2 entry with "
        "simulator 'isaac' names as isaac.cord_json and from which Stage 3 (execute_isaac_deformables) starts."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "path", "description": "The MuJoCo scene (bench + rigid objects)."},
        {"name": "camera", "type": "path | dict", "description": "camera.json in the rig frame."},
        {"name": "deformables", "type": "list[dict]", "description": "Cords to add: name, points [[x,y,z],...] rig metres, radius, rgb, density (1100), fix_start, fix_end, bend_stiffness (0.004), bend_damping (0.002), joint_friction (0.0015); optional preset ('thin_cord' | 'segmented_rope'), physics overrides, n_links, cap_rgb, cap_length_a_m, cap_length_b_m, grasp."},
        {"name": "out_dir", "type": "path", "description": "Receives export.json, <tag>_sim.png, <tag>_scene.usda, <tag>_report.json, settled centrelines."},
        {"name": "qpos", "type": "list | path | None", "description": "Scene state to export (npy or list); default the model's initial state."},
        {"name": "junction", "type": "dict | None", "description": "{center, radius, rgb}: a small rigid body the strands' first links are jointed to."},
        {"name": "segment_m", "type": "float | None", "description": "Link length in metres. Default None = the link length rule: 8 mm, lengthened so no articulation has more than max_links links."},
        {"name": "max_links", "type": "int", "description": "Most links PhysX gets in one articulation (one cord, or every strand tied to one junction). Default 80."},
        {"name": "max_mean_drift_m", "type": "float", "description": "A cord whose links move more than this on average while settling (it was floating, buried or blew up) fails the run. Default 0.10."},
        {"name": "cord_cast_shadows", "type": "bool | None", "description": "Whether cord links cast shadows. Default None: only cords of radius >= 3 mm do (a thinner cord is a few pixels wide and its hard shadow strip beside it makes it read as black)."},
        {"name": "settle_s", "type": "float", "description": "Settling time under gravity. Default 1.5."},
        {"name": "self_collision", "type": "bool | None", "description": "Cord self-collision between its own links, for every cord. Default None: each cord's preset decides (thin_cord off: its 8 mm links overlap and would push the chain apart; segmented_rope on: its capsules are trimmed so non-neighbours never touch at rest). The table and the rigid objects always collide with every link."},
        {"name": "tag", "type": "str", "description": "Output file prefix. Default 'isaac'."},
        {"name": "isaac_python", "type": "path | None", "description": "Isaac Sim's Python; default DEXAGENT_ISAAC_PYTHON."},
        {"name": "isaac_home", "type": "path | None", "description": "HOME for Isaac's caches; default DEXAGENT_ISAAC_HOME or out_dir/isaac_home."},
        {"name": "timeout_s", "type": "float", "description": "Default 1800."},
        {"name": "frame", "type": "path | None", "description": "Video frame for a 50/50 overlay of the render."},
        {"name": "exclude_bodies", "type": "list[str] | None", "description": "Body name prefixes left out of the export (bodies named like a deformable are always left out: Isaac rebuilds the cord)."},
    ],
    "outputs": [
        {"name": "sim_png / scene_usda / report_json / export_json", "type": "path", "description": "Written files."},
        {"name": "cord_json", "type": "dict", "description": "{cord name: <tag>_<name>_cord.json}: the resolved specification (schema dexagent.isaac_cord.v1) with link_poses {centres, quats} of the settled state."},
        {"name": "report", "type": "dict", "description": "ok, issues; per cord: n_links, segment_m, length_m, radius_m, preset, finite, mean_drift_m, max_drift_m, link_stretch, min_z_m, settled_centreline, settled_quats; '_' holds settle_s, physics_dt, physx_invalid_transforms, physx_errors."},
        {"name": "overlay_png", "type": "path | None", "description": "Render blended with the frame when given."},
    ],
    "status": "tested",
    "entrypoint": "build_isaac_scene",
    "requirements": ["mujoco>=3.1", "numpy", "scipy", "Isaac Sim 5.0 (pip isaacsim) in its own Python for the simulation step"],
    "simulators": ["Isaac Sim", "MuJoCo"],
    "tags": ["isaac sim", "deformable", "rope", "cable", "physx", "stage 2"],
    "used_in": ["Knot the Rope", "Scissor Cutting"],
    "cli": "python -m tool_library.skills.build_isaac_scene scene.xml camera.json deformables.json out/ --isaac-python /path/to/isaac/bin/python",
}

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from tool_library.common import isaac_cord as IC
from tool_library.common import rigcam as RC

SCRIPT = Path(__file__).resolve().parents[1] / "common" / "isaac_scene_script.py"
TYPE_NAMES = {int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
              int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}


def export_scene(scene_xml, qpos=None, table_geom="g_table", exclude_prefixes=()):
    """Every visible geom of the scene at qpos as a JSON-ready list in the world (= rig) frame; bodies whose
    name starts with one of exclude_prefixes are left out (the MuJoCo twin of a cord that Isaac rebuilds)."""
    m = mujoco.MjModel.from_xml_path(str(scene_xml)); d = mujoco.MjData(m)
    if qpos is not None:
        q = np.load(qpos) if isinstance(qpos, (str, Path)) else np.asarray(qpos, float)
        d.qpos[:len(q)] = q
    elif m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    geoms = []
    for g in range(m.ngeom):
        name = m.geom(g).name or f"g{g}"
        if m.geom_group[g] > 2 or "_collision_" in name or m.geom_rgba[g][3] < 0.05:
            continue
        t = int(m.geom_type[g])
        if t not in TYPE_NAMES:
            continue
        bname = m.body(int(m.geom_bodyid[g])).name
        if any(bname.startswith(px) or bname == px for px in exclude_prefixes):
            continue
        rgba = m.geom_rgba[g].copy()
        if m.geom_matid[g] >= 0:
            rgba = m.mat_rgba[int(m.geom_matid[g])].copy()
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        q = Rot.from_matrix(R).as_quat()
        rec = {"name": name, "body": m.body(int(m.geom_bodyid[g])).name, "type": TYPE_NAMES[t], "size": [float(x) for x in m.geom_size[g]],
               "rgba": [float(x) for x in rgba], "pos": [float(x) for x in p], "quat": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
               "is_table": name == table_geom}
        if TYPE_NAMES[t] == "mesh":
            mid = int(m.geom_dataid[g]); va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid]); fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            rec["vertices"] = np.round(m.mesh_vert[va:va + vn], 5).tolist(); rec["faces"] = m.mesh_face[fa:fa + fn].tolist()
        geoms.append(rec)
    return geoms


LINK_M = 0.008            # the default link length: a laid cord's curvature is resolved at 8 mm
MAX_LINKS = 80            # links per articulation PhysX settles reliably (thin cords explode from ~120, see TOOL)
THIN_CORD_M = 0.003       # below this radius a cord casts no shadow (its shadow strip would render it black)


def link_plan(deformables, junction=None, segment_m=None, max_links=MAX_LINKS):
    """{cord name: (n_links, link length m)} by the link length rule. Strands tied to one junction are one articulation,
    so they share the budget (in proportion to their lengths)."""
    import math
    lengths = {c["name"]: float(np.linalg.norm(np.diff(np.asarray(c["points"], float), axis=0), axis=1).sum()) for c in deformables}
    for n, L in lengths.items():
        if not np.isfinite(L) or L <= 0:
            raise ValueError(f"deformable {n!r}: its points have zero or non-finite length")
    groups = [list(lengths)] if junction else [[n] for n in lengths]
    plan = {}
    for g in groups:
        total = sum(lengths[n] for n in g)
        if segment_m is not None:
            seg = float(segment_m)
            if seg <= 0:
                raise ValueError("segment_m must be > 0")
            need = sum(max(4, int(round(lengths[n] / seg))) for n in g)
            if need > int(max_links):
                raise ValueError(f"segment_m {seg * 1000:.1f} mm makes {need} links in one articulation ({', '.join(g)}, "
                                 f"{total:.2f} m); PhysX blows up long thin chains: use segment_m >= {total / int(max_links):.4f} "
                                 f"(or leave it out for the automatic rule), or raise max_links for a thick cord")
        else:
            seg = max(LINK_M, total / int(max_links))
        for n in g:
            k = max(4, int(round(lengths[n] / seg)))
            if segment_m is None and len(g) > 1:
                k = max(4, min(k, int(math.floor(int(max_links) * lengths[n] / total))))
            plan[n] = (k, lengths[n] / k)
    return plan


def physx_log_issues(text):
    """Counts of PhysX trouble in an Isaac log: invalid transforms (a NaN pose PhysX refused to write back) and PhysX
    errors (Illegal BroadPhaseUpdateData and the like)."""
    inv = text.count("Invalid PhysX transform")
    err = sum(1 for line in text.splitlines() if "PhysX error" in line)
    return inv, err


def judge_report(report, log_text, max_mean_drift_m=0.10, max_link_stretch=0.25):
    """The issues that make a settled Isaac scene garbage (empty list = trustworthy)."""
    issues = []
    inv, err = physx_log_issues(log_text)
    if inv or err:
        issues.append(f"PhysX logged {inv} invalid transforms and {err} errors: the cord articulation blew up and its poses were "
                      f"never written back, so the render and the drift are of the cord as laid, not as simulated; shorten "
                      f"the chain (fewer, longer links: max_links / segment_m) or check that the cord is not laid inside an object")
    for n, r in report.items():
        if n == "_" or not isinstance(r, dict):
            continue
        if r.get("finite") is False or any(r.get(k) is not None and not np.isfinite(r[k]) for k in ("mean_drift_m", "max_drift_m")):
            issues.append(f"cord {n!r}: the settled centreline is not finite (the simulation diverged)")
            continue
        md = r.get("mean_drift_m")
        if md is not None and md > float(max_mean_drift_m):
            issues.append(f"cord {n!r} moved {md:.3g} m on average while settling (limit {max_mean_drift_m} m): it exploded, fell "
                          f"off an object or was laid floating / inside geometry; check its points against the scene")
        st = r.get("link_stretch")
        if st is not None and st > float(max_link_stretch):
            issues.append(f"cord {n!r}: neighbouring links ended {100 * st:.0f}% further apart than one link length, so the chain "
                          f"tore (an unstable articulation)")
    return issues


def resolve_cords(deformables, junction=None, segment_m=None, max_links=MAX_LINKS, self_collision=None, cord_cast_shadows=None):
    """Each cord's full specification (isaac_cord.cord_spec): an explicit n_links wins, a preset other than thin_cord
    takes its own link length, thin cords follow the link length rule (link_plan, strands of one junction sharing one
    articulation's budget). A strand tied to a junction starts at the junction."""
    plain = [c for c in deformables if not c.get("n_links") and str(c.get("preset") or "thin_cord") == "thin_cord"]
    plan = link_plan(plain, junction, segment_m, max_links) if plain else {}
    out = []
    for c in deformables:
        c = dict(c)
        if junction:
            P = np.asarray(c["points"], float); kc = np.asarray(junction["center"], float)
            if np.linalg.norm(P[-1, :2] - kc[:2]) < np.linalg.norm(P[0, :2] - kc[:2]):
                c["points"] = P[::-1].tolist()
        if c.get("n_links") and int(c["n_links"]) > int(max_links):
            raise ValueError(f"deformable {c['name']!r}: n_links {c['n_links']} is above max_links {max_links}; PhysX blows up long "
                             f"thin chains (raise max_links only for a thick cord)")
        if self_collision is not None:
            c["self_collision"] = bool(self_collision)
        if cord_cast_shadows is not None:
            c["cast_shadows"] = bool(cord_cast_shadows)
        elif c.get("cast_shadows") is None:
            c["cast_shadows"] = float(c["radius"]) >= THIN_CORD_M
        spec = IC.cord_spec(c, n_links=plan[c["name"]][0] if c["name"] in plan else None)
        if spec["n_links"] > int(max_links):
            raise ValueError(f"deformable {c['name']!r}: {spec['n_links']} links of its preset's length is above max_links {max_links}")
        out.append(spec)
    return out


def build_isaac_scene(scene_xml, camera, deformables, out_dir, qpos=None, junction=None, segment_m=None, settle_s=1.5, self_collision=None,
                      tag="isaac", isaac_python=None, isaac_home=None, timeout_s=1800.0, frame=None, exclude_bodies=None,
                      max_links=MAX_LINKS, max_mean_drift_m=0.10, cord_cast_shadows=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cam = RC.load(camera) if isinstance(camera, (str, Path)) else camera
    if not deformables:
        raise ValueError("deformables is empty: build_isaac_scene is for scenes with at least one cord-like object")
    for c in deformables:
        for k in ("name", "points", "radius"):
            if k not in c:
                raise ValueError(f"deformable {c.get('name', '?')!r} needs {k}")
        if len(c["points"]) < 2:
            raise ValueError(f"deformable {c['name']!r} needs at least two centreline points")
        col = c.get("rgb") or c.get("rgba") or c.get("color") or c.get("colour")
        if col is not None:
            c["rgb"] = [float(v) for v in list(col)[:3]]
            if max(c["rgb"]) > 1.0:                  # 0..255 colours (a measured pixel) -> 0..1
                c["rgb"] = [v / 255.0 for v in c["rgb"]]
        if not np.isfinite(np.asarray(c["points"], float)).all():
            raise ValueError(f"deformable {c['name']!r}: its points contain NaN or inf")
    specs = resolve_cords(deformables, junction, segment_m, max_links, self_collision, cord_cast_shadows)
    plan = {sp["name"]: (sp["n_links"], sp["segment_m"]) for sp in specs}
    # a cord that also exists in the MuJoCo scene (as a straight cable asset) must not be exported twice
    prefixes = tuple(exclude_bodies or ()) + tuple(f"{c['name']}" for c in deformables) + tuple(f"{c['name']}_" for c in deformables)
    export = {"camera": {k: cam[k] for k in ("width", "height", "fovy_deg", "pos", "R_cv")}, "geoms": export_scene(scene_xml, qpos, exclude_prefixes=prefixes),
              "deformables": specs, "junction": junction, "segment_m": LINK_M if segment_m is None else float(segment_m),
              "settle_s": float(settle_s), "tag": tag, "scene_xml": str(scene_xml)}
    exp = out / "export.json"; exp.write_text(json.dumps(export))
    py = isaac_python or os.environ.get("DEXAGENT_ISAAC_PYTHON")
    if not py or not Path(py).exists():
        raise FileNotFoundError("Isaac Sim's Python not found: pass isaac_python or set DEXAGENT_ISAAC_PYTHON (dexagent.yaml environments.isaac)")
    home = Path(isaac_home or os.environ.get("DEXAGENT_ISAAC_HOME") or (out / "isaac_home")); home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home), OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PYTHONUNBUFFERED="1", EXPORT=str(exp), OUT=str(out))
    env.setdefault("TMPDIR", str(out / "tmp")); Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    log = out / f"{tag}_isaac.log"
    with log.open("w") as fh:
        r = subprocess.run([str(py), str(SCRIPT)], env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=float(timeout_s))
    text = log.read_text()
    if r.returncode != 0 or "ISAAC_DONE" not in text:
        raise RuntimeError(f"Isaac Sim run failed (returncode {r.returncode}); log tail:\n{text[-3000:]}")
    report = json.loads((out / f"{tag}_report.json").read_text())
    issues = judge_report(report, text, max_mean_drift_m)
    inv, err = physx_log_issues(text)
    report.setdefault("_", {}).update(physx_invalid_transforms=inv, physx_errors=err)
    report["ok"], report["issues"] = not issues, issues
    (out / f"{tag}_report.json").write_text(json.dumps(report, indent=1))
    if issues:
        raise RuntimeError("build_isaac_scene: the settled scene is not trustworthy: " + " | ".join(issues)
                           + f" (report {out / f'{tag}_report.json'}, log {log}, render {out / f'{tag}_sim.png'})")
    # each cord's specification with its settled link poses: the asset Stage 3's Isaac episode and Stage 4's replays start from
    cord_json = {}
    for sp in specs:
        row = report.get(sp["name"]) or {}
        if not row.get("settled_centreline") or not row.get("settled_quats"):
            raise RuntimeError(f"build_isaac_scene: the Isaac run returned no settled link poses for {sp['name']!r} (report {out / f'{tag}_report.json'})")
        cj = out / f"{tag}_{sp['name']}_cord.json"
        IC.save_spec(dict(sp, link_poses={"centres": row["settled_centreline"], "quats": row["settled_quats"]},
                          settled={"settle_s": float(settle_s), "mean_drift_m": row.get("mean_drift_m"), "report": f"{tag}_report.json"}), cj)
        cord_json[sp["name"]] = str(cj)
    result = {"ok": True, "issues": [], "sim_png": str(out / f"{tag}_sim.png"), "scene_usda": str(out / f"{tag}_scene.usda"),
              "report_json": str(out / f"{tag}_report.json"), "export_json": str(exp), "log": str(log), "report": report,
              "cord_json": cord_json, "links": {n: {"n_links": k, "segment_m": round(seg, 5)} for n, (k, seg) in plan.items()}}
    if frame is not None:
        import cv2
        a = cv2.imread(str(frame)); b = cv2.imread(result["sim_png"])
        if a is not None and b is not None:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
            ov = out / f"{tag}_overlay.png"; cv2.imwrite(str(ov), (0.5 * a + 0.5 * b).astype(np.uint8)); result["overlay_png"] = str(ov)
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=TOOL["summary"])
    ap.add_argument("scene_xml"); ap.add_argument("camera"); ap.add_argument("deformables_json"); ap.add_argument("out_dir")
    ap.add_argument("--qpos"); ap.add_argument("--isaac-python"); ap.add_argument("--frame"); ap.add_argument("--settle-s", type=float, default=1.5)
    a = ap.parse_args(argv)
    d = json.loads(Path(a.deformables_json).read_text())
    r = build_isaac_scene(a.scene_xml, a.camera, d.get("deformables", d) if isinstance(d, dict) else d, a.out_dir, a.qpos, (d.get("junction") if isinstance(d, dict) else None),
                          settle_s=a.settle_s, isaac_python=a.isaac_python, frame=a.frame)
    r["report"] = {k: {kk: vv for kk, vv in v.items() if kk != "settled_centreline"} if isinstance(v, dict) else v for k, v in r["report"].items()}
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.exit(main())
