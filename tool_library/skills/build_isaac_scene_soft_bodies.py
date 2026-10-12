"""Isaac Sim scene for compact soft objects (a plush toy, a sponge, a bean bag): the MuJoCo bench is exported as
static geometry, and each named soft body becomes a dynamic PhysX rigid body built from its own MuJoCo geoms with a
compliant-contact material, settled under gravity, rendered through the registered camera, with a drift report.

    from tool_library.skills.build_isaac_scene_soft_bodies import build_isaac_scene_soft_bodies
    r = build_isaac_scene_soft_bodies("scene.xml", "camera.json", [{"body": "toy_toy"}], "isaac",
                                      isaac_python="/path/to/isaac/python")
    r["report"]["toy_toy"]["mean_drift_m"]
"""
from __future__ import annotations

TOOL = {
    "name": "build_isaac_scene_soft_bodies",
    "title": "Build Isaac scene with compact soft bodies",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Exports a MuJoCo scene to Isaac Sim with each compact soft object (plush, sponge) as a dynamic PhysX body "
               "of its own geoms on a compliant-contact material, settles it, renders through the video camera and "
               "reports each soft body's settling drift.",
    "description": (
        "The cord route (build_isaac_scene) only rebuilds rope-like deformables; compact soft objects such as stuffed "
        "toys are modelled in MuJoCo as light free bodies with soft contact (solref time constant and damping ratio). "
        "This tool carries the same model into PhysX. Every visible geom of the scene at its initial keyframe (or "
        "`qpos`) is exported in the rig frame, as build_isaac_scene does (table with collision and friction, other "
        "non-robot objects with static convex-hull collision). Each body named in `soft_bodies` is left out of the "
        "static export and rebuilt as ONE dynamic rigid body at its MuJoCo pose and mass: boxes, spheres, capsules and "
        "cylinders as USD primitives, ellipsoids and meshes as convex-hull meshes, visual-only geoms without collision. "
        "Its physics material is compliant: stiffness k = m (2 pi / tc)^2 and damping c = 2 zeta m (2 pi / tc), with tc "
        "and zeta read from the geoms' solref (or given per body), so a palm pressing it sinks as in MuJoCo. The scene "
        "settles for `settle_s` under gravity; the report gives per soft body the start and end position, mean_drift_m "
        "(= max_drift_m, one body), the rotation change and the contact parameters. The Isaac script is written next "
        "to the export and run with Isaac Sim's Python. Assumes rig frame = stage frame, z up, metres."
    ),
    "inputs": [
        {"name": "scene_xml", "type": "path", "description": "The MuJoCo scene (bench + objects)."},
        {"name": "camera", "type": "path | dict", "description": "camera.json in the rig frame (width, height, fovy_deg, pos, R_cv)."},
        {"name": "soft_bodies", "type": "list[dict]", "description": "[{body, stiffness_N_per_m?, damping_Ns_per_m?, friction?}]: MuJoCo body names to rebuild as soft dynamic bodies."},
        {"name": "out_dir", "type": "path", "description": "Receives export.json, the Isaac script, <tag>_sim.png, <tag>_scene.usda, <tag>_report.json."},
        {"name": "qpos", "type": "path | list | None", "description": "Scene state to export; default keyframe 0 or qpos0."},
        {"name": "settle_s", "type": "float", "description": "Settling time under gravity. Default 1.5."},
        {"name": "tag", "type": "str", "description": "Output file prefix. Default 'isaac'."},
        {"name": "isaac_python", "type": "str | None", "description": "Isaac Sim's Python; default env DEXAGENT_ISAAC_PYTHON."},
        {"name": "isaac_home", "type": "str | None", "description": "HOME for Isaac's caches; default env DEXAGENT_ISAAC_HOME or out_dir/isaac_home."},
        {"name": "timeout_s", "type": "float", "description": "Default 1800."},
        {"name": "frame", "type": "path | None", "description": "Video frame for a 50/50 overlay of the render."},
        {"name": "dry_run", "type": "bool", "description": "Write export.json and the script but do not launch Isaac. Default False."},
    ],
    "outputs": [
        {"name": "sim_png / scene_usda / report_json / export_json / script", "type": "str", "description": "Written files (sim_png, scene_usda, report_json absent on a dry run)."},
        {"name": "report", "type": "dict", "description": "Per soft body: start_pos, end_pos, mean_drift_m, max_drift_m, rot_change_deg, mass_kg, stiffness_N_per_m, damping_Ns_per_m, n_geoms; '_' holds settle_s and physics_dt."},
        {"name": "soft", "type": "list[dict]", "description": "The soft bodies as exported (pose, mass, contact parameters, geoms)."},
        {"name": "overlay_png", "type": "str", "description": "Render blended with the frame when given."},
    ],
    "status": "tested",
    "entrypoint": "build_isaac_scene_soft_bodies",
    "requirements": ["numpy", "scipy", "mujoco", "opencv-python", "Isaac Sim 5.0 Python (for the run)"],
    "simulators": ["mujoco", "isaac"],
    "tags": ["isaac", "physx", "soft", "plush", "sponge", "deformable", "compliant contact", "export", "settle", "drift"],
    "used_in": ["Place Toy in Drawer"],
    "cli": "python -m tool_library.skills.build_isaac_scene_soft_bodies <scene_xml> <camera_json> <body> <out_dir>",
}

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

PRIM = {"box", "sphere", "capsule", "cylinder"}


def _geom_type_names():
    import mujoco
    G = mujoco.mjtGeom
    return {int(G.mjGEOM_BOX): "box", int(G.mjGEOM_CAPSULE): "capsule", int(G.mjGEOM_CYLINDER): "cylinder",
            int(G.mjGEOM_SPHERE): "sphere", int(G.mjGEOM_MESH): "mesh", int(G.mjGEOM_ELLIPSOID): "ellipsoid"}


def _quat_wxyz(R):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(R).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _ellipsoid_mesh(a, b, c, n_lat=8, n_lon=14):
    V = [[0.0, 0.0, c]]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            V.append([a * math.sin(th) * math.cos(ph), b * math.sin(th) * math.sin(ph), c * math.cos(th)])
    V.append([0.0, 0.0, -c])
    F = []
    for j in range(n_lon):
        F.append([0, 1 + j, 1 + (j + 1) % n_lon])
    for i in range(n_lat - 2):
        r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            F.append([r0 + j, r1 + j, r1 + j1]); F.append([r0 + j, r1 + j1, r0 + j1])
    last = len(V) - 1; r0 = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        F.append([r0 + j, last, r0 + (j + 1) % n_lon])
    return V, F


def _subtree(m, root):
    out = []
    for b in range(m.nbody):
        x = b
        while x > 0 and x != root:
            x = int(m.body_parentid[x])
        if x == root:
            out.append(b)
    return out


def export_scene(scene_xml, qpos=None, soft=(), table_geom="g_table"):
    """Static geoms (soft bodies' subtrees excluded) and the soft bodies as dynamic records."""
    import mujoco
    TN = _geom_type_names()
    m = mujoco.MjModel.from_xml_path(str(scene_xml)); d = mujoco.MjData(m)
    if qpos is not None:
        q = np.load(qpos) if isinstance(qpos, (str, Path)) else np.asarray(qpos, float)
        d.qpos[:len(q)] = q
    elif m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    soft_ids, soft_recs = {}, []
    for s in soft:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s["body"])
        if bid < 0:
            raise ValueError(f"soft_bodies: body {s['body']!r} is not in {Path(scene_xml).name}")
        for b in _subtree(m, bid):
            soft_ids[b] = bid
        Rb = d.xmat[bid].reshape(3, 3); pb = d.xpos[bid].copy()
        mass = float(m.body_subtreemass[bid])
        geoms, tcs, zs, fr = [], [], [], []
        for g in range(m.ngeom):
            if soft_ids.get(int(m.geom_bodyid[g])) != bid:
                continue
            t = TN.get(int(m.geom_type[g]))
            if t is None:
                continue
            Rg = d.geom_xmat[g].reshape(3, 3); pg = d.geom_xpos[g]
            coll = bool(m.geom_contype[g] or m.geom_conaffinity[g])
            rec = {"name": m.geom(g).name or f"g{g}", "type": t, "size": [float(x) for x in m.geom_size[g]],
                   "rgba": [float(x) for x in m.geom_rgba[g]], "pos": [float(x) for x in Rb.T @ (pg - pb)],
                   "quat": _quat_wxyz(Rb.T @ Rg), "collide": coll}
            if t == "ellipsoid":
                rec["vertices"], rec["faces"] = _ellipsoid_mesh(*rec["size"])
            if t == "mesh":
                mid = int(m.geom_dataid[g]); va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                rec["vertices"] = np.round(m.mesh_vert[va:va + vn], 5).tolist(); rec["faces"] = m.mesh_face[fa:fa + fn].tolist()
            geoms.append(rec)
            if coll:
                tcs.append(float(m.geom_solref[g][0])); zs.append(float(m.geom_solref[g][1])); fr.append(float(m.geom_friction[g][0]))
        if not any(r["collide"] for r in geoms):
            raise ValueError(f"soft_bodies: body {s['body']!r} has no colliding geom")
        tc = float(np.median(tcs)) if tcs and np.median(tcs) > 0 else 0.02
        zeta = float(np.median(zs)) if zs and np.median(zs) > 0 else 1.0
        w = 2 * math.pi / tc
        k = float(s.get("stiffness_N_per_m", mass * w * w))
        c = float(s.get("damping_Ns_per_m", 2 * zeta * mass * w))
        soft_recs.append({"body": s["body"], "pos": [float(x) for x in pb], "quat": _quat_wxyz(Rb), "mass_kg": mass,
                          "stiffness_N_per_m": k, "damping_Ns_per_m": c, "friction": float(s.get("friction", np.median(fr) if fr else 1.0)),
                          "solref_timeconst": tc, "solref_dampratio": zeta, "geoms": geoms})
    statics = []
    for g in range(m.ngeom):
        name = m.geom(g).name or f"g{g}"
        if m.geom_group[g] > 2 or "_collision_" in name or m.geom_rgba[g][3] < 0.05:
            continue
        t = TN.get(int(m.geom_type[g]))
        if t is None:
            continue
        b = int(m.geom_bodyid[g])
        if b in soft_ids:
            continue
        rgba = m.geom_rgba[g].copy()
        if m.geom_matid[g] >= 0:
            rgba = m.mat_rgba[int(m.geom_matid[g])].copy()
        R = d.geom_xmat[g].reshape(3, 3)
        rec = {"name": name, "body": m.body(b).name, "type": t, "size": [float(x) for x in m.geom_size[g]],
               "rgba": [float(x) for x in rgba], "pos": [float(x) for x in d.geom_xpos[g]], "quat": _quat_wxyz(R),
               "is_table": name == table_geom}
        if t == "ellipsoid":
            rec["vertices"], rec["faces"] = _ellipsoid_mesh(*rec["size"]); rec["type"] = "mesh"
        elif t == "mesh":
            mid = int(m.geom_dataid[g]); va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            rec["vertices"] = np.round(m.mesh_vert[va:va + vn], 5).tolist(); rec["faces"] = m.mesh_face[fa:fa + fn].tolist()
        statics.append(rec)
    return statics, soft_recs


_SCRIPT = r'''
import json, math, os
import numpy as np
EXPORT = os.environ["EXPORT"]; OUT = os.environ["OUT"]
S = json.load(open(EXPORT)); cam = S["camera"]; W, H = int(cam["width"]), int(cam["height"])
SETTLE_S = float(S.get("settle_s", 1.5)); PDT = 1.0 / float(S.get("physics_hz", 480)); TAG = S.get("tag", "isaac")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": W, "height": H})
import omni
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from scipy.spatial.transform import Rotation as SciRot
import imageio.v2 as imageio
world = World(physics_dt=PDT, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

def set_xform(prim, pos, quat, scale=(1, 1, 1)):
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*[float(v) for v in quat]))
    if tuple(scale) != (1, 1, 1):
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(s) for s in scale]))

mats = {}
def mat_for(rgba, rough=0.85, metal=0.0):
    key = (tuple(np.round(rgba[:3], 3)), rough, metal)
    if key not in mats:
        p = f"/World/mats/m{len(mats)}"; mat = UsdShade.Material.Define(stage, p); sh = UsdShade.Shader.Define(stage, p + "/shader")
        sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgba[:3]]))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface"); mats[key] = mat
    return mats[key]

def make_geom(path, r):
    t = r["type"]
    if t in ("mesh", "ellipsoid"):
        V = np.array(r["vertices"], float); F = np.array(r["faces"], int); g = UsdGeom.Mesh.Define(stage, path + "/mesh")
        g.CreatePointsAttr([Gf.Vec3f(*v) for v in V]); g.CreateFaceVertexCountsAttr([3] * len(F))
        g.CreateFaceVertexIndicesAttr([int(i) for f3 in F for i in f3]); g.CreateSubdivisionSchemeAttr("none")
    elif t == "box":
        g = UsdGeom.Cube.Define(stage, path + "/box"); g.CreateSizeAttr(2.0)
        UsdGeom.Xformable(g.GetPrim()).AddScaleOp().Set(Gf.Vec3d(*[float(s) for s in r["size"]]))
    elif t == "capsule":
        g = UsdGeom.Capsule.Define(stage, path + "/cap"); g.CreateAxisAttr("Z"); g.CreateRadiusAttr(float(r["size"][0])); g.CreateHeightAttr(2 * float(r["size"][1]))
    elif t == "cylinder":
        g = UsdGeom.Cylinder.Define(stage, path + "/cyl"); g.CreateAxisAttr("Z"); g.CreateRadiusAttr(float(r["size"][0])); g.CreateHeightAttr(2 * float(r["size"][1]))
    elif t == "sphere":
        g = UsdGeom.Sphere.Define(stage, path + "/sph"); g.CreateRadiusAttr(float(r["size"][0]))
    else:
        return None
    prim = g.GetPrim(); UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat_for(r["rgba"]))
    return prim

def collide(prim):
    UsdPhysics.CollisionAPI.Apply(prim)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim); pc.CreateContactOffsetAttr(0.002); pc.CreateRestOffsetAttr(0.0)

dome = UsdLux.DomeLight.Define(stage, "/World/dome"); dome.CreateIntensityAttr(float(S.get("dome", 350))); dome.CreateColorAttr(Gf.Vec3f(0.62, 0.62, 0.62))
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(float(S.get("sun", 600))); sun.CreateAngleAttr(5.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55, 10, 30))
tmat = UsdShade.Material.Define(stage, "/World/mats/table_phys"); tpm = UsdPhysics.MaterialAPI.Apply(tmat.GetPrim())
tpm.CreateStaticFrictionAttr(1.2); tpm.CreateDynamicFrictionAttr(1.0); tpm.CreateRestitutionAttr(0.0)
for k, r in enumerate(S["geoms"]):
    gp = f"/World/static/g{k:04d}"; xf = UsdGeom.Xform.Define(stage, gp); set_xform(xf.GetPrim(), r["pos"], r["quat"])
    prim = make_geom(gp, r)
    if prim is None:
        continue
    if r.get("is_table"):
        collide(prim); UsdShade.MaterialBindingAPI.Apply(prim).Bind(tmat, UsdShade.Tokens.weakerThanDescendants, "physics")
    elif not r["body"].startswith("robot") and r["body"] not in ("table", "", "world"):
        collide(prim)

bodies = {}
for si, sb in enumerate(S["soft"]):
    bp = f"/World/soft/b{si:02d}"; bx = UsdGeom.Xform.Define(stage, bp); set_xform(bx.GetPrim(), sb["pos"], sb["quat"])
    UsdPhysics.RigidBodyAPI.Apply(bx.GetPrim()); UsdPhysics.MassAPI.Apply(bx.GetPrim()).CreateMassAttr(float(sb["mass_kg"]))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(bx.GetPrim()); prb.CreateMaxDepenetrationVelocityAttr(0.05); prb.CreateSleepThresholdAttr(0.0)
    pm_ = UsdShade.Material.Define(stage, f"/World/mats/soft{si}"); um = UsdPhysics.MaterialAPI.Apply(pm_.GetPrim())
    um.CreateStaticFrictionAttr(float(sb["friction"])); um.CreateDynamicFrictionAttr(0.9 * float(sb["friction"])); um.CreateRestitutionAttr(0.0)
    xm = PhysxSchema.PhysxMaterialAPI.Apply(pm_.GetPrim())
    xm.CreateCompliantContactStiffnessAttr(float(sb["stiffness_N_per_m"])); xm.CreateCompliantContactDampingAttr(float(sb["damping_Ns_per_m"]))
    for gi, r in enumerate(sb["geoms"]):
        gp = f"{bp}/g{gi:02d}"; gx = UsdGeom.Xform.Define(stage, gp); set_xform(gx.GetPrim(), r["pos"], r["quat"])
        prim = make_geom(gp, r)
        if prim is not None and r["collide"]:
            collide(prim); UsdShade.MaterialBindingAPI.Apply(prim).Bind(pm_, UsdShade.Tokens.weakerThanDescendants, "physics")
    bodies[sb["body"]] = bx.GetPrim()

Rc = np.array(cam["R_cv"]); pc_ = np.array(cam["pos"]); qc = SciRot.from_matrix(Rc @ np.diag([1, -1, -1.0])).as_quat()
uc = UsdGeom.Camera.Define(stage, "/World/cam"); set_xform(uc.GetPrim(), pc_, [qc[3], qc[0], qc[1], qc[2]])
focal = 18.0; vap = 2 * focal * math.tan(math.radians(cam["fovy_deg"]) / 2)
uc.CreateFocalLengthAttr(focal); uc.CreateVerticalApertureAttr(vap); uc.CreateHorizontalApertureAttr(vap * W / H); uc.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))
camera = Camera(prim_path="/World/cam", resolution=(W, H))

def pose(prim):
    T = UsdGeom.XformCache().GetLocalToWorldTransform(prim); q = T.ExtractRotationQuat()
    return np.array([*T.ExtractTranslation()]), np.array([q.GetReal(), *q.GetImaginary()])

world.reset(); camera.initialize()
for _ in range(3):
    world.render()
P0 = {n: pose(p) for n, p in bodies.items()}
for _ in range(int(SETTLE_S / PDT)):
    world.step(render=False)
for _ in range(int(S.get("warm_frames", 20))):
    world.step(render=True)
P1 = {n: pose(p) for n, p in bodies.items()}
rgba = None
for _ in range(12):
    world.render(); rgba = camera.get_rgba()
    if rgba is not None and getattr(rgba, "size", 0) and np.asarray(rgba)[..., :3].std() > 2.0:
        break
img = np.asarray(rgba)[..., :3]
imageio.imwrite(f"{OUT}/{TAG}_sim.png", img)
stage.GetRootLayer().Export(f"{OUT}/{TAG}_scene.usda")
report = {}
for sb in S["soft"]:
    n = sb["body"]; (p0, q0), (p1, q1) = P0[n], P1[n]
    dot = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    dr = float(np.linalg.norm(p1 - p0))
    report[n] = {"start_pos": np.round(p0, 5).tolist(), "end_pos": np.round(p1, 5).tolist(), "mean_drift_m": dr, "max_drift_m": dr,
                 "rot_change_deg": float(np.degrees(2 * np.arccos(min(1.0, dot)))), "mass_kg": sb["mass_kg"],
                 "stiffness_N_per_m": sb["stiffness_N_per_m"], "damping_Ns_per_m": sb["damping_Ns_per_m"], "n_geoms": len(sb["geoms"])}
report["_"] = {"settle_s": SETTLE_S, "physics_dt": PDT, "n_static_geoms": len(S["geoms"])}
json.dump(report, open(f"{OUT}/{TAG}_report.json", "w"), indent=1)
print(f"ISAAC_DONE soft {len(bodies)} img std {float(img.std()):.1f}", flush=True)
app.close()
'''


def build_isaac_scene_soft_bodies(scene_xml, camera, soft_bodies, out_dir, qpos=None, settle_s=1.5, tag="isaac",
                                  isaac_python=None, isaac_home=None, timeout_s=1800.0, frame=None, dry_run=False):
    if not soft_bodies:
        raise ValueError("soft_bodies is empty: name at least one MuJoCo body to rebuild as a soft dynamic body")
    allowed = {"body", "stiffness_N_per_m", "damping_Ns_per_m", "friction"}
    for s in soft_bodies:
        if not isinstance(s, dict) or "body" not in s:
            raise ValueError("soft_bodies: each entry needs {'body': <MuJoCo body name>}")
        bad = set(s) - allowed
        if bad:
            raise ValueError(f"soft_bodies[{s['body']}]: unknown key(s) {sorted(bad)}; allowed {sorted(allowed)}")
    if float(settle_s) <= 0:
        raise ValueError("settle_s must be positive")
    if not Path(scene_xml).exists():
        raise ValueError(f"scene_xml: {scene_xml} does not exist")
    cam = json.loads(Path(camera).read_text()) if isinstance(camera, (str, Path)) else dict(camera)
    for k in ("width", "height", "fovy_deg", "pos", "R_cv"):
        if k not in cam:
            raise ValueError(f"camera: {k} missing (write camera.json with register_table_camera)")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    statics, soft = export_scene(scene_xml, qpos, soft_bodies)
    export = {"camera": {k: cam[k] for k in ("width", "height", "fovy_deg", "pos", "R_cv")}, "geoms": statics, "soft": soft,
              "settle_s": float(settle_s), "tag": tag, "scene_xml": str(scene_xml)}
    exp = out / "export.json"; exp.write_text(json.dumps(export))
    script = out / f"{tag}_soft_scene_script.py"; script.write_text(_SCRIPT)
    result = {"export_json": str(exp), "script": str(script), "n_static_geoms": len(statics),
              "soft": [{k: v for k, v in s.items() if k != "geoms"} | {"n_geoms": len(s["geoms"])} for s in soft]}
    if dry_run:
        return result
    py = isaac_python or os.environ.get("DEXAGENT_ISAAC_PYTHON")
    if not py or not Path(py).exists():
        raise FileNotFoundError("Isaac Sim's Python not found: pass isaac_python or set DEXAGENT_ISAAC_PYTHON")
    home = Path(isaac_home or os.environ.get("DEXAGENT_ISAAC_HOME") or (out / "isaac_home")); home.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home), OMNI_KIT_ACCEPT_EULA="YES", ACCEPT_EULA="Y", PYTHONUNBUFFERED="1", EXPORT=str(exp), OUT=str(out))
    env.setdefault("TMPDIR", str(out / "tmp")); Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    log = out / f"{tag}_isaac.log"
    with log.open("w") as fh:
        r = subprocess.run([str(py), str(script)], env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=float(timeout_s))
    text = log.read_text()
    if r.returncode != 0 or "ISAAC_DONE" not in text:
        raise RuntimeError(f"Isaac Sim run failed (returncode {r.returncode}); log tail:\n{text[-3000:]}")
    result.update({"sim_png": str(out / f"{tag}_sim.png"), "scene_usda": str(out / f"{tag}_scene.usda"),
                   "report_json": str(out / f"{tag}_report.json"), "log": str(log),
                   "report": json.loads((out / f"{tag}_report.json").read_text())})
    if frame is not None:
        import cv2
        a = cv2.imread(str(frame)); b = cv2.imread(result["sim_png"])
        if a is not None and b is not None:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
            ov = out / f"{tag}_overlay.png"; cv2.imwrite(str(ov), (0.5 * a + 0.5 * b).astype(np.uint8)); result["overlay_png"] = str(ov)
    return result


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if len(a) < 4:
        print(__doc__); return 2
    r = build_isaac_scene_soft_bodies(a[0], a[1], [{"body": a[2]}], a[3])
    print(json.dumps({k: v for k, v in r.items() if k != "soft"}, indent=1)); return 0


if __name__ == "__main__":
    sys.exit(main())
