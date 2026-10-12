"""Isaac Sim 5.0 (PhysX) scene from an export.json: static geoms of a MuJoCo scene, deformable cords as jointed
capsule chains (isaac_cord.build_cord, from each cord's resolved specification), settled under gravity and rendered
through the registered camera. Launched by skills/build_isaac_scene.py inside the Isaac Sim Python; its only project
import is isaac_cord, from this folder.

    EXPORT=export.json OUT=out_dir python isaac_scene_script.py
"""
import json
import math
import os
import sys

import numpy as np

EXPORT = os.environ["EXPORT"]
OUT = os.environ["OUT"]
os.makedirs(OUT, exist_ok=True)
S = json.load(open(EXPORT))
cam = S["camera"]
W, H = int(cam["width"]), int(cam["height"])
SEG_M = float(S.get("segment_m", 0.008))
SETTLE_S = float(S.get("settle_s", 1.5))
PDT = 1.0 / float(S.get("physics_hz", 480))
TAG = S.get("tag", "isaac")

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True, "width": W, "height": H})
import omni  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from scipy.spatial.transform import Rotation as SciRot  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

world = World(physics_dt=PDT, rendering_dt=1.0 / 30.0, stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()


def set_xform(prim, pos, quat):
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if not ops:
        ops = [xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble), xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)]
    ops[0].Set(Gf.Vec3d(*[float(v) for v in pos]))
    ops[1].Set(Gf.Quatd(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))
    if len(ops) > 2:
        ops[2].Set(Gf.Vec3d(1, 1, 1))


def make_material(path, rgb, roughness=0.7, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


mats = {}


def mat_for(rgba, rough=0.85, metal=0.0):
    key = (tuple(np.round(rgba[:3], 3)), rough, metal)
    if key not in mats:
        mats[key] = make_material(f"/World/mats/m{len(mats)}", rgba[:3], roughness=rough, metallic=metal)
    return mats[key]


dome = UsdLux.DomeLight.Define(stage, "/World/dome"); dome.CreateIntensityAttr(float(S.get("dome", 350))); dome.CreateColorAttr(Gf.Vec3f(0.62, 0.62, 0.62))
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(float(S.get("sun", 600))); sun.CreateAngleAttr(5.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55, 10, 30))

# the table's contact with the cords: the first cord's specification (build_isaac_scene resolves them all)
TP = (S["deformables"][0].get("physics") or {}) if S.get("deformables") else {}
TP = {"table_static_friction": 1.2, "table_dynamic_friction": 1.0, "table_friction_combine": None, "table_contact_offset_m": 0.002, **TP}

# ---- static geoms (rig frame == stage frame)
obj_prims = {}
for k, r in enumerate(S["geoms"]):
    gp = f"/World/static/g{k:03d}"
    xf = UsdGeom.Xform.Define(stage, gp)
    set_xform(xf.GetPrim(), r["pos"], r["quat"])
    t = r["type"]
    if t == "mesh":
        V = np.array(r["vertices"], float); F = np.array(r["faces"], int)
        mesh = UsdGeom.Mesh.Define(stage, gp + "/mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in V]); mesh.CreateFaceVertexCountsAttr([3] * len(F)); mesh.CreateFaceVertexIndicesAttr([int(i) for f3 in F for i in f3])
        mesh.CreateSubdivisionSchemeAttr("none"); prim = mesh.GetPrim()
        bind(prim, mat_for(r["rgba"], 0.6, 0.5 if (r["rgba"][0] > 0.6 and r["body"].startswith("robot_")) else 0.0))
    elif t == "box":
        g = UsdGeom.Cube.Define(stage, gp + "/box"); g.CreateSizeAttr(2.0)
        UsdGeom.Xformable(g.GetPrim()).AddScaleOp().Set(Gf.Vec3d(*[float(s) for s in r["size"]])); prim = g.GetPrim()
        bind(prim, mat_for(r["rgba"], 0.97 if r.get("is_table") else 0.85))
    elif t == "capsule":
        g = UsdGeom.Capsule.Define(stage, gp + "/cap"); g.CreateAxisAttr("Z"); g.CreateRadiusAttr(float(r["size"][0])); g.CreateHeightAttr(2 * float(r["size"][1])); prim = g.GetPrim(); bind(prim, mat_for(r["rgba"]))
    elif t == "cylinder":
        g = UsdGeom.Cylinder.Define(stage, gp + "/cyl"); g.CreateAxisAttr("Z"); g.CreateRadiusAttr(float(r["size"][0])); g.CreateHeightAttr(2 * float(r["size"][1])); prim = g.GetPrim(); bind(prim, mat_for(r["rgba"]))
    elif t == "sphere":
        g = UsdGeom.Sphere.Define(stage, gp + "/sph"); g.CreateRadiusAttr(float(r["size"][0])); prim = g.GetPrim(); bind(prim, mat_for(r["rgba"]))
    else:
        continue
    obj_prims.setdefault(r["body"], []).append(prim)
    if r.get("is_table"):
        UsdPhysics.CollisionAPI.Apply(prim)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim); pc.CreateContactOffsetAttr(float(TP["table_contact_offset_m"])); pc.CreateRestOffsetAttr(0.0)
        mp = UsdShade.Material.Define(stage, "/World/mats/table_phys"); pm = UsdPhysics.MaterialAPI.Apply(mp.GetPrim())
        pm.CreateStaticFrictionAttr(float(TP["table_static_friction"])); pm.CreateDynamicFrictionAttr(float(TP["table_dynamic_friction"]))
        pm.CreateRestitutionAttr(0.0)
        if TP.get("table_friction_combine"):
            PhysxSchema.PhysxMaterialAPI.Apply(mp.GetPrim()).CreateFrictionCombineModeAttr(TP["table_friction_combine"])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mp, UsdShade.Tokens.weakerThanDescendants, "physics")
# rigid objects the cords may touch get collision (convex hulls)
for bname, prims in obj_prims.items():
    if bname.startswith("robot") or bname in ("table", "", "world"):
        continue
    for prim in prims:
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")


# ---- deformables: capsule chains on D6 joints, built from each cord's resolved specification (isaac_cord.cord_spec:
# geometry and every PhysX coefficient), the same builder Stage 3's Isaac episodes and Stage 4's replays use
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaac_cord as IC  # noqa: E402

USD = {"Usd": Usd, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics, "UsdShade": UsdShade, "PhysxSchema": PhysxSchema, "Gf": Gf, "Sdf": Sdf}


def d6(path, b0, b1, p0, p1, q0, q1, stiff=0.004, damp=0.002, fric=0.0015):
    """A strand's first link to the junction body."""
    jp = UsdPhysics.Joint.Define(stage, path); jprim = jp.GetPrim()
    jp.CreateBody0Rel().SetTargets([b0.GetPath()]); jp.CreateBody1Rel().SetTargets([b1.GetPath()])
    jp.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in p0])); jp.CreateLocalPos1Attr(Gf.Vec3f(*[float(v) for v in p1]))
    jp.CreateLocalRot0Attr(Gf.Quatf(*[float(v) for v in q0])); jp.CreateLocalRot1Attr(Gf.Quatf(*[float(v) for v in q1]))
    for ax in ("transX", "transY", "transZ"):
        lim = UsdPhysics.LimitAPI.Apply(jprim, ax); lim.CreateLowAttr(1.0); lim.CreateHighAttr(-1.0)
    for ax in ("rotX", "rotY", "rotZ"):
        drv = UsdPhysics.DriveAPI.Apply(jprim, ax); drv.CreateTypeAttr("force"); drv.CreateStiffnessAttr(stiff); drv.CreateDampingAttr(damp); drv.CreateTargetPositionAttr(0.0)
    PhysxSchema.PhysxJointAPI.Apply(jprim)
    for ax in ("rotX", "rotY", "rotZ"):
        jprim.AddAppliedSchema(f"PhysxJointAxisAPI:{ax}")
        for attr, val in ((f"physxJointAxis:{ax}:staticFrictionEffort", fric), (f"physxJointAxis:{ax}:dynamicFrictionEffort", fric * 0.8), (f"physxJointAxis:{ax}:maxJointVelocity", 800.0)):
            jprim.CreateAttribute(attr, Sdf.ValueTypeNames.Float).Set(float(val))


SPECS = [c if c.get("schema") == IC.SCHEMA else IC.cord_spec(c) for c in S["deformables"]]
UsdGeom.Xform.Define(stage, "/World/cords")
all_links, report, chains = [], {}, {}
junction_prim = None
if S.get("junction"):
    # strands tied to one junction body are ONE articulation: the root is /World/cords, the strands are authored under
    # it without roots of their own; self-collision is off (the strands meet at the junction, where it explodes the chain)
    root = stage.GetPrimAtPath("/World/cords"); UsdPhysics.ArticulationRootAPI.Apply(root)
    PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(False)
    J = S["junction"]; kc = np.array(J["center"], float); kr = float(J["radius"]); kc[2] = max(kc[2], kr)
    kp = UsdGeom.Xform.Define(stage, "/World/cords/junction"); set_xform(kp.GetPrim(), kc, [1, 0, 0, 0])
    sph = UsdGeom.Sphere.Define(stage, "/World/cords/junction/geom"); sph.CreateRadiusAttr(kr); bind(sph.GetPrim(), make_material("/World/mats/junction", J.get("rgb") or [0.7, 0.2, 0.2], 0.95))
    UsdPhysics.RigidBodyAPI.Apply(kp.GetPrim()); UsdPhysics.MassAPI.Apply(kp.GetPrim()).CreateMassAttr(0.003)
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(kp.GetPrim()); prb.CreateLinearDampingAttr(3.0); prb.CreateAngularDampingAttr(4.0); prb.CreateMaxDepenetrationVelocityAttr(0.05)
    junction_prim = kp.GetPrim()

for spec in SPECS:
    name = spec["name"]
    if junction_prim is not None:
        kc = np.array(S["junction"]["center"])
        P = np.asarray(spec["points"], float)
        if np.linalg.norm(P[-1, :2] - kc[:2]) < np.linalg.norm(P[0, :2] - kc[:2]):
            spec = dict(spec, points=P[::-1].tolist())       # a strand starts at the junction
    built = IC.build_cord(stage, USD, f"/World/cords/{name}", spec, lambda rgb, r, m: mat_for(list(rgb) + [1.0], r, m),
                          articulation=junction_prim is None)
    links = built["links"]
    if junction_prim is not None:
        jc = np.array(S["junction"]["center"]); jc[2] = max(jc[2], float(S["junction"]["radius"]))
        start = built["centres"][0] - 0.5 * spec["segment_m"] * IC.quat_to_mat(built["quats"][0])[:, 0]
        d6(f"/World/cords/{name}/joints/junction", junction_prim, links[0], start - jc, (-spec["segment_m"] / 2, 0, 0), built["quats"][0],
           (1, 0, 0, 0), stiff=0.01, damp=0.004, fric=0.003)
    all_links += links; chains[name] = links
    report[name] = {"n_links": spec["n_links"], "segment_m": spec["segment_m"], "length_m": spec["length_m"], "radius_m": spec["radius"],
                    "preset": spec["preset"]}
    print("cord", name, "links", spec["n_links"], "seg", round(spec["segment_m"] * 1000, 1), "mm", "preset", spec["preset"], flush=True)

# ---- camera (OpenCV axes in the rig frame -> USD camera: x right, y up, looks along -z)
Rc = np.array(cam["R_cv"]); pc_ = np.array(cam["pos"]); Rusd = Rc @ np.diag([1, -1, -1.0]); qc = SciRot.from_matrix(Rusd).as_quat()
uc = UsdGeom.Camera.Define(stage, "/World/cam"); set_xform(uc.GetPrim(), pc_, [qc[3], qc[0], qc[1], qc[2]])
focal = 18.0; vap = 2 * focal * math.tan(math.radians(cam["fovy_deg"]) / 2)
uc.CreateFocalLengthAttr(focal); uc.CreateVerticalApertureAttr(vap); uc.CreateHorizontalApertureAttr(vap * W / H); uc.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))
camera = Camera(prim_path="/World/cam", resolution=(W, H))


def centreline(links):
    xc = UsdGeom.XformCache()
    return np.array([[*xc.GetLocalToWorldTransform(l).ExtractTranslation()] for l in links])


world.reset(); camera.initialize()
for _ in range(3):
    world.render()
C0 = {n: centreline(l) for n, l in chains.items()}
for k in range(int(SETTLE_S / PDT)):
    world.step(render=False)
for _ in range(int(S.get("warm_frames", 20))):
    world.step(render=True)
C1 = {n: centreline(l) for n, l in chains.items()}
Q1 = {n: IC.link_poses_usd(USD, l)[1] for n, l in chains.items()}
rgba = None
for _ in range(12):
    world.render(); rgba = camera.get_rgba()
    if rgba is not None and getattr(rgba, "size", 0) and np.asarray(rgba)[..., :3].std() > 2.0:
        break
img = np.asarray(rgba)[..., :3]
imageio.imwrite(f"{OUT}/{TAG}_sim.png", img)
stage.GetRootLayer().Export(f"{OUT}/{TAG}_scene.usda")
for n in chains:
    np.save(f"{OUT}/{TAG}_{n}_centreline_settled.npy", C1[n])
    dr = np.linalg.norm(C1[n] - C0[n], axis=1)
    finite = bool(np.isfinite(C1[n]).all())
    report[n]["finite"] = finite
    report[n]["mean_drift_m"] = float(dr.mean()) if finite else None
    report[n]["max_drift_m"] = float(dr.max()) if finite else None
    # neighbouring link centres are at most one link length apart however the cord bends (SEG cos(bend / 2)): more
    # than that is a torn chain, the signature of an articulation that blew up
    gaps = np.linalg.norm(np.diff(C1[n], axis=0), axis=1)
    report[n]["link_stretch"] = float(gaps.max() / report[n]["segment_m"] - 1.0) if finite and len(gaps) else None
    report[n]["min_z_m"] = float(C1[n][:, 2].min()) if finite else None
    report[n]["settled_centreline"] = np.round(C1[n], 5).tolist() if finite else None
    report[n]["settled_quats"] = np.round(Q1[n], 6).tolist() if finite and np.isfinite(Q1[n]).all() else None
report["_"] = {"settle_s": SETTLE_S, "physics_dt": PDT, "segment_m": SEG_M, "n_links_total": len(all_links)}
json.dump(report, open(f"{OUT}/{TAG}_report.json", "w"), indent=1)
print(f"ISAAC_DONE links {len(all_links)} img std {float(img.std()):.1f}", flush=True)
app.close()
