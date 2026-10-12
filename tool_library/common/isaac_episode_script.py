"""A Stage 3 episode's cords in Isaac Sim (PhysX): the robot and the rigid objects move as MuJoCo executed them, the
cords are simulated. Launched by skills/execute_isaac_deformables.py inside Isaac Sim's Python (which also has
mujoco); its only project import is isaac_cord, from this folder.

    JOB=job.json python isaac_episode_script.py

Per video frame the MuJoCo trajectory row is put through MuJoCo forward kinematics: every visible geom is posed for the
render, every rigid object (a free body outside the robot) is a kinematic PhysX collider, and each hand drives a
kinematic anchor. The poses are interpolated over the physics substeps of the frame (a kinematic body that jumps once
per video frame hands PhysX the whole frame's displacement in one substep, a velocity spike that kicks whatever it
touches). A grasp welds the anchor to the one link nearest the hand's pinch point, with the cord's compliant weld
gains, when that link is within the capture distance (a pinch that closes on air holds nothing); the weld lets go
where the plan opens the hand, or earlier when the link is pulled further than the slip distance from where the hand
holds it. Records every link's pose per frame, the welds, the strains and the link speeds, and renders the frames
through the registered camera when asked.
"""
import json
import math
import os
import sys
import time

import numpy as np

JOB = json.load(open(os.environ["JOB"]))
OUT = JOB["out_dir"]
os.makedirs(OUT, exist_ok=True)
RENDER = JOB.get("render") or None
W, H = (int(RENDER["width"]), int(RENDER["height"])) if RENDER else (64, 48)
FPS = float(JOB["fps"])
SUB = int(JOB.get("substeps", 16))
PDT = 1.0 / (FPS * SUB)

from isaacsim import SimulationApp  # noqa: E402
app = SimulationApp({"headless": True, "width": W, "height": H})
import omni  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
import mujoco  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaac_cord as IC  # noqa: E402

USD = {"Usd": Usd, "UsdGeom": UsdGeom, "UsdPhysics": UsdPhysics, "UsdShade": UsdShade, "PhysxSchema": PhysxSchema, "Gf": Gf, "Sdf": Sdf}
world = World(physics_dt=PDT, rendering_dt=1.0 / FPS, stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

m = mujoco.MjModel.from_xml_path(JOB["scene_xml"])
d = mujoco.MjData(m)
Q = np.load(JOB["trajectory"], allow_pickle=True)["qpos"].astype(float)
if Q.shape[1] != m.nq:
    raise SystemExit(f"trajectory has {Q.shape[1]} qpos columns, the scene nq={m.nq}")
T = len(Q) if JOB.get("frames") is None else min(len(Q), int(JOB["frames"]))
PREFIX = JOB.get("robot_prefix", "robot_")
SPECS = [IC.cord_spec(c) if c.get("schema") != IC.SCHEMA else c for c in JOB["cords"]]
P0 = SPECS[0]["physics"]

_xops = {}


def set_xf(prim, pos, quat):
    """Write a prim's pose into translate/orient ops authored once (re-authoring ops every frame is ten times slower)."""
    key = prim.GetPath().pathString
    ops = _xops.get(key)
    if ops is None:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        ops = _xops[key] = (xf.AddTranslateOp(), xf.AddOrientOp())
    ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    ops[1].Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))


_mats = {}


def mat_for(rgb, rough=0.7, metal=0.0):
    key = (tuple(np.round(rgb[:3], 3)), round(float(rough), 2), round(float(metal), 2))
    if key not in _mats:
        path = f"/World/mats/m{len(_mats)}"
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb[:3]]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        _mats[key] = mat
    return _mats[key]


def geom_prim(path, g):
    """A USD prim of MuJoCo geom g in its own frame (posed through its parent Xform)."""
    t, size = int(m.geom_type[g]), m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        va, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, nf = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
        mesh.CreatePointsAttr([Gf.Vec3f(*[float(x) for x in v]) for v in m.mesh_vert[va:va + nv]])
        mesh.CreateFaceVertexCountsAttr([3] * nf)
        mesh.CreateFaceVertexIndicesAttr([int(i) for f3 in m.mesh_face[fa:fa + nf] for i in f3])
        mesh.CreateSubdivisionSchemeAttr("none")
        return mesh.GetPrim()
    if t in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_PLANE):
        b = UsdGeom.Cube.Define(stage, path + "/box"); b.CreateSizeAttr(2.0)
        sc = [float(s) for s in size] if t == mujoco.mjtGeom.mjGEOM_BOX else [3.0, 3.0, 0.001]
        UsdGeom.Xformable(b.GetPrim()).AddScaleOp().Set(Gf.Vec3d(*sc))
        return b.GetPrim()
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        c = UsdGeom.Capsule.Define(stage, path + "/cap"); c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(float(size[0])); c.CreateHeightAttr(2 * float(size[1]))
        return c.GetPrim()
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        c = UsdGeom.Cylinder.Define(stage, path + "/cyl"); c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(float(size[0])); c.CreateHeightAttr(2 * float(size[1]))
        return c.GetPrim()
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        s_ = UsdGeom.Sphere.Define(stage, path + "/sph"); s_.CreateRadiusAttr(float(size[0]))
        return s_.GetPrim()
    return None


def quat_of(R):
    return IC.mat_to_quat(np.asarray(R).reshape(3, 3))


def body_is_free(b):
    """A body outside the robot that moves (a free joint at it or above it)."""
    while b > 0:
        if m.body_jntnum[b] > 0 and int(m.jnt_type[m.body_jntadr[b]]) == mujoco.mjtJoint.mjJNT_FREE:
            return True
        b = int(m.body_parentid[b])
    return False


# ------------------------------------------------------------------------------------------------- the stage
phys = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
UsdPhysics.Scene(phys).CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
UsdPhysics.Scene(phys).CreateGravityMagnitudeAttr(float(-m.opt.gravity[2]) or 9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(phys)
px.CreateSolverTypeAttr("TGS"); px.CreateEnableGPUDynamicsAttr(False); px.CreateBroadphaseTypeAttr("MBP")
dome = UsdLux.DomeLight.Define(stage, "/World/dome"); dome.CreateIntensityAttr(float((RENDER or {}).get("dome", 320)))
sun = UsdLux.DistantLight.Define(stage, "/World/sun"); sun.CreateIntensityAttr(float((RENDER or {}).get("sun", 700))); sun.CreateAngleAttr(5.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55, 10, 30))

mujoco.mj_forward(m, d)
d.qpos[:] = Q[0]
mujoco.mj_forward(m, d)
cord_names = {s["name"] for s in SPECS}
visual, colliders = [], []                 # (geom, prim) posed per frame / kinematic object colliders
TABLE = JOB.get("table_geom", "g_table")
for g in range(m.ngeom):
    bname = m.body(int(m.geom_bodyid[g])).name or ""
    gname = m.geom(g).name or f"g{g}"
    if any(bname.startswith(n + "B_") for n in cord_names):
        continue
    rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
    is_table = gname == TABLE
    moving = bname.startswith(PREFIX) or body_is_free(int(m.geom_bodyid[g]))
    if is_table:
        root = UsdGeom.Xform.Define(stage, "/World/table").GetPrim()
        set_xf(root, d.geom_xpos[g], quat_of(d.geom_xmat[g]))
        prim = geom_prim("/World/table", g)
        UsdPhysics.CollisionAPI.Apply(prim)
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr(float(P0["table_contact_offset_m"])); pc.CreateRestOffsetAttr(0.0)
        tm = UsdShade.Material.Define(stage, "/World/mats/table_phys")
        pm = UsdPhysics.MaterialAPI.Apply(tm.GetPrim())
        pm.CreateStaticFrictionAttr(float(P0["table_static_friction"])); pm.CreateDynamicFrictionAttr(float(P0["table_dynamic_friction"]))
        pm.CreateRestitutionAttr(0.0)
        if P0.get("table_friction_combine"):
            PhysxSchema.PhysxMaterialAPI.Apply(tm.GetPrim()).CreateFrictionCombineModeAttr(P0["table_friction_combine"])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(tm, UsdShade.Tokens.weakerThanDescendants, "physics")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat_for(rgba, 0.95))
        continue
    if int(m.geom_group[g]) > 2 or rgba[3] < 0.05:
        if not (moving and not bname.startswith(PREFIX) and m.geom_contype[g]):
            continue
    gp = f"/World/{'robot' if bname.startswith(PREFIX) else 'objects'}/g{g:04d}"
    xf = UsdGeom.Xform.Define(stage, gp).GetPrim()
    set_xf(xf, d.geom_xpos[g], quat_of(d.geom_xmat[g]))
    prim = geom_prim(gp, g)
    if prim is None:
        continue
    spec_ = float(m.mat_specular[m.geom_matid[g]]) if m.geom_matid[g] >= 0 else 0.2
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat_for(rgba, float(np.clip(1 - spec_, 0.25, 0.95))))
    if int(m.geom_group[g]) > 2 or rgba[3] < 0.05:
        UsdGeom.Imageable(prim).MakeInvisible()
    if moving:
        visual.append((g, xf))
    # rigid objects (and, when asked, the robot's arm links) push the cords: kinematic convex hulls
    collide = (not bname.startswith(PREFIX) and moving and m.geom_contype[g]) or \
              (bname in set(JOB.get("robot_collider_bodies") or []) and m.geom_contype[g])
    if collide:
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
        rb = UsdPhysics.RigidBodyAPI.Apply(xf); rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(xf).CreateMassAttr(1.0)
        colliders.append((g, xf))
print(f"[scene] {len(visual)} moving geoms, {len(colliders)} kinematic colliders", flush=True)

cords = []
for s in SPECS:
    c = IC.build_cord(stage, USD, f"/World/cords/{s['name']}", s, mat_for, link_poses=s.get("link_poses"))
    c["spec"] = s
    cords.append(c)
    print(f"[cord] {s['name']}: {s['n_links']} links of {s['segment_m'] * 1000:.1f} mm, preset {s['preset']}", flush=True)
by_name = {c["spec"]["name"]: c for c in cords}

HANDS = JOB["hands"]                       # side -> {anchor_body, pinch_sites}
grip = IC.GraspRig(stage, USD)
for side in HANDS:
    grip.add_anchor(side)


def palm(side):
    b = m.body(HANDS[side]["anchor_body"]).id
    return d.xpos[b].copy(), quat_of(d.xmat[b])


def pinch(side):
    return np.mean([d.site_xpos[m.site(s).id] for s in HANDS[side]["pinch_sites"]], axis=0)


# ---- the smooth visual cord: a finely resampled tube along the simulated centreline (the 27 mm physics capsules
# read as beads on camera); render only
VIS = []
if RENDER:
    k_vis = int(RENDER.get("tube_per_link", 4))
    for c in cords:
        s = c["spec"]
        for lk in c["links"]:
            for ch in Usd.PrimRange(lk):
                if ch.GetName() == "geom":
                    UsdGeom.Imageable(ch).MakeInvisible()
        n_v = k_vis * s["n_links"]
        seg_v = s["length_m"] / n_v
        ca, cb = IC.cap_links(s)
        n_cap_a, n_cap_b = ca * k_vis, cb * k_vis
        prims = []
        for i in range(n_v):
            xf = UsdGeom.Xform.Define(stage, f"/World/cord_vis/{s['name']}/v{i:04d}")
            cap = UsdGeom.Capsule.Define(stage, f"/World/cord_vis/{s['name']}/v{i:04d}/geom")
            cap.CreateAxisAttr("X"); cap.CreateHeightAttr(seg_v); cap.CreateRadiusAttr(s["radius"])
            gold = i < n_cap_a or i >= n_v - n_cap_b
            UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(
                mat_for(s.get("cap_rgb") or s["rgb"], 0.35, 0.85) if gold and s.get("cap_rgb") else mat_for(s["rgb"], 0.95))
            if not s.get("cast_shadows", True):
                cap.GetPrim().CreateAttribute("primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)
            prims.append(xf.GetPrim())
        VIS.append((c, prims))


def draw_cords():
    for c, prims in VIS:
        C, _ = IC.link_poses_usd(USD, c["links"])
        h = 0.5 * c["spec"]["segment_m"]
        t0 = (C[0] - C[1]) / max(np.linalg.norm(C[0] - C[1]), 1e-9)
        t1 = (C[-1] - C[-2]) / max(np.linalg.norm(C[-1] - C[-2]), 1e-9)
        P = np.vstack([C[0] + h * t0, C, C[-1] + h * t1])
        sarc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        want = np.linspace(0.0, sarc[-1], len(prims) + 1)
        path = np.stack([np.interp(want, sarc, P[:, i]) for i in range(3)], 1)
        cen, tang = 0.5 * (path[:-1] + path[1:]), np.diff(path, axis=0)
        for i, prim in enumerate(prims):
            set_xf(prim, cen[i], IC.quat_x_to(tang[i]))


camera = None
if RENDER:
    cam = RENDER["camera"]
    Rc = np.asarray(cam["R_cv"], float); pc_ = np.asarray(cam["pos"], float)
    qc = IC.mat_to_quat(Rc @ np.diag([1.0, -1.0, -1.0]))      # OpenCV axes -> USD camera (x right, y up, looks along -z)
    uc = UsdGeom.Camera.Define(stage, "/World/cam"); set_xf(uc.GetPrim(), pc_, qc)
    fovy = float(cam.get("fovy_deg") or math.degrees(2 * math.atan(0.5 * float(cam["height"]) / float(cam["fy"]))))
    focal = 18.0; vap = 2 * focal * math.tan(math.radians(fovy) / 2)
    uc.CreateFocalLengthAttr(focal); uc.CreateVerticalApertureAttr(vap); uc.CreateHorizontalApertureAttr(vap * W / H)
    uc.CreateClippingRangeAttr(Gf.Vec2f(0.02, 20.0))
    from isaacsim.sensors.camera import Camera  # noqa: E402
    camera = Camera(prim_path="/World/cam", resolution=(W, H))
    import imageio.v2 as imageio  # noqa: E402
    os.makedirs(f"{OUT}/frames", exist_ok=True)

world.reset()
if camera is not None:
    camera.initialize()
    for g, xf in visual:
        set_xf(xf, d.geom_xpos[g], quat_of(d.geom_xmat[g]))
    draw_cords()
    for _ in range(int(RENDER.get("warm_frames", 24))):
        world.step(render=True)

# ------------------------------------------------------------------------------------------------- the episode
GRASPS = sorted(JOB.get("grasps") or [], key=lambda g: int(g["weld_frame"]))
EVENTS = JOB.get("events") or []
for gr in GRASPS:
    gr.update(state="planned", weld=None, link=None, released_at=None, off=None)
log_pose = {c["spec"]["name"]: (np.zeros((T, c["spec"]["n_links"], 3), np.float32), np.zeros((T, c["spec"]["n_links"], 4), np.float32)) for c in cords}
sides = list(HANDS)
log_anchor = np.zeros((T, len(sides), 3), np.float32)
log_held = -np.ones((T, len(sides)), np.int16)
log_strain = np.zeros((T, len(sides)), np.float32)
speeds = np.zeros(T, np.float32)
events_log = []
parked, calm_until, active = {s: None for s in sides}, -1, {s: None for s in sides}
anchor_now = {}


def note(k, msg):
    events_log.append({"frame": int(k), "t": round(k / FPS, 3), "msg": msg})
    print(f"[event {k:5d}] {msg}", flush=True)


def poses_at(k):
    d.qpos[:] = Q[k]
    mujoco.mj_forward(m, d)
    return ({s: palm(s) for s in sides}, {g: (d.geom_xpos[g].copy(), quat_of(d.geom_xmat[g])) for g, _ in colliders})


def lerp_pose(a, b, u):
    p = (1 - u) * a[0] + u * b[0]
    q0, q1 = a[1], b[1] if float(np.dot(a[1], b[1])) >= 0 else -b[1]
    q = (1 - u) * q0 + u * q1
    return p, q / max(np.linalg.norm(q), 1e-12)


def set_kinematics(pa, pb, ga, gb, u):
    for s in sides:
        p, q = parked[s] if parked[s] is not None else lerp_pose(pa[s], pb[s], u)
        grip.move_anchor(s, p, q)
        anchor_now[s] = (p, q)
    for g, xf in colliders:
        p, q = lerp_pose(ga[g], gb[g], u)
        set_xf(xf, p, q)


pa, ga = poses_at(0)
set_kinematics(pa, pa, ga, ga, 0.0)
for _ in range(SUB):
    world.step(render=False)
prev = {n: IC.link_poses_usd(USD, c["links"])[0] for n, c in by_name.items()}
t_start = time.time()
for k in range(T):
    pb, gb = poses_at(k)
    if k > 0:
        for j in range(1, SUB + 1):
            set_kinematics(pa, pb, ga, gb, j / SUB)
            world.step(render=False)
    pa, ga = pb, gb
    for ev in EVENTS:
        if int(ev["frame"]) == k and ev.get("bend_stiffness") is not None:
            for a in by_name[ev["cord"]]["drive_stiffness_attrs"]:
                a.Set(float(ev["bend_stiffness"]))
            note(k, f"{ev['cord']}: bend stiffness -> {ev['bend_stiffness']}")
    live = {n: IC.link_poses_usd(USD, c["links"]) for n, c in by_name.items()}
    released_now = False
    for gr in GRASPS:
        s, c = gr["side"], by_name[gr["cord"]]
        gp = c["spec"]["grasp"]
        C = live[gr["cord"]][0]
        if gr["state"] == "planned" and k == int(gr["weld_frame"]):
            pt = pinch(s)
            dist = np.linalg.norm(C - pt, axis=1)
            j = int(gr["link"]) if gr.get("link") is not None and dist[int(gr["link"])] <= float(gr.get("capture_m", gp["capture_m"])) \
                else int(np.argmin(dist))
            if dist[j] > float(gr.get("capture_m", gp["capture_m"])):
                gr["state"] = "missed"
                note(k, f"{s}: closes {dist[j] * 1000:.0f} mm from the nearest link of {gr['cord']}: nothing held")
                continue
            gr["weld"] = grip.weld(s, c["links"][j], gp, float(gr.get("stiff", 1.0)))
            gr["link"], gr["state"] = j, "held"
            ap, aq = anchor_now[s]
            gr["off"] = IC.quat_to_mat(aq).T @ (C[j] - ap)
            active[s] = gr
            note(k, f"{s}: holds link {j} of {gr['cord']} ({dist[j] * 1000:.0f} mm from the pinch)")
        if gr["state"] == "held":
            ap, aq = anchor_now[s]
            strain = float(np.linalg.norm(C[gr["link"]] - (ap + IC.quat_to_mat(aq) @ gr["off"])))
            log_strain[k, sides.index(s)] = strain
            slipped = k > int(gr["weld_frame"]) + 6 and strain > float(gr.get("slip_m", gp["slip_m"]))
            if k >= int(gr["release_frame"]) or slipped:
                grip.release(gr["weld"], k, int(gp["fade_frames"]))
                gr["state"], gr["released_at"] = ("slipped" if slipped else "released"), k
                parked[s] = anchor_now[s]
                released_now = True
                note(k, f"{s}: lets go of link {gr['link']} of {gr['cord']}" +
                     (f" (pulled {strain * 1000:.0f} mm out of the hand)" if slipped else ""))
        if gr["state"] in ("released", "slipped") and k >= gr["released_at"] + int(gp["fade_frames"]) + 1 and active[s] is gr:
            parked[s], active[s] = None, None
    if released_now:
        for c in cords:
            for prb in c["body_apis"]:
                prb.GetLinearDampingAttr().Set(float(c["spec"]["grasp"]["calm_damping"][0]))
                prb.GetAngularDampingAttr().Set(float(c["spec"]["grasp"]["calm_damping"][1]))
        calm_until = k + int(SPECS[0]["grasp"]["calm_frames"])
    if calm_until >= 0 and k >= calm_until:
        for c in cords:
            for prb in c["body_apis"]:
                prb.GetLinearDampingAttr().Set(float(c["spec"]["physics"]["linear_damping"]))
                prb.GetAngularDampingAttr().Set(float(c["spec"]["physics"]["angular_damping"]))
        calm_until = -1
    grip.update(k)
    vmax = 0.0
    for n, c in by_name.items():
        C, Qc = IC.link_poses_usd(USD, c["links"])
        log_pose[n][0][k], log_pose[n][1][k] = C, Qc
        if k > 0:
            vmax = max(vmax, float(np.linalg.norm(C - prev[n], axis=1).max() * FPS))
        prev[n] = C
    speeds[k] = vmax
    for i, s in enumerate(sides):
        log_anchor[k, i] = anchor_now[s][0]
        log_held[k, i] = active[s]["link"] if active[s] is not None and active[s]["state"] == "held" else -1
    if camera is not None and (k % int(RENDER.get("every", 1)) == 0):
        for g, xf in visual:
            set_xf(xf, d.geom_xpos[g], quat_of(d.geom_xmat[g]))
        draw_cords()
        img = None
        for _ in range(8):
            world.render(); img = camera.get_rgba()
            if img is not None and getattr(img, "size", 0) and np.asarray(img)[..., :3].std() > 3.0:
                break
        imageio.imwrite(f"{OUT}/frames/f{k:05d}.png", np.asarray(img)[..., :3])
    if k % 100 == 0:
        print(f"[{k:5d}/{T}] peak link speed {vmax:5.2f} m/s  held {log_held[k].tolist()}  {time.time() - t_start:5.0f}s", flush=True)

rec = {"fps": np.float64(FPS), "substeps": np.int64(SUB), "anchor": log_anchor, "held": log_held, "strain": log_strain,
       "link_speed_max": speeds, "sides": np.array(sides), "cords": np.array([c["spec"]["name"] for c in cords])}
for n, (C, Qc) in log_pose.items():
    rec[f"cord_{n}_centres"] = C
    rec[f"cord_{n}_quats"] = Qc
np.savez_compressed(f"{OUT}/isaac_recording.npz", **rec)
grasp_rows = [{k_: (v if not isinstance(v, np.ndarray) else v.tolist()) for k_, v in gr.items() if k_ not in ("weld", "off")} for gr in GRASPS]
json.dump({"events": events_log, "grasps": grasp_rows, "frames": T, "seconds": round(time.time() - t_start, 1),
           "peak_link_speed": float(speeds.max()) if T else 0.0}, open(f"{OUT}/isaac_events.json", "w"), indent=1)
print(f"ISAAC_EPISODE_DONE frames {T} peak link speed {float(speeds.max()):.2f} m/s in {time.time() - t_start:.0f}s", flush=True)
app.close()
