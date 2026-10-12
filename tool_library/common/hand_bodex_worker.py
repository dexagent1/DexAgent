"""BODex force-closure grasp synthesis worker. Runs INSIDE the BODex conda
environment (torch / warp / cuRobo), as a subprocess of
skills/grasp_prior_bodex.py; the two talk through a job.json and an .npz,
never through imports.

    python hand_bodex_worker.py job.json

job.json:
  {"bodex_dir": <BODex checkout>, "out_dir": ..., "seed_num": 20,
   "objects": [{"obj_id": ..., "mesh": <metric .obj/.stl>, "T_settled": 4x4 world pose}],
   "hands":   [{"key": "<robot>_<side>", "robot_yml": ..., "manip_dir": ...}]}

Per (object, hand) it writes {out_dir}/{obj_id}__{key}.npz with
  base_pose_rel (nseed, 3, 4, 4)  BODex base_link pose per stage
                                 [pre, grasp, squeeze], OBJECT-LOCAL
  joints (nseed, 3, nj), joint_names (nj,), grasp_error (nseed,),
  dist_error (nseed,), base_link (str), T_syn (4, 4)
The object sits at its settled pose on a table plane whose top is z=0, so
grasps from below or through the table are excluded by the collision
world itself.
"""
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as et

import numpy as np
import trimesh
import yaml


def quat_wxyz(R):
    T = np.eye(4)
    T[:3, :3] = R
    return trimesh.transformations.quaternion_from_matrix(T)


def build_asset(spec, dgn):
    obj_id = spec["obj_id"]
    T_set = np.asarray(spec["T_settled"], float)
    proc = f"{dgn}/processed_data/{obj_id}"
    scene_dir = f"{dgn}/scene_cfg/{obj_id}/table"
    shutil.rmtree(f"{dgn}/processed_data/{obj_id}", ignore_errors=True)
    shutil.rmtree(f"{dgn}/scene_cfg/{obj_id}", ignore_errors=True)
    for d in (f"{proc}/mesh", f"{proc}/info", f"{proc}/urdf/meshes", scene_dir):
        os.makedirs(d, exist_ok=True)
    m = trimesh.load(spec["mesh"], force="mesh")
    V = np.asarray(m.vertices, float)
    center = (V.max(0) + V.min(0)) / 2
    half_diag = float(np.linalg.norm(V.max(0) - V.min(0)) / 2)
    m.vertices = (V - center) / half_diag              # BODex canonical: half-diagonal 1
    simp = f"{proc}/mesh/simplified.obj"
    m.export(simp)
    pieces = []
    try:
        import coacd
        parts = coacd.run_coacd(coacd.Mesh(np.asarray(m.vertices), np.asarray(m.faces)), threshold=0.05)
        merged = []
        for i, (pv, pf) in enumerate(parts):
            p = f"{proc}/urdf/meshes/convex_piece_{i}.obj"
            trimesh.Trimesh(np.asarray(pv), np.asarray(pf), process=False).export(p)
            pieces.append(p)
            merged.append(trimesh.Trimesh(np.asarray(pv), np.asarray(pf), process=False))
        trimesh.util.concatenate(merged).export(f"{proc}/urdf/coacd.obj")
    except Exception as e:                      # noqa: BLE001
        print(f"[bodex_worker] CoACD failed ({e}) -> convex hull", flush=True)
        p = f"{proc}/urdf/meshes/convex_piece_0.obj"
        m.convex_hull.export(p)
        pieces.append(p)
        m.convex_hull.export(f"{proc}/urdf/coacd.obj")
    root = et.Element("robot", name="root")
    for p in pieces:
        rel = os.path.relpath(p, f"{proc}/urdf")
        link = et.SubElement(root, "link", name=f"link_{os.path.basename(p)}")
        ine = et.SubElement(link, "inertial")
        et.SubElement(ine, "origin", xyz="0 0 0", rpy="0 0 0")
        et.SubElement(ine, "mass", value="0.1")
        et.SubElement(ine, "inertia", ixx="1e-3", ixy="0", ixz="0", iyy="1e-3", iyz="0", izz="1e-3")
        for tag in ("visual", "collision"):
            el = et.SubElement(link, tag)
            et.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
            geo = et.SubElement(el, "geometry")
            et.SubElement(geo, "mesh", filename=rel, scale="1.0 1.0 1.0")
    et.ElementTree(root).write(f"{proc}/urdf/coacd.urdf")
    mj = et.Element("mujoco", model="root")
    asset = et.SubElement(mj, "asset")
    body = et.SubElement(et.SubElement(mj, "worldbody"), "body", name="object")
    for i, p in enumerate(pieces):
        rel = os.path.relpath(p, f"{proc}/urdf")
        et.SubElement(asset, "mesh", name=f"piece_{i}", file=rel)
        et.SubElement(body, "geom", type="mesh", mesh=f"piece_{i}")
    et.ElementTree(mj).write(f"{proc}/urdf/coacd.xml")
    sm = trimesh.load(simp, force="mesh")
    try:
        gc, dens, mass = sm.center_mass.tolist(), float(sm.density), float(sm.mass)
    except Exception:                           # noqa: BLE001
        gc, dens, mass = sm.centroid.tolist(), 1000.0, 0.1
    json.dump({"gravity_center": gc, "obb": sm.bounding_box_oriented.primitive.extents.tolist(),
               "scale": 1.0, "density": dens, "mass": mass},
              open(f"{proc}/info/simplified.json", "w"), indent=1)
    R_s = T_set[:3, :3]
    t_s = R_s @ center + T_set[:3, 3]
    q = quat_wxyz(R_s)
    tag = f"scale{int(round(half_diag * 100)):03d}"
    scene = {"scene": {obj_id: {"type": "rigid_object",
                                "file_path": os.path.relpath(simp, scene_dir),
                                "xml_path": os.path.relpath(f"{proc}/urdf/coacd.xml", scene_dir),
                                "urdf_path": os.path.relpath(f"{proc}/urdf/coacd.urdf", scene_dir),
                                "info_path": os.path.relpath(f"{proc}/info/simplified.json", scene_dir),
                                "scale": np.array([half_diag] * 3),
                                "pose": np.array([t_s[0], t_s[1], t_s[2], q[0], q[1], q[2], q[3]])},
                       "table_plane": {"type": "plane", "pose": np.array([0.0, 0, 0, 1, 0, 0, 0])}},
             "scene_id": f"{obj_id}/{tag}",
             "task": {"type": "force_closure", "obj_name": obj_id}}
    np.save(f"{scene_dir}/{tag}", scene)
    return tag


def synth(bodex, out, manip, obj_id, tag, hand, seeds):
    fc = yaml.safe_load(open(f"{manip}/{hand['manip_dir']}/fc.yml"))
    fc["world"]["template_path"] = f"object/DGN_2k/scene_cfg/{obj_id}/table/*.npy"
    fc["world"]["start"] = 0
    fc["world"]["end"] = 10
    fc["exp_name"] = "lib"
    fc["seed_num"] = int(seeds)
    cfg_name = f"{hand['manip_dir']}/lib_{obj_id}.yml"
    yaml.safe_dump(fc, open(f"{manip}/{cfg_name}", "w"))
    save_folder = f"{out}/raw/{obj_id}__{hand['key']}"
    npy = f"{save_folder}/grasp_data/{obj_id}/{tag}_grasp.npy"
    if os.path.exists(npy):
        return npy
    cmd = [sys.executable, "example_grasp/plan_batch_env.py", "-c", cfg_name, "-w", "1", "-f", save_folder]
    r = subprocess.run(cmd, cwd=bodex, capture_output=True, text=True, timeout=3600)
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-12:]), flush=True)
    if not os.path.exists(npy):
        raise RuntimeError(f"synthesis produced no {npy}")
    return npy


def export(bodex, out, npy, obj_id, hand, T_set):
    d = np.load(npy, allow_pickle=True).item()
    rp = np.asarray(d["robot_pose"][0], float)
    ge = np.asarray(d["grasp_error"][0], float)[:, 0]
    de = np.asarray(d["dist_error"][0], float)[:, 0]
    jn = [str(x) for x in d["joint_names"]]
    Tinv = np.linalg.inv(np.asarray(T_set, float))
    rel = np.zeros((rp.shape[0], rp.shape[1], 4, 4))
    for i in range(rp.shape[0]):
        for k in range(rp.shape[1]):
            T = trimesh.transformations.quaternion_matrix(rp[i, k, 3:7])
            T[:3, 3] = rp[i, k, :3]
            rel[i, k] = Tinv @ T
    base_link = yaml.safe_load(open(f"{bodex}/src/curobo/content/configs/robot/{hand['robot_yml']}"))[
        "robot_cfg"]["kinematics"]["base_link"]
    path = f"{out}/{obj_id}__{hand['key']}.npz"
    np.savez(path, base_pose_rel=rel, joints=rp[:, :, 7:], joint_names=np.array(jn), grasp_error=ge,
             dist_error=de, base_link=np.array(base_link), T_syn=np.asarray(T_set, float))
    print(f"[bodex_worker] {obj_id} x {hand['key']}: {rp.shape[0]} seeds, "
          f"{int((ge < 0.2).sum())} force-closure -> {path}", flush=True)


def main():
    job = json.load(open(sys.argv[1]))
    bodex, out = job["bodex_dir"], job["out_dir"]
    dgn = f"{bodex}/src/curobo/content/assets/object/DGN_2k"
    manip = f"{bodex}/src/curobo/content/configs/manip"
    os.makedirs(out, exist_ok=True)
    for spec in job["objects"]:
        tag = build_asset(spec, dgn)
        for hand in job["hands"]:
            npy = synth(bodex, out, manip, spec["obj_id"], tag, hand, job.get("seed_num", 20))
            export(bodex, out, npy, spec["obj_id"], hand, spec["T_settled"])
    print("BODEX_WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
