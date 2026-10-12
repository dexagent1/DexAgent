"""Compose a tabletop world: table, floor, lights, cameras, and asset MJCFs placed on it."""
from __future__ import annotations

TOOL = {
    "name": "build_tabletop_world",
    "title": "Build tabletop world",
    "type": "skill",
    "category": "Scene Runtime & Rendering",
    "summary": "Writes one compiled MJCF world from a table description, named cameras, lights and a list of asset MJCFs attached at given poses with name prefixes.",
    "description": (
        "The scene a trajectory runs in is a fixed world, not a reconstruction: a table of "
        "known size and height, a floor, lights, the cameras the renders will use, and the "
        "reconstructed or constructed assets attached where perception put them. Each asset is "
        "attached through an MjSpec frame with a unique prefix so names never collide and the "
        "asset's own joints, actuators and contacts come along. Lights inside asset files are "
        "stripped so a world keeps a single lighting setup. The result compiles, and the summary "
        "lists bodies, joints, cameras and the table so later tools can refer to them by name."
    ),
    "inputs": [
        {"name": "out_xml", "type": "path", "description": "Where the compiled-ready MJCF is written (meshes referenced by absolute path)."},
        {"name": "table", "type": "dict", "description": "{'size': [half_x, half_y], 'top_z': m, 'thickness': m, 'center': [x, y], 'rgba': [...]}; defaults 0.6 x 0.4, top 0.0, 4 cm thick."},
        {"name": "assets", "type": "list[dict]", "description": "[{'name', 'xml', 'pos': [x,y,z], 'quat': [w,x,y,z] | 'yaw_deg'}] placed in world coordinates; z omitted = table top."},
        {"name": "cameras", "type": "dict", "description": "{name: {'pos','quat','fovy'} | {'pos','target','fovy'}} fixed world cameras."},
        {"name": "lights", "type": "list[dict] | None", "description": "Light overrides; default one directional and one fill light."},
        {"name": "floor", "type": "bool", "description": "Add a ground plane 0.75 m below the table (default True)."},
    ],
    "outputs": [
        {"name": "summary", "type": "dict", "description": "out_xml, nq, nbody, bodies per asset, joints, cameras, table."},
    ],
    "status": "tested",
    "entrypoint": "build_tabletop_world",
    "requirements": ["mujoco >= 3.2 (MjSpec)"],
    "tags": ["world", "MJCF", "table", "camera"],
    "used_in": ["Place Cup in Bowl", "Multi-Cup Grasping", "Place Toy in Drawer", "Giftbox Packing", "Knot the Rope",
                "Battery Insertion", "Drawing and Wiping", "Computer Installation", "Open the Bottle", "Scissor Cutting",
                "Biology Experiment"],
}

import json
import re
import sys
from pathlib import Path

import numpy as np

from tool_library.common.render import lookat_quat

TABLE_DEFAULT = {"size": [0.6, 0.4], "top_z": 0.0, "thickness": 0.04, "center": [0.0, 0.0],
                 "rgba": [0.78, 0.74, 0.68, 1.0]}
WORLD_TEMPLATE = """<mujoco model="tabletop_world">
  <compiler angle="radian" meshdir="." autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><map znear="0.01" zfar="30"/><global offwidth="1920" offheight="1080"/></visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.9 0.9 0.92" rgb2="0.6 0.62 0.68" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.35 0.35 0.36" rgb2="0.4 0.4 0.41" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody/>
</mujoco>
"""


def _yaw_quat(deg):
    h = np.radians(float(deg)) / 2
    return [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))]


def _strip_lights(xml_path, work_dir):
    """A copy of the asset without <light> elements (one lighting setup per world)."""
    src = Path(xml_path)
    txt = src.read_text()
    txt2 = re.sub(r"[ \t]*<light\b[^>]*/>\s*\n?", "", txt)
    if txt2 == txt:
        return src
    dst = Path(work_dir) / f"{src.stem}_nolight.xml"
    # keep relative mesh/texture paths valid from the new location
    txt2 = re.sub(r'(meshdir|texturedir)="([^"]*)"',
                  lambda m: f'{m.group(1)}="{(src.parent / m.group(2)).resolve()}"', txt2)
    if "meshdir=" not in txt2:
        txt2 = txt2.replace("<compiler", f'<compiler meshdir="{src.parent.resolve()}"', 1) if "<compiler" in txt2 \
            else re.sub(r"(<mujoco\b[^>]*>)", lambda m: f'{m.group(1)}<compiler meshdir="{src.parent.resolve()}"/>', txt2, count=1)
    dst.write_text(txt2)
    return dst


def build_tabletop_world(out_xml, table=None, assets=(), cameras=None, lights=None, floor=True):
    import mujoco
    tb = {**TABLE_DEFAULT, **(table or {})}
    out = Path(out_xml)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = mujoco.MjSpec.from_string(WORLD_TEMPLATE)
    wb = spec.worldbody
    hx, hy = (float(v) for v in tb["size"])
    cx, cy = (float(v) for v in tb["center"])
    top, th = float(tb["top_z"]), float(tb["thickness"])
    g = wb.add_geom()
    g.name, g.type = "table_top", mujoco.mjtGeom.mjGEOM_BOX
    g.size, g.pos = [hx, hy, th / 2], [cx, cy, top - th / 2]
    g.rgba = [float(v) for v in tb["rgba"]]
    g.friction = [1.0, 0.005, 0.0001]
    if floor:
        f = wb.add_geom()
        f.name, f.type = "floor", mujoco.mjtGeom.mjGEOM_PLANE
        f.size, f.pos, f.material = [0, 0, 0.05], [0, 0, top - 0.75], "floor_mat"
    for li in (lights or [{"pos": [0.3, -0.8, 1.6], "dir": [-0.2, 0.5, -1.0], "diffuse": [0.8, 0.8, 0.8]},
                          {"pos": [-0.6, 0.6, 1.4], "dir": [0.4, -0.4, -1.0], "diffuse": [0.35, 0.35, 0.35]}]):
        L = wb.add_light()
        L.pos, L.dir = [float(v) for v in li["pos"]], [float(v) for v in li["dir"]]
        L.diffuse = [float(v) for v in li.get("diffuse", [0.7, 0.7, 0.7])]
        L.castshadow = bool(li.get("castshadow", True))
    cams = {}
    for name, c in (cameras or {}).items():
        cam = wb.add_camera()
        cam.name = name
        cam.pos = [float(v) for v in c["pos"]]
        cam.quat = [float(v) for v in (c["quat"] if "quat" in c else lookat_quat(c["pos"], c["target"], c.get("up", (0, 0, 1))))]
        cam.fovy = float(c.get("fovy", 45.0))
        cams[name] = {"pos": list(cam.pos), "quat": list(cam.quat), "fovy": cam.fovy}
    placed = {}
    for a in assets:
        child = mujoco.MjSpec.from_file(str(_strip_lights(a["xml"], out.parent)))
        fr = wb.add_frame()
        pos = list(a.get("pos", [0.0, 0.0]))
        if len(pos) == 2:
            pos.append(top)
        fr.pos = [float(v) for v in pos]
        fr.quat = [float(v) for v in a["quat"]] if "quat" in a else _yaw_quat(a.get("yaw_deg", 0.0))
        before = {b.name for b in spec.bodies}
        spec.attach(child, prefix=f"{a['name']}_", frame=fr)
        placed[a["name"]] = {"xml": str(a["xml"]), "pos": list(fr.pos), "quat": list(fr.quat),
                             "bodies": sorted({b.name for b in spec.bodies} - before)}
    model = spec.compile()
    out.write_text(spec.to_xml())
    joints = [model.joint(j).name for j in range(model.njnt)]
    return {"out_xml": str(out), "nq": int(model.nq), "nbody": int(model.nbody), "ngeom": int(model.ngeom),
            "assets": placed, "joints": joints, "cameras": cams,
            "table": {"top_z": top, "half_size": [hx, hy], "center": [cx, cy]}}


if __name__ == "__main__":
    a = sys.argv[1:]
    spec = json.loads(Path(a[1]).read_text()) if len(a) > 1 else {}
    print(json.dumps(build_tabletop_world(a[0], **spec), indent=1))
