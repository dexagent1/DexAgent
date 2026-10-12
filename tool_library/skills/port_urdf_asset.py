"""Port a URDF (ArtiCraft output, PartNet-Mobility, a hand-written cabinet)
to a metric MuJoCo asset with usable joints.

MuJoCo can read a URDF, but a URDF says nothing about how its joints feel or
how its surfaces collide, and a construction fitted in a clip's depth frame
is often not in metres. This port walks the URDF's link tree, keeps every
box, cylinder, sphere and mesh with its colour, multiplies lengths by
`scale`, turns prismatic/revolute/continuous joints into slide/hinge joints
with limits, damping, dry friction and armature, gives the root a fixed or
free base, and writes `<name>.xml` plus `<name>_info.json`. The info lists
every joint (type, axis, range, parent, child), the footprint, and for each
moving link the longest cylinder it carries as a handle site, which is what
a grasp planner wants from a drawer or a door.

    from tool_library.skills.port_urdf_asset import port_urdf_asset
    info = port_urdf_asset("model.urdf", out_dir, name="cabinet", scale=1.0)
    info["xml"], info["joints"]
"""
from __future__ import annotations

TOOL = {
    "name": "port_urdf_asset",
    "title": "Port URDF asset",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "URDF (ArtiCraft, PartNet-Mobility, hand-written) to a metric MJCF asset: all "
               "geometry and colours, slide/hinge joints with limits, damping and friction, "
               "handle sites, and a joint/footprint info sheet.",
    "description": (
        "Parses the URDF with the standard library, finds the root link, and emits one MJCF "
        "body per link nested by the joint tree. Geometry comes from the <visual> elements "
        "(or <collision> with geometry='collision'); boxes, cylinders, spheres and meshes are "
        "kept with their origins and material colours, every length multiplied by `scale`; "
        "mesh files are copied beside the XML. prismatic -> slide and revolute/continuous -> "
        "hinge joints keep their axis and limits (scaled for slides) and get the damping, "
        "frictionloss and armature given; fixed joints merge the child into its parent's "
        "frame; a floating joint (below the root by construction) raises ValueError, since MuJoCo allows free joints "
        "only on top-level bodies: port that link as its own asset. URDF inertials are kept when present. The "
        "root body is fixed to the world by default or given a freejoint. A default contact "
        "class (condim 4, friction 0.8, solref 0.004) makes the surfaces graspable. The info "
        "sheet lists joints, per-link handle sites (the longest cylinder on each moving "
        "link), the carcass footprint and the drawer measurements (front face, bin interior, "
        "bin floor) when the link follows the ArtiCraft cabinet naming (front/bottom/side_*/"
        "back panels)."
    ),
    "inputs": [
        {"name": "urdf", "type": "str", "description": "Path of the URDF file."},
        {"name": "out_dir", "type": "str", "description": "Directory for <name>.xml, meshes and <name>_info.json."},
        {"name": "name", "type": "str", "description": "Asset/root body name. Default 'articulated'."},
        {"name": "scale", "type": "float", "description": "Multiplier applied to every length. Default 1.0."},
        {"name": "joint_damping", "type": "float", "description": "Damping on every moving joint. Default 3.0 (a drawer slide)."},
        {"name": "joint_frictionloss", "type": "float", "description": "Dry friction on every moving joint. Default 0.4."},
        {"name": "joint_armature", "type": "float", "description": "Armature on every moving joint. Default 0.01."},
        {"name": "fixed_base", "type": "bool", "description": "Root fixed to the world (True) or on a freejoint. Default True."},
        {"name": "geometry", "type": "str", "description": "'visual' (default) or 'collision': which URDF elements become geoms."},
        {"name": "moving_mass", "type": "float | None",
         "description": "Override mass for every moving link (kg); None keeps the URDF inertials."},
        {"name": "pos", "type": "list[float]", "description": "World position of the root. Default [0, 0, 0]."},
        {"name": "yaw_deg", "type": "float", "description": "World yaw of the root. Default 0."},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane and lights. Default False."},
    ],
    "outputs": [
        {"name": "xml", "type": "str", "description": "The MJCF asset."},
        {"name": "info_json", "type": "str", "description": "<name>_info.json with everything below."},
        {"name": "joints", "type": "list[dict]",
         "description": "name, type (slide|hinge), axis, range, parent, child, origin_m for every moving joint."},
        {"name": "handles", "type": "dict", "description": "link -> {site, centre_m, radius_m, length_m} for links with a cylinder handle."},
        {"name": "drawers", "type": "dict", "description": "link -> front/bin measurements when the ArtiCraft panel names are present."},
        {"name": "footprint_m", "type": "dict", "description": "Root link's box extents in x, y, z."},
    ],
    "status": "tested",
    "entrypoint": "port_urdf_asset",
    "requirements": ["numpy", "scipy", "mujoco"],
    "simulators": ["MuJoCo"],
    "tags": ["urdf", "articulated", "drawer", "cabinet", "articraft", "mjcf", "reconstruct", "object reconstruction"],
    "used_in": ["Place Toy in Drawer"],
    "cli": "python -m tool_library.skills.port_urdf_asset <model.urdf> <out_dir> [name] [scale]",
}

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _rpy_quat(rpy):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_euler("xyz", [float(v) for v in rpy]).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def _origin(el):
    o = el.find("origin") if el is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return np.asarray(xyz, float), np.asarray(rpy, float)


def _fmt(v):
    return " ".join(f"{float(x):.6f}" for x in v)


def _rgba_of(el, materials):
    mat = el.find("material")
    if mat is None:
        return None
    col = mat.find("color")
    if col is not None and col.get("rgba"):
        return col.get("rgba")
    return materials.get(mat.get("name"))


def _resolve_mesh(fn: str, urdf_dir: Path) -> Path:
    if fn.startswith("package://"):
        fn = fn[len("package://"):]
        fn = fn.split("/", 1)[1] if "/" in fn else fn
    if fn.startswith("file://"):
        fn = fn[len("file://"):]
    p = Path(fn)
    return p if p.is_absolute() else (urdf_dir / p)


def port_urdf_asset(urdf, out_dir, name: str = "articulated", scale: float = 1.0,
                    joint_damping: float = 3.0, joint_frictionloss: float = 0.4,
                    joint_armature: float = 0.01, fixed_base: bool = True,
                    geometry: str = "visual", moving_mass=None, pos=(0.0, 0.0, 0.0),
                    yaw_deg: float = 0.0, include_floor: bool = False) -> dict:
    from tool_library.common.mjcf import mjcf_header, FLOOR_XML, LIGHTS_XML, load
    urdf, out_dir = Path(urdf), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = float(scale)
    root = ET.parse(str(urdf)).getroot()
    materials = {}
    for mat in root.findall("material"):
        col = mat.find("color")
        if col is not None and col.get("rgba"):
            materials[mat.get("name")] = col.get("rgba")
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")
    children = {}
    parent_of = {}
    for j in joints:
        p_, c_ = j.find("parent").get("link"), j.find("child").get("link")
        children.setdefault(p_, []).append(j)
        parent_of[c_] = j
    roots = [n for n in links if n not in parent_of]
    if len(roots) != 1:
        raise ValueError(f"{urdf}: expected one root link, found {roots}")
    root_link = roots[0]
    which = {"visual": "visual", "collision": "collision"}[geometry]

    meshes = {}          # asset name -> (file, scale)
    info_joints, handles, drawers = [], {}, {}
    mesh_dir = out_dir / "meshes"

    def geoms_of(link_name: str, indent: str) -> list[str]:
        link = links[link_name]
        out = []
        for i, el in enumerate(link.findall(which)):
            gname = f"{link_name}_{el.get('name', f'{which}{i}')}"
            xyz, rpy = _origin(el)
            geo = el.find("geometry")
            rgba = _rgba_of(el, materials) or "0.6 0.6 0.6 1"
            pose = f'pos="{_fmt(xyz * s)}" quat="{_fmt(_rpy_quat(rpy))}" rgba="{rgba}"'
            if geo.find("box") is not None:
                sz = np.asarray([float(v) for v in geo.find("box").get("size").split()]) * s / 2
                out.append(f'{indent}<geom name="{gname}" type="box" size="{_fmt(sz)}" {pose}/>')
            elif geo.find("cylinder") is not None:
                c = geo.find("cylinder")
                out.append(f'{indent}<geom name="{gname}" type="cylinder" '
                           f'size="{float(c.get("radius"))*s:.6f} {float(c.get("length"))*s/2:.6f}" {pose}/>')
            elif geo.find("sphere") is not None:
                out.append(f'{indent}<geom name="{gname}" type="sphere" '
                           f'size="{float(geo.find("sphere").get("radius"))*s:.6f}" {pose}/>')
            elif geo.find("mesh") is not None:
                me = geo.find("mesh")
                src = _resolve_mesh(me.get("filename"), urdf.parent)
                if not src.exists():
                    raise FileNotFoundError(f"mesh {src} referenced by {link_name}")
                msc = np.asarray([float(v) for v in me.get("scale", "1 1 1").split()]) * s
                mesh_dir.mkdir(exist_ok=True)
                dst = mesh_dir / src.name
                if not dst.exists():
                    shutil.copy(src, dst)
                aname = f"{name}_{src.stem}"
                meshes[aname] = (dst.name, msc)
                out.append(f'{indent}<geom name="{gname}" type="mesh" mesh="{aname}" {pose}/>')
            else:
                raise ValueError(f"unhandled geometry in link {link_name}")
        return out

    def handle_of(link_name: str):
        bar = None
        for el in links[link_name].findall(which):
            c = el.find("geometry").find("cylinder")
            if c is None:
                continue
            xyz, rpy = _origin(el)
            L = float(c.get("length"))
            if bar is None or L > bar["L"]:
                bar = {"L": L, "r": float(c.get("radius")), "xyz": xyz, "rpy": rpy}
        return bar

    def drawer_of(link_name: str):
        inner, front, floor_z = {}, None, None
        for el in links[link_name].findall(which):
            nm = el.get("name", "")
            b = el.find("geometry").find("box")
            if b is None:
                continue
            sz = [float(v) for v in b.get("size").split()]
            xyz, _ = _origin(el)
            if nm == "bottom":
                floor_z = xyz[2] + sz[2] / 2
            if nm == "front":
                front = (xyz, sz)
                inner["y_hi"] = xyz[1] - sz[1] / 2
                inner["z_hi"] = xyz[2] + sz[2] / 2
            if nm == "side_left":
                inner["x_lo"] = xyz[0] + sz[0] / 2
            if nm == "side_right":
                inner["x_hi"] = xyz[0] - sz[0] / 2
            if nm == "back":
                inner["y_lo"] = xyz[1] + sz[1] / 2
        if front is None or floor_z is None:
            return None
        fxyz, fsz = front
        return {"front_width_m": fsz[0] * s, "front_height_m": fsz[2] * s,
                "front_face_y_link_m": float((fxyz[1] + fsz[1] / 2) * s),
                "bin_inner_m": {k: float(v * s) for k, v in inner.items()},
                "bin_floor_z_link_m": float(floor_z * s)}

    def inertial_xml(link_name: str, indent: str, moving: bool) -> str:
        ine = links[link_name].find("inertial")
        if moving and moving_mass is not None:
            return (f'{indent}<inertial pos="0 0 0" mass="{float(moving_mass):.5f}" '
                    f'diaginertia="0.004 0.003 0.004"/>')
        if ine is None or ine.find("mass") is None:
            return ""
        xyz, rpy = _origin(ine)
        mass = float(ine.find("mass").get("value"))
        it = ine.find("inertia")
        if it is None:
            return f'{indent}<inertial pos="{_fmt(xyz*s)}" mass="{mass * s**3:.6f}" diaginertia="1e-4 1e-4 1e-4"/>'
        # the URDF tensor is in the inertial frame given by rpy; MuJoCo takes
        # either a full tensor OR an orientation, so rotate the tensor into
        # the body frame and give it as fullinertia (mass scales with s^3,
        # inertia with s^5 when the whole object is scaled)
        from scipy.spatial.transform import Rotation as Rot
        I = np.array([[float(it.get("ixx", 0)), float(it.get("ixy", 0)), float(it.get("ixz", 0))],
                      [float(it.get("ixy", 0)), float(it.get("iyy", 0)), float(it.get("iyz", 0))],
                      [float(it.get("ixz", 0)), float(it.get("iyz", 0)), float(it.get("izz", 0))]])
        R = Rot.from_euler("xyz", rpy).as_matrix()
        Ib = R @ I @ R.T * s ** 5
        mass_s = mass * s ** 3
        w, _ = np.linalg.eigh(Ib)
        if w.min() <= 0 or mass_s <= 0:
            return ""
        full = " ".join(f"{v:.10f}" for v in (Ib[0, 0], Ib[1, 1], Ib[2, 2], Ib[0, 1], Ib[0, 2], Ib[1, 2]))
        return (f'{indent}<inertial pos="{_fmt(xyz*s)}" mass="{mass_s:.6f}" fullinertia="{full}"/>')

    def body_xml(link_name: str, joint, depth: int) -> str:
        ind = "  " * (depth + 2)
        if joint is None:
            jpos = np.zeros(3)
            jq = np.array([1.0, 0, 0, 0])
            jxml = ""
            moving = False
        else:
            jxyz, jrpy = _origin(joint)
            jpos, jq = jxyz * s, _rpy_quat(jrpy)
            jt = joint.get("type")
            jn = f"{name}_{joint.get('name')}"
            axis = [float(v) for v in (joint.find("axis").get("xyz").split()
                                        if joint.find("axis") is not None else "1 0 0".split())]
            lim = joint.find("limit")
            moving = jt in ("prismatic", "revolute", "continuous", "floating")
            if jt == "prismatic":
                lo, hi = float(lim.get("lower", 0.0)) * s, float(lim.get("upper", 0.0)) * s
                rng = f'range="{lo:.6f} {hi:.6f}"'
                jxml = (f'{ind}  <joint name="{jn}" type="slide" axis="{_fmt(axis)}" {rng} '
                        f'damping="{joint_damping}" frictionloss="{joint_frictionloss}" '
                        f'armature="{joint_armature}"/>')
                info_joints.append({"name": jn, "type": "slide", "axis": axis, "range": [lo, hi],
                                    "parent": joint.find("parent").get("link"), "child": link_name,
                                    "origin_m": [float(v) for v in jpos]})
            elif jt in ("revolute", "continuous"):
                if jt == "revolute" and lim is not None:
                    lo, hi = float(lim.get("lower", 0.0)), float(lim.get("upper", 0.0))
                    rng = f'range="{np.degrees(lo):.4f} {np.degrees(hi):.4f}"'
                    rr = [lo, hi]
                else:
                    rng, rr = 'limited="false"', None
                jxml = (f'{ind}  <joint name="{jn}" type="hinge" axis="{_fmt(axis)}" {rng} '
                        f'damping="{joint_damping}" frictionloss="{joint_frictionloss}" '
                        f'armature="{joint_armature}"/>')
                info_joints.append({"name": jn, "type": "hinge", "axis": axis, "range": rr,
                                    "parent": joint.find("parent").get("link"), "child": link_name,
                                    "origin_m": [float(v) for v in jpos]})
            elif jt == "floating":
                raise ValueError(f"{urdf}: joint {joint.get('name')!r} is floating, but link {link_name!r} sits below the "
                                 "root and MuJoCo allows free joints only on top-level bodies; port that link as its own asset")
            else:                                   # fixed
                jxml = ""
        parts = [f'{ind}<body name="{name}_{link_name}" pos="{_fmt(jpos)}" quat="{_fmt(jq)}">']
        if jxml:
            parts.append(jxml)
        ie = inertial_xml(link_name, ind + "  ", moving)
        if ie:
            parts.append(ie)
        parts += geoms_of(link_name, ind + "  ")
        if moving:
            bar = handle_of(link_name)
            if bar is not None:
                site = f"{name}_{link_name}_handle"
                parts.append(f'{ind}  <site name="{site}" pos="{_fmt(bar["xyz"]*s)}" '
                             f'quat="{_fmt(_rpy_quat(bar["rpy"]))}" size="0.004" rgba="0 1 0 0"/>')
                handles[link_name] = {"site": site, "centre_link_m": [float(v) for v in bar["xyz"] * s],
                                      "radius_m": bar["r"] * s, "length_m": bar["L"] * s}
            dr = drawer_of(link_name)
            if dr is not None:
                drawers[link_name] = dr
        for cj in children.get(link_name, []):
            parts.append(body_xml(cj.find("child").get("link"), cj, depth + 1))
        parts.append(f"{ind}</body>")
        return "\n".join(parts)

    body = body_xml(root_link, None, 0)
    # the root body: fixed to the world, or free
    root_tag = f'<body name="{name}_{root_link}" pos="{_fmt(np.zeros(3))}" quat="{_fmt([1,0,0,0])}">'
    yaw_q = _fmt([np.cos(np.radians(yaw_deg) / 2), 0, 0, np.sin(np.radians(yaw_deg) / 2)])
    free = "" if fixed_base else f'\n      <freejoint name="{name}_free"/>'
    body = body.replace(root_tag, f'<body name="{name}_{root_link}" pos="{_fmt(pos)}" quat="{yaw_q}">{free}', 1)
    asset = ""
    if meshes:
        asset = "  <asset>\n" + "\n".join(
            f'    <mesh name="{a}" file="meshes/{f}" scale="{_fmt(sc)}"/>' for a, (f, sc) in meshes.items()) + "\n  </asset>\n"
    xml = mjcf_header(name, timestep=0.002) + f"""
  <default>
    <geom condim="4" friction="0.8 0.02 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"/>
  </default>
{asset}  <worldbody>{LIGHTS_XML + FLOOR_XML if include_floor else ""}
{body}
  </worldbody>
</mujoco>
"""
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(xml)
    m = load(xml_path)                         # compiles, or raises
    # footprint of the root link's own geometry
    xs, ys, zs = [], [], []
    for el in links[root_link].findall(which):
        b = el.find("geometry").find("box")
        if b is None:
            continue
        sz = [float(v) for v in b.get("size").split()]
        xyz, _ = _origin(el)
        xs += [xyz[0] - sz[0] / 2, xyz[0] + sz[0] / 2]
        ys += [xyz[1] - sz[1] / 2, xyz[1] + sz[1] / 2]
        zs += [xyz[2] - sz[2] / 2, xyz[2] + sz[2] / 2]
    info = {
        "xml": str(xml_path), "source_urdf": str(urdf), "scale": s, "name": name,
        "root_body": f"{name}_{root_link}", "fixed_base": bool(fixed_base),
        "joints": info_joints, "handles": handles, "drawers": drawers,
        "footprint_m": ({"x": [min(xs) * s, max(xs) * s], "y": [min(ys) * s, max(ys) * s],
                         "z": [min(zs) * s, max(zs) * s]} if xs else None),
        "n_bodies": int(m.nbody), "n_geoms": int(m.ngeom),
        "note": "bodies are '<name>_<link>', joints '<name>_<joint>'; a slide's range is in "
                "metres (scaled), a hinge's in radians in this sheet and degrees in the XML",
    }
    (out_dir / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    info["info_json"] = str(out_dir / f"{name}_info.json")
    return info


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    print(json.dumps(port_urdf_asset(a[0], a[1], a[2] if len(a) > 2 else "articulated",
                                     float(a[3]) if len(a) > 3 else 1.0), indent=1))
