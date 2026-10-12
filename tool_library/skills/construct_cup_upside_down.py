TOOL = {
    "name": "construct_cup_upside_down",
    "title": "Upside-down cup (mouth on the table)",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "A construct_cup vessel turned mouth-down: the open rim rests on the support and the closed base faces up, with the body origin at the rim contact so it can be staged at z = 0.",
    "description": (
        "Builds the cup with construct_cup from the same parameters (base_radius = the closed base, rim_radius = the open mouth, "
        "height, wall, colours, mass, ...) and then wraps every geom and site of the root body in a frame rotated 180 degrees about x "
        "and lifted by the height, so the mouth ring sits at body z = 0 and the closed base at z = height. The free joint stays on the "
        "root body, so the asset is staged with stage_on_bench like any standing object and needs no roll. Collision stays the hollow "
        "wedge model (the inside under the cup is empty). A params json with upside_down = true is written next to the XML. "
        "Assumes MuJoCo >= 3.1 (<frame> element). Unknown parameters raise ValueError (they are passed to construct_cup's validation)."
    ),
    "inputs": [
        {"name": "out_dir", "type": "str", "description": "Directory for the XML, meshes and params json."},
        {"name": "params", "type": "kwargs", "description": "Any construct_cup parameter (name, base_radius = closed base radius, rim_radius = mouth radius, height, wall, lip, bottom, body_rgba, inner_rgba, mass, density, friction, condim, pos, yaw_deg, fixed, include_floor, ...)."},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "construct_cup's dict (xml, params_json, body, free_joint, sizes) plus orientation='mouth_down', top_z_m (closed base height above the support), mouth_radius_m, params_json_upside_down."},
    ],
    "status": "tested",
    "entrypoint": "construct_cup_upside_down",
    "tags": ["cup", "inverted", "upside down", "stacking cup", "vessel", "rigid"],
    "used_in": ["Place Cup in Bowl"],
    "requirements": ["mujoco>=3.1"],
}

import json
import os
import xml.etree.ElementTree as ET


def construct_cup_upside_down(out_dir, **params) -> dict:
    from tool_library.skills.construct_cup import construct_cup, DEFAULTS
    unknown = [k for k in params if k not in DEFAULTS and k not in ("preview",)]
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; valid: {sorted(DEFAULTS)}")
    params.pop("preview", None)
    built = construct_cup(out_dir, **params)
    h = float(built["height_m"])
    tree = ET.parse(built["xml"])
    root = tree.getroot()
    body = root.find("./worldbody/body[@name='%s']" % built["body"])
    if body is None:
        raise ValueError("construct_cup produced no root body named %r" % built["body"])
    frame = ET.Element("frame", {"name": built["body"] + "_flip", "pos": f"0 0 {h:.6f}", "euler": "180 0 0"})
    moved = [c for c in list(body) if c.tag in ("geom", "site", "body")]
    for c in moved:
        body.remove(c)
        frame.append(c)
    body.append(frame)
    ET.indent(tree, space="  ")
    tree.write(built["xml"])
    with open(built["params_json"]) as f:
        pj = json.load(f)
    pj["upside_down"] = True
    pj_path = os.path.join(out_dir, f"{built['name']}_params_upside_down.json")
    with open(pj_path, "w") as f:
        json.dump(pj, f, indent=1)
    built.update({"orientation": "mouth_down", "top_z_m": h, "mouth_radius_m": float(built["rim_radius_m"]),
                  "params_json_upside_down": pj_path,
                  "note": built.get("note", "") + " Turned mouth-down: rim at body z=0, closed base at z=height; sites moved with the geoms."})
    return built
