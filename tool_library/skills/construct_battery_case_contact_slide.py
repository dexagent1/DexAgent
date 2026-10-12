"""Construct battery holder with one prismatic spring contact per slot.

Fork of construct_battery_case: the same case, slots and spring coils, but each
slot's coil chain is driven by ONE slide joint carrying the spring's whole
compression travel with the series-equivalent stiffness, so the spring contact
is a single prismatic part (what a manipulation planner and articulation checks
expect) instead of N one-millimetre coil joints.
"""
from __future__ import annotations

TOOL = {
    "name": "construct_battery_case_contact_slide",
    "title": "Construct battery holder with single-slide spring contacts",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "An open N-slot battery holder whose spring contact in every slot is one prismatic joint with the spring's full compression travel and series-equivalent stiffness.",
    "description": "Builds the holder with the construct_battery_case builder (cell size, slot count, spring ends, walls, colours, wires), then rewrites each slot's spring chain: the first coil joint keeps its axis and becomes the contact slide with range [-total_travel, upper_slack] (total_travel = sum of the coil travels, negative = compressed), stiffness 1/sum(1/k_i) and one coil's damping; the other coil joints are removed so the coils ride on the contact rigidly. upper_slack (default 1e-5 m) lets the contact sit at its free length without resting on a hard stop. Assumes the builder names coil joints <name>_s<i>spring_c<k>. Fails with ValueError if the requested min_travel is not reached; raise spring_free_length or lower spring_wire_radius / spring_n_coils to get more travel.",
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives the MJCF, meshes and <name>_params.json."},
        {"name": "preview", "type": "bool", "description": "Render preview stills when the base builder supports it. Default False."},
        {"name": "upper_slack", "type": "float", "description": "Upper limit of the contact slide above the free length, metres. Default 1e-5."},
        {"name": "min_travel", "type": "float", "description": "Required compression travel of each contact slide, metres; ValueError below it. Default 0.0 (no requirement)."},
        {"name": "**params", "type": "dict", "description": "Any construct_battery_case parameter (battery_diameter, battery_length, n_slots, springs_at, slot_clearance, seated_preload, spring_stiffness, spring_free_length, spring_coil_radius, spring_wire_radius, spring_n_coils, colours, with_wires, include_floor, ...); unknown keys raise."},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "The construct_battery_case build dict with, per slot, spring_joints = [the single contact joint], contact_joint, contact_travel_m and contact_stiffness_N_per_m; xml and params_json paths."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "trimesh"],
    "simulators": ["MuJoCo"],
    "tags": ["battery", "holder", "spring", "prismatic", "contact", "articulated", "object reconstruction"],
    "used_in": ["Battery Insertion"],
    "status": "tested",
    "entrypoint": "construct_battery_case_contact_slide",
    "cli": "python -m tool_library.skills.construct_battery_case_contact_slide out_dir",
}

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from tool_library.common import battery_case as _core

DEFAULTS = dict(_core.DEFAULTS)


def _collapse(xml_path, slots):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parent = {c: p for p in root.iter() for c in p}
    joints = {j.get("name"): j for j in root.iter("joint")}
    out = []
    for s in slots:
        names = list(s["spring_joints"])
        missing = [n for n in names if n not in joints]
        if missing:
            raise ValueError(f"spring joints {missing} not found in {xml_path}")
        lo = sum(float(joints[n].get("range").split()[0]) for n in names)
        ks = [float(joints[n].get("stiffness", "0")) for n in names]
        keq = 1.0 / sum(1.0 / k for k in ks) if all(k > 0 for k in ks) else 0.0
        ds = [float(joints[n].get("damping", "0")) for n in names]
        deq = max(ds) if ds else 0.0   # keep one coil's damping: the collapsed slide must settle as fast as the chain
        j1 = joints[names[0]]
        j1.set("range", f"{lo:.6f} {{slack}}")
        j1.set("stiffness", f"{keq:.4f}")
        j1.set("damping", f"{deq:.5f}")
        for n in names[1:]:
            parent[joints[n]].remove(joints[n])
        out.append({"joint": names[0], "travel": -lo, "k": keq})
    return tree, out


def construct_battery_case_contact_slide(out_dir, preview=False, upper_slack=1e-5, min_travel=0.0, **params):
    """Build the holder (construct_battery_case parameters) and collapse every
    slot's coil chain into one prismatic contact joint. Returns the build dict."""
    if upper_slack < 0 or upper_slack > 0.001:
        raise ValueError("upper_slack must be in [0, 0.001] m")
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown parameters {sorted(unknown)}; allowed: {sorted(DEFAULTS)}")
    built = _core.build(dict(params), out_dir)
    xml_path = Path(built["xml"])
    if not xml_path.is_absolute():
        xml_path = Path(out_dir) / xml_path.name
    tree, info = _collapse(xml_path, built["slots"])
    for el in tree.getroot().iter("joint"):
        if el.get("range", "").endswith("{slack}"):
            el.set("range", el.get("range").replace("{slack}", f"{upper_slack:.6f}"))
    tree.write(xml_path)
    for s, inf in zip(built["slots"], info):
        if inf["travel"] < min_travel:
            raise ValueError(f"slot {s['index']} contact travel {inf['travel']:.4f} m < min_travel {min_travel}; "
                             "raise spring_free_length or lower spring_wire_radius / spring_n_coils")
        s["spring_joints"] = [inf["joint"]]
        s["contact_joint"] = inf["joint"]
        s["contact_travel_m"] = round(inf["travel"], 6)
        s["contact_stiffness_N_per_m"] = round(inf["k"], 3)
    built["contact_upper_slack_m"] = upper_slack
    pj = built.get("params_json")
    if pj:
        p = Path(pj) if Path(pj).is_absolute() else Path(out_dir) / Path(pj).name
        try:
            d = json.loads(p.read_text())
        except Exception:
            d = {}
        d["contact_slide"] = {"upper_slack": upper_slack, "slots": [
            {"joint": i["joint"], "travel_m": i["travel"], "stiffness": i["k"]} for i in info]}
        p.write_text(json.dumps(d, indent=1))
    if preview and hasattr(_core, "preview"):
        try:
            built["stills"] = _core.preview(built, out_dir)
        except Exception as e:  # preview is optional
            built["preview_error"] = str(e)
    return built


if __name__ == "__main__":
    import sys
    print(json.dumps(construct_battery_case_contact_slide(sys.argv[1] if len(sys.argv) > 1 else "."), indent=1, default=str))
