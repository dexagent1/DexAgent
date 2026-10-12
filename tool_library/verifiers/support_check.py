"""Support check: does everything rest on something?"""
from __future__ import annotations

TOOL = {
    "name": "support_check",
    "title": "Support check",
    "type": "verifier",
    "category": "Scene Compile Checks",
    "summary": "After settling, every free body touches something, and every fixed object stands "
               "on a surface rather than hanging in mid-air or over an edge.",
    "description": (
        "Two ways an object can float. A free body that has no contact at all after the scene "
        "settles has fallen out of the world or hovers where no support was built. A fixed "
        "body, which the settle test cannot expose because it never falls, is probed with "
        "downward rays across its footprint: most of the footprint must have a surface "
        "directly below (`min_supported_frac`, 60 percent), and that surface must be close "
        "(`max_gap`, 15 mm). Static bodies whose name matches `support_names` (the table, the "
        "floor) are the supports themselves and are skipped. The report gives, per object, "
        "the supported fraction, the gap and the footprint, which is enough to decide whether "
        "to move the object or extend the surface."
    ),
    "inputs": [
        {"name": "scene", "type": "str | mujoco.MjModel", "description": "MJCF file, XML string, or compiled model."},
        {"name": "seconds", "type": "float", "description": "Settle time before judging free bodies. Default 2.0."},
        {"name": "support_names", "type": "list[str]", "description": "Regular expressions of bodies that are supports and are not checked. Default table|floor|ground|world."},
        {"name": "min_supported_frac", "type": "float", "description": "Fraction of a fixed object's footprint that needs a surface below. Default 0.6."},
        {"name": "max_gap", "type": "float, metres", "description": "Largest hover of a fixed object over its surface. Default 0.015."},
    ],
    "outputs": [
        {"name": "report", "type": "dict",
         "description": "ok; floating: free bodies with no contact after settling; static: {body: "
                        "{supported_frac, gap_m, base_z, footprint_x, footprint_y, verdict}}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["compile", "support", "floating", "placement"],
    "used_in": ["Place Cup in Bowl", "Place Toy in Drawer", "Computer Installation", "Biology Experiment"],
    "status": "tested",
    "entrypoint": "support_check",
    "cli": "python -m tool_library.verifiers.support_check scene.xml",
}

import json
import re
import sys

from tool_library.common import scene as S

DEFAULT_SUPPORTS = (r"table", r"floor", r"ground", r"^world$")


def check_support(scene, seconds=2.0, support_names=DEFAULT_SUPPORTS,
                  min_supported_frac=0.6, max_gap=0.015):
    model = S.load_model(scene)
    data = S.forward(model)
    free = set(S.free_bodies(model))
    skip = [re.compile(p) for p in support_names]

    # A free joint owns a whole articulated subtree. A cable's first segment
    # may be above the table while the rest supports it. Contacts internal to
    # that same object do not establish support.
    S.settle(model, data, seconds)
    pairs = S.contact_pairs(model, data)
    supported_by = {}
    for root in free:
        names = {model.body(b).name for b in S.subtree(model, root)}
        external = [(a, b) for a, b, _ in pairs if (a in names) != (b in names)]
        supported_by[model.body(root).name] = external
    floating = [name for name, contacts in supported_by.items() if not contacts]

    # fixed objects: every top-level body welded to the world, taken with the
    # parts hanging off it (a door on its hinge belongs to its cabinet), probed
    # across the footprint of their collidable geometry
    static = {}
    for b in range(1, model.nbody):
        if int(model.body_parentid[b]) != 0 or int(model.body_weldid[b]) != 0:
            continue
        name = model.body(b).name
        if any(p.search(name) for p in skip):
            continue
        gids = S.collidable_geoms(model, S.subtree(model, b))
        if not gids:
            continue
        rec = S.support_of_group(model, data, gids)
        frac, gap = rec["supported_frac"], rec["gap_m"]
        if frac == 0.0:
            rec["verdict"] = "nothing below"
        elif frac < min_supported_frac:
            rec["verdict"] = "over the edge"
        elif gap is not None and gap > max_gap:
            rec["verdict"] = "hovering"
        else:
            rec["verdict"] = "supported"
        static[name] = rec
    bad_static = [n for n, r in static.items() if r["verdict"] != "supported"]
    return {"ok": not floating and not bad_static, "floating": floating, "static": static,
            "unsupported_static": bad_static, "free_object_contacts": supported_by}


def main(argv=None):
    rep = check_support(sys.argv[1] if argv is None else argv[0])
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


support_check = check_support   # the library calls every tool by its own name
