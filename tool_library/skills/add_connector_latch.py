"""Add a connector latch to a staged scene: the inactive weld that holds a plug in its port once it is seated."""
from __future__ import annotations

TOOL = {
    "name": "add_connector_latch",
    "title": "Add a connector latch (plug held in its port)",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "Writes the spring latch of a plug-in connector (a USB-C plug in a phone) into a staged Stage 2 scene: an inactive "
               "weld between the port body and the plug body that the Stage 3 executor switches on when the plug bottoms out in "
               "the port and off when it is pulled out harder than release_N; returns the executor rule and the scene_recon.json "
               "fields that carry it.",
    "description": (
        "A real USB-C plug is held by spring latches, not by friction alone, and a gram-scale friction fit is fragile in "
        "contact physics (its hold depends on the contact softness, the noslip iterations and the timestep). The latch is the "
        "executor's run-time rule kind 'latch' (common.episode.latch_rules_live / latch_step, applied every physics step by "
        "servo_rollout in Stage 3 and in Stage 4's physics replay): when the plug tip site is within seat_m of the port "
        "floor site, axes within 20 degrees, has stopped advancing and no longer presses on the port harder than 1 N (so "
        "the push of the hand that seated it is not locked in: a plug the hand still presses latches when the hand lets "
        "go), the weld's relpose is set to the bodies' current relative pose and the weld is switched on (no snap); when the weld carries a pull-out load (its force on the plug "
        "along the port's insertion axis, low-passed over 5 ms) above release_N it is switched off and stays off until the "
        "plug has left the port. An engaged latch is handed from subgoal to subgoal. This skill writes the weld into the "
        "scene file (body1 = port body, body2 = plug body, anchor at the plug tip site, relpose = the seated pose: the tip "
        "site's frame on the floor site's frame turned roll_deg about the insertion axis, torquescale 1, a nearly hard solref "
        "/ solimp, active=false; a scene whose video starts with the plug in its port gives active=True and the relpose is "
        "read from keyframe `key`, where the plug must be seated). It re-compiles the scene and checks the weld. Copy "
        "`scene_fields` into the plug's (the cable's) scene_recon.json entry: execute_subgoal_plan (latch_rules_from_recon) "
        "and write_stage4_config read the rule from the entry's `latch`. construct_phone's <name>_port_floor and "
        "construct_plug_cable's <name>_plug_tip are the two sites (with the scene's name prefix after stage_on_bench); prove "
        "the port with plug_port_retention_test, whose default trials run the same latch rule in the episode's physics."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "Staged Stage 2 scene MJCF (scene.xml after stage_on_bench)."},
        {"name": "plug_body", "type": "str", "description": "Body that carries the plug (construct_plug_cable's plug.body, scene-prefixed)."},
        {"name": "plug_site", "type": "str", "description": "Site at the plug tip, z along the insertion direction, on plug_body (construct_plug_cable's plug.tip_site)."},
        {"name": "port_body", "type": "str", "description": "Body that carries the port (construct_phone's body, scene-prefixed)."},
        {"name": "port_site", "type": "str", "description": "Site on the port floor, z along the insertion direction, on port_body (construct_phone's sites.port_floor)."},
        {"name": "out_xml", "type": "str | None", "description": "Where to write the scene with the latch. Default: scene itself (in place)."},
        {"name": "name", "type": "str | None", "description": "Weld name. Default latch_<plug_body>. A weld of that name already in the scene is replaced."},
        {"name": "seat_m", "type": "float", "description": "Tip-to-floor distance that counts as seated, m. Default 0.0015"},
        {"name": "release_N", "type": "float", "description": "Pull-out load at which the latch lets go, N (USB-C: 8-20 N). Default 15"},
        {"name": "roll_deg", "type": "float", "description": "Roll of the seated plug about the insertion axis relative to the floor site's frame (a reversible plug seats at 0 or 180). Default 0"},
        {"name": "active", "type": "bool", "description": "Start the latch engaged (the plug sits in its port at keyframe `key`). Default false"},
        {"name": "key", "type": "int", "description": "Keyframe read when active=True. Default 0"},
        {"name": "solref, solimp", "type": "list", "description": "Weld softness. Default [0.004, 1] and [0.998, 0.999, 0.001, 0.5, 2] (about 0.006 mm per N on a 6 g plug)."},
    ],
    "outputs": [
        {"name": "result", "type": "dict", "description": "xml (the written scene), equality, rule ({kind: 'latch', eq, site_a, site_b, seat_m, release_N}: the executor rule), "
                                                         "scene_fields ({latch: {eq, site_a, site_b, seat_m, release_N, plug_body, port_body}}: copy into the plug's "
                                                         "scene_recon.json entry), eq_data (anchor 3, relpose 7, torquescale 1), check {active0, quat_norm, torquescale, "
                                                         "seated_gap_m (active=True)}."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["connector", "latch", "plug", "port", "usb-c", "weld", "equality", "stage2", "insertion"],
    "status": "tested",
    "entrypoint": "add_connector_latch",
    "cli": "python -m tool_library.skills.add_connector_latch scene.xml plug_body plug_site port_body port_site",
}

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tool_library.common import episode as E


def _s(v):
    return " ".join("%.9g" % float(x) for x in v)


def add_connector_latch(scene, plug_body, plug_site, port_body, port_site, out_xml=None, name=None, seat_m=E.LATCH_SEAT_M,
                        release_N=E.LATCH_RELEASE_N, roll_deg=0.0, active=False, key=0, solref=None, solimp=None):
    import mujoco
    scene = Path(scene)
    out = Path(out_xml) if out_xml else scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    ids = {}
    for kind, nm, obj in (("body", plug_body, mujoco.mjtObj.mjOBJ_BODY), ("body", port_body, mujoco.mjtObj.mjOBJ_BODY),
                          ("site", plug_site, mujoco.mjtObj.mjOBJ_SITE), ("site", port_site, mujoco.mjtObj.mjOBJ_SITE)):
        i = mujoco.mj_name2id(model, obj, str(nm))
        if i < 0:
            raise ValueError(f"add_connector_latch: no {kind} {nm!r} in {scene.name}")
        ids[nm] = int(i)
    if ids[plug_body] == ids[port_body]:
        raise ValueError("add_connector_latch: plug_body and port_body are the same body")
    for site, body in ((plug_site, plug_body), (port_site, port_body)):
        if int(model.site_bodyid[ids[site]]) != ids[body]:
            raise ValueError(f"add_connector_latch: site {site!r} is on body {model.body(int(model.site_bodyid[ids[site]])).name!r}, "
                             f"not on {body!r} (the weld joins the bodies that carry the two sites)")
    if float(seat_m) <= 0 or float(release_N) <= 0:
        raise ValueError("add_connector_latch: seat_m and release_N must be > 0")
    eq_name = str(name or f"latch_{plug_body}")
    solref = list(E.LATCH_WELD_SOLREF if solref is None else solref)
    solimp = list(E.LATCH_WELD_SOLIMP if solimp is None else solimp)
    check = {}
    data_eq = E.latch_seated_eq_data(model, plug_site, port_site, roll_deg)
    if active:
        # the scene starts plugged in: the weld holds the pose the plug has at the keyframe, which must be seated
        d = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, d, int(key))
        mujoco.mj_forward(model, d)
        gap = float(np.linalg.norm(d.site_xpos[ids[plug_site]] - d.site_xpos[ids[port_site]]))
        if gap > float(seat_m):
            raise ValueError(f"add_connector_latch: active=True but at keyframe {key} the plug tip is {1000 * gap:.2f} mm from the port "
                             f"floor (> seat_m {1000 * float(seat_m):.1f} mm): seat the plug in the keyframe or leave the latch inactive")
        b1, b2 = ids[port_body], ids[plug_body]
        R1 = d.xmat[b1].reshape(3, 3); R2 = d.xmat[b2].reshape(3, 3)
        anchor = data_eq[0:3]
        pos = R1.T @ (d.xpos[b2] + R2 @ anchor - d.xpos[b1])
        q = np.zeros(4); mujoco.mju_mulQuat(q, np.array([d.xquat[b1][0], *(-d.xquat[b1][1:])]), d.xquat[b2])
        data_eq = np.concatenate([anchor, pos, q / np.linalg.norm(q), [1.0]])
        check["seated_gap_m"] = round(gap, 6)

    root = ET.parse(str(scene)).getroot()
    eqs = root.find("equality")
    if eqs is None:
        eqs = ET.SubElement(root, "equality")
    for w in list(eqs):
        if w.get("name") == eq_name:
            eqs.remove(w)
    ET.SubElement(eqs, "weld", {"name": eq_name, "body1": str(port_body), "body2": str(plug_body), "anchor": _s(data_eq[0:3]),
                                "relpose": _s(data_eq[3:10]), "torquescale": "1", "active": "true" if active else "false",
                                "solref": _s(solref), "solimp": _s(solimp)})
    ET.indent(root, space="  ")
    # a scene written elsewhere keeps its relative asset paths only if it stays in the same folder
    if out.resolve().parent != scene.resolve().parent:
        raise ValueError("add_connector_latch: out_xml must be in the scene's folder (its meshes are referenced relative to it)")
    out.write_text(ET.tostring(root, encoding="unicode") + "\n")

    m2 = mujoco.MjModel.from_xml_path(str(out))
    e = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
    if e < 0:
        raise RuntimeError(f"add_connector_latch: the weld {eq_name!r} is not in the written scene")
    ed = np.asarray(m2.eq_data[e], float)
    check.update({"active0": bool(m2.eq_active0[e]), "quat_norm": round(float(np.linalg.norm(ed[6:10])), 9),
                  "torquescale": float(ed[10]), "eq_data_matches": bool(np.allclose(ed, data_eq, atol=1e-7))})
    if not check["eq_data_matches"] or abs(check["quat_norm"] - 1.0) > 1e-6 or check["active0"] != bool(active):
        raise RuntimeError(f"add_connector_latch: the compiled weld differs from what was written: {check}")
    # the rule must load on the written scene (sites on the weld's bodies, a body-body weld)
    rule = {"kind": "latch", "eq": eq_name, "site_a": str(plug_site), "site_b": str(port_site),
            "seat_m": float(seat_m), "release_N": float(release_N)}
    E.latch_rules_live(m2, [rule])
    fields = {"latch": {**{k: v for k, v in rule.items() if k != "kind"}, "plug_body": str(plug_body), "port_body": str(port_body)}}
    return {"xml": str(out), "equality": eq_name, "rule": rule, "scene_fields": fields,
            "eq_data": [round(float(v), 9) for v in data_eq], "check": check}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    print(json.dumps(add_connector_latch(*a[:5]), indent=1))
