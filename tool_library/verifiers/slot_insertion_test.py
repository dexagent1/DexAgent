"""Slot insertion test: a free part pushed into a receptacle's slot must slide in, seat and stay."""
TOOL = {
    "name": "slot_insertion_test",
    "title": "Slot insertion test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Pushes a free part (a drive tray, a card, a cartridge) into each named slot of a fixed receptacle with a capped force, releases it and shoves it sideways: it must slide in to the slot centre, stay there unheld and be penned in by the slot walls.",
    "description": (
        "Composes one throwaway model from the receptacle MJCF (its root held fixed: a free joint is removed) and the "
        "insert MJCF (its first body, kept free). Each slot is given by two sites in the receptacle: a floor site at the "
        "seated centre of the slot on its floor, and a mouth site in front of the opening; the insertion axis is the "
        "horizontal direction mouth -> floor site. The insert is turned so its nose axis (insert_nose_local, or the "
        "longest horizontal extent of its collision geometry when omitted) points along that axis, set on the slot floor "
        "with its leading face start_offset_m past the mouth site, and pushed by a plain body force along the axis while "
        "support_frac of its weight is carried (as a hand would), then released, then pushed sideways both ways. A slot "
        "passes when the insert's centre reaches the floor site within seat_tol_m along the axis, drifts less than "
        "max_drift_m after release, stays within max_rest_z_m of resting on the slot floor, and moves less than "
        "max_lateral_m under the side force. Assumes metres, z up, primitive or asset-free geometry in the insert."),
    "inputs": [
        {"name": "receptacle_xml", "type": "path", "description": "MJCF of the receptacle (rack, cage, case) containing the slot sites."},
        {"name": "insert_xml", "type": "path", "description": "MJCF whose first worldbody body is the part to insert."},
        {"name": "slots", "type": "list[[str, str]]", "description": "Per slot [floor_site, mouth_site] names in the receptacle."},
        {"name": "insert_nose_local", "type": "list[3] | None", "description": "Insert-body axis that goes in first. Default None = longest horizontal collision extent."},
        {"name": "push_force_N", "type": "float", "description": "Push along the axis. Default 6."},
        {"name": "push_s", "type": "float", "description": "Push duration. Default 1.5."},
        {"name": "support_frac", "type": "float", "description": "Fraction of the weight carried during the push. Default 0.9."},
        {"name": "release_s", "type": "float", "description": "Unheld rest after the push. Default 1.0."},
        {"name": "lateral_force_N", "type": "float", "description": "Side shove after release, each way. Default None = 2.5 x the insert weight, enough to beat floor friction so only walls can hold it."},
        {"name": "start_offset_m", "type": "float", "description": "Leading face past the mouth site at the start. Default 0.04."},
        {"name": "seat_tol_m", "type": "float", "description": "Allowed shortfall of the centre to the floor site. Default 0.02."},
        {"name": "max_drift_m", "type": "float", "description": "Allowed motion while unheld. Default 0.005."},
        {"name": "max_lateral_m", "type": "float", "description": "Allowed side motion under the shove. Default 0.012."},
        {"name": "max_rest_z_m", "type": "float", "description": "Allowed gap between the insert's bottom and the slot floor at rest. Default 0.008."},
        {"name": "out_json", "type": "path | None", "description": "Also write the report here."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; per slot: depth_error_m (centre vs floor site along the axis, negative = short), drift_m, lateral_m, rest_gap_m, seated, held, penned, ok; insert size and axis."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["slot", "insert", "drive bay", "rack", "cartridge", "seat", "mechanism test"],
    "used_in": ["Computer Installation"],
    "status": "tested",
    "entrypoint": "slot_insertion_test",
    "cli": "python -m tool_library.verifiers.slot_insertion_test receptacle.xml insert.xml floor_site mouth_site",
}

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _compose(receptacle_xml, insert_xml):
    rx = ET.parse(str(receptacle_xml)).getroot()
    ix = ET.parse(str(insert_xml)).getroot()
    rwb = rx.find("worldbody")
    for b in rwb.findall("body"):
        for j in list(b.findall("freejoint")) + [j for j in b.findall("joint") if j.get("type") == "free"]:
            b.remove(j)
    ibody = ix.find("worldbody").find("body")
    if ibody is None:
        raise ValueError("insert_xml has no body in its worldbody; pass the part's MJCF")
    if ibody.find("freejoint") is None and not any(j.get("type") == "free" for j in ibody.findall("joint")):
        ET.SubElement(ibody, "freejoint")
    for tag in ("asset", "default"):
        src = ix.find(tag)
        if src is not None and len(src):
            dst = rx.find(tag)
            if dst is None:
                dst = ET.SubElement(rx, tag)
            for c in src:
                dst.append(c)
    ibody.set("pos", "0 0 0"); ibody.set("quat", "1 0 0 0")
    for k in ("euler", "axisangle", "xyaxes", "zaxis"):
        ibody.attrib.pop(k, None)
    rwb.append(ibody)
    return ET.tostring(rx, encoding="unicode"), ibody.get("name")


def _local_aabb(model, data, bid):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    R = data.xmat[bid].reshape(3, 3); p = data.xpos[bid]
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0):
            continue
        c = R.T @ (data.geom_xpos[g] - p); Rg = R.T @ data.geom_xmat[g].reshape(3, 3)
        hs = model.geom_aabb[g][3:]
        ext = np.abs(Rg) @ hs
        cc = c + Rg @ model.geom_aabb[g][:3]
        lo = np.minimum(lo, cc - ext); hi = np.maximum(hi, cc + ext)
    if not np.isfinite(lo).all():
        raise ValueError("the insert body has no collision geoms")
    return lo, hi


def _yaw_quat(theta):
    return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)])


def slot_insertion_test(receptacle_xml, insert_xml, slots, insert_nose_local=None, push_force_N=6.0, push_s=1.5,
                        support_frac=0.9, release_s=1.0, lateral_force_N=None, start_offset_m=0.04, seat_tol_m=0.02,
                        max_drift_m=0.005, max_lateral_m=0.012, max_rest_z_m=0.008, out_json=None):
    import mujoco
    if not slots or any(len(s) != 2 for s in slots):
        raise ValueError("slots must be a non-empty list of [floor_site, mouth_site] pairs")
    if push_force_N <= 0 or push_s <= 0:
        raise ValueError("push_force_N and push_s must be positive")
    xml, iname = _compose(receptacle_xml, insert_xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    bid = model.body(iname).id
    adr = int(model.jnt_qposadr[model.body_jntadr[bid]])
    mujoco.mj_forward(model, data)
    lo, hi = _local_aabb(model, data, bid)
    if insert_nose_local is None:
        k = 0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else 1
        nose = np.zeros(3); nose[k] = 1.0
    else:
        nose = np.asarray(insert_nose_local, float); nose[2] = 0.0
        if np.linalg.norm(nose) < 1e-9:
            raise ValueError("insert_nose_local must have a horizontal component")
        nose /= np.linalg.norm(nose)
    lead = float(max(np.dot(lo, nose), np.dot(hi, nose)))          # leading face along the nose
    ctr_off = 0.5 * (lo + hi)
    length = float(abs(np.dot(hi - lo, nose)))
    mg = float(model.body_subtreemass[bid]) * 9.81
    if lateral_force_N is None:
        lateral_force_N = 2.5 * mg
    lateral_force_N = float(lateral_force_N)
    issues, per = [], []
    q0 = data.qpos.copy()
    for floor_site, mouth_site in slots:
        try:
            fs, ms = model.site(floor_site).id, model.site(mouth_site).id
        except KeyError as e:
            raise ValueError(f"site {e} not found in receptacle_xml; check the slots names")
        mujoco.mj_resetData(model, data); data.qpos[:] = q0; mujoco.mj_forward(model, data)
        F = data.site_xpos[fs].copy(); Mo = data.site_xpos[ms].copy()
        ax = F - Mo; ax[2] = 0.0
        if np.linalg.norm(ax) < 1e-6:
            raise ValueError(f"floor site {floor_site} and mouth site {mouth_site} coincide horizontally")
        ax /= np.linalg.norm(ax)
        lat = np.array([-ax[1], ax[0], 0.0])
        th = np.arctan2(ax[1], ax[0]) - np.arctan2(nose[1], nose[0])
        q = _yaw_quat(th)
        Rq = np.zeros(9); mujoco.mju_quat2Mat(Rq, q); Rq = Rq.reshape(3, 3)
        # world position of the body origin: leading face at mouth + start_offset along the axis,
        # centred laterally on the slot line, bottom 1 mm above the floor site
        lead_w = Rq @ (nose * lead)
        c_w = Rq @ ctr_off
        target_lead = Mo + ax * start_offset_m
        pos = np.zeros(3)
        pos[:2] = F[:2] + ax[:2] * (np.dot(target_lead - F, ax)) - lead_w[:2]
        lat_err = np.dot(pos + c_w - F, lat); pos[:2] -= lat[:2] * lat_err
        pos[2] = F[2] - lo[2] + 0.001
        data.qpos[adr:adr + 3] = pos; data.qpos[adr + 3:adr + 7] = q
        data.qvel[:] = 0; mujoco.mj_forward(model, data)
        dt = model.opt.timestep
        bad = False
        def run(seconds, force):
            nonlocal bad
            for _ in range(int(seconds / dt)):
                data.xfrc_applied[bid, :3] = force
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all():
                    bad = True; return
            data.xfrc_applied[bid, :3] = 0
        run(push_s, push_force_N * ax + np.array([0, 0, support_frac * mg]))
        run(0.3, np.zeros(3))
        p_rel = data.xpos[bid].copy()  # body origin
        run(release_s, np.zeros(3))
        p_rest = data.xpos[bid].copy()
        ctr = p_rest + data.xmat[bid].reshape(3, 3) @ ctr_off
        depth_err = float(np.dot(ctr - F, ax))
        drift = float(np.linalg.norm(p_rest - p_rel))
        bottom = float(ctr[2] - (hi[2] - lo[2]) / 2)
        rest_gap = float(bottom - F[2])
        run(0.5, lateral_force_N * lat); pa = data.xpos[bid].copy()
        run(0.5, -lateral_force_N * lat); pb = data.xpos[bid].copy()
        lateral = float(max(abs(np.dot(pa - p_rest, lat)), abs(np.dot(pb - p_rest, lat))))
        seated = depth_err >= -seat_tol_m and abs(rest_gap) <= max_rest_z_m
        held = drift <= max_drift_m
        penned = lateral <= max_lateral_m
        ok = bool(seated and held and penned and not bad)
        r = {"slot": floor_site, "depth_error_m": round(depth_err, 4), "drift_m": round(drift, 4),
             "rest_gap_m": round(rest_gap, 4), "lateral_m": round(lateral, 4), "seated": bool(seated),
             "held": bool(held), "penned": bool(penned), "unstable": bad, "ok": ok}
        per.append(r)
        if bad:
            issues.append(f"{floor_site}: simulation went non-finite; check the solref/timestep of the receptacle")
        if not seated:
            issues.append(f"{floor_site}: insert stopped {(-depth_err) * 1000:.0f} mm short of the slot centre or rests "
                          f"{rest_gap * 1000:.0f} mm off the floor; widen/heighten the slot (slot_w, slot_h) or shrink the insert")
        if not held:
            issues.append(f"{floor_site}: insert moved {drift * 1000:.1f} mm after release; add a back stop or raise friction")
        if not penned:
            issues.append(f"{floor_site}: a {lateral_force_N:.1f} N side shove moved it {lateral * 1000:.0f} mm; the slot needs side walls or rails")
    rep = {"ok": all(r["ok"] for r in per), "issues": issues, "slots": per,
           "insert": {"body": iname, "length_m": round(length, 4), "nose_local": nose.tolist(),
                      "mass_kg": round(mg / 9.81, 4)},
           "push_force_N": push_force_N, "lateral_force_N": round(lateral_force_N, 3)}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    import sys
    print(json.dumps(slot_insertion_test(sys.argv[1], sys.argv[2], [[sys.argv[3], sys.argv[4]]]), indent=1))
