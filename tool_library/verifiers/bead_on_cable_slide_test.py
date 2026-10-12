"""Verifier: a holed bead (ring, washer, spacer) threads onto a cable from its free end, slides along it, stops
against a fatter stop section (a column of beads already strung, a knot, a ferrule) and cannot leave it sideways."""
from __future__ import annotations

TOOL = {
    "name": "bead_on_cable_slide_test",
    "title": "Bead on cable slide test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Force-only trial of a holed bead on a cable: pushed coaxially at the cable's free tip it must thread on, slide along the cable, come to rest against the cable's fat stop section (a cap or bead column) without passing it, and stay threaded under a sideways push.",
    "description": (
        "Rebuilds the cable straight with tool_library.common.cable from the given cross-section (radius, length, "
        "segments, stop section of cap_radius over cap_length at end b; or reads them from a construct_rope params "
        "JSON), pins both ends to the world with welds (on end links made 1e4 times heavier: a weld's give is about "
        "F tc^2 / m, which on a thin cord's 0.03 g end link let a 0.2 N push swing the whole 'pinned' cable 10 cm), "
        "switches gravity off, and inserts the bead asset (any MJCF "
        "whose bead body has a free joint; its actuators, contacts, assets and equalities are copied along) with its "
        "bore axis along the cable, its centre a few millimetres beyond the free tip (end a). Phase 1 pushes the bead "
        "along +x with a speed-regulated force capped at push_force_N: it must pass the tip (thread on), travel at "
        "least half of the thin section, and stop with its centre before the start of the stop section and no more "
        "than half its height plus the stop radius plus 2 mm short of it. Phase 2 keeps the axial push and adds "
        "lateral_force_N sideways: the bead centre must stay within the bore radius plus tol_m of the cable's current "
        "centreline (it is threaded, not resting beside the cable). While the bead is at rest (below 1 cm/s) no bead-cable contact may "
        "penetrate deeper than max_penetration_m; the transient peak of impacts is reported, not judged. Assumes the bead's bore axis is a principal axis of its body frame (bead_axis) "
        "and its collision geoms are distributed symmetrically about the bore (their mean position is the bore centre)."
    ),
    "inputs": [
        {"name": "bead_xml", "type": "path", "description": "MJCF of the bead asset (one free body; e.g. construct_magnetic_ring with magnet_force_N 0)."},
        {"name": "bore_radius", "type": "float", "description": "Radius of the bead's hole, metres (the free radius the cable passes through)."},
        {"name": "bead_height", "type": "float", "description": "Length of the bead along its bore axis, metres."},
        {"name": "bead_body", "type": "str | None", "description": "Name of the bead body; None takes the first top-level body with a free joint. Default None"},
        {"name": "bead_axis", "type": "str", "description": "Bore axis in the bead body frame: 'x', 'y' or 'z'. Default 'z'"},
        {"name": "cable_params_json", "type": "path | None", "description": "A construct_rope params JSON to read radius, cap_radius_scale, cap_length_b, density, friction and stiffness from (its length is replaced by cable_length). Default None"},
        {"name": "cable_radius", "type": "float | None", "description": "Cable radius, metres (required without cable_params_json)."},
        {"name": "cap_radius", "type": "float | None", "description": "Radius of the stop section at end b, metres; None or <= cable_radius for no stop. Default None"},
        {"name": "cap_length", "type": "float", "description": "Length of the stop section at end b, metres (0 for none). Default 0.04"},
        {"name": "cable_length", "type": "float | None", "description": "Length of the straight test cable, metres. Default: the stop section plus 15 cm (at least 0.16)"},
        {"name": "segments", "type": "int", "description": "Capsule segments of the test cable. Default 24"},
        {"name": "push_force_N", "type": "float", "description": "Cap on the axial push, newtons. Default 0.05"},
        {"name": "push_speed", "type": "float", "description": "Target sliding speed of the push, m/s. Default 0.08"},
        {"name": "lateral_force_N", "type": "float", "description": "Sideways push in phase 2, newtons. Default 0.2"},
        {"name": "tol_m", "type": "float", "description": "Slack on the threaded-distance bound, metres. Default 0.0007"},
        {"name": "max_penetration_m", "type": "float", "description": "Deepest allowed bead-cable penetration, metres. Default 0.001"},
        {"name": "timestep", "type": "float", "description": "Simulation timestep, seconds. Default 0.002"},
        {"name": "out_dir", "type": "path | None", "description": "Folder for the merged test MJCF; None uses a temporary folder. Default None"},
        {"name": "out_json", "type": "path | None", "description": "Write the report here as JSON. Default None"},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; cable_pin_drift_m (how far the pinned cable ends moved; over 1 cm voids the trial); insertion {passed_tip, travel_m, final_gap_to_stop_m, stopped_at_stop, has_stop}; lateral {max_offset_from_centreline_m, bound_m, threaded}; max_penetration_m (while the bead is at rest, judged); peak_transient_penetration_m (impacts, reported); params used; merged_xml."},
    ],
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["bead", "ring", "thread", "cable", "rope", "slide", "insert", "threading", "mechanism test", "verifier"],
    "used_in": ["EgoDex thread_unthread_bead_necklace"],
    "status": "tested",
    "entrypoint": "bead_on_cable_slide_test",
    "cli": "python -m tool_library.verifiers.bead_on_cable_slide_test bead_xml bore_radius bead_height cable_radius",
}

import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_ALLOWED = {"bead_body", "bead_axis", "cable_params_json", "cable_radius", "cap_radius", "cap_length", "cable_length",
            "segments", "push_force_N", "push_speed", "lateral_force_N", "tol_m", "max_penetration_m", "timestep",
            "out_dir", "out_json"}


def _first_free_body(root):
    wb = root.find("worldbody")
    for b in (wb if wb is not None else []):
        if b.tag == "body" and (b.find("freejoint") is not None or any(j.get("type") == "free" for j in b.findall("joint"))):
            return b
    return None


def _axis_quat(axis):
    h = math.sqrt(0.5)
    return {"z": [h, 0.0, h, 0.0], "y": [h, 0.0, 0.0, -h], "x": [1.0, 0.0, 0.0, 0.0]}[axis]


def _qrot(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, float)


def bead_on_cable_slide_test(bead_xml, bore_radius, bead_height, **kw):
    import mujoco
    from tool_library.common import cable as CB

    unknown = sorted(set(kw) - _ALLOWED)
    if unknown:
        raise ValueError(f"bead_on_cable_slide_test: unknown parameter(s) {unknown}; allowed: {sorted(_ALLOWED)}")
    axis = kw.get("bead_axis", "z")
    if axis not in ("x", "y", "z"):
        raise ValueError("bead_on_cable_slide_test: bead_axis must be 'x', 'y' or 'z'")
    issues = []
    cp = {}
    if kw.get("cable_params_json"):
        cp = json.load(open(kw["cable_params_json"]))
    r_c = kw.get("cable_radius") if kw.get("cable_radius") is not None else cp.get("radius")
    if r_c is None:
        raise ValueError("bead_on_cable_slide_test: give cable_radius or cable_params_json")
    r_c = float(r_c)
    cap_r = kw.get("cap_radius")
    if cap_r is None and cp:
        cap_r = r_c * float(cp.get("cap_radius_scale", 1.0))
    cap_len = float(kw.get("cap_length", cp.get("cap_length_b", 0.04) if cp else 0.04) or 0.0)
    has_stop = cap_r is not None and float(cap_r) > r_c + 1e-6 and cap_len > 0
    cap_r = float(cap_r) if has_stop else r_c
    # the free part of the test cable must leave the bead room to slide before the stop: by default 15 cm of it
    L = float(kw["cable_length"]) if kw.get("cable_length") is not None else max(0.16, (cap_len if has_stop else 0.0) + 0.15)
    nseg = int(kw.get("segments", 24))
    if has_stop and L - cap_len < 0.08:
        raise ValueError(f"bead_on_cable_slide_test: cable_length {L:.3f} m leaves {L - cap_len:.3f} m before the {cap_len:.3f} m stop; "
                         f"give at least 8 cm (omit cable_length for 15 cm)")
    bore, h = float(bore_radius), float(bead_height)
    if bore <= 0 or h <= 0:
        raise ValueError("bead_on_cable_slide_test: bore_radius and bead_height must be > 0")
    Fmax = float(kw.get("push_force_N", 0.05)); vt = float(kw.get("push_speed", 0.08))
    Flat = float(kw.get("lateral_force_N", 0.2)); tol = float(kw.get("tol_m", 0.0007))
    pen_max = float(kw.get("max_penetration_m", 0.001)); dt = float(kw.get("timestep", 0.002))

    out_dir = Path(kw.get("out_dir") or tempfile.mkdtemp(prefix="beadcable_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    z0 = 0.05
    cparams = {"name": "tcable", "length": L, "segments": nseg, "radius": r_c, "cap_length": 0.0, "cap_length_a": 0.0,
               "cap_length_b": cap_len if has_stop else 0.0, "cap_radius_scale": (cap_r / r_c) if has_stop else 1.0,
               "grips": False, "include_floor": False, "pos": [0.0, 0.0, z0], "timestep": dt}
    for k in ("density", "friction", "bend_stiffness", "twist_stiffness", "joint_damping", "contact_solref", "condim"):
        if k in cp:
            cparams[k] = cp[k]
    built = CB.build(cparams, out_dir / "cable")
    cx = ET.parse(built["xml"]).getroot()
    bx = ET.parse(bead_xml).getroot()
    bead = None
    if kw.get("bead_body"):
        for b in bx.iter("body"):
            if b.get("name") == kw["bead_body"]:
                bead = b
                break
    else:
        bead = _first_free_body(bx)
    if bead is None:
        raise ValueError("bead_on_cable_slide_test: no bead body found (set bead_body to a body with a free joint)")
    bname = bead.get("name")
    # bore centre in the bead body frame: mean position of the body's own colliding geoms
    probe = mujoco.MjModel.from_xml_path(str(bead_xml))
    bid0 = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_BODY, bname)
    gs = [g for g in range(probe.ngeom) if probe.geom_bodyid[g] == bid0 and (probe.geom_contype[g] or probe.geom_conaffinity[g])]
    if not gs:
        raise ValueError("bead_on_cable_slide_test: the bead body has no colliding geoms")
    c_loc = probe.geom_pos[gs].mean(axis=0)
    q = _axis_quat(axis)
    start = np.array([-(h / 2 + 0.004), 0.0, z0])
    pos = start - _qrot(q, c_loc)
    bead.set("pos", " ".join(f"{v:.7g}" for v in pos))
    bead.set("quat", " ".join(f"{v:.7g}" for v in q))
    for a in ("euler", "axisangle", "xyaxes", "zaxis"):
        bead.attrib.pop(a, None)
    cx.find("worldbody").append(bead)
    for sec in ("asset", "actuator", "contact", "equality", "tendon", "sensor"):
        s = bx.find(sec)
        if s is not None and len(s):
            t = cx.find(sec)
            if t is None:
                t = ET.SubElement(cx, sec)
            t.extend(list(s))
    opt = cx.find("option")
    if opt is None:
        opt = ET.SubElement(cx, "option")
    opt.set("gravity", "0 0 0"); opt.set("timestep", f"{dt:.6g}")
    eq = cx.find("equality")
    if eq is None:
        eq = ET.SubElement(cx, "equality")
    first, last = built["bodies"]["first"], built["bodies"]["last"]
    pin_ref = f"{2.0 * dt:.6g} 1"            # as stiff as the timestep allows
    ET.SubElement(eq, "weld", {"name": "pin_a", "body1": first, "solref": pin_ref, "solimp": "0.99 0.999 0.0001 0.5 2"})
    ET.SubElement(eq, "weld", {"name": "pin_b", "body1": last, "solref": pin_ref, "solimp": "0.99 0.999 0.0001 0.5 2"})
    merged = out_dir / "bead_on_cable.xml"
    merged.write_text(ET.tostring(cx, encoding="unicode"))
    m = mujoco.MjModel.from_xml_path(str(merged))
    # PINNED MEANS PINNED. A weld is a soft constraint whose give under a force F is about F tc^2 / m: on a 1.25 mm
    # cord's end link (0.03 g) the 0.2 N sideways push moved the "pinned" ends 100 mm, the whole cable swung away and the
    # bead was thrown off or kept by luck depending on the timestep. The two pinned links are made 1e4 times heavier
    # (gravity is off, so only their inertia changes), which turns the welds' give into microns; the links in between,
    # which the bead actually slides on, keep their real mass.
    for bname_pin in (first, last):
        pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname_pin)
        m.body_mass[pb] *= 1e4
        m.body_inertia[pb] *= 1e4
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    jadr = None
    for j in range(m.njnt):
        if m.jnt_bodyid[j] == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            jadr = int(m.jnt_dofadr[j])
    cab = [i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("tcableB")]
    bead_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == bid}
    # the test cable is straight and pinned, so its self-collision only adds the stop section's fat capsules pressing on
    # their non-adjacent neighbours (a 4 mm stop on 6 mm links overlaps by 1.5 mm at rest, pushing the chain from
    # inside): cable geoms keep colliding with the bead and stop colliding with each other
    bead_bits = int(np.bitwise_or.reduce([int(m.geom_contype[g]) for g in bead_geoms] + [0]))
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) in set(cab):
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = bead_bits or 1
    cab_set = set(cab)
    mass = float(m.body_subtreemass[bid])
    kv = mass / 0.05
    x_tip = 0.0
    x_stop = L - cap_len if has_stop else L

    def centre():
        return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ c_loc

    def cable_pts():
        pts = [d.xpos[i] for i in cab]
        pts.append(d.xpos[cab[-1]] + d.xmat[cab[-1]].reshape(3, 3) @ np.array([L / nseg, 0, 0]))
        return np.asarray(pts)

    def dist_to_polyline(p, P):
        best = np.inf
        for a, b in zip(P[:-1], P[1:]):
            ab = b - a; t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            best = min(best, float(np.linalg.norm(p - (a + t * ab))))
        return best

    def penetration():
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in bead_geoms and m.geom_bodyid[g2] in cab_set) or (g2 in bead_geoms and m.geom_bodyid[g1] in cab_set):
                worst = min(worst, float(c.dist))
        return -worst

    pins = [cab[0], cab[-1]]
    pin0 = d.xpos[pins].copy()
    pin_drift = [0.0]

    def track_pins():
        pin_drift[0] = max(pin_drift[0], float(np.linalg.norm(d.xpos[pins] - pin0, axis=1).max()))

    pen0 = penetration()
    max_pen = pen0          # resting penetration (bead speed below 0.01 m/s), judged
    peak_pen = pen0         # transient peak during impacts, reported only
    x0c = float(centre()[0])
    # phase 1: thread on and slide to the stop
    T1 = (x_stop - x0c + 0.02) / vt + 1.5
    still, xs = 0, []
    for k in range(int(T1 / dt)):
        v = float(d.qvel[jadr]) if jadr is not None else 0.0
        F = float(np.clip(kv * (vt - v), -Fmax, Fmax))
        d.xfrc_applied[bid, :3] = [F, 0.0, 0.0]
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            issues.append("simulation diverged during the push: check the bead's mass and contact solref against the timestep")
            break
        pk = penetration(); peak_pen = max(peak_pen, pk)
        if abs(v) < 0.01:
            max_pen = max(max_pen, pk)
        xs.append(float(centre()[0]))
        track_pins()
        still = still + 1 if abs(v) < 0.002 else 0
        if still * dt > 0.3 and xs[-1] > x_tip + 0.01:
            break
    xc = float(centre()[0]) if np.all(np.isfinite(d.qpos)) else float("nan")
    passed_tip = bool(np.isfinite(xc) and xc > x_tip + h / 2 + 0.002 and dist_to_polyline(centre(), cable_pts()) <= bore + tol)
    travel = float(xc - x0c) if np.isfinite(xc) else float("nan")
    gap = float(x_stop - xc) if np.isfinite(xc) else float("nan")
    lo = h / 2 + cap_r + 0.002
    stopped = bool(has_stop and np.isfinite(gap) and -1e-4 <= gap <= lo)
    if not passed_tip:
        issues.append(f"the bead did not thread onto the cable tip (centre at x={xc:.4f} m, tip at 0): the bore radius {bore:.4f} m "
                      f"must exceed the cable radius {r_c:.4f} m with clearance, or the push ({Fmax} N) is too weak")
    elif travel < 0.5 * (x_stop - x_tip):
        issues.append(f"the bead stuck after {travel:.4f} m of the {x_stop:.4f} m thin section: friction or a tight bore holds it; "
                      "lower the bead-cable friction or widen the bore")
    if has_stop and passed_tip and not stopped:
        issues.append(f"the bead did not come to rest against the stop section: gap {gap:.4f} m (allowed 0..{lo:.4f}); "
                      + ("it passed into the stop, so the stop radius must exceed the bore radius" if gap < 0 else "it stopped short"))
    # phase 2: sideways push while held against the stop
    max_off = float("nan")
    if passed_tip and np.all(np.isfinite(d.qpos)):
        max_off = 0.0
        for k in range(int(0.5 / dt)):
            v = float(d.qvel[jadr]) if jadr is not None else 0.0
            d.xfrc_applied[bid, :3] = [float(np.clip(kv * (vt - v), -Fmax, Fmax)), Flat, 0.0]
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)):
                issues.append("simulation diverged during the sideways push")
                break
            pk = penetration(); peak_pen = max(peak_pen, pk)
            if k * dt > 0.2 and float(np.linalg.norm(d.qvel[jadr:jadr + 3])) < 0.01:
                max_pen = max(max_pen, pk)
            if k % 5 == 0:
                max_off = max(max_off, dist_to_polyline(centre(), cable_pts()))
            track_pins()
    threaded = bool(np.isfinite(max_off) and max_off <= bore + tol)
    if passed_tip and not threaded:
        issues.append(f"the bead left the cable under a {Flat} N sideways push (centre {max_off:.4f} m from the cable, bound "
                      f"{bore + tol:.4f} m): the cable is not passing through its bore; check bore_radius and the bead's wall geoms")
    if max_pen > pen_max:
        issues.append(f"bead and cable interpenetrate by {max_pen * 1000:.2f} mm (allowed {pen_max * 1000:.2f} mm): stiffen the "
                      "contact (solref) or lower the push force")
    if pin_drift[0] > 0.01:
        # the trial's own premise: a cable that swings away under the push says nothing about the bead. The threading
        # bound is taken against the cable's current centreline, so a slow give of the pinned ends (2 mm under 0.2 N on a
        # 1.25 mm cord, as the joints' dry friction creeps) does not corrupt it; the 10 cm swing of soft pins did
        issues.append(f"the pinned cable ends moved {pin_drift[0] * 1000:.1f} mm during the trial, so it is not a test of the bead; "
                      "lower lateral_force_N / push_force_N or the timestep")
    rep = {"ok": not issues, "issues": issues, "cable_pin_drift_m": float(pin_drift[0]),
           "insertion": {"passed_tip": passed_tip, "travel_m": travel, "final_gap_to_stop_m": gap, "stopped_at_stop": stopped,
                         "has_stop": bool(has_stop)},
           "lateral": {"max_offset_from_centreline_m": max_off, "bound_m": bore + tol, "threaded": threaded},
           "max_penetration_m": float(max_pen), "peak_transient_penetration_m": float(peak_pen), "initial_penetration_m": float(pen0),
           "params": {"bore_radius": bore, "bead_height": h, "cable_radius": r_c, "cap_radius": cap_r, "cap_length": cap_len,
                      "cable_length": L, "bead_mass_kg": mass},
           "merged_xml": str(merged)}
    if kw.get("out_json"):
        Path(kw["out_json"]).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    print(json.dumps(bead_on_cable_slide_test(a[0], float(a[1]), float(a[2]), cable_radius=float(a[3])), indent=1))
