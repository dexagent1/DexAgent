"""Stud stack seat test: does a toy brick seat on another brick's studs, centre itself, interlock and stay?

Force-only trials on two brick assets (any MJCF whose root body is a free brick with
its bottom face at the body origin, such as construct_stud_brick builds). The lower
brick stands on a floor; the upper one is released above it and must come to rest
SEATED (its bottom on the lower brick's top face, not perched on the studs), level,
and where the studs put it. Then a capped sideways push must not slide it off: the
studs hold it, where a plain block on a block would skate.
"""
from __future__ import annotations

TOOL = {
    "name": "stud_stack_seat_test",
    "title": "Stud stack seat test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Drops one brick asset onto another (aligned, misaligned by a few millimetres, and shifted by whole stud pitches) and checks that it seats over the studs, centres itself, stays level and resists a sideways shove, so stacked bricks interlock instead of resting block-on-block.",
    "description": (
        "Composes a test world from two brick MJCFs (the lower and the upper, the same file allowed) with a "
        "floor: names are prefixed, the lower brick stands on the floor at the origin, the upper brick is "
        "placed above it at each trial offset (dx, dy in the lower brick's frame, metres) with a small drop gap "
        "and released under gravity, then pressed down with press_N for press_s at the centre of the part of its "
        "top that lies over the lower brick (a hand pressing the supported half of an overhanging brick). A trial passes when, "
        "after settling, the upper brick's bottom is within seat_tol_m of the lower brick's top face "
        "(body_height_m, read from the built dict or measured from the lower brick's highest non-stud geom), "
        "its tilt is under max_tilt_deg, and its lateral position is within max_lateral_m of the nearest "
        "stud-grid position (offset rounded to whole pitches). A shove trial then pushes the seated aligned "
        "upper brick sideways with shove_N (default 0.75 x its weight) for shove_s; it passes when the upper brick moves less than "
        "max_shove_m relative to the lower one (the lower brick may slide along with it on the floor). Both bricks must be free bodies. The report gives "
        "every trial's measured seat height error, tilt, lateral error and the shove displacement."
    ),
    "inputs": [
        {"name": "lower", "type": "dict | path", "description": "construct_stud_brick build dict (or its params json, or an MJCF path) of the bottom brick."},
        {"name": "upper", "type": "dict | path | None", "description": "The brick dropped on top; default the same as lower."},
        {"name": "offsets_m", "type": "list[[dx, dy]] | None", "description": "Release offsets of the upper brick in the lower brick's frame. Default: aligned, a lead-in offset of 0.3 x stud radius on both axes, and one pitch along x."},
        {"name": "drop_gap_m", "type": "float", "description": "Release height of the upper brick's bottom above the lower brick's stud tops. Default 0.002"},
        {"name": "press_N", "type": "float", "description": "Downward press on the upper brick after release, applied over its supported part (a hand pressing it on). Default 3.0"},
        {"name": "press_s", "type": "float", "description": "Press duration. Default 0.3"},
        {"name": "settle_s", "type": "float", "description": "Settling time after the press. Default 1.0"},
        {"name": "seat_tol_m", "type": "float", "description": "Allowed seat height error. Default 0.0015"},
        {"name": "max_tilt_deg", "type": "float", "description": "Allowed tilt of the seated brick. Default 3.0"},
        {"name": "max_lateral_m", "type": "float", "description": "Allowed distance from the nearest stud-grid position. Default 0.0015"},
        {"name": "shove_N", "type": "float | None", "description": "Sideways shove on the seated brick at its centre of mass, newtons (0 skips). Default None = 0.75 x the upper brick's weight, more than plain plastic-on-plastic friction holds"},
        {"name": "shove_s", "type": "float", "description": "Shove duration. Default 0.5"},
        {"name": "max_shove_m", "type": "float", "description": "Allowed displacement under the shove. Default 0.002"},
        {"name": "timestep", "type": "float", "description": "Simulation timestep. Default 0.001"},
        {"name": "out_json", "type": "path | None", "description": "Optional report file."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; trials: [{offset, seated, seat_err_m, tilt_deg, lateral_err_m, ok}]; shove: {force_N, displacement_m, ok}; body_height_m; pitch_m."},
    ],
    "requirements": ["mujoco>=3.1", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["brick", "lego", "stud", "stack", "seat", "interlock", "drop test", "mechanism test"],
    "status": "tested",
    "entrypoint": "stud_stack_seat_test",
}

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REF_ATTRS = ("material", "mesh", "texture", "hfield")


def _load_built(b):
    if b is None:
        return None
    if isinstance(b, dict):
        return dict(b)
    p = Path(b)
    if not p.exists():
        raise ValueError(f"brick file not found: {b}")
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        d = d.get("built", d)
        if "xml" not in d:
            raise ValueError(f"{b}: json has no 'xml' entry")
        return d
    return {"xml": str(p)}


def _prefix_tree(root, pre):
    names = {}
    for el in root.iter():
        n = el.get("name")
        if n is not None and el.tag != "mujoco":
            names[n] = pre + n
            el.set("name", pre + n)
    for el in root.iter():
        for a in _REF_ATTRS + ("joint", "body1", "body2", "geom1", "geom2", "site", "body"):
            v = el.get(a)
            if v is not None and v in names and el.get("name") != names.get(v):
                el.set(a, names[v])
    return root


def _compose(parts, timestep):
    """parts: list of (xml_path, prefix, pos[3]). Returns (xml string, root body names)."""
    world = ET.Element("worldbody")
    asset = ET.Element("asset")
    roots = []
    for path, pre, pos in parts:
        r = _prefix_tree(ET.parse(path).getroot(), pre)
        for a in r.findall("asset"):
            for ch in a:
                asset.append(copy.deepcopy(ch))
        wb = r.find("worldbody")
        if wb is None:
            raise ValueError(f"{path}: no worldbody")
        bodies = [b for b in wb.findall("body")]
        if len(bodies) != 1:
            raise ValueError(f"{path}: expected exactly one root body, found {len(bodies)}")
        b = copy.deepcopy(bodies[0])
        b.set("pos", " ".join(f"{float(v):.6g}" for v in pos))
        b.attrib.pop("quat", None)
        world.append(b)
        roots.append(b.get("name"))
    top = ET.Element("mujoco", model="stud_stack_seat_test")
    ET.SubElement(top, "compiler", angle="radian")
    ET.SubElement(top, "option", timestep=f"{timestep}")
    sz = ET.SubElement(top, "size"); sz.set("memory", "64M")
    if len(asset):
        top.append(asset)
    world.insert(0, ET.Element("geom", name="floor", type="plane", size="1 1 0.01", friction="0.8 0.01 0.001"))
    top.append(world)
    return ET.tostring(top, encoding="unicode"), roots


def _brick_dims(model, body_id, built):
    import mujoco
    H = built.get("top_z_m") or (built.get("size_m") or [None, None, None])[2]
    P = built.get("pitch")
    rs = built.get("stud_radius_m")
    if H is None:
        # highest top of a colliding box geom of the body = the top face
        tops = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body_id and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and model.geom_contype[g]:
                tops.append(model.geom_pos[g][2] + model.geom_size[g][2])
        if not tops:
            raise ValueError("cannot find the brick's top face; pass a construct_stud_brick build dict")
        H = max(tops)
    return float(H), (float(P) if P else None), (float(rs) if rs else None)


def stud_stack_seat_test(lower, upper=None, offsets_m=None, drop_gap_m=0.002, press_N=3.0, press_s=0.3,
                         settle_s=1.0, seat_tol_m=0.0015, max_tilt_deg=3.0, max_lateral_m=0.0015, shove_N=None,
                         shove_s=0.5, max_shove_m=0.002, timestep=0.001, out_json=None):
    import mujoco
    lo = _load_built(lower)
    if lo is None:
        raise ValueError("lower brick is required")
    up = _load_built(upper) or dict(lo)
    for k, v in (("drop_gap_m", drop_gap_m), ("settle_s", settle_s), ("timestep", timestep)):
        if not float(v) > 0:
            raise ValueError(f"{k} must be > 0")
    issues, trials = [], []
    report = {"ok": False, "issues": issues, "trials": trials}

    # dims from a probe compile
    probe_xml, roots = _compose([(lo["xml"], "L_", [0, 0, 0]), (up["xml"], "U_", [0, 0, 1.0])], timestep)
    try:
        m = mujoco.MjModel.from_xml_string(probe_xml)
    except Exception as e:  # report, do not raise: a broken asset is a verdict
        issues.append(f"the two bricks do not compile together: {e}")
        return report
    bl, bu = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, r) for r in roots)
    for b, nm in ((bl, "lower"), (bu, "upper")):
        if m.body_jntnum[b] < 1 or m.jnt_type[m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
            issues.append(f"the {nm} brick's root body has no free joint; build it with free=True")
    if issues:
        return report
    H, P, rs = _brick_dims(m, bl, lo)
    hs = float(lo.get("stud_height_m") or 0.0)
    sz_l = lo.get("size_m"); sz_u = up.get("size_m")
    report.update(body_height_m=H, pitch_m=P)
    if offsets_m is None:
        lead = 0.3 * (rs or 0.004)
        offsets_m = [[0.0, 0.0], [lead, lead]] + ([[P, 0.0]] if P else [])

    def run(off, press, shove=None):
        xml, rts = _compose([(lo["xml"], "L_", [0, 0, 0]),
                             (up["xml"], "U_", [off[0], off[1], H + hs + drop_gap_m])], timestep)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        b_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rts[0])
        b_u = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rts[1])
        mujoco.mj_forward(model, data)
        n_press = int(press_s / timestep)
        n_set = int(settle_s / timestep)
        # press where a hand would: on the upper brick's top, over the part of it that the lower brick supports
        # (the centre of the two footprints' overlap), so an overhanging brick is not levered off by the test
        ov = np.array([0.0, 0.0])
        if sz_l is not None and sz_u is not None:
            lo_x = max(-sz_l[0] / 2, off[0] - sz_u[0] / 2); hi_x = min(sz_l[0] / 2, off[0] + sz_u[0] / 2)
            lo_y = max(-sz_l[1] / 2, off[1] - sz_u[1] / 2); hi_y = min(sz_l[1] / 2, off[1] + sz_u[1] / 2)
            if hi_x > lo_x and hi_y > lo_y:
                ov = np.array([(lo_x + hi_x) / 2, (lo_y + hi_y) / 2])
        for i in range(n_press + n_set):
            data.xfrc_applied[b_u] = 0
            if i < n_press and press > 0:
                F = np.array([0.0, 0.0, -press])
                pt = np.array([data.xpos[b_l][0] + ov[0], data.xpos[b_l][1] + ov[1], data.xpos[b_u][2] + H])
                data.xfrc_applied[b_u, :3] = F
                data.xfrc_applied[b_u, 3:] = np.cross(pt - data.xipos[b_u], F)
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)):
                return None, "simulation diverged"
        res = {"pos_l": data.xpos[b_l].copy(), "pos_u": data.xpos[b_u].copy(),
               "R_u": data.xmat[b_u].reshape(3, 3).copy(), "R_l": data.xmat[b_l].reshape(3, 3).copy()}
        if shove is not None:
            # the lower brick is free too: the shove is judged on the RELATIVE motion, so a brick that drags
            # the one below along passes and a brick that skates over it fails
            p0 = data.xpos[b_u].copy()
            n_sh = int(shove_s / timestep)
            peak = 0.0
            for i in range(n_sh):
                data.xfrc_applied[b_u] = 0
                data.xfrc_applied[b_u, 0] = shove
                data.xfrc_applied[b_u, 2] = 0
                mujoco.mj_step(model, data)
                peak = max(peak, float(np.linalg.norm(data.xpos[b_u][:2] - p0[:2])))
            res["shove_disp"] = peak
            res["shove_rel"] = float(np.linalg.norm((data.xpos[b_u] - data.xpos[b_l])[:2] - (p0 - res["pos_l"])[:2]))
        return res, None

    for off in offsets_m:
        off = [float(off[0]), float(off[1])]
        res, err = run(off, press_N)
        if res is None:
            trials.append({"offset": off, "ok": False, "error": err}); issues.append(f"offset {off}: {err}"); continue
        R_l = res["R_l"]
        rel = R_l.T @ (res["pos_u"] - res["pos_l"])
        seat_err = float(rel[2] - H)
        tilt = float(math.degrees(math.acos(max(-1.0, min(1.0, res["R_u"][2, 2])))))
        if P:
            target = np.round(rel[:2] / P) * P
        else:
            target = np.zeros(2)
        lat = float(np.linalg.norm(rel[:2] - target))
        ok = abs(seat_err) <= seat_tol_m and tilt <= max_tilt_deg and lat <= max_lateral_m
        t = {"offset": off, "rel_pos": [round(float(v), 5) for v in rel], "seat_err_m": round(seat_err, 5),
             "tilt_deg": round(tilt, 3), "lateral_err_m": round(lat, 5), "grid_target": [round(float(v), 5) for v in target],
             "seated": abs(seat_err) <= seat_tol_m, "ok": bool(ok)}
        trials.append(t)
        if not ok:
            why = []
            if abs(seat_err) > seat_tol_m:
                why.append(f"rests {seat_err * 1000:+.1f} mm from the seated height (perched on the studs or sunk)")
            if tilt > max_tilt_deg:
                why.append(f"tilted {tilt:.1f} deg")
            if lat > max_lateral_m:
                why.append(f"{lat * 1000:.1f} mm off the stud grid")
            issues.append(f"offset {off}: " + "; ".join(why) +
                          " -> check that the underside clearance and the stud lead-in let the brick slide over the studs")
    shove = None
    if shove_N is None:
        shove_N = 0.75 * float(m.body_subtreemass[bu]) * 9.81
    if shove_N and shove_N > 0:
        res, err = run([0.0, 0.0], press_N, shove=float(shove_N))
        if res is None:
            shove = {"ok": False, "error": err}; issues.append(f"shove: {err}")
        else:
            d = float(res["shove_rel"])
            shove = {"force_N": float(shove_N), "displacement_m": round(d, 5), "peak_m": round(float(res["shove_disp"]), 5),
                     "ok": d <= max_shove_m}
            if d > max_shove_m:
                issues.append(f"a {shove_N} N sideways shove moved the seated brick {d * 1000:.1f} mm: the studs do not "
                              f"interlock (underside too loose or studs too low)")
    report["shove"] = shove
    report["ok"] = not issues
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=1))
    return report
