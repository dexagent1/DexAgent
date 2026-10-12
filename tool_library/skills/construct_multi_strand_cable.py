TOOL = {
    "name": "construct_multi_strand_cable",
    "title": "Construct multi-strand cable (strands joined at one end, optionally plaited)",
    "type": "skill",
    "category": "Object Reconstruction",
    "summary": "Several flexible strands (flat ribbons or round cords) joined at one end by a small knot body, each an elastic MuJoCo cable laid along a measured tail polyline, optionally plaited near the joined end into a three-strand braid with a chosen number of alternating crossings.",
    "description": (
        "Builds one free object: a junction body (the knot, stitch or tape that joins the strands) with N elastic "
        "cables (mujoco.elasticity.cable) attached to it by ball joints, so the link graph is one connected tree "
        "with N free ends. Each strand has its own width, thickness, colour and mass per metre; profile 'flat' "
        "makes every link a box (width x thickness, a ribbon or strap that lies flat), 'round' a capsule of "
        "radius width/2 (a cord). Strand centrelines come from the caller: either a full 'centreline' per strand "
        "starting at the junction, or, with 'braid', a three-strand plait generated along a braid axis polyline "
        "(phases 0, 120, 240 degrees: lateral a*sin(t+phi), height b*(1 -/+ sin 2(t+phi))/2, which gives exactly "
        "`crossings` alternating crossings, outer strand over the middle for style 'over', under for 'under') "
        "followed by each strand's measured 'tail' polyline from the braid's lower end to its tip. Strands are "
        "matched to the plait positions by where their tails go (left to right across the axis). Inside the braid "
        "the link boxes are narrowed by braid_width_frac (a ribbon bunches in a tight plait) and the lateral "
        "amplitude and lift are chosen so that no two strands overlap at rest; crossings between tails on the "
        "table are lifted locally, the later strand over the earlier. The asset is written in the frame of the "
        "positions given (stage it at x = y = 0, yaw 0 when they are rig coordinates). Self-collision between "
        "strands is on (contype/conaffinity 1); adjacent links are excluded by MuJoCo's parent-child rule. "
        "MuJoCo's <composite type=\"cable\"> adds a white skin over box links that hides their colour and replaces the geoms in segmentation renders (silhouette checks then see only the end links); this version deletes those skins (keep_skin=False) or paints them the strand colour (keep_skin=True). Assumes the strands rest on a horizontal support at z = 0."
    ),
    "inputs": [
        {"name": "out_dir", "type": "path", "description": "Folder that receives <name>.xml and <name>_params.json."},
        {"name": "name", "type": "str", "description": "Asset name and body prefix. Default 'strands'."},
        {"name": "strands", "type": "list[dict]", "description": "One dict per strand: width (m, required), rgba, label, thickness, mass_per_m, and either centreline [[x,y,z],...] from the junction (no braid) or tail [[x,y(,z)],...] from the braid's lower end to the strand's tip (with braid)."},
        {"name": "profile", "type": "str", "description": "'flat' (box links, ribbons) or 'round' (capsule links, cords). Default 'flat'."},
        {"name": "thickness", "type": "float", "description": "Default link thickness for flat strands, metres. Default 0.0012."},
        {"name": "junction_pos", "type": "list[2|3] | None", "description": "Where the strands join (x, y on the support): the strands start there (a braid is moved so its axis starts there); default the braid axis start or the mean first centreline point."},
        {"name": "junction_size", "type": "list[3] | None", "description": "Junction box half-extents (along the strands, across, up), metres; default sized to hold the strands."},
        {"name": "junction_rgba", "type": "list[4] | None", "description": "Junction colour; default the first strand's."},
        {"name": "braid", "type": "dict | None", "description": "{axis: [[x,y],...] from the junction, crossings: int, lateral_m: amplitude or None (auto), lift_m: None (auto), style: 'over' | 'under'}; exactly three strands."},
        {"name": "braid_width_frac", "type": "float", "description": "Link width inside the braid as a fraction of the strand width. Default 0.6."},
        {"name": "segment_m", "type": "float", "description": "Target link length, metres. Default 0.012."},
        {"name": "bend_stiffness", "type": "float", "description": "Cable plugin bending modulus, Pa. Default 2e5."},
        {"name": "twist_stiffness", "type": "float", "description": "Cable plugin twisting modulus, Pa. Default 1e5."},
        {"name": "joint_damping", "type": "float", "description": "Link joint damping. Default 0.0005."},
        {"name": "joint_armature", "type": "float", "description": "Link joint armature. Default 2e-6."},
        {"name": "mass_per_m", "type": "float", "description": "Default strand mass per metre, kg/m (satin ribbon about 0.004-0.01). Default 0.006."},
        {"name": "junction_mass", "type": "float", "description": "Junction mass, kg. Default 0.002."},
        {"name": "friction", "type": "list[3]", "description": "Sliding, torsional, rolling friction of every link. Default [0.6, 0.005, 0.0001]."},
        {"name": "condim", "type": "int", "description": "Contact dimensionality. Default 3."},
        {"name": "contact_solref", "type": "list[2]", "description": "Contact solref. Default [0.004, 1.0]."},
        {"name": "free", "type": "bool", "description": "Junction is a free body (else fixed where built). Default True."},
        {"name": "lift_tail_crossings", "type": "bool", "description": "Lift a tail where it crosses another strand's tail. Default True."},
        {"name": "include_floor", "type": "bool", "description": "Add a floor plane (for testing the asset alone). Default False."},
        {"name": "timestep", "type": "float", "description": "Model timestep, seconds. Default 0.002."},
        {"name": "keep_skin", "type": "bool", "description": "Keep the composite skins of flat strands, painted the strand colour, instead of deleting them. Default False."},
    ],
    "outputs": [
        {"name": "built", "type": "dict", "description": "xml, params_json, root_body (the junction), bodies (all), joint (the junction free joint or None), strands {label: {first, last, links, n_links, length_m, width_m, thickness_m, rgba, centreline (N,3) in the build frame}}, braid {crossings, lateral_m, lift_m, style, order} or None, tail_crossings_lifted, n_links, total_mass_kg."},
    ],
    "status": "tested",
    "entrypoint": "construct_multi_strand_cable",
    "requirements": ["mujoco>=3.2", "numpy"],
    "simulators": ["MuJoCo"],
    "tags": ["rope", "cable", "ribbon", "braid", "plait", "strands", "deformable", "parametric", "object reconstruction"],
    "used_in": ["EgoDex braid_unbraid"],
    "cli": "python -m tool_library.skills.construct_multi_strand_cable out_dir params.json",
}

import json
import re
from pathlib import Path

import numpy as np

from tool_library.common.mjcf import FLOOR_XML, LIGHTS_XML, mjcf_header, rgba, save_params, vec

DEFAULTS = {
    "name": "strands",
    "strands": None,
    "profile": "flat",
    "thickness": 0.0012,
    "junction_pos": None,
    "junction_size": None,
    "junction_rgba": None,
    "braid": None,
    "braid_width_frac": 0.6,
    "segment_m": 0.012,
    "bend_stiffness": 2e5,
    "twist_stiffness": 1e5,
    "joint_damping": 0.0005,
    "joint_armature": 2e-6,
    "mass_per_m": 0.006,
    "junction_mass": 0.002,
    "friction": [0.6, 0.005, 0.0001],
    "condim": 3,
    "contact_solref": [0.004, 1.0],
    "free": True,
    "lift_tail_crossings": True,
    "include_floor": False,
    "timestep": 0.002,
    "keep_skin": False,
}
STRAND_KEYS = {"width", "rgba", "label", "thickness", "mass_per_m", "centreline", "tail"}
BRAID_KEYS = {"axis", "crossings", "lateral_m", "lift_m", "style"}


def _xy3(P, z=None):
    P = np.asarray(P, float)
    if P.ndim != 2 or P.shape[1] not in (2, 3) or len(P) < 2:
        raise ValueError("a polyline must be a list of >= 2 [x, y] or [x, y, z] points")
    if P.shape[1] == 2:
        P = np.column_stack([P, np.full(len(P), 0.0 if z is None else z)])
    return P


def _clean(P):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-7])
    return P[keep]


def _resample(P, step):
    P = _clean(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(3, int(round(s[-1] / step)) + 1)
    w = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(w, s, P[:, i]) for i in range(3)], 1), float(s[-1])


def _seg_intersect(p, q, r, s):
    """2D segment intersection parameters (t on p->q, u on r->s) or None."""
    d1, d2 = q - p, s - r
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return None
    w = r - p
    t = (w[0] * d2[1] - w[1] * d2[0]) / den
    u = (w[0] * d1[1] - w[1] * d1[0]) / den
    if 0 < t < 1 and 0 < u < 1:
        return t, u
    return None


def plait_paths(axis, crossings, widths, thickness, lateral=None, lift=None, style="over", width_frac=0.6, per_crossing=14):
    """Three plait centrelines along an axis polyline (z = 0 support). Returns (paths, info)."""
    A = _xy3(axis)[:, :2]
    seg = np.linalg.norm(np.diff(A, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-3:
        raise ValueError("braid.axis is shorter than a millimetre")
    crossings = int(crossings)
    if crossings < 1:
        raise ValueError("braid.crossings must be >= 1")
    if style not in ("over", "under"):
        raise ValueError("braid.style must be 'over' (outer strand over the middle) or 'under'")
    n = max(25, crossings * per_crossing + 1)
    u = np.linspace(0.0, 1.0, n)
    pts = np.stack([np.interp(u * s[-1], s, A[:, i]) for i in range(2)], 1)
    tang = np.gradient(pts, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)                  # left of the direction of travel
    t = u * crossings * np.pi / 3.0
    phases = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
    sign = -1.0 if style == "over" else 1.0
    we = [float(w) * float(width_frac) for w in widths]
    if lateral is None:
        # where two strands are level (A + B = pi/2) their lateral gap is 1.225 a: keep their edges apart there with margin
        lateral = max((we[i] + we[j]) / 2.0 for i in range(3) for j in range(i + 1, 3))
    lateral = float(lateral)
    lat = [lateral * np.sin(t + ph) for ph in phases]
    zf = [(1.0 + sign * np.sin(2 * (t + ph))) / 2.0 for ph in phases]
    need = 2.0 * thickness
    for i in range(3):
        for j in range(i + 1, 3):
            close = np.abs(lat[i] - lat[j]) < (we[i] + we[j]) / 2.0
            if close.any():
                dz = np.abs(zf[i] - zf[j])[close]
                need = max(need, 1.7 * thickness / max(float(dz.min()), 1e-3))
    b = float(lift) if lift is not None else need
    if b > 0.03:
        raise ValueError(f"the plait needs a lift of {b * 1000:.0f} mm to keep the strands apart: raise braid.lateral_m "
                         f"(now {lateral * 1000:.1f} mm) or lower braid_width_frac")
    paths = [np.column_stack([pts + lat[k][:, None] * nrm, thickness / 2.0 + b * zf[k]]) for k in range(3)]
    end_lat = [float(lat[k][-1]) for k in range(3)]
    return paths, {"lateral_m": round(lateral, 5), "lift_m": round(b, 5), "crossings": crossings, "style": style,
                   "end_lateral_m": [round(v, 5) for v in end_lat], "axis_length_m": round(float(s[-1]), 4),
                   "end_normal": nrm[-1].tolist(), "end_point": pts[-1].tolist()}


def _bump_tail_crossings(cl, tail_start, thickness, widths):
    """Lift later strands over earlier ones where their tails cross on the support. Returns the count."""
    lifted = 0
    k = len(cl)
    for b in range(k):
        for a in range(b):
            Pa, Pb = cl[a], cl[b]
            hits = []
            for i in range(tail_start[b], len(Pb) - 1):
                for j in range(tail_start[a], len(Pa) - 1):
                    r = _seg_intersect(Pb[i, :2], Pb[i + 1, :2], Pa[j, :2], Pa[j + 1, :2])
                    if r is not None:
                        hits.append(Pb[i, :2] + r[0] * (Pb[i + 1, :2] - Pb[i, :2]))
            for h in hits:
                d = np.linalg.norm(Pb[:, :2] - h, axis=1)
                half = max(widths[a], widths[b]) * 1.2 + 0.01
                bump = np.where(d < half, 0.5 * (1.0 + np.cos(np.pi * np.minimum(d / half, 1.0))), 0.0)
                bump = np.where(np.arange(len(Pb)) >= tail_start[b], bump, 0.0)
                Pb[:, 2] = np.maximum(Pb[:, 2], thickness / 2.0 + 1.6 * thickness * bump)
                lifted += 1
    return lifted


def _quat_yaw(yaw):
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def construct_multi_strand_cable(out_dir, **params):
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; valid: {sorted(DEFAULTS)}")
    p = dict(DEFAULTS); p.update(params)
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(p["name"]))
    if p["profile"] not in ("flat", "round"):
        raise ValueError("profile must be 'flat' or 'round'")
    strands = p["strands"]
    if not strands or not isinstance(strands, (list, tuple)):
        raise ValueError("strands: give a list of at least one strand dict (width, and centreline or tail)")
    for k, st in enumerate(strands):
        bad = sorted(set(st) - STRAND_KEYS)
        if bad:
            raise ValueError(f"strands[{k}] has unknown key(s) {bad}; valid: {sorted(STRAND_KEYS)}")
        if "width" not in st or float(st["width"]) <= 0:
            raise ValueError(f"strands[{k}] needs a positive width (metres)")
    braid = p["braid"]
    T0 = float(p["thickness"])
    labels = [re.sub(r"[^A-Za-z0-9_]", "_", str(st.get("label", f"s{k}"))) for k, st in enumerate(strands)]
    if len(set(labels)) != len(labels):
        raise ValueError("strand labels must be unique")
    widths = [float(st["width"]) for st in strands]
    thick = [float(st.get("thickness", T0)) if p["profile"] == "flat" else float(st["width"]) for st in strands]
    tmax = max(thick)
    seg = float(p["segment_m"])
    if seg <= 0.002:
        raise ValueError("segment_m must be above 2 mm")

    braid_info, tail_start, cl = None, [], []
    jp_given = None                     # resolved first: the strands are laid out from the junction they start at
    if p["junction_pos"] is not None:
        jp = np.asarray(p["junction_pos"], float)
        if jp.shape not in ((2,), (3,)):
            raise ValueError("junction_pos must be [x, y] or [x, y, z]")
        jp_given = jp[:2]
    if braid is not None:
        bad = sorted(set(braid) - BRAID_KEYS)
        if bad:
            raise ValueError(f"braid has unknown key(s) {bad}; valid: {sorted(BRAID_KEYS)}")
        if len(strands) != 3:
            raise ValueError("braid needs exactly three strands (a three-strand plait)")
        if "axis" not in braid or "crossings" not in braid:
            raise ValueError("braid needs axis and crossings")
        for k, st in enumerate(strands):
            if "tail" not in st:
                raise ValueError(f"strands[{k}] needs a tail polyline (braid lower end -> tip) when braid is given")
        paths, braid_info = plait_paths(braid["axis"], braid["crossings"], widths, tmax, braid.get("lateral_m"), braid.get("lift_m"),
                                        braid.get("style", "over"), float(p["braid_width_frac"]))
        # match strands to plait positions: tails sorted by where they go across the axis (left = +normal)
        nrm, endp = np.array(braid_info["end_normal"]), np.array(braid_info["end_point"])
        tail_side = [float((_xy3(st["tail"])[-1, :2] - endp) @ nrm) for st in strands]
        order_strands = list(np.argsort(tail_side)[::-1])               # leftmost tail first
        order_pos = list(np.argsort(braid_info["end_lateral_m"])[::-1])  # leftmost plait position first
        assign = {int(order_strands[r]): int(order_pos[r]) for r in range(3)}
        braid_info["order"] = {labels[k]: int(assign[k]) for k in range(3)}
        for k, st in enumerate(strands):
            bp = paths[assign[k]]
            tail = _xy3(st["tail"], z=thick[k] / 2.0)
            tail = tail.copy(); tail[:, 2] = np.maximum(tail[:, 2], thick[k] / 2.0)
            off = bp[-1, :2] - tail[0, :2]                               # tail starts where this strand leaves the plait
            ramp = np.linspace(1.0, 0.0, len(tail))[:, None]
            tail[:, :2] = tail[:, :2] + off * ramp
            full = np.vstack([bp, tail[1:]])
            r, _ = _resample(full, seg)
            # index where the tail starts in the resampled polyline
            sb = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(_clean(bp), axis=0), axis=1))])[-1]
            sr = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(r, axis=0), axis=1))])
            tail_start.append(int(np.searchsorted(sr, sb)))
            cl.append(r)
        jxy = np.asarray(_xy3(braid["axis"])[0, :2])
        if jp_given is not None:            # the plait (and its tails) moved so its axis starts at the junction
            for c in cl:
                c[:, :2] += jp_given - jxy
            jxy = jp_given
        a0 = _xy3(braid["axis"])
        d0 = a0[1, :2] - a0[0, :2]
        yaw = float(np.arctan2(d0[1], d0[0]))
    else:
        for k, st in enumerate(strands):
            if "centreline" not in st:
                raise ValueError(f"strands[{k}] needs a centreline from the junction when no braid is given")
            c = _xy3(st["centreline"], z=thick[k] / 2.0).copy()
            c[:, 2] = np.maximum(c[:, 2], thick[k] / 2.0)
            cl.append(c); tail_start.append(0)
        firsts = np.array([c[0, :2] for c in cl])
        jxy = firsts.mean(0) if jp_given is None else jp_given
        d0 = np.mean([c[-1, :2] - c[0, :2] for c in cl], axis=0)
        yaw = float(np.arctan2(d0[1], d0[0])) if np.linalg.norm(d0) > 1e-6 else 0.0
        # side by side at the junction, ordered by where each strand goes across the mean direction (left first)
        nrm0 = np.array([-np.sin(yaw), np.cos(yaw)])
        side = [float((c[-1, :2] - jxy) @ nrm0) for c in cl]
        order = list(np.argsort(side)[::-1])
        gap = 0.0015
        span = sum(widths) + gap * (len(cl) - 1)
        off, cur = {}, span / 2.0
        for k in order:
            off[int(k)] = cur - widths[k] / 2.0
            cur -= widths[k] + gap
        for k in range(len(cl)):
            c = cl[k].copy()
            c[0, :2] = jxy + off[k] * nrm0
            lead = c[0].copy(); lead[:2] = lead[:2] + 0.02 * np.array([np.cos(yaw), np.sin(yaw)])
            if np.linalg.norm(c[1, :2] - c[0, :2]) > 0.025:
                c = np.vstack([c[:1], lead[None], c[1:]])
            cl[k], _ = _resample(c, seg)
    lifted = _bump_tail_crossings(cl, tail_start, tmax, widths) if p["lift_tail_crossings"] else 0

    # junction box
    if p["junction_size"] is not None:
        js = [float(v) for v in p["junction_size"]]
        if len(js) != 3 or min(js) <= 0:
            raise ValueError("junction_size must be three positive half-extents")
    else:
        spread = max(float(np.max(np.abs((c[0, :2] - jxy) @ np.array([-np.sin(yaw), np.cos(yaw)])))) for c in cl)
        top = max(float(c[0, 2]) for c in cl)
        js = [0.008, spread + max(widths) * 0.5 * float(p["braid_width_frac"] if braid is not None else 1.0), max(top / 2.0 + tmax, 0.003)]
    jz = js[2]
    jpos = np.array([jxy[0], jxy[1], jz])
    c, s_ = np.cos(yaw), np.sin(yaw)
    Rj = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    jrgba = p["junction_rgba"] or (strands[0].get("rgba") or [0.8, 0.3, 0.3, 1.0])

    comps, info = [], {}
    total_mass = float(p["junction_mass"])
    for k, st in enumerate(strands):
        lab = labels[k]; pre = f"{name}_{lab}"
        loc = (cl[k] - jpos) @ Rj
        L = float(np.linalg.norm(np.diff(cl[k], axis=0), axis=1).sum())
        mpm = float(st.get("mass_per_m", p["mass_per_m"]))
        if p["profile"] == "flat":
            area = widths[k] * thick[k]
            gsize = f"0.001 {widths[k] / 2:.5f} {thick[k] / 2:.5f}"; gtype = "box"
        else:
            area = np.pi * (widths[k] / 2) ** 2
            gsize = f"{widths[k] / 2:.5f}"; gtype = "capsule"
        dens = mpm / area
        total_mass += mpm * L
        col = st.get("rgba") or [0.8, 0.2, 0.2, 1.0]
        comps.append(f"""
      <composite prefix="{pre}" type="cable" initial="ball" vertex="{' '.join(f'{v:.6f}' for v in loc.ravel())}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="{float(p['twist_stiffness'])}"/>
          <config key="bend" value="{float(p['bend_stiffness'])}"/>
          <config key="vmax" value="0"/>
        </plugin>
        <joint kind="main" damping="{float(p['joint_damping'])}" armature="{float(p['joint_armature'])}"/>
        <geom type="{gtype}" size="{gsize}" rgba="{rgba(col)}" density="{dens:.4f}" condim="{int(p['condim'])}"
              friction="{vec(p['friction'])}" solref="{vec(p['contact_solref'])}" contype="1" conaffinity="1"/>
      </composite>""")
        n = len(cl[k])
        info[lab] = {"first": f"{pre}B_first", "last": f"{pre}B_last",
                     "links": [f"{pre}B_first"] + [f"{pre}B_{i}" for i in range(1, n - 2)] + [f"{pre}B_last"],
                     "n_links": n - 1, "length_m": round(L, 4), "width_m": widths[k], "thickness_m": thick[k],
                     "rgba": [float(v) for v in col], "mass_kg": round(mpm * L, 5),
                     "centreline": np.round(cl[k], 5).tolist(), "tail_start_index": int(tail_start[k])}
    floor = FLOOR_XML if p["include_floor"] else ""
    jvol = 8 * js[0] * js[1] * js[2]
    freej = f'<freejoint name="{name}_free"/>' if p["free"] else ""
    xml = mjcf_header(f"{name}", timestep=float(p["timestep"])) + f"""
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>{LIGHTS_XML}{floor}
    <body name="{name}_junction" pos="{vec(jpos)}" quat="{vec(_quat_yaw(yaw))}">
      {freej}
      <geom name="{name}_junction_g" type="box" size="{vec(js)}" rgba="{rgba(jrgba)}" density="{float(p['junction_mass']) / jvol:.3f}"
            condim="{int(p['condim'])}" friction="{vec(p['friction'])}" solref="{vec(p['contact_solref'])}"/>{''.join(comps)}
    </body>
  </worldbody>
</mujoco>
"""
    import mujoco
    spec = mujoco.MjSpec.from_string(xml)
    if braid is not None and abs(float(p["braid_width_frac"]) - 1.0) > 1e-9 and p["profile"] == "flat":
        # narrow the links that lie in the plait
        for k, lab in enumerate(labels):
            pre = f"{name}_{lab}"
            for g in spec.geoms:
                m = re.fullmatch(rf"{re.escape(pre)}G(\d+)", g.name or "")
                if m and int(m.group(1)) < max(tail_start[k] - 1, 0):
                    g.size = [g.size[0], widths[k] / 2 * float(p["braid_width_frac"]), g.size[2]]
    # <composite> frames parallel-transport along a 3D curve and roll the boxes onto their edges after a plait:
    # give every link the frame x = its edge, y = horizontal, z = up-ish (relative to its parent)
    def _frame(e):
        e = e / (np.linalg.norm(e) + 1e-12)
        y = np.cross([0.0, 0.0, 1.0], e)
        if np.linalg.norm(y) < 1e-6:
            y = np.array([0.0, 1.0, 0.0])
        y = y / np.linalg.norm(y)
        return np.column_stack([e, y, np.cross(e, y)])
    def _mat2quat(R):
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel()); return q
    for k, lab in enumerate(labels):
        pre = f"{name}_{lab}"
        P = cl[k]
        frames = [_frame(P[i + 1] - P[i]) for i in range(len(P) - 1)]
        names = info[lab]["links"]
        prev = Rj
        for i, bn in enumerate(names):
            bspec = spec.body(bn)
            if bspec is None:
                raise RuntimeError(f"expanded cable has no body {bn}")
            bspec.quat = _mat2quat(prev.T @ frames[i]).tolist()
            prev = frames[i]
    # composite box cables carry a white skin; delete it (or paint it the strand colour)
    skins = list(spec.skins)
    n_skins = len(skins)
    for sk in skins:
        lab = next((l for l in labels if (sk.name or "").startswith(f"{name}_{l}") or f"{name}_{l}" in (sk.name or "")), None)
        if p["keep_skin"] and lab is not None:
            sk.rgba = info[lab]["rgba"]
        else:
            spec.delete(sk)
    model = spec.compile()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    xml_path = out / f"{name}.xml"
    xml_path.write_text(spec.to_xml())
    for lab in labels:                                    # the compiled names are the truth
        ids = [model.body(b).id for b in info[lab]["links"]]
        assert len(ids) == info[lab]["n_links"]
    bodies = [f"{name}_junction"] + [b for lab in labels for b in info[lab]["links"]]
    pj = dict(p); pj["name"] = name
    save_params(out / f"{name}_params.json", json.loads(json.dumps(pj, default=float)))
    return {"xml": str(xml_path), "params_json": str(out / f"{name}_params.json"), "root_body": f"{name}_junction",
            "joint": f"{name}_free" if p["free"] else None, "bodies": bodies, "strands": info, "braid": braid_info,
            "junction": {"pos": jpos.round(5).tolist(), "yaw_deg": round(float(np.degrees(yaw)), 2), "half_size": js},
            "tail_crossings_lifted": int(lifted), "n_links": int(sum(info[l]["n_links"] for l in labels)),
            "total_mass_kg": round(total_mass, 5), "profile": p["profile"], "skins_found": n_skins, "skins_kept": int(model.nskin)}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    prm = json.loads(Path(a[1]).read_text()) if len(a) > 1 else {"strands": [{"width": 0.01, "centreline": [[0, 0], [0.2, 0]]}]}
    print(json.dumps(construct_multi_strand_cable(a[0] if a else ".", **prm), indent=1, default=str)[:4000])
