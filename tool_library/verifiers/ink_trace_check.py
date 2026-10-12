"""Replay an executed episode's nibs and felt pads on a live-ink surface and judge what the surface shows."""
from __future__ import annotations

TOOL = {
    "name": "ink_trace_check",
    "title": "Ink trace check",
    "type": "verifier",
    "category": "Task & Rollout Gates",
    "summary": "Replay an executed episode frame by frame on a writable surface's live ink texture (nib sites ink where "
               "they touch it, felt pads lighten what they cover) and check that named regions end inked or clean.",
    "description": (
        "Writing and erasing are outcomes on a texture, not on a body pose, so the subgoal outcome check cannot see them. "
        "This verifier resolves the surface geom and its texture in the compiled scene (suffix match, so an asset "
        "attached under a prefix is found), resets the texture to its pristine state, and walks the trajectory: on every "
        "frame each nib site inks the surface where its gap to the writing face is at most `max_gap_m` (joined to its "
        "previous inked sample unless the jump is larger than `max_jump_m`, so a lift-and-move draws nothing), and each "
        "pad geom whose bottom face lies flat within `pad_gap_m` of the face lightens the rotated rectangle it covers by "
        "`efficiency` per frame (one pass does not erase; back-and-forth wiping does). Regions are rectangles in WORLD "
        "x/y on the surface (converted to the surface's own frame at frame 0), each with an expectation 'inked' "
        "(ink fraction >= min_ink) or 'clean' (<= max_ink), judged after replaying up to `at_frame` (default the last "
        "frame), so 'the 4 was written' and 'the 4 was wiped off' are separate gates on one replay. It also reports per "
        "nib the frames in contact and the inked path length, and writes the final texture as a PNG for a look. Assumes "
        "the surface is a box geom whose +z face is the writing face and whose texture maps its x/y extent (the "
        "construct_whiteboard convention, shared through tool_library.common.ink.InkSurface)."
    ),
    "inputs": [
        {"name": "scene", "type": "str", "description": "MJCF scene (the scene the episode was executed in)."},
        {"name": "trajectory", "type": "str | ndarray", "description": "trajectory.npz (qpos [T, nq]) or the qpos array."},
        {"name": "surface_geom", "type": "str", "description": "Writing-face geom name (or its unprefixed suffix, e.g. 'board_panel')."},
        {"name": "texture", "type": "str", "description": "Its live texture name (or suffix, e.g. 'board_ink')."},
        {"name": "nibs", "type": "list[str]", "description": "Site names that write (or suffixes). Default: every non-robot site whose name ends with 'tip'."},
        {"name": "pads", "type": "list[str] | None", "description": "Box geoms whose bottom (-z) face wipes (felt pads). Default none."},
        {"name": "regions", "type": "list[dict]", "description": "[{name, rect: [x0, x1, y0, y1] world metres, expect: 'inked'|'clean', min_ink? (default 0.02), max_ink? (default 0.004), at_frame? (int, default last)}]. At least one: with none nothing is judged and the check fails."},
        {"name": "frames", "type": "list[int] | None", "description": "[first, last] frame window to replay. Default the whole episode."},
        {"name": "nib_radius_m", "type": "float", "description": "Ink line radius. Default 0.0018."},
        {"name": "max_gap_m", "type": "float", "description": "Largest nib height above the face that still inks. Default 0.0012."},
        {"name": "max_jump_m", "type": "float", "description": "Largest step between inked samples that is joined. Default 0.03."},
        {"name": "pad_gap_m", "type": "float", "description": "Largest pad-face height above the surface that wipes. Default 0.0025."},
        {"name": "efficiency", "type": "float", "description": "Fraction of the remaining ink one frame of pad contact removes. Default 0.16."},
        {"name": "ink_rgb", "type": "list[int] | None", "description": "Ink colour 0-255. Default [18, 18, 22]."},
        {"name": "board_rgb", "type": "list[int] | None", "description": "Blank colour 0-255. Default [249, 249, 246]."},
        {"name": "out_png", "type": "str | None", "description": "Write the final texture here."},
    ],
    "outputs": [
        {"name": "report", "type": "dict", "description": "ok; issues; regions {name: {expect, ink, at_frame, ok}}; nibs {site: {contact_frames, first, last, inked_path_m}}; pads {geom: {contact_frames}}; n_frames; png."},
    ],
    "requirements": ["mujoco>=3.2", "numpy", "Pillow"],
    "simulators": ["MuJoCo"],
    "tags": ["ink", "whiteboard", "write", "wipe", "erase", "drawing", "texture", "outcome", "stage3"],
    "used_in": ["Drawing and Wiping"],
    "status": "tested",
    "entrypoint": "ink_trace_check",
}

from pathlib import Path

import numpy as np

_KNOWN = {"scene", "trajectory", "surface_geom", "texture", "nibs", "pads", "regions", "frames", "nib_radius_m",
          "max_gap_m", "max_jump_m", "pad_gap_m", "efficiency", "ink_rgb", "board_rgb", "out_png"}


def _resolve(names, want, kind):
    if want in names:
        return want
    hits = [n for n in names if n and n.endswith(want)]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(f"{kind} {want!r}: {'no match' if not hits else 'ambiguous: ' + ', '.join(hits[:5])}; give the full name")


def ink_trace_check(scene, trajectory, surface_geom, texture, nibs=None, pads=None, regions=None, frames=None,
                    nib_radius_m=0.0018, max_gap_m=0.0012, max_jump_m=0.03, pad_gap_m=0.0025, efficiency=0.16,
                    ink_rgb=None, board_rgb=None, out_png=None, **unknown):
    if unknown:
        raise ValueError(f"unknown parameter(s) {sorted(unknown)}; known: {sorted(_KNOWN)}")
    import mujoco
    from tool_library.common.ink import InkSurface
    if regions is not None and not isinstance(regions, list):
        raise ValueError("regions must be a list of {name, rect, expect}")
    model = scene if isinstance(scene, mujoco.MjModel) else mujoco.MjModel.from_xml_path(str(scene))
    if isinstance(trajectory, (str, Path)):
        qpos = np.asarray(np.load(str(trajectory), allow_pickle=True)["qpos"], float)
    else:
        qpos = np.asarray(trajectory, float)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"trajectory qpos has shape {qpos.shape}; the scene has nq={model.nq}")
    gnames = [model.geom(i).name for i in range(model.ngeom)]
    tnames = [model.texture(i).name for i in range(model.ntex)]
    snames = [model.site(i).name for i in range(model.nsite)]
    gname = _resolve(gnames, surface_geom, "surface_geom")
    tname = _resolve(tnames, texture, "texture")
    built = {"surface_geom": gname, "texture": tname, "ink_rgb": ink_rgb or [18, 18, 22], "board_rgb": board_rgb or [249, 249, 246]}
    ink = InkSurface(model, built)
    entry = ink.buf.copy()                       # the caller's texture is restored on return: a replay never leaves ink behind
    if nibs is None:
        nibs = [s for s in snames if s.endswith("tip") and not model.body(int(model.site_bodyid[model.site(s).id])).name.startswith("robot_")]
    nib_ids = {s: model.site(_resolve(snames, s, "nib site")).id for s in nibs}
    pad_ids = {p: model.geom(_resolve(gnames, p, "pad geom")).id for p in (pads or [])}
    for p, gid in pad_ids.items():
        if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError(f"pad {p!r} is not a box geom")
    T = len(qpos)
    f0, f1 = (0, T - 1) if frames is None else (max(0, int(frames[0])), min(T - 1, int(frames[1])))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[f0]
    mujoco.mj_kinematics(model, data)
    Rg = data.geom_xmat[ink.gid].reshape(3, 3).copy(); cg = data.geom_xpos[ink.gid].copy()

    def local_rect(rect):
        x0, x1, y0, y1 = map(float, rect)
        pts = [Rg.T @ (np.array([x, y, cg[2]]) - cg) for x in (x0, x1) for y in (y0, y1)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), max(xs), min(ys), max(ys)

    regs = []
    for i, r in enumerate(regions or []):
        if "rect" not in r or r.get("expect") not in ("inked", "clean"):
            raise ValueError(f"regions[{i}] needs rect [x0, x1, y0, y1] and expect 'inked' or 'clean'")
        at = int(r.get("at_frame", f1))
        regs.append({"name": r.get("name", f"region_{i}"), "loc": local_rect(r["rect"]), "expect": r["expect"],
                     "min_ink": float(r.get("min_ink", 0.02)), "max_ink": float(r.get("max_ink", 0.004)),
                     "at": min(max(at, f0), f1)})
    nib_stat = {s: {"contact_frames": 0, "first": None, "last": None, "inked_path_m": 0.0, "_prev": None} for s in nib_ids}
    pad_stat = {p: {"contact_frames": 0, "first": None, "last": None} for p in pad_ids}
    results = {}
    for f in range(f0, f1 + 1):
        data.qpos[:] = qpos[f]
        mujoco.mj_kinematics(model, data)
        for s, sid in nib_ids.items():
            p = data.site_xpos[sid].copy()
            st = nib_stat[s]
            if ink.stroke(data, p, radius_m=float(nib_radius_m), key=s, max_gap_m=float(max_gap_m), max_jump_m=float(max_jump_m)):
                st["contact_frames"] += 1
                st["first"] = f if st["first"] is None else st["first"]; st["last"] = f
                if st["_prev"] is not None and np.linalg.norm(p - st["_prev"]) <= max_jump_m:
                    st["inked_path_m"] += float(np.linalg.norm(p - st["_prev"]))
                st["_prev"] = p
            else:
                st["_prev"] = None
        for pname, gid in pad_ids.items():
            R = data.geom_xmat[gid].reshape(3, 3)
            if abs(float(R[:, 2] @ Rg[:, 2])) < 0.9:          # the pad face is not lying on the surface
                continue
            down = -R[:, 2] if float(R[:, 2] @ Rg[:, 2]) > 0 else R[:, 2]
            face = data.geom_xpos[gid] + down * float(model.geom_size[gid][2])
            if ink.gap(data, face) > float(pad_gap_m) or ink.uv(data, face) is None:
                continue
            ink.wipe(data, face, R, float(model.geom_size[gid][0]), float(model.geom_size[gid][1]),
                     max_gap_m=float(pad_gap_m), efficiency=float(efficiency))
            ps = pad_stat[pname]; ps["contact_frames"] += 1
            ps["first"] = f if ps["first"] is None else ps["first"]; ps["last"] = f
        for r in regs:
            if r["at"] == f:
                v = ink.darkness(*r["loc"])
                ok = v >= r["min_ink"] if r["expect"] == "inked" else v <= r["max_ink"]
                results[r["name"]] = {"expect": r["expect"], "ink": round(v, 4), "at_frame": f, "ok": bool(ok),
                                      "threshold": r["min_ink"] if r["expect"] == "inked" else r["max_ink"]}
    issues = []
    if not regs:
        issues.append("no regions given: nothing was judged; pass regions [{name, rect, expect: 'inked'|'clean'}] for "
                      "what the surface must show")
    for name, v in results.items():
        if not v["ok"]:
            if v["expect"] == "inked":
                issues.append(f"region {name}: ink {v['ink']:.3f} < {v['threshold']} at frame {v['at_frame']}: the nib did not "
                              f"touch the face there (put the nib on the surface: tip below the face by about 1 mm, or check the "
                              f"stroke passes through the region)")
            else:
                issues.append(f"region {name}: ink {v['ink']:.3f} > {v['threshold']} at frame {v['at_frame']}: the pad did not "
                              f"wipe it clean (more back-and-forth passes over it, the pad flat on the face, or a wider sweep)")
    for s, st in nib_stat.items():
        st.pop("_prev", None); st["inked_path_m"] = round(st["inked_path_m"], 4)
    png = None
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        png = ink.snapshot(out_png)
    ink.buf[:] = entry
    return {"ok": not issues, "issues": issues, "regions": results, "nibs": nib_stat, "pads": pad_stat,
            "surface": {"geom": gname, "texture": tname}, "frames": [f0, f1], "n_frames": f1 - f0 + 1, "png": png}
