"""Drop a free item into an opened drawer, check it comes to rest inside the drawer's bin,
then push the drawer shut with a capped force and check it closes with the item enclosed.

    from tool_library.verifiers.drawer_receive_and_close_test import drawer_receive_and_close_test
    rep = drawer_receive_and_close_test("assets/drawer/drawer.xml", "assets/toy/toy.xml")
    rep["ok"], rep["issues"], rep["rest"], rep["close"]
"""
from __future__ import annotations

TOOL = {
    "name": "drawer_receive_and_close_test",
    "title": "Drawer receive-and-close test",
    "type": "verifier",
    "category": "Reconstruction Mechanism Tests",
    "summary": "Opens a drawer's slide joint, drops a free item over the exposed bin, checks it rests inside the bin, "
               "then pushes the drawer shut with a capped force and checks it closes with the item enclosed.",
    "description": (
        "Composes a test world from two MJCF files: the drawer asset (a fixed or free root with a slide joint "
        "whose child body is the drawer bin, e.g. port_urdf_asset output) and an item asset with a free joint "
        "(a plush toy, a cube, any rigid or soft body). A free drawer root is held fixed by removing its free "
        "joint. The slide joint (the first slide joint, or `joint`) is set to `open_frac` of its range. The "
        "exposed part of the bin is the part of the moving body's geometry that lies beyond the fixed geometry "
        "along the joint axis (world-space geom AABBs); the item is dropped from `drop_height` above the bin's "
        "top over the middle of that exposed part and settles for `settle_s` under gravity. It must end inside "
        "the moving body's local bounding box (between its floor and rim, within its walls) and at rest "
        "(speed below `rest_speed`). Then a force of `close_force_N` is applied to the drawer along the closing "
        "direction for `close_s`: the joint must return to within `close_tol_frac` of its closed stop, and the "
        "item must still be inside the bin and inside the fixed housing's extent along the axis (enclosed). "
        "Assumes the drawer opens toward +q of the slide joint and that the closed stop is the lower limit."
    ),
    "inputs": [
        {"name": "drawer_xml", "type": "str", "description": "MJCF of the drawer asset (slide joint on the bin body)."},
        {"name": "item_xml", "type": "str", "description": "MJCF of the item; its first body must carry a free joint."},
        {"name": "joint", "type": "str | None", "description": "Slide joint name in drawer_xml; default the first slide joint."},
        {"name": "open_frac", "type": "float", "description": "Opening as a fraction of the joint range. Default 0.9."},
        {"name": "drop_height", "type": "float, metres", "description": "Gap between the bin's top and the item's lowest point at release. Default 0.02."},
        {"name": "settle_s", "type": "float, seconds", "description": "Settling time after the drop. Default 1.5."},
        {"name": "close_force_N", "type": "float, newtons", "description": "Capped pushing force along the closing direction. Default 10."},
        {"name": "close_s", "type": "float, seconds", "description": "Pushing time. Default 1.5."},
        {"name": "close_tol_frac", "type": "float", "description": "Closed when q <= lower + this fraction of the range. Default 0.1."},
        {"name": "rest_speed", "type": "float, m/s", "description": "The item counts as at rest below this speed. Default 0.02."},
        {"name": "out_json", "type": "str | None", "description": "Optional path for the report."},
    ],
    "outputs": [
        {"name": "ok", "type": "bool", "description": "The item rested inside the open bin and stayed enclosed after the drawer closed."},
        {"name": "issues", "type": "list[str]", "description": "What failed and what to change (bin size, opening, force)."},
        {"name": "joint", "type": "str", "description": "The slide joint tested."},
        {"name": "exposed_m", "type": "float", "description": "Length of the bin exposed at the opening, along the axis."},
        {"name": "item_extent_m", "type": "list[float]", "description": "The item's bounding box size."},
        {"name": "bin_local_bounds", "type": "dict", "description": "lo/hi of the bin body's geometry in its own frame."},
        {"name": "rest", "type": "dict", "description": "inside, speed, item position in the bin frame after settling."},
        {"name": "close", "type": "dict", "description": "q_end, closed, inside_bin, enclosed after pushing."},
    ],
    "status": "tested",
    "entrypoint": "drawer_receive_and_close_test",
    "requirements": ["numpy", "mujoco"],
    "simulators": ["mujoco"],
    "tags": ["drawer", "slide", "container", "place", "drop", "enclose", "articulated", "verifier", "mechanism"],
    "used_in": ["Place Toy in Drawer"],
    "cli": "python -m tool_library.verifiers.drawer_receive_and_close_test <drawer_xml> <item_xml>",
}

import json
import re
import sys
from pathlib import Path

import numpy as np


def _strip_root_free_joint(xml_text: str) -> tuple[str, bool]:
    new, n = re.subn(r"<freejoint\b[^>]*/>", "", xml_text, count=1)
    if n:
        return new, True
    new, n = re.subn(r"<joint\b[^>]*type=\"free\"[^>]*/>", "", xml_text, count=1)
    return new, bool(n)


def _geom_world_corners(m, d, g):
    c = m.geom_aabb[g, :3]
    h = m.geom_aabb[g, 3:]
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    local = c + signs * h
    R = d.geom_xmat[g].reshape(3, 3)
    return d.geom_xpos[g] + local @ R.T


def _subtree(m, root):
    out = []
    for b in range(m.nbody):
        x = b
        while x > 0 and x != root:
            x = m.body_parentid[x]
        if x == root:
            out.append(b)
    return out


def drawer_receive_and_close_test(drawer_xml, item_xml, joint=None, open_frac=0.9, drop_height=0.02,
                                  settle_s=1.5, close_force_N=10.0, close_s=1.5, close_tol_frac=0.1,
                                  rest_speed=0.02, out_json=None):
    import mujoco

    for k, v in (("open_frac", open_frac), ("close_tol_frac", close_tol_frac)):
        if not (0.0 < float(v) <= 1.0):
            raise ValueError(f"{k}={v} must be in (0, 1]")
    for k, v in (("settle_s", settle_s), ("close_s", close_s), ("close_force_N", close_force_N),
                 ("rest_speed", rest_speed)):
        if float(v) <= 0:
            raise ValueError(f"{k}={v} must be positive")
    if float(drop_height) < 0:
        raise ValueError("drop_height must be >= 0")
    dpath, ipath = Path(drawer_xml), Path(item_xml)
    for k, p in (("drawer_xml", dpath), ("item_xml", ipath)):
        if not p.exists():
            raise ValueError(f"{k}: {p} does not exist")

    issues = []
    dtext, held = _strip_root_free_joint(dpath.read_text())
    dspec = mujoco.MjSpec.from_string(dtext)
    dspec.meshdir = str(dpath.parent / (dspec.meshdir or ""))
    ispec = mujoco.MjSpec.from_file(str(ipath))
    world = mujoco.MjSpec()
    world.option.timestep = min(0.002, float(dspec.option.timestep))
    world.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    world.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    fl = world.worldbody.add_geom()
    fl.name = "test_floor"
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [1.0, 1.0, 0.01]
    fl.pos = [0.0, 0.0, -0.0005]
    fd = world.worldbody.add_frame()
    world.attach(dspec, prefix="d_", frame=fd)
    fi = world.worldbody.add_frame()
    world.attach(ispec, prefix="i_", frame=fi)
    m = world.compile()
    d = mujoco.MjData(m)

    slides = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
              and m.joint(j).name.startswith("d_")]
    if joint is not None:
        slides = [j for j in slides if m.joint(j).name in ("d_" + joint, joint)]
    if not slides:
        raise ValueError(f"joint: no slide joint {joint!r} in {dpath.name}")
    j = slides[0]
    jname = m.joint(j).name[2:]
    lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (0.0, 0.0))
    if hi - lo <= 1e-6:
        raise ValueError(f"joint {jname}: needs a limited range")
    frees = [k for k in range(m.njnt) if m.jnt_type[k] == mujoco.mjtJoint.mjJNT_FREE
             and m.joint(k).name.startswith("i_")]
    if not frees:
        raise ValueError(f"item_xml: {ipath.name} has no free joint")
    fj = frees[0]
    ibody = m.jnt_bodyid[fj]
    mbody = m.jnt_bodyid[j]
    mov_bodies = set(_subtree(m, mbody))
    item_bodies = set(_subtree(m, ibody))
    drawer_geoms = [g for g in range(m.ngeom) if m.geom(g).name.startswith("d_") or
                    (m.geom_bodyid[g] not in item_bodies and m.geom_bodyid[g] != 0)]
    mov_geoms = [g for g in drawer_geoms if m.geom_bodyid[g] in mov_bodies and m.geom_contype[g] | m.geom_conaffinity[g]]
    fix_geoms = [g for g in drawer_geoms if m.geom_bodyid[g] not in mov_bodies and m.geom_contype[g] | m.geom_conaffinity[g]]
    item_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in item_bodies]

    q_open = lo + float(open_frac) * (hi - lo)
    d.qpos[m.jnt_qposadr[j]] = q_open
    mujoco.mj_forward(m, d)
    axis = d.xaxis[j] / np.linalg.norm(d.xaxis[j])
    Rm = d.xmat[mbody].reshape(3, 3)
    pm = d.xpos[mbody].copy()
    mov_c = np.vstack([_geom_world_corners(m, d, g) for g in mov_geoms])
    fix_c = np.vstack([_geom_world_corners(m, d, g) for g in fix_geoms]) if fix_geoms else None
    loc = (mov_c - pm) @ Rm
    blo, bhi = loc.min(0), loc.max(0)
    pa_m = mov_c @ axis
    m1 = pa_m.max()
    f1 = (fix_c @ axis).max() if fix_c is not None else pa_m.min()
    f0 = (fix_c @ axis).min() if fix_c is not None else pa_m.min()
    exposed = float(max(0.0, m1 - max(pa_m.min(), f1)))
    # item extent at its own origin
    it_c = np.vstack([_geom_world_corners(m, d, g) for g in item_geoms])
    i_lo, i_hi = it_c.min(0), it_c.max(0)
    item_ext = (i_hi - i_lo)
    item_root = d.xpos[ibody].copy()
    # drop point: middle of the exposed segment, laterally at the bin centre, above the rim
    centre = mov_c.mean(0)
    s_mid = (max(pa_m.min(), f1) + m1) / 2.0
    target = centre + (s_mid - centre @ axis) * axis
    z_rel = item_root[2] - i_lo[2]
    target[2] = mov_c[:, 2].max() + float(drop_height) + z_rel
    lat_item = float(max(item_ext[0], item_ext[1]))
    if exposed < min(item_ext[0], item_ext[1]):
        issues.append(f"only {exposed:.3f} m of the bin is exposed at open_frac={open_frac}, the item is "
                      f"{min(item_ext[0], item_ext[1]):.3f} m across: open further or build a longer travel")
    qa = m.jnt_qposadr[fj]
    d.qpos[qa:qa + 3] = target
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    # hold the drawer at its opening while the item drops
    dof = m.jnt_dofadr[j]
    qadr = m.jnt_qposadr[j]
    n_settle = int(float(settle_s) / m.opt.timestep)
    for _ in range(n_settle):
        d.qpos[qadr] = q_open
        d.qvel[dof] = 0.0
        mujoco.mj_step(m, d)

    def in_bin(eps=0.003):
        mujoco.mj_forward(m, d)
        Rb = d.xmat[mbody].reshape(3, 3)
        p = (d.xipos[ibody] - d.xpos[mbody]) @ Rb
        inside = bool(np.all(p >= blo - eps) and np.all(p <= bhi + eps))
        return inside, p

    inside, p_rest = in_bin()
    speed = float(np.linalg.norm(d.qvel[m.jnt_dofadr[fj]:m.jnt_dofadr[fj] + 3]))
    rest = {"inside": inside, "speed_m_s": round(speed, 4), "item_in_bin_frame_m": np.round(p_rest, 4).tolist()}
    if not inside:
        issues.append(f"the item did not land inside the bin (bin-frame position {np.round(p_rest, 3).tolist()} "
                      f"outside {np.round(blo, 3).tolist()}..{np.round(bhi, 3).tolist()}): the bin is too small "
                      f"for the item ({np.round(item_ext, 3).tolist()} m) or the opening too short")
    if speed > float(rest_speed):
        issues.append(f"the item is still moving at {speed:.3f} m/s after {settle_s} s: it bounces or slides; "
                      f"check the item's contact softness or lengthen settle_s")

    # push the drawer shut
    d.xfrc_applied[:] = 0
    n_close = int(float(close_s) / m.opt.timestep)
    for _ in range(n_close):
        d.xfrc_applied[mbody, :3] = -float(close_force_N) * axis
        mujoco.mj_step(m, d)
    d.xfrc_applied[:] = 0
    for _ in range(int(0.3 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    q_end = float(d.qpos[qadr])
    closed = q_end <= lo + float(close_tol_frac) * (hi - lo)
    inside2, p_end = in_bin()
    s_item = float(d.xipos[ibody] @ axis)
    enclosed = bool(inside2 and (fix_c is None or (f0 - 0.003 <= s_item <= f1 + 0.003)))
    close = {"q_end": round(q_end, 4), "range": [float(lo), float(hi)], "closed": bool(closed),
             "inside_bin": inside2, "enclosed": enclosed, "item_in_bin_frame_m": np.round(p_end, 4).tolist()}
    if not closed:
        issues.append(f"the drawer stopped at q={q_end:.3f} (range {lo:.3f}..{hi:.3f}) under {close_force_N} N: "
                      f"the item jams it (too tall for the housing opening) or the force is too small")
    if not enclosed:
        issues.append("after closing, the item is not enclosed in the housing: it was pushed out or rides on the rim")
    rep = {"ok": not issues, "issues": issues, "joint": jname, "held_free_root": held,
           "open_q": round(q_open, 4), "exposed_m": round(exposed, 4), "item_extent_m": np.round(item_ext, 4).tolist(),
           "item_lateral_m": round(lat_item, 4),
           "bin_local_bounds": {"lo": np.round(blo, 4).tolist(), "hi": np.round(bhi, 4).tolist()},
           "rest": rest, "close": close}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None):
    a = argv if argv is not None else sys.argv[1:]
    if len(a) < 2:
        print(__doc__)
        return 2
    r = drawer_receive_and_close_test(a[0], a[1])
    print(json.dumps(r, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
