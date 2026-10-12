"""Tests for construct_bead_pile: three build variations plus misuse, each printing [PASS] <case>."""
import sys
import tempfile

import mujoco
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_bead_pile import construct_bead_pile


def visual_inside_collision(model, data, body, tol=1e-6):
    """Every visual bead's horizontal disk and height lie inside the union of the body's collision geoms (boxes or
    cylinders), checked on sample points of each visual geom."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    col = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_contype[g] | model.geom_conaffinity[g]]
    vis = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and not (model.geom_contype[g] | model.geom_conaffinity[g])]
    worst = 0.0
    for g in vis:
        r, hh = model.geom_size[g][0], model.geom_size[g][1]
        c = data.geom_xpos[g]
        for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            for dz in (-hh, hh):
                pt = c + np.array([r * np.cos(ang), r * np.sin(ang), dz])
                best = np.inf
                for k in col:
                    R = data.geom_xmat[k].reshape(3, 3)
                    q = R.T @ (pt - data.geom_xpos[k])
                    sz = model.geom_size[k]
                    if model.geom_type[k] == mujoco.mjtGeom.mjGEOM_BOX:
                        out = np.maximum(np.abs(q) - sz, 0)
                        d = float(np.linalg.norm(out))
                    else:
                        d = max(float(np.hypot(q[0], q[1]) - sz[0]), float(abs(q[2]) - sz[1]), 0.0)
                    best = min(best, d)
                worst = max(worst, best)
    return worst <= tol + 1e-4, worst, len(col), len(vis)


def main():
    ok_all = True
    tmp = tempfile.mkdtemp()

    # 1. defaults: fixed pile with cell collision, colours from the default palette
    b = construct_bead_pile(tmp + "/a")
    m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    inside, worst, nc, nv = visual_inside_collision(m, d, b["body"])
    c1 = (b["joint"] is None and m.njnt == 0 and b["n_beads"] == 60 and nc == b["n_collision_geoms"] < 20 and inside
          and abs(m.body_subtreemass[1] - b["mass_kg"]) < 1e-4 * b["mass_kg"])
    print(("[PASS]" if c1 else "[FAIL]"), "default fixed pile: 60 beads, %d collision cells, visual inside collision (worst %.2g m), mass %.4f kg" % (nc, worst, b["mass_kg"]))
    ok_all &= c1

    # 2. free pile, per-bead collision, two layers, keep_out circle, measured rgb palette with weights; it rests on a floor
    ko = [[0.01, 0.0, 0.012]]
    b = construct_bead_pile(tmp + "/b", name="heap", length=0.07, width=0.05, n_beads=25, layers=2, cell=0.0, fixed=False,
                            keep_out=ko, palette=[[0.4, 0.23, 0.16], [0.6, 0.49, 0.1]], palette_weights=[3, 1],
                            include_floor=True, seed=3)
    m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    import json
    beads = json.load(open(b["beads_json"]))
    clear = all(np.hypot(e["x"] - ko[0][0], e["y"] - ko[0][1]) >= ko[0][2] + 0.004 - 1e-9 for e in beads)
    z0 = d.qpos[2]
    for _ in range(500):
        mujoco.mj_step(m, d)
    drift = float(np.linalg.norm(d.qpos[:3] - np.array([0, 0, z0])))
    two = any(e["z"] > 0 for e in beads)
    c2 = (b["joint"] == "heap_free" and b["n_collision_geoms"] == b["n_beads"] and clear and two and drift < 0.003
          and np.all(np.isfinite(d.qpos)))
    print(("[PASS]" if c2 else "[FAIL]"), "free two-layer pile with keep_out and per-bead collision rests on a floor (drift %.4f m, %d beads)" % (drift, b["n_beads"]))
    ok_all &= c2

    # 3. a big footprint caps at the grid capacity; small beads; solid beads (no hole); yaw and position applied
    b = construct_bead_pile(tmp + "/c", name="pellets", length=0.03, width=0.02, bead_radius=0.002, bead_height=0.003,
                            hole_radius=0.0, n_beads=500, cell=0.01, yaw_deg=90, pos=[0.1, 0.2, 0.0])
    m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    inside, worst, nc, nv = visual_inside_collision(m, d, "pellets")
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pellets")
    xs = [d.geom_xpos[g] for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    ext = np.ptp(np.asarray(xs), axis=0)
    c3 = (b["n_beads"] < 500 and nv == b["n_beads"] and inside and ext[1] > ext[0] and np.allclose(d.xpos[bid], [0.1, 0.2, 0.0]))
    print(("[PASS]" if c3 else "[FAIL]"), "capacity-capped solid pellets (%d of 500), yawed 90 deg (extent x %.3f < y %.3f), visual inside collision" % (b["n_beads"], ext[0], ext[1]))
    ok_all &= c3

    # 4. cell collision with keep_out spots: no collision box reaches into a spot (the cell boxes used to cover the
    #    beads' bounding box, hole included, and a free bead placed in the spot started 4.4 mm inside the heap)
    ko = [[0.012, 0.004, 0.0045], [-0.025, -0.01, 0.0045]]
    b = construct_bead_pile(tmp + "/d", name="necklace_pile", length=0.10, width=0.07, n_beads=80, cell=0.025, keep_out=ko, seed=1)
    m = mujoco.MjModel.from_xml_path(b["xml"]); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    worst_in = 0.0
    for g in (m.geom(n).id for n in b["collision_geoms"]):
        c, sz = d.geom_xpos[g], m.geom_size[g]
        for kx, ky, kr in ko:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                dist = float(np.hypot(np.clip(kx, c[0] - sz[0], c[0] + sz[0]) - kx, np.clip(ky, c[1] - sz[1], c[1] + sz[1]) - ky))
            else:
                dist = float(np.hypot(c[0] - kx, c[1] - ky)) - sz[0]
            worst_in = max(worst_in, kr - dist)
    inside, worst, nc, nv = visual_inside_collision(m, d, b["body"])
    # a free bead of the keep_out radius dropped into each spot touches nothing of the heap
    spot = "".join(f'<body name="free{i}" pos="{kx} {ky} {0.003 + 0.0002}"><freejoint/><geom type="cylinder" size="{kr * 0.98} 0.003" '
                   f'mass="0.0003"/></body>' for i, (kx, ky, kr) in enumerate(ko))
    txt = open(b["xml"]).read().replace("</worldbody>", spot + "</worldbody>")
    m2 = mujoco.MjModel.from_xml_string(txt); d2 = mujoco.MjData(m2); mujoco.mj_forward(m2, d2)
    pen = max([-d2.contact.dist[i] for i in range(d2.ncon)] + [0.0])
    c4 = worst_in <= 1e-9 and inside and pen < 1e-4 and b["n_collision_geoms"] < 40
    print(("[PASS]" if c4 else "[FAIL]"), "cell collision carves the keep_out spots (deepest box reach into a spot %.2g m, free bead penetration %.2g m, %d collision geoms, visual inside collision %s)" % (worst_in, pen, b["n_collision_geoms"], inside))
    ok_all &= c4

    # 5. misuse raises ValueError naming the parameter
    errs = 0
    for bad in ({"colour": [1, 0, 0]}, {"hole_radius": 0.005}, {"layers": 3}, {"cell": 0.001}, {"palette_weights": [1]}):
        try:
            construct_bead_pile(tmp + "/bad", **bad)
        except ValueError:
            errs += 1
    c5 = errs == 5
    print(("[PASS]" if c5 else "[FAIL]"), "misuse raises ValueError (%d/5)" % errs)
    ok_all &= c5
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
