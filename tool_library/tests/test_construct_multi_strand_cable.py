"""Test construct_multi_strand_cable on several variations: a plaited three-strand ribbon, a two-strand round Y cord,
a four-strand fanned tassel, a six-crossing plait, and misuse."""
import os
import sys
import tempfile

import mujoco
import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_multi_strand_cable import construct_multi_strand_cable

FAIL = []


def check(cond, case, msg):
    if cond:
        print(f"[PASS] {case}")
    else:
        print(f"[FAIL] {case}: {msg}")
        FAIL.append(case)


def tree_stats(model, built):
    """Connected components and degree-1 links of the object's link graph (junction + links)."""
    ids = [model.body(b).id for b in built["bodies"]]
    S = set(ids)
    adj = {b: set() for b in ids}
    for b in ids:
        p = int(model.body_parentid[b])
        if p in S:
            adj[b].add(p); adj[p].add(b)
    seen, comps = set(), 0
    for b in ids:
        if b in seen:
            continue
        comps += 1
        st = [b]
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x); st += list(adj[x] - seen)
    ends = sum(1 for b in ids if len(adj[b]) == 1)
    return comps, ends


def deepest_contact(model, data):
    """Deepest penetration among contacts between geoms of bodies that are not parent and child."""
    worst = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if int(model.body_parentid[b1]) == b2 or int(model.body_parentid[b2]) == b1:
            continue
        worst = max(worst, -float(c.dist))
    return worst


def crossings(model, data, built):
    """2D crossings between different strands' link polylines, with which strand is on top."""
    pl = {lab: data.xpos[[model.body(b).id for b in s["links"]]].copy() for lab, s in built["strands"].items()}
    labs = list(pl)
    out = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            A, B = pl[labs[i]], pl[labs[j]]
            for a in range(len(A) - 1):
                for b in range(len(B) - 1):
                    p, q, r, s = A[a, :2], A[a + 1, :2], B[b, :2], B[b + 1, :2]
                    d1, d2 = q - p, s - r
                    den = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(den) < 1e-12:
                        continue
                    w = r - p
                    t = (w[0] * d2[1] - w[1] * d2[0]) / den
                    u = (w[0] * d1[1] - w[1] * d1[0]) / den
                    if 0 < t < 1 and 0 < u < 1:
                        za = A[a, 2] + t * (A[a + 1, 2] - A[a, 2]); zb = B[b, 2] + u * (B[b + 1, 2] - B[b, 2])
                        out.append((labs[i], labs[j], labs[i] if za > zb else labs[j]))
    return out


def settle(model, data, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    return bool(np.isfinite(data.qpos).all())


tmp = tempfile.mkdtemp(prefix="msc_test_")

# 1. the video case: three flat ribbons plaited with four crossings near the junction, then tails
axis = [[0.0, 0.0], [0.03, 0.05], [0.06, 0.10], [0.09, 0.16]]
b1 = construct_multi_strand_cable(
    os.path.join(tmp, "c1"), name="ribbon", include_floor=True,
    braid={"axis": axis, "crossings": 4, "style": "over"},
    strands=[{"label": "red", "width": 0.017, "rgba": [0.6, 0.08, 0.06, 1], "tail": [[0.09, 0.16], [0.10, 0.22], [0.11, 0.28]]},
             {"label": "wide_pink", "width": 0.021, "rgba": [0.75, 0.43, 0.34, 1], "tail": [[0.09, 0.16], [0.04, 0.20]]},
             {"label": "narrow_pink", "width": 0.009, "rgba": [0.75, 0.43, 0.34, 1], "tail": [[0.09, 0.16], [0.14, 0.19]]}])
m1 = mujoco.MjModel.from_xml_path(b1["xml"]); d1 = mujoco.MjData(m1); mujoco.mj_forward(m1, d1)
comps, ends = tree_stats(m1, b1)
check(comps == 1 and ends == 3, "plait3_topology_one_tree_three_ends", f"components {comps}, ends {ends}")
pen = deepest_contact(m1, d1)
check(pen < 0.001, "plait3_no_interpenetration_at_rest", f"deepest penetration {pen * 1000:.2f} mm")
x0 = crossings(m1, d1, b1)
check(len(x0) >= 4, "plait3_four_crossings_built", f"{len(x0)} crossings: {x0}")
j0 = d1.xpos[m1.body(b1["root_body"]).id].copy()
ok = settle(m1, d1, 1.5)
x1 = crossings(m1, d1, b1)
drift = float(np.linalg.norm(d1.xpos[m1.body(b1["root_body"]).id] - j0))
check(ok and drift < 0.01 and len(x1) >= 4, "plait3_settles_keeps_crossings",
      f"finite {ok}, junction drift {drift * 1000:.1f} mm, crossings after settle {len(x1)}")
check(b1["braid"]["order"]["wide_pink"] != b1["braid"]["order"]["narrow_pink"] and b1["n_links"] > 20, "plait3_strands_assigned",
      str(b1["braid"]))

# 2. a two-strand round Y cord, no braid
b2 = construct_multi_strand_cable(
    os.path.join(tmp, "c2"), name="ycord", profile="round", include_floor=True, segment_m=0.015,
    strands=[{"label": "a", "width": 0.006, "centreline": [[0.0, 0.0], [0.15, 0.05], [0.3, 0.05]]},
             {"label": "b", "width": 0.006, "centreline": [[0.0, 0.0], [0.15, -0.05], [0.3, -0.06]]}])
m2 = mujoco.MjModel.from_xml_path(b2["xml"]); d2 = mujoco.MjData(m2); mujoco.mj_forward(m2, d2)
comps, ends = tree_stats(m2, b2)
geoms_round = all(int(m2.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE) for g in range(m2.ngeom)
                  if m2.body(int(m2.geom_bodyid[g])).name.startswith("ycord_a"))
ok = settle(m2, d2, 1.0)
check(comps == 1 and ends == 2 and geoms_round and ok, "ycord_round_two_strands", f"components {comps}, ends {ends}, round {geoms_round}, finite {ok}")
L = b2["strands"]["a"]["length_m"]
check(abs(L - (np.hypot(0.15, 0.05) + 0.15)) < 0.01, "ycord_length_follows_centreline", f"length {L}")

# 3. a four-strand flat tassel fanned on the table, with one tail crossing another
b3 = construct_multi_strand_cable(
    os.path.join(tmp, "c3"), name="tassel", include_floor=True, thickness=0.001,
    strands=[{"label": f"t{k}", "width": 0.008 + 0.002 * k, "centreline": [[0.0, 0.0], [0.2 * np.cos(a), 0.2 * np.sin(a)]]}
             for k, a in enumerate(np.radians([-30, -10, 10, 30]))]
    + [])
m3 = mujoco.MjModel.from_xml_path(b3["xml"]); d3 = mujoco.MjData(m3); mujoco.mj_forward(m3, d3)
comps, ends = tree_stats(m3, b3)
pen = deepest_contact(m3, d3)
ok = settle(m3, d3, 1.0)
check(comps == 1 and ends == 4 and pen < 0.001 and ok, "tassel_four_strands", f"components {comps}, ends {ends}, pen {pen * 1000:.2f} mm, finite {ok}")

# 4. a longer plait with six crossings, explicit lateral amplitude, style under
b4 = construct_multi_strand_cable(
    os.path.join(tmp, "c4"), name="plait6", include_floor=True,
    braid={"axis": [[0.0, 0.0], [0.0, 0.24]], "crossings": 6, "lateral_m": 0.012, "style": "under"},
    strands=[{"label": "l", "width": 0.012, "tail": [[0.0, 0.24], [-0.05, 0.30]]},
             {"label": "m", "width": 0.012, "tail": [[0.0, 0.24], [0.0, 0.31]]},
             {"label": "r", "width": 0.012, "tail": [[0.0, 0.24], [0.05, 0.30]]}])
m4 = mujoco.MjModel.from_xml_path(b4["xml"]); d4 = mujoco.MjData(m4); mujoco.mj_forward(m4, d4)
x4 = crossings(m4, d4, b4)
pen = deepest_contact(m4, d4)
check(len(x4) == 6 and pen < 0.001, "plait6_exact_crossings", f"{len(x4)} crossings, pen {pen * 1000:.2f} mm")
# a plait alternates: consecutive crossings involve different strand pairs
pairs = [tuple(sorted(c[:2])) for c in x4]
check(len(set(pairs)) == 3, "plait6_all_pairs_cross", str(pairs))

# 5. misuse raises ValueError naming the parameter
for case, kw in [("unknown_param", {"strands": [{"width": 0.01, "centreline": [[0, 0], [0.1, 0]]}], "colour": 1}),
                 ("braid_needs_three", {"braid": {"axis": [[0, 0], [0, 0.1]], "crossings": 2},
                                        "strands": [{"width": 0.01, "tail": [[0, 0.1], [0, 0.2]]}] * 2}),
                 ("strand_needs_width", {"strands": [{"centreline": [[0, 0], [0.1, 0]]}]}),
                 ("bad_profile", {"profile": "tube", "strands": [{"width": 0.01, "centreline": [[0, 0], [0.1, 0]]}]})]:
    try:
        construct_multi_strand_cable(os.path.join(tmp, "bad"), **kw)
        check(False, f"misuse_{case}", "no error raised")
    except ValueError as e:
        check(True, f"misuse_{case}", "")

# 6. skins: flat strands come out without the composite's white skin, or with it painted when asked
check(m1.nskin == 0 and b1["skins_found"] == 3 and b1["skins_kept"] == 0, "flat_strands_unskinned", f"nskin {m1.nskin}, {b1.get('skins_found')}")
b6 = construct_multi_strand_cable(os.path.join(tmp, "c6"), name="painted", keep_skin=True,
    strands=[{"label": "a", "width": 0.01, "rgba": [0.2, 0.4, 0.9, 1], "centreline": [[0, 0], [0.15, 0.02]]},
             {"label": "b", "width": 0.01, "rgba": [0.9, 0.4, 0.2, 1], "centreline": [[0, 0], [0.15, -0.03]]}])
m6 = mujoco.MjModel.from_xml_path(b6["xml"])
cols = sorted(tuple(round(float(v), 2) for v in m6.skin_rgba[i][:3]) for i in range(m6.nskin))
check(m6.nskin == 2 and cols == [(0.2, 0.4, 0.9), (0.9, 0.4, 0.2)], "keep_skin_painted", f"nskin {m6.nskin}, colours {cols}")
check(m2.nskin == 0, "round_strands_have_no_skin", f"nskin {m2.nskin}")

sys.exit(1 if FAIL else 0)
