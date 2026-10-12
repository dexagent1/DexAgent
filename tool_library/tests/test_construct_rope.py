"""construct_rope built along a measured centreline (the name measure_rope and the Stage 2 prompt use).

    python tool_library/tests/test_construct_rope.py

`centreline` was silently ignored: the result said "laid": false and the rope came out straight, 0.6 m long, at the
default pose, half inside the table; only the undeclared `rest_vertices` worked. Now a centreline (a point list,
measure_rope's result dict or its JSON) is resampled into the cable's rest vertices, and parameters the points
decide are refused when they contradict them.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from tool_library.skills.construct_rope import TOOL, construct_rope  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_rope_", dir=os.environ.get("TMPDIR")))
FAILS = 0
R = 0.003


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def body_chain(built):
    b = built["bodies"]
    return [b["first"]] + b["segments"] + [b["last"]]


def dist_to_poly(P, C):
    a, b = C[:-1], C[1:]
    ab = b - a
    out = []
    for p in P:
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        out.append(np.min(np.linalg.norm(a + ab * t[:, None] - p, axis=1)))
    return np.asarray(out)


# a measured S-curve lying on the table (z = radius), as measure_rope returns it in the rig frame
s = np.linspace(0, 1, 90)
C = np.stack([0.05 + 0.30 * s, 0.35 + 0.08 * np.sin(2 * np.pi * s), np.full_like(s, R)], axis=1)
L_true = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
meas = {"length_m": round(L_true, 4), "radius_m": R, "centreline_rig": C.tolist(), "points": C.tolist(), "centreline_table": (C * 0).tolist()}

# 1. a measure_rope result dict: built along the curve, length from the points, resting on the table
b = construct_rope(TMP / "dict", centreline=meas, radius=R)
m = mujoco.MjModel.from_xml_path(b["xml"])
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
P = np.array([d.xpos[m.body(n).id] for n in body_chain(b)])
dev = dist_to_poly(P, C)
pj = json.loads(Path(b["params_json"]).read_text())
check("measure_rope dict: laid, every link on the measured centreline (max 1 mm), length within 1%",
      b["laid"] and dev.max() < 0.001 and abs(pj["length"] - L_true) / L_true < 0.01 and len(P) >= 20,
      f"laid {b['laid']} max dev {dev.max() * 1000:.2f} mm length {pj['length']:.4f} vs {L_true:.4f}, {len(P)} links")
for _ in range(500):                                         # 1 s on the asset's own floor (z = 0)
    mujoco.mj_step(m, d)
P1 = np.array([d.xpos[m.body(n).id] for n in body_chain(b)])
check("... and it rests on the table instead of half inside it (axis stays ~one radius up, millimetre drift)",
      P1[:, 2].min() > R - 0.0015 and np.linalg.norm(P1 - P, axis=1).mean() < 0.003,
      f"min z {P1[:, 2].min() * 1000:.2f} mm drift {np.linalg.norm(P1 - P, axis=1).mean() * 1000:.2f} mm")

# 2. the same as a JSON file path, and as bare 2-D table points (z = radius)
jp = TMP / "rope_meas.json"
jp.write_text(json.dumps(meas))
bj = construct_rope(TMP / "json", centreline=str(jp), radius=R, segments=31)
check("measure_rope JSON path with segments=31: 31 vertices along the curve", bj["laid"] and len(body_chain(bj)) == 30,
      f"{len(body_chain(bj))} capsules")
b2 = construct_rope(TMP / "flat", centreline=C[:, :2].tolist(), radius=R)
m2 = mujoco.MjModel.from_xml_path(b2["xml"]); d2 = mujoco.MjData(m2); mujoco.mj_forward(m2, d2)
z2 = np.array([d2.xpos[m2.body(n).id][2] for n in body_chain(b2)])
check("2-D centreline points lie on the table (z = radius)", np.allclose(z2, R, atol=1e-6), f"z {z2.min():.4f}..{z2.max():.4f}")

# 3. a long thin cord (the charging cable): 1.9 m gets 1 vertex per 15 mm, capped at 160
t = np.linspace(0, 1, 400)
coil = np.stack([0.1 * np.cos(2 * np.pi * 3 * t) * (0.5 + 0.5 * t), 0.4 + 0.1 * np.sin(2 * np.pi * 3 * t) * (0.5 + 0.5 * t),
                 np.full_like(t, 0.00125)], 1)
coil[:, :2] = [0, 0.4] + (coil[:, :2] - [0, 0.4]) * 1.9 / np.linalg.norm(np.diff(coil, axis=0), axis=1).sum()
bc = construct_rope(TMP / "coil", centreline=coil.tolist(), radius=0.00125)
check("1.9 m cord: 128 vertices (15 mm capsules)", len(body_chain(bc)) == 127, f"{len(body_chain(bc))} capsules")

# 4. contradictions are refused, never ignored
for kw, word in ((dict(length=0.6), "length"), (dict(pos=[0, 0, 0.32]), "pos"), (dict(yaw_deg=30.0), "yaw_deg"),
                 (dict(rest_vertices=C[:5].tolist()), "not both")):
    try:
        construct_rope(TMP / "bad", centreline=C.tolist(), radius=R, **kw)
        check(f"centreline + contradicting {list(kw)[0]} is refused", False)
    except ValueError as e:
        check(f"centreline + contradicting {list(kw)[0]} is refused", word in str(e), str(e)[:90])
bl = construct_rope(TMP / "consistent", centreline=C.tolist(), radius=R, length=round(L_true, 3), pos=C[0].tolist())
check("a length and pos that agree with the centreline are accepted", bl["laid"])
try:
    construct_rope(TMP / "bad2", rest_vertices=C[::10].tolist(), segments=40, radius=R)
    check("rest_vertices with a contradicting segments count is refused", False)
except ValueError as e:
    check("rest_vertices with a contradicting segments count is refused", "segments" in str(e))
bs = construct_rope(TMP / "straight", radius=R)
check("no centreline: the straight default rope as before", not bs["laid"] and len(body_chain(bs)) == 39)

# 5. both are declared inputs (the harness builds the model's schema from them)
names = {i["name"] for i in TOOL["inputs"]}
check("centreline and rest_vertices are declared inputs", {"centreline", "rest_vertices"} <= names)

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
