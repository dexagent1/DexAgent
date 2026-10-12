import os
import sys
import tempfile

import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.strand_crossing_check import strand_crossing_check

TMP = tempfile.mkdtemp(prefix="scc_braid_order_")
FAIL = []


def write_scene(paths, name):
    """Each strand is a row of static marker bodies at the given points (x, y, z)."""
    body = []
    strands = {}
    for lab, P in paths.items():
        strands[lab] = []
        for i, p in enumerate(P):
            bn = f"{lab}_{i}"
            strands[lab].append(bn)
            body.append(f'<body name="{bn}" pos="{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"><geom type="sphere" size="0.001" '
                        f'contype="0" conaffinity="0" mass="0.001"/></body>')
    xml = f'<mujoco model="{name}"><option timestep="0.002"/><worldbody>{"".join(body)}</worldbody></mujoco>'
    path = os.path.join(TMP, name + ".xml")
    with open(path, "w") as f:
        f.write(xml)
    return path, strands


def plait(crossings, length=0.18, lateral=0.01, lift=0.006, style="over", bend=0.0, labels=("left", "right", "middle")):
    n = crossings * 15 + 1
    u = np.linspace(0.0, 1.0, n)
    t = u * crossings * np.pi / 3.0
    sign = -1.0 if style == "over" else 1.0
    out = {}
    for lab, ph in zip(labels, [0.0, 2 * np.pi / 3, 4 * np.pi / 3]):
        x = u * length
        y = lateral * np.sin(t + ph) + bend * (u * length) ** 2       # optional curved axis
        z = 0.001 + lift * (1.0 + sign * np.sin(2 * (t + ph))) / 2.0
        out[lab] = np.column_stack([x, y, z])
    return out


def check(case, cond, rep):
    if cond:
        print(f"[PASS] {case}")
    else:
        FAIL.append(case)
        print(f"[FAIL] {case}: {rep}")


# 1. a five-crossing over-plait: five crossings, consecutive pairs differ (the parent verifier reported repeats here)
scene, st = write_scene(plait(5), "plait5")
rep = strand_crossing_check(scene, st, min_crossings=4, max_crossings=6, expect_alternating=True)
check("plait_5_over_alternates", rep["ok"] and rep["n_crossings"] == 5 and rep["alternating"]
      and all(rep["crossings"][i]["along_mean"] <= rep["crossings"][i + 1]["along_mean"] for i in range(4)), rep)

# 2. a three-crossing under-plait on a curved axis, with the strands listed in another order
p = plait(3, length=0.12, lateral=0.008, style="under", bend=1.5, labels=("b", "c", "a"))
scene, st = write_scene({k: p[k] for k in ("c", "a", "b")}, "plait3_curved")
rep = strand_crossing_check(scene, st, min_crossings=3, max_crossings=3, expect_alternating=True)
check("plait_3_under_curved_axis", rep["ok"] and rep["n_crossings"] == 3 and rep["alternating"], rep)

# 3. persistence path: settle a static plait; nothing may be lost or flipped
scene, st = write_scene(plait(4), "plait4")
rep = strand_crossing_check(scene, st, min_crossings=4, expect_alternating=True, settle_s=0.1)
check("plait_4_settled_persists", rep["ok"] and rep["after_settle"]["n_crossings"] == 4 and not rep["after_settle"]["lost"]
      and not rep["after_settle"]["flipped"], rep)

# 4. a twisted pair plus a straight third strand: crossings all between the same pair -> not a plait
u = np.linspace(0, 1, 61)
t = u * 3 * np.pi
tw = {"A": np.column_stack([u * 0.2, 0.01 * np.cos(t), 0.004 + 0.003 * np.sin(t)]),
      "B": np.column_stack([u * 0.2, -0.01 * np.cos(t), 0.004 - 0.003 * np.sin(t)]),
      "C": np.column_stack([u * 0.2, np.full_like(u, 0.05), np.full_like(u, 0.001)])}
scene, st = write_scene(tw, "twisted_pair")
rep_alt = strand_crossing_check(scene, st, min_crossings=1, expect_alternating=True)
rep_any = strand_crossing_check(scene, st, min_crossings=1, expect_alternating=False)
check("twisted_pair_not_a_plait", (not rep_alt["ok"]) and rep_alt["n_crossings"] == 3 and not rep_alt["alternating"] and rep_any["ok"],
      (rep_alt, rep_any))

# 5. an unbraided fan of three strands from one junction: zero crossings is the goal state, and fails a braid requirement
fan = {lab: np.column_stack([u * 0.2 * np.cos(a), u * 0.2 * np.sin(a), np.full_like(u, 0.001)])
       for lab, a in (("l", 0.4), ("m", 0.0), ("r", -0.4))}
scene, st = write_scene(fan, "fan")
rep0 = strand_crossing_check(scene, st, min_crossings=0, max_crossings=0)
rep1 = strand_crossing_check(scene, st, min_crossings=1)
rep2 = strand_crossing_check(scene, st, min_crossings=0, junction_m=0.0)   # the shared start counts when asked
check("unbraided_fan_zero_crossings", rep0["ok"] and rep0["n_crossings"] == 0 and (not rep1["ok"]) and rep2["n_crossings"] == 3,
      (rep0, rep1, rep2))

# 6. misuse fails loudly
errs = 0
for kwargs in ({"strands": {"only": ["l_0", "l_1"]}}, {"strands": {"l": ["l_0", "nope"], "m": ["m_0", "m_1"]}},
               {"strands": st, "min_crossings": 3, "max_crossings": 1}):
    try:
        strand_crossing_check(scene, **kwargs)
    except ValueError:
        errs += 1
check("misuse_raises_value_error", errs == 3, errs)

# 7. a scene that does not compile is a verdict, not a crash
bad = os.path.join(TMP, "bad.xml")
with open(bad, "w") as f:
    f.write("<mujoco><worldbody><body name='x'><geom type='nonsense'/></body></worldbody></mujoco>")
rep = strand_crossing_check(bad, {"a": ["a0", "a1"], "b": ["b0", "b1"]})
check("non_compiling_scene_reported", (not rep["ok"]) and "compile" in rep["issues"][0], rep)

sys.exit(1 if FAIL else 0)
