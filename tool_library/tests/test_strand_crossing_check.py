"""Test strand_crossing_check on synthetic strands built as plain body chains: one X crossing, a three-strand plait,
parallel strands, a crossing that falls through (no strand contact), a settled crossing that holds, and misuse."""
import os
import sys
import tempfile

import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.strand_crossing_check import strand_crossing_check

FAIL = []


def check(cond, case, msg=""):
    print(f"[PASS] {case}" if cond else f"[FAIL] {case}: {msg}")
    if not cond:
        FAIL.append(case)


def chain_xml(name, pts, radius=0.004, contype=1, joint=True):
    """A chain of capsule links along pts (rig metres): nested bodies, ball joints, the first link free."""
    pts = np.asarray(pts, float)
    xml, close = "", ""
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        rel = a if i == 0 else a - pts[i - 1]
        j = '<freejoint/>' if i == 0 else ('<joint type="ball" damping="0.001"/>' if joint else "")
        e = b - a
        xml += (f'<body name="{name}_{i}" pos="{rel[0]:.5f} {rel[1]:.5f} {rel[2]:.5f}">{j}'
                f'<geom type="capsule" fromto="0 0 0 {e[0]:.5f} {e[1]:.5f} {e[2]:.5f}" size="{radius}" density="300" '
                f'contype="{contype}" conaffinity="{contype}"/>')
        close += "</body>"
    return xml + close, [f"{name}_{i}" for i in range(len(pts) - 1)]


def scene(path, chains):
    body = "".join(c[0] for c in chains)
    open(path, "w").write(f'<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/>{body}'
                          f'</worldbody></mujoco>')
    return path


tmp = tempfile.mkdtemp(prefix="scc_test_")
R = 0.004
lin = np.linspace(-0.1, 0.1, 11)

# 1. one X crossing, strand b lifted over strand a
a = np.column_stack([lin, np.zeros_like(lin), np.full_like(lin, R)])
b = np.column_stack([np.zeros_like(lin), lin, R + 2.2 * R * np.exp(-(lin / 0.03) ** 2)])
ca, cb = chain_xml("a", a), chain_xml("b", b)
p1 = scene(os.path.join(tmp, "x.xml"), [ca, cb])
r = strand_crossing_check(p1, {"a": ca[1], "b": cb[1]}, min_crossings=1)
check(r["ok"] and r["n_crossings"] == 1 and r["crossings"][0]["over"] == "b", "x_crossing_one_with_b_on_top", str(r))

# 2. a three-strand plait with four crossings (outer over middle)
t = np.linspace(0, 4 * np.pi / 3, 49)
ys = np.linspace(0, 0.2, 49)
plait = {}
chains = []
for k, ph in enumerate([0, 2 * np.pi / 3, 4 * np.pi / 3]):
    P = np.column_stack([0.012 * np.sin(t + ph), ys, R + 0.012 * (1 - np.sin(2 * (t + ph))) / 2])
    c = chain_xml(f"s{k}", P[::3], radius=R)
    chains.append(c); plait[f"s{k}"] = c[1]
p2 = scene(os.path.join(tmp, "plait.xml"), chains)
r = strand_crossing_check(p2, plait, min_crossings=4, expect_alternating=True)
check(r["ok"] and r["n_crossings"] == 4 and r["alternating"], "plait_four_alternating", str({k: r[k] for k in ("n_crossings", "pairs_sequence", "issues")}))

# 3. parallel strands: no crossing -> reported, not raised; and max_crossings=0 passes them (an unbraided goal)
c1 = chain_xml("p", np.column_stack([lin, np.zeros_like(lin), np.full_like(lin, R)]))
c2 = chain_xml("q", np.column_stack([lin, np.full_like(lin, 0.03), np.full_like(lin, R)]))
p3 = scene(os.path.join(tmp, "par.xml"), [c1, c2])
r = strand_crossing_check(p3, {"p": c1[1], "q": c2[1]}, min_crossings=1)
check((not r["ok"]) and r["n_crossings"] == 0 and r["issues"], "parallel_reports_zero", str(r))
r = strand_crossing_check(p3, {"p": c1[1], "q": c2[1]}, min_crossings=0, max_crossings=0)
check(r["ok"], "parallel_passes_unbraided_goal", str(r))

# 4. settling: with contact the lifted strand rests on the other (crossing kept); without contact it falls through
p4 = scene(os.path.join(tmp, "x_ok.xml"), [ca, cb])
r = strand_crossing_check(p4, {"a": ca[1], "b": cb[1]}, settle_s=1.0)
check(r["ok"] and not r["after_settle"]["lost"] and not r["after_settle"]["flipped"], "settle_crossing_holds", str(r.get("after_settle")))
cb0 = chain_xml("b", b, contype=0)
ca2 = chain_xml("a", a, contype=1)
p5 = scene(os.path.join(tmp, "x_ghost.xml"), [ca2, cb0])
r = strand_crossing_check(p5, {"a": ca2[1], "b": cb0[1]}, settle_s=1.0)
check(not r["ok"], "settle_pass_through_detected", str(r.get("after_settle")))

# 5. misuse raises ValueError
try:
    strand_crossing_check(p1, {"a": ca[1], "b": ["nope_0", "nope_1"]})
    check(False, "misuse_missing_body", "no error")
except ValueError:
    check(True, "misuse_missing_body")
try:
    strand_crossing_check(p1, {"a": ca[1]})
    check(False, "misuse_one_strand", "no error")
except ValueError:
    check(True, "misuse_one_strand")

sys.exit(1 if FAIL else 0)
