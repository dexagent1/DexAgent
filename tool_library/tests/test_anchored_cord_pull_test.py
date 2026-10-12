import sys, tempfile
import numpy as np
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.anchored_cord_pull_test import anchored_cord_pull_test

fails = 0


def check(case, cond, msg=""):
    global fails
    if cond:
        print(f"[PASS] {case}")
    else:
        fails += 1
        print(f"[FAIL] {case} {msg}")


def scene(path, n=20, seg=0.012, r=0.0012, curve=0.08, anchor="weld", anchor_to="block", block_free=False):
    """A curved cord of n capsule links on a floor; link c0 sits next to a block and is (optionally) anchored to it."""
    body = ""
    for i in range(n):
        if i == 0:
            body += f'<body name="c0" pos="0 0 {r}"><freejoint/>'
        else:
            body += f'<body name="c{i}" pos="{seg} 0 0" euler="0 0 {curve}"><joint type="ball" damping="1e-3" armature="1e-5"/>'
        body += f'<geom type="capsule" fromto="0 0 0 {seg} 0 0" size="{r}" density="1100" friction="0.7 0.02 0.001"/>'
    body += "</body>" * n
    bj = "<freejoint/>" if block_free else ""
    eq = ""
    if anchor == "weld":
        eq = f'<equality><weld body1="c0" body2="{anchor_to}" solref="0.004 1" solimp="0.99 0.999 0.0001 0.5 2"/></equality>'
    elif anchor == "connect":
        eq = f'<equality><connect body1="c0" body2="{anchor_to}" anchor="0 0 0" solref="0.004 1" solimp="0.99 0.999 0.0001 0.5 2"/></equality>'
    xml = (f'<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody><geom name="floor" type="plane" size="2 2 0.01"/>'
           f'<body name="block" pos="-0.03 0 0.02">{bj}<geom type="box" size="0.02 0.03 0.02" mass="0.5"/></body>'
           f'{body}</worldbody>{eq}</mujoco>')
    open(path, "w").write(xml)
    return f"c{n - 1}"


with tempfile.TemporaryDirectory() as tmp:
    # 1) welded thread-like cord: the end holds and the cord straightens
    free = scene(f"{tmp}/a.xml", curve=0.15)
    r = anchored_cord_pull_test(f"{tmp}/a.xml", "c0", free, out_json=f"{tmp}/a.json")
    check("welded cord holds and goes taut", r["ok"] and r["taut_ratio"] > 0.9 and r["end_to_end_start_m"] < 0.9 * r["length_m"],
          str(r["issues"]) + f" ratio {r.get('taut_ratio')} start {r.get('end_to_end_start_m')} len {r.get('length_m')}")

    # 2) longer rope on a connect anchor, larger force and its own direction
    free2 = scene(f"{tmp}/b.xml", n=36, seg=0.01, r=0.002, curve=0.05, anchor="connect")
    r2 = anchored_cord_pull_test(f"{tmp}/b.xml", "c0", free2, pull_force_N=1.0, hold_s=2.0)
    check("connect-anchored longer rope holds under 1 N", r2["ok"] and r2["anchor"]["type"] == "connect", str(r2["issues"]))

    # 3) negative: no anchor at all -> the end is dragged away, the verdict says so
    free3 = scene(f"{tmp}/c.xml", anchor="none")
    r3 = anchored_cord_pull_test(f"{tmp}/c.xml", "c0", free3)
    check("unanchored cord is reported", (not r3["ok"]) and any("no active weld" in s for s in r3["issues"])
          and any("anchored end moved" in s for s in r3["issues"]), str(r3["issues"]))

    # 4) negative: anchored to a block that is itself free -> refused as an anchor
    free4 = scene(f"{tmp}/d.xml", block_free=True)
    r4 = anchored_cord_pull_test(f"{tmp}/d.xml", "c0", free4)
    check("anchor on a movable body is reported", (not r4["ok"]) and any("can move" in s for s in r4["issues"]), str(r4["issues"]))

    # 5) misuse
    try:
        anchored_cord_pull_test(f"{tmp}/a.xml", "c0", free, force=2)
        check("unknown parameter refused", False, "no error")
    except ValueError as e:
        check("unknown parameter refused", "unknown" in str(e))
    r5 = anchored_cord_pull_test(f"{tmp}/a.xml", "c0", "nope")
    check("missing body reported, not raised", (not r5["ok"]) and "not a body" in r5["issues"][0])

sys.exit(1 if fails else 0)
