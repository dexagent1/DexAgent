import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from tool_library.verifiers.hinge_drive_check import hinge_drive_check

# a flap hinged about x at 5 cm, a free "finger" sphere on three slides; qpos = [fx, fy, fz, hinge]
XML = """<mujoco model="flapcheck"><option timestep="0.002"/><worldbody>
<geom name="floor" type="plane" size="1 1 0.1"/>
<body name="finger"><joint name="fx" type="slide" axis="1 0 0"/><joint name="fy" type="slide" axis="0 1 0"/>
<joint name="fz" type="slide" axis="0 0 1"/><geom name="tip" type="sphere" size="0.008" mass="0.05"/></body>
<body name="flap" pos="0 0 0.05"><joint name="hinge" type="hinge" axis="1 0 0" range="-30 200"/>
<geom name="plate" type="box" size="0.08 0.06 0.0015" pos="0 0.06 0" mass="0.05"/></body>
<body name="drawer" pos="0 -0.3 0.03"><joint name="slide" type="slide" axis="0 1 0" range="-0.2 0.2"/>
<geom name="front" type="box" size="0.05 0.01 0.02" mass="0.2"/></body>
</worldbody></mujoco>"""
tmp = Path(tempfile.mkdtemp()); scene = tmp / "flap.xml"; scene.write_text(XML)

def under(q, r=0.09, gap=-0.0005):
    u = np.array([0, np.cos(q), np.sin(q)]); n = np.array([0, -np.sin(q), np.cos(q)])
    return np.array([0, 0, 0.05]) + r * u - (0.0015 + 0.008 + gap) * n

def traj(name, qs, finger=None, slide=None):
    Q = np.zeros((len(qs), 5))
    for k, q in enumerate(qs):
        Q[k, :3] = finger(k, q) if finger else under(q); Q[k, 3] = q
        Q[k, 4] = 0.0 if slide is None else slide[k]
    p = tmp / f"{name}.npz"
    np.savez(p, qpos=Q, labels=np.array([f"left: lift flap"] * len(qs)), fps=30.0)
    return str(p)

fails = 0
def case(name, fn):
    global fails
    try:
        fn(); print(f"[PASS] {name}")
    except Exception as e:
        fails += 1; print(f"[FAIL] {name}: {e}")

up = np.radians(np.linspace(0, 80, 60))
def carried():
    r = hinge_drive_check(str(scene), traj("carried", up), "hinge", "finger")
    assert r["ok"], r; m = r["measured"]
    assert m["direction"] == 1 and m["travel"] > 79 and m["driven_frac"] > 0.99 and m["backslide"] < 0.5, m
def near_within_tol():
    t = traj("near", up, finger=lambda k, q: under(q, gap=0.002))
    r = hinge_drive_check(str(scene), t, "hinge", "finger", contact_tol_m=0.003)
    assert r["ok"], r
    r2 = hinge_drive_check(str(scene), t, "hinge", "finger", contact_tol_m=0.001)
    assert not r2["ok"] and r2["measured"]["driven_frac"] < 0.1, r2["measured"]
def dropped():
    q = np.radians(np.concatenate([np.linspace(0, 50, 30), np.linspace(50, 20, 30)]))
    path = np.radians(np.linspace(0, 80, 60))
    r = hinge_drive_check(str(scene), traj("dropped", q, finger=lambda k, qq: under(path[k])), "hinge", "finger", direction=1)
    assert not r["ok"] and r["measured"]["backslide"] > 25 and any("fell back" in s for s in r["issues"]), r
def batted():
    r = hinge_drive_check(str(scene), traj("batted", up, finger=lambda k, q: np.array([0.0, 0.1, -0.05])), "hinge", "finger")
    assert not r["ok"] and r["measured"]["driven_frac"] == 0.0 and any("on its own" in s for s in r["issues"]), r
def closing_direction():
    down = up[::-1]
    r = hinge_drive_check(str(scene), traj("closing", down), "hinge", "finger", min_travel_deg=60)
    assert r["ok"] and r["measured"]["direction"] == -1, r
def drawer_slide():
    s = np.linspace(0.0, 0.08, 40)
    fing = lambda k, q: np.array([0.0, -0.3 + s[k] - 0.01 - 0.008 + 0.0004, 0.03])
    r = hinge_drive_check(str(scene), traj("drawer", np.zeros(40), finger=fing, slide=s), "slide", "finger", min_travel_deg=60)
    assert r["ok"] and r["measured"]["unit"] == "mm" and r["measured"]["travel"] > 79, r
def misuse():
    t = traj("m", up)
    for kw, word in ((dict(joint="nope"), "joint"), (dict(holder="nobody"), "holder"), (dict(frames=[3]), "two frames"), (dict(direction=2), "direction")):
        args = dict(joint="hinge", holder="finger"); args.update(kw)
        try:
            hinge_drive_check(str(scene), t, **args)
        except ValueError as e:
            assert word in str(e), str(e); continue
        raise AssertionError(f"no ValueError for {kw}")
case("a finger under a 16x12 cm flap carries it 80 deg up", carried)
case("a finger 2 mm off the flap counts within a 3 mm tolerance, not within 1 mm", near_within_tol)
case("a flap that falls back 30 deg off the hand fails on the backslide", dropped)
case("a flap that turns with the hand 5 cm away fails as not driven", batted)
case("a flap pushed shut (decreasing angle) is recognised as driven in -1", closing_direction)
case("a fingertip pushing a drawer front 80 mm along its slide passes in mm", drawer_slide)
case("unknown joint / holder, a one-frame window and a bad direction are refused", misuse)
sys.exit(1 if fails else 0)
