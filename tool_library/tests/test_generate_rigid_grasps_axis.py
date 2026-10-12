"""generate_rigid_grasps builds its wraps about the object's own axis: a rod (24 mm, 18 cm, 80 g, pipette-like) leaning
14 deg in a rack is wrapped about its tilted axis (every side wrap approaches perpendicular to it), and a feasible opposed
wrap closes on it without turning it. A vertical-axis control approaching across the lean turns the rod; importantly,
the positive case must come from the self-collision-feasible candidates, not assume that a particular azimuth survives
the physical hand's joint and collision constraints. The same rod standing upright keeps the
vertical exactly; out_json writes the ranked list like the other pose tools.

    python tool_library/tests/test_generate_rigid_grasps_axis.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import tool_library.skills.generate_rigid_grasps as G  # noqa: E402
from tool_library.common import grasp_robust as GR  # noqa: E402
from tool_library.common import hand as H  # noqa: E402

HAND = str(ROOT / "setup" / "robot" / "sharpa_right.xml")
TMP = Path(tempfile.mkdtemp(prefix="rigid_axis_", dir=os.environ.get("TMPDIR")))
FAILS = 0
R_ROD, HL_ROD, R_BAR, LEAN_DEG = 0.012, 0.09, 0.006, 14.0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def rack_scene(lean_deg):
    """A free rod whose bottom rests on the table against a stop and whose shaft leans on a fixed bar."""
    th = np.radians(lean_deg)
    a = np.array([0.0, -np.sin(th), np.cos(th)]); n = np.array([0.0, -np.cos(th), -np.sin(th)])
    B = np.array([0.0, 0.0, R_ROD * np.sin(th) + 0.0005])
    C = B + a * HL_ROD
    rack = ""
    if lean_deg > 0:
        bar = B + a * 0.14 + n * (R_ROD + R_BAR + 0.0002)
        stop = B + np.array([0.0, R_ROD + 0.006, 0.006])
        rack = (f'<body name="rack"><geom name="bar" type="capsule" size="{R_BAR} 0.04" pos="{bar[0]} {bar[1]} {bar[2]}" euler="0 90 0"/>'
                f'<geom name="post" type="box" size="0.006 0.006 {bar[2] / 2}" pos="0.045 {bar[1]} {bar[2] / 2}"/>'
                f'<geom name="stop" type="box" size="0.02 0.004 0.006" pos="{stop[0]} {stop[1]} {stop[2]}"/></body>')
    p = TMP / f"rod{int(lean_deg)}.xml"
    p.write_text(f'<mujoco model="rod"><worldbody>{rack}<body name="rod" pos="{C[0]} {C[1]} {C[2]}" euler="{lean_deg} 0 0"><freejoint/>'
                 f'<geom name="g" type="cylinder" size="{R_ROD} {HL_ROD}" mass="0.08" friction="1 0.01 0.001"/></body></worldbody></mujoco>')
    return {"hand_xml": HAND, "side": "right", "floor": True, "grasp_body": "lean_rod",
            "objects": [{"name": "lean", "xml": str(p), "pos": [0.0, 0.45, 0.0]}]}


def closing_turn(scene, pose):
    case = GR.Case(GR.load_scene(scene))
    return GR.close_like_executor(case, pose)["max_rot_deg"]


def main():
    kw = dict(include_top=False, include_pinch=False, yaw_step_deg=90.0, thumb_oppose_fracs=(0.0,))
    lean = rack_scene(LEAN_DEG)
    model, data0, rig, ids = H.open_scene(lean)
    b = H.grasp_body(model, lean, ids)
    true_axis = data0.xmat[b].reshape(3, 3)[:, 2]
    # 1. the axis
    u, tilt = G.object_axis(model, data0, sorted(ids["grasp_gids"]), b)
    check("the leaning rod's axis is found", abs(tilt - LEAN_DEG) < 0.5 and float(u @ true_axis) > np.cos(np.radians(0.5)),
          f"tilt {tilt:.2f} deg, axis {np.round(u, 3).tolist()}")
    # 2. every side wrap approaches perpendicular to it, and out_json writes the ranked list
    out = TMP / "lean_grasps.json"
    g = G.generate_rigid_grasps(lean, out_json=str(out), **kw)
    side = [c for c in g["grasps"] if c["family"].startswith("side")]
    worst = max((abs(float(np.asarray(c["approach"]) @ u)) for c in side), default=1.0)
    check("every side wrap of the leaning rod approaches perpendicular to its axis", side and worst < 1e-6 and abs(g["object_axis"]["tilt_deg"] - LEAN_DEG) < 0.5,
          f"{len(side)} wraps, worst |approach . axis| {worst:.2e}")
    written = json.loads(out.read_text()) if out.exists() else None
    check("out_json writes the ranked grasp list", written is not None and len(written) == len(g["grasps"]) and g.get("out_json") == str(out)
          and all(w["family"] == c["family"] for w, c in zip(written, g["grasps"])), f"{None if written is None else len(written)} poses")
    # 3. an opposed, physically feasible wrap closes without turning the rod.
    # A fixed across-lean azimuth used to pass only because hand self contacts
    # were disabled; do not restore that invalid grasp as a positive fixture.
    opposed = [c for c in side if c["hold"]["opposed"]]
    t_axis = min((closing_turn(lean, c) for c in opposed), default=None)
    G.AXIS_MIN_TILT_DEG = 90.0
    try:
        gv = G.generate_rigid_grasps(lean, **kw)
    finally:
        G.AXIS_MIN_TILT_DEG = 3.0
    across_v = [c for c in gv["grasps"] if c["family"].startswith("side") and abs(c["approach"][0]) > 0.9]
    t_vert = min((closing_turn(lean, c) for c in across_v), default=None)
    wrong_axis = max((abs(float(np.asarray(c["approach"]) @ u)) for c in gv["grasps"]), default=0.)
    check("the vertical-axis control violates perpendicular approach to the tilted rod", wrong_axis > np.sin(np.radians(10)), wrong_axis)
    check("an opposed axis-aligned wrap closes gently; the across-lean vertical control turns the rod", t_axis is not None and t_axis < 5.0
          and t_vert is not None and t_vert > 10.0, f"turned {t_axis if t_axis is None else round(t_axis, 1)} deg about its axis, "
          f"{t_vert if t_vert is None else round(t_vert, 1)} deg about the vertical")
    # 4. upright: the vertical exactly, wraps horizontal
    up = rack_scene(0.0)
    mu, d_u, rig_u, ids_u = H.open_scene(up)
    uu, tu = G.object_axis(mu, d_u, sorted(ids_u["grasp_gids"]), H.grasp_body(mu, up, ids_u))
    gu = G.generate_rigid_grasps(up, **kw)
    flat = max((abs(c["approach"][2]) for c in gu["grasps"] if c["family"].startswith("side")), default=1.0)
    check("an upright rod keeps the vertical exactly and horizontal wraps", list(uu) == [0.0, 0.0, 1.0] and gu["n_feasible"] > 0 and flat == 0.0,
          f"tilt {tu:.2f} deg, {gu['n_feasible']} grasps")
    print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
