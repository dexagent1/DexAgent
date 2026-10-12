"""Tests for bead_on_cable_slide_test: two beads that must pass and one whose bore is too small, each [PASS] <case>."""
import math
import sys
import tempfile
from pathlib import Path

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.bead_on_cable_slide_test import bead_on_cable_slide_test


def ring_xml(path, r_in, r_out, h, n=8, mass=0.0003, solref="0.005 1"):
    """A plain holed bead: n box segments around a vertical bore on a free body (origin at the bottom centre)."""
    t = (r_out - r_in) / 2
    rm = (r_in + r_out) / 2
    half_w = rm * math.tan(math.pi / n) * 1.02
    geoms = []
    for i in range(n):
        a = 2 * math.pi * i / n
        geoms.append(f'<geom type="box" size="{t:.6g} {half_w:.6g} {h / 2:.6g}" pos="{rm * math.cos(a):.6g} {rm * math.sin(a):.6g} {h / 2:.6g}" '
                     f'euler="0 0 {a:.6g}" mass="{mass / n:.6g}" solref="{solref}" friction="0.3 0.01 0.001"/>')
    Path(path).write_text('<mujoco><compiler angle="radian"/><worldbody><body name="bead" pos="0 0 0"><freejoint/>'
                          + "".join(geoms) + '</body></worldbody></mujoco>')
    return path


def main():
    tmp = Path(tempfile.mkdtemp())
    ok_all = True

    # 1. pony-bead-sized ring (2 mm bore) on a 1.25 mm cord with a 4 mm bead column: threads on, slides, stops, stays on
    x = ring_xml(tmp / "b1.xml", 0.002, 0.004, 0.006)
    r = bead_on_cable_slide_test(str(x), 0.002, 0.006, cable_radius=0.00125, cap_radius=0.004, cap_length=0.04,
                                 cable_length=0.15, out_dir=str(tmp / "o1"))
    c1 = r["ok"] and r["insertion"]["passed_tip"] and r["insertion"]["stopped_at_stop"] and r["lateral"]["threaded"]
    print(("[PASS]" if c1 else "[FAIL]"), "pony bead on cord: threads, travels %.3f m, stops %.4f m short of the column, offset %.4f m" % (
        r["insertion"]["travel_m"], r["insertion"]["final_gap_to_stop_m"], r["lateral"]["max_offset_from_centreline_m"]), r["issues"])
    ok_all &= c1

    # 2. bore smaller than the cord: must be reported as not threading
    x = ring_xml(tmp / "b2.xml", 0.0009, 0.004, 0.006)
    r = bead_on_cable_slide_test(str(x), 0.0009, 0.006, cable_radius=0.00125, cap_radius=0.004, cap_length=0.04,
                                 cable_length=0.15, out_dir=str(tmp / "o2"))
    c2 = (not r["ok"]) and (not r["insertion"]["passed_tip"]) and any("did not thread" in s for s in r["issues"])
    print(("[PASS]" if c2 else "[FAIL]"), "bore 0.9 mm on a 1.25 mm cord is refused (passed_tip=%s)" % r["insertion"]["passed_tip"])
    ok_all &= c2

    # 3. a bigger washer-like ring (3 mm bore, 12 segments, heavier) on a 2 mm cable with a 6 mm stop, horizontal-axis bead frame
    x = ring_xml(tmp / "b3.xml", 0.003, 0.006, 0.004, n=12, mass=0.001)
    txt = Path(x).read_text().replace('<body name="bead" pos="0 0 0">', '<body name="bead" pos="0 0 0"><!-- vertical bore -->')
    Path(x).write_text(txt)
    r = bead_on_cable_slide_test(str(x), 0.003, 0.004, bead_axis="z", cable_radius=0.002, cap_radius=0.006, cap_length=0.05,
                                 cable_length=0.2, segments=30, push_force_N=0.1, out_dir=str(tmp / "o3"))
    c3 = r["ok"] and r["insertion"]["stopped_at_stop"] and r["lateral"]["threaded"]
    print(("[PASS]" if c3 else "[FAIL]"), "washer on a 2 mm cable with a 6 mm stop: travel %.3f m, gap %.4f m" % (
        r["insertion"]["travel_m"], r["insertion"]["final_gap_to_stop_m"]), r["issues"])
    ok_all &= c3

    # 4. a 0.25 g ring bead (construct_magnetic_ring without a magnet) under the 0.2 N sideways push: the "pinned" test
    #    cable used to swing ~10 cm on its soft end welds (0.03 g end links), so the bead stayed on or was thrown off
    #    depending on the timestep; now the pins hold and two timesteps give the same verdict
    from tool_library.skills.construct_magnetic_ring import construct_magnetic_ring
    xr = construct_magnetic_ring(tmp / "mr", name="bead", outer_radius=0.004, inner_radius=0.0018, height=0.005, n_segments=16,
                                 mass=0.00025, magnet_force_N=0.0)["xml"]
    rs = [bead_on_cable_slide_test(xr, 0.0018, 0.005, cable_radius=0.00125, cap_radius=0.004, cap_length=0.04, cable_length=0.15,
                                   timestep=dt, out_dir=str(tmp / f"o4_{dt}")) for dt in (0.002, 0.001)]
    c4 = all(r["ok"] and r["cable_pin_drift_m"] < 0.005 and r["lateral"]["threaded"] for r in rs)
    print(("[PASS]" if c4 else "[FAIL]"), "0.25 g ring bead, 0.2 N sideways: pins hold (drift %s mm), threaded at dt 0.002 and 0.001 (offsets %s mm)" % (
        [round(r["cable_pin_drift_m"] * 1000, 3) for r in rs], [round(r["lateral"]["max_offset_from_centreline_m"] * 1000, 2) for r in rs]),
        [r["issues"] for r in rs if r["issues"]])
    ok_all &= c4

    # 5. misuse
    try:
        bead_on_cable_slide_test(str(tmp / "b1.xml"), 0.002, 0.006, cable_radius=0.00125, colour="red")
        c5 = False
    except ValueError:
        c5 = True
    print(("[PASS]" if c5 else "[FAIL]"), "unknown parameter raises ValueError")
    ok_all &= c5
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
