import sys
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.verifiers.box_enclosure_check_tiled_bottom import box_enclosure_check_tiled_bottom as chk


def box_xml(L, W, H, t=0.004, tiled=False, walls=("xp", "xn", "yp", "yn"), lid=False, solid=False):
    g = []
    if solid:
        g.append(f'<geom type="box" size="{L/2} {W/2} {H/2}" pos="0 0 {H/2}"/>')
    elif tiled:
        r = 0.2 * min(L, W)
        g.append(f'<geom name="fa" type="box" size="{L/2 - r} {W/2} {t/2}" pos="0 0 {t/2}"/>')
        g.append(f'<geom name="fb" type="box" size="{r/2} {W/2 - r} {t/2}" pos="{L/2 - r/2} 0 {t/2}"/>')
        g.append(f'<geom name="fc" type="box" size="{r/2} {W/2 - r} {t/2}" pos="{-L/2 + r/2} 0 {t/2}"/>')
    else:
        g.append(f'<geom name="f" type="box" size="{L/2} {W/2} {t/2}" pos="0 0 {t/2}"/>')
    if not solid:
        if "xp" in walls: g.append(f'<geom type="box" size="{t/2} {W/2} {H/2}" pos="{L/2 - t/2} 0 {H/2}"/>')
        if "xn" in walls: g.append(f'<geom type="box" size="{t/2} {W/2} {H/2}" pos="{-L/2 + t/2} 0 {H/2}"/>')
        if "yp" in walls: g.append(f'<geom type="box" size="{L/2} {t/2} {H/2}" pos="0 {W/2 - t/2} {H/2}"/>')
        if "yn" in walls: g.append(f'<geom type="box" size="{L/2} {t/2} {H/2}" pos="0 {-W/2 + t/2} {H/2}"/>')
    if lid:
        g.append(f'<geom type="box" size="{L/2} {W/2} {t/2}" pos="0 0 {H + t/2}"/>')
    return f'<mujoco><worldbody><body name="bx" pos="0.1 0.2 0" euler="0 0 30"><freejoint/>{"".join(g)}</body></worldbody></mujoco>'


fails = 0
def case(name, cond):
    global fails
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name}"); fails += 1

r = chk(box_xml(0.30, 0.20, 0.10), "bx", expect_top="open")
case("single-slab open box passes", r["ok"] and len(r["walls_found"]) == 4)
r = chk(box_xml(0.21, 0.15, 0.06, tiled=True), "bx", expect_top="open")
case("rounded dish with tiled bottom passes, all four walls", r["ok"] and len(r["walls_found"]) == 4 and len(r["bottom_tiles"]) == 3)
r = chk(box_xml(0.12, 0.40, 0.05, tiled=True, lid=True), "bx", expect_top="closed")
case("lidded tiled box reads closed", r["ok"] and r["top_state"] == "closed")
r = chk(box_xml(0.21, 0.15, 0.06, tiled=True, walls=("yp", "yn")), "bx")
case("tray with two walls fails naming x walls", (not r["ok"]) and any("x_pos" in i for i in r["issues"]))
r = chk(box_xml(0.2, 0.2, 0.1, solid=True), "bx")
case("solid block fails", not r["ok"])
r = chk(box_xml(0.2, 0.2, 0.1), "nothing")
case("unknown body reported, not raised", (not r["ok"]) and r["issues"])
try:
    chk(box_xml(0.2, 0.2, 0.1), "bx", expect_top="ajar"); case("bad expect_top raises", False)
except ValueError:
    case("bad expect_top raises", True)
sys.exit(1 if fails else 0)
