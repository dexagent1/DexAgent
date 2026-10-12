import sys
import tempfile
from pathlib import Path

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_articulated import construct_articulated
from tool_library.skills.port_urdf_asset import port_urdf_asset
from tool_library.skills.construct_plush_toy import construct_plush_toy
from tool_library.verifiers.drawer_receive_and_close_test import drawer_receive_and_close_test


def drawer(tmp, tag, width, depth, height, rows, moving, fixed_base, travel_frac=0.7):
    out = Path(tmp) / tag
    a = construct_articulated(str(out / "urdf"), width=width, depth=depth, height=height, rows=rows,
                              moving=moving, travel_frac=travel_frac, backend="builtin", preview=False,
                              name=tag, description="test drawer")
    p = port_urdf_asset(a["urdf"], str(out), name=tag, fixed_base=fixed_base, include_floor=False,
                        joint_damping=0.5, joint_frictionloss=0.05)
    return p["xml"], tag + "_" + a["moving_joint"]


def item(tmp, tag, length, width, height):
    return construct_plush_toy(str(Path(tmp) / tag), name=tag, length=length, width=width, height=height,
                               include_floor=False)["xml"]


fails = 0


def case(name, cond, rep):
    global fails
    if cond:
        print(f"[PASS] {name}")
    else:
        fails += 1
        print(f"[FAIL] {name}: {rep.get('issues')} rest={rep.get('rest')} close={rep.get('close')}")


with tempfile.TemporaryDirectory() as tmp:
    # 1: small single-drawer box (fixed base) with a small plush
    dx, _ = drawer(tmp, "boxA", 0.091, 0.19, 0.061, [{"cols": 1}], {"row": 0, "col": 0}, True)
    r = drawer_receive_and_close_test(dx, item(tmp, "toyA", 0.038, 0.033, 0.030))
    case("single drawer box + small plush", r["ok"] and r["close"]["closed"] and r["rest"]["inside"], r)

    # 2: two-row cabinet with a FREE root (held by the test), lower drawer moved, explicit joint
    dx, j = drawer(tmp, "cabB", 0.30, 0.28, 0.24, [{"cols": 1}, {"cols": 1}], {"row": 1, "col": 0}, False)
    r = drawer_receive_and_close_test(dx, item(tmp, "toyB", 0.08, 0.06, 0.05), joint=j, open_frac=0.8)
    case("two-row cabinet, free root held, lower drawer", r["ok"] and r["held_free_root"] and r["joint"] == j, r)

    # 3: wide 2-column top row cabinet, larger plush, partial opening
    dx, _ = drawer(tmp, "cabC", 0.40, 0.30, 0.20, [{"cols": 2}], {"row": 0, "col": 0}, True)
    r = drawer_receive_and_close_test(dx, item(tmp, "toyC", 0.10, 0.07, 0.05), open_frac=0.9, close_force_N=15)
    case("two-column drawer + larger plush", r["ok"], r)

    # 4: edge case: an item far larger than the bin must be reported, not passed
    dx, _ = drawer(tmp, "boxD", 0.09, 0.18, 0.06, [{"cols": 1}], {"row": 0, "col": 0}, True)
    r = drawer_receive_and_close_test(dx, item(tmp, "toyD", 0.25, 0.20, 0.15))
    case("oversized item is rejected with issues", (not r["ok"]) and len(r["issues"]) > 0, r)

    # 5: misuse raises ValueError naming the parameter
    try:
        drawer_receive_and_close_test(dx, item(tmp, "toyE", 0.03, 0.03, 0.02), open_frac=1.5)
        case("bad open_frac raises", False, {})
    except ValueError as e:
        case("bad open_frac raises", "open_frac" in str(e), {})

sys.exit(1 if fails else 0)
