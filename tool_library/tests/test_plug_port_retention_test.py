"""plug_port_retention_test on construct_phone ports: by default in the episode's physics (dt 0.002, implicitfast,
elliptic cones, noslip 5) with the executor's connector latch on the test plug; and as a pure friction fit in the
fine-timestep regime the port was first tuned in."""
import sys
import tempfile

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.construct_phone import construct_phone
from tool_library.verifiers.plug_port_retention_test import plug_port_retention_test

# the friction regime the port's grip was tuned in (the old defaults): fine timestep, no noslip, the asset's own options
FRICTION_REGIME = dict(latch=False, timestep=0.0002, noslip_iterations=0, integrator="Euler", cone="pyramidal", impratio=1.0)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    fail += 0 if ok else 1


with tempfile.TemporaryDirectory() as td:
    r = plug_port_retention_test(construct_phone(td + "/a"))
    lr = r.get("latch_report") or {}
    check("default phone port, latch, episode physics: seats, latches, holds, pulls out near release_N, blocked",
          r["ok"] and r["retention"] == "latch" and r["physics"]["episode_physics"] and r["insert_depth_m"] > 0.0057
          and lr.get("engaged_in_insert") and lr.get("engaged_through_hold") and not lr.get("engaged_in_blocked")
          and 12.0 <= r["extract_force_N"] <= 20.0 and r["hold_slip_m"] < 0.0005 and "latch" in r["verdict"] and "episode's physics" in r["verdict"],
          f"seats {r.get('insert_depth_m', 0)*1000:.2f} mm, slip {r.get('hold_slip_m', 0)*1000:.3f} mm, out at {r.get('extract_force_N')} N; {r['verdict'][:90]}")
    # the hold report is self-consistent: start depth - end depth = slip, the creep is in the verdict too (the report
    # used to show a full 'seated depth' next to a 3 mm slip and a verdict saying it holds by friction)
    cons = (abs(r["hold_start_depth_m"] - r["hold_end_depth_m"] - r["hold_slip_m"]) < 2e-5 and "hold_seated_depth_m" not in r)
    if r["hold_creep_warning"]:
        cons = cons and "creeps out" in r["verdict"] and r["hold_seconds_to_release"] is not None and r["hold_creep_m_per_s"] > 0
    check("hold report consistent", cons, f"{r['hold_start_depth_m']*1000:.2f} -> {r['hold_end_depth_m']*1000:.2f} mm, slip {r['hold_slip_m']*1000:.3f} mm")

    r8 = plug_port_retention_test(construct_phone(td + "/a8"), release_N=8.0)
    check("release_N sets the pull-out force", r8["ok"] and 6.5 <= r8["extract_force_N"] <= 10.0 and r8["latch_report"]["release_force_N"] is not None,
          f"release_N 8 -> out at {r8['extract_force_N']} N (latch let go at {r8['latch_report']['release_force_N']} N)")

    r = plug_port_retention_test(construct_phone(td + "/b", length=0.165, width=0.078, thickness=0.0085, port_offset_x=0.003))
    check("larger phone with an offset port", r["ok"], r["verdict"][:120])

    r = plug_port_retention_test(construct_phone(td + "/c", port_grip_m=0.0))
    check("loose port (no grip) is held by the latch", r["ok"] and r["latch_report"]["engaged_through_hold"], r["verdict"][:120])
    r = plug_port_retention_test(construct_phone(td + "/c2", port_grip_m=0.0), **FRICTION_REGIME)
    check("loose port as a pure friction fit is caught: plug falls out", (not r["ok"]) and r["retention"] == "friction"
          and r["latch_report"] is None and any("fell out" in i for i in r["issues"]) and "no latch" in r["verdict"], r["verdict"][:120])

    r = plug_port_retention_test(construct_phone(td + "/f"), **FRICTION_REGIME)
    check("default port as a friction fit in its tuning regime: seats, holds, pulls out", r["ok"] and r["retention"] == "friction"
          and not r["physics"]["episode_physics"] and r["extract_force_N"] >= 1.0, r["verdict"][:120])
    r = plug_port_retention_test(construct_phone(td + "/g"), latch=False)
    check("friction fit in the episode's physics: the report names the retention and physics tested",
          r["retention"] == "friction" and r["physics"]["episode_physics"] and r["latch_report"] is None and "friction fit, no latch" in r["verdict"]
          and "dt 0.002" in r["verdict"], r["verdict"][:140])

    r = plug_port_retention_test(construct_phone(td + "/d", port_grip_m=0.0002))
    check("over-tight port is caught: plug cannot seat", (not r["ok"]) and any("too tight" in i for i in r["issues"]), r["verdict"][:120])

    r = plug_port_retention_test(construct_phone(td + "/e"), seat_m=0.00002)
    check("a latch that never engages is caught", (not r["ok"]) and any("did not engage" in i for i in r["issues"]), r["verdict"][:140])

    r = plug_port_retention_test({"xml": td + "/missing.xml", "body": "x", "port": {}})
    check("bad input reported, not raised", (not r["ok"]) and r["issues"] and r["verdict"] == "NOT RUN")
sys.exit(1 if fail else 0)
