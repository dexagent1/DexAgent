"""table_workspace_bounds_audit: a box slid on the bench passes; a box pushed off the table edge, sunk into the
tabletop, lifted 80 cm, and moved outside a work area fail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.table_workspace_bounds_audit import table_workspace_bounds_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.03, 0.03, 0.03], pos=[0.0, 0.45, 0.03])])
q0 = S.home(m)
a = S.free_adr(m, "box")
T = 60


def path(dx=0.0, dy=0.0, dz=0.0):
    Q = np.repeat(q0[None], T, 0)
    Q[:, a:a + 3] = S.smooth(q0[a:a + 3], q0[a:a + 3] + [dx, dy, dz], T)
    return Q


C = S.Cases()


def slide_passes():
    r = table_workspace_bounds_audit(scene, S.save(tmp / "ok.npz", path(dx=0.1), "right: push box #1"))
    assert r["ok"], r["issues"]
    assert abs(r["table"]["top_z"]) < 1e-6 and r["bodies"]["box"]["lowest_mm"] > -1, r


def off_the_edge_fails():
    Q = path(dy=-0.5)                                                    # past the near edge at y = 0.05, and down
    Q[40:, a + 2] = np.linspace(0.03, -0.3, 20)
    r = table_workspace_bounds_audit(scene, S.save(tmp / "off.npz", Q, "right: push box #1"))
    kinds = {i.split(": ")[1].split(" on ")[0] for i in r["issues"]}
    assert not r["ok"] and {"off table", "fell"} <= kinds, r["issues"]


def sunk_and_high_fail():
    r = table_workspace_bounds_audit(scene, S.save(tmp / "sunk.npz", path(dz=-0.02), "right: press box #1"))
    assert not r["ok"] and r["worst"]["kind"] == "under_tabletop" and r["worst"]["value"] > 15, r["worst"]
    r = table_workspace_bounds_audit(scene, S.save(tmp / "high.npz", path(dz=0.8), "right: lift box #1"))
    assert not r["ok"] and any("too high" in s for s in r["issues"]), r["issues"]


def work_area():
    t = S.save(tmp / "wa.npz", path(dx=0.3), "right: push box #1")
    assert table_workspace_bounds_audit(scene, t)["ok"]
    r = table_workspace_bounds_audit(scene, t, work_area={"x": [-0.4, 0.2], "y": [0.05, 1.16]})
    assert not r["ok"] and any("outside work area" in s for s in r["issues"]), r["issues"]


def misuse():
    t = S.save(tmp / "m.npz", path(), "x: y")
    S.expect_value_error(lambda: table_workspace_bounds_audit(scene, t, work_area={"x": [1, 0], "y": [0, 1]}), "work_area")


C.run("a box slid 10 cm on the bench passes", slide_passes)
C.run("a box pushed off the near edge and falling fails (off table, fell)", off_the_edge_fails)
C.run("a box 2 cm inside the tabletop and one lifted 80 cm fail", sunk_and_high_fail)
C.run("a work area moves the verdict", work_area)
C.run("an inverted work area is refused", misuse)
C.exit()
