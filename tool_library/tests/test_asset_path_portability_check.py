"""asset_path_portability_check: relative, existing asset paths pass; absolute, missing, out-of-root and included ones fail.

    python tool_library/tests/test_asset_path_portability_check.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.asset_path_portability_check import asset_path_portability_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="asset_paths_"))
SETUP = ROOT / "setup" / "scene.xml"

# 1. the shipped setup references its meshes relatively
r = asset_path_portability_check(str(SETUP))
check("setup/scene.xml: every mesh path relative and present", r["ok"] and r["n_refs"] > 50, str(r["issues"])[:300])

# 2. a reused copy whose paths were made absolute (what reused scenes carry today)
txt = SETUP.read_text().replace('file="assets/', f'file="{ROOT}/setup/assets/')
(tmp / "abs").mkdir()
(tmp / "abs" / "scene.xml").write_text(txt)
r = asset_path_portability_check(str(tmp / "abs" / "scene.xml"))
check("absolute mesh paths fail with a relative replacement suggested",
      not r["ok"] and r["counts"]["absolute"] > 50 and r["worst"]["suggest"] and not os.path.isabs(r["worst"]["suggest"]), str(r)[:300])

# 3. a small primitive scene with a relative mesh passes; a missing mesh fails
cube = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
(tmp / "rel" / "meshes").mkdir(parents=True)
(tmp / "rel" / "meshes" / "tet.obj").write_text(cube)
scene = ('<mujoco><compiler meshdir="meshes"/><asset><mesh name="t" file="{f}" scale=".05 .05 .05"/></asset>'
         '<worldbody><body><freejoint/><geom type="mesh" mesh="t"/></body></worldbody></mujoco>')
(tmp / "rel" / "ok.xml").write_text(scene.format(f="tet.obj"))
(tmp / "rel" / "missing.xml").write_text(scene.format(f="nope.obj"))
r1 = asset_path_portability_check(str(tmp / "rel" / "ok.xml"))
r2 = asset_path_portability_check(str(tmp / "rel" / "missing.xml"))
check("relative mesh under meshdir passes; a missing one fails as missing",
      r1["ok"] and not r2["ok"] and r2["counts"]["missing"] == 1, f"{r1['issues']} {r2['issues']}")

# 4. containment: a path that leaves the allowed root, and an absolute path inside an <include>d file
(tmp / "outside.obj").write_text(cube)
(tmp / "rel" / "escape.xml").write_text(scene.format(f="../../outside.obj"))
r3 = asset_path_portability_check(str(tmp / "rel" / "escape.xml"), root=str(tmp / "rel"))
r3b = asset_path_portability_check(str(tmp / "rel" / "escape.xml"))
(tmp / "rel" / "assets.xml").write_text(f'<mujoco><asset><mesh name="t" file="{tmp / "outside.obj"}"/></asset></mujoco>')
(tmp / "rel" / "main.xml").write_text('<mujoco><include file="assets.xml"/><worldbody><geom type="mesh" mesh="t"/></worldbody></mujoco>')
r4 = asset_path_portability_check(str(tmp / "rel" / "main.xml"))
check("a path escaping root fails only when root is given; an absolute path in an included file is found",
      not r3["ok"] and r3["counts"]["outside_root"] == 1 and r3b["ok"] and not r4["ok"] and r4["counts"]["absolute"] == 1,
      f"{r3['issues']} {r3b['issues']} {r4['issues']}")

# 5. misuse: an XML string has no folder
try:
    asset_path_portability_check("<mujoco/>")
    check("an XML string raises ValueError", False)
except ValueError:
    check("an XML string raises ValueError", True)

sys.exit(1 if fails else 0)
