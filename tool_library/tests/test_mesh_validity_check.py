"""mesh_validity_check: plausible closed meshes pass; millimetre-scale slips, open volume-massed meshes and flat collision
meshes fail.

    python tool_library/tests/test_mesh_validity_check.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.mesh_validity_check import mesh_validity_check  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


tmp = Path(tempfile.mkdtemp(prefix="mesh_validity_"))
V = "v -1 -1 -1\nv 1 -1 -1\nv 1 1 -1\nv -1 1 -1\nv -1 -1 1\nv 1 -1 1\nv 1 1 1\nv -1 1 1\n"
TOP = "f 5 6 7\nf 5 7 8\n"
SIDES = "f 1 3 2\nf 1 4 3\nf 1 2 6\nf 1 6 5\nf 2 3 7\nf 2 7 6\nf 3 4 8\nf 3 8 7\nf 4 1 5\nf 4 5 8\n"
(tmp / "box.obj").write_text(V + SIDES + TOP)          # closed cube, outward faces
(tmp / "openbox.obj").write_text(V + SIDES)             # a cup-like box without its top
(tmp / "flat.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n")
# a 4 cm square plate 5 micrometres thick (closed)
(tmp / "thin.obj").write_text("v 0 0 0\nv .04 0 0\nv .04 .04 0\nv 0 .04 0\nv 0 0 5e-6\nv .04 0 5e-6\nv .04 .04 5e-6\nv 0 .04 5e-6\n"
                              + SIDES + TOP)


def scene(name, mesh, scale, extra="", inertia="legacy", body_extra=""):
    xml = (f'<mujoco><asset><mesh name="m" file="{mesh}" scale="{scale} {scale} {scale}" inertia="{inertia}"/></asset>'
           f'<worldbody><geom name="floor" type="plane" size="1 1 .1"/>'
           f'<body name="obj" pos="0 0 .1"><freejoint/>{body_extra}<geom name="g" type="mesh" mesh="m" {extra}/>'
           f'<geom name="handle" type="box" size=".01 .01 .03" pos=".05 0 0" mass="0.01"/></body></worldbody></mujoco>')
    p = tmp / f"{name}.xml"
    p.write_text(xml)
    return str(p)


# 1. setup scene (robot meshes, explicit inertials) and a closed 4 cm cube pass
r0 = mesh_validity_check(str(ROOT / "setup" / "scene.xml"))
r1 = mesh_validity_check(scene("good", "box.obj", 0.02))
check("setup scene and a closed 4 cm cube pass", r0["ok"] and r1["ok"], f"{r0['issues']} {r1['issues']}")

# 2. unit slips: authored in millimetres (40 m across) and scaled twice (0.04 mm across)
r2 = mesh_validity_check(scene("mm_as_m", "box.obj", 20.0))
r3 = mesh_validity_check(scene("double_scaled", "box.obj", 0.00002))
check("a 40 m cube and a 0.04 mm cube are reported as unit slips",
      not r2["ok"] and "millimetres" in r2["issues"][0] and not r3["ok"] and "scaled twice" in r3["issues"][0], f"{r2['issues']} {r3['issues']}")

# 3. an open mesh carrying mass from its volume fails; the same mesh with convex inertia, explicit mass or an
#    explicit body inertial passes
r4 = mesh_validity_check(scene("open_legacy", "openbox.obj", 0.04, extra='density="1000"'))
r5 = mesh_validity_check(scene("open_convex", "openbox.obj", 0.04, extra='density="1000"', inertia="convex"))
r6 = mesh_validity_check(scene("open_mass", "openbox.obj", 0.04, extra='mass="0.1"'))
r7 = mesh_validity_check(scene("open_inertial", "openbox.obj", 0.04, body_extra='<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'))
check("an open volume-massed mesh fails; convex inertia, an explicit mass or an explicit inertial pass",
      not r4["ok"] and "open surface" in r4["issues"][0] and r5["ok"] and r6["ok"] and r7["ok"],
      f"{r4['issues']} | {r5['issues']} | {r6['issues']} | {r7['issues']}")

# 4. a coplanar mesh is refused by MuJoCo (reported as a compile failure); a 5 micrometre thin collision mesh compiles
#    (shell inertia) but is flagged as degenerate; the same thin mesh as a visual geom passes
r8 = mesh_validity_check(scene("flat", "flat.obj", 0.05, extra='mass="0"', body_extra='<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'))
INERT = '<inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>'
r9 = mesh_validity_check(scene("thin_col", "thin.obj", 1.0, extra='mass="0"', inertia="shell", body_extra=INERT))
r10 = mesh_validity_check(scene("thin_vis", "thin.obj", 1.0, extra='mass="0" contype="0" conaffinity="0"', inertia="shell", body_extra=INERT))
check("a coplanar mesh fails at compile; a 5 um thin collision mesh is degenerate; the same mesh as a visual geom passes",
      not r8["ok"] and "coplanar" in str(r8.get("compile_error")) and not r9["ok"] and "degenerate" in r9["issues"][0] and r10["ok"],
      f"{r8['issues']} | {r9['issues']} | {r10['issues']}")

sys.exit(1 if fails else 0)
