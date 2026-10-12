"""arm_workspace_map caches solved cells on disk per (robot signature, side, orientation, cell).

A workspace map took about ten minutes per side on a shared node, and agents asked for the same map again and again.
These cases check that a repeated call answers from the cache with the same numbers, that a scene that only adds
objects to the same bench reuses it, that moving a station invalidates it, and that cache=False solves everything.

    python tool_library/tests/test_arm_workspace_map_cache.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

from tool_library.verifiers.arm_workspace_map import SHIPPED_CACHE, arm_workspace_map, robot_signature  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_wscache_", dir=os.environ.get("TMPDIR")))
FAILS = 0
T0 = time.time()
SCENE = ROOT / "setup" / "scene.xml"


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def absolute_assets(spec, xml_path):
    base = Path(xml_path).resolve().parent
    for m in spec.meshes:
        if m.file and not Path(m.file).is_absolute():
            m.file = str(base / (spec.meshdir or "") / m.file)
    for t in spec.textures:
        if t.file and not Path(t.file).is_absolute():
            t.file = str(base / (spec.texturedir or "") / t.file)
    spec.meshdir = ""; spec.texturedir = ""
    return spec


def variant(name, edit):
    sp = absolute_assets(mujoco.MjSpec.from_file(str(SCENE)), SCENE)
    edit(sp)
    p = TMP / f"{name}.xml"; p.write_text(sp.to_xml())
    return str(p)


# a grid the shipped cache does not hold (z = 0.135), so the first call solves
KW = dict(orientations=["palm_down"], points={"spot": [0.15, 0.45, 0.135]}, x_range=[0.1, 0.2], y_range=[0.4, 0.5], z_m=0.135, step_m=0.1)
cache = TMP / "cache.json"

t = time.time(); cold = arm_workspace_map(str(SCENE), "right", cache=str(cache), **KW); t_cold = time.time() - t
t = time.time(); warm = arm_workspace_map(str(SCENE), "right", cache=str(cache), **KW); t_warm = time.time() - t
cells = lambda r: r["orientations"]["palm_down"]["cells"]
check("first call solves 4 cells + 1 point and writes the cache", cold["cache"]["solved"] == 5 and cold["cache"]["hits"] == 0 and cache.exists(),
      f"{cold['cache']} in {t_cold:.1f} s")
check("repeated call: all from the cache, same cells and verdict, faster", warm["cache"]["hits"] == 5 and warm["cache"]["solved"] == 0
      and cells(warm) == cells(cold) and warm["verdict"] == cold["verdict"] and warm["orientations"]["palm_down"]["points"] == cold["orientations"]["palm_down"]["points"]
      and t_warm < 0.5 * t_cold, f"{t_cold:.1f} s -> {t_warm:.1f} s")


def add_box(sp):
    b = sp.worldbody.add_body(); b.name = "extra_box"; b.pos = [0.0, 0.5, 0.03]
    g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [0.03, 0.03, 0.03]


with_obj = variant("with_object", add_box)
r = arm_workspace_map(with_obj, "right", cache=str(cache), **KW)
check("a scene that only adds an object has the same signature and reuses the cache", r["cache"]["hits"] == 5 and r["cache"]["solved"] == 0
      and r["cache"]["signature"] == cold["cache"]["signature"])


def move_station(sp):
    m = sp.compile()
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot_right_arm_gripper")
    b = g
    while int(m.body_parentid[b]) != 0:
        b = int(m.body_parentid[b])
    body = sp.body(m.body(b).name)
    body.pos = [body.pos[0], body.pos[1] + 0.05, body.pos[2]]


moved = variant("moved_station", move_station)
r = arm_workspace_map(moved, "right", cache=str(cache), **KW)
check("moving the station changes the signature: nothing is reused", r["cache"]["hits"] == 0 and r["cache"]["solved"] == 5
      and r["cache"]["signature"] != cold["cache"]["signature"])

off = arm_workspace_map(str(SCENE), "right", cache=False, **KW)
check("cache=False solves every cell and reads/writes nothing", off["cache"]["solved"] == 5 and off["cache"]["hits"] == 0 and off["cache"]["file"] is None
      and [c[:3] + [c[5]] for c in cells(off)] == [c[:3] + [c[5]] for c in cells(cold)])

# the shipped cache (the default grid of the default setup) is valid and matches this setup's signatures
ok_shipped, det = False, "missing"
if SHIPPED_CACHE.exists():
    d = json.loads(SHIPPED_CACHE.read_text())
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    sigs = {s: robot_signature(m, s) for s in ("left", "right")}
    keys = list(d.get("entries", {}))
    ok_shipped = all(any(k.startswith(f"{sigs[s]}|{s}|") for k in keys) for s in sigs) and SHIPPED_CACHE.stat().st_size < 2_000_000
    det = f"{len(keys)} entries, {SHIPPED_CACHE.stat().st_size // 1024} KB"
check("the shipped cache covers both stations of the default setup", ok_shipped, det)

print(f"{'ALL PASS' if not FAILS else f'{FAILS} FAILED'} ({time.time() - T0:.0f}s)")
sys.exit(1 if FAILS else 0)
