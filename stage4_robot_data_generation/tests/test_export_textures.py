"""Textures in the Blender export: every textured geom is drawn with its texture or named as dropped.

A scene with a textured box (a whiteboard's ink on a primitive), a textured cylinder, a textured mesh that has no
texture coordinates (it used to render white), a mesh with its own UVs, and a box with a cube-map texture (which
Stage 4 cannot draw). The export must give the primitives a projection, box-project UVs onto the bare mesh,
keep the mapped mesh's UVs, and name the cube-mapped box as dropped, in the log and in every sample.json; when
Blender (bpy) is installed the built meshes must carry UV layers that cover the texture.

    cd stage4_robot_data_generation && python tests/test_export_textures.py
"""
import json
import os
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import synthetic  # noqa: E402
from stage4 import export  # noqa: E402
from stage4.config import load_config  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="stage4_tex_", dir=os.environ.get("TMPDIR")))
FAILS = 0

XML = """<mujoco>
  <compiler angle="radian"/>
  <asset>
    <texture name="ink" type="2d" builtin="checker" width="64" height="64" rgb1="1 1 1" rgb2="0.1 0.1 0.6"/>
    <texture name="cubemap" type="cube" builtin="checker" width="32" height="32" rgb1="1 0 0" rgb2="0 1 0"/>
    <material name="board" texture="ink"/>
    <material name="can" texture="ink" texrepeat="2 1"/>
    <material name="cupmat" texture="ink"/>
    <material name="cubemat" texture="cubemap"/>
    <mesh name="bare" vertex="0 0 0  0.05 0 0  0 0.05 0  0 0 0.05"/>
    <mesh name="mapped" vertex="0 0 0  0.05 0 0  0 0.05 0  0 0 0.05" face="0 2 1  0 1 3  0 3 2  1 2 3" texcoord="0 0  1 0  0 1  1 1"/>
  </asset>
  <worldbody>
    <body name="board" pos="0.2 0.4 0.01"><freejoint/><geom name="board" type="box" size="0.1 0.06 0.01" material="board"/></body>
    <body name="can" pos="0.4 0.4 0.03"><freejoint/><geom name="can" type="cylinder" size="0.02 0.03" material="can"/></body>
    <body name="cup" pos="0.0 0.4 0.0"><freejoint/><geom name="cup" type="mesh" mesh="bare" material="cupmat"/></body>
    <body name="sticker" pos="0.0 0.6 0.0"><freejoint/><geom name="sticker" type="mesh" mesh="mapped" material="cupmat"/></body>
    <body name="dice" pos="0.3 0.6 0.02"><freejoint/><geom name="dice" type="box" size="0.02 0.02 0.02" material="cubemat"/></body>
    <body name="plain" pos="0.5 0.6 0.02"><freejoint/><geom name="plain" type="box" size="0.02 0.02 0.02" rgba="0.2 0.8 0.2 1"/></body>
  </worldbody>
</mujoco>
"""


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def main():
    task = TMP / "task"
    task.mkdir()
    (task / "scene.xml").write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(task / "scene.xml"))
    d = mujoco.MjData(m)
    Q = np.tile(d.qpos.copy(), (4, 1))
    np.savez_compressed(task / "trajectory.npz", qpos=Q, fps=30.0, labels=np.array(["left: wait"] * 4))
    synthetic.write_common(task, {"robot": {"end_effectors": {"x": "board"}}, "num_samples": 1})
    cfg = load_config(task / "stage4.json")
    rep = {}
    geoms = {e["name"]: e for e in export.export_geoms(m, cfg, TMP / "shared", rep)}

    b, c = geoms["board"], geoms["can"]
    check("box and cylinder textures are projected", b.get("uv") == "planar_xy" and Path(b.get("texture", "")).exists()
          and c.get("uv") == "cylindrical" and c.get("texrepeat") == [2.0, 1.0], f"{b.get('uv')} {c.get('uv')}")
    cup = geoms["cup"]
    obj = Path(cup["obj"]).read_text()
    nvt = sum(1 for line in obj.splitlines() if line.startswith("vt "))
    check("a mesh without UVs gets box-projected UVs", cup.get("uv") == "box_projection" and "texture" in cup and nvt == 3 * m.mesh_facenum[0]
          and all("/" in line for line in obj.splitlines() if line.startswith("f ")), f"{nvt} vt")
    st = geoms["sticker"]
    check("a mesh with its own UVs keeps them", "texture" in st and "uv" not in st and "vt 0.000000 1.000000" in Path(st["obj"]).read_text())
    check("a cube-map texture is named as dropped, an untextured geom is left alone",
          len(rep["dropped"]) == 1 and "dice" in rep["dropped"][0] and "texture" not in geoms["dice"] and "texture" not in geoms["plain"]
          and sorted(rep["generated_uv"]) == ["cup/cup/bare"] and len(rep["projected"]) == 2, json.dumps(rep)[:160])
    uv, ft = export.box_projection_uv(m.mesh_vert[:4], m.mesh_face[:4])
    check("box projection spans the texture", uv.min() >= 0.0 and uv.max() <= 1.0 and np.ptp(uv[:, 0]) > 0.9 and ft.shape == (4, 3))

    # the export step records the texture report in every sample.json and logs every dropped texture
    s0 = cfg.out / "samples" / "0000"
    s0.mkdir(parents=True)
    (s0 / "scene_physics.xml").write_bytes((task / "scene.xml").read_bytes())
    np.savez_compressed(s0 / "trajectory.npz", qpos=Q, fps=30.0, labels=np.array(["left: wait"] * 4))
    (s0 / "sample.json").write_text(json.dumps({"layout": {}, "body_overrides": {}}))
    (cfg.out / "plate").mkdir(parents=True)
    cv2.imwrite(str(cfg.out / "plate" / "plate.png"), np.full((48, 64, 3), 128, np.uint8))
    cfg["cameras"]["width"], cfg["cameras"]["height"] = 64, 48
    logs = []
    # Texture-only fixture has no robot dynamics; release-boundary tests live
    # in test_verified_outputs.py and test_runtime_replay.py.
    with patch.object(export, "validate_sample"):
        export.run(cfg, log=logs.append)
    info = json.loads((s0 / "sample.json").read_text())
    check("export.run records textures in sample.json and warns on drops", info.get("textures", {}).get("dropped") == rep["dropped"]
          and any("warning: texture dropped on dice" in line for line in logs), logs[0][:120])

    try:
        import bpy  # noqa: F401
    except ImportError:
        print("[SKIP] bpy not installed: Blender UV layers not checked")
    else:
        from stage4 import render_cycles as RC
        import bpy
        mb, mc, mm = RC.build_mesh(bpy, b), RC.build_mesh(bpy, cup), RC.build_mesh(bpy, c)
        uvb = np.array([d_.uv[:] for d_ in mb.uv_layers[0].data]) if mb.uv_layers else np.zeros((0, 2))
        check("Blender meshes carry the UV layers", len(mb.uv_layers) == 1 and len(mc.uv_layers) == 1 and len(mm.uv_layers) == 1
              and np.allclose([uvb.min(), uvb.max()], [0.0, 1.0]), f"box uv range {uvb.min():.2f}..{uvb.max():.2f}")
    print(f"{'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
