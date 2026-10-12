"""Tests for plate_edge_pinch_pose on synthetic plates (a flat panel on a post, the same panel pinched from another
edge with the thumb underneath, and a tilted hinged panel with a pivot preference), plus misuse errors."""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

HAND = os.environ.get("DEXAGENT_HAND_XML", str(ROOT / "setup" / "robot" / "sharpa_right.xml"))

PANEL = """<mujoco model="panel"><worldbody>
  <body name="panel" pos="0 0 {z}" euler="{tilt} 0 0"><freejoint/>
    <geom name="panel_plate" type="box" size="0.12 0.10 0.0015" mass="0.1" rgba="0.8 0.2 0.2 1"/>
    <geom name="panel_post" type="cylinder" size="0.01 {post}" pos="0 0 -{post}" mass="0.05"/>
  </body></worldbody></mujoco>"""


def scene(tmp, z=0.09, tilt=0.0):
    p = Path(tmp) / f"panel_{z}_{tilt}.xml"
    p.write_text(PANEL.format(z=z, tilt=tilt, post=z / 2 - 0.001))
    return {"hand_xml": HAND, "side": "right", "objects": [{"name": "panel", "xml": str(p), "pos": [0, 0, 0]}], "floor": True}


def main():
    if not Path(HAND).exists():
        print(f"[SKIP] hand model not found at {HAND}")
        return 0
    from tool_library.skills.plate_edge_pinch_pose import plate_edge_pinch_pose
    from tool_library.common import hand as H
    import mujoco
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        cases = [
            ("flat panel, +x edge, thumb on top", scene(tmp), dict(edge="+x", s=0.5, thumb_face="up")),
            ("flat panel, -y edge, thumb underneath", scene(tmp), dict(edge="-y", s=0.3, thumb_face="down")),
            ("tilted panel, +x edge, pivot + finger axis", scene(tmp, z=0.11, tilt=12.0),
             dict(edge="+x", s=0.8, thumb_face="up", finger_axis_max_deg=60, pivot={"point": [0, 0.1, 0.11], "axis": [1, 0, 0]})),
        ]
        for name, sc, kw in cases:
            try:
                # A stale recorded anchor must never replace the materialized
                # object's current transform used to solve these contacts.
                sc["grasp_body_pose"] = dict(body="panel_panel", pos=[7., 8., 9.], quat_wxyz=[1., 0., 0., 0.])
                p = plate_edge_pinch_pose(sc, "panel_plate", yaw_step_deg=45, pitches_deg=(-60, -30, 0, 30, 60), iters=150,
                                          min_out_m=0.01, open_floor_clear_m=-0.01, **kw)
                ok = p["worst_residual_mm"] <= 3.0 and p["outside_m"] >= 0.01 and np.dot(p["approach"], p["edge_normal"]) > 0.5 \
                    and set(p["grip_fingers"]) == {"thumb", "index", "middle"} and p["finger_targets_squeezed"]
                if kw.get("finger_axis_max_deg") is not None:
                    ok = ok and p["finger_axis_deg"] <= kw["finger_axis_max_deg"] and "pivot_dist_m" in p
                model, data, rig, _ = H.open_scene(sc)
                gid = next(i for i in range(model.ngeom) if model.geom(i).name.endswith("panel_plate"))
                body = int(model.geom_bodyid[gid]); rot = data.geom_xmat[gid].reshape(3, 3)
                np.testing.assert_allclose(p["anchor"]["pos"], data.xpos[body], atol=1e-8)
                self_quat = np.zeros(4); mujoco.mju_mat2Quat(self_quat, data.xmat[body])
                np.testing.assert_allclose(p["anchor"]["quat_wxyz"], self_quat, atol=1e-8)
                assert p["anchor"]["body"] == model.body(body).name
                for finger, point in p["points"].items():
                    normal = np.asarray(p["normals"][finger])
                    local = rot.T @ (np.asarray(point) - data.geom_xpos[gid])
                    np.testing.assert_allclose(abs(local[2]), model.geom_size[gid, 2], atol=1e-8)
                    np.testing.assert_allclose(np.linalg.norm(normal), 1., atol=1e-8)
                    np.testing.assert_allclose(np.asarray(p["targets"][finger]) - point,
                                               normal * (H.pad_radius(model, rig, finger) + .001), atol=1e-8)
                np.testing.assert_allclose(p["normals"]["thumb"], -np.asarray(p["normals"]["index"]), atol=1e-8)
                print(("[PASS] " if ok else "[FAIL] ") + name, p["worst_residual_mm"], p["outside_m"], p.get("finger_axis_deg"))
                fails += 0 if ok else 1
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {name}: {type(e).__name__}: {e}"); fails += 1
        for name, kw in [("misuse: thickness axis as edge", dict(edge="+z")), ("misuse: bad thumb_face", dict(edge="+x", thumb_face="left")),
                         ("misuse: thumb not first", dict(edge="+x", fingers=["index", "thumb"]))]:
            try:
                plate_edge_pinch_pose(scene(tmp), "panel_plate", **kw)
                print(f"[FAIL] {name}: no error"); fails += 1
            except ValueError:
                print(f"[PASS] {name}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
