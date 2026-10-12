"""place_egodex_camera_for_reach: where EgoDex's head camera stands so both arms reach every object.

Fixture (tests/fixtures/place_egodex_camera_for_reach/): EgoDex build_unstack_lego/59 read by read_egodex_episode
with the default cam_xy (0.019, 0.05): camera.json, episode.json, three frames of hands.npz, and the eight bricks'
frame-0 masks plus hands.png from that episode's Stage 2. With the default foot one brick is out of reach (the run's
agent swept cam_xy by hand to (-0.001, 0.095)).
Run: MUJOCO_GL=egl python tool_library/tests/test_place_egodex_camera_for_reach.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.common import rigcam as RC  # noqa: E402
from tool_library.skills.place_egodex_camera_for_reach import apply_transform, place_egodex_camera_for_reach  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "place_egodex_camera_for_reach"
FAILS = []
BRICK_H = 0.034           # a 2x2 brick with its studs: every object below shares one reach-field height


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
    if not cond:
        FAILS.append(name)


def fp(*centers):
    return {f"o{i}": {"center": list(c), "radius": 0.03, "height": BRICK_H} for i, c in enumerate(centers)}


def main():
    t0 = time.time()
    cam = RC.load(FIX / "camera.json")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # 1. objects beyond the band: the camera foot moves so they come in, verified by the exact arm IK
        t = time.time()
        r = place_egodex_camera_for_reach(d / "far", camera=cam, footprints=fp((-0.12, 0.74), (0.12, 0.76), (0.0, 0.72)),
                                          field_step_m=0.06, y_range=(-0.3, 0.1), yaw_range_deg=(0, 0))
        field = json.loads(Path(r["files"]["reach_field_json"]).read_text())
        o = r["objects"]
        check("objects beyond the band are brought into reach",
              r["ok"] and not all(v["reachable_before"] for v in o.values()) and all(v["reachable_after"] for v in o.values())
              and r["worst_margin_after_m"] >= 0.03 and r["transform"]["dy"] < -0.05,
              f"cam_xy {r['cam_xy']}, worst margin {1000 * r['worst_margin_before_m']:.0f} -> {1000 * r['worst_margin_after_m']:.0f} mm, "
              f"reachable before {sum(v['reachable_before'] for v in o.values())}/3, {time.time() - t:.0f} s")
        # 2. objects already well inside the band: the foot hardly moves
        from tool_library.common import human_prior as HP
        xs, ys, maps = HP.field_margin_maps(field)
        deep = []
        for side in ("left", "right"):                       # each station's deepest point of its band
            M = maps[side][0]
            iy, ix = np.unravel_index(int(np.argmax(M)), M.shape)
            deep.append((float(xs[ix]), float(ys[iy])))
        r = place_egodex_camera_for_reach(d / "fine", camera=cam, footprints=fp(*deep), field=field)
        mv = float(np.hypot(r["transform"]["dx"], r["transform"]["dy"]))
        check("objects already in the band: the camera stays (smallest move wins)",
              r["ok"] and mv <= 0.02 and abs(r["yaw_deg"]) <= 2.5, f"moved {1000 * mv:.0f} mm, yaw {r['yaw_deg']}, worst margin {1000 * r['worst_margin_after_m']:.0f} mm")
        # 3. two objects further apart than the band is deep: no placement satisfies both; report the best compromise
        r = place_egodex_camera_for_reach(d / "split", camera=cam, footprints=fp((0.0, 0.25), (0.0, 0.95)), field=field, verify=False)
        check("an impossible layout returns the best compromise and ok=False",
              (not r["ok"]) and "no placement" in r["reason"] and r["worst_margin_after_m"] >= r["worst_margin_before_m"],
              f"worst margin {1000 * r['worst_margin_before_m']:.0f} -> {1000 * r['worst_margin_after_m']:.0f} mm")
        # 4. the real lego episode from its masks: re-framed files consistent, every brick reachable
        t = time.time()
        r = place_egodex_camera_for_reach(d / "lego", episode_dir=FIX, masks=FIX / "masks", heights={k: BRICK_H for k in (
            "cream_brick", "lavender_brick_back", "lavender_brick_front", "lime_green_brick", "pink_brick", "sage_green_brick_back",
            "sage_green_brick_right", "yellow_brick")}, field=field)
        o = r["objects"]
        n_before = sum(v["reachable_before"] for v in o.values())
        check("real EgoDex lego episode: every brick in reach after the move (exact IK), worst margin raised",
              all(v["reachable_after"] for v in o.values()) and r["worst_margin_after_m"] > max(r["worst_margin_before_m"], 0.0),
              f"cam_xy {r['cam_xy']} yaw {r['yaw_deg']} (agent's manual sweep: [-0.001, 0.095], yaw 0); reachable {n_before}/8 -> "
              f"{sum(v['reachable_after'] for v in o.values())}/8; worst margin {1000 * r['worst_margin_before_m']:.0f} -> "
              f"{1000 * r['worst_margin_after_m']:.0f} mm; {time.time() - t:.0f} s")
        # re-framing: the new hands.npz / episode.json / camera.json describe the same world in the moved rig
        with np.load(FIX / "hands.npz") as a, np.load(r["files"]["hands_npz"]) as b:
            z0, z1 = {k: a[k] for k in a.files}, {k: b[k] for k in b.files}
        P0 = np.asarray(z0["rig_xyz"][0], float)
        Pw = np.asarray(z0["origin_world"]) + P0 @ np.asarray(z0["R_world_rig"]).T
        P1 = (Pw - np.asarray(z1["origin_world"])) @ np.asarray(z1["R_world_rig"])
        moved = np.c_[apply_transform(r["transform"], P0[:, :2]), P0[:, 2]]
        c1 = RC.load(r["files"]["camera_json"])
        ep = json.loads(Path(r["files"]["episode_json"]).read_text())
        cw0 = np.asarray(z0["origin_world"]) + np.asarray(cam["pos"]) @ np.asarray(z0["R_world_rig"]).T
        cw1 = np.asarray(ep["rig"]["origin_world"]) + np.asarray(c1["pos"]) @ np.asarray(ep["rig"]["R_world_rig"]).T
        err = max(float(np.abs(P1 - np.asarray(z1["rig_xyz"][0])).max()), float(np.abs(moved - np.asarray(z1["rig_xyz"][0])).max()))
        check("re-framed hands.npz, episode.json and camera.json agree with the world and with the transform",
              err < 1e-4 and np.allclose(c1["pos"][:2], r["cam_xy"], atol=1e-6) and np.allclose(cw0, cw1, atol=1e-6) and ep["rig"]["cam_xy"] == r["cam_xy"],
              f"worst joint mismatch {1000 * err:.3f} mm")
    # 5. an object lying flat (a ribbon across the frame): footprint on the table, not a tall column
    import cv2
    from tool_library.skills.place_egodex_camera_for_reach import _flat_footprint, _footprints_from_masks
    W, H = int(cam["width"]), int(cam["height"])
    strip = np.zeros((H, W), np.uint8)
    cv2.line(strip, (int(0.2 * W), int(0.75 * H)), (int(0.8 * W), int(0.55 * H)), 255, max(3, H // 60))
    sp = Path(tempfile.mkdtemp()) / "ribbon.png"; cv2.imwrite(str(sp), strip)
    flat = _flat_footprint(sp, cam, 0.02)
    standing, _ = _footprints_from_masks({"ribbon": str(sp)}, cam, None)
    on_strip = RC.project(cam, np.array([[*flat["center"], 0.0]]))[0][0]
    check("a lying strip's flat footprint is 2 cm high, its centre on the strip, narrower than the standing reading",
          flat["height"] == 0.02 and strip[int(round(on_strip[1])), int(round(on_strip[0]))] > 0 and flat["radius"] <= 0.05
          and standing["ribbon"]["height"] > 0.05,
          f"flat r {flat['radius']:.3f} m; standing reading h {standing['ribbon']['height']:.3f} m")
    # a long lying strip is placed as zones along its length, its ends included
    from tool_library.skills.place_egodex_camera_for_reach import _flat_zones
    zones = _flat_zones(sp, cam, 0.02)
    ends = [np.asarray(z["center"]) for z in zones.values()]
    span = max(np.linalg.norm(a - b) for a in ends for b in ends) if len(ends) > 1 else 0.0
    e0, e1 = RC.pixel_on_plane(cam, np.array([[0.2 * W, 0.75 * H], [0.8 * W, 0.55 * H]]), 0.0)[:, :2]
    length = float(np.linalg.norm(e1 - e0))
    check("a long lying strip is split into zones along its length", len(zones) >= 2 and span > 0.5 * length,
          f"{len(zones)} zones spanning {span:.3f} m of a {length:.3f} m strip")
    # 6. only the listed objects are placed: a helper mask in the folder is ignored
    only, _ = _footprints_from_masks({"ribbon": str(sp), "ribbon_helper": str(sp)}, cam, None, flat=["ribbon"], objects=["ribbon"])
    check("objects= keeps only the listed masks", all(k.startswith("ribbon") and not k.startswith("ribbon_helper") for k in only), str(list(only)))
    # 7. misuse
    bad = 0
    for kw in ({}, {"camera": cam}, {"footprints": {"a": {"radius": 0.03}}, "camera": cam}):
        try:
            place_egodex_camera_for_reach(tempfile.mkdtemp(), **kw)
        except ValueError:
            bad += 1
    check("misuse raises ValueError (no inputs, no objects, footprint without centre)", bad == 3, f"{bad}/3")
    print(f"{len(FAILS)} failed, {time.time() - t0:.0f} s")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
