"""build_isaac_scene on long thin cords: the link length rule, the verdict on garbage results, the cord colour; every cord
resolved through isaac_cord (presets, explicit link counts), and a thick segmented rope's cord specification with its
settled link poses, accepted by isaac_deformable_rest_check.

    DEXAGENT_ISAAC_PYTHON=/path/to/isaac/bin/python python tool_library/tests/test_build_isaac_scene.py

A run reported success on garbage: at 206 links of a 1.25 mm cord PhysX printed tens of thousands of 'Invalid PhysX
transform detected' and 'Illegal BroadPhaseUpdateData' yet mean_drift_m came back ~0 (the poses were never written
back), at 120 links the drift was 1e7 m and still ok. The pure-Python sections (link plan, verdict) always run; the
Isaac sections run when DEXAGENT_ISAAC_PYTHON points at Isaac Sim's Python (about 30 s each).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tool_library.common import isaac_cord as IC  # noqa: E402
from tool_library.skills.build_isaac_scene import build_isaac_scene, judge_report, link_plan, resolve_cords  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_isaac_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def spiral(L, R, turns=2.5, centre=(-0.1, 0.35)):
    t = np.linspace(0, 1, 400)
    rad = L / (2 * np.pi * turns)
    P = np.stack([centre[0] + rad * np.cos(2 * np.pi * turns * t) * (0.6 + 0.4 * t),
                  centre[1] + rad * np.sin(2 * np.pi * turns * t) * (0.6 + 0.4 * t), np.full_like(t, R)], 1)
    P[:, :2] = centre + (P[:, :2] - centre) * L / np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    return P


# ---------------------------------------------------------------- link length rule (no Isaac)
thin19 = {"name": "cord", "points": spiral(1.9, 0.00125).tolist(), "radius": 0.00125}
plan = link_plan([thin19])
check("a 1.9 m cord gets 80 links of ~24 mm (8 mm would be 237 links, where PhysX blows up)",
      plan["cord"][0] == 80 and 0.022 < plan["cord"][1] < 0.025, str(plan))
short = {"name": "short", "points": spiral(0.4, 0.003).tolist(), "radius": 0.003}
check("a 0.4 m cord keeps 8 mm links", link_plan([short])["short"][0] == 50, str(link_plan([short])))
try:
    link_plan([thin19], segment_m=0.008)
    check("an explicit segment_m needing 237 links is refused", False)
except ValueError as e:
    check("an explicit segment_m needing 237 links is refused", "segment_m >=" in str(e), str(e)[:120])
strands = [{"name": f"s{i}", "points": spiral(0.6, 0.002, centre=(0.1 * i, 0.3)).tolist(), "radius": 0.002} for i in range(3)]
pj = link_plan(strands, junction={"center": [0, 0.3, 0.002], "radius": 0.004})
check("strands tied to one junction share one articulation's budget", sum(k for k, _ in pj.values()) <= 80, str(pj))

# ---------------------------------------------------------------- every cord resolved through isaac_cord (no Isaac)
U = np.stack([0.02 + 0.25 * np.cos(np.linspace(0, np.pi, 200)), 0.45 + 0.30 * np.sin(np.linspace(0, np.pi, 200)), np.full(200, 0.01386)], 1)
rope = {"name": "rope", "points": U.tolist(), "radius": 0.01386, "preset": "segmented_rope"}
specs = resolve_cords([thin19, rope, dict(rope, name="rope41", n_links=41)])
by = {sp["name"]: sp for sp in specs}
check("a thin cord keeps the link length rule and the thin_cord coefficients (the old model)",
      by["cord"]["n_links"] == 80 and by["cord"]["physics"]["bend_stiffness"] == 0.004 and by["cord"]["physics"]["linear_damping"] == 3.0
      and by["cord"]["physics"]["self_collision"] is False, str({k: by["cord"][k] for k in ("n_links", "preset")}))
check("a segmented_rope takes its own ~27 mm links, self-collision and the sticky-table contact",
      by["rope"]["n_links"] == round(by["rope"]["length_m"] / 0.027353) and by["rope"]["physics"]["self_collision"] is True
      and by["rope"]["physics"]["table_friction_combine"] == "max", str(by["rope"]["n_links"]))
check("an explicit n_links wins", by["rope41"]["n_links"] == 41)
try:
    resolve_cords([dict(rope, n_links=120)])
    check("n_links above max_links is refused", False)
except ValueError as e:
    check("n_links above max_links is refused", "max_links" in str(e))
check("self_collision given to the call overrides every cord", all(sp["physics"]["self_collision"] is False for sp in resolve_cords([rope], self_collision=False)))

# ---------------------------------------------------------------- verdict on a finished run (no Isaac)
good = {"cord": {"finite": True, "mean_drift_m": 0.0005, "max_drift_m": 0.001, "link_stretch": 0.001, "segment_m": 0.02}, "_": {}}
check("a clean report and log pass", judge_report(good, "ISAAC_DONE") == [])
log_bad = "\n".join(["[Warning] [omni.physx.plugin] Invalid PhysX transform detected for /World/cords/cord/link_101."] * 50
                    + ["[Error] [omni.physx.plugin] PhysX error: Illegal BroadPhaseUpdateData"])
iss = judge_report(good, log_bad)
check("invalid transforms in the log fail the run even with a tiny drift (the 206-link case)", len(iss) == 1 and "50 invalid" in iss[0], str(iss)[:160])
exploded = {"cord": dict(good["cord"], mean_drift_m=1.43e7, max_drift_m=2e7)}
check("a 1e7 m drift fails (the 120-link case)", any("moved" in s for s in judge_report(exploded, "")))
nan = {"cord": dict(good["cord"], finite=False, mean_drift_m=None)}
check("a non-finite centreline fails", any("not finite" in s for s in judge_report(nan, "")))
torn = {"cord": dict(good["cord"], link_stretch=0.8)}
check("a torn chain fails", any("tore" in s for s in judge_report(torn, "")))

# ---------------------------------------------------------------- Isaac runs
isaac_py = os.environ.get("DEXAGENT_ISAAC_PYTHON")
if isaac_py and Path(isaac_py).exists():
    import imageio.v2 as imageio
    from tool_library.common import rigcam as RC
    from tool_library.skills.register_table_camera import register_table_camera
    world = TMP / "world.xml"
    world.write_text("""<mujoco model="t"><option timestep="0.002"/><visual><global offwidth="1280" offheight="960"/></visual>
<worldbody><light pos="0 0 2" dir="0 0 -1"/>
<geom name="g_table" type="box" size="1 1 0.02" pos="0 0.6 -0.02" rgba="0.4 0.3 0.2 1"/>
<body name="box" pos="0.10 0.45 0.03"><freejoint/><geom type="box" size="0.06 0.04 0.03" rgba="0.8 0.1 0.1 1" mass="0.2"/></body>
</worldbody></mujoco>""")
    cam = register_table_camera(TMP / "cam.json", "estimate", 640, 480, fovy_deg=45.0, pitch_deg=40.0, roll_deg=0.0, height_m=0.6,
                                yaw_deg=90.0, foot_xy=[0.0, -0.2])["camera"]

    def dark_fraction(out, name="cord"):
        """Share of pixels within 3 px of the settled centreline darker than 60% of the table: a cord's black shadow strip."""
        im = imageio.imread(out / "isaac_sim.png")[..., :3].astype(float)
        uv, _ = RC.project(cam, np.load(out / f"isaac_{name}_centreline_settled.npy"))
        band = np.zeros(im.shape[:2], bool)
        for u, v in np.round(uv).astype(int):
            band[max(0, v - 3):v + 4, max(0, u - 3):u + 4] = True
        table = np.median(im[100:150, 100:200].reshape(-1, 3), 0).sum()
        return float(((im.sum(-1) < 0.6 * table) & band).sum() / max(band.sum(), 1))

    # 1. the 1.9 m thin cord with the automatic rule settles, and renders in its colour (no black shadow strip)
    try:
        out = TMP / "long_thin"
        r = build_isaac_scene(world, cam, [dict(thin19, rgb=[0.697, 0.64, 0.5])], out, settle_s=1.0, isaac_python=isaac_py)
        rc = r["report"]["cord"]
        df = dark_fraction(out)
        check("1.9 m x 1.25 mm cord: 80 links, settles with millimetre drift, no PhysX errors, ok",
              r["ok"] and rc["n_links"] == 80 and rc["mean_drift_m"] < 0.005 and r["report"]["_"]["physx_invalid_transforms"] == 0,
              f"links {rc['n_links']} drift {rc['mean_drift_m'] * 1000:.2f} mm")
        check("the thin beige cord is not rendered dark (shadow strip share < 3%)", df < 0.03, f"dark share {df:.3f}")
    except Exception as exc:                                     # noqa: BLE001
        check("1.9 m x 1.25 mm cord settles with the automatic link rule", False, repr(exc)[:300])

    # 2. forcing the old 8 mm links (206 links) reproduces the PhysX failure, which now fails the call
    try:
        build_isaac_scene(world, cam, [{"name": "cord", "points": spiral(1.65, 0.00125).tolist(), "radius": 0.00125}],
                          TMP / "forced206", segment_m=0.008, max_links=300, settle_s=1.0, isaac_python=isaac_py)
        check("206 links of a 1.25 mm cord: the PhysX blow-up raises instead of reporting ok", False, "returned ok")
    except RuntimeError as exc:
        check("206 links of a 1.25 mm cord: the PhysX blow-up raises instead of reporting ok", "invalid transforms" in str(exc), str(exc)[:160])
    # 3. a thick segmented rope (41 links) settles and writes its cord specification with the settled link poses, which
    # isaac_deformable_rest_check accepts
    try:
        import json
        from PIL import Image
        from tool_library.common import scene_assert as SA
        from tool_library.verifiers.isaac_deformable_rest_check import isaac_deformable_rest_check
        cam_b = RC.load(ROOT / "examples" / "default" / "knot_the_rope" / "stage2" / "camera.json")
        ws = TMP / "rope41"
        r = build_isaac_scene(ROOT / "setup" / "scene.xml", cam_b, [dict(rope, n_links=41, cap_rgb=[0.54, 0.51, 0.29], cap_length_a_m=0.075,
                                                                        cap_length_b_m=0.075)], ws / "isaac", settle_s=1.0, isaac_python=isaac_py)
        sp = IC.load_spec(r["cord_json"]["rope"])
        lp = sp["link_poses"]
        check("segmented_rope, 41 links: settles, writes isaac_rope_cord.json with 41 settled centres and orientations",
              r["ok"] and sp["n_links"] == 41 and len(lp["centres"]) == 41 and len(lp["quats"]) == 41 and r["report"]["rope"]["mean_drift_m"] < 0.005,
              f"drift {r['report']['rope']['mean_drift_m']}")
        (ws / "masks").mkdir(exist_ok=True)
        Image.fromarray((SA.cord_tube_mask(cam_b, lp["centres"], sp["radius"], sp["segment_m"], quats=lp["quats"]) * 255).astype(np.uint8)).save(ws / "masks" / "rope.png")
        RC.save(cam_b, ws / "camera.json")
        (ws / "scene_recon.json").write_text(json.dumps({"camera_json": "camera.json", "objects": [
            {"id": "rope", "type": "deformable", "simulator": "isaac", "mask": "masks/rope.png",
             "isaac": {"cord_json": "isaac/isaac_rope_cord.json", "report_json": "isaac/isaac_report.json",
                       "mean_drift_m": r["report"]["rope"]["mean_drift_m"]}}]}))
        rc = isaac_deformable_rest_check(str(ROOT / "setup" / "scene.xml"), str(ws / "scene_recon.json"))
        check("isaac_deformable_rest_check re-measures the settled rope from its files and passes it", rc["ok"], rc["issues"])
    except Exception as exc:                                     # noqa: BLE001
        check("segmented_rope, 41 links settles and writes its cord specification", False, repr(exc)[:300])
else:
    print("[SKIP] Isaac runs: set DEXAGENT_ISAAC_PYTHON")

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
