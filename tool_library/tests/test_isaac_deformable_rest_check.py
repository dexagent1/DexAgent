"""isaac_deformable_rest_check and the Isaac-only deformable entry of scene_recon_manifest_check: a cord laid on the
setup's bench, specified as build_isaac_scene writes it (cord json with settled link poses, settling report), passes;
missing files, PhysX errors, drift, sinking, a torn chain, self-overlap, a MuJoCo copy, a cord inside an object and a
misplaced cord each fail. No Isaac run: the cord files are written from the specification itself.

    python tool_library/tests/test_isaac_deformable_rest_check.py
"""
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from tool_library.common import isaac_cord as IC  # noqa: E402
from tool_library.common import rigcam as RC  # noqa: E402
from tool_library.common import scene_assert as SA  # noqa: E402
from tool_library.verifiers.isaac_deformable_rest_check import isaac_deformable_rest_check  # noqa: E402
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check  # noqa: E402

SETUP = ROOT / "setup"
TMP = Path(tempfile.mkdtemp(prefix="isaac_rest_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  {str(detail)[:500]}"))
    FAILS += 0 if ok else 1


def bench(name, objects="", key_extra=""):
    """setup/scene.xml with objects appended (meshes resolved from setup/)."""
    xml = (SETUP / "scene.xml").read_text()
    xml = xml.replace('<compiler angle="radian" />', f'<compiler angle="radian" meshdir="{SETUP}" texturedir="{SETUP}" />', 1)
    xml = xml.replace("</worldbody>", objects + "\n  </worldbody>", 1)
    if key_extra:
        xml = re.sub(r'(<key name="home" qpos=")([^"]*)', lambda m: m.group(1) + m.group(2) + " " + key_extra, xml, count=1)
    p = TMP / f"{name}.xml"
    p.write_text(xml)
    return str(p)


cfg = yaml.safe_load((SETUP / "setup.yaml").read_text())
e = cfg["cameras"]["ego"]
cam = RC.camera_from_mujoco(640, 480, e["fovy_deg"], e["pos"], e["quat_wxyz"])

R = 0.01386
t = np.linspace(0, np.pi, 200)
U = np.stack([0.02 + 0.25 * np.cos(t), 0.45 + 0.30 * np.sin(t), np.full_like(t, R)], 1)


def workspace(name, centres_edit=None, report_edit=None, points=U, mask_shift=None, scene=None, entry_edit=None):
    ws = TMP / name
    (ws / "isaac").mkdir(parents=True, exist_ok=True)
    (ws / "masks").mkdir(exist_ok=True)
    spec = IC.cord_spec({"name": "rope", "points": points.tolist(), "radius": R, "preset": "segmented_rope", "n_links": 41})
    C, Q = IC.lay_poses(spec["points"], spec["n_links"], R)
    if centres_edit:
        C = centres_edit(C.copy())
    spec["link_poses"] = {"centres": C.tolist(), "quats": Q.tolist()}
    IC.save_spec(spec, ws / "isaac" / "isaac_rope_cord.json")
    rep = {"rope": {"n_links": 41, "segment_m": spec["segment_m"], "radius_m": R, "finite": True, "mean_drift_m": 0.0005},
           "_": {"physx_invalid_transforms": 0, "physx_errors": 0}, "ok": True, "issues": []}
    if report_edit:
        report_edit(rep)
    (ws / "isaac" / "isaac_report.json").write_text(json.dumps(rep))
    for f in ("isaac_sim.png", "isaac_scene.usda"):
        (ws / "isaac" / f).write_text("x")
    Cm, Qm = IC.lay_poses(U, 41, R)
    if mask_shift is not None:
        Cm = Cm + np.asarray(mask_shift)
    m = SA.cord_tube_mask(cam, Cm, R, spec["segment_m"], quats=Qm)
    Image.fromarray((m * 255).astype(np.uint8)).save(ws / "masks" / "rope.png")
    RC.save(cam, ws / "camera.json")
    (ws / "compare.png").write_text("x")
    entry = {"id": "rope", "type": "deformable", "simulator": "isaac", "skill": "build_isaac_scene", "skill_origin": "library",
             "simulation_model": {"kind": "isaac_cord", "approximation": "segmented capsule chain"},
             "pose": {"x": 0.02, "y": 0.6, "yaw_deg": 0.0}, "mask": "masks/rope.png", "iou": 0.9,
             "verification_plan": {"isaac_deformable_rest_check": "rest and placement"},
             "verifiers": {"isaac_deformable_rest_check": {"ok": True}},
             "properties_covered": {"A thick flexible rope.": "segmented_rope preset; isaac_deformable_rest_check"},
             "isaac": {"cord_json": "isaac/isaac_rope_cord.json", "report_json": "isaac/isaac_report.json", "sim_png": "isaac/isaac_sim.png",
                       "scene_usda": "isaac/isaac_scene.usda", "mean_drift_m": 0.0005}}
    if entry_edit:
        entry_edit(entry)
    rec = {"scene_semantic": "sem.json", "scene_xml": scene or bench("bench"), "camera_json": "camera.json", "frame0_compare_png": "compare.png",
           "simulator": "mujoco+isaac", "objects": [entry]}
    (ws / "scene_recon.json").write_text(json.dumps(rec))
    return ws, rec


def run(ws, rec):
    return isaac_deformable_rest_check(rec["scene_xml"], str(ws / "scene_recon.json"))


ws, rec = workspace("clean")
r = run(ws, rec)
row = r["objects"].get("rope", {})
check("a cord laid on the bench, with its specification and a clean settling report, passes", r["ok"] and row.get("iou", 0) > 0.9, r)

ws, rec = workspace("no_cord", entry_edit=lambda e: e["isaac"].pop("cord_json"))
check("an entry without isaac.cord_json fails", not run(ws, rec)["ok"])

ws, rec = workspace("physx", report_edit=lambda rp: rp["_"].update(physx_errors=3))
r = run(ws, rec)
check("PhysX errors while settling fail", (not r["ok"]) and any("PhysX" in i for i in r["issues"]), r["issues"])

ws, rec = workspace("floating", centres_edit=lambda C: C + [0, 0, 0.04])
r = run(ws, rec)
check("a cord 4 cm above where it was laid (drift) fails, and the report's drift claim is caught", (not r["ok"]) and
      any("drift" in i for i in r["issues"]) and any("claims" in i for i in r["issues"]), r["issues"])

ws, rec = workspace("sunk", centres_edit=lambda C: C - [0, 0, 0.012])
r = run(ws, rec)
check("a cord sunk 12 mm into the table fails", (not r["ok"]) and any("sinks" in i for i in r["issues"]), r["issues"])


def tear(C):
    C[20] += [0.0, 0.02, 0.0]
    return C


ws, rec = workspace("torn", centres_edit=tear)
r = run(ws, rec)
check("a link moved 2 cm off the chain (broken joint) fails", (not r["ok"]) and any("broken" in i for i in r["issues"]), r["issues"])

s = np.linspace(0, 2 * np.pi, 300)
EIGHT = np.stack([0.02 + 0.12 * np.sin(s), 0.55 + 0.10 * np.sin(2 * s), np.full_like(s, R)], 1)
ws, rec = workspace("eight", points=EIGHT)
r = run(ws, rec)
check("a cord that crosses itself flat on the table (links inside each other) fails", (not r["ok"]) and any("overlap" in i for i in r["issues"]), r["issues"])

copy_xml = bench("with_copy", '<body name="ropeB_first" pos="0 0.3 0.02"><freejoint/><geom type="capsule" size="0.01 0.02" mass="0.01"/></body>',
                 "0 0.3 0.02 1 0 0 0")
ws, rec = workspace("mujoco_copy", scene=copy_xml)
r = run(ws, rec)
check("scene.xml with a MuJoCo cable of the same cord fails", (not r["ok"]) and any("MuJoCo bodies" in i for i in r["issues"]), r["issues"])

box_xml = bench("with_box", '<body name="block" pos="0.27 0.45 0.03"><freejoint/><geom name="block_g" type="box" size="0.04 0.04 0.03" mass="0.2"/></body>',
                "0.27 0.45 0.03 1 0 0 0")
ws, rec = workspace("inside_box", scene=box_xml)
r = run(ws, rec)
check("a cord laid through a rigid object fails", (not r["ok"]) and any("inside" in i for i in r["issues"]), r["issues"])

ws, rec = workspace("misplaced", mask_shift=[0.0, 0.15, 0.0])
r = run(ws, rec)
check("a cord 15 cm from where its mask shows it fails the placement score", (not r["ok"]) and any("silhouette" in i for i in r["issues"]), r["issues"])

# ---------------------------------------------------------------- scene_recon_manifest_check on the Isaac-only entry
sem = {"objects": [{"id": "rope", "type": "deformable", "role": "manipulated", "properties": ["A thick flexible rope."]}],
       "subgoals": [{"index": 1, "acting_hand": "both", "motion": "tie", "object_refs": ["rope"], "time_window_s": [0, 1]}]}


def manifest(ws, rec, semantic=sem):
    (ws / "sem.json").write_text(json.dumps(semantic))
    return scene_recon_manifest_check(str(ws / "scene_recon.json"))


ws, rec = workspace("m_clean")
r = manifest(ws, rec)
check("manifest: an Isaac-only cord with its files and no MuJoCo body passes the bookkeeping", r["ok"] and
      r["per_object"]["rope"].get("simulation_scope") == "isaac_cord", r["issues"])
ws, rec = workspace("m_nocord", entry_edit=lambda e: e["isaac"].pop("cord_json"))
r = manifest(ws, rec)
check("manifest: without isaac.cord_json it fails", (not r["ok"]) and any("cord_json" in i for i in r["issues"]), r["issues"])
ws, rec = workspace("m_sim", entry_edit=lambda e: None)
rec2 = copy.deepcopy(rec); rec2["simulator"] = "mujoco"
(ws / "scene_recon.json").write_text(json.dumps(rec2))
r = manifest(ws, rec2)
check("manifest: an Isaac cord in a scene that claims simulator 'mujoco' fails", (not r["ok"]) and any("mujoco+isaac" in i for i in r["issues"]), r["issues"])
ws, rec = workspace("m_copy", scene=copy_xml)
r = manifest(ws, rec)
check("manifest: a MuJoCo copy of the cord in scene.xml fails", (not r["ok"]) and any("MuJoCo bodies" in i for i in r["issues"]), r["issues"])
ws, rec = workspace("m_cut")
cut = copy.deepcopy(sem); cut["subgoals"][0]["motion"] = "cut"
r = manifest(ws, rec, cut)
check("manifest: a cord a subgoal cuts cannot be an Isaac cord (no cutting in Isaac Sim)", (not r["ok"]) and
      any("construct_cuttable_rope" in i for i in r["issues"]), r["issues"])
ws, rec = workspace("m_report", report_edit=lambda rp: rp.update(ok=False, issues=["exploded"]))
r = manifest(ws, rec)
check("manifest: a failed settling report fails", (not r["ok"]) and any("settling report" in i for i in r["issues"]), r["issues"])

print(f"\n{'ALL PASS' if not FAILS else f'{FAILS} FAILED'}  ({TMP})")
sys.exit(1 if FAILS else 0)
