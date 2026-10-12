"""stage2_gate_suite: a clean Stage 2 workspace (the setup + one box) passes; widened robot ranges, a missing keyframe,
stronger fingers and a skipped check are each handled.

    python tool_library/tests/test_stage2_gate_suite.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from agent.reference_evidence import begin  # noqa: E402
from tool_library.common.video import write_video  # noqa: E402
from tool_library.skills.segment_source_frame import segment_source_frame  # noqa: E402
from tool_library.verifiers.stage2_gate_suite import stage2_gate_suite  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="s2suite_", dir=os.environ.get("TMPDIR")))
FAILS = []
# the source video: a bright square (the box) on a dark table, three frames
VIDEO = TMP / "video.mp4"
_frame = np.full((64, 64, 3), 40, dtype=np.uint8)
_frame[20:44, 20:44] = 230
write_video([_frame] * 3, VIDEO, fps=30)


def ok(name, cond, extra=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (f"  ({extra})" if extra else ""))
    if not cond:
        FAILS.append(name)


def workspace(name, tamper=None):
    ws = TMP / name
    shutil.copytree(ROOT / "setup", ws / "setup_copy")            # the setup's meshes resolve from the copy
    root = ET.parse(ws / "setup_copy" / "scene.xml").getroot()
    wb = root.find("worldbody")
    body = ET.SubElement(wb, "body", {"name": "box_box", "pos": "0.0 0.45 0.03"})
    ET.SubElement(body, "freejoint", {"name": "box_box_free"})
    ET.SubElement(body, "geom", {"name": "box_box_geom", "type": "box", "size": "0.03 0.03 0.03", "mass": "0.2", "rgba": "0.8 0.3 0.2 1"})
    for k in root.iter("key"):
        k.set("qpos", k.get("qpos", "").strip() + " 0.0 0.45 0.03 1 0 0 0")
    if tamper:
        tamper(root)
    (ws / "setup_copy" / "scene.xml").write_text(ET.tostring(root, encoding="unicode"))
    source_reference(ws)
    (ws / "scene_recon.json").write_text(json.dumps({"scene_xml": "setup_copy/scene.xml", "frame0": "ref/source.png", "objects": [
        {"id": "box", "type": "rigid", "root_body": "box_box", "skill": "construct_cubes", "pose": {"x": 0.0, "y": 0.45, "z": 0.0},
         "mask": "ref/mask.png"}]}))
    return ws


def source_reference(ws, extra=()):
    """The box's frame-0 mask and source frame, with trusted receipts minted the way the Stage 2 harness does it:
    reference_evidence.begin on the pinned video, then prepare / complete around an approved tool call
    (segment_source_frame). Saved masks or written provenance alone cannot certify Stage 2. `extra`: more
    out_dir names segmented the same way (another object's mask)."""
    evidence = begin(ws, VIDEO)
    for out in ("ref", *extra):
        args = {"video": str(VIDEO), "frame_index": 0, "box": [14, 14, 50, 50], "out_dir": str(ws / out)}
        token = evidence.prepare("segment_source_frame", args)
        evidence.complete(token, {"ok": True, "result": segment_source_frame(**args)})


sem = TMP / "scene_semantics.json"
sem.write_text(json.dumps({"objects": [{"id": "box", "name": "orange box", "type": "rigid", "role": "manipulated", "properties": ["A rigid box."]}],
                           "subgoals": [{"index": 1, "acting_hand": "right", "motion": "grasp_lift", "object_refs": ["box"], "time_window_s": [0, 1]}]}))

r = stage2_gate_suite(str(workspace("clean")), str(sem))
ok("the setup plus a box passes", r["ok"], "; ".join(r["issues"][:3]))
ok("every check reports", len(r["reports"]) >= 15, f"{len(r['reports'])} checks")
ok("the acting arm reach is a required check", r['reports']['object_reach_check']['objects']['box_box']['required_sides'] == ['right'])

def unreachable(root):
    root.find(".//body[@name='box_box']").set('pos', '0 1.2 .03')
    for key in root.iter('key'):
        q = key.get('qpos').split(); q[-6] = '1.2'; key.set('qpos', ' '.join(q))

r = stage2_gate_suite(str(workspace('unreachable', unreachable)), str(sem))
ok('unreachable Stage 1 object blocks Stage 2', not r['ok'] and not r['reports']['object_reach_check']['ok'])


def widen(root):
    for j in root.iter("joint"):
        if re.search(r"robot_.*joint[1-6]$", j.get("name") or "") and j.get("range"):
            j.set("range", "-6.28 6.28")


r = stage2_gate_suite(str(workspace("widened", widen)), str(sem))
ok("widened arm ranges fail", not r["ok"] and any(i.startswith("robot_joint_range_check") for i in r["issues"]))


def nokey(root):
    for kf in list(root.iter("keyframe")):
        root.remove(kf)


r = stage2_gate_suite(str(workspace("nokey", nokey)), str(sem))
ok("a scene without its keyframe fails", not r["ok"] and any(i.startswith("initial_keyframe_state_check") for i in r["issues"]))


def stronger(root):
    for a in root.iter("general"):
        if (a.get("name") or "").startswith("robot_") and "index" in (a.get("name") or ""):
            g = [float(x) for x in a.get("gainprm", "1").split()]
            b = [float(x) for x in a.get("biasprm", "0 0 0").split()]
            a.set("gainprm", " ".join(str(x * 10) for x in g)); a.set("biasprm", " ".join(str(x * 10) for x in b))


r = stage2_gate_suite(str(workspace("stronger", stronger)), str(sem))
ok("stronger fingers fail", not r["ok"] and any(i.startswith("actuator_limit_integrity_check") for i in r["issues"]))

r = stage2_gate_suite(str(workspace("skip", widen)), str(sem), skip=["robot_joint_range_check", "robot_model_integrity_check"])
ok("a skipped check is reported as skipped", "robot_joint_range_check" in r["skipped"])

# a deformable simulated in Isaac Sim: re-measured from its cord files by isaac_deformable_rest_check, never by the MuJoCo
# cable checks (it has no MuJoCo body); a MuJoCo copy of it in scene.xml fails
from tool_library.common import isaac_cord as IC  # noqa: E402

sem_rope = TMP / "scene_semantics_rope.json"
_s = json.loads(sem.read_text())
_s["objects"].append({"id": "rope", "name": "red rope", "type": "deformable", "role": "manipulated", "properties": ["A thick flexible rope."]})
sem_rope.write_text(json.dumps(_s))


def isaac_rope(name, tamper=None):
    ws = workspace(name, tamper)
    source_reference(ws, extra=("ref_rope",))           # the rope's own receipted frame-0 mask
    rec = json.loads((ws / "scene_recon.json").read_text())
    R, t = 0.01386, np.linspace(0, np.pi, 100)
    U = np.stack([0.25 * np.cos(t), 0.5 + 0.2 * np.sin(t), np.full_like(t, R)], 1)
    spec = IC.cord_spec({"name": "rope", "points": U.tolist(), "radius": R, "preset": "segmented_rope", "n_links": 30})
    C, Q = IC.lay_poses(spec["points"], 30, R)
    spec["link_poses"] = {"centres": C.tolist(), "quats": Q.tolist()}
    (ws / "isaac").mkdir(exist_ok=True)
    IC.save_spec(spec, ws / "isaac" / "isaac_rope_cord.json")
    (ws / "isaac" / "isaac_report.json").write_text(json.dumps({"rope": {"n_links": 30, "radius_m": R, "mean_drift_m": 0.0005},
                                                                 "_": {"physx_invalid_transforms": 0, "physx_errors": 0}, "ok": True, "issues": []}))
    rec["simulator"] = "mujoco+isaac"
    rec["objects"].append({"id": "rope", "type": "deformable", "simulator": "isaac", "skill": "build_isaac_scene", "mask": "ref_rope/mask.png",
                           "isaac": {"cord_json": "isaac/isaac_rope_cord.json", "report_json": "isaac/isaac_report.json", "mean_drift_m": 0.0005}})
    (ws / "scene_recon.json").write_text(json.dumps(rec))
    return ws


r = stage2_gate_suite(str(isaac_rope("isaac_rope")), str(sem_rope))
iso = r["reports"].get("isaac_deformable_rest_check") or {}
failing = sorted(k for k, v in r["reports"].items() if isinstance(v, dict) and v.get("ok") is not True)
ok("an Isaac Sim cord is re-measured by isaac_deformable_rest_check, not by the MuJoCo cable checks",
   bool(iso) and "deformable_rest_check" not in r["reports"] and "deformable_topology_check" not in r["reports"], sorted(r["reports"]))
ok("... whose only finding here is the placement it cannot check without camera.json",
   failing == ["isaac_deformable_rest_check"] and all("placement cannot be checked" in i for i in iso.get("issues", [])), f"{failing} {iso.get('issues')}")


def rope_copy(root):
    body = ET.SubElement(root.find("worldbody"), "body", {"name": "ropeB_first", "pos": "0.3 0.3 0.02"})
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "geom", {"type": "capsule", "size": "0.01 0.02", "mass": "0.01"})
    for k in root.iter("key"):
        k.set("qpos", k.get("qpos", "").strip() + " 0.3 0.3 0.02 1 0 0 0")


r = stage2_gate_suite(str(isaac_rope("isaac_rope_copy", rope_copy)), str(sem_rope))
ok("a MuJoCo copy of an Isaac Sim cord in scene.xml fails", any("MuJoCo bodies" in i for i in (r["reports"].get("isaac_deformable_rest_check") or {}).get("issues", [])),
   (r["reports"].get("isaac_deformable_rest_check") or {}).get("issues"))

shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if FAILS else 0)
