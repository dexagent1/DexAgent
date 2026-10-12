"""Objects hidden at source frame 0 (tool_library/common/hidden_objects.py) on a synthetic scene: a watch inside a closed box.

  provenance (agent/reference_evidence): a declared hidden object's receipted, EMPTY frame-0 mask passes and its reveal
      mask must be a receipted mask of the declared reveal frame; an undeclared empty mask, an unreceipted mask and a
      reveal mask of another frame still fail; the hands rule is unchanged.
  scene_qc / silhouette_iou_check: the hidden watch renders invisible with the lid shut and passes; with the lid open it
      is visible and fails; placed out of view it fails; a visible object's frame-0 mask cannot be declared hidden.
  scene_recon_manifest_check: a complete declaration needs no IoU; a static object or a declaration without its reason
      and reveal frame fails; an undeclared object keeps the IoU floor.

    python tool_library/tests/test_hidden_objects.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from agent import reference_evidence as RE  # noqa: E402
from tool_library.common import scene as S  # noqa: E402
from tool_library.verifiers.scene_qc import run_scene_qc  # noqa: E402
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check  # noqa: E402
from tool_library.verifiers.silhouette_iou_check import check_silhouette_iou  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


def scene_xml(lid_open=False, watch_xy=(0.0, 0.0)):
    """A 20 x 12 x 5 cm box (floor, four walls, a lid hinged on its far edge) with a watch on its floor."""
    wall = 'type="box" rgba=".6 .42 .2 1"'
    lid_euler = "-120 0 0" if lid_open else "0 0 0"
    return f'''<mujoco><option timestep="0.002"/><worldbody><light pos="0 0 2" dir="0 0 -1"/>
      <camera name="video_cam" pos="0 -.35 .45" euler="38 0 0" fovy="50"/>
      <body name="table"><geom name="table_top" type="box" size=".6 .6 .02" pos="0 0 -.02" rgba=".3 .3 .3 1"/></body>
      <body name="box_box" pos="0 0 0"><freejoint/>
        <geom {wall} size=".1 .06 .0015" pos="0 0 .0015" mass=".02"/>
        <geom {wall} size=".1 .0015 .025" pos="0 .0585 .025" mass=".01"/><geom {wall} size=".1 .0015 .025" pos="0 -.0585 .025" mass=".01"/>
        <geom {wall} size=".0015 .06 .025" pos=".0985 0 .025" mass=".01"/><geom {wall} size=".0015 .06 .025" pos="-.0985 0 .025" mass=".01"/>
        <body name="box_box_lidb" pos="0 .06 .05" euler="{lid_euler}"><joint name="box_lid" type="hinge" axis="1 0 0" range="-150 0" damping="1"/>
          <geom {wall} size=".1 .06 .0015" pos="0 -.06 .0015" mass=".02"/></body>
      </body>
      <body name="watch_watch" pos="{watch_xy[0]} {watch_xy[1]} .0085"><freejoint/>
        <geom type="box" size=".022 .018 .0055" rgba=".1 .1 .1 1" mass=".04"/></body>
    </worldbody></mujoco>'''


tmp = Path(tempfile.mkdtemp(prefix="hidden_objects_", dir=os.environ.get("TMPDIR")))
ws = tmp / "stage2"
(ws / "masks").mkdir(parents=True)
(ws / "frames").mkdir()
(ws / "assets").mkdir()
(tmp / "stage1").mkdir()
W, H = 320, 240
for name, kw in (("scene.xml", {}), ("open.xml", {"lid_open": True}), ("away.xml", {"watch_xy": (0.0, 3.0)})):
    (ws / name).write_text(scene_xml(**kw))


def render(xml, body_names):
    m = mujoco.MjModel.from_xml_path(str(ws / xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bm = S.render_body_masks(m, d, W, H, "video_cam")
    out = np.zeros((H, W), bool)
    for n in body_names:
        out |= bm.get(n, np.zeros((H, W), bool))
    return out


def save_mask(path, m):
    Image.fromarray((m * 255).astype(np.uint8)).save(path)


box_mask = render("scene.xml", ["box_box", "box_box_lidb"])
watch_visible = render("open.xml", ["watch_watch"])
check("setup: the closed box hides the watch, the open box shows it", render("scene.xml", ["watch_watch"]).sum() == 0
      and watch_visible.sum() > 200, f"{render('scene.xml', ['watch_watch']).sum()} {watch_visible.sum()}")

# ------------------------------------------------------------------------------- receipts (the harness's approved path)
video = tmp / "source.mp4"
video.write_bytes(b"synthetic source video bytes")
frame0, frame7 = ws / "frames" / "frame00000.jpg", ws / "frames" / "frame00007.jpg"
Image.fromarray(np.full((H, W, 3), 90, np.uint8)).save(frame0)
Image.fromarray(np.full((H, W, 3), 120, np.uint8)).save(frame7)
save_mask(ws / "masks" / "box.png", box_mask)
save_mask(ws / "masks" / "watch.png", np.zeros((H, W), bool))                  # the approved frame-0 segmentation: empty
save_mask(ws / "masks" / "watch_reveal.png", watch_visible)                    # the same segmentation of frame 7
authority = RE.begin(ws, video)


def receipt(mask, frame, index):
    """What the harness records around an approved segment_source_frame call (simulated tool output)."""
    args = {"video": str(video), "frame_index": index, "box": [0, 0, W - 1, H - 1], "out_dir": str(ws / "seg")}
    token = authority.prepare("segment_source_frame", args)
    authority.complete(token, {"ok": True, "result": {"source_frame": str(frame), "mask": str(mask)}})


receipt(ws / "masks" / "box.png", frame0, 0)
receipt(ws / "masks" / "watch.png", frame0, 0)
receipt(ws / "masks" / "watch_reveal.png", frame7, 7)

PROPS = {"box": ["A closed cardboard box with a lid hinged on its far edge."], "watch": ["A smartwatch lying on the box floor."]}
sem = {"task_prompt": "take the watch out of the box", "task_summary": "The right hand opens the box and takes the watch out.",
       "environment": {"support_surface": "a grey table"}, "hands": {"used": ["right"]},
       "objects": [{"id": "box", "name": "cardboard box", "type": "rigid", "role": "container", "properties": PROPS["box"], "evidence": "0 s"},
                   {"id": "watch", "name": "black smartwatch", "type": "rigid", "role": "manipulated", "properties": PROPS["watch"], "evidence": "2 s"}],
       "subgoals": [{"index": 1, "acting_hand": "right", "action": "Lift the watch out.", "object_refs": ["watch"], "motion": "grasp_lift",
                     "hand_roles": {"left": "Rests.", "right": "Lifts the watch."}, "goal": "The watch is held.", "time_window_s": [2.0, 3.0]}],
       "task_success": "The watch is held above the table."}
(tmp / "stage1" / "scene_semantics.json").write_text(json.dumps(sem))
(tmp / "stage1" / "static.json").write_text(json.dumps({**sem, "objects": [sem["objects"][0], {**sem["objects"][1], "role": "static"}]}))
(ws / "camera.json").write_text(json.dumps({"fx": 300, "fy": 300, "cx": 160, "cy": 120, "pos": [0, -0.35, 0.45], "quat_wxyz": [1, 0, 0, 0],
                                            "width": W, "height": H}))
Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(ws / "frame0_compare.png")
index = tmp / "index.json"
index.write_text(json.dumps({"tools": [{"name": "construct_box", "category": "Object Reconstruction", "type": "skill"}]}))
HIDDEN = {"reason": "inside the closed box, under its lid", "reveal_frame": 7, "reveal_mask": "masks/watch_reveal.png"}


def entry(oid, root, hidden=None, iou=0.9):
    (ws / "assets" / f"{oid}.xml").write_text("<mujoco/>")
    e = {"id": oid, "type": "rigid", "skill": "construct_box", "skill_origin": "library", "asset_xml": f"assets/{oid}.xml",
         "root_body": root, "pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}, "mask": f"masks/{oid}.png",
         "verification_plan": {"settle_stability_check": "rests"},
         "verifiers": {"settle_stability_check": {"ok": True}, "box_enclosure_check": {"ok": True}},
         "properties_covered": {p: "settle_stability_check" for p in PROPS[oid]}}
    if iou is not None:
        e["iou"] = iou
    if hidden is not None:
        e["hidden_at_frame0"] = hidden
    return e


def write_recon(objects, sem_name="scene_semantics.json", scene="scene.xml", name="scene_recon.json"):
    p = ws / name
    p.write_text(json.dumps({"scene_semantic": f"../stage1/{sem_name}", "scene_xml": scene, "camera_json": "camera.json",
                             "frame0": "frames/frame00000.jpg", "frame0_compare_png": "frame0_compare.png", "simulator": "mujoco",
                             "objects": objects}, indent=1))
    return p


def provenance(objects):
    write_recon(objects)
    return RE.check_workspace(ws, video)


r = provenance([entry("box", "box_box"), entry("watch", "watch_watch", HIDDEN, iou=None)])
check("provenance: a declared hidden object's receipted empty frame-0 mask and its reveal-frame mask pass",
      r["ok"] and any(c.get("hidden_at_frame0") for c in r["checked"]) and any(c["role"] == "reveal" and c["source_frame_index"] == 7 for c in r["checked"]),
      r["issues"])
r = provenance([entry("box", "box_box"), entry("watch", "watch_watch")])
check("provenance: the same empty mask without the declaration still fails (nonempty required)",
      not r["ok"] and any("nonempty" in i for i in r["issues"]), r["issues"])
save_mask(ws / "masks" / "watch_painted.png", np.zeros((H, W), bool) | (np.arange(W)[None] < 3))
e = entry("watch", "watch_watch", HIDDEN, iou=None)
e["mask"] = "masks/watch_painted.png"
r = provenance([entry("box", "box_box"), e])
check("provenance: an unreceipted mask of a hidden object still fails", not r["ok"] and any("no trusted source receipt" in i for i in r["issues"]), r["issues"])
r = provenance([entry("box", "box_box"), entry("watch", "watch_watch", {**HIDDEN, "reveal_frame": 9}, iou=None)])
check("provenance: a reveal mask of another frame than the declared reveal_frame fails",
      not r["ok"] and any("reveal_mask" in i and "frame 9" in i for i in r["issues"]), r["issues"])
e = entry("watch", "watch_watch", HIDDEN, iou=None)
e["mask"] = "masks/watch_reveal.png"
r = provenance([entry("box", "box_box"), e])
check("provenance: the reveal frame's mask cannot stand in for the frame-0 mask", not r["ok"]
      and any("different source frame" in i for i in r["issues"]), r["issues"])

# ------------------------------------------------------------------------------------------ silhouettes and scene_qc
masks = {"box_box": str(ws / "masks" / "box.png"), "watch_watch": str(ws / "masks" / "watch.png")}
r = run_scene_qc(str(ws / "scene.xml"), required=["box_box", "watch_watch"], masks=masks, camera="video_cam", hidden=["watch_watch"])
h = r["checks"]["silhouette_iou"]["hidden"]["watch_watch"]
check("scene_qc: the hidden watch renders invisible inside the shut box and passes (the box keeps its IoU)",
      r["ok"] and h["ok"] and h["visible_px"] == 0 and h["unoccluded_px"] > 200 and r["checks"]["silhouette_iou"]["iou"]["box_box"] > 0.97,
      f"{r['failed']} {h}")
r = check_silhouette_iou(str(ws / "open.xml"), masks, "video_cam", hidden=["watch_watch"])
check("silhouette: with the lid open the 'hidden' watch is visible and fails, with the reason",
      not r["ok"] and "watch_watch" in r["failures"] and "visible in the render" in r["hidden"]["watch_watch"].get("reason", ""), r["hidden"])
r = check_silhouette_iou(str(ws / "away.xml"), masks, "video_cam", hidden=["watch_watch"])
check("silhouette: a 'hidden' watch placed out of the view fails", not r["ok"] and "out of the camera's view" in r["hidden"]["watch_watch"].get("reason", ""),
      r["hidden"])
r = check_silhouette_iou(str(ws / "scene.xml"), {"watch_watch": str(ws / "masks" / "watch_reveal.png")}, "video_cam", hidden=["watch_watch"])
check("silhouette: an object whose frame-0 mask shows it cannot be declared hidden",
      not r["ok"] and "frame-0 mask shows" in r["hidden"]["watch_watch"].get("reason", ""), r["hidden"])
r = check_silhouette_iou(str(ws / "scene.xml"), masks, "video_cam")
check("silhouette: without the declaration the empty-mask watch fails as before", not r["ok"] and r["iou"].get("watch_watch") == 0.0, r["iou"])

# ----------------------------------------------------------------------------------------- scene_recon_manifest_check
r = scene_recon_manifest_check(write_recon([entry("box", "box_box"), entry("watch", "watch_watch", HIDDEN, iou=None)]), library_index=index)
check("manifest: a complete hidden declaration needs no IoU", r["ok"] and r["per_object"]["watch"].get("hidden_at_frame0"), r["issues"])
r = scene_recon_manifest_check(write_recon([entry("box", "box_box"), entry("watch", "watch_watch", {"reveal_mask": "masks/watch_reveal.png"}, iou=None)],
                                           name="r_bad.json"), library_index=index)
check("manifest: a declaration without its reason and reveal frame fails",
      not r["ok"] and any("reason" in i for i in r["issues"]) and any("reveal_frame" in i for i in r["issues"]), r["issues"])
r = scene_recon_manifest_check(write_recon([entry("box", "box_box"), entry("watch", "watch_watch", HIDDEN, iou=None)], "static.json",
                                           name="r_static.json"), library_index=index)
check("manifest: an object Stage 1 calls static cannot be declared hidden", not r["ok"] and any("static" in i for i in r["issues"]), r["issues"])
r = scene_recon_manifest_check(write_recon([entry("box", "box_box"), entry("watch", "watch_watch", iou=0.1)], name="r_plain.json"),
                               library_index=index)
check("manifest: an undeclared object keeps the IoU floor", not r["ok"] and any("below" in i for i in r["issues"]), r["issues"])

print(f"{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}: hidden objects ({tmp})")
sys.exit(1 if fails else 0)
