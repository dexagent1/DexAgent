"""Group objects (several separate, loose pieces under one Stage 1 object; tool_library/common/piece_groups.py) end to end
on a synthetic scene: a 3-bar stack of loose bars and a 3-card fan, each piece its own free body.

  Stage 1: `pieces` validates in the schema and the consistency check; a one-piece group is refused.
  Stage 2: declared `parts` pass free_joint_object_check, scene_recon_manifest_check, silhouette_iou_check / scene_qc (union
           silhouette), object_count_vs_stage1_check (piece counts) and hand_object_contact_consistency_check (nearest
           piece); an undeclared loose body, a fixed piece Stage 1 calls loose, a wrong count, a root outside its parts, a
           piece of two objects and dissimilar 'pieces' without Stage 1 `pieces` still fail; a single-body scene is unchanged.
  Stage 3: goal resolution takes the piece the subgoal acted on, the whole group otherwise.

    python tool_library/tests/test_group_objects.py
"""
import copy
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

from tool_library.common import scene as S  # noqa: E402
from tool_library.verifiers.free_joint_object_check import free_joint_object_check  # noqa: E402
from tool_library.verifiers.hand_object_contact_consistency_check import hand_object_contact_consistency_check  # noqa: E402
from tool_library.verifiers.object_count_vs_stage1_check import object_count_vs_stage1_check  # noqa: E402
from tool_library.verifiers.scene_qc import run_scene_qc  # noqa: E402
from tool_library.verifiers.scene_recon_manifest_check import scene_recon_manifest_check  # noqa: E402
from tool_library.verifiers.scene_semantic_consistency_check import scene_semantic_consistency_check  # noqa: E402
from tool_library.verifiers.scene_semantic_schema_check import scene_semantic_schema_check  # noqa: E402
from tool_library.verifiers.silhouette_iou_check import check_silhouette_iou  # noqa: E402

fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {msg}"))
    fails += 0 if cond else 1


BAR = '<geom type="box" size=".0375 .0125 .0075" mass=".017" rgba=".8 .6 .4 1"/>'
CARD = '<geom type="box" size=".0445 .0318 .0005" mass=".0018" rgba=".9 .9 .9 1"/>'


def scene_xml(top_free=True, cap=False):
    """A stack of 3 loose bars (two side by side, one across them) and a fan of 3 cards on a table."""
    fj = "<freejoint/>"
    parts = [
        f'<body name="stack_c0s0" pos="0 -.0126 .0075">{fj}{BAR}</body>',
        f'<body name="stack_c0s1" pos="0 .0126 .0075">{fj}{BAR}</body>',
        f'<body name="stack_c1s0" pos="0 0 .0225" euler="0 0 90">{fj if top_free else ""}{BAR}</body>',
        f'<body name="fan_card0" pos=".16 0 .0005">{fj}{CARD}</body>',
        f'<body name="fan_card1" pos=".24 .01 .0005" euler="0 0 12">{fj}{CARD}</body>',
        f'<body name="fan_card2" pos=".32 .03 .0005" euler="0 0 24">{fj}{CARD}</body>',
    ]
    if cap:   # a bottle and its loose cap: not a group of pieces
        parts += ['<body name="bottle_body" pos="-.15 0 .1"><freejoint/><geom type="cylinder" size=".035 .1" mass=".3"/></body>',
                  '<body name="bottle_cap" pos="-.15 0 .21"><freejoint/><geom type="cylinder" size=".016 .01" mass=".005"/></body>']
    return ('<mujoco><option timestep="0.002"/><worldbody><light pos="0 0 2" dir="0 0 -1"/>'
            '<camera name="video_cam" pos=".15 -.3 .6" euler=".46 0 0" fovy="50"/>'
            '<body name="table"><geom name="table_top" type="box" size=".6 .6 .02" pos="0 0 -.02" rgba=".3 .3 .3 1"/></body>'
            + "".join(parts) + '</worldbody></mujoco>')


STACK = ["stack_c0s0", "stack_c0s1", "stack_c1s0"]
FAN = ["fan_card0", "fan_card1", "fan_card2"]
PROPS = {"stack": ["Three separate wooden bars stacked in two courses, held only by gravity."],
         "fan": ["Three separate playing cards fanned out flat on the table."]}

tmp = Path(tempfile.mkdtemp(prefix="group_objects_"))
(tmp / "stage1").mkdir()
ws = tmp / "stage2"
(ws / "masks").mkdir(parents=True)
(ws / "assets").mkdir()


def stage1(stack_pieces=None, fan_pieces=None, success="Two cards lie apart from the stack."):
    objs = [{"id": "stack", "name": "wooden bar stack", "type": "rigid", "role": "target", "properties": PROPS["stack"],
             "initial_state": "standing on the table", "evidence": "visible from 0.0 s to 1.0 s"},
            {"id": "fan", "name": "fan of playing cards", "type": "rigid", "role": "manipulated", "properties": PROPS["fan"],
             "initial_state": "lying on the table", "evidence": "visible from 0.0 s to 1.0 s"}]
    if stack_pieces:
        objs[0]["pieces"] = stack_pieces
    if fan_pieces:
        objs[1]["pieces"] = fan_pieces
    return {"task_prompt": "Take a card from the fan next to the bar stack", "task_summary": "The right hand takes a card from the fan "
            "next to the bar stack.", "environment": {"support_surface": "a grey table"}, "hands": {"used": ["right"]},
            "objects": objs,
            "subgoals": [{"index": 1, "acting_hand": "right", "action": "Touch the bar stack.", "object_refs": ["stack"],
                          "motion": "hold", "hand_roles": {"left": "Rests on the table.", "right": "Touches the stack."},
                          "goal": "The fingertips are on the stack.", "time_window_s": [0.3, 0.6]},
                         {"index": 2, "acting_hand": "right", "action": "Pick up one card from the fan.", "object_refs": ["fan"],
                          "motion": "grasp_lift", "hand_roles": {"left": "Rests on the table.", "right": "Lifts a card."},
                          "goal": "One card is held above the table.", "time_window_s": [0.7, 0.9]}],
            "task_success": success}


def write_stage1(sem, name="scene_semantics.json"):
    p = tmp / "stage1" / name
    p.write_text(json.dumps(sem, indent=1))
    return p


(ws / "scene.xml").write_text(scene_xml())
model = mujoco.MjModel.from_xml_path(str(ws / "scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
W, H = 320, 240
bm = S.render_body_masks(model, data, W, H, "video_cam")
for oid, names in (("stack", STACK), ("fan", FAN)):
    m = np.zeros((H, W), bool)
    for n in names:
        m |= bm.get(n, np.zeros((H, W), bool))      # a bar hidden behind the others adds nothing
    Image.fromarray((m * 255).astype(np.uint8)).save(ws / "masks" / f"{oid}.png")
    (ws / "assets" / f"{oid}.xml").write_text("<mujoco/>")
(ws / "camera.json").write_text(json.dumps({"fx": 300, "fy": 300, "cx": 160, "cy": 120, "pos": [0.15, -0.3, 0.6],
                                            "quat_wxyz": [1, 0, 0, 0], "width": W, "height": H}))
Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(ws / "frame0_compare.png")
index = tmp / "index.json"
index.write_text(json.dumps({"tools": [{"name": "construct_bars", "category": "Object Reconstruction", "type": "skill"}]}))


def entry(oid, names, parts=True, root=None):
    e = {"id": oid, "type": "rigid", "skill": "construct_bars", "skill_origin": "library", "asset_xml": f"assets/{oid}.xml",
         "root_body": root or names[0], "bodies": list(names), "pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
         "mask": f"masks/{oid}.png", "iou": 0.9, "verification_plan": {"settle_stability_check": "the pieces stay put"},
         "verifiers": {"settle_stability_check": {"ok": True}}, "properties_covered": {p: "settle_stability_check" for p in PROPS[oid]}}
    if parts:
        e["parts"] = list(names)
    return e


def write_recon(objects, sem_name="scene_semantics.json", name="scene_recon.json"):
    p = ws / name
    p.write_text(json.dumps({"scene_semantic": f"../stage1/{sem_name}", "scene_xml": "scene.xml", "camera_json": "camera.json",
                             "frame0_compare_png": "frame0_compare.png", "simulator": "mujoco", "objects": objects}, indent=1))
    return p


LOOSE3 = {"count": 3, "kind": "bar", "loose": True}
CARDS3 = {"count": 3, "kind": "card", "loose": True}
sem_p = write_stage1(stage1(LOOSE3, CARDS3))
recon = write_recon([entry("stack", STACK), entry("fan", FAN)])
scene = str(ws / "scene.xml")

# ---------------------------------------------------------------------------------------------------------- Stage 1
r = scene_semantic_schema_check(sem_p)
c = scene_semantic_consistency_check(sem_p)
check("Stage 1: pieces validate in the schema and the consistency check", r["ok"] and c["ok"], f"{r['issues']} {c['issues']}")
bad = stage1({"count": 1, "kind": "bar"}, {"count": 3})
r = scene_semantic_schema_check(write_stage1(bad, "bad.json"))
c = scene_semantic_consistency_check(tmp / "stage1" / "bad.json")
check("Stage 1: a one-piece group and a piece without a kind are refused", not r["ok"] and not c["ok"]
      and any("pieces.count" in i for i in c["issues"]) and any("kind" in i for i in r["issues"]), f"{r['issues']} {c['issues']}")
legacy = stage1()
r = scene_semantic_schema_check(write_stage1(legacy, "legacy.json"))
check("Stage 1: an object without pieces is valid as before", r["ok"], r["issues"])

# ------------------------------------------------------------------------------------------ free_joint_object_check
r = free_joint_object_check(scene, objects=str(recon))
check("free joints: the declared loose bars and cards pass", r["ok"] and r["objects"]["stack"]["parts"] == STACK, r["issues"])
r = free_joint_object_check(scene, objects=str(write_recon([entry("stack", STACK, parts=False), entry("fan", FAN, parts=False)], name="noparts.json")))
check("free joints: the same pieces without `parts` fail (untied)", not r["ok"] and any("stack" in i for i in r["issues"])
      and any("fan" in i for i in r["issues"]), r["issues"])
e = entry("stack", STACK)
e["parts"] = STACK[:2]
r = free_joint_object_check(scene, objects=str(write_recon([e, entry("fan", FAN)], name="partial.json")))
check("free joints: an undeclared loose bar still fails", not r["ok"] and r["objects"]["stack"]["untied_free_bodies"] == ["stack_c1s0"], r["issues"])
(ws / "fixed_top.xml").write_text(scene_xml(top_free=False))
r = free_joint_object_check(str(ws / "fixed_top.xml"), objects=str(recon))
check("free joints: a declared piece without its own free joint fails when Stage 1 calls the pieces loose",
      not r["ok"] and any("stack_c1s0" in i and "loose" in i for i in r["issues"]), r["issues"])

# ----------------------------------------------------------------------------------------- scene_recon_manifest_check
r = scene_recon_manifest_check(recon, library_index=index)
check("manifest: group entries with parts matching Stage 1 counts pass", r["ok"] and r["per_object"]["stack"]["parts"] == 3, r["issues"])
sem4 = write_stage1(stage1({"count": 4, "kind": "bar"}, CARDS3), "four.json")
r = scene_recon_manifest_check(write_recon([entry("stack", STACK), entry("fan", FAN)], "four.json", "r_four.json"), library_index=index)
check("manifest: 3 parts where Stage 1 says 4 bars fail", not r["ok"] and any("Stage 1 says 4" in i for i in r["issues"]), r["issues"])
sem_al = write_stage1(stage1({"count": 2, "at_least": True, "kind": "bar"}, CARDS3), "atleast.json")
r = scene_recon_manifest_check(write_recon([entry("stack", STACK), entry("fan", FAN)], "atleast.json", "r_al.json"), library_index=index)
check("manifest: at_least 2 bars accepts 3", r["ok"], r["issues"])
r = scene_recon_manifest_check(write_recon([entry("stack", STACK, parts=False), entry("fan", FAN)], name="r_np.json"), library_index=index)
check("manifest: Stage 1 loose pieces without parts fail", not r["ok"] and any("stack" in i and "parts" in i for i in r["issues"]), r["issues"])
e = entry("stack", STACK)
e["root_body"], e["parts"] = "stack_c1s0", STACK[:2] + ["fan_card0"]
r = scene_recon_manifest_check(write_recon([e, entry("fan", FAN)], name="r_root.json"), library_index=index)
check("manifest: a root outside the parts and a piece claimed by two objects fail",
      not r["ok"] and any("must be one of the pieces" in i for i in r["issues"]) and any("already belongs" in i for i in r["issues"]), r["issues"])
# without Stage 1 pieces (an older description): similar pieces are accepted, an object and its cap are not
(ws / "cap.xml").write_text(scene_xml(cap=True))
legacy_sem = stage1()
legacy_sem["objects"].append({"id": "bottle", "name": "blue bottle", "type": "rigid", "role": "static", "properties": ["A bottle with a cap."],
                              "evidence": "visible from 0.0 s"})
write_stage1(legacy_sem, "legacy_cap.json")
PROPS["bottle"] = ["A bottle with a cap."]
(ws / "assets" / "bottle.xml").write_text("<mujoco/>")
(ws / "masks" / "bottle.png").write_bytes((ws / "masks" / "fan.png").read_bytes())
rc = json.loads(write_recon([entry("stack", STACK), entry("fan", FAN), entry("bottle", ["bottle_body", "bottle_cap"])],
                            "legacy_cap.json", "r_cap.json").read_text())
rc["scene_xml"] = "cap.xml"
(ws / "r_cap.json").write_text(json.dumps(rc))
r = scene_recon_manifest_check(ws / "r_cap.json", library_index=index)
iss = [i for i in r["issues"] if "parts" in i]
check("manifest: without Stage 1 pieces, similar bars and cards pass and a bottle with its cap as 'parts' fails",
      len(iss) == 1 and "bottle" in iss[0] and "not alike" in iss[0], r["issues"])

# --------------------------------------------------------------------------------------- silhouettes and scene_qc
masks = {"stack": str(ws / "masks" / "stack.png"), "fan": str(ws / "masks" / "fan.png")}
r = check_silhouette_iou(scene, masks, "video_cam", groups={"stack": STACK, "fan": FAN})
check("silhouette: a group is scored by the union of its pieces", r["ok"] and min(r["iou"].values()) > 0.97, r["iou"])
r0 = check_silhouette_iou(scene, {"stack_c0s0": masks["stack"], "fan_card0": masks["fan"]}, "video_cam")
check("silhouette: one root body against the group's mask scores low (the old behaviour)", max(r0["iou"].values()) < 0.6, r0["iou"])
r = run_scene_qc(scene, required=STACK + FAN, masks=masks, camera="video_cam", groups={"stack": STACK, "fan": FAN})
check("scene_qc: groups pass silhouette while settle and support still judge every piece",
      r["ok"] and r["checks"]["settle_stability"]["ok"] and r["checks"]["support"]["ok"], r["failed"])
# single-body objects are unchanged: the same score with or without a groups argument
one = {"fan_card1": str(ws / "masks" / "card1.png")}
Image.fromarray((bm["fan_card1"] * 255).astype(np.uint8)).save(ws / "masks" / "card1.png")
a, b = check_silhouette_iou(scene, one, "video_cam"), check_silhouette_iou(scene, one, "video_cam", groups={})
check("silhouette: a single-body object scores the same as before", a["iou"] == b["iou"] and a["iou"]["fan_card1"] > 0.97, f"{a['iou']} {b['iou']}")

# ------------------------------------------------------------------------------------- object_count_vs_stage1_check
r = object_count_vs_stage1_check(scene, sem_p, recon)
rows = {(x["object"], x["noun"]): x for x in r["counts"]}
check("count: 3 parts against Stage 1's 3 bars and 3 cards pass; 'two cards' in the task is met by the fan's pieces",
      r["ok"] and rows[("stack", "bar")]["verdict"] == "ok" and rows[("fan", "card")]["verdict"] == "ok"
      and rows[("task", "card")]["scene"] == 3, r["counts"])
r = object_count_vs_stage1_check(scene, sem4, write_recon([entry("stack", STACK), entry("fan", FAN)], "four.json", "r_four.json"))
check("count: Stage 1's 4 bars against 3 parts fail", not r["ok"] and any("4 bar" in i for i in r["issues"]), r["issues"])
# single-body: two card objects where the task says three is still an exact mismatch
single = stage1(success="Three cards lie apart from the stack.")
single["objects"] = [{"id": "left_card", "name": "left card", "type": "rigid", "role": "manipulated", "properties": ["a card"], "evidence": "0 s"},
                     {"id": "right_card", "name": "right card", "type": "rigid", "role": "manipulated", "properties": ["a card"], "evidence": "0 s"}]
sp = write_stage1(single, "single.json")
rs = write_recon([dict(entry("fan", FAN, parts=False), id="left_card", root_body="fan_card0", bodies=["fan_card0"]),
                  dict(entry("fan", FAN, parts=False), id="right_card", root_body="fan_card1", bodies=["fan_card1"])], "single.json", "r_single.json")
r = object_count_vs_stage1_check(scene, sp, rs)
check("count: single-body objects keep the exact comparison (3 cards stated, 2 in the scene)", not r["ok"]
      and any(x["object"] == "task" and x["scene"] == 2 and not x.get("at_least") for x in r["counts"]), r["counts"])

# ------------------------------------------------------------------------------ hand_object_contact_consistency_check
names = ["rightThumbTip", "rightIndexFingerTip", "rightMiddleFingerTip", "rightRingFingerTip", "rightLittleFingerTip"]
X = np.full((30, len(names), 3), np.nan)
X[:, :, :] = [0.5, 0.5, 0.3]                                      # far away, except:
X[9:18, 1] = [0.0, 0.0126 + 0.0125 + 0.002, 0.0075]                 # subgoal 1 (0.3-0.6 s): index on the far face of stack_c0s1
X[21:27, 0] = [0.32, 0.03, 0.002]                                   # subgoal 2 (0.7-0.9 s): thumb on fan_card2
np.savez(ws / "hands.npz", names=np.array(names), rig_xyz=X)
r = hand_object_contact_consistency_check(scene, str(ws / "hands.npz"), str(sem_p), str(recon), fps=30.0, pad_s=0.0)
check("contact: a group's first contact is measured to its nearest piece", r["ok"] and r["objects"]["stack"]["part"] == "stack_c0s1"
      and r["objects"]["fan"]["part"] == "fan_card2", r["issues"] or r["objects"])
r = hand_object_contact_consistency_check(scene, str(ws / "hands.npz"), str(sem_p), str(ws / "noparts.json"), fps=30.0, pad_s=0.0)
check("contact: without parts the root body alone is too far (the old behaviour)", not r["ok"] and r["objects"]["stack"]["part"] is None,
      r["objects"])

# ---------------------------------------------------------------------------------------- Stage 3 goal resolution
from tool_library.common import task_gates as G  # noqa: E402
gc = G.GoalContext(G.Episode(model, qpos=data.qpos.copy()), json.loads(sem_p.read_text()), json.loads(recon.read_text()))
card2 = model.body("fan_card2").id
S_acted, name_acted, b_acted = G._resolve_set(gc, "fan", [card2])
S_all, name_all, b_all = G._resolve_set(gc, "fan", ())
fan_ids = {model.body(n).id for n in FAN}
check("Stage 3: 'a card of the fan' is the card the subgoal acted on; the fan as a whole otherwise",
      b_acted == card2 and S_acted == {card2} and b_all == model.body("fan_card0").id and fan_ids <= S_all, f"{name_acted} {S_acted} {S_all}")
r = G.recon_objects(json.loads(recon.read_text()))
check("Stage 3: recon_objects carries the parts", r["fan"]["parts"] == FAN and set(FAN) <= set(r["fan"]["bodies"]), r["fan"])

print(f"{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}: group objects ({tmp})")
sys.exit(1 if fails else 0)
