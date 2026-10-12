"""asset_scene_consistency_check: the asset recorded for an object is the object the scene holds (bodies and the geoms they
carry), objects sharing one asset are compared together.

    python tool_library/tests/test_asset_scene_consistency_check.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tool_library.verifiers.asset_scene_consistency_check import asset_scene_consistency_check as check_asc  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="dexagent_asc_", dir=os.environ.get("TMPDIR")))
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    FAILS += 0 if ok else 1


def write(name, text):
    (TMP / name).write_text(text)
    return name


CORD = '<body name="cordB_first"><freejoint/><geom type="capsule" size="0.002 0.01"/><body name="cordB_last" pos="0.02 0 0"><joint type="ball"/><geom type="capsule" size="0.002 0.01"/>{plug}</body></body>'
PLUG = '<geom type="box" size="0.004 0.006 0.002" pos="0.012 0 0"/><geom type="box" size="0.002 0.004 0.001" pos="0.018 0 0"/>'
write("cord_bare.xml", f"<mujoco><worldbody>{CORD.format(plug='')}</worldbody></mujoco>")
write("cord_plug.xml", f"<mujoco><worldbody>{CORD.format(plug=PLUG)}</worldbody></mujoco>")
scene = CORD.format(plug=PLUG).replace('name="cord', 'name="cable_cord')
write("scene.xml", f'<mujoco><worldbody><body name="table"><geom type="box" size="0.5 0.5 0.01"/></body>{scene}'
                   '<body name="box_box" pos="0.2 0 0"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>'
                   '<body name="box_lid" pos="0.2 0 0.05"><freejoint/><geom type="box" size="0.02 0.02 0.002"/></body></worldbody></mujoco>')
write("box.xml", '<mujoco><worldbody><body name="box"><freejoint/><geom type="box" size="0.02 0.02 0.02"/></body>'
                 '<body name="lid" pos="0 0 0.05"><freejoint/><geom type="box" size="0.02 0.02 0.002"/></body></worldbody></mujoco>')


def recon(cord_asset):
    return {"objects": [{"id": "cable", "type": "deformable", "root_body": "cable_cordB_first", "asset_xml": cord_asset},
                        {"id": "box", "type": "rigid", "root_body": "box_box", "asset_xml": "box.xml"},
                        {"id": "lid", "type": "rigid", "root_body": "box_lid", "asset_xml": "box.xml"}]}


(TMP / "r1.json").write_text(json.dumps(recon("cord_plug.xml")))
r = check_asc(TMP / "scene.xml", TMP / "r1.json")
check("the asset the scene was staged from passes; a box and its free lid sharing one asset pass together", r["ok"], str(r["issues"]))
(TMP / "r2.json").write_text(json.dumps(recon("cord_bare.xml")))
r = check_asc(TMP / "scene.xml", TMP / "r2.json")
check("a cable asset without the plug the scene carries fails, naming the body", not r["ok"] and "cordB_last" in r["issues"][0], str(r["issues"]))
(TMP / "r3.json").write_text(json.dumps({"objects": [{"id": "box", "type": "rigid", "root_body": "box_box", "asset_xml": "cord_plug.xml"}]}))
r = check_asc(TMP / "scene.xml", TMP / "r3.json")
check("an asset that is another object entirely fails", not r["ok"], str(r["issues"])[:160])
(TMP / "r4.json").write_text(json.dumps({"objects": [{"id": "rope", "type": "deformable", "root_body": "rope_isaac_only", "asset_xml": "cord_plug.xml"}]}))
r = check_asc(TMP / "scene.xml", TMP / "r4.json")
check("a deformable simulated only in Isaac is skipped", r["ok"] and "rope" in r["skipped"], str(r))

print("ALL PASS" if not FAILS else f"{FAILS} FAILED")
sys.exit(1 if FAILS else 0)
