"""Smoke test for stage_station_integrity_check: setup and recorded station shifts pass; unrecorded or different shifts, a moved table and an off-home keyframe fail."""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SETUP = ROOT / "setup"
TMP = Path(tempfile.mkdtemp())
fails = 0


def check(name, cond, msg=""):
    global fails
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + ("" if cond else f"  {str(msg)[:600]}"))
    fails += 0 if cond else 1


def bench(name, objects="", key_extra="", edit=None):
    """The setup's bench (setup/scene.xml) with objects added to the worldbody and their free-joint qpos appended to
    keyframe 'home', written to a temp file (meshes resolved from setup/)."""
    xml = (SETUP / "scene.xml").read_text()
    xml = xml.replace('<compiler angle="radian" />', f'<compiler angle="radian" meshdir="{SETUP}" texturedir="{SETUP}" />', 1)
    xml = xml.replace("</worldbody>", objects + "\n  </worldbody>", 1)
    if key_extra:
        xml = re.sub(r'(<key name="home" qpos=")([^"]*)', lambda m: m.group(1) + m.group(2) + " " + key_extra, xml, count=1)
    if edit:
        xml = edit(xml)
    p = TMP / f"{name}.xml"
    p.write_text(xml)
    return str(p)


CUBE = ('<body name="cube" pos="0 0.5 0.025"><freejoint name="cube_free"/>'
        '<geom name="cube_geom" type="box" size="0.025 0.025 0.025" mass="0.1" rgba="0.8 0.2 0.2 1"/></body>')
CUBE_Q = "0 0.5 0.025 1 0 0 0"


import json

import mujoco
import numpy as np

from tool_library.verifiers.stage_station_integrity_check import stage_station_integrity_check as check_st


def shifted(name, dx, dy, sides=("left", "right"), table_dz=0.0, finger=None, per=None):
    """place_arm_stations' edit: move the station roots and each hand's home x/y with them (`per` {side: (dx, dy)}
    gives each station its own shift)."""
    per = per or {s: (dx, dy) for s in sides}
    spec = mujoco.MjSpec.from_file(bench(name + "_src", CUBE, CUBE_Q))
    m = spec.compile()
    q = np.array(spec.keys[0].qpos, float)
    c = np.array(spec.keys[0].ctrl, float)
    for s, (sx, sy) in per.items():
        b = spec.body(f"robot_{s}_arm_root")
        b.pos = list(np.asarray(b.pos, float) + [sx, sy, 0.0])
        for ax, d in (("x", sx), ("y", sy)):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot_{s}_{s}_pos_{ax}")
            q[m.jnt_qposadr[j]] += d
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot_{s}_{s}_pos_{ax}_position")
            c[a] += d
    if finger:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, finger)
        q[m.jnt_qposadr[j]] += 0.3
    if table_dz:
        t = spec.body("table")
        t.pos = list(np.asarray(t.pos, float) + [0, 0, table_dz])
    spec.keys[0].qpos = q.tolist(); spec.keys[0].ctrl = c.tolist()
    p = TMP / f"{name}.xml"
    p.write_text(spec.to_xml())
    return str(p)


def report(dx, dy, name):
    p = TMP / f"{name}.json"
    p.write_text(json.dumps({"shift": [dx, dy], "scene": "scene_stations.xml"}))
    return str(p)


r = check_st(str(SETUP / "scene.xml"))
check("the setup's own stations, table and home pass", r["ok"] and r["home"]["joints_compared"] >= 60, r["issues"])
r = check_st(bench("cube", CUBE, CUBE_Q))
check("a Stage 2 bench with an object passes", r["ok"], r["issues"])

p = shifted("both", 0.0, -0.075)
r = check_st(p, stations_report=report(0.0, -0.075, "rep_both"))
check("both stations moved by a recorded place_arm_stations shift pass", r["ok"] and all(v["explained"] for v in r["stations"].values()), r["issues"])
r = check_st(p)
check("the same move without a record passes as a station shift, with a warning", r["ok"] and r.get("warnings"), r["issues"][:1])
r = check_st(p, unrecorded_shift_ok=False)
check("the same move without a record fails when unrecorded shifts are not allowed", (not r["ok"]) and not r["stations"]["left"]["explained"], r["issues"][:1])
r = check_st(shifted("both", 0.0, 0.8))
check("an unrecorded move beyond place_arm_stations' range fails", not r["ok"], r["issues"][:1])
r = check_st(p, stations_report=report(0.0, 0.2, "rep_other"))
check("a record of a different shift fails", (not r["ok"]) and "recorded shift" in " ".join(r["issues"]), r["issues"][:1])

p = shifted("one", -0.1, 0.05, sides=("right",))
r = check_st(p, stations_report=report(-0.1, 0.05, "rep_one"))
check("one station moved alone (stations=['right']) passes with its record", r["ok"] and r["stations"]["left"]["shift"] == [0.0, 0.0, 0.0], r["issues"])

# two place_arm_stations calls: both stations +0.15/+0.10, then the right one a further +0.10 in x
p = shifted("chain", 0, 0, per={"left": (0.15, 0.10), "right": (0.25, 0.10)})
rp = TMP / "rep_chain.json"
rp.write_text(json.dumps({"shift": [0.10, 0.0], "stations_moved": ["right"], "station_shifts": {"left": [0.15, 0.10], "right": [0.25, 0.10]}}))
r = check_st(p, stations_report=str(rp))
check("chained calls pass with each station's recorded total", r["ok"], r["issues"])
r = check_st(p, stations_report=report(0.10, 0.0, "rep_last_only"))
check("a record of only the last call's shift fails", not r["ok"], r["issues"][:1])
rp.write_text(json.dumps({"shift": [0.10, 0.0], "station_shifts": {"left": [0.15, 0.10], "right": [0.25, 0.10]}}))
r = check_st(shifted("chain_undone", 0, 0, per={"right": (0.25, 0.10)}), stations_report=str(rp))
check("a station put back although its record says it moved fails", not r["ok"] and not r["stations"]["left"]["explained"], r["issues"][:1])

r = check_st(bench("cube_probe", CUBE, CUBE_Q), stations_report=report(0.0, -0.025, "rep_probe"))
check("a report of a probe shift beside a scene whose stations all stand at the setup is ignored, with a warning",
      r["ok"] and any("describes another scene" in w for w in r.get("warnings", [])), r["issues"][:1])

p = shifted("table", 0.0, 0.0, table_dz=-0.03)
r = check_st(p)
check("a lowered table is caught", (not r["ok"]) and r["table"]["changed"], r["issues"][:1])
p = shifted("finger", 0.0, 0.0, finger="robot_left_left_index_PIP")
r = check_st(p)
check("a finger off the home pose in keyframe 0 is caught", (not r["ok"]) and any(c["joint"] == "robot_left_left_index_PIP" for c in r["home"]["changed"]), r["issues"][:1])

print(f"{'OK' if fails == 0 else 'FAILED'}: stage_station_integrity_check ({fails} failure(s))")
sys.exit(1 if fails else 0)
