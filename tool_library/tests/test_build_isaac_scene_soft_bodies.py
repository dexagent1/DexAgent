import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.build_isaac_scene_soft_bodies import build_isaac_scene_soft_bodies

CAM = {"width": 320, "height": 256, "fovy_deg": 45.0, "pos": [0.0, -0.45, 0.45],
       "R_cv": [[1, 0, 0], [0, -0.7071, 0.7071], [0, -0.7071, -0.7071]]}
SOFT = 'solref="0.03 1" solimp="0.5 0.95 0.012" friction="1 0.01 0.001" condim="4"'


def plush(name, x, y, z, a, b, c, rgba="0.8 0.7 0.5 1", mass=0.03):
    return (f'<body name="{name}" pos="{x} {y} {z + c}"><freejoint/>'
            f'<geom name="{name}_torso" type="ellipsoid" size="{a} {b} {c}" mass="{mass}" rgba="{rgba}" {SOFT}/>'
            f'<geom name="{name}_head" type="sphere" size="{0.6 * c}" pos="{a} 0 {0.5 * c}" mass="{0.2 * mass}" rgba="{rgba}" {SOFT}/>'
            f'<geom name="{name}_eye" type="sphere" size="0.002" pos="{a + 0.6 * c} 0 {0.6 * c}" contype="0" conaffinity="0" mass="0" rgba="0 0 0 1"/>'
            f'</body>')


def scene(path, bodies, extra=""):
    xml = ('<mujoco><option timestep="0.002"/><worldbody>'
           '<body name="table" pos="0 0 -0.02"><geom name="g_table" type="box" size="0.5 0.4 0.02" rgba="0.9 0.9 0.9 1"/></body>'
           f'{extra}{bodies}</worldbody><keyframe/></mujoco>').replace("<keyframe/>", "")
    Path(path).write_text(xml)
    return str(path)


py = os.environ.get("DEXAGENT_ISAAC_PYTHON")
if not py or not Path(py).exists():
    # an optional simulator: report the coverage gap (run_all counts it as incomplete), not a failure
    print("[SKIP] soft-body variations: set DEXAGENT_ISAAC_PYTHON to Isaac Sim's Python")
    sys.exit(0)
fails = 0


def case(name, cond, info=""):
    global fails
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else f": {info}"))
    fails += 0 if cond else 1


with tempfile.TemporaryDirectory() as tmp:
    t = Path(tmp)
    # 1: one plush on the table: settles in place
    s1 = scene(t / "s1.xml", plush("toy", 0.0, 0.05, 0.0, 0.03, 0.025, 0.02))
    r = build_isaac_scene_soft_bodies(s1, CAM, [{"body": "toy"}], str(t / "o1"))
    rep = r["report"]["toy"]
    case("single plush settles on the table", rep["mean_drift_m"] < 0.01 and Path(r["scene_usda"]).exists()
         and Path(r["sim_png"]).exists() and rep["n_geoms"] == 3, rep)

    # 2: two soft bodies, one with explicit stiffness, plus a static box (collides, not exported as soft)
    box = '<body name="crate" pos="0.12 0.0 0.03"><geom name="crate_g" type="box" size="0.04 0.04 0.03" rgba="0.4 0.4 0.45 1"/></body>'
    s2 = scene(t / "s2.xml", plush("a", -0.1, 0.0, 0.0, 0.04, 0.03, 0.025) + plush("b", 0.0, 0.1, 0.0, 0.02, 0.02, 0.015, "0.9 0.3 0.3 1"), box)
    r = build_isaac_scene_soft_bodies(s2, CAM, [{"body": "a"}, {"body": "b", "stiffness_N_per_m": 400.0}], str(t / "o2"), settle_s=1.0)
    ra, rb = r["report"]["a"], r["report"]["b"]
    case("two soft bodies with a given stiffness", ra["mean_drift_m"] < 0.01 and rb["mean_drift_m"] < 0.01
         and abs(rb["stiffness_N_per_m"] - 400.0) < 1e-6 and ra["stiffness_N_per_m"] > 0, (ra, rb))

    # 3: a plush resting ON a static box: it must stay on top (box collision exported), not fall through
    s3 = scene(t / "s3.xml", plush("top", 0.0, 0.0, 0.06, 0.03, 0.025, 0.02), box.replace("0.12 0.0 0.03", "0.0 0.0 0.03"))
    r = build_isaac_scene_soft_bodies(s3, CAM, [{"body": "top"}], str(t / "o3"))
    rep = r["report"]["top"]
    case("plush resting on a static box stays on it", rep["end_pos"][2] > 0.05 and rep["mean_drift_m"] < 0.01, rep)

    # 4: dry run writes the export and the script without Isaac; misuse raises
    r = build_isaac_scene_soft_bodies(s1, CAM, [{"body": "toy"}], str(t / "o4"), dry_run=True)
    exp = json.loads(Path(r["export_json"]).read_text())
    case("dry run exports soft body separately from statics", Path(r["script"]).exists() and len(exp["soft"]) == 1
         and all(g["body"] != "toy" for g in exp["geoms"]), r)
    for bad, kw in (("empty soft_bodies", dict(soft_bodies=[])), ("unknown body", dict(soft_bodies=[{"body": "nope"}])),
                    ("unknown key", dict(soft_bodies=[{"body": "toy", "colour": 1}]))):
        try:
            build_isaac_scene_soft_bodies(s1, CAM, out_dir=str(t / "o5"), dry_run=True, **kw)
            case(f"misuse raises: {bad}", False, "no error")
        except ValueError:
            case(f"misuse raises: {bad}", True)

sys.exit(1 if fails else 0)
