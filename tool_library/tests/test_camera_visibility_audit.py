"""camera_visibility_audit through the bench's ego camera: a box moved on the table passes; a box carried out of the
picture fails (camera json and scene camera); misuse is refused."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import json
import numpy as np
from tool_library.verifiers.camera_visibility_audit import camera_visibility_audit

tmp = S.tmpdir()
scene, m = S.bench(tmp, [dict(name="box", size=[0.03, 0.03, 0.03], pos=[0.0, 0.45, 0.03])])
q0 = S.home(m)
a = S.free_adr(m, "box")
cam = {"width": 640, "height": 480, "fovy_deg": 38.0, "pos": [0.01879, -0.18293, 0.55779],
       "quat_wxyz": [0.881567, 0.470897, 0.024947, 0.021763]}                 # setup.yaml's ego camera
camp = tmp / "camera.json"; camp.write_text(json.dumps(cam))
T = 90


def carry(to):
    Q = np.repeat(q0[None], T, 0)
    Q[:30, a:a + 3] = S.smooth(q0[a:a + 3], to, 30)
    Q[30:, a:a + 3] = to
    Q[30:, a] += np.linspace(0, 0.001, T - 30)                          # still moving slowly: the frames matter
    Q[30:, a] = to[0] + 0.03 * np.sin(np.linspace(0, 6, T - 30))
    return Q


C = S.Cases()


def on_table_passes():
    r = camera_visibility_audit(scene, S.save(tmp / "ok.npz", carry([0.1, 0.5, 0.03]), "right: push box #1"), camera=str(camp))
    assert r["ok"], r["issues"]
    assert r["bodies"]["box"]["out_of_view"] == 0, r["bodies"]


def out_of_picture_fails():
    t = S.save(tmp / "out.npz", carry([1.2, 0.6, 0.3]), "right: carry box #1")
    r = camera_visibility_audit(scene, t, camera=str(camp))
    assert not r["ok"] and r["worst"]["body"] == "box" and r["worst"]["out_frac"] > 0.5, r["worst"]
    r2 = camera_visibility_audit(scene, t, camera="ego")
    assert not r2["ok"] and r2["camera"]["source"] == "scene camera ego", r2["camera"]


def slack_at_the_border():
    t = S.save(tmp / "edge.npz", carry([0.3, 0.45, 0.03]), "right: push box #1")
    r = camera_visibility_audit(scene, t, camera=cam, outside_frac=0.0)
    r2 = camera_visibility_audit(scene, t, camera=cam)
    assert r2["ok"] and r["bodies"]["box"]["out_of_view"] >= r2["bodies"]["box"]["out_of_view"], (r["bodies"], r2["bodies"])


def misuse():
    t = S.save(tmp / "m.npz", carry([0.1, 0.5, 0.03]), "x: y")
    S.expect_value_error(lambda: camera_visibility_audit(scene, t, camera="nope"), "camera")
    S.expect_value_error(lambda: camera_visibility_audit(scene, t, camera={"width": 640}), "camera")


C.run("a box moved on the table stays in view of the ego camera", on_table_passes)
C.run("a box carried 1.2 m to the side fails (camera json and scene camera)", out_of_picture_fails)
C.run("the border slack only ever adds frames in view", slack_at_the_border)
C.run("an unknown camera name and an incomplete json are refused", misuse)
C.exit()
