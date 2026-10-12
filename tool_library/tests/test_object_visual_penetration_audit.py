"""object_visual_penetration_audit on meshes the collision geometry does not follow: a cup standing inside a bowl
passes; a cup half inside the bowl's wall fails; a puck whose mesh hangs 10 mm under its collision box sinks into the
table and fails; an open mesh is reported, not judged as a container."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
import trimesh
from tool_library.verifiers.object_visual_penetration_audit import object_visual_penetration_audit

tmp = S.tmpdir()
wall = trimesh.creation.annulus(r_min=0.045, r_max=0.05, height=0.04)      # the bowl's wall, standing on the table
wall.apply_translation([0, 0, 0.018])
cup = trimesh.creation.cylinder(radius=0.015, height=0.04)
puck = trimesh.creation.cylinder(radius=0.02, height=0.03)
puck.apply_translation([0, 0, -0.005])                                      # 10 mm lower than its collision box's bottom
objs = [dict(name="bowl", size=[0.05, 0.05, 0.002], pos=[0.0, 0.5, 0.002], visual_mesh=(wall.vertices, wall.faces)),
        dict(name="cup", size=[0.01, 0.01, 0.02], pos=[0.0, 0.5, 0.024], visual_mesh=(cup.vertices, cup.faces)),
        dict(name="puck", size=[0.02, 0.02, 0.01], pos=[0.3, 0.5, 0.01], visual_mesh=(puck.vertices, puck.faces))]
scene, m = S.bench(tmp, objs)
q0 = S.home(m)
ca, pa = S.free_adr(m, "cup"), S.free_adr(m, "puck")
T = 10
C = S.Cases()


def cup_inside_passes():
    Q = np.repeat(q0[None], T, 0)
    Q[:, pa + 2] = 0.025                                              # puck mesh clear of the table
    r = object_visual_penetration_audit(scene, S.save(tmp / "ok.npz", Q, "left: set down cup #1"))
    assert r["ok"], r["issues"]
    assert r["pairs"]["bowl|cup"]["depth_mm"] < 2, r["pairs"]


def cup_in_the_wall_fails():
    Q = np.repeat(q0[None], T, 0)
    Q[:, pa + 2] = 0.025
    Q[:, ca] = 0.045                                                   # the cup's side through the wall
    r = object_visual_penetration_audit(scene, S.save(tmp / "wall.npz", Q, "left: set down cup #1"))
    assert not r["ok"] and set(r["worst"]["objects"]) == {"bowl", "cup"} and r["worst"]["depth_mm"] > 5, r["worst"]


def puck_sinks_into_table_fails():
    Q = np.repeat(q0[None], T, 0)
    r = object_visual_penetration_audit(scene, S.save(tmp / "puck.npz", Q, "left: set down cup #1"))
    assert not r["ok"] and r["table"]["puck"]["depth_mm"] > 8 and any("tabletop" in s for s in r["issues"]), r["table"]
    assert object_visual_penetration_audit(scene, S.save(tmp / "puck2.npz", Q, "x: y"), table_tol_m=0.02)["ok"]


def misuse():
    S.expect_value_error(lambda: object_visual_penetration_audit(scene, S.save(tmp / "m.npz", np.repeat(q0[None], T, 0), "x: y"), pitch_m=0), "pitch_m")


C.run("a cup standing inside a bowl (meshes apart) passes", cup_inside_passes)
C.run("a cup whose mesh goes through the bowl's wall fails though collision geoms do not touch", cup_in_the_wall_fails)
C.run("a puck mesh hanging 10 mm below its collision box fails as sunk into the table", puck_sinks_into_table_fails)
C.run("a zero voxel pitch is refused", misuse)
C.exit()
