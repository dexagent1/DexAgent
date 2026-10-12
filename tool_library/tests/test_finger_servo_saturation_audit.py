"""finger_servo_saturation_audit on a two-pad gripper (position servos capped at 10 N): a light box carried between the
pads needs a few percent of the cap and passes; a 5 kg box needs more than the cap and fails; a cap given as a
parameter is used when the scene sets none."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures" / "episode_audits"))
import synth as S
import numpy as np
from tool_library.verifiers.finger_servo_saturation_audit import finger_servo_saturation_audit

tmp = S.tmpdir()


def gripper(mass, limited=True):
    lim = 'forcelimited="true" forcerange="-10 10"' if limited else ""
    xml = f"""<mujoco model="gripper"><option timestep="0.002"/><worldbody>
<body name="table"><geom name="g_table" type="box" size="0.5 0.5 0.02" pos="0 0 -0.02"/></body>
<body name="robot_right_hand">
  <joint name="robot_right_pos_x" type="slide" axis="1 0 0"/><joint name="robot_right_pos_y" type="slide" axis="0 1 0"/>
  <joint name="robot_right_pos_z" type="slide" axis="0 0 1"/>
  <geom name="palm" type="box" size="0.04 0.01 0.01" pos="0 0 0.08" contype="0" conaffinity="0" mass="0.2"/>
  <body name="robot_right_finger_l" pos="-0.03 0 0.02"><joint name="robot_right_finger_l_slide" type="slide" axis="1 0 0" range="-0.02 0.03"/>
    <geom name="pad_l" type="box" size="0.005 0.01 0.015" mass="0.02"/></body>
  <body name="robot_right_finger_r" pos="0.03 0 0.02"><joint name="robot_right_finger_r_slide" type="slide" axis="-1 0 0" range="-0.02 0.03"/>
    <geom name="pad_r" type="box" size="0.005 0.01 0.015" mass="0.02"/></body>
</body>
<body name="box" pos="0 0 0.02"><freejoint name="box_free"/><geom name="box_geom" type="box" size="0.02 0.02 0.02" mass="{mass}"/></body>
</worldbody><actuator>
<position name="fl" joint="robot_right_finger_l_slide" kp="1000" {lim}/><position name="fr" joint="robot_right_finger_r_slide" kp="1000" {lim}/>
</actuator></mujoco>"""
    p = tmp / f"gripper_{mass}_{limited}.xml"
    p.write_text(xml)
    return str(p)


T = 60
Q = np.zeros((T, 3 + 2 + 7))
Q[:, 3:5] = 0.0053                                                   # both pads 0.3 mm into the box's sides
Q[:, 2] = S.smooth([0], [0.10], T)[:, 0]                             # the hand lifts 10 cm
Q[:, 5:8] = np.c_[np.zeros(T), np.zeros(T), 0.02 + Q[:, 2]]
Q[:, 8] = 1.0
L = ["right: grip box #1"] * 10 + ["right: carry box #1"] * 50
C = S.Cases()


def light_box_passes():
    sc = gripper(0.05)
    r = finger_servo_saturation_audit(sc, S.save(tmp / "light.npz", Q, L), stride=1)
    assert r["ok"], r["issues"]
    w = r["windows"][0]
    assert w["object"] == "box" and w["judged"] > 30 and w["median_s"] < 0.1, w


def heavy_box_fails():
    sc = gripper(5.0)
    r = finger_servo_saturation_audit(sc, S.save(tmp / "heavy.npz", Q, L), stride=1)
    assert not r["ok"] and r["worst"]["median_s"] > 1.5 and r["worst"]["saturated_frac"] > 0.9, r["worst"]
    assert "robot_right_finger" in (r["worst"]["limiting_joint"] or ""), r["worst"]


def parameter_cap():
    sc = gripper(0.5, limited=False)                                  # no forcerange in the scene
    t = S.save(tmp / "free.npz", Q, L)
    r = finger_servo_saturation_audit(sc, t, stride=2)
    assert r["ok"] and "no finger actuator" in (r.get("note") or ""), r
    r = finger_servo_saturation_audit(sc, t, finger_cap_nm=1.0, stride=2)            # 1 N cap: 4.9 N of weight needs 2.45 N per pad
    assert not r["ok"] and r["worst"]["median_s"] > 2, r["worst"]


def misuse():
    sc = gripper(0.05)
    S.expect_value_error(lambda: finger_servo_saturation_audit(sc, S.save(tmp / "m.npz", Q, L), sat_frac=0), "sat_frac")
    S.expect_value_error(lambda: finger_servo_saturation_audit(sc, S.save(tmp / "m2.npz", Q, L), spin_m=-1), "spin_m")


C.run("a 50 g box carried between 10 N pads passes", light_box_passes)
C.run("a 5 kg box between the same pads fails (needs about 2.5x the cap)", heavy_box_fails)
C.run("without a scene cap nothing is judged; finger_cap_nm=1 N judges and fails a 0.5 kg box", parameter_cap)
C.run("a zero saturation share and a negative torsion length are refused", misuse)
C.exit()
