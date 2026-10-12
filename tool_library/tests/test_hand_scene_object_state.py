import json, sys, tempfile
from pathlib import Path
import numpy as np
import mujoco
from pathlib import Path as _RepoPath
sys.path.insert(0, str(_RepoPath(__file__).resolve().parents[2]))  # runnable from anywhere
from tool_library.skills.hand_scene_object_state import hand_scene_object_state

XML = """<mujoco><worldbody>
<body name="rope_B0" pos="0 0 0.1"><freejoint name="rope_root"/><geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.01"/>
 <body name="rope_B1" pos="0.05 0 0"><joint name="rope_J1" type="ball"/><geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.01"/>
  <body name="rope_B2" pos="0.05 0 0"><joint name="rope_J2" type="ball"/><geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.01"/></body></body></body>
<body name="box_base" pos="0.5 0 0"><geom type="box" size="0.05 0.05 0.02"/>
 <body name="box_lid" pos="0 0 0.03"><joint name="box_hinge" type="hinge" axis="0 1 0"/><geom type="box" size="0.05 0.05 0.005"/></body>
 <body name="box_drawer" pos="0 0 -0.01"><joint name="box_slide" type="slide" axis="1 0 0"/><geom type="box" size="0.04 0.04 0.005"/></body></body>
</worldbody></mujoco>"""

fails = 0
with tempfile.TemporaryDirectory() as td:
    td = Path(td); (td / "s.xml").write_text(XML)
    m = mujoco.MjModel.from_xml_path(str(td / "s.xml"))
    T = 5; Q = np.zeros((T, m.nq))
    for t in range(T):
        d = mujoco.MjData(m); q = d.qpos.copy()
        q[m.jnt_qposadr[m.joint("rope_J1").id]:][:4] = [np.cos(0.1 * t), 0, 0, np.sin(0.1 * t)]
        q[m.jnt_qposadr[m.joint("rope_J2").id]:][:4] = [np.cos(0.2 * t), 0, np.sin(0.2 * t), 0]
        q[m.jnt_qposadr[m.joint("box_hinge").id]] = 0.3 * t
        q[m.jnt_qposadr[m.joint("box_slide").id]] = 0.01 * t
        Q[t] = q
    np.savez(td / "traj.npz", qpos=Q)
    hs = {"hand_xml": "h.xml", "side": "right", "objects": [{"name": "rope"}], "joint_init": {"other": 0.5}}
    # case 1: cable ball joints, last frame, written to json
    r = hand_scene_object_state(hs, str(td / "s.xml"), str(td / "traj.npz"), ["rope_"], out_json=str(td / "o.json"))
    ji = json.loads((td / "o.json").read_text())["joint_init"]
    # the root's free joint comes along (7 values) so the rope lands where the bench has it
    ok = r["n_joints"] == 3 and len(ji.get("rope_root", [])) == 7 and np.allclose(ji["rope_J1"], [np.cos(0.4), 0, 0, np.sin(0.4)]) and ji["other"] == 0.5
    print("[PASS] cable ball joints and root" if ok else "[FAIL] cable ball joints and root"); fails += not ok
    r0 = hand_scene_object_state(hs, str(td / "s.xml"), str(td / "traj.npz"), ["rope_"], include_free=False)
    ok = r0["n_joints"] == 2 and "rope_root" not in r0["hand_scene"]["joint_init"]
    print("[PASS] include_free=False leaves the root to the scene" if ok else "[FAIL] include_free=False leaves the root to the scene"); fails += not ok
    # case 2: hinge + slide at a middle frame, with grasp_body
    r = hand_scene_object_state(hs, str(td / "s.xml"), str(td / "traj.npz"), ["box_"], frame=2, grasp_body="box_lid")
    ji = r["hand_scene"]["joint_init"]
    ok = abs(ji["box_hinge"] - 0.6) < 1e-9 and abs(ji["box_slide"] - 0.02) < 1e-9 and r["hand_scene"]["grasp_body"] == "box_lid"
    print("[PASS] hinge and slide at frame 2" if ok else "[FAIL] hinge and slide"); fails += not ok
    # case 3: misuse raises with the parameter named
    bad = 0
    for kw, word in [({"prefixes": ["nothing_"]}, "prefixes"), ({"prefixes": ["rope_"], "grasp_body": "nope"}, "grasp_body")]:
        try:
            hand_scene_object_state(hs, str(td / "s.xml"), str(td / "traj.npz"), **kw)
        except ValueError as e:
            bad += word in str(e)
    np.savez(td / "bad.npz", qpos=np.zeros((3, m.nq + 1)))
    try:
        hand_scene_object_state(hs, str(td / "s.xml"), str(td / "bad.npz"), ["rope_"])
    except ValueError as e:
        bad += "trajectory" in str(e)
    ok = bad == 3
    print("[PASS] misuse raises" if ok else "[FAIL] misuse raises"); fails += not ok
sys.exit(1 if fails else 0)
