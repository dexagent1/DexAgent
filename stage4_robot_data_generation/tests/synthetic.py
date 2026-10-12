"""Small synthetic Stage 4 task folders for the tests (no external data).

stack_task: a free "hand" (a box on six joints) carries box A onto box B and then box C onto A, so C ends on A on
B: a two-level tower whose goals are relative to B. Kinematic only, builds in milliseconds.

bench_task: the real bench (setup/scene.xml: two YAM arms with Sharpa hands, parked at home) with a box and a
small cube resting on it, as a legacy floating-wrist recording (qpos only): Stage 4 must refuse it.

rigid_bench_task: the same bench compiled to the rigid arm-hand runtime with a complete servo recording (what Stage 3
hands off), for the physics replay: the hands do nothing, so the only question is whether the cube is still
supported in the new layout.

Every task folder gets a small decodable human video (the handoff check decodes it).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]

CAMERA = {"pos": [0.0, -0.4, 0.6], "quat_wxyz": [0.92, 0.38, 0.0, 0.0], "fovy": 45.0}
TABLE = {"top_z": 0.0, "center": [0.0, 0.5], "half_size": [0.8, 0.6]}

STACK_XML = """<mujoco model="stack">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="robot_right_right_base_yaw" pos="0 0 0">
      <joint name="hx" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="hy" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="hz" type="slide" axis="0 0 1" range="-2 2"/>
      <joint name="hrx" type="hinge" axis="1 0 0" range="-3.1 3.1"/>
      <joint name="hry" type="hinge" axis="0 1 0" range="-3.1 3.1"/>
      <joint name="hrz" type="hinge" axis="0 0 1" range="-3.1 3.1"/>
      <geom name="palm" type="box" size="0.03 0.03 0.01" pos="0 0 0.012" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
    </body>
    <body name="A" pos="0.10 0.40 0.02"><freejoint name="A"/><geom name="A" type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"/></body>
    <body name="B" pos="0.30 0.40 0.01"><freejoint name="B"/><geom name="B" type="box" size="0.05 0.05 0.01" rgba="0 1 0 1"/></body>
    <body name="C" pos="0.10 0.60 0.015"><freejoint name="C"/><geom name="C" type="box" size="0.015 0.015 0.015" rgba="0 0 1 1"/></body>
    {extra}
  </worldbody>
</mujoco>
"""

HALF = {"A": 0.02, "B": 0.01, "C": 0.015}


def _qadr(model, body):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return int(model.jnt_qposadr[model.body_jntadr[b]])


def stack_trajectory(model, sink_m=0.0005):
    """(Q, labels, phases): the hand carries A onto B, then C onto A. Objects sit sink_m into their support so the
    final state has contacts; a carried object hangs right under the palm."""
    hand = [0, 1, 2, 3, 4, 5]
    pos0 = {n: model.body(n).pos.copy() for n in ("A", "B", "C")}
    home = np.array([0.0, 0.2, 0.25])
    Q, labels, phases = [], [], []
    state = {n: pos0[n].copy() for n in pos0}
    hp = home.copy()
    held = None

    def frame(label):
        q = np.zeros(model.nq)
        q[hand[:3]] = hp
        for n, p in state.items():
            a = _qadr(model, n)
            q[a:a + 3] = p
            q[a + 3:a + 7] = [1, 0, 0, 0]
        Q.append(q)
        labels.append(label)

    def move(label, target, n=20, hand_name="right", objects=(), anchor=None, subgoal=1):
        nonlocal hp
        start = hp.copy()
        t0 = len(Q)
        for k in range(1, n + 1):
            u = k / n
            hp = (1 - u) * start + u * np.asarray(target, float)
            if held:
                state[held] = hp - np.array([0.0, 0.0, HALF[held]])
            frame(label)
        phases.append({"label": label, "hand": hand_name, "objects": list(objects), "anchor": anchor,
                       "subgoal": subgoal, "frames": [[t0, len(Q) - 1]]})

    def carry(obj, onto, below, sg):
        nonlocal held
        top = pos0[obj] + np.array([0.0, 0.0, HALF[obj]])
        move(f"right: reach for {obj} #{sg}", top + [0, 0, 0.08], objects=[obj], anchor=obj, subgoal=sg)
        move(f"right: descend on {obj} #{sg}", top, objects=[obj], anchor=obj, subgoal=sg)
        move(f"right: grip {obj} #{sg}", top, n=10, objects=[obj], anchor=obj, subgoal=sg)
        held = obj
        move(f"right: lift {obj} #{sg}", top + [0, 0, 0.06], objects=[obj], anchor=obj, subgoal=sg)
        goal_xy = state[onto][:2]
        z_goal = state[onto][2] + HALF[onto] + 2 * HALF[obj] - sink_m
        move(f"right: carry {obj} #{sg + 1}", [goal_xy[0], goal_xy[1], z_goal + 0.06], n=30, objects=[obj], anchor=onto, subgoal=sg + 1)
        move(f"right: set down {obj} #{sg + 1}", [goal_xy[0], goal_xy[1], z_goal], objects=[obj], anchor=onto, subgoal=sg + 1)
        held = None
        move(f"right: let go of {obj} #{sg + 1}", [goal_xy[0], goal_xy[1], z_goal], n=10, objects=[obj], anchor=onto, subgoal=sg + 1)
        move(f"right: retreat #{sg + 1}", [goal_xy[0], goal_xy[1], z_goal + 0.08], objects=[obj], anchor=onto, subgoal=sg + 1)

    frame("right: reach for A #1")
    phases.append({"label": "right: reach for A #1", "hand": "right", "objects": ["A"], "anchor": "A", "subgoal": 1, "frames": [[0, 0]]})
    carry("A", "B", None, 1)
    carry("C", "A", None, 3)
    move("right: return home #5", home, n=30, objects=[], anchor=None, subgoal=5)
    # merge the duplicate first phase entry
    phases[0]["frames"] = [[0, phases[1]["frames"][0][1]]]
    phases.pop(1)
    return np.array(Q), labels, phases


def stack_segments():
    """Anchors Stage 4 needs for the stack task: the hand follows A for A's grasp, B for its set-down, and so on."""
    return {"right": [
        {"phases": ["right: reach for A #1"], "blend": ["world", "A"]},
        {"phases": ["right: descend on A #1", "right: grip A #1", "right: lift A #1"], "anchor": "A"},
        {"phases": ["right: carry A #2"], "blend": ["A", "B"]},
        {"phases": ["right: set down A #2", "right: let go of A #2", "right: retreat #2"], "anchor": "B"},
        {"phases": ["right: reach for C #3"], "blend": ["B", "C"]},
        {"phases": ["right: descend on C #3", "right: grip C #3", "right: lift C #3"], "anchor": "C"},
        {"phases": ["right: carry C #4"], "blend": ["C", "A"]},
        {"phases": ["right: set down C #4", "right: let go of C #4", "right: retreat #4"], "anchor": "A"},
        {"phases": ["right: return home #5"], "blend": ["A", "world"]}]}


def write_video(path, frames=8, size=(64, 48), fps=30):
    """A small grey video (the randomize step never reads it; the handoff check and the plate step decode it)."""
    import imageio
    w = imageio.get_writer(str(path), fps=fps, macro_block_size=1)
    for i in range(frames):
        w.append_data(np.full((size[1], size[0], 3), 110 + i, np.uint8))
    w.close()


def write_common(task, extra_cfg):
    task.mkdir(parents=True, exist_ok=True)
    write_video(task / "human_video.mp4")
    cfg = {"scene": "scene.xml", "trajectory": "trajectory.npz", "human_video": "human_video.mp4",
           "num_samples": 2, "seed": 0, "cameras": {"ego": CAMERA}, "table": TABLE}
    cfg.update(extra_cfg)
    (task / "stage4.json").write_text(json.dumps(cfg, indent=1))
    return task / "stage4.json"


def stack_task(task, objects=None, segments=None, extra_xml="", checks=None):
    """Write the stack task into folder `task`; returns (config path, labels, phases)."""
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    (task / "scene.xml").write_text(STACK_XML.replace("{extra}", extra_xml))
    model = mujoco.MjModel.from_xml_path(str(task / "scene.xml"))
    Q, labels, phases = stack_trajectory(model)
    np.savez_compressed(task / "trajectory.npz", qpos=Q, fps=30.0, labels=np.array(labels))
    (task / "phases.json").write_text(json.dumps(phases, indent=1))
    if objects is None:
        objects = {"A": {"carrier": "right", "release": "right: let go of A #2", "rests_on": "B", "dx": [-0.01, 0.01], "dy": [-0.01, 0.01]},
                   "B": {"dx": [0.02, 0.04], "dy": [-0.02, 0.02], "dyaw_deg": [-10, 10]},
                   "C": {"carrier": "right", "release": "right: let go of C #4", "rests_on": "A", "dx": [-0.01, 0.01], "dy": [-0.01, 0.01]}}
    cfg = write_common(task, {"robot": {"end_effectors": {"right": "robot_right_right_base_yaw"}, "init_jitter_rad": 0.0},
                              "randomize": {"objects": objects}, "segments": stack_segments() if segments is None else segments,
                              "checks": {"physics_replay": False, **(checks or {})}})
    return cfg, labels, phases


def bench_task(task, attached=True, frames=60, base_dx=(0.045, 0.05), expect=None):
    """The bench with a 6x6x3 cm box at (0.15, 0.42) and a 2 cm cube on it; the robot holds its home pose.
    attached=True moves the cube with the box (attached_to); False leaves it where it was, over thin air once the
    box moved more than 4 cm. A manifest asks the cube to stay put and at rest over the whole clip."""
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    setup = REPO / "setup"
    txt = (setup / "scene.xml").read_text()
    txt = txt.replace('file="assets/', f'file="{(setup / "assets").as_posix()}/')
    bodies = ('<body name="box" pos="0.15 0.42 0.015"><freejoint name="box"/>'
              '<geom name="box" type="box" size="0.03 0.03 0.015" rgba="0.3 0.5 0.8 1" mass="0.2"/></body>'
              '<body name="cube" pos="0.15 0.42 0.0399"><freejoint name="cube"/>'
              '<geom name="cube" type="box" size="0.01 0.01 0.01" rgba="0.9 0.6 0.1 1" mass="0.02"/></body>')
    i = txt.rindex("</worldbody>")
    txt = txt[:i] + bodies + txt[i:]
    extra = " 0.15 0.42 0.015 1 0 0 0 0.15 0.42 0.0399 1 0 0 0"
    txt = re.sub(r'(<key name="home" qpos=")([^"]*)(")', lambda m: m.group(1) + m.group(2) + extra + m.group(3), txt)
    (task / "scene.xml").write_text(txt)
    model = mujoco.MjModel.from_xml_path(str(task / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    Q = np.tile(data.qpos.copy(), (frames, 1))
    labels = ["left: wait #1"] * frames
    np.savez_compressed(task / "trajectory.npz", qpos=Q, fps=30.0, labels=np.array(labels), reference=Q.copy())
    ex = expect or {"object": "cube", "at_rest": True, "untouched": ["cube"]}
    (task / "trajectory_manifest.json").write_text(json.dumps(
        {"subgoals": [{"index": 1, "motion": "hold", "side": "left", "object": "cube", "frames": [0, frames - 1], "expect": ex}]}))
    objs = {"box": {"dx": list(base_dx), "dy": [0.0, 0.0], "dyaw_deg": [0.0, 0.0]}}
    if attached:
        objs["cube"] = {"attached_to": "box"}
    cfg = write_common(task, {"robot": {"end_effectors": {"left": "robot_left_left_base_yaw", "right": "robot_right_right_base_yaw"},
                                        "init_jitter_rad": 0.0},
                              "randomize": {"objects": objs}, "segments": {},
                              "manifest": "trajectory_manifest.json", "checks": {"physics_replay": True}})
    return cfg


def rigid_bench_task(task, attached=True, frames=60, base_dx=(0.045, 0.05), expect=None, cube_z=0.0399):
    """bench_task on the rigid arm-hand runtime: the bench compiled with the hands fixed to the arm flanges, a 6x6x3 cm
    box at (0.15, 0.42) and a 2 cm cube at height cube_z (0.0399 rests on the box), and a complete servo recording of
    the robot holding its home pose (controls, velocities, integration state, scene hash), as Stage 3 hands it off.
    attached=True moves the cube with the box (attached_to); False leaves it where it was, over thin air once the box
    moved more than 4 cm. A manifest asks the cube to stay put and at rest over the whole clip, and Stage 1's goal
    (scene_semantics.json, goal_objects.json) is that the cube rests on top of the box."""
    import hashlib
    import sys
    import xml.etree.ElementTree as ET
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tool_library.common import episode as E
    from tool_library.common import rigid_robot as R
    from tool_library.skills.write_stage4_config import _copy_scene
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    root = ET.fromstring(R.compile_rigid(REPO / "setup" / "scene.xml").xml)
    wb = root.find("worldbody")
    for name, pos, size, rgba, mass in (("box", (0.15, 0.42, 0.015), (0.03, 0.03, 0.015), "0.3 0.5 0.8 1", "0.2"),
                                        ("cube", (0.15, 0.42, cube_z), (0.01, 0.01, 0.01), "0.9 0.6 0.1 1", "0.02")):
        b = ET.SubElement(wb, "body", name=name, pos=" ".join(map(str, pos)))
        ET.SubElement(b, "freejoint", name=name)
        ET.SubElement(b, "geom", name=name, type="box", size=" ".join(map(str, size)), rgba=rgba, mass=mass)
    for key in root.iter("key"):
        if key.get("qpos"):
            key.set("qpos", key.get("qpos") + f" 0.15 0.42 0.015 1 0 0 0 0.15 0.42 {cube_z} 1 0 0 0")
    (task / "raw.xml").write_text(ET.tostring(root, encoding="unicode"))
    _copy_scene(task / "raw.xml", task)                   # scene.xml with its assets bundled next to it
    (task / "raw.xml").unlink()
    spec = mujoco.MjSpec.from_file(str(task / "scene.xml"))
    model = spec.compile()
    R.register_model(model, spec)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    reference = np.tile(data.qpos.copy(), (frames, 1))
    result = E.servo_rollout(model, reference, 30, qpos_start=data.qpos, qvel_start=data.qvel)
    arrays = R.episode_arrays(model, result, reference)
    arrays.update(scene_sha256=np.array(hashlib.sha256((task / "scene.xml").read_bytes()).hexdigest()),
                  fps=np.array(30.0), labels=np.array(["left: wait #1"] * frames))
    np.savez_compressed(task / "trajectory.npz", **arrays)
    ex = expect or {"object": "cube", "at_rest": True, "untouched": ["cube"]}
    (task / "trajectory_manifest.json").write_text(json.dumps(
        {"subgoals": [{"index": 1, "motion": "hold", "side": "left", "object": "cube", "frames": [0, frames - 1], "expect": ex,
                       "stage1_indices": [1]}]}))
    (task / "scene_semantics.json").write_text(json.dumps({
        "objects": [{"id": "cube", "name": "orange cube", "type": "rigid"}, {"id": "box", "name": "blue box", "type": "rigid"}],
        "subgoals": [{"index": 1, "acting_hand": "left", "motion": "place", "object_refs": ["cube", "box"],
                      "goal": "The cube rests on top of the box."}],
        "task_success": "The cube rests on top of the box."}))
    (task / "goal_objects.json").write_text(json.dumps({"objects": [{"id": "cube", "root_body": "cube", "bodies": ["cube"]},
                                                                    {"id": "box", "root_body": "box", "bodies": ["box"]}]}))
    objs = {"box": {"dx": list(base_dx), "dy": [0.0, 0.0], "dyaw_deg": [0.0, 0.0]}}
    if attached:
        objs["cube"] = {"attached_to": "box"}
    return write_common(task, {"robot": {"end_effectors": {"left": "robot_left_left_base_yaw", "right": "robot_right_right_base_yaw"},
                                         "init_jitter_rad": 0.0},
                               "randomize": {"objects": objs}, "segments": {},
                               "manifest": "trajectory_manifest.json", "checks": {"physics_replay": True},
                               "goal": {"semantics": "scene_semantics.json", "objects": "goal_objects.json"},
                               "physics": {"mujoco_version": mujoco.__version__}})


RELAY_XML = """<mujoco model="relay">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    {hands}
    <body name="A" pos="-0.20 0.40 0.02"><freejoint name="A"/><geom name="A" type="box" size="0.02 0.02 0.02" rgba="1 0 0 1"/></body>
    <body name="B" pos="0.25 0.45 0.01"><freejoint name="B"/><geom name="B" type="box" size="0.05 0.05 0.01" rgba="0 1 0 1"/></body>
  </worldbody>
</mujoco>
"""
HAND_XML = """<body name="robot_{s}_{s}_base_yaw" pos="0 0 0">
      <joint name="{s}x" type="slide" axis="1 0 0" range="-2 2"/>
      <joint name="{s}y" type="slide" axis="0 1 0" range="-2 2"/>
      <joint name="{s}z" type="slide" axis="0 0 1" range="-2 2"/>
      <joint name="{s}rx" type="hinge" axis="1 0 0" range="-3.1 3.1"/>
      <joint name="{s}ry" type="hinge" axis="0 1 0" range="-3.1 3.1"/>
      <joint name="{s}rz" type="hinge" axis="0 0 1" range="-3.1 3.1"/>
      <geom name="{s}_palm" type="box" size="0.03 0.03 0.01" pos="0 0 0.012" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
    </body>"""


def relay_task(task, handover=False):
    """Two free hands pass box A along and put it on box B. The left hand picks A up and either sets it down at a fixed
    place in the middle of the table (handover=False: the right hand picks it up from there) or holds it up in the air
    there while the right hand takes it (handover=True: the left lets go once the right holds it). The right hand then
    stacks A on B. A has two carriers; its goal follows B. Returns (config path, labels)."""
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    (task / "scene.xml").write_text(RELAY_XML.replace("{hands}", "\n    ".join(HAND_XML.format(s=s) for s in ("left", "right"))))
    model = mujoco.MjModel.from_xml_path(str(task / "scene.xml"))
    qa = {n: _qadr(model, n) for n in ("A", "B")}
    hand = {"left": [0, 1, 2], "right": [6, 7, 8]}
    pos = {n: model.body(n).pos.copy() for n in ("A", "B")}
    hp = {"left": np.array([-0.25, 0.15, 0.25]), "right": np.array([0.25, 0.15, 0.25])}
    held = {"left": None, "right": None}
    Q, labels = [], []
    mid = np.array([0.0, 0.42, 0.02 + (0.10 if handover else 0.0)])

    def frame(label):
        q = np.zeros(model.nq)
        for s, ix in hand.items():
            q[ix] = hp[s]
        for n, p in pos.items():
            q[qa[n]:qa[n] + 3] = p
            q[qa[n] + 3:qa[n] + 7] = [1, 0, 0, 0]
        Q.append(q)
        labels.append(label)

    def move(side, label, target, n=20, other=None):
        """Move one hand (and what it holds) to target; `other` = (side, target) moves the second hand at once."""
        starts = {side: hp[side].copy()}
        goals = {side: np.asarray(target, float)}
        if other:
            starts[other[0]], goals[other[0]] = hp[other[0]].copy(), np.asarray(other[1], float)
        for k in range(1, n + 1):
            u = k / n
            for s in starts:
                hp[s] = (1 - u) * starts[s] + u * goals[s]
                if held[s]:
                    pos[held[s]] = hp[s] - np.array([0.0, 0.0, 0.02])
            frame(label)

    top = lambda p: np.asarray(p, float) + np.array([0.0, 0.0, 0.02])       # the palm sits on the box's top face
    frame("left: reach for A #1")
    move("left", "left: reach for A #1", top(pos["A"]) + [0, 0, 0.08])
    move("left", "left: descend on A #1", top(pos["A"]))
    move("left", "left: grip A #1", top(pos["A"]), n=10)
    held["left"] = "A"
    move("left", "left: lift A #1", top(pos["A"]) + [0, 0, 0.06])
    move("left", "left: carry A #2", top(mid) + [0, 0, 0.06], n=30)
    if not handover:
        move("left", "left: set down A #2", top(mid))
        held["left"] = None
        move("left", "left: let go of A #2", top(mid), n=10)
        move("left", "left: retreat #2", top(mid) + [0, 0, 0.08])
        move("left", "left: return home #2", [-0.25, 0.15, 0.25], n=20)
        grip_at = top(mid)
    else:
        move("left", "left: carry A (hold) #2", top(mid), n=10)
        grip_at = top(mid) + [0.0, 0.0, -0.04]                               # the right palm comes under the box
    move("right", "right: reach for A #3", grip_at + [0.06, 0, 0.0])
    move("right", "right: descend on A #3", grip_at)
    move("right", "right: grip A #3", grip_at, n=10)
    if handover:
        held["left"] = None
        held["right"] = "A"
        move("left", "left: let go of A #3", top(mid), n=10)
        move("left", "left: retreat #3", top(mid) + [-0.08, 0, 0.0])
        move("left", "left: return home #3", [-0.25, 0.15, 0.25], n=20)
    held["right"] = "A"
    off = top(pos["A"]) - pos["A"] if not handover else grip_at - pos["A"]
    goal_box = pos["B"] + np.array([0.0, 0.0, 0.01 + 0.02])                  # A's centre on B
    move("right", "right: lift A #4", hp["right"] + [0, 0, 0.06])
    move("right", "right: carry A #4", goal_box + off + [0, 0, 0.06], n=30)
    move("right", "right: set down A #4", goal_box + off)
    held["right"] = None
    move("right", "right: let go of A #4", goal_box + off, n=10)
    move("right", "right: retreat #4", goal_box + off + [0, 0, 0.08])
    move("right", "right: return home #4", [0.25, 0.15, 0.25], n=20)
    np.savez_compressed(task / "trajectory.npz", qpos=np.array(Q), fps=30.0, labels=np.array(labels))
    left_rel = "left: let go of A #3" if handover else "left: let go of A #2"
    seg_left = [{"phases": ["left: reach for A #1"], "blend": ["world", "A"]},
                {"phases": ["left: descend on A #1", "left: grip A #1", "left: lift A #1"], "anchor": "A"},
                {"phases": ["left: carry A #2"], "blend": ["A", "world"]}]
    held_phases = ["left: let go of A #3", "left: retreat #3", "left: return home #3"] if handover else []
    seg_right = [{"phases": ["right: reach for A #3"], "blend": ["world", "A"]},
                 {"phases": ["right: descend on A #3", "right: grip A #3"] + held_phases + ["right: lift A #4"], "anchor": "A"},
                 {"phases": ["right: carry A #4"], "blend": ["A", "B"]},
                 {"phases": ["right: set down A #4", "right: let go of A #4", "right: retreat #4"], "anchor": "B"},
                 {"phases": ["right: return home #4"], "blend": ["B", "world"]}]
    objects = {"A": {"dx": [-0.02, 0.02], "dy": [-0.02, 0.02], "dyaw_deg": [0, 0], "carrier": ["left", "right"],
                     "release": [left_rel, "right: let go of A #4"], "rests_on": "B"},
               "B": {"dx": [-0.03, 0.03], "dy": [-0.02, 0.02], "dyaw_deg": [-10, 10]}}
    cfg = write_common(task, {"robot": {"end_effectors": {"left": "robot_left_left_base_yaw", "right": "robot_right_right_base_yaw"},
                                        "init_jitter_rad": 0.0},
                              "randomize": {"objects": objects}, "segments": {"left": seg_left, "right": seg_right},
                              "checks": {"physics_replay": False}})
    return cfg, labels


def beside_task(task, next_to=True):
    """The stack scene's free hand puts box A down 5 mm to the right of box B (a world target right beside B), then
    picks B up and carries it away to a world place. A's goal is beside B where B was when A was put down: when B is
    moved by the layout A must follow it, and B's later carry must not drag A along. Returns (config path, labels,
    phases)."""
    task = Path(task)
    task.mkdir(parents=True, exist_ok=True)
    (task / "scene.xml").write_text(STACK_XML.replace("{extra}", ""))
    model = mujoco.MjModel.from_xml_path(str(task / "scene.xml"))
    pos = {n: model.body(n).pos.copy() for n in ("A", "B", "C")}
    qa = {n: _qadr(model, n) for n in pos}
    hp = np.array([0.0, 0.2, 0.25])
    held = [None]
    Q, labels, phases = [], [], []

    def frame(label):
        q = np.zeros(model.nq)
        q[:3] = hp
        for n, p in pos.items():
            q[qa[n]:qa[n] + 3] = p
            q[qa[n] + 3:qa[n] + 7] = [1, 0, 0, 0]
        Q.append(q)
        labels.append(label)

    def move(label, target, n=20, objects=(), anchor=None):
        nonlocal hp
        start, t0 = hp.copy(), len(Q)
        for k in range(1, n + 1):
            hp = (1 - k / n) * start + k / n * np.asarray(target, float)
            if held[0]:
                pos[held[0]] = hp - np.array([0.0, 0.0, HALF[held[0]]])
            frame(label)
        phases.append({"label": label, "hand": "right", "objects": list(objects), "anchor": anchor, "frames": [[t0, len(Q) - 1]]})

    def pick(obj, sg):
        top = pos[obj] + np.array([0.0, 0.0, HALF[obj]])
        move(f"right: reach for {obj} #{sg}", top + [0, 0, 0.08], objects=[obj], anchor=obj)
        move(f"right: descend on {obj} #{sg}", top, objects=[obj], anchor=obj)
        move(f"right: grip {obj} #{sg}", top, n=10, objects=[obj], anchor=obj)
        held[0] = obj
        move(f"right: lift {obj} #{sg}", top + [0, 0, 0.06], objects=[obj], anchor=obj)

    def place(obj, xy, sg, anchor):
        z = HALF[obj] * 2 - 0.0005
        move(f"right: carry {obj} #{sg}", [xy[0], xy[1], z + 0.06], n=30, objects=[obj], anchor=anchor)
        move(f"right: set down {obj} #{sg}", [xy[0], xy[1], z], objects=[obj], anchor=anchor)
        held[0] = None
        move(f"right: let go of {obj} #{sg}", [xy[0], xy[1], z], n=10, objects=[obj], anchor=anchor)
        move(f"right: retreat #{sg}", [xy[0], xy[1], z + 0.08], objects=[obj], anchor=anchor)

    frame("right: reach for A #1")
    pick("A", 1)
    beside = pos["B"][:2] + np.array([0.05 + 0.02 + 0.005, 0.0])             # A's right next to B, 5 mm apart
    place("A", beside, 2, None)
    pick("B", 3)
    place("B", [0.30, 0.65], 4, None)
    move("right: return home #5", [0.0, 0.2, 0.25], n=30)
    phases[0]["frames"] = [[0, phases[0]["frames"][0][1]]]
    np.savez_compressed(task / "trajectory.npz", qpos=np.array(Q), fps=30.0, labels=np.array(labels))
    (task / "phases.json").write_text(json.dumps(phases, indent=1))
    segs = {"right": [
        {"phases": ["right: reach for A #1"], "blend": ["world", "A"]},
        {"phases": ["right: descend on A #1", "right: grip A #1", "right: lift A #1"], "anchor": "A"},
        {"phases": ["right: carry A #2"], "blend": ["A", "B"]},
        {"phases": ["right: set down A #2", "right: let go of A #2", "right: retreat #2", "right: reach for B #3",
                    "right: descend on B #3", "right: grip B #3", "right: lift B #3"], "anchor": "B"},
        {"phases": ["right: carry B #4"], "blend": ["B", "world"]}]}
    objects = {"A": {"dx": [-0.01, 0.01], "dy": [-0.01, 0.01], "carrier": "right", "release": "right: let go of A #2"},
               "B": {"dx": [0.02, 0.04], "dy": [-0.02, 0.02], "dyaw_deg": [-10, 10], "carrier": "right", "release": "right: let go of B #4"}}
    if next_to:
        objects["A"]["next_to"] = "B"
    cfg = write_common(task, {"robot": {"end_effectors": {"right": "robot_right_right_base_yaw"}, "init_jitter_rad": 0.0},
                              "randomize": {"objects": objects}, "segments": segs, "checks": {"physics_replay": False}})
    return cfg, labels, phases
