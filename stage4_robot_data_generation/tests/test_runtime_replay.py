"""Actual rigid-arm dynamics and portable handoff, without a GPU or a task solver."""
import hashlib
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage4_robot_data_generation"))
from tool_library.common import episode as E, rigid_robot as R
from tool_library.common.stage4 import load_runtime_pair, copy_runtime_scene
from tool_library.skills.write_stage4_config import _copy_scene, write_stage4_config
from tool_library.verifiers.physics_recording_check import physics_recording_check
from stage4.config import load_config
from stage4 import physics, randomize


class RuntimeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="stage4_runtime_")      # $TMPDIR, else the system default
        cls.path = Path(cls.tmp.name)
        adapter = R.compile_rigid(ROOT / "setup/scene.xml")
        cls.scene = cls.path / "scene.xml"
        raw = cls.path / "raw.xml"
        root = ET.fromstring(adapter.xml)
        fixed = ET.SubElement(root.find("worldbody"), "body", name="fixture", pos="1.5 1.5 .1")
        ET.SubElement(fixed, "geom", type="box", size=".02 .02 .02", mass=".1")
        raw.write_text(ET.tostring(root, encoding="unicode"))
        _copy_scene(raw, cls.path)
        spec = mujoco.MjSpec.from_file(str(cls.scene)); m = spec.compile(); R.register_model(m, spec)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        reference = np.tile(d.qpos, (7, 1))
        result = E.servo_rollout(m, reference, 30, qpos_start=d.qpos, qvel_start=d.qvel)
        cls.arrays = R.episode_arrays(m, result, reference)
        cls.arrays.update(scene_sha256=np.array(hashlib.sha256(cls.scene.read_bytes()).hexdigest()),
                          fps=np.array(30.), labels=np.array(["left: hold #1"] * 3 + ["left: hold #2"] * 4))
        cls.traj = cls.path / "trajectory.npz"
        np.savez_compressed(cls.traj, **cls.arrays)
        (cls.path / "unused.mp4").write_bytes(b"unit fixture: replay does not decode the human video")
        (cls.path / "manifest.json").write_text(json.dumps({"subgoals": [
            {"index": 1, "frames": [0, 2], "motion": "hold", "expect": {"hand": "left"}},
            {"index": 2, "frames": [3, 6], "motion": "hold", "expect": {"hand": "left"}}]}))
        (cls.path / "stage4.json").write_text(json.dumps({
            "scene": "scene.xml", "trajectory": "trajectory.npz", "human_video": "unused.mp4",
            "cameras": {"ego": {"pos": [0, 1, 1], "quat_wxyz": [1, 0, 0, 0], "fovy_deg": 45}},
            "table": {"body": "table", "top_z": 0, "xy": [-1, 1, -1, 1]},
            "manifest": "manifest.json", "robot": {"end_effectors": {
                s: f"robot_{s}_{s}_base_yaw" for s in ("left", "right")}},
            "randomize": {"objects": {"fixture": {"dx": [0, .01]}}}, "physics": {"mujoco_version": mujoco.__version__}}))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runtime_recording_and_scene_identity(self):
        _, _, m, arrays = load_runtime_pair(self.scene, self.traj)
        self.assertTrue(R.is_rigid(m))
        self.assertEqual(arrays["ctrl"].shape, (7, m.nu))
        changed = self.path / "changed.xml"
        changed.write_text(self.scene.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "scene hash"):
            load_runtime_pair(changed, self.traj)

    def test_anonymous_object_joint_names_match_physics_recording(self):
        from types import SimpleNamespace
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="a"><freejoint/><geom size=".02" mass=".1"/></body>
          <body name="b"><freejoint/><geom size=".03" mass=".2"/></body>
          </worldbody></mujoco>''')
        names = randomize.Episode.joint_names(SimpleNamespace(model=model))
        self.assertEqual(names, E.joint_names(model))
        self.assertEqual(len(set(names)), model.nq)

    def test_assets_are_portable_and_model_unchanged(self):
        bundle = self.path / "bundle"
        count = _copy_scene(self.scene, bundle)
        self.assertGreater(count, 0)
        self.assertNotIn(str(ROOT), (bundle / "scene.xml").read_text())
        before = mujoco.MjModel.from_xml_path(str(self.scene))
        after = mujoco.MjModel.from_xml_path(str(bundle / "scene.xml"))
        for key in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia",
                    "jnt_range", "actuator_ctrlrange", "actuator_gainprm", "actuator_biasprm"):
            np.testing.assert_allclose(getattr(after, key), getattr(before, key), rtol=0, atol=1e-10)

    def test_handoff_preserves_actual_runtime_arrays(self):
        episode = self.path / "episode"
        episode.mkdir(exist_ok=True)
        copy_runtime_scene(self.scene, episode / "scene_physics.xml")
        np.savez_compressed(episode / "trajectory.npz", **self.arrays)
        (episode / "phases.json").write_text(json.dumps([
            {"label": "left: hold #1", "hand": "left", "frames": [[0, 2]], "anchor": None},
            {"label": "left: hold #2", "hand": "left", "frames": [[3, 6]], "anchor": None}]))
        (self.path / "camera.json").write_text(json.dumps({"pos": [0, 1, 1], "quat_wxyz": [1, 0, 0, 0], "fovy_deg": 45}))
        recon = self.path / "scene_recon.json"
        recon.write_text(json.dumps({"objects": [], "camera_json": "camera.json"}))
        output = self.path / "handoff"
        write_stage4_config(str(recon), str(episode), str(output), str(self.path / "unused.mp4"),
                            manifest=str(self.path / "manifest.json"))
        _, _, model, copied = load_runtime_pair(output / "scene.xml", output / "trajectory.npz")
        self.assertTrue(R.is_rigid(model))
        for key in self.arrays:
            if key != "scene_sha256":
                np.testing.assert_array_equal(copied[key], self.arrays[key], err_msg=key)

    def test_segment_replay_has_no_duplicate_times_or_state_resets(self):
        cfg = load_config(self.path / "stage4.json")
        ep = randomize.Episode(cfg)
        replay = physics.Replay(ep, cfg, log=lambda *_: None)
        _, actual, _, unstable, stopped = replay.replay(ep.Q, replay.Qref)
        self.assertIsNone(unstable)
        self.assertIsNone(stopped)
        record = dict(replay.last_execution, joint_names=np.array(ep.joint_names()))
        dest = self.path / "replay.npz"
        np.savez_compressed(dest, **record)
        report = physics_recording_check(str(self.scene), str(dest))
        self.assertTrue(report["ok"], report)
        self.assertEqual(actual.shape, ep.Q.shape)
        self.assertTrue(np.all(np.diff(record["time_s"]) > 0))
        # A single servo segment is the same physical evolution as partitioning
        # at frame 2, including the integration state carried across the boundary.
        _, one, _, _, _ = replay.replay(ep.Q, replay.Qref, segments=[(0, ep.T - 1, {})])
        np.testing.assert_allclose(actual, one, rtol=0, atol=2e-5)

    def test_default_and_fixed_body_scenes_match_audit_and_saved_recording(self):
        from stage4 import export
        cfg = load_config(self.path / "stage4.json")
        ep = randomize.Episode(cfg)
        replay = physics.Replay(ep, cfg, log=lambda *_: None)
        for index, overrides in enumerate(({}, {"fixture": {"pos": [1.52, 1.5, .1], "quat_wxyz": [1, 0, 0, 0]}})):
            model, actual, _, unstable, _ = replay.replay(ep.Q, replay.Qref, overrides)
            self.assertIsNone(unstable)
            audits = replay._physical_audits(overrides)
            self.assertIn("physical arm speed and mount", audits)
            self.assertTrue(all(v["ok"] for v in audits.values()), audits)
            output = self.path / f"sample_{index}"
            randomize.save_sample(ep, output, ep.Q, {}, overrides,
                                  {"ok": True, "physics": {"ok": True, "physical_audits": audits}},
                                  index, replay=actual, execution=replay.last_execution)
            _, _, published, arrays = load_runtime_pair(output / "scene_physics.xml", output / "trajectory.npz")
            np.testing.assert_array_equal(published.body_pos, model.body_pos)
            self.assertEqual(str(arrays["scene_sha256"]), hashlib.sha256((output / "scene_physics.xml").read_bytes()).hexdigest())
            export.validate_sample(output, json.loads((output / "sample.json").read_text()), self.scene)
            if not overrides:
                self.assertEqual((output / "scene_physics.xml").read_bytes(), self.scene.read_bytes())

    def test_mechanism_release_survives_segment_boundary(self):
        from tool_library.common.stage4 import physics_rules
        directory = self.path / "mechanism"
        _copy_scene(self.scene, directory)
        scene = directory / "scene.xml"
        root = ET.fromstring(scene.read_bytes())
        body = ET.SubElement(root.find("worldbody"), "body", name="cap", pos="1.5 1.5 .3")
        ET.SubElement(body, "joint", name="cap_lift", type="slide", axis="0 0 1")
        ET.SubElement(body, "geom", type="sphere", size=".01", mass=".1")
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(equality, "joint", name="engaged", joint1="cap_lift", polycoef="0 0 0 0 0")
        for key in root.findall("./keyframe/key"):
            if key.get("qpos"):
                key.set("qpos", key.get("qpos") + " .02")
        scene.write_text(ET.tostring(root, encoding="unicode"))
        spec = mujoco.MjSpec.from_file(str(scene)); model = spec.compile(); R.register_model(model, spec)
        data = mujoco.MjData(model); mujoco.mj_resetDataKeyframe(model, data, 0)
        recon = directory / "mechanism_scene_recon.json"
        recon.write_text(json.dumps({"objects": [{"equalities": {"engaged": "engaged"},
                           "joints": {"lift": "cap_lift"}, "params": {"release_travel_m": .01}}]}))
        rules = physics_rules(recon)
        reference = np.tile(data.qpos, (7, 1))
        result = E.servo_rollout(model, reference, 30, qpos_start=data.qpos, qvel_start=data.qvel, rules=rules)
        arrays = R.episode_arrays(model, result, reference)
        arrays.update(scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest(), fps=30., labels=self.arrays["labels"])
        np.savez_compressed(directory / "trajectory.npz", **arrays)
        config = json.loads((self.path / "stage4.json").read_text())
        config.update(human_video=str(self.path / "unused.mp4"), manifest=str(self.path / "manifest.json"),
                      physics={"rules": rules, "scene_recon_json": str(recon),
                               "scene_recon_sha256": hashlib.sha256(recon.read_bytes()).hexdigest()})
        (directory / "stage4.json").write_text(json.dumps(config))
        cfg = load_config(directory / "stage4.json")
        ep = randomize.Episode(cfg); replay = physics.Replay(ep, cfg, log=lambda *_: None)
        replay.replay(ep.Q, replay.Qref)
        execution = replay.last_execution
        self.assertEqual(execution["eq_active"][:, 0].tolist(), [1, 0, 0, 0, 0, 0, 0])
        np.savez_compressed(directory / "replayed.npz", **{k: v for k, v in execution.items() if not k.startswith("_")},
                            joint_names=np.array(ep.joint_names()))
        report = physics_recording_check(str(scene), str(directory / "replayed.npz"), str(recon))
        self.assertTrue(report["ok"], report)
        self.assertIn("engaged", json.loads(str(execution["mechanism_final"]))["inactive"])


if __name__ == "__main__":
    unittest.main()
