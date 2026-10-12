"""Negative acceptance cases for plural inputs and actual-state combined holds."""
import copy
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common.evidence import file_digest, record_tool_evidence, tool_inputs, value_digest
from tool_library.verifiers.trajectory_manifest_check import _verify_stress_evidence, _combined_objects_from_execution
from tool_library.verifiers.combined_hold_state_test import _recorded_state, _held_groups

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "tool_library"


class CombinedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dexagent_combined_evidence_")
        self.ws = Path(self.tmp.name)
        self.mod = importlib.import_module("tool_library.verifiers.combined_hold_state_test")

    def tearDown(self):
        self.tmp.cleanup()

    def test_plural_snapshot_resolves_files_and_does_not_alias_values(self):
        p1 = {"pos": [0, 0, .1], "grip_fingers": ["thumb", "index"]}
        p2 = {"pos": [.1, 0, .1], "grip_fingers": ["middle", "ring", "pinky"]}
        (self.ws / "first.json").write_text(json.dumps(p1))
        args = {"poses": ["first.json", p2], "objects": ["a", "b"], "scene": {"side": "right"}}
        before = tool_inputs(args, self.ws)
        self.assertEqual(before["poses"]["value"], [p1, p2])
        self.assertEqual(before["poses"]["sha256"], value_digest([p1, p2]))
        p2["pos"][0] = .2
        self.assertEqual(before["poses"]["value"][1]["pos"][0], .1)
        self.assertNotEqual(before, tool_inputs(args, self.ws))
        with self.assertRaisesRegex(ValueError, "mutated a bound input"):
            record_tool_evidence(LIB, self.mod.TOOL, args, {"ok": True}, self.ws, before)

    def test_pose_file_edit_and_order_changes_cannot_keep_plural_evidence(self):
        (self.ws / "p.json").write_text('{"pos":[0,0,0]}')
        args = {"poses": ["p.json", {"pos": [1, 0, 0]}], "objects": ["a", "b"]}
        before = tool_inputs(args, self.ws)
        for key in ("poses", "objects"):
            changed = copy.deepcopy(args)
            changed[key].reverse()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mutated"):
                record_tool_evidence(LIB, self.mod.TOOL, changed, {"ok": True}, self.ws, before)
        (self.ws / "p.json").write_text('{"pos":[0,0,1]}')
        with self.assertRaisesRegex(ValueError, "mutated"):
            record_tool_evidence(LIB, self.mod.TOOL, args, {"ok": True}, self.ws, before)

    def _bound(self):
        (self.ws / "scene.xml").write_text("<mujoco/>")
        (self.ws / "trajectory.npz").write_bytes(b"identity-only orchestration fixture")
        first = {"index": 1, "frames": [0, 20], "motion": "grasp_lift", "side": "right", "object": "a", "ok": True}
        more = {"index": 2, "frames": [21, 40], "motion": "grasp_more", "side": "right", "object": "b", "ok": True}
        execution = {"ok": True, "subgoals": [first, more], "scene_sha256": file_digest(self.ws / "scene.xml"),
                     "trajectory_sha256": file_digest(self.ws / "trajectory.npz")}
        context = {"execution": execution, "selected": more, "scene": self.ws / "scene.xml", "trajectory": self.ws / "trajectory.npz"}
        args = {"scene": str(context["scene"]), "trajectory": str(context["trajectory"]), "frame": 40,
                "side": "right", "objects": ["a", "b"]}
        return more, context, args

    def _verify(self, report, entry, context):
        return _verify_stress_evidence("combined_hold_state_test", report, {}, entry, None, self.ws, LIB,
            trusted_verifiers={"combined_hold_state_test": file_digest(LIB / "verifiers/combined_hold_state_test.py")},
            execution_context=context)

    def test_forged_success_replays_from_exact_recorded_combined_state(self):
        entry, context, args = self._bound()
        report = record_tool_evidence(LIB, self.mod.TOOL, args, {"ok": True}, self.ws)
        with patch.object(self.mod, "combined_hold_state_test", return_value={"ok": False, "issues": ["object a slips"]}) as fn:
            with self.assertRaisesRegex(ValueError, "independent actual-state.*failed"):
                self._verify(report, entry, context)
            fn.assert_called_once_with(**args)

    def test_positive_state_binding_and_wrong_frame_or_omitted_grip_rejected(self):
        entry, context, args = self._bound()
        report = record_tool_evidence(LIB, self.mod.TOOL, args, {"ok": True}, self.ws)
        with patch.object(self.mod, "combined_hold_state_test", return_value={"ok": True}) as fn:
            self._verify(report, entry, context)
            fn.assert_called_once_with(**args)
        for key, value in (("objects", ["b"]), ("objects", ["b", "a"]), ("frame", 20), ("side", "left")):
            bad = record_tool_evidence(LIB, self.mod.TOOL, {**args, key: value}, {"ok": True}, self.ws)
            with self.subTest(key=key, value=value), patch.object(self.mod, "combined_hold_state_test") as fn:
                with self.assertRaisesRegex(ValueError, f"different {key}"):
                    self._verify(bad, entry, context)
                fn.assert_not_called()
        (self.ws / "trajectory.npz").write_bytes(b"changed physical recording")
        with self.assertRaisesRegex(ValueError, "trajectory changed"):
            self._verify(report, entry, context)

    def test_single_object_and_original_scene_regrasp_cannot_certify_grasp_more(self):
        entry, context, args = self._bound()
        for name in ("grasp_hold_test", "multi_object_hold_test"):
            mod = importlib.import_module("tool_library.verifiers." + name)
            report = record_tool_evidence(LIB, mod.TOOL, args, {"ok": True}, self.ws)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "requires combined_hold_state_test"):
                _verify_stress_evidence(name, report, {}, entry, None, self.ws, LIB, execution_context=context)

    def test_ambiguous_retention_and_missing_execution_fail_closed(self):
        entry, context, _ = self._bound()
        self.assertEqual(_combined_objects_from_execution(context, "right"), ["a", "b"])
        with self.assertRaisesRegex(ValueError, "bound successful"):
            _combined_objects_from_execution(None, "right")
        context["execution"]["subgoals"][0]["motion"] = "both"
        with self.assertRaisesRegex(ValueError, "simultaneous"):
            _combined_objects_from_execution(context, "right")

    def test_artificial_attachment_and_gravity_compensation_are_not_grasps(self):
        bodies = '<body name="a"><freejoint/><geom size=".01" mass=".1"/></body><body name="b" pos=".1 0 0"><freejoint/><geom size=".01" mass=".1"/></body>'
        xml = '<mujoco><worldbody>' + bodies + '</worldbody>{}</mujoco>'
        plain = mujoco.MjModel.from_xml_string(xml.format(''))
        groups, _ = _held_groups(plain, ['a', 'b'])
        self.assertEqual(set(groups), {'a', 'b'})
        glued = mujoco.MjModel.from_xml_string(xml.format('<equality><weld body1="a" body2="b"/></equality>'))
        with self.assertRaisesRegex(ValueError, "non-contact constraint"):
            _held_groups(glued, ['a', 'b'])
        plain.body_gravcomp[plain.body('a').id] = 1
        with self.assertRaisesRegex(ValueError, "gravity compensation"):
            _held_groups(plain, ['a', 'b'])


class RecordedStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tool_library.common import rigid_robot as RR
        cls.tmp = tempfile.TemporaryDirectory(prefix="dexagent_combined_recording_")
        cls.ws = Path(cls.tmp.name)
        runtime = RR.compile_rigid(str(ROOT / "setup/scene.xml"))
        cls.scene = cls.ws / "scene.xml"
        cls.scene.write_text(runtime.xml)
        m = runtime.model
        reference = np.repeat(m.key_qpos[0:1], 6, axis=0)
        roll = RR.servo_rollout(m, reference, 30.)
        cls.arrays = RR.episode_arrays(m, roll, reference)
        cls.path = cls.ws / "trajectory.npz"
        np.savez_compressed(cls.path, **cls.arrays)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_control_replay_restores_exact_hidden_state(self):
        model = mujoco.MjModel.from_xml_path(str(self.scene))
        data, report = _recorded_state(model, self.scene, self.path, 5)
        self.assertEqual(report["max_qpos_error"], 0.)
        self.assertEqual(report["max_qvel_error"], 0.)
        state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.testing.assert_array_equal(state, self.arrays["final_state"])

    def test_changed_controls_and_edited_frame_fail_independent_replay(self):
        for key in ("substep_ctrl", "qpos"):
            arrays = {k: v.copy() for k, v in self.arrays.items()}
            arrays[key][-1, 0] += .01
            changed = self.ws / (key + ".npz")
            np.savez_compressed(changed, **arrays)
            model = mujoco.MjModel.from_xml_path(str(self.scene))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "replay|selected frame"):
                _recorded_state(model, self.scene, changed, 5)


if __name__ == "__main__":
    unittest.main()
