"""Contact and arm screens must use the same recorded articulated-object state."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from tool_library.common import episode as E, hand as H, plan as P
from tool_library.skills.hand_scene_from_recon import hand_scene_from_recon, resolve_bench_state
from tool_library.skills.hand_scene_object_state import hand_scene_object_state
from tool_library.skills.generate_flat_pinches import generate_flat_pinches
from tool_library.skills.generate_rigid_grasps import generate_rigid_grasps
from tool_library.skills.optimize_grasp_pose import optimize_grasp_pose
from tool_library.skills.stage_on_bench import stage_on_bench


class GraspStateProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ws = Path(cls.tmp.name)
        cls.asset = cls.ws / "cable.xml"
        pieces = []
        for i in range(7):
            joint = '<freejoint/>' if i == 0 else f'<joint name="bend{i}" type="ball" damping=".02"/>'
            pieces.append(f'<body name="link{i}" pos="{0 if i == 0 else .03} 0 {0.008 if i == 0 else 0}">'
                          f'{joint}<geom name="segment{i}" type="capsule" fromto="0 0 0 .03 0 0" '
                          'size=".008" mass=".01"/>')
        cls.asset.write_text('<mujoco><worldbody>' + ''.join(pieces) + '</body>' * 7 + '</worldbody></mujoco>')
        cls.scene = cls.ws / "scene.xml"
        stage_on_bench(str(ROOT / "setup/scene.xml"),
                       assets=[{"name": "cable", "xml": str(cls.asset), "x": -.09, "y": .4}],
                       out_xml=str(cls.scene), rig={"origin": [0, 0, 0], "R": np.eye(3).tolist()})
        cls.manifest = cls.ws / "scene_recon.json"
        cls.manifest.write_text(json.dumps({"scene_xml": "scene.xml", "objects": [
            {"id": "cable", "type": "deformable", "asset_xml": "cable.xml", "root_body": "cable_link0",
             "bodies": [f"cable_link{i}" for i in range(7)]}]}))
        cls.model = E.build_execution_model(str(cls.scene))[0]
        d = mujoco.MjData(cls.model)
        mujoco.mj_resetDataKeyframe(cls.model, d, 0)
        cls.initial = d.qpos.copy()
        cls.bent = cls.initial.copy()
        adr = int(cls.model.jnt_qposadr[cls.model.joint("cable_bend3").id])
        cls.bent[adr:adr+4] = [np.cos(.35), 0, 0, np.sin(.35)]
        cls.hs = hand_scene_from_recon(str(cls.manifest), "left", "cable", grasp_body="cable_link6",
                                      state=cls.bent.tolist())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_saved_state_keeps_bent_cable_in_both_models(self):
        q = resolve_bench_state(self.hs, self.model)
        np.testing.assert_array_equal(q, self.bent)
        bench = mujoco.MjData(self.model)
        bench.qpos[:] = q
        mujoco.mj_forward(self.model, bench)
        hand_model, hand, _, _ = H.open_scene(self.hs, settle_s=0)
        for i in range(7):
            name = f"cable_link{i}"
            np.testing.assert_allclose(hand.body(name).xpos, bench.body(name).xpos, atol=1e-6)
            np.testing.assert_allclose(hand.body(name).xmat, bench.body(name).xmat, atol=1e-6)
        initial = mujoco.MjData(self.model)
        initial.qpos[:] = self.initial
        mujoco.mj_forward(self.model, initial)
        self.assertGreater(np.linalg.norm(bench.body("cable_link6").xpos - initial.body("cable_link6").xpos), .05)

    def test_conflicting_object_state_rejected_but_robot_update_allowed(self):
        with self.assertRaisesRegex(ValueError, "different object states"):
            resolve_bench_state(self.hs, self.model, self.initial)
        q = self.bent.copy()
        q[E.robot_parts(self.model)["left"]["arm_qadr"][0]] += .05
        np.testing.assert_array_equal(resolve_bench_state(self.hs, self.model, q), q)

    def test_public_grasp_searches_reject_mixed_states(self):
        for tool in (generate_flat_pinches, generate_rigid_grasps, optimize_grasp_pose):
            with self.subTest(tool=tool.__name__):
                with self.assertRaisesRegex(ValueError, "different object states"):
                    tool(self.hs, bench_scene=str(self.scene), state=self.initial.tolist())

    def test_edited_object_state_cannot_retain_old_provenance(self):
        hs = json.loads(json.dumps(self.hs))
        hs["joint_init"]["cable_bend3"] = [1, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "joint_init changed"):
            resolve_bench_state(hs, self.model)

    def test_object_state_updates_named_recording_and_provenance(self):
        q = self.bent.copy()
        adr = int(self.model.jnt_qposadr[self.model.joint("cable_bend4").id])
        q[adr:adr + 4] = [np.cos(.2), 0, np.sin(.2), 0]
        recording = self.ws / "reordered.npz"
        np.savez(recording, qpos=q[None, ::-1], joint_names=E.joint_names(self.model)[::-1])
        updated = hand_scene_object_state(self.hs, str(self.scene), str(recording), ["cable_"])["hand_scene"]
        np.testing.assert_array_equal(resolve_bench_state(updated, self.model), q)
        bench = mujoco.MjData(self.model)
        bench.qpos[:] = q
        mujoco.mj_forward(self.model, bench)
        hm, hd, _, _ = H.open_scene(updated, settle_s=0)
        np.testing.assert_allclose(hd.body("cable_link6").xpos, bench.body("cable_link6").xpos, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "outside the copied prefixes"):
            hand_scene_object_state(self.hs, str(self.scene), str(recording), ["cable_bend1"])

    def test_coordinate_order_and_quaternion_sign_do_not_change_state(self):
        hs = json.loads(json.dumps(self.hs))
        saved = hs["bench_state"]
        saved["joint_names"].reverse()
        saved["qpos"].reverse()
        np.testing.assert_array_equal(resolve_bench_state(hs, self.model), self.bent)
        same = self.bent.copy()
        adr = int(self.model.jnt_qposadr[self.model.joint("cable_bend3").id])
        same[adr:adr+4] *= -1
        np.testing.assert_array_equal(resolve_bench_state(hs, self.model, same), same)

    def test_kinematic_cable_carry_is_not_a_deformation_prediction(self):
        planner = P.Planner(self.model, self.initial)
        before = np.array([planner.data.body(f"cable_link{i}").xpos.copy() for i in range(7)])
        planner.attach("right", "cable_link1")
        pos, rotation = planner.wrist("right")
        displacement = np.array([.08, .04, .05])
        planner.move_wrist("right", pos + displacement, rotation)
        after = np.array([planner.data.body(f"cable_link{i}").xpos.copy() for i in range(7)])
        # This documents why a downstream grasp needs an actual physical
        # prefix: even the distant free end is moved rigidly by the proposal.
        np.testing.assert_allclose(after - before, np.tile(displacement, (7, 1)), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
