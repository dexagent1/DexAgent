"""Final task success cannot be undone after individually successful subgoals."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
from tool_library.common import task_gates as G


class TaskSuccessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "scene.xml"
        path.write_text('<mujoco><compiler angle="radian"/><worldbody><body name="table"><geom type="box" size="1 1 .02" pos="0 0 -.02"/></body>'
                        '<body name="panel" pos="0 0 .2"><joint name="panel_hinge" range="0 1" limited="true"/>'
                        '<geom type="box" size=".1 .02 .1" mass=".1"/></body></worldbody></mujoco>')
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.sem = {"objects": [{"id": "panel", "name": "panel", "type": "articulated"}],
                    "subgoals": [{"index": 1, "motion": "open_joint", "object_refs": ["panel"], "goal": "The panel is open.", "acting_hand": "left"},
                                 {"index": 2, "motion": "close_joint", "object_refs": ["panel"], "goal": "The panel is closed.", "acting_hand": "left"}],
                    "task_success": "The panel is closed."}
        self.recon = {"objects": [{"id": "panel", "root_body": "panel", "joints": {"hinge": "panel_hinge"}}]}
        self.man = {"subgoals": [{"index": 1, "stage1_indices": [1], "side": "left", "object": "panel", "motion": "open_joint", "frames": [0, 19]},
                                 {"index": 2, "stage1_indices": [2], "side": "left", "object": "panel", "motion": "close_joint", "frames": [20, 39]}]}

    def tearDown(self):
        self.tmp.cleanup()

    def episode(self, undone=False):
        # Use the actual compiled joint range for both states.
        limit = float(self.model.jnt_range[0, 1])
        q = np.zeros((60, 1)); q[:20, 0] = limit
        if undone:
            q[40:, 0] = limit
        return G.Episode(self.model, qpos=q, fps=30., labels=["left: act"] * len(q))

    def report(self, sem=None, man=None, undone=False):
        return G.stage1_goal_report(self.episode(undone), sem or self.sem, man or self.man, self.recon)

    def test_successful_intermediate_steps_do_not_certify_undone_final_state(self):
        good = self.report()
        self.assertTrue(good["ok"], good)
        # Opening is transient and must not be re-enforced at episode end.
        self.assertTrue(all(r["ok"] for r in good["subgoals"]))
        bad = self.report(undone=True)
        self.assertTrue(all(r["ok"] for r in bad["subgoals"]), bad)
        self.assertFalse(bad["ok"])
        self.assertFalse(bad["task_success"]["ok"])
        self.assertTrue(any(p["measured"].get("frame", 59) == 59 for p in bad["task_success"]["predicates"]))

    def test_unknown_partial_or_missing_final_semantics_fail_closed(self):
        for text in (None, "", "The panel is closed and emits a melody.", "The panel signifies completion.", "The panel is closed and open."):
            sem = copy.deepcopy(self.sem); sem["task_success"] = text
            result = self.report(sem)
            self.assertFalse(result["ok"], (text, result))
            self.assertTrue(all(r["ok"] for r in result["subgoals"]), result)

    def test_declarations_never_waive_unknown_language(self):
        sem = copy.deepcopy(self.sem); sem["task_success"] = "The panel is ready."
        predicate = {"class": "joint_target", "bodies": {"A": "panel"}, "params": {"joint": "panel_hinge", "state": "closed"}}
        man = copy.deepcopy(self.man); man["task_success_predicates"] = [predicate]
        self.assertFalse(self.report(sem, man)["ok"])
        sem["task_success_predicates"] = [predicate]
        self.assertFalse(self.report(sem, man)["ok"])
        self.assertFalse(self.report(sem, man, undone=True)["ok"])
        sem["task_success_predicates"][0]["frames"] = [20, 39]
        self.assertFalse(self.report(sem, man, undone=True)["ok"])

    def test_reviewed_predicates_cannot_replace_inferred_final_relation(self):
        sem = copy.deepcopy(self.sem)
        sem["task_success_predicates"] = [{"class": "joint_target", "bodies": {"A": "panel"},
                                            "params": {"joint": "panel_hinge", "state": "open"}}]
        bad = self.report(sem, undone=True)
        self.assertFalse(bad["ok"])
        self.assertEqual([p["ok"] for p in bad["task_success"]["predicates"]], [False, True])

    def test_has_not_moved_cannot_move_and_return_to_start(self):
        sem = copy.deepcopy(self.sem); sem["task_success"] = "The panel has not moved."
        q = np.zeros((60, 1))
        for excursion in (False, True):
            if excursion:
                q[20:40, 0] = self.model.jnt_range[0, 1]
            ep = G.Episode(self.model, qpos=q, fps=30., labels=["left: act"] * len(q))
            report = G.task_success_report(G.GoalContext(ep, sem, self.recon), sem)
            self.assertEqual(report["ok"], not excursion, report)

    def test_release_must_remain_supported_at_actual_episode_end(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="table" pos="0 0 -.02">'
                    '<geom type="box" size="1 1 .02"/></body><body name="block" pos="0 0 .02"><freejoint/>'
                    '<geom type="box" size=".02 .02 .02" mass=".1"/></body></worldbody></mujoco>')
        sem = {"objects": [{"id": "block", "name": "block"}], "task_success": "The block rests on the table."}
        recon = {"objects": [{"id": "block", "root_body": "block"}]}
        q = np.repeat(model.qpos0[None], 60, axis=0)
        for falling in (False, True):
            if falling:
                q[-1, 2] += .2
            ep = G.Episode(model, qpos=q, fps=30., labels=["left: act"] * len(q))
            report = G.task_success_report(G.GoalContext(ep, sem, recon), sem)
            self.assertEqual(report["ok"], not falling, report)

    def test_holding_must_include_the_final_frame(self):
        from tool_library.tests.test_bimanual_sync_gate import XML, build, RECON, SEM
        folder = Path(self.tmp.name); (folder / "scene.xml").write_text(XML)
        trajectory = build(folder, (0, 59), None, "final_hold", T=60)
        model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
        with np.load(trajectory) as archive:
            q = archive["qpos"].copy()
        sem = copy.deepcopy(SEM); sem["task_success"] = "The pink brick is held in the left hand."
        for dropped in (False, True):
            if dropped:
                for pad in ("robot_left_pad_a", "robot_left_pad_b"):
                    adr = int(model.jnt_qposadr[model.body(pad).jntadr[0]])
                    q[-1, adr] += .2
            ep = G.Episode(model, qpos=q, fps=30., labels=["left: hold"] * len(q))
            report = G.task_success_report(G.GoalContext(ep, sem, RECON), sem)
            self.assertEqual(report["ok"], not dropped, report)


if __name__ == "__main__":
    unittest.main()
