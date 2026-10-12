"""Example reuse (certificate, replay plan paths) and the goal phrasings Stage 1 examples use.

Synthetic scenes and files only, except the certificate test, which uses a shipped example and its shipped video.
"""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tool_library.common import examples as EX
from tool_library.common import task_gates as G
from tool_library.common.final_goal_language import compile_final, invariant_tol_m

CAMERA = '<camera name="video_cam" pos="0 -.7 1" quat=".9238795 .3826834 0 0"/>'


class GoalPhraseTests(unittest.TestCase):
    def scene(self, bowl_radius=.08):
        walls = "".join(f'<geom type="box" size=".01 .02 .03" pos="{bowl_radius * np.cos(a):.4f} {bowl_radius * np.sin(a):.4f} .03" '
                        f'euler="0 0 {np.degrees(a):.1f}"/>' for a in np.linspace(0, 2 * np.pi, 12, endpoint=False))
        xml = ('<mujoco><worldbody><geom name="g_table" type="plane" size="2 2 .1"/>' + CAMERA +
               '<body name="cup" pos="-.2 0 .03"><freejoint/><geom type="cylinder" size=".03 .03" mass=".05"/></body>'
               f'<body name="bowl" pos="0 0 0"><geom type="cylinder" size="{bowl_radius} .004" pos="0 0 .004"/>{walls}</body>'
               '</worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        sem = {"objects": [{"id": "cup", "name": "blue cup", "type": "rigid"}, {"id": "bowl", "name": "green bowl", "type": "rigid"}]}
        recon = {"objects": [{"id": o["id"], "root_body": o["id"]} for o in sem["objects"]]}
        return G.GoalContext(G.Episode(model, qpos=np.repeat(model.qpos0[None], 10, axis=0), fps=30.), sem, recon)

    def test_shipped_example_phrasings_compile(self):
        gc = self.scene()
        for text, classes in [("The cup is held clear of the table.", {"held", "lift_clearance"}),
                              ("The cup is above the bowl's interior.", {"final_over_opening"}),
                              ("The cup is held upright in the left hand, clear of the table.", {"held", "final_orientation", "lift_clearance"}),
                              ("The cup is held level above the table.", {"held", "lift_clearance"})]:
            predicates, coverage, issues = compile_final(gc, text)
            self.assertEqual(issues, [], text)
            self.assertTrue(classes <= {p["class"] for p in predicates}, (text, predicates))

    def test_unknown_phrase_still_fails_closed(self):
        _, _, issues = compile_final(self.scene(), "The cup is somewhere nice.")
        self.assertTrue(issues)

    def test_unmoved_tolerance_scales_with_the_object(self):
        small, large = self.scene(.04), self.scene(.09)
        self.assertAlmostEqual(invariant_tol_m(small, "cup"), .005)                 # a 6 cm cup: the 5 mm floor
        self.assertGreater(invariant_tol_m(large, "bowl"), invariant_tol_m(small, "bowl"))
        self.assertLessEqual(invariant_tol_m(large, "bowl"), .01)                   # never above 10 mm


class ReplayPathTests(unittest.TestCase):
    def test_attach_and_prepare_rewrite_machine_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = tmp / "run"; s3 = run / "stage3"; (s3 / "poses").mkdir(parents=True); (run / "stage2" / "assets" / "cup").mkdir(parents=True)
            (run / "stage2" / "assets" / "cup" / "cup.xml").write_text("<mujoco/>")
            hs = {"hand_xml": str(tmp / "repo" / "setup" / "robot" / "sharpa_left.xml"), "bench_scene": str(run / "stage2" / "scene.xml"),
                  "objects": [{"name": "cup", "xml": str(run / "stage2" / "assets" / "cup" / "cup.xml")}],
                  "note": f"re-anchored from {run}/stage3/poses/old.json with a turn"}
            (s3 / "poses" / "hs.json").write_text(json.dumps(hs))
            (s3 / "poses" / "pose.json").write_text(json.dumps({"side": "left", "pos": [0, 0, .1]}))
            (s3 / "plan.json").write_text(json.dumps({"scene_recon_json": str(run / "stage2" / "scene_recon.json"), "subgoals": [
                {"index": 1, "motion": "grasp_lift", "pose": "poses/pose.json", "hand_scene": str(s3 / "poses" / "hs.json")}]}))
            (s3 / "place_arm_stations.json").write_text(json.dumps({"shift": [0.0, 0.1]}))
            ex = tmp / "example"; (ex / "stage3").mkdir(parents=True)
            rep = EX.attach_replay(ex, s3)
            self.assertEqual(rep["stations"], [0.0, 0.1])
            shipped = (ex / "stage3" / "replay").rglob("*.json")
            self.assertFalse(any(str(tmp) in p.read_text() for p in shipped), "a machine path was shipped")
            ws, s2 = tmp / "ws", tmp / "newrun" / "stage2"
            (s2 / "objects" / "cup").mkdir(parents=True); (s2 / "objects" / "cup" / "cup.xml").write_text("<mujoco/>")
            out = EX.prepare_replay(ex, ws, s2, tmp / "repo2" / "setup")
            plan = json.loads(Path(out["plan"]).read_text())
            self.assertEqual(plan["scene_recon_json"], str(s2.resolve() / "scene_recon.json"))
            sg = plan["subgoals"][0]
            self.assertTrue(Path(sg["pose"]).is_file() and Path(sg["hand_scene"]).is_file())
            got = json.loads(Path(sg["hand_scene"]).read_text())
            self.assertEqual(got["objects"][0]["xml"], str(s2.resolve() / "objects" / "cup" / "cup.xml"))   # reuse's layout
            self.assertTrue(got["hand_xml"].endswith("repo2/setup/robot/sharpa_left.xml"))
            self.assertEqual(out["stations"], [0.0, 0.1])


class CertificateTests(unittest.TestCase):
    def test_same_recording_certifies_and_other_footage_or_tampering_does_not(self):
        from agent.reference_evidence import certify_example
        ex = REPO / "examples" / "default" / "knot_the_rope"
        video = REPO / "stage1_semantic_understanding" / "examples" / "knot_the_rope" / "video.mp4"
        other = REPO / "stage1_semantic_understanding" / "examples" / "place_cup_in_bowl" / "video.mp4"
        if not (ex / "stage2" / "example_certificate.json").is_file():
            self.skipTest("the example ships no certificate")
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "stage2"
            EX.reuse_example(str(ex), "2", ws, video=str(video))
            self.assertTrue(certify_example(ws, video, ex)["ok"])
            ws2 = Path(tmp) / "other" / "stage2"
            EX.reuse_example(str(ex), "2", ws2, video=str(other))
            r = certify_example(ws2, other, ex)
            self.assertFalse(r["ok"]); self.assertIn("first frame", " ".join(r["issues"]))
            ws3 = Path(tmp) / "tampered" / "stage2"
            EX.reuse_example(str(ex), "2", ws3, video=str(video))
            mask = ws3 / "masks" / "rope.png"
            import cv2
            m = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED); m[:5, :5] = 255 - m[:5, :5]; cv2.imwrite(str(mask), m)
            r = certify_example(ws3, video, ex)
            self.assertFalse(r["ok"]); self.assertIn("rope.png", " ".join(r["issues"]))


if __name__ == "__main__":
    unittest.main()
