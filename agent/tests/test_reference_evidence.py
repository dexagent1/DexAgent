"""Real source pixels and immutable receipts, including adversarial substitutions."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent import reference_evidence as E
from agent.library import call_tool
from tool_library.common.evidence import file_digest
from tool_library.skills.extract_frames import extract_frames
from tool_library.skills.segment_source_frame import segment_source_frame
from tool_library.skills.segment_with_text_prompts import segment_with_text_prompts


class ReferenceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ws = self.root / "stage2"
        self.ws.mkdir()
        self.video = self.root / "source.avi"
        self.other = self.root / "render.avi"
        for path, offset in ((self.video, 0), (self.other, 15)):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10., (80, 64))
            self.assertTrue(writer.isOpened())
            for i in range(3):
                pixels = np.full((64, 80, 3), 180, np.uint8)
                pixels[17:43, 23 + offset:43 + offset] = (15, 25, 230 - i * 20)
                cv2.circle(pixels, (60, 45), 6, (30, 200, 30), -1)
                writer.write(pixels)
            writer.release()
        self.auth = E.begin(self.ws, self.video)

    def tearDown(self):
        E._AUTHORITIES.pop(str(self.ws.resolve()), None)
        self.tmp.cleanup()

    def cpu(self, frame_index=0, box=None, name="object"):
        args = {"video": str(self.video), "frame_index": frame_index,
                "box": box or [18, 12, 49, 49], "out_dir": str(self.ws / name)}
        token = self.auth.prepare("segment_source_frame", args)
        result = segment_source_frame(**args)
        self.auth.complete(token, {"ok": True, "result": result})
        return result

    def recon(self, obj, hands=None):
        masks = self.ws / "masks"
        masks.mkdir(exist_ok=True)
        shutil.copyfile(obj["mask"], masks / "object.png")
        if hands:
            shutil.copyfile(hands["mask"], masks / "hands.png")
        value = {"frame0": obj["source_frame"], "scene_xml": "scene.xml", "camera_json": "camera.json",
                 "objects": [{"id": "object", "type": "rigid", "root_body": "object", "mask": "masks/object.png"}]}
        (self.ws / "scene_recon.json").write_text(json.dumps(value))
        return value

    class SourceProcessor:
        """Deterministic source-pixel fixture, not a claim of real SAM accuracy."""
        model_identity = {"version": "sam3", "repo_id": "facebook/sam3", "checkpoint_sha256": "a" * 64}

        def set_image(self, image):
            self.pixels = np.asarray(image).astype(float)
            return None

        def set_text_prompt(self, prompt, state):
            p = self.pixels
            channel = 0 if prompt == "red object" else 1
            other = 1 if channel == 0 else 0
            mask = (p[:, :, channel] > p[:, :, other] + 60) & (p[:, :, channel] > p[:, :, 2] + 60)
            return {"masks": mask[None], "scores": np.array([.9])}

    def sam_fixture(self, *, instances=False, egodex=False):
        args = {"video": str(self.video), "out_dir": str(self.ws / "frames"), "start": 0, "max_frames": 1, "quality": 71}
        if egodex:
            frames = self.ws / "frames"
            frames.mkdir()
            cap = cv2.VideoCapture(str(self.video))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = cap.read()
            cap.release()
            self.assertTrue(ok)
            path = frames / "frame00000.jpg"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            token = self.auth.prepare("read_egodex_episode", {"hdf5": "unused.hdf5", "mp4": str(self.video), "out_dir": str(frames)})
            self.auth.complete(token, {"ok": True, "result": {"frames": [str(path)]}})
        else:
            token = self.auth.prepare("extract_frames", args)
            self.auth.complete(token, {"ok": True, "result": extract_frames(**args)})
            path = Path(args["out_dir"]) / "frame00000.jpg"
        seg = {"frames_dir": str(path.parent), "prompts": ["red object", "green object"] if instances else ["red object"],
               "out_dir": str(self.ws / "sam"), "instances": instances, "min_score": .4, "downscale": .5, "version": "sam3"}
        token = self.auth.prepare("segment_with_text_prompts", seg)
        with patch("tool_library.skills.segment_with_text_prompts.load_sam3", return_value=self.SourceProcessor()):
            result = segment_with_text_prompts(**seg)
        self.auth.complete(token, {"ok": True, "result": result})
        selected = result["instances"]["red object"][0][0]["mask"] if instances else str(self.ws / "sam/frame00000.png")
        recon = self.recon({"mask": selected, "source_frame": str(path)})
        if instances:
            recon["objects"].append({"id": "second", "mask": result["instances"]["green object"][0][0]["mask"]})
            (self.ws / "scene_recon.json").write_text(json.dumps(recon))
        self.assertTrue(self.auth.check()["ok"])
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        return recon

    def replay_dispatch(self, lib, index, name, args, **kwargs):
        self.assertEqual(name, "segment_with_text_prompts")
        self.assertTrue(kwargs["force_subprocess"])
        self.assertEqual(len(list(Path(args["frames_dir"]).iterdir())), 1)
        self.assertNotEqual(Path(args["frames_dir"]).parent, self.ws)
        with patch("tool_library.skills.segment_with_text_prompts.load_sam3", return_value=self.SourceProcessor()):
            return {"ok": True, "result": segment_with_text_prompts(**args)}

    def sam_role_mask(self, prompts, *, empty=False, full=False):
        serial = getattr(self, "_role_serial", 0)
        self._role_serial = serial + 1
        frames, out = self.ws / f"role_frames{serial}", self.ws / f"role_masks{serial}"
        args = {"video": str(self.video), "out_dir": str(frames), "max_frames": 1}
        token = self.auth.prepare("extract_frames", args)
        self.auth.complete(token, {"ok": True, "result": extract_frames(**args)})
        class Processor(self.SourceProcessor):
            def set_text_prompt(self, prompt, state):
                if empty or full:
                    return {"masks": np.full((1, *self.pixels.shape[:2]), full, bool), "scores": np.array([.9])}
                return super().set_text_prompt(prompt, state)
        args = {"frames_dir": str(frames), "prompts": prompts, "out_dir": str(out)}
        token = self.auth.prepare("segment_with_text_prompts", args)
        with patch("tool_library.skills.segment_with_text_prompts.load_sam3", return_value=Processor()):
            result = segment_with_text_prompts(**args)
        self.auth.complete(token, {"ok": True, "result": result})
        return {"mask": str(out / "frame00000.png"), "source_frame": str(frames / "frame00000.jpg")}

    def static_mask_fixture(self, *, identical_frames=False):
        cap = cv2.VideoCapture(str(self.video))
        ok, image = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.video = self.root / "static.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (80, 64))
        for index in range(3):
            pixels = image.copy()
            if not identical_frames:
                pixels[:8, :8] = 40 + index * 30  # changing background; object pixels stay identical
            writer.write(pixels)
        writer.release()
        self.auth = E.begin(self.ws, self.video)
        args = {"video": str(self.video), "out_dir": str(self.ws / "frames")}
        token = self.auth.prepare("extract_frames", args)
        self.auth.complete(token, {"ok": True, "result": extract_frames(**args)})
        seg = {"frames_dir": args["out_dir"], "prompts": ["red object"], "out_dir": str(self.ws / "sam")}
        token = self.auth.prepare("segment_with_text_prompts", seg)
        self.assertEqual([row["record"]["source_frame_index"] for row in token["frames"]], [0, 1, 2])
        with patch("tool_library.skills.segment_with_text_prompts.load_sam3", return_value=self.SourceProcessor()):
            result = segment_with_text_prompts(**seg)
        self.auth.complete(token, {"ok": True, "result": result})
        digests = {file_digest(self.ws / f"sam/frame{i:05d}.png") for i in range(3)}
        self.assertEqual(len(digests), 1, "fixture must generate byte-identical masks at distinct source times")
        mask_record = self.auth.masks[next(iter(digests))]
        self.assertEqual({row["source_frame_index"] for row in E._bindings(mask_record)}, {0, 1, 2})
        self.recon({"mask": str(self.ws / "sam/frame00000.png"), "source_frame": str(self.ws / "frames/frame00000.jpg")})
        return next(iter(digests))

    def test_real_cpu_source_mask_and_byte_copy_pass(self):
        obj = self.cpu()
        self.recon(obj)
        self.assertTrue(E.check_workspace(self.ws, self.video)["ok"])
        mask = cv2.imread(obj["mask"], 0) > 0
        target = np.zeros_like(mask)
        target[17:43, 23:43] = True
        self.assertGreater((mask & target).sum() / (mask | target).sum(), .8)

    def test_legacy_masks_and_forged_ok_reports_fail_direct_suite(self):
        obj = self.cpu()
        self.recon(obj)
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        (self.ws / "reference_provenance.json").write_text(json.dumps({"ok": True, "trusted": True}))
        semantic = self.root / "semantic.json"
        semantic.write_text('{"objects":[],"subgoals":[]}')
        from tool_library.verifiers.stage2_gate_suite import stage2_gate_suite
        with patch("tool_library.verifiers.stage2_gate_suite._call", return_value={"ok": True}):
            result = stage2_gate_suite(self.ws, semantic)
        self.assertFalse(result["ok"])
        self.assertFalse(result["reports"]["source_reference_provenance_check"]["ok"])

    def test_render_replacement_cannot_forge_recipe_pass(self):
        obj = self.cpu()
        self.recon(obj)
        mask_path = self.ws / "masks" / "object.png"
        simulated = np.zeros((64, 80), np.uint8)
        simulated[2:15, 2:30] = 255
        cv2.imwrite(str(mask_path), simulated)
        saved = json.loads((self.ws / "reference_evidence_recipes.json").read_text())
        claimed = dict(next(iter(saved["masks"].values())), ok=True, trusted=True)
        saved["masks"][file_digest(mask_path)] = claimed
        saved["ok"] = True
        (self.ws / "reference_evidence_recipes.json").write_text(json.dumps(saved))
        self.assertFalse(E.check_workspace(self.ws)["ok"])
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        self.assertFalse(E.revalidate_cpu_recipes(self.ws, self.video)["ok"])

    def test_edited_mask_pixels_fail(self):
        obj = self.cpu()
        self.recon(obj)
        path = self.ws / "masks" / "object.png"
        pixels = cv2.imread(str(path), 0)
        pixels[0, 0] = 255
        cv2.imwrite(str(path), pixels)
        self.assertFalse(self.auth.check()["ok"])

    def test_fabricated_hands_exclusion_fails(self):
        obj = self.cpu()
        hands = self.sam_role_mask(["human hand"])
        self.recon(obj, hands)
        self.assertTrue(self.auth.check()["ok"])
        cv2.imwrite(str(self.ws / "masks" / "hands.png"), np.ones((64, 80), np.uint8) * 255)
        self.assertFalse(self.auth.check()["ok"])

    def test_mixed_source_frames_fail(self):
        obj = self.cpu()
        hands = self.cpu(frame_index=1, box=[50, 35, 70, 55], name="hands")
        self.recon(obj, hands)
        self.assertFalse(self.auth.check()["ok"])

    def test_modified_reference_frame_fails(self):
        obj = self.cpu()
        self.recon(obj)
        pixels = cv2.imread(obj["source_frame"])
        pixels[0, 0] = 0
        cv2.imwrite(obj["source_frame"], pixels)
        self.assertFalse(self.auth.check()["ok"])

    def test_another_video_and_processor_injection_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned original"):
            self.auth.prepare("segment_source_frame", {"video": str(self.other), "frame_index": 0, "box": [18, 12, 49, 49]})
        with self.assertRaisesRegex(ValueError, "injected"):
            self.auth.prepare("segment_with_text_prompts", {"processor": "fake", "frames_dir": "frames"})

    def test_source_video_mutation_revokes_receipt(self):
        obj = self.cpu()
        self.recon(obj)
        shutil.copyfile(self.other, self.video)
        self.assertFalse(self.auth.check()["ok"])

    def test_fresh_process_recipe_is_recomputed_not_imported(self):
        obj = self.cpu()
        hands = self.cpu(box=[50, 35, 70, 55], name="hands")
        recon = self.recon(obj)
        recon["objects"].append({"id": "second", "mask": hands["mask"]})
        (self.ws / "scene_recon.json").write_text(json.dumps(recon))
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        self.assertFalse(E.check_workspace(self.ws)["ok"])
        with patch("tool_library.skills.segment_source_frame.segment_source_frame", wraps=segment_source_frame) as replay:
            result = E.revalidate_cpu_recipes(self.ws, self.video)
        self.assertTrue(result["ok"], result)
        self.assertEqual(replay.call_count, 2)

    def test_source_extraction_and_sam_receipts_bind_exact_input(self):
        args = {"video": str(self.video), "out_dir": str(self.ws / "frames"), "max_frames": 1}
        token = self.auth.prepare("extract_frames", args)
        self.auth.complete(token, {"ok": True, "result": extract_frames(**args)})
        seg_args = {"frames_dir": args["out_dir"], "prompts": ["red object"], "out_dir": str(self.ws / "sam")}
        class Processor:
            def set_image(self, image):
                self.pixels = np.asarray(image)
                return None
            def set_text_prompt(self, prompt, state):
                p = self.pixels.astype(float)
                return {"masks": ((p[:, :, 0] > p[:, :, 1] + 60) & (p[:, :, 0] > p[:, :, 2] + 60))[None]}
        token = self.auth.prepare("segment_with_text_prompts", seg_args)
        # Generic test fixture replaces a costly model, but the production
        # schema never accepts a processor and the harness rejects injection.
        with patch("tool_library.skills.segment_with_text_prompts.load_sam3", return_value=Processor()):
            output = segment_with_text_prompts(**seg_args)
        self.auth.complete(token, {"ok": True, "result": output})
        self.recon({"mask": str(self.ws / "sam/frame00000.png"), "source_frame": str(self.ws / "frames/frame00000.jpg")})
        self.assertTrue(self.auth.check()["ok"])
        cv2.imwrite(str(self.ws / "frames/frame00000.jpg"), np.zeros((64, 80, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "source receipt"):
            self.auth.prepare("segment_with_text_prompts", seg_args)

    def test_detection_does_not_certify_subsequent_mask(self):
        args = {"video": str(self.video), "out_dir": str(self.ws / "frames"), "max_frames": 1}
        token = self.auth.prepare("extract_frames", args)
        self.auth.complete(token, {"ok": True, "result": extract_frames(**args)})
        token = self.auth.prepare("detect_objects", {"frames_dir": args["out_dir"], "keywords": ["object"]})
        self.auth.complete(token, {"ok": True, "result": {"ok": True, "detections": {}}})
        self.assertEqual(self.auth.masks, {})

    def test_empty_or_malformed_segmentation_rejected(self):
        for box in ([0, 0, 80, 64], [18.5, 12, 49, 49], [49, 12, 18, 49]):
            with self.assertRaises(ValueError):
                segment_source_frame(self.video, 0, box, self.ws / "bad")
        with patch("tool_library.skills.segment_source_frame.cv2.grabCut", return_value=None):
            with self.assertRaisesRegex(ValueError, "empty"):
                segment_source_frame(self.video, 0, [18, 12, 49, 49], self.ws / "bad")

    def test_generated_tool_child_cannot_read_parent_authority(self):
        lib = self.root / "isolated_tools"
        (lib / "skills").mkdir(parents=True)
        (lib / "__init__.py").write_text("")
        (lib / "skills/__init__.py").write_text("")
        (lib / "skills/read_authority.py").write_text("def read_authority():\n    from agent.reference_evidence import _AUTHORITIES\n    return {'authority_count': len(_AUTHORITIES)}\n")
        index = {"tools": [{"name": "read_authority", "type": "skill", "inputs": [], "entrypoint": "read_authority"}]}
        (lib / "index.json").write_text(json.dumps(index))
        self.assertGreater(len(E._AUTHORITIES), 0)
        result = call_tool(lib, index, "read_authority", {}, cwd=self.ws, force_subprocess=True)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["result"]["authority_count"], 0)

    def test_stage3_refuses_legacy_upstream_before_running_agent(self):
        from stage3_trajectory_generation import run_stage3 as S3
        obj = self.cpu()
        self.recon(obj)
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        (self.ws / "reference_evidence_recipes.json").unlink()
        semantic = self.root / "scene_semantics.json"
        semantic.write_text('{}')
        config = {"inputs": {"video": str(self.video)}, "run": {"name": "case", "output_dir": str(self.root / "runs")}, "stage3": {}}
        with patch.object(S3, "load_config", return_value=config), \
             patch("agent.input_identity.video_verifier", return_value=lambda _: {"ok": True}), \
             patch("agent.llm.make_llm") as model:
            rc = S3.main(["unused", "--recon", str(self.ws / "scene_recon.json"), "--semantic", str(semantic)])
        self.assertEqual(rc, 1)
        model.assert_not_called()
        report = json.loads((self.root / "runs/case/stage3/run.json").read_text())
        self.assertEqual(report["upstream"]["stage"], "stage2")
        self.assertFalse(report["finished"])

    def test_runner_captures_direct_tool_receipt_and_blocks_forged_finish(self):
        from agent.runner import StageRunner
        from agent.llm import MockLLM
        lib = Path(__file__).resolve().parents[2] / "tool_library"
        config = {"inputs": {"video": str(self.video), "library": str(lib)}, "agent": {"max_rounds": 1}}
        runner = StageRunner(config, MockLLM([]), self.ws, "test", [lambda _: {"ok": True}], stage="stage2")
        args = {"video": str(self.video), "frame_index": 0, "box": [18, 12, 49, 49], "out_dir": "direct"}
        _, result = runner.run_tool("segment_source_frame", args)
        self.assertTrue(result["ok"], result)
        self.recon(result["result"])
        self.assertTrue(runner._verify()["ok"])
        cv2.imwrite(str(self.ws / "masks/object.png"), np.ones((64, 80), np.uint8) * 255)
        self.assertFalse(runner._verify()["ok"])
        with patch("agent.runner.call_tool", return_value={"ok": True, "result": {}}) as dispatch:
            runner.run_tool("arbitrary_created_skill", {})
        self.assertTrue(dispatch.call_args.kwargs["force_subprocess"])

    def test_sam_fresh_replay_preserves_extraction_quality_and_routing(self):
        self.sam_fixture()
        cfg = {"environments": {"sam3": "/configured/sam/python"}, "tool_environments": {"segment_with_text_prompts": "sam3"}}
        lib = Path(__file__).resolve().parents[2] / "tool_library"
        with patch("agent.library.call_tool", side_effect=self.replay_dispatch) as dispatch:
            result = E.revalidate_source_recipes(self.ws, self.video, library=lib, cfg=cfg)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["replay"]["source_frames"], 1)
        self.assertEqual(result["replay"]["sam_calls"], 1)
        self.assertIs(dispatch.call_args.kwargs["cfg"], cfg)
        self.assertEqual(dispatch.call_args.args[0], lib)

    def test_sam_instance_references_share_one_independent_inference(self):
        self.sam_fixture(instances=True)
        with patch("agent.library.call_tool", side_effect=self.replay_dispatch) as dispatch:
            result = E.revalidate_source_recipes(self.ws, self.video)
        self.assertTrue(result["ok"], result)
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(len(result["checked"]), 2)

    def test_egodex_source_seek_recipe_replays_only_selected_frame(self):
        self.sam_fixture(egodex=True)
        with patch("agent.library.call_tool", side_effect=self.replay_dispatch):
            result = E.revalidate_source_recipes(self.ws, self.video)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["replay"]["source_frames"], 1)

    def test_sam_forged_paths_processors_code_and_version_fail_before_inference(self):
        self.sam_fixture()
        recipe_path = self.ws / "reference_evidence_recipes.json"
        original = json.loads(recipe_path.read_text())
        digest = file_digest(self.ws / "masks/object.png")
        modifications = [
            lambda r: r["args"].update(processor="injected"),
            lambda r: r["args"].update(out_dir="/arbitrary/output"),
            lambda r: r["args"].update(code="raise SystemExit"),
            lambda r: r["args"].update(version="unsupported-model"),
            lambda r: r["frame"]["recipe"].update(video=str(self.other)),
            lambda r: r.update(tool="arbitrary_created_tool"),
            lambda r: r["frame"].update(sha256="0" * 64),
        ]
        for change in modifications:
            with self.subTest(change=change):
                E._AUTHORITIES.pop(str(self.ws.resolve()), None)
                saved = json.loads(json.dumps(original))
                change(saved["masks"][digest]["recipe"])
                recipe_path.write_text(json.dumps(saved))
                with patch("agent.library.call_tool") as dispatch:
                    self.assertFalse(E.revalidate_source_recipes(self.ws, self.video)["ok"])
                dispatch.assert_not_called()

    def test_sam_checkpoint_mismatch_rejects_even_identical_mask(self):
        self.sam_fixture()
        def other_model(*args, **kwargs):
            result = self.replay_dispatch(*args, **kwargs)
            result["result"]["model_identity"] = dict(self.SourceProcessor.model_identity, checkpoint_sha256="b" * 64)
            return result
        with patch("agent.library.call_tool", side_effect=other_model):
            result = E.revalidate_source_recipes(self.ws, self.video)
        self.assertFalse(result["ok"])
        self.assertIn("checkpoint", " ".join(result["issues"]))

    def test_sam_single_pixel_nondeterminism_is_not_approximately_accepted(self):
        self.sam_fixture()
        def changed_pixel(*args, **kwargs):
            result = self.replay_dispatch(*args, **kwargs)
            path = Path(result["result"]["out_dir"]) / "frame00000.png"
            pixels = cv2.imread(str(path), 0)
            pixels[0, 0] = 255
            cv2.imwrite(str(path), pixels)
            return result
        with patch("agent.library.call_tool", side_effect=changed_pixel):
            result = E.revalidate_source_recipes(self.ws, self.video)
        self.assertFalse(result["ok"])
        self.assertIn("exact artifact bytes", " ".join(result["issues"]))

    def test_sam_unavailable_runtime_is_unverified_not_skipped(self):
        self.sam_fixture()
        with patch("agent.library.call_tool", return_value={"ok": False, "error": "checkpoint unavailable"}):
            result = E.revalidate_source_recipes(self.ws, self.video)
        self.assertFalse(result["ok"])
        self.assertIn("checkpoint unavailable", " ".join(result["issues"]))

    def test_sam_instance_result_cannot_reference_old_or_external_mask(self):
        recon = self.sam_fixture(instances=True)
        def escaped_path(*args, **kwargs):
            result = self.replay_dispatch(*args, **kwargs)
            result["result"]["instances"]["red object"][0][0]["mask"] = str(self.ws / "masks/object.png")
            return result
        with patch("agent.library.call_tool", side_effect=escaped_path):
            result = E.revalidate_source_recipes(self.ws, self.video)
        self.assertFalse(result["ok"])
        self.assertIn("escaped", " ".join(result["issues"]))

    def test_identical_masks_preserve_first_frame_binding_and_fresh_replay(self):
        self.static_mask_fixture()
        self.assertTrue(self.auth.check()["ok"])
        self.assertEqual(self.auth.check()["checked"][0]["source_frame_index"], 0)
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        with patch("agent.library.call_tool", side_effect=self.replay_dispatch) as dispatch:
            result = E.revalidate_source_recipes(self.ws, self.video)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["checked"][0]["source_frame_index"], 0)
        self.assertEqual(dispatch.call_count, 1)
        pixels = cv2.imread(str(self.ws / "masks/object.png"), 0)
        pixels[0, 0] = 255
        cv2.imwrite(str(self.ws / "masks/object.png"), pixels)
        self.assertFalse(E.check_workspace(self.ws)["ok"])

    def test_identical_source_images_preserve_exact_temporal_binding(self):
        self.static_mask_fixture(identical_frames=True)
        frame_digests = {file_digest(self.ws / f"frames/frame{i:05d}.jpg") for i in range(3)}
        self.assertEqual(len(frame_digests), 1)
        for index in (2, 0):
            recon = json.loads((self.ws / "scene_recon.json").read_text())
            recon["frame0"] = str(self.ws / f"frames/frame{index:05d}.jpg")
            (self.ws / "scene_recon.json").write_text(json.dumps(recon))
            report = self.auth.check()
            if index == 0:
                self.assertTrue(report["ok"], report)
                self.assertEqual(report["checked"][0]["source_frame_index"], index)
            else:
                self.assertFalse(report["ok"])
                self.assertIn("source frame index 0", " ".join(report["issues"]))
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        with patch("agent.library.call_tool", side_effect=self.replay_dispatch) as dispatch:
            report = E.revalidate_source_recipes(self.ws, self.video)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["checked"][0]["source_frame_index"], 0)
        self.assertEqual(dispatch.call_count, 1)

    def test_byte_copy_alias_requires_one_common_binding_across_all_masks(self):
        self.static_mask_fixture(identical_frames=True)
        # Restrict otherwise genuine source-generated masks to disjoint valid
        # temporal bindings. Equal reference pixels cannot justify mixing them.
        red_digest = file_digest(self.ws / "masks/object.png")
        red_zero = next(row for row in E._bindings(self.auth.masks[red_digest]) if row["source_frame_index"] == 0)
        self.auth.masks[red_digest] = red_zero
        green = np.zeros((64, 80), np.uint8)
        green[40:45, 57:62] = 255
        path = self.ws / "masks/second.png"
        cv2.imwrite(str(path), green)
        frame_one = self.auth.frame_record(self.ws / "frames/frame00001.jpg", file_digest(self.ws / "frames/frame00001.jpg"))
        self.auth._mask(path, frame_one)  # trusted synthetic receipt fixture
        copied = self.ws / "copied_reference.jpg"
        shutil.copyfile(self.ws / "frames/frame00000.jpg", copied)
        recon = json.loads((self.ws / "scene_recon.json").read_text())
        recon["frame0"] = str(copied)
        recon["objects"].append({"id": "second", "mask": str(path)})
        (self.ws / "scene_recon.json").write_text(json.dumps(recon))
        report = self.auth.check()
        self.assertFalse(report["ok"])
        self.assertIn("different source frame", " ".join(report["issues"]))

    def test_empty_approved_hands_mask_is_valid_and_replayed(self):
        obj = self.cpu()
        hands = self.sam_role_mask(["human hand", "arm", "sleeve"], empty=True)
        self.recon(obj, hands)
        self.assertTrue(self.auth.check()["ok"])
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        def empty_dispatch(lib, index, name, args, **kwargs):
            class Empty(self.SourceProcessor):
                def set_text_prompt(self, prompt, state):
                    return {"masks": np.zeros((0, *self.pixels.shape[:2]), bool)}
            with patch("tool_library.skills.segment_with_text_prompts.load_sam3", return_value=Empty()):
                return {"ok": True, "result": segment_with_text_prompts(**args)}
        with patch("agent.library.call_tool", side_effect=empty_dispatch):
            report = E.revalidate_source_recipes(self.ws, self.video)
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["replay"]["sam_calls"], 1)

    def test_empty_source_object_mask_is_never_accepted(self):
        obj = self.sam_role_mask(["red object"], empty=True)
        self.recon(obj)
        report = self.auth.check()
        self.assertFalse(report["ok"])
        self.assertIn("nonempty", " ".join(report["issues"]))

    def test_cpu_object_mask_cannot_be_renamed_hands_even_with_forged_role(self):
        obj = self.cpu()
        hands = self.cpu(box=[50, 35, 70, 55], name="hands")
        self.recon(obj, hands)
        self.assertFalse(self.auth.check()["ok"])
        path = self.ws / "reference_evidence_recipes.json"
        recipes = json.loads(path.read_text())
        recipes["masks"][file_digest(hands["mask"])]["origin"] = {"tool": "segment_with_text_prompts", "prompts": ["human hand"]}
        path.write_text(json.dumps(recipes))
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        report = E.revalidate_source_recipes(self.ws, self.video)
        self.assertFalse(report["ok"])
        self.assertIn("CPU box", " ".join(report["issues"]))

    def test_unrelated_or_mixed_sam_prompts_cannot_certify_hands(self):
        obj = self.cpu()
        for prompts in (["green object"], ["hand sanitizer"], ["human hand", "green object"]):
            with self.subTest(prompts=prompts):
                hands = self.sam_role_mask(prompts)
                self.recon(obj, hands)
                report = self.auth.check()
                self.assertFalse(report["ok"])
                self.assertIn("prompt provenance", " ".join(report["issues"]))

    def test_nonempty_human_hand_prompt_replays_with_role_proof(self):
        obj = self.cpu()
        hands = self.sam_role_mask(["person's left hand"])
        self.recon(obj, hands)
        self.assertTrue(self.auth.check()["ok"])
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        with patch("agent.library.call_tool", side_effect=self.replay_dispatch):
            report = E.revalidate_source_recipes(self.ws, self.video)
        self.assertTrue(report["ok"], report)
        record = E._AUTHORITIES[str(self.ws.resolve())].masks[file_digest(hands["mask"])]
        self.assertTrue(any(E._hands_origin(row) for row in E._bindings(record)))

    def test_full_source_derived_hands_mask_is_rejected(self):
        obj = self.cpu()
        hands = self.sam_role_mask(["human hand"], full=True)
        self.recon(obj, hands)
        self.assertFalse(self.auth.check()["ok"])

    def test_later_coherent_reference_and_masks_do_not_replace_frame_zero(self):
        obj = self.cpu(frame_index=1)
        self.recon(obj)
        report = self.auth.check()
        self.assertFalse(report["ok"])
        self.assertIn("source frame index 0", " ".join(report["issues"]))
        E._AUTHORITIES.pop(str(self.ws.resolve()))
        self.assertFalse(E.revalidate_source_recipes(self.ws, self.video)["ok"])


if __name__ == "__main__":
    unittest.main()
