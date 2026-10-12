"""Image SAM uses compatible complete weights and does not leak inference modes."""
import importlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repository root: runs as a script too
D = importlib.import_module("tool_library.skills.detect_objects")
S = importlib.import_module("tool_library.skills.segment_with_text_prompts")


class SamImageLoadingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.checkpoint = self.path / "sam3.pt"
        self.checkpoint.write_bytes(b"generic checkpoint fixture")
        self.model = Mock()
        self.model.state_dict.return_value = {"weight": "initial"}
        self.build = Mock(return_value=self.model)
        self.download = Mock(return_value=str(self.checkpoint))
        self.processor = Mock()
        self.processor_type = Mock(return_value=self.processor)
        self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                           __version__="test", load=Mock(return_value={"model": {"detector.weight": "loaded"}}))
        self.cache = Mock(return_value=str(self.checkpoint))
        self.modules = {"torch": self.torch,
                        "huggingface_hub": types.SimpleNamespace(try_to_load_from_cache=self.cache),
                        "sam3": types.ModuleType("sam3"), "sam3.model": types.ModuleType("sam3.model"),
                        "sam3.model.sam3_image_processor": types.SimpleNamespace(Sam3Processor=self.processor_type),
                        "sam3.model_builder": types.SimpleNamespace(build_sam3_image_model=self.build, download_ckpt_from_hf=self.download)}

    def load(self, **kwargs):
        with patch.dict(sys.modules, self.modules), patch.object(D.importlib.metadata, "version", return_value="test"):
            return D.load_sam3(**kwargs)

    def test_cached_image_weights_loaded_strictly_and_cpu_device_agrees(self):
        result = self.load()
        self.cache.assert_called_once_with("facebook/sam3", "sam3.pt")
        self.download.assert_not_called()
        self.build.assert_called_once_with(checkpoint_path=None, load_from_HF=False, device="cpu")
        self.model.load_state_dict.assert_called_once_with({"weight": "loaded"}, strict=True)
        self.model.eval.assert_called_once()
        self.processor_type.assert_called_once_with(self.model, device="cpu", confidence_threshold=.1)
        self.assertEqual(result.model_identity["version"], "sam3")
        self.assertEqual(result.model_identity["repo_id"], "facebook/sam3")
        self.assertEqual(len(result.model_identity["checkpoint_sha256"]), 64)

    def test_uncached_checkpoint_downloads_correct_image_version(self):
        self.cache.return_value = None
        self.load()
        self.download.assert_called_once_with(version="sam3")

    def test_only_disabled_interactive_tracker_neck_extras_are_ignored(self):
        self.torch.load.return_value = {"detector.weight": "loaded", "detector.backbone.vision_backbone.sam2_convs.0.weight": "unused"}
        self.load()
        self.model.load_state_dict.assert_called_once_with({"weight": "loaded"}, strict=True)
        self.torch.load.return_value["detector.unrecognized.weight"] = "wrong architecture"
        with self.assertRaisesRegex(ValueError, "unsupported detector weights"):
            self.load()

    def test_multiplex_video_and_missing_or_partial_weights_fail(self):
        with self.assertRaisesRegex(ValueError, "multiplex video"):
            D.load_sam3(version="sam3.1")
        self.torch.load.return_value = {"tracker.weight": "wrong model"}
        with self.assertRaisesRegex(ValueError, "no detector weights"):
            self.load()
        self.model.load_state_dict.assert_not_called()
        self.torch.load.return_value = {"detector.weight": "partial"}
        self.model.load_state_dict.side_effect = RuntimeError("Missing required model weights")
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            self.load()
        self.processor_type.assert_not_called()

    def test_autocast_and_inference_are_scoped_even_on_exception(self):
        active = []
        @contextmanager
        def context(name):
            active.append(name)
            try:
                yield
            finally:
                self.assertEqual(active.pop(), name)
        torch = types.SimpleNamespace(inference_mode=lambda: context("inference"),
            autocast=lambda device, dtype: context("autocast"), bfloat16="bfloat16", float16="float16",
            cuda=types.SimpleNamespace(is_bf16_supported=lambda: True))
        processor = Mock()
        def image(*args):
            self.assertEqual(active, ["inference", "autocast"])
            return "state"
        def text(**kwargs):
            self.assertEqual(active, ["inference", "autocast"])
            raise RuntimeError("inference failure")
        processor.set_image.side_effect = image
        processor.set_text_prompt.side_effect = text
        result = D._InferenceProcessor(processor, torch, "cuda", {})
        self.assertEqual(result.set_image("image"), "state")
        self.assertEqual(active, [])
        with self.assertRaisesRegex(RuntimeError, "inference failure"):
            result.set_text_prompt(prompt="object", state="state")
        self.assertEqual(active, [])

    def test_segmentation_forwards_image_version_and_records_identity(self):
        frames = self.path / "frames"
        frames.mkdir()
        Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(frames / "frame00000.png")
        masks = np.zeros((1, 1, 20, 30), bool)
        masks[0, 0, 5:15, 5:20] = True
        proc = Mock(model_identity={"version": "sam3", "checkpoint_sha256": "a" * 64})
        proc.set_text_prompt.return_value = {"masks": masks, "scores": np.array([.9])}
        with patch.object(S, "load_sam3", return_value=proc) as load:
            result = S.segment_with_text_prompts(frames, ["object"], self.path / "masks", instances=True)
        load.assert_called_once_with(.3, "sam3")
        self.assertEqual(result["model_identity"], proc.model_identity)
        self.assertEqual(result["n_instances"]["object"], [1])
        self.assertEqual(result["instances"]["object"][0][0]["area_px"], 150)
        self.assertEqual(json.loads((self.path / "masks" / "meta.json").read_text())["model_identity"], proc.model_identity)


if __name__ == "__main__":
    unittest.main()
