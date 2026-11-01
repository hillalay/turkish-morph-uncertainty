import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

from turkish_llm_morphology.models import nafie


class NafieLoaderTests(unittest.TestCase):
    def test_reviewed_revision_and_artifacts_are_pinned(self):
        self.assertRegex(nafie.REVISION, r"^[a-f0-9]{40}$")
        self.assertIn("configuration_nafie.py", nafie.REVIEWED_SHA256)
        self.assertIn("modeling_nafie.py", nafie.REVIEWED_SHA256)
        self.assertIn("model.safetensors", nafie.REVIEWED_SHA256)
        self.assertFalse(any(name.endswith(".bin") for name in nafie.REVIEWED_SHA256))

    def test_changed_snapshot_is_rejected_before_code_execution(self):
        content = b"reviewed bytes\r\n"
        expected = hashlib.sha256(content).hexdigest()
        with patch.object(nafie, "REVIEWED_SHA256", {"modeling_nafie.py": expected}):
            with patch.object(Path, "open", mock_open(read_data=content)):
                self.assertEqual(nafie.verify_snapshot(Path("unused")), {"modeling_nafie.py": expected})
            with patch.object(Path, "open", mock_open(read_data=b"different bytes")), \
                    self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                nafie.verify_snapshot(Path("unused"))

    def test_cuda_unavailable_never_falls_back_or_downloads(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": fake_torch}), \
                self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
            nafie.load_nafie()

    def test_integrity_failure_stops_before_custom_model_import(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        snapshot_download = Mock(return_value="untrusted-snapshot")
        model_factory = Mock()
        with patch.dict("sys.modules", {
            "torch": fake_torch,
            "huggingface_hub": SimpleNamespace(snapshot_download=snapshot_download),
            "transformers": SimpleNamespace(AutoModelForCausalLM=model_factory),
        }), patch.object(nafie, "verify_snapshot", side_effect=RuntimeError("SHA-256 mismatch")), \
                self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            nafie.load_nafie()
        model_factory.from_pretrained.assert_not_called()
        snapshot_download.assert_called_once_with(
            nafie.MODEL_ID, revision=nafie.REVISION, cache_dir=nafie.CACHE_DIR,
            allow_patterns=list(nafie.REVIEWED_SHA256), local_files_only=True,
        )

    def test_transformers_loading_sets_are_saved_as_json_arrays(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.parameters.return_value = [SimpleNamespace(device=SimpleNamespace(type="cuda"), dtype="fp16")]
        loading_info = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []}
        with patch.dict("sys.modules", {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), float16="fp16"),
            "huggingface_hub": SimpleNamespace(snapshot_download=Mock(return_value="reviewed-snapshot")),
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value="tokenizer")),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=(model, loading_info))),
            ),
        }), patch.object(nafie, "verify_snapshot", return_value={}):
            loaded, tokenizer, metadata = nafie.load_nafie()
        self.assertIs(loaded, model)
        self.assertEqual(tokenizer, "tokenizer")
        self.assertEqual(json.loads(json.dumps(metadata))["loading_info"], {key: [] for key in loading_info})


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch environment is not installed")
class LogitsTests(unittest.TestCase):
    def test_expected_logits_shape_is_accepted(self):
        import torch

        nafie.validate_logits(torch.zeros(1, 3, 7), (1, 3), 7)

    def test_wrong_batch_sequence_or_vocab_is_rejected(self):
        import torch

        for shape in ((3, 7), (2, 3, 7), (1, 2, 7), (1, 3, 8)):
            with self.subTest(shape=shape), self.assertRaisesRegex(RuntimeError, "Unexpected logits shape"):
                nafie.validate_logits(torch.zeros(shape), (1, 3), 7)

    def test_nan_and_infinity_are_rejected(self):
        import torch

        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "NaN or infinity"):
                nafie.validate_logits(torch.full((1, 3, 7), value), (1, 3), 7)

    def test_probe_records_cuda_failure_without_model_loading(self):
        with patch("torch.cuda.is_available", return_value=False), \
                patch.object(nafie, "load_nafie") as loader:
            report = nafie.run_forward_probe()
        loader.assert_not_called()
        self.assertEqual(report["status"], "error")
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["stage"], "cuda_check")
        self.assertIn("CPU fallback is disabled", report["error"]["message"])
        self.assertEqual(report["model_revision"], nafie.REVISION)


if __name__ == "__main__":
    unittest.main()
