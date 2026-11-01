import copy
import hashlib
import importlib.util
import io
import json
import math
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import run_inference
from scripts.validate_dataset import ROOT, read_jsonl, validate_result_records
from turkish_llm_morphology.models.nafie import MODEL_ID, REVISION
from turkish_llm_morphology.token_boundary import (
    MEASUREMENT_POLICY, TOKENIZATION_SETTINGS, check_alignment, tokenize_aligned_target,
)
from turkish_llm_morphology.token_nll import compute_token_nll, evaluate_sample


class CharacterTokenizer:
    is_fast = True
    all_special_ids = [0, 1, 2, 3]

    def __call__(self, text, *, add_special_tokens, padding, truncation,
                 return_offsets_mapping=False):
        assert (add_special_tokens, padding, truncation) == (False, False, False)
        encoded = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert (skip_special_tokens, clean_up_tokenization_spaces) == (False, False)
        return "".join(chr(value) for value in ids)


def make_model():
    import torch

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.config = SimpleNamespace(vocab_size=512, max_position_embeddings=1024)
            self.seen = []
            self.inference_flags = []
            self.nonfinite = False

        def forward(self, *, input_ids, attention_mask, return_dict, output_hidden_states):
            assert return_dict and not output_hidden_states
            assert input_ids.shape[0] == 1 and attention_mask.eq(1).all().item()
            self.seen.append(input_ids.clone())
            self.inference_flags.append(torch.is_inference_mode_enabled())
            logits = torch.zeros(*input_ids.shape, self.config.vocab_size)
            if self.nonfinite:
                logits[0, 0, 0] = float("nan")
            return SimpleNamespace(logits=logits)

    return FakeModel()


def tokenization_config():
    return {
        "tokenizer_id": MODEL_ID, "tokenizer_revision": REVISION,
        "transformers_version": "test-version", "tokenizers_version": "test-version",
        **TOKENIZATION_SETTINGS,
    }


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch environment is not installed")
class TokenNLLTests(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.ids = torch.tensor([[0, 0, 2, 1]], dtype=torch.long)
        self.logits = torch.tensor([
            [[8.0, 1.0, 1.0], [1.0, 3.0, 6.0], [2.0, 5.0, 3.0], [9.0, 0.5, 0.5]]
        ], dtype=torch.float64).log()

    def test_single_target_token_uses_previous_logit_position(self):
        result = compute_token_nll(self.logits, self.ids, [3], 3)
        self.assertAlmostEqual(result["token_logprobs"][0], math.log(0.5), places=6)
        self.assertEqual(result["token_count"], 1)
        self.assertEqual(result["word_nll"], result["token_normalized_nll"])
        self.assertGreater(abs(result["token_logprobs"][0] - math.log(0.05)), 2)

    def test_multiple_target_tokens_and_nll_equations(self):
        result = compute_token_nll(self.logits, self.ids, [2, 3], 3)
        self.assertEqual(result["token_count"], 2)
        for actual, expected in zip(result["token_logprobs"], (math.log(0.6), math.log(0.5))):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(result["word_nll"], -math.log(0.3), places=6)
        self.assertEqual(result["word_nll"], -sum(result["token_logprobs"]))
        self.assertEqual(result["token_normalized_nll"], result["word_nll"] / 2)

    def test_final_token_own_logits_do_not_affect_its_probability(self):
        expected = compute_token_nll(self.logits, self.ids, [3], 3)
        self.logits[0, -1] = self.torch.tensor([100.0, -100.0, -100.0])
        self.assertEqual(compute_token_nll(self.logits, self.ids, [3], 3), expected)

    def test_fp16_normalization_is_promoted_before_log_softmax(self):
        logits = self.torch.tensor([[[65504.0, -65504.0], [0.0, 0.0]]], dtype=self.torch.float16)
        ids = self.torch.tensor([[0, 1]], dtype=self.torch.long)
        result = compute_token_nll(logits, ids, [1], 2)
        self.assertEqual(result["token_logprobs"], [-131008.0])
        self.assertEqual(result["word_nll"], 131008.0)

    def test_empty_or_insufficient_context_and_invalid_indices(self):
        for indices in ([], [0, 1, 2, 3], [1, 3], [2], [4], [2.0, 3], [True, 2, 3]):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "preceding context token"):
                compute_token_nll(self.logits, self.ids, indices, 3)
        for ids, logits in (
            (self.torch.empty((1, 0), dtype=self.torch.long), self.torch.empty((1, 0, 3))),
            (self.torch.tensor([[0]]), self.torch.zeros(1, 1, 3)),
        ):
            with self.subTest(shape=ids.shape), self.assertRaisesRegex(ValueError, "context and one target"):
                compute_token_nll(logits, ids, [0], 3)

    def test_nonfinite_logits_are_rejected_even_outside_target_rows(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "NaN or infinity"):
                logits = self.logits.clone()
                logits[0, 0, 0] = value
                compute_token_nll(logits, self.ids, [3], 3)

    def test_fp32_cast_or_log_softmax_overflow_is_not_a_valid_nll(self):
        for logits in (
            self.torch.tensor([[[1e100, -1e100], [0, 0]]], dtype=self.torch.float64),
            self.torch.tensor([[[3e38, -3e38], [0, 0]]], dtype=self.torch.float32),
        ):
            with self.subTest(dtype=logits.dtype), self.assertRaises(RuntimeError):
                compute_token_nll(logits, self.torch.tensor([[0, 1]]), [1], 2)

    def test_bad_logits_shapes_and_nonfloating_logits(self):
        for logits in (self.torch.zeros(4, 3), self.torch.zeros(1, 3, 3),
                       self.torch.zeros(1, 4, 4), self.torch.zeros(1, 4, 3, dtype=self.torch.long)):
            with self.subTest(shape=logits.shape), self.assertRaises((ValueError, RuntimeError)):
                compute_token_nll(logits, self.ids, [3], 3)

    def test_bad_input_id_shape_type_or_vocabulary(self):
        for ids in (self.ids.float(), self.ids.expand(2, -1), self.ids[0],
                    self.torch.tensor([[0, 0, -1, 1]]), self.torch.tensor([[0, 0, 3, 1]])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                compute_token_nll(self.logits, ids, [3], 3)


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch environment is not installed")
class SampleScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = read_jsonl(ROOT / "data/pilot/pilot_dataset.jsonl")

    def test_checked_ids_are_reused_and_old_alignment_api_is_unchanged(self):
        sample = self.dataset[0]
        tokenizer = CharacterTokenizer()
        encoded = tokenize_aligned_target(tokenizer, sample["prefix"], sample["target"])
        self.assertEqual(encoded.pop("input_ids"), [ord(char) for char in sample["prefix"] + sample["target"]])
        self.assertEqual(encoded, check_alignment(tokenizer, sample["prefix"], sample["target"]))

    def test_full_gold_sequence_is_teacher_forced_without_gradients(self):
        sample = self.dataset[0]
        model = make_model()
        result = evaluate_sample(model, CharacterTokenizer(), sample, tokenization_config())
        self.assertEqual(result["measurement_policy"], MEASUREMENT_POLICY)
        self.assertEqual(result["token_count"], len(" " + sample["target"]))
        self.assertEqual(model.seen[0][0].tolist(), [ord(char) for char in sample["prefix"] + sample["target"]])
        self.assertEqual(model.inference_flags, [True])
        self.assertFalse(model.training)
        self.assertAlmostEqual(result["word_nll"], result["token_count"] * math.log(512), places=5)
        validate_result_records([result], self.dataset)

    def test_alignment_errors_produce_valid_v2_error_records_without_forward(self):
        for prefix in ("", " ", "Dün", "Dün  "):
            with self.subTest(prefix=prefix):
                sample = copy.deepcopy(self.dataset[0])
                sample["prefix"] = prefix
                model = make_model()
                result = evaluate_sample(model, CharacterTokenizer(), sample, tokenization_config())
                self.assertEqual(result["status"], "alignment_error")
                for field in ("token_count", "token_logprobs", "word_nll", "token_normalized_nll", "alignment"):
                    self.assertIsNone(result[field])
                self.assertEqual(model.seen, [])
                validate_result_records([result], [sample])

    def test_context_window_overflow_is_not_truncated(self):
        model = make_model()
        model.config.max_position_embeddings = 2
        with self.assertRaisesRegex(ValueError, "context window"):
            evaluate_sample(model, CharacterTokenizer(), self.dataset[0], tokenization_config())
        self.assertEqual(model.seen, [])

    def test_nonfinite_forward_does_not_return_a_success_record(self):
        model = make_model()
        model.nonfinite = True
        with self.assertRaisesRegex(RuntimeError, "NaN or infinity"):
            evaluate_sample(model, CharacterTokenizer(), self.dataset[0], tokenization_config())

    def run_stub_pilot(self, model):
        with patch.object(run_inference, "load_nafie", return_value=(model, CharacterTokenizer(), {})), \
                patch("torch.cuda.get_device_name", return_value="unit-test CPU stub"), \
                patch("torch.cuda.reset_peak_memory_stats"), \
                patch.object(run_inference, "cuda_memory", return_value={}):
            return run_inference.run_pilot()

    def test_pilot_scores_each_sample_separately_and_validates_v2(self):
        model = make_model()
        records, report = self.run_stub_pilot(model)
        self.assertEqual(report["status"], "ok")
        self.assertTrue(report["results_validated"])
        self.assertEqual(report["ok_count"], 5)
        self.assertEqual(report["computation_errors"], [])
        self.assertEqual(len(model.seen), 5)
        validate_result_records(records, self.dataset)

    def test_computation_failures_are_separate_null_metric_diagnostics(self):
        model = make_model()
        model.nonfinite = True
        records, report = self.run_stub_pilot(model)
        self.assertEqual(records, [])
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["ok_count"], 0)
        self.assertEqual(report["computation_error_count"], 5)
        for failure in report["computation_errors"]:
            self.assertEqual(failure["status"], "computation_error")
            self.assertEqual(failure["error"]["code"], "NLL_COMPUTATION_FAILED")
            self.assertIsNone(failure["word_nll"])
            self.assertIsNone(failure["token_logprobs"])

    def test_saved_result_hash_matches_exact_utf8_bytes(self):
        record = evaluate_sample(make_model(), CharacterTokenizer(), self.dataset[0], tokenization_config())
        report = {"results_validated": True, "status": "ok"}
        result_path, report_path = Mock(), Mock()
        result_path.name = "results.jsonl"
        with patch.object(run_inference, "run_pilot", return_value=([record], report)), \
                patch.object(run_inference, "RESULTS_PATH", result_path), \
                patch.object(run_inference, "REPORT_PATH", report_path), \
                redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as outcome:
            run_inference.main()
        self.assertEqual(outcome.exception.code, 0)
        payload = result_path.with_suffix.return_value.write_bytes.call_args.args[0]
        saved_report = json.loads(report_path.write_text.call_args.args[0])
        self.assertEqual(saved_report["results_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(json.loads(payload.decode("utf-8")), record)
        self.assertNotIn(b"\r\n", payload)


if __name__ == "__main__":
    unittest.main()
