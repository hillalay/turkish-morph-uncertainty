import copy
import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jsonschema import Draft202012Validator

from scripts import check_boundary_policy
from scripts.validate_dataset import (
    ContractValidationError, ROOT, SCHEMA_DIR, read_jsonl, validate_result_records,
)
from turkish_llm_morphology.token_boundary import (
    MEASUREMENT_POLICY, RESULT_SCHEMA_ID, TOKENIZATION_SETTINGS,
    check_alignment, evaluation_config_id,
)


class FakeTokenizer:
    """Deterministic code-point offsets, with no external tokenizer dependency."""

    is_fast = True
    all_special_ids = [99]

    def __init__(self, prefix="Dün ", target="geldim", single=False):
        self.context = prefix[:-1]
        self.text = prefix + target
        start, end = len(self.context), len(self.text)
        self.context_ids = [10]
        self.full_ids = [10, 11] if single else [10, 11, 12]
        self.offsets = [(0, start), (start, end)] if single else [
            (0, start), (start, start + 4), (start + 4, end)
        ]
        self.decoded_full = self.text
        self.decoded_context = self.context

    def __call__(self, text, *, add_special_tokens, padding, truncation,
                 return_offsets_mapping=False):
        assert (add_special_tokens, padding, truncation) == (False, False, False)
        if text == self.context:
            return {"input_ids": self.context_ids.copy()}
        assert text == self.text and return_offsets_mapping
        return {"input_ids": self.full_ids.copy(), "offset_mapping": copy.deepcopy(self.offsets)}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert (skip_special_tokens, clean_up_tokenization_spaces) == (False, False)
        return self.decoded_context if ids == self.context_ids else self.decoded_full


def tokenization_config():
    return {
        "tokenizer_id": "example/tokenizer",
        "tokenizer_revision": "a" * 40,
        "transformers_version": "test-version",
        "tokenizers_version": "test-version",
        **TOKENIZATION_SETTINGS,
    }


class TokenBoundaryTests(unittest.TestCase):
    def assert_alignment_error(self, tokenizer, prefix="Dün ", target="geldim", **kwargs):
        result = check_alignment(tokenizer, prefix, target, **kwargs)
        self.assertEqual(result["status"], "alignment_error")
        self.assertIsNone(result["alignment"])
        self.assertEqual(result["error"]["code"], "TARGET_ALIGNMENT_FAILED")
        self.assertTrue(result["error"]["message"])
        return result

    def test_single_token_target(self):
        result = check_alignment(FakeTokenizer(single=True), "Dün ", "geldim")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["alignment"]["target_token_indices"], [1])
        self.assertEqual(result["alignment"]["target_token_char_spans"], [[3, 10]])

    def test_multiple_tokens_and_separate_linguistic_span(self):
        result = check_alignment(FakeTokenizer(), "Dün ", "geldim")
        self.assertEqual(result, {
            "status": "ok", "error": None,
            "alignment": {
                "linguistic_target_char_span": [4, 10],
                "measured_target_char_span": [3, 10],
                "target_token_indices": [1, 2],
                "target_token_char_spans": [[3, 7], [7, 10]],
            },
        })

    def test_turkish_unicode_uses_code_points(self):
        prefix, target = "İçeri ", "gördüğümüz"
        result = check_alignment(FakeTokenizer(prefix, target), prefix, target)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["alignment"]["measured_target_char_span"], [5, 16])
        self.assertEqual(result["alignment"]["linguistic_target_char_span"], [6, 16])

    def test_unsupported_trailing_whitespace_is_never_skipped(self):
        for prefix in ("Dün", "Dün  ", "Dün\t", "Dün\u00a0", "Dün\t ", "Dün\u00a0 "):
            with self.subTest(prefix=prefix):
                tokenizer = Mock()
                self.assert_alignment_error(tokenizer, prefix)
                tokenizer.assert_not_called()

    def test_empty_context_cannot_be_scored_without_bos(self):
        tokenizer = FakeTokenizer(" ")
        tokenizer.context_ids = []
        self.assert_alignment_error(tokenizer, " ")

    def test_unknown_policy_never_falls_back(self):
        self.assert_alignment_error(FakeTokenizer(), measurement_policy="strict_target_v1")

    def test_non_nfc_and_invalid_targets_are_rejected(self):
        for prefix, target in (("Du\u0308n ", "geldim"), ("Dün ", ""), ("Dün ", "gel dim")):
            with self.subTest(prefix=prefix, target=target):
                self.assert_alignment_error(Mock(), prefix, target)

    def test_context_ids_must_be_stable(self):
        tokenizer = FakeTokenizer()
        tokenizer.full_ids[0] = 20
        self.assert_alignment_error(tokenizer)

    def test_target_offsets_must_be_exact_nonempty_and_contiguous(self):
        for spans in (
            [(4, 7), (7, 10)],  # Missing leading space.
            [(3, 6), (7, 10)],  # Gap.
            [(3, 8), (7, 10)],  # Overlap.
            [(3, 7), (7, 9)],   # Missing final character.
            [(2, 7), (7, 10)],  # Context leakage.
            [(3, 3), (3, 10)],  # Empty span.
            [(3, 8), (8, 7)],   # Reversed span.
            [(3, 7), (7, 11)],  # Byte-like/out-of-range offsets.
        ):
            with self.subTest(spans=spans):
                tokenizer = FakeTokenizer()
                tokenizer.offsets[1:] = spans
                self.assert_alignment_error(tokenizer)

    def test_context_offset_cannot_cross_measured_boundary(self):
        tokenizer = FakeTokenizer()
        tokenizer.offsets[0] = (0, 4)
        self.assert_alignment_error(tokenizer)

    def test_offset_array_length_and_types(self):
        for offsets in ([(0, 3)], [(0, 3), (3.0, 7), (7, 10)], [None]):
            with self.subTest(offsets=offsets):
                tokenizer = FakeTokenizer()
                tokenizer.offsets = offsets
                self.assert_alignment_error(tokenizer)

    def test_decode_must_preserve_full_text_and_context(self):
        for field in ("decoded_full", "decoded_context"):
            with self.subTest(field=field):
                tokenizer = FakeTokenizer()
                setattr(tokenizer, field, "normalized text")
                self.assert_alignment_error(tokenizer)

    def test_special_tokens_are_rejected(self):
        tokenizer = FakeTokenizer()
        tokenizer.full_ids[-1] = 99
        self.assert_alignment_error(tokenizer)

    def test_slow_or_unavailable_offsets_are_alignment_errors(self):
        tokenizer = FakeTokenizer()
        tokenizer.is_fast = False
        self.assert_alignment_error(tokenizer)
        for error in (NotImplementedError("no offsets"), ValueError("bad input"), KeyError("offset_mapping")):
            with self.subTest(error=error):
                tokenizer = Mock(is_fast=True, side_effect=error)
                self.assert_alignment_error(tokenizer)

    def test_no_target_tokens_is_alignment_error(self):
        tokenizer = FakeTokenizer()
        tokenizer.full_ids = [10]
        tokenizer.offsets = [(0, 3)]
        self.assert_alignment_error(tokenizer)

    def test_cli_reports_diagnostics_without_loading_models(self):
        samples = read_jsonl(ROOT / "data/pilot/pilot_dataset.jsonl")[:1]
        sample = samples[0]
        factory = Mock(return_value=FakeTokenizer(sample["prefix"], sample["target"]))
        output = io.StringIO()
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=factory))}), \
                patch.object(sys, "argv", ["check_boundary_policy.py", "--revision", "a" * 40]), \
                patch.object(check_boundary_policy, "read_jsonl", return_value=samples), \
                patch.object(check_boundary_policy, "version", return_value="test-version"), \
                redirect_stdout(output):
            check_boundary_policy.main()
        diagnostic, summary = output.getvalue().splitlines()
        diagnostic = json.loads(diagnostic)
        self.assertEqual(diagnostic["status"], "ok")
        self.assertEqual(summary, "TOTAL: 1/1 PASS")
        self.assertEqual(diagnostic["evaluation_config_id"], evaluation_config_id(
            diagnostic["measurement_policy"], diagnostic["tokenization_config"]
        ))
        factory.assert_called_once_with(
            "nafie-ai/nafie-473M", revision="a" * 40, trust_remote_code=False, use_fast=True
        )

    def test_cli_rejects_mutable_revision_before_loading_tokenizer(self):
        with patch.object(sys, "argv", ["check_boundary_policy.py", "--revision", "main"]), \
                patch.object(check_boundary_policy, "read_jsonl") as read, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            check_boundary_policy.main()
        self.assertEqual(error.exception.code, 2)
        read.assert_not_called()


class ResultV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = read_jsonl(ROOT / "data/pilot/pilot_dataset.jsonl")

    def result(self):
        sample = self.dataset[0]
        config = tokenization_config()
        return {
            "schema_id": RESULT_SCHEMA_ID,
            "sample_id": sample["sample_id"],
            "model_id": "example/model", "model_revision": "abc123",
            "measurement_policy": MEASUREMENT_POLICY,
            "tokenization_config": config,
            "evaluation_config_id": evaluation_config_id(MEASUREMENT_POLICY, config),
            "token_count": 2, "token_logprobs": [-0.4, -0.6],
            "word_nll": 1.0, "token_normalized_nll": 0.5,
            **check_alignment(FakeTokenizer(sample["prefix"], sample["target"]), sample["prefix"], sample["target"]),
        }

    def validate(self, result, dataset=None):
        validate_result_records([result], self.dataset if dataset is None else dataset)

    def legacy_result(self):
        result = self.result()
        for field in ("schema_id", "measurement_policy", "tokenization_config"):
            del result[field]
        result["evaluation_config_id"] = "baseline-v1"
        alignment = result["alignment"]
        alignment["target_char_span"] = alignment.pop("linguistic_target_char_span")
        del alignment["measured_target_char_span"]
        alignment["target_token_char_spans"][0][0] += 1
        return result

    def test_valid_v2_result(self):
        self.validate(self.result())

    def test_schema_is_valid_and_has_distinct_id(self):
        schemas = [json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8")) for name in (
            "model_result.schema.json", "model_result.v2.schema.json"
        )]
        for schema in schemas:
            Draft202012Validator.check_schema(schema)
        self.assertNotEqual(schemas[0]["$id"], schemas[1]["$id"])
        self.assertEqual(schemas[1]["$id"], RESULT_SCHEMA_ID)

    def test_v1_and_v2_validate_in_same_file(self):
        validate_result_records([self.legacy_result(), self.result()], self.dataset)

    def test_v1_cannot_pass_v2_schema_with_only_schema_id_added(self):
        result = self.legacy_result()
        result["schema_id"] = RESULT_SCHEMA_ID
        with self.assertRaises(ContractValidationError):
            self.validate(result)

    def test_unknown_null_or_missing_v2_schema_id_is_rejected(self):
        for schema_id in (None, "v2", "urn:turkish-llm-morphology:schema:model-result:v3", "missing"):
            with self.subTest(schema_id=schema_id):
                result = self.result()
                if schema_id == "missing":
                    del result["schema_id"]
                else:
                    result["schema_id"] = schema_id
                with self.assertRaises(ContractValidationError):
                    self.validate(result)

    def test_v1_alignment_cannot_be_relabeled_v2(self):
        result = self.result()
        alignment = result["alignment"]
        alignment["target_char_span"] = alignment.pop("linguistic_target_char_span")
        del alignment["measured_target_char_span"]
        with self.assertRaises(ContractValidationError):
            self.validate(result)

    def test_each_required_v2_field(self):
        for field in self.result():
            with self.subTest(field=field):
                result = self.result()
                del result[field]
                with self.assertRaises(ContractValidationError):
                    self.validate(result)

    def test_wrong_character_spans(self):
        for field in ("linguistic_target_char_span", "measured_target_char_span"):
            with self.subTest(field=field):
                result = self.result()
                result["alignment"][field][0] += 1
                with self.assertRaisesRegex(ContractValidationError, field):
                    self.validate(result)

    def test_unsupported_prefix_cannot_have_success_metrics(self):
        for prefix in ("Dün", "Dün  "):
            with self.subTest(prefix=prefix):
                dataset = copy.deepcopy(self.dataset)
                dataset[0]["prefix"] = prefix
                with self.assertRaisesRegex(ContractValidationError, "exactly one ASCII space"):
                    self.validate(self.result(), dataset)

    def test_invalid_token_arrays_and_nll(self):
        for field, value in (
            ("token_count", 1), ("token_logprobs", [-1.0]),
            ("token_logprobs", [0.1, -1.1]), ("word_nll", 2.0),
            ("token_normalized_nll", 0.9), ("token_logprobs", [float("nan"), -0.6]),
            ("word_nll", float("inf")), ("token_normalized_nll", float("nan")),
        ):
            with self.subTest(field=field, value=value):
                result = self.result()
                result[field] = value
                with self.assertRaises(ContractValidationError):
                    self.validate(result)

    def test_malformed_alignment_arrays(self):
        for field, value in (
            ("target_token_indices", [1, 3]), ("target_token_indices", [0, 1]),
            ("target_token_char_spans", [[3, 6], [7, 10]]),
            ("target_token_char_spans", [[3, 8], [7, 10]]),
            ("target_token_char_spans", []),
        ):
            with self.subTest(field=field, value=value):
                result = self.result()
                start = result["alignment"]["measured_target_char_span"][0]
                if field == "target_token_char_spans" and value:
                    value = [[start, start + value[0][1] - 3],
                             [start + 4, result["alignment"]["measured_target_char_span"][1]]]
                result["alignment"][field] = value
                with self.assertRaises(ContractValidationError):
                    self.validate(result)

    def test_config_identity_covers_policy_and_all_settings(self):
        config = tokenization_config()
        original = evaluation_config_id(MEASUREMENT_POLICY, config)
        self.assertEqual(original, evaluation_config_id(MEASUREMENT_POLICY, dict(reversed(list(config.items())))))
        self.assertNotEqual(original, evaluation_config_id("strict_target_v1", config))
        for field, value in config.items():
            with self.subTest(field=field):
                changed = {**config, field: not value if isinstance(value, bool) else value + "-changed"}
                self.assertNotEqual(original, evaluation_config_id(MEASUREMENT_POLICY, changed))
                result = self.result()
                result["tokenization_config"] = changed
                with self.assertRaises(ContractValidationError):
                    self.validate(result)

    def test_unknown_policy_and_changed_fixed_settings_are_rejected_even_with_new_hash(self):
        result = self.result()
        result["measurement_policy"] = "strict_target_v1"
        result["evaluation_config_id"] = evaluation_config_id("strict_target_v1", result["tokenization_config"])
        with self.assertRaises(ContractValidationError):
            self.validate(result)
        for field, value in TOKENIZATION_SETTINGS.items():
            with self.subTest(field=field):
                result = self.result()
                result["tokenization_config"][field] = not value
                result["evaluation_config_id"] = evaluation_config_id(MEASUREMENT_POLICY, result["tokenization_config"])
                with self.assertRaises(ContractValidationError):
                    self.validate(result)

    def test_legacy_and_v2_cannot_share_config_id(self):
        v1, v2 = self.legacy_result(), self.result()
        v1["evaluation_config_id"] = v2["evaluation_config_id"]
        with self.assertRaisesRegex(ContractValidationError, "mixes measurement policies/settings"):
            validate_result_records([v1, v2], self.dataset)

    def test_duplicate_v2_key_and_unknown_sample_are_rejected(self):
        result = self.result()
        with self.assertRaisesRegex(ContractValidationError, "duplicate result key"):
            validate_result_records([result, copy.deepcopy(result)], self.dataset)
        result["sample_id"] = "unknown-sample"
        with self.assertRaisesRegex(ContractValidationError, "unknown sample_id"):
            self.validate(result)

    def test_alignment_error_record_and_null_metrics(self):
        result = self.result()
        result.update(check_alignment(FakeTokenizer(), "Dün", "geldim"))
        for field in ("token_count", "token_logprobs", "word_nll", "token_normalized_nll"):
            result[field] = None
        self.validate(result)
        for field, value in (("token_count", 1), ("alignment", {}), ("error", None)):
            with self.subTest(field=field):
                invalid = {**result, field: value}
                with self.assertRaises(ContractValidationError):
                    self.validate(invalid)
        result["evaluation_config_id"] = "untracked"
        with self.assertRaises(ContractValidationError):
            self.validate(result)


if __name__ == "__main__":
    unittest.main()
