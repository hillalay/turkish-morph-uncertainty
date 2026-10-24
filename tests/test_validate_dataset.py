import copy
import unittest

from scripts.validate_dataset import (
    ContractValidationError,
    ROOT,
    read_jsonl,
    validate_dataset_records,
    validate_result_records,
)


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = read_jsonl(ROOT / "data" / "pilot" / "pilot_dataset.jsonl")

    def test_pilot_dataset_is_valid(self):
        validate_dataset_records(self.dataset)

    def test_duplicate_sample_id_is_rejected(self):
        duplicate = copy.deepcopy(self.dataset[:2])
        duplicate[1]["sample_id"] = duplicate[0]["sample_id"]
        with self.assertRaisesRegex(ContractValidationError, "duplicate sample_id"):
            validate_dataset_records(duplicate)

    def test_invalid_sample_id_is_rejected(self):
        invalid = copy.deepcopy(self.dataset[:1])
        invalid[0]["sample_id"] = "Pilot 001"
        with self.assertRaises(ContractValidationError):
            validate_dataset_records(invalid)

    def test_missing_required_field_is_rejected(self):
        invalid = copy.deepcopy(self.dataset[:1])
        del invalid[0]["target"]
        with self.assertRaisesRegex(ContractValidationError, "required property"):
            validate_dataset_records(invalid)

    def test_selected_analysis_must_match_top_level_fields(self):
        invalid = copy.deepcopy(self.dataset[:1])
        invalid[0]["lemma"] = "yanlış"
        with self.assertRaisesRegex(ContractValidationError, "must match the selected analysis"):
            validate_dataset_records(invalid)

    def test_affix_count_must_match_selected_analysis(self):
        invalid = copy.deepcopy(self.dataset[:1])
        invalid[0]["affix_count"] += 1
        with self.assertRaisesRegex(ContractValidationError, "affix_count"):
            validate_dataset_records(invalid)

    def test_derivational_affix_is_rejected_even_with_false_scope_flag(self):
        invalid = copy.deepcopy(self.dataset[:1])
        invalid[0]["morphological_analyses"][0]["affixes"][0]["type"] = "derivational"
        with self.assertRaisesRegex(ContractValidationError, "derivational affixes"):
            validate_dataset_records(invalid)

    def test_experiment_type_must_match_morphology(self):
        invalid = copy.deepcopy(self.dataset[:1])
        invalid[0]["scope"]["experiment_type"] = "noun_case"
        with self.assertRaisesRegex(ContractValidationError, "overtly case-marked noun"):
            validate_dataset_records(invalid)

    def test_multiword_target_is_rejected(self):
        invalid = copy.deepcopy(self.dataset[:1])
        invalid[0]["target"] = "gelmiş idim"
        with self.assertRaisesRegex(ContractValidationError, "exactly one word"):
            validate_dataset_records(invalid)

    def test_out_of_scope_flag_is_rejected(self):
        invalid = copy.deepcopy(self.dataset[:1])
        invalid[0]["scope"]["has_copula"] = True
        with self.assertRaises(ContractValidationError):
            validate_dataset_records(invalid)

    def test_valid_model_result(self):
        validate_result_records([self._valid_result()], self.dataset)

    def test_bad_alignment_is_rejected(self):
        invalid = self._valid_result()
        invalid["alignment"]["target_char_span"][0] -= 1
        with self.assertRaisesRegex(ContractValidationError, "target_char_span"):
            validate_result_records([invalid], self.dataset)

    def test_token_span_crossing_target_boundary_is_rejected(self):
        invalid = self._valid_result()
        invalid["alignment"]["target_token_char_spans"][0][0] -= 1
        with self.assertRaisesRegex(ContractValidationError, "cover the target exactly"):
            validate_result_records([invalid], self.dataset)

    def test_gap_between_token_spans_is_rejected(self):
        invalid = self._valid_result()
        invalid["alignment"]["target_token_char_spans"][0][1] -= 1
        with self.assertRaisesRegex(ContractValidationError, "non-empty and contiguous"):
            validate_result_records([invalid], self.dataset)

    def test_bad_nll_is_rejected(self):
        invalid = self._valid_result()
        invalid["word_nll"] = 99.0
        with self.assertRaisesRegex(ContractValidationError, "word_nll"):
            validate_result_records([invalid], self.dataset)

    def test_duplicate_result_key_is_rejected(self):
        result = self._valid_result()
        with self.assertRaisesRegex(ContractValidationError, "duplicate result key"):
            validate_result_records([result, copy.deepcopy(result)], self.dataset)

    def test_same_sample_with_different_model_is_allowed(self):
        first = self._valid_result()
        second = copy.deepcopy(first)
        second["model_id"] = "example/other-model"
        validate_result_records([first, second], self.dataset)

    def test_unknown_sample_id_is_rejected(self):
        invalid = self._valid_result()
        invalid["sample_id"] = "pilot-999"
        with self.assertRaisesRegex(ContractValidationError, "unknown sample_id"):
            validate_result_records([invalid], self.dataset)

    def test_alignment_error_has_no_metrics(self):
        result = self._valid_result()
        result.update(
            status="alignment_error",
            token_count=None,
            token_logprobs=None,
            word_nll=None,
            token_normalized_nll=None,
            alignment=None,
            error={"code": "TARGET_ALIGNMENT_FAILED", "message": "boundary split"},
        )
        validate_result_records([result], self.dataset)

    def _valid_result(self):
        sample = self.dataset[0]
        start = len(sample["prefix"])
        middle = start + 3
        end = start + len(sample["target"])
        return {
            "sample_id": sample["sample_id"],
            "model_id": "example/model",
            "model_revision": "abc123",
            "evaluation_config_id": "baseline-v1",
            "status": "ok",
            "token_count": 2,
            "token_logprobs": [-0.4, -0.6],
            "word_nll": 1.0,
            "token_normalized_nll": 0.5,
            "alignment": {
                "target_char_span": [start, end],
                "target_token_indices": [3, 4],
                "target_token_char_spans": [[start, middle], [middle, end]],
            },
            "error": None,
        }


if __name__ == "__main__":
    unittest.main()
