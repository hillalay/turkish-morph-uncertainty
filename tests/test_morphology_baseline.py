"""Experiment orchestration tests; actual scoring is covered by test_token_nll."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import run_morphology_baseline as baseline
from tests.test_token_nll import CharacterTokenizer, make_model


class MorphologyBaselineTests(unittest.TestCase):
    def setUp(self):
        self.samples = baseline.preflight()

    def run_fake(self, output, *, evaluator=None, interrupt_at=None):
        import torch

        model = make_model()
        model.register_parameter("fixture_parameter", torch.nn.Parameter(torch.zeros(1)))
        model.config.model_type = "test"
        model.config.architectures = ["FakeModel"]
        original_evaluate = baseline.evaluate_sample
        calls = []

        def score(*args):
            calls.append(args[2]["sample_id"])
            if interrupt_at == len(calls):
                raise KeyboardInterrupt()
            if evaluator is not None:
                return evaluator(*args)
            return original_evaluate(*args)

        # These are synthetic references for the CPU uniform-logit fixture only.
        # Production references are tested independently below and never altered on disk.
        references = {s["sample_id"]: (s["target"], len(s["target"]) + 1,
                       (len(s["target"]) + 1) * 6.2383246421813965, 6.2383246421813965)
                      for s in self.samples[:4]}
        with ExitStack() as stack:
            for name in ("set_device", "reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch(f"torch.cuda.{name}"))
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(patch("torch.cuda.get_device_name", return_value="test GPU"))
            stack.enter_context(patch.object(baseline, "cuda_memory", return_value={}))
            loader = stack.enter_context(patch.object(baseline, "load_nafie", return_value=(
                model, CharacterTokenizer(), {"verified_sha256": {}, "loading_info": {}})))
            stack.enter_context(patch.object(baseline, "REFERENCES", references))
            stack.enter_context(patch.object(baseline, "evaluate_sample", side_effect=score))
            stack.enter_context(redirect_stdout(io.StringIO()))
            metadata = baseline.run_experiment(output)
        return metadata, loader.call_count, calls

    def test_full_run_loads_once_scores_all_and_roundtrips_nested_alternatives(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            metadata, loads, calls = self.run_fake(output)
            self.assertEqual(metadata["status"], "ok", metadata.get("error"))
            self.assertEqual(loads, 1)
            self.assertEqual(len(calls), 40)
            self.assertEqual(metadata["successful_count"], 40)
            rows = baseline.read_jsonl(output / "results.jsonl")
            self.assertEqual(baseline.read_results_csv(output / "results.csv"), rows)
            self.assertEqual(sum(len(r["morphological_analyses"]) for r in rows), 79)
            self.assertEqual(metadata["validation"]["v2_schema_valid_count"], 40)
            self.assertEqual({p.name for p in output.iterdir()}, {
                "metadata.json", "results.jsonl", "results.csv", "experiment_report.md"})

    def test_interruption_preserves_completed_journal_records(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            metadata, loads, calls = self.run_fake(output, interrupt_at=3)
            self.assertEqual(metadata["status"], "interrupted")
            self.assertEqual(len(baseline.read_jsonl(output / "results.jsonl")), 2)
            self.assertEqual(metadata["successful_count"], 2)
            self.assertEqual(metadata["not_attempted_count"], 38)
            self.assertFalse(metadata["validation"]["exact_40_ids"])
            self.assertEqual(loads, 1)

    def test_forward_failure_is_persisted_with_successful_alignment(self):
        real_evaluate = baseline.evaluate_sample

        def score(model, tokenizer, sample, config):
            if sample["sample_id"] == "morph-v1-012":
                raise RuntimeError("simulated forward failure")
            return real_evaluate(model, tokenizer, sample, config)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            metadata, _, calls = self.run_fake(output, evaluator=score)
            self.assertEqual(metadata["status"], "validation_failed")
            self.assertEqual(len(calls), 40)
            self.assertEqual(metadata["failed_count"], 1)
            row = baseline.read_jsonl(output / "results.jsonl")[11]
            self.assertEqual(row["status"], "computation_error")
            self.assertEqual(row["alignment_check"]["status"], "ok")
            self.assertIsNone(row["model_result"])
            self.assertIn("simulated forward failure", row["error"]["message"])
            self.assertEqual(metadata["validation"]["v2_schema_valid_count"], 39)

    def test_existing_directory_is_never_overwritten_or_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "results.jsonl"
            sentinel.write_text("existing results", encoding="utf-8")
            with patch.object(baseline, "load_nafie") as loader:
                with self.assertRaises(FileExistsError):
                    baseline.run_experiment(output)
                loader.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing results")

    def test_stale_review_and_pending_approval_rejected_before_experiment(self):
        review = json.loads(baseline.REVIEW.read_text(encoding="utf-8"))
        for field, value in (("sample_sha256", "0" * 64), ("status", "pending")):
            stale = copy.deepcopy(review)
            stale["samples"][0][field] = value
            with patch.object(baseline.json, "loads", return_value=stale):
                # Do not patch read_jsonl: its parser also calls json.loads.
                with patch.object(baseline, "read_jsonl", return_value=self.samples):
                    with self.assertRaises(ValueError):
                        baseline.preflight()

    def test_missing_or_duplicate_id_rejected(self):
        for samples in (self.samples[:-1], self.samples[:-1] + [self.samples[0]]):
            with patch.object(baseline, "read_jsonl", return_value=samples):
                with self.assertRaises(ValueError):
                    baseline.preflight()

    def test_reference_tolerance_and_token_count_are_enforced(self):
        rows = []
        for sample_id, (target, count, word, normalized) in baseline.REFERENCES.items():
            rows.append({"sample_id": sample_id, "target": target, "status": "ok",
                         "model_result": {"token_count": count, "word_nll": word + 1e-6,
                                          "token_normalized_nll": normalized + 1e-6}})
        self.assertTrue(all(r["pass"] for r in baseline.compare_references(rows)))
        rows[0]["model_result"]["word_nll"] += 0.01
        rows[1]["model_result"]["token_count"] += 1
        decisions = baseline.compare_references(rows)
        self.assertFalse(decisions[0]["pass"])
        self.assertFalse(decisions[1]["pass"])
        self.assertFalse(all(r["pass"] for r in baseline.compare_references(rows[:2])))

    def test_reference_mismatch_halts_after_four_durable_records(self):
        real_evaluate = baseline.evaluate_sample

        def score(model, tokenizer, sample, config):
            result = real_evaluate(model, tokenizer, sample, config)
            # A different but internally consistent NLL is schema-valid, reference-invalid.
            result["token_logprobs"] = [value - 0.1 for value in result["token_logprobs"]]
            result["word_nll"] = -sum(result["token_logprobs"])
            result["token_normalized_nll"] = result["word_nll"] / result["token_count"]
            return result

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            metadata, _, calls = self.run_fake(output, evaluator=score)
            self.assertEqual(len(calls), 4)
            self.assertEqual(metadata["status"], "error")
            self.assertEqual(metadata["not_attempted_count"], 36)
            self.assertIn("reference mismatch", metadata["error"]["message"])

    def test_descriptive_statistics_use_sample_stdev_and_keep_groups(self):
        rows = [{"pos": "VERB", "affix_count": 2, "status": "ok", "model_result": {
            "word_nll": word, "token_normalized_nll": word / 2, "token_count": 2}}
            for word in (2.0, 4.0)]
        summary = baseline.summaries(rows)
        self.assertEqual(summary["overall"]["word_nll_mean"], 3)
        self.assertAlmostEqual(summary["overall"]["word_nll_sample_stdev"], 2 ** 0.5)
        self.assertEqual(summary["by_pos"]["VERB"]["n"], 2)
        self.assertEqual(summary["by_affix_count"]["2"]["normalized_nll_mean"], 1.5)
        self.assertIsNone(baseline.describe(rows[:1])["word_nll_sample_stdev"])

    def test_csv_rejects_metric_disagreement(self):
        row = {**self.samples[0], "sample_sha256": "test", "status": "computation_error",
               "alignment_check": None, "model_result": None, "error": {"message": "test"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            baseline.write_results_csv([row], path)
            self.assertEqual(baseline.read_results_csv(path), [row])
            import csv
            with path.open(encoding="utf-8", newline="") as stream:
                cells = list(csv.DictReader(stream))
            cells[0]["word_nll"] = "123"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=baseline.CSV_FIELDS)
                writer.writeheader()
                writer.writerows(cells)
            with self.assertRaisesRegex(ValueError, "metrics disagree"):
                baseline.read_results_csv(path)


if __name__ == "__main__":
    unittest.main()
