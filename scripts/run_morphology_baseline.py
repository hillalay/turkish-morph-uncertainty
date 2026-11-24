"""Run the approved 40-sample Nafie baseline into a new, exclusive directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from turkish_llm_morphology.models.nafie import MODEL_ID, REVISION, cuda_memory, load_nafie
from turkish_llm_morphology.morphology.dataset import FIELDS, record_sha256, validate_manual_review
from turkish_llm_morphology.token_boundary import (
    MEASUREMENT_POLICY, TOKENIZATION_SETTINGS, check_alignment, target_char_spans,
)
from turkish_llm_morphology.token_nll import evaluate_sample

if __package__:
    from .validate_dataset import ROOT, read_jsonl, validate_dataset_records, validate_result_records
else:
    from validate_dataset import ROOT, read_jsonl, validate_dataset_records, validate_result_records


DATASET = ROOT / "data/pilot/morphology_pilot_dataset.jsonl"
REVIEW = ROOT / "data/pilot/morphology_pilot_review.json"
EVIDENCE = ROOT / "data/pilot/morphology_pilot_analyses.jsonl"
OUTPUT = ROOT / "experiments/experiment_001_nafie_morphology"
EXPECTED_IDS = {f"morph-v1-{i:03d}" for i in range(1, 41)}
REFERENCES = {
    "morph-v1-001": ("geldim", 1, 5.2880983353, 5.2880983353),
    "morph-v1-002": ("geldik", 1, 9.1152467728, 9.1152467728),
    "morph-v1-003": ("gelmedim", 2, 7.9363057613, 3.9681528807),
    "morph-v1-004": ("gelmedik", 2, 11.6550559998, 5.8275279999),
}
REFERENCE_TOLERANCE = {"abs_tol": 1e-4, "rel_tol": 1e-5}
METRICS = ("token_count", "token_logprobs", "word_nll", "token_normalized_nll")
EXTRAS = ("sample_sha256", "status", "alignment_check", "model_result", "error")
CSV_FIELDS = (*FIELDS, *EXTRAS, *METRICS)
WARNING_ASSESSMENT = (
    "AutoTokenizer uses trust_remote_code=False. In Transformers 5.22.1, "
    "tokenization_auto.py catches the unknown/custom AutoConfig ValueError and falls "
    "back to PreTrainedConfig (model_type=''). configuration_utils.py then warns "
    "about nafie versus the empty model_type. Reproduced with tokenizer-only loading; "
    "this warning does not identify the causal model class. The causal model is loaded "
    "separately with pinned, SHA-256-verified custom code and trust_remote_code=True. "
    "Check runtime model/config classes, empty loading diagnostics and finite forwards "
    "below. No architecture, checkpoint, tokenizer settings or logging filters changed."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def save_metadata(output, metadata):
    temporary = output / "metadata.json.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "metadata.json")


def preflight():
    samples = read_jsonl(DATASET)
    validate_dataset_records(samples)
    if len(samples) != 40 or {s["sample_id"] for s in samples} != EXPECTED_IDS:
        raise ValueError("Expected exactly morph-v1-001 through morph-v1-040")
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    validate_manual_review(samples, review)
    for sample in samples:
        target_char_spans(sample["prefix"], sample["target"], MEASUREMENT_POLICY)
        if not sample["prefix"][:-1].strip():
            raise ValueError(f"{sample['sample_id']}: context is empty")
    return sorted(samples, key=lambda s: s["sample_id"])


def compare_references(rows):
    by_id = {r["sample_id"]: r for r in rows}
    comparisons = []
    for sample_id, (target, count, word, normalized) in REFERENCES.items():
        row = by_id.get(sample_id)
        result = (row or {}).get("model_result") or {}
        ok = bool(row and row["status"] == "ok" and row["target"] == target
                  and result.get("token_count") == count)
        for field, expected in (("word_nll", word), ("token_normalized_nll", normalized)):
            value = result.get(field)
            ok = ok and isinstance(value, (int, float)) and math.isclose(
                value, expected, **REFERENCE_TOLERANCE)
        comparisons.append({
            "sample_id": sample_id, "target": target, "pass": bool(ok),
            "expected_token_count": count, "actual_token_count": result.get("token_count"),
            "expected_word_nll": word, "actual_word_nll": result.get("word_nll"),
            "word_nll_delta": None if result.get("word_nll") is None else result["word_nll"] - word,
            "expected_normalized_nll": normalized,
            "actual_normalized_nll": result.get("token_normalized_nll"),
            "normalized_nll_delta": None if result.get("token_normalized_nll") is None
            else result["token_normalized_nll"] - normalized,
        })
    return comparisons


def csv_metrics(row):
    result = row["model_result"] or {}
    return {field: result.get(field) for field in METRICS}


def write_results_csv(rows, path):
    # Complex cells are JSON; scalar metrics stay directly readable in a spreadsheet.
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: encode(value) for key, value in {**row, **csv_metrics(row)}.items()})


def read_results_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("Experiment CSV header mismatch")
        rows = []
        for cells in reader:
            row = {key: json.loads(value) for key, value in cells.items()}
            projected = {field: row.pop(field) for field in METRICS}
            if projected != csv_metrics(row):
                raise ValueError("CSV metrics disagree with the v2 model_result payload")
            rows.append(row)
    return rows


def describe(rows):
    if not rows:
        return {"n": 0}
    word = [r["model_result"]["word_nll"] for r in rows]
    normalized = [r["model_result"]["token_normalized_nll"] for r in rows]
    return {
        "n": len(rows), "word_nll_mean": statistics.mean(word),
        "word_nll_median": statistics.median(word),
        "word_nll_sample_stdev": statistics.stdev(word) if len(word) > 1 else None,
        "word_nll_min": min(word), "word_nll_max": max(word),
        "normalized_nll_mean": statistics.mean(normalized),
        "normalized_nll_median": statistics.median(normalized),
        "token_count_mean": statistics.mean(r["model_result"]["token_count"] for r in rows),
        "affix_count_mean": statistics.mean(r["affix_count"] for r in rows),
    }


def summaries(rows):
    good = [r for r in rows if r["status"] == "ok"]
    return {"overall": describe(good), **{
        f"by_{field}": {str(value): describe([r for r in good if r[field] == value])
                         for value in sorted({r[field] for r in good})}
        for field in ("pos", "affix_count")
    }}


def validate_outputs(rows, samples, output, input_hashes):
    by_id = {s["sample_id"]: s for s in samples}
    validated = alignment_ok = hashes_ok = 0
    for row in rows:
        sample = by_id[row["sample_id"]]
        if {field: row[field] for field in FIELDS} != sample:
            raise ValueError("Experiment morphology differs from the approved dataset")
        hashes_ok += row["sample_sha256"] == record_sha256(sample)
        result = row["model_result"]
        if result is not None:
            validate_result_records([result], samples)
            if result["sample_id"] != row["sample_id"] or result["status"] != row["status"]:
                raise ValueError("Envelope differs from model_result")
            if result["alignment"] != row["alignment_check"]["alignment"]:
                raise ValueError("Alignment precheck differs from model_result")
            validated += 1
        alignment_ok += (row["alignment_check"] or {}).get("status") == "ok"
    # Also enforce uniqueness/configuration consistency across the entire v2 set.
    validate_result_records([r["model_result"] for r in rows if r["model_result"] is not None], samples)
    return {
        "exact_40_ids": len(rows) == 40 and {r["sample_id"] for r in rows} == EXPECTED_IDS,
        "alignment_ok_count": alignment_ok, "v2_schema_valid_count": validated,
        "sample_hash_match_count": hashes_ok,
        "inputs_unchanged": all(sha256(ROOT / path) == value for path, value in input_hashes.items()),
        "csv_jsonl_equal": read_results_csv(output / "results.csv") == rows,
        # The existing v2 validator checks finite, nonpositive logprobs, both NLL
        # equations, token-array lengths and exact target spans for every ok row.
        "finite_valid_nll_count": sum(r["status"] == "ok" for r in rows),
    }


class LoadWarningRecorder(logging.Handler):
    """Observe warnings alongside the normal console handler; never suppress them."""

    def __init__(self, messages):
        super().__init__(logging.WARNING)
        self.messages = messages

    def emit(self, record):
        self.messages.append({
            "logger": record.name, "message": record.getMessage(),
            "transformers_stack": [
                {"file": f.filename.split("site-packages")[-1].lstrip("/\\"),
                 "line": f.lineno, "function": f.name}
                for f in traceback.extract_stack() if "site-packages" in f.filename
                and "transformers" in f.filename
            ],
        })


def write_report(output, metadata, rows):
    def number(value):
        return "undefined (n=1)" if value is None else f"{value:.6f}"

    lines = ["# Experiment 001 — Nafie morphology baseline", "",
             f"Status: **{metadata['status']}**. Successful: {metadata['successful_count']}; "
             f"failed: {metadata['failed_count']}; not attempted: {metadata['not_attempted_count']}.", "",
             "## Reproduction and measurement", "", "```powershell",
             ".venv\\Scripts\\python.exe -B scripts/run_morphology_baseline.py",
             "# For another run, specify a fresh directory:",
             ".venv\\Scripts\\python.exe -B scripts/run_morphology_baseline.py --output-dir experiments/experiment_001_repeat",
             "```", "",
             f"Model: `{MODEL_ID}`; model and tokenizer revision: `{REVISION}`.",
             "The existing load_nafie/evaluate_sample/compute_token_nll functions are reused. "
             "Policy leading_space_in_target_v1 measures the leading space with the target, "
             "uses causal logits i-1, CUDA FP16 forwards and FP32 log_softmax. "
             "Batch size 1; no special tokens, padding or truncation. NLL units are nats.", "",
             "Each JSONL row retains the full approved morphology record, its sample hash, "
             "the alignment precheck and the unchanged v2 payload under model_result. "
             "The v2 validator is applied to that payload, not to the experiment envelope. "
             "CSV includes the same row plus directly accessible metric columns; complex cells "
             "and strings use JSON encoding for lossless import (read_results_csv).", "",
             "results.jsonl is flushed and fsynced after every completed sample. Metadata is "
             "checkpointed afterward. KeyboardInterrupt finalizes partial CSV/report; an OS kill "
             "may leave metadata marked running, but completed JSONL lines survive. No automatic "
             "resume or overwrite is performed; use a new directory for another run.", "",
             "## Validation", "", "```json", encode(metadata.get("validation", {})), "```", "",
             "## First four recomputed references", "",
             f"Tolerance: absolute {REFERENCE_TOLERANCE['abs_tol']} nats and relative "
             f"{REFERENCE_TOLERANCE['rel_tol']} (math.isclose); token counts and targets must match exactly. "
             "These tolerances accommodate small floating-point changes, not changed scoring policies.", "",
             "| ID | Tokens old/new | Word NLL old/new | Normalized NLL old/new | Word delta | PASS |",
             "|---|---|---|---|---|---|"]
    for r in metadata.get("reference_comparisons", []):
        lines.append(f"| {r['sample_id']} | {r['expected_token_count']}/{r['actual_token_count']} | "
                     f"{r['expected_word_nll']}/{r['actual_word_nll']} | "
                     f"{r['expected_normalized_nll']}/{r['actual_normalized_nll']} | "
                     f"{r['word_nll_delta']} | {r['pass']} |")
    lines += ["", "## Descriptive statistics", "",
              "Sample standard deviation uses n-1; it is undefined for singleton groups. "
              "Summaries include successful rows only. These 40 controlled examples support "
              "descriptive comparisons only; no correlation, significance or causal claim is made.", "",
              "| Group | n | Word mean | Median | SD | Min | Max | Normalized mean | Median | Tokens mean | Affixes mean |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    stats = metadata.get("statistics", summaries(rows))
    groups = [("All", stats["overall"])]
    groups += [(f"{field}={key}", value) for field in ("by_pos", "by_affix_count")
               for key, value in stats[field].items()]
    for label, group in groups:
        if group["n"]:
            lines.append(f"| {label} | {group['n']} | " + " | ".join(number(group[field]) for field in (
                "word_nll_mean", "word_nll_median", "word_nll_sample_stdev", "word_nll_min",
                "word_nll_max", "normalized_nll_mean", "normalized_nll_median",
                "token_count_mean", "affix_count_mean")) + " |")
    lines += ["", "## Runtime and memory", "", "```json", encode({key: metadata.get(key) for key in (
        "started_at_utc", "finished_at_utc", "duration_seconds", "load_seconds", "inference_seconds",
        "gpu", "memory_before_load", "memory_after_load", "memory_after_evaluation", "environment")}),
        "```", "", "## Transformers warning assessment", "", WARNING_ASSESSMENT, "", "```json",
        encode({key: metadata.get(key) for key in ("runtime_architecture", "loading_info", "load_warnings")}),
        "```", "", "Snapshot config declares NafieForCausalLM with NafieConfig, model_type=nafie, "
        "36 layers, hidden size 1024, vocabulary 65536, maximum sequence length 1024. "
        "The custom model config_class is NafieConfig. All seven snapshot files are hash-verified "
        "by the existing loader before custom code runs. Source checkpoint dtype=float32 is "
        "intentionally loaded as FP16 under the existing scoring profile.", "",
        "## Failures and limits", "", "```json", encode({
            "run_error": metadata.get("error"), "validation_error": metadata.get("validation_error"),
            "unsuccessful_rows": [{"sample_id": r["sample_id"], "status": r["status"], "error": r["error"]}
                                  for r in rows if r["status"] != "ok"],
            "failed_references": [r for r in metadata.get("reference_comparisons", []) if not r["pass"]],
        }), "```", "",
        "Reference values were supplied as terminal observations, without a separately archived "
        "old environment manifest. This run records the pinned revision, scoring profile and "
        "installed versions explicitly. Reproducibility across other GPU/software versions is "
        "not established. The upstream attention mask limitation requires single unpadded sequences.",
        "", "## Regression checks", "", encode(metadata.get("regression_tests", {"status": "not recorded"})),
        "", "## Remaining work", "", "Human review of these experiment artifacts and commit approval. "
        "No commit is created by this runner.", ""]
    (output / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_experiment(output=OUTPUT):
    # All input checks and the exclusive directory claim precede expensive loading.
    samples = preflight()
    output = Path(output)
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (DATASET, REVIEW, EVIDENCE)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    started = perf_counter()
    metadata = {
        "experiment_id": "experiment_001_nafie_morphology", "status": "running",
        "started_at_utc": utc_now(), "finished_at_utc": None,
        "model_id": MODEL_ID, "model_revision": REVISION, "tokenizer_revision": REVISION,
        "measurement_policy": MEASUREMENT_POLICY,
        "scoring_profile": "single_cuda_fp16_logsoftmax_fp32_v1",
        "scoring_settings": {"batch_size": 1, "model_dtype": "torch.float16",
                             "log_softmax_dtype": "torch.float32", "causal_logit_shift": -1,
                             "nll_reduction": "Python sum over FP32 token log probabilities"},
        "input_sha256": input_hashes, "dataset_sha256": sha256(DATASET),
        "approved_count": 40, "pending_count": 0,
        "morphological_analysis_count": sum(len(s["morphological_analyses"]) for s in samples),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "scripts/validate_dataset.py",
            ROOT / "src/turkish_llm_morphology/token_nll.py",
            ROOT / "src/turkish_llm_morphology/token_boundary.py",
            ROOT / "src/turkish_llm_morphology/models/nafie.py",
            ROOT / "src/turkish_llm_morphology/morphology/dataset.py")},
        "reference_tolerance": REFERENCE_TOLERANCE, "load_warnings": [],
        "warning_assessment": WARNING_ASSESSMENT,
        "successful_count": 0, "failed_count": 0, "not_attempted_count": 40,
    }
    save_metadata(output, metadata)
    rows = []
    try:
        with (output / "results.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
            import torch

            metadata["environment"] = {
                "python": platform.python_version(), "platform": platform.platform(),
                **{name: version(name) for name in (
                    "torch", "transformers", "tokenizers", "huggingface_hub", "safetensors")},
                "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            }
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")
            torch.cuda.set_device(0)
            torch.cuda.reset_peak_memory_stats(0)
            metadata["gpu"] = torch.cuda.get_device_name(0)
            metadata["memory_before_load"] = cuda_memory()
            config = {"tokenizer_id": MODEL_ID, "tokenizer_revision": REVISION,
                      "transformers_version": version("transformers"),
                      "tokenizers_version": version("tokenizers"), **TOKENIZATION_SETTINGS}
            metadata["tokenization_config"] = config
            handler = LoadWarningRecorder(metadata["load_warnings"])
            logger = logging.getLogger("transformers.configuration_utils")
            logger.addHandler(handler)
            load_started = perf_counter()
            try:
                model, tokenizer, load_metadata = load_nafie()
            finally:
                logger.removeHandler(handler)
            torch.cuda.synchronize(0)
            metadata.update(load_metadata)
            metadata["load_seconds"] = perf_counter() - load_started
            metadata["memory_after_load"] = cuda_memory()
            metadata["runtime_architecture"] = {
                "model_class": type(model).__name__, "config_class": type(model.config).__name__,
                "model_type": model.config.model_type, "architectures": model.config.architectures,
                "tokenizer_class": type(tokenizer).__name__, "model_dtype": str(next(model.parameters()).dtype),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "config_dimensions": {name: getattr(model.config, name, None) for name in (
                    "vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings")},
            }
            save_metadata(output, metadata)
            inference_started = perf_counter()
            try:
                for sample in samples:
                    row = {**sample, "sample_sha256": record_sha256(sample), "status": "computation_error",
                           "alignment_check": None, "model_result": None, "error": None}
                    try:
                        row["alignment_check"] = check_alignment(tokenizer, sample["prefix"], sample["target"])
                        result = evaluate_sample(model, tokenizer, sample, config)
                        validate_result_records([result], samples)
                        row.update(model_result=result, status=result["status"], error=result["error"])
                    except Exception as exc:
                        row["error"] = {"type": type(exc).__name__, "message": str(exc),
                                        "traceback": traceback.format_exc()}
                    journal.write(encode(row) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    rows.append(row)
                    metadata.update(successful_count=sum(r["status"] == "ok" for r in rows),
                                    failed_count=sum(r["status"] != "ok" for r in rows),
                                    not_attempted_count=40 - len(rows))
                    save_metadata(output, metadata)
                    print(f"{sample['sample_id']}: {row['status']}", flush=True)
                    if len(rows) == 4 and not all(r["pass"] for r in compare_references(rows)):
                        raise RuntimeError("First-four reference mismatch; remaining inference halted for investigation")
            finally:
                torch.cuda.synchronize(0)
                metadata["inference_seconds"] = perf_counter() - inference_started
                metadata["memory_after_evaluation"] = cuda_memory()
    except (Exception, KeyboardInterrupt) as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        metadata["error"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
    # Re-read the durable journal; validation must cover persisted bytes, not only memory.
    rows = read_jsonl(output / "results.jsonl")
    write_results_csv(rows, output / "results.csv")
    metadata["reference_comparisons"] = compare_references(rows)
    metadata["statistics"] = summaries(rows)
    try:
        metadata["validation"] = validate_outputs(rows, samples, output, input_hashes)
        checks = metadata["validation"]
        passed = (checks["exact_40_ids"] and checks["alignment_ok_count"] == 40
                  and checks["v2_schema_valid_count"] == 40 and checks["sample_hash_match_count"] == 40
                  and checks["finite_valid_nll_count"] == 40 and checks["inputs_unchanged"]
                  and checks["csv_jsonl_equal"] and all(r["pass"] for r in metadata["reference_comparisons"]))
        if metadata["status"] == "running":
            metadata["status"] = "ok" if passed else "validation_failed"
    except Exception as exc:
        metadata["status"] = "validation_failed"
        metadata["validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
    metadata["finished_at_utc"] = utc_now()
    metadata["duration_seconds"] = perf_counter() - started
    metadata["output_sha256"] = {name: sha256(output / name) for name in ("results.jsonl", "results.csv")}
    save_metadata(output, metadata)
    write_report(output, metadata, rows)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="must not already exist")
    args = parser.parse_args()
    metadata = run_experiment(args.output_dir)
    print(encode({key: metadata[key] for key in (
        "status", "successful_count", "failed_count", "not_attempted_count", "duration_seconds")}))
    raise SystemExit(0 if metadata["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
