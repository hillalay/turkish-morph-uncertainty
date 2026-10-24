"""Validate dataset and model-result JSONL files against the shared contract."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "data" / "schema"
SAMPLE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


class ContractValidationError(ValueError):
    """Raised when a JSONL record violates the data contract."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractValidationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ContractValidationError(f"{path}:{line_number}: record must be a JSON object")
            records.append(record)
    return records


def _validate_schema(records: list[dict[str, Any]], schema_name: str) -> None:
    schema = json.loads((SCHEMA_DIR / schema_name).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for index, record in enumerate(records, start=1):
        errors = sorted(
            validator.iter_errors(record),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.path) or "<record>"
            raise ContractValidationError(f"record {index}, {location}: {error.message}")


def validate_dataset_records(records: list[dict[str, Any]]) -> None:
    _validate_schema(records, "dataset.schema.json")
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        sample_id = record["sample_id"]
        if SAMPLE_ID_PATTERN.fullmatch(sample_id) is None:
            raise ContractValidationError(f"record {index}: invalid sample_id {sample_id!r}")
        if sample_id in seen:
            raise ContractValidationError(f"record {index}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)

        for field in ("sample_id", "prefix", "target", "lemma"):
            value = record[field]
            if value != unicodedata.normalize("NFC", value):
                raise ContractValidationError(f"record {index}: {field} must use Unicode NFC")

        selected_index = record["selected_analysis_index"]
        analyses = record["morphological_analyses"]
        if selected_index >= len(analyses):
            raise ContractValidationError(f"record {index}: selected_analysis_index is out of range")
        selected = analyses[selected_index]
        for field in ("lemma", "pos", "morphological_features"):
            if record[field] != selected[field]:
                raise ContractValidationError(
                    f"record {index}: top-level {field} must match the selected analysis"
                )
        if record["affix_count"] != len(selected["affixes"]):
            raise ContractValidationError(
                f"record {index}: affix_count must equal the selected analysis affix count"
            )

        if any(character.isspace() for character in record["target"]):
            raise ContractValidationError(f"record {index}: target must be exactly one word")
        if any(affix["type"] != "inflectional" for affix in selected["affixes"]):
            raise ContractValidationError(f"record {index}: derivational affixes are out of scope")

        features = selected["morphological_features"]
        experiment_type = record["scope"]["experiment_type"]
        if experiment_type == "simple_verb_inflection":
            if selected["pos"] != "VERB" or features["case"] is not None:
                raise ContractValidationError(
                    f"record {index}: simple_verb_inflection requires VERB with no case"
                )
            required_features = ("tense", "mood", "polarity", "person", "number")
            if record["affix_count"] == 0 or any(features[field] is None for field in required_features):
                raise ContractValidationError(
                    f"record {index}: simple_verb_inflection requires an affix and finite verb features"
                )
        elif experiment_type == "noun_case":
            irrelevant_features = ("tense", "mood", "polarity", "person")
            if (
                selected["pos"] not in {"NOUN", "PROPN"}
                or features["case"] in {None, "nominative"}
                or any(features[field] is not None for field in irrelevant_features)
                or record["affix_count"] == 0
            ):
                raise ContractValidationError(
                    f"record {index}: noun_case requires an overtly case-marked noun"
                )


def validate_result_records(
    records: list[dict[str, Any]], dataset_records: list[dict[str, Any]]
) -> None:
    validate_dataset_records(dataset_records)
    _validate_schema(records, "model_result.schema.json")
    samples = {record["sample_id"]: record for record in dataset_records}
    seen: set[tuple[str, str, str, str]] = set()

    for index, record in enumerate(records, start=1):
        key = tuple(
            record[field]
            for field in ("sample_id", "model_id", "model_revision", "evaluation_config_id")
        )
        if key in seen:
            raise ContractValidationError(f"record {index}: duplicate result key {key!r}")
        seen.add(key)

        sample = samples.get(record["sample_id"])
        if sample is None:
            raise ContractValidationError(f"record {index}: unknown sample_id {record['sample_id']!r}")
        if record["status"] == "alignment_error":
            continue

        token_count = record["token_count"]
        logprobs = record["token_logprobs"]
        alignment = record["alignment"]
        indices = alignment["target_token_indices"]
        spans = alignment["target_token_char_spans"]
        if not (token_count == len(logprobs) == len(indices) == len(spans)):
            raise ContractValidationError(f"record {index}: token-aligned arrays must match token_count")
        if indices != list(range(indices[0], indices[0] + token_count)):
            raise ContractValidationError(f"record {index}: target token indices must be contiguous")
        if any(not math.isfinite(value) or value > 0 for value in logprobs):
            raise ContractValidationError(f"record {index}: token_logprobs must be finite and <= 0")

        expected_char_span = [len(sample["prefix"]), len(sample["prefix"]) + len(sample["target"])]
        if alignment["target_char_span"] != expected_char_span:
            raise ContractValidationError(f"record {index}: target_char_span does not match prefix + target")
        if spans[0][0] != expected_char_span[0] or spans[-1][1] != expected_char_span[1]:
            raise ContractValidationError(f"record {index}: token spans do not cover the target exactly")
        if any(start >= end for start, end in spans) or any(
            spans[position][1] != spans[position + 1][0] for position in range(len(spans) - 1)
        ):
            raise ContractValidationError(f"record {index}: target token spans must be non-empty and contiguous")

        expected_nll = -sum(logprobs)
        if not math.isclose(record["word_nll"], expected_nll, rel_tol=1e-9, abs_tol=1e-9):
            raise ContractValidationError(f"record {index}: word_nll does not equal -sum(token_logprobs)")
        expected_normalized = expected_nll / token_count
        if not math.isclose(
            record["token_normalized_nll"], expected_normalized, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ContractValidationError(
                f"record {index}: token_normalized_nll does not equal word_nll / token_count"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("dataset", "results"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--dataset", type=Path, help="input dataset; required for result validation")
    args = parser.parse_args()

    records = read_jsonl(args.path)
    if args.kind == "dataset":
        validate_dataset_records(records)
    else:
        if args.dataset is None:
            parser.error("--dataset is required when kind is 'results'")
        validate_result_records(records, read_jsonl(args.dataset))
    print(f"OK: {len(records)} {args.kind} record(s) validated")


if __name__ == "__main__":
    main()
