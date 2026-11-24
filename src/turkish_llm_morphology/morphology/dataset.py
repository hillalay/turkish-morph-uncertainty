"""Lossless v1 CSV export and content-bound human review checks."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

FIELDS = (
    "sample_id", "prefix", "target", "lemma", "pos", "morphological_features",
    "affix_count", "morphological_analyses", "selected_analysis_index", "scope",
)
JSON_FIELDS = ("morphological_features", "morphological_analyses", "scope")
INT_FIELDS = ("affix_count", "selected_analysis_index")


def record_sha256(record: dict) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = dict(record)
            for field in JSON_FIELDS:
                row[field] = json.dumps(row[field], ensure_ascii=False, allow_nan=False)
            writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("CSV header must match the v1 dataset fields exactly")
        records = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("CSV row has missing or extra columns")
            for field in JSON_FIELDS:
                row[field] = json.loads(row[field])
            for field in INT_FIELDS:
                row[field] = int(row[field])
            records.append(row)
    return records


def review_template(records: list[dict]) -> dict:
    return {"dataset_version": "morphology-pilot-v1-draft", "samples": [
        {"sample_id": r["sample_id"], "sample_sha256": record_sha256(r),
         "status": "pending", "reviewer": None, "reviewed_at": None, "notes": ""}
        for r in records
    ]}


def validate_manual_review(records: list[dict], review: dict) -> None:
    """Only an explicit reviewer attestation can make the draft release-ready."""
    entries = review["samples"]
    by_id = {entry["sample_id"]: entry for entry in entries}
    if len(by_id) != len(entries) or set(by_id) != {r["sample_id"] for r in records}:
        raise ValueError("Review entries must match the dataset IDs without duplicates")
    if len(records) < 30:
        raise ValueError("At least 30 reviewed records are required")
    for record in records:
        entry = by_id[record["sample_id"]]
        if entry["sample_sha256"] != record_sha256(record):
            raise ValueError(f"Stale review: {record['sample_id']}")
        if (entry["status"] != "approved" or not isinstance(entry["reviewer"], str)
                or not entry["reviewer"].strip()):
            raise ValueError(f"Human review pending: {record['sample_id']}")
        if not isinstance(entry["reviewed_at"], str):
            raise ValueError(f"Review date missing: {record['sample_id']}")
        datetime.fromisoformat(entry["reviewed_at"])
