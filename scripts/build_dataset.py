"""Build/check the morphology draft; never overwrite the original five-sample pilot."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import logging

from turkish_llm_morphology.morphology import TurkishMorphAnalyzer
from turkish_llm_morphology.morphology.affix_counter import (
    count_affixes, to_contract_analysis, v1_scope,
)
from turkish_llm_morphology.morphology.dataset import (
    read_csv, record_sha256, review_template, validate_manual_review, write_csv,
)
from turkish_llm_morphology.morphology.feature_extractor import extract_features

if __package__:
    from .validate_dataset import ROOT, load_nafie_tokenizer, read_jsonl, validate_morphology_records
else:
    from validate_dataset import ROOT, load_nafie_tokenizer, read_jsonl, validate_morphology_records


PILOT = ROOT / "data/pilot"
SEEDS = PILOT / "morphology_pilot_seeds.jsonl"
DATASET = PILOT / "morphology_pilot_dataset.jsonl"
CSV = PILOT / "pilot_dataset.csv"
ANALYSES = PILOT / "morphology_pilot_analyses.jsonl"
REVIEW = PILOT / "morphology_pilot_review.json"


def build_records(seeds, analyzer):
    records, evidence = [], []
    for seed in seeds:
        analyses = analyzer.analyze(seed["target"])
        matches = [i for i, a in enumerate(analyses) if (
            a.lemma == seed["lemma"] and a.pos == seed["pos"]
            and [m.id for m in a.morphemes] == seed["morphemes"]
        )]
        if len(matches) != 1:
            raise ValueError(f"{seed['sample_id']}: expected one explicit analysis, found {len(matches)}")
        index = matches[0]
        selected = analyses[index]
        projected = [to_contract_analysis(a) for a in analyses]
        record = {
            "sample_id": seed["sample_id"], "prefix": seed["prefix"], "target": seed["target"],
            **{key: projected[index][key] for key in ("lemma", "pos", "morphological_features")},
            "affix_count": count_affixes(selected), "morphological_analyses": projected,
            "selected_analysis_index": index, "scope": v1_scope(selected),
        }
        records.append(record)
        evidence.append({
            "sample_id": record["sample_id"], "sample_sha256": record_sha256(record),
            "zeyrek_version": version("zeyrek"), "nltk_version": version("nltk"),
            "analyses": [{**asdict(a), "features": extract_features(a)} for a in analyses],
        })
    return records, evidence


def jsonl(records):
    return "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated files without writing")
    parser.add_argument("--check-nafie-alignment", action="store_true")
    parser.add_argument("--require-reviewed", action="store_true", help="fail until all >=30 samples have human approval")
    args = parser.parse_args()
    logging.getLogger("zeyrek.rulebasedanalyzer").setLevel(logging.ERROR)
    analyzer = TurkishMorphAnalyzer()
    records, evidence = build_records(read_jsonl(SEEDS), analyzer)
    if not 30 <= len(records) <= 50:
        raise ValueError("The controlled pilot must contain 30–50 samples")
    tokenizer = load_nafie_tokenizer() if args.check_nafie_alignment else None
    validate_morphology_records(records, analyzer, tokenizer)
    review = json.loads(REVIEW.read_text(encoding="utf-8")) if REVIEW.exists() else review_template(records)
    # Refuse stale attestations before changing any generated file.
    expected_reviews = review_template(records)["samples"]
    actual_hashes = [(r["sample_id"], r["sample_sha256"]) for r in review["samples"]]
    if actual_hashes != [(r["sample_id"], r["sample_sha256"]) for r in expected_reviews]:
        raise ValueError("Review manifest is stale; re-review changed samples explicitly")
    if args.require_reviewed:
        validate_manual_review(records, review)
    if args.check:
        if read_jsonl(DATASET) != records or read_jsonl(ANALYSES) != json.loads(json.dumps(evidence)):
            raise ValueError("Saved dataset/evidence does not match current seeds and real analyses")
    else:
        DATASET.write_text(jsonl(records), encoding="utf-8", newline="\n")
        ANALYSES.write_text(jsonl(evidence), encoding="utf-8", newline="\n")
        write_csv(records, CSV)
        if not REVIEW.exists():
            REVIEW.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if read_csv(CSV) != records:
        raise ValueError("CSV/JSONL round-trip mismatch")
    approved = sum(r["status"] == "approved" for r in review["samples"])
    print(f"OK: {len(records)} schema/morphology-valid draft samples; CSV round-trip PASS")
    print(f"Human review: {approved}/{len(records)} marked approved; "
          f"release gate {'PASS' if args.require_reviewed else 'NOT CHECKED (draft)'}")
    if tokenizer is not None:
        print(f"Nafie alignment: {len(records)}/{len(records)} PASS (tokenizer only)")


if __name__ == "__main__":
    main()
