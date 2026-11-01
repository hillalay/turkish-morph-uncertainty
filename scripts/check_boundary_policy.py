"""Run tokenizer-only leading-space diagnostics; this does not produce NLL results."""

import argparse
import json
import re
from importlib.metadata import version
from pathlib import Path

from turkish_llm_morphology.token_boundary import (
    MEASUREMENT_POLICY,
    TOKENIZATION_SETTINGS,
    check_alignment,
    evaluation_config_id,
)

if __package__:
    from .validate_dataset import ROOT, read_jsonl, validate_dataset_records
else:
    from validate_dataset import ROOT, read_jsonl, validate_dataset_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="nafie-ai/nafie-473M")
    parser.add_argument("--revision", required=True, help="immutable tokenizer snapshot/commit")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/pilot/pilot_dataset.jsonl")
    args = parser.parse_args()
    if re.fullmatch(r"[a-f0-9]{40}", args.revision) is None:
        parser.error("--revision must be a full immutable Hugging Face commit hash (40 lowercase hex characters)")
    samples = read_jsonl(args.dataset)
    validate_dataset_records(samples)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.revision, trust_remote_code=False, use_fast=True,
    )
    config = {
        "tokenizer_id": args.model_id,
        "tokenizer_revision": args.revision,
        "transformers_version": version("transformers"),
        "tokenizers_version": version("tokenizers"),
        **TOKENIZATION_SETTINGS,
    }
    config_id = evaluation_config_id(MEASUREMENT_POLICY, config)
    passed = 0
    for sample in samples:
        diagnostic = check_alignment(tokenizer, sample["prefix"], sample["target"])
        passed += diagnostic["status"] == "ok"
        print(json.dumps({
            "sample_id": sample["sample_id"],
            "measurement_policy": MEASUREMENT_POLICY,
            "tokenization_config": config,
            "evaluation_config_id": config_id,
            **diagnostic,
        }, ensure_ascii=False))
    print(f"TOTAL: {passed}/{len(samples)} PASS")


if __name__ == "__main__":
    main()
