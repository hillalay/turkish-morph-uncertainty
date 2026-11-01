
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "nafie-ai/nafie-473M"
DATASET_PATH = Path("data/pilot/pilot_dataset.jsonl")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    use_fast=True,
)

with DATASET_PATH.open(encoding="utf-8") as f:
    samples = [json.loads(line) for line in f if line.strip()]

for sample in samples:
    prefix = sample["prefix"]
    target = sample["target"]
    full_text = prefix + target

    target_start = len(prefix)
    target_end = len(full_text)

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = encoded["offset_mapping"]

    target_indices = [
        i for i, (start, end) in enumerate(offsets)
        if start < target_end and end > target_start
    ]

    target_spans = [offsets[i] for i in target_indices]

    aligned = (
        bool(target_spans)
        and target_spans[0][0] == target_start
        and target_spans[-1][1] == target_end
        and all(
            a[1] == b[0]
            for a, b in zip(target_spans, target_spans[1:])
        )
        and all(
            start >= target_start and end <= target_end
            for start, end in target_spans
        )
    )

    print(f"\nSample: {sample['sample_id']}")
    print(f"Full text: {full_text!r}")
    print(f"Target: {target!r}")
    print(f"Target span: [{target_start}, {target_end})")

    for i, (token, offset) in enumerate(zip(tokens, offsets)):
        selected = "*" if i in target_indices else " "
        print(f"{selected} {i}: {token!r} {offset}")

    print("Target token count:", len(target_indices))
    print("Alignment:", "PASS" if aligned else "ERROR")
    
    """
    print("\n=== PREFIX/TARGET TOKENIZATION CHECK ===")

    for sample in samples:
        prefix = sample["prefix"]
        target = sample["target"]
        full_text = prefix + target

        prefix_ids = tokenizer(
            prefix, add_special_tokens=False
        )["input_ids"]

        full_ids = tokenizer(
            full_text, add_special_tokens=False
        )["input_ids"]

        prefix_is_stable = (
            full_ids[:len(prefix_ids)] == prefix_ids
        )

        print(f"\n{sample['sample_id']}: {target}")
        print("Prefix token IDs:", prefix_ids)
        print("Full token IDs:", full_ids)
        print("Prefix stable:", prefix_is_stable)
        print("Decoded full:", repr(tokenizer.decode(full_ids)))
"""