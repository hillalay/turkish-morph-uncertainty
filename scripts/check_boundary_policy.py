
import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL_ID = "nafie-ai/nafie-473M"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=False,
    use_fast=True,
)

path = Path("data/pilot/pilot_dataset.jsonl")

samples = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

passed = 0

for sample in samples:
    prefix = sample["prefix"]
    target = sample["target"]

    # Yeni politika yalnızca tam bir sondaki boşluk için
    # test ediliyor.
    if not prefix.endswith(" "):
        print(sample["sample_id"], "SKIP: trailing space yok")
        continue

    context = prefix[:-1]
    measured_target = " " + target
    full_text = context + measured_target

    context_ids = tokenizer(
        context, add_special_tokens=False
    )["input_ids"]

    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    full_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    stable = full_ids[:len(context_ids)] == context_ids

    target_ids = full_ids[len(context_ids):]
    target_start = len(context)

    spans = offsets[len(context_ids):]

    exact_coverage = (
        bool(spans)
        and spans[0][0] == target_start
        and spans[-1][1] == len(full_text)
        and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    )

    decoded = tokenizer.decode(
        full_ids, skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )

    roundtrip = decoded == full_text
    ok = stable and exact_coverage and roundtrip

    passed += int(ok)

    print(f"\n{sample['sample_id']} - {target}")
    print("Context:", repr(context))
    print("Measured target:", repr(measured_target))
    print("Target IDs:", target_ids)
    print("Prefix stable:", stable)
    print("Exact coverage:", exact_coverage)
    print("Roundtrip:", roundtrip)
    print("Result:", "PASS" if ok else "ERROR")

print(f"\nTOTAL: {passed}/{len(samples)} PASS")
