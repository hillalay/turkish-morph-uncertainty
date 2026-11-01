"""Score the five pilot samples separately with the pinned CUDA FP16 Nafie model."""

import hashlib
import json
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import version

from turkish_llm_morphology.models.nafie import MODEL_ID, REVISION, cuda_memory, load_nafie
from turkish_llm_morphology.token_boundary import (
    MEASUREMENT_POLICY, TOKENIZATION_SETTINGS, evaluation_config_id,
)
from turkish_llm_morphology.token_nll import evaluate_sample

if __package__:
    from .validate_dataset import ROOT, read_jsonl, validate_dataset_records, validate_result_records
else:
    from validate_dataset import ROOT, read_jsonl, validate_dataset_records, validate_result_records


DATASET_PATH = ROOT / "data/pilot/pilot_dataset.jsonl"
RESULTS_PATH = ROOT / "results/raw/nafie_473m_pilot_nll.v2.jsonl"
REPORT_PATH = ROOT / "results/raw/nafie_473m_pilot_nll_run.json"


def run_pilot() -> tuple[list[dict], dict]:
    records = []
    report = {
        "report_kind": "nll_run_not_model_result", "status": "error",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "model_revision": REVISION, "code_revision": REVISION,
        "scoring_profile": "single_cuda_fp16_logsoftmax_fp32_v1",
        "scoring_settings": {
            "batch_size": 1, "padding": False, "model_dtype": "torch.float16",
            "log_softmax_dtype": "torch.float32", "causal_logit_shift": -1,
            "nll_reduction": "Python sum over FP32 token log probabilities",
        },
        "stage": "dataset_validation", "computation_errors": [], "results_validated": False,
    }
    try:
        samples = read_jsonl(DATASET_PATH)
        validate_dataset_records(samples)
        if not samples:
            raise ValueError("The pilot dataset must not be empty")
        report["input_count"] = len(samples)
        report["dataset_sha256"] = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        report["stage"] = "environment"
        import torch

        report["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in (
                "torch", "transformers", "tokenizers", "huggingface_hub", "safetensors"
            )},
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        }
        config = {
            "tokenizer_id": MODEL_ID, "tokenizer_revision": REVISION,
            "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
            **TOKENIZATION_SETTINGS,
        }
        # shortcut: v2 hashes tokenization only, version the contract before comparing scoring precisions.
        report["evaluation_config_id"] = evaluation_config_id(MEASUREMENT_POLICY, config)
        report["stage"] = "model_loading"
        model, tokenizer, metadata = load_nafie()
        report.update(metadata)
        report["gpu"] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats(0)
        report["stage"] = "sample_evaluation"
        for sample in samples:
            try:
                record = evaluate_sample(model, tokenizer, sample, config)
                validate_result_records([record], samples)
            except Exception as exc:
                report["computation_errors"].append({
                    "sample_id": sample["sample_id"], "status": "computation_error",
                    "token_count": None, "token_logprobs": None,
                    "word_nll": None, "token_normalized_nll": None,
                    "error": {"code": "NLL_COMPUTATION_FAILED", "type": type(exc).__name__,
                              "message": str(exc), "traceback": traceback.format_exc()},
                })
                continue
            records.append(record)
        validate_result_records(records, samples)
        report.update(
            results_validated=True, result_count=len(records),
            ok_count=sum(record["status"] == "ok" for record in records),
            alignment_error_count=sum(record["status"] == "alignment_error" for record in records),
            computation_error_count=len(report["computation_errors"]),
            memory_after_evaluation=cuda_memory(), stage="complete",
        )
        if report["ok_count"] == len(samples):
            report["status"] = "ok"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return records, report


def main() -> None:
    records, report = run_pilot()
    if report["results_validated"]:
        text = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records)
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = RESULTS_PATH.with_suffix(".jsonl.tmp")
        temporary.write_bytes(text.encode("utf-8"))
        temporary.replace(RESULTS_PATH)
        report["results_file"] = RESULTS_PATH.name
        report["results_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
