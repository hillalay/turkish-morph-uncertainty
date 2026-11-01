"""One pinned, audited Nafie snapshot; CUDA FP16 forward only, with no scoring."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from turkish_llm_morphology.token_boundary import TOKENIZATION_SETTINGS


ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / ".cache/huggingface/hub"
REPORT_PATH = ROOT / "results/raw/nafie_473m_forward_probe.json"
MODEL_ID = "nafie-ai/nafie-473M"
REVISION = "20477766abe4ca9e5ef3ef8891164a346887bba8"
TEXT = "Dün okula geldim"
# Review and provenance: docs/nafie_forward_probe.md. Re-review before changing these.
REVIEWED_SHA256 = {
    "configuration_nafie.py": "5e504f0db6380cc3f1a9a216d5939505a617911deb8bdd9a4433e2a9beb033f9",
    "modeling_nafie.py": "6ca6801bfe72db3b53388dad055e8362afe129e4465046eb01359caf29480120",
    "config.json": "fd9518bb374c5495f86c76fb8f1c844ec8bab7806de09a00d6f2273e2e47f5a9",
    "generation_config.json": "2a72d248d738a6f4650e95f8f84ca0d29244647f7f2487b33ac7a65b64edbcdb",
    "tokenizer.json": "01125b1e34c915563546552555283a7fd183a522b66f856e8a9cebf0ab70b42a",
    "tokenizer_config.json": "a1a584f73974811f7eff57a83203e499bf54186ed49f0897b50b32700c5ce183",
    "model.safetensors": "3e5089e03079b26a1836b355189eceb158daefe2a1248043b5bfd8226ce3f963",
}


def verify_snapshot(snapshot: Path) -> dict[str, str]:
    verified = {}
    for name, expected in REVIEWED_SHA256.items():
        digest = hashlib.sha256()
        with (snapshot / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"Reviewed snapshot SHA-256 mismatch: {name}")
        verified[name] = actual
    return verified


def load_nafie():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(
        MODEL_ID, revision=REVISION, cache_dir=CACHE_DIR,
        allow_patterns=list(REVIEWED_SHA256), local_files_only=True,
    ))
    verified = verify_snapshot(snapshot)
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".cache/huggingface/modules"))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, cache_dir=CACHE_DIR,
        local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, code_revision=REVISION, cache_dir=CACHE_DIR,
        local_files_only=True, trust_remote_code=True, use_safetensors=True,
        weights_only=True, dtype=torch.float16, output_loading_info=True,
    )
    if any(loading_info.get(key) for key in (
        "missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"
    )):
        raise RuntimeError(f"Incomplete checkpoint loading: {loading_info}")
    model = model.to("cuda:0").eval()
    if any(p.device.type != "cuda" or p.dtype != torch.float16 for p in model.parameters()):
        raise RuntimeError("All model parameters must be CUDA FP16")
    return model, tokenizer, {
        "verified_sha256": verified,
        "loading_info": {key: sorted(value) for key, value in loading_info.items()},
    }


def validate_logits(logits, input_shape: tuple[int, int], vocab_size: int) -> None:
    import torch

    expected = (*input_shape, vocab_size)
    if tuple(logits.shape) != expected:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}; expected {expected}")
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Logits contain NaN or infinity")


def cuda_memory() -> dict[str, int]:
    import torch

    free, total = torch.cuda.mem_get_info(0)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "device_free_bytes": free, "device_total_bytes": total,
    }


def run_forward_probe() -> dict:
    report = {
        "report_kind": "forward_probe_not_model_result",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID, "model_revision": REVISION, "code_revision": REVISION,
        "text": TEXT, "status": "error", "stage": "environment", "model_loaded": False,
    }
    try:
        import torch

        report["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: version(name) for name in (
                "torch", "transformers", "tokenizers", "huggingface_hub", "safetensors"
            )},
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        }
        report["stage"] = "cuda_check"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; CPU fallback is disabled")
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
        report["memory_before_load"] = cuda_memory()
        report["stage"] = "model_loading"
        started = perf_counter()
        model, tokenizer, metadata = load_nafie()
        torch.cuda.synchronize(0)
        report.update(metadata)
        report.update(model_loaded=True, model_class=type(model).__name__,
                      model_dtype=str(next(model.parameters()).dtype),
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      load_seconds=perf_counter() - started,
                      memory_after_load=cuda_memory())
        report["stage"] = "tokenization"
        inputs = tokenizer(
            TEXT, **{key: TOKENIZATION_SETTINGS[key] for key in (
                "add_special_tokens", "padding", "truncation"
            )}, return_tensors="pt", return_token_type_ids=False,
        )
        ids = inputs["input_ids"][0].tolist()
        if any(token in tokenizer.all_special_ids for token in ids):
            raise RuntimeError("Special tokens are forbidden in the boundary-v2-compatible input")
        if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != TEXT:
            raise RuntimeError("Tokenizer decode does not preserve the input text")
        # shortcut: upstream attention_mask is ignored, restrict to one unpadded sequence until adapted.
        if inputs["input_ids"].shape[0] != 1 or not inputs["attention_mask"].eq(1).all().item():
            raise RuntimeError("This prototype supports only one unpadded sequence")
        report["tokenization"] = {
            "settings": TOKENIZATION_SETTINGS,
            "input_ids": ids, "tokens": tokenizer.convert_ids_to_tokens(ids),
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "bos_in_input": tokenizer.bos_token_id in ids,
            "eos_in_input": tokenizer.eos_token_id in ids,
            "ids_with_add_special_tokens_true": tokenizer(TEXT, add_special_tokens=True)["input_ids"],
        }
        inputs = inputs.to("cuda:0")
        report["stage"] = "forward"
        started = perf_counter()
        with torch.inference_mode():
            outputs = model(**inputs, return_dict=True, output_hidden_states=False)
            logits = outputs.logits
            validate_logits(logits, tuple(inputs["input_ids"].shape), model.config.vocab_size)
            if logits.device.type != "cuda":
                raise RuntimeError("Logits must reside on CUDA")
        torch.cuda.synchronize(0)
        report.update(
            status="ok", stage="complete", logits_shape=list(logits.shape),
            logits_dtype=str(logits.dtype), logits_device=str(logits.device), logits_all_finite=True,
            logits_requires_grad=logits.requires_grad, forward_seconds=perf_counter() - started,
            memory_after_forward=cuda_memory(),
        )
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> None:
    report = run_forward_probe()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
