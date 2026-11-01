"""Teacher-forced target NLL, with one unpadded sequence and FP32 log-softmax."""

import math

from turkish_llm_morphology.models.nafie import MODEL_ID, REVISION, validate_logits
from turkish_llm_morphology.token_boundary import (
    MEASUREMENT_POLICY, RESULT_SCHEMA_ID, evaluation_config_id, tokenize_aligned_target,
)


def compute_token_nll(logits, input_ids, target_token_indices: list[int], vocab_size: int) -> dict:
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 2:
        raise ValueError("One sequence with at least one context and one target token is required")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must have dtype torch.long")
    if logits.ndim != 3 or not torch.is_floating_point(logits):
        raise ValueError("logits must be a floating-point [1, sequence_length, vocab_size] tensor")
    validate_logits(logits, tuple(input_ids.shape), vocab_size)
    if logits.device != input_ids.device:
        raise ValueError("logits and input_ids must be on the same device")
    if vocab_size < 1 or input_ids.min().item() < 0 or input_ids.max().item() >= vocab_size:
        raise ValueError("input_ids are outside the model vocabulary")
    if (not target_token_indices
            or any(type(index) is not int for index in target_token_indices)
            or target_token_indices[0] < 1
            or target_token_indices != list(range(target_token_indices[0], input_ids.shape[1]))):
        raise ValueError("Target indices must be a contiguous suffix with a preceding context token")

    start = target_token_indices[0]
    with torch.inference_mode():
        shifted_logits = logits[0, start - 1:-1, :].float()
        if not torch.isfinite(shifted_logits).all().item():
            raise RuntimeError("Target logits cannot be represented as finite FP32 values")
        target_ids = input_ids[0, start:]
        logprobs = torch.log_softmax(shifted_logits, dim=-1).gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1).tolist()
    if any(not math.isfinite(value) or value > 0 for value in logprobs):
        raise RuntimeError("Target token log probabilities must be finite and <= 0")
    word_nll = -sum(logprobs)
    return {
        "token_count": len(logprobs), "token_logprobs": logprobs,
        "word_nll": word_nll, "token_normalized_nll": word_nll / len(logprobs),
    }


def evaluate_sample(model, tokenizer, sample: dict, tokenization_config: dict) -> dict:
    """Return a v2 record; computation failures raise instead of fabricating a result."""
    aligned = tokenize_aligned_target(tokenizer, sample["prefix"], sample["target"])
    full_ids = aligned.pop("input_ids", None)
    record = {
        "schema_id": RESULT_SCHEMA_ID, "sample_id": sample["sample_id"],
        "model_id": MODEL_ID, "model_revision": REVISION,
        "measurement_policy": MEASUREMENT_POLICY,
        "tokenization_config": tokenization_config,
        "evaluation_config_id": evaluation_config_id(MEASUREMENT_POLICY, tokenization_config),
        "token_count": None, "token_logprobs": None,
        "word_nll": None, "token_normalized_nll": None,
        **aligned,
    }
    if record["status"] == "alignment_error":
        return record

    import torch

    if len(full_ids) > model.config.max_position_embeddings:
        raise ValueError("Input exceeds the model context window; truncation is disabled")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    model.eval()
    # Upstream ignores attention_mask; this single sequence has no padding.
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                        return_dict=True, output_hidden_states=False)
        record.update(compute_token_nll(
            outputs.logits, input_ids, record["alignment"]["target_token_indices"],
            model.config.vocab_size,
        ))
    return record
