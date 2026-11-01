"""Versioned tokenizer alignment only; no model loading or metric computation."""

from __future__ import annotations

import hashlib
import json
import unicodedata


MEASUREMENT_POLICY = "leading_space_in_target_v1"
RESULT_SCHEMA_ID = "urn:turkish-llm-morphology:schema:model-result:v2"
TOKENIZATION_SETTINGS = {
    "use_fast": True,
    "add_special_tokens": False,
    "padding": False,
    "truncation": False,
    "skip_special_tokens": False,
    "clean_up_tokenization_spaces": False,
}


class AlignmentError(ValueError):
    """The requested target cannot be scored safely under this policy."""


def evaluation_config_id(measurement_policy: str, tokenization_config: dict) -> str:
    payload = json.dumps(
        {"measurement_policy": measurement_policy, "tokenization_config": tokenization_config},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return f"{measurement_policy}:sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def target_char_spans(prefix: str, target: str, measurement_policy: str) -> tuple[list, list]:
    if measurement_policy != MEASUREMENT_POLICY:
        raise AlignmentError(f"Unsupported measurement_policy: {measurement_policy!r}")
    if not prefix.endswith(" ") or (len(prefix) > 1 and prefix[-2].isspace()):
        raise AlignmentError("prefix must end in exactly one ASCII space (U+0020)")
    if not target or any(character.isspace() for character in target):
        raise AlignmentError("target must be exactly one non-empty word")
    if any(text != unicodedata.normalize("NFC", text) for text in (prefix, target)):
        raise AlignmentError("prefix and target must use Unicode NFC")
    end = len(prefix) + len(target)
    return [len(prefix), end], [len(prefix) - 1, end]


def validate_token_spans(indices: list, spans: list, target_span: list) -> None:
    """Check coverage after the caller has validated index/offset types."""
    if not indices or len(indices) != len(spans):
        raise AlignmentError("target token indices and spans must have equal non-zero lengths")
    if indices != [indices[0] + position for position in range(len(indices))]:
        raise AlignmentError("target token indices must be contiguous")
    if spans[0][0] != target_span[0] or spans[-1][1] != target_span[1]:
        raise AlignmentError("token spans do not cover the target exactly")
    if any(start >= end for start, end in spans) or any(
        left[1] != right[0] for left, right in zip(spans, spans[1:])
    ):
        raise AlignmentError("target token spans must be non-empty and contiguous")


def check_alignment(tokenizer, prefix: str, target: str,
                    measurement_policy: str = MEASUREMENT_POLICY) -> dict:
    """Return an alignment diagnostic; callers must gate scoring on status='ok'."""
    try:
        linguistic_span, measured_span = target_char_spans(prefix, target, measurement_policy)
        if not tokenizer.is_fast:
            raise AlignmentError("a fast tokenizer with Unicode code-point offsets is required")
        context = prefix[:-1]
        full_text = prefix + target
        encode_settings = {key: TOKENIZATION_SETTINGS[key] for key in (
            "add_special_tokens", "padding", "truncation"
        )}
        context_ids = tokenizer(context, **encode_settings)["input_ids"]
        encoded = tokenizer(full_text, return_offsets_mapping=True, **encode_settings)
        full_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if not context_ids:
            raise AlignmentError("no context token precedes the target; BOS is disabled")
        if any(type(value) is not int or value < 0 for value in context_ids + full_ids):
            raise AlignmentError("token IDs must be non-negative integers")
        if full_ids[:len(context_ids)] != context_ids:
            raise AlignmentError("context token IDs changed in full-text tokenization")
        if any(value in tokenizer.all_special_ids for value in context_ids + full_ids):
            raise AlignmentError("special tokens, including BOS/EOS, are not allowed")
        if len(full_ids) != len(offsets) or any(
            len(span) != 2
            or any(type(value) is not int for value in span)
            or not 0 <= span[0] < span[1] <= len(full_text)
            for span in offsets
        ):
            raise AlignmentError("invalid Unicode code-point offsets")
        if any(end > measured_span[0] for _, end in offsets[:len(context_ids)]):
            raise AlignmentError("context token offsets cross the measured target boundary")
        indices = list(range(len(context_ids), len(full_ids)))
        spans = [list(span) for span in offsets[len(context_ids):]]
        validate_token_spans(indices, spans, measured_span)
        decode_settings = {key: TOKENIZATION_SETTINGS[key] for key in (
            "skip_special_tokens", "clean_up_tokenization_spaces"
        )}
        if (tokenizer.decode(full_ids, **decode_settings) != full_text
                or tokenizer.decode(context_ids, **decode_settings) != context):
            raise AlignmentError("tokenizer decode does not preserve the original text")
        return {
            "status": "ok",
            "alignment": {
                "linguistic_target_char_span": linguistic_span,
                "measured_target_char_span": measured_span,
                "target_token_indices": indices,
                "target_token_char_spans": spans,
            },
            "error": None,
        }
    except (AlignmentError, ValueError, TypeError, KeyError, AttributeError, NotImplementedError) as exc:
        return {
            "status": "alignment_error",
            "alignment": None,
            "error": {"code": "TARGET_ALIGNMENT_FAILED", "message": str(exc) or type(exc).__name__},
        }
