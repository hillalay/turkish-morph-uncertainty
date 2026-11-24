# Experiment 001 — Nafie morphology baseline

Status: **ok**. Successful: 40; failed: 0; not attempted: 0.

## Reproduction and measurement

```powershell
.venv\Scripts\python.exe -B scripts/run_morphology_baseline.py
# For another run, specify a fresh directory:
.venv\Scripts\python.exe -B scripts/run_morphology_baseline.py --output-dir experiments/experiment_001_repeat
```

Model: `nafie-ai/nafie-473M`; model and tokenizer revision: `20477766abe4ca9e5ef3ef8891164a346887bba8`.
The existing load_nafie/evaluate_sample/compute_token_nll functions are reused. Policy leading_space_in_target_v1 measures the leading space with the target, uses causal logits i-1, CUDA FP16 forwards and FP32 log_softmax. Batch size 1; no special tokens, padding or truncation. NLL units are nats.

Each JSONL row retains the full approved morphology record, its sample hash, the alignment precheck and the unchanged v2 payload under model_result. The v2 validator is applied to that payload, not to the experiment envelope. CSV includes the same row plus directly accessible metric columns; complex cells and strings use JSON encoding for lossless import (read_results_csv).

results.jsonl is flushed and fsynced after every completed sample. Metadata is checkpointed afterward. KeyboardInterrupt finalizes partial CSV/report; an OS kill may leave metadata marked running, but completed JSONL lines survive. No automatic resume or overwrite is performed; use a new directory for another run.

## Validation

```json
{"exact_40_ids": true, "alignment_ok_count": 40, "v2_schema_valid_count": 40, "sample_hash_match_count": 40, "inputs_unchanged": true, "csv_jsonl_equal": true, "finite_valid_nll_count": 40}
```

## First four recomputed references

Tolerance: absolute 0.0001 nats and relative 1e-05 (math.isclose); token counts and targets must match exactly. These tolerances accommodate small floating-point changes, not changed scoring policies.

| ID | Tokens old/new | Word NLL old/new | Normalized NLL old/new | Word delta | PASS |
|---|---|---|---|---|---|
| morph-v1-001 | 1/1 | 5.2880983353/5.288098335266113 | 5.2880983353/5.288098335266113 | -3.388667124681888e-11 | True |
| morph-v1-002 | 1/1 | 9.1152467728/9.115246772766113 | 9.1152467728/9.115246772766113 | -3.388755942523858e-11 | True |
| morph-v1-003 | 2/2 | 7.9363057613/7.93630576133728 | 3.9681528807/3.96815288066864 | 3.7280400988493057e-11 | True |
| morph-v1-004 | 2/2 | 11.6550559998/11.65505599975586 | 5.8275279999/5.82752799987793 | -4.4140691102256824e-11 | True |

## Descriptive statistics

Sample standard deviation uses n-1; it is undefined for singleton groups. Summaries include successful rows only. These 40 controlled examples support descriptive comparisons only; no correlation, significance or causal claim is made.

| Group | n | Word mean | Median | SD | Min | Max | Normalized mean | Median | Tokens mean | Affixes mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 40 | 8.214675 | 9.030365 | 4.059394 | 1.831229 | 16.469409 | 6.970266 | 6.027921 | 1.225000 | 1.775000 |
| by_pos=NOUN | 20 | 9.654618 | 9.906992 | 3.825207 | 3.238821 | 16.469409 | 8.676069 | 9.309726 | 1.150000 | 1.300000 |
| by_pos=VERB | 20 | 6.774732 | 5.590914 | 3.850625 | 1.831229 | 13.140431 | 5.264462 | 5.256721 | 1.300000 | 2.250000 |
| by_affix_count=1 | 17 | 8.710577 | 9.722688 | 4.140833 | 2.395123 | 14.222039 | 8.319856 | 9.673970 | 1.058824 | 1.000000 |
| by_affix_count=2 | 15 | 6.093759 | 5.225344 | 3.340476 | 1.831229 | 11.752516 | 5.636161 | 4.694028 | 1.133333 | 2.000000 |
| by_affix_count=3 | 8 | 11.137602 | 11.334981 | 3.209330 | 5.893729 | 16.469409 | 6.603831 | 6.027921 | 1.750000 | 3.000000 |

## Runtime and memory

```json
{"started_at_utc": "2026-11-24T13:53:56.555591+00:00", "finished_at_utc": "2026-11-24T13:54:14.917568+00:00", "duration_seconds": 18.361997500000143, "load_seconds": 9.696395299999494, "inference_seconds": 4.571478500000012, "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU", "memory_before_load": {"allocated_bytes": 0, "reserved_bytes": 0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "device_free_bytes": 5379194880, "device_total_bytes": 6441926656}, "memory_after_load": {"allocated_bytes": 957573120, "reserved_bytes": 1012924416, "peak_allocated_bytes": 957573120, "peak_reserved_bytes": 1012924416, "device_free_bytes": 4366270464, "device_total_bytes": 6441926656}, "memory_after_evaluation": {"allocated_bytes": 967010304, "reserved_bytes": 1015021568, "peak_allocated_bytes": 971926016, "peak_reserved_bytes": 1015021568, "device_free_bytes": 4328521728, "device_total_bytes": 6441926656}, "environment": {"python": "3.13.5", "platform": "Windows-11-10.0.26200-SP0", "torch": "2.14.1+cu126", "transformers": "5.22.1", "tokenizers": "0.23.3", "huggingface_hub": "2.4.0", "safetensors": "0.8.0", "cuda_runtime": "12.6", "cuda_available": true}}
```

## Transformers warning assessment

AutoTokenizer uses trust_remote_code=False. In Transformers 5.22.1, tokenization_auto.py catches the unknown/custom AutoConfig ValueError and falls back to PreTrainedConfig (model_type=''). configuration_utils.py then warns about nafie versus the empty model_type. Reproduced with tokenizer-only loading; this warning does not identify the causal model class. The causal model is loaded separately with pinned, SHA-256-verified custom code and trust_remote_code=True. Check runtime model/config classes, empty loading diagnostics and finite forwards below. No architecture, checkpoint, tokenizer settings or logging filters changed.

```json
{"runtime_architecture": {"model_class": "NafieForCausalLM", "config_class": "NafieConfig", "model_type": "nafie", "architectures": ["NafieForCausalLM"], "tokenizer_class": "TokenizersBackend", "model_dtype": "torch.float16", "parameter_count": 473019392, "config_dimensions": {"vocab_size": 65536, "hidden_size": 1024, "num_hidden_layers": 36, "max_position_embeddings": 1024}}, "loading_info": {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}, "load_warnings": [{"logger": "transformers.configuration_utils", "message": "You are using a model of type `nafie` to instantiate a model of type ``. This may be expected if you are loading a checkpoint that shares a subset of the architecture (e.g., loading a `sam2_video` checkpoint into `Sam2Model`), but is otherwise not supported and can yield errors. Please verify that the checkpoint is compatible with the model you are instantiating.", "transformers_stack": [{"file": "transformers\\models\\auto\\tokenization_auto.py", "line": 807, "function": "from_pretrained"}, {"file": "transformers\\configuration_utils.py", "line": 896, "function": "from_pretrained"}]}]}
```

Snapshot config declares NafieForCausalLM with NafieConfig, model_type=nafie, 36 layers, hidden size 1024, vocabulary 65536, maximum sequence length 1024. The custom model config_class is NafieConfig. All seven snapshot files are hash-verified by the existing loader before custom code runs. Source checkpoint dtype=float32 is intentionally loaded as FP16 under the existing scoring profile.

## Failures and limits

```json
{"run_error": null, "validation_error": null, "unsuccessful_rows": [], "failed_references": []}
```

Reference values were supplied as terminal observations, without a separately archived old environment manifest. This run records the pinned revision, scoring profile and installed versions explicitly. Reproducibility across other GPU/software versions is not established. The upstream attention mask limitation requires single unpadded sequences.

## Regression checks

{"status": "PASS", "command": ".venv\\Scripts\\python.exe -B -m unittest discover -s tests -v", "tests_run": 116, "existing_tests": 106, "new_tests": 10, "failures": 0, "errors": 0, "skipped": 0, "duration_seconds": 10.201, "executed_before_inference": true, "execution_note": "Normal Windows temporary-directory access was required; initial sandbox-only attempt failed on temporary-folder permissions.", "test_source_sha256": "09364456246079e09bb4adc151f9be8984f95217a66b3f0d2e83a5bb2be2da08"}

## Remaining work

Human review of these experiment artifacts and commit approval. No commit is created by this runner.
