# Nafie-473M: model loading and forward probe

This diagnostic is separate from the v1/v2 model-result contracts. It computes
no NLL, entropy, loss, generation or training updates. Dataset, morphology and
token-boundary contracts are unchanged.

## Reviewed snapshot

- Repository: `nafie-ai/nafie-473M`.
- Code, config, tokenizer and weights revision:
  `20477766abe4ca9e5ef3ef8891164a346887bba8`.
- Review date: 2026-11-01.
- Sources reviewed as text before importing or executing:
  [configuration_nafie.py](https://huggingface.co/nafie-ai/nafie-473M/blob/20477766abe4ca9e5ef3ef8891164a346887bba8/configuration_nafie.py),
  [modeling_nafie.py](https://huggingface.co/nafie-ai/nafie-473M/blob/20477766abe4ca9e5ef3ef8891164a346887bba8/modeling_nafie.py).
- Pinned repository metadata lists one `model.safetensors`, 1,892,120,016 bytes,
  LFS SHA-256 `3e5089e03079b26a1836b355189eceb158daefe2a1248043b5bfd8226ce3f963`.

The two Python files import PyTorch, Transformers and the relative config module.
Their top-level statements define model classes. Review found no network calls,
subprocesses, filesystem writes/deletes, dynamic evaluation, obfuscation or extra
custom imports. This is a scoped source review, not a guarantee against bugs in
the framework or CUDA stack. Only this reviewed snapshot is allowed by the loader.

Every required artifact, including both Python sources, config, tokenizer and
Safetensors weights, is checked against a fixed SHA-256 before custom code is
imported. Git blob IDs for the small downloaded files were also compared against
the pinned repository metadata during review. The full expected SHA-256 list is
in `models/nafie.py`. A changed snapshot fails closed. `trust_remote_code=True`
is used only for the model, with both `revision` and `code_revision` pinned;
the tokenizer uses `trust_remote_code=False`. Pickled weights are not loaded.

## Compatibility and scope

The existing Python 3.13.5, PyTorch 2.14.1+cu126, Transformers 5.22.1 and
Tokenizers 0.23.3 environment is preserved. No packages are installed or changed.
The custom config's `use_return_dict` access is deprecated but still available
in this Transformers version; the probe passes `return_dict=True` explicitly.
The custom tied-weights mapping is a dictionary, matching the installed API.

The architecture accepts but ignores `attention_mask`; its attention is causal
SDPA. The probe therefore permits only one unpadded sequence, with an all-ones
mask. Padded batching must not reuse this implementation without a separate
architecture adaptation and validation. No KV cache or generation is tested.

Loading is explicitly FP16, followed by `.to("cuda:0").eval()`. Missing,
unexpected or mismatched checkpoint tensors fail the probe. CUDA availability
and every parameter's CUDA/FP16 placement are checked; there is no CPU fallback.
Weights may be staged in CPU memory during loading, but forward runs on CUDA.

The only input is `Dün okula geldim`. Tokenization reuses the boundary module's
`add_special_tokens=False`, `padding=False`, `truncation=False` settings. No
template, BOS or EOS is inserted. The report also records the IDs produced with
`add_special_tokens=True`, so default special-token insertion can be compared
with the actual input. Decode must reproduce the original text exactly.

Under `torch.inference_mode()`, the probe makes one model call, with no labels,
and validates all elements of `outputs.logits` for finiteness and the exact
shape `[batch_size, sequence_length, model.config.vocab_size]`. No scoring or
probability transformation follows.

## Reproduction

The loader is intentionally local-only. Download the exact allowlisted snapshot
before running it (the existing environment already contains the HF libraries):

```powershell
@'
from huggingface_hub import snapshot_download
from turkish_llm_morphology.models.nafie import CACHE_DIR, MODEL_ID, REVISION, REVIEWED_SHA256
snapshot_download(MODEL_ID, revision=REVISION, cache_dir=CACHE_DIR,
                  allow_patterns=list(REVIEWED_SHA256))
'@ | .venv/Scripts/python.exe -

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.venv/Scripts/python.exe -m turkish_llm_morphology.models.nafie
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

If Xet transfer stalls, set `HF_HUB_DISABLE_XET=1` for the download process to
use standard HTTP. This changes transport only, not the artifacts or versions.
In this session both Hub-client transfers stalled; direct HTTPS Range transfer
completed the same pinned file, and its full byte count and LFS SHA-256 matched.

The machine-readable report is `results/raw/nafie_473m_forward_probe.json`.
It records success/failure stage, exact revisions and verified artifact hashes,
package/OS/CUDA versions, input IDs, special tokens, logits validation, timings,
and GPU allocated/reserved/peak/free/total bytes before and after loading/forward.
On failure it records the original exception and traceback, and the process
exits nonzero. Re-running replaces this diagnostic report. It is not a v2
measurement record and must not be passed to the model-result validator.

All downloaded files and dynamically cached model code stay in `.cache/`, which
is ignored by Git. No weights or HF cache artifacts are staged or committed.

## Observed result: 2026-11-01

The offline CUDA FP16 probe succeeded on the NVIDIA GeForce RTX 3060 Laptop GPU
(6 GB, compute capability 8.6; NVIDIA driver observed as 610.47). The saved JSON
contains the exact successful run at 16:53 UTC / 19:53 Europe/Istanbul.

| Check | Observed value |
| --- | --- |
| Model | `NafieForCausalLM`, 473,019,392 parameters, CUDA FP16 |
| Checkpoint loading | No missing, unexpected or mismatched keys; no loading errors |
| Actual input IDs | `[17035, 9856, 17547]` |
| BOS / EOS IDs | `0` / `1`; neither appears in the actual input |
| Automatic special tokens | `add_special_tokens=True` produces the same three IDs; no BOS/EOS insertion |
| Logits | `[1, 3, 65536]`, `torch.float16`, `cuda:0`, every value finite |
| Gradient tracking | `logits.requires_grad=False` |
| GPU allocation after load | 913.213 MiB |
| GPU allocation after forward | 922.589 MiB |
| Peak PyTorch GPU allocation | 923.526 MiB |
| Peak PyTorch GPU reservation | 966 MiB |
| Timings | Load plus verification: 11.884 s; forward plus validation: 0.672 s |
| Tests | 61/61 passed, including all previous 52 tests |

MiB uses 1,048,576 bytes. Allocation/reservation describe PyTorch's allocator;
they exclude driver/context memory. The report also records device-wide free
memory, so allocator figures should not be treated as total GPU process usage.
Timings are a single smoke test, not a throughput benchmark.

The tokenizer still emits the generic Transformers warning about instantiating
an unknown `nafie` config with `trust_remote_code=False`. It did not prevent
tokenization or the separately trusted custom model load/forward in this pinned
environment. No model architecture patch or package version change was needed.

This validates only the specified three-token, unpadded example. Longer inputs,
padded batches, numerical agreement with FP32, NLL and entropy remain untested.
