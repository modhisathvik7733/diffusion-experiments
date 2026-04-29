"""DreamOn-v0-7B inference wrapper.

Two code paths, gated by the FAST_DLLM env var:

- Default (FAST_DLLM unset/0): upstream Dream via AutoModel + trust_remote_code.
  Uses alg=entropy + number_transfer_tokens=1 (sequential decode).
  Keeps DreamOn's variable-length expansion (4 initial masks grow during
  generation up to max_new_tokens). This is the baseline 88.67 Pass@1 path.

- FAST_DLLM=1: vendored Fast-dLLM model under server/fast_dllm_model/.
  Uses alg=confidence_threshold + threshold=0.9 (parallel decode).
  Pre-allocates the full mask region up-front because Fast-dLLM's _sample
  loop operates on a fixed mask_index — no variable-length expansion.
  Phase-1 speedup target: ~1.8x on long bucket (parallel decode only).
  Block-wise KV-cache (Method 5) is L-to-R and breaks infill, so it's
  intentionally not wired up here.
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoTokenizer

from postprocess import clean_completion

MODEL_ID = "Dream-org/DreamOn-v0-7B"
DEVICE = "cuda"

USE_FAST_DLLM = os.environ.get("FAST_DLLM", "").lower() in ("1", "true", "yes")

print(f"[inference] loading {MODEL_ID} (bf16, fast_dllm={USE_FAST_DLLM})...")
_t0 = time.time()
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if USE_FAST_DLLM:
    from fast_dllm_model.modeling_dream import DreamModel

    _model = (
        DreamModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        .to(DEVICE)
        .eval()
    )
    _GEN_KWARGS = dict(
        alg="confidence_threshold",
        threshold=0.9,
        temperature=0.2,
        top_p=0.9,
        return_dict_in_generate=True,
        output_history=False,
    )
else:
    from transformers import AutoModel

    _model = (
        AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
        .to(DEVICE)
        .eval()
    )
    _GEN_KWARGS = dict(
        temperature=0.2,
        alg="entropy",
        alg_temp=0,
        top_p=0.9,
        return_dict_in_generate=True,
        output_history=False,
        number_transfer_tokens=1,
    )

print(
    f"[inference] loaded in {time.time() - _t0:.1f}s; "
    f"VRAM={torch.cuda.memory_allocated() / 1e9:.2f} GB"
)


def _encode(prefix: str, suffix: str, n_mask: int) -> tuple[torch.Tensor, int]:
    pids = [_tokenizer.bos_token_id] + _tokenizer.encode(prefix, add_special_tokens=False)
    mids = [_tokenizer.mask_token_id] * n_mask
    sids = _tokenizer.encode(suffix, add_special_tokens=False) + [_tokenizer.eos_token_id]
    return torch.LongTensor([pids + mids + sids]).to(DEVICE), len(pids)


def _pick_n_mask_and_steps(cap_tokens: int) -> tuple[int, int]:
    """Round cap up to a multiple of 8 and pick steps so each step
    transfers ~8 tokens. Fast-dLLM's _sample asserts that n_mask is
    divisible by steps."""
    n_mask = max(8, ((cap_tokens + 7) // 8) * 8)
    steps = max(1, n_mask // 8)
    return n_mask, steps


def infill(
    prefix: str,
    suffix: str,
    max_tokens: int = 64,
    language: str | None = None,
) -> dict:
    """Generate the middle text between prefix and suffix.

    Returns {completion, tokens, latency_ms, finish_reason}.
    """
    cap_tokens = max(4, min(int(max_tokens), 128))

    if USE_FAST_DLLM:
        # Pre-allocate the full mask region. max_length=current_len
        # disables right-padding (which would otherwise add masks
        # after eos and waste compute on post-suffix tokens).
        n_mask, steps = _pick_n_mask_and_steps(cap_tokens)
        input_ids, prompt_len = _encode(prefix, suffix, n_mask)
        gen_kwargs = dict(
            _GEN_KWARGS,
            steps=steps,
            max_length=input_ids.shape[1],
        )
    else:
        # 4 initial masks + max_new_tokens cap → DreamOn variable-length
        # expansion (the 88.67 Pass@1 baseline pattern).
        n_mask = 4
        input_ids, prompt_len = _encode(prefix, suffix, n_mask)
        gen_kwargs = dict(_GEN_KWARGS, max_new_tokens=cap_tokens)

    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = _model.diffusion_generate(input_ids, **gen_kwargs)
    torch.cuda.synchronize()
    latency_ms = int((time.perf_counter() - t) * 1000)

    full = _tokenizer.decode(out.sequences[0], skip_special_tokens=False)
    raw_completion = clean_completion(full, prefix, suffix, language)

    tokens = len(_tokenizer.encode(raw_completion, add_special_tokens=False))
    finish_reason = "stop" if "<|endoftext|>" in full else "length"

    return {
        "completion": raw_completion,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "finish_reason": finish_reason,
    }


def vram_usage() -> tuple[float, float]:
    """Return (used_gb, total_gb) for GPU 0."""
    used = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return used, total
