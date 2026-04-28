"""DreamOn-v0-7B inference wrapper.

Loads the model once at import time, exposes a single infill() entry point
that mirrors the settings used in our verified 88.67 Pass@1 baseline:
alg=entropy, alg_temp=0, top_p=0.9, mask_expansion via DreamOn's modeling.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModel, AutoTokenizer

from postprocess import clean_completion

MODEL_ID = "Dream-org/DreamOn-v0-7B"
DEVICE = "cuda"

print(f"[inference] loading {MODEL_ID} (bf16)...")
_t0 = time.time()
_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
_model = (
    AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
    .to(DEVICE)
    .eval()
)
print(
    f"[inference] loaded in {time.time() - _t0:.1f}s; "
    f"VRAM={torch.cuda.memory_allocated() / 1e9:.2f} GB"
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


def _encode(prefix: str, suffix: str, n_mask: int) -> torch.Tensor:
    pids = [_tokenizer.bos_token_id] + _tokenizer.encode(prefix, add_special_tokens=False)
    mids = [_tokenizer.mask_token_id] * n_mask
    sids = _tokenizer.encode(suffix, add_special_tokens=False) + [_tokenizer.eos_token_id]
    return torch.LongTensor([pids + mids + sids]).to(DEVICE), len(pids)


def infill(
    prefix: str,
    suffix: str,
    max_tokens: int = 64,
    language: str | None = None,
) -> dict:
    """Generate the middle text between prefix and suffix.

    Returns {completion, tokens, latency_ms, finish_reason}.

    Uses small initial mask count (4) and max_new_tokens=max_tokens as the
    expansion cap — matches the upstream eval pattern (min_gen_len=4,
    max_gen_len=64) that produced our 88.67 Pass@1 baseline. Tying initial
    masks to max_tokens kills DreamOn's variable-length expansion mechanism
    and produces wrong-length completions.
    """
    initial_masks = 4
    cap_tokens = max(initial_masks, min(int(max_tokens), 128))
    input_ids, prompt_len = _encode(prefix, suffix, initial_masks)

    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = _model.diffusion_generate(input_ids, max_new_tokens=cap_tokens, **_GEN_KWARGS)
    torch.cuda.synchronize()
    latency_ms = int((time.perf_counter() - t) * 1000)

    full = _tokenizer.decode(out.sequences[0], skip_special_tokens=False)
    raw_completion = clean_completion(full, prefix, suffix, language)

    # Token count of just the completion (rough; includes specials stripped above)
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
