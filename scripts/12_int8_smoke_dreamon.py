"""
INT8 quantization smoke test for DreamOn-v0-7B on RTX 5090 (Blackwell sm_120).

Goal: confirm bitsandbytes 8-bit loading works on this hardware AND that
DreamOn's custom modeling/generation still functions correctly. Reports VRAM
savings and latency vs bf16. If smoke passes, we proceed to the full INT8
HumanEval-Infill eval. If it fails, we fall back to torchao or HQQ.

Run: python scripts/12_int8_smoke_dreamon.py
"""
import time
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

MODEL = "Dream-org/DreamOn-v0-7B"

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

# --- Step 1: load INT8 ---
print("\n==> Loading DreamOn with bitsandbytes 8-bit")
quant_config = BitsAndBytesConfig(load_in_8bit=True)

t0 = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="cuda:0",
    ).eval()
    load_time = time.time() - t0
    vram_loaded = torch.cuda.memory_allocated() / 1e9
    print(f"loaded in {load_time:.1f}s; VRAM={vram_loaded:.2f} GB "
          f"(baseline bf16 was 15.28 GB)")
    print(f"INT8 VRAM reduction: {(15.28 - vram_loaded) / 15.28 * 100:.0f}%")
except Exception as e:
    print(f"\n!!! INT8 load FAILED on this hardware: {type(e).__name__}")
    print(f"    {e}")
    print("\nLikely cause: bitsandbytes lacks Blackwell sm_120 kernels.")
    print("Fallback options: torchao int8, HQQ, or skip INT8 for v1.")
    raise SystemExit(1)


# --- Step 2: smoke generation ---
print("\n==> Smoke generation: HumanEval-style infilling")


def encode_infill(prefix: str, suffix: str, n_mask: int):
    pids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_mask
    sids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return torch.LongTensor([pids + mids + sids]).to("cuda"), len(pids)


prefix = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if any two numbers are closer than threshold. """
    for idx, elem in enumerate(numbers):
'''
suffix = '''        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
'''

ids, plen = encode_infill(prefix, suffix, 4)

# Warmup
with torch.no_grad():
    _ = model.diffusion_generate(
        ids, temperature=0.2, alg="entropy", alg_temp=0, top_p=0.9,
        max_new_tokens=64, return_dict_in_generate=True,
        output_history=False, number_transfer_tokens=1,
    )
torch.cuda.synchronize()

# Real run
torch.cuda.reset_peak_memory_stats()
t = time.perf_counter()
with torch.no_grad():
    out = model.diffusion_generate(
        ids, temperature=0.2, alg="entropy", alg_temp=0, top_p=0.9,
        max_new_tokens=64, return_dict_in_generate=True,
        output_history=False, number_transfer_tokens=1,
    )
torch.cuda.synchronize()
dt_ms = (time.perf_counter() - t) * 1000

generated = tokenizer.decode(out.sequences[0][plen:plen + 32], skip_special_tokens=True)
peak_vram = torch.cuda.max_memory_allocated() / 1e9

print(f"\n--- Result ---")
print(f"Generated: {generated.strip()[:200]}")
print(f"Latency:   {dt_ms:.0f} ms (bf16 baseline was ~3892 ms for similar prompt at 256 steps)")
print(f"Peak VRAM: {peak_vram:.2f} GB (bf16 was 18.83 GB)")
print(f"\n=== Quality check (visual): does generated text look like valid Python that fills the loop? ===")
print(f"Expected pattern: 'for idx2, elem2 in enumerate(numbers):' + comparison block")
print(f"\nIf the output above is coherent code -> INT8 works on your Blackwell GPU.")
print(f"  Next step: bash scripts/13_int8_eval_dreamon.sh (full HumanEval-Infill at INT8)")
print(f"If output is garbage / repeated tokens / errors -> bitsandbytes Blackwell support is broken.")
print(f"  Fallback: try torchao int8 or HQQ (different quant libraries).")
