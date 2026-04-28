"""
Concurrent / batched load test for DreamOn-v0-7B.

Question: how many simultaneous autocomplete requests can the 5090 serve?
Tests batched generation at batch sizes {1, 2, 4, 8} with realistic
short-autocomplete prompts. Reports per-batch latency, per-request latency,
throughput (req/s), and VRAM under load.

Run: python scripts/17_concurrent_load.py
"""
import statistics
import time
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Dream-org/DreamOn-v0-7B"
BATCH_SIZES = [1, 2, 4, 8]
TRIALS_PER_BATCH = 5

print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    .to("cuda")
    .eval()
)
print(f"loaded in {time.time() - t0:.1f}s; VRAM={torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Realistic short autocomplete prompts (roughly the inline-completion case)
PROMPTS = [
    ("def add(a, b):\n    return ", "\n"),
    ("x = ", "\nprint(x)\n"),
    ("squares = [", " for x in range(10)]\n"),
    ("def is_palindrome(s: str) -> bool:\n    return ", "\n"),
    ("for item in items:\n    ", "\n    process(item)\n"),
    ("result = compute(items, threshold=", ", verbose=True)\n"),
    ("class Stack:\n    def push(self, item):\n        ", "\n"),
    ("config = {'host': 'localhost', 'port': ", "}\n"),
]


def encode_infill(prefix, suffix, n_mask=8):
    pids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_mask
    sids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return pids + mids + sids


GEN_KWARGS = dict(
    temperature=0.2, alg="entropy", alg_temp=0, top_p=0.9,
    max_new_tokens=32, return_dict_in_generate=True,
    output_history=False, number_transfer_tokens=1,
)


def make_batch(batch_size: int):
    """Build a padded batch of input_ids."""
    seqs = [encode_infill(*PROMPTS[i % len(PROMPTS)]) for i in range(batch_size)]
    max_len = max(len(s) for s in seqs)
    padded = [s + [tokenizer.pad_token_id] * (max_len - len(s)) for s in seqs]
    return torch.LongTensor(padded).to("cuda")


# Warmup
print("\n==> Warmup")
warm = make_batch(1)
for _ in range(3):
    with torch.no_grad():
        model.diffusion_generate(warm, **GEN_KWARGS)
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n==> Benchmarking batched generation")
print(f"{'batch':>6}  {'p50_ms':>8}  {'p95_ms':>8}  {'per_req_ms':>11}  {'req_per_s':>10}  {'peak_vram_gb':>13}")
print("-" * 70)

results = {}
for bs in BATCH_SIZES:
    try:
        ids = make_batch(bs)
    except Exception as e:
        print(f"{bs:>6}  ERROR creating batch: {e}")
        continue

    lats = []
    err = None
    for _ in range(TRIALS_PER_BATCH):
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            with torch.no_grad():
                _ = model.diffusion_generate(ids, **GEN_KWARGS)
        except torch.cuda.OutOfMemoryError as e:
            err = "OOM"
            torch.cuda.empty_cache()
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            break
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)

    if err:
        print(f"{bs:>6}  {err}")
        continue

    p50 = statistics.median(lats)
    p95 = sorted(lats)[min(len(lats) - 1, int(round(0.95 * (len(lats) - 1))))]
    per_req = p50 / bs
    req_per_s = (bs * 1000.0) / p50
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    results[bs] = dict(p50=p50, p95=p95, per_req=per_req, throughput=req_per_s, vram=peak_vram)
    print(f"{bs:>6}  {p50:>8.0f}  {p95:>8.0f}  {per_req:>11.0f}  {req_per_s:>10.2f}  {peak_vram:>13.2f}")

print("\n==> Verdict")
if 1 in results:
    base = results[1]["throughput"]
    print(f"  Single-request throughput baseline: {base:.2f} req/s")
    for bs, r in results.items():
        if bs == 1:
            continue
        scaling = r["throughput"] / base
        ideal = bs
        eff = scaling / ideal * 100
        print(f"  Batch {bs}: {r['throughput']:.2f} req/s  ({scaling:.2f}× single, "
              f"{eff:.0f}% of linear scaling)")

print("\nReading the table:")
print("  per_req_ms = p50_ms / batch — the per-user latency at that concurrency")
print("  req_per_s  = throughput cap for that batch size")
print("  peak_vram  = how much VRAM that batch size used")
print("\nFor production sizing:")
print("  - Pick batch where per_req_ms is < your latency budget")
print("  - req_per_s × number_of_GPUs = max users you can serve")
print("  - Stay below 28 GB VRAM on a 32 GB card to leave headroom")
