"""
Production-realistic latency benchmark for DreamOn-v0-7B autocomplete.

Measures p50/p95/p99 over a mix of short/medium/long infilling scenarios that
mirror real editor autocomplete cases (single-line, function body, class
method). Uses the same DreamOn settings as the HumanEval-Infill eval so the
quality result generalizes.

Run: python scripts/06_latency_bench_dreamon.py
"""
import statistics
import time
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Dream-org/DreamOn-v0-7B"

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    .to("cuda")
    .eval()
)
print(f"loaded in {time.time() - t0:.1f}s; "
      f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def encode_infill(prefix: str, suffix: str, n_mask: int):
    pids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_mask
    sids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return torch.LongTensor([pids + mids + sids]).to("cuda")


# (name, prefix, suffix, n_initial_masks, max_new_tokens)
# Mirrors realistic editor autocomplete: cursor sits between prefix and suffix.
PROMPTS = [
    # SHORT — single token / single line completions (most common autocomplete case)
    ("short_var_assign",
     "user_name = ",
     "\nprint(f'Hello, {user_name}')\n",
     4, 16),
    ("short_return_expr",
     "def double(x: int) -> int:\n    return ",
     "\n",
     4, 16),
    ("short_list_comp",
     "squares = [",
     " for x in range(10)]\n",
     4, 16),
    ("short_call_arg",
     "result = compute_score(items, threshold=",
     ", verbose=True)\n",
     4, 16),
    ("short_type_annot",
     "def get_user(user_id: int) -> ",
     ":\n    pass\n",
     4, 16),

    # MEDIUM — function body / single block (common Tab completion)
    ("medium_function_body",
     "def is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s is a palindrome.\"\"\"\n    ",
     "\n",
     8, 32),
    ("medium_class_method",
     "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        ",
     "\n\n    def pop(self):\n        return self.items.pop()\n",
     8, 32),
    ("medium_loop_body",
     "results = []\nfor item in dataset:\n    ",
     "\n    results.append(value)\nreturn results\n",
     8, 32),
    ("medium_dict_literal",
     "config = {\n    'host': 'localhost',\n    'port': ",
     ",\n    'timeout': 30,\n}\n",
     4, 16),

    # LONG — multi-line block / complex infill
    ("long_quicksort_body",
     "def quicksort(arr: list) -> list:\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    ",
     "\n    return quicksort(left) + middle + quicksort(right)\n",
     16, 64),
    ("long_api_handler",
     "@app.route('/users/<int:user_id>')\ndef get_user(user_id: int):\n    \"\"\"Return user JSON or 404.\"\"\"\n    ",
     "\n    return jsonify(user.to_dict())\n",
     16, 64),
    ("long_data_class",
     "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    id: int\n    name: str\n    ",
     "\n\n    def __repr__(self) -> str:\n        return f'User({self.id}, {self.name})'\n",
     16, 64),
]

# DreamOn generation kwargs — same as upstream eval used (steps=256 etc.)
GEN_KWARGS = dict(
    temperature=0.2,
    alg="entropy",
    alg_temp=0,
    top_p=0.9,
    return_dict_in_generate=True,
    output_history=False,
    number_transfer_tokens=1,
)

# Warmup — first call is slow due to compilation/caching
print("\n==> Warmup (3 runs)")
warmup = PROMPTS[0]
warmup_ids = encode_infill(warmup[1], warmup[2], warmup[3])
for _ in range(3):
    with torch.no_grad():
        model.diffusion_generate(warmup_ids, max_new_tokens=warmup[4], **GEN_KWARGS)
torch.cuda.synchronize()
torch.cuda.empty_cache()

# Benchmark
print("\n==> Benchmarking ({} prompts × 3 trials each)".format(len(PROMPTS)))
results = {}
for name, prefix, suffix, n_mask, max_new in PROMPTS:
    ids = encode_infill(prefix, suffix, n_mask)
    lats = []
    tok_out = []
    for trial in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            out = model.diffusion_generate(ids, max_new_tokens=max_new, **GEN_KWARGS)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t) * 1000)
        tok_out.append(out.sequences.shape[1] - ids.shape[1])
    results[name] = {"lats": lats, "tokens": tok_out, "max_new": max_new}
    print(f"  {name:24s}  p50={statistics.median(lats):6.0f}ms  "
          f"min={min(lats):6.0f}ms  max={max(lats):6.0f}ms  "
          f"tok_out={statistics.median(tok_out):3.0f}")

# Aggregate
print("\n==> Aggregate latency over all {} runs".format(sum(len(r["lats"]) for r in results.values())))
all_lats = sorted(l for r in results.values() for l in r["lats"])
def pct(p):
    idx = min(len(all_lats) - 1, int(round(p * (len(all_lats) - 1))))
    return all_lats[idx]
print(f"  mean: {statistics.mean(all_lats):.0f} ms")
print(f"  p50 : {pct(0.50):.0f} ms")
print(f"  p95 : {pct(0.95):.0f} ms")
print(f"  p99 : {pct(0.99):.0f} ms")
print(f"  min : {min(all_lats):.0f} ms")
print(f"  max : {max(all_lats):.0f} ms")

# Per-category aggregate
print("\n==> By category")
for cat in ("short", "medium", "long"):
    cat_lats = sorted(l for n, r in results.items() if n.startswith(cat) for l in r["lats"])
    if not cat_lats:
        continue
    print(f"  {cat:6s}  count={len(cat_lats):2d}  "
          f"p50={statistics.median(cat_lats):6.0f}ms  "
          f"p95={cat_lats[min(len(cat_lats)-1, int(round(0.95*(len(cat_lats)-1))))]:6.0f}ms  "
          f"p99={cat_lats[min(len(cat_lats)-1, int(round(0.99*(len(cat_lats)-1))))]:6.0f}ms")

print(f"\n==> Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"==> Verdict thresholds for autocomplete UX:")
print(f"    p99 < 500ms  : ship vanilla")
print(f"    p99 < 1500ms : usable, distill for v1.5")
print(f"    p99 > 1500ms : distillation required for ship")
