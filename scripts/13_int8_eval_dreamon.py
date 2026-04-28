"""
HumanEval-Infill Pass@1 evaluation for DreamOn-v0-7B.

Uses oracle middle length (one mask per ground-truth token), single-GPU,
no DDP. Set DREAMON_INT8=1 to load in 8-bit (bitsandbytes), otherwise loads
bf16. Run BOTH back-to-back to compare INT8 vs bf16 on identical settings.

Run:
  python scripts/13_int8_eval_dreamon.py            # bf16 baseline
  DREAMON_INT8=1 python scripts/13_int8_eval_dreamon.py   # INT8

Note: numbers won't exactly match the published 88.67 baseline because that
used DreamOn's mask_expansion mechanism (min_gen_len=4, max_gen_len=64). The
bf16 reading from THIS script is the apples-to-apples comparison for INT8.
"""
import json
import os
import subprocess
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from human_eval_infilling.data import read_problems

MODEL = "Dream-org/DreamOn-v0-7B"
USE_INT8 = os.environ.get("DREAMON_INT8", "0") == "1"
MODE = "INT8" if USE_INT8 else "bf16"

ROOT = os.environ.get("ROOT", "/workspace/diffusion-llm")
EXPDIR = f"{ROOT}/diffusion-experiments"
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_eval_{MODE.lower()}"
RUN_DIR = f"{EXPDIR}/runs/{RUN_ID}"
os.makedirs(RUN_DIR, exist_ok=True)
PRED_FILE = f"{RUN_DIR}/predictions.jsonl"

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0)}")
print(f"Mode: {MODE}")
print(f"Run dir: {RUN_DIR}")

# Load model
print(f"\n==> Loading DreamOn in {MODE}")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if USE_INT8:
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(
        MODEL,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map={"": 0},
    ).eval()
else:
    model = AutoModel.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
load_vram = torch.cuda.memory_allocated() / 1e9
print(f"loaded in {time.time() - t0:.1f}s; VRAM={load_vram:.2f} GB")

# Load problems
print("\n==> Loading HumanEval-Infill (single-line)")
problems = read_problems(benchmark_name="single-line")
task_ids = list(problems.keys())
print(f"loaded {len(task_ids)} problems")

# Generation config
GEN_KWARGS = dict(
    temperature=0.2,
    alg="entropy",
    alg_temp=0,
    top_p=0.9,
    max_new_tokens=64,
    return_dict_in_generate=True,
    output_history=False,
    number_transfer_tokens=1,
)

# Eval loop with oracle middle length
results = []
start = time.time()
torch.cuda.reset_peak_memory_stats()

for task_id in tqdm(task_ids, desc=f"{MODE} eval"):
    p = problems[task_id]
    prefix = p["prompt"]
    suffix = p["suffix"]
    canonical = p["canonical_solution"]
    # Oracle: use ground truth length
    n_mask = max(4, len(tokenizer.encode(canonical, add_special_tokens=False)))

    pids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_mask
    sids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    total = len(pids) + len(mids) + len(sids)

    # Skip if total exceeds context (DreamOn's 2048; rare for HumanEval)
    if total > 2048:
        results.append({"task_id": task_id, "completion": ""})
        continue

    input_ids = torch.LongTensor([pids + mids + sids]).to("cuda")

    try:
        with torch.no_grad():
            out = model.diffusion_generate(input_ids, **GEN_KWARGS)
    except Exception as e:
        print(f"\n  ERROR on {task_id}: {type(e).__name__}: {e}")
        results.append({"task_id": task_id, "completion": ""})
        continue

    # Decode + extract middle
    full = tokenizer.decode(out.sequences[0], skip_special_tokens=False)
    for stop in ("<|endoftext|>", "<|eos|>"):
        if stop in full:
            full = full.split(stop)[0]
    for special in ("<|beginoftext|>", "<|mask|>", "<|expand|>"):
        full = full.replace(special, "")

    # Find middle: between prefix and suffix
    completion = ""
    if prefix in full:
        after_prefix = full[full.index(prefix) + len(prefix):]
        # Look for suffix starting somewhere in after_prefix
        suffix_stripped = suffix.lstrip("\n")
        # Try to match the first 20 chars of suffix
        marker = suffix_stripped[:20].strip() if suffix_stripped.strip() else None
        if marker and marker in after_prefix:
            completion = after_prefix.split(marker)[0]
        else:
            # Fallback: take first n_mask*2 chars
            completion = after_prefix[: n_mask * 6]
    else:
        # Hard fallback: decode the slot we asked for
        gen_slice = out.sequences[0][len(pids): len(pids) + n_mask]
        completion = tokenizer.decode(gen_slice, skip_special_tokens=True)

    completion = completion.rstrip("!").rstrip()
    results.append({"task_id": task_id, "completion": completion})

eval_time = time.time() - start
peak_vram = torch.cuda.max_memory_allocated() / 1e9
ms_per_problem = eval_time / max(1, len(task_ids)) * 1000

print(f"\n==> Eval done in {eval_time:.0f}s ({ms_per_problem:.0f} ms/problem)")
print(f"Peak VRAM: {peak_vram:.2f} GB")

# Save
with open(PRED_FILE, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved predictions: {PRED_FILE}")

# Score via the upstream tool
print("\n==> Scoring")
res = subprocess.run(
    ["evaluate_infilling_functional_correctness", "single-line", PRED_FILE],
    capture_output=True, text=True,
)
print("STDOUT:", res.stdout.strip())
if res.stderr:
    # Last 30 lines of stderr (progress bars etc.)
    tail = "\n".join(res.stderr.strip().splitlines()[-5:])
    print("STDERR (tail):", tail)

print(f"\n=== {MODE} SUMMARY ===")
print(f"  load VRAM:    {load_vram:.2f} GB")
print(f"  peak VRAM:    {peak_vram:.2f} GB")
print(f"  eval time:    {eval_time:.0f} s")
print(f"  ms/problem:   {ms_per_problem:.0f} ms")
print(f"  see Pass@1 in STDOUT above")
