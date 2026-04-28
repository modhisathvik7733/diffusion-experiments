"""
Chat-style code generation test for DreamOn-v0-7B.

DreamOn is specialized for infilling (prefix + masks + suffix). This script
tests how it behaves in chat-style "write me a function" mode where there's
only a request and no suffix. We try two formats so we can see which (if any)
DreamOn handles well:

  Format A: [BOS] + prompt + N_masks + [EOS]   (treat as "infill before EOS")
  Format B: chat template if tokenizer supplies one

Run: python scripts/07_chat_test_dreamon.py
"""
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
      f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"chat_template available: {tokenizer.chat_template is not None}")

GEN_KWARGS = dict(
    temperature=0.2,
    alg="entropy",
    alg_temp=0,
    top_p=0.9,
    max_new_tokens=256,
    return_dict_in_generate=True,
    output_history=False,
    number_transfer_tokens=1,
)


def format_a(prompt: str, n_masks: int = 128):
    """[BOS] + prompt + masks + [EOS]"""
    pids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_masks
    sids = [tokenizer.eos_token_id]
    return torch.LongTensor([pids + mids + sids]).to("cuda"), len(pids)


def format_b(prompt: str, n_masks: int = 128):
    """Chat template + masks before EOS, if template exists."""
    if tokenizer.chat_template is None:
        return None, None
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pids = tokenizer.encode(text, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_masks
    sids = [tokenizer.eos_token_id]
    return torch.LongTensor([pids + mids + sids]).to("cuda"), len(pids)


PROMPTS = [
    "Write a Python function to check if a number is prime.",
    "Implement a simple Stack class in Python with push, pop, and peek methods.",
    "Write a function that takes a string and returns it reversed.",
    "Write a Python function to compute the nth Fibonacci number iteratively.",
]


def run(name: str, ids: torch.Tensor, prompt_len: int):
    if ids is None:
        print(f"\n--- {name}: SKIPPED (no chat template) ---")
        return
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = model.diffusion_generate(ids, **GEN_KWARGS)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) * 1000
    full = tokenizer.decode(out.sequences[0], skip_special_tokens=False)
    just_generated = tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)
    print(f"\n--- {name}  ({dt:.0f} ms) ---")
    print("[generated only]")
    print(just_generated.strip())


for i, p in enumerate(PROMPTS, 1):
    print(f"\n{'=' * 70}\nPROMPT {i}: {p}\n{'=' * 70}")
    a_ids, a_plen = format_a(p)
    run("Format A (raw + masks)", a_ids, a_plen)
    b_ids, b_plen = format_b(p)
    run("Format B (chat template)", b_ids, b_plen)
