"""
Test DreamOn on a specific autocomplete scenario the user asked about:

  if __name__ == "__main__":
      print(f"2 + 3 = {add

Cursor is at end of `add`. Expected: model completes to something like
`(2, 3)}")`. We try a few mask-count choices since the right number isn't
known a-priori in real autocomplete.

Run: python scripts/10_autocomplete_specific.py
"""
import time
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "Dream-org/DreamOn-v0-7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = (
    AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    .to("cuda")
    .eval()
)
print(f"loaded; VRAM={torch.cuda.memory_allocated() / 1e9:.2f} GB\n")

# Without an `add` function defined, the model has to assume one exists.
# Test variant 1: just the line, no add() defined upstream
PREFIX_ALONE = '''if __name__ == "__main__":
    print(f"2 + 3 = {add'''

# Test variant 2: with `add` properly defined upstream (more realistic context)
PREFIX_WITH_DEF = '''def add(a, b):
    return a + b


if __name__ == "__main__":
    print(f"2 + 3 = {add'''

GEN_KWARGS = dict(
    temperature=0.2,
    alg="entropy",
    alg_temp=0,
    top_p=0.9,
    max_new_tokens=32,
    return_dict_in_generate=True,
    output_history=False,
    number_transfer_tokens=1,
)


def autocomplete(prefix: str, n_masks: int):
    """Treat as infilling with empty suffix: prefix + masks + EOS."""
    pids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    mids = [tokenizer.mask_token_id] * n_masks
    sids = [tokenizer.eos_token_id]
    input_ids = torch.LongTensor([pids + mids + sids]).to("cuda")

    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        out = model.diffusion_generate(input_ids, **GEN_KWARGS)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) * 1000

    # Decode the full output, then extract just what the model added after the prefix
    full = tokenizer.decode(out.sequences[0], skip_special_tokens=False)
    # Strip after first <|endoftext|>
    for stop in ("<|endoftext|>", "<|eos|>"):
        if stop in full:
            full = full.split(stop)[0]
    # Strip BOS / mask leftovers
    full = full.replace("<|beginoftext|>", "").replace("<|mask|>", "").replace("<|expand|>", "")
    # Find where prefix ends, return only the continuation
    idx = full.find(prefix)
    completion = full[idx + len(prefix):] if idx >= 0 else full
    # Strip any trailing `!`s (unfilled mask leftovers in some tokenizers)
    completion = completion.rstrip("!").rstrip()
    return completion, dt


for label, prefix in [
    ("[NO add() defined]", PREFIX_ALONE),
    ("[WITH add() defined]", PREFIX_WITH_DEF),
]:
    print("=" * 72)
    print(label)
    print("=" * 72)
    print("PREFIX (cursor at end):")
    print(prefix + "█")
    print()
    for n_masks in (4, 8, 16):
        completion, dt = autocomplete(prefix, n_masks)
        result = prefix + completion
        print(f"--- n_masks={n_masks}  ({dt:.0f} ms) ---")
        print("COMPLETION (model added):")
        print(repr(completion))
        print("FULL RESULT:")
        print(result)
        print()
