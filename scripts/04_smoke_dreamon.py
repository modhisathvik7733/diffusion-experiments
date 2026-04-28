"""
Smoke test for DreamOn-v0-7B (Dream-Coder fine-tuned for variable-length infilling).

Mirrors the HF model card example. Confirms the 8B model loads on the 5090, the
infilling input format is correct, and `diffusion_generate` produces output before
we commit to a 30-60 min full eval.

Run: python scripts/04_smoke_dreamon.py
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
n_params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"loaded in {time.time() - t0:.1f}s; params={n_params:.1f}B; "
      f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def process_infilling_prompt(prefix: str, suffix: str, tokenizer, number_of_mask: int):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    middle_ids = [tokenizer.mask_token_id] * number_of_mask
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return prefix_ids + middle_ids + suffix_ids


prefix = '''from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    """
    for idx, elem in enumerate(numbers):
'''

suffix = '''        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
'''

input_ids = torch.LongTensor(
    [process_infilling_prompt(prefix, suffix, tokenizer, number_of_mask=4)]
).to("cuda")
prompt_len = input_ids.shape[1]

t1 = time.time()
with torch.no_grad():
    out = model.diffusion_generate(
        input_ids,
        temperature=0.2,
        alg="entropy",
        alg_temp=0,
        top_p=0.9,
        max_new_tokens=64,
        return_dict_in_generate=True,
        output_history=True,
        number_transfer_tokens=1,
    )
print(f"generated in {time.time() - t1:.1f}s")

print("=" * 60)
print("FULL OUTPUT (prompt + filled middle):")
print(tokenizer.decode(out.sequences[0], skip_special_tokens=True))
print("=" * 60)
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
