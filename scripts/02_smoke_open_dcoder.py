"""
1-minute smoke test: load Open-dCoder 0.5B, generate a quicksort, print it.

Mirrors upstream Open-dLLM/sample.py (the README's quickstart is incomplete -
it omits mask_token_id which is required by MDMGenerationConfig).

Run: python scripts/02_smoke_open_dcoder.py
"""
import time
import torch
from transformers import AutoTokenizer
from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig

MODEL = "fredzzp/open-dcoder-0.5B"

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = (
    Qwen2ForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    .to("cuda")
    .eval()
)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"loaded in {time.time() - t0:.1f}s; params={n_params:.1f}M")

if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    model.resize_token_embeddings(len(tokenizer))
    print("added [MASK] token")

prompt = "Write a quick sort algorithm in Python.\n"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
prompt_len = input_ids.shape[1]

cfg = MDMGenerationConfig(
    mask_token_id=tokenizer.mask_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
    steps=128,
    temperature=0.5,
    top_k=200,
    alg="p2",
    alg_temp=0.5,
    num_return_sequences=1,
    return_dict_in_generate=True,
    output_history=False,
)

t1 = time.time()
with torch.no_grad():
    out = model.diffusion_generate(inputs=input_ids, generation_config=cfg)
print(f"generated in {time.time() - t1:.1f}s")

generated = tokenizer.decode(out.sequences[0][prompt_len:], skip_special_tokens=True)
print("=" * 60)
print(generated)
print("=" * 60)
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
