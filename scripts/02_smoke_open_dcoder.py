"""
1-minute smoke test: load Open-dCoder 0.5B, generate a quicksort, print it.

Purpose: confirm that the model downloads, loads on GPU, diffusion_generate
runs end-to-end, and output decodes. If this passes we know the toolchain is
fine before committing to a 10-30 minute HumanEval run.

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
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
mdl = (
    Qwen2ForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True)
    .to("cuda")
    .eval()
)
n_params = sum(p.numel() for p in mdl.parameters()) / 1e6
print(f"loaded in {time.time() - t0:.1f}s; params={n_params:.1f}M")

prompt = "Write a quick sort algorithm in python."
ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
cfg = MDMGenerationConfig(max_new_tokens=128, steps=200, temperature=0.7)

t1 = time.time()
with torch.no_grad():
    out = mdl.diffusion_generate(inputs=ids, generation_config=cfg)
print(f"generated in {time.time() - t1:.1f}s")
print("=" * 60)
print(tok.decode(out.sequences[0], skip_special_tokens=True))
print("=" * 60)
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
