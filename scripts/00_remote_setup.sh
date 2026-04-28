#!/usr/bin/env bash
# Idempotent env setup for Vast.ai (vastai/pytorch_cuda-12.8.1-auto, RTX 5090 sm_120).
# Strategy: keep the image's torch (Blackwell-tuned), install Open-dLLM with --no-deps
# so its requirements.txt (flash-attn, older transformers) can't downgrade torch.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
cd "$ROOT"

echo "==> Restore torch + torchvision from cu128 channel (in case prior run downgraded them)"
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision

echo "==> Verify torch + CUDA + Blackwell capability"
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
cap = torch.cuda.get_device_capability(0)
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0), "cap", cap)
assert cap[0] >= 9, f"GPU compute capability {cap} too old; expected sm_90+ (Blackwell sm_120)"
PY

echo "==> Pip toolchain"
pip install --upgrade pip wheel ninja packaging setuptools

echo "==> Runtime deps (explicit; no torch pins => won't touch image torch)"
pip install --upgrade \
  "transformers==4.54.1" accelerate datasets peft hf-transfer \
  "triton>=3.1.0" \
  codetiming hydra-core pandas "pyarrow>=15.0.0" pylatexenc \
  wandb "liger-kernel==0.5.8" \
  "diffusers>=0.30.0,<=0.31.0" tiktoken blobfile

echo "==> Open-dLLM editable, --no-deps (skip its flash-attn + conflicting pins)"
cd "$ROOT/Open-dLLM"
pip install --no-deps -e .

echo "==> Eval harnesses"
pip install -e lm-evaluation-harness
pip install --no-build-isolation -e human-eval-infilling

echo "==> Final torch sanity check"
python - <<'PY'
import torch
print("final torch", torch.__version__, "cuda", torch.cuda.is_available())
assert torch.cuda.is_available()
PY

echo "==> RTX 5090 (sm_120) note: flash-attn skipped (no Blackwell wheel for 2.7.x)."
echo "    Phase 1 eval runs in eager attention. Phase 2 training: revisit flash-attn>=2.8."

echo "==> Setup OK."
