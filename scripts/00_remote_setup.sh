#!/usr/bin/env bash
# Idempotent env setup for the Vast.ai remote (vastai/pytorch_cuda-12.8.1-auto image).
# Uses the image's pre-installed torch (already tuned for Blackwell sm_120).
# Assumes /workspace/diffusion-llm/{LLaDA,Open-dLLM,dllm,Dream,diffusion-experiments} exist.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
cd "$ROOT"

echo "==> Verify base image torch + CUDA"
python - <<'PY'
import torch, sys
assert torch.cuda.is_available(), "CUDA not available — wrong image?"
print("torch", torch.__version__, "device", torch.cuda.get_device_name(0),
      "cap", torch.cuda.get_device_capability(0))
PY

echo "==> Upgrade pip toolchain"
pip install --upgrade pip wheel ninja

echo "==> Install Open-dLLM upstream Python deps"
pip install \
  "transformers==4.54.1" accelerate datasets peft hf-transfer \
  tensordict torchdata "triton>=3.1.0" \
  codetiming hydra-core pandas "pyarrow>=15.0.0" pylatexenc \
  wandb "liger-kernel==0.5.8" \
  pytest yapf py-spy pyext pre-commit ruff packaging

echo "==> Install Open-dLLM editable + its eval harnesses"
cd "$ROOT/Open-dLLM"
pip install -e .
pip install -e lm-evaluation-harness
pip install -e human-eval-infilling

echo "==> RTX 5090 (sm_120) note: flash-attn 2.7.x has no prebuilt Blackwell wheel."
echo "    Phase 1 eval works without flash-attn (eager attention)."
echo "    For Phase 2 we'll revisit: try flash-attn>=2.8 or build from source."

echo "==> Setup OK."
