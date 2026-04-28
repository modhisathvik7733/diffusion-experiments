#!/usr/bin/env bash
# Idempotent env setup for the Vast.ai remote (Linux, CUDA 12.8, RTX 5090).
# Assumes /workspace/diffusion-llm/{LLaDA,Open-dLLM,dllm,Dream,diffusion-experiments} already exist.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$ROOT/.venv" ]; then
  uv venv "$ROOT/.venv" --python 3.10
fi
source "$ROOT/.venv/bin/activate"

uv pip install --upgrade pip wheel ninja

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

uv pip install \
  "transformers==4.54.1" accelerate datasets peft hf-transfer \
  tensordict torchdata "triton>=3.1.0" \
  codetiming hydra-core pandas "pyarrow>=15.0.0" pylatexenc \
  wandb "liger-kernel==0.5.8" \
  pytest yapf py-spy pyext pre-commit ruff packaging \
  bitsandbytes

cd "$ROOT/Open-dLLM"
uv pip install -e .
uv pip install -e lm-evaluation-harness
uv pip install -e human-eval-infilling

echo "==> RTX 5090 (sm_120) note: flash-attn 2.7.x has no prebuilt wheel for Blackwell."
echo "    Phase 1 eval works without flash-attn (model runs in eager attention)."
echo "    For Phase 2 training we will revisit: try flash-attn>=2.8 or build from source."

echo "==> Setup OK. Activate later with: source $ROOT/.venv/bin/activate"
