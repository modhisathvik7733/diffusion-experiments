#!/usr/bin/env bash
# DreamOn-v0-7B HumanEval-Infill eval at INT8, matching the 88.67 baseline
# methodology exactly (mask_expansion=true, min_gen_len=4, max_gen_len=64,
# steps=256, alg=entropy). The ONLY difference vs the 88.67 run is INT8
# weights — so the Pass@1 delta is purely the cost of quantization.
#
# Strategy: patch DreamOn/eval/evaluate.py in place to load model in 8-bit,
# trap restore on exit, run scripts/05 (which uses DreamOn's eval). Restore.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_int8_full_eval"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

EVAL_PY="$ROOT/DreamOn/eval/evaluate.py"
BACKUP="$EVAL_PY.preint8.backup"

if [ ! -f "$EVAL_PY" ]; then
  echo "ERROR: $EVAL_PY not found. Run scripts/05 first to clone DreamOn repo."
  exit 1
fi

# Restore stale backup if exists (previous crashed run)
if [ -f "$BACKUP" ]; then
  echo "==> Stale backup detected — restoring before patching"
  mv "$BACKUP" "$EVAL_PY"
fi

cp "$EVAL_PY" "$BACKUP"
restore_eval() {
  if [ -f "$BACKUP" ]; then
    mv "$BACKUP" "$EVAL_PY"
    echo "==> Restored evaluate.py"
  fi
}
trap restore_eval EXIT INT TERM

echo "==> Patching $EVAL_PY for INT8 loading"
python - <<PYEOF
import re
path = "$EVAL_PY"
with open(path) as f:
    code = f.read()

# Patch 1: ensure BitsAndBytesConfig is imported
if "BitsAndBytesConfig" not in code:
    # find any line that does 'from transformers import' and append
    code = re.sub(
        r"^(from transformers import [^\n]+)",
        r"\1\nfrom transformers import BitsAndBytesConfig",
        code,
        count=1,
        flags=re.MULTILINE,
    )

# Patch 2: replace AutoModel.from_pretrained call to use INT8 + device_map
old = "AutoModel.from_pretrained(cfg.ckpt, torch_dtype=torch.bfloat16, \n                                       trust_remote_code=True)"
new = ('AutoModel.from_pretrained('
       'cfg.ckpt, '
       'quantization_config=BitsAndBytesConfig(load_in_8bit=True), '
       'device_map={"": local_rank}, '
       'trust_remote_code=True)')
if old in code:
    code = code.replace(old, new)
else:
    # Try a broader regex match (handle whitespace variations)
    pattern = re.compile(
        r"AutoModel\.from_pretrained\(\s*cfg\.ckpt\s*,\s*"
        r"torch_dtype=torch\.bfloat16\s*,\s*"
        r"trust_remote_code=True\s*\)",
        re.DOTALL,
    )
    if pattern.search(code):
        code = pattern.sub(new, code)
    else:
        raise SystemExit("ERROR: could not locate AutoModel.from_pretrained call to patch")

# Patch 3: skip the .to(local_rank) call — bnb 8-bit models can't be moved
code = code.replace(
    "model = model.to(local_rank)  # Move to proper device before DDP",
    "# [INT8-patch] skipped .to() — bnb 8-bit model already placed via device_map"
)

# Patch 4: skip DDP wrap if model is loaded in 8-bit (DDP+bnb is fragile)
code = code.replace(
    "model = DDP(model, device_ids=[local_rank])",
    "# [INT8-patch] skipped DDP wrap — single GPU, bnb 8-bit\n"
    "    # model = DDP(model, device_ids=[local_rank])"
)

with open(path, "w") as f:
    f.write(code)
print("  Patched: import + from_pretrained + .to() + DDP")
PYEOF

# Sanity check: the patched code must still parse
python -c "import ast; ast.parse(open('$EVAL_PY').read()); print('==> Patched evaluate.py parses OK')"

echo ""
echo "==> Running DreamOn HumanEval-Infill eval with INT8-patched config"
echo "    (this uses scripts/05 which runs the SAME eval as the 88.67 baseline)"
echo ""

bash "$EXPDIR/scripts/05_baseline_humaneval_infill_dreamon.sh" 2>&1 | tee "$RUN_DIR/eval.log"

echo ""
echo "==> DONE. Vanilla bf16 baseline (same eval): Pass@1 = 88.67"
echo "    INT8 result: see tail above"
echo "    Logs: $RUN_DIR/eval.log"
