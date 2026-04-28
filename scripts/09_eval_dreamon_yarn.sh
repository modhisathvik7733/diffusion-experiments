#!/usr/bin/env bash
# HumanEval-Infill on DreamOn-v0-7B with YaRN context extension applied.
# Strategy: patch the cached config.json to add rope_scaling, run script 05
# unmodified (it'll pick up the patched config on model load), then restore.
#
# Compares to vanilla baseline (your previous run hit Pass@1 = 88.67).
# Result tells us if YaRN-extended DreamOn is shippable at 4x context.
set -euo pipefail

ROOT=${ROOT:-/workspace/diffusion-llm}
EXPDIR="$ROOT/diffusion-experiments"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_yarn_eval"
RUN_DIR="$EXPDIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

YARN_FACTOR=${YARN_FACTOR:-4}

# Find DreamOn's cached config.json — ask huggingface_hub for the actual path
# (your HF_HOME may be /workspace/.hf_home, /root/.cache/..., or elsewhere)
CONFIG=$(python - <<'PY'
from huggingface_hub import try_to_load_from_cache
p = try_to_load_from_cache("Dream-org/DreamOn-v0-7B", "config.json")
print(p if p else "")
PY
)
if [ -z "$CONFIG" ] || [ ! -f "$CONFIG" ]; then
  echo "ERROR: DreamOn config.json not found in HF cache."
  echo "huggingface_hub.try_to_load_from_cache returned: '$CONFIG'"
  echo "Tried searching for the model in HF_HOME=${HF_HOME:-<unset>}"
  echo ""
  echo "Diagnostics — checking common cache roots:"
  for d in \
    "$HOME/.cache/huggingface/hub" \
    "/workspace/.hf_home/hub" \
    "/root/.cache/huggingface/hub" \
    "${HF_HOME:-/dev/null}/hub"; do
    if [ -d "$d" ]; then
      echo "  exists: $d"
      ls "$d" 2>/dev/null | head -5
    fi
  done
  echo ""
  echo "Run scripts/04_smoke_dreamon.py or scripts/05_baseline_humaneval_infill_dreamon.sh first to download."
  exit 1
fi
echo "==> Found config: $CONFIG"
CONFIG_BACKUP="$CONFIG.preyarn.backup"

# Defensive: if a stale backup exists (previous crashed run), restore first
if [ -f "$CONFIG_BACKUP" ]; then
  echo "==> Stale backup detected — restoring before patching"
  mv "$CONFIG_BACKUP" "$CONFIG"
fi

# Backup + register trap to restore on ANY exit (success, failure, signal)
cp "$CONFIG" "$CONFIG_BACKUP"
restore_config() {
  if [ -f "$CONFIG_BACKUP" ]; then
    mv "$CONFIG_BACKUP" "$CONFIG"
    echo "==> Restored original config.json"
  fi
}
trap restore_config EXIT INT TERM

echo "==> Patching config.json with YaRN factor=$YARN_FACTOR"
python - <<PY
import json
p = "$CONFIG"
with open(p) as f:
    c = json.load(f)
orig_max = c["max_position_embeddings"]
factor = float("$YARN_FACTOR")
c["rope_scaling"] = {
    "type": "yarn",
    "factor": factor,
    "original_max_position_embeddings": orig_max,
}
c["max_position_embeddings"] = int(orig_max * factor)
with open(p, "w") as f:
    json.dump(c, f, indent=2)
print(f"  rope_scaling=yarn factor={factor}")
print(f"  max_position_embeddings: {orig_max} -> {c['max_position_embeddings']}")
PY

echo ""
echo "==> Running HumanEval-Infill eval with YaRN-patched config"
echo "    (reusing scripts/05; it'll load the model with patched config automatically)"
echo ""

# Run the existing eval script. Output goes to its own runs/ dir + we tee a copy here.
bash "$EXPDIR/scripts/05_baseline_humaneval_infill_dreamon.sh" 2>&1 | tee "$RUN_DIR/eval.log"

echo ""
echo "==> DONE. Vanilla baseline was Pass@1 = 88.67."
echo "    YaRN factor=$YARN_FACTOR result is in the tail above."
echo "    Logs: $RUN_DIR/eval.log"
