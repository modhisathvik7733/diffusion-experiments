# diffusion-experiments

Research code for grafted hybrid-Mamba diffusion language models, layered on top of upstream baselines.

## Layout

```
diffusion-experiments/
├── scripts/    # shell entrypoints for setup, eval, training
├── models/     # custom modeling code (hybrid Mamba blocks etc.)
├── configs/    # experiment configs
└── runs/       # local outputs (gitignored)
```

Upstream baselines (`LLaDA`, `Open-dLLM`, `dllm`, `Dream`) live as siblings of this folder, cloned read-only:

```
/workspace/diffusion-llm/
├── LLaDA/
├── Open-dLLM/
├── dllm/
├── Dream/
└── diffusion-experiments/   <-- this repo
```

## Workflow

Local (Mac) — edit and push:

```
git pull
# edit files
git add -A && git commit -m "..." && git push
```

Remote (Vast RTX 5090) — pull and run:

```
cd /workspace/diffusion-llm/diffusion-experiments
git pull
bash scripts/01_baseline_humaneval.sh
```

## Phases

- **Phase 0 — env setup.** `scripts/00_remote_setup.sh`. Idempotent install of upstream Open-dLLM deps in a venv at `/workspace/diffusion-llm/.venv`.
- **Phase 1 — reproduce baseline.** `scripts/01_baseline_humaneval.sh`. Run Open-dCoder 0.5B HumanEval and confirm we hit the published Pass@1 ≈ 20.8.
- **Phase 2 — hybrid Mamba graft.** `models/hybrid_mamba.py` + a continued-pretrain script (TBD). Replace 1–2 attention layers in Open-dCoder with bidirectional Mamba-2, zero-init the new output projection, short continued pretrain on FineCode.

Phase 1 must pass before we touch Phase 2.

## Hardware target

1× RTX 5090 (Blackwell, sm_120), 31.8 GB VRAM, CUDA 12.8. Open-dLLM 0.5B fits comfortably. Larger checkpoints (LLaDA-8B / Dream-7B) are inference-only on this card.
