# scripts

Run order on a fresh Vast remote box:

```
bash scripts/00_remote_setup.sh             # once per fresh instance (~5 min)
python scripts/02_smoke_open_dcoder.py      # 1-min smoke test: model loads + generates
bash scripts/01_baseline_humaneval.sh       # 10-30 min: HumanEval Pass@1 baseline
```

Outputs land in `runs/<UTC-timestamp>/` (gitignored).
