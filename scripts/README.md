# scripts

Run order on a fresh Vast remote box:

```
bash scripts/00_remote_setup.sh                       # once per fresh instance (~5 min)
python scripts/02_smoke_open_dcoder.py                # 1-min smoke test
bash scripts/01_baseline_humaneval.sh                 # 20-30 min: HumanEval Pass@1
bash scripts/03_baseline_infilling_open_dcoder.sh     # 15-25 min: HumanEval-Infill Pass@1
```

Outputs land in `runs/<UTC-timestamp>/` (gitignored).
