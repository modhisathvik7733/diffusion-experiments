# scripts

Run order on a fresh Vast remote box:

```
bash scripts/00_remote_setup.sh                          # once per fresh instance (~5 min)
python scripts/02_smoke_open_dcoder.py                   # 1-min smoke test (0.5B)
bash scripts/01_baseline_humaneval.sh                    # 20-30 min: 0.5B HumanEval
bash scripts/03_baseline_infilling_open_dcoder.sh        # 15-25 min: 0.5B HumanEval-Infill
python scripts/04_smoke_dreamon.py                       # 2-3 min: DreamOn 8B smoke
bash scripts/05_baseline_humaneval_infill_dreamon.sh     # 30-60 min: DreamOn HumanEval-Infill
python scripts/06_latency_bench_dreamon.py               # 5-10 min: p50/p95/p99 latency
```

Outputs land in `runs/<UTC-timestamp>/` (gitignored).
