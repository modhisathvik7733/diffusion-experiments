# scripts

Run order on a fresh Vast remote box:

```
bash scripts/00_remote_setup.sh        # once per fresh instance
bash scripts/01_baseline_humaneval.sh  # reproduce Open-dCoder HumanEval baseline
```

Outputs land in `runs/<UTC-timestamp>/` (gitignored).
