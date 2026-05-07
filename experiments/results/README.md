# Experiments — bench logs

All run artifacts live under
`experiments/results/<YYYY-MM-DD>/level-N/E<k>-<slug>.json`.

Per RESEARCH.md "Honest negatives" standard, failed runs are committed
alongside passing ones. Each artifact carries:

- `meta` — protocol ID, hypothesis, pass bar, named seed, warmup, trials
- `env` — git SHA, timestamp, runner, platform, hardware
- `status` — `pass` | `fail` | `noisy`
- `diagnosis` — short failure reason when `status != "pass"`
- `summary` — aggregated metrics (median, p90, max, peak ratio, etc.)
- `rows` — per-trial / per-cell observations

## Index (chronological)

| Date | Level | Experiment | Status | Headline |
|------|-------|-----------|--------|----------|
| 2026-05-07 | L1 | E1-ion-xs-match | pass | peak_ratio 0.9987, median 8.46e-4, p90 1.78e-2 vs G4EMLOW Born σ_ion |
| 2026-05-07 | L1 | E2-exc-xs-match | pass | peak_ratio 0.9970, median 2.42e-4, p90 3.51e-3 vs G4EMLOW Emfietzoglou σ_exc |
