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
| 2026-05-07 | L1 | E1-ion-xs-match       | pass | peak_ratio 0.9987, median 8.46e-4, p90 1.78e-2 vs G4EMLOW Born σ_ion |
| 2026-05-07 | L1 | E2-exc-xs-match       | pass | peak_ratio 0.9970, median 2.42e-4, p90 3.51e-3 vs G4EMLOW Emfietzoglou σ_exc |
| 2026-05-07 | L1 | E3-elastic-xs-match   | pass | peak_ratio 0.9751, median 1.25e-4, p90 7.78e-4 vs G4EMLOW Champion σ_el (retroactive 334× catcher) |
| 2026-05-07 | L1 | E4-vib-xs-match       | pass | peak_ratio 1.0000, median 2.6e-16, max 6e-16 vs G4EMLOW Sanche σ_vib total |
| 2026-05-07 | L1 | E4b-vib-mode-fractions | pass | 342 (energy × mode) pairs vs raw σ_mode/σ_total; max sum dev 4e-8 (closes L1 fully) |
| 2026-05-07 | L2 | E5-csda-vs-g4-ntuple  | pass | CSDA 2714.4 vs 2756.5 (0.985×, 4.61σ); E-cons 100% vs 100% — surfaces 1.5% CSDA bias as statistically significant |
| 2026-05-07 | L2 | E6-mfp-vs-g4-ntuple   | pass | 6 energy bins (100 eV → 10 keV), MFP_total ratios [0.895, 0.965] (median 0.926), confirms README "MFP within 2-14%" claim numerically |
| 2026-05-07 | L4 | E10-irt-vs-karamitros | pass | 5 energies (1/3/5/10/20 keV) × 5 species vs Karamitros 2011 — surfaced G(eaq) V-shape at 1-3 keV (real track-end effect, ~40σ outside MC noise) |
