# Level 2 — Track structure vs Geant4 11.4.1 ntuple

## Status: protocol only
No experiments implemented in stage 1. This file defines the falsifiable
claims so the work can be picked up directly.

## Thesis fragment
> Per-primary track-structure metrics — CSDA range, mean free path,
> ionizations per primary, secondary energy spectrum — match the Geant4
> 11.4.1 dnaphysics ntuple within 2σ at every energy in [100 eV, 20 keV].

## Baseline
The repo currently scores these metrics ad-hoc via `validation/compare.py`
against a single 4096-primary 10 keV ntuple. Level 2 promotes that to a
research-grade experiment ladder: per-energy, per-metric, with named
seeds, mean ± SEM, and explicit pass bars.

## Experiments

### E5 — CSDA range vs Geant4 ntuple
- **Hypothesis:** Mean CSDA range over N=4096 primaries agrees with the
  Geant4 11.4.1 ntuple mean within 2σ at every energy in
  {100, 300, 500, 1000, 3000, 5000, 10000, 20000} eV.
- **Method:** Run the WebGPU primary-tracking pipeline N=4096 times per
  energy with seed `E5_CSDA`; compute mean and SEM. Compare to the Geant4
  ntuple's mean for the same N.
- **Pass bar:** `|μ_wgsl − μ_g4| / σ_g4 < 2` AND ratio `μ_wgsl / μ_g4 ∈ [0.9, 1.1]`.
- **Why both:** the σ-bar catches statistical drift; the ratio bar
  catches systematic biases that statistical noise might mask at large N.

### E6 — Mean free path vs Geant4 ntuple
- **Hypothesis:** Per-energy mean free path (path length / N_steps) at
  primary energies in {100, 300, 500, 1000, 3000, 5000, 10000, 20000} eV
  matches the Geant4 ntuple's MFP within 15%.
- **Method:** Same primary set as E5; bin by primary energy at the start
  of each step.
- **Pass bar:** `|MFP_wgsl − MFP_g4| / MFP_g4 < 0.15` at every energy bin.
  Current README claims 2-14% across all 8 energies — bar is the upper
  edge of that band.

### E7 — Ions per primary vs Geant4 ntuple
- **Hypothesis:** Total ionizations per primary at 10 keV agrees with
  the Geant4 ntuple's 509.1 within 1% (current: 509 ≈ 0.998×).
- **Method:** Sum primary + cascade ionizations per primary; mean over
  N=4096.
- **Pass bar:** `|μ_wgsl − μ_g4| / μ_g4 < 1e-2`.
- **Note:** This is the cleanest hard match in the validation suite —
  any drift here means primary-electron physics has regressed.

### E8 — Secondary kinetic energy spectrum vs Geant4 ntuple
- **Hypothesis:** Histogram of secondary KE at production matches the
  Geant4 ntuple's secondary spectrum (KS test p > 0.05 per energy bin).
- **Method:** Collect all secondaries created by 4096 primaries at
  10 keV; histogram in log-spaced KE bins from 1 eV to 5 keV; KS-test
  vs the ntuple's same-binned histogram.
- **Pass bar:** KS p-value > 0.05 at every primary energy.
- **Why:** Mean transfer matches at 57.15 eV per ARCHITECTURE.md; the
  full distribution test catches shape regressions that the mean misses.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-2/E<k>-<slug>.json`. Same shape
as Level 1 — `{ meta, env, status, diagnosis, rows }`. The `env` block
on GPU runs adds `adapter` and `limits` from `navigator.gpu.requestAdapter()`.
