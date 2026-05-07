# Level 6 — Performance

## Status: protocol only
No experiments implemented in stage 1. Awaits a same-machine Geant4
single-thread baseline (currently the repo cites "~6 sec for 4096 primaries
on laptop GPU vs hours on CPU" qualitatively — Level 6 promotes that to
a measured number with proper sync timing).

## Thesis fragment
> The fused per-primary WGSL dispatch is bandwidth-limited at production
> N (≥ 4096 primaries) and beats Geant4-DNA single-thread CPU on the
> same machine by ≥ 10² wall-clock for a 10 keV 4096-primary run.

## Baseline
Per the kernel-fusion thesis (kernelfusion.dev), single-dispatch fused
kernels reach 71× on Apple Silicon and 56× on NVIDIA on launch-bound
workloads. Whether that magnitude transfers to webgpu-dna is the open
question this level answers. Existing claim "hours → 6 seconds" is
~10³× but unmeasured head-to-head.

## Experiments

### E15 — Phase A dispatch overhead α + per-amplitude β
- **Hypothesis:** Phase A wall time decomposes as `T(N) = α + β·N` with
  α ∈ [10, 500] μs (fixed submit + sync cost) and β > 0. At N ≥ 1024
  the variable term dominates.
- **Method:** Dispatch Phase A at N ∈ {1, 4, 16, 64, 256, 1024, 4096,
  16384}; W=5 warmup + T=20 trials with forced GPU sync before/after
  each measurement; OLS fit medians.
- **Pass bar:** α ∈ [10, 500] μs AND β > 0 AND R² ≥ 0.85. NOISY if
  > 50% of cells flag std/median > 0.1.
- **Why this matters:** the floor that the fusion thesis attacks. A
  measured α gives the speedup magnitude.

### E16 — Fused vs naive per-step dispatch (synthetic baseline)
- **Hypothesis:** The fused single-dispatch path is ≥ 10² faster than
  a "naive" baseline that submits one dispatch per physics step.
- **Method:** Implement a naive variant for measurement purposes only
  (one dispatch per step, no fusion); benchmark against the production
  fused path at N=4096, E=10 keV; report speedup.
- **Pass bar:** `t_naive / t_fused ≥ 100` AND fused path is bandwidth-
  bound by E15's α/β decomposition.
- **Why:** This is the kernel-fusion thesis demonstrated within
  webgpu-dna. Without it, the kernelfusion.dev framing on the site
  is unsupported speculation.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-6/E<k>-<slug>.json`. GPU runs;
artifact carries the full adapter info + limits block per webgpu-q's
shape.
