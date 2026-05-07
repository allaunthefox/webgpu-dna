# Level 4 — Chemistry G-values at 1 μs

## Status: protocol only
No experiments implemented in stage 1. Awaits ingest of Tran 2024 (Med
Phys, ESA BioRad III review) as the canonical comparison set across all
Geant4-DNA chemistry constructors.

## Thesis fragment
> G-values at t = 1 μs from the Karamitros 2011 9-reaction IRT in
> `public/irt-worker.js` reproduce the published reference values from
> Karamitros 2011 / Tran 2024, with explicit accounting for the
> LET-induced deficit at 10 keV vs the ~1 MeV low-LET reference.

## Baseline
Current 10 keV results vs Karamitros 2011 (low-LET, ~1 MeV):
- G(OH)   = 1.55 → 0.62× (deficit expected at 10 keV LET)
- G(e⁻aq) = 1.41 → 0.56× (deficit expected at 10 keV LET)
- G(H)    = 0.71 → 1.24× (within ~25%)
- G(H₂O₂) = 0.60 → 0.83× (within ~17%)
- G(H₂)   = 0.47 → 1.11× (within ~11%)

The OH and e⁻aq deficits are physically expected — Karamitros's reference
is for low-LET radiation where track-core radical recombination is lower.
A research-grade pass bar must encode this expectation; comparing 10 keV
results to low-LET references with a flat 10% bar would always fail for
the wrong reason.

## Experiments

### E10 — IRT G-values at 1 μs vs Karamitros 2011 / Tran 2024
- **Hypothesis:** At 10 keV, G(H), G(H₂), G(H₂O₂) match Karamitros 2011
  within 25%; G(OH) and G(e⁻aq) sit at 0.5×–0.8× Karamitros (LET deficit
  band) AND match Tran 2024 chem6 at 10 keV within 20%.
- **Method:** Run N ≥ 4096 primaries through the full pipeline; record
  per-species G(t=1 μs). Average over two seeds (`E10_IRT_G_VALUES`
  and a derived companion) per the existing two-run convention.
- **Pass bar:**
  1. G(H), G(H₂), G(H₂O₂): `|G_wgsl / G_karamitros − 1| < 0.25`.
  2. G(OH), G(e⁻aq): ratio ∈ [0.5, 0.8] vs Karamitros (LET-band).
  3. AND `|G_wgsl / G_tran_chem6_10keV − 1| < 0.20` for all 5 species.
- **Why two references:** Karamitros pins the chemistry kernel; Tran
  2024 chem6 at 10 keV pins the LET interpretation. Without the second
  reference, the LET claim isn't falsifiable.
- **Blocker:** ingest Tran 2024's 10 keV table.

### E11 — GPU chemistry backend vs IRT worker
- **Hypothesis:** The `chemBackend: 'gpu'` path produces G-values within
  10% of the IRT worker at the same N and same time checkpoints.
- **Method:** Run the same primary set through both backends; compare
  G(species, t) at all 7 time checkpoints (0.1 ps → 1 μs).
- **Pass bar:** `|G_gpu − G_irt| / G_irt < 0.10` at every (species, t).
- **Expected outcome:** **fail.** CLAUDE.md flags the GPU backend as
  undercounting long-time reactions because the spatial-hash search
  radius is narrower than the diffusion σ at 30 ns timesteps. This
  experiment exists to *quantify* the deficit and commit it as
  `status: "fail"` with a diagnosis pointing at the diffusion-σ vs
  hash-radius math — the marquee honest negative for Level 4.
- **Why ship the failure:** webgpu-q's pattern. A documented failure
  with reproducible numbers is research-grade; a one-line "known gap"
  in CLAUDE.md is a footnote.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-4/E<k>-<slug>.json`.
