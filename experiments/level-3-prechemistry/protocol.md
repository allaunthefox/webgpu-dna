# Level 3 — Pre-chemistry initial G-values

## Status: protocol only
No experiments implemented in stage 1. Awaits a Geant4 chem6 reference
ntuple (currently the repo only holds the dnaphysics ntuple).

## Thesis fragment
> Initial G-values at ~1 ps — the radical yields produced before the
> chemistry phase begins, after mother-molecule and product displacement
> but before reactions — reproduce a Geant4 chem6 ntuple with the
> documented Geant4-DNA convention (2.0 nm RMS mother displacement;
> species-specific σ for OH / e⁻aq / H; e⁻aq thermalization at 1.7 eV).

## Baseline
The repo currently quotes initial G-values
`{G_OH=4.51, G_eaq=3.79, G_H=0.79}` at 10 keV (validation/compare.py
WEBGPU dict). Geant4 doesn't expose initial G in dnaphysics — this is
why a chem6 ntuple is required as a reference.

## Experiments

### E9 — Initial G(OH), G(e⁻aq), G(H) at 1 ps vs Geant4 chem6
- **Hypothesis:** At 10 keV primary energy, initial G-values at the
  pre-chemistry → chemistry handoff (1 ps) match a Geant4 chem6 ntuple
  within 5% per species.
- **Method:** Run N=4096 primaries through Phase A + B + pre-chemistry
  with seed `E9_PRECHEM_G_INIT`. Score G(species) at the handoff time.
  Compare against the Geant4 chem6 ntuple at the same primary energy
  and N.
- **Pass bar:** `|G_wgsl − G_g4| / G_g4 < 0.05` per species.
- **Failure = evidence:** committing the JSON with the per-species
  ratios surfaces whether a deficit is uniform (calibration) or
  species-specific (branching ratio bug).
- **Blocker:** need a Geant4 chem6 ntuple at 10 keV with initial-G
  scoring enabled.

## Artifacts
`experiments/results/<YYYY-MM-DD>/level-3/E<k>-<slug>.json`.
