# Level 1 — Cross-section bit-match

## Thesis fragment
> The WGSL cross-section tables in `public/cross_sections.wgsl` agree with
> the raw G4EMLOW source data (`data/g4emlow/dna/sigma_*_e_*.dat`) within
> the noise of log-log subsampling, at every tabulated source energy in
> the WGSL grid's support.

## Baseline
The converter (`tools/convert_g4data.py`) takes 50–100 raw points per
species, applies a known scale factor, caps at 30 keV, and subsamples to
100 log-spaced points via log-log linear interpolation. The bit-match
check is therefore not strict equality — it is **relative agreement at
the source energies** under the same log-log interp kernel.

A passing Level 1 run shows:
- **Scale ratio** `peak_sigma_wgsl / peak_sigma_raw ∈ [0.95, 1.05]` —
  catches scale-factor regressions (the historical Champion 334× bug
  would land at ≈ 0.003 or ≈ 334).
- **Median relative error** at non-zero source energies < 1e-3 — log-log
  subsampling on a smooth curve introduces only sub-permille noise. A
  failure here would mean a systematic shift (wrong unit, dropped shell,
  off-by-one column) — exactly the bug class this experiment exists for.
- **p90 relative error** < 5e-2 — 90% of rows are within 5% even when
  near-threshold rows aren't. Catches non-systematic but widespread
  errors (e.g. partial shell sum).
- **Max relative error** < 1e-1, **reported as informational** —
  log-log linear interp on 100 grid points fundamentally produces up to
  ~8% rel_err *at* the water shell openings (1b₁ 10.79 / 3a₁ 13.39 /
  1b₂ 16.05 / 2a₁ 32.30 / 1a₁ 539.0 eV). This is a measured property
  of the converter's subsampling, not a bug. The hard cap at 10% is
  there to fail loudly if a bug bumps the noise above its baseline.

Below those bars the experiment fails with `status: "fail"` and a
diagnosis pointing at the worst-offending energy + species. Per the
RESEARCH.md "Honest negatives" standard, the JSON is committed in any
case with the failure surfaced.

## Experiments

### E1 — Born ionization total cross section vs G4EMLOW
- **Hypothesis:** For every energy E in
  `sigma_ionisation_e_born.dat ∩ [XE[0], XE[-1]]`, the WGSL `XI` table
  evaluated via log-log interp at E agrees with the raw shell-summed
  σ × scale factor (2.993 × 10⁻⁵ nm² per molecule per Geant4 unit).
- **Method:** Parse `data/g4emlow/dna/sigma_ionisation_e_born.dat` →
  83 rows × 6 cols (E + 5 shells). Compute `σ_total_raw(E) = Σ_shells × 2.993e-5`.
  Parse `public/cross_sections.wgsl` → extract `XE[100]` and `XI[100]`.
  For each raw E in the WGSL support, compute `σ_total_wgsl(E)` via
  log-log interp on `(XE, XI)`. Record relative error per row.
- **Pass bar (all required):**
  1. `peak_sigma_wgsl / peak_sigma_raw ∈ [0.95, 1.05]`, AND
  2. `median rel_err < 1e-3` over rows where `σ_total_raw > 0`, AND
  3. `p90 rel_err < 5e-2` over rows where `σ_total_raw > 1e-6 nm²`, AND
  4. `max rel_err < 1e-1` over the same meaningful rows (loose cap —
     log-log subsampling near shell openings can reach ~7-8% legitimately).
- **Failure = evidence:** Commit the JSON artifact with `status: "fail"`,
  `diagnosis: "<reason>"`, and the worst rows still in `rows[]`.
- **Diagnosis surface on fail:** scale ratio, median / p90 / max relative
  error, energy of worst row, σ_raw and σ_wgsl at that row.

### E2 — Emfietzoglou excitation total XS vs G4EMLOW
- **Status:** **Deferred.** Same shape as E1 against
  `sigma_excitation_e_emfietzoglou.dat` and the WGSL `XC` table.
- **Hypothesis (when implemented):** matches E1's structure; same scale
  factor (2.993e-5 nm²); same three pass-bar criteria.
- **Why this matters:** Emfietzoglou excitation drives the
  initial G(H) value through dissociative branching; a scale-factor bug
  here would silently shift G(H) at the 10–25% level — exactly the band
  webgpu-dna currently sits in.

### E3 — Champion elastic total XS vs G4EMLOW
- **Status:** **Deferred.** Compares WGSL `XL` to
  `sigma_elastic_e_champion.dat` with the **Champion-specific scale
  factor** `1e-16 cm² = 0.01 nm²` (NOT the Emfietzoglou 2.993e-5).
- **Why this matters:** This is exactly the experiment that would have
  caught the historical Champion 334× regression captured in
  memory/cross_section_fix.md. Its absence is the strongest argument
  for shipping E1 first as a template for the others.

### E4 — Sanche vibrational total XS vs G4EMLOW
- **Status:** **Deferred.** Compares the 9-mode Sanche table against
  `sigma_excitationvib_e_sanche.dat` with the 2× liquid-phase factor
  applied per CLAUDE.md.

## Artifacts
Each experiment writes
`experiments/results/<YYYY-MM-DD>/level-1/E<k>-<slug>.json`. The artifact
shape follows `experiments/lib/artifact.mjs` —
`{ meta, env, status, diagnosis, rows }`.

## Running
```bash
node experiments/runner.mjs E1
# or
npm run experiments -- E1
```

The runner writes the JSON to `experiments/results/<today>/level-1/`,
prints a summary line to stdout, and exits 0 on pass / 1 on fail.
