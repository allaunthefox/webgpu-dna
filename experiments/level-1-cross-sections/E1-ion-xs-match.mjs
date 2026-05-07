// E1 — Born ionization total cross section vs G4EMLOW.
//
// Hypothesis:
//   For every energy E in sigma_ionisation_e_born.dat that falls within
//   the WGSL XE support, σ_total_wgsl(E) — evaluated via log-log interp
//   on (XE, XI) — agrees with σ_total_raw(E) = Σ_shells × 2.993e-5 nm².
//
// Pass bar (all four required):
//   1. peak σ ratio (wgsl / raw) ∈ [0.95, 1.05]      — catches scale-factor bugs
//   2. median rel_err < 1e-3 on rows with σ_raw > 0  — catches systematic shifts
//   3. p90 rel_err < 5e-2 on rows with σ_raw > 1e-6  — catches widespread errors
//   4. max rel_err < 1e-1 on the same meaningful set — loose cap; log-log
//      subsampling near water shell openings (10.79 / 13.39 / 16.05 / 32.30
//      / 539.0 eV) can legitimately produce ~7–8% rel_err and is a measured
//      property of the 100-point WGSL grid, not a bug.
//
// See experiments/level-1-cross-sections/protocol.md for the full spec.

import { join } from 'node:path';
import { SEEDS } from '../lib/seeds.mjs';
import { captureEnv } from '../lib/env.mjs';
import { runXsBitMatch } from '../lib/xs-bitmatch.mjs';

// Geant4-DNA Born ionization scale factor (G4DNABornIonisationModel.cc):
//   scaleFactor = (1e-22 / 3.343) m² = 2.993e-23 m² = 2.993e-5 nm²
const BORN_SCALE_NM2 = 2.993e-5;

const REPO_ROOT = join(import.meta.dirname, '..', '..');

export async function runE1() {
  const result = runXsBitMatch({
    rawPath: join(REPO_ROOT, 'data', 'g4emlow', 'dna', 'sigma_ionisation_e_born.dat'),
    scaleNm2: BORN_SCALE_NM2,
    wgslPath: join(REPO_ROOT, 'public', 'cross_sections.wgsl'),
    wgslArrayName: 'XI',
    // Born ionization physical range: lowest shell (1b₁) at 10.79 eV
    // (per CLAUDE.md), upper bound 1 MeV. The WGSL grid is capped at
    // 30 keV by convert_g4data.py — meaningful comparisons live in
    // [10.79 eV, 30 keV].
    dataRangeMinEv: 10.79,
    dataRangeMaxEv: 30000,
  });

  return {
    meta: {
      protocol: 'E1-ion-xs-match',
      hypothesis:
        'σ_total_wgsl(E) via log-log interp on (XE, XI) agrees with σ_total_raw(E) = Σ_shells × 2.993e-5 nm² at every raw energy in the WGSL support.',
      passBar:
        'peak σ ratio ∈ [0.95, 1.05] AND median rel_err < 1e-3 over σ_raw > 0 AND p90 rel_err < 5e-2 over σ_raw > 1e-6 nm² AND max rel_err < 1e-1 (loose cap; log-log subsampling near shell openings).',
      seed: `E1_ION_XS=0x${SEEDS.E1_ION_XS.toString(16).toUpperCase()}`,
      warmup: 0,
      trials: 1,
      sources: {
        raw: 'data/g4emlow/dna/sigma_ionisation_e_born.dat',
        wgsl: 'public/cross_sections.wgsl',
        wgslArray: 'XI',
        scaleFactor: BORN_SCALE_NM2,
      },
    },
    env: captureEnv(),
    ...result,
  };
}
