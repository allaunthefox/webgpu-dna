// E2 — Emfietzoglou excitation total cross section vs G4EMLOW.
//
// Hypothesis:
//   For every energy E in sigma_excitation_e_emfietzoglou.dat that falls
//   within the WGSL XE support, σ_total_wgsl(E) — evaluated via log-log
//   interp on (XE, XC) — agrees with σ_total_raw(E) = Σ_levels × 2.993e-5 nm².
//
// Why excitation matters: Emfietzoglou excitation drives the initial
// G(H) value through dissociative branching. A scale-factor bug here
// would silently shift G(H) at the 10-25% level — the same band that
// webgpu-dna's chemistry currently sits in vs Karamitros 2011.
//
// Same pass bar as E1; same scale factor (2.993e-5 nm² per molecule).
// The Emfietzoglou raw file uses bare CR line terminators (legacy Mac
// format) — `parseRawShellSum` handles all three line-ending styles.
//
// See experiments/level-1-cross-sections/protocol.md for the full spec.

import { join } from 'node:path';
import { SEEDS } from '../lib/seeds.mjs';
import { captureEnv } from '../lib/env.mjs';
import { runXsBitMatch } from '../lib/xs-bitmatch.mjs';

// Same scale factor as Born (G4EMLOW Emfietzoglou inherits the
// 1e-22/3.343 m² Geant4-internal-units convention):
//   scaleFactor = 2.993e-23 m² = 2.993e-5 nm²
const EMFIETZOGLOU_SCALE_NM2 = 2.993e-5;

const REPO_ROOT = join(import.meta.dirname, '..', '..');

export async function runE2() {
  const result = runXsBitMatch({
    rawPath: join(REPO_ROOT, 'data', 'g4emlow', 'dna', 'sigma_excitation_e_emfietzoglou.dat'),
    scaleNm2: EMFIETZOGLOU_SCALE_NM2,
    wgslPath: join(REPO_ROOT, 'public', 'cross_sections.wgsl'),
    wgslArrayName: 'XC',
    // Emfietzoglou excitation physical range: lowest level A¹B₁ at
    // 8.22 eV (per CLAUDE.md), upper bound 10 keV (where the source
    // data ends). Outside this range the converter zeros out, producing
    // boundary artifacts that aren't bugs:
    //   - E < 8.22 eV: raw file has sub-threshold tails; WGSL XC[0]=0 is
    //     the physically correct behavior.
    //   - E ≥ 10000 eV: WGSL transitions to 0 at XE[86]=10181, so
    //     log-log interp at exactly 10000 falls in the zero-out zone.
    dataRangeMinEv: 8.22,
    dataRangeMaxEv: 9999,
  });

  return {
    meta: {
      protocol: 'E2-exc-xs-match',
      hypothesis:
        'σ_total_wgsl(E) via log-log interp on (XE, XC) agrees with σ_total_raw(E) = Σ_levels × 2.993e-5 nm² at every raw energy in the WGSL support.',
      passBar:
        'peak σ ratio ∈ [0.95, 1.05] AND median rel_err < 1e-3 over σ_raw > 0 AND p90 rel_err < 5e-2 over σ_raw > 1e-6 nm² AND max rel_err < 1e-1.',
      seed: `E2_EXC_XS=0x${SEEDS.E2_EXC_XS.toString(16).toUpperCase()}`,
      warmup: 0,
      trials: 1,
      sources: {
        raw: 'data/g4emlow/dna/sigma_excitation_e_emfietzoglou.dat',
        wgsl: 'public/cross_sections.wgsl',
        wgslArray: 'XC',
        scaleFactor: EMFIETZOGLOU_SCALE_NM2,
      },
    },
    env: captureEnv(),
    ...result,
  };
}
