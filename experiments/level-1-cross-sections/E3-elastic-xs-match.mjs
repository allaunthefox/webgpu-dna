// E3 — Champion elastic total cross section vs G4EMLOW.
//
// Hypothesis:
//   For every energy E in sigma_elastic_e_champion.dat that falls within
//   the WGSL XE support, σ_total_wgsl(E) — evaluated via log-log interp
//   on (XE, XL) — agrees with σ_raw(E) × 0.01 nm² (Champion scale).
//
// Why this experiment matters: Champion elastic has its OWN scale factor,
// distinct from Born ionization / Emfietzoglou excitation. Per
// G4DNAChampionElasticModel.cc:
//   scaleFactor = 1e-16 cm² = 1e-20 m² = 0.01 nm²
// The Born/Emfietzoglou scale (2.993e-5 nm²) is **334× smaller** than
// the Champion scale. Past regression: cross_section_fix.md memory
// captures a historical bug where convert_g4data.py applied the Born
// scale to Champion data — a silent 334× error in elastic XS that
// shortened CSDA tracks. This experiment, run before that commit
// would have shipped, would have caught it instantly via the
// peak_ratio bar:
//   peak_ratio = 1 / 334 ≈ 0.003  →  ∉ [0.95, 1.05]  →  hard fail
// Same cap structure for the inverse error (using 0.01 instead of
// 2.993e-5 on Born/Emfietzoglou): peak_ratio = 334.
//
// File shape: 2 columns (E, σ), 101 rows, 7.4 eV - 10 MeV. parseRawShellSum
// handles single-shell files transparently — sum of one column = that column.
//
// See experiments/level-1-cross-sections/protocol.md for the full spec.

import { join } from 'node:path';
import { SEEDS } from '../lib/seeds.mjs';
import { captureEnv } from '../lib/env.mjs';
import { runXsBitMatch } from '../lib/xs-bitmatch.mjs';

// Champion elastic scale factor (G4DNAChampionElasticModel.cc):
//   scaleFactor = 1e-16 cm² = 1e-20 m² = 0.01 nm²
const CHAMPION_SCALE_NM2 = 0.01;

const REPO_ROOT = join(import.meta.dirname, '..', '..');

export async function runE3() {
  const result = runXsBitMatch({
    rawPath: join(REPO_ROOT, 'data', 'g4emlow', 'dna', 'sigma_elastic_e_champion.dat'),
    scaleNm2: CHAMPION_SCALE_NM2,
    wgslPath: join(REPO_ROOT, 'public', 'cross_sections.wgsl'),
    wgslArrayName: 'XL',
    // Champion elastic has no physical threshold (elastic scattering
    // happens at all energies). The data file spans 7.4 eV - 10 MeV;
    // the WGSL grid spans 8 eV - 30 keV. The intersection is the
    // effective comparison range. Setting dataRangeMinEv = 8 to match
    // the WGSL lower bound (the 7.4 eV row is auto-excluded by the
    // support filter anyway); dataRangeMaxEv = 30000 to match the
    // WGSL upper bound.
    dataRangeMinEv: 8,
    dataRangeMaxEv: 30000,
  });

  return {
    meta: {
      protocol: 'E3-elastic-xs-match',
      hypothesis:
        'σ_wgsl(E) via log-log interp on (XE, XL) agrees with σ_raw(E) × 0.01 nm² (Champion scale) at every raw energy in the WGSL support.',
      passBar:
        'peak σ ratio ∈ [0.95, 1.05] (catches the historical 334× scale-factor regression) AND median rel_err < 1e-3 AND p90 rel_err < 5e-2 AND max rel_err < 1.5e-1.',
      seed: `E3_ELASTIC_XS=0x${SEEDS.E3_ELASTIC_XS.toString(16).toUpperCase()}`,
      warmup: 0,
      trials: 1,
      sources: {
        raw: 'data/g4emlow/dna/sigma_elastic_e_champion.dat',
        wgsl: 'public/cross_sections.wgsl',
        wgslArray: 'XL',
        scaleFactor: CHAMPION_SCALE_NM2,
      },
    },
    env: captureEnv(),
    ...result,
  };
}
