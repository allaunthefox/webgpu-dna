// E4 — Sanche vibrational total cross section vs G4EMLOW.
//
// Hypothesis:
//   For every energy E in sigma_excitationvib_e_sanche.dat that falls
//   within the WGSL XVE support and is above the E=0 placeholder row,
//   σ_total_wgsl(E) = XVS[i] agrees with σ_total_raw(E) =
//   Σ_modes(E) × 0.01 nm² × 2.0 (liquid factor) at near-fp32 precision.
//
// Why the bars are MUCH tighter than E1/E2/E3: the converter does NOT
// subsample Sanche — it passes the raw 38-point grid through unchanged.
// So the only difference between σ_wgsl and σ_raw at each grid energy
// is fp32 round-off (~1e-7 relative). Tighter bars catch any
// missing-row / scale / liquid-factor regression.
//
// Source-data quirks:
//   - File starts at E=0 (first row) which is sub-physical; excluded
//     via dataRangeMinEv=1.7 (next grid point).
//   - File shape: 38 rows × 10 cols (E + 9 modes), bare-LF separated.
//   - 9 modes: vibrational thresholds 0.01 / 0.024 / 0.061 / 0.092 /
//     0.204 / 0.417 / 0.460 / 0.500 / 0.835 eV (per VIB_LEV in
//     convert_g4data.py, hardcoded from G4DNASancheExcitationModel.cc).
//
// Why this matters: vibrational excitation is the dominant
// energy-loss channel for sub-100 eV thermalizing secondaries. A
// missing 2× liquid-phase factor would silently halve secondary
// thermalization range — caught by the peak-ratio bar.
//
// Per-mode fractions (XVMF[38*9]) are NOT validated by E4 — that's
// the deferred E4b (would compare each mode's σ × 0.02 against the
// fraction reconstruction). Out of scope for closing L1.

import { join } from 'node:path';
import { SEEDS } from '../lib/seeds.mjs';
import { captureEnv } from '../lib/env.mjs';
import { runXsBitMatch } from '../lib/xs-bitmatch.mjs';

// Sanche scale factor:
//   G4DNASancheExcitationModel.cc:  scaleFactor = 1e-16 cm² = 0.01 nm²
//   Liquid-phase factor:            VIB_LIQUID_FACTOR = 2.0
//   Effective total:                0.01 × 2.0 = 0.02 nm² per raw unit
const SANCHE_SCALE_NM2 = 0.02;

// Tight bars — Sanche WGSL is non-subsampled, only fp32 round-off
// applies. Loose bars from the L1 default (15% max) would miss real
// regressions on this XS family.
const SANCHE_PASS_BAR = Object.freeze({
  peakRatioMin: 0.99,
  peakRatioMax: 1.01,
  medianRelErrMax: 1e-5,
  p90RelErrMax: 1e-4,
  maxRelErrMax: 1e-3,
  meaningfulSigmaNm2: 1e-6,
});

const REPO_ROOT = join(import.meta.dirname, '..', '..');

export async function runE4() {
  const result = runXsBitMatch({
    rawPath: join(REPO_ROOT, 'data', 'g4emlow', 'dna', 'sigma_excitationvib_e_sanche.dat'),
    scaleNm2: SANCHE_SCALE_NM2,
    wgslPath: join(REPO_ROOT, 'public', 'cross_sections.wgsl'),
    wgslArrayName: 'XVS',
    energyArrayName: 'XVE',
    // Exclude the E=0 placeholder row (first entry of the source file
    // and the WGSL grid). The lowest physical Sanche threshold (mode 0
    // = 0.01 eV) is far below 1.7 eV, but the file's grid begins
    // E=0 → 1.7 eV → 3.2 eV → ..., and 0 eV is not a meaningful
    // primary energy. dataRangeMaxEv = 100 matches the source upper
    // limit (= XVE[-1] = 100 eV).
    dataRangeMinEv: 1.7,
    dataRangeMaxEv: 100,
    passBar: SANCHE_PASS_BAR,
  });

  return {
    meta: {
      protocol: 'E4-vib-xs-match',
      hypothesis:
        'σ_total_wgsl(E) = XVS[i] agrees with σ_total_raw(E) = Σ_modes(E) × 0.02 nm² (= 0.01 × 2.0 liquid factor) at every raw energy in the WGSL support, to fp32 round-off precision (Sanche WGSL is non-subsampled).',
      passBar:
        'peak σ ratio ∈ [0.99, 1.01] AND median rel_err < 1e-5 AND p90 rel_err < 1e-4 AND max rel_err < 1e-3.',
      seed: `E4_VIB_XS=0x${SEEDS.E4_VIB_XS.toString(16).toUpperCase()}`,
      warmup: 0,
      trials: 1,
      sources: {
        raw: 'data/g4emlow/dna/sigma_excitationvib_e_sanche.dat',
        wgsl: 'public/cross_sections.wgsl',
        wgslArray: 'XVS',
        energyArray: 'XVE',
        scaleFactor: SANCHE_SCALE_NM2,
        scaleBreakdown: '0.01 nm² (G4DNASancheExcitationModel.cc) × 2.0 (liquid-phase factor)',
      },
    },
    env: captureEnv(),
    ...result,
  };
}
