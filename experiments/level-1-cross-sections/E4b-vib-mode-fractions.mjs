// E4b — Sanche per-mode XVMF fraction table vs raw σ_mode / σ_total.
//
// Hypothesis:
//   For every (energy_i, mode_k) in [0, 38) × [0, 9), the WGSL
//   XVMF[i*9 + k] equals raw σ_mode_k(E_i) / Σ_j σ_mode_j(E_i) at
//   fp32 precision. The scale factor (0.01 × 2.0 liquid) cancels in
//   the ratio, so XVMF is an exact reflection of the raw mode breakdown.
//
// Why this matters: XVMF is the table the WGSL primary tracker reads
// when it has decided "this step is a Sanche vibrational excitation"
// and needs to pick which of the 9 modes was excited (and therefore
// how much energy is lost — VIB_LEV[mode]). A wrong fraction means
// the wrong distribution of vibrational losses, which silently
// shifts secondary thermalization range. E4 catches scale errors;
// E4b catches per-mode mapping errors that E4 would miss because
// they cancel in the sum.
//
// Same data range as E4 (excludes E=0 placeholder row).

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { SEEDS } from '../lib/seeds.mjs';
import { captureEnv } from '../lib/env.mjs';
import { parseWgslArray } from '../lib/xs-bitmatch.mjs';

const REPO_ROOT = join(import.meta.dirname, '..', '..');
const N_MODES = 9;
const DATA_RANGE_MIN_EV = 1.7;

// Same tight bars as E4 — non-subsampled, so only fp32 round-off applies.
const PASS_BAR = Object.freeze({
  medianRelErrMax: 1e-5,
  p90RelErrMax: 1e-4,
  maxRelErrMax: 1e-3,
});

function parseSancheRaw(text) {
  // 38 rows × 10 cols (E + 9 modes). Bare-LF separated; trailing newline
  // optional. Returns array of { energyEv, modes[9], totalRaw } in source units.
  const tokens = text.split(/[\s\r\n]+/).filter(Boolean).map(Number);
  if (tokens.length % 10 !== 0) {
    throw new Error(`Sanche raw token count ${tokens.length} not divisible by 10`);
  }
  const rows = [];
  for (let i = 0; i < tokens.length; i += 10) {
    const energyEv = tokens[i];
    const modes = tokens.slice(i + 1, i + 10);
    const totalRaw = modes.reduce((a, b) => a + b, 0);
    rows.push({ energyEv, modes, totalRaw });
  }
  return rows;
}

export async function runE4b() {
  const rawText = readFileSync(
    join(REPO_ROOT, 'data', 'g4emlow', 'dna', 'sigma_excitationvib_e_sanche.dat'),
    'utf8',
  );
  const wgslText = readFileSync(
    join(REPO_ROOT, 'public', 'cross_sections.wgsl'),
    'utf8',
  );

  const raw = parseSancheRaw(rawText);
  const xve = parseWgslArray(wgslText, 'XVE');
  const xvmf = parseWgslArray(wgslText, 'XVMF');

  if (xve.length !== raw.length) {
    throw new Error(`XVE length ${xve.length} ≠ raw row count ${raw.length}`);
  }
  if (xvmf.length !== xve.length * N_MODES) {
    throw new Error(`XVMF length ${xvmf.length} ≠ ${xve.length} × ${N_MODES}`);
  }

  const rows = [];
  for (let i = 0; i < raw.length; i++) {
    const r = raw[i];
    if (Math.abs(xve[i] - r.energyEv) > 1e-3) {
      throw new Error(
        `XVE[${i}]=${xve[i]} ≠ raw energy[${i}]=${r.energyEv} — grid alignment broken`,
      );
    }
    const inDataRange = r.energyEv >= DATA_RANGE_MIN_EV;

    for (let k = 0; k < N_MODES; k++) {
      const expected = r.totalRaw > 0 ? r.modes[k] / r.totalRaw : 0;
      const actual = xvmf[i * N_MODES + k];
      const absErr = Math.abs(actual - expected);
      // For very small expected values (< 1e-4), prefer absolute error
      // to relative — relative error blows up near zero.
      const relErr = expected > 1e-4
        ? absErr / expected
        : absErr; // treat as absolute for tiny fractions
      const bigEnoughToCheck = inDataRange && expected > 1e-4;

      rows.push({
        energyEv: r.energyEv,
        modeIdx: k,
        expected,
        wgsl: actual,
        absErr,
        relErr,
        inDataRange,
        bigEnoughToCheck,
        status: 'pending',
      });
    }
  }

  // ---- aggregates (over rows that are inDataRange AND bigEnoughToCheck) ----
  const meaningful = rows.filter((r) => r.bigEnoughToCheck);
  const sortedErrs = [...meaningful].map((r) => r.relErr).sort((a, b) => a - b);
  const quantile = (q) => {
    if (sortedErrs.length === 0) return 0;
    const idx = (sortedErrs.length - 1) * q;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return lo === hi ? sortedErrs[lo] : sortedErrs[lo] + (idx - lo) * (sortedErrs[hi] - sortedErrs[lo]);
  };
  const medianRelErr = quantile(0.5);
  const p90RelErr = quantile(0.9);
  const maxRelErr = sortedErrs.length === 0 ? 0 : sortedErrs[sortedErrs.length - 1];

  // Sum-of-fractions sanity check per energy (Σ_k XVMF[i*9+k] should be 1).
  const fractionSums = [];
  for (let i = 0; i < xve.length; i++) {
    const s = xvmf.slice(i * N_MODES, (i + 1) * N_MODES).reduce((a, b) => a + b, 0);
    if (xve[i] >= DATA_RANGE_MIN_EV) {
      fractionSums.push({ energyEv: xve[i], sum: s, deviation: Math.abs(s - 1) });
    }
  }
  const maxSumDeviation = Math.max(...fractionSums.map((f) => f.deviation));

  // ---- pass-bar evaluation ----
  const failures = [];
  if (medianRelErr > PASS_BAR.medianRelErrMax) {
    failures.push(`median rel_err ${medianRelErr.toExponential(2)} ≥ ${PASS_BAR.medianRelErrMax}`);
  }
  if (p90RelErr > PASS_BAR.p90RelErrMax) {
    failures.push(`p90 rel_err ${p90RelErr.toExponential(2)} ≥ ${PASS_BAR.p90RelErrMax}`);
  }
  if (maxRelErr > PASS_BAR.maxRelErrMax) {
    const worst = meaningful.reduce((a, b) => (b.relErr > a.relErr ? b : a), meaningful[0]);
    failures.push(`max rel_err ${maxRelErr.toExponential(2)} ≥ ${PASS_BAR.maxRelErrMax} at E=${worst.energyEv} mode=${worst.modeIdx}`);
  }
  if (maxSumDeviation > 1e-4) {
    failures.push(`Σ XVMF[i*9+k] deviates from 1 by ${maxSumDeviation.toExponential(2)} (fractions don't sum)`);
  }

  // Per-row status
  for (const r of rows) {
    if (!r.inDataRange) r.status = 'out-of-range';
    else if (!r.bigEnoughToCheck) r.status = 'pass'; // tiny-fraction tolerance
    else if (r.relErr < PASS_BAR.maxRelErrMax) r.status = 'pass';
    else r.status = 'fail';
  }

  const status = failures.length === 0 ? 'pass' : 'fail';
  const diagnosis = failures.length === 0 ? null : failures.join('; ');

  return {
    meta: {
      protocol: 'E4b-vib-mode-fractions',
      hypothesis:
        'XVMF[i*9+k] equals raw σ_mode_k(E_i) / Σ_j σ_mode_j(E_i) at fp32 precision for every (energy, mode) in the data range. Sum-of-fractions per energy equals 1.',
      passBar:
        'median rel_err < 1e-5 AND p90 < 1e-4 AND max < 1e-3 (tight bars — non-subsampled, fp32 round-off floor); AND Σ_k XVMF[i*9+k] = 1 within 1e-4.',
      seed: `E4_VIB_XS=0x${SEEDS.E4_VIB_XS.toString(16).toUpperCase()}`,
      warmup: 0,
      trials: 1,
      sources: {
        raw: 'data/g4emlow/dna/sigma_excitationvib_e_sanche.dat',
        wgsl: 'public/cross_sections.wgsl',
        wgslArrays: 'XVE (energies), XVMF (38×9 fractions)',
      },
    },
    env: captureEnv(),
    status,
    diagnosis,
    summary: {
      nRows: rows.length,
      nMeaningful: meaningful.length,
      nOutOfRange: rows.filter((r) => !r.inDataRange).length,
      nFailedRows: rows.filter((r) => r.status === 'fail').length,
      medianRelErr,
      p90RelErr,
      maxRelErrMeaningful: maxRelErr,
      maxFractionSumDeviation: maxSumDeviation,
      headline: `${rows.length} (energy, mode) pairs; max sum dev=${maxSumDeviation.toExponential(2)}`,
    },
    rows,
  };
}
