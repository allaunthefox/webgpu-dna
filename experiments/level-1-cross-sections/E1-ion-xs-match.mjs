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

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { SEEDS } from '../lib/seeds.mjs';
import { captureEnv } from '../lib/env.mjs';

// Geant4-DNA Born ionization scale factor (G4DNABornIonisationModel.cc):
//   scaleFactor = (1e-22 / 3.343) m² = 2.993e-23 m² = 2.993e-5 nm²
const BORN_SCALE_NM2 = 2.993e-5;

const REPO_ROOT = join(import.meta.dirname, '..', '..');
const RAW_PATH = join(REPO_ROOT, 'data', 'g4emlow', 'dna', 'sigma_ionisation_e_born.dat');
const WGSL_PATH = join(REPO_ROOT, 'public', 'cross_sections.wgsl');

// ---------- parsing ----------

function parseRawIonisation(text) {
  // Format: per-line "E s0 s1 s2 s3 s4" — energy in eV, σ_shell in raw units.
  const rows = [];
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t || t.startsWith('#')) continue;
    const cols = t.split(/\s+/).map(Number);
    if (cols.length < 2 || cols.some(Number.isNaN)) continue;
    rows.push(cols);
  }
  return rows.map(([E, ...shells]) => ({
    energyEv: E,
    shells,
    sigmaTotalNm2: shells.reduce((a, b) => a + b, 0) * BORN_SCALE_NM2,
  }));
}

function parseWgslArray(text, name) {
  // Matches: const NAME=array<f32,N>(v0,v1,...,vN-1);
  const re = new RegExp(`const\\s+${name}\\s*=\\s*array<f32,\\d+>\\(([^)]*)\\)`);
  const m = text.match(re);
  if (!m) throw new Error(`WGSL array ${name} not found in cross_sections.wgsl`);
  return m[1].split(',').map((s) => Number(s.trim()));
}

// ---------- log-log interp ----------

function logLogInterp(xs, ys, x) {
  // Returns ys interpolated at x using log-log linear interp on (xs, ys).
  // Outside [xs[0], xs[N-1]] returns 0.
  if (x <= 0) return 0;
  if (x < xs[0] || x > xs[xs.length - 1]) return 0;
  // Binary search for the bracketing interval.
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid; else hi = mid;
  }
  const y0 = ys[lo], y1 = ys[hi];
  if (y0 <= 0 || y1 <= 0) {
    // Linear interp through any zero — log-log undefined.
    const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return y0 + t * (y1 - y0);
  }
  const lx0 = Math.log(xs[lo]), lx1 = Math.log(xs[hi]);
  const ly0 = Math.log(y0), ly1 = Math.log(y1);
  const t = (Math.log(x) - lx0) / (lx1 - lx0);
  return Math.exp(ly0 + t * (ly1 - ly0));
}

// ---------- experiment ----------

export async function runE1() {
  const rawText = readFileSync(RAW_PATH, 'utf8');
  const wgslText = readFileSync(WGSL_PATH, 'utf8');

  const raw = parseRawIonisation(rawText);
  const xe = parseWgslArray(wgslText, 'XE');
  const xi = parseWgslArray(wgslText, 'XI');

  if (xe.length !== xi.length) {
    throw new Error(`XE/XI length mismatch: ${xe.length} vs ${xi.length}`);
  }
  if (xe.length === 0) {
    throw new Error('XE is empty — converter output looks broken');
  }

  const xeMin = xe[0];
  const xeMax = xe[xe.length - 1];

  const rows = [];
  for (const r of raw) {
    if (r.energyEv < xeMin || r.energyEv > xeMax) continue;

    const sigmaWgsl = logLogInterp(xe, xi, r.energyEv);
    const sigmaRaw = r.sigmaTotalNm2;

    let relErr;
    if (sigmaRaw === 0 && sigmaWgsl === 0) {
      relErr = 0;
    } else if (sigmaRaw === 0) {
      relErr = sigmaWgsl > 1e-10 ? Infinity : 0;
    } else {
      relErr = Math.abs(sigmaWgsl - sigmaRaw) / sigmaRaw;
    }

    rows.push({
      energyEv: r.energyEv,
      sigmaRawNm2: sigmaRaw,
      sigmaWgslNm2: sigmaWgsl,
      relErr,
      // status filled in below once we know the per-row pass criteria.
      status: 'pending',
    });
  }

  if (rows.length === 0) {
    throw new Error('No raw rows fell within the WGSL support — converter range mismatch?');
  }

  // ---- aggregate metrics ----
  const peakRaw = Math.max(...raw.map((r) => r.sigmaTotalNm2));
  const peakWgsl = Math.max(...xi);
  const peakRatio = peakRaw > 0 ? peakWgsl / peakRaw : 0;

  const nonZero = rows.filter((r) => r.sigmaRawNm2 > 0);
  const meaningful = rows.filter((r) => r.sigmaRawNm2 > 1e-6);

  const quantile = (xs, q) => {
    if (xs.length === 0) return 0;
    const sorted = [...xs].sort((a, b) => a - b);
    const idx = (sorted.length - 1) * q;
    const lo = Math.floor(idx), hi = Math.ceil(idx);
    return lo === hi ? sorted[lo] : sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
  };

  const nonZeroErrs = nonZero.map((r) => r.relErr);
  const meaningfulErrs = meaningful.map((r) => r.relErr);

  const medianRelErr = quantile(nonZeroErrs, 0.5);
  const p90RelErr = quantile(meaningfulErrs, 0.9);
  const maxRelErrMeaningful = meaningfulErrs.length === 0 ? 0 : Math.max(...meaningfulErrs);

  // ---- pass-bar evaluation ----
  const PASS_BAR = {
    peakRatioMin: 0.95,
    peakRatioMax: 1.05,
    medianRelErrMax: 1e-3,
    p90RelErrMax: 5e-2,
    maxRelErrMax: 1e-1, // loose: log-log subsampling near shell openings
  };

  const peakOk = peakRatio >= PASS_BAR.peakRatioMin && peakRatio <= PASS_BAR.peakRatioMax;
  const medianOk = medianRelErr < PASS_BAR.medianRelErrMax;
  const p90Ok = p90RelErr < PASS_BAR.p90RelErrMax;
  const maxOk = maxRelErrMeaningful < PASS_BAR.maxRelErrMax;

  const failures = [];
  if (!peakOk) failures.push(`scale ratio ${peakRatio.toFixed(4)} ∉ [0.95, 1.05]`);
  if (!medianOk) failures.push(`median rel_err ${medianRelErr.toExponential(2)} ≥ 1e-3`);
  if (!p90Ok) failures.push(`p90 rel_err ${p90RelErr.toExponential(2)} ≥ 5e-2`);
  if (!maxOk) {
    const worst = meaningful.reduce((a, b) => (b.relErr > a.relErr ? b : a), meaningful[0]);
    failures.push(
      `max rel_err ${maxRelErrMeaningful.toExponential(2)} ≥ 1e-1 at ${worst.energyEv.toFixed(2)} eV`,
    );
  }

  // Per-row status — pass if rel_err < 1e-1 OR row is in zero region.
  for (const r of rows) {
    if (r.sigmaRawNm2 === 0 && r.sigmaWgslNm2 < 1e-10) {
      r.status = 'pass';
    } else if (Number.isFinite(r.relErr) && r.relErr < PASS_BAR.maxRelErrMax) {
      r.status = 'pass';
    } else {
      r.status = 'fail';
    }
  }

  const status = failures.length === 0 ? 'pass' : 'fail';
  const diagnosis = failures.length === 0 ? null : failures.join('; ');

  const summary = {
    nRows: rows.length,
    peakSigmaRawNm2: peakRaw,
    peakSigmaWgslNm2: peakWgsl,
    peakRatio,
    medianRelErr,
    p90RelErr,
    maxRelErrMeaningful,
    nNonZero: nonZero.length,
    nMeaningful: meaningful.length,
  };

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
        scaleFactor: BORN_SCALE_NM2,
      },
    },
    env: captureEnv(),
    status,
    diagnosis,
    summary,
    rows,
  };
}
